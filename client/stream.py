"""One streaming request to an OpenAI-compatible server (vLLM), timed on the client.

Timing points (perf_counter):
  started_at      just before the first attempt is sent (retries stay inside the latency:
                  the user waited for them too)
  first_chunk_at  first SSE `data:` chunk of any kind
  first_token_at  first chunk carrying generated text -> TTFT
  token_timestamps every chunk carrying generated text -> ITL
  completed_at    `data: [DONE]`

Token counts come from the server's `usage` (stream_options.include_usage), never from
counting chunks: one chunk is usually one token in vLLM, but that is not guaranteed.

Retries: only failures that happened before any generated token, with exponential backoff
(optionally capped) and jitter:
  - the connection could not be established: refused / unreachable (ConnectError) or not
    established in time (ConnectTimeout), i.e. the request was never sent;
  - the connection dropped before the first token (ReadError, WriteError, RemoteProtocolError);
  - HTTP 429/502/503/504.
A stream that broke after the first token is an error, not a retry: re-sending it would
double-count work and hide the failure. A read timeout is not retried either.

Every attempt is logged in `record.attempts` (start/end, outcome, planned backoff before the
next one): enough to rebuild arrivals -> failed attempts -> retry traffic on a time axis.
Latency still counts from the first attempt: the user waited for the retries too.
"""

from __future__ import annotations

import asyncio
import json
import random
import time

import httpx

from client.records import RequestRecord

RETRY_HTTP_STATUS = {429, 502, 503, 504}

# Attempt outcomes (record.attempts[i]["outcome"]):
#   ok                  completed with [DONE] and usage
#   http_error          non-200 status (retried if in RETRY_HTTP_STATUS)
#   connect_error       connection refused / unreachable, nothing sent (retried)
#   connect_timeout     connection not established within the connect timeout (retried)
#   disconnect          connection dropped while waiting for or reading the stream
#   timeout             read / write / pool timeout (not retried)
#   incomplete          stream ended without [DONE] or without usage
#   error               anything else (bad JSON, other httpx errors)
# disconnect and incomplete are retried only before the first token; "mid_stream" in the
# attempt says which side of the first token it happened on.

Prompt = str | list[int]


def build_payload(
    endpoint: str,
    model: str,
    prompt: Prompt,
    max_tokens: int,
    ignore_eos: bool,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True  # vLLM extension: always generate exactly max_tokens
    if endpoint == "chat":
        if not isinstance(prompt, str):
            raise ValueError("chat endpoint needs a text prompt")
        payload["messages"] = [{"role": "user", "content": prompt}]
        return "/v1/chat/completions", payload
    if endpoint == "completions":
        payload["prompt"] = prompt  # text or a list of token ids
        return "/v1/completions", payload
    raise ValueError(f"unknown endpoint: {endpoint}")


def _chunk_text(endpoint: str, choice: dict) -> str:
    if endpoint == "chat":
        return (choice.get("delta") or {}).get("content") or ""
    return choice.get("text") or ""


def _reset(record: RequestRecord) -> None:
    record.first_chunk_at = record.first_token_at = record.completed_at = None
    record.prompt_tokens = record.output_tokens = record.finish_reason = None
    record.http_status = record.error = None
    record.status = "error"
    record.token_timestamps = []


async def _attempt(
    client: httpx.AsyncClient, record: RequestRecord, path: str, payload: dict
) -> tuple[str, bool]:
    """One HTTP attempt -> (outcome, retryable)."""
    try:
        async with client.stream("POST", path, json=payload) as resp:
            record.http_status = resp.status_code
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                record.error = body[:500]
                return "http_error", resp.status_code in RETRY_HTTP_STATUS
            async for line in resp.aiter_lines():
                now = time.perf_counter()
                if not line.startswith("data:"):
                    continue  # blank separators, SSE comments
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    record.completed_at = now
                    break
                chunk = json.loads(data)
                if record.first_chunk_at is None:
                    record.first_chunk_at = now
                for choice in chunk.get("choices") or []:
                    if _chunk_text(record.endpoint, choice):
                        if record.first_token_at is None:
                            record.first_token_at = now
                        record.token_timestamps.append(now)
                    if choice.get("finish_reason"):
                        record.finish_reason = choice["finish_reason"]
                if usage := chunk.get("usage"):
                    record.prompt_tokens = usage.get("prompt_tokens")
                    record.output_tokens = usage.get("completion_tokens")
    except httpx.ConnectTimeout as e:  # before the generic timeout: nothing was sent yet
        record.error = f"{type(e).__name__}: {e}"
        return "connect_timeout", True
    except httpx.TimeoutException as e:
        record.status = "timeout"
        record.error = f"{type(e).__name__}: {e}"
        return "timeout", False
    except httpx.ConnectError as e:
        record.error = f"{type(e).__name__}: {e}"
        return "connect_error", True
    except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
        record.error = f"{type(e).__name__}: {e}"
        return "disconnect", record.first_token_at is None
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        record.error = f"{type(e).__name__}: {e}"
        return "error", False

    if record.completed_at is None:
        record.error = "stream ended without [DONE]"
    elif record.output_tokens is None:
        record.error = "no usage in stream"
    else:
        record.status = "ok"
        return "ok", False
    return "incomplete", record.first_token_at is None


def backoff_delay(
    k: int, backoff_s: float, backoff_max_s: float | None, rng: random.Random
) -> float:
    """Pause before retry k (1-based): min(backoff_s * 2**(k-1), backoff_max_s) * U(0.5, 1.5).

    The cap applies before the jitter, so capped retries stay spread over 0.5-1.5 x the cap
    instead of lining up exactly at it.
    """
    base = backoff_s * 2 ** (k - 1)
    if backoff_max_s is not None:
        base = min(base, backoff_max_s)
    return base * rng.uniform(0.5, 1.5)


async def send_request(
    client: httpx.AsyncClient,
    record: RequestRecord,
    model: str,
    prompt: Prompt,
    ignore_eos: bool = True,
    temperature: float = 0.0,
    retries: int = 0,
    backoff_s: float = 0.5,
    rng: random.Random | None = None,
    backoff_max_s: float | None = None,
) -> RequestRecord:
    """Fill `record` in place with timings, usage, status and attempts; never raises on request
    errors. Retry pauses: `backoff_delay` (jitter, so clients that failed together do not
    retry in lockstep)."""
    path, payload = build_payload(
        record.endpoint, model, prompt, record.max_tokens, ignore_eos, temperature
    )
    rng = rng or random.Random()
    record.started_at_unix = time.time()
    record.started_at = time.perf_counter()
    for attempt in range(retries + 1):
        if attempt:
            _reset(record)
            record.retry_count = attempt
            await asyncio.sleep(record.attempts[-1]["retry_in_s"])
        start = time.perf_counter()
        outcome, retryable = await _attempt(client, record, path, payload)
        retry = retryable and attempt < retries
        record.attempts.append({
            "start": start,
            "end": time.perf_counter(),
            "outcome": outcome,
            "http_status": record.http_status,
            "mid_stream": record.first_token_at is not None,
            "error": None if record.error is None else record.error[:200],
            "retry_in_s": backoff_delay(attempt + 1, backoff_s, backoff_max_s, rng)
            if retry else None,
        })  # fmt: skip
        if not retry:
            break
    return record
