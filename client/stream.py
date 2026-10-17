"""One streaming request to an OpenAI-compatible server (vLLM), timed on the client.

Timing points (perf_counter):
  started_at      just before the request is sent
  first_chunk_at  first SSE `data:` chunk of any kind
  first_token_at  first chunk carrying generated text -> TTFT
  token_timestamps every chunk carrying generated text -> ITL
  completed_at    `data: [DONE]`

Token counts come from the server's `usage` (stream_options.include_usage), never from
counting chunks: one chunk is usually one token in vLLM, but that is not guaranteed.
"""

from __future__ import annotations

import json
import time

import httpx

from client.records import RequestRecord


def build_payload(
    endpoint: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool
) -> tuple[str, dict]:
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True  # vLLM extension: always generate exactly max_tokens
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
        return "/v1/chat/completions", payload
    if endpoint == "completions":
        payload["prompt"] = prompt
        return "/v1/completions", payload
    raise ValueError(f"unknown endpoint: {endpoint}")


def _chunk_text(endpoint: str, choice: dict) -> str:
    if endpoint == "chat":
        return (choice.get("delta") or {}).get("content") or ""
    return choice.get("text") or ""


async def send_request(
    client: httpx.AsyncClient,
    record: RequestRecord,
    model: str,
    prompt: str,
    ignore_eos: bool = True,
) -> RequestRecord:
    """Fill `record` in place with timings, usage and status; never raises on request errors."""
    path, payload = build_payload(record.endpoint, model, prompt, record.max_tokens, ignore_eos)
    record.started_at_unix = time.time()
    record.started_at = time.perf_counter()
    try:
        async with client.stream("POST", path, json=payload) as resp:
            record.http_status = resp.status_code
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                record.error = body[:500]
                return record
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
    except httpx.TimeoutException as e:
        record.status = "timeout"
        record.error = f"{type(e).__name__}: {e}"
        return record
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        record.error = f"{type(e).__name__}: {e}"
        return record

    if record.completed_at is None:
        record.error = "stream ended without [DONE]"
    elif record.output_tokens is None:
        record.error = "no usage in stream"
    else:
        record.status = "ok"
    return record
