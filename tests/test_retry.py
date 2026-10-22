"""Retry semantics, backoff cap and the per-attempt log (client.stream), without a server.

httpx.MockTransport makes every failure deterministic: the handler decides per call whether
the attempt fails to connect, times out, breaks mid-stream or succeeds.
"""

import json
import random

import httpx
import pytest

from client.records import RequestRecord, ttft
from client.stream import backoff_delay, send_request


def record() -> RequestRecord:
    return RequestRecord(request_id="r", run_id="t", scenario="t", phase="measure",
                         endpoint="completions", max_tokens=2, started_at=0.0)  # fmt: skip


def sse(n_tokens: int = 2, done: bool = True) -> bytes:
    chunks = [{"choices": [{"index": 0, "text": f" t{i}", "finish_reason": None}]}
              for i in range(n_tokens)]  # fmt: skip
    if done:
        chunks.append({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": n_tokens}})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


class BrokenStream(httpx.AsyncByteStream):
    """Sends `head`, then the connection drops."""

    def __init__(self, head: bytes):
        self.head = head

    async def __aiter__(self):
        if self.head:
            yield self.head
        raise httpx.ReadError("connection reset by peer")


def client_with(script: list) -> httpx.AsyncClient:
    """Each call takes the next step: an exception class to raise, or a Response factory."""
    calls = iter(script)

    def handler(request: httpx.Request) -> httpx.Response:
        step = next(calls)
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("scripted", request=request)
        return step()

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock")


def ok() -> httpx.Response:
    return httpx.Response(200, content=sse())


async def send(script: list, **kw) -> RequestRecord:
    kw = {"retries": 5, "backoff_s": 0.001, "rng": random.Random(0), **kw}
    async with client_with(script) as c:
        return await send_request(c, record(), "m", [1, 2], **kw)


async def test_connect_timeout_is_retried():
    rec = await send([httpx.ConnectTimeout, httpx.ConnectTimeout, ok])
    assert rec.status == "ok" and rec.retry_count == 2
    assert [a["outcome"] for a in rec.attempts] == ["connect_timeout", "connect_timeout", "ok"]


async def test_connect_refused_is_retried():
    rec = await send([httpx.ConnectError, ok])
    assert rec.status == "ok"
    assert [a["outcome"] for a in rec.attempts] == ["connect_error", "ok"]


async def test_read_timeout_is_not_retried():
    rec = await send([httpx.ReadTimeout, ok])
    assert rec.status == "timeout" and rec.retry_count == 0
    assert [a["outcome"] for a in rec.attempts] == ["timeout"]


async def test_disconnect_before_first_token_is_retried():
    rec = await send([lambda: httpx.Response(200, stream=BrokenStream(b"")), ok])
    assert rec.status == "ok"
    assert rec.attempts[0]["outcome"] == "disconnect" and not rec.attempts[0]["mid_stream"]


async def test_mid_stream_disconnect_is_not_retried():
    head = sse(n_tokens=1, done=False)
    rec = await send([lambda: httpx.Response(200, stream=BrokenStream(head)), ok])
    assert rec.status == "error" and rec.retry_count == 0
    assert rec.first_token_at is not None  # the user saw a token: re-sending would hide it
    (a,) = rec.attempts
    assert a["outcome"] == "disconnect" and a["mid_stream"] and a["retry_in_s"] is None


async def test_incomplete_stream_mid_stream_is_error():
    rec = await send([lambda: httpx.Response(200, content=sse(done=False)), ok])
    assert rec.status == "error" and rec.error == "stream ended without [DONE]"
    assert [a["outcome"] for a in rec.attempts] == ["incomplete"]


async def test_retries_exhausted_and_503():
    rec = await send([lambda: httpx.Response(503, text="busy")] * 3, retries=2)
    assert rec.status == "error" and rec.http_status == 503 and rec.retry_count == 2
    assert [a["http_status"] for a in rec.attempts] == [503, 503, 503]
    assert rec.attempts[-1]["retry_in_s"] is None  # no retry planned after the last one


def test_backoff_cap_keeps_jitter():
    rng = random.Random(1)
    uncapped = [backoff_delay(k, 0.5, None, rng) for k in range(1, 12)]
    assert uncapped[-1] > 100  # 0.5 * 2**10 * U(0.5, 1.5) >= 256
    rng = random.Random(1)
    capped = [backoff_delay(k, 0.5, 8.0, rng) for k in range(1, 40)]
    assert all(d <= 8.0 * 1.5 for d in capped)
    late = capped[10:]  # all at the cap: spread by the jitter, not lined up at 8.0
    assert min(late) >= 4.0 and max(late) - min(late) > 2.0
    assert capped[:4] == pytest.approx(uncapped[:4])  # below the cap nothing changes


async def test_attempt_timeline_and_latency_from_first_attempt():
    rec = await send([httpx.ConnectError] * 3 + [ok], backoff_s=0.01, backoff_max_s=0.02)
    atts = rec.attempts
    assert len(atts) == 4 and rec.retry_count == 3
    assert atts[0]["start"] >= rec.started_at
    for a, b in zip(atts, atts[1:], strict=False):
        assert a["start"] <= a["end"] <= b["start"]
        assert 0 < a["retry_in_s"] <= 0.02 * 1.5  # cap applied
        assert b["start"] - a["end"] >= a["retry_in_s"] * 0.9  # the planned pause was taken
    # TTFT counts from the first attempt, so it includes the failed attempts and pauses
    assert ttft(rec) >= sum(a["retry_in_s"] for a in atts[:-1])
    assert atts[-1]["outcome"] == "ok" and atts[-1]["retry_in_s"] is None


def test_old_records_without_attempts_still_load():
    old = {k: v for k, v in record().to_dict().items() if k != "attempts"}
    assert RequestRecord.from_dict(old).attempts == []
