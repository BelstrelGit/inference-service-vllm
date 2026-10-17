import json

import httpx
import pytest

from client import mock_server
from client.records import RequestRecord, e2e, itls, percentile, summarize, tpot, ttft
from client.run import parse_args, run
from client.stream import send_request

TTFT, ITL = 0.15, 0.02


@pytest.fixture(scope="module")
def base_url():
    server = mock_server.start(ttft=TTFT, itl=ITL)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def new_record(endpoint="completions", max_tokens=10) -> RequestRecord:
    return RequestRecord(
        request_id="r1",
        run_id="t",
        scenario="test",
        phase="measure",
        endpoint=endpoint,
        max_tokens=max_tokens,
        started_at=0.0,
    )


@pytest.mark.parametrize("endpoint", ["completions", "chat"])
async def test_stream_timings_match_mock(base_url, endpoint):
    async with httpx.AsyncClient(base_url=base_url) as c:
        rec = await send_request(c, new_record(endpoint), mock_server.MODEL, "a b c d")
    assert rec.status == "ok", rec.error
    assert rec.http_status == 200
    assert (rec.prompt_tokens, rec.output_tokens) == (4, 10)
    assert rec.finish_reason == "length"
    assert len(rec.token_timestamps) == 10
    # mock sleeps exactly TTFT / ITL; allow scheduling jitter on the slow side only
    assert TTFT <= ttft(rec) < TTFT + 0.1
    assert ITL * 0.9 <= tpot(rec) < ITL + 0.02
    assert e2e(rec) >= TTFT + 9 * ITL


async def test_chat_role_chunk_is_not_first_token(base_url):
    async with httpx.AsyncClient(base_url=base_url) as c:
        rec = await send_request(c, new_record("chat"), mock_server.MODEL, "x")
    # role-only chunk arrives before prefill is done; TTFT must wait for real text
    assert rec.first_chunk_at < rec.first_token_at
    assert rec.first_token_at - rec.first_chunk_at >= TTFT * 0.9


async def test_http_error_recorded(base_url):
    async with httpx.AsyncClient(base_url=base_url) as c:
        rec = await send_request(c, new_record(), "no-such-model", "x")
    assert rec.status == "error"
    assert rec.http_status == 404
    assert "not found" in rec.error


async def test_timeout_recorded(base_url):
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(TTFT / 3)) as c:
        rec = await send_request(c, new_record(), mock_server.MODEL, "x")
    assert rec.status == "timeout"
    assert ttft(rec) is None


def test_derived_metrics_formulas():
    r = new_record()
    r.started_at, r.first_token_at, r.completed_at, r.output_tokens = 10.0, 10.5, 11.4, 10
    r.token_timestamps = [10.5, 10.6, 10.8]
    assert ttft(r) == pytest.approx(0.5)
    assert e2e(r) == pytest.approx(1.4)
    assert tpot(r) == pytest.approx(0.1)  # 0.9 s / (10 - 1): first token excluded
    assert itls(r) == pytest.approx([0.1, 0.2])
    r.output_tokens = 1
    assert tpot(r) is None


def test_percentile_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([3.0], 99) == 3.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile(list(range(101)), 90) == pytest.approx(90)


def test_summary_excludes_warmup_and_errors():
    recs = []
    for phase, status, t in (
        ("warmup", "ok", 9.0),
        ("measure", "ok", 1.0),
        ("measure", "error", None),
    ):
        r = new_record()
        r.phase, r.status = phase, status
        if t is not None:
            r.started_at, r.first_token_at, r.completed_at, r.output_tokens = 0.0, t, t + 1, 5
        recs.append(r)
    s = summarize(recs)
    assert (s["requests"], s["ok"], s["errors"]) == (2, 1, 1)
    assert s["ttft_s"]["p50"] == pytest.approx(1.0)


async def test_run_writes_jsonl_and_meta(base_url, tmp_path):
    args = parse_args(
        [
            "--base-url",
            base_url,
            "-n",
            "3",
            "--warmup",
            "1",
            "--max-tokens",
            "5",
            "--out-dir",
            str(tmp_path),
            "--series",
            "s",
            "--run-id",
            "rid",
        ]
    )
    path = await run(args)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x["phase"] for x in lines] == ["warmup", "measure", "measure", "measure"]
    assert all(x["status"] == "ok" and x["output_tokens"] == 5 for x in lines)
    assert len({x["request_id"] for x in lines}) == 4
    meta = json.loads((tmp_path / "s" / "rid.meta.json").read_text())
    assert meta["model"] == mock_server.MODEL
    assert meta["server"]["version"] == {"version": "mock"}
    assert meta["summary"]["ok"] == 3
