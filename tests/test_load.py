import json
import random

import httpx
import pytest

from client import mock_server
from client.load import Run, Scenario, execute, load_scenario, parse_set
from client.prompts import PromptFactory, PromptSpec
from client.records import RequestRecord
from client.server_metrics import metric, parse_prometheus
from client.stream import send_request


@pytest.fixture
def mock():
    server = mock_server.start(ttft=0.05, itl=0.005, max_num_seqs=4)
    yield server
    server.shutdown()


def url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def scenario(**kw) -> Scenario:
    base = dict(name="t", series="s", num_requests=8, warmup_requests=2, input_len=16,
                output_len=4, cooldown_s=0, metrics_interval_s=0.05)  # fmt: skip
    base.update(kw)
    sc = Scenario.from_dict(base)
    sc.validate()
    return sc


# --- prompts -------------------------------------------------------------------------------


async def test_token_id_prompts_exact_unique_deterministic():
    spec = PromptSpec(input_len=50, seed=7)
    a, b = PromptFactory(spec), PromptFactory(spec)
    pa = [await a.next() for _ in range(3)]
    pb = [await b.next() for _ in range(3)]
    assert pa == pb  # same seed -> same prompts
    assert all(len(p) == 50 for p in pa)
    assert len({tuple(p) for p in pa}) == 3
    assert all(spec.token_id_low <= t < spec.token_id_high for p in pa for t in p)


async def test_shared_prefix():
    f = PromptFactory(PromptSpec(input_len=20, shared_prefix_len=8))
    p1, p2 = await f.next(), await f.next()
    assert p1[:8] == p2[:8] and p1[8:] != p2[8:]


async def test_text_prompts_fit_exact_length(mock):
    async with httpx.AsyncClient(base_url=url(mock)) as c:
        f = PromptFactory(PromptSpec(input_len=30, format="text"), c, mock_server.MODEL)
        text = await f.next()
        r = await c.post("/tokenize", json={"model": mock_server.MODEL, "prompt": text})
    assert r.json()["count"] == 30


# --- retries -------------------------------------------------------------------------------


def _record() -> RequestRecord:
    return RequestRecord(request_id="r", run_id="t", scenario="t", phase="measure",
                         endpoint="completions", max_tokens=3, started_at=0.0)  # fmt: skip


async def test_retry_on_503_then_success():
    server = mock_server.start(ttft=0.01, itl=0.001, fail_first=2)
    try:
        async with httpx.AsyncClient(base_url=url(server)) as c:
            rec = await send_request(c, _record(), mock_server.MODEL, [1, 2, 3], retries=3,
                                     backoff_s=0.01, rng=random.Random(0))  # fmt: skip
    finally:
        server.shutdown()
    assert rec.status == "ok" and rec.retry_count == 2
    assert rec.error is None and len(rec.token_timestamps) == 3


async def test_no_retry_on_404_and_retries_exhausted():
    server = mock_server.start(ttft=0.01, itl=0.001, fail_first=5)
    try:
        async with httpx.AsyncClient(base_url=url(server)) as c:
            rec = await send_request(c, _record(), "wrong-model", [1], retries=3, backoff_s=0.01)
            assert rec.http_status == 404 and rec.retry_count == 0
            rec = await send_request(
                c, _record(), mock_server.MODEL, [1], retries=1, backoff_s=0.01
            )
            assert rec.http_status == 503 and rec.retry_count == 1 and rec.status == "error"
    finally:
        server.shutdown()


# --- load modes ----------------------------------------------------------------------------


async def test_closed_loop_concurrency_and_counts(mock, tmp_path):
    sc = scenario(concurrency=3)
    meta = await execute(sc, url(mock), tmp_path, ("c3", 3), rep=1)
    recs = [
        json.loads(x) for x in (tmp_path / "s" / f"{meta['run_id']}.jsonl").read_text().splitlines()
    ]
    assert [r["phase"] for r in recs].count("warmup") == 2
    assert [r["phase"] for r in recs].count("measure") == 8
    assert all(r["status"] == "ok" and r["prompt_tokens"] == 16 for r in recs)
    assert mock.state.max_running_seen == 3  # never more in flight than concurrency
    assert meta["summary"]["ok"] == 8 and meta["summary"]["output_token_throughput"] > 0
    assert meta["server_metrics"]["snapshots"] > 0
    assert 0 < meta["client"]["cpu_util"] < 2
    assert (tmp_path / "s" / f"{meta['run_id']}.server.jsonl").exists()


async def test_closed_loop_queueing_shows_in_ttft_and_metrics(mock, tmp_path):
    # mock runs 4 at a time: with 8 clients, half wait in the server queue
    sc = scenario(concurrency=8, num_requests=16, warmup_requests=0, output_len=20)
    meta = await execute(sc, url(mock), tmp_path, ("c8", 8), rep=1)
    s = meta["summary"]
    # mock TTFT is 0.05 s; one service turn is 0.05 + 19 * 0.005 = 0.145 s. In steady state
    # every request waits about one turn in the queue, and that wait is part of TTFT.
    assert s["ttft_s"]["p50"] > 0.05 + 0.1
    snaps = [
        json.loads(x)
        for x in (tmp_path / "s" / f"{meta['run_id']}.server.jsonl").read_text().splitlines()
    ]
    assert max(metric(x["samples"], "vllm:num_requests_waiting") or 0 for x in snaps) > 0


async def test_open_loop_poisson_schedule(mock, tmp_path):
    sc = scenario(mode="open", rate=40.0, num_requests=30, warmup_requests=0)
    meta = await execute(sc, url(mock), tmp_path, ("r40", 40.0), rep=1)
    recs = [
        json.loads(x) for x in (tmp_path / "s" / f"{meta['run_id']}.jsonl").read_text().splitlines()
    ]
    assert len(recs) == 30
    sched = sorted(r["scheduled_at"] for r in recs)
    gaps = [b - a for a, b in zip(sched, sched[1:], strict=False)]
    assert 0.5 / 40 < sum(gaps) / len(gaps) < 2 / 40  # mean inter-arrival ~ 1/rate
    # median, not p99: macOS timer coalescing can stall a background process for ~0.4 s
    assert meta["summary"]["send_lag_s"]["p50"] < 0.02  # the generator kept up


def test_claim_is_exact_for_back_to_back_arrivals():
    sc = scenario(num_requests=5, warmup_requests=2)
    run = Run(sc, "r", "m", PromptFactory(PromptSpec(input_len=4)), rep=1)
    claims = []
    while (c := run.claim(0.0)) is not None:  # no awaits between claims
        claims.append(c)
    assert [p for _, p in claims] == ["warmup"] * 2 + ["measure"] * 5
    assert [i for i, _ in claims] == list(range(7))


def test_duration_based_stop():
    sc = scenario(num_requests=None, duration_s=10.0, warmup_requests=1)
    run = Run(sc, "r", "m", PromptFactory(PromptSpec(input_len=4)), rep=1)
    assert run.claim(0.0)[1] == "warmup"
    assert run.claim(1.0)[1] == "measure"  # measure window starts here
    assert run.claim(10.9)[1] == "measure"
    assert run.claim(11.0) is None


def test_scenario_validation_and_overrides(tmp_path):
    with pytest.raises(ValueError, match="unknown scenario keys"):
        Scenario.from_dict({"name": "x", "series": "s", "concurency": 4})
    with pytest.raises(ValueError, match="exactly one"):
        scenario(duration_s=10)
    with pytest.raises(ValueError, match="needs `rate`"):
        scenario(mode="open")
    p = tmp_path / "s.yaml"
    p.write_text("name: x\nseries: s\nconcurrency: [1, 2]\n")
    sc = load_scenario(str(p), ["concurrency=[4,8]", "repeats=2"])
    assert sc.levels() == [("c4", 4), ("c8", 8)] and sc.repeats == 2
    assert parse_set(["rate=0.5"]) == {"rate": 0.5}


# --- /metrics parsing ----------------------------------------------------------------------


def test_parse_prometheus():
    text = """# HELP vllm:x
vllm:num_requests_running{engine="0",model_name="a b"} 3.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="a"} 2.0
vllm:time_to_first_token_seconds_sum{model_name="a"} 0.5
vllm:prompt_tokens_created{model_name="a"} 1.7e9
python_gc_objects_collected_total{generation="0"} 1.0
vllm:plain 7
"""
    s = parse_prometheus(text)
    assert metric(s, "vllm:num_requests_running") == 3.0
    assert metric(s, "vllm:time_to_first_token_seconds_sum") == 0.5
    assert metric(s, "vllm:plain") == 7.0
    assert metric(s, "vllm:time_to_first_token_seconds_bucket") is None
    assert metric(s, "vllm:prompt_tokens_created") is None
    assert not any(k.startswith("python_") for k in s)


async def test_runs_get_distinct_prompts(mock, tmp_path):
    # Regression: two levels with the same repeat number used to get identical prompts, so the
    # second run was served from the server's prefix cache.
    sc = scenario(concurrency=[1, 2], num_requests=2, warmup_requests=0)
    metas = [await execute(sc, url(mock), tmp_path, lvl, rep=1) for lvl in sc.levels()]
    assert metas[0]["prompt_seed"] != metas[1]["prompt_seed"]
    first = [PromptFactory(PromptSpec(input_len=16, seed=m["prompt_seed"])) for m in metas]
    assert await first[0].next() != await first[1].next()


async def test_stagger_spreads_closed_loop_starts(mock, tmp_path):
    sc = scenario(
        concurrency=4,
        num_requests=4,
        warmup_requests=0,
        warmup_s=0.4,
        stagger_s=0.4,
        output_len=100,  # 0.05 + 99 * 0.005 = 0.55 s > stagger: first 4 starts = 4 workers
    )
    meta = await execute(sc, url(mock), tmp_path, ("c4", 4), rep=1)
    recs = [
        json.loads(x) for x in (tmp_path / "s" / f"{meta['run_id']}.jsonl").read_text().splitlines()
    ]
    starts = sorted(r["started_at"] for r in recs)[:4]
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert all(0.07 < g < 0.2 for g in gaps)  # ~0.1 s apart instead of all at once
    with pytest.raises(ValueError, match="warmup_s"):
        scenario(concurrency=4, stagger_s=10, warmup_s=5)
