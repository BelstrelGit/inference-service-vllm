import json

import pytest

from analysis.aggregate import aggregate, server_stats, steady_output_tok_s
from analysis.compare_vllm_bench import compare
from client import mock_server
from client.load import Scenario, execute
from client.records import RequestRecord


def snap(t: float, **samples: float) -> dict:
    return {"t_unix": t, "t_perf": t, "samples": {f"vllm:{k}": v for k, v in samples.items()}}


def test_server_stats_window_deltas_and_gauges():
    snaps = [
        snap(0, time_to_first_token_seconds_sum=1.0, time_to_first_token_seconds_count=10,
             num_requests_waiting=0, generation_tokens_total=0),
        snap(1, time_to_first_token_seconds_sum=1.5, time_to_first_token_seconds_count=12,
             num_requests_waiting=4, generation_tokens_total=50),
        snap(2, time_to_first_token_seconds_sum=3.0, time_to_first_token_seconds_count=20,
             num_requests_waiting=2, generation_tokens_total=200),
        snap(3, time_to_first_token_seconds_sum=9.0, time_to_first_token_seconds_count=30,
             num_requests_waiting=9, generation_tokens_total=999),
    ]  # fmt: skip
    s = server_stats(snaps, (0.5, 2.5))  # baseline = snapshot at t=0, end = snapshot at t=2
    assert s["server_ttft_mean_ms"] == pytest.approx((3.0 - 1.0) / (20 - 10) * 1000)
    assert s["server_gen_tok_s"] == pytest.approx(200 / 2)
    assert s["server_waiting_max"] == 4 and s["server_waiting_mean"] == pytest.approx(3)
    assert s["server_snapshots_in_window"] == 2
    assert s["server_kv_tokens_per_running"] is None  # no cache_config_info -> no KV accounting


def test_server_stats_kv_tokens_from_cache_config_info():
    info = 'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="53696"}'
    snaps = [snap(t, num_requests_running=r, kv_cache_usage_perc=kv)
             for t, r, kv in ((0, 10, 0.7), (1, 10, 0.8), (2, 12, 0.9))]  # fmt: skip
    for s in snaps:
        s["samples"][info] = 1.0
    s = server_stats(snaps, (0.5, 2.5))
    assert s["server_kv_capacity_tokens"] == 53696
    assert s["server_kv_tokens_mean"] == pytest.approx(0.85 * 53696)
    assert s["server_kv_tokens_per_running"] == pytest.approx(0.85 * 53696 / 11)


def test_steady_throughput_counts_all_phases_between_measure_sends():
    def rec(phase: str, start: float, tokens: list[float]) -> RequestRecord:
        return RequestRecord("id", "run", "s", phase, "completions", 4, start,
                             status="ok", token_timestamps=tokens)  # fmt: skip

    records = [
        rec("warmup", 0.0, [1.0, 9.0, 11.0]),  # 1.0 is before the window; 9.0 counts
        rec("measure", 2.0, [3.0, 4.0]),
        rec("measure", 12.0, [13.0, 14.0]),  # last send at 12: tokens after it are drain
    ]
    assert steady_output_tok_s(records) == pytest.approx(4 / 10)  # 3, 4, 9, 11 in [2, 12]
    assert steady_output_tok_s(records[:2]) is None


def test_compare_with_vllm_bench():
    summary = {"ttft_s": {"p50": 0.100, "p95": 0.2}, "tpot_s": {"p50": 0.05},
               "output_token_throughput": 120.0}  # fmt: skip
    bench = {"median_ttft_ms": 95.0, "p95_ttft_ms": 250.0, "median_tpot_ms": 50.0,
             "output_throughput": 118.0}  # fmt: skip
    rows = {r["metric"]: r for r in compare(summary, bench)}
    assert (
        rows["TTFT p50, ms"]["diff"] == pytest.approx(100 / 95 - 1) and rows["TTFT p50, ms"]["ok"]
    )
    assert not rows["TTFT p95, ms"]["ok"]
    assert rows["ITL p50, ms"]["diff"] is None  # missing on both sides -> not ok, not a crash


async def test_end_to_end_mock_sweep_aggregate(tmp_path):
    server = mock_server.start(ttft=0.03, itl=0.003, max_num_seqs=2)
    try:
        sc = Scenario.from_dict(dict(name="e2e", series="ser", concurrency=[1, 4], num_requests=6,
                                     warmup_requests=1, repeats=2, input_len=8, output_len=5,
                                     metrics_interval_s=0.05, cooldown_s=0))  # fmt: skip
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for lvl in sc.levels():
            for rep in (1, 2):
                await execute(sc, base, tmp_path / "raw", lvl, rep)
    finally:
        server.shutdown()
    out_dir, levels = aggregate(tmp_path / "raw" / "ser", tmp_path / "agg")
    assert [lv["level"] for lv in levels] == ["c1", "c4"]
    assert all(lv["repeats"] == 2 and lv["ok"] == 6 for lv in levels)
    c1, c4 = levels
    # 4 clients on 2 server slots: TTFT includes queueing; throughput still higher than c=1
    assert c4["ttft_p95_ms"] > c1["ttft_p95_ms"]
    assert c4["output_token_throughput"] > c1["output_token_throughput"]
    assert c4["server_waiting_max"] > 0
    assert c4["server_waiting_capacity_max"] == c4["server_waiting_max"]
    assert c4["server_waiting_deferred_max"] == 0
    assert 0 < c4["client_cpu_util"] < 2
    runs = (out_dir / "runs.csv").read_text().splitlines()
    assert len(runs) == 1 + 4
    assert "| c4 |" in (out_dir / "levels.md").read_text()
    json.loads(next((tmp_path / "raw" / "ser").glob("*.meta.json")).read_text())
