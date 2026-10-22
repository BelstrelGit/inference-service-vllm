"""Exp5 tooling without a GPU: mock-server outage under open-loop load, analysis/restart.py on
the result, and scripts/fault.sh against a fake kubectl."""

import asyncio
import json
import os
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from analysis import restart
from analysis.restart import Criterion, Req, cover_counts, recovery, segments
from client import mock_server
from client.load import Scenario, execute

ROOT = Path(__file__).resolve().parent.parent


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --- helpers of the analysis ---------------------------------------------------------------


def test_parse_ts_kubernetes_and_kubelet_formats():
    assert restart.parse_ts("2026-10-21T05:22:46Z") == pytest.approx(1792560166.0)
    assert restart.parse_ts("2026-10-21T05:22:46.123456789Z") == pytest.approx(1792560166.123456)
    assert restart.parse_ts("nonsense") is None


def test_counter_segments_are_not_glued():
    series = [(0, 0), (1, 4), (2, 10), (10, 0), (11, 3)]  # gap + reset = new process
    segs = segments(series, max_gap_s=5)
    assert [s["delta"] for s in segs] == [10, 3]
    series = [(0, 5), (1, 7), (2, 1), (3, 2)]  # reset without a gap
    assert [s["delta"] for s in segments(series, max_gap_s=5)] == [2, 1]


def test_cover_counts_matches_bruteforce():
    ivs = [(0.2, 3.7), (1.0, 1.4), (2.5, 9.0), (5.0, 5.0)]
    got = cover_counts(ivs, start=0.0, bin_s=1.0, n=10)
    want = [sum(a <= i + 0.5 < b for a, b in ivs) for i in range(10)]
    assert got == want


def req(arrival: float, ttft: float | None, ok: bool = True, attempts: int = 1) -> Req:
    atts = [{"start": arrival + k, "end": arrival + k + 0.1, "outcome": "connect_error",
             "mid_stream": False} for k in range(attempts - 1)]  # fmt: skip
    end = arrival + (ttft or 0) + 1
    atts.append({"start": arrival + attempts - 1, "end": end,
                 "outcome": "ok" if ok else "connect_error", "mid_stream": False})  # fmt: skip
    first = None if ttft is None else arrival + ttft
    return Req(arrival, end, ok, "measure", ttft, first, atts[-1]["outcome"], False, atts)


def snaps(points: list[tuple[float, float]]) -> list[dict]:
    return [{"t_unix": t, "samples": {"vllm:num_requests_waiting": w}} for t, w in points]


def test_recovery_criterion_on_synthetic_timeline():
    # baseline: 1 req/s, TTFT 100 ms, until the fault at t=100; Ready at t=200
    reqs = [req(float(t), 0.1) for t in range(0, 100)]
    reqs += [req(float(t), None, ok=False) for t in range(100, 200)]  # outage: all fail
    reqs += [req(float(t), 2.0) for t in range(200, 220)]  # backlog: slow
    reqs += [req(float(t), 0.1) for t in range(220, 300)]  # normal again
    waiting = [(float(t), 5.0 if 200 <= t < 225 else 0.0) for t in range(0, 300) if
               not 100 <= t < 200]  # fmt: skip
    c = Criterion(window_s=10, baseline_s=60)
    r = recovery(reqs, snaps(waiting), t_fault=100.0, t_ready=200.0, c=c)
    assert r["baseline_ttft_p95_ms"] == pytest.approx(100)
    assert r["baseline_pass_share"] == 1.0  # steady pre-fault windows all pass
    assert r["t_recovered"] == 225.0  # TTFT ok from 220, but waiting > 0 until 225
    # one error inside a later window blocks it
    reqs2 = reqs + [req(232.5, None, ok=False)]
    r2 = recovery(reqs2, snaps(waiting), 100.0, 200.0, c=c)
    assert r2["t_recovered"] == 233.0


def test_recovery_baseline_noise_is_reported():
    # every 20th pre-fault request is slow: baseline p95 barely moves, but a 10 s window with
    # one slow request (of ~10) fails the p95 test -- the share of passing windows shows it
    reqs = [req(float(t), 0.5 if t % 20 == 0 else 0.1) for t in range(0, 100)]
    waiting = [(float(t), 0.0) for t in range(0, 100)]
    r = recovery(reqs, snaps(waiting), 100.0, None, c=Criterion(window_s=10, baseline_s=100))
    assert 0 < r["baseline_pass_share"] < 1
    assert "note" in r  # no Ready time: recovery not evaluated


def test_prometheus_series_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/query_range"
        if "UTIL" in request.url.params["query"]:
            vals = [[100.0, "95"], [105.0, "0"]]
            return httpx.Response(200, json={"data": {"result": [{"values": vals}]}})
        return httpx.Response(200, json={"data": {"result": []}})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://p") as c:
        out = restart.prometheus_series(c, 100, 110)
    assert out["gpu_util"] == [(100.0, 95.0), (105.0, 0.0)]
    assert out["sm_active"] == []


# --- mock server outage end to end ---------------------------------------------------------

# Seconds, not milliseconds: macOS timer coalescing can stall a background process for ~0.4 s
# (see test_open_loop_poisson_schedule), which would blur a shorter outage.
OUTAGE_S = 1.0
# Mechanics only: a stall of the test process inflates mock TTFT many times, so the threshold
# here is loose; the criterion itself is tested on synthetic data above.
CRIT = Criterion(window_s=1.0, ttft_factor=50.0, baseline_s=2.0, step_s=0.05)


async def run_with_outage(tmp_path: Path, retries: int) -> tuple[dict, float, float]:
    server = mock_server.start(ttft=0.02, itl=0.02, max_num_seqs=64)
    port = server.server_address[1]
    sc = Scenario.from_dict(dict(
        name=f"exp5_mock_r{retries}", series="mock-restart", mode="open", rate=30.0,
        duration_s=8.0, num_requests=None, warmup_requests=0, input_len=8, output_len=20,
        retries=retries, backoff_s=0.02, backoff_max_s=0.1, timeout_s=10,
        metrics_interval_s=0.1, cooldown_s=0,
    ))  # fmt: skip
    sc.validate()
    task = asyncio.create_task(execute(sc, f"http://127.0.0.1:{port}", tmp_path, ("r30", 30.0), 1))
    await asyncio.sleep(2.5)
    t_fault = time.time()
    await asyncio.to_thread(server.kill)  # in-flight streams cut, port closed (blocks ~0.5 s)
    await asyncio.sleep(OUTAGE_S)
    new = mock_server.start(port=port, ttft=0.02, itl=0.02, max_num_seqs=64)
    t_ready = time.time()
    try:
        meta = await task
    finally:
        new.shutdown()
        new.server_close()
    return meta, t_fault, t_ready


def write_fault_artifacts(d: Path, t_fault: float, t_ready: float) -> Path:
    """What scripts/fault.sh would leave for a delete, with the mock's times."""
    conditions = [
        {"type": "PodScheduled", "status": "True", "lastTransitionTime": iso(t_fault + 0.1)},
        {"type": "Ready", "status": "True", "lastTransitionTime": iso(t_ready)},
    ]
    running = {"running": {"startedAt": iso(t_fault + 0.2)}}
    pod = {
        "metadata": {"name": "vllm-b", "creationTimestamp": iso(t_fault + 0.05)},
        "status": {"conditions": conditions, "containerStatuses": [{"state": running}]},
    }
    (d / "f.pod_after.json").write_text(json.dumps(pod))
    (d / "f.log_after.txt").write_text(
        f"{iso(t_fault + 0.3)} INFO Loading weights took 0.10 seconds\n"
        f"{iso(t_fault + 0.4)} INFO Model loading took 14.29 GiB memory and 0.20 seconds\n"
        f"{iso(t_ready - 0.05)} INFO:     Application startup complete.\n"
    )
    fault = {
        "mode": "delete",
        "t_fault_unix": t_fault,
        "t_old_pod_gone_unix": t_fault + 0.03,
        "pod_after": "vllm-b",
        "files": {"pod_after": "f.pod_after.json", "log_after": "f.log_after.txt"},
    }
    path = d / "20990101-000000_fault_delete.json"
    path.write_text(json.dumps(fault))
    return path


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


async def test_mock_outage_with_retries_end_to_end(tmp_path):
    meta, t_fault, t_ready = await run_with_outage(tmp_path / "raw", retries=100)
    run = tmp_path / "raw" / "mock-restart" / f"{meta['run_id']}.jsonl"
    recs = load(run)
    retried = [r for r in recs if r["retry_count"] > 0]
    assert retried and all(r["status"] == "ok" for r in retried)
    outcomes = {a["outcome"] for r in retried for a in r["attempts"]}
    assert "connect_error" in outcomes  # port closed during the outage
    # streams cut after the first token are errors, never retried
    mid = [r for r in recs if r["status"] != "ok"]
    assert mid and all(r["attempts"][-1]["mid_stream"] and r["retry_count"] == 0 for r in mid)

    write_fault_artifacts(run.parent, t_fault, t_ready)  # found by time, no --fault needed
    s = restart.analyze(run, None, tmp_path / "agg", tmp_path / "fig", bin_s=0.1,
                        criterion=CRIT)  # fmt: skip
    assert s["fault_file"] == "20990101-000000_fault_delete.json"
    assert s["failed"] == len(mid) and s["requests_retried"] == len(retried)
    assert list(s["phases_rel_s"])[:3] == ["fault", "old_pod_gone", "pod_created"]
    # phases are rounded to 0.1 s
    assert s["phases_rel_s"]["Ready"] == pytest.approx(t_ready - t_fault, abs=0.06)
    assert "model_loaded" in s["phases_rel_s"] and "startup_complete" in s["phases_rel_s"]
    o = s["outage"]
    # in-flight streams are cut at the fault; the server answers again after the outage (loose
    # bounds: a loaded laptop can stall this process for a second)
    assert 0 <= o["t_first_failure"] < o["t_first_token_after"]
    assert OUTAGE_S * 0.8 < o["t_first_token_after"] < OUTAGE_S + 1.5
    assert o["attempts_per_request"] > 1  # retry amplification is visible
    assert s["recovery"].get("recovered_after_ready_s", -1) >= 0, s["recovery"]
    # the new server's counters start a new segment after the outage: never glued across it
    segs = s["preemption_segments"]
    assert any(g["t_end"] <= 0.5 for g in segs)
    assert any(g["t_start"] >= OUTAGE_S * 0.8 for g in segs)
    assert not any(g["t_start"] < 0 and g["t_end"] > OUTAGE_S for g in segs)
    csv_path = tmp_path / "agg" / "mock-restart" / "restart" / f"{meta['run_id']}.timeline.csv"
    rows = csv_path.read_text().splitlines()
    assert rows[0].startswith("t_rel_s,arrivals,attempts,retries")
    assert max(int(r.split(",")[1]) for r in rows[1:]) > 0


async def test_mock_outage_without_retries(tmp_path):
    meta, t_fault, t_ready = await run_with_outage(tmp_path / "raw", retries=0)
    run = tmp_path / "raw" / "mock-restart" / f"{meta['run_id']}.jsonl"
    recs = load(run)
    refused = [r for r in recs if r["attempts"][-1]["outcome"] == "connect_error"]
    # ~30 req/s x 1 s outage, Poisson: well above zero, all failed on the first attempt
    assert len(refused) >= 10 and all(r["retry_count"] == 0 for r in refused)
    s = restart.analyze(run, write_fault_artifacts(run.parent, t_fault, t_ready),
                        tmp_path / "agg", tmp_path / "fig", bin_s=0.1, criterion=CRIT)  # fmt: skip
    assert s["failed_by_outcome"]["connect_error"] == len(refused)
    assert s["requests_retried"] == 0


def test_plot_writes_figure(tmp_path):
    pytest.importorskip("matplotlib")
    reqs = [req(float(t), 0.1) for t in range(0, 20)] + [req(25.0, 3.0, attempts=3)]
    rows = restart.timeline(reqs, snaps([(float(t), 0.0) for t in range(30)]), 10.0, 1.0)
    path = restart.plot(rows, reqs, {"fault": {"t": 10.0}, "Ready": {"t": 24.0}}, 10.0, {},
                        tmp_path / "f.png", 120.0)  # fmt: skip
    assert path.stat().st_size > 1000


# --- scripts/fault.sh against a fake kubectl -------------------------------------------------

FAKE_KUBECTL = r"""
import json, os, sys, time
state = os.environ["FAKE_STATE"]
args = [a for a in sys.argv[1:]]
if args[:2] == ["-n", "vllm"]:
    args = args[2:]
def marker(name):
    p = os.path.join(state, name)
    return float(open(p).read()) if os.path.exists(p) else None
def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
now = time.time()
t_del, t_kill = marker("deleted"), marker("killed")
if args[0] == "delete":
    open(os.path.join(state, "deleted"), "w").write(str(now))
elif args[0] == "exec":
    if "kill -TERM 1" in args[-1]:
        open(os.path.join(state, "killed"), "w").write(str(now))
    elif args[-1] == "/proc/1/cmdline":
        sys.stdout.write("/usr/bin/python3\0/usr/local/bin/vllm\0serve\0Qwen\0")
elif args[0] == "logs":
    print(f"{iso(now - 60)} INFO Model loading took 14.29 GiB memory and 106.1 seconds")
    print(f"{iso(now - 1)} INFO:     Application startup complete.")
elif args[:2] == ["get", "pod"] and "-l" in args:
    if t_del is not None:
        dt = now - t_del
        if dt < 2:
            print(f"vllm-a|{iso(t_del)}|Running|false|0")
            print("vllm-b||Pending||")
        else:
            print(f"vllm-b||Running|{'true' if dt > 4 else 'false'}|0")
    elif t_kill is not None:
        dt = now - t_kill
        print(f"vllm-a||Running|{'true' if dt > 3 else 'false'}|{1 if dt > 1 else 0}")
    else:
        print("vllm-a||Running|true|0")
elif args[:2] == ["get", "pod"] and "jsonpath={.spec.nodeName}" in args:
    print("node-1", end="")
elif args[:2] == ["get", "pod"]:
    t0 = t_del or t_kill or now
    print(json.dumps({"metadata": {"name": args[2], "creationTimestamp": iso(t0 + 1)},
        "status": {"conditions": [{"type": "Ready", "status": "True",
                                   "lastTransitionTime": iso(now)}]}}))
elif args[:2] == ["get", "node"]:
    print(json.dumps({"metadata": {"creationTimestamp": iso(now - 3600)}}))
elif args[:2] == ["get", "events"]:
    print(json.dumps({"items": [{"reason": "Scheduled", "involvedObject": {"name": "vllm-b"},
                                 "firstTimestamp": iso((t_del or now) + 1), "message": "ok"}]}))
else:
    print("{}")
"""


@pytest.mark.parametrize("mode", ["delete", "kill", "observe"])
def test_fault_script_with_fake_kubectl(tmp_path, mode):
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(f"#!{sys.executable}\n{FAKE_KUBECTL}")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_STATE": str(state)}
    out = tmp_path / "raw"
    p = subprocess.run(
        [str(ROOT / "scripts" / "fault.sh"), mode, "--save", str(out), "--after", "0",
         "--timeout", "30"],
        env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert p.returncode == 0, p.stderr + p.stdout
    (fault_json,) = out.glob(f"*_fault_{mode}.json")
    f = json.loads(fault_json.read_text())
    assert not f["timed_out"]
    assert all((out / name).exists() for name in f["files"].values() if name)
    ph = restart.phases(fault_json)
    assert {"Ready", "model_loaded", "startup_complete", "node_created"} <= set(ph)
    if mode == "observe":  # R0: no fault, just the Pod that came up
        assert f["pod_before"] is None and f["t_fault_unix"] is None
        assert f["pod_after"] == "vllm-a" and "fault" not in ph
        return
    assert f["pod_before"] == "vllm-a" and "fault" in ph
    assert f["pod_after"] == ("vllm-b" if mode == "delete" else "vllm-a")
    assert f["t_fault_unix"] < f["t_ready_seen_unix"]
    if mode == "delete":
        assert f["t_fault_unix"] < f["t_old_pod_gone_unix"] < f["t_ready_seen_unix"]
    else:
        assert "vllm" in f["pid1_cmdline"]
