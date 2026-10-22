"""Exp5: one open-loop run with a fault (pod / container restart) -> timeline, phases, recovery.

    uv run python -m analysis.restart results/raw/<series>/<run_id>.jsonl [--plot]
    uv run python -m analysis.restart results/raw/<series>/<run_id>.jsonl \
        --fault results/raw/<series>/<stamp>_fault_delete.json --prometheus http://localhost:9090
    uv run python -m analysis.restart --phases results/raw/<series>/<stamp>_fault_observe.json

Inputs: the run's request records (with per-attempt log, client.stream) and /metrics
snapshots; the fault artifacts written by scripts/fault.sh (fault time, pod JSON, events,
vLLM log with kubelet timestamps). Without --fault the newest *_fault_*.json in the run's
directory whose fault time falls inside the run is used. All times are converted to UTC epoch
seconds and reported relative to the fault (t = 0).

Server counters restart from zero with the new vLLM process: they are never glued into one
series. Gauges (running / waiting / KV usage) are plotted as sampled, with a gap during the
outage; counters are reported per segment (a segment ends at a gap or a decrease).

Recovery (EXP_006, Definitions): the first 30 s window [a, a + 30) with a >= Ready of the
new pod such that every request that arrived in it succeeded, server waiting = 0 in every
snapshot of it, and TTFT p95 of those requests <= 1.2 x the pre-fault TTFT p95. The same test
on pre-fault windows (baseline_pass_share) shows how noisy the criterion is at this rate.

Outputs: results/aggregated/<series>/restart/<run_id>.{timeline.csv,summary.json,summary.md}
and, with --plot, results/figures/<series>/<run_id>_restart.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from analysis.aggregate import load_records, load_snapshots
from client.records import RequestRecord, percentile, ttft
from client.server_metrics import metric


@dataclass(frozen=True)
class Criterion:
    """Recovery criterion (EXP_006). Defaults are the pre-registered values."""

    window_s: float = 30.0  # continuous window after Ready
    ttft_factor: float = 1.2  # TTFT p95 in the window <= factor x pre-fault TTFT p95
    baseline_s: float = 120.0  # pre-fault baseline: requests that arrived in this span
    step_s: float = 1.0  # slide of the window


PREREGISTERED = Criterion()

# vLLM 0.30 startup log lines (kubectl logs --timestamps) -> phase name
LOG_MARKERS = {
    "download_done": re.compile(r"Time spent downloading weights for .*?: ([\d.]+) seconds"),
    "weights_loaded": re.compile(r"Loading weights took ([\d.]+) seconds"),
    "model_loaded": re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
    "engine_ready": re.compile(r"init engine .*? took ([\d.]+) seconds"),
    "startup_complete": re.compile(r"Application startup complete"),
}

PROM_QUERIES = {
    "gpu_util": "max(DCGM_FI_DEV_GPU_UTIL)",
    "sm_active": "max(DCGM_FI_PROF_SM_ACTIVE)",
    "fb_used_mib": "max(DCGM_FI_DEV_FB_USED)",
    "power_w": "max(DCGM_FI_DEV_POWER_USAGE)",
}


def parse_ts(s: str | None) -> float | None:
    """RFC 3339 (Kubernetes, kubelet log prefix; up to ns precision) -> epoch seconds."""
    if not s:
        return None
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:\d\d)$", s.strip())
    if not m:
        return None
    frac = (m.group(2) or ".0")[:7]  # microseconds are enough
    tz = "+00:00" if m.group(3) == "Z" else m.group(3)
    return datetime.fromisoformat(m.group(1) + frac + tz).timestamp()


# --- requests on the wall clock --------------------------------------------------------------


@dataclass
class Req:
    """One request on the UTC epoch axis."""

    arrival: float  # first attempt started (latency counts from here)
    end: float  # last attempt ended
    ok: bool
    phase: str
    ttft: float | None
    first_token: float | None
    outcome: str  # outcome of the last attempt
    mid_stream: bool
    attempts: list[dict]  # start / end on the epoch axis


def to_wall(r: RequestRecord) -> Req:
    off = r.started_at_unix - r.started_at
    atts = r.attempts or [  # records written before attempts were logged: one attempt
        {
            "start": r.started_at,
            "end": r.completed_at or r.started_at,
            "outcome": "ok" if r.status == "ok" else r.status,
            "mid_stream": r.first_token_at is not None,
            "retry_in_s": None,
        }
    ]
    atts = [{**a, "start": a["start"] + off, "end": a["end"] + off} for a in atts]
    last = atts[-1]
    return Req(
        arrival=r.started_at + off,
        end=last["end"],
        ok=r.status == "ok",
        phase=r.phase,
        ttft=ttft(r),
        first_token=None if r.first_token_at is None else r.first_token_at + off,
        outcome=last["outcome"],
        mid_stream=bool(last.get("mid_stream")),
        attempts=atts,
    )


def snap_series(snaps: list[dict], name: str) -> list[tuple[float, float]]:
    out = []
    for s in snaps:
        v = metric(s["samples"], name)
        if v is not None:
            out.append((s["t_unix"], v))
    return out


def snapshot_interval(series: list[tuple[float, float]]) -> float:
    """Median spacing of /metrics snapshots (1 s in the scenarios)."""
    gaps = sorted(b - a for (a, _), (b, _) in zip(series, series[1:], strict=False))
    return gaps[len(gaps) // 2] if gaps else 1.0


def segments(series: list[tuple[float, float]], max_gap_s: float) -> list[dict]:
    """Split a counter at gaps and decreases (process restart): delta per segment, not glued."""
    out: list[dict] = []
    for t, v in series:
        if out and t - out[-1]["t_end"] <= max_gap_s and v >= out[-1]["v_end"]:
            out[-1].update(t_end=t, v_end=v)
        else:
            out.append({"t_start": t, "t_end": t, "v_start": v, "v_end": v})
    return [{**s, "delta": s["v_end"] - s["v_start"]} for s in out]


# --- fault artifacts -------------------------------------------------------------------------


def find_fault(run_path: Path, t0: float, t1: float) -> Path | None:
    hits = []
    for p in run_path.parent.glob("*_fault_*.json"):
        f = json.loads(p.read_text())
        t = f.get("t_fault_unix")
        if t is not None and t0 <= t <= t1:
            hits.append((t, p))
    return max(hits)[1] if hits else None


def log_markers(log_path: Path) -> dict[str, dict]:
    """First occurrence of each startup marker in a `kubectl logs --timestamps` file."""
    out: dict[str, dict] = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text(errors="replace").splitlines():
        ts, _, msg = line.partition(" ")
        t = parse_ts(ts)
        if t is None:
            continue
        out.setdefault("log_first_line", {"t": t})
        for name, rx in LOG_MARKERS.items():
            if name not in out and (m := rx.search(msg)):
                out[name] = {"t": t, "value_s": float(m.group(1)) if m.groups() else None}
    return out


def phases(fault_path: Path) -> dict[str, dict]:
    """Phase timestamps (epoch s) from scripts/fault.sh artifacts; missing ones are skipped."""
    f = json.loads(fault_path.read_text())
    base = fault_path.parent
    ph: dict[str, dict] = {}

    def put(name: str, t: float | None, **kw) -> None:
        if t is not None:
            ph[name] = {"t": t, **kw}

    put("fault", f.get("t_fault_unix"))
    put("old_pod_gone", f.get("t_old_pod_gone_unix"))
    pod_file = base / f["files"]["pod_after"] if f.get("files", {}).get("pod_after") else None
    if pod_file and pod_file.exists():
        pod = json.loads(pod_file.read_text())
        put("pod_created", parse_ts(pod["metadata"].get("creationTimestamp")))
        for c in pod.get("status", {}).get("conditions", []):
            if c.get("status") == "True":
                put(c["type"], parse_ts(c.get("lastTransitionTime")))
        for cs in pod.get("status", {}).get("containerStatuses", []):
            term = (cs.get("lastState") or {}).get("terminated") or {}
            put("container_exited", parse_ts(term.get("finishedAt")),
                exit_code=term.get("exitCode"))  # fmt: skip
            put("container_started", parse_ts(((cs.get("state") or {}).get("running") or {})
                                              .get("startedAt")))  # fmt: skip
    node_file = base / f["files"]["node"] if f.get("files", {}).get("node") else None
    if node_file and node_file.exists():
        node = json.loads(node_file.read_text())
        put("node_created", parse_ts(node["metadata"].get("creationTimestamp")))
    ev_file = base / f["files"]["events"] if f.get("files", {}).get("events") else None
    if ev_file and ev_file.exists():
        pod_name = f.get("pod_after")
        for e in json.loads(ev_file.read_text()).get("items", []):
            if pod_name and e.get("involvedObject", {}).get("name") != pod_name:
                continue
            reason = e.get("reason")
            t = parse_ts(e.get("eventTime") or e.get("firstTimestamp"))
            if reason in ("TriggeredScaleUp", "Scheduled", "Pulling", "Pulled", "BackOff"):
                if f"event_{reason}" not in ph:
                    put(f"event_{reason}", t, message=(e.get("message") or "")[:200])
    if f.get("files", {}).get("log_after"):
        for name, v in log_markers(base / f["files"]["log_after"]).items():
            ph.setdefault(name, v)
    return dict(sorted(ph.items(), key=lambda kv: kv[1]["t"]))


# --- timeline and recovery -------------------------------------------------------------------


def cover_counts(intervals: list[tuple[float, float]], start: float, bin_s: float,
                 n: int) -> list[int]:  # fmt: skip
    """How many [a, b) intervals cover the centre of each bin (difference array, O(n + k))."""
    diff = [0] * (n + 1)
    for a, b in intervals:
        lo = max(0, math.ceil((a - start) / bin_s - 0.5))
        hi = min(n, math.ceil((b - start) / bin_s - 0.5))  # first bin whose centre is >= b
        if lo < hi:
            diff[lo] += 1
            diff[hi] -= 1
    out, acc = [], 0
    for d in diff[:n]:
        acc += d
        out.append(acc)
    return out


def _p(values: list[float], q: float) -> float | None:
    return percentile(values, q) if values else None


def timeline(reqs: list[Req], snaps: list[dict], t_ref: float, bin_s: float) -> list[dict]:
    if not reqs:
        return []
    t_lo = min(r.arrival for r in reqs)
    t_hi = max(r.end for r in reqs)
    if snaps:
        t_hi = max(t_hi, snaps[-1]["t_unix"])
    n = int(math.ceil((t_hi - t_lo) / bin_s)) + 1
    start = t_lo
    rows = [{"t_rel_s": round(start + i * bin_s - t_ref, 3), "arrivals": 0, "attempts": 0,
             "retries": 0, "attempts_failed": 0, "ok": 0, "failed": 0, "failed_mid_stream": 0,
             "_ttft": []} for i in range(n)]  # fmt: skip
    fail_kinds: Counter = Counter()

    def idx(t: float) -> int:
        return min(n - 1, max(0, int((t - start) // bin_s)))

    for r in reqs:
        b = rows[idx(r.arrival)]
        b["arrivals"] += 1
        if r.ok and r.ttft is not None:
            b["_ttft"].append(r.ttft)
        for k, a in enumerate(r.attempts):
            rows[idx(a["start"])]["attempts"] += 1
            if k:
                rows[idx(a["start"])]["retries"] += 1
            if a["outcome"] != "ok":
                rows[idx(a["end"])]["attempts_failed"] += 1
                key = f"attempts_failed_{a['outcome']}"
                rows[idx(a["end"])][key] = rows[idx(a["end"])].get(key, 0) + 1
                fail_kinds[key] += 1
        e = rows[idx(r.end)]
        if r.ok:
            e["ok"] += 1
        else:
            e["failed"] += 1
            e["failed_mid_stream"] += r.mid_stream
    in_flight = cover_counts([(r.arrival, r.end) for r in reqs], start, bin_s, n)
    in_backoff = cover_counts([(a["end"], b["start"]) for r in reqs
                               for a, b in zip(r.attempts, r.attempts[1:], strict=False)],
                              start, bin_s, n)  # fmt: skip
    for i, row in enumerate(rows):
        row["client_in_flight"] = in_flight[i]
        row["client_in_backoff"] = in_backoff[i]
        ts = row.pop("_ttft")
        row["ttft_p50_ms"] = None if not ts else round(_p(ts, 50) * 1000, 1)
        row["ttft_p95_ms"] = None if not ts else round(_p(ts, 95) * 1000, 1)
        for k in fail_kinds:
            row.setdefault(k, 0)
    for name, col in (("vllm:num_requests_running", "server_running"),
                      ("vllm:num_requests_waiting", "server_waiting"),
                      ("vllm:kv_cache_usage_perc", "server_kv_usage")):  # fmt: skip
        acc: dict[int, list[float]] = {}
        for t, v in snap_series(snaps, name):
            acc.setdefault(idx(t), []).append(v)
        for i, row in enumerate(rows):
            row[col] = round(max(acc[i]), 4) if i in acc else None
    return rows


def ttft_p95(reqs: list[Req]) -> float | None:
    return _p([r.ttft for r in reqs if r.ok and r.ttft is not None], 95)


def window_ok(reqs: list[Req], waits: list[tuple[float, float]], a: float, b: float,
              limit_s: float, min_snaps: int) -> bool:  # fmt: skip
    """All requests that arrived in [a, b) succeeded, waiting = 0 in every snapshot, and their
    TTFT p95 <= limit_s."""
    inside = [r for r in reqs if a <= r.arrival < b]
    if not inside or not all(r.ok for r in inside):
        return False
    w = [v for t, v in waits if a <= t < b]
    if len(w) < min_snaps or any(v > 0 for v in w):
        return False
    p95 = ttft_p95(inside)
    return p95 is not None and p95 <= limit_s


def recovery(reqs: list[Req], snaps: list[dict], t_fault: float, t_ready: float | None,
             c: Criterion = PREREGISTERED) -> dict:  # fmt: skip
    measured = [r for r in reqs if r.phase == "measure"]
    pre = [r for r in measured
           if t_fault - c.baseline_s <= r.arrival < t_fault and r.end < t_fault]  # fmt: skip
    base = ttft_p95(pre)
    waits = snap_series(snaps, "vllm:num_requests_waiting")
    interval = snapshot_interval(waits)
    min_snaps = max(1, int(c.window_s / interval / 2))  # at least half of the expected snapshots
    out: dict = {
        "baseline_ttft_p95_ms": None if base is None else base * 1000,
        "baseline_requests": len(pre),
        "window_s": c.window_s,
        "ttft_factor": c.ttft_factor,
    }
    if base is None:
        return out
    limit = c.ttft_factor * base
    # the same test on pre-fault windows: share that passes = how noisy the criterion is
    pre_start = min((r.arrival for r in pre), default=t_fault)
    n_pre = int((t_fault - c.window_s - pre_start) / c.step_s) + 1
    passed = [window_ok(measured, waits, a, a + c.window_s, limit, min_snaps)
              for a in (pre_start + k * c.step_s for k in range(max(0, n_pre)))]  # fmt: skip
    out["baseline_windows"] = len(passed)
    out["baseline_pass_share"] = sum(passed) / len(passed) if passed else None
    if t_ready is None:
        out["note"] = "no Ready time of the new pod: recovery not evaluated"
        return out
    last = max(r.arrival for r in measured)
    a = t_ready
    while a + c.window_s <= last:
        if window_ok(measured, waits, a, a + c.window_s, limit, min_snaps):
            out["t_recovered"] = a
            break
        a += c.step_s
    else:
        out["note"] = "no window met the criterion before the last arrival"
    return out


def outage(reqs: list[Req], t_fault: float) -> dict:
    """Client view: first failed attempt at/after the fault -> first token of a request whose
    last attempt started after that (the server answers again)."""
    failed = [a for r in reqs for a in r.attempts if a["outcome"] != "ok" and a["end"] >= t_fault]
    if not failed:
        return {}
    t0 = min(a["end"] for a in failed)
    firsts = [r.first_token for r in reqs
              if r.first_token is not None and r.attempts[-1]["start"] >= t0]  # fmt: skip
    t1 = min(firsts) if firsts else None
    hit = [r for r in reqs if t0 <= r.arrival < (t1 or math.inf)]
    retried_ok = [r for r in hit if r.ok]
    out = {"t_first_failure": t0, "t_first_token_after": t1,
           "requests_arrived_during": len(hit),
           "attempts_during": sum(t0 <= a["start"] < (t1 or math.inf)
                                  for r in reqs for a in r.attempts)}  # fmt: skip
    if hit:
        out["attempts_per_request"] = out["attempts_during"] / len(hit)
    if retried_ok:
        out["last_backlog_first_token"] = max(r.first_token for r in retried_ok if r.first_token)
    return out


def summarize_run(reqs: list[Req], snaps: list[dict], ph: dict[str, dict], t_fault: float,
                  c: Criterion = PREREGISTERED) -> dict:  # fmt: skip
    measured = [r for r in reqs if r.phase == "measure"]
    failed = [r for r in measured if not r.ok]
    t_ready = ph.get("Ready", {}).get("t")
    rel = {k: round(v["t"] - t_fault, 1) for k, v in ph.items()}
    s: dict = {
        "requests": len(measured),
        "ok": len(measured) - len(failed),
        "failed": len(failed),
        "failed_by_outcome": dict(Counter(
            f"{r.outcome}{'_mid_stream' if r.mid_stream else ''}" for r in failed)),
        "attempts": sum(len(r.attempts) for r in measured),
        "attempts_failed_by_outcome": dict(Counter(
            a["outcome"] for r in measured for a in r.attempts if a["outcome"] != "ok")),
        "requests_retried": sum(len(r.attempts) > 1 for r in measured),
        "max_attempts_per_request": max((len(r.attempts) for r in measured), default=0),
        "phases_rel_s": rel,
    }  # fmt: skip
    o = outage(reqs, t_fault)
    s["outage"] = {k: (round(v - t_fault, 1) if k.startswith("t_") or k.startswith("last_")
                       else v) for k, v in o.items()}  # fmt: skip
    if o.get("t_first_token_after"):
        s["outage"]["duration_s"] = round(o["t_first_token_after"] - o["t_first_failure"], 1)
    rec = recovery(reqs, snaps, t_fault, t_ready, c)
    if "t_recovered" in rec:
        rec["recovered_after_fault_s"] = round(rec["t_recovered"] - t_fault, 1)
        rec["recovered_after_ready_s"] = round(rec["t_recovered"] - t_ready, 1)
        del rec["t_recovered"]
    s["recovery"] = rec
    if t_ready is not None:
        after = [s_ for s_ in snaps if s_["t_unix"] >= t_ready]
        for name, key in (("vllm:num_requests_running", "running"),
                          ("vllm:num_requests_waiting", "waiting"),
                          ("vllm:kv_cache_usage_perc", "kv_usage")):  # fmt: skip
            vals = [v for _, v in snap_series(after, name)]
            s[f"server_{key}_max_after_ready"] = max(vals) if vals else None
    waits = snap_series(snaps, "vllm:num_requests_waiting")
    interval = snapshot_interval(waits)
    s["preemption_segments"] = [
        {"t_start": round(g["t_start"] - t_fault, 1), "t_end": round(g["t_end"] - t_fault, 1),
         "delta": g["delta"]}
        for g in segments(snap_series(snaps, "vllm:num_preemptions_total"), 5 * interval)
    ]  # fmt: skip
    return s


# --- Prometheus (optional) -------------------------------------------------------------------


def prometheus_series(client: httpx.Client, start: float, end: float,
                      step_s: float = 5.0) -> dict[str, list[tuple[float, float]]]:  # fmt: skip
    """GPU metrics over the run from the cluster's Prometheus (port-forward), one series each."""
    out: dict[str, list[tuple[float, float]]] = {}
    for key, q in PROM_QUERIES.items():
        r = client.get("/api/v1/query_range",
                       params={"query": q, "start": start, "end": end, "step": step_s})  # fmt: skip
        r.raise_for_status()
        res = r.json().get("data", {}).get("result", [])
        out[key] = [(float(t), float(v)) for t, v in res[0]["values"]] if res else []
    return out


# --- output ----------------------------------------------------------------------------------


def markdown(run_id: str, s: dict) -> str:
    lines = [f"# {run_id}", "", "| phase | t from fault, s |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["phases_rel_s"].items()]
    lines += ["", "| metric | value |", "|---|---|"]
    for k in (
        "requests",
        "ok",
        "failed",
        "failed_by_outcome",
        "attempts",
        "attempts_failed_by_outcome",
        "requests_retried",
        "max_attempts_per_request",
    ):
        lines.append(f"| {k} | {s[k]} |")
    for group in ("outage", "recovery"):
        for k, v in s[group].items():
            lines.append(f"| {group}.{k} | {round(v, 3) if isinstance(v, float) else v} |")
    for k in (
        "server_running_max_after_ready",
        "server_waiting_max_after_ready",
        "server_kv_usage_max_after_ready",
    ):
        if k in s:
            lines.append(f"| {k} | {s[k]} |")
    lines.append(f"| preemption_segments | {s['preemption_segments']} |")
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], reqs: list[Req], ph: dict, t_fault: float, gpu: dict,
         path: Path, limit_ms: float | None) -> Path:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t_rel_s"] for r in rows]
    n_panels = 5 if gpu else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.4 * n_panels), sharex=True)
    ax = axes[0]
    ax.plot(t, [r["arrivals"] for r in rows], label="arrivals", lw=1)
    ax.plot(t, [r["ok"] for r in rows], label="ok", lw=1)
    ax.plot(t, [r["failed"] for r in rows], label="failed (final)", lw=1, color="red")
    ax.plot(t, [r["retries"] for r in rows], label="retry attempts", lw=1, color="gray")
    ax.set_ylabel("per bin")
    ax.legend(fontsize=7, ncol=4)
    ax = axes[1]
    ax.plot(t, [r["client_in_flight"] for r in rows], label="client in flight", lw=1)
    ax.plot(t, [r["client_in_backoff"] for r in rows], label="client in backoff", lw=1)
    ax.plot(t, [r["server_running"] for r in rows], label="server running", lw=1)
    ax.plot(t, [r["server_waiting"] for r in rows], label="server waiting", lw=1)
    ax.set_ylabel("requests")
    ax.legend(fontsize=7, ncol=4)
    ax = axes[2]
    ok = [r for r in reqs if r.ok and r.ttft is not None]
    ax.scatter([r.arrival - t_fault for r in ok], [r.ttft * 1000 for r in ok], s=3)
    if limit_ms:
        ax.axhline(limit_ms, color="gray", ls="--", lw=0.8,
                   label="recovery limit: factor x pre-fault p95")  # fmt: skip
        ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("TTFT, ms (by arrival)")
    ax = axes[3]
    ax.plot(t, [r["server_kv_usage"] for r in rows], lw=1)
    ax.set_ylabel("KV usage")
    if gpu:
        ax = axes[4]
        for key in ("gpu_util", "sm_active"):
            pts = gpu.get(key) or []
            scale = 100 if key == "sm_active" else 1
            ax.plot([a - t_fault for a, _ in pts], [v * scale for _, v in pts], label=key, lw=1)
        ax.set_ylabel("GPU, %")
        ax.legend(fontsize=7)
    for name, v in ph.items():
        x = v["t"] - t_fault
        if not t[0] <= x <= t[-1]:  # e.g. PodScheduled of the same pod long before (R2)
            continue
        for a in axes:
            a.axvline(x, color="k", lw=0.5, alpha=0.4)
        axes[0].annotate(name, (x, 1), xycoords=("data", "axes fraction"), rotation=90,
                         fontsize=6, va="top")  # fmt: skip
    axes[-1].set_xlabel("s from fault")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def analyze(run_path: Path, fault_path: Path | None, agg_root: Path, fig_root: Path,
            bin_s: float = 1.0, prometheus: str | None = None, make_plot: bool = False,
            criterion: Criterion = PREREGISTERED) -> dict:  # fmt: skip
    run_id = run_path.name.removesuffix(".jsonl")
    series = run_path.parent.name
    reqs = [to_wall(r) for r in load_records(run_path)]
    snaps = load_snapshots(run_path.with_name(f"{run_id}.server.jsonl"))
    t0, t1 = min(r.arrival for r in reqs), max(r.end for r in reqs)
    fault_path = fault_path or find_fault(run_path, t0, t1)
    ph = phases(fault_path) if fault_path else {}
    t_fault = ph.get("fault", {}).get("t")
    if t_fault is None:  # no fault artifacts: anchor on the first failed attempt
        fails = [a["end"] for r in reqs for a in r.attempts if a["outcome"] != "ok"]
        t_fault = min(fails) if fails else t0
    s = summarize_run(reqs, snaps, ph, t_fault, criterion)
    s["run_id"], s["fault_file"] = run_id, None if fault_path is None else fault_path.name
    rows = timeline(reqs, snaps, t_fault, bin_s)
    gpu = {}
    if prometheus:
        with httpx.Client(base_url=prometheus, timeout=30) as c:
            gpu = prometheus_series(c, t0, t1)
        for key, pts in gpu.items():
            for row in rows:
                tt = row["t_rel_s"] + t_fault
                vals = [v for a, v in pts if tt <= a < tt + bin_s]
                row[key] = vals[0] if vals else None
    out = agg_root / series / "restart"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"{run_id}.timeline.csv", rows)
    (out / f"{run_id}.summary.json").write_text(json.dumps(s, indent=2) + "\n")
    (out / f"{run_id}.summary.md").write_text(markdown(run_id, s))
    if make_plot:
        base = s["recovery"].get("baseline_ttft_p95_ms")
        limit = None if base is None else criterion.ttft_factor * base
        fig = fig_root / series / f"{run_id}_restart.png"
        s["figure"] = str(plot(rows, reqs, ph, t_fault, gpu, fig, limit))
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run", nargs="?", type=Path, help="results/raw/<series>/<run_id>.jsonl")
    p.add_argument("--fault", type=Path, help="fault.sh JSON (default: found by time)")
    p.add_argument("--phases", type=Path, help="only print phases of a fault.sh JSON (R0)")
    p.add_argument("--bin", type=float, default=1.0, help="timeline bin, s")
    p.add_argument("--prometheus", help="Prometheus URL for GPU metrics, e.g. port-forward")
    p.add_argument("--plot", action="store_true", help="write the figure (analysis extra)")
    p.add_argument(
        "--window",
        type=float,
        default=Criterion.window_s,
        help="recovery window, s (not the pre-registered 30: report as a deviation)",
    )
    p.add_argument(
        "--ttft-factor",
        type=float,
        default=Criterion.ttft_factor,
        help="recovery TTFT p95 limit vs pre-fault p95 (pre-registered 1.2)",
    )
    p.add_argument("--agg-root", type=Path, default=Path("results/aggregated"))
    p.add_argument("--fig-root", type=Path, default=Path("results/figures"))
    a = p.parse_args()
    if a.phases:
        ph = phases(a.phases)
        t_ref = min(v["t"] for v in ph.values()) if ph else 0
        for k, v in ph.items():
            print(f"{k:24s} +{v['t'] - t_ref:8.1f} s  {v.get('value_s') or v.get('message') or ''}")
        return
    if not a.run:
        p.error("run path required (or --phases)")
    crit = Criterion(window_s=a.window, ttft_factor=a.ttft_factor)
    s = analyze(a.run, a.fault, a.agg_root, a.fig_root, a.bin, a.prometheus, a.plot, crit)
    print(markdown(s["run_id"], s))


if __name__ == "__main__":
    main()
