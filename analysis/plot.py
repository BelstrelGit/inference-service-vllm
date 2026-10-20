"""Figures from aggregated runs (needs the `analysis` extra: uv sync --extra analysis).

    uv run --extra analysis python -m analysis.plot sweep results/aggregated/<series>
    uv run --extra analysis python -m analysis.plot timeseries \
        results/raw/<series>/<run_id>.server.jsonl
    uv run --extra analysis python -m analysis.plot context results/aggregated/<series>
    uv run --extra analysis python -m analysis.plot output results/aggregated/<series>
    uv run --extra analysis python -m analysis.plot kv results/aggregated/<series> ...

sweep       median over repeats, error bars = min..max across repeats:
              throughput vs level, TTFT p50/p95 vs level, TPOT p50/p95 vs level,
              latency-throughput curve (TTFT p95 / E2E p95 vs output tok/s),
              server queue / KV usage vs level
timeseries  running / waiting requests and KV usage over one run (/metrics snapshots)
context     Exp2, closed-loop runs of all prompt lengths in the series:
              c=1: TTFT p50 / server prefill and TPOT p50 vs prompt length;
              running mean vs concurrency per prompt length, dashed = KV pool / (P + out/2);
              KV tokens per running sequence vs expected P + out/2 (diagonal = match)
output      Exp3, closed-loop runs with P=512 for every output length in the series:
              c=1: TPOT p50 and E2E p50 vs out, x = TTFT p50 + (out - 1) * TPOT p50;
              vs concurrency per out: steady tok/s, steady req/s (dashed: prefill-only ceiling
              1 / server prefill at c=1), TPOT p50, running mean (dashed: KV pool / (P + out/2)),
              KV usage mean, preemptions per 100 measured requests;
              TPOT p50 vs the indirect estimate d(n) + n * prefill / out, d(n) from out=128 runs
              without preemption, held flat outside their range of n
              (diagonal = match; an estimate, not a measurement)
kv          Exp4, closed-loop runs with P=2048 across several series (one per KV pool size /
            KV dtype), output in results/figures/kv/:
              vs concurrency per series: running mean (dashed: K / (P + out/2)), steady tok/s,
              TPOT p50, TTFT p50, KV usage mean, preemptions per 100 measured requests;
              vs pool size K: plateau running (max over levels; line K / (P + out/2), dotted
              0.93 x) and max steady tok/s
Output: results/figures/<series>/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from client.server_metrics import metric  # noqa: E402


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_runs(agg_dir: Path) -> list[dict]:
    with (agg_dir / "runs.csv").open() as f:
        return list(csv.DictReader(f))


def by_level(runs: list[dict], col: str) -> tuple[list[float], list[float], list[list[float]]]:
    groups: dict[float, list[float]] = {}
    for r in runs:
        x, y = _num(r["value"]), _num(r.get(col))
        if x is not None and y is not None:
            groups.setdefault(x, []).append(y)
    xs = sorted(groups)
    med = [statistics.median(groups[x]) for x in xs]
    err = [[m - min(groups[x]) for x, m in zip(xs, med, strict=True)],
           [max(groups[x]) - m for x, m in zip(xs, med, strict=True)]]  # fmt: skip
    return xs, med, err


def _level_axis(ax, runs: list[dict]) -> None:
    mode = runs[0]["mode"]
    ax.set_xlabel({"closed": "concurrency", "open": "arrival rate, req/s"}.get(mode, "level"))
    if mode == "closed":
        ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)


def sweep(agg_dir: Path, out_dir: Path) -> list[Path]:
    """One set of figures per scenario; with several scenarios also overlays of the headline
    metrics (e.g. exp1_pilot vs exp1_pilot_stagger)."""
    by_scenario: dict[str, list[dict]] = {}
    for r in load_runs(agg_dir):
        by_scenario.setdefault(r["scenario"], []).append(r)
    paths = []
    for runs in by_scenario.values():
        paths += _sweep_one(agg_dir, out_dir, runs)
    if len(by_scenario) > 1:
        paths += _compare(agg_dir, out_dir, by_scenario)
    return paths


def _compare(agg_dir: Path, out_dir: Path, by_scenario: dict[str, list[dict]]) -> list[Path]:
    paths = []
    for name, col, ylabel in (
        ("throughput", "output_token_throughput", "output tok/s"),
        ("steady_throughput", "steady_output_tok_s", "steady output tok/s"),
        ("ttft_p50", "ttft_p50_ms", "TTFT p50, ms"),
        ("ttft_p95", "ttft_p95_ms", "TTFT p95, ms"),
        ("tpot_p50", "tpot_p50_ms", "TPOT p50, ms"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for scenario, runs in by_scenario.items():
            xs, med, err = by_level(runs, col)
            if xs:
                ax.errorbar(xs, med, yerr=err, marker="o", capsize=3, label=scenario)
        _level_axis(ax, next(iter(by_scenario.values())))
        if col.startswith("ttft"):
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.suptitle(f"{agg_dir.name}: {', '.join(by_scenario)}", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"compare_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def _sweep_one(agg_dir: Path, out_dir: Path, runs: list[dict]) -> list[Path]:
    r0 = runs[0]
    title = f"{agg_dir.name} ({r0['scenario']}, in={r0['input_len']} out={r0['output_len']})"
    paths = []

    def save(fig, name: str) -> None:
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        p = out_dir / f"{runs[0]['scenario']}_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    for name, cols, ylabel in (
        (
            "throughput",
            [("output_token_throughput", "output tok/s"), ("steady_output_tok_s", "steady")],
            "tokens/s",
        ),
        ("ttft", [("ttft_p50_ms", "TTFT p50"), ("ttft_p95_ms", "TTFT p95")], "ms"),
        ("tpot", [("tpot_p50_ms", "TPOT p50"), ("tpot_p95_ms", "TPOT p95")], "ms"),
        (
            "server_queue",
            [("server_waiting_max", "waiting max"), ("server_running_max", "running max")],
            "requests",
        ),
        ("kv_usage", [("server_kv_usage_max", "KV usage max")], "fraction"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for col, label in cols:
            xs, med, err = by_level(runs, col)
            if xs:
                ax.errorbar(xs, med, yerr=err, marker="o", capsize=3, label=label)
        _level_axis(ax, runs)
        ax.set_ylabel(ylabel)
        ax.legend()
        save(fig, name)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs_t, thr, _ = by_level(runs, "output_token_throughput")
    for col, label in (("ttft_p95_ms", "TTFT p95"), ("e2e_p95_ms", "E2E p95")):
        xs, lat, _ = by_level(runs, col)
        pts = [(t, latency) for x, t in zip(xs_t, thr, strict=True)
               for x2, latency in zip(xs, lat, strict=True) if x == x2]  # fmt: skip
        if pts:
            ax.plot(*zip(*pts, strict=True), marker="o", label=label)
    ax.set_xlabel("output tokens/s")
    ax.set_ylabel("ms")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "latency_vs_throughput")
    return paths


def _median_by(runs: list[dict], key, col: str) -> dict:
    groups: dict = {}
    for r in runs:
        k, y = key(r), _num(r.get(col))
        if k is not None and y is not None:
            groups.setdefault(k, []).append(y)
    return {k: statistics.median(v) for k, v in groups.items()}


def context(agg_dir: Path, out_dir: Path) -> list[Path]:
    runs = [r for r in load_runs(agg_dir) if r["mode"] == "closed" and _num(r.get("input_len"))]
    plen = lambda r: int(float(r["input_len"]))  # noqa: E731
    paths = []

    def save(fig, name: str) -> None:
        fig.suptitle(f"{agg_dir.name}: context length", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"context_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    c1 = [r for r in runs if _num(r["value"]) == 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for col, label in (("ttft_p50_ms", "TTFT p50"), ("server_prefill_mean_ms", "server prefill")):
        m = _median_by(c1, plen, col)
        ax1.plot(sorted(m), [m[k] for k in sorted(m)], marker="o", label=label)
    m = _median_by(c1, plen, "tpot_p50_ms")
    ax2.plot(sorted(m), [m[k] for k in sorted(m)], marker="o", label="TPOT p50")
    for ax, ylabel in ((ax1, "ms"), (ax2, "ms")):
        ax.set_xlabel("prompt tokens (c=1)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    ax2.set_ylim(bottom=0)
    save(fig, "c1")

    fig, ax = plt.subplots(figsize=(6, 4))
    for P in sorted({plen(r) for r in runs}):
        sub = [r for r in runs if plen(r) == P]
        if len({r["value"] for r in sub}) < 2:
            continue
        xs, med, err = by_level(sub, "server_running_mean")
        line = ax.errorbar(xs, med, yerr=err, marker="o", capsize=3, label=f"P={P}")
        cap = _median_by(sub, lambda r: 0, "server_kv_capacity_tokens").get(0)
        out = _median_by(sub, lambda r: 0, "output_len").get(0)
        if cap and out:
            ax.axhline(cap / (P + out / 2), ls="--", lw=1, color=line[0].get_color())
    ax.set_xlabel("concurrency")
    ax.set_xscale("log", base=2)
    ax.set_ylabel("running mean (dashed: KV pool / (P + out/2))")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "running")

    fig, ax = plt.subplots(figsize=(5, 5))
    for P in sorted({plen(r) for r in runs}):
        sub = [r for r in runs if plen(r) == P and (_num(r.get("server_running_mean")) or 0) >= 0.5]
        ys = [_num(r.get("server_kv_tokens_per_running")) for r in sub]
        pts = [(P + _num(r["output_len"]) / 2, y) for r, y in zip(sub, ys, strict=True) if y]
        if pts:
            ax.scatter(*zip(*pts, strict=True), label=f"P={P}", s=18)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--")
    ax.set_xlabel("expected tokens per sequence, P + out/2")
    ax.set_ylabel("KV tokens / running (measured)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "kv_accounting")
    return paths


def _preemptions_per_100(r: dict) -> float | None:
    # Per measured request rather than per run: runs differ in window and request count.
    pre, req = _num(r.get("server_preemptions")), _num(r.get("requests"))
    return 100 * pre / req if pre is not None and req else None


def output(agg_dir: Path, out_dir: Path, prompt_len: int = 512) -> list[Path]:
    runs = [r for r in load_runs(agg_dir)
            if r["mode"] == "closed" and _num(r.get("input_len")) == prompt_len]  # fmt: skip
    olen = lambda r: int(float(r["output_len"]))  # noqa: E731
    for r in runs:
        tok_s = _num(r.get("steady_output_tok_s"))
        r["steady_req_s"] = tok_s / olen(r) if tok_s is not None else None
        r["preemptions_per_100_req"] = _preemptions_per_100(r)
    lengths = sorted({olen(r) for r in runs})
    c1 = [r for r in runs if _num(r["value"]) == 1]
    prefill_ms = _median_by(c1, lambda r: 0, "server_prefill_mean_ms").get(0)
    cap = _median_by(runs, lambda r: 0, "server_kv_capacity_tokens").get(0)
    paths = []

    def save(fig, name: str) -> None:
        fig.suptitle(f"{agg_dir.name}: output length, P={prompt_len}", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"output_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    tpot, e2e, ttft = (
        _median_by(c1, olen, col) for col in ("tpot_p50_ms", "e2e_p50_ms", "ttft_p50_ms")
    )
    xs = sorted(tpot)
    ax1.plot(xs, [tpot[o] for o in xs], marker="o", label="TPOT p50")
    ax1.set_ylim(bottom=0)
    ax2.plot(sorted(e2e), [e2e[o] for o in sorted(e2e)], marker="o", label="E2E p50")
    pred = [o for o in xs if o in ttft]
    ax2.scatter(pred, [ttft[o] + (o - 1) * tpot[o] for o in pred], marker="x", color="black",
                zorder=3, label="TTFT + (out - 1) x TPOT")  # fmt: skip
    for ax in (ax1, ax2):
        ax.set_xlabel("output tokens (c=1)")
        ax.set_ylabel("ms")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, "c1")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = (
        ("steady_output_tok_s", "steady output tok/s"),
        ("steady_req_s", "steady req/s (dashed: 1 / prefill at c=1)"),
        ("tpot_p50_ms", "TPOT p50, ms"),
        ("server_running_mean", "running mean (dashed: KV pool / (P + out/2))"),
        ("server_kv_usage_mean", "KV usage mean (dashed: 1 - P / K)"),
        ("preemptions_per_100_req", "preemptions per 100 requests"),
    )
    for ax, (col, ylabel) in zip(axes.flat, panels, strict=True):
        for out in lengths:
            sub = [r for r in runs if olen(r) == out]
            if len({r["value"] for r in sub}) < 2:
                continue
            xs, med, err = by_level(sub, col)
            line = ax.errorbar(xs, med, yerr=err, marker="o", capsize=3, label=f"out={out}")
            if cap and col == "server_running_mean":
                ax.axhline(cap / (prompt_len + out / 2), ls="--", lw=1, color=line[0].get_color())
        if prefill_ms and col == "steady_req_s":
            ax.axhline(1000 / prefill_ms, ls="--", lw=1, color="grey")
        if col == "steady_output_tok_s":
            ax.set_yscale("log")
        ax.set_xlabel("concurrency")
        ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, "sweep")

    clean = lambda r: _num(r.get("server_preemptions")) == 0  # noqa: E731
    base = sorted(
        (n, _num(r["tpot_p50_ms"]) - n * prefill_ms / 128)
        for r in runs if olen(r) == 128 and clean(r) and prefill_ms
        and (n := _num(r.get("server_running_mean"))) and _num(r.get("tpot_p50_ms"))
    )  # fmt: skip
    est: dict[int, list[tuple[float, float]]] = {}
    for r in runs:
        n, y = _num(r.get("server_running_mean")), _num(r.get("tpot_p50_ms"))
        if len(base) >= 2 and olen(r) != 128 and clean(r) and n and y:
            d = float(np.interp(n, [b[0] for b in base], [b[1] for b in base]))
            est.setdefault(olen(r), []).append((d + n * prefill_ms / olen(r), y))
    if est:
        fig, ax = plt.subplots(figsize=(5, 5))
        for out in sorted(est):
            ax.scatter(*zip(*est[out], strict=True), label=f"out={out}", s=18)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--")
        ax.set_xlabel("estimate d(n) + n x prefill / out, ms")
        ax.set_ylabel("TPOT p50 (measured), ms")
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, "tpot_model")
    return paths


def kv(agg_dirs: list[Path], out_dir: Path, prompt_len: int = 2048) -> list[Path]:
    series = []
    for agg_dir in agg_dirs:
        runs = [r for r in load_runs(agg_dir)
                if r["mode"] == "closed" and _num(r.get("input_len")) == prompt_len]  # fmt: skip
        for r in runs:
            r["preemptions_per_100_req"] = _preemptions_per_100(r)
        cap = _median_by(runs, lambda r: 0, "server_kv_capacity_tokens").get(0)
        out = _median_by(runs, lambda r: 0, "output_len").get(0)
        if runs and cap and out:
            # F: mean KV footprint per running sequence, + half a 16-token block of rounding
            series.append((agg_dir.name, runs, cap, prompt_len + out / 2 + 8))
    series.sort(key=lambda s: s[2])
    paths = []

    def save(fig, name: str) -> None:
        fig.suptitle(f"KV pool size, P={prompt_len}: {', '.join(s[0] for s in series)}", fontsize=8)
        fig.tight_layout()
        p = out_dir / f"kv_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = (
        ("server_running_mean", "running mean (dashed: (K - P) / F, dotted: K / F)"),
        ("steady_output_tok_s", "steady output tok/s"),
        ("tpot_p50_ms", "TPOT p50, ms"),
        ("ttft_p50_ms", "TTFT p50, ms"),
        ("server_kv_usage_mean", "KV usage mean (dashed: 1 - P / K)"),
        ("preemptions_per_100_req", "preemptions per 100 requests"),
    )
    for ax, (col, ylabel) in zip(axes.flat, panels, strict=True):
        for name, runs, cap, footprint in series:
            xs, med, err = by_level(runs, col)
            if not xs:
                continue
            line = ax.errorbar(xs, med, yerr=err, marker="o", capsize=3,
                               label=f"{name} (K={cap:.0f})")  # fmt: skip
            # vLLM 0.30 admits a waiting request only if its whole prompt fits in free blocks
            # (EXP_003): the queue starts at ~(K - P) / F running, the pool is full at K / F.
            color = line[0].get_color()
            if col == "server_running_mean":
                ax.axhline((cap - prompt_len) / footprint, ls="--", lw=1, color=color)
                ax.axhline(cap / footprint, ls=":", lw=1, color=color)
            elif col == "server_kv_usage_mean":
                ax.axhline(1 - prompt_len / cap, ls="--", lw=1, color=color)
        if col == "ttft_p50_ms":
            ax.set_yscale("log")
        ax.set_xlabel("concurrency")
        ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    save(fig, "sweep")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, runs, cap, _ in series:
        for ax, col in ((ax1, "server_running_mean"), (ax2, "steady_output_tok_s")):
            _, med, _ = by_level(runs, col)
            if med:
                ax.scatter([cap], [max(med)], s=30, label=name, zorder=3)
    if series:
        ks = [0, max(s[2] for s in series) * 1.05]
        footprint = series[0][3]
        ax1.plot(ks, [k / footprint for k in ks], color="grey", lw=1, ls=":",
                 label="K / F (pool full)")  # fmt: skip
        ax1.plot(ks, [max(k - prompt_len, 0) / footprint for k in ks], color="grey", lw=1, ls="--",
                 label="(K - P) / F (admission)")  # fmt: skip
    ax1.set_ylabel("plateau running (max of level medians)")
    ax2.set_ylabel("max steady output tok/s (max of level medians)")
    for ax in (ax1, ax2):
        ax.set_xlabel("KV pool K, tokens")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    save(fig, "capacity")
    return paths


def timeseries(server_jsonl: Path, out_dir: Path) -> Path:
    snaps = [json.loads(line) for line in server_jsonl.read_text().splitlines() if line.strip()]
    t0 = snaps[0]["t_unix"]
    t = [s["t_unix"] - t0 for s in snaps]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for name, label in (
        ("vllm:num_requests_running", "running"),
        ("vllm:num_requests_waiting", "waiting"),
    ):
        ax1.plot(t, [metric(s["samples"], name) for s in snaps], label=label)
    ax1.set_ylabel("requests")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(t, [metric(s["samples"], "vllm:kv_cache_usage_perc") for s in snaps], color="tab:red")
    ax2.set_ylabel("KV cache usage")
    ax2.set_xlabel("seconds since first snapshot")
    ax2.grid(alpha=0.3)
    run_id = server_jsonl.name.removesuffix(".server.jsonl")
    fig.suptitle(run_id, fontsize=9)
    fig.tight_layout()
    p = out_dir / f"{run_id}_timeseries.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("kind", choices=["sweep", "timeseries", "context", "output", "kv"])
    p.add_argument(
        "paths", type=Path, nargs="+",
        help="results/aggregated/<series> (several for kv) or a .server.jsonl file",
    )  # fmt: skip
    p.add_argument("--out-root", type=Path, default=Path("results/figures"))
    a = p.parse_args()
    if a.kind != "kv" and len(a.paths) != 1:
        p.error(f"{a.kind} takes one path")
    a.path = a.paths[0]
    if a.kind == "kv":
        out_dir = a.out_root / "kv"
    else:
        out_dir = a.out_root / (a.path.parent.name if a.kind == "timeseries" else a.path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.kind == "kv":
        paths = kv(a.paths, out_dir)
    elif a.kind == "timeseries":
        paths = [timeseries(a.path, out_dir)]
    else:
        paths = {"sweep": sweep, "context": context, "output": output}[a.kind](a.path, out_dir)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
