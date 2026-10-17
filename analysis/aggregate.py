"""Raw runs of a series -> results/aggregated/<series>/{runs,levels}.csv (+ a Markdown table).

    uv run python -m analysis.aggregate results/raw/<series>

Everything is recomputed from raw JSONL (meta summaries are not trusted), so a fix in a
formula re-applies to old data. Server-side columns come from the /metrics snapshots taken
during the run, restricted to the measurement window:
  window   = first measure request sent -> last measure request completed (client clock,
             mapped to wall clock via started_at_unix - started_at)
  counters / histogram _sum,_count: (last snapshot <= window end) - (last snapshot <= start)
  gauges   : mean and max over snapshots inside the window
levels.csv = median over repeats per level, plus min/max of the headline metrics.

output_token_throughput (from client.records.summarize) = measure-phase output tokens / window.
It is biased when the window is short relative to E2E: at the start the server still works on
warm-up requests whose tokens are not counted, and at the end the load drains. Hence also
  steady_output_tok_s = all output tokens (any phase) emitted between the first and the last
                        measure request sent / that interval (full load, no drain)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from client.records import RequestRecord, summarize
from client.server_metrics import metric

HEADLINE = (
    "output_token_throughput", "steady_output_tok_s",
    "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "e2e_p95_ms",
)  # fmt: skip


def load_records(path: Path) -> list[RequestRecord]:
    with path.open() as f:
        return [RequestRecord.from_dict(json.loads(line)) for line in f if line.strip()]


def load_snapshots(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def window(records: list[RequestRecord]) -> tuple[float, float] | None:
    m = [r for r in records if r.phase == "measure"]
    ends = [r.completed_at for r in m if r.completed_at is not None]
    if not m or not ends:
        return None
    offset = m[0].started_at_unix - m[0].started_at
    return min(r.started_at for r in m) + offset, max(ends) + offset


def steady_output_tok_s(records: list[RequestRecord]) -> float | None:
    sends = [r.started_at for r in records if r.phase == "measure"]
    if len(sends) < 2 or max(sends) <= min(sends):
        return None
    a, b = min(sends), max(sends)
    n = sum(a <= t <= b for r in records for t in (r.token_timestamps or []))
    return n / (b - a)


def server_stats(snaps: list[dict], win: tuple[float, float] | None) -> dict:
    if not snaps or win is None:
        return {}
    start, end = win
    before = [s for s in snaps if s["t_unix"] <= start] or snaps[:1]
    upto = [s for s in snaps if s["t_unix"] <= end] or snaps[:1]
    a, b = before[-1]["samples"], upto[-1]["samples"]
    inside = [s["samples"] for s in snaps if start <= s["t_unix"] <= end]

    def delta(name: str) -> float | None:
        x, y = metric(a, name), metric(b, name)
        return None if x is None or y is None else y - x

    def hist_mean_ms(name: str) -> float | None:
        s, c = delta(f"{name}_sum"), delta(f"{name}_count")
        return s / c * 1000 if s is not None and c else None

    def gauge(name: str) -> tuple[float | None, float | None]:
        vals = [v for v in (metric(s, name) for s in inside) if v is not None]
        return (statistics.fmean(vals), max(vals)) if vals else (None, None)

    out: dict = {"server_snapshots_in_window": len(inside)}
    for col, name in (
        ("server_ttft_mean_ms", "vllm:time_to_first_token_seconds"),
        ("server_queue_mean_ms", "vllm:request_queue_time_seconds"),
        ("server_prefill_mean_ms", "vllm:request_prefill_time_seconds"),
        ("server_decode_mean_ms", "vllm:request_decode_time_seconds"),
        ("server_itl_mean_ms", "vllm:inter_token_latency_seconds"),
        ("server_e2e_mean_ms", "vllm:e2e_request_latency_seconds"),
    ):
        out[col] = hist_mean_ms(name)
    out["server_preemptions"] = delta("vllm:num_preemptions_total")
    hits, queries = delta("vllm:prefix_cache_hits_total"), delta("vllm:prefix_cache_queries_total")
    out["server_prefix_hit_rate"] = hits / queries if hits is not None and queries else None
    gen = delta("vllm:generation_tokens_total")
    span = upto[-1]["t_unix"] - before[-1]["t_unix"]
    out["server_gen_tok_s"] = gen / span if gen is not None and span > 0 else None
    for col, name in (
        ("running", "vllm:num_requests_running"),
        ("waiting", "vllm:num_requests_waiting"),
        ("kv_usage", "vllm:kv_cache_usage_perc"),
    ):
        out[f"server_{col}_mean"], out[f"server_{col}_max"] = gauge(name)
    # vLLM 0.30: reason="capacity" (no scheduling capacity: KV blocks, max_num_seqs, token
    # budget are NOT distinguished) or "deferred" (LoRA budget, KV transfer, blocked)
    for reason in ("capacity", "deferred"):
        vals = [
            v
            for s in inside
            for k, v in s.items()
            if k.startswith("vllm:num_requests_waiting_by_reason{") and f'reason="{reason}"' in k
        ]
        out[f"server_waiting_{reason}_max"] = max(vals) if vals else None
    return out


def flatten(summary: dict) -> dict:
    row: dict = {}
    for k, v in summary.items():
        if isinstance(v, dict):  # distribution of seconds -> ms columns
            base = k.removesuffix("_s")
            for stat, x in v.items():
                if stat == "n":
                    continue
                row[f"{base}_{stat}_ms"] = x * 1000 if x is not None else None
        else:
            row[k] = v
    return row


def run_row(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    run_id = meta["run_id"]
    records = load_records(meta_path.with_name(f"{run_id}.jsonl"))
    level = meta.get("level") or {"label": "single", "mode": "sequential", "value": 1}
    row = {
        "series": meta.get("series_id"),
        "run_id": run_id,
        "scenario": records[0].scenario if records else None,
        "mode": level["mode"],
        "level": level["label"],
        "value": level["value"],
        "repeat": meta.get("repeat", 1),
    }
    sc = meta.get("scenario") if isinstance(meta.get("scenario"), dict) else {}
    row["input_len"] = sc.get("input_len")
    row["output_len"] = sc.get("output_len") or meta.get("workload", {}).get("max_tokens")
    row.update(flatten(summarize(records)))
    row["steady_output_tok_s"] = steady_output_tok_s(records)
    row["client_cpu_util"] = meta.get("client", {}).get("cpu_util")
    snaps = load_snapshots(meta_path.with_name(f"{run_id}.server.jsonl"))
    row.update(server_stats(snaps, window(records)))
    return row


def level_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(
            (r["scenario"], r["mode"], r["value"], r["input_len"], r["output_len"]), []
        ).append(r)
    out = []
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), float(k[2]))):
        g = groups[key]
        lvl = {
            k: g[0][k]
            for k in ("series", "scenario", "mode", "level", "value", "input_len", "output_len")
        }
        lvl["repeats"] = len(g)
        for col in g[0]:
            if col in lvl or col in ("run_id", "repeat"):
                continue
            vals = [
                r[col]
                for r in g
                if isinstance(r.get(col), int | float) and not isinstance(r.get(col), bool)
            ]
            lvl[col] = statistics.median(vals) if vals else None
            if col in HEADLINE and vals:
                lvl[f"{col}_min"], lvl[f"{col}_max"] = min(vals), max(vals)
        out.append(lvl)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def markdown(levels: list[dict]) -> str:
    cols = [
        ("level", "level", None), ("repeats", "n", None),
        ("output_token_throughput", "out tok/s", 1), ("steady_output_tok_s", "steady tok/s", 1),
        ("request_throughput", "req/s", 2),
        ("ttft_p50_ms", "TTFT p50", 1), ("ttft_p95_ms", "TTFT p95", 1),
        ("tpot_p50_ms", "TPOT p50", 1), ("e2e_p95_ms", "E2E p95", 0),
        ("error_rate", "err", 3), ("server_waiting_max", "wait max", 0),
        ("server_kv_usage_max", "KV max", 2), ("server_preemptions", "preempt", 0),
    ]  # fmt: skip
    if len({r["scenario"] for r in levels}) > 1:
        cols.insert(0, ("scenario", "scenario", None))
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in levels:
        cells = []
        for key, _, digits in cols:
            v = r.get(key)
            cells.append(
                "-" if v is None else (f"{v:.{digits}f}" if digits is not None else str(v))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate(
    raw_dir: Path, out_root: Path = Path("results/aggregated")
) -> tuple[Path, list[dict]]:
    metas = sorted(raw_dir.glob("*.meta.json"))
    if not metas:
        raise SystemExit(f"no *.meta.json in {raw_dir}")
    rows = [run_row(m) for m in metas]
    levels = level_rows(rows)
    out_dir = out_root / raw_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "runs.csv", rows)
    write_csv(out_dir / "levels.csv", levels)
    (out_dir / "levels.md").write_text(markdown(levels) + "\n")
    return out_dir, levels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("raw_dir", type=Path, help="results/raw/<series>")
    p.add_argument("--out-root", type=Path, default=Path("results/aggregated"))
    a = p.parse_args()
    out_dir, levels = aggregate(a.raw_dir, a.out_root)
    print(markdown(levels))
    # Prompts are random and unique per run: hits mean prompts were reused (or a shared prefix).
    for lv in levels:
        hit = lv.get("server_prefix_hit_rate")
        if hit is not None and hit > 0.05:
            print(f"WARNING: {lv['level']}: prefix cache hit rate {hit:.1%} - check prompt reuse")
    print(f"\nwrote {out_dir}/runs.csv, levels.csv, levels.md")


if __name__ == "__main__":
    main()
