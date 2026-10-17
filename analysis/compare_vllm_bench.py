"""Compare a client.load run with `vllm bench serve --save-result` JSON (EXPERIMENT_PROTOCOL s. 8).

    uv run python -m analysis.compare_vllm_bench results/raw/<series>/<run_id>.meta.json \\
        results/raw/<series>/<run_id>.vllm-bench.json

Expected agreement: within ~5-10%. Known definitional differences (small at these lengths):
  - vllm bench: TTFT at the first chunk with `choices` (even empty text); ours: first
    non-empty text. Same thing for /v1/completions, which sends no role-only chunk.
  - vllm bench: request latency ends at the last token chunk; ours at `data: [DONE]`.
  - vllm bench sends all requests of the run; warm-ups are separate (--num-warmups).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from client.records import RequestRecord, summarize

TOLERANCE = 0.10

# (label, our summary path, vllm bench key)
PAIRS = [
    ("TTFT mean, ms", ("ttft_s", "mean", 1000), "mean_ttft_ms"),
    ("TTFT p50, ms", ("ttft_s", "p50", 1000), "median_ttft_ms"),
    ("TTFT p95, ms", ("ttft_s", "p95", 1000), "p95_ttft_ms"),
    ("TPOT p50, ms", ("tpot_s", "p50", 1000), "median_tpot_ms"),
    ("ITL p50, ms", ("itl_s", "p50", 1000), "median_itl_ms"),
    ("E2E p50, ms", ("e2e_s", "p50", 1000), "median_e2el_ms"),
    ("output tok/s", ("output_token_throughput", None, 1), "output_throughput"),
    ("request/s", ("request_throughput", None, 1), "request_throughput"),
]


def ours(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    jsonl = meta_path.with_name(f"{meta['run_id']}.jsonl")
    recs = [RequestRecord.from_dict(json.loads(x)) for x in jsonl.read_text().splitlines() if x]
    return summarize(recs)


def compare(summary: dict, bench: dict) -> list[dict]:
    rows = []
    for label, (key, stat, scale), bkey in PAIRS:
        v = summary.get(key)
        if stat is not None:
            v = (v or {}).get(stat)
        mine = v * scale if v is not None else None
        theirs = bench.get(bkey)
        diff = (mine - theirs) / theirs if mine is not None and theirs else None
        rows.append({"metric": label, "ours": mine, "vllm_bench": theirs, "diff": diff,
                     "ok": diff is not None and abs(diff) <= TOLERANCE})  # fmt: skip
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("meta", type=Path, help="our <run_id>.meta.json")
    p.add_argument("bench", type=Path, help="vllm bench serve result JSON")
    a = p.parse_args()
    bench = json.loads(a.bench.read_text())
    rows = compare(ours(a.meta), bench)
    print(f"| metric | ours | vllm bench | diff | within {TOLERANCE:.0%} |\n|---|---|---|---|---|")
    for r in rows:
        f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        d = "-" if r["diff"] is None else f"{r['diff']:+.1%}"
        ok = "yes" if r["ok"] else "NO"
        print(f"| {r['metric']} | {f(r['ours'])} | {f(r['vllm_bench'])} | {d} | {ok} |")
    print(f"\nvllm bench: completed={bench.get('completed')} failed={bench.get('failed')} "
          f"duration={bench.get('duration')}")  # fmt: skip


if __name__ == "__main__":
    main()
