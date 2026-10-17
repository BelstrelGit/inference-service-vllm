"""M1 client: N sequential streaming requests -> JSONL records + meta.json.

Sequential (concurrency 1) on purpose: this measures single-request latency. Concurrency
and open-loop arrival come with the load generator (M2).

    uv run python -m client.run --base-url http://localhost:8000 -n 20 --warmup 3 --max-tokens 128

Writes results/raw/<series>/<run_id>.jsonl and <run_id>.meta.json (docs/EXPERIMENT_PROTOCOL.md).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from client.records import RequestRecord, e2e, summarize, tpot, ttft
from client.stream import send_request

BASE_PROMPT = (
    "Write a detailed technical explanation of how a large language model inference server "
    "schedules requests, manages GPU memory for the key-value cache and streams tokens back "
    "to clients."
)


def make_prompt(request_id: str) -> str:
    # Unique prefix per request: otherwise prefix caching would serve repeats from cache and
    # every request after the first would show an artificially low TTFT.
    return f"[{request_id}] {BASE_PROMPT}"


def git_state() -> dict:
    # In-cluster runs have no .git: the launcher passes the state in via env.
    if commit := os.environ.get("CLIENT_GIT_COMMIT"):
        return {"commit": commit, "dirty": os.environ.get("CLIENT_GIT_DIRTY") == "1"}

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


async def server_info(client: httpx.AsyncClient) -> dict:
    info: dict = {}
    for key, path in (("version", "/version"), ("models", "/v1/models")):
        try:
            r = await client.get(path)
            info[key] = r.json() if r.status_code == 200 else {"http_status": r.status_code}
        except (httpx.HTTPError, ValueError) as e:
            info[key] = {"error": str(e)}
    return info


def _ms(v: float | None) -> str:
    return f"{v * 1000:8.1f}" if v is not None else "       -"


async def run(args: argparse.Namespace) -> Path:
    started = datetime.now(UTC)
    run_id = (
        args.run_id or f"{started:%Y%m%d-%H%M%S}_{args.scenario}_n{args.num_requests}_r{args.rep}"
    )
    out_dir = Path(args.out_dir) / args.series
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_id}.jsonl"

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    records: list[RequestRecord] = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        info = await server_info(client)
        model = args.model or info.get("models", {}).get("data", [{}])[0].get("id")
        if not model:
            sys.exit(f"cannot determine model from {args.base_url}/v1/models: {info}")

        meta = {
            "run_id": run_id,
            "series_id": args.series,
            "scenario": args.scenario,
            "started_at_utc": started.isoformat(),
            "base_url": args.base_url,
            "model": model,
            "server": info,
            "workload": {
                "load_mode": "sequential",
                "concurrency": 1,
                "endpoint": args.endpoint,
                "num_requests": args.num_requests,
                "warmup_requests": args.warmup,
                "max_tokens": args.max_tokens,
                "ignore_eos": not args.no_ignore_eos,
                "temperature": 0,
                "prompt": "unique prefix + fixed text (prompt tokens from server usage)",
            },
            "client": {
                "git": git_state(),
                "timeout_s": args.timeout,
                "retries": 0,
                "python": sys.version.split()[0],
                "httpx": httpx.__version__,
            },
            "notes": args.notes,
        }

        print(
            f"run {run_id}  model={model}  endpoint={args.endpoint}  max_tokens={args.max_tokens}"
        )
        print(
            f"{'#':>4} {'phase':7} {'status':7} {'in':>5} {'out':>5} "
            f"{'TTFT ms':>8} {'TPOT ms':>8} {'E2E ms':>8}"
        )
        with jsonl_path.open("w") as f:
            for i in range(args.warmup + args.num_requests):
                phase = "warmup" if i < args.warmup else "measure"
                rid = uuid.uuid4().hex[:12]
                rec = RequestRecord(
                    request_id=rid,
                    run_id=run_id,
                    scenario=args.scenario,
                    phase=phase,
                    endpoint=args.endpoint,
                    max_tokens=args.max_tokens,
                    started_at=0.0,
                )
                await send_request(client, rec, model, make_prompt(rid), not args.no_ignore_eos)
                f.write(json.dumps(rec.to_dict()) + "\n")
                f.flush()
                records.append(rec)
                print(
                    f"{i:>4} {phase:7} {rec.status:7} {rec.prompt_tokens or '-':>5} "
                    f"{rec.output_tokens or '-':>5} "
                    f"{_ms(ttft(rec))} {_ms(tpot(rec))} {_ms(e2e(rec))}"
                    + (f"  {rec.error[:80]}" if rec.error else "")
                )

    meta["finished_at_utc"] = datetime.now(UTC).isoformat()
    meta["summary"] = summarize(records)
    (out_dir / f"{run_id}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    s = meta["summary"]
    print(f"\nmeasure: {s['ok']}/{s['requests']} ok")
    for k in ("ttft_s", "tpot_s", "e2e_s"):
        print(
            f"  {k:7} p50={_ms(s[k]['p50'])} ms  p90={_ms(s[k]['p90'])} ms  "
            f"mean={_ms(s[k]['mean'])} ms"
        )
    print(f"saved {jsonl_path}")
    return jsonl_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", help="default: first id from /v1/models")
    p.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    p.add_argument("-n", "--num-requests", type=int, default=10)
    p.add_argument(
        "--warmup", type=int, default=2, help="requests sent first and marked phase=warmup"
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--no-ignore-eos", action="store_true", help="allow stopping at EOS (output length varies)"
    )
    p.add_argument("--timeout", type=float, default=300.0, help="per-request read timeout, s")
    p.add_argument("--series", default="m1-dev")
    p.add_argument("--scenario", default="single")
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--run-id")
    p.add_argument("--out-dir", default="results/raw")
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
