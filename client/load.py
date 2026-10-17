"""M2 load generator: YAML scenario -> closed/open-loop runs -> JSONL + meta + server metrics.

    uv run python -m client.load configs/scenarios/exp1_concurrency.yaml
    uv run python -m client.load configs/scenarios/exp1_concurrency.yaml \
        --set concurrency=8 --set repeats=1

Load modes (docs/EXPERIMENT_PROTOCOL.md, section 4):
  closed  `concurrency` workers; each sends its next request as soon as the previous one
          finished. Load adapts to the server: when it slows down, clients wait instead of
          queueing (coordinated omission), so queues never grow without bound.
  open    requests arrive as a Poisson process with `rate` req/s, independent of responses.
          If rate > capacity, the queue (and TTFT) grows: that is the point.

`concurrency` / `rate` may be lists: one run per level x `repeats` (a sweep). Each run writes
  <out_dir>/<series>/<run_id>.jsonl         request records
  <out_dir>/<series>/<run_id>.meta.json     scenario, server, client, summary
  <out_dir>/<series>/<run_id>.server.jsonl  /metrics snapshots
with run_id = <YYYYMMDD-HHMMSS>_<scenario>_<level>_r<rep>.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from client.common import git_state, server_info
from client.prompts import PromptFactory, PromptSpec
from client.records import RequestRecord, summarize
from client.server_metrics import MetricsScraper
from client.stream import send_request


@dataclass
class Scenario:
    name: str
    series: str
    mode: str = "closed"  # closed | open
    concurrency: int | list[int] = 1  # closed
    rate: float | list[float] | None = None  # open, req/s
    num_requests: int | None = 100  # measure-phase size (count-based) ...
    duration_s: float | None = None  # ... or measure-phase length (duration-based)
    warmup_requests: int = 10  # warm-up = first N requests AND first warmup_s seconds
    warmup_s: float = 0.0
    repeats: int = 1
    cooldown_s: float = 5.0  # pause between runs of a sweep (server drains)
    endpoint: str = "completions"
    input_len: int = 512
    output_len: int = 128
    prompt_format: str = "token_ids"  # token_ids | text
    shared_prefix_len: int = 0
    ignore_eos: bool = True
    temperature: float = 0.0
    timeout_s: float = 300.0
    retries: int = 0
    backoff_s: float = 0.5
    seed: int = 0
    metrics_interval_s: float = 1.0  # 0 disables /metrics snapshots
    max_connections: int = 1024  # open loop: must exceed peak in-flight, see validate()
    model: str | None = None  # default: first id from /v1/models
    notes: str = ""
    extra: dict = field(default_factory=dict)  # free-form, copied to meta

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        return cls(**d)

    def validate(self) -> None:
        if self.mode not in ("closed", "open"):
            raise ValueError(f"mode must be closed|open, got {self.mode}")
        if self.mode == "open" and self.rate is None:
            raise ValueError("open mode needs `rate`")
        if (self.num_requests is None) == (self.duration_s is None):
            raise ValueError("set exactly one of num_requests / duration_s")
        if self.prompt_format == "text" and self.num_requests is None:
            raise ValueError("text prompts are pre-built: use num_requests, not duration_s")
        if self.endpoint == "chat" and self.prompt_format != "text":
            raise ValueError("chat endpoint needs prompt_format: text")

    def levels(self) -> list[tuple[str, int | float]]:
        if self.mode == "closed":
            vals = self.concurrency if isinstance(self.concurrency, list) else [self.concurrency]
            return [(f"c{v}", int(v)) for v in vals]
        vals = self.rate if isinstance(self.rate, list) else [self.rate]
        return [(f"r{v:g}", float(v)) for v in vals]


class Run:
    """State of one run (one level, one repeat)."""

    def __init__(self, sc: Scenario, run_id: str, model: str, prompts: PromptFactory, rep: int):
        self.sc, self.run_id, self.model, self.prompts = sc, run_id, model, prompts
        self.rng = random.Random(sc.seed * 1000 + rep)  # retry jitter
        self.records: list[RequestRecord] = []
        self.sent = 0
        self.measure_sent = 0
        self.t0 = 0.0
        self.measure_t0: float | None = None
        self.prebuilt: list | None = None

    def claim(self, now: float) -> tuple[int, str] | None:
        """Reserve the next request: (index, phase), or None if the run is complete.

        Synchronous on purpose: counting happens before any await, so concurrent workers and
        back-to-back arrivals can never overshoot num_requests.
        """
        sc = self.sc
        if self.sent < sc.warmup_requests or now - self.t0 < sc.warmup_s:
            phase = "warmup"
        else:
            if self.measure_t0 is None:
                self.measure_t0 = now
            if sc.num_requests is not None and self.measure_sent >= sc.num_requests:
                return None
            if sc.duration_s is not None and now - self.measure_t0 >= sc.duration_s:
                return None
            phase = "measure"
            self.measure_sent += 1
        self.sent += 1
        return self.sent - 1, phase

    async def one(self, client, out, idx: int, phase: str, scheduled_at: float | None = None):
        if self.prebuilt is not None:
            prompt = self.prebuilt[idx]
        else:
            prompt = await self.prompts.next()
        rid = uuid.uuid4().hex[:12]
        rec = RequestRecord(
            request_id=rid,
            run_id=self.run_id,
            scenario=self.sc.name,
            phase=phase,
            endpoint=self.sc.endpoint,
            max_tokens=self.sc.output_len,
            started_at=0.0,
            scheduled_at=scheduled_at,
        )
        await send_request(
            client, rec, self.model, prompt,
            ignore_eos=self.sc.ignore_eos, temperature=self.sc.temperature,
            retries=self.sc.retries, backoff_s=self.sc.backoff_s, rng=self.rng,
        )  # fmt: skip
        out.write(json.dumps(rec.to_dict()) + "\n")
        self.records.append(rec)

    async def closed_loop(self, client, out, concurrency: int) -> None:
        async def worker() -> None:
            while (claim := self.claim(time.perf_counter())) is not None:
                await self.one(client, out, *claim)

        self.t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))

    async def open_loop(self, client, out, rate: float) -> None:
        arrivals = random.Random(self.sc.seed)  # same arrival schedule for every repeat
        tasks: list[asyncio.Task] = []
        self.t0 = next_t = time.perf_counter()
        while True:
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            claim = self.claim(next_t)
            if claim is None:
                break
            tasks.append(asyncio.create_task(self.one(client, out, *claim, scheduled_at=next_t)))
            next_t += arrivals.expovariate(rate)
        await asyncio.gather(*tasks)


async def execute(sc: Scenario, base_url: str, out_dir: Path, level: tuple, rep: int) -> dict:
    label, value = level
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}_{sc.name}_{label}_r{rep}"
    series_dir = out_dir / sc.series
    series_dir.mkdir(parents=True, exist_ok=True)

    conns = value if sc.mode == "closed" else sc.max_connections
    limits = httpx.Limits(max_connections=conns, max_keepalive_connections=conns)
    timeout = httpx.Timeout(sc.timeout_s, connect=10.0)
    async with (
        httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as client,
        httpx.AsyncClient(base_url=base_url, timeout=10.0) as aux,  # metrics, tokenizer
    ):
        info = await server_info(aux)
        model = sc.model or info.get("models", {}).get("data", [{}])[0].get("id")
        if not model:
            sys.exit(f"cannot determine model from {base_url}/v1/models: {info}")
        spec = PromptSpec(
            input_len=sc.input_len, format=sc.prompt_format,
            shared_prefix_len=sc.shared_prefix_len, seed=sc.seed * 1000 + rep,
        )  # fmt: skip
        run = Run(sc, run_id, model, PromptFactory(spec, aux, model), rep)
        if sc.prompt_format == "text":
            total = sc.warmup_requests + sc.num_requests
            run.prebuilt = [await run.prompts.next() for _ in range(total)]

        scraper = None
        if sc.metrics_interval_s > 0:
            scraper = MetricsScraper(
                aux, series_dir / f"{run_id}.server.jsonl", sc.metrics_interval_s
            )
            scraper.start()
        with (series_dir / f"{run_id}.jsonl").open("w") as out:
            if sc.mode == "closed":
                await run.closed_loop(client, out, int(value))
            else:
                await run.open_loop(client, out, float(value))
        if scraper:
            await scraper.stop()

    summary = summarize(run.records)
    meta = {
        "run_id": run_id,
        "series_id": sc.series,
        "scenario": asdict(sc),
        "level": {"label": label, "mode": sc.mode, "value": value},
        "repeat": rep,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "model": model,
        "server": info,
        "client": {
            "git": git_state(),
            "python": sys.version.split()[0],
            "httpx": httpx.__version__,
            "max_connections": conns,
        },
        "server_metrics": None
        if scraper is None
        else {"snapshots": scraper.snapshots, "errors": scraper.errors},
        "summary": summary,
    }
    (series_dir / f"{run_id}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _fmt(v: float | None, scale: float = 1000.0, width: int = 7) -> str:
    return f"{v * scale:{width}.1f}" if v is not None else " " * (width - 1) + "-"


def print_run(meta: dict) -> None:
    s = meta["summary"]
    print(
        f"{meta['run_id']}: ok {s['ok']}/{s['requests']}"
        f"  out_tok/s {_fmt(s.get('output_token_throughput'), 1)}"
        f"  TTFT p50/p95 {_fmt(s['ttft_s']['p50'])}/{_fmt(s['ttft_s']['p95'])} ms"
        f"  TPOT p50 {_fmt(s['tpot_s']['p50'])} ms"
        + (f"  send-lag p99 {_fmt(s['send_lag_s']['p99'])} ms" if "send_lag_s" in s else "")
    )


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item}")
        out[k] = yaml.safe_load(v)
    return out


def load_scenario(path: str, overrides: list[str]) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text())
    raw.update(parse_set(overrides))
    sc = Scenario.from_dict(raw)
    sc.validate()
    return sc


async def main_async(args: argparse.Namespace) -> list[dict]:
    sc = load_scenario(args.scenario, args.set)
    plan = [(lvl, rep) for lvl in sc.levels() for rep in range(1, sc.repeats + 1)]
    print(f"scenario {sc.name} -> series {sc.series}: {len(plan)} runs, mode={sc.mode}, "
          f"in={sc.input_len} out={sc.output_len}, base_url={args.base_url}")  # fmt: skip
    if args.dry_run:
        for lvl, rep in plan:
            print(f"  {lvl[0]} r{rep}")
        return []
    metas = []
    for i, (lvl, rep) in enumerate(plan):
        if i and sc.cooldown_s:
            await asyncio.sleep(sc.cooldown_s)
        meta = await execute(sc, args.base_url, Path(args.out_dir), lvl, rep)
        print_run(meta)
        metas.append(meta)
    return metas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("scenario", help="YAML scenario file")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--out-dir", default="results/raw")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a scenario key (value parsed as YAML), repeatable")  # fmt: skip
    p.add_argument("--dry-run", action="store_true", help="print the run plan and exit")
    return p.parse_args(argv)


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
