"""Periodic snapshots of vLLM /metrics during a run -> <run_id>.server.jsonl.

Each line: {"t_unix", "t_perf", "samples": {"<name>{<labels>}": value}}.
Kept: every `vllm:` sample except histogram buckets and `_created` timestamps. Gauges give
the time series (running / waiting / KV usage); counters and histogram _sum/_count give
averages over any window as (end - start) differences, e.g. mean server-side TTFT =
d(time_to_first_token_seconds_sum) / d(time_to_first_token_seconds_count).
Buckets are dropped to keep files small; percentiles come from client-side records.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx


def parse_prometheus(text: str, prefix: str = "vllm:") -> dict[str, float]:
    """Minimal Prometheus text parser: `name{labels} value [timestamp]` -> {"name{labels}": v}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        # labels may contain spaces inside quotes: split on the last '}' when present
        if "}" in line:
            key, _, rest = line.rpartition("}")
            key += "}"
        else:
            key, _, rest = line.partition(" ")
        name = key.split("{", 1)[0]
        if name.endswith(("_bucket", "_created")):
            continue
        try:
            out[key] = float(rest.split()[0])
        except (IndexError, ValueError):
            continue
    return out


def metric(samples: dict[str, float], name: str) -> float | None:
    """Sum of all label sets of `name` (one engine, one model -> just the value)."""
    vals = [v for k, v in samples.items() if k.split("{", 1)[0] == name]
    return sum(vals) if vals else None


class MetricsScraper:
    def __init__(self, client: httpx.AsyncClient, path: Path, interval_s: float = 1.0):
        self.client = client
        self.path = path
        self.interval_s = interval_s
        self.snapshots = 0
        self.errors = 0
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        with self.path.open("w") as f:
            next_t = time.perf_counter()
            while True:
                try:
                    t_perf, t_unix = time.perf_counter(), time.time()
                    r = await self.client.get("/metrics", timeout=self.interval_s * 5)
                    r.raise_for_status()
                    snap = {"t_unix": t_unix, "t_perf": t_perf, "samples": parse_prometheus(r.text)}
                    f.write(json.dumps(snap) + "\n")
                    f.flush()
                    self.snapshots += 1
                except httpx.HTTPError:
                    self.errors += 1
                next_t += self.interval_s
                await asyncio.sleep(max(0.0, next_t - time.perf_counter()))

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Take one last snapshot so the window end is covered, then stop."""
        if self._task is None:
            return
        await asyncio.sleep(self.interval_s)
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
