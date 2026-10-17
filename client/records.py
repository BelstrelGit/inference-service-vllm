"""Request record (docs/EXPERIMENT_PROTOCOL.md, section 7) and metrics derived from it.

Raw records keep timestamps only; TTFT/TPOT/E2E are always derived here, so a fix in a
formula re-applies to old raw data without editing it.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields


@dataclass
class RequestRecord:
    request_id: str
    run_id: str
    scenario: str
    phase: str  # "warmup" | "measure"
    endpoint: str  # "completions" | "chat"
    max_tokens: int
    # perf_counter() seconds: only differences between them are meaningful
    started_at: float
    # open-loop only: when the arrival process wanted this request sent (started_at - scheduled_at
    # = client lag; large values mean the load generator itself could not keep up)
    scheduled_at: float | None = None
    first_chunk_at: float | None = None  # any SSE data chunk (vLLM chat sends a role-only chunk)
    first_token_at: float | None = None  # first chunk with non-empty generated text = TTFT point
    completed_at: float | None = None  # [DONE] received
    # wall clock (UTC epoch seconds) of started_at: to line up with server /metrics snapshots
    started_at_unix: float = 0.0
    prompt_tokens: int | None = None  # from server usage
    output_tokens: int | None = None  # from server usage
    finish_reason: str | None = None
    status: str = "error"  # "ok" | "error" | "timeout"
    http_status: int | None = None
    error: str | None = None
    retry_count: int = 0
    # arrival time of every chunk that carried generated text (one chunk ~ one token in vLLM)
    token_timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RequestRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def ttft(r: RequestRecord) -> float | None:
    if r.first_token_at is None:
        return None
    return r.first_token_at - r.started_at


def e2e(r: RequestRecord) -> float | None:
    if r.completed_at is None:
        return None
    return r.completed_at - r.started_at


def tpot(r: RequestRecord) -> float | None:
    """Decode time per output token, first token excluded (it belongs to TTFT)."""
    if r.first_token_at is None or r.completed_at is None or not r.output_tokens:
        return None
    if r.output_tokens < 2:
        return None
    return (r.completed_at - r.first_token_at) / (r.output_tokens - 1)


def itls(r: RequestRecord) -> list[float]:
    ts = r.token_timestamps
    return [b - a for a, b in zip(ts, ts[1:], strict=False)]


def send_lag(r: RequestRecord) -> float | None:
    if r.scheduled_at is None:
        return None
    return r.started_at - r.scheduled_at


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default), p in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


PERCENTILES = (50, 90, 95, 99)


def distribution(values: list[float]) -> dict:
    out: dict = {"n": len(values), "mean": statistics.fmean(values) if values else None}
    for p in PERCENTILES:
        out[f"p{p}"] = percentile(values, p)
    out["max"] = max(values) if values else None
    return out


LATENCY_METRICS: dict[str, Callable[[RequestRecord], float | None]] = {
    "ttft_s": ttft,
    "tpot_s": tpot,
    "e2e_s": e2e,
}


def summarize(records: list[RequestRecord]) -> dict:
    """Run-level summary of the measure phase (EXPERIMENT_PROTOCOL.md, section 5).

    Measurement window: first measure request sent -> last measure request completed.
    Latency percentiles use successful requests only; errors are counted separately.
    """
    measured = [r for r in records if r.phase == "measure"]
    ok = [r for r in measured if r.status == "ok"]
    out: dict = {
        "requests": len(measured),
        "ok": len(ok),
        "errors": len(measured) - len(ok),
        "error_rate": (len(measured) - len(ok)) / len(measured) if measured else None,
        "timeouts": sum(r.status == "timeout" for r in measured),
        "retries": sum(r.retry_count for r in measured),
    }
    ends = [r.completed_at for r in ok if r.completed_at is not None]
    duration = (max(ends) - min(r.started_at for r in measured)) if ends else None
    out["duration_s"] = duration
    if duration:
        out_tok = sum(r.output_tokens or 0 for r in ok)
        in_tok = sum(r.prompt_tokens or 0 for r in ok)
        out["request_throughput"] = len(ok) / duration
        out["output_token_throughput"] = out_tok / duration
        out["total_token_throughput"] = (out_tok + in_tok) / duration
    for name, fn in LATENCY_METRICS.items():
        out[name] = distribution([v for v in map(fn, ok) if v is not None])
    out["itl_s"] = distribution([x for r in ok for x in itls(r)])
    lags = [v for v in map(send_lag, measured) if v is not None]
    if lags:
        out["send_lag_s"] = distribution(lags)
    return out
