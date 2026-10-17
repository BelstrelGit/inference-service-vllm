"""Request record (docs/EXPERIMENT_PROTOCOL.md, section 7) and metrics derived from it.

Raw records keep timestamps only; TTFT/TPOT/E2E are always derived here, so a fix in a
formula re-applies to old raw data without editing it.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field


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


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default), p in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(records: list[RequestRecord]) -> dict:
    """Quick per-run summary of the measure phase. Proper aggregation lives in analysis/."""
    measured = [r for r in records if r.phase == "measure"]
    ok = [r for r in measured if r.status == "ok"]
    out: dict = {"requests": len(measured), "ok": len(ok), "errors": len(measured) - len(ok)}
    for name, fn in (("ttft_s", ttft), ("tpot_s", tpot), ("e2e_s", e2e)):
        vals = [v for v in map(fn, ok) if v is not None]
        out[name] = {
            "mean": statistics.fmean(vals) if vals else None,
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "p99": percentile(vals, 99),
        }
    return out
