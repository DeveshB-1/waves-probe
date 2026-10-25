from __future__ import annotations

import math


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]. NaN on empty input.

    Callers pool raw samples across runs and compute once. Averaging
    per-run p95s gives a number that is not the p95 of anything.
    """
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return math.nan
    rank = max(1, math.ceil(q / 100 * len(xs)))
    return xs[rank - 1]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else math.nan,
    }
