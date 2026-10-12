"""Uncertainty estimates. Pure Python and seeded, so they reproduce exactly."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z_95 = 1.959963984540054


def mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion. More reliable than the normal
    approximation when n is small or the rate is near 0 or 1."""
    if n <= 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be within [0, n], got {successes}/{n}")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)  # exact at the extremes
    return (low, high)


def bootstrap_mean_interval(
    values: Sequence[float], *, resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for a mean. Same inputs give the same output."""
    n = len(values)
    if n == 0:
        return None
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    # random.random() is the one generator method Python guarantees to reproduce across
    # versions for a given seed (randrange is not), so indices are derived from it.
    rng = random.Random(seed)
    data = list(values)
    means = sorted(sum(data[int(rng.random() * n)] for _ in range(n)) / n for _ in range(resamples))
    lo_index = math.floor((alpha / 2) * (resamples - 1))
    hi_index = math.ceil((1 - alpha / 2) * (resamples - 1))
    return (means[lo_index], means[hi_index])


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100])."""
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)
