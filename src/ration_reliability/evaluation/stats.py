"""Monte Carlo intervals for independent draws from one fixed declared model.

These are simulation-error intervals, not parameter/assumption uncertainty and not
confidence intervals for clustered empirical records. Invalid inputs fail closed:
counts are never rounded, probabilities are never clipped and NaN is never accepted.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from scipy import stats

__all__ = ["clopper_pearson", "one_sided_upper", "zero_event_upper_bound", "mc_standard_error"]


def _count(value: int, name: str, *, positive: bool = False) -> int:
    """Accept finite nonnegative integral real values (including NumPy scalars).

    Integral-valued floats remain supported for CSV-derived data; fractions,
    booleans and strings are rejected rather than silently truncated/coerced.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an integer count, not {value!r}")
    if isinstance(value, Integral):
        out = int(value)
    else:
        x = float(value)
        if not math.isfinite(x) or not x.is_integer():
            raise ValueError(f"{name} must be a finite integer count")
        out = int(x)
    if out < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return out


def _probability(value: float, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability")
    x = float(value)
    valid = 0.0 < x < 1.0 if open_interval else 0.0 <= x <= 1.0
    if not math.isfinite(x) or not valid:
        brackets = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be finite and in {brackets}")
    return x


def _binomial_inputs(k: int, n: int, confidence: float) -> tuple[int, int, float]:
    k, n = _count(k, "k"), _count(n, "n", positive=True)
    if k > n:
        raise ValueError("need 0 <= k <= n and n > 0")
    return k, n, _probability(confidence, "confidence", open_interval=True)


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for a binomial proportion."""
    k, n, confidence = _binomial_inputs(k, n, confidence)
    a = 1.0 - confidence
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def one_sided_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper Clopper-Pearson bound."""
    k, n, confidence = _binomial_inputs(k, n, confidence)
    return 1.0 if k == n else float(stats.beta.ppf(confidence, k + 1, n - k))


def zero_event_upper_bound(n: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1-alpha) upper bound after zero events.

    expm1 avoids cancellation for very large n; alpha must be strictly inside (0,1).
    """
    n = _count(n, "n", positive=True)
    alpha = _probability(alpha, "alpha", open_interval=True)
    return -math.expm1(math.log(alpha) / n)


def mc_standard_error(p_hat: float, n: int) -> float:
    """sqrt(p*(1-p)/n) for independent draws; no clipping of invalid p."""
    n = _count(n, "n", positive=True)
    p = _probability(p_hat, "p_hat")
    return math.sqrt(p * (1.0 - p) / n)
