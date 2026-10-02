"""Supply quantities of contract T2.

::

    D(q, theta)   = sum_i q_i d_i              # supplied DM, kg DM/head/d
    N_j(q, theta) = sum_i q_i d_i a_ij         # supplied nutrient j (kg/d or Mcal/d)
    C(q)          = sum_i p_i q_i              # feed cost per head per day (as-fed prices)

All functions accept a single state (``d: [I]``, ``theta: [I, J]``) or a batch of scenarios
(``d: [S, I]``, ``theta: [S, I, J]``).  Missing values (NaN) only propagate if the ingredient is
actually used (``q_i != 0``); a NaN is never silently replaced by zero for a used ingredient.
Nothing here renormalises the realised DM formula ``q * d`` (T2.1).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "realized_dm_formula",
    "dm_supply",
    "nutrient_supply",
    "concentrations",
    "ration_cost",
]


def _q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be a 1-D array [I]")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must be finite")
    return q


def realized_dm_formula(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Realised DM amounts ``x_real = q * d`` (kg DM/head/d), **not** renormalised.

    Entries of unused ingredients (``q_i == 0``) are exactly 0 even if ``d_i`` is NaN.
    """
    q = _q(q)
    d = np.asarray(d, dtype=float)
    used = q != 0.0
    with np.errstate(invalid="ignore"):
        return np.where(used, d * q, 0.0)


def dm_supply(q: np.ndarray, d: np.ndarray) -> np.ndarray | float:
    """``D = sum_i q_i d_i``; shape ``[S]`` for batched ``d`` or scalar for ``d: [I]``."""
    x = realized_dm_formula(q, d)
    out = x.sum(axis=-1)
    return float(out) if np.ndim(out) == 0 else out


def nutrient_supply(q: np.ndarray, d: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """``N_j = sum_i q_i d_i a_ij``; shape ``[S, J]`` (batched) or ``[J]``."""
    q = _q(q)
    theta = np.asarray(theta, dtype=float)
    x = realized_dm_formula(q, d)  # [.., I]
    used = (q != 0.0)
    th = np.where(used[:, None], theta, 0.0) if theta.ndim == 2 else np.where(used[None, :, None], theta, 0.0)
    return np.einsum("...i,...ij->...j", x, th)


def concentrations(q: np.ndarray, d: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Realised concentrations ``N_j / D`` (per kg of *realised* DM); NaN where ``D <= 0``."""
    N = nutrient_supply(q, d, theta)
    D = np.asarray(dm_supply(q, d), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = N / D[..., None]
    return np.where(D[..., None] > 0, out, np.nan)


def ration_cost(q: np.ndarray, prices_per_kg_as_fed: np.ndarray) -> float:
    """``C(q) = sum_i p_i q_i`` in currency/head/d (deterministic under as-fed pricing, T2.2)."""
    q = _q(q)
    p = np.asarray(prices_per_kg_as_fed, dtype=float)
    if p.shape != q.shape:
        raise ValueError("prices and q must have the same shape")
    return float(p @ q)
