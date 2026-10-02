"""Shared LP assembly helpers for methods that solve in DM space (x) and execute q = x / d_hat.

T2.1: an optimiser may work in ``x_i`` (kg DM/head/d) but the executed decision is
``q_i = x_i / d_hat_i`` with the *decision-time* estimate ``d_hat``.  A q-space row
``A_q @ q - b`` becomes ``(A_q / d_hat) @ x - b`` and the cost ``p @ q`` becomes ``(p / d_hat) @ x``.

Only ``structural_hard`` and ``probabilistic_nutrition`` constraints enter an optimisation model;
``diagnostic_only`` constraints are reported by the evaluator but never imposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..datamodel import ConstraintClass
from ..nutrition.constraints import CompiledConstraints, LinearRows

__all__ = ["XSpaceLP", "optimization_indices", "assemble_x_space_lp", "big_m_from_box", "residuals_by_constraint"]


@dataclass(frozen=True)
class XSpaceLP:
    """LP data in x-space: ``min c @ x`` s.t. ``A_ub x <= b_ub``, ``A_eq x = b_eq``, ``lb <= x <= ub``."""

    c: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    ub_constraint_ids: tuple[str, ...]
    eq_constraint_ids: tuple[str, ...]


def optimization_indices(cc: CompiledConstraints) -> np.ndarray:
    """Indices of constraints imposed in optimisation (structural + probabilistic)."""
    return cc.indices(ConstraintClass.STRUCTURAL_HARD, ConstraintClass.PROBABILISTIC_NUTRITION)


def assemble_x_space_lp(A_q: np.ndarray, b: np.ndarray, is_eq: np.ndarray, constraint_ids,
                        d_hat: np.ndarray, prices: np.ndarray) -> XSpaceLP:
    """Build x-space LP data from q-space rows ``A_q [K, I]``, ``b [K]`` (non-negativity built in)."""
    A_q = np.asarray(A_q, dtype=float)
    d_hat = np.asarray(d_hat, dtype=float)
    if A_q.ndim != 2 or A_q.shape[1] != d_hat.shape[0]:
        raise ValueError("assemble_x_space_lp: A_q must be [K, I]")
    if np.any(~np.isfinite(A_q)):
        raise ValueError("assemble_x_space_lp: non-finite coefficients (missing data?)")
    A_x = A_q / d_hat[None, :]
    is_eq = np.asarray(is_eq, dtype=bool)
    ids = tuple(constraint_ids)
    I = d_hat.shape[0]
    return XSpaceLP(
        c=np.asarray(prices, dtype=float) / d_hat,
        A_ub=A_x[~is_eq], b_ub=np.asarray(b, dtype=float)[~is_eq],
        A_eq=A_x[is_eq], b_eq=np.asarray(b, dtype=float)[is_eq],
        lb=np.zeros(I), ub=np.full(I, np.inf),
        ub_constraint_ids=tuple(i for i, e in zip(ids, is_eq) if not e),
        eq_constraint_ids=tuple(i for i, e in zip(ids, is_eq) if e),
    )


def big_m_from_box(A: np.ndarray, b: np.ndarray, q_lb: np.ndarray, q_ub: np.ndarray) -> np.ndarray:
    """Tight Big-M per scenario and row: ``M[s,k] = max_{q_lb <= q <= q_ub} A[s,k,:] @ q - b[k]``.

    Derived from variable bounds as required by T4 (no unexplained 1e6).  Needs finite bounds.
    """
    q_lb, q_ub = np.asarray(q_lb, float), np.asarray(q_ub, float)
    if not (np.all(np.isfinite(q_lb)) and np.all(np.isfinite(q_ub))):
        raise ValueError("big_m_from_box: finite variable bounds are required")
    A = np.asarray(A, dtype=float)
    top = np.maximum(A * q_lb[None, None, :], A * q_ub[None, None, :]).sum(axis=2)
    return top - np.asarray(b, dtype=float)[None, :]


def residuals_by_constraint(rows: LinearRows, q: np.ndarray, scenario: int = 0) -> dict[str, float]:
    """Linearised residual ``g_k(q)`` for one scenario (``<= 0`` satisfied; eq rows as ``|g|``)."""
    g = rows.residual(q)[scenario]
    return {cid: float(v) for cid, v in zip(rows.constraint_ids, g)}
