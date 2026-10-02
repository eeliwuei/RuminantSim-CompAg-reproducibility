"""HiGHS (via SciPy) wrappers with an explicit, audited status mapping.

Status mapping (HiGHS model status parsed from SciPy's message ``"(HiGHS Status N: ...)"``):

=====================================  ===============================================
HiGHS model status                     engine :class:`SolveStatus`
=====================================  ===============================================
kOptimal (7)                           ``optimal`` if the engine's residual check passes,
                                       else ``numerical_error``
kTimeLimit (13), kIterationLimit (14),  ``feasible_time_limit`` if a candidate exists and
kSolutionLimit (16), kInterrupt (17)   passes the residual check, else
                                       ``no_feasible_solution_found``
kInfeasible (8)                        ``proven_infeasible``
kUnboundedOrInfeasible (9)             re-solve once with presolve off, then map again;
                                       still ambiguous -> ``numerical_error``
kUnbounded (10), kModelError (2)       ``invalid_input`` (a ration LP with finite prices and
                                       non-negative amounts is mis-specified if unbounded)
anything else                          ``numerical_error``
=====================================  ===============================================

No relaxation or repair is ever attempted.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from ..datamodel import SolveStatus, SolverOptions

__all__ = ["LPOutcome", "solver_versions", "solver_version_string", "primal_residual", "run_linprog", "run_milp"]

_K_OPTIMAL, _K_INFEASIBLE, _K_UNB_OR_INF, _K_UNBOUNDED, _K_MODEL_ERROR = 7, 8, 9, 10, 2
_K_LIMITS = {13, 14, 16, 17}
_STATUS_RE = re.compile(r"HiGHS Status (\d+)")


def solver_versions() -> dict[str, str]:
    """Versions of SciPy and of the HiGHS library bundled with it."""
    out = {"scipy": scipy.__version__, "numpy": np.__version__}
    try:  # private module; version constants are exported by the HiGHS bindings
        from scipy.optimize._highspy import _core  # type: ignore

        h = _core._Highs()
        v = h.version()
        g = h.githash() if hasattr(h, "githash") else ""
        out["highs"] = f"{v}" + (f" (git {g})" if g else "")
    except Exception:  # pragma: no cover - depends on SciPy internals
        out["highs"] = f"unknown (bundled with scipy {scipy.__version__})"
    return out


def solver_version_string() -> str:
    """One-line solver version, e.g. ``HiGHS 1.12.0 (git 4f96ee8) via scipy 1.17.1``."""
    v = solver_versions()
    return f"HiGHS {v['highs']} via scipy {v['scipy']}"


@dataclass
class LPOutcome:
    """Raw result of one HiGHS call after status mapping."""

    status: SolveStatus
    x: Optional[np.ndarray]
    fun: Optional[float]
    iterations: Optional[int]
    message: str
    highs_model_status: Optional[int]
    wall_time_s: float
    max_rel_residual: Optional[float] = None
    max_abs_residual: Optional[float] = None
    mip_gap: Optional[float] = None
    resolved_without_presolve: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def primal_residual(A_ub, b_ub, A_eq, b_eq, lb, ub, x) -> tuple[float, float]:
    """Max absolute and max scaled violation of ``A_ub x <= b_ub``, ``A_eq x = b_eq`` and bounds.

    Scaled violation of a row = violation / (1 + |b| + sum_i |A_i x_i|); of a bound =
    violation / (1 + |bound|).
    """
    x = np.asarray(x, dtype=float)
    abs_v, rel_v = [0.0], [0.0]
    if A_ub is not None and len(b_ub):
        r = A_ub @ x - b_ub
        v = np.maximum(r, 0.0)
        sc = 1.0 + np.abs(b_ub) + np.abs(A_ub) @ np.abs(x)
        abs_v.append(float(v.max()))
        rel_v.append(float((v / sc).max()))
    if A_eq is not None and len(b_eq):
        r = np.abs(A_eq @ x - b_eq)
        sc = 1.0 + np.abs(b_eq) + np.abs(A_eq) @ np.abs(x)
        abs_v.append(float(r.max()))
        rel_v.append(float((r / sc).max()))
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    vl = np.where(np.isfinite(lb), np.maximum(lb - x, 0.0), 0.0)
    vu = np.where(np.isfinite(ub), np.maximum(x - ub, 0.0), 0.0)
    abs_v += [float(vl.max(initial=0.0)), float(vu.max(initial=0.0))]
    rel_v += [float((vl / (1 + np.abs(np.where(np.isfinite(lb), lb, 0)))).max(initial=0.0)),
              float((vu / (1 + np.abs(np.where(np.isfinite(ub), ub, 0)))).max(initial=0.0))]
    return max(abs_v), max(rel_v)


def _highs_status(message: str) -> Optional[int]:
    m = _STATUS_RE.search(message or "")
    return int(m.group(1)) if m else None


def _map(hs: Optional[int], scipy_status: int, x, res_ok: bool) -> SolveStatus:
    if hs == _K_OPTIMAL or (hs is None and scipy_status == 0):
        return SolveStatus.OPTIMAL if (x is not None and res_ok) else SolveStatus.NUMERICAL_ERROR
    if hs in _K_LIMITS or (hs is None and scipy_status == 1):
        return SolveStatus.FEASIBLE_TIME_LIMIT if (x is not None and res_ok) else \
            SolveStatus.NO_FEASIBLE_SOLUTION_FOUND
    if hs == _K_INFEASIBLE:
        return SolveStatus.PROVEN_INFEASIBLE
    if hs in (_K_UNBOUNDED, _K_MODEL_ERROR):
        return SolveStatus.INVALID_INPUT
    return SolveStatus.NUMERICAL_ERROR


def _dense(A):
    if A is None:
        return None
    A = np.asarray(A, dtype=float)
    return A if A.size else None


def run_linprog(c, A_ub, b_ub, A_eq, b_eq, lb, ub, options: SolverOptions) -> LPOutcome:
    """Solve ``min c x  s.t.  A_ub x <= b_ub, A_eq x = b_eq, lb <= x <= ub`` with HiGHS."""
    c = np.asarray(c, dtype=float)
    A_ub, A_eq = _dense(A_ub), _dense(A_eq)
    b_ub = None if A_ub is None else np.asarray(b_ub, dtype=float)
    b_eq = None if A_eq is None else np.asarray(b_eq, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    bounds = list(zip([None if not np.isfinite(v) else float(v) for v in lb],
                      [None if not np.isfinite(v) else float(v) for v in ub]))

    def once(opts: SolverOptions) -> LPOutcome:
        hopts: dict[str, Any] = {
            "presolve": bool(opts.presolve),
            "primal_feasibility_tolerance": float(opts.primal_feasibility_tolerance),
            "dual_feasibility_tolerance": float(opts.dual_feasibility_tolerance),
            "disp": False,
        }
        if opts.time_limit_s is not None:
            hopts["time_limit"] = float(opts.time_limit_s)
        t0 = time.perf_counter()
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
                    options=hopts)
        wt = time.perf_counter() - t0
        hs = _highs_status(r.message)
        x = None if r.x is None else np.array(r.x, dtype=float)
        ra = rr = None
        ok = False
        if x is not None and np.all(np.isfinite(x)):
            ra, rr = primal_residual(A_ub, b_ub, A_eq, b_eq, lb, ub, x)
            ok = rr <= opts.residual_check_rel_tol
        st = _map(hs, int(r.status), x, ok)
        keep_x = x if st in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT) else None
        return LPOutcome(st, keep_x, None if keep_x is None else float(r.fun), int(getattr(r, "nit", 0) or 0),
                         str(r.message), hs, wt, rr, ra, extra={"raw_x": None if x is None else x.tolist()})

    out = once(options)
    if out.highs_model_status == _K_UNB_OR_INF and options.presolve:
        out2 = once(replace(options, presolve=False))
        out2.resolved_without_presolve = True
        out2.wall_time_s += out.wall_time_s
        out2.message = f"[first pass: {out.message}] {out2.message}"
        return out2
    return out


def run_milp(c, A_ub, b_ub, A_eq, b_eq, lb, ub, integrality, options: SolverOptions) -> LPOutcome:
    """Solve a MILP with HiGHS (``integrality[i] = 1`` for integer variables).

    Provided for methods such as joint chance-constrained SAA (M2).  ``mip_gap`` is recorded.
    """
    c = np.asarray(c, dtype=float)
    A_ub, A_eq = _dense(A_ub), _dense(A_eq)
    cons = []
    if A_ub is not None:
        cons.append(LinearConstraint(A_ub, -np.inf, np.asarray(b_ub, dtype=float)))
    if A_eq is not None:
        be = np.asarray(b_eq, dtype=float)
        cons.append(LinearConstraint(A_eq, be, be))
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    hopts: dict[str, Any] = {"presolve": bool(options.presolve), "disp": False}
    if options.time_limit_s is not None:
        hopts["time_limit"] = float(options.time_limit_s)
    if options.mip_rel_gap is not None:
        hopts["mip_rel_gap"] = float(options.mip_rel_gap)
    t0 = time.perf_counter()
    r = milp(c, constraints=cons or None, integrality=np.asarray(integrality), bounds=Bounds(lb, ub),
             options=hopts)
    wt = time.perf_counter() - t0
    hs = _highs_status(r.message)
    x = None if r.x is None else np.array(r.x, dtype=float)
    ra = rr = None
    ok = False
    if x is not None and np.all(np.isfinite(x)):
        ra, rr = primal_residual(A_ub, None if A_ub is None else np.asarray(b_ub, float),
                                 A_eq, None if A_eq is None else np.asarray(b_eq, float), lb, ub, x)
        integ = np.asarray(integrality) > 0
        int_err = float(np.max(np.abs(x[integ] - np.round(x[integ])), initial=0.0))
        ok = rr <= options.residual_check_rel_tol and int_err <= 1e-6
    st = _map(hs, int(r.status), x, ok)
    keep_x = x if st in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT) else None
    gap = getattr(r, "mip_gap", None)
    return LPOutcome(st, keep_x, None if keep_x is None else float(r.fun), None, str(r.message), hs, wt, rr, ra,
                     mip_gap=None if gap is None else float(gap),
                     extra={"mip_node_count": getattr(r, "mip_node_count", None),
                            "mip_dual_bound": getattr(r, "mip_dual_bound", None),
                            "raw_x": None if x is None else x.tolist()})
