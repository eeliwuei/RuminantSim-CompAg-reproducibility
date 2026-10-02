"""M2: joint chance-constrained SAA, and M2b: marginal chance constraints with Bonferroni allocation.

Contract T4 M2, section 9 M2.  Both solve a sample-average approximation (SAA) on the ``opt``
draws only; every returned ration is scored by the public evaluator, never here.

M2 -- joint chance constraint (``M2_joint_chance_saa``)
------------------------------------------------------
Variables: ``x_i >= 0`` (kg DM/head/d; executed ``q = x / d_hat``, T2.1) and one binary
``z_s`` per training scenario ``s = 1..N``::

    min  sum_i (p_i / d_hat_i) x_i
    s.t. structural_hard rows (deterministic, d_hat; never relaxed)
         A_x[s,k,:] @ x - M[s,k] z_s <= b_k      for every probabilistic_nutrition row k, every s
         sum_s z_s <= m = floor(alpha_train * N)
         z_s in {0, 1}

All probabilistic rows of scenario ``s`` share the **same** ``z_s``, so ``z_s = 0`` means "all
constraints hold in scenario s": the training joint violation rate is ``<= m / N <= alpha_train``
(the joint event ``I`` of T3).  ``A_x[s]`` are the engine's linearised rows
(:func:`~ration_reliability.nutrition.linear_rows`) with the scenario's own DM ``d_s`` and
composition ``theta_s``, divided by ``d_hat`` (x-space).  ``floor`` is computed exactly from the
decimal representation of ``alpha`` (``0.29 * 100 -> 29``, not 28).

Big-M derivation (T4: "Big-M 从变量界与约束系数推导")
------------------------------------------------------
Write ``g_{s,k}(x) = A_x[s,k,:] @ x - b_k``.  A Big-M is *valid* when no SAA-feasible ``(x, z)``
is cut off, i.e. ``g_{s,k}(x) <= M[s,k]`` for every SAA-feasible ``x`` with ``z_s = 1``.

1. Box: ``[lo_i, hi_i]`` = min / max of ``x_i`` over the structural polytope
   ``P = {x >= 0 : structural rows}`` (2 I small LPs).  ``P`` must be bounded (e.g. a DM-offer rule);
   otherwise ``invalid_input``.  If ``P`` is empty the problem is ``proven_infeasible``.  The box
   (padded by 1e-9 relative, to absorb LP round-off) is used only to derive M; it is implied by the
   structural rows and is not added as variable bounds (bounds at padded values would create
   tolerance-level artefacts, e.g. a DM offer exceeded by 2e-8 kg).
2. Base M, valid for **every** ``x in P``:

   * ``"box"``: ``M_box[s,k] = max_{lo <= x <= hi} g_{s,k}(x)``
     (:func:`~ration_reliability.optimization.lp_builder.big_m_from_box`); ``P`` lies in the box, so
     ``M_box >= max_{x in P} g`` (not too tight);
   * ``"lp_tight"``: ``max_{x in P} g_{s,k}(x)`` itself (one LP per row; tightest base M).

   Rows with base ``M <= 0`` hold for every ``x in P``; they are dropped (exact).  A negative M
   must never be used with ``z``: it would *tighten* the row when ``z_s = 1``.
3. ``"box_quantile"`` (default) additionally uses the budget.  With
   ``D_{s,t} = max_{lo <= x <= hi} (A_x[s,k,:] - A_x[t,k,:]) @ x`` (closed form on the box; ``D_{s,s} = 0``)
   and ``m_k`` the number of scenarios that may violate row ``k`` (``m`` for M2, ``m_k`` for M2b),
   ``Mq[s,k] = (m_k + 1)``-th smallest of ``{D_{s,t}}_t``.  *Validity:* in an SAA-feasible point at
   most ``m_k`` scenarios violate row ``k``, so among the ``m_k + 1`` scenarios ``t`` with the
   smallest ``D_{s,t}`` one satisfies it (``g_t <= 0``; ``t != s`` when ``z_s = 1``); then
   ``g_s(x) = (g_s - g_t)(x) + g_t(x) <= D_{s,t} <= Mq``.  The model uses ``min(M_box, Mq)``.
   Rows with ``M_box > 0`` but ``min(M_box, Mq) <= 0`` hold in **every SAA-feasible** point (not in
   every point of ``P``); they are imposed as hard rows without ``z`` ("hard_by_quantile").  This
   is the quantile strengthening known from the chance-constraint MIP literature (e.g. Luedtke 2014;
   Qiu, Ahmed, Dey & Wolsey 2014 -- references to be verified in the literature stage); here it
   is derived and tested directly (tests/numerical/test_chance_saa.py: equal optimum to "box",
   "lp_tight" and brute-force enumeration).
4. Scale check (``diagnostics["big_m"]``): counts of dropped / hard / Big-M rows, ``M_max``,
   coefficient range, the relaxation ``1e-6 * M_max`` that an integrality error of 1e-6 in ``z``
   could leak into a row, the median tightening ``M_used / M_box`` and, for box modes, a validity
   and looseness check ``M_box`` vs the polytope maximum on the up-to-32 rows with the largest
   ``M_box``.  Because of the leak, the MILP solution is **polished** by default: ``z`` is rounded
   and the LP with every hard row and every row of the ``z = 0`` scenarios imposed exactly is
   re-solved (its optimum is <= the MILP cost of that pattern).  Both objectives are recorded.

Status (SolveResult; no relaxation, no repair)
----------------------------------------------
``optimal``                    HiGHS proved optimality within ``mip_rel_gap`` (``mip_gap`` recorded).
``feasible_time_limit``        time/node limit reached with an incumbent that passes the residual
                               check (not proven optimal; gap recorded).
``no_feasible_solution_found`` limit reached without an incumbent (this is *not* infeasibility).
``proven_infeasible``          the SAA instance (these N scenarios, this m) has no solution, or the
                               structural rows alone are infeasible.
``invalid_input`` / ``numerical_error`` as in the engine table (``optimization/highs.py``).

All statuses refer to the SAA instance on the given ``opt`` scenarios, not to the true chance-
constrained problem; satisfying the training budget is no guarantee on other draws (T4).
``solver_options.time_limit_s`` applies to the MILP only; the small auxiliary LPs (box,
``lp_tight`` Big-M, scale report, polish) run without a time limit, so that a short MILP limit is
reported as ``feasible_time_limit`` / ``no_feasible_solution_found`` and never as an auxiliary failure
(``lp_tight`` solves N x K_p LPs and can be slow for large N).

M2b -- marginal chance constraints with Bonferroni allocation (``M2b_marginal_bonferroni_saa``)
------------------------------------------------------------------------------------------------
One binary ``z_{s,k}`` per scenario *and row*, with ``sum_s z_{s,k} <= floor(alpha_k N)`` and
``sum_k alpha_k <= alpha_train`` (default ``alpha_k = alpha_train / K_p``).  By the union bound the
training joint violation rate is ``<= sum_k floor(alpha_k N) / N <= alpha_train``.  It is a separate,
usually more conservative model -- **not** an implementation of the joint SAA -- and is named so.
Allocations with ``sum_k alpha_k > alpha_train`` are rejected (``invalid_input``): per-row ``alpha``
cannot be reported as a joint ``alpha``.

Scale / performance: the MILP is assembled sparsely (``scipy.sparse``) and solved with
``scipy.optimize.milp`` (HiGHS).  Large ``N`` belongs on the server (user rule: Mac = delivery only).
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from ..datamodel import ConstraintClass, RationDecision, RationProblem, SolverOptions, SolveResult, SolveStatus
from ..hashing import stable_hash
from ..nutrition.constraints import linear_rows
from ..uncertainty.base import DrawSet, require_stream
from . import highs as H
from .highs import LPOutcome, run_linprog, solver_version_string
from .lp_builder import assemble_x_space_lp, big_m_from_box, optimization_indices
from .safety_margin import ValidationSelection, select_parameter_on_validation

__all__ = [
    "METHOD_ID_JOINT",
    "METHOD_ID_BONFERRONI",
    "PARAM_DEFAULTS_JOINT",
    "PARAM_DEFAULTS_BONFERRONI",
    "BIG_M_MODES",
    "INT_TOL_ASSUMED",
    "allowed_violations",
    "SAAModel",
    "build_saa_model",
    "quantile_big_m",
    "solve",
    "solve_marginal_bonferroni",
    "select_alpha_train_on_validation",
]

METHOD_ID_JOINT = "M2_joint_chance_saa"
METHOD_ID_BONFERRONI = "M2b_marginal_bonferroni_saa"
BIG_M_MODES = ("box_quantile", "box", "lp_tight")
PARAM_DEFAULTS_JOINT: dict[str, Any] = {
    "alpha_train": None,          # required, in [0, 1)
    "n_scenarios": None,          # None = all opt draws; else the first N (nested ladder)
    "big_m_mode": "box_quantile",  # "box_quantile" | "box" | "lp_tight"
    "polish": True,               # re-solve LP with z fixed (removes integrality leak)
    "mip_node_limit": None,       # optional HiGHS node limit (deterministic stop, for audits/tests)
    "information_state": "t0_reference_only",
}
PARAM_DEFAULTS_BONFERRONI: dict[str, Any] = dict(PARAM_DEFAULTS_JOINT, alpha_allocation="equal")
#: HiGHS default ``mip_feasibility_tolerance`` (integrality); used in the residual check and report.
INT_TOL_ASSUMED = 1e-6
_BOX_PAD_REL = 1e-9
_LOOSENESS_SAMPLE = 32
_ALLOWED_OPT_STREAMS = ("opt",)
_SOLVER_NAME = "scipy.optimize.milp (HiGHS branch-and-cut)"


def _frac(a: float) -> Fraction:
    return Fraction(repr(float(a)))


def allowed_violations(alpha: float, n_scenarios: int) -> int:
    """``floor(alpha * N)`` computed exactly from the decimal representation of ``alpha``."""
    a = float(alpha)
    if not (0.0 <= a < 1.0):
        raise ValueError("alpha must be in [0, 1)")
    return int(math.floor(_frac(a) * int(n_scenarios)))


# ---------------------------------------------------------------------------------------------
# model assembly
# ---------------------------------------------------------------------------------------------

class _Stop(Exception):
    def __init__(self, status: SolveStatus, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


@dataclass(frozen=True)
class SAAModel:
    """SAA data in x-space (built by :func:`build_saa_model`; exposed for audits and tests).

    ``A_prob [N, Kp, I]``/``b_prob [Kp]``: scenario rows ``g = A_prob[s,k] @ x - b_prob[k] <= 0``.
    ``x_lo/x_hi``: box of ``x`` over the structural polytope (unpadded).  ``M [N, Kp]``: base Big-M
    valid for every ``x`` in the structural polytope (``box`` or ``lp_tight``; before ``max(., 0)``).
    ``A_s_ub, b_s_ub, A_s_eq, b_s_eq``: structural rows; ``c``: cost per kg DM.
    """

    constraint_ids_struct_ub: tuple[str, ...]
    constraint_ids_struct_eq: tuple[str, ...]
    prob_ids: tuple[str, ...]
    c: np.ndarray
    A_s_ub: np.ndarray
    b_s_ub: np.ndarray
    A_s_eq: np.ndarray
    b_s_eq: np.ndarray
    A_prob: np.ndarray
    b_prob: np.ndarray
    x_lo: np.ndarray
    x_hi: np.ndarray
    M: np.ndarray
    big_m_mode: str
    n_aux_lps: int

    @property
    def padded_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Box used to derive M (relative padding ``1e-9``; not imposed as variable bounds)."""
        lo = np.maximum(0.0, self.x_lo - _BOX_PAD_REL * np.maximum(1.0, np.abs(self.x_lo)))
        hi = self.x_hi + _BOX_PAD_REL * np.maximum(1.0, np.abs(self.x_hi))
        return lo, hi


def _nz(A):
    return A if A is not None and A.shape[0] else None


def _aux_options(opts: SolverOptions) -> SolverOptions:
    """Options for the small auxiliary LPs: same tolerances, no time limit."""
    return dataclasses.replace(opts, time_limit_s=None)


def _max_over_polytope(row: np.ndarray, model_or_rows, I: int, opts: SolverOptions) -> LPOutcome:
    A_ub, b_ub, A_eq, b_eq = model_or_rows
    return run_linprog(-np.asarray(row, float), _nz(A_ub), b_ub, _nz(A_eq), b_eq, np.zeros(I), np.full(I, np.inf),
                       opts)


def build_saa_model(problem: RationProblem, opt_draws: DrawSet, *, d_hat: Optional[np.ndarray] = None,
                    n_scenarios: Optional[int] = None, big_m_mode: str = "box_quantile",
                    solver_options: Optional[SolverOptions] = None) -> SAAModel:
    """Assemble the SAA rows, the structural box and the base Big-M (see module doc, steps 1-2).

    Raises the internal ``_Stop`` with the status to report for unbounded / infeasible structural
    polytopes, missing scenario data or auxiliary LP failures.
    """
    if big_m_mode not in BIG_M_MODES:
        raise _Stop(SolveStatus.INVALID_INPUT, f"unknown big_m_mode {big_m_mode!r} (use one of {BIG_M_MODES})")
    opts = _aux_options(solver_options or SolverOptions())
    ids = problem.ingredient_ids
    I = len(ids)
    dh = problem.dm_estimates() if d_hat is None else np.asarray(d_hat, dtype=float)
    cc = problem.compiled.subset(optimization_indices(problem.compiled))
    s_idx = [k for k, c in enumerate(cc.classes) if c is ConstraintClass.STRUCTURAL_HARD]
    p_idx = [k for k, c in enumerate(cc.classes) if c is ConstraintClass.PROBABILISTIC_NUTRITION]
    cc_s, cc_p = cc.subset(s_idx), cc.subset(p_idx)
    mu = problem.nominal_theta()
    rows_s = linear_rows(cc_s, np.where(np.isnan(mu), 0.0, mu), dh, d_hat=dh)  # W == 0 for structural rows
    lp_s = assemble_x_space_lp(rows_s.A[0], rows_s.b, rows_s.is_eq, cc_s.constraint_ids, dh, problem.price_vector())
    N = opt_draws.n_draws if n_scenarios is None else int(n_scenarios)
    rows_p = linear_rows(cc_p, opt_draws.theta[:N], opt_draws.d[:N], d_hat=dh)
    if np.any(rows_p.missing):
        bad = sorted({cc_p.constraint_ids[k] for k in np.flatnonzero(rows_p.missing.any(axis=0))})
        n_bad = int(rows_p.missing.any(axis=1).sum())
        raise _Stop(SolveStatus.INVALID_INPUT,
                    f"missing composition/DM in {n_bad} of {N} opt scenarios for rows {bad}; "
                    "SAA cannot impose undefined rows (never set to 0)")
    A_prob = rows_p.A / dh[None, None, :]
    b_prob = np.array(rows_p.b, dtype=float)
    srows = (lp_s.A_ub, lp_s.b_ub, lp_s.A_eq, lp_s.b_eq)

    lo, hi = np.zeros(I), np.zeros(I)
    n_lps = 0
    for i in range(I):
        e = np.zeros(I)
        e[i] = 1.0
        for sign in (1.0, -1.0):
            out = run_linprog(sign * e, _nz(lp_s.A_ub), lp_s.b_ub, _nz(lp_s.A_eq), lp_s.b_eq, np.zeros(I),
                              np.full(I, np.inf), opts)
            n_lps += 1
            if out.status is SolveStatus.PROVEN_INFEASIBLE:
                raise _Stop(SolveStatus.PROVEN_INFEASIBLE, "structural_hard constraints alone are infeasible "
                            f"(bound LP for {ids[i]}): {out.message}")
            if out.status is SolveStatus.INVALID_INPUT and sign < 0:
                raise _Stop(SolveStatus.INVALID_INPUT, f"x[{ids[i]}] is unbounded under the structural "
                            "constraints; Big-M needs finite bounds (declare a DM-offer rule or upper bounds)")
            if out.status is not SolveStatus.OPTIMAL:
                raise _Stop(SolveStatus.NUMERICAL_ERROR, f"bound LP for {ids[i]} ended with {out.status}: "
                            f"{out.message}")
            if sign > 0:
                lo[i] = max(0.0, float(out.x[i]))
            else:
                hi[i] = float(out.x[i])
    Kp = len(p_idx)
    tmp = SAAModel(tuple(lp_s.ub_constraint_ids), tuple(lp_s.eq_constraint_ids), tuple(cc_p.constraint_ids),
                   lp_s.c, lp_s.A_ub, lp_s.b_ub, lp_s.A_eq, lp_s.b_eq, A_prob, b_prob, lo, hi,
                   np.zeros((N, Kp)), big_m_mode, n_lps)
    lo_p, hi_p = tmp.padded_box
    if big_m_mode in ("box", "box_quantile"):
        M = big_m_from_box(A_prob, b_prob, lo_p, hi_p) if Kp else np.zeros((N, 0))
    else:  # lp_tight
        M = np.zeros((N, Kp))
        for s in range(N):
            for k in range(Kp):
                out = _max_over_polytope(A_prob[s, k], srows, I, opts)
                n_lps += 1
                if out.status is not SolveStatus.OPTIMAL:
                    raise _Stop(SolveStatus.NUMERICAL_ERROR, f"lp_tight Big-M LP (s={s}, k={k}) ended with "
                                f"{out.status}: {out.message}")
                M[s, k] = -float(out.fun) - b_prob[k]
    for a in (A_prob, b_prob, lo, hi, M):
        a.setflags(write=False)
    return SAAModel(tmp.constraint_ids_struct_ub, tmp.constraint_ids_struct_eq, tmp.prob_ids, tmp.c, tmp.A_s_ub,
                    tmp.b_s_ub, tmp.A_s_eq, tmp.b_s_eq, A_prob, b_prob, lo, hi, M, big_m_mode, n_lps)


def quantile_big_m(A_prob: np.ndarray, lo: np.ndarray, hi: np.ndarray, budgets: Sequence[int]) -> np.ndarray:
    """Quantile-strengthened Big-M ``Mq [N, Kp]`` (module doc, step 3).

    ``Mq[s,k]`` = ``(budgets[k] + 1)``-th smallest over ``t`` of
    ``D_{s,t} = max_{lo <= x <= hi} (A_prob[s,k] - A_prob[t,k]) @ x``; ``+inf`` if ``budgets[k] >= N``.
    Valid (never cuts an SAA-feasible point) whenever at most ``budgets[k]`` scenarios may violate
    row ``k``.  Memory is bounded by processing scenarios in chunks.
    """
    A = np.asarray(A_prob, dtype=float)
    N, Kp, I = A.shape
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = np.full((N, Kp), np.inf)
    chunk = max(1, int(2_000_000 // max(1, N * I)))
    for k in range(Kp):
        m = int(budgets[k])
        if m >= N:
            continue
        Ak = A[:, k, :]
        for s0 in range(0, N, chunk):
            diff = Ak[s0:s0 + chunk, None, :] - Ak[None, :, :]               # [c, N, I]
            D = np.maximum(diff * lo, diff * hi).sum(axis=2)                  # [c, N]
            out[s0:s0 + chunk, k] = np.partition(D, m, axis=1)[:, m]
    return out


def _bigm_report(model: SAAModel, M_used: np.ndarray, dropped: np.ndarray, hard: np.ndarray, bigm: np.ndarray,
                 opts: SolverOptions, polish: bool) -> tuple[dict, int]:
    """Scale / validity / looseness report of the Big-M values actually used."""
    Mb = model.M
    Mk = M_used[bigm]
    A = model.A_prob[bigm]
    nz = np.abs(A[A != 0])
    derivation = {
        "box": "M = max over the (padded) structural box of g_{s,k}; box = min/max of each x_i over the "
               "structural polytope (2I LPs)",
        "lp_tight": "M = max over the structural polytope of g_{s,k} (one LP per row)",
        "box_quantile": "M = min(M_box, Mq), Mq[s,k] = (m_k+1)-th smallest over t of max over the box of "
                        "(A[s,k]-A[t,k]) @ x; valid for every SAA-feasible point",
    }[model.big_m_mode]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = M_used[bigm] / Mb[bigm]
    rep: dict[str, Any] = {
        "mode": model.big_m_mode,
        "derivation": derivation + "; rows with base M <= 0 are dropped (hold on the whole structural polytope)",
        "x_box_lo": model.x_lo.tolist(),
        "x_box_hi": model.x_hi.tolist(),
        "n_rows_total": int(Mb.size),
        "n_rows_dropped_always_satisfied": int(dropped.sum()),
        "n_rows_hard_by_quantile": int(hard.sum()),
        "n_rows_big_m": int(bigm.sum()),
        "median_M_used_over_M_base": float(np.median(ratio)) if ratio.size else None,
        "M_max": float(Mk.max()) if Mk.size else None,
        "M_min": float(Mk.min()) if Mk.size else None,
        "M_median": float(np.median(Mk)) if Mk.size else None,
        "coef_abs_max": float(nz.max()) if nz.size else None,
        "coef_abs_min_nonzero": float(nz.min()) if nz.size else None,
        "int_tol_assumed": INT_TOL_ASSUMED,
        "int_leak_linear_max": float(INT_TOL_ASSUMED * Mk.max()) if Mk.size else 0.0,
        "polish_applied": bool(polish),
    }
    flags = []
    if Mk.size and nz.size and Mk.max() / nz.min() > 1e9:
        flags.append("dynamic_range(M_max / min|coef|) > 1e9")
    if Mk.size and INT_TOL_ASSUMED * Mk.max() > 1e-4 and not polish:
        flags.append("integrality leak > 1e-4 (linear units) without polish")
    n_lps = 0
    live = Mb > 0
    if model.big_m_mode in ("box", "box_quantile") and live.any():
        srows = (model.A_s_ub, model.b_s_ub, model.A_s_eq, model.b_s_eq)
        flat = np.argsort(-np.where(live, Mb, -np.inf), axis=None)[:min(_LOOSENESS_SAMPLE, int(live.sum()))]
        ratios, box_only = [], 0
        for f in flat:
            s, k = np.unravel_index(int(f), Mb.shape)
            out = _max_over_polytope(model.A_prob[s, k], srows, model.A_prob.shape[2], opts)
            n_lps += 1
            if out.status is not SolveStatus.OPTIMAL:
                continue
            m_lp = -float(out.fun) - model.b_prob[k]
            if Mb[s, k] < m_lp - 1e-9 * max(1.0, abs(m_lp)):
                flags.append(f"INVALID: box M below polytope max at (s={s}, k={k})")
            if m_lp > 1e-12:
                ratios.append(float(Mb[s, k] / m_lp))
            else:
                box_only += 1
        rep["box_validity_sample_size"] = int(len(flat))
        rep["box_looseness_ratio_max"] = max(ratios) if ratios else None
        rep["box_rows_positive_but_polytope_nonpositive"] = int(box_only)
    rep["flags"] = flags
    return rep, n_lps


def _run_milp_sparse(c, A_ub, b_ub, A_eq, b_eq, lb, ub, integrality, options: SolverOptions,
                     node_limit: Optional[int]) -> LPOutcome:
    """``scipy.optimize.milp`` on sparse matrices with the engine's status mapping and residual check.

    Same mapping (:func:`ration_reliability.optimization.highs._map`) and residual check as
    :func:`ration_reliability.optimization.highs.run_milp`, which densifies its inputs;
    ``node_limit`` is passed to HiGHS when given.
    """
    A_ub, A_eq = _nz(A_ub), _nz(A_eq)
    cons = []
    if A_ub is not None:
        cons.append(LinearConstraint(A_ub, -np.inf, np.asarray(b_ub, float)))
    if A_eq is not None:
        be = np.asarray(b_eq, float)
        cons.append(LinearConstraint(A_eq, be, be))
    hopts: dict[str, Any] = {"presolve": bool(options.presolve), "disp": False}
    if options.time_limit_s is not None:
        hopts["time_limit"] = float(options.time_limit_s)
    if options.mip_rel_gap is not None:
        hopts["mip_rel_gap"] = float(options.mip_rel_gap)
    if node_limit is not None:
        hopts["node_limit"] = int(node_limit)
    t0 = time.perf_counter()
    r = milp(np.asarray(c, float), constraints=cons or None, integrality=np.asarray(integrality),
             bounds=Bounds(np.asarray(lb, float), np.asarray(ub, float)), options=hopts)
    wt = time.perf_counter() - t0
    hs = H._highs_status(r.message)
    x = None if r.x is None else np.array(r.x, dtype=float)
    ra = rr = None
    ok = False
    int_err = None
    if x is not None and np.all(np.isfinite(x)):
        ra, rr = H.primal_residual(A_ub, None if A_ub is None else np.asarray(b_ub, float),
                                   A_eq, None if A_eq is None else np.asarray(b_eq, float), lb, ub, x)
        integ = np.asarray(integrality) > 0
        int_err = float(np.max(np.abs(x[integ] - np.round(x[integ])), initial=0.0))
        ok = rr <= options.residual_check_rel_tol and int_err <= INT_TOL_ASSUMED
    st = H._map(hs, int(r.status), x, ok)
    keep_x = x if st in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT) else None
    gap = getattr(r, "mip_gap", None)
    return LPOutcome(st, keep_x, None if keep_x is None else float(r.fun), None, str(r.message), hs, wt, rr, ra,
                     mip_gap=None if gap is None else float(gap),
                     extra={"mip_node_count": getattr(r, "mip_node_count", None),
                            "mip_dual_bound": getattr(r, "mip_dual_bound", None),
                            "integrality_error": int_err,
                            "raw_x": None if x is None else x.tolist()})


# ---------------------------------------------------------------------------------------------
# core solver (joint or marginal budgets)
# ---------------------------------------------------------------------------------------------

def _solve_saa(problem: RationProblem, *, method_id: str, mode: str, d_hat, opt_draws: Optional[DrawSet],
               p: Mapping[str, Any], opts: SolverOptions, row_budgets: Optional[Sequence[int]] = None,
               extra_diag: Optional[Mapping[str, Any]] = None, t0: Optional[float] = None) -> SolveResult:
    """Solve the SAA with ``mode="joint"`` (common ``z_s``, budget ``m``) or ``mode="marginal"``
    (``z_{s,k}``, budgets ``row_budgets[k]``).  Parameter validation happens in the public wrappers."""
    t0 = time.perf_counter() if t0 is None else t0
    currency = problem.prices.currency
    is_synth = bool(problem.is_synthetic or (opt_draws is not None and opt_draws.is_synthetic))
    diag: dict[str, Any] = dict(extra_diag or {})
    streams: tuple[str, ...] = ()
    input_hash = stable_hash(method_id, "pre", problem.problem_id, problem.ingredient_ids, dict(p), opts.to_dict())

    def fail(status: SolveStatus, msg: str, mip_gap: Optional[float] = None) -> SolveResult:
        # reads input_hash / streams at call time (they are refined as the build proceeds)
        return SolveResult(method_id=method_id, status=status, decision=None, objective=None,
                           objective_unit=f"{currency}/head/d", solver=_SOLVER_NAME,
                           solver_version=solver_version_string(), tolerances=_tolerance_record(opts),
                           wall_time_s=time.perf_counter() - t0, input_hash=input_hash, mip_gap=mip_gap,
                           message=msg, params=dict(p), streams_used=streams, diagnostics=diag,
                           is_synthetic=is_synth)

    if opt_draws is None:
        return fail(SolveStatus.INVALID_INPUT, f"{method_id} requires opt_draws (the SAA training scenarios)")
    require_stream(opt_draws, _ALLOWED_OPT_STREAMS, method_id)  # LeakageError, never a status
    streams = (opt_draws.stream_id,)
    ids = problem.ingredient_ids
    I = len(ids)
    if opt_draws.ingredient_ids != ids or opt_draws.nutrient_ids != problem.nutrient_ids:
        return fail(SolveStatus.INVALID_INPUT, "opt_draws labels differ from the problem")
    dh = problem.dm_estimates() if d_hat is None else np.asarray(d_hat, dtype=float)
    if I == 0 or dh.shape != (I,) or np.any(~np.isfinite(dh)) or np.any(dh <= 0) or np.any(dh > 1):
        return fail(SolveStatus.INVALID_INPUT, "d_hat must be finite in (0, 1] with shape [I]")
    N = opt_draws.n_draws if p["n_scenarios"] is None else int(p["n_scenarios"])
    if not (1 <= N <= opt_draws.n_draws):
        return fail(SolveStatus.INVALID_INPUT, f"n_scenarios must be in [1, {opt_draws.n_draws}]")
    prices = problem.price_vector()

    try:
        model = build_saa_model(problem, opt_draws, d_hat=dh, n_scenarios=N, big_m_mode=str(p["big_m_mode"]),
                                solver_options=opts)
    except _Stop as st:
        return fail(st.status, st.msg)
    Kp = len(model.prob_ids)
    alpha = float(p["alpha_train"])
    if mode == "joint":
        m = allowed_violations(alpha, N)
        budgets = [m] * Kp
        nz = N
    else:
        budgets = [int(v) for v in (row_budgets or [])]
        if len(budgets) != Kp:
            return fail(SolveStatus.INVALID_INPUT, "row_budgets must have one entry per probabilistic row")
        nz = N * Kp
    input_hash = stable_hash(method_id, "v1", problem.problem_id, ids, problem.nutrient_ids,
                             problem.compiled.fingerprint, dh, prices, dict(p), opts.to_dict(),
                             opt_draws.fingerprint, N, mode, budgets)

    lo_p, hi_p = model.padded_box
    dropped = model.M <= 0.0                                            # hold on the whole polytope
    if model.big_m_mode == "box_quantile" and Kp:
        M_used = np.minimum(model.M, quantile_big_m(model.A_prob, lo_p, hi_p, budgets))
    else:
        M_used = np.array(model.M, dtype=float)
    hard = (~dropped) & (M_used <= 0.0)                                 # hold in every SAA-feasible point
    bigm = (~dropped) & (M_used > 0.0)
    bigm_rep, n_aux = _bigm_report(model, M_used, dropped, hard, bigm, _aux_options(opts), bool(p["polish"]))

    # --- sparse MILP ---------------------------------------------------------------------
    hs_, hk_ = np.nonzero(hard)
    bs_, bk_ = np.nonzero(bigm)
    zcol = bs_ if mode == "joint" else bs_ * Kp + bk_
    blocks, rhs = [], []
    Ks_ub = model.A_s_ub.shape[0]
    if Ks_ub:
        blocks.append(sparse.hstack([sparse.csr_matrix(model.A_s_ub), sparse.csr_matrix((Ks_ub, nz))]))
        rhs.append(model.b_s_ub)
    if len(hs_):
        blocks.append(sparse.hstack([sparse.csr_matrix(model.A_prob[hs_, hk_, :]),
                                     sparse.csr_matrix((len(hs_), nz))]))
        rhs.append(model.b_prob[hk_])
    if len(bs_):
        Az = sparse.csr_matrix((-M_used[bs_, bk_], (np.arange(len(bs_)), zcol)), shape=(len(bs_), nz))
        blocks.append(sparse.hstack([sparse.csr_matrix(model.A_prob[bs_, bk_, :]), Az]))
        rhs.append(model.b_prob[bk_])
    if mode == "joint":
        blocks.append(sparse.hstack([sparse.csr_matrix((1, I)), sparse.csr_matrix(np.ones((1, nz)))]))
        rhs.append(np.array([float(budgets[0] if Kp else 0)]))
    else:
        cols = np.arange(nz)
        Bz = sparse.csr_matrix((np.ones(nz), (cols % Kp, cols)), shape=(Kp, nz))
        blocks.append(sparse.hstack([sparse.csr_matrix((Kp, I)), Bz]))
        rhs.append(np.array(budgets, dtype=float))
    A_ub_all = sparse.vstack(blocks).tocsr()
    b_ub_all = np.concatenate(rhs)
    Ke = model.A_s_eq.shape[0]
    A_eq_all = sparse.hstack([sparse.csr_matrix(model.A_s_eq), sparse.csr_matrix((Ke, nz))]).tocsr() if Ke else None
    z_used = np.zeros(nz, dtype=bool)
    z_used[zcol] = True
    lb = np.zeros(I + nz)
    ub = np.concatenate([np.full(I, np.inf), z_used.astype(float)])
    cvec = np.concatenate([model.c, np.zeros(nz)])
    integrality = np.concatenate([np.zeros(I), np.ones(nz)])
    out = _run_milp_sparse(cvec, A_ub_all, b_ub_all, A_eq_all, model.b_s_eq if Ke else None, lb, ub, integrality,
                           opts, p["mip_node_limit"])
    diag.update({
        "joint_event": ("common z_s over all probabilistic_nutrition rows of scenario s" if mode == "joint" else
                        "per-row z_{s,k} (marginal chance constraints); joint rate bounded by union bound"),
        "saa_status_scope": "status refers to the SAA instance on these opt scenarios, not the true "
                            "chance-constrained problem",
        "n_scenarios_used": N,
        "alpha_train": alpha,
        "allowed_violations": (budgets[0] if Kp else 0) if mode == "joint" else None,
        "row_budgets": None if mode == "joint" else dict(zip(model.prob_ids, budgets)),
        "imposed_constraint_ids": list(model.constraint_ids_struct_eq + model.constraint_ids_struct_ub
                                       + model.prob_ids),
        "probabilistic_constraint_ids": list(model.prob_ids),
        "big_m": bigm_rep,
        "n_aux_lps": model.n_aux_lps + n_aux,
        "milp_size": {"n_rows_ub": int(A_ub_all.shape[0]), "n_rows_eq": int(Ke), "n_x": I,
                      "n_binary_free": int(z_used.sum()), "n_binary_total": int(nz),
                      "nnz": int(A_ub_all.nnz + (A_eq_all.nnz if A_eq_all is not None else 0))},
        "highs_model_status": out.highs_model_status,
        "mip_node_count": out.extra.get("mip_node_count"),
        "mip_dual_bound_x_space": out.extra.get("mip_dual_bound"),
        "mip_integrality_error": out.extra.get("integrality_error"),
        "mip_node_limit": p["mip_node_limit"],
        "milp_wall_time_s": out.wall_time_s,
        "max_rel_residual_milp": out.max_rel_residual,
        "representation": "x_space_kg_dm; q = x / d_hat",
    })
    if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT):
        diag["raw_candidate"] = out.extra.get("raw_x")
        return fail(out.status, out.message, out.mip_gap)

    x_milp = out.x[:I]
    z = np.round(out.x[I:]).astype(int)
    relaxed = z[zcol] == 1                                   # Big-M rows switched off
    enforced = np.ones((N, Kp), dtype=bool)
    enforced[bs_[relaxed], bk_[relaxed]] = False
    x_final = x_milp
    polish_info: dict[str, Any] = {"applied": False}
    if p["polish"]:
        es, ek = np.nonzero(enforced & ~dropped)
        rows = ([model.A_s_ub] if Ks_ub else []) + ([model.A_prob[es, ek, :]] if len(es) else [])
        rb = ([model.b_s_ub] if Ks_ub else []) + ([model.b_prob[ek]] if len(es) else [])
        pol = run_linprog(model.c, np.vstack(rows) if rows else None, np.concatenate(rb) if rb else None,
                          model.A_s_eq if Ke else None, model.b_s_eq if Ke else None, np.zeros(I),
                          np.full(I, np.inf), _aux_options(opts))
        polish_info = {"applied": True, "status": str(pol.status), "objective_x_space": pol.fun,
                       "max_rel_residual": pol.max_rel_residual, "wall_time_s": pol.wall_time_s}
        if pol.status is SolveStatus.OPTIMAL:
            x_final = pol.x
            if pol.fun > out.fun + 1e-7 * max(1.0, abs(out.fun)):
                polish_info["warning"] = "polished cost above MILP cost (numerical); polished ration kept"
        else:
            polish_info["warning"] = "polish LP failed; MILP x kept (z=0 rows hold only to MILP tolerance)"
    q = np.maximum(x_final, 0.0) / dh
    decision = RationDecision(ids, q, dh, method_id, information_state=str(p["information_state"]))
    objective = float(prices @ q)

    # consistency diagnostics (solver audit, not a score): linearised rows at the final x
    g = np.einsum("ski,i->sk", model.A_prob, x_final) - model.b_prob[None, :]           # [N, Kp]
    scale = 1.0 + np.abs(model.b_prob)[None, :] + np.einsum("ski,i->sk", np.abs(model.A_prob), np.abs(x_final))
    viol_lin = g > opts.residual_check_rel_tol * scale
    diag.update({
        "milp_objective_x_space": out.fun,
        "polish": polish_info,
        "n_z_one": int(z[z_used].sum()),
        "relaxed_scenarios": (sorted(set(int(v) for v in bs_[relaxed])) if mode == "joint" else
                              {cid: sorted(int(s) for s in bs_[relaxed & (bk_ == k)])
                               for k, cid in enumerate(model.prob_ids)}),
        "n_train_scenarios_violated_linearised": int(viol_lin.any(axis=1).sum()),
        "n_enforced_rows_violated_linearised": int((viol_lin & enforced).sum()),
        "constraint_residuals_semantics": "structural: g(q); probabilistic: max over enforced (not relaxed) "
                                          "scenarios of the linearised g_{s,k}(q) (<= 0 satisfied)",
    })
    residuals: dict[str, float] = {}
    if Ks_ub:
        gs = model.A_s_ub @ x_final - model.b_s_ub
        residuals.update({cid: float(v) for cid, v in zip(model.constraint_ids_struct_ub, gs)})
    if Ke:
        ge = np.abs(model.A_s_eq @ x_final - model.b_s_eq)
        residuals.update({cid: float(v) for cid, v in zip(model.constraint_ids_struct_eq, ge)})
    for k, cid in enumerate(model.prob_ids):
        col = np.where(enforced[:, k], g[:, k], -np.inf)
        residuals[cid] = float(col.max()) if np.isfinite(col.max()) else float("nan")
    if not np.any(q > 0):
        diag["warning"] = "all-zero ration is optimal: check that a DM-offer/intake rule is declared"
    return SolveResult(method_id=method_id, status=out.status, decision=decision, objective=objective,
                       objective_unit=f"{currency}/head/d", solver=_SOLVER_NAME,
                       solver_version=solver_version_string(), tolerances=_tolerance_record(opts),
                       wall_time_s=time.perf_counter() - t0, input_hash=input_hash, mip_gap=out.mip_gap,
                       iterations=None, n_evaluations=None, message=out.message, constraint_residuals=residuals,
                       params=dict(p), streams_used=streams, diagnostics=diag, is_synthetic=is_synth)


# ---------------------------------------------------------------------------------------------
# public methods
# ---------------------------------------------------------------------------------------------

def _common_param_errors(p: Mapping[str, Any], defaults: Mapping[str, Any]) -> Optional[str]:
    unknown = sorted(set(p) - set(defaults))
    if unknown:
        return f"unknown params {unknown}"
    a = p["alpha_train"]
    if a is None:
        return "param alpha_train is required (calibrate with select_alpha_train_on_validation)"
    if isinstance(a, bool):
        return "alpha_train must be a number"
    try:
        a = float(a)
    except (TypeError, ValueError):
        return "alpha_train must be a number"
    if not (0.0 <= a < 1.0):
        return "alpha_train must be in [0, 1)"
    n = p["n_scenarios"]
    if n is not None and (isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1):
        return "n_scenarios must be None or a positive integer"
    if p["big_m_mode"] not in BIG_M_MODES:
        return f"unknown big_m_mode {p['big_m_mode']!r} (use one of {BIG_M_MODES})"
    if not isinstance(p["polish"], (bool, np.bool_)):
        return "polish must be a bool"
    nl = p["mip_node_limit"]
    if nl is not None and (isinstance(nl, bool) or not isinstance(nl, (int, np.integer)) or nl < 0):
        return "mip_node_limit must be None or an integer >= 0"
    return None


def _invalid(method_id: str, msg: str, problem: RationProblem, p, opts, opt_draws, t0) -> SolveResult:
    if opt_draws is not None:
        require_stream(opt_draws, _ALLOWED_OPT_STREAMS, method_id)   # leakage is never a status
    return SolveResult(method_id=method_id, status=SolveStatus.INVALID_INPUT, decision=None, objective=None,
                       objective_unit=f"{problem.prices.currency}/head/d", solver=_SOLVER_NAME,
                       solver_version=solver_version_string(), tolerances=_tolerance_record(opts),
                       wall_time_s=time.perf_counter() - t0,
                       input_hash=stable_hash(method_id, "pre", problem.problem_id, problem.ingredient_ids,
                                              dict(p), opts.to_dict()),
                       message=msg, params=dict(p),
                       is_synthetic=bool(problem.is_synthetic or (opt_draws is not None and opt_draws.is_synthetic)))


def _tolerance_record(opts: SolverOptions) -> dict[str, Any]:
    """Solver options plus whether the MILP gap was set explicitly (red-team C11)."""
    d = opts.to_dict()
    d["mip_rel_gap_source"] = "explicit" if opts.mip_rel_gap is not None else "solver_default_not_set_explicitly"
    return d


def solve(problem: RationProblem, *, d_hat: Optional[np.ndarray] = None, opt_draws: Optional[DrawSet] = None,
          params: Optional[Mapping[str, Any]] = None,
          solver_options: Optional[SolverOptions] = None) -> SolveResult:
    """M2 joint chance-constrained SAA (see module doc).

    Parameters (``params``; unknown keys -> ``invalid_input``)
    ----------------------------------------------------------
    alpha_train : float in [0, 1), **required** (calibrated on ``validation``, then frozen).
    n_scenarios : None (all ``opt`` draws) or N <= number of draws (first N; nested ladder).
    big_m_mode : ``"box_quantile"`` (default), ``"box"`` or ``"lp_tight"``.
    polish : bool, default True.   mip_node_limit : None or int >= 0.
    information_state : label copied into the decision.

    ``solver_options.time_limit_s`` and ``mip_rel_gap`` are passed to HiGHS; the gap reached is
    ``SolveResult.mip_gap``.  Set ``mip_rel_gap`` explicitly for research runs (HiGHS default 1e-4).
    """
    t0 = time.perf_counter()
    opts = solver_options or SolverOptions()
    p = dict(PARAM_DEFAULTS_JOINT)
    p.update(dict(params or {}))
    err = _common_param_errors(p, PARAM_DEFAULTS_JOINT)
    if err:
        return _invalid(METHOD_ID_JOINT, err, problem, p, opts, opt_draws, t0)
    return _solve_saa(problem, method_id=METHOD_ID_JOINT, mode="joint", d_hat=d_hat, opt_draws=opt_draws, p=p,
                      opts=opts, t0=t0)


def solve_marginal_bonferroni(problem: RationProblem, *, d_hat: Optional[np.ndarray] = None,
                              opt_draws: Optional[DrawSet] = None, params: Optional[Mapping[str, Any]] = None,
                              solver_options: Optional[SolverOptions] = None) -> SolveResult:
    """M2b: marginal chance-constrained SAA with Bonferroni allocation ``sum_k alpha_k <= alpha_train``.

    ``params["alpha_allocation"]``: ``"equal"`` (default, ``alpha_k = alpha_train / K_p``, budgets
    computed exactly as ``floor(alpha_train N / K_p)``) or a mapping ``{constraint_id: alpha_k}``
    covering exactly the probabilistic rows, with ``alpha_k >= 0`` and ``sum alpha_k <= alpha_train``.
    Other parameters as :func:`solve`.
    """
    t0 = time.perf_counter()
    opts = solver_options or SolverOptions()
    p = dict(PARAM_DEFAULTS_BONFERRONI)
    p.update(dict(params or {}))

    def invalid(msg: str) -> SolveResult:
        return _invalid(METHOD_ID_BONFERRONI, msg, problem, p, opts, opt_draws, t0)

    err = _common_param_errors(p, PARAM_DEFAULTS_BONFERRONI)
    if err:
        return invalid(err)
    cc = problem.compiled
    prob_ids = [cc.constraint_ids[k] for k in cc.indices(ConstraintClass.PROBABILISTIC_NUTRITION)]
    Kp = len(prob_ids)
    alpha_f = _frac(float(p["alpha_train"]))
    if p["n_scenarios"] is not None:
        N = int(p["n_scenarios"])
    else:
        N = opt_draws.n_draws if opt_draws is not None else 0
    alloc = p["alpha_allocation"]
    if isinstance(alloc, str) and alloc == "equal":
        fr = {cid: alpha_f / Kp for cid in prob_ids} if Kp else {}
        alloc_record = {cid: float(v) for cid, v in fr.items()}
        rule = "equal (alpha_train / K_p)"
    elif isinstance(alloc, Mapping):
        if set(alloc) != set(prob_ids):
            return invalid(f"alpha_allocation keys must equal the probabilistic constraint ids {sorted(prob_ids)}")
        try:
            fr = {cid: _frac(float(alloc[cid])) for cid in prob_ids}
        except (TypeError, ValueError):
            return invalid("alpha_allocation values must be numbers")
        if any(v < 0 or v >= 1 for v in fr.values()):
            return invalid("each alpha_k must be in [0, 1)")
        if sum(fr.values(), Fraction(0)) > alpha_f:
            return invalid("Bonferroni allocation violates sum_k alpha_k <= alpha_train; per-row alpha cannot "
                           "be reported as a joint alpha")
        alloc_record = {cid: float(alloc[cid]) for cid in prob_ids}
        rule = "explicit"
    else:
        return invalid("alpha_allocation must be 'equal' or a mapping {constraint_id: alpha_k}")
    budgets = [int(math.floor(fr[cid] * N)) for cid in prob_ids]
    extra = {"bonferroni_allocation": alloc_record, "bonferroni_rule": rule,
             "bonferroni_sum_alpha_k": float(sum(fr.values(), Fraction(0))),
             "training_joint_rate_bound": (sum(budgets) / N) if N else None,
             "naming_note": "marginal chance constraints with Bonferroni risk allocation; not the joint SAA"}
    return _solve_saa(problem, method_id=METHOD_ID_BONFERRONI, mode="marginal", d_hat=d_hat, opt_draws=opt_draws,
                      p=p, opts=opts, row_budgets=budgets, extra_diag=extra, t0=t0)


def select_alpha_train_on_validation(problem: RationProblem, validation_draws: DrawSet, *, opt_draws: DrawSet,
                                     grid: Sequence[float], target_alpha: float, screening_rule: str,
                                     confidence: Optional[float] = None, params: Optional[Mapping[str, Any]] = None,
                                     d_hat: Optional[np.ndarray] = None,
                                     solver_options: Optional[SolverOptions] = None,
                                     marginal_bonferroni: bool = False) -> ValidationSelection:
    """Calibrate ``alpha_train`` of M2 (or M2b) on ``validation`` draws; SAA scenarios stay ``opt``.

    Same screen and tie rules as
    :func:`~ration_reliability.optimization.safety_margin.select_parameter_on_validation`.
    """
    base = dict(params or {})
    if "alpha_train" in base:
        raise ValueError("do not pass alpha_train in params; it is the calibrated parameter")
    fn = solve_marginal_bonferroni if marginal_bonferroni else solve
    return select_parameter_on_validation(fn, problem, validation_draws, param_name="alpha_train",
                                          grid=[float(v) for v in grid], target_alpha=target_alpha,
                                          screening_rule=screening_rule, confidence=confidence, base_params=base,
                                          opt_draws=opt_draws, d_hat=d_hat, solver_options=solver_options,
                                          consumer="select_alpha_train_on_validation")
