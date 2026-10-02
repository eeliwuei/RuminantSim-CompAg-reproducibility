"""M1: calibrated safety margin on content coefficients (contract T4 M1, section 9 M1).

Margin object and direction (T4: "先定义余量作用于要求量、含量系数还是统计标准差")
--------------------------------------------------------------------------------
This implementation puts the margin on the **content coefficients** of every
``probabilistic_nutrition`` row -- never on the requirement ``K`` and never on the SD itself:

* composition ``a_ij`` (canonical, DM basis) is replaced by ``mu_ij + k * spread_ij * sgn_kij``;
* if ``apply_to_dm`` (**required**, no default), the DM fraction ``d_i`` of the row is replaced by
  ``d_hat_i + k * spread_d_i * sgn_d_ki`` ("DM 同理").

``mu`` is the nominal (table) composition ``IngredientRecord.composition`` and the DM centre is
the decision-time estimate ``d_hat`` -- i.e. the "table value + margin" practice; with ``k = 0``
the rows are exactly those of ``M0_nominal`` in ``coefficient_mode="nominal_point"``.

``spread`` is selected by ``margin_scale`` (**required**, no default; red-team fix D02: with a
default ``"sd"`` the contract grid ``{0 .. 0.10}`` silently became a ``<= 0.1 sigma`` margin, i.e. an
M1 that is M0 in disguise):

``"sd"``
    ``spread = sigma``: ``mu - k sigma`` for a lower-bound (``ge``) term, ``mu + k sigma`` for an
    upper-bound (``le``) term.  ``sigma`` is the sample SD of the ``opt`` draws
    (``sd_source="opt_draws"``, ``ddof`` = ``sd_ddof``) or given explicitly
    (``sd_source="explicit"``, ``theta_sd`` ``[I][J]`` and ``d_sd`` ``[I]`` in canonical units,
    e.g. the declared parameters of the uncertainty model).
``"relative"``
    ``spread = |mu|`` (and ``d_hat`` for DM): ``mu (1 -/+ k)``.  This is the natural reading of
    the development grid ``{0, 0.025, 0.05, 0.075, 0.10}`` in contract T4 / ``configs/protocol.yaml``
    (``methods.safety_margin_development_grid``); on the ``sd`` scale that grid would be
    ``<= 0.1 sigma``, which is not a reasonable margin (section 9: "不人为设弱"), so an ``sd`` grid
    must be fixed separately before the protocol freeze.

Direction ("按上限/下限的正确方向收紧").  Every row is ``g_k(q) = A_k(theta, d) @ q - b_k <= 0``
with ``A_ki = s_k d_i (c_ki(theta) - K_k [concentration]) (+ s_k v_ki)``, ``s_k = -1`` for ``ge`` and
``+1`` for ``le``, ``c_ki = sum_j W_kij a_ij + w0_ki``.  Since ``q_i d_i >= 0``:

* ``sgn_kij = sign(s_k W_kij)``: for ``ge`` a positively weighted content goes *down*, for ``le``
  *up*; mixed-sign rows such as ``starch - 2 fNDF <= -K3`` are handled term by term (starch up,
  forage NDF down).  Cells with ``W_kij = 0`` are not touched.
* ``sgn_d_ki = sign(s_k (c_ki^cons - K_k))`` for concentration rows (``sign(s_k c_ki^cons)`` for
  supply rows), evaluated at the already-conservative content.

Because ``g_k`` is, for each ingredient separately, bilinear in ``(d_i, a_i.)`` with ``d_i > 0``,
this choice is the exact row-wise maximum of ``g_k`` over the box
``{|a - mu| <= k spread, |d - d_hat| <= k spread_d}``: the M1 LP at ``k`` **is** the (Soyster-type)
box-robust counterpart of that box.  A robust method (M3) built on the same box ``mu +/- k sigma``
is therefore the same mathematical problem and must not be reported as a different method
(contract D06); M1 differs only in that ``k`` is calibrated on the ``validation`` stream.

Physical clipping: mass-fraction contents are clipped to ``[0, 1]`` and DM to ``[DM_FLOOR, 1]``;
``energy_density`` columns are **not clipped** (FINAL fix of the anti-conservative energy clip,
round 3).  Such a column (``NEL_fixedDMI`` of :mod:`ration_reliability.nutrition.energy`) is a
signed per-feed contribution, negative for minerals; the former clip to ``[0, inf)`` lifted a
mineral's margin-shifted (more negative) contribution on a ``ge`` energy row to 0, i.e. made the
row *weaker* than at ``k = 0``.  The exemption is the one M3 uses
(``robust._UNCLIPPED_DIMENSIONS``, review round 2 FIX_B / B-K7-1).  With ``apply_to_dm`` a negative
contributor on a ``ge`` supply row now also gets its DM moved *up* (``sgn_d = +1``; the clip at 0 had
made it ``0``).  The number of clipped cells is reported in ``diagnostics`` (clipping makes the
margin smaller than ``k spread`` in those cells).

Structural rows are never changed; ``diagnostic_only`` rows are not imposed.  If a threshold
already contains a safety factor, M1 adds conservatism on top of it; record this in the protocol
(T4 M1, "记录潜在重复保守性").

Calibration
-----------
:func:`select_margin_on_validation` solves M1 for every ``k`` of a grid, scores each ration with
the public evaluator on **validation** draws only, and returns the lowest-cost ``k`` that meets
the declared risk screen, or ``status="not_met"`` (no fallback, no use of ``test``).
:func:`select_parameter_on_validation` is the same routine for any registered method and
parameter (used for ``alpha_train`` of M2 as well).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..datamodel import (
    ConstraintClass,
    ConstraintKind,
    RationDecision,
    RationProblem,
    Sense,
    SolverOptions,
    SolveResult,
    SolveStatus,
)
from ..errors import InvalidProblemError
from ..evaluation.evaluator import evaluate_drawset
from ..evaluation.stats import one_sided_upper
from ..hashing import stable_hash
from ..nutrition.constraints import linear_rows
from ..uncertainty.base import DrawSet, require_stream
from .highs import run_linprog, solver_version_string
from .lp_builder import assemble_x_space_lp, optimization_indices
# signed columns never clipped (shared with M3; robust.py does not import this module, so no cycle)
from .robust import _UNCLIPPED_DIMENSIONS

__all__ = [
    "METHOD_ID",
    "PARAM_DEFAULTS",
    "DEVELOPMENT_GRID_RELATIVE",
    "DM_FLOOR",
    "SCREENING_RULES",
    "MarginRows",
    "margin_spread",
    "build_margin_rows",
    "solve",
    "CandidateRecord",
    "ValidationSelection",
    "select_parameter_on_validation",
    "select_margin_on_validation",
    "check_margin_grid",
]

METHOD_ID = "M1_safety_margin"
PARAM_DEFAULTS: dict[str, Any] = {
    "k": None,                      # required, >= 0
    "margin_scale": None,           # required: "sd" | "relative" (no default, red-team D02)
    "apply_to_dm": None,            # required: bool (DM margin undecided in configs/methods.yaml)
    "sd_source": "opt_draws",       # "opt_draws" | "explicit" (only for margin_scale="sd")
    "sd_ddof": 1,
    "theta_sd": None,               # [I][J] canonical units, only with sd_source="explicit"
    "d_sd": None,                   # [I], only with sd_source="explicit"
    "information_state": "t0_reference_only",
}
#: Contract T4 development grid (``configs/protocol.yaml`` methods.safety_margin_development_grid);
#: meaningful as a *relative* scale.  Not a nutrition recommendation.
DEVELOPMENT_GRID_RELATIVE: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10)
#: Lower clip of a conservative DM fraction (same order as ``IndependentNormalModel.d_lower``).
DM_FLOOR = 1e-6
#: Validation screening rules (see :func:`select_parameter_on_validation`).
SCREENING_RULES = ("rate_upper_le_alpha", "cp_upper_le_alpha")
_ALLOWED_OPT_STREAMS = ("opt",)


def _equivalences(method_id: str):
    from . import equivalence_annotations   # local import: the package imports modules lazily
    return equivalence_annotations(method_id)


# ---------------------------------------------------------------------------------------------
# spread and conservative rows
# ---------------------------------------------------------------------------------------------

def margin_spread(problem: RationProblem, *, margin_scale: str, sd_source: str = "opt_draws",
                  opt_draws: Optional[DrawSet] = None, theta_sd=None, d_sd=None, ddof: int = 1,
                  d_hat: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(theta_spread [I, J], d_spread [I], info)`` for the chosen scale.

    ``NaN`` spread means "unknown"; it is an error only where a probabilistic row needs it
    (checked in :func:`build_margin_rows`).  Raises :class:`InvalidProblemError` for bad input and
    :class:`~ration_reliability.errors.LeakageError` for non-``opt`` draws.
    """
    I, J = len(problem.ingredient_ids), len(problem.nutrient_ids)
    dh = problem.dm_estimates() if d_hat is None else np.asarray(d_hat, dtype=float)
    info: dict[str, Any] = {"margin_scale": margin_scale}
    if margin_scale == "relative":
        info["spread_source"] = "abs(nominal composition); d_hat for DM"
        return np.abs(problem.nominal_theta()), dh.copy(), info
    if margin_scale != "sd":
        raise InvalidProblemError(f"unknown margin_scale {margin_scale!r} (use 'sd' or 'relative')")
    if sd_source == "opt_draws":
        if opt_draws is None:
            raise InvalidProblemError("margin_scale='sd' with sd_source='opt_draws' requires opt_draws")
        require_stream(opt_draws, _ALLOWED_OPT_STREAMS, METHOD_ID)
        if opt_draws.ingredient_ids != problem.ingredient_ids or opt_draws.nutrient_ids != problem.nutrient_ids:
            raise InvalidProblemError("opt_draws labels differ from the problem")
        if opt_draws.n_draws < int(ddof) + 1:
            raise InvalidProblemError(f"need at least {int(ddof) + 1} opt draws to estimate an SD")
        th_sd = np.std(opt_draws.theta, axis=0, ddof=int(ddof))   # NaN where any draw is NaN
        dd_sd = np.std(opt_draws.d, axis=0, ddof=int(ddof))
        # constant cells (deterministic inputs) get an exact 0 instead of a 1e-13 rounding residue
        th_sd = np.where(np.ptp(opt_draws.theta, axis=0) == 0, 0.0, th_sd)
        dd_sd = np.where(np.ptp(opt_draws.d, axis=0) == 0, 0.0, dd_sd)
        info.update(spread_source="sample SD of opt draws", sd_ddof=int(ddof), n_opt_draws=opt_draws.n_draws,
                    opt_stream_id=opt_draws.stream_id)
        return th_sd, dd_sd, info
    if sd_source == "explicit":
        if theta_sd is None or d_sd is None:
            raise InvalidProblemError("sd_source='explicit' requires params theta_sd [I][J] and d_sd [I]")
        th_sd = np.array(theta_sd, dtype=float)
        dd_sd = np.array(d_sd, dtype=float)
        if th_sd.shape != (I, J) or dd_sd.shape != (I,):
            raise InvalidProblemError(f"theta_sd must be [{I}][{J}] and d_sd [{I}]")
        if np.any(th_sd[np.isfinite(th_sd)] < 0) or np.any(dd_sd[np.isfinite(dd_sd)] < 0):
            raise InvalidProblemError("explicit SDs must be >= 0")
        info["spread_source"] = "explicit SD (params theta_sd, d_sd)"
        return th_sd, dd_sd, info
    raise InvalidProblemError(f"unknown sd_source {sd_source!r} (use 'opt_draws' or 'explicit')")


@dataclass(frozen=True)
class MarginRows:
    """Rows of the M1 LP in q-space (structural rows unchanged, probabilistic rows tightened).

    ``theta_states[cid]`` / ``d_states[cid]`` hold the conservative composition ``[I, J]`` and DM
    ``[I]`` used for probabilistic row ``cid``; ``missing`` lists ``constraint:ingredient:item``
    cells whose centre or spread is unknown (the LP must then not be solved).
    """

    constraint_ids: tuple[str, ...]
    A_q: np.ndarray
    b: np.ndarray
    is_eq: np.ndarray
    probabilistic_ids: tuple[str, ...]
    theta_states: Mapping[str, np.ndarray]
    d_states: Mapping[str, np.ndarray]
    n_theta_clipped: int
    n_d_clipped: int
    missing: tuple[str, ...]


def build_margin_rows(problem: RationProblem, k: float, theta_spread: np.ndarray, d_spread: np.ndarray, *,
                      apply_to_dm: bool, d_hat: Optional[np.ndarray] = None) -> MarginRows:
    """Tightened linear rows for margin ``k`` (see module doc for object and direction)."""
    k = float(k)
    if not (np.isfinite(k) and k >= 0):
        raise InvalidProblemError("margin k must be finite and >= 0")
    cc_all = problem.compiled
    cc = cc_all.subset(optimization_indices(cc_all))
    ids, nids = problem.ingredient_ids, problem.nutrient_ids
    I, J = len(ids), len(nids)
    dh = problem.dm_estimates() if d_hat is None else np.asarray(d_hat, dtype=float)
    mu = problem.nominal_theta()
    th_sp = np.asarray(theta_spread, dtype=float)
    d_sp = np.asarray(d_spread, dtype=float)
    if th_sp.shape != (I, J) or d_sp.shape != (I,):
        raise InvalidProblemError("spread shapes do not match the problem")
    # per-column physical clip: mass fractions [0, 1]; signed columns (energy_density) unclipped -- FINAL fix of the
    # anti-conservative energy clip (module doc), same exemption as M3 (robust._UNCLIPPED_DIMENSIONS)
    lower = np.array([-np.inf if n.dimension in _UNCLIPPED_DIMENSIONS else 0.0 for n in problem.nutrients])
    upper = np.array([1.0 if n.dimension == "mass_fraction" else np.inf for n in problem.nutrients])

    K = cc.n_constraints
    base = linear_rows(cc, np.where(np.isnan(mu), 0.0, mu), dh, d_hat=dh)  # structural rows (W == 0)
    A = np.array(base.A[0], dtype=float)
    b = np.array(base.b, dtype=float)
    is_eq = np.array(base.is_eq, dtype=bool)
    theta_states: dict[str, np.ndarray] = {}
    d_states: dict[str, np.ndarray] = {}
    missing: list[str] = []
    n_th_clip = n_d_clip = 0
    prob_ids = []
    for kk in range(K):
        if cc.classes[kk] is not ConstraintClass.PROBABILISTIC_NUTRITION:
            continue
        cid = cc.constraint_ids[kk]
        prob_ids.append(cid)
        s = -1.0 if cc.senses[kk] is Sense.GE else 1.0   # eq is rejected at compile time
        Wk = cc.W[kk]
        need = Wk != 0
        bad = need & (np.isnan(mu) | (np.isnan(th_sp) & (k > 0)))
        for i, j in np.argwhere(bad):
            missing.append(f"{cid}:{ids[i]}:{nids[j]}")
        sgn = np.sign(s * Wk)
        th = np.array(mu, dtype=float)
        if k > 0:
            shift = np.where(need, k * np.nan_to_num(th_sp, nan=0.0) * sgn, 0.0)
            th = np.where(need, th + shift, th)
            clipped = np.clip(th, lower[None, :], upper[None, :])
            n_th_clip += int(np.sum(need & np.isfinite(th) & (clipped != th)))
            th = np.where(np.isfinite(th), clipped, th)
        c = (Wk * np.where(np.isnan(th), 0.0, th)).sum(axis=1) + cc.w0[kk]
        dd = np.array(dh, dtype=float)
        if apply_to_dm and k > 0:
            if cc.kinds[kk] is ConstraintKind.CONCENTRATION:
                slope = s * (c - cc.bound[kk])
            else:
                slope = s * c
            sgn_d = np.sign(slope)
            bad_d = (sgn_d != 0) & np.isnan(d_sp)
            for i in np.flatnonzero(bad_d):
                missing.append(f"{cid}:{ids[i]}:DM")
            dd = dh + k * np.nan_to_num(d_sp, nan=0.0) * sgn_d
            clipped_d = np.clip(dd, DM_FLOOR, 1.0)
            n_d_clip += int(np.sum(clipped_d != dd))
            dd = clipped_d
        theta_states[cid] = th
        d_states[cid] = dd
        lr = linear_rows(cc.subset([kk]), th, dd, d_hat=dh)
        A[kk] = lr.A[0, 0]
        b[kk] = lr.b[0]
    for arr in list(theta_states.values()) + list(d_states.values()):
        arr.setflags(write=False)
    A.setflags(write=False)
    return MarginRows(tuple(cc.constraint_ids), A, b, is_eq, tuple(prob_ids), theta_states, d_states,
                      n_th_clip, n_d_clip, tuple(sorted(set(missing))))


# ---------------------------------------------------------------------------------------------
# method
# ---------------------------------------------------------------------------------------------

def _fail(status: SolveStatus, msg: str, t0: float, input_hash: str, params: Mapping[str, Any],
          opts: SolverOptions, is_synth: bool, currency: str, streams=(), diagnostics=None) -> SolveResult:
    return SolveResult(method_id=METHOD_ID, status=status, decision=None, objective=None,
                       objective_unit=f"{currency}/head/d", solver="scipy.optimize.linprog(method='highs')",
                       solver_version=solver_version_string(), tolerances=opts.to_dict(),
                       wall_time_s=time.perf_counter() - t0, input_hash=input_hash, message=msg,
                       params=dict(params), streams_used=tuple(streams), diagnostics=dict(diagnostics or {}),
                       is_synthetic=is_synth)


def solve(problem: RationProblem, *, d_hat: Optional[np.ndarray] = None, opt_draws: Optional[DrawSet] = None,
          params: Optional[Mapping[str, Any]] = None,
          solver_options: Optional[SolverOptions] = None) -> SolveResult:
    """Least-cost LP with content-coefficient safety margin ``params["k"]``.

    Parameters (``params``; unknown keys -> ``invalid_input``)
    ----------------------------------------------------------
    k : float >= 0, **required**.
    margin_scale : ``"sd"`` or ``"relative"``, **required** (no default).
    apply_to_dm : bool, **required** (does the DM fraction get the margin too).
    sd_source : ``"opt_draws"`` (default; needs ``opt`` draws) or ``"explicit"``.
    sd_ddof : int, default 1.   theta_sd, d_sd : explicit SDs (canonical units).
    information_state : label copied into the decision.

    Only ``structural_hard`` and ``probabilistic_nutrition`` constraints are imposed; the LP is
    solved in DM space and executed as ``q = x / d_hat``.  Failures carry no ration.
    """
    t0 = time.perf_counter()
    opts = solver_options or SolverOptions()
    p = dict(PARAM_DEFAULTS)
    p.update(dict(params or {}))
    is_synth = bool(problem.is_synthetic or (opt_draws is not None and opt_draws.is_synthetic))
    currency = problem.prices.currency
    pre_hash = stable_hash(METHOD_ID, "pre", problem.problem_id, problem.ingredient_ids, p, opts.to_dict())

    def bad(msg: str, streams=()) -> SolveResult:
        return _fail(SolveStatus.INVALID_INPUT, msg, t0, pre_hash, p, opts, is_synth, currency, streams)

    if opt_draws is not None:
        # hygiene first: whatever the mode or parameters, a method may only ever receive opt draws
        # (LeakageError otherwise, before any parameter check)
        require_stream(opt_draws, _ALLOWED_OPT_STREAMS, METHOD_ID)
    unknown = sorted(set(p) - set(PARAM_DEFAULTS))
    if unknown:
        return bad(f"unknown params {unknown}")
    try:
        k = float(p["k"]) if p["k"] is not None else None
    except (TypeError, ValueError):
        return bad("param k must be a number")
    if k is None:
        return bad("param k (margin multiplier) is required; select it with select_margin_on_validation")
    if not (np.isfinite(k) and k >= 0):
        return bad("param k must be finite and >= 0")
    if p["margin_scale"] is None:
        return bad("param margin_scale is required ('sd' | 'relative'); it has no default because the scale decides "
                   "what the grid means (red-team D02; configs/methods.yaml M1 margin_semantics)")
    if p["margin_scale"] not in ("sd", "relative"):
        return bad(f"unknown margin_scale {p['margin_scale']!r}")
    if p["apply_to_dm"] is None:
        return bad("param apply_to_dm (bool) is required; the DM margin is an open protocol decision "
                   "(configs/methods.yaml M1 dm_margin)")
    if not isinstance(p["apply_to_dm"], (bool, np.bool_)):
        return bad("apply_to_dm must be a bool")
    if p["sd_ddof"] not in (0, 1) or isinstance(p["sd_ddof"], bool):
        return bad("sd_ddof must be 0 or 1")
    if p["sd_source"] not in ("opt_draws", "explicit"):
        return bad(f"unknown sd_source {p['sd_source']!r}")
    uses_draws = p["margin_scale"] == "sd" and p["sd_source"] == "opt_draws"
    explicit = p["margin_scale"] == "sd" and p["sd_source"] == "explicit"
    if not explicit and (p["theta_sd"] is not None or p["d_sd"] is not None):
        return bad("theta_sd/d_sd are only used with margin_scale='sd' and sd_source='explicit'")
    try:
        ids = problem.ingredient_ids
        prices = problem.price_vector()
        dh = problem.dm_estimates() if d_hat is None else np.asarray(d_hat, dtype=float)
        if dh.shape != (len(ids),) or np.any(~np.isfinite(dh)) or np.any(dh <= 0) or np.any(dh > 1):
            raise InvalidProblemError("d_hat must be finite in (0, 1] with shape [I]")
        if len(ids) == 0:
            raise InvalidProblemError("no ingredients")
        th_sp, d_sp, sp_info = margin_spread(problem, margin_scale=p["margin_scale"], sd_source=p["sd_source"],
                                             opt_draws=opt_draws if uses_draws else None,
                                             theta_sd=p["theta_sd"], d_sd=p["d_sd"], ddof=int(p["sd_ddof"]),
                                             d_hat=dh)
    except InvalidProblemError as exc:
        return bad(str(exc))
    # LeakageError from margin_spread (non-opt draws) propagates: it is never a status.
    streams = (opt_draws.stream_id,) if uses_draws else ()
    draws_fp = opt_draws.fingerprint if uses_draws else None

    mr = build_margin_rows(problem, k, th_sp, d_sp, d_hat=dh, apply_to_dm=bool(p["apply_to_dm"]))
    cc = problem.compiled.subset(optimization_indices(problem.compiled))
    input_hash = stable_hash(METHOD_ID, "v1", problem.problem_id, ids, problem.nutrient_ids, cc.fingerprint,
                             mr.A_q, mr.b, mr.is_eq, dh, prices, p, opts.to_dict(), draws_fp)
    if mr.missing:
        return _fail(SolveStatus.INVALID_INPUT,
                     "missing nominal value or spread (constraint:ingredient:item) " + ", ".join(mr.missing),
                     t0, input_hash, p, opts, is_synth, currency, streams)

    lp = assemble_x_space_lp(mr.A_q, mr.b, mr.is_eq, mr.constraint_ids, dh, prices)
    out = run_linprog(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, lp.ub, opts)
    diag: dict[str, Any] = {
        "margin_object": "content coefficients a_ij" + (" and DM d_i" if p["apply_to_dm"] else "")
                         + " of probabilistic_nutrition rows; requirement K unchanged",
        "margin_direction": "per row and term: a -> mu + k*spread*sign(s_k W_kij), s=-1 (ge) / +1 (le); "
                            "d -> d_hat + k*spread_d*sign(s_k (c_ki - K_k))",
        "equivalent_methods": [r["pair_id"] for r in _equivalences(METHOD_ID)],
        "equivalence_note": "row-wise worst case over the box mu +/- k*spread = box-robust counterpart of "
                            "that box (same problem as a box-robust M3 on the same box)",
        "double_conservatism_note": "check whether constraint thresholds already contain a safety factor",
        "spread": sp_info,
        "n_theta_cells_clipped": mr.n_theta_clipped,
        "n_dm_cells_clipped": mr.n_d_clipped,
        "highs_model_status": out.highs_model_status,
        "resolved_without_presolve": out.resolved_without_presolve,
        "max_rel_residual_x_space": out.max_rel_residual,
        "max_abs_residual_x_space": out.max_abs_residual,
        "imposed_constraint_ids": list(mr.constraint_ids),
        "tightened_constraint_ids": list(mr.probabilistic_ids),
        "solver_wall_time_s": out.wall_time_s,
        "representation": "x_space_kg_dm; q = x / d_hat",
    }
    if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT):
        diag["raw_candidate_x"] = out.extra.get("raw_x")
        return _fail(out.status, out.message, t0, input_hash, p, opts, is_synth, currency, streams, diag)

    q = out.x / dh
    decision = RationDecision(ids, q, dh, METHOD_ID, information_state=str(p["information_state"]))
    objective = float(prices @ q)
    diag["lp_objective_x_space"] = out.fun
    diag["total_q_as_fed_kg"] = float(q.sum())
    if not np.any(q > 0):
        diag["warning"] = "all-zero ration is optimal: check that a DM-offer/intake rule is declared"
    g = mr.A_q @ q - mr.b
    g = np.where(mr.is_eq, np.abs(g), g)
    residuals = {cid: float(v) for cid, v in zip(mr.constraint_ids, g)}
    return SolveResult(method_id=METHOD_ID, status=out.status, decision=decision, objective=objective,
                       objective_unit=f"{currency}/head/d", solver="scipy.optimize.linprog(method='highs')",
                       solver_version=solver_version_string(), tolerances=opts.to_dict(),
                       wall_time_s=time.perf_counter() - t0, input_hash=input_hash, mip_gap=None,
                       iterations=out.iterations, n_evaluations=None, message=out.message,
                       constraint_residuals=residuals, params=p, streams_used=streams, diagnostics=diag,
                       is_synthetic=is_synth)


# ---------------------------------------------------------------------------------------------
# validation-stream selection (generic; used for M1 k and M2 alpha_train)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateRecord:
    """One grid point scored on the validation stream (``solve_result`` is kept for audit)."""

    value: Any
    status: str
    cost: Optional[float]
    n_draws: int
    n_violated: Optional[int]
    n_unknown: Optional[int]
    rate_lower: Optional[float]
    rate_upper: Optional[float]
    screen_statistic: Optional[float]
    structural_ok: Optional[bool]
    meets_screen: bool
    solve_result: SolveResult

    def to_dict(self) -> dict[str, Any]:
        """Plain dict without the embedded :class:`SolveResult`."""
        return {"value": self.value, "status": self.status, "cost": self.cost, "n_draws": self.n_draws,
                "n_violated": self.n_violated, "n_unknown": self.n_unknown, "rate_lower": self.rate_lower,
                "rate_upper": self.rate_upper, "screen_statistic": self.screen_statistic,
                "structural_ok": self.structural_ok, "meets_screen": self.meets_screen,
                "input_hash": self.solve_result.input_hash}


@dataclass(frozen=True)
class ValidationSelection:
    """Result of a validation-stream selection.

    ``status`` is ``"selected"`` or ``"not_met"`` (no grid point met the screen; ``selected_value``
    and ``selected_result`` are then ``None`` -- nothing is tuned further and ``test`` is never used).
    ``selected_result`` is the chosen :class:`SolveResult` with the grid, the criterion and the
    validation stream id written into ``params["selection"]`` and ``streams_used``.
    """

    method_id: str
    param_name: str
    grid: tuple[Any, ...]
    target_alpha: float
    screening_rule: str
    confidence: Optional[float]
    validation_stream_id: str
    validation_n_draws: int
    status: str
    selected_value: Any
    selected_result: Optional[SolveResult]
    candidates: tuple[CandidateRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {"method_id": self.method_id, "param_name": self.param_name, "grid": list(self.grid),
                "target_alpha": self.target_alpha, "screening_rule": self.screening_rule,
                "confidence": self.confidence, "validation_stream_id": self.validation_stream_id,
                "validation_n_draws": self.validation_n_draws, "status": self.status,
                "selected_value": self.selected_value,
                "candidates": [c.to_dict() for c in self.candidates]}


def select_parameter_on_validation(solve_fn: Callable[..., SolveResult], problem: RationProblem,
                                   validation_draws: DrawSet, *, param_name: str, grid: Sequence[Any],
                                   target_alpha: float, screening_rule: str, confidence: Optional[float] = None,
                                   base_params: Optional[Mapping[str, Any]] = None,
                                   opt_draws: Optional[DrawSet] = None, d_hat: Optional[np.ndarray] = None,
                                   solver_options: Optional[SolverOptions] = None,
                                   consumer: str = "select_parameter_on_validation") -> ValidationSelection:
    """Pick the lowest-cost grid value whose ration meets the risk screen on ``validation`` draws.

    For each value ``v`` in ``grid`` (all are solved; no early stop), ``solve_fn`` is called with
    ``params = {**base_params, param_name: v}`` and ``opt_draws``; a ration is scored by the public
    evaluator on ``validation_draws`` (joint event ``I`` over ``probabilistic_nutrition`` rows).

    A candidate *meets the screen* iff it has a ration, all structural constraints hold, and

    * ``screening_rule="rate_upper_le_alpha"``: ``(n_violated + n_unknown) / S <= target_alpha``;
    * ``screening_rule="cp_upper_le_alpha"``: the exact one-sided Clopper-Pearson upper bound at
      ``confidence`` of ``(n_violated + n_unknown)`` in ``S`` draws is ``<= target_alpha``
      (Monte Carlo error of the fixed declared distribution only, T8.2/T8.3).

    Unknown (missing-data) draws are counted against the candidate in the screen (conservative);
    both rates are recorded.  Among candidates meeting the screen the lowest cost wins; ties go to
    the lower ``rate_upper``, then to the earlier grid position.  If none meets the screen, the
    result has ``status="not_met"``.  ``target_alpha`` and ``screening_rule`` have no defaults:
    they are protocol decisions.

    Raises :class:`~ration_reliability.errors.LeakageError` unless ``validation_draws`` come from
    the ``validation`` stream (``test`` is never accepted).
    """
    require_stream(validation_draws, ("validation",), consumer)
    if screening_rule not in SCREENING_RULES:
        raise ValueError(f"screening_rule must be one of {SCREENING_RULES}")
    a = float(target_alpha)
    if not (0.0 <= a < 1.0):
        raise ValueError("target_alpha must be in [0, 1)")
    if screening_rule == "cp_upper_le_alpha":
        if confidence is None or not (0.0 < float(confidence) < 1.0):
            raise ValueError("cp_upper_le_alpha needs confidence in (0, 1)")
    grid = tuple(grid)
    if not grid:
        raise ValueError("empty grid")
    if solve_fn is solve and param_name == "k":
        check_margin_grid(grid, (base_params or {}).get("margin_scale"))
    if validation_draws.ingredient_ids != problem.ingredient_ids or \
            validation_draws.nutrient_ids != problem.nutrient_ids:
        raise ValueError("validation draws labels differ from the problem")
    S = validation_draws.n_draws
    base = dict(base_params or {})
    records: list[CandidateRecord] = []
    for v in grid:
        prm = dict(base)
        prm[param_name] = v
        res = solve_fn(problem, d_hat=d_hat, opt_draws=opt_draws, params=prm, solver_options=solver_options)
        if not res.has_solution:
            records.append(CandidateRecord(v, str(res.status), None, S, None, None, None, None, None, None,
                                           False, res))
            continue
        ev = evaluate_drawset(res.decision, validation_draws, problem.compiled, prices=problem.prices)
        nv = int(ev.joint_violation.sum())
        nu = int(ev.joint_unknown.sum())
        lower, upper = nv / S, (nv + nu) / S
        stat = upper if screening_rule == "rate_upper_le_alpha" else one_sided_upper(nv + nu, S, float(confidence))
        meets = bool(ev.structural_ok and stat <= a)
        records.append(CandidateRecord(v, str(res.status), ev.cost, S, nv, nu, lower, upper, float(stat),
                                       bool(ev.structural_ok), meets, res))
    ok = [(r.cost, r.rate_upper, pos, r) for pos, r in enumerate(records) if r.meets_screen]
    method_id = records[0].solve_result.method_id
    if not ok:
        return ValidationSelection(method_id, param_name, grid, a, screening_rule,
                                   None if confidence is None else float(confidence), validation_draws.stream_id,
                                   S, "not_met", None, None, tuple(records))
    ok.sort(key=lambda t: (t[0], t[1], t[2]))
    best = ok[0][3]
    sel_info = {
        "param_name": param_name,
        "candidate_grid": list(grid),
        "criterion": f"lowest cost among grid values with {screening_rule} (target_alpha={a}"
                     + (f", confidence={float(confidence)}" if screening_rule == "cp_upper_le_alpha" else "")
                     + "); unknown draws counted as violations; ties -> lower rate_upper, then grid order",
        "screening_rule": screening_rule,
        "target_alpha": a,
        "confidence": None if confidence is None else float(confidence),
        "selected_on_stream": validation_draws.stream_id,
        "validation_n_draws": S,
        "selected_value": best.value,
        "candidate_table": [r.to_dict() for r in records],
    }
    res = best.solve_result
    params_new = dict(res.params)
    params_new["selection"] = sel_info
    streams_new = tuple(res.streams_used) + (validation_draws.stream_id,)
    selected = dataclasses.replace(res, params=params_new, streams_used=streams_new)
    return ValidationSelection(method_id, param_name, grid, a, screening_rule,
                               None if confidence is None else float(confidence), validation_draws.stream_id, S,
                               "selected", best.value, selected, tuple(records))


def check_margin_grid(grid: Sequence[Any], margin_scale: Any) -> None:
    """Validator for an M1 calibration grid (red-team fix D02).

    * ``margin_scale`` must be declared (``"sd"`` or ``"relative"``);
    * the contract development grid ``{0, 0.025, 0.05, 0.075, 0.10}`` is a *relative* grid (T4,
      ``configs/methods.yaml`` M1 ``coef_directional``): on the ``sd`` scale a grid whose largest
      multiplier is ``<= max(DEVELOPMENT_GRID_RELATIVE)`` would only test ``<= 0.1 sigma`` margins and
      is rejected -- an ``sd`` grid must be declared separately before the protocol freeze;
    * every grid value must be finite and ``>= 0``.
    """
    if margin_scale not in ("sd", "relative"):
        raise ValueError("M1 grid: params['margin_scale'] must be declared as 'sd' or 'relative' (no default)")
    try:
        vals = [float(v) for v in grid]
    except (TypeError, ValueError):
        raise ValueError("M1 grid values must be numbers") from None
    if not vals or any((not np.isfinite(v)) or v < 0 for v in vals):
        raise ValueError("M1 grid values must be finite and >= 0")
    if margin_scale == "sd" and max(vals) <= max(DEVELOPMENT_GRID_RELATIVE) + 1e-15:
        raise ValueError(f"M1 grid {tuple(vals)} on margin_scale='sd' tests at most {max(vals):g} sigma: the contract "
                         "development grid is a relative grid (use margin_scale='relative') or declare an sd grid "
                         "separately before the freeze")


def select_margin_on_validation(problem: RationProblem, validation_draws: DrawSet, *, grid: Sequence[float],
                                target_alpha: float, screening_rule: str, confidence: Optional[float] = None,
                                params: Optional[Mapping[str, Any]] = None, opt_draws: Optional[DrawSet] = None,
                                d_hat: Optional[np.ndarray] = None,
                                solver_options: Optional[SolverOptions] = None) -> ValidationSelection:
    """Calibrate the M1 multiplier ``k`` over ``grid`` on ``validation`` draws.

    ``params`` are the other M1 parameters (e.g. ``{"margin_scale": "sd"}``); ``opt_draws`` supply
    the SD when ``sd_source="opt_draws"``.  See :func:`select_parameter_on_validation`.
    """
    base = dict(params or {})
    if "k" in base:
        raise ValueError("do not pass k in params; it is the calibrated parameter")
    return select_parameter_on_validation(solve, problem, validation_draws, param_name="k",
                                          grid=[float(v) for v in grid], target_alpha=target_alpha,
                                          screening_rule=screening_rule, confidence=confidence, base_params=base,
                                          opt_draws=opt_draws, d_hat=d_hat, solver_options=solver_options,
                                          consumer="select_margin_on_validation")
