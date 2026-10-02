"""M0: nominal least-cost ration (contract section 9 M0, T4).

The LP is solved in DM space ``x`` (kg DM/head/d) with HiGHS and converted to the executed
decision ``q = x / d_hat`` (kg as-fed/head/d) using the decision-time DM estimate.

Coefficient modes (``params["coefficient_mode"]``)
--------------------------------------------------
``nominal_point`` (default)
    Rows evaluated at the nominal state ``theta = IngredientRecord.composition``, ``d = d_hat``.
``draw_mean``
    Rows averaged over ``opt`` draws: ``E[g_k(q)] = mean_s(A[s,k,:]) @ q - b_k``.  Because the
    rows contain the products ``d_i a_ij`` per draw, this uses the joint mean of the products,
    not ``E[d] E[a]`` (T4).  Only the ``opt`` stream is accepted (:class:`LeakageError` otherwise).
    With zero uncertainty centred on the nominal state both modes define the same LP; they must
    then be reported as one method (contract section 9, "均值方法的等价性").

Only ``structural_hard`` and ``probabilistic_nutrition`` constraints are imposed.  Infeasible or
failed problems return no ration and no cost; nothing is relaxed automatically.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import numpy as np

from ..datamodel import RationDecision, RationProblem, SolverOptions, SolveResult, SolveStatus
from ..errors import InvalidProblemError
from ..hashing import stable_hash
from ..nutrition.constraints import linear_rows
from ..uncertainty.base import DrawSet, require_stream
from .highs import run_linprog, solver_version_string
from .lp_builder import assemble_x_space_lp, optimization_indices

__all__ = ["METHOD_ID", "PARAM_DEFAULTS", "solve"]

METHOD_ID = "M0_nominal"
PARAM_DEFAULTS: dict[str, Any] = {"coefficient_mode": "nominal_point", "information_state": "t0_reference_only"}
_ALLOWED_OPT_STREAMS = ("opt",)


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
    """Solve the nominal least-cost LP.

    Parameters
    ----------
    problem : the shared :class:`RationProblem`.
    d_hat : decision-time DM estimates ``[I]``; default ``problem.dm_estimates()``.
    opt_draws : ``opt``-stream draws, required only for ``coefficient_mode="draw_mean"``.
    params : ``{"coefficient_mode": ..., "information_state": ...}``.
    solver_options : :class:`SolverOptions` (time limit, tolerances, presolve).

    Returns
    -------
    SolveResult
        ``decision`` / ``objective`` only for ``optimal`` or ``feasible_time_limit``.
    """
    t0 = time.perf_counter()
    opts = solver_options or SolverOptions()
    p = dict(PARAM_DEFAULTS)
    p.update(dict(params or {}))
    is_synth = bool(problem.is_synthetic or (opt_draws is not None and opt_draws.is_synthetic))
    currency = problem.prices.currency
    unknown = sorted(set(p) - set(PARAM_DEFAULTS))
    pre_hash = stable_hash(METHOD_ID, "pre", problem.problem_id, problem.ingredient_ids, p, opts.to_dict())
    if unknown:
        return _fail(SolveStatus.INVALID_INPUT, f"unknown params {unknown}", t0, pre_hash, p, opts, is_synth, currency)
    mode = p["coefficient_mode"]
    if mode not in ("nominal_point", "draw_mean"):
        return _fail(SolveStatus.INVALID_INPUT, f"unknown coefficient_mode {mode!r}", t0, pre_hash, p, opts,
                     is_synth, currency)

    try:
        cc_all = problem.compiled
        ids = problem.ingredient_ids
        prices = problem.price_vector()
        dh = problem.dm_estimates() if d_hat is None else np.asarray(d_hat, dtype=float)
        if dh.shape != (len(ids),) or np.any(~np.isfinite(dh)) or np.any(dh <= 0) or np.any(dh > 1):
            raise InvalidProblemError("d_hat must be finite in (0, 1] with shape [I]")
        if len(ids) == 0:
            raise InvalidProblemError("no ingredients")
    except InvalidProblemError as exc:
        return _fail(SolveStatus.INVALID_INPUT, str(exc), t0, pre_hash, p, opts, is_synth, currency)

    idx = optimization_indices(cc_all)
    cc = cc_all.subset(idx)
    streams: tuple[str, ...] = ()
    if mode == "nominal_point":
        rows = linear_rows(cc, problem.nominal_theta(), dh, d_hat=dh)
        draws_fp = None
    else:
        if opt_draws is None:
            return _fail(SolveStatus.INVALID_INPUT, "coefficient_mode='draw_mean' requires opt_draws", t0,
                         pre_hash, p, opts, is_synth, currency)
        # raises LeakageError for validation/test/outer draws (never downgraded to a status)
        require_stream(opt_draws, _ALLOWED_OPT_STREAMS, METHOD_ID)
        if opt_draws.ingredient_ids != ids or opt_draws.nutrient_ids != problem.nutrient_ids:
            return _fail(SolveStatus.INVALID_INPUT, "opt_draws labels differ from the problem", t0, pre_hash, p,
                         opts, is_synth, currency)
        rows = linear_rows(cc, opt_draws.theta, opt_draws.d, d_hat=dh).mean()
        draws_fp = opt_draws.fingerprint
        streams = (opt_draws.stream_id,)

    input_hash = stable_hash(METHOD_ID, "v1", problem.problem_id, ids, problem.nutrient_ids, cc.fingerprint,
                             rows.A, rows.b, rows.is_eq, dh, prices, p, opts.to_dict(), draws_fp)
    A_q = rows.A[0]
    if np.any(rows.missing[0]):
        bad = []
        nan_cells = np.argwhere(np.isnan(A_q))
        for k, i in nan_cells:
            bad.append(f"{cc.constraint_ids[k]}:{ids[i]}")
        return _fail(SolveStatus.INVALID_INPUT,
                     "missing nominal coefficients (constraint:ingredient) " + ", ".join(sorted(set(bad))),
                     t0, input_hash, p, opts, is_synth, currency, streams)

    lp = assemble_x_space_lp(A_q, rows.b, rows.is_eq, cc.constraint_ids, dh, prices)
    out = run_linprog(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, lp.ub, opts)
    from . import equivalence_annotations   # local import (package imports modules lazily)
    diag = {
        "coefficient_mode": mode,
        "equivalent_methods": [r["pair_id"] for r in equivalence_annotations(METHOD_ID)],
        "highs_model_status": out.highs_model_status,
        "resolved_without_presolve": out.resolved_without_presolve,
        "max_rel_residual_x_space": out.max_rel_residual,
        "max_abs_residual_x_space": out.max_abs_residual,
        "n_rows_ub": int(lp.A_ub.shape[0]),
        "n_rows_eq": int(lp.A_eq.shape[0]),
        "imposed_constraint_ids": list(cc.constraint_ids),
        "solver_wall_time_s": out.wall_time_s,
        "representation": "x_space_kg_dm; q = x / d_hat",
    }
    if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT):
        diag["raw_candidate_x"] = out.extra.get("raw_x")
        return _fail(out.status, out.message, t0, input_hash, p, opts, is_synth, currency, streams, diag)

    x = out.x
    q = x / dh
    decision = RationDecision(ids, q, dh, METHOD_ID, information_state=str(p["information_state"]))
    objective = float(prices @ q)
    diag["lp_objective_x_space"] = out.fun
    diag["total_q_as_fed_kg"] = float(q.sum())
    if not np.any(q > 0):
        # mathematically optimal for the stated problem, but almost surely a mis-specified problem
        # (e.g. no DM-offer rule); reported, never altered
        diag["warning"] = "all-zero ration is optimal: check that a DM-offer/intake rule is declared"
    g = rows.residual(q)[0]
    residuals = {cid: float(v) for cid, v in zip(cc.constraint_ids, g)}
    return SolveResult(method_id=METHOD_ID, status=out.status, decision=decision, objective=objective,
                       objective_unit=f"{currency}/head/d", solver="scipy.optimize.linprog(method='highs')",
                       solver_version=solver_version_string(), tolerances=opts.to_dict(),
                       wall_time_s=time.perf_counter() - t0, input_hash=input_hash, mip_gap=None,
                       iterations=out.iterations, n_evaluations=None, message=out.message,
                       constraint_residuals=residuals, params=p, streams_used=streams, diagnostics=diag,
                       is_synthetic=is_synth)
