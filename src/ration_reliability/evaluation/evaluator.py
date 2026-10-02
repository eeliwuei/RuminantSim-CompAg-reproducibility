"""The single public evaluator shared by every method (contract section 8, T2, T3).

``evaluate(q, theta_draws, d_draws, constraints)`` scores one executed ration ``q``
(kg as-fed/head/d) in every drawn state.  It is deliberately independent of how ``q`` was
produced: methods may optimise in any representation but must hand over ``q``.

What the evaluator does
-----------------------
* realised DM per ingredient ``x_real = q * d_true`` -- **never renormalised** to the planned DM
  and never re-derived from ``q`` with the hidden true DM (T2.1);
* realised ``D = sum_i x_real_i`` and ``N_j = sum_i x_real_i a_ij``;
* for each probabilistic / diagnostic constraint, the natural-unit value ``E`` (concentration
  ``S / D`` with the scenario's own ``D``; supply ``S``), the signed margin in the constraint's
  declared unit, the violation amount and the violation indicator ``-margin > tolerance``;
* the joint indicator ``I(q, theta) = 1{any probabilistic_nutrition constraint violated}`` (T3);
* missing composition/DM of a *used* ingredient -> ``missing_data`` / ``data_quality_flag``;
  such a cell is ``undefined`` and is not counted as a nutrient failure;
* structural constraints are checked once, deterministically, from ``q`` and the decision-time
  ``d_hat`` (never from draws), independently of any risk budget;
* cost ``C(q) = p @ q`` (as-fed prices; unaffected by draws).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..datamodel import (
    ConstraintClass,
    ConstraintKind,
    DMSource,
    EvaluationResult,
    PriceScenario,
    RationDecision,
    Sense,
)
from ..hashing import stable_hash
from ..nutrition.constraints import CompiledConstraints
from ..uncertainty.base import DrawSet

__all__ = ["evaluate", "evaluate_drawset", "structural_check"]

QLike = Union[np.ndarray, RationDecision]


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _margins(kinds, senses, bound, factor, Sv, D):
    """Natural-unit values and declared-unit margins for a block of constraints."""
    n, K = Sv.shape
    E = np.array(Sv, dtype=float)
    conc = np.array([k is ConstraintKind.CONCENTRATION for k in kinds], dtype=bool)
    bad_d = np.zeros((n, K), dtype=bool)
    if conc.any():
        Dm = D[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            E[:, conc] = Sv[:, conc] / Dm
        bad_d[:, conc] = ~(Dm > 0)
    ge = np.array([s is Sense.GE for s in senses], dtype=bool)
    le = np.array([s is Sense.LE for s in senses], dtype=bool)
    m = np.where(ge[None, :], E - bound[None, :],
                 np.where(le[None, :], bound[None, :] - E, -np.abs(E - bound[None, :])))
    return E, m / factor[None, :], bad_d


def structural_check(Q: np.ndarray, d_hat: Optional[np.ndarray], constraints: CompiledConstraints, *,
                     q_nonneg_tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic structural check of rations ``Q [n, I]`` under decision-time DM ``d_hat``.

    ``d_hat`` is ``[I]`` (the same estimate for every ration) or ``[n, I]`` (one estimate per
    ration, e.g. the decision-time information of different signal bins).  Returns
    ``(margin [n, Ks], violated [n, Ks], nonneg_ok [n])`` in the constraints' declared units.
    This is the code path :func:`evaluate` uses for its structural block, so a structural verdict
    obtained here is identical to the public evaluator's (contract D05, C09).
    """
    cc = constraints
    I = len(cc.ingredient_ids)
    Qa = np.asarray(Q, dtype=float)
    if Qa.ndim == 1:
        Qa = Qa[None]
    if Qa.ndim != 2 or Qa.shape[1] != I or not np.all(np.isfinite(Qa)):
        raise ValueError(f"structural_check: Q must be finite [n, {I}]")
    n = Qa.shape[0]
    st = cc.indices(ConstraintClass.STRUCTURAL_HARD)
    Ks = len(st)
    nonneg = np.all(Qa >= -abs(q_nonneg_tol), axis=1)
    if Ks == 0:
        return np.empty((n, 0)), np.zeros((n, 0), dtype=bool), nonneg
    need_dhat = any(cc.dm_sources[k] is DMSource.DECISION_ESTIMATE and np.any(cc.w0[k] != 0) for k in st)
    if d_hat is None:
        if need_dhat:
            raise ValueError("evaluate: structural DM terms need d_hat (pass a RationDecision or d_hat)")
        dstruct = np.zeros((n, I))
    else:
        dh = np.asarray(d_hat, dtype=float)
        if dh.ndim == 1:
            dh = np.broadcast_to(dh, (n, I))
        if dh.shape != (n, I) or np.any(~np.isfinite(dh)) or np.any(dh <= 0) or np.any(dh > 1):
            raise ValueError("structural_check: d_hat must be finite in (0, 1] with shape [I] or [n, I]")
        dstruct = dh
    xs = dstruct * Qa
    Ds = xs.sum(axis=1)
    Svs = xs @ cc.w0[st].T + Qa @ cc.v[st].T  # W == 0 for structural rows (compile rule)
    _, ms, bad = _margins([cc.kinds[k] for k in st], [cc.senses[k] for k in st], cc.bound[st],
                          cc.unit_factor[st], Svs, Ds)
    s_margin = np.where(bad, np.nan, ms)
    s_tol = cc.tol[st] / cc.unit_factor[st]
    with np.errstate(invalid="ignore"):
        s_viol = bad | (-s_margin > s_tol[None, :])
    return s_margin, s_viol, nonneg


def evaluate(q: QLike, theta_draws: np.ndarray, d_draws: np.ndarray, constraints: CompiledConstraints, *,
             d_hat: Optional[np.ndarray] = None,
             prices: Optional[Union[np.ndarray, PriceScenario]] = None,
             draw_stream_id: Optional[str] = None,
             is_synthetic: bool = False,
             chunk_size: int = 20000,
             q_nonneg_tol: float = 1e-9) -> EvaluationResult:
    """Evaluate the executed ration ``q`` under drawn states.

    Parameters
    ----------
    q : ``[I]`` kg as-fed/head/d, or a :class:`RationDecision` (its ``d_hat`` is then used for
        structural constraints).  Ingredient order must equal ``constraints.ingredient_ids``.
    theta_draws : ``[S, I, J]`` (or ``[I, J]``) composition, canonical, DM basis.
    d_draws : ``[S, I]`` (or ``[I]``) true DM fractions of the batch being fed.
    constraints : compiled constraint set (all classes; structural ones are checked separately).
    d_hat : decision-time DM estimates, needed only for structural DM terms (planned ration).
    prices : price vector per kg as-fed or a :class:`PriceScenario` (for ``cost``).
    draw_stream_id, is_synthetic : provenance copied into the result.
    chunk_size : scenarios processed per block (memory bound only; results do not depend on it).
    q_nonneg_tol : tolerance of the built-in structural rule ``q >= 0``.
    """
    cc = constraints
    ids = cc.ingredient_ids
    I, J = len(ids), len(cc.nutrient_ids)

    if isinstance(q, RationDecision):
        if q.ingredient_ids != ids:
            raise ValueError("evaluate: decision ingredient order differs from constraints; "
                             "use decision.reordered(constraints.ingredient_ids)")
        if d_hat is not None and not np.array_equal(np.asarray(d_hat, float), q.d_hat):
            raise ValueError("evaluate: d_hat conflicts with the decision's recorded d_hat")
        d_hat = q.d_hat
        qv = np.array(q.q_as_fed, dtype=float)
    else:
        qv = np.array(q, dtype=float)
    if qv.shape != (I,) or not np.all(np.isfinite(qv)):
        raise ValueError(f"evaluate: q must be finite with shape ({I},)")
    if d_hat is not None:
        d_hat = np.asarray(d_hat, dtype=float)
        if d_hat.shape != (I,) or np.any(~np.isfinite(d_hat)) or np.any(d_hat <= 0) or np.any(d_hat > 1):
            raise ValueError("evaluate: d_hat must be finite in (0, 1] with shape [I]")

    theta = np.asarray(theta_draws, dtype=float)
    d = np.asarray(d_draws, dtype=float)
    if theta.ndim == 2:
        theta = theta[None]
    if d.ndim == 1:
        d = d[None]
    if theta.ndim != 3 or theta.shape[1:] != (I, J) or d.shape != (theta.shape[0], I):
        raise ValueError(f"evaluate: expected theta [S,{I},{J}] and d [S,{I}]")
    S = theta.shape[0]

    ev = cc.indices(ConstraintClass.PROBABILISTIC_NUTRITION, ConstraintClass.DIAGNOSTIC_ONLY)
    st = cc.indices(ConstraintClass.STRUCTURAL_HARD)
    K = len(ev)

    # ---------------- evaluated (random) constraints --------------------------------------
    Wk = cc.W[ev]
    w0k = cc.w0[ev]
    vk = cc.v[ev]
    kinds = [cc.kinds[k] for k in ev]
    senses = [cc.senses[k] for k in ev]
    bound = cc.bound[ev]
    factor = cc.unit_factor[ev]
    tol_decl = cc.tol[ev] / factor
    conc = np.array([k is ConstraintKind.CONCENTRATION for k in kinds], dtype=bool)
    Wflat = Wk.reshape(K, I * J)
    Wnz = (Wflat != 0).astype(float)
    dep_d = ((Wk != 0).any(axis=2) | (w0k != 0)).astype(float)  # [K, I]
    used = qv != 0.0
    s_af = vk @ qv  # [K]

    margin = np.empty((S, K))
    missing = np.zeros((S, K), dtype=bool)
    undefined = np.zeros((S, K), dtype=bool)
    Dall = np.empty(S)
    Nall = np.empty((S, J))
    step = max(1, int(chunk_size))
    for s0 in range(0, S, step):
        s1 = min(S, s0 + step)
        th, dd = theta[s0:s1], d[s0:s1]
        n = s1 - s0
        nan_t, nan_d = np.isnan(th), np.isnan(dd)
        th0 = np.where(nan_t, 0.0, th)
        d0 = np.where(nan_d, 0.0, dd)
        xd = d0 * qv[None, :]                       # realised DM per ingredient, NOT renormalised
        D = xd.sum(axis=1)
        Y = th0 * xd[:, :, None]
        Sv = Y.reshape(n, I * J) @ Wflat.T + xd @ w0k.T + s_af[None, :]
        miss_t = ((nan_t & used[None, :, None]).reshape(n, I * J).astype(float) @ Wnz.T) > 0
        nd_used = (nan_d & used[None, :]).astype(float)
        miss_d = (nd_used @ dep_d.T) > 0
        miss_any_d = nd_used.any(axis=1)
        miss = miss_t | miss_d
        miss[:, conc] |= miss_any_d[:, None]
        _, m, bad_d = _margins(kinds, senses, bound, factor, Sv, D)
        und = miss | bad_d
        margin[s0:s1] = np.where(und, np.nan, m)
        missing[s0:s1] = miss
        undefined[s0:s1] = und
        N = Y.sum(axis=1)
        n_miss = (nan_t & used[None, :, None]).any(axis=1) | miss_any_d[:, None]
        Nall[s0:s1] = np.where(n_miss, np.nan, N)
        Dall[s0:s1] = np.where(miss_any_d, np.nan, D)

    with np.errstate(invalid="ignore"):
        violated = (~undefined) & (-margin > tol_decl[None, :])
        amount = np.where(undefined, 0.0, np.maximum(0.0, -np.nan_to_num(margin, nan=0.0)))
    classes = tuple(cc.classes[k].value for k in ev)
    prob = np.array([c == ConstraintClass.PROBABILISTIC_NUTRITION.value for c in classes], dtype=bool)
    joint_v = violated[:, prob].any(axis=1)
    joint_u = (~joint_v) & undefined[:, prob].any(axis=1)
    dq = missing[:, prob].any(axis=1)

    # ---------------- structural constraints (deterministic, decision-time information) ----
    sm_all, sv_all, nn_all = structural_check(qv[None, :], d_hat, cc, q_nonneg_tol=q_nonneg_tol)
    s_margin, s_viol = sm_all[0], sv_all[0]
    nonneg_ok = bool(nn_all[0])

    cost = None
    if prices is not None:
        p = prices.vector(ids) if isinstance(prices, PriceScenario) else np.asarray(prices, dtype=float)
        if p.shape != (I,):
            raise ValueError("evaluate: prices must have shape [I]")
        cost = float(p @ qv)

    return EvaluationResult(
        ingredient_ids=ids,
        nutrient_ids=cc.nutrient_ids,
        q_as_fed=_ro(qv.copy()),
        q_hash=stable_hash("q/v1", ids, qv),
        constraint_ids=tuple(cc.constraint_ids[k] for k in ev),
        constraint_classes=classes,
        constraint_units=tuple(cc.units[k] for k in ev),
        tolerances=_ro(tol_decl.copy()),
        margin=_ro(margin),
        violation_amount=_ro(amount),
        violated=_ro(violated),
        undefined=_ro(undefined),
        missing_data=_ro(missing),
        joint_violation=_ro(joint_v),
        joint_unknown=_ro(joint_u),
        data_quality_flag=_ro(dq),
        dm_supply=_ro(Dall),
        nutrient_supply=_ro(Nall),
        structural_ids=tuple(cc.constraint_ids[k] for k in st),
        structural_margin=_ro(np.asarray(s_margin, dtype=float)),
        structural_violated=_ro(np.asarray(s_viol, dtype=bool)),
        nonnegativity_ok=nonneg_ok,
        cost=cost,
        n_draws=S,
        draw_stream_id=draw_stream_id,
        is_synthetic=bool(is_synthetic),
    )


def evaluate_drawset(q: QLike, draws: DrawSet, constraints: CompiledConstraints, **kw) -> EvaluationResult:
    """:func:`evaluate` on a :class:`DrawSet` (labels are checked; stream id and synthetic flag
    are propagated)."""
    if draws.ingredient_ids != constraints.ingredient_ids or draws.nutrient_ids != constraints.nutrient_ids:
        raise ValueError("evaluate_drawset: draw labels differ from constraint labels; reorder the draws")
    kw.setdefault("draw_stream_id", draws.stream_id)
    kw["is_synthetic"] = bool(kw.get("is_synthetic", False) or draws.is_synthetic)
    return evaluate(q, draws.theta, draws.d, constraints, **kw)
