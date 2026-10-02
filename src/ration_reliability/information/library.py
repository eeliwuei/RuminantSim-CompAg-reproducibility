"""Finite candidate ration library ``K`` and its risk table (contract T7.3).

The information-value policy model chooses, for every signal bin, one ration from a finite
library ``K`` built during development.  The library must

* contain no-information candidates (so that the constant policy -- the no-information decision --
  is part of every with-information policy class, T7.1);
* cover the observable information states (candidates designed for bin-conditional compositions);
* contain only structurally feasible rations (structural constraints are never relaxed, C09);
* be frozen before any test evaluation (formal tests evaluate the frozen policy only).

Actions are executed as-fed rations ``q`` (kg as-fed/head/d).  A candidate designed after a DM
assay stores the DM estimate it was designed with (``RationDecision.d_hat``); nutrition is always
evaluated with the true state's DM by the public evaluator (T2.1).

Decision-time structural feasibility (red-team fix C05/F02, 2026-09-24)
-----------------------------------------------------------------------
Entering the library only requires structural feasibility under the candidate's *own* design
``d_hat`` (a candidate must be executable in the information state it was designed for).  Whether
a candidate is a legal action of a given decision is decided separately, with the DM estimate
available *at that decision* (contract T1, T2.1; ``configs/constraints.yaml`` SH-DM-PLAN: after a
DM assay only the assayed ingredient's ``d_hat_i`` is updated):

* the no-information decision (V0) and every structure that observes no DM component use the t0
  estimate ``problem.dm_estimates()`` -- a ration designed for a DM bin whose planned DM under the
  t0 estimate violates SH-DM-PLAN is **not** available to them;
* a structure that observes the DM of ingredient ``i`` uses, in bin ``z``, ``d_hat_z`` equal to the
  t0 estimate except ``d_hat_z,i = E[d_i | z]`` (prior weights x likelihood, the same arithmetic as
  :func:`generate_conditional_library`).

:func:`decision_time_action_masks` returns these masks; :func:`~.value.compute_information_value`
applies them.  With information-dependent masks the constant no-information policy need not be
available after a DM assay, so constant-policy containment (T7.1) is *checked*, not assumed.

Risk entries come exclusively from :func:`ration_reliability.evaluation.evaluate` (one shared
evaluator for every method, D05).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..datamodel import ConstraintClass, RationDecision, RationProblem, Sense, SolverOptions, SolveResult
from ..errors import InvalidProblemError
from ..evaluation import evaluate, structural_check
from ..hashing import stable_hash
from ..nutrition.constraints import CompiledConstraints
from .prior import DM_COMPONENT, PriorStates

__all__ = [
    "UNKNOWN_POLICIES",
    "CandidateLibrary",
    "RiskTable",
    "DecisionTimeActions",
    "decision_time_action_masks",
    "ConditioningSpec",
    "compute_risk_table",
    "generate_conditional_library",
    "conditioning_from_likelihood",
    "adjustment_mask",
]

#: How an ``undefined`` joint outcome (missing data) enters the risk: never silently.
UNKNOWN_POLICIES = ("error_if_any", "count_as_violation", "count_as_satisfied")


@dataclass(frozen=True)
class CandidateLibrary:
    """Frozen finite set of candidate rations.

    ``origins[k]`` records how candidate ``k`` was produced (method, conditioning label, tightening
    parameter, streams).  ``failures`` records attempted generations that returned no ration
    (status kept; never replaced by an empty ration or zero cost).
    """

    candidate_ids: tuple[str, ...]
    decisions: tuple[RationDecision, ...]
    costs: np.ndarray
    ingredient_ids: tuple[str, ...]
    origins: tuple[Mapping[str, Any], ...]
    failures: tuple[Mapping[str, Any], ...] = ()
    merged_duplicates: tuple[Mapping[str, Any], ...] = ()
    is_synthetic: bool = False
    currency: str = ""
    t0_d_hat: Optional[np.ndarray] = None   # decision-time DM estimate without any assay (problem.dm_estimates())

    def __post_init__(self) -> None:
        c = np.array(self.costs, dtype=float)
        c.setflags(write=False)
        object.__setattr__(self, "costs", c)
        if self.t0_d_hat is not None:
            d0 = np.array(self.t0_d_hat, dtype=float)
            if d0.shape != (len(self.ingredient_ids),) or np.any(~np.isfinite(d0)) or np.any(d0 <= 0) \
                    or np.any(d0 > 1):
                raise InvalidProblemError("CandidateLibrary: t0_d_hat must be finite in (0, 1] with shape [I]")
            d0.setflags(write=False)
            object.__setattr__(self, "t0_d_hat", d0)
        K = len(self.candidate_ids)
        if not (len(self.decisions) == len(self.origins) == c.shape[0] == K):
            raise InvalidProblemError("CandidateLibrary: inconsistent lengths")
        if K == 0:
            raise InvalidProblemError("CandidateLibrary: empty library")
        if len(set(self.candidate_ids)) != K:
            raise InvalidProblemError("CandidateLibrary: duplicate candidate ids")
        for dcs in self.decisions:
            if dcs.ingredient_ids != tuple(self.ingredient_ids):
                raise InvalidProblemError("CandidateLibrary: decision ingredient order differs from library")

    @property
    def size(self) -> int:
        """Number of candidates ``K``."""
        return len(self.candidate_ids)

    @property
    def Q(self) -> np.ndarray:
        """As-fed amounts ``[K, I]``."""
        return np.stack([d.q_as_fed for d in self.decisions])

    @property
    def fingerprint(self) -> str:
        """Content hash of the rations (q, d_hat) and ids."""
        return stable_hash("CandidateLibrary/v2", self.candidate_ids, self.ingredient_ids,
                           tuple((d.q_as_fed, d.d_hat) for d in self.decisions), self.costs, self.t0_d_hat)

    @classmethod
    def from_decisions(cls, problem: RationProblem, decisions: Sequence[RationDecision],
                       candidate_ids: Optional[Sequence[str]] = None,
                       origins: Optional[Sequence[Mapping[str, Any]]] = None, *,
                       failures: Sequence[Mapping[str, Any]] = (), dedup_tol: float = 1e-9) -> "CandidateLibrary":
        """Build a library, rejecting structurally infeasible rations and merging duplicates.

        Structural feasibility is checked by the public evaluator from ``q`` and the decision's own
        ``d_hat`` (independent of draws).  A structurally infeasible ration raises
        :class:`InvalidProblemError` (it must not silently enter the action space).
        """
        cc = problem.compiled
        ids = list(candidate_ids) if candidate_ids is not None else [f"k{n:03d}" for n in range(len(decisions))]
        ors = list(origins) if origins is not None else [{} for _ in decisions]
        if not (len(ids) == len(ors) == len(decisions)):
            raise InvalidProblemError("from_decisions: lengths of decisions, ids and origins differ")
        theta0 = np.where(np.isnan(problem.nominal_theta()), 0.0, problem.nominal_theta())
        prices = problem.price_vector()
        keep_d: list[RationDecision] = []
        keep_id: list[str] = []
        keep_o: list[Mapping[str, Any]] = []
        merged: list[dict] = []
        for cid, dec, org in zip(ids, decisions, ors):
            dec = dec if dec.ingredient_ids == problem.ingredient_ids else dec.reordered(problem.ingredient_ids)
            ev = evaluate(dec, theta0, problem.dm_estimates(), cc, prices=prices)
            if not ev.structural_ok:
                bad = [c for c, v in zip(ev.structural_ids, ev.structural_violated) if v]
                raise InvalidProblemError(f"candidate {cid}: structurally infeasible ({bad or 'q >= 0'}); "
                                          "structural constraints are never relaxed")
            dup = None
            for kid, kd in zip(keep_id, keep_d):
                if np.max(np.abs(kd.q_as_fed - dec.q_as_fed)) <= dedup_tol and \
                        np.max(np.abs(kd.d_hat - dec.d_hat)) <= dedup_tol:
                    dup = kid
                    break
            if dup is not None:
                merged.append({"dropped": cid, "kept": dup, "origin": dict(org)})
                continue
            keep_d.append(dec)
            keep_id.append(cid)
            keep_o.append(dict(org))
        costs = np.array([float(prices @ d.q_as_fed) for d in keep_d])
        return cls(tuple(keep_id), tuple(keep_d), costs, problem.ingredient_ids, tuple(keep_o),
                   tuple(dict(f) for f in failures), tuple(merged), bool(problem.is_synthetic),
                   problem.prices.currency, problem.dm_estimates())

    def subset(self, mask: np.ndarray) -> "CandidateLibrary":
        """Library restricted to ``mask`` (for K-enlargement checks)."""
        m = np.asarray(mask, dtype=bool)
        if m.shape != (self.size,) or not m.any():
            raise InvalidProblemError("subset: mask must be [K] with at least one True")
        idx = np.flatnonzero(m)
        return CandidateLibrary(tuple(self.candidate_ids[k] for k in idx), tuple(self.decisions[k] for k in idx),
                                self.costs[idx], self.ingredient_ids, tuple(self.origins[k] for k in idx),
                                self.failures, self.merged_duplicates, self.is_synthetic, self.currency,
                                self.t0_d_hat)


@dataclass(frozen=True)
class RiskTable:
    """Joint-violation outcomes ``[S, K]`` of every candidate in every prior state.

    ``violated[s, k]``: at least one ``probabilistic_nutrition`` constraint violated (event I, T3).
    ``unknown[s, k]``: not violated but undefined because of missing data (data-quality flag).

    Decision-time structural data (needed by :func:`decision_time_action_masks`): the executed
    rations ``q_as_fed [K, I]``, the t0 DM estimate ``t0_d_hat [I]``, the structural rows of the
    compiled constraints and ``t0_structural_ok [K]`` (legal action of the no-information decision).
    """

    violated: np.ndarray
    unknown: np.ndarray
    costs: np.ndarray
    candidate_ids: tuple[str, ...]
    prior_fingerprint: str
    library_fingerprint: str
    constraints_fingerprint: str
    is_synthetic: bool
    q_as_fed: Optional[np.ndarray] = None
    t0_d_hat: Optional[np.ndarray] = None
    structural_constraints: Optional[CompiledConstraints] = None
    t0_structural_ok: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for n in ("violated", "unknown"):
            a = np.array(getattr(self, n), dtype=bool)
            a.setflags(write=False)
            object.__setattr__(self, n, a)
        c = np.array(self.costs, dtype=float)
        c.setflags(write=False)
        object.__setattr__(self, "costs", c)
        if self.violated.shape != self.unknown.shape or self.violated.shape[1] != c.shape[0]:
            raise InvalidProblemError("RiskTable: inconsistent shapes")
        for n, dt in (("q_as_fed", float), ("t0_d_hat", float), ("t0_structural_ok", bool)):
            v = getattr(self, n)
            if v is not None:
                a = np.array(v, dtype=dt)
                a.setflags(write=False)
                object.__setattr__(self, n, a)
        if self.q_as_fed is not None and self.q_as_fed.shape[0] != c.shape[0]:
            raise InvalidProblemError("RiskTable: q_as_fed must be [K, I]")
        if self.t0_structural_ok is not None and self.t0_structural_ok.shape != c.shape:
            raise InvalidProblemError("RiskTable: t0_structural_ok must be [K]")

    def indicator(self, unknown_policy: str = "error_if_any") -> np.ndarray:
        """Risk indicator ``I[s, k]`` as floats under the declared treatment of unknown outcomes."""
        if unknown_policy not in UNKNOWN_POLICIES:
            raise InvalidProblemError(f"unknown_policy must be one of {UNKNOWN_POLICIES}")
        if unknown_policy == "error_if_any" and self.unknown.any():
            n = int(self.unknown.sum())
            raise InvalidProblemError(f"{n} state x candidate outcome(s) are undefined (missing data); choose "
                                      "unknown_policy='count_as_violation' (upper) or 'count_as_satisfied' (lower) "
                                      "explicitly and report both")
        if unknown_policy == "count_as_violation":
            return (self.violated | self.unknown).astype(float)
        return self.violated.astype(float)

    @property
    def fingerprint(self) -> str:
        """Content hash."""
        return stable_hash("RiskTable/v2", self.prior_fingerprint, self.library_fingerprint,
                           self.constraints_fingerprint, self.violated, self.unknown, self.costs, self.q_as_fed,
                           self.t0_d_hat, self.t0_structural_ok)


def compute_risk_table(library: CandidateLibrary, prior: PriorStates, constraints: CompiledConstraints,
                       prices: Any = None) -> RiskTable:
    """Evaluate every candidate in every prior state with the public evaluator."""
    if tuple(constraints.ingredient_ids) != tuple(library.ingredient_ids) or \
            tuple(prior.ingredient_ids) != tuple(library.ingredient_ids) or \
            tuple(prior.nutrient_ids) != tuple(constraints.nutrient_ids):
        raise InvalidProblemError("compute_risk_table: ingredient/nutrient labels differ")
    S, K = prior.n_states, library.size
    viol = np.zeros((S, K), dtype=bool)
    unk = np.zeros((S, K), dtype=bool)
    costs = np.zeros(K)
    for k, dec in enumerate(library.decisions):
        ev = evaluate(dec, prior.theta, prior.d, constraints, prices=prices, draw_stream_id=prior.stream_id,
                      is_synthetic=prior.is_synthetic)
        if not ev.structural_ok:
            raise InvalidProblemError(f"candidate {library.candidate_ids[k]} is structurally infeasible")
        viol[:, k] = ev.joint_violation
        unk[:, k] = ev.joint_unknown
        costs[k] = library.costs[k] if ev.cost is None else ev.cost
    if not np.allclose(costs, library.costs, rtol=0, atol=1e-12):
        raise InvalidProblemError("compute_risk_table: evaluator cost differs from library cost (price mismatch)")
    if library.t0_d_hat is None:
        raise InvalidProblemError("compute_risk_table: the library has no t0 decision-time DM estimate; build it with "
                                  "CandidateLibrary.from_decisions / generate_conditional_library")
    structural = constraints.subset(constraints.indices(ConstraintClass.STRUCTURAL_HARD))
    Q = library.Q
    _, sviol0, nn0 = structural_check(Q, library.t0_d_hat, structural)
    t0_ok = (~sviol0.any(axis=1)) & nn0
    return RiskTable(viol, unk, library.costs, library.candidate_ids, prior.fingerprint, library.fingerprint,
                     constraints.fingerprint, bool(prior.is_synthetic or library.is_synthetic), Q,
                     library.t0_d_hat, structural, t0_ok)


# ---------------------------------------------------------------------------------------------
# decision-time action sets (red-team fix C05/F02)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionTimeActions:
    """Legal actions of the no-information decision and of every signal bin of one structure.

    * ``t0_mask [K]``: candidates structurally feasible under the t0 DM estimate (V0's action set).
    * ``bin_masks [Z, K]``: candidates structurally feasible under bin ``z``'s decision-time DM
      estimate ``d_hat_bins[z]`` (VT's action set in bin ``z``).  Bins without development
      probability have an undefined ``E[d | z]`` when DM is observed: their row is all False and
      ``d_hat_bins[z]`` is NaN (their frozen fallback action is reported as unverified).
    * ``action_space``: ``"t0_fixed"`` (no DM observed; every bin uses the t0 set) or
      ``"information_dependent"`` (DM observed).
    * ``constant_policies_contained``: every V0 action is legal in every active bin (constant-policy
      containment of T7.1).  When False, ``V0 - VT`` compares two different action spaces and is not
      an information value (T7.2: report it as a cost-risk comparison).
    """

    dm_observed: tuple[str, ...]
    d_hat_bins: np.ndarray
    t0_mask: np.ndarray
    bin_masks: np.ndarray
    active_bins: np.ndarray
    action_space: str
    constant_policies_contained: bool
    n_active_bins_without_candidate: int

    def summary(self, candidate_ids: Sequence[str] = ()) -> dict[str, Any]:
        """JSON-friendly summary (no arrays except small index lists)."""
        za = self.active_bins
        lost = [int(k) for k in np.flatnonzero(self.t0_mask) if not self.bin_masks[za, k].all()]
        return {"dm_observed": list(self.dm_observed), "action_space": self.action_space,
                "constant_policies_contained": bool(self.constant_policies_contained),
                "n_t0_feasible_candidates": int(self.t0_mask.sum()),
                "n_active_bins": int(za.size),
                "n_active_bins_without_candidate": int(self.n_active_bins_without_candidate),
                "t0_candidates_not_legal_in_some_active_bin": ([candidate_ids[k] for k in lost] if candidate_ids
                                                               else lost),
                "rule": "structural rows checked with the decision-time d_hat: t0 estimate for V0 and for "
                        "ingredients whose DM is not observed; E[d_i | z] for observed DM (SH-DM-PLAN note)"}


def decision_time_action_masks(prior: PriorStates, lik, risk: RiskTable, *,
                               user_mask: Optional[np.ndarray] = None) -> DecisionTimeActions:
    """Structural action masks of V0 and of every bin of the structure described by ``lik``.

    ``user_mask [K]`` (e.g. :func:`adjustment_mask`) is intersected with both.  Needs a risk table
    from :func:`compute_risk_table` (it carries the rations, the t0 estimate and the structural rows).
    """
    if risk.q_as_fed is None or risk.t0_d_hat is None or risk.structural_constraints is None \
            or risk.t0_structural_ok is None:
        raise InvalidProblemError("risk table lacks decision-time structural data; rebuild it with compute_risk_table")
    if lik.prior_fingerprint != prior.fingerprint or risk.prior_fingerprint != prior.fingerprint:
        raise InvalidProblemError("decision_time_action_masks: likelihood / risk table were built on another prior")
    K = risk.costs.shape[0]
    um = np.ones(K, dtype=bool) if user_mask is None else np.asarray(user_mask, dtype=bool)
    if um.shape != (K,):
        raise InvalidProblemError("decision_time_action_masks: user mask must be [K]")
    Q, d0, cons = risk.q_as_fed, risk.t0_d_hat, risk.structural_constraints
    I = d0.shape[0]
    t0 = np.array(risk.t0_structural_ok, dtype=bool) & um
    Z = lik.n_bins
    Pz = prior.weights @ lik.L
    active = np.flatnonzero(Pz > 0)
    ids = tuple(prior.ingredient_ids)
    dm_obs = tuple(sorted({c.ingredient_id for c in lik.components if c.component == DM_COMPONENT}))
    if not dm_obs:
        d_bins = np.broadcast_to(d0, (Z, I)).copy()
        masks = np.broadcast_to(t0, (Z, K)).copy()
        space = "t0_fixed"
    else:
        idx = [ids.index(i) for i in dm_obs]
        d_bins = np.full((Z, I), np.nan)
        masks = np.zeros((Z, K), dtype=bool)
        for z in active:
            w = lik.L[:, z].copy() * prior.weights      # same arithmetic as generate_conditional_library
            _, d_bar = prior.weighted_mean(w)
            dz = np.array(d0, dtype=float)
            dz[idx] = d_bar[idx]
            d_bins[z] = dz
            _, vz, nnz = structural_check(Q, dz, cons)
            masks[z] = (~vz.any(axis=1)) & nnz & um
        space = "information_dependent"
    contained = bool(np.all(masks[np.ix_(active, np.flatnonzero(t0))])) if active.size else True
    n_empty = int(np.sum(~masks[active].any(axis=1))) if active.size else 0
    for a in (d_bins, masks, t0, active):
        a.setflags(write=False)
    return DecisionTimeActions(dm_obs, d_bins, t0, masks, active, space, contained, n_empty)


# ---------------------------------------------------------------------------------------------
# library generation (development only)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditioningSpec:
    """One information state for which candidates are designed.

    ``weights[s]`` >= 0 are *likelihood* weights of the prior states in this information state
    (0/1 for a perfect-information bin; ``L[s, z]`` for a noisy bin; all ones for the
    no-information state).  The posterior weight used for design is ``pi_s * weights[s]``.
    ``dm_observed`` lists ingredients whose DM is known in this state (their design ``d_hat``
    becomes the conditional mean DM).
    """

    label: str
    weights: np.ndarray
    dm_observed: tuple[str, ...] = ()


def conditioning_from_likelihood(prior: PriorStates, lik, *, min_bin_probability: float = 0.0,
                                 prefix: str = "") -> list[ConditioningSpec]:
    """One :class:`ConditioningSpec` per active bin of a :class:`~.likelihood.SignalLikelihood`."""
    dm_obs = tuple(sorted({c.ingredient_id for c in lik.components if c.component == DM_COMPONENT}))
    if lik.prior_fingerprint != prior.fingerprint:
        raise InvalidProblemError("conditioning_from_likelihood: likelihood was built on another prior")
    out = []
    pz = prior.weights @ lik.L
    for z in range(lik.n_bins):
        if pz[z] > min_bin_probability:
            out.append(ConditioningSpec(f"{prefix}{lik.kind}:{lik.bin_labels[z]}", lik.L[:, z].copy(), dm_obs))
    return out


def _tightened_problem(problem: RationProblem, theta_bar: np.ndarray, d_hat: np.ndarray,
                       shifts: Mapping[str, float], tag: str) -> RationProblem:
    ings = []
    for i, g in enumerate(problem.ingredients):
        comp = {n.nutrient_id: float(theta_bar[i, j]) for j, n in enumerate(problem.nutrients)
                if not np.isnan(theta_bar[i, j])}
        ings.append(dataclasses.replace(g, composition=comp, dm_estimate=float(d_hat[i])))
    cons = []
    for c in problem.constraints:
        if c.constraint_id in shifts and shifts[c.constraint_id] != 0.0:
            s = float(shifts[c.constraint_id])
            nb = float(c.bound) + s if Sense(c.sense) is Sense.GE else float(c.bound) - s
            cons.append(dataclasses.replace(c, bound=nb))
        else:
            cons.append(c)
    return dataclasses.replace(problem, problem_id=f"{problem.problem_id}|{tag}", ingredients=tuple(ings),
                               constraints=tuple(cons))


def generate_conditional_library(problem: RationProblem, prior: PriorStates,
                                 conditioning: Sequence[ConditioningSpec], *,
                                 tightening_grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                                 include_no_information: bool = True,
                                 solve_fn: Optional[Callable[..., SolveResult]] = None,
                                 solver_options: Optional[SolverOptions] = None,
                                 extra_decisions: Iterable[tuple[str, RationDecision, Mapping[str, Any]]] = (),
                                 dedup_tol: float = 1e-9) -> CandidateLibrary:
    """Design candidates for each information state (development heuristic, contract T7.3).

    For each conditioning state ``g`` (plus the unconditional state if ``include_no_information``):

    1. ``theta_bar_g``, ``d_bar_g`` = weighted means of the prior states (development stream only);
       the design DM estimate is ``d_bar_g`` for ingredients whose DM is observed, else the
       problem's decision-time ``d_hat``;
    2. ``q_ref`` = M0 nominal solution with ``theta_bar_g`` (``tightening 0``);
    3. for every ``gamma`` in ``tightening_grid``: every ``probabilistic_nutrition`` bound is moved
       in the conservative direction by ``gamma * sd_gk``, where ``sd_gk`` is the weighted SD (in
       the constraint's declared unit) of the constraint value of ``q_ref`` over the states of
       ``g``; the tightened nominal LP is solved with M0.

    The grid is a *library-construction device*, not a safety-margin method (M1 is separate) and
    not a nutritional recommendation.  Failed solves are recorded in ``failures`` and never
    replaced.  ``extra_decisions`` (e.g. M1/M2/M3 rations) can be added.  All candidates share one
    library; which of them a decision may use is decided by :func:`decision_time_action_masks`
    (structural feasibility under that decision's DM information), not by the design ``d_hat``.
    """
    from ..optimization import get_method  # local import (optional dependency on the registry)

    if tuple(prior.ingredient_ids) != problem.ingredient_ids or tuple(prior.nutrient_ids) != problem.nutrient_ids:
        raise InvalidProblemError("generate_conditional_library: prior labels differ from the problem")
    solve = solve_fn or get_method("M0_nominal")
    cc = problem.compiled
    prob_idx = cc.indices(ConstraintClass.PROBABILISTIC_NUTRITION)
    ev_idx = cc.indices(ConstraintClass.PROBABILISTIC_NUTRITION, ConstraintClass.DIAGNOSTIC_ONLY)
    # position of each probabilistic constraint inside the evaluator's (prob + diag) block
    pos_in_ev = {int(k): int(np.flatnonzero(ev_idx == k)[0]) for k in prob_idx}
    states = list(conditioning)
    if include_no_information:
        states = [ConditioningSpec("no_information", np.ones(prior.n_states), ())] + states
    decisions: list[RationDecision] = []
    ids: list[str] = []
    origins: list[dict] = []
    failures: list[dict] = []
    d_hat0 = problem.dm_estimates()
    for g_no, spec in enumerate(states):
        w = np.asarray(spec.weights, dtype=float) * prior.weights
        if w.shape != (prior.n_states,) or w.sum() <= 0:
            failures.append({"conditioning": spec.label, "status": "invalid_input", "message": "zero weight"})
            continue
        th_bar, d_bar = prior.weighted_mean(w)
        d_design = d_hat0.copy()
        for iid in spec.dm_observed:
            i = problem.ingredient_ids.index(iid)
            d_design[i] = d_bar[i]
        ref = None
        for gamma in tightening_grid:
            shifts: dict[str, float] = {}
            if gamma != 0.0:
                if ref is None:
                    failures.append({"conditioning": spec.label, "gamma": float(gamma), "status": "skipped",
                                     "message": "no reference ration (tightening 0 failed)"})
                    continue
                ev = evaluate(ref, prior.theta, prior.d, cc)
                wn = w / w.sum()
                for k in prob_idx:
                    col = ev.margin[:, pos_in_ev[int(k)]]
                    ok = ~np.isnan(col)
                    if not ok.any():
                        continue
                    ww = wn[ok] / wn[ok].sum()
                    mu = float(ww @ col[ok])
                    sd = float(np.sqrt(max(ww @ (col[ok] - mu) ** 2, 0.0)))
                    shifts[cc.constraint_ids[int(k)]] = float(gamma) * sd
            tag = f"g{g_no}:gamma={gamma:g}"
            sub = _tightened_problem(problem, th_bar, d_design, shifts, tag)
            res = solve(sub, d_hat=d_design, params={"coefficient_mode": "nominal_point",
                                                    "information_state": spec.label},
                        solver_options=solver_options)
            origin = {"conditioning": spec.label, "gamma": float(gamma), "method": res.method_id,
                      "prior_stream_id": prior.stream_id, "bound_shifts_declared_units": dict(shifts),
                      "dm_observed": list(spec.dm_observed)}
            if not res.has_solution:
                failures.append(dict(origin, status=str(res.status), message=res.message))
                continue
            dec = RationDecision(problem.ingredient_ids, res.decision.q_as_fed, d_design,
                                 method_id=f"P8_library[{res.method_id}]", information_state=spec.label)
            if gamma == 0.0:
                ref = dec
            decisions.append(dec)
            ids.append(f"g{g_no:03d}_t{len(ids):04d}")
            origins.append(origin)
    for cid, dec, org in extra_decisions:
        decisions.append(dec)
        ids.append(str(cid))
        origins.append(dict(org))
    if not decisions:
        raise InvalidProblemError("generate_conditional_library: no candidate could be generated; failures: "
                                  f"{failures[:5]}")
    return CandidateLibrary.from_decisions(problem, decisions, ids, origins, failures=failures, dedup_tol=dedup_tol)


def adjustment_mask(library: CandidateLibrary, q_current: np.ndarray, *, max_abs_change_kg: Optional[float] = None,
                    max_rel_change: Optional[float] = None) -> np.ndarray:
    """Candidates reachable from the current ration within adjustment limits (applied to V0 and VT).

    The same mask must be passed to the no-information and with-information problems so that both
    use the same action space (T7.1).  Limits are research-setting parameters (must be sourced or
    declared as scenario assumptions by the caller).
    """
    q0 = np.asarray(q_current, dtype=float)
    Q = library.Q
    if q0.shape != (Q.shape[1],):
        raise InvalidProblemError("adjustment_mask: q_current must be [I]")
    m = np.ones(library.size, dtype=bool)
    if max_abs_change_kg is not None:
        m &= np.max(np.abs(Q - q0[None, :]), axis=1) <= float(max_abs_change_kg) + 1e-12
    if max_rel_change is not None:
        rel = np.abs(Q - q0[None, :]) / np.maximum(np.abs(q0[None, :]), 1e-12)
        rel = np.where(q0[None, :] == 0, np.where(Q == 0, 0.0, np.inf), rel)
        m &= np.max(rel, axis=1) <= float(max_rel_change) + 1e-12
    return m
