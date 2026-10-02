"""Assay-selection strategies compared at the same budget (contract §12.3, T7.6, F08, F13).

Strategies (all use only information available at t1: the declared prior on development states,
the no-information ration ``q0`` and the fixed library -- never the realised batch state):

==============================  =================================================================
strategy id                     rule (pre-declared scores)
==============================  =================================================================
``none``                        buy nothing
``max_marginal_variance``       largest prior SD (or CV) of an option's components
``max_inclusion``               largest planned DM inclusion ``x0_i = q0_i d_hat_i`` of the ingredient
``constraint_sensitivity``      variance share of the component in the linearised margins of the
                                probabilistic constraints at ``q0``, weighted by each constraint's
                                violation probability at ``q0``
``random``                      uniformly random order (named stream ``assay_strategy_random``)
``decision_value``              greedy on the *joint* value of the selected set under an
                                **explicitly declared** ``value_definition`` (no default; second
                                review R1, ``docs/value_definition.md``); sets are re-valued jointly,
                                single values are never added (F13); stops below a declared
                                materiality threshold; operational ranking excludes randomisation
                                channels (FIX_A)
``oracle_reference``            uses the realised true state -- explicit oracle reference only,
                                never a feasible strategy (T7.6); ranks by the number of realised
                                constraint violations an option explains (no cross-unit sums, T3)
==============================  =================================================================

Selections are *fixed-order* plans decided at t1.  Adaptive sequential testing (second assay
chosen after the first result) is ``planned``, not implemented.

Red-team FIX_A (2026-09-25): ranked by the operational definition, a selection is eligible only if its
randomisation channel can be assessed, it is not ``randomization_only`` (pure-noise or garbled assays:
the randomised same-class reference is 0, or a state-independent device with the same bin
probabilities reaches the same saving) and the marginal gain of its
``heuristic_min_of_two_policy_values = min(operational, randomised reference)`` exceeds the declared
materiality threshold; other selections are listed in ``details["randomization_channel_excluded"]``.
Every greedy step records the randomisation-channel assessment (status, benchmark, diagnostic ratio)
and the identification label of the chosen set; a gain within the solver cost tolerance never
triggers a purchase.

Third review R3B (2026-09-25; ``docs/value_semantics_decision.md``): the heuristic minimum (formerly
``executable_information_supported_value``) is no longer "the recommended ranking definition for assay
priorities".  Ranking *by* it is possible only when the caller names it; the selection is then labelled
``value_role="heuristic"`` and ``excluded_from_paper_main_results=True``.  The FIX_A screen of the
operational ranking is kept as a refusal-only filter (it can only exclude selections, never adds value)
but it is a heuristic, too: the output says ``heuristic_screen_applied=True`` and is excluded from paper
main results, and the screen does not make the ranking garbling-safe (a pure garbling of an
informative signal passes it; ``tests/unit/test_garbling_counterexample.py``).

Round-3 red team FIX3_BC (2026-09-25; findings B-1, B-4): no ranking produced here is an assay priority
unless it comes from a complete decision problem (``docs/value_semantics_decision.md`` §2).  Without one
(``decision_problem=None``) every decision-value selection is labelled ``assay_priority_basis =
"none_without_complete_decision_problem"`` and ``excluded_from_paper_main_results=True``, whatever the
definition.  With ``decision_problem=`` a :class:`~.economics.CompleteDecisionProblem` the ranking definition
must be the problem's one policy-class value, alpha must match, and the FIX_A screen (which uses the minimum
of two policy classes) is **not** applied: an operational selection is refused only when its saving is
entirely reachable without information (``randomization_only``) or cannot be assessed (refusal-only); the
materiality threshold applies to the problem's own value.  ``ranking`` scores are ``None`` (never ``-inf``)
for options whose value is undefined or that were screened out; ``details["ranking_status"]`` gives the
reason (``undefined:...`` / ``screened_out:...``).

A decision-value ranking computed on development states is in-sample; strategy comparisons must be
evaluated on an independent stream (validation during development; frozen policies on ``test``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..datamodel import ConstraintClass, RationDecision
from ..errors import InvalidProblemError
from ..evaluation import evaluate
from ..nutrition.constraints import CompiledConstraints
from ..uncertainty.streams import RandomStreams
from .binning import SignalBinning
from .economics import CompleteDecisionProblem
from .library import RiskTable
from .prior import ObservedComponent, PriorStates
from .signal import ComponentErrorModel, SignalModel
from .value import (
    EXCLUDED_FROM_PAPER_MAIN_RESULTS,
    EXECUTABLE_RANDOMIZATION_STATUSES,
    LEGACY_VALUE_KIND_ALIASES,
    VALUE_DEFINITION_NOTES,
    VALUE_DEFINITION_ROLES,
    DefinitionLike,
    InformationStructure,
    InformationValueResult,
    ValueDefinition,
    compute_information_value,
    resolve_value_definition,
)

__all__ = [
    "STRATEGY_IDS",
    "VALUE_KINDS",
    "THRESHOLD_STATUSES",
    "AssayOption",
    "AssayBudget",
    "StrategySelection",
    "combined_structure",
    "strategy_none",
    "strategy_max_marginal_variance",
    "strategy_max_inclusion",
    "strategy_constraint_sensitivity",
    "strategy_random",
    "strategy_decision_value",
    "strategy_oracle_reference",
    "evaluate_selection_value",
]

STRATEGY_IDS = ("none", "max_marginal_variance", "max_inclusion", "constraint_sensitivity", "random",
                "decision_value", "oracle_reference")


@dataclass(frozen=True)
class AssayOption:
    """One purchasable assay/panel on one ingredient batch.

    ``signal_model=None`` means ideal (error-free) observation of the components (used for
    perfect-information rankings only).  ``binning`` holds the fixed development bins of the
    option's components.  ``cost_per_panel`` is per sample/panel (not per nutrient); ``None`` =
    unknown.
    """

    option_id: str
    ingredient_id: str
    components: tuple[str, ...]
    binning: SignalBinning
    signal_model: Optional[SignalModel] = None
    cost_per_panel: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        oc = self.observed
        if tuple(self.binning.components) != oc:
            raise InvalidProblemError(f"option {self.option_id}: binning components must equal the option components")
        if self.signal_model is not None and self.signal_model.components != oc:
            raise InvalidProblemError(f"option {self.option_id}: signal components must equal the option components")

    @property
    def observed(self) -> tuple[ObservedComponent, ...]:
        """Observed components."""
        return tuple(ObservedComponent(self.ingredient_id, c) for c in self.components)


@dataclass(frozen=True)
class AssayBudget:
    """Same budget for every strategy: number of panels and/or total cost (all costs must be known)."""

    max_panels: Optional[int] = None
    max_cost: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_panels is None and self.max_cost is None:
            raise InvalidProblemError("AssayBudget: give max_panels and/or max_cost")


@dataclass(frozen=True)
class StrategySelection:
    """Ordered selection of assay options."""

    strategy_id: str
    selected: tuple[str, ...]
    ranking: tuple[tuple[str, float], ...]
    information_used: str
    is_oracle: bool = False
    notes: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


def _apply_budget(order: Sequence[str], options: Mapping[str, AssayOption], budget: AssayBudget) -> tuple[str, ...]:
    out: list[str] = []
    spent = 0.0
    for oid in order:
        if budget.max_panels is not None and len(out) >= int(budget.max_panels):
            break
        if budget.max_cost is not None:
            c = options[oid].cost_per_panel
            if c is None:
                raise InvalidProblemError(f"cost budget given but the cost of {oid} is unknown")
            if spent + float(c) > float(budget.max_cost) + 1e-12:
                continue
            spent += float(c)
        out.append(oid)
    return tuple(out)


def _index(options: Sequence[AssayOption]) -> dict[str, AssayOption]:
    d = {o.option_id: o for o in options}
    if len(d) != len(options):
        raise InvalidProblemError("duplicate option ids")
    return d


def _rank(scores: Mapping[str, Optional[float]]) -> list[tuple[str, Optional[float]]]:
    """Descending scores; ``None`` (undefined or screened out, FIX3_BC B-4) after every defined score."""
    return sorted(((k, None if v is None else float(v)) for k, v in scores.items()),
                  key=lambda kv: (kv[1] is None, 0.0 if kv[1] is None else -kv[1], kv[0]))


def strategy_none(options: Sequence[AssayOption], budget: AssayBudget) -> StrategySelection:
    """No assay (the no-information decision)."""
    return StrategySelection("none", (), (), "t0_prior_only")


def strategy_max_marginal_variance(options: Sequence[AssayOption], prior: PriorStates, budget: AssayBudget, *,
                                   scale: str = "sd") -> StrategySelection:
    """Rank by the largest prior SD (``scale='sd'``, canonical units) or CV of an option's components."""
    if scale not in ("sd", "cv"):
        raise InvalidProblemError("scale must be 'sd' or 'cv'")
    opts = _index(options)
    scores = {}
    w = prior.weights
    for oid, o in opts.items():
        vals = prior.component_values(o.observed)
        mu = w @ vals
        sd = np.sqrt(np.maximum(w @ (vals - mu) ** 2, 0.0))
        s = sd if scale == "sd" else sd / np.maximum(np.abs(mu), 1e-12)
        scores[oid] = float(np.max(s))
    rk = _rank(scores)
    return StrategySelection("max_marginal_variance", _apply_budget([k for k, _ in rk], opts, budget), tuple(rk),
                             "t0_prior_only", details={"scale": scale})


def strategy_max_inclusion(options: Sequence[AssayOption], q0: RationDecision, budget: AssayBudget) -> StrategySelection:
    """Rank by planned DM inclusion ``q0_i * d_hat_i`` of the option's ingredient in the no-info ration."""
    opts = _index(options)
    x0 = dict(zip(q0.ingredient_ids, q0.x_planned_dm))
    scores = {oid: float(x0[o.ingredient_id]) for oid, o in opts.items()}
    rk = _rank(scores)
    return StrategySelection("max_inclusion", _apply_budget([k for k, _ in rk], opts, budget), tuple(rk),
                             "t0_no_information_ration", details={"basis": "planned kg DM/head/d"})


def strategy_constraint_sensitivity(options: Sequence[AssayOption], prior: PriorStates, q0: RationDecision,
                                    constraints: CompiledConstraints, budget: AssayBudget,
                                    rel_step: float = 1e-4) -> StrategySelection:
    """Variance share of each component in the linearised constraint margins at ``q0``.

    ``share_kc = (dm_k/dc * sd_c)^2 / sum_c' (dm_k/dc' * sd_c')^2`` over all stochastic components
    (independence approximation, a declared simplification), weighted by
    ``P(violation of constraint k at q0)`` on the prior states (equal weights if all are 0).  The
    option score is the sum over its components.  Derivatives by central differences at the prior
    mean state.
    """
    opts = _index(options)
    ev_all = evaluate(q0, prior.theta, prior.d, constraints)
    prob = np.array([c == ConstraintClass.PROBABILISTIC_NUTRITION.value for c in ev_all.constraint_classes])
    if not prob.any():
        raise InvalidProblemError("no probabilistic constraint to be sensitive to")
    w = prior.weights
    pviol = (w @ ev_all.violated[:, prob].astype(float))
    wk = pviol if pviol.sum() > 0 else np.ones(prob.sum())
    wk = wk / wk.sum()
    th_bar, d_bar = prior.weighted_mean(np.ones(prior.n_states))
    comps = prior.stochastic_components()
    vals = prior.component_values(comps)
    mu = w @ vals
    sd = np.sqrt(np.maximum(w @ (vals - mu) ** 2, 0.0))
    grads = np.zeros((int(prob.sum()), len(comps)))
    for ci, comp in enumerate(comps):
        i, j = prior.component_index(comp)
        h = rel_step * max(sd[ci], abs(mu[ci]), 1e-6)
        th_p, th_m = th_bar.copy(), th_bar.copy()
        d_p, d_m = d_bar.copy(), d_bar.copy()
        if j is None:
            d_p[i] += h
            d_m[i] -= h
        else:
            th_p[i, j] += h
            th_m[i, j] -= h
        mp = evaluate(q0, th_p, d_p, constraints).margin[0, prob]
        mm = evaluate(q0, th_m, d_m, constraints).margin[0, prob]
        grads[:, ci] = (mp - mm) / (2 * h)
    contrib = (grads * sd[None, :]) ** 2
    tot = contrib.sum(axis=1, keepdims=True)
    share = np.where(tot > 0, contrib / np.where(tot > 0, tot, 1.0), 0.0)
    comp_score = dict(zip(comps, wk @ share))
    scores = {oid: float(sum(comp_score.get(c, 0.0) for c in o.observed)) for oid, o in opts.items()}
    rk = _rank(scores)
    return StrategySelection("constraint_sensitivity", _apply_budget([k for k, _ in rk], opts, budget), tuple(rk),
                             "t0_prior_and_no_information_ration",
                             details={"constraint_weights": dict(zip(np.array(ev_all.constraint_ids)[prob].tolist(),
                                                                     wk.tolist()))})


def strategy_random(options: Sequence[AssayOption], budget: AssayBudget, streams: RandomStreams,
                    replicate: int = 0, stream: str = "assay_strategy_random") -> StrategySelection:
    """Uniformly random order from a named stream (reproducible)."""
    opts = _index(options)
    ids = sorted(opts)
    perm = streams.generator(stream, int(replicate)).permutation(len(ids))
    order = [ids[p] for p in perm]
    rk = tuple((oid, float(len(order) - n)) for n, oid in enumerate(order))
    return StrategySelection("random", _apply_budget(order, opts, budget), rk, "none (random)",
                             details={"stream_id": streams.stream_id(stream, int(replicate))})


def combined_structure(selected: Sequence[AssayOption], structure_id: str) -> InformationStructure:
    """Joint structure of several options (errors of ideal options = 0; product-grid bins)."""
    if not selected:
        raise InvalidProblemError("combined_structure: empty selection")
    errs: list[ComponentErrorModel] = []
    comps: list[ObservedComponent] = []
    edges: list[tuple[float, ...]] = []
    noisy = False
    synth = False
    for o in selected:
        for n, c in enumerate(o.observed):
            if c in comps:
                raise InvalidProblemError(f"component {c.label()} is observed by two selected options")
            comps.append(c)
            edges.append(o.binning.interior_edges[n])
            if o.signal_model is None:
                errs.append(ComponentErrorModel(c, 0.0, 0.0))
            else:
                e = o.signal_model.errors[n]
                errs.append(e)
                noisy = noisy or e.sampling_sd > 0 or e.lab_repeatability_sd > 0 or e.lab_bias_sd > 0
                synth = synth or o.signal_model.is_synthetic
    binning = SignalBinning(tuple(comps), tuple(edges), "combined option bins",
                            max_bins=max(o.binning.max_bins for o in selected))
    if not noisy and all(e.bias_mean == 0.0 for e in errs):
        return InformationStructure(structure_id, "perfect_partial", tuple(comps), binning)
    protocols = {o.signal_model.protocol for o in selected if o.signal_model is not None}
    if len(protocols) > 1:
        raise InvalidProblemError("combined_structure: selected options use different sampling protocols")
    proto = protocols.pop() if protocols else None
    sm = SignalModel(structure_id, tuple(errs), proto, is_synthetic=synth) if proto is not None else \
        SignalModel(structure_id, tuple(errs), is_synthetic=synth)
    return InformationStructure(structure_id, "sample", tuple(comps), binning, sm)


def evaluate_selection_value(selected_ids: Sequence[str], options: Sequence[AssayOption], prior: PriorStates,
                             risk: RiskTable, alpha: float, *, prior_variance_basis: Mapping[ObservedComponent, str],
                             unknown_policy: str = "error_if_any", action_mask: Optional[np.ndarray] = None,
                             value_definition: Optional[DefinitionLike] = None,
                             **kw) -> Optional[InformationValueResult]:
    """Joint values of a selection on the given states (``None`` for the empty selection).

    The result carries every definition; ``value_definition`` (optional) only marks its
    ``primary_value`` -- without it ``primary_value`` is ``None`` (R1).
    """
    opts = _index(options)
    if not selected_ids:
        return None
    sel = [opts[o] for o in selected_ids]
    st = combined_structure(sel, "+".join(selected_ids))
    # the double-count guard is run inside compute_information_value on this structure's own signal (F06)
    pvb = prior_variance_basis if st.info_type == "sample" else None
    return compute_information_value(st, prior, risk, alpha, unknown_policy=unknown_policy, action_mask=action_mask,
                                     prior_variance_basis=pvb, value_definition=value_definition, **kw)


#: DEPRECATED ``value_kind`` names (engine before the second review); use ``value_definition`` instead.
#: ``net_of_randomization`` maps to the *diagnostic* ``contrast_vs_matched_uninformative_bins``.
VALUE_KINDS = ("net_of_randomization", "deterministic", "randomized_reference")
THRESHOLD_STATUSES = ("sourced", "research_scenario_assumption", "synthetic_test_only")
_LEGACY_NAME = {d.value: k for k, d in LEGACY_VALUE_KIND_ALIASES.items()}


def _operational_block(res: InformationValueResult) -> dict[str, Any]:
    """Cost, ex-ante joint risk and feasibility of the executable deterministic policies (R1, point 4)."""
    def blk(p):
        if p is None:
            return None
        return {"status": p.status, "expected_cost": p.expected_cost, "ex_ante_joint_risk": p.ex_ante_risk,
                "meets_risk_target": None if p.ex_ante_risk is None else bool(p.ex_ante_risk <= p.alpha + p.risk_tol)}
    return {"V0_constant": blk(res.V0), "VT_signal_policy": blk(res.VT),
            "operational_deterministic_cost_difference": res.gross_value,
            "information_value_semantics": res.information_value_semantics,
            "randomization_channel": res.randomization_channel,
            "error_model_identification": res.error_model_identification}


def strategy_decision_value(options: Sequence[AssayOption], prior: PriorStates, risk: RiskTable, alpha: float,
                            budget: AssayBudget, *, prior_variance_basis: Mapping[ObservedComponent, str],
                            min_marginal_value: float, min_marginal_value_status: str,
                            value_definition: Optional[DefinitionLike] = None, value_kind: Optional[str] = None,
                            unknown_policy: str = "error_if_any",
                            action_mask: Optional[np.ndarray] = None, stop_if_no_positive_value: bool = True,
                            decision_problem: Optional[CompleteDecisionProblem] = None,
                            **kw) -> StrategySelection:
    """Greedy selection on the joint value of the selected set (development states only).

    At each step every remaining option is valued *jointly* with the options already selected
    (F13).  ``value_definition`` is **required** (second review R1; there is no default ranking
    value, ``docs/value_definition.md``):

    * ``operational_deterministic_cost_difference``: ``V0 - VT`` over deterministic policies (T7.3
      default class; executable; *includes* the randomisation channel of deterministic maps);
    * ``randomized_same_class_information_reference``: ``V0_rand - VT_rand`` (theoretical reference;
      0 for state-independent signals; the randomised policies are never feeding recommendations);
    * ``contrast_vs_matched_uninformative_bins``: DIAGNOSTIC contrast ``VT(U) - VT`` (any sign); a
      ranking by it is labelled ``value_role="diagnostic"`` and is not an assay priority;
    * ``heuristic_min_of_two_policy_values`` (R3B; deprecated name ``executable_information_supported_value``):
      ``min(operational, randomised reference)``, a HEURISTIC of two policy classes (not
      garbling-monotone); a ranking by it is labelled ``value_role="heuristic"``, excluded from paper
      main results and is not an established assay priority.

    ``value_kind`` is the deprecated name (``deterministic`` / ``randomized_reference`` /
    ``net_of_randomization``) and still works with a :class:`DeprecationWarning`.  Every greedy step
    also records the cost, ex-ante joint risk and feasibility of the executable deterministic
    policies (``greedy_history[i]["operational"]``).

    ``min_marginal_value`` (currency/head/d, >= 0) is the declared materiality threshold: the
    strategy stops when no option adds more than it.  In-sample values of nearly useless signals
    are small but positive (finite-state optimism), so the threshold is a protocol parameter and must
    be declared with ``min_marginal_value_status`` (no default number is invented).  Structures whose
    value is not an information value (information-dependent action space, see
    :func:`~.value.compute_information_value`) are never selected by value; they are listed in
    ``details["not_information_values"]``.

    FIX_A: with the operational definition a set is eligible only if it is not ``randomization_only``
    or ``not_assessable`` and the marginal gain of its ``heuristic_min_of_two_policy_values``
    (``min(operational, randomised reference)``) exceeds ``min_marginal_value``; other sets are never
    selected and are listed in ``details["randomization_channel_excluded"]``.  Each step records
    ``randomization_channel`` (status, benchmark, diagnostic ratio) and ``error_model_identification``.

    R3B: that screen is a heuristic refusal filter (``details["heuristic_screen_applied"]=True``,
    ``details["excluded_from_paper_main_results"]=True``); it is not garbling-safe.  Ranking by the
    heuristic minimum itself requires naming it and is labelled ``heuristic``.  No ranking produced here
    is an established assay priority: that needs a complete decision problem
    (``docs/value_semantics_decision.md``).

    FIX3_BC (B-1, B-4): ``decision_problem`` (a :class:`~.economics.CompleteDecisionProblem`) makes the
    selection the problem's own: the definition must be the problem's (element (a)), alpha must match, and
    the FIX_A screen is replaced by the refusal-only rule (``randomization_only`` / ``not_assessable`` are
    never selected; nothing of the other policy class decides eligibility).  Without it the selection is
    labelled ``assay_priority_basis="none_without_complete_decision_problem"`` and excluded from paper main
    results.  ``ranking`` holds ``None`` for undefined or screened-out options (reason in
    ``details["ranking_status"]``).
    """
    if prior.stream not in ("opt", "validation", "discrete_exact_prior"):
        raise InvalidProblemError("decision_value ranking must use development states")
    if value_kind is not None and value_kind not in VALUE_KINDS:
        raise InvalidProblemError(f"value_kind must be one of {VALUE_KINDS} (deprecated; use value_definition)")
    vdef = resolve_value_definition(value_definition, legacy_value_kind=value_kind, context="strategy_decision_value",
                                    allow_legacy=False)
    role = VALUE_DEFINITION_ROLES[vdef.value]
    if decision_problem is not None:
        if not isinstance(decision_problem, CompleteDecisionProblem):
            raise InvalidProblemError("strategy_decision_value: decision_problem must be a CompleteDecisionProblem")
        if vdef.value != decision_problem.value_definition:
            raise InvalidProblemError(
                f"strategy_decision_value: decision problem {decision_problem.problem_id} values "
                f"{decision_problem.value_definition} (its information processing "
                f"{decision_problem.information_processing.choice}); a ranking by {vdef.value} is not its ranking")
        if abs(float(alpha) - float(decision_problem.alpha)) > 1e-12:
            raise InvalidProblemError(f"strategy_decision_value: decision problem {decision_problem.problem_id} is "
                                      f"declared for alpha {decision_problem.alpha:g}, not {alpha:g}")
        if decision_problem.risk_timing.choice != "ex_ante_joint_risk":
            raise InvalidProblemError("strategy_decision_value: the engine ranks under the ex-ante joint risk only")
    if vdef is ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE and \
            kw.get("include_randomized_reference") is False:
        raise InvalidProblemError("randomized_same_class_information_reference needs include_randomized_reference")
    if vdef is ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS and \
            kw.get("include_randomization_benchmark") is False:
        raise InvalidProblemError("contrast_vs_matched_uninformative_bins needs include_randomization_benchmark")
    thr = float(min_marginal_value)
    if not np.isfinite(thr) or thr < 0:
        raise InvalidProblemError("min_marginal_value must be finite and >= 0")
    if min_marginal_value_status not in THRESHOLD_STATUSES:
        raise InvalidProblemError(f"min_marginal_value_status must be one of {THRESHOLD_STATUSES}")
    opts = _index(options)
    chosen: list[str] = []
    history: list[dict] = []
    first_scores: dict[str, Optional[float]] = {}
    ranking_status: dict[str, str] = {}
    not_values: list[dict] = []
    excluded: list[dict] = []
    current = 0.0
    current_eis = 0.0            # heuristic_min_of_two_policy_values of the chosen set (FIX_A screen; R3B name)
    spent = 0.0
    #: FIX_A screen (min of two policy classes) only without a complete decision problem; with one, refusal-only
    screen = vdef is ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE and decision_problem is None
    refusal_only = vdef is ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE and decision_problem is not None
    while True:
        if budget.max_panels is not None and len(chosen) >= int(budget.max_panels):
            break
        best = None
        best_any = None
        for oid in sorted(opts):
            if oid in chosen:
                continue
            if budget.max_cost is not None:
                c = opts[oid].cost_per_panel
                if c is None:
                    raise InvalidProblemError(f"cost budget given but the cost of {oid} is unknown")
                if spent + float(c) > float(budget.max_cost) + 1e-12:
                    continue
            res = evaluate_selection_value(chosen + [oid], options, prior, risk, alpha,
                                           prior_variance_basis=prior_variance_basis, unknown_policy=unknown_policy,
                                           action_mask=action_mask, value_definition=vdef, **kw)
            v_raw = res.primary_value                               # value under the explicit definition
            if not res.is_information_value:
                not_values.append({"selection": chosen + [oid], "semantics": res.information_value_semantics,
                                   "cost_difference_V0_minus_VT": res.cost_difference_V0_minus_VT})
            v = -np.inf if v_raw is None else float(v_raw)          # internal comparisons only; never reported
            eligible = True
            screened_reason = None
            eis = res.heuristic_min_of_two_policy_values
            if screen and res.is_information_value:
                rc = res.randomization_channel
                eis_gain = None if eis is None else float(eis) - current_eis
                if rc["status"] in ("randomization_only", "not_assessable") or eis_gain is None or not (eis_gain > thr):
                    # FIX_A screen (R3B: a heuristic refusal filter): the marginal gain of min(operational, randomised
                    # reference) must be material; it only excludes selections and never adds value
                    eligible = False
                    screened_reason = (f"FIX_A heuristic screen: randomisation channel {rc['status']}, heuristic-minimum "
                                       f"marginal gain {eis_gain}")
                    excluded.append({"selection": chosen + [oid], "operational_value": v_raw, "status": rc["status"],
                                     "reason": rc["reason"],
                                     "heuristic_screen_marginal_gain": eis_gain,
                                     "randomized_same_class_information_reference":
                                         rc["randomized_same_class_information_reference"],
                                     "heuristic_min_of_two_policy_values":
                                         rc["heuristic_min_of_two_policy_values"],
                                     "randomization_benchmark_value": rc["randomization_benchmark_value"],
                                     "matched_uninformative_contrast_ratio":
                                         rc["matched_uninformative_contrast_ratio"]})
            elif refusal_only and res.is_information_value:
                rc = res.randomization_channel
                if rc["status"] in ("randomization_only", "not_assessable"):
                    # FIX3_BC: refusal-only (a saving reachable without information is never bought); the other
                    # policy class never decides eligibility beyond this refusal
                    eligible = False
                    screened_reason = f"refusal-only rule of the decision problem: randomisation channel {rc['status']}"
                    excluded.append({"selection": chosen + [oid], "operational_value": v_raw, "status": rc["status"],
                                     "reason": rc["reason"], "rule": "refusal_only_decision_problem",
                                     "randomization_benchmark_value": rc["randomization_benchmark_value"],
                                     "matched_uninformative_contrast_ratio":
                                         rc["matched_uninformative_contrast_ratio"]})
            if not chosen:
                # FIX3_BC (B-4): None + reason, never -inf; undefined and screened out are kept apart
                first_scores[oid] = float(v_raw) if (eligible and v_raw is not None) else None
                ranking_status[oid] = ("ok" if (eligible and v_raw is not None) else
                                       f"undefined:{res.primary_value_status}" if v_raw is None else
                                       f"screened_out:{screened_reason}")
            if best_any is None or v > best_any[1]:
                best_any = (oid, v, _operational_block(res), eligible)
            if eligible and (best is None or v > best[1]):
                best = (oid, v, _operational_block(res), eligible, eis, res.randomization_channel["tolerance"])
        if best is None:
            if best_any is not None:
                ja = None if not np.isfinite(best_any[1]) else best_any[1]
                history.append({"step": len(chosen) + 1, "option": best_any[0], "joint_value": ja,
                                "marginal_gain": None if ja is None else ja - current, "value_definition": vdef.value,
                                "operational": best_any[2], "eligible": False,
                                "randomization_channel": best_any[2]["randomization_channel"],
                                "error_model_identification": best_any[2]["error_model_identification"],
                                "stopped": ("no remaining selection passes the operational screen (randomisation "
                                            "channel or immaterial heuristic minimum, FIX_A/R3B); see "
                                            "randomization_channel_excluded" if screen else
                                            "no remaining selection is eligible (refusal-only rule or undefined value); "
                                            "see randomization_channel_excluded / ranking_status")})
            break
        gain = best[1] - current
        defined = bool(np.isfinite(best[1]))
        history.append({"step": len(chosen) + 1, "option": best[0], "joint_value": best[1] if defined else None,
                        "marginal_gain": gain if defined else None,
                        "value_definition": vdef.value, "operational": best[2], "eligible": True,
                        "randomization_channel": best[2]["randomization_channel"],
                        "error_model_identification": best[2]["error_model_identification"]})
        if not defined:
            # FIX3_BC (B-4): an undefined value (None + reason) never triggers a purchase and is not reported as -inf
            history[-1]["stopped"] = "joint value undefined under the chosen definition (None, see ranking_status)"
            break
        if stop_if_no_positive_value and not (gain > thr and gain > best[5]):
            # FIX_A: a gain within the solver cost tolerance (floating-point residue, e.g. 4e-16 for a pure-noise
            # assay under heuristic_min_of_two_policy_values) never triggers a purchase; values are not modified
            history[-1]["stopped"] = (f"marginal joint value {gain:.6g} <= materiality threshold {thr:g}"
                                      + ("" if gain <= thr else f" (within the numerical cost tolerance {best[5]:.3g})"))
            break
        chosen.append(best[0])
        current = best[1]
        current_eis = 0.0 if best[4] is None else float(best[4])
        if budget.max_cost is not None:
            spent += float(opts[best[0]].cost_per_panel)
    rk = _rank(first_scores)
    notes = ""
    if role == "diagnostic":
        notes = ("ranked by the diagnostic contrast_vs_matched_uninformative_bins: not an information value and not an "
                 "assay priority for reporting (docs/value_definition.md)")
    elif role == "theoretical_reference":
        notes = ("ranked by the randomised same-class reference (theoretical); the executable deterministic policies' "
                 "cost, joint risk and feasibility are in greedy_history[i]['operational']")
    elif role == "operational" and screen:
        notes = ("ranked by the operational deterministic difference after the FIX_A screen (a heuristic refusal filter "
                 "on min(operational, randomised reference); randomization_channel_excluded); the screen is not "
                 "garbling-safe and the selection is not an established assay priority (R3B, "
                 "docs/value_semantics_decision.md)")
    elif role == "operational":
        notes = (f"ranked by the operational deterministic difference of the complete decision problem "
                 f"{decision_problem.problem_id} (deterministic policy class; refusal-only rule for randomisation "
                 "channels; not garbling-monotone, not a pure information value; FIX3_BC)")
    elif role == "heuristic":
        notes = ("ranked by the explicitly named HEURISTIC heuristic_min_of_two_policy_values = min(operational, "
                 "randomised reference) of two policy classes; not garbling-monotone; not an assay priority for paper "
                 "main results (R3B, docs/value_semantics_decision.md)")
    heuristic_output = screen or role == "heuristic"
    if decision_problem is None:
        priority_basis = "none_without_complete_decision_problem"
        excluded_main = True
        exclusion = ("no complete decision problem: not an assay priority (docs/value_semantics_decision.md §2); RQ3 is "
                     "an exploratory appendix (§1)")
    else:
        priority_basis = "complete_decision_problem"
        excluded_main = bool(decision_problem.excluded_from_paper_main_results or heuristic_output
                             or vdef.value in EXCLUDED_FROM_PAPER_MAIN_RESULTS)
        exclusion = decision_problem.exclusion_reason()
    return StrategySelection("decision_value", tuple(chosen), tuple(rk), "t0_prior_library_risk_on_development_states",
                             notes=notes,
                             details={"value_definition": vdef.value, "value_role": role,
                                      "ranking_status": ranking_status,
                                      "assay_priority_basis": priority_basis,
                                      "decision_problem": None if decision_problem is None else
                                      decision_problem.to_dict(),
                                      "exclusion_reason": exclusion,
                                      "value_definition_note": VALUE_DEFINITION_NOTES[vdef.value],
                                      "value_kind": _LEGACY_NAME.get(vdef.value),  # deprecated legacy name, kept
                                      "greedy_history": history,
                                      "min_marginal_value": thr, "min_marginal_value_status": min_marginal_value_status,
                                      "not_information_values": not_values,
                                      "randomization_channel_excluded": excluded,
                                      "randomization_channel_rule": (
                                          "operational definition: a set is eligible only if its randomisation channel "
                                          "is assessable, it is not randomization_only, and the marginal gain of "
                                          "heuristic_min_of_two_policy_values (= min(operational, randomised "
                                          "reference)) exceeds the declared materiality threshold; otherwise it is "
                                          "listed in randomization_channel_excluded and never selected (FIX_A screen; "
                                          "R3B: a heuristic refusal filter, not a ranking basis and not "
                                          "garbling-safe; docs/value_semantics_decision.md)"
                                          if screen else
                                          ("operational definition of a complete decision problem: refusal-only -- a set "
                                           "that is randomization_only or not_assessable is never selected; the "
                                           "materiality threshold applies to the problem's own value; the randomised "
                                           "reference does not decide eligibility (FIX3_BC)" if refusal_only else
                                           "recorded only (not the operational definition)")),
                                      "heuristic_screen_applied": screen,
                                      "heuristic": role == "heuristic",
                                      "excluded_from_paper_main_results": excluded_main,
                                      "executable_randomization_statuses": list(EXECUTABLE_RANDOMIZATION_STATUSES),
                                      "note": "joint values of sets; single values never summed"})


def strategy_oracle_reference(options: Sequence[AssayOption], true_theta: np.ndarray, true_d: np.ndarray,
                              prior: PriorStates, q0: RationDecision, constraints: CompiledConstraints,
                              budget: AssayBudget, *, acknowledge_oracle: bool = False) -> StrategySelection:
    """ORACLE REFERENCE ONLY: rank by how many realised constraint violations at ``q0`` each option's
    components explain.  Never a feasible strategy (T7.6).

    Score of an option = (number of ``probabilistic_nutrition`` constraints violated at the true
    state) - (number violated when the option's components are set to their prior means).  Counts
    of constraint events are dimensionless; deficits of different constraints are **not** added
    (their units differ, contract T3).  The per-constraint deficit change in each constraint's own
    unit is reported in ``details["per_constraint_deficit_change"]``.
    """
    if not acknowledge_oracle:
        raise InvalidProblemError("oracle_reference uses the realised true state; pass acknowledge_oracle=True and "
                                  "report it only as an explicit oracle reference")
    opts = _index(options)
    th_bar, d_bar = prior.weighted_mean(np.ones(prior.n_states))
    th_true = np.asarray(true_theta, dtype=float)
    d_true = np.asarray(true_d, dtype=float)

    def outcome(th, dd):
        ev = evaluate(q0, th, dd, constraints)
        prob = np.array([c == ConstraintClass.PROBABILISTIC_NUTRITION.value for c in ev.constraint_classes])
        ids = tuple(np.array(ev.constraint_ids)[prob].tolist())
        units = tuple(np.array(ev.constraint_units)[prob].tolist())
        return ev.violated[0, prob].copy(), ev.violation_amount[0, prob].copy(), ids, units

    v_base, a_base, cids, units = outcome(th_true, d_true)
    scores = {}
    per_con: dict[str, dict[str, Any]] = {}
    for oid, o in opts.items():
        th, dd = th_true.copy(), d_true.copy()
        for c in o.observed:
            i, j = prior.component_index(c)
            if j is None:
                dd[i] = d_bar[i]
            else:
                th[i, j] = th_bar[i, j]
        v_o, a_o, _, _ = outcome(th, dd)
        scores[oid] = float(int(v_base.sum()) - int(v_o.sum()))
        per_con[oid] = {cid: {"unit": u, "deficit_true": float(ab), "deficit_with_prior_mean": float(ao),
                              "change": float(ab - ao)} for cid, u, ab, ao in zip(cids, units, a_base, a_o)}
    rk = _rank(scores)
    return StrategySelection("oracle_reference", _apply_budget([k for k, _ in rk], opts, budget), tuple(rk),
                             "ORACLE: realised true state", is_oracle=True,
                             notes="explicit oracle reference; not an implementable strategy",
                             details={"score": "number of realised probabilistic-constraint violations explained "
                                               "(dimensionless count; deficits of different units never summed)",
                                      "per_constraint_deficit_change": per_con})
