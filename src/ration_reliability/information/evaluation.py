"""Freeze a signal policy in development and evaluate it on independent states (contract T7.3, T8.4).

Time order (T1, §12.2): the assay action and the bin -> ration map are fixed before any test
state is seen; at evaluation, each test state produces a signal (true value + simulated error from a
named noise stream), the signal is binned with the frozen edges, the frozen map picks a ration, and
the public evaluator scores that ration against the *hidden* test state.  The lookup never receives
the state itself (only the bin index), so unobserved components cannot influence the action.

Only binned structures can be frozen (``perfect_partial_binned`` and sample information).  Full
perfect information and exact-value partial information are defined by development state
identities and have no frozen policy for new states: they remain ideal/oracle references.

The Monte Carlo interval reported here is the fixed-distribution simulation interval only (T8.3).

Red-team fixes (2026-09-24):

* D09/F04 -- ``realised_cost_saving_per_head_day`` is given only when the frozen policy *and* the
  no-information candidate both meet the target ``alpha`` on the evaluation states (point estimate,
  same absolute tolerance as the policy problem); otherwise it is ``None`` and the raw difference is
  reported as ``cost_difference_per_head_day`` with ``saving_status="cost_difference_at_unequal_risk"``.
* F04 -- :func:`calibrate_alpha_train` additionally returns ``alpha_train_common`` (largest grid level
  at which both policies meet the target, i.e. one decision model); the separately calibrated levels
  are labelled for cost-risk comparisons only and must not be combined into ``V0 - VT``.
* C05/F02 -- for DM-observing structures, fallback actions of bins without development probability
  cannot be checked against the (undefined) decision-time DM estimate; such bins are listed and the
  evaluation states falling into them are counted.

Second review R1 (2026-09-25; ``docs/value_definition.md``): a frozen policy is the *deterministic*
with-information policy, the only kind that may be fed.  :func:`freeze_policy` and
:func:`calibrate_alpha_train` therefore require an explicit
``value_definition="operational_deterministic_cost_difference"`` (other definitions are refused: the
randomised reference policies are theoretical and never frozen, the contrast is a diagnostic);
:func:`evaluate_frozen_policy` refuses a policy without that recorded definition.  The development
record keeps the other definitions only as labelled references/diagnostics.

Red-team FIX_A (2026-09-25): a policy whose operational value is a randomisation channel only (a
pure-noise or garbled assay used as a coin: the randomised same-class reference is 0, or a
state-independent device with the same bin probabilities reaches the same saving), or whose positive
operational value cannot be assessed, is never frozen (:class:`~.value.RandomizationChannelError`);
the development record carries the assessment (status, benchmark, diagnostic ratio) and the
identification label.  :func:`calibrate_alpha_train` therefore computes both references and records
refused levels instead of freezing them.

Third review R3B (2026-09-25; ``docs/value_semantics_decision.md``): the development record keeps the
heuristic minimum under ``heuristic_min_of_two_policy_values_development_heuristic`` (formerly
``executable_information_supported_value_development``) and the matched-benchmark ratio as
``matched_uninformative_contrast_ratio``; neither is described as an assay's executable value any more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..errors import InvalidProblemError, LeakageError
from ..evaluation import evaluate
from ..evaluation.stats import clopper_pearson
from ..hashing import stable_hash
from ..nutrition.constraints import CompiledConstraints
from ..uncertainty.base import DrawSet
from ..uncertainty.streams import RandomStreams
from .library import CandidateLibrary, RiskTable
from .policy import RISK_TOL_DEFAULT
from .prior import DEVELOPMENT_STREAMS, ObservedComponent, PriorStates
from .value import (
    DefinitionLike,
    InformationStructure,
    InformationValueResult,
    RandomizationChannelError,
    ValueDefinition,
    compute_information_value,
    require_executable_operational_value,
    resolve_value_definition,
)

_FROZEN_DEFINITION = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE


def _frozen_value_definition(value_definition: Optional[DefinitionLike], context: str) -> ValueDefinition:
    """Explicit definition for a frozen (fed) policy: only the operational deterministic difference (R1)."""
    d = resolve_value_definition(value_definition, context=context, allow_legacy=False)
    if d is not _FROZEN_DEFINITION:
        raise InvalidProblemError(
            f"{context}: a frozen policy is the deterministic bin -> ration map that may be fed; its value is "
            f"{_FROZEN_DEFINITION.value}. {d.value} is "
            + ("a theoretical reference over randomised policies, which are never frozen or recommended for feeding"
               if d is ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE
               else "a heuristic minimum of two values from two different policy classes, not the value of a policy "
                    "(R3B)" if d is ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES
               else "a diagnostic contrast, not a policy value")
            + " (docs/value_definition.md)")
    return d

__all__ = ["FrozenSignalPolicy", "freeze_policy", "FrozenPolicyEvaluation", "evaluate_frozen_policy",
           "signal_bins_for_states", "calibrate_alpha_train"]


@dataclass(frozen=True)
class FrozenSignalPolicy:
    """Development-frozen bin -> candidate map plus the no-information candidate."""

    policy_id: str
    structure: InformationStructure
    library: CandidateLibrary
    assignment: np.ndarray
    no_information_candidate: int
    alpha: float
    unknown_policy: str
    development: Mapping[str, Any]

    def __post_init__(self) -> None:
        a = np.array(self.assignment, dtype=int)
        a.setflags(write=False)
        object.__setattr__(self, "assignment", a)
        if self.structure.binning is None or a.shape != (self.structure.binning.n_bins,):
            raise InvalidProblemError("FrozenSignalPolicy: one action per frozen bin is required")
        if np.any(a < 0) or np.any(a >= self.library.size):
            raise InvalidProblemError("FrozenSignalPolicy: every bin needs a valid candidate (fallback declared)")
        if not (0 <= int(self.no_information_candidate) < self.library.size):
            raise InvalidProblemError("FrozenSignalPolicy: invalid no-information candidate")

    @property
    def fingerprint(self) -> str:
        """Hash of structure (signal + bins), library and map."""
        sm = self.structure.signal_model
        return stable_hash("FrozenSignalPolicy/v1", self.policy_id, self.structure.info_type,
                           self.structure.binning.fingerprint, None if sm is None else sm.fingerprint(),
                           self.library.fingerprint, self.assignment, int(self.no_information_candidate),
                           float(self.alpha), self.unknown_policy)


def freeze_policy(result: InformationValueResult, structure: InformationStructure, library: CandidateLibrary,
                  policy_id: Optional[str] = None, *,
                  value_definition: Optional[DefinitionLike] = None) -> FrozenSignalPolicy:
    """Freeze the deterministic with-information policy of a development value result.

    ``value_definition`` must be given explicitly and must be
    ``operational_deterministic_cost_difference`` (R1): the frozen map is deterministic.  The
    development record stores the operational value as ``development_value`` and keeps the
    randomised reference and the contrast only as labelled theoretical/diagnostic fields.

    FIX_A: refuses (:class:`~.value.RandomizationChannelError`) a policy whose operational value is a
    randomisation channel only, or whose positive operational value cannot be assessed (references
    not computed); records the assessment and the identification label.
    """
    vdef = _frozen_value_definition(value_definition, "freeze_policy")
    if structure.structure_id != result.structure_id:
        raise InvalidProblemError("freeze_policy: result belongs to another structure")
    rc = require_executable_operational_value(result, "freeze_policy")
    if result.likelihood_kind not in ("perfect_partial_binned", "sample_analytic_gaussian", "sample_monte_carlo"):
        raise InvalidProblemError(f"freeze_policy: {result.likelihood_kind} has no frozen policy for new states "
                                  "(ideal/oracle reference only)")
    if result.provenance.get("library_fingerprint") != library.fingerprint:
        raise InvalidProblemError("freeze_policy: library differs from the one used in development")
    if not (result.VT.has_solution and result.V0.has_solution):
        raise InvalidProblemError("freeze_policy: V0 or VT has no solution; nothing to freeze")
    k0 = int(result.V0.diagnostics["chosen_index"])
    p_bin = np.asarray(result.occupancy.get("p_bin", []), dtype=float)
    dm_obs = list(result.action_space.get("dm_observed", []))
    inactive = [int(z) for z in np.flatnonzero(p_bin <= 0)] if p_bin.size else []
    dev = {"prior_fingerprint": result.provenance.get("prior_fingerprint"),
           "prior_stream_id": result.provenance.get("prior_stream_id"),
           "risk_table_fingerprint": result.provenance.get("risk_table_fingerprint"),
           "value_definition": vdef.value,
           "development_value": result.value(vdef),
           "gross_value_development": result.gross_value,
           "cost_difference_V0_minus_VT_development": result.cost_difference_V0_minus_VT,
           "information_value_semantics": result.information_value_semantics,
           "randomized_same_class_information_reference_development_theoretical":
               result.gross_value_randomized_reference,
           "contrast_vs_matched_uninformative_bins_development_diagnostic":
               result.contrast_vs_matched_uninformative_bins,
           "VT_development": result.VT.expected_cost, "V0_development": result.V0.expected_cost,
           "VT_ex_ante_risk_development": result.VT.ex_ante_risk,
           "V0_ex_ante_risk_development": result.V0.ex_ante_risk,
           "VT_method": result.VT.method, "mip_gap": result.VT.mip_gap,
           "randomization_channel": rc, "error_model_identification": result.error_model_identification,
           "heuristic_min_of_two_policy_values_development_heuristic": result.heuristic_min_of_two_policy_values,
           "development_result_fingerprint": result.fingerprint,
           "action_space": result.action_space.get("action_space"), "dm_observed": dm_obs,
           "fallback_bins": inactive,
           "fallback_bins_structurally_unverified": inactive if dm_obs else [],
           "fallback_note": ("bins without development probability get the no-information ration; with DM observed "
                             "its planned DM under the bin's DM estimate cannot be checked (E[d|z] undefined)")
           if dm_obs and inactive else ""}
    return FrozenSignalPolicy(policy_id or f"frozen:{structure.structure_id}", structure, library,
                              np.asarray(result.VT.assignment, dtype=int), k0, result.alpha, result.unknown_policy, dev)


def _component_values(draws: DrawSet, comps: tuple[ObservedComponent, ...]) -> np.ndarray:
    cols = []
    for c in comps:
        i = draws.ingredient_ids.index(c.ingredient_id)
        if c.component == "DM":
            cols.append(draws.d[:, i])
        else:
            cols.append(draws.theta[:, i, draws.nutrient_ids.index(c.component)])
    return np.stack(cols, axis=1)


def signal_bins_for_states(policy: FrozenSignalPolicy, draws: DrawSet, streams: RandomStreams,
                           noise_stream: str = "assay_signal_eval") -> np.ndarray:
    """Signal bin of every evaluation state (signal simulated from ``noise_stream`` for sample info)."""
    st = policy.structure
    t = _component_values(draws, st.components)
    if st.info_type == "sample":
        z = st.signal_model.simulate(t, streams.generator(noise_stream))
    else:
        z = t
    return st.binning.bin_index(z)


@dataclass(frozen=True)
class FrozenPolicyEvaluation:
    """Outcome of a frozen policy vs. the no-information candidate on the same evaluation states."""

    policy_id: str
    draw_stream_id: str
    noise_stream_id: Optional[str]
    n_states: int
    policy_mean_cost: float
    constant_mean_cost: float
    realised_cost_saving_per_head_day: Optional[float]
    cost_difference_per_head_day: float
    saving_status: str
    target_alpha: float
    policy_joint_violation: Mapping[str, Any]
    constant_joint_violation: Mapping[str, Any]
    risk_target_check: Mapping[str, Any]
    bin_counts: np.ndarray
    action_counts: Mapping[str, int]
    in_sample: bool
    is_synthetic: bool
    notes: str = ("MC interval = fixed-distribution simulation error only; realised saving is the ex-post average "
                  "over independent states under the declared model, not a field result")
    extra: Mapping[str, Any] = field(default_factory=dict)


def _joint_summary(viol: np.ndarray, unk: np.ndarray, unknown_policy: str) -> dict[str, Any]:
    n = int(viol.size)
    nv, nu = int(viol.sum()), int(unk.sum())
    k = nv + nu if unknown_policy == "count_as_violation" else nv
    lo, hi = clopper_pearson(k, n)
    return {"n": n, "n_violated": nv, "n_unknown": nu, "rate_lower": nv / n, "rate_upper": (nv + nu) / n,
            "rate_used": k / n, "mc_clopper_pearson_95": [lo, hi], "interval_type": "MC_only_fixed_distribution"}


def evaluate_frozen_policy(policy: FrozenSignalPolicy, draws: DrawSet, constraints: CompiledConstraints,
                           streams: RandomStreams, *, prices: Any = None,
                           noise_stream: str = "assay_signal_eval", target_alpha: Optional[float] = None,
                           risk_tol: float = RISK_TOL_DEFAULT) -> FrozenPolicyEvaluation:
    """Evaluate a frozen policy and the no-information candidate on independent states.

    Formal evaluation uses ``test`` draws after the protocol freeze; development-stream draws are
    accepted but flagged ``in_sample``.  The noise stream must differ from the development noise
    stream used to build Monte Carlo likelihoods.

    ``target_alpha`` (default: the policy's training level) is the risk target of the comparison.
    ``realised_cost_saving_per_head_day`` is reported only if both the policy and the
    no-information candidate meet it on these states (``rate_used <= target_alpha + risk_tol``);
    otherwise it is ``None`` and only ``cost_difference_per_head_day`` (``C_0 - mean policy cost``,
    at unequal risk) is given (contract §11 E1, T7.1, D09).

    The policy must carry the explicit ``value_definition`` recorded by :func:`freeze_policy`
    (``operational_deterministic_cost_difference``, R1); the realised saving is the operational
    deterministic cost difference on these states.
    """
    rec = policy.development.get("value_definition") if isinstance(policy.development, Mapping) else None
    if rec is None:
        raise InvalidProblemError("evaluate_frozen_policy: the frozen policy records no explicit value_definition; "
                                  "freeze it with freeze_policy(..., value_definition="
                                  f"{_FROZEN_DEFINITION.value!r}) (R1)")
    _frozen_value_definition(rec, "evaluate_frozen_policy")
    rcd = policy.development.get("randomization_channel")
    if not isinstance(rcd, Mapping) or "status" not in rcd:
        raise InvalidProblemError("evaluate_frozen_policy: the frozen policy carries no randomisation-channel record; "
                                  "freeze it with freeze_policy (FIX_A)")
    if rcd["status"] == "randomization_only" or (rcd["status"] == "not_assessable"
                                                 and policy.development.get("information_value_semantics")
                                                 == "information_value_within_library"):
        raise RandomizationChannelError(f"evaluate_frozen_policy: the frozen policy's operational value is "
                                        f"{rcd['status']} ({rcd.get('reason')}); it is not evaluated as an assay policy "
                                        "(FIX_A)")
    if noise_stream in ("assay_signal_dev", "opt", "validation", "test", "outer"):
        raise InvalidProblemError("use a dedicated evaluation noise stream (not a core or development stream)")
    if tuple(draws.ingredient_ids) != policy.library.ingredient_ids:
        raise InvalidProblemError("evaluate_frozen_policy: draw ingredient order differs from the library")
    unknown_policy = policy.unknown_policy
    z = signal_bins_for_states(policy, draws, streams, noise_stream)
    k_of_state = policy.assignment[z]                     # action depends on the bin only
    S = draws.n_draws
    viol = np.zeros(S, dtype=bool)
    unk = np.zeros(S, dtype=bool)
    cost = np.zeros(S)
    for k in np.unique(k_of_state):
        idx = np.flatnonzero(k_of_state == k)
        ev = evaluate(policy.library.decisions[k], draws.theta[idx], draws.d[idx], constraints, prices=prices,
                      draw_stream_id=draws.stream_id, is_synthetic=draws.is_synthetic)
        viol[idx] = ev.joint_violation
        unk[idx] = ev.joint_unknown
        cost[idx] = policy.library.costs[k]
    k0 = int(policy.no_information_candidate)
    ev0 = evaluate(policy.library.decisions[k0], draws.theta, draws.d, constraints, prices=prices,
                   draw_stream_id=draws.stream_id, is_synthetic=draws.is_synthetic)
    if unknown_policy == "error_if_any" and (unk.any() or ev0.joint_unknown.any()):
        raise InvalidProblemError("undefined outcomes on evaluation states; declare an explicit unknown_policy")
    c0 = float(policy.library.costs[k0])
    counts = np.bincount(z, minlength=policy.structure.binning.n_bins)
    acts = {policy.library.candidate_ids[int(k)]: int(np.sum(k_of_state == k)) for k in np.unique(k_of_state)}
    pj = _joint_summary(viol, unk, unknown_policy)
    cj = _joint_summary(ev0.joint_violation, ev0.joint_unknown, unknown_policy)
    ta = float(policy.alpha) if target_alpha is None else float(target_alpha)
    if not (0.0 <= ta <= 1.0):
        raise InvalidProblemError("target_alpha must be in [0, 1]")
    p_meets = bool(pj["rate_used"] <= ta + risk_tol)
    c_meets = bool(cj["rate_used"] <= ta + risk_tol)
    diff = float(c0 - cost.mean())
    if p_meets and c_meets:
        saving, saving_status = diff, "both_meet_target_alpha_point"
    else:
        saving, saving_status = None, "cost_difference_at_unequal_risk"
    unver = list(policy.development.get("fallback_bins_structurally_unverified", []))
    n_unver = int(np.isin(z, unver).sum()) if unver else 0
    target = {"alpha_policy_training_level": float(policy.alpha), "target_alpha": ta,
              "policy_meets_target_point": p_meets, "constant_meets_target_point": c_meets,
              "policy_point_exceeds_alpha": bool(pj["rate_used"] > policy.alpha),
              "policy_mc_lower_exceeds_alpha": bool(pj["mc_clopper_pearson_95"][0] > policy.alpha),
              "constant_point_exceeds_alpha": bool(cj["rate_used"] > policy.alpha),
              "constant_mc_lower_exceeds_alpha": bool(cj["mc_clopper_pearson_95"][0] > policy.alpha),
              "note": "the development risk constraint holds on development states only; exceedance on independent "
                      "states is in-sample optimism of the finite-state policy (see calibrate_alpha_train)"}
    return FrozenPolicyEvaluation(
        policy.policy_id, draws.stream_id,
        streams.stream_id(noise_stream) if policy.structure.info_type == "sample" else None, S,
        float(cost.mean()), c0, saving, diff, saving_status, ta, pj, cj, target, counts, acts,
        draws.stream in DEVELOPMENT_STREAMS, bool(draws.is_synthetic or policy.library.is_synthetic),
        extra={"policy_fingerprint": policy.fingerprint, "development": dict(policy.development),
               "n_states_in_structurally_unverified_fallback_bins": n_unver,
               "information_value_semantics": policy.development.get("information_value_semantics"),
               "value_definition": _FROZEN_DEFINITION.value,
               "value_definition_note": "realised saving / cost difference of the frozen deterministic policy vs the "
                                        "no-information candidate (operational); no randomised policy is evaluated",
               "development_randomization_channel_status": rcd["status"],
               "development_matched_uninformative_contrast_ratio": rcd.get("matched_uninformative_contrast_ratio"),
               "development_heuristic_min_of_two_policy_values_heuristic":
                   policy.development.get("heuristic_min_of_two_policy_values_development_heuristic"),
               "realised_saving_note": "the realised operational saving is the cost difference of the frozen "
                                       "deterministic policy; it can contain a randomisation channel (the matched "
                                       "benchmark ratio above is a diagnostic, not a randomisation share, and a zero "
                                       "benchmark does not show that no channel exists); the development heuristic "
                                       "minimum is a heuristic, not an assay's value (R3B; "
                                       "docs/value_semantics_decision.md)"})


def calibrate_alpha_train(structure: InformationStructure, prior: PriorStates, risk: RiskTable,
                          library: CandidateLibrary, validation_draws: DrawSet, constraints: CompiledConstraints,
                          streams: RandomStreams, alpha: float, grid: Any, *, prices: Any = None,
                          criterion: str = "point", noise_stream: str = "assay_signal_validation",
                          value_definition: Optional[DefinitionLike] = None,
                          **value_kw) -> dict[str, Any]:
    """Development calibration of the training risk level (analogue of M2's ``alpha_train``, T4/T8.1).

    For every ``a`` in ``grid`` (each ``<= alpha``) the constant and the signal policy are optimised on
    the development states at risk level ``a``, frozen, and evaluated on ``validation`` draws.  The
    largest ``a`` whose validation risk meets ``alpha`` (``criterion='point'``: point estimate;
    ``'mc_upper'``: 95 % Clopper-Pearson upper bound) is reported separately for the constant and the
    signal policy.  Only the ``validation`` stream is accepted (never ``test``).

    ``alpha_train_common`` is the largest level at which *both* policies meet the target: a gross
    value ``V0 - VT`` must be computed at one common level (one decision model, T7.1).  The
    separately calibrated levels (``alpha_train_signal_policy`` / ``alpha_train_constant_policy``)
    may only feed a cost-risk comparison (``usage`` field), never an information value (F04).

    ``value_definition`` must be ``operational_deterministic_cost_difference`` (explicit; R1): the
    calibrated objects are frozen deterministic policies.

    FIX_A: both references are computed at every level (unless the caller explicitly passes
    ``include_randomized_reference`` / ``include_randomization_benchmark``); a level whose operational
    value is a randomisation channel only is not frozen: its row records ``frozen=False`` with the
    reason and never counts as meeting the target.
    """
    if validation_draws.stream != "validation":
        raise LeakageError(f"calibrate_alpha_train needs 'validation' draws, got {validation_draws.stream!r}")
    vdef = _frozen_value_definition(value_definition, "calibrate_alpha_train")
    if criterion not in ("point", "mc_upper"):
        raise InvalidProblemError("criterion must be 'point' or 'mc_upper'")
    levels = sorted({float(a) for a in grid})
    if not levels or levels[-1] > float(alpha) + 1e-15 or levels[0] < 0:
        raise InvalidProblemError("grid must be non-empty with 0 <= a <= alpha")
    rows: list[dict[str, Any]] = []
    kw = {"include_randomized_reference": True, "include_randomization_benchmark": True, **value_kw}
    for a in levels:
        res = compute_information_value(structure, prior, risk, a, value_definition=vdef, **kw)
        row: dict[str, Any] = {"alpha_train": a, "V0_dev": res.V0.expected_cost, "VT_dev": res.VT.expected_cost,
                               "status": res.gross_value_status,
                               "randomization_channel_status": res.randomization_channel["status"]}
        if res.V0.has_solution and res.VT.has_solution:
            try:
                pol = freeze_policy(res, structure, library, value_definition=vdef)
            except RandomizationChannelError as exc:
                row.update({"frozen": False, "refused": str(exc)})
                rows.append(row)
                continue
            row["frozen"] = True
            ev = evaluate_frozen_policy(pol, validation_draws, constraints, streams, prices=prices,
                                        noise_stream=noise_stream)
            key = 1 if criterion == "mc_upper" else None
            pr = ev.policy_joint_violation["mc_clopper_pearson_95"][1] if key else ev.policy_joint_violation["rate_used"]
            cr = ev.constant_joint_violation["mc_clopper_pearson_95"][1] if key else \
                ev.constant_joint_violation["rate_used"]
            row.update({"policy_validation_risk": pr, "constant_validation_risk": cr,
                        "policy_validation_cost": ev.policy_mean_cost, "constant_validation_cost": ev.constant_mean_cost,
                        "policy_meets": bool(pr <= alpha), "constant_meets": bool(cr <= alpha)})
        rows.append(row)
    ok_p = [r["alpha_train"] for r in rows if r.get("policy_meets")]
    ok_c = [r["alpha_train"] for r in rows if r.get("constant_meets")]
    ok_b = [r["alpha_train"] for r in rows if r.get("policy_meets") and r.get("constant_meets")]
    return {"alpha_target": float(alpha), "criterion": criterion, "value_definition": vdef.value, "table": rows,
            "alpha_train_signal_policy": max(ok_p) if ok_p else None,
            "alpha_train_constant_policy": max(ok_c) if ok_c else None,
            "alpha_train_common": max(ok_b) if ok_b else None,
            "usage": {"alpha_train_common": "operational_deterministic_cost_difference V0 - VT (one decision model; "
                                            "explicit value_definition, R1)",
                      "alpha_train_signal_policy/alpha_train_constant_policy":
                          "cost-risk comparison only; never combine separately calibrated levels into V0 - VT"},
            "validation_stream_id": validation_draws.stream_id,
            "note": "None = no grid level meets the target on validation (reported, not relaxed)"}
