"""Red-team lens A regressions (round 2, FIX_A).  ALL NUMBERS ARE SYNTHETIC unless stated.

R1 -- the operational (deterministic) value contains a randomisation channel: a pure-noise or garbled
assay gets a positive operational value.  These tests pin that such assays are never bought (money
conversions), ranked (decision-value strategy) or frozen on the operational basis, that
``executable_information_supported_value = min(operational, randomised reference)`` behaves as the
minimum (R3B: renamed ``heuristic_min_of_two_policy_values``; the FIX_A reading "threshold-free executable
basis" is withdrawn -- it is a heuristic, not garbling-monotone), that money conversions only take values
bound to their result and keep scenario labels (FIX3_BC: and need a complete decision problem or an
explicit, labelled development-diagnostic flag), and that the matched uninformative-bins policy is
reported as a hypothetical randomisation device, not an executable feeding policy.

R2 -- identification needs *verified* object metadata: a copied model fingerprint does not link a
truth record, explicit metadata of a non-synthetic world never identify, a de-convolved true state is
checked against its numeric record and the states, the signal must name its measurement process, a
truth given as prior states must be the same world, sourced error values are traced to the locator
table, perfect information carries an identification label, and the ``EnergyColumnModel`` route of the
development case keeps the factory metadata (non-synthetic draws without resolvable metadata raise).

Fabricated source strings below (``TEST-FABRICATED-...``) are deliberately fake inputs whose only
purpose is to be refused; they are not sources.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import norm

from engine_test_helpers import conc, dm_offer, ing, problem, two_ingredient_problem
from ration_reliability.datamodel import Provenance, RationDecision, ValueStatus
from ration_reliability.errors import InvalidProblemError
from ration_reliability.information import (
    AssayBudget,
    AssayOption,
    BatchCoverage,
    CandidateLibrary,
    ComponentErrorModel,
    ComponentMetadata,
    InformationStructure,
    MetadataConflictError,
    ObservedComponent,
    PerHeadDayValue,
    PriorStates,
    RandomizationChannelError,
    SamplingProtocol,
    SignalBinning,
    SignalModel,
    StateMetadata,
    ValueDefinition,
    batch_gross_value,
    break_even_max_cost_per_batch,
    calibrate_alpha_train,
    compute_information_value,
    compute_risk_table,
    conditioning_from_likelihood,
    decompose_observed_variance,
    default_error_locators,
    double_count_guard,
    freeze_policy,
    generate_conditional_library,
    inventory_coverage_check,
    load_error_locators,
    net_value_per_batch,
    per_head_day_value,
    perfect_partial_likelihood,
    quantile_edges,
    randomization_channel_assessment,
    resolve_truth_link,
    strategy_decision_value,
    verify_state_metadata,
)
from ration_reliability.nutrition import energy as E
from ration_reliability.uncertainty import DrawSet, IndependentNormalModel, RandomStreams
from ration_reliability.uncertainty.factory import build_uncertainty_model
from ration_reliability.uncertainty.spec import UncertaintySpec

REPO = Path(__file__).resolve().parents[2]
OP = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE
RR = ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE
CON = ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS
EIS = ValueDefinition.EXECUTABLE_INFORMATION_SUPPORTED_VALUE
SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-FIX-A")
FCP, CCP = ObservedComponent("F", "CP"), ObservedComponent("C", "CP")
HIST = "synthetic_measurement:FIX-A-HIST-LAB"
COSTS_UNKNOWN = {k: None for k in ("sampling", "laboratory", "logistics", "waiting", "reformulation")}
#: FIX3_BC (round-3 red team B-1): money conversions need a complete decision problem; the successful conversions
#: below are labelled development diagnostics.  Only this keyword was added to them; no expected value changed.
DIAG = {"diagnostic_without_decision_problem": True}


# =================================================================================================
# fixtures
# =================================================================================================

def _md_counterexample(basis="true_batch_state", label="fix_a", **kw):
    """Synthetic object metadata of the review counterexample world (F:CP stochastic, C:CP a point value)."""
    base = StateMetadata.synthetic((FCP,), basis, label=label, **kw)
    ccp = ComponentMetadata(CCP, "not_applicable", None, "none", None, None, True, "synthetic_test_only",
                            f"SYNTHETIC:{label}", None)
    return StateMetadata(base.components + (ccp,), "explicit_prior_definition", is_synthetic=True)


def _cx(metadata="default", theta_f=(0.08, 0.12), is_synthetic=True):
    """The review counterexample: candidates A (2.90) / B (3.56); F:CP 8 % or 12 % with prior 1/2."""
    prob = problem([ing("F", 0.40, {"CP": 0.10}, forage=1.0), ing("C", 0.80, {"CP": 0.40})], ["CP"],
                   [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0)], {"F": 0.04, "C": 0.32},
                   problem_id="fix_a_counterexample")
    ids, dh = prob.ingredient_ids, prob.dm_estimates()
    lib = CandidateLibrary.from_decisions(prob, [RationDecision(ids, np.array([42.5, 3.75]), dh, "synthetic_A"),
                                                 RationDecision(ids, np.array([37.0, 6.50]), dh, "synthetic_B")],
                                          ["A", "B"])
    md = _md_counterexample() if metadata == "default" else metadata
    prior = PriorStates.from_discrete(np.array([[[theta_f[0]], [0.40]], [[theta_f[1]], [0.40]]]),
                                      np.array([[0.4, 0.8]] * 2), [0.5, 0.5], ids, ["CP"], label="fix_a_cx",
                                      is_synthetic=is_synthetic, metadata=md)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    return prob, lib, prior, risk


def _sig(comp, sd, sid, lab=0.0, mm=None, prov=SYN, synthetic=True, protocol=SamplingProtocol()):
    pv = {k: prov for k, v in (("sampling_sd", sd), ("lab_repeatability_sd", lab)) if v}
    return SignalModel(sid, (ComponentErrorModel(comp, sd, lab, provenance=pv, is_synthetic=synthetic,
                                                 measurement_model_id=mm),), protocol, is_synthetic=synthetic)


def _st(comp, sd, edges, sid, **kw):
    return InformationStructure(sid, "sample", binning=SignalBinning((comp,), (tuple(edges),), "synthetic edges"),
                                signal_model=_sig(comp, sd, sid, **kw))


def _review_signal():
    sd = 0.04 / (norm.ppf(0.3) - norm.ppf(0.1))
    return sd, 0.08 + sd * norm.ppf(0.3)


def _noise_on_ccp(sid="pure_noise_C_CP"):
    """An assay of C:CP, which does not vary in the prior: bins 0.2 / 0.8 whatever the state (pure noise)."""
    sdn = 0.05
    return sdn, 0.40 + sdn * norm.ppf(0.2)


def _garbled_edge(sd):
    """F:CP signal with error ``sd``; the edge makes P(low | bad) just below 0.2 (risk of (A, B) <= 0.1)."""
    return 0.08 + sd * norm.ppf(0.2) - 1e-9 * sd


def _cov_inv(lib):
    cov = BatchCoverage(100.0, 14.0, "synthetic_test_only", "synthetic_test_only", 14.0)
    return cov, inventory_coverage_check(lib.Q, lib.ingredient_ids, cov, {"F": 1e9, "C": 1e9})


def _continuous(n=300, seed=1103):
    """Continuous (many-state) synthetic world of the engine red-team tests (F: CP/NDF/DM uncertain)."""
    F = ing("F", 0.35, {"CP": 0.10, "NDF": 0.45}, forage=1.0)
    C = ing("C", 0.88, {"CP": 0.40, "NDF": 0.20})
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            conc("fndf_min", {"G:forage:NDF": 1.0}, "ge", 18.0)]
    prob = problem([F, C], ["CP", "NDF"], cons, {"F": 0.035, "C": 0.30}, problem_id="fix_a_continuous")
    model = IndependentNormalModel("fix_a_continuous", prob.ingredient_ids, prob.nutrient_ids,
                                   np.array([[0.10, 0.45], [0.40, 0.20]]), np.array([[0.012, 0.03], [0.0, 0.0]]),
                                   np.array([0.35, 0.88]), np.array([0.02, 0.0]), is_synthetic=True)
    FNDF, FDM = ObservedComponent("F", "NDF"), ObservedComponent("F", "DM")
    prior = PriorStates.from_drawset(model.draw(RandomStreams(seed), "opt", n),
                                     metadata=StateMetadata.synthetic((FCP, FNDF, FDM), "true_batch_state",
                                                                      label="fix_a_continuous"))
    bcp = SignalBinning((FCP,), (quantile_edges(prior.component_values((FCP,)), 4),), "q opt")
    bdm = SignalBinning((FDM,), (quantile_edges(prior.component_values((FDM,)), 3),), "q opt")
    cond = conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FCP,), bcp), prefix="cp|") + \
        conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FDM,), bdm), prefix="dm|")
    lib = generate_conditional_library(prob, prior, cond)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    return prob, model, prior, bcp, lib, risk


@pytest.fixture(scope="module")
def cx():
    return _cx()


@pytest.fixture(scope="module")
def continuous():
    return _continuous()


# =================================================================================================
# R1-1  pure noise: positive operational value, never bought / ranked / frozen
# =================================================================================================

def test_pure_noise_assay_is_never_bought_ranked_or_frozen_on_the_operational_basis(cx):
    prob, lib, prior, risk = cx
    sdn, edgen = _noise_on_ccp()
    st = _st(CCP, sdn, [edgen], "pure_noise_C_CP")
    rn = compute_information_value(st, prior, risk, 0.10)
    # the numbers of the finding are unchanged (nothing truncated): operational 0.132 = benchmark, reference 0
    assert rn.gross_value == pytest.approx(0.132, abs=1e-12)
    assert rn.gross_value_randomized_reference == pytest.approx(0.0, abs=1e-9)
    assert rn.randomization_benchmark_value == pytest.approx(0.132, abs=1e-12)
    rc = rn.randomization_channel
    assert rc["status"] == "randomization_only" and rc["executable_use_allowed"] is False
    assert rc["operational_information_supported"] is False
    assert rn.executable_information_supported_value == pytest.approx(0.0, abs=1e-9)
    assert any("randomization channel (randomization_only)" in w for w in rn.warnings)
    # money: the operational value is refused.  R3B/FIX3_BC: the heuristic minimum (formerly called the "executable
    # information-supported value") is 0 here, but it is not a money basis; converting it needs an explicit
    # development-diagnostic flag and is labelled (no complete decision problem exists)
    cov, inv = _cov_inv(lib)
    with pytest.raises(RandomizationChannelError, match="randomisation channel"):
        per_head_day_value(rn, OP)
    with pytest.raises(RandomizationChannelError):
        break_even_max_cost_per_batch(rn, cov, inventory_check=inv, value_definition=OP)
    with pytest.raises(RandomizationChannelError):
        net_value_per_batch(rn, cov, dict(COSTS_UNKNOWN, laboratory=80.0), inventory_check=inv, value_definition=OP)
    be = break_even_max_cost_per_batch(rn, cov, inventory_check=inv, value_definition=EIS, **DIAG)
    assert float(be) == pytest.approx(0.0, abs=1e-6) and be.binding.randomization_channel["status"] == \
        "randomization_only"
    assert be.binding.excluded_from_paper_main_results is True and \
        be.binding.money_basis == "development_diagnostic_without_decision_problem"                    # FIX3_BC
    # freezing: refused at alpha 0.10 and at 0.12 (the red-team frozen case)
    with pytest.raises(RandomizationChannelError):
        freeze_policy(rn, st, lib, value_definition=OP)
    r12 = compute_information_value(st, prior, risk, 0.12)
    assert r12.randomization_channel["status"] == "randomization_only"
    with pytest.raises(RandomizationChannelError):
        freeze_policy(r12, st, lib, value_definition=OP)
    # ranking: operational and executable definitions select nothing; the randomised reference picks the precise assay
    sd0, edge0 = _review_signal()
    opts = [AssayOption("precise_F_CP", "F", ("CP",), SignalBinning((FCP,), ((edge0,),), "syn"),
                        _sig(FCP, sd0, "p"), 60.0),
            AssayOption("pure_noise_C_CP", "C", ("CP",), SignalBinning((CCP,), ((edgen,),), "syn"),
                        _sig(CCP, sdn, "n"), 60.0)]
    thr = {"min_marginal_value": 0.0, "min_marginal_value_status": "synthetic_test_only"}
    s_op = strategy_decision_value(opts, prior, risk, 0.10, AssayBudget(max_panels=1), prior_variance_basis={},
                                   value_definition=OP, **thr)
    assert s_op.selected == ()
    assert "pure_noise_C_CP" in [x["selection"][-1] for x in s_op.details["randomization_channel_excluded"]]
    # FIX3_BC (B-4): a screened-out option scores None with its reason (was -inf, which also meant "undefined")
    assert dict(s_op.ranking)["pure_noise_C_CP"] is None
    assert s_op.details["ranking_status"]["pure_noise_C_CP"].startswith("screened_out:")
    s_eis = strategy_decision_value(opts, prior, risk, 0.10, AssayBudget(max_panels=1), prior_variance_basis={},
                                    value_definition=EIS, **thr)
    # R3B (third review): the renamed definition's role is "heuristic" (was "executable_decision_basis")
    assert s_eis.selected == () and s_eis.details["value_role"] == "heuristic"
    s_rr = strategy_decision_value(opts, prior, risk, 0.10, AssayBudget(max_panels=1), prior_variance_basis={},
                                   value_definition=RR, **thr)
    assert s_rr.selected == ("precise_F_CP",)


def test_definition_report_keeps_the_randomisation_device_out_of_the_executable_block(cx):
    prob, lib, prior, risk = cx
    sdn, edgen = _noise_on_ccp()
    rep = compute_information_value(_st(CCP, sdn, [edgen], "noise"), prior, risk, 0.10).definition_report()
    det = rep["deterministic_policies"]
    assert set(det) == {"role", "V0_constant", "VT_signal_policy"}
    assert "feeding candidate only if" in det["role"]
    assert det["VT_signal_policy"]["feeding_candidate"] is False            # the coin-driven (A, B) map
    assert det["VT_signal_policy"]["assignment_candidate_ids"] == ["A", "B"]
    dev = rep["diagnostic_randomisation_devices"]
    assert dev["role"] == "hypothetical_randomisation_device" and dev["not_a_feeding_recommendation"] is True
    assert dev["VT_matched_uninformative_bins_policy"]["assignment_candidate_ids"] == ["A", "B"]
    assert rep["randomization_channel"]["status"] == "randomization_only"
    # an informative signal at the review alpha keeps a feeding candidate (its saving is 0, nothing randomised)
    sd0, edge0 = _review_signal()
    rep0 = compute_information_value(_st(FCP, sd0, [edge0], "review"), prior, risk, 0.10).definition_report()
    assert rep0["deterministic_policies"]["VT_signal_policy"]["feeding_candidate"] is True
    assert rep0["randomization_channel"]["status"] == "no_operational_saving"


# =================================================================================================
# R1-2  garbled signals (discrete) and the review signal's alpha sweep
# =================================================================================================

@pytest.mark.parametrize("sd", [0.2, 1.0, 5.0])
def test_garbled_signals_are_randomisation_channels_and_are_not_bought(cx, sd):
    prob, lib, prior, risk = cx
    r = compute_information_value(_st(FCP, sd, [_garbled_edge(sd)], f"garbled{sd}"), prior, risk, 0.10)
    assert r.gross_value > 0.1 and r.gross_value_randomized_reference < 0.005          # red-team numbers
    rc = r.randomization_channel
    assert rc["status"] == "randomization_only" and rc["contrast_vs_matched_uninformative_bins"] <= rc["tolerance"]
    cov, inv = _cov_inv(lib)
    with pytest.raises(RandomizationChannelError):
        per_head_day_value(r, OP)
    v = per_head_day_value(r, EIS, **DIAG)                          # = the (tiny) randomised reference
    assert float(v) == pytest.approx(r.gross_value_randomized_reference, abs=1e-12)
    assert float(break_even_max_cost_per_batch(v, cov, inventory_check=inv, value_definition=EIS, **DIAG)) < \
        1400 * 0.005


def test_operational_ranking_no_longer_prefers_the_noisier_laboratory(cx):
    """exp_r1b: the operational ranking chose ``noisy_lab_sd5`` over ``precise_lab``; the reference did not."""
    prob, lib, prior, risk = cx
    sd0, edge0 = _review_signal()
    opts = [AssayOption("precise_lab", "F", ("CP",), SignalBinning((FCP,), ((edge0,),), "syn"), _sig(FCP, sd0, "p"),
                        60.0),
            AssayOption("noisy_lab_sd5", "F", ("CP",), SignalBinning((FCP,), ((_garbled_edge(5.0),),), "syn"),
                        _sig(FCP, 5.0, "g"), 60.0)]
    thr = {"min_marginal_value": 0.01, "min_marginal_value_status": "synthetic_test_only"}
    kw = dict(prior_variance_basis={FCP: "true_batch_state"}, **thr)
    op = strategy_decision_value(opts, prior, risk, 0.10, AssayBudget(max_panels=1), value_definition=OP, **kw)
    assert op.selected == ()
    assert {x["selection"][-1] for x in op.details["randomization_channel_excluded"]} == {"precise_lab",
                                                                                           "noisy_lab_sd5"}
    assert op.details["greedy_history"][-1]["eligible"] is False
    eis = strategy_decision_value(opts, prior, risk, 0.10, AssayBudget(max_panels=1), value_definition=EIS, **kw)
    assert eis.selected == ()
    rr = strategy_decision_value(opts, prior, risk, 0.10, AssayBudget(max_panels=1), value_definition=RR, **kw)
    assert rr.selected == ("precise_lab",)


@pytest.mark.parametrize("alpha", [0.15, 0.2, 0.25, 0.3, 0.4])
def test_review_signal_alpha_sweep_operational_equals_benchmark(cx, alpha):
    prob, lib, prior, risk = cx
    sd0, edge0 = _review_signal()
    st = _st(FCP, sd0, [edge0], "review")
    r = compute_information_value(st, prior, risk, alpha)
    assert r.gross_value == pytest.approx(r.randomization_benchmark_value, abs=1e-12)     # red-team E1
    assert r.randomization_channel["status"] == "randomization_only"
    assert r.executable_information_supported_value == pytest.approx(r.gross_value_randomized_reference, abs=1e-12)
    with pytest.raises(RandomizationChannelError):
        per_head_day_value(r, OP)
    with pytest.raises(RandomizationChannelError):
        freeze_policy(r, st, lib, value_definition=OP)


# =================================================================================================
# R1-3  continuous (many-state) prior: a near-noise signal has a tiny positive contrast
# =================================================================================================

def test_near_noise_signal_in_a_continuous_prior_is_not_bought(continuous):
    prob, model, prior, bcp, lib, risk = continuous
    noise = InformationStructure("noise50", "sample", binning=bcp, signal_model=_sig(FCP, 50.0, "noise50"))
    good = InformationStructure("good", "sample", binning=bcp, signal_model=_sig(FCP, 0.006, "good"))
    rn = compute_information_value(noise, prior, risk, 0.05)
    rg = compute_information_value(good, prior, risk, 0.05)
    rcn = rn.randomization_channel
    # a status-only gate would let it through: contrast and reference are tiny but positive in a finite prior ...
    assert rcn["status"] == "partly_randomization_channel" and rcn["randomization_share_of_operational"] > 0.99
    # ... the operational value is refused as a money basis (R3B/FIX3_BC: the heuristic minimum below is only a
    # labelled diagnostic, not an "information-supported value")
    assert rcn["operational_information_supported"] is False
    assert rn.executable_information_supported_value < 1e-4 < rn.gross_value
    with pytest.raises(RandomizationChannelError, match="executable_information_supported_value"):
        per_head_day_value(rn, OP)
    cov = BatchCoverage(100.0, 14.0, "synthetic_test_only", "synthetic_test_only", 14.0)
    inv = inventory_coverage_check(lib.Q, lib.ingredient_ids, cov, {"F": 1e12, "C": 1e12})
    assert float(break_even_max_cost_per_batch(rn, cov, inventory_check=inv, value_definition=EIS, **DIAG)) < \
        1400 * 1e-4
    # the informative assay: the partial channel travels with the (diagnostic, heuristic) money output
    rcg = rg.randomization_channel
    assert rcg["status"] == "partly_randomization_channel" and 0 < rcg["randomization_share_of_operational"] < 1
    nv = net_value_per_batch(rg, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=EIS, **DIAG)
    eis = min(rg.gross_value, rg.gross_value_randomized_reference)
    assert nv.break_even_max_cost_per_batch == pytest.approx(1400 * eis, rel=1e-12)
    # R3B: role "heuristic" (was "executable_decision_basis")
    assert nv.value_role == "heuristic" and nv.value_binding == "bound_to_information_value_result"
    assert nv.randomization_channel_status == "partly_randomization_channel"
    assert nv.randomization_share_of_operational == pytest.approx(rg.randomization_benchmark_value / rg.gross_value)
    assert nv.result_fingerprint == rg.fingerprint and nv.error_model_identification == "identified_by_declared_sources"
    # ranking: the near-noise assay is excluded (operational) / worth nothing material (executable)
    thr = {"min_marginal_value": 1e-3, "min_marginal_value_status": "synthetic_test_only"}
    o_noise = AssayOption("noise50", "F", ("CP",), bcp, _sig(FCP, 50.0, "noise50"))
    o_good = AssayOption("good", "F", ("CP",), bcp, _sig(FCP, 0.006, "good"))
    only = strategy_decision_value([o_noise], prior, risk, 0.05, AssayBudget(max_panels=1), prior_variance_basis={},
                                   value_definition=OP, **thr)
    assert only.selected == ()
    step = only.details["greedy_history"][0]
    assert step["joint_value"] == pytest.approx(rn.gross_value) and step["eligible"] is False   # value kept, not bought
    both = strategy_decision_value([o_noise, o_good], prior, risk, 0.05, AssayBudget(max_panels=1),
                                   prior_variance_basis={}, value_definition=OP, **thr)
    assert both.selected == ("good",)
    assert both.details["greedy_history"][0]["randomization_channel"]["status"] == "partly_randomization_channel"
    e = strategy_decision_value([o_noise, o_good], prior, risk, 0.05, AssayBudget(max_panels=1),
                                prior_variance_basis={}, value_definition=EIS, **thr)
    assert e.selected == ("good",) and dict(e.ranking)["noise50"] < 1e-4


def test_executable_information_supported_value_is_the_minimum(cx, continuous):
    prob, lib, prior, risk = cx
    sd0, edge0 = _review_signal()
    res = [compute_information_value(_st(FCP, sd0, [edge0], "review"), prior, risk, 0.10),
           compute_information_value(InformationStructure("u", "uninformative", bin_probabilities=(0.2, 0.8)), prior,
                                     risk, 0.10),
           compute_information_value(InformationStructure("pp", "perfect_partial", (FCP,)), prior, risk, 0.10)]
    for r in res:
        op, rr, eis = r.gross_value, r.gross_value_randomized_reference, r.executable_information_supported_value
        assert eis == min(op, rr) and eis <= op and eis <= rr
        assert eis >= -1e-9 * max(1.0, abs(r.V0.expected_cost))                 # >= 0 under containment
    assert res[1].executable_information_supported_value == pytest.approx(0.0, abs=1e-9)   # state-independent
    chosen = compute_information_value(_st(FCP, sd0, [edge0], "review"), prior, risk, 0.10, value_definition=EIS)
    assert chosen.primary_value == pytest.approx(0.0, abs=1e-12) and chosen.primary_value_role == \
        "heuristic"                                                   # R3B: was "executable_decision_basis"
    with pytest.raises(InvalidProblemError, match="include_randomized_reference"):
        compute_information_value(_st(FCP, sd0, [edge0], "review"), prior, risk, 0.10, value_definition=EIS,
                                  include_randomized_reference=False)


def test_alpha_calibration_records_randomisation_only_levels_instead_of_freezing(cx):
    prob, lib, prior, risk = cx
    sdn, edgen = _noise_on_ccp()
    st = _st(CCP, sdn, [edgen], "noise")
    val = DrawSet(np.repeat(prior.theta, 50, axis=0), np.repeat(prior.d, 50, axis=0), "validation",
                  "synthetic/fix_a/validation", "fix_a_cx", "synthetic", prior.ingredient_ids, prior.nutrient_ids, True)
    out = calibrate_alpha_train(st, prior, risk, lib, val, prob.compiled, RandomStreams(1103), 0.10, [0.05, 0.10],
                                prices=prob.price_vector(), value_definition=OP)
    rows = {r["alpha_train"]: r for r in out["table"]}
    assert rows[0.10]["frozen"] is False and "randomisation channel" in rows[0.10]["refused"]
    assert rows[0.10]["randomization_channel_status"] == "randomization_only" and "policy_meets" not in rows[0.10]
    assert rows[0.05]["frozen"] is True and rows[0.05]["randomization_channel_status"] == "no_operational_saving"
    # a frozen policy whose development record is tampered into a randomisation channel is not evaluated either
    from ration_reliability.information import evaluate_frozen_policy
    r05 = compute_information_value(st, prior, risk, 0.05)
    pol = freeze_policy(r05, st, lib, value_definition=OP)
    assert pol.development["randomization_channel"]["status"] == "no_operational_saving"
    ev = evaluate_frozen_policy(pol, val, prob.compiled, RandomStreams(1103), prices=prob.price_vector())
    assert ev.extra["development_randomization_channel_status"] == "no_operational_saving"
    bad = dataclasses.replace(pol, development=dict(pol.development, randomization_channel=dict(
        pol.development["randomization_channel"], status="randomization_only")))
    with pytest.raises(RandomizationChannelError):
        evaluate_frozen_policy(bad, val, prob.compiled, RandomStreams(1103), prices=prob.price_vector())
    no_rec = dataclasses.replace(pol, development={k: v for k, v in pol.development.items()
                                                   if k != "randomization_channel"})
    with pytest.raises(InvalidProblemError, match="randomisation-channel record"):
        evaluate_frozen_policy(no_rec, val, prob.compiled, RandomStreams(1103), prices=prob.price_vector())


# =================================================================================================
# R1-4  money conversions take bound values and keep scenario labels
# =================================================================================================

def test_money_conversions_take_values_bound_to_their_result(cx):
    prob, lib, prior, risk = cx
    sd0, edge0 = _review_signal()
    res = compute_information_value(_st(FCP, sd0, [edge0], "review"), prior, risk, 0.10)
    assert res.error_model_identification == "identified_by_declared_sources"
    cov, inv = _cov_inv(lib)
    # red-team E6: the diagnostic contrast (-0.132) handed over as a float labelled operational is refused ...
    with pytest.raises(InvalidProblemError, match="bare float"):
        batch_gross_value(res.contrast_vs_matched_uninformative_bins, cov, inventory_check=inv, value_definition=OP)
    # ... and a randomised-reference value cannot be relabelled operational
    rr = per_head_day_value(res, RR, **DIAG)
    assert isinstance(rr, PerHeadDayValue) and rr.binding.value_role == "theoretical_reference"
    with pytest.raises(InvalidProblemError, match="cannot be relabelled"):
        break_even_max_cost_per_batch(rr, cov, inventory_check=inv, value_definition=OP)
    with pytest.raises(InvalidProblemError, match="diagnostic contrast"):
        batch_gross_value(res, cov, inventory_check=inv, value_definition=CON)
    # bound: the output remembers the result
    nv = net_value_per_batch(res, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=RR, **DIAG)
    assert nv.break_even_max_cost_per_batch == pytest.approx(1400 * res.gross_value_randomized_reference, abs=1e-9)
    assert (nv.value_binding, nv.structure_id, nv.result_fingerprint) == ("bound_to_information_value_result",
                                                                          "review", res.fingerprint)
    assert nv.error_model_identification == "identified_by_declared_sources" and not nv.identification_scenario
    # an explicitly declared bare float is labelled as such, never as an operational value
    b = batch_gross_value(0.01, cov, inventory_check=inv, value_definition=OP, unbound_value_declaration="arithmetic",
                          **DIAG)
    assert float(b) == pytest.approx(14.0) and b.binding.binding == "caller_declared_unbound_float"
    assert b.binding.value_role == "unbound_caller_declaration"
    nvu = net_value_per_batch(0.01, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=OP,
                              unbound_value_declaration="arithmetic", **DIAG)
    assert nvu.value_role == "unbound_caller_declaration" and "caller-declared float" in nvu.value_interpretation


def test_scenario_values_need_an_explicit_scenario_flag_and_keep_the_label():
    prob, lib, prior, risk = _cx(metadata=None)                  # no object metadata: a declaration only
    sd0, edge0 = _review_signal()
    ru = compute_information_value(_st(FCP, sd0, [edge0], "review"), prior, risk, 0.10,
                                   prior_variance_basis={FCP: "true_batch_state"})
    assert ru.error_model_identification == "unidentified_scenario"
    cov, inv = _cov_inv(lib)
    with pytest.raises(InvalidProblemError, match="scenario=True"):
        break_even_max_cost_per_batch(ru, cov, inventory_check=inv, value_definition=RR)          # red-team E7
    nv = net_value_per_batch(ru, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=RR, scenario=True, **DIAG)
    assert nv.identification_scenario is True and nv.error_model_identification == "unidentified_scenario"
    assert "SCENARIO: error_model_identification=unidentified_scenario" in nv.value_interpretation


# =================================================================================================
# R2-1  truth links need verification; explicit non-synthetic metadata never identify
# =================================================================================================

def test_a_copied_model_fingerprint_does_not_link_a_truth_record():
    pr = two_ingredient_problem()
    inm = IndependentNormalModel("fix_a_inm", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(),
                                 np.array([[0.0125], [0.0]]), pr.dm_estimates(), np.zeros(2), is_synthetic=True)
    p = PriorStates.from_drawset(inm.draw(RandomStreams(1103), "opt", 300))
    assert p.metadata is None
    decs = [RationDecision(pr.ingredient_ids, np.array([(20 - 20 * (c - .10) / .30) / .40, 20 * (c - .10) / .30 / .80]),
                           pr.dm_estimates(), f"cp{c:.2f}") for c in (0.18, 0.16)]
    lb = CandidateLibrary.from_decisions(pr, decs, ["safe", "nominal"])
    rk = compute_risk_table(lb, p, pr.compiled, pr.price_vector())
    st = _st(FCP, 0.006, [0.10], "s", lab=0.0045)
    forged = dataclasses.replace(StateMetadata.synthetic((FCP,), "true_batch_state", label="forged",
                                                         origin="explicit_truth_record"),
                                 model_fingerprint=p.generator_fingerprint)
    link = resolve_truth_link(forged, p)
    assert link.linked is True and link.verified is False and link.origin.endswith("claim_unverified")
    r = compute_information_value(st, p, rk, 0.05, truth=forged)                               # red-team (c)
    assert r.error_model_identification == "unidentified_scenario"
    assert r.error_model_report["truth_verified"] is False
    assert r.error_model_report["basis_sources"] == {"F:CP": "caller_declaration_unbound"}
    # the prior's own states are a verified truth; other states with the same metadata are another world (B6)
    prob, lib, prior, risk = _cx()
    assert resolve_truth_link(prior, prior).verified is True
    other = _cx(theta_f=(0.00, 0.20))[2]
    assert other.metadata.fingerprint() == prior.metadata.fingerprint()
    sd0, edge0 = _review_signal()
    with pytest.raises(InvalidProblemError, match="another world"):
        compute_information_value(_st(FCP, sd0, [edge0], "review"), prior, risk, 0.10, truth=other)


def _factory_spec(basis="true_batch_state", *, mm=None, dec="none", dec_src=None, sd=0.01, synthetic=True,
                  spec_id="fix_a_world"):
    pr = two_ingredient_problem()
    rule = {"primary_family": "TN_MM", "fallback_families": [], "on_exhausted": "error",
            "status": "synthetic_test_only" if synthetic else "research_scenario_assumption",
            "rationale": "unit test (FIX_A)", "selection_basis": "declared_rule"}
    if synthetic:
        prov = {"data_fingerprint": "synthetic:fix_a", "provenance_status": "synthetic_test_only", "source_id": None,
                "locator": None}
    else:            # deliberately fabricated strings: this world exists only to be refused
        prov = {"data_fingerprint": "TEST-FABRICATED-ROW-HASH", "provenance_status": "sourced",
                "source_id": "TEST-FABRICATED-NOT-A-SOURCE", "locator": "fabricated for a refusal test"}
    defaults = {"variance_basis": basis, "decomposition_id": dec, "decomposition_source": dec_src,
                "measurement_model_id": mm, **prov}
    spec = UncertaintySpec.from_arrays(
        spec_id, list(pr.ingredient_ids), list(pr.nutrient_ids), pr.nominal_theta(), np.array([[sd], [0.0]]),
        pr.dm_estimates(), np.array([0.0, 0.0]), purpose="unit_test", moment_semantics="target_marginal_moments",
        is_synthetic=synthetic, family_rule=rule, cell_defaults=defaults, theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0))
    return pr, spec


def _factory_world(n=400, **kw):
    pr, spec = _factory_spec(**kw)
    model = build_uncertainty_model(spec)
    prior = model.prior_states(RandomStreams(1103), "opt", n)
    decs = [RationDecision(pr.ingredient_ids, np.array([(20 - 20 * (c - .10) / .30) / .40, 20 * (c - .10) / .30 / .80]),
                           pr.dm_estimates(), f"cp{c:.2f}") for c in (0.18, 0.16)]
    lb = CandidateLibrary.from_decisions(pr, decs, ["safe", "nominal"])
    return model, prior, compute_risk_table(lb, prior, pr.compiled, pr.price_vector())


def test_factory_metadata_are_verified_against_the_registry():
    model, prior, risk = _factory_world(basis="observed_incl_sampling_and_lab", mm=HIST, spec_id="fix_a_obs")
    assert verify_state_metadata(prior.metadata)[0] == "factory_registry_verified"
    # a forged factory claim (same origin, same fingerprint, other basis) contradicts the registry: refused
    cm = prior.component_metadata(FCP)
    forged_cm = dataclasses.replace(cm, variance_basis="true_batch_state")
    forged = dataclasses.replace(prior.metadata, components=tuple(forged_cm if c.component == FCP else c
                                                                  for c in prior.metadata.components))
    assert verify_state_metadata(forged)[0] == "factory_claim_contradicted"
    with pytest.raises(MetadataConflictError, match="contradict the registered factory model"):
        dataclasses.replace(prior, metadata=forged)
    # a factory claim nobody registered in this process cannot identify
    pr = two_ingredient_problem()
    inm = IndependentNormalModel("fix_a_inm2", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(),
                                 np.array([[0.01], [0.0]]), pr.dm_estimates(), np.zeros(2), is_synthetic=True)
    d = inm.draw(RandomStreams(1103), "opt", 300)
    claim = dataclasses.replace(StateMetadata.synthetic((FCP,), "true_batch_state", label="claim"),
                                origin="factory_model_metadata", model_fingerprint=d.model_fingerprint)
    assert verify_state_metadata(claim)[0] == "factory_claim_unverifiable_in_this_process"
    p = PriorStates(d.theta, d.d, np.full(300, 1 / 300), d.ingredient_ids, d.nutrient_ids, d.stream, d.stream_id,
                    d.fingerprint, True, claim, d.model_fingerprint)
    decs = [RationDecision(pr.ingredient_ids, np.array([(20 - 20 * (c - .10) / .30) / .40, 20 * (c - .10) / .30 / .80]),
                           pr.dm_estimates(), f"cp{c:.2f}") for c in (0.18, 0.16)]
    lb = CandidateLibrary.from_decisions(pr, decs, ["safe", "nominal"])
    r = compute_information_value(_st(FCP, 0.004, [0.10], "s"), p, compute_risk_table(lb, p, pr.compiled,
                                                                                       pr.price_vector()), 0.05)
    assert r.error_model_identification == "unidentified_scenario"
    assert r.error_model_report["object_metadata_unverified"] == ["F:CP (prior: factory_claim_unverifiable_in_this_process)"]
    # factory metadata cannot be put on states that were not drawn from the model
    with pytest.raises(MetadataConflictError, match="not drawn from a model"):
        _cx(metadata=StateMetadata.from_model_metadata(model.metadata, model_fingerprint=model.fingerprint())
            .restricted_to(("F", "C"), ("CP",)))


def test_explicit_metadata_identify_only_as_the_definition_of_a_synthetic_world():
    # synthetic: explicit metadata are the definition of the test world (red-team (d); documented)
    pr = two_ingredient_problem()
    inm = IndependentNormalModel("fix_a_inm3", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(),
                                 np.array([[0.01], [0.0]]), pr.dm_estimates(), np.zeros(2), is_synthetic=True)
    p = PriorStates.from_drawset(inm.draw(RandomStreams(1103), "opt", 300),
                                 metadata=StateMetadata.synthetic((FCP,), "true_batch_state", label="def"))
    decs = [RationDecision(pr.ingredient_ids, np.array([(20 - 20 * (c - .10) / .30) / .40, 20 * (c - .10) / .30 / .80]),
                           pr.dm_estimates(), f"cp{c:.2f}") for c in (0.18, 0.16)]
    lb = CandidateLibrary.from_decisions(pr, decs, ["safe", "nominal"])
    r = compute_information_value(_st(FCP, 0.004, [0.10], "s"), p, compute_risk_table(lb, p, pr.compiled,
                                                                                       pr.price_vector()), 0.05)
    assert r.error_model_identification == "identified_by_declared_sources" and r.is_synthetic
    assert r.error_model_report["identification_basis"] == "synthetic_world_definition"
    model, prior, risk = _factory_world(spec_id="fix_a_true")
    rf = compute_information_value(_st(FCP, 0.004, [0.10], "s"), prior, risk, 0.05)
    assert rf.error_model_report["identification_basis"] == "factory_registry_verified"
    # non-synthetic explicit metadata (red-team B1): never an identification
    obs = ComponentMetadata(FCP, "observed_incl_lab_only", "TEST-FABRICATED-ROW-HASH", "none", None,
                            "TEST-FABRICATED-LAB", False, "sourced", "TEST-FABRICATED-NOT-A-SOURCE", "nowhere")
    ccp = ComponentMetadata(CCP, "not_applicable", None, "none", None, None, False)
    prob, lib, pn, rn = _cx(metadata=StateMetadata((obs, ccp), "explicit_prior_definition", is_synthetic=False),
                            is_synthetic=False)
    sd0, edge0 = _review_signal()
    rr = compute_information_value(_st(FCP, sd0, [edge0], "review"), pn, rn, 0.10)
    assert rr.error_model_identification == "unidentified_scenario"
    assert rr.error_model_report["object_metadata_unverified"] == ["F:CP (prior: explicit_non_factory_unverifiable)"]
    # ... and with a claimed de-convolution but no numeric record it is refused (red-team B1 as submitted)
    fab = ComponentMetadata(FCP, "true_batch_state", "TEST-FABRICATED-ROW-HASH", "DEC-FABRICATED",
                            "no such decomposition", "MM-FABRICATED", False, "sourced", "TEST-FABRICATED-NOT-A-SOURCE",
                            "nowhere")
    prob, lib, pf, rf2 = _cx(metadata=StateMetadata((fab, ccp), "explicit_prior_definition", is_synthetic=False),
                             is_synthetic=False)
    with pytest.raises(InvalidProblemError, match="must carry its numeric VarianceDecomposition record"):
        compute_information_value(_st(FCP, sd0, [edge0], "review"), pf, rf2, 0.10)


def test_non_synthetic_draws_without_resolvable_metadata_raise():
    th = np.full((20, 2, 1), 0.1)
    d = DrawSet(th, np.full((20, 2), 0.5), "opt", "root=1/opt", "some_real_model", "f" * 64, ("F", "C"), ("CP",),
                False)
    with pytest.raises(InvalidProblemError, match="could not be resolved from a factory model"):
        PriorStates.from_drawset(d)
    with pytest.raises(InvalidProblemError, match="explicit metadata of a non-synthetic world"):
        PriorStates.from_drawset(d, metadata=StateMetadata((ComponentMetadata(FCP, "true_batch_state", "x", "D", "s",
                                                                               "M", False),), "explicit_prior_definition"))


# =================================================================================================
# R2-2  a de-convolved true state is checked numerically
# =================================================================================================

def test_deconvolved_true_state_must_match_its_record_and_the_signal():
    # red-team B2: record observed 0.0125 -> true 0.010, but the states have SD 0.02; same process in the signal
    rec = decompose_observed_variance(0.0125, sampling_sd=0.006, lab_sd=0.0045, component=FCP,
                                      decomposition_id="DEC-X", decomposition_source="synthetic", measurement_model_id=HIST)
    cm = ComponentMetadata(FCP, "true_batch_state", "synthetic:x", "DEC-X", "synthetic", HIST, True,
                           "synthetic_test_only", "SYN", None, rec)
    ccp = ComponentMetadata(CCP, "not_applicable", None, "none", None, None, True, "synthetic_test_only", "SYN", None)
    prob, lib, prior, risk = _cx(metadata=StateMetadata((cm, ccp), "explicit_prior_definition", is_synthetic=True))
    sd0, edge0 = _review_signal()
    same_proc = _st(FCP, 0.006, [edge0], "hist", lab=0.0045, mm=HIST)
    with pytest.raises(InvalidProblemError, match="not generated from the recorded true-state SD"):
        compute_information_value(same_proc, prior, risk, 0.10)
    # a record that matches the states (true SD 0.02): same process with the same errors identifies ...
    rec2 = decompose_observed_variance(float(np.sqrt(0.02 ** 2 + 0.006 ** 2 + 0.0045 ** 2)), sampling_sd=0.006,
                                       lab_sd=0.0045, component=FCP, decomposition_id="DEC-X",
                                       decomposition_source="synthetic", measurement_model_id=HIST)
    cm2 = dataclasses.replace(cm, decomposition=rec2)
    prob, lib, prior2, risk2 = _cx(metadata=StateMetadata((cm2, ccp), "explicit_prior_definition", is_synthetic=True))
    ok = compute_information_value(same_proc, prior2, risk2, 0.10)
    assert ok.error_model_identification == "identified_by_declared_sources"
    assert ok.error_model_report["checks"]["true_state_checks"]["F:CP"]["measurement_link"] == "same_process"
    # ... the same process with other error magnitudes is refused; another process is recorded
    with pytest.raises(InvalidProblemError, match="same measurement process"):
        compute_information_value(_st(FCP, 0.008, [edge0], "hist2", lab=0.0045, mm=HIST), prior2, risk2, 0.10)
    other = compute_information_value(_st(FCP, 0.008, [edge0], "lab2", lab=0.0045, mm="synthetic:OTHER"), prior2,
                                      risk2, 0.10)
    assert other.error_model_report["checks"]["true_state_checks"]["F:CP"]["measurement_link"] == "different_process"


def test_claimed_decomposition_without_a_record_does_not_identify_or_is_refused():
    # red-team B3: synthetic factory cell labelled de-convolved, SD = the observed SD, no numeric record
    model, prior, risk = _factory_world(dec="DEC-CLAIMED", dec_src="claimed, never applied", mm=HIST, sd=0.0125,
                                        spec_id="fix_a_b3")
    r = compute_information_value(_st(FCP, 0.006, [0.10], "h", lab=0.0045, mm=HIST), prior, risk, 0.05)
    assert r.error_model_identification == "unidentified_scenario"
    assert r.error_model_report["true_state_unverified"] == ["F:CP"]
    # red-team B3b (reverse regression): non-synthetic, SD unchanged, only the label says de-convolved -> refused
    model_n, prior_n, risk_n = _factory_world(dec="DEC-CLAIMED", dec_src="claimed", mm=HIST, sd=0.0125,
                                              synthetic=False, spec_id="fix_a_b3b")
    with pytest.raises(InvalidProblemError, match="must carry its numeric VarianceDecomposition record"):
        compute_information_value(_st(FCP, 0.006, [0.10], "h", lab=0.0045, mm=HIST), prior_n, risk_n, 0.05)


def test_drawn_prior_record_is_checked_with_a_declared_monte_carlo_tolerance():
    pr = two_ingredient_problem()
    inm = IndependentNormalModel("fix_a_inm4", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(),
                                 np.array([[0.012], [0.0]]), pr.dm_estimates(), np.zeros(2), is_synthetic=True)
    d = inm.draw(RandomStreams(7), "opt", 2000)
    decs = [RationDecision(pr.ingredient_ids, np.array([(20 - 20 * (c - .10) / .30) / .40, 20 * (c - .10) / .30 / .80]),
                           pr.dm_estimates(), f"cp{c:.2f}") for c in (0.18, 0.16)]
    lb = CandidateLibrary.from_decisions(pr, decs, ["safe", "nominal"])
    for true_sd, ok in ((0.012, True), (0.006, False)):
        rec = decompose_observed_variance(float(np.sqrt(true_sd ** 2 + 0.004 ** 2)), sampling_sd=0.004,
                                          component=FCP, decomposition_id="D", decomposition_source="synthetic",
                                          measurement_model_id=HIST)
        cm = ComponentMetadata(FCP, "true_batch_state", "synthetic:d", "D", "synthetic", HIST, True,
                               "synthetic_test_only", "SYN", None, rec)
        p = PriorStates.from_drawset(d, metadata=StateMetadata((cm,), "explicit_prior_definition", is_synthetic=True))
        rk = compute_risk_table(lb, p, pr.compiled, pr.price_vector())
        if ok:
            r = compute_information_value(_st(FCP, 0.004, [0.10], "s"), p, rk, 0.05)
            chk = r.error_model_report["checks"]["true_state_checks"]["F:CP"]
            assert chk["status"] == "verified" and chk["tolerance_kind"].startswith("monte_carlo_6")
        else:
            with pytest.raises(InvalidProblemError, match="not generated from the recorded true-state SD"):
                compute_information_value(_st(FCP, 0.004, [0.10], "s"), p, rk, 0.05)


# =================================================================================================
# R2-3  measurement process must be declared; unlinked negative decomposition rejects a member
# =================================================================================================

def test_signal_must_name_its_measurement_process_when_the_prior_names_one():
    prob, lib, prior, risk = _cx(metadata=_md_counterexample("unidentified", measurement_model_id=HIST))
    big = dict(lab=0.03, sid="big")                      # single-result error variance 0.05^2 + 0.03^2 > prior 4e-4
    with pytest.raises(InvalidProblemError, match="must declare measurement_model_id"):
        compute_information_value(_st(FCP, 0.05, [0.10], **big), prior, risk, 0.10)            # red-team B7
    r = compute_information_value(_st(FCP, 0.05, [0.10], mm="renamed-lab", **big), prior, risk, 0.10)   # B7b
    assert r.error_model_identification == "unidentified_scenario"
    assert r.error_model_report["checks"]["decomposition"]["F:CP"]["deconvolved_member"] == \
        "rejected_inconsistent_negative"
    assert r.error_model_report["scenario_member"].startswith("all_true_variation_end_only")
    with pytest.raises(InvalidProblemError, match="inconsistent variance decomposition"):
        compute_information_value(_st(FCP, 0.05, [0.10], mm=HIST, **big), prior, risk, 0.10)


# =================================================================================================
# R2-4  sourced error values are traced to the locator table
# =================================================================================================

LOCATORS = REPO / "sources" / "error_source_locators.csv"


def _pick(loc, entries, components, nutrient):
    return next(pid for pid, r in loc.items()
                if r["error_components"] == components and r["nutrient"] == nutrient and r["scale"] == "additive"
                and r["unit"] == "percentage_points" and r["value_status"] == "value_present"
                and r["verification"] != "blocked" and entries.get(pid, {}).get("value") is not None)


def test_sourced_error_values_are_traced_before_any_value_is_reported():
    loc = load_error_locators(LOCATORS)
    assert default_error_locators() is not None
    entries = {e["param_id"]: e for e in
               yaml.safe_load((REPO / "configs" / "assays.yaml").read_text(encoding="utf-8"))["error_parameters"]["entries"]}
    FNDF = ObservedComponent("F", "NDF")
    prob = problem([ing("F", 0.40, {"NDF": 0.45}, forage=1.0), ing("C", 0.80, {"NDF": 0.20})], ["NDF"],
                   [dm_offer(20.0), conc("fndf_min", {"G:forage:NDF": 1.0}, "ge", 18.0)], {"F": 0.04, "C": 0.32},
                   problem_id="fix_a_trace")
    ids, dh = prob.ingredient_ids, prob.dm_estimates()
    lib = CandidateLibrary.from_decisions(prob, [RationDecision(ids, np.array([40.0, 5.0]), dh, "a"),
                                                 RationDecision(ids, np.array([45.0, 2.5]), dh, "b")], ["a", "b"])
    md = StateMetadata.synthetic((FNDF,), "true_batch_state", label="trace")
    prior = PriorStates.from_discrete(np.array([[[0.40], [0.20]], [[0.50], [0.20]]]), np.array([[0.4, 0.8]] * 2),
                                      [0.5, 0.5], ids, ["NDF"], label="fix_a_trace", is_synthetic=True, metadata=md)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())

    def em(fields, *, semantics="repeatability_only", mm=None, r=1):
        kw = {"sampling_sd": 0.0, "lab_repeatability_sd": 0.0}
        prov = {}
        for fname, pid in fields.items():
            kw[fname] = float(entries[pid]["value"]) / 100.0                # values from configs/assays.yaml
            prov[fname] = Provenance(ValueStatus.SOURCED, source_id=loc[pid]["registry_source_id"].split("|")[0],
                                     locator=f"sources/error_source_locators.csv#{pid}")
        e = ComponentErrorModel(FNDF, lab_error_semantics=semantics, provenance=prov, is_synthetic=False,
                                measurement_model_id=mm, **kw)
        sm = SignalModel("trace", (e,), SamplingProtocol(1, True, r), is_synthetic=False)
        return InformationStructure("trace", "sample", binning=SignalBinning((FNDF,), ((0.45,),), "e"), signal_model=sm)

    batch = _pick(loc, entries, "batch_true", "NDF")
    total = _pick(loc, entries, "lab_random|lab_bias", "NDF")
    lab = _pick(loc, entries, "lab_random", "NDF")
    samp = _pick(loc, entries, "sampling", "NDF")
    # red-team B8: true batch variation used as a sampling error, and a total single-result SD shrunk by 16 replicates
    with pytest.raises(InvalidProblemError, match="does not trace.*true batch variation"):
        compute_information_value(em({"sampling_sd": batch}), prior, risk, 0.10, error_locators=loc)
    with pytest.raises(InvalidProblemError, match="do not match field lab_repeatability_sd"):
        compute_information_value(em({"lab_repeatability_sd": total}, r=16), prior, risk, 0.10, error_locators=loc)
    rep = double_count_guard(em({"sampling_sd": batch}).signal_model, prior=prior, error_locators=loc)
    assert rep.status == "violation" and rep.checks["error_provenance"]["status"] == "issues"
    # correctly traced values identify; the report carries the trace rows
    good = em({"sampling_sd": samp, "lab_repeatability_sd": lab}, mm=lab)
    ok = compute_information_value(good, prior, risk, 0.10, error_locators=loc)
    assert ok.error_model_identification == "identified_by_declared_sources"
    trace = ok.error_model_report["checks"]["error_provenance"]
    assert trace["status"] == "traced" and {r["param_id"] for r in trace["rows"]} == {samp, lab}
    ok_default = compute_information_value(good, prior, risk, 0.10)                     # default table
    assert ok_default.error_model_report["checks"]["error_provenance"]["locator_table"]["path"] == \
        "sources/error_source_locators.csv"
    # an arbitrary measurement id cannot (un)link processes for real error values
    with pytest.raises(InvalidProblemError, match="not the locator param_id"):
        compute_information_value(em({"lab_repeatability_sd": lab}, mm="renamed-lab"), prior, risk, 0.10,
                                  error_locators=loc)


def test_without_a_locator_table_sourced_errors_are_unsourced(monkeypatch):
    import ration_reliability.information.signal as S

    monkeypatch.setattr(S, "_default_locator_path", lambda: REPO / "sources" / "__missing_locator_table__.csv")
    prob, lib, prior, risk = _cx()
    prov = Provenance(ValueStatus.SOURCED, source_id="SRC-V-SPW2015", locator="sources/error_source_locators.csv#X")
    sd0, edge0 = _review_signal()
    st = _st(FCP, sd0, [edge0], "s", prov=prov, synthetic=False)
    r = compute_information_value(st, prior, risk, 0.10)
    assert r.error_model_identification == "error_model_unsourced"
    assert r.error_model_report["checks"]["error_provenance"]["status"] == "locator_table_unavailable"


# =================================================================================================
# R2-5  perfect information carries an identification label
# =================================================================================================

def test_perfect_information_is_labelled_by_what_the_prior_sd_represents():
    model, prior, risk = _factory_world(basis="observed_incl_sampling_and_lab", mm=HIST, sd=0.0125,
                                        spec_id="fix_a_b9")
    ppi = compute_information_value(InformationStructure("ppi", "perfect_partial", (FCP,),
                                                         SignalBinning((FCP,), ((0.10,),), "e")), prior, risk, 0.05)
    assert ppi.error_model_identification == "observed_state_scenario"                        # red-team B9
    assert any("observed_state_scenario" in w for w in ppi.warnings)
    pf = compute_information_value(InformationStructure("pf", "perfect_full"), prior, risk, 0.05)
    assert pf.error_model_identification == "observed_state_scenario"
    _, prior_t, risk_t = _factory_world(spec_id="fix_a_b9_true")
    ppt = compute_information_value(InformationStructure("ppi", "perfect_partial", (FCP,),
                                                         SignalBinning((FCP,), ((0.10,),), "e")), prior_t, risk_t, 0.05)
    assert ppt.error_model_identification == "perfect_information_on_true_state"
    prob, lib, pn, rn = _cx(metadata=None)
    pp_none = compute_information_value(InformationStructure("pp", "perfect_partial", (FCP,)), pn, rn, 0.10)
    assert pp_none.error_model_identification == "unidentified_scenario"
    un = compute_information_value(InformationStructure("u", "uninformative", bin_probabilities=(0.5, 0.5)), pn, rn,
                                   0.10)
    assert un.error_model_identification == "not_applicable"
    cov, inv = _cov_inv(lib)
    with pytest.raises(InvalidProblemError, match="scenario=True"):
        per_head_day_value(ppi.with_value_definition(EIS), EIS)


# =================================================================================================
# R2-6  the EnergyColumnModel route keeps the factory metadata
# =================================================================================================

def _energy_world(basis="observed_incl_sampling_and_lab"):
    feeds = (E.FeedEnergyInputs("syn_forage", ndf=40.0, lignin=4.0, starch=20.0, fa=3.0, cp=10.0, ash=6.0,
                                rup_pct_cp=30.0, drup_pct_rup=70.0, dstarch_base=0.88, dfa=0.73),
             E.FeedEnergyInputs("syn_protein", ndf=12.0, lignin=1.0, starch=2.0, fa=1.5, cp=50.0, ash=7.0,
                                rup_pct_cp=35.0, drup_pct_rup=90.0, dstarch_base=0.95, dfa=0.73))
    st = E.FixedDMISettings(dmi_kg_d=20.0, body_weight_kg=600.0, starch_ref_pct=24.0, milk_cp_kg_d=1.0,
                            body_gain_cp_kg_d=0.0)
    nut = ("NDF", "starch", "CP", "ash")
    lin = E.linearise_nel_fixed_dmi(feeds, st, nutrient_map={"NDF": "ndf", "starch": "starch", "CP": "cp",
                                                             "ash": "ash"})
    ids = tuple(f.ingredient_id for f in feeds)
    fld = {"NDF": "ndf", "starch": "starch", "CP": "cp", "ash": "ash"}
    mean = np.array([[getattr(f, fld[n]) / 100.0 for n in nut] for f in feeds])
    sd = np.array([[0.01, 0.01, 0.01, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rule = {"primary_family": "TN_MM", "fallback_families": [], "on_exhausted": "error",
            "status": "synthetic_test_only", "rationale": "unit test (FIX_A)", "selection_basis": "declared_rule"}
    defaults = {"variance_basis": basis, "data_fingerprint": "synthetic:fix_a_energy", "decomposition_id": "none",
                "decomposition_source": None, "measurement_model_id": HIST, "provenance_status": "synthetic_test_only",
                "source_id": None, "locator": None}
    spec = UncertaintySpec.from_arrays("fix_a_energy", list(ids), list(nut), mean, sd, np.array([0.35, 0.88]),
                                       np.array([0.01, 0.0]), purpose="unit_test",
                                       moment_semantics="target_marginal_moments", is_synthetic=True, family_rule=rule,
                                       cell_defaults=defaults, theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0))
    fm = build_uncertainty_model(spec)
    return fm, E.EnergyColumnModel(fm, lin), ids


def test_energy_wrapped_world_keeps_the_factory_metadata():
    fm, w, ids = _energy_world()
    draws = w.draw(RandomStreams(1103), "opt", 200)
    comp = ObservedComponent(ids[0], "starch")
    p = PriorStates.from_drawset(draws, model=w)                       # the run_dev_case_v1 route, with model=
    md = p.metadata
    assert md.origin == "factory_model_metadata" and md.model_fingerprint == w.fingerprint()
    assert md.derived_from_model_fingerprint == fm.fingerprint()
    assert verify_state_metadata(md)[0] == "factory_registry_verified"
    assert p.component_metadata(comp).variance_basis == "observed_incl_sampling_and_lab"
    assert p.component_metadata(ObservedComponent(ids[0], E.ENERGY_COLUMN_ID)) is None     # derived column
    sm = _sig(comp, 0.006, "h", lab=0.0045, mm=HIST)
    with pytest.raises(MetadataConflictError):                                # red-team exp_r2c
        double_count_guard(sm, {comp: "true_batch_state"}, prior=p)
    assert double_count_guard(sm, prior=p).status == "violation"
    link = resolve_truth_link(w, p)                                    # the wrapper object: same metadata
    assert link.linked and link.verified and link.origin.endswith(("same_as_prior_metadata", "linked_by_model_object"))
    other_fm, other_w, _ = _energy_world(basis="true_batch_state")
    with pytest.raises(InvalidProblemError, match="not the model that generated"):
        resolve_truth_link(other_w, p)
    # synthetic draws without model= are not resolved (metadata None); non-synthetic ones raise (see below)
    assert PriorStates.from_drawset(draws).metadata is None


_E0 = REPO / "experiments" / "E0_verification"
_DEV = REPO / "data" / "restricted_local" / "dev_case_v1"


@pytest.mark.skipif(not (_DEV / "uncertainty_cells.csv").exists() or not (_E0 / "run_dev_case_v1.py").exists(),
                    reason="restricted dev_case_v1 inputs not present (data/restricted_local/ is not distributed)")
def test_real_dev_case_route_resolves_the_factory_metadata(monkeypatch):
    """The actual route of experiments/E0_verification/run_dev_case_v1.py (EnergyColumnModel of the NASEM factory
    world).  Asserts on origins, fingerprints, counts and labels only -- never on restricted values."""
    monkeypatch.chdir(REPO)
    if str(_E0) not in sys.path:
        sys.path.insert(0, str(_E0))
    import run_dev_case_v1 as D   # noqa: E402
    from ration_reliability.hashing import file_sha256
    from ration_reliability.io import load_problem

    problem_, _ = load_problem(D.PROBLEM_YAML, mode="pilot")
    lin = D.load_linearisation(problem_)
    cells = D.read_cells()
    fm = build_uncertainty_model(D.build_h0_spec(problem_.ingredient_ids, cells, file_sha256(D.CELLS_CSV)),
                                 model_id="DEV_CASE_V1_H0_factory_TN_MM")
    w = E.EnergyColumnModel(fm, lin)
    pd_ = w.draw(RandomStreams(1103), "opt", 64)
    assert not pd_.is_synthetic
    with pytest.raises(InvalidProblemError, match="pass model="):
        PriorStates.from_drawset(pd_)                              # line 1080 as written: no silent metadata=None
    p = PriorStates.from_drawset(pd_, model=w)
    md = p.metadata
    assert md.origin == "factory_model_metadata" and md.derived_from_model_fingerprint == fm.fingerprint()
    assert verify_state_metadata(md)[0] == "factory_registry_verified"
    stoch = [c for c in md.components if c.is_stochastic]
    assert stoch and all(c.variance_basis == "observed_incl_sampling_and_lab" for c in stoch)
    comp = ObservedComponent(D.CS, "starch")
    assert p.component_metadata(comp).is_stochastic
    edges = quantile_edges(p.component_values((comp,)), 4, p.weights)
    lik_st = InformationStructure("PPI", "perfect_partial", (comp,), SignalBinning((comp,), (edges,), "q"))
    from ration_reliability.information.value import _perfect_information_identification
    label, rep = _perfect_information_identification(lik_st, p, None)
    assert label == "observed_state_scenario" and rep["prior_metadata_verification"] == "factory_registry_verified"
    sm = _sig(comp, 0.004, "real", lab=0.003, mm="synthetic:X")
    with pytest.raises(MetadataConflictError):
        double_count_guard(sm, {comp: "true_batch_state"}, prior=p)
    assert double_count_guard(sm, prior=p).status == "violation"
