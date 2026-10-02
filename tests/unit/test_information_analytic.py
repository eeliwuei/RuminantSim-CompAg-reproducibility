"""Hand-calculated information values on a solvable toy problem (synthetic numbers only).

Problem (all values synthetic, currency XXX):

* forage F: d = 0.40, CP uncertain in {8 %, 12 %} DM with prior 1/2 each, price 0.04 /kg as-fed;
* concentrate C: d = 0.80, CP 40 % DM, price 0.32 /kg as-fed;
* planned DM offered = 20 kg/d (structural); CP >= 16 % DM (probabilistic, the joint event I).

Candidates (DM share s of C):  A: s = 0.15 -> q = (42.5, 3.75), cost 2.90;
                               B: s = 0.26 -> q = (37.0, 6.50), cost 3.56.
A violates CP >= 16 % iff CP_F = 8 % (12.8 %); B never violates (16.32 % at CP_F = 8 %).

Hand results with alpha = 0.05:
  V0 = 3.56 (B), perfect info: VT = 0.5*2.90 + 0.5*3.56 = 3.23, EVPI = EVPPI = 0.33.
  randomised references: V0_rand = 3.56 - 0.1*0.66 = 3.494, VT_rand(perfect) = 1.45 + 0.5*3.494 = 3.197,
  EVPI_rand = 0.297.
  Sample signal Z = CP_F + e, e ~ N(0, sd^2), bins split at 10 %: p = Phi(0.02/sd).
  The deterministic policy "high -> A, low -> B" has cost 3.23 and ex-ante risk 0.5 (1-p); it is
  feasible iff p >= 0.9, so EVSI_det = 0.33 if sd <= 0.02/Phi^-1(0.9) else 0.
  VT_rand(sd) = 3.56 - 0.33 (1 + (p-0.9)/p) if p >= 0.9 else 3.56 - 0.033/(1-p).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from engine_test_helpers import conc, dm_offer, ing, problem
from ration_reliability.datamodel import Provenance, RationDecision, ValueStatus
from ration_reliability.information import (
    CandidateLibrary,
    ComponentErrorModel,
    InformationStructure,
    ObservedComponent,
    PolicyProblem,
    PriorStates,
    SignalBinning,
    SignalModel,
    StateMetadata,
    build_likelihood,
    check_theoretical_order,
    compute_information_value,
    compute_risk_table,
    double_count_guard,
    per_outcome_conditional_risk_rule,
    solve_signal_policy,
)

SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-P8-TEST")
FCP = ObservedComponent("F", "CP")
ALPHA = 0.05
SD_STAR = 0.02 / stats.norm.ppf(0.9)


def _setup(alpha_unused=None, states=((0.08, 0.5), (0.12, 0.5))):
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    prob = problem([F, C], ["CP"], [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0)],
                   {"F": 0.04, "C": 0.32}, problem_id="p8_analytic")
    ids, dh = prob.ingredient_ids, prob.dm_estimates()
    kA = RationDecision(ids, np.array([17.0 / 0.4, 3.0 / 0.8]), dh, "synthetic_candidate_A")
    kB = RationDecision(ids, np.array([14.8 / 0.4, 5.2 / 0.8]), dh, "synthetic_candidate_B")
    lib = CandidateLibrary.from_decisions(prob, [kA, kB], ["A", "B"])
    theta = np.array([[[v], [0.40]] for v, _ in states])
    d = np.tile([0.4, 0.8], (len(states), 1))
    prior = PriorStates.from_discrete(theta, d, [w for _, w in states], ids, ["CP"], label="p8_analytic",
                                      is_synthetic=True,
                                      metadata=StateMetadata.synthetic((FCP,), "true_batch_state", label="p8_analytic"))
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    return prob, lib, prior, risk


def _sample_structure(sd, sid=None):
    em = ComponentErrorModel(FCP, sd, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True)
    sm = SignalModel(sid or f"noise_sd={sd:g}", (em,), is_synthetic=True)
    b = SignalBinning((FCP,), ((0.10,),), "synthetic edge at 10 %")
    return InformationStructure(sid or f"sample_sd={sd:g}", "sample", binning=b, signal_model=sm), sm


def _sample_value(prior, risk, sd, alpha=ALPHA, **kw):
    st, sm = _sample_structure(sd)
    return compute_information_value(st, prior, risk, alpha,
                                     double_count_report=double_count_guard(sm, {FCP: "true_batch_state"}), **kw)


def test_setup_costs_and_risk_table_match_hand_values():
    prob, lib, prior, risk = _setup()
    np.testing.assert_allclose(lib.costs, [2.90, 3.56], atol=1e-12)
    # state 0 = CP 8 %: A violates, B satisfies; state 1 = CP 12 %: both satisfy
    assert risk.violated.tolist() == [[True, False], [False, False]]
    assert not risk.unknown.any()


@pytest.mark.parametrize("st", [InformationStructure("full", "perfect_full"),
                                InformationStructure("partial_exact", "perfect_partial", (FCP,)),
                                InformationStructure("partial_binned", "perfect_partial", (FCP,),
                                                     SignalBinning((FCP,), ((0.10,),), "syn"))])
def test_perfect_information_hand_values(st):
    _, _, prior, risk = _setup()
    r = compute_information_value(st, prior, risk, ALPHA)
    assert r.V0.expected_cost == pytest.approx(3.56, abs=1e-12)
    assert r.V0.ex_ante_risk == pytest.approx(0.0, abs=1e-15)
    assert r.VT.expected_cost == pytest.approx(3.23, abs=1e-12)
    assert r.gross_value == pytest.approx(0.33, abs=1e-12)
    assert r.V0_randomized.expected_cost == pytest.approx(3.494, abs=1e-9)
    assert r.VT_randomized.expected_cost == pytest.approx(3.197, abs=1e-9)
    assert r.gross_value_randomized_reference == pytest.approx(0.297, abs=1e-9)
    assert r.gross_value_status == "ok"
    assert r.containment_check["holds"] is True
    assert r.randomization_benchmark_value == pytest.approx(0.0, abs=1e-12)
    assert r.is_synthetic
    assert "not a continuous global" in r.scope_label


def test_deterministic_policy_milp_equals_exhaustive_enumeration():
    _, _, prior, risk = _setup()
    for sd in (0.005, SD_STAR * 0.99, SD_STAR * 1.01, 0.05):
        st, _ = _sample_structure(sd)
        pp = PolicyProblem.build(prior, build_likelihood(st, prior), risk, ALPHA)
        a = solve_signal_policy(pp, method="milp")
        b = solve_signal_policy(pp, method="enumeration")
        assert a.status == b.status == "optimal"
        assert a.expected_cost == pytest.approx(b.expected_cost, abs=1e-12)
        assert a.mip_gap is not None and a.mip_gap <= 1e-9


def test_sample_information_threshold_hand_value():
    _, _, prior, risk = _setup()
    below = _sample_value(prior, risk, SD_STAR * 0.99)
    above = _sample_value(prior, risk, SD_STAR * 1.01)
    assert below.gross_value == pytest.approx(0.33, abs=1e-12)
    p = stats.norm.cdf(0.02 / (SD_STAR * 0.99))
    assert below.VT.ex_ante_risk == pytest.approx(0.5 * (1 - p), abs=1e-12)
    assert above.gross_value == pytest.approx(0.0, abs=1e-12)
    assert below.error_model_identification == "identified_by_declared_sources"


@pytest.mark.parametrize("sd", [0.005, 0.01, 0.014, 0.02, 0.05, 0.2])
def test_randomized_reference_closed_form(sd):
    _, _, prior, risk = _setup()
    r = _sample_value(prior, risk, sd)
    p = stats.norm.cdf(0.02 / sd)
    vt = 3.56 - 0.33 * (1 + (p - 0.9) / p) if p >= 0.9 else 3.56 - 0.033 / (1 - p)
    assert r.VT_randomized.expected_cost == pytest.approx(vt, abs=1e-8)
    assert r.gross_value_randomized_reference == pytest.approx(3.494 - vt, abs=1e-8)


def test_noise_to_zero_sample_value_tends_to_partial_perfect_value():
    _, _, prior, risk = _setup()
    evppi = compute_information_value(InformationStructure("pp", "perfect_partial", (FCP,)), prior, risk, ALPHA)
    for sd in (1e-4, 1e-8, 1e-12):
        r = _sample_value(prior, risk, sd)
        assert r.gross_value == pytest.approx(evppi.gross_value, abs=1e-10)
    r = _sample_value(prior, risk, 1e-12)
    assert r.gross_value_randomized_reference == pytest.approx(evppi.gross_value_randomized_reference, abs=1e-8)
    # exactly zero noise is the perfect (binned) observation
    r0 = _sample_value(prior, risk, 0.0)
    assert r0.gross_value == pytest.approx(0.33, abs=1e-12)


def test_noise_to_infinity_sample_value_tends_to_zero():
    _, _, prior, risk = _setup()
    sds = [0.02, 0.05, 0.1, 1.0, 1e3, 1e6, 1e9]
    vals = [_sample_value(prior, risk, sd) for sd in sds]
    rnd = [v.gross_value_randomized_reference for v in vals]
    assert all(a >= b - 1e-10 for a, b in zip(rnd, rnd[1:]))          # non-increasing in the noise SD
    assert abs(rnd[-1]) <= 1e-9
    # deterministic: equals the uninformative-signal (randomisation) benchmark in the limit, here 0
    assert vals[-1].gross_value == pytest.approx(0.0, abs=1e-12)
    assert vals[-1].randomization_benchmark_value == pytest.approx(0.0, abs=1e-12)


def test_noise_limit_deterministic_value_is_randomization_benchmark_not_information():
    """alpha = 0.30: a pure-noise signal lets the deterministic policy mix A and B 50/50.

    V0 = 3.56 (A has risk 0.5 > 0.3); with a 50/50 randomisation device the policy (A | B) has risk
    0.25 <= 0.3 and cost 3.23, so the deterministic 'value' of an uninformative signal is 0.33,
    while the randomised reference (information net of randomisation) tends to 0.
    """
    _, _, prior, risk = _setup()
    r = _sample_value(prior, risk, 1e9, alpha=0.30)
    assert r.gross_value == pytest.approx(0.33, abs=1e-9)
    assert r.randomization_benchmark_value == pytest.approx(0.33, abs=1e-9)
    assert abs(r.gross_value_randomized_reference) <= 1e-8
    assert any("randomisation value" in w for w in r.warnings)


def test_irrelevant_perfect_information_deterministic_positive_randomized_zero():
    """Perfect knowledge of a component that no constraint uses (F NDF) has randomisation value only."""
    F = ing("F", 0.40, {"CP": 0.10, "NDF": 0.5}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40, "NDF": 0.2})
    prob = problem([F, C], ["CP", "NDF"], [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0)],
                   {"F": 0.04, "C": 0.32}, problem_id="p8_irrelevant")
    ids, dh = prob.ingredient_ids, prob.dm_estimates()
    lib = CandidateLibrary.from_decisions(prob, [RationDecision(ids, np.array([42.5, 3.75]), dh, "A"),
                                                 RationDecision(ids, np.array([37.0, 6.5]), dh, "B")], ["A", "B"])
    theta, w = [], []
    for cp in (0.08, 0.12):
        for ndf in (0.40, 0.60):
            theta.append([[cp, ndf], [0.40, 0.2]])
            w.append(0.25)
    prior = PriorStates.from_discrete(np.array(theta), np.tile([0.4, 0.8], (4, 1)), w, ids, ["CP", "NDF"],
                                      label="irrelevant", is_synthetic=True)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    r = compute_information_value(InformationStructure("ndf", "perfect_partial", (ObservedComponent("F", "NDF"),)),
                                  prior, risk, 0.30)
    assert r.gross_value == pytest.approx(0.33, abs=1e-12)          # deterministic: randomisation only
    assert abs(r.gross_value_randomized_reference) <= 1e-9            # information net of randomisation: 0
    rel = compute_information_value(InformationStructure("cp", "perfect_partial", (ObservedComponent("F", "CP"),)),
                                    prior, risk, 0.30)
    assert rel.gross_value_randomized_reference > 1e-3                # relevant information has value


def test_per_outcome_conditional_rule_is_separate_and_named():
    _, _, prior, risk = _setup()
    # sd with 0.90 <= p < 0.95: ex-ante policy uses A in the high bin, the per-outcome rule cannot
    sd = 0.02 / stats.norm.ppf(0.925)
    st, _ = _sample_structure(sd)
    pp = PolicyProblem.build(prior, build_likelihood(st, prior), risk, ALPHA)
    ex_ante = solve_signal_policy(pp)
    rule = per_outcome_conditional_risk_rule(pp)
    assert ex_ante.expected_cost == pytest.approx(3.23, abs=1e-12)
    assert rule.rule == "per_outcome_conditional_risk_rule_T7_2"
    assert rule.is_information_value is False
    assert "not an EVSI" in rule.notes
    assert rule.expected_cost == pytest.approx(3.56, abs=1e-12)
    assert rule.cost_difference_vs_no_information_rule == pytest.approx(0.0, abs=1e-12)
    # with p >= 0.95 the rule also uses A in the high bin; its ex-ante risk is 0.5 (1 - p)
    sd2 = 0.01
    st2, _ = _sample_structure(sd2)
    rule2 = per_outcome_conditional_risk_rule(PolicyProblem.build(prior, build_likelihood(st2, prior), risk, ALPHA))
    p = stats.norm.cdf(2.0)
    assert rule2.expected_cost == pytest.approx(3.23, abs=1e-12)
    assert rule2.ex_ante_risk_of_rule == pytest.approx(0.5 * (1 - p), abs=1e-12)
    assert np.nanmax(rule2.conditional_risk_of_choice) <= ALPHA


def test_order_checks_on_analytic_case_hold_without_modification():
    _, _, prior, risk = _setup()
    res = {
        "full": compute_information_value(InformationStructure("full", "perfect_full"), prior, risk, ALPHA),
        "partial": compute_information_value(InformationStructure("partial", "perfect_partial", (FCP,)), prior, risk,
                                             ALPHA),
    }
    for sd in (0.005, 0.02, 1.0):
        r = _sample_value(prior, risk, sd)
        res[r.structure_id] = r
    before = {k: (v.gross_value, v.gross_value_randomized_reference) for k, v in res.items()}
    checks = check_theoretical_order(res)
    assert checks and all(c.holds for c in checks if c.holds is not None)
    theorem_rand = [c for c in checks if c.relation == "order" and c.policy_class == "randomized"
                    and c.lesser.startswith("sample") and c.greater == "partial"]
    assert theorem_rand and all(c.guarantee == "theorem" for c in theorem_rand)   # exact partial -> garbling
    det = [c for c in checks if c.relation == "order" and c.policy_class == "deterministic"
           and c.lesser.startswith("sample")]
    assert det and all(c.guarantee == "not_guaranteed_randomization_channel" for c in det)
    assert before == {k: (v.gross_value, v.gross_value_randomized_reference) for k, v in res.items()}


def test_order_violation_is_reported_not_truncated():
    """One prior state, B safe, A always violates; a noise signal with bins (0.25, 0.75) lets the
    deterministic policy use A in the 0.25 bin (risk 0.25 <= 0.30): 'EVSI_det' > 0 = EVPI_det."""
    _, _, prior1, risk1 = _setup(states=((0.08, 1.0),))
    full = compute_information_value(InformationStructure("full", "perfect_full"), prior1, risk1, 0.30)
    em = ComponentErrorModel(FCP, 0.01, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True)
    sm = SignalModel("noise", (em,), is_synthetic=True)
    edge = 0.08 + 0.01 * stats.norm.ppf(0.25)
    st = InformationStructure("sample_noise", "sample", binning=SignalBinning((FCP,), ((edge,),), "syn"),
                              signal_model=sm)
    samp = compute_information_value(st, prior1, risk1, 0.30,
                                     double_count_report=double_count_guard(sm, {FCP: "true_batch_state"}))
    assert full.gross_value == pytest.approx(0.0, abs=1e-12)
    assert samp.gross_value == pytest.approx(0.25 * 0.66, abs=1e-9)
    checks = check_theoretical_order({"full": full, "sample_noise": samp})
    viol = [c for c in checks if c.holds is False]
    assert len(viol) == 1
    v = viol[0]
    assert (v.lesser, v.greater, v.policy_class) == ("sample_noise", "full", "deterministic")
    assert v.guarantee == "not_guaranteed_randomization_channel"
    assert v.violation_size == pytest.approx(0.165, abs=1e-9)
    assert samp.gross_value == pytest.approx(0.165, abs=1e-9)      # value itself left unchanged
    rnd = [c for c in checks if c.policy_class == "randomized" and c.relation == "order"]
    assert rnd and all(c.holds for c in rnd)


def test_order_check_refuses_theoretical_order_across_different_models():
    _, _, prior, risk = _setup()
    a = compute_information_value(InformationStructure("partial", "perfect_partial", (FCP,)), prior, risk, ALPHA)
    b = compute_information_value(InformationStructure("full", "perfect_full"), prior, risk, 0.10)   # other alpha
    rel = [c for c in check_theoretical_order({"partial": a, "full": b}) if c.relation == "order"]
    assert rel and all(c.guarantee == "not_guaranteed_not_a_garbling" for c in rel)
    assert all("no theoretical order" in c.note for c in rel)


def test_information_structure_validation():
    with pytest.raises(Exception):
        InformationStructure("x", "evsi")                                    # unknown type
    with pytest.raises(Exception):
        InformationStructure("x", "perfect_partial")                          # needs components
    with pytest.raises(Exception):
        InformationStructure("x", "sample", (FCP,))                           # needs signal model and bins
    st, _ = _sample_structure(0.01)
    with pytest.raises(Exception):
        InformationStructure("x", "sample", (ObservedComponent("C", "CP"),), st.binning, st.signal_model)
