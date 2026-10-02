"""Second-review counterexample for the information-value definitions (R1).  ALL NUMBERS ARE SYNTHETIC.

Source of the case: ``docs/review_20260924_round2/RuminantSim_Second_Review/independent_probes.py``
(section "negative_default_primary_value"); the hand calculation below is repeated independently.
It is a mathematical test case, not a feed, farm or biological result.

Setup
-----
* forage F: DM fraction 0.40, CP either 8 % DM ("bad") or 12 % DM ("good"), prior 1/2 each;
  concentrate C: DM 0.80, CP 40 % DM; planned DM offered = 20 kg/d (structural); CP >= 16 % DM is the
  only probabilistic constraint (joint violation event I).
* candidates: A = q (42.5, 3.75) kg as-fed, cost 2.90, violates only in the bad state;
              B = q (37.0, 6.50) kg as-fed, cost 3.56, never violates.
* alpha = 0.10 (ex-ante joint risk, risk_tol 1e-8).
* sample signal Z = CP_F + e, e ~ N(0, sd^2) with one bin edge; sd and the edge are chosen so that
  P(low | bad) = 0.30 and P(low | good) = 0.10, hence P(low) = 0.20 and P(high) = 0.80.

Hand calculation
----------------
Deterministic policies (action in the low bin, action in the high bin)::

    (B, B): cost 3.560 = 3.56                      risk 0                       feasible
    (A, B): cost 3.428 = 0.2*2.90 + 0.8*3.56       risk 0.15 = 0.5*0.30         infeasible
    (B, A): cost 3.032 = 0.2*3.56 + 0.8*2.90       risk 0.35 = 0.5*0.70         infeasible
    (A, A): cost 2.900                             risk 0.50                    infeasible

=> V0 = VT = 3.56; operational_deterministic_cost_difference = 0.
A state-independent signal with the same bin probabilities 0.2 / 0.8 allows (A, B) with risk
0.2 * 0.5 = 0.10 <= alpha: VT(U) = 3.428.
=> contrast_vs_matched_uninformative_bins = VT(U) - VT = -0.132 (kept negative);
   randomisation benchmark V0 - VT(U) = +0.132;  operational = benchmark + contrast.
Randomised policies: V0_rand = 3.428 (A with probability 0.2, risk 0.1).  VT_rand: A in the low bin
saves 0.132 per 0.15 risk, in the high bin 0.528 per 0.35 risk (better ratio) -> w(high, A) = 0.1/0.35,
VT_rand = 3.56 - 0.528/3.5 = 3.4091428571...
=> randomized_same_class_information_reference = 0.132/7 = 0.0188571428571...

Running this file as a script writes ``reports/value_definition_counterexample_results.json``
(``python tests/unit/test_value_definition_counterexample.py --write``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

if __name__ == "__main__":                                    # script mode: same paths as tests/conftest.py
    _here = Path(__file__).resolve()
    for _p in (str(_here.parents[2] / "src"), str(_here.parents[1])):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    sys.dont_write_bytecode = True

import numpy as np
import pytest
from scipy.stats import norm

from engine_test_helpers import conc, dm_offer, ing, problem
from ration_reliability.datamodel import Provenance, RationDecision, ValueStatus
from ration_reliability.errors import InvalidProblemError
from ration_reliability.information import (
    AssayBudget,
    AssayOption,
    BatchCoverage,
    CandidateLibrary,
    ComponentErrorModel,
    InformationStructure,
    ObservedComponent,
    PolicyProblem,
    PriorStates,
    SignalBinning,
    SignalModel,
    ValueDefinition,
    VALUE_DEFINITION_ROLES,
    batch_gross_value,
    break_even_max_cost_per_batch,
    build_likelihood,
    check_theoretical_order,
    compute_information_value,
    compute_risk_table,
    evaluate_frozen_policy,
    freeze_policy,
    inventory_coverage_check,
    net_value_per_batch,
    per_head_day_value,
    solve_signal_policy,
    strategy_decision_value,
)
from ration_reliability.uncertainty import DrawSet, RandomStreams

OP = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE
RND = ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE
CON = ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS

SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-R1-VALUE-DEFINITION-COUNTEREXAMPLE")
FCP = ObservedComponent("F", "CP")
ALPHA = 0.10
P_LOW_GIVEN_BAD, P_LOW_GIVEN_GOOD = 0.30, 0.10
COST = {"A": 2.90, "B": 3.56}

#: hand table: (low-bin action, high-bin action) -> (expected cost, ex-ante joint risk, feasible at alpha)
HAND_TABLE = {
    ("B", "B"): (3.56, 0.0, True),
    ("A", "B"): (0.2 * 2.90 + 0.8 * 3.56, 0.5 * 0.30, False),
    ("B", "A"): (0.2 * 3.56 + 0.8 * 2.90, 0.5 * 0.70, False),
    ("A", "A"): (2.90, 0.5, False),
}
HAND = {
    "V0": 3.56, "VT": 3.56, "VT_matched_uninformative_bins": 0.2 * 2.90 + 0.8 * 3.56,
    "V0_randomized": 3.56 - 0.2 * 0.66, "VT_randomized": 3.56 - 0.8 * 0.66 * (0.10 / 0.35),
    OP.value: 0.0, RND.value: 0.132 / 7.0, CON.value: -0.132, "randomization_benchmark": 0.132,
}
THR0 = {"min_marginal_value": 0.0, "min_marginal_value_status": "synthetic_test_only"}


def build_counterexample():
    """Problem, library, prior, risk table, signal structure and option of the counterexample."""
    prob = problem([ing("F", 0.40, {"CP": 0.10}, forage=1.0), ing("C", 0.80, {"CP": 0.40})], ["CP"],
                   [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0)], {"F": 0.04, "C": 0.32},
                   problem_id="r1_value_definition_counterexample")
    ids, dh = prob.ingredient_ids, prob.dm_estimates()
    lib = CandidateLibrary.from_decisions(prob, [RationDecision(ids, np.array([42.5, 3.75]), dh, "synthetic_A"),
                                                 RationDecision(ids, np.array([37.0, 6.50]), dh, "synthetic_B")],
                                          ["A", "B"])
    prior = PriorStates.from_discrete(np.array([[[0.08], [0.40]], [[0.12], [0.40]]]), np.array([[0.4, 0.8]] * 2),
                                      [0.5, 0.5], ids, ["CP"], label="r1_counterexample", is_synthetic=True)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    sd = 0.04 / (norm.ppf(P_LOW_GIVEN_BAD) - norm.ppf(P_LOW_GIVEN_GOOD))
    edge = 0.08 + sd * norm.ppf(P_LOW_GIVEN_BAD)
    em = ComponentErrorModel(FCP, sd, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True)
    sm = SignalModel("r1_counterexample_signal", (em,), is_synthetic=True)
    binning = SignalBinning((FCP,), ((edge,),), "synthetic single edge (P(low|bad)=0.3, P(low|good)=0.1)")
    st = InformationStructure("r1_counterexample", "sample", binning=binning, signal_model=sm)
    option = AssayOption("r1_counterexample", "F", ("CP",), binning, sm)
    return {"problem": prob, "library": lib, "prior": prior, "risk": risk, "structure": st, "option": option,
            "sd": float(sd), "edge": float(edge)}


def _value(case, **kw):
    """The one place where the counterexample is valued (prior variance basis declared as in the probe)."""
    return compute_information_value(case["structure"], case["prior"], case["risk"], ALPHA,
                                     prior_variance_basis={FCP: "true_batch_state"}, **kw)


@pytest.fixture(scope="module")
def case():
    return build_counterexample()


@pytest.fixture(scope="module")
def res(case):
    return _value(case)


# =================================================================================================
# the hand table
# =================================================================================================

def test_signal_bins_and_hand_table_of_the_four_deterministic_policies(case):
    lik = build_likelihood(case["structure"], case["prior"])
    assert lik.L[0] == pytest.approx([P_LOW_GIVEN_BAD, 1 - P_LOW_GIVEN_BAD], abs=1e-12)     # bad state
    assert lik.L[1] == pytest.approx([P_LOW_GIVEN_GOOD, 1 - P_LOW_GIVEN_GOOD], abs=1e-12)   # good state
    pp = PolicyProblem.build(case["prior"], lik, case["risk"], ALPHA)
    assert pp.Pz == pytest.approx([0.2, 0.8], abs=1e-12)
    assert list(pp.candidate_ids) == ["A", "B"] and pp.costs == pytest.approx([2.90, 3.56], abs=1e-12)
    idx = {"A": 0, "B": 1}
    for (lo, hi), (cost, risk, feasible) in HAND_TABLE.items():
        c, r = pp.policy_cost_risk(np.array([idx[lo], idx[hi]]))
        assert c == pytest.approx(cost, abs=1e-12) and r == pytest.approx(risk, abs=1e-12)
        assert (r <= ALPHA + pp.risk_tol) is feasible
    for method in ("milp", "enumeration"):                     # solver = exhaustive enumeration
        vt = solve_signal_policy(pp, method=method)
        assert vt.expected_cost == pytest.approx(3.56, abs=1e-12)
        assert [pp.candidate_ids[k] for k in vt.assignment] == ["B", "B"]


# =================================================================================================
# the three definitions on the counterexample
# =================================================================================================

def test_values_under_each_definition_match_the_hand_calculation(res):
    assert res.info_type == "sample" and res.is_synthetic and res.is_information_value
    assert res.V0.expected_cost == pytest.approx(HAND["V0"], abs=1e-12)
    assert res.VT.expected_cost == pytest.approx(HAND["VT"], abs=1e-12)
    assert res.VT.ex_ante_risk == pytest.approx(0.0, abs=1e-15)
    assert res.gross_value_status == "ok" and res.containment_check["holds"] is True
    # operational deterministic cost difference: 0 (floating point ~ -4.4e-16 is inside the tolerance)
    assert res.value(OP) == pytest.approx(HAND[OP.value], abs=1e-12)
    assert res.operational_deterministic_cost_difference == res.gross_value
    # randomised same-class reference: +0.132/7
    assert res.V0_randomized.expected_cost == pytest.approx(HAND["V0_randomized"], abs=1e-9)
    assert res.VT_randomized.expected_cost == pytest.approx(HAND["VT_randomized"], abs=1e-9)
    assert res.value(RND) == pytest.approx(0.018857142857142857, abs=1e-9)
    assert res.value(RND) == pytest.approx(HAND[RND.value], abs=1e-9)
    # diagnostic contrast: -0.132, kept negative (never truncated)
    assert res.value(CON) == pytest.approx(HAND[CON.value], abs=1e-12)
    assert res.contrast_vs_matched_uninformative_bins < 0
    assert res.VT_uninformative_benchmark.expected_cost == pytest.approx(HAND["VT_matched_uninformative_bins"],
                                                                         abs=1e-12)
    assert res.VT_uninformative_benchmark.ex_ante_risk == pytest.approx(0.10, abs=1e-12)
    assert [res.candidate_ids[k] for k in res.VT_uninformative_benchmark.assignment] == ["A", "B"]
    assert res.randomization_benchmark_value == pytest.approx(HAND["randomization_benchmark"], abs=1e-12)
    # identity: operational = benchmark + contrast (the contrast is not an additive "pure information" part)
    assert res.gross_value == pytest.approx(res.randomization_benchmark_value + res.contrast_vs_matched_uninformative_bins,
                                            abs=1e-12)
    assert any("contrast_vs_matched_uninformative_bins" in w and "< 0" in w for w in res.warnings)


def test_primary_value_is_unavailable_without_an_explicit_definition(res):
    assert res.value_definition is None
    assert res.primary_value is None
    assert res.primary_value_status == "no_value_definition_selected"
    assert res.primary_value_role is None
    with pytest.raises(InvalidProblemError, match="no value_definition was chosen"):
        res.require_primary_value()
    d = res.to_dict()
    assert d["primary_value"] is None and d["value_definition"] is None
    assert "value_net_of_randomization" not in d and "value_net_of_randomization" in d["deprecated_aliases"]
    assert d["values_by_definition"][CON.value] == pytest.approx(-0.132, abs=1e-12)
    text = repr(res)
    assert "primary_value=None [no_value_definition_selected]" in text
    assert "contrast_vs_matched_uninformative_bins[diagnostic" in text


def test_explicit_definitions_select_the_primary_value(case, res):
    expect = {OP: HAND[OP.value], RND: HAND[RND.value], CON: HAND[CON.value]}
    for d, v in expect.items():
        tol = 1e-9 if d is RND else 1e-12
        chosen = _value(case, value_definition=d)
        assert chosen.value_definition == d.value and chosen.primary_value_status == "ok"
        assert chosen.primary_value == pytest.approx(v, abs=tol)
        assert chosen.require_primary_value() == pytest.approx(v, abs=tol)
        assert chosen.primary_value_role == VALUE_DEFINITION_ROLES[d.value]
        again = res.with_value_definition(d.value)                  # canonical string also accepted
        assert again.primary_value == pytest.approx(v, abs=tol)
    assert VALUE_DEFINITION_ROLES[CON.value] == "diagnostic"
    with pytest.raises(InvalidProblemError, match="unknown value_definition"):
        _value(case, value_definition="net_of_randomization")      # legacy names are not definitions
    with pytest.raises(InvalidProblemError, match="include_randomized_reference"):
        _value(case, value_definition=RND, include_randomized_reference=False)
    with pytest.raises(InvalidProblemError, match="include_randomization_benchmark"):
        _value(case, value_definition=CON, include_randomization_benchmark=False)


def test_deprecated_alias_is_the_diagnostic_contrast_and_warns(res):
    with pytest.warns(DeprecationWarning, match="diagnostic"):
        v = res.value_net_of_randomization
    assert v == res.contrast_vs_matched_uninformative_bins
    with pytest.warns(DeprecationWarning):
        assert res.value("net_of_randomization") == res.contrast_vs_matched_uninformative_bins
    with pytest.warns(DeprecationWarning):
        assert res.value("deterministic") == res.gross_value


def test_definition_report_pairs_values_with_deterministic_cost_risk_and_feasibility(res):
    rep = res.definition_report(RND)
    assert rep["selected_value_definition"] == RND.value
    assert rep["selected_value"] == pytest.approx(HAND[RND.value], abs=1e-9)
    assert rep["values"][CON.value]["role"] == "diagnostic"
    assert rep["values"][RND.value]["role"] == "theoretical_reference"
    det = rep["deterministic_policies"]
    assert det["V0_constant"]["expected_cost"] == pytest.approx(3.56, abs=1e-12)
    assert det["V0_constant"]["meets_risk_target"] is True
    assert det["VT_signal_policy"]["assignment_candidate_ids"] == ["B", "B"]
    assert det["VT_signal_policy"]["ex_ante_joint_risk"] == pytest.approx(0.0, abs=1e-15)
    assert det["VT_signal_policy"]["meets_risk_target"] is True
    # FIX_A: the matched uninformative-bins policy is a hypothetical randomisation device, not an executable
    # feeding policy -- same assertion, now in the diagnostic block
    assert "VT_matched_uninformative_bins_policy" not in det
    dev = rep["diagnostic_randomisation_devices"]
    assert dev["role"] == "hypothetical_randomisation_device" and dev["not_a_feeding_recommendation"] is True
    assert dev["VT_matched_uninformative_bins_policy"]["assignment_candidate_ids"] == ["A", "B"]
    rnd = rep["randomized_reference_policies"]
    assert rnd["not_a_feeding_recommendation"] is True
    assert rnd["VT_randomized"]["expected_cost"] == pytest.approx(HAND["VT_randomized"], abs=1e-9)
    w = np.asarray(rnd["VT_randomized"]["w"])                       # A with probability 0.1/0.35 in the high bin
    assert w[1, 0] == pytest.approx(0.10 / 0.35, abs=1e-7) and w[0, 0] == pytest.approx(0.0, abs=1e-7)


def test_theoretical_order_checks_use_operational_and_randomized_only(case, res):
    pp = compute_information_value(InformationStructure("pp_exact", "perfect_partial", (FCP,)), case["prior"],
                                   case["risk"], ALPHA)
    assert pp.gross_value == pytest.approx(0.33, abs=1e-12)                  # 0.5*3.56 + 0.5*2.90 = 3.23
    assert pp.gross_value_randomized_reference == pytest.approx(3.428 - (0.5 * 2.90 + 0.5 * 3.428), abs=1e-9)
    checks = check_theoretical_order({"sample": res, "pp_exact": pp})
    decided = [c for c in checks if c.holds is not None]
    assert len(decided) == len(checks) == 6 and all(c.holds for c in decided)   # 4 nonnegativity + 2 order checks
    assert {c.policy_class for c in checks} == {"deterministic", "randomized"}   # the contrast is never order-checked


def test_uninformative_structure_operational_positive_randomized_zero_contrast_zero(case):
    """Same bin probabilities, no information: the operational value is pure randomisation channel."""
    un = compute_information_value(InformationStructure("u", "uninformative", bin_probabilities=(0.2, 0.8)),
                                   case["prior"], case["risk"], ALPHA)
    assert un.value(OP) == pytest.approx(0.132, abs=1e-12)
    assert un.value(RND) == pytest.approx(0.0, abs=1e-9)
    assert un.value(CON) == 0.0


# =================================================================================================
# call sites: strategies, economics, frozen policies
# =================================================================================================

def test_strategy_ranking_requires_an_explicit_definition(case):
    args = ([case["option"]], case["prior"], case["risk"], ALPHA, AssayBudget(max_panels=1))
    kw = dict(prior_variance_basis={FCP: "true_batch_state"}, **THR0)
    with pytest.raises(InvalidProblemError, match="explicit value_definition"):
        strategy_decision_value(*args, **kw)
    op = strategy_decision_value(*args, value_definition=OP, **kw)
    assert op.selected == () and op.details["value_role"] == "operational"
    assert op.details["greedy_history"][0]["joint_value"] == pytest.approx(0.0, abs=1e-12)
    rnd = strategy_decision_value(*args, value_definition=RND, **kw)
    assert rnd.selected == ("r1_counterexample",) and rnd.details["value_role"] == "theoretical_reference"
    step = rnd.details["greedy_history"][0]
    assert step["joint_value"] == pytest.approx(HAND[RND.value], abs=1e-9)
    # the executable deterministic policy of the selected assay saves nothing: reported next to the reference
    assert step["operational"]["VT_signal_policy"]["expected_cost"] == pytest.approx(3.56, abs=1e-12)
    assert step["operational"]["VT_signal_policy"]["meets_risk_target"] is True
    assert step["operational"]["operational_deterministic_cost_difference"] == pytest.approx(0.0, abs=1e-12)
    con = strategy_decision_value(*args, value_definition=CON, **kw)
    assert con.selected == () and con.details["value_role"] == "diagnostic" and "diagnostic" in con.notes
    assert con.details["greedy_history"][0]["joint_value"] == pytest.approx(-0.132, abs=1e-12)
    with pytest.warns(DeprecationWarning):
        legacy = strategy_decision_value(*args, value_kind="net_of_randomization", **kw)
    assert legacy.details["value_definition"] == CON.value and legacy.details["value_kind"] == "net_of_randomization"
    with pytest.raises(InvalidProblemError, match="disagree"), pytest.warns(DeprecationWarning):
        strategy_decision_value(*args, value_definition=OP, value_kind="net_of_randomization", **kw)


def test_economic_conversions_require_an_explicit_admissible_definition(res):
    cov = BatchCoverage(100.0, 14.0, "synthetic_test_only", "synthetic_test_only", 14.0)
    inv = inventory_coverage_check(np.array([[37.0, 6.5]]), ("F", "C"), cov, {"F": 1e9, "C": 1e9})
    with pytest.raises(InvalidProblemError, match="explicit value_definition"):
        batch_gross_value(0.01, cov, inventory_check=inv)
    with pytest.raises(InvalidProblemError, match="explicit value_definition"):
        break_even_max_cost_per_batch(0.01, cov, inventory_check=inv)
    with pytest.raises(InvalidProblemError, match="diagnostic contrast"):
        break_even_max_cost_per_batch(0.01, cov, inventory_check=inv, value_definition=CON)
    with pytest.raises(InvalidProblemError, match="explicit value_definition"):
        per_head_day_value(res, None)
    with pytest.raises(InvalidProblemError, match="diagnostic contrast"):
        per_head_day_value(res, CON)
    # FIX_A: this counterexample prior has no object metadata -> unidentified_scenario; money needs scenario=True
    with pytest.raises(InvalidProblemError, match="scenario=True"):
        per_head_day_value(res, OP)
    # FIX3_BC (round-3 red team B-1): no money without a complete decision problem; the conversions below are
    # labelled development diagnostics (only the keyword was added; expected values unchanged)
    diag = {"diagnostic_without_decision_problem": True}
    dc_op = per_head_day_value(res, OP, scenario=True, **diag)
    assert dc_op == pytest.approx(0.0, abs=1e-12)
    assert break_even_max_cost_per_batch(dc_op, cov, inventory_check=inv, value_definition=OP, **diag) == \
        pytest.approx(0.0, abs=1e-8)
    dc_rnd = per_head_day_value(res, RND, scenario=True, **diag)
    nv = net_value_per_batch(dc_rnd, cov, {k: None for k in ("sampling", "laboratory", "logistics", "waiting",
                                                             "reformulation")}, inventory_check=inv,
                             value_definition=RND, **diag)
    assert nv.break_even_max_cost_per_batch == pytest.approx(1400.0 * HAND[RND.value], abs=1e-6)
    assert nv.value_role == "theoretical_reference" and "never recommended" in nv.value_interpretation
    assert nv.net_value_per_batch is None
    fake = dataclasses.replace(res, information_value_semantics="cost_risk_comparison_action_spaces_differ",
                               gross_value=None)
    with pytest.raises(InvalidProblemError, match="not an information value"):
        per_head_day_value(fake, OP)


def test_frozen_policy_requires_the_operational_definition(case, res):
    st, lib = case["structure"], case["library"]
    with pytest.raises(InvalidProblemError, match="explicit value_definition"):
        freeze_policy(res, st, lib)
    with pytest.raises(InvalidProblemError, match="never frozen"):
        freeze_policy(res, st, lib, value_definition=RND)
    with pytest.raises(InvalidProblemError, match="diagnostic contrast"):
        freeze_policy(res, st, lib, value_definition=CON)
    pol = freeze_policy(res, st, lib, value_definition=OP)
    dev = pol.development
    assert dev["value_definition"] == OP.value
    assert dev["development_value"] == pytest.approx(0.0, abs=1e-12)
    assert dev["contrast_vs_matched_uninformative_bins_development_diagnostic"] == pytest.approx(-0.132, abs=1e-12)
    assert [lib.candidate_ids[k] for k in pol.assignment] == ["B", "B"]
    prior = case["prior"]
    draws = DrawSet(np.repeat(prior.theta, 50, axis=0), np.repeat(prior.d, 50, axis=0), "validation",
                    "synthetic/r1_counterexample/validation", "r1_counterexample", "synthetic", prior.ingredient_ids,
                    prior.nutrient_ids, True)
    streams = RandomStreams(1103)
    ev = evaluate_frozen_policy(pol, draws, case["problem"].compiled, streams, prices=case["problem"].price_vector())
    assert ev.extra["value_definition"] == OP.value
    assert ev.cost_difference_per_head_day == pytest.approx(0.0, abs=1e-12)          # (B, B) vs B
    unmarked = dataclasses.replace(pol, development={k: v for k, v in dev.items() if k != "value_definition"})
    with pytest.raises(InvalidProblemError, match="no explicit value_definition"):
        evaluate_frozen_policy(unmarked, draws, case["problem"].compiled, streams)


# =================================================================================================
# script mode: write the synthetic results file
# =================================================================================================

def _code_identity(root: Path) -> dict:
    """CLOSE2: bind the results file to the whole code state that produced it, not only to the five files listed in
    ``source_sha256`` (the generator also imports the prior / signal / library, datamodel and uncertainty modules)."""
    from ration_reliability.io.run_record import code_manifest, code_manifest_subset_sha256

    man = code_manifest(root)
    sub = code_manifest_subset_sha256(man)
    return {"code_manifest_sha256": man["manifest_sha256"], "code_manifest_file_count": len(man.get("files", [])),
            "code_manifest_complete": man.get("complete"), "code_subset_sha256": sub["subset_sha256"],
            "code_subset_areas": sub["areas"], "code_subset_file_count": sub["file_count"],
            "function": "ration_reliability.io.run_record.code_manifest",
            "note": "source_sha256 lists the five files named by the review; code_identity covers the full code "
                    "manifest scope (src experiments scripts tests configs + three root files)"}


def counterexample_record() -> dict:
    """All numbers of the counterexample (hand and computed); every value is synthetic."""
    import platform

    import scipy

    c = build_counterexample()
    r = _value(c)
    pp = PolicyProblem.build(c["prior"], build_likelihood(c["structure"], c["prior"]), c["risk"], ALPHA)
    idx = {"A": 0, "B": 1}
    table = []
    for (lo, hi), (hc, hr, hf) in HAND_TABLE.items():
        cc, cr = pp.policy_cost_risk(np.array([idx[lo], idx[hi]]))
        table.append({"policy_low_bin": lo, "policy_high_bin": hi, "hand_expected_cost": hc, "hand_ex_ante_risk": hr,
                      "hand_feasible_at_alpha": hf, "computed_expected_cost": cc, "computed_ex_ante_risk": cr,
                      "computed_feasible_at_alpha": bool(cr <= ALPHA + pp.risk_tol), "is_synthetic": True})
    args = ([c["option"]], c["prior"], c["risk"], ALPHA, AssayBudget(max_panels=1))
    kw = dict(prior_variance_basis={FCP: "true_batch_state"}, **THR0)
    strat = {}
    for d in (OP, RND, CON):
        s = strategy_decision_value(*args, value_definition=d, **kw)
        strat[d.value] = {"selected": list(s.selected), "value_role": s.details["value_role"],
                          "joint_value": s.details["greedy_history"][0]["joint_value"],
                          "operational_block": s.details["greedy_history"][0]["operational"], "is_synthetic": True}
    root = Path(__file__).resolve().parents[2]
    src = ["src/ration_reliability/information/value.py", "src/ration_reliability/information/strategies.py",
           "src/ration_reliability/information/economics.py", "src/ration_reliability/information/evaluation.py",
           "tests/unit/test_value_definition_counterexample.py"]
    return {
        "title": "R1 value-definition counterexample (second review)",
        "status": "unit_passed",
        "value_status": "synthetic_test_only",
        "is_synthetic": True,
        "scope": ("synthetic mathematical counterexample; no feed, farm, price or biological claim; not a pilot or "
                  "official result; must not enter the manuscript"),
        "origin": "docs/review_20260924_round2/RuminantSim_Second_Review/independent_probes.py "
                  "(negative_default_primary_value); hand calculation repeated in the generator docstring",
        "setup": {"states": {"bad": {"F_CP_fraction_DM": 0.08, "prior": 0.5}, "good": {"F_CP_fraction_DM": 0.12,
                                                                                       "prior": 0.5}},
                  "candidates": {"A": {"q_as_fed_kg": [42.5, 3.75], "cost_per_head_day": 2.90,
                                       "violates_in": ["bad"]},
                                 "B": {"q_as_fed_kg": [37.0, 6.50], "cost_per_head_day": 3.56, "violates_in": []}},
                  "alpha": ALPHA, "risk_tol": pp.risk_tol, "risk_definition": "ex-ante joint violation probability",
                  "signal_sd": c["sd"], "signal_edge": c["edge"],
                  "p_low_given_bad_good": [float(x) for x in build_likelihood(c["structure"], c["prior"]).L[:, 0]],
                  "bin_probabilities_low_high": [float(x) for x in pp.Pz], "currency": "synthetic units/head/d",
                  "is_synthetic": True},
        "hand_table_deterministic_policies": table,
        "hand_values": {k: float(v) for k, v in HAND.items()},
        "computed": {
            "V0": r.V0.expected_cost, "VT": r.VT.expected_cost, "VT_ex_ante_risk": r.VT.ex_ante_risk,
            "VT_matched_uninformative_bins": r.VT_uninformative_benchmark.expected_cost,
            "VT_matched_uninformative_bins_risk": r.VT_uninformative_benchmark.ex_ante_risk,
            "V0_randomized": r.V0_randomized.expected_cost, "VT_randomized": r.VT_randomized.expected_cost,
            "values_by_definition": r.values_by_definition(),
            "value_definition_roles": dict(VALUE_DEFINITION_ROLES),
            "randomization_benchmark_V0_minus_VT_uninformative": r.randomization_benchmark_value,
            "identity_operational_equals_benchmark_plus_contrast_abs_error":
                abs(r.gross_value - r.randomization_benchmark_value - r.contrast_vs_matched_uninformative_bins),
            "primary_value_without_definition": r.primary_value,
            "primary_value_status_without_definition": r.primary_value_status,
            "contrast_kept_negative": bool(r.contrast_vs_matched_uninformative_bins < 0),
            "warnings": list(r.warnings), "is_synthetic": True},
        "definition_report": json.loads(json.dumps(r.definition_report(), default=float)),
        "strategy_decision_value_by_definition": strat,
        "economics": {"admissible_definitions": [OP.value, RND.value,
                                                 ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value],
                      "refused_definitions": [CON.value],
                      "money_basis_required": ("FIX3_BC (round-3 red team B-1): a CompleteDecisionProblem (its one "
                                               "policy class; heuristic refused) or an explicit, labelled development "
                                               "diagnostic; every money output excluded from paper main results"),
                      "operational_refused_on_the_diagnostic_path_if": (
                          "operational > randomised reference + tol, or randomization_only (FIX_A refusal, kept "
                          "refusal-only; passing it is not a permission; docs/value_definition.md §8.2)"),
                      "per_head_day_value_operational": float(per_head_day_value(
                          r, OP, scenario=True, diagnostic_without_decision_problem=True)),
                      "per_head_day_value_randomized_reference_theoretical": float(per_head_day_value(
                          r, RND, scenario=True, diagnostic_without_decision_problem=True)),
                      "money_basis_note": "FIX3_BC: development diagnostic without a complete decision problem; "
                                          "excluded from paper main results",
                      "identification_note": "unidentified_scenario (no object metadata on the prior): converted "
                                             "with scenario=True (FIX_A)",
                      "is_synthetic": True},
        "generator": "tests/unit/test_value_definition_counterexample.py --write",
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in src},
        "code_identity": _code_identity(root),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }


if __name__ == "__main__":
    import datetime

    rec = counterexample_record()
    rec["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = json.dumps(rec, ensure_ascii=False, indent=2, default=float) + "\n"
    if "--write" in sys.argv:
        out = Path(__file__).resolve().parents[2] / "reports" / "value_definition_counterexample_results.json"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
