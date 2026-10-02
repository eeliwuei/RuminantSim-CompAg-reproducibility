"""Third-review garbling counterexample (R3B; review §3.2), rebuilt independently.  ALL NUMBERS ARE SYNTHETIC.

Source: ``docs/review_20260925_round3/RuminantSim_第三次复核报告_20260925.md`` §3.2 (the review's
``evidence/probe_information_garbling.*`` files were not shipped; every parameter below is taken from the
report text and the hand calculation is repeated here).  It is a mathematical test case, not a feed, farm,
price or biological result, and it must not enter the manuscript results.

Setup (unchanged from the review; alpha, policies and bins are *not* modified)
------------------------------------------------------------------------------
* two states ("bad": forage CP 8 % DM, "good": 12 % DM), prior 1/2 each; one probabilistic constraint
  CP >= 16 % DM; planned DM 20 kg/d;
* candidate A: cost 2.90, violates only in the bad state; candidate B: cost 3.56, never violates;
* ex-ante joint risk target alpha = 0.10;
* original signal ``L = [[0.3, 0.7], [0.1, 0.9]]`` (rows: bad, good; columns: low bin, high bin);
  pure garbling ``G = [[1, 0], [5/7, 2/7]]`` (a row-stochastic re-labelling of the observed bin that
  adds no state information); garbled signal ``L' = L G = [[0.8, 0.2], [26/35, 9/35]]``.
  Both are produced by the public API: a Gaussian sample signal ``Z = CP_F + e`` with one bin edge
  (:class:`SignalModel` + :class:`SignalBinning`, analytic likelihood).

Hand calculation (deterministic policies = (action in the low bin, action in the high bin))
------------------------------------------------------------------------------------------
Original L (P_low = 0.2, P_high = 0.8)::

    (B, B) 3.560  risk 0     feasible      (A, B) 3.428  risk 0.15  infeasible
    (B, A) 3.032  risk 0.35  infeasible    (A, A) 2.900  risk 0.50  infeasible
    => V0 = VT = 3.56, operational = 0; matched device (A, B) at risk 0.2*0.5 = 0.1 -> B = 0.132;
       V0_rand = 3.428, VT_rand = 3.56 - 0.528/3.5 -> randomised reference = 0.132/7 = 0.018857;
       min(operational, reference) = 0.

Garbled L' (P_low = 27/35, P_high = 8/35)::

    (B, B) 3.560              risk 0    feasible      (A, B) 106.78/35 = 3.050857  risk 0.4  infeasible
    (B, A) 119.32/35 = 3.409143  risk 0.1  feasible   (A, A) 2.900                 risk 0.5  infeasible
    => VT = 3.409143, operational = 5.28/35 = 0.150857; matched device: (A, B) risk 27/70, (B, A) risk
       8/70 > 0.1 -> VT(U) = 3.56, B = 0; VT_rand = 3.56 - 5.28/35 (high bin first, ratio 1.509 > 1.273)
       -> randomised reference = 0.018857 (unchanged); min(operational, reference) = 0.018857.

A pure garbling leaves the randomised same-class reference unchanged and *raises* the heuristic minimum
from 0 to 0.018857; the matched benchmark reports ``no_randomization_channel_measured`` although the
operational saving (0.150857) exceeds the randomised reference by 0.132.  This is why
``executable_information_supported_value`` was renamed ``heuristic_min_of_two_policy_values`` and
withdrawn as a payment bound / assay priority (``docs/value_semantics_decision.md``).

Running this file as a script writes ``reports/garbling_counterexample.json``
(``python tests/unit/test_garbling_counterexample.py --write``).
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
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
    DecisionProblemRequiredError,
    InformationStructure,
    ObservedComponent,
    PolicyProblem,
    PriorStates,
    RandomizationChannelError,
    SamplingProtocol,
    SignalBinning,
    SignalModel,
    ValueDefinition,
    break_even_max_cost_per_batch,
    build_likelihood,
    compute_information_value,
    compute_risk_table,
    inventory_coverage_check,
    net_value_per_batch,
    per_head_day_value,
    solve_no_information,
    solve_no_information_randomized,
    solve_signal_policy,
    solve_signal_policy_randomized,
    strategy_decision_value,
)

OP = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE
RND = ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE
HEUR = ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES
SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-R3B-GARBLING-COUNTEREXAMPLE")
FCP = ObservedComponent("F", "CP")
ALPHA = 0.10
PVB = {FCP: "true_batch_state"}
COST = {"A": 2.90, "B": 3.56}
COSTS_UNKNOWN = {k: None for k in ("sampling", "laboratory", "logistics", "waiting", "reformulation")}
#: FIX3_BC (round-3 red team B-1): money needs a complete decision problem; labelled development diagnostic otherwise
DIAG = {"diagnostic_without_decision_problem": True}

L_MATRIX = np.array([[0.3, 0.7], [0.1, 0.9]])
G_MATRIX = np.array([[1.0, 0.0], [5.0 / 7.0, 2.0 / 7.0]])
LG_EXACT = np.array([[0.8, 0.2], [26.0 / 35.0, 9.0 / 35.0]])
#: FIX3_BC (round-3 red team B-3): the review's garbled policy (B, A) has risk exactly alpha = 0.10 and is feasible
#: only through risk_tol = 1e-8.  A strong interior garbling keeps the counterexample away from that boundary:
#: ``L G_int = [[0.804, 0.196], [0.748, 0.252]]``, (B, A) risk 0.098 < alpha; heuristic minimum 0 -> 0.018388.
G_INTERIOR = np.array([[1.0, 0.0], [0.72, 0.28]])
LG_INTERIOR = np.array([[0.804, 0.196], [0.748, 0.252]])
INTERIOR_ALPHAS = (0.10, 0.0999, 0.0985)

F = Fraction
#: exact hand values (fractions); the costs 2.90 / 3.56 enter as 29/10 and 89/25
_CA, _CB = F(29, 10), F(89, 25)


def _hand(p_low: Fraction, lik_bad_low: Fraction) -> dict:
    """Hand table and values of one two-bin signal (exact fractions; see module docstring)."""
    p_high = 1 - p_low
    r_low_a, r_high_a = F(1, 2) * lik_bad_low, F(1, 2) * (1 - lik_bad_low)       # A violates in the bad state only
    table = {("B", "B"): (_CB, F(0)), ("A", "B"): (p_low * _CA + p_high * _CB, r_low_a),
             ("B", "A"): (p_low * _CB + p_high * _CA, r_high_a), ("A", "A"): (_CA, F(1, 2))}
    a = F(1, 10)
    feas = {k: v for k, v in table.items() if v[1] <= a}
    vt = min(c for c, _ in feas.values())
    v0 = _CB                                                                 # A alone has risk 1/2 > alpha
    # randomised: V0_rand mixes A with weight alpha / (1/2); VT_rand fills the better saving/risk ratio first
    v0r = _CB - (a / F(1, 2)) * (_CB - _CA)
    bins = sorted([(p_low * (_CB - _CA) / r_low_a, p_low * (_CB - _CA), r_low_a),
                   (p_high * (_CB - _CA) / r_high_a, p_high * (_CB - _CA), r_high_a)], reverse=True)
    budget, vtr = a, _CB
    for _, save, risk in bins:
        w = min(F(1), budget / risk)
        vtr -= w * save
        budget -= w * risk
    # matched state-independent device: same bin probabilities, risk of A in bin z = P_z / 2
    dev = {("B", "B"): (_CB, F(0)), ("A", "B"): (p_low * _CA + p_high * _CB, p_low / 2),
           ("B", "A"): (p_low * _CB + p_high * _CA, p_high / 2), ("A", "A"): (_CA, F(1, 2))}
    vtu = min(c for c, r in dev.values() if r <= a)
    op, rr = v0 - vt, v0r - vtr
    return {"table": table, "V0": v0, "VT": vt, "V0_randomized": v0r, "VT_randomized": vtr, "VT_matched_device": vtu,
            "operational": op, "randomized_reference": rr, "benchmark_B": v0 - vtu, "contrast": vtu - vt,
            "heuristic_min": min(op, rr)}


HAND = {"original": _hand(F(1, 5), F(3, 10)), "garbled": _hand(F(27, 35), F(4, 5))}

#: numbers printed in review §3.2 (6 decimals); ``Δ_op ≈ 0`` and ``min ≈ 0`` there are exact zeros here
REVIEW_REPORTED = {
    "original": {"V0": 3.56, "VT": 3.56, "operational": 0.0, "randomized_reference": 0.018857,
                 "heuristic_min": 0.0, "benchmark_B": 0.132},
    "garbled": {"V0": 3.56, "VT": 3.409143, "operational": 0.150857, "randomized_reference": 0.018857,
                "heuristic_min": 0.018857, "benchmark_B": 0.0},
}


# =================================================================================================
# construction (public API only)
# =================================================================================================

def build_world(cp_max: float | None = None):
    """Problem, library {A, B}, two-state prior and risk table (``cp_max`` only for the undefined-value case)."""
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0)]
    if cp_max is not None:
        cons.append(conc("cp_max", {"CP": 1.0}, "le", cp_max))
    prob = problem([ing("F", 0.40, {"CP": 0.10}, forage=1.0), ing("C", 0.80, {"CP": 0.40})], ["CP"], cons,
                   {"F": 0.04, "C": 0.32}, problem_id="r3b_garbling_counterexample")
    ids, dh = prob.ingredient_ids, prob.dm_estimates()
    lib = CandidateLibrary.from_decisions(prob, [RationDecision(ids, np.array([42.5, 3.75]), dh, "synthetic_A"),
                                                 RationDecision(ids, np.array([37.0, 6.50]), dh, "synthetic_B")],
                                          ["A", "B"])
    prior = PriorStates.from_discrete(np.array([[[0.08], [0.40]], [[0.12], [0.40]]]), np.array([[0.4, 0.8]] * 2),
                                      [0.5, 0.5], ids, ["CP"], label="r3b_garbling", is_synthetic=True)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    return prob, lib, prior, risk


def gaussian_signal(p_low_bad: float, p_low_good: float, sid: str):
    """``Z = CP_F + e``, ``e ~ N(0, sd^2)``, one bin edge: P(low | bad) = p_low_bad, P(low | good) = p_low_good."""
    sd = 0.04 / (norm.ppf(p_low_bad) - norm.ppf(p_low_good))
    edge = 0.08 + sd * norm.ppf(p_low_bad)
    sm = SignalModel(sid, (ComponentErrorModel(FCP, sd, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True),),
                     SamplingProtocol(), is_synthetic=True)
    binning = SignalBinning((FCP,), ((edge,),), "synthetic single edge (R3B)")
    return InformationStructure(sid, "sample", binning=binning, signal_model=sm), sd, edge


def structures():
    """Original signal L and its pure garbling L' = L G (both through the Gaussian-signal API)."""
    s0, sd0, e0 = gaussian_signal(0.3, 0.1, "original_L")
    s1, sd1, e1 = gaussian_signal(0.8, 26.0 / 35.0, "garbled_L_G")
    return {"original": (s0, sd0, e0), "garbled": (s1, sd1, e1)}


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture(scope="module")
def results(world):
    prob, lib, prior, risk = world
    return {k: compute_information_value(s, prior, risk, ALPHA, prior_variance_basis=PVB)
            for k, (s, _, _) in structures().items()}


def _enumerate(pp: PolicyProblem) -> dict:
    idx = {"A": 0, "B": 1}
    return {(lo, hi): pp.policy_cost_risk(np.array([idx[lo], idx[hi]])) for lo in "AB" for hi in "AB"}


# =================================================================================================
# 1  the garbling and the program likelihoods
# =================================================================================================

def test_garbling_is_row_stochastic_and_program_likelihoods_equal_the_matrices(world):
    prob, lib, prior, risk = world
    assert np.all(G_MATRIX >= 0) and np.allclose(G_MATRIX.sum(axis=1), 1.0, atol=0, rtol=0)
    assert np.max(np.abs(L_MATRIX @ G_MATRIX - LG_EXACT)) <= 1e-15
    st = structures()
    L0 = build_likelihood(st["original"][0], prior).L
    L1 = build_likelihood(st["garbled"][0], prior).L
    assert np.max(np.abs(L0 - L_MATRIX)) <= 1e-15
    assert np.max(np.abs(L1 - L_MATRIX @ G_MATRIX)) <= 1e-15
    assert np.max(np.abs(L1 - LG_EXACT)) <= 1e-15
    # the candidates behave as declared: A violates only in the bad state, B never; costs 2.90 / 3.56
    assert risk.indicator().tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert lib.costs.tolist() == pytest.approx([COST["A"], COST["B"]], abs=1e-12)


# =================================================================================================
# 2  four deterministic policies: hand = enumeration = solver; discrete structure = Gaussian structure
# =================================================================================================

@pytest.mark.parametrize("case", ["original", "garbled"])
def test_four_deterministic_policies_by_hand_enumeration_and_solver(world, case):
    prob, lib, prior, risk = world
    st = structures()[case][0]
    pp = PolicyProblem.build(prior, build_likelihood(st, prior), risk, ALPHA)
    enum = _enumerate(pp)
    for pol, (hc, hr) in HAND[case]["table"].items():
        cc, cr = enum[pol]
        assert cc == pytest.approx(float(hc), abs=1e-12) and cr == pytest.approx(float(hr), abs=1e-12)
        assert (cr <= ALPHA + pp.risk_tol) == (hr <= F(1, 10))
    best = min(c for c, r in enum.values() if r <= ALPHA + pp.risk_tol)
    for method in ("milp", "enumeration"):
        vt = solve_signal_policy(pp, method=method)
        assert vt.has_solution and vt.expected_cost == pytest.approx(best, abs=1e-12)
    assert solve_no_information(pp).expected_cost == pytest.approx(float(HAND[case]["V0"]), abs=1e-12)
    # the same numbers from the plain discrete structure (matrix likelihood, no Gaussian signal)
    Lm = L_MATRIX if case == "original" else L_MATRIX @ G_MATRIX
    ppm = PolicyProblem.from_arrays(prior.weights, Lm, risk.indicator(), risk.costs, ALPHA)
    assert solve_signal_policy(ppm).expected_cost == pytest.approx(float(HAND[case]["VT"]), abs=1e-12)
    assert solve_signal_policy_randomized(ppm).expected_cost == pytest.approx(float(HAND[case]["VT_randomized"]),
                                                                              abs=1e-9)
    assert solve_no_information_randomized(ppm).expected_cost == pytest.approx(float(HAND[case]["V0_randomized"]),
                                                                               abs=1e-9)


# =================================================================================================
# 3  the values before / after the garbling (regression; nothing truncated, alpha / policies / bins unchanged)
# =================================================================================================

def test_values_before_and_after_the_garbling(results):
    tol = {"V0": 1e-12, "VT": 1e-12, "operational": 1e-12, "randomized_reference": 1e-9, "heuristic_min": 1e-9,
           "benchmark_B": 1e-12, "contrast": 1e-12, "V0_randomized": 1e-9, "VT_randomized": 1e-9}
    for case, r in results.items():
        got = {"V0": r.V0.expected_cost, "VT": r.VT.expected_cost, "operational": r.gross_value,
               "randomized_reference": r.gross_value_randomized_reference,
               "heuristic_min": r.heuristic_min_of_two_policy_values, "benchmark_B": r.randomization_benchmark_value,
               "contrast": r.contrast_vs_matched_uninformative_bins, "V0_randomized": r.V0_randomized.expected_cost,
               "VT_randomized": r.VT_randomized.expected_cost}
        for k, v in got.items():
            assert v == pytest.approx(float(HAND[case][k]), abs=tol[k]), (case, k)
        for k, v in REVIEW_REPORTED[case].items():               # the review printed 6 decimals
            assert got[k] == pytest.approx(v, abs=5e-7), (case, k)
        assert r.alpha == ALPHA and r.n_bins == 2 and r.n_candidates == 2
    o, g = results["original"], results["garbled"]
    assert [o.candidate_ids[k] for k in o.VT.assignment] == ["B", "B"]
    assert [g.candidate_ids[k] for k in g.VT.assignment] == ["B", "A"]
    # garbling: the randomised same-class reference does not grow (theorem) -- here it is unchanged ...
    assert g.gross_value_randomized_reference <= o.gross_value_randomized_reference + 1e-9
    assert g.VT_randomized.expected_cost >= o.VT_randomized.expected_cost - 1e-9
    # ... while the heuristic minimum RISES under a pure garbling: it is not garbling-monotone
    assert g.heuristic_min_of_two_policy_values > o.heuristic_min_of_two_policy_values + 0.018
    assert g.value(HEUR) == pytest.approx(0.132 / 7.0, abs=1e-9) and o.value(HEUR) == pytest.approx(0.0, abs=1e-12)


def test_deprecated_names_still_resolve_to_the_heuristic(results):
    g = results["garbled"]
    with pytest.warns(DeprecationWarning, match="heuristic"):
        old_member = ValueDefinition.EXECUTABLE_INFORMATION_SUPPORTED_VALUE
    assert old_member is HEUR
    with pytest.warns(DeprecationWarning, match="heuristic"):
        assert ValueDefinition("executable_information_supported_value") is HEUR
    with pytest.warns(DeprecationWarning, match="heuristic"):
        assert g.executable_information_supported_value == g.heuristic_min_of_two_policy_values
    with pytest.warns(DeprecationWarning):
        assert g.value("executable_information_supported_value") == g.heuristic_min_of_two_policy_values
    rc = g.randomization_channel
    with pytest.warns(DeprecationWarning, match="matched_uninformative_contrast_ratio"):
        assert rc["randomization_share_of_operational"] == rc["matched_uninformative_contrast_ratio"]
    with pytest.warns(DeprecationWarning):
        assert rc.get("executable_information_supported_value") == rc["heuristic_min_of_two_policy_values"]
    assert "randomization_share_of_operational" not in dict(rc)          # only current keys are stored
    assert "executable_information_supported_value" not in g.values_by_definition()


# =================================================================================================
# 4  what the statuses and ratios mean
# =================================================================================================

def test_matched_benchmark_status_is_not_a_statement_that_no_channel_exists(results):
    g, o = results["garbled"], results["original"]
    rc = g.randomization_channel
    assert rc["status"] == "no_randomization_channel_measured"
    assert "does not show that no randomisation mechanism exists" in rc["reason"]
    assert "do not identify the causal source" in rc["status_scope"]
    assert rc["matched_uninformative_contrast_ratio"] == pytest.approx(0.0, abs=1e-12)     # B / op = 0 / 0.150857
    assert "not an identified share" in rc["matched_uninformative_contrast_ratio_note"]
    # yet the operational saving exceeds the randomised reference by 0.132 (same number as the original's B)
    assert rc["operational_excess_over_randomized_reference"] == pytest.approx(0.132, abs=1e-9)
    assert rc["operational_not_above_randomized_reference"] is False
    assert o.randomization_channel["status"] == "no_operational_saving"
    # money on the operational value is refused, with no recommended substitute
    with pytest.raises(RandomizationChannelError, match="not a substitute basis") as exc:
        per_head_day_value(g, OP, scenario=True)
    assert "complete decision problem" in str(exc.value)


# =================================================================================================
# 5  the heuristic is never a default and is labelled wherever it is used
# =================================================================================================

def test_heuristic_money_conversion_only_when_named_and_labelled(world, results):
    prob, lib, prior, risk = world
    g = results["garbled"]
    cov = BatchCoverage(100.0, 14.0, "synthetic_test_only", "synthetic_test_only", 14.0)
    inv = inventory_coverage_check(lib.Q, lib.ingredient_ids, cov, {"F": 1e9, "C": 1e9})
    with pytest.raises(InvalidProblemError, match="explicit value_definition"):
        break_even_max_cost_per_batch(g, cov, inventory_check=inv, scenario=True)          # no default
    # FIX3_BC (round-3 red team B-1): naming the heuristic is not enough -- no money without a complete decision
    # problem; a complete decision problem refuses it (it is not the value of any single decision problem)
    with pytest.raises(DecisionProblemRequiredError, match="complete decision problem"):
        break_even_max_cost_per_batch(g, cov, inventory_check=inv, value_definition=HEUR, scenario=True)
    with pytest.warns(UserWarning, match="HEURISTIC"):
        be = break_even_max_cost_per_batch(g, cov, inventory_check=inv, value_definition=HEUR, scenario=True, **DIAG)
    # the review's consequence: 1 400 head-days x 0.018857 = 26.4 synthetic units for the *garbled* (worse) signal
    assert float(be) == pytest.approx(1400.0 * 0.132 / 7.0, abs=1e-9) and float(be) == pytest.approx(26.4, abs=1e-9)
    b = be.binding
    assert b.value_role == "heuristic" and b.heuristic is True and b.excluded_from_paper_main_results is True
    assert b.to_dict()["heuristic"] is True and "HEURISTIC" in repr(be)
    with pytest.warns(UserWarning, match="HEURISTIC"):
        nv = net_value_per_batch(g, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=HEUR, scenario=True,
                                 **DIAG)
    assert nv.heuristic is True and nv.excluded_from_paper_main_results is True and nv.value_role == "heuristic"
    assert "not garbling-monotone" in nv.value_interpretation and nv.net_value_per_batch is None
    # the deprecated name still converts, with a deprecation warning, to the same labelled heuristic
    with pytest.warns(DeprecationWarning), pytest.warns(UserWarning, match="HEURISTIC"):
        old = per_head_day_value(g, "executable_information_supported_value", scenario=True, **DIAG)
    assert old.binding.value_definition == HEUR.value and old.binding.heuristic is True
    # the randomised reference remains a separately labelled theoretical number (not a heuristic)
    rr = per_head_day_value(g, RND, scenario=True, **DIAG)
    assert rr.binding.value_role == "theoretical_reference" and rr.binding.heuristic is False
    # FIX3_BC: every money output of this version is excluded from paper main results, with the reason
    assert rr.binding.excluded_from_paper_main_results is True and "decision problem" in rr.binding.exclusion_reason


def test_rankings_that_use_the_minimum_are_labelled_and_prefer_the_garbled_assay(world):
    prob, lib, prior, risk = world
    st = structures()
    opts = [AssayOption(name, "F", ("CP",), s.binning, s.signal_model, 60.0)
            for name, (s, _, _) in (("a_original_L", st["original"]), ("b_garbled_L_G", st["garbled"]))]
    kw = dict(prior_variance_basis=PVB, min_marginal_value=1e-3, min_marginal_value_status="synthetic_test_only")
    with pytest.raises(InvalidProblemError, match="explicit value_definition"):
        strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), **kw)   # no default ranking
    heur = strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), value_definition=HEUR, **kw)
    assert heur.selected == ("b_garbled_L_G",)                     # the heuristic buys the *garbled* assay
    assert heur.details["value_role"] == "heuristic" and heur.details["heuristic"] is True
    assert heur.details["excluded_from_paper_main_results"] is True and "not garbling-monotone" in heur.notes
    op = strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), value_definition=OP, **kw)
    assert op.selected == ("b_garbled_L_G",)                       # the FIX_A screen does not stop a garbling
    assert op.details["heuristic_screen_applied"] is True and op.details["excluded_from_paper_main_results"] is True
    assert "not garbling-safe" in op.notes
    excluded = {x["selection"][-1]: x for x in op.details["randomization_channel_excluded"]}
    assert set(excluded) == {"a_original_L"} and "heuristic_min_of_two_policy_values" in excluded["a_original_L"]
    rnd = strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), value_definition=RND, **kw)
    assert rnd.details["heuristic"] is False and rnd.details["heuristic_screen_applied"] is False
    scores = dict(rnd.ranking)                                     # equal randomised references (garbling theorem)
    assert scores["a_original_L"] == pytest.approx(scores["b_garbled_L_G"], abs=1e-12)
    # FIX3_BC (B-1): no decision-value ranking is an assay priority without a complete decision problem
    for sel in (heur, op, rnd):
        assert sel.details["assay_priority_basis"] == "none_without_complete_decision_problem"
        assert sel.details["excluded_from_paper_main_results"] is True


# =================================================================================================
# 5A  strong interior garbling (FIX3_BC, round-3 red team B-3): not an artefact of the risk tolerance
# =================================================================================================

def _interior_hand(alpha: Fraction) -> dict:
    """Exact hand values of the strong interior garbling L G_int at ``alpha`` (0.098 < alpha < 0.15)."""
    p_low, lik_bad_low = F(776, 1000), F(804, 1000)             # P(low) = (0.804 + 0.748) / 2
    p_high = 1 - p_low
    r_low_a, r_high_a = F(1, 2) * lik_bad_low, F(1, 2) * (1 - lik_bad_low)        # 0.402, 0.098
    vt = p_low * _CB + p_high * _CA if r_high_a <= alpha else _CB                   # (B, A) if feasible
    v0r = _CB - (alpha / F(1, 2)) * (_CB - _CA)
    bins = sorted([(p_low * (_CB - _CA) / r_low_a, p_low * (_CB - _CA), r_low_a),
                   (p_high * (_CB - _CA) / r_high_a, p_high * (_CB - _CA), r_high_a)], reverse=True)
    budget, vtr = alpha, _CB
    for _, save, risk in bins:
        w = min(F(1), budget / risk)
        vtr -= w * save
        budget -= w * risk
    op, rr = _CB - vt, v0r - vtr
    return {"operational": op, "randomized_reference": rr, "heuristic_min": min(op, rr), "risk_B_A": r_high_a}


def test_strong_interior_garbling_keeps_the_counterexample_away_from_the_tolerance(world):
    prob, lib, prior, risk = world
    assert np.max(np.abs(L_MATRIX @ G_INTERIOR - LG_INTERIOR)) <= 1e-15
    assert np.all(G_INTERIOR >= 0) and np.allclose(G_INTERIOR.sum(axis=1), 1.0, rtol=0, atol=0)
    s_int, _, _ = gaussian_signal(0.804, 0.748, "garbled_interior_L_G")
    assert np.max(np.abs(build_likelihood(s_int, prior).L - LG_INTERIOR)) <= 1e-14
    s_org = structures()["original"][0]
    s_bnd = structures()["garbled"][0]
    for alpha in INTERIOR_ALPHAS:
        r_int = compute_information_value(s_int, prior, risk, alpha, prior_variance_basis=PVB)
        r_org = compute_information_value(s_org, prior, risk, alpha, prior_variance_basis=PVB)
        r_bnd = compute_information_value(s_bnd, prior, risk, alpha, prior_variance_basis=PVB)
        h = _interior_hand(F(str(alpha)))
        # the chosen deterministic policy (B, A) has risk 0.098: strictly inside alpha, not decided by risk_tol
        assert [r_int.candidate_ids[k] for k in r_int.VT.assignment] == ["B", "A"]
        assert r_int.VT.ex_ante_risk == pytest.approx(0.098, abs=1e-12) and r_int.VT.ex_ante_risk < alpha - 1e-4
        assert r_int.gross_value == pytest.approx(float(h["operational"]), abs=1e-12)
        assert r_int.gross_value_randomized_reference == pytest.approx(float(h["randomized_reference"]), abs=1e-9)
        # the heuristic minimum rises under the pure garbling (0 -> about 0.0184), at every alpha of the sweep
        assert r_org.heuristic_min_of_two_policy_values == pytest.approx(0.0, abs=1e-9)
        assert r_int.heuristic_min_of_two_policy_values == pytest.approx(float(h["heuristic_min"]), abs=1e-9)
        assert r_int.heuristic_min_of_two_policy_values > 0.018
        # the review's boundary version only survives at alpha = 0.10 (its (B, A) risk is exactly 0.10)
        if alpha < 0.1:
            assert r_bnd.heuristic_min_of_two_policy_values == pytest.approx(0.0, abs=1e-9)
    r10 = compute_information_value(s_int, prior, risk, 0.10, prior_variance_basis=PVB)
    assert r10.gross_value == pytest.approx(0.14784, abs=1e-12)                        # finding: Δ_op = 0.14784
    assert r10.heuristic_min_of_two_policy_values == pytest.approx(0.018388, abs=5e-7)  # finding: 0.018388


# =================================================================================================
# 6  the two policy classes are shown side by side, never combined
# =================================================================================================

def test_policy_class_comparison_keeps_the_classes_apart(results):
    g = results["garbled"]
    cmp_ = g.policy_class_comparison()
    det, rnd = cmp_["deterministic_policy_class"], cmp_["randomized_same_class_reference"]
    assert det["no_information_policy"]["expected_cost"] == pytest.approx(3.56, abs=1e-12)
    assert det["with_information_policy"]["expected_cost"] == pytest.approx(119.32 / 35.0, abs=1e-12)
    assert det["with_information_policy"]["assignment_candidate_ids"] == ["B", "A"]
    assert det["with_information_policy"]["ex_ante_joint_risk"] == pytest.approx(0.1, abs=1e-12)
    assert det["cost_difference_V0_minus_VT"] == pytest.approx(5.28 / 35.0, abs=1e-12)
    assert rnd["no_information_policy"]["expected_cost"] == pytest.approx(3.428, abs=1e-9)
    assert rnd["no_information_policy"]["mixture"] == pytest.approx([0.2, 0.8], abs=1e-7)
    assert rnd["cost_difference_V0_minus_VT"] == pytest.approx(0.132 / 7.0, abs=1e-9)
    assert det["role"] == "operational" and rnd["role"] == "theoretical_reference"
    assert "no minimum, maximum or difference across the classes" in cmp_["not_combined"]
    assert not any("min_of" in k or "heuristic" in k for k in cmp_ if k != "not_combined")
    rep = g.definition_report()
    assert rep["policy_class_comparison"]["deterministic_policy_class"]["cost_difference_V0_minus_VT"] == \
        det["cost_difference_V0_minus_VT"]
    assert rep["values"][HEUR.value]["role"] == "heuristic"
    assert rep["values"][HEUR.value]["excluded_from_paper_main_results"] is True
    assert rep["values"][OP.value]["excluded_from_paper_main_results"] is False


# =================================================================================================
# 7  no feasible policy with or without information: undefined (None + reason), never 0
# =================================================================================================

def test_undefined_cost_value_is_none_with_a_reason_and_risk_is_reported_without_money():
    """Separate synthetic case (not the counterexample): an upper CP bound of 18 % DM makes B violate in the
    good state, so every candidate has joint risk 1/2 > alpha; the review signal cannot bring the library
    below 0.4.  Both V0 and VT (deterministic and randomised) are infeasible."""
    prob, lib, prior, risk = build_world(cp_max=18.0)
    assert risk.indicator().tolist() == [[1.0, 0.0], [0.0, 1.0]]
    st = structures()["original"][0]
    r = compute_information_value(st, prior, risk, ALPHA, prior_variance_basis=PVB)
    assert not r.V0.has_solution and not r.VT.has_solution
    assert r.gross_value_status == "V0_and_VT_infeasible_within_library"
    for d in ValueDefinition:
        assert r.value(d) is None
    rep = r.definition_report()
    assert rep["values"][OP.value]["status"] == "undefined:V0_and_VT_infeasible_within_library"
    assert rep["values"][HEUR.value]["status"].startswith("undefined:")
    chosen = r.with_value_definition(OP)
    assert chosen.primary_value is None and chosen.primary_value_status.startswith("undefined_under_definition:")
    with pytest.raises(InvalidProblemError, match="undefined"):
        chosen.require_primary_value()
    with pytest.raises(InvalidProblemError, match="undefined"):
        per_head_day_value(r, RND, scenario=True)
    cmp_ = r.policy_class_comparison()
    assert cmp_["deterministic_policy_class"]["cost_difference_V0_minus_VT"] is None
    assert cmp_["deterministic_policy_class"]["cost_difference_status"] == \
        "undefined:V0_and_VT_infeasible_within_library"
    assert cmp_["randomized_same_class_reference"]["cost_difference_V0_minus_VT"] is None
    mr = r.min_attainable_ex_ante_joint_risk
    assert mr["no_information"] == pytest.approx(0.5, abs=1e-12)
    assert mr["with_information"] == pytest.approx(0.5 * (0.1 + 0.7), abs=1e-12)          # min(.15,.05)+min(.35,.45)
    assert mr["reduction_with_information"] == pytest.approx(0.1, abs=1e-12)
    assert mr["no_information_reaches_alpha"] is False and mr["with_information_reaches_alpha"] is False
    assert "never converted to money" in mr["money"] and "not computed here" in mr["whole_decision_space"]
    assert any("undefined (None" in w and "not 0" in w for w in r.warnings)
    assert r.to_dict()["min_attainable_ex_ante_joint_risk"]["no_information"] == pytest.approx(0.5, abs=1e-12)


# =================================================================================================
# 8  the committed JSON record matches this code
# =================================================================================================

def test_committed_record_matches_recomputation():
    path = Path(__file__).resolve().parents[2] / "reports" / "garbling_counterexample.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["is_synthetic"] is True and rec["evidence_level"] == "code_tested"
    now = counterexample_record(with_identity=False)
    for case in ("original", "garbled"):
        for k, v in now["values"][case].items():
            if isinstance(v, float):
                assert rec["values"][case][k] == pytest.approx(v, abs=1e-12), (case, k)
            else:
                assert rec["values"][case][k] == v, (case, k)
    for a_, blk in now["strong_interior_garbling"]["by_alpha"].items():           # FIX3_BC (B-3)
        for k, v in blk.items():
            if isinstance(v, float):
                assert rec["strong_interior_garbling"]["by_alpha"][a_][k] == pytest.approx(v, abs=1e-12), (a_, k)
            elif k != "hand":
                assert rec["strong_interior_garbling"]["by_alpha"][a_][k] == v, (a_, k)


# =================================================================================================
# script mode: write the synthetic record
# =================================================================================================

def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Fraction):
        return float(v)
    return v


def counterexample_record(with_identity: bool = True) -> dict:
    """All numbers of the counterexample (hand and computed); every value is synthetic."""
    import platform
    import warnings

    import scipy

    prob, lib, prior, risk = build_world()
    st = structures()
    values, policies, channels, classes, lik = {}, {}, {}, {}, {}
    for case, (s, sd, edge) in st.items():
        L = build_likelihood(s, prior).L
        ref = L_MATRIX if case == "original" else L_MATRIX @ G_MATRIX
        lik[case] = {"signal_sd": sd, "signal_edge": edge, "program_likelihood": L.tolist(),
                     "matrix_likelihood": ref.tolist(), "program_minus_matrix_max_abs": float(np.max(np.abs(L - ref)))}
        r = compute_information_value(s, prior, risk, ALPHA, prior_variance_basis=PVB)
        pp = PolicyProblem.build(prior, build_likelihood(s, prior), risk, ALPHA)
        enum = _enumerate(pp)
        policies[case] = [{"low_bin_action": lo, "high_bin_action": hi, "hand_expected_cost": float(hc),
                           "hand_ex_ante_risk": float(hr), "hand_feasible": bool(hr <= F(1, 10)),
                           "computed_expected_cost": enum[(lo, hi)][0], "computed_ex_ante_risk": enum[(lo, hi)][1],
                           "computed_feasible": bool(enum[(lo, hi)][1] <= ALPHA + pp.risk_tol)}
                          for (lo, hi), (hc, hr) in HAND[case]["table"].items()]
        values[case] = {"V0_deterministic": r.V0.expected_cost, "VT_deterministic": r.VT.expected_cost,
                        "VT_assignment": [r.candidate_ids[k] for k in r.VT.assignment],
                        "operational_delta_op": r.gross_value,
                        "randomized_reference_delta_R": r.gross_value_randomized_reference,
                        "heuristic_min_of_two_policy_values": r.heuristic_min_of_two_policy_values,
                        "matched_benchmark_B": r.randomization_benchmark_value,
                        "contrast_vs_matched_uninformative_bins": r.contrast_vs_matched_uninformative_bins,
                        "V0_randomized": r.V0_randomized.expected_cost, "VT_randomized": r.VT_randomized.expected_cost,
                        "randomization_channel_status": r.randomization_channel["status"],
                        "error_model_identification": r.error_model_identification}
        channels[case] = dict(r.randomization_channel)
        classes[case] = r.to_dict()["policy_class_comparison"]
    s_int, sd_int, edge_int = gaussian_signal(0.804, 0.748, "garbled_interior_L_G")
    interior = {"G": G_INTERIOR.tolist(), "L_G": LG_INTERIOR.tolist(), "signal_sd": sd_int, "signal_edge": edge_int,
                "note": ("FIX3_BC (round-3 red team B-3): strong interior garbling; the garbled deterministic policy (B, A) "
                         "has ex-ante risk 0.098 < alpha, so the result does not depend on risk_tol"),
                "by_alpha": {}}
    for a_ in INTERIOR_ALPHAS:
        ri = compute_information_value(s_int, prior, risk, a_, prior_variance_basis=PVB)
        ro = compute_information_value(st["original"][0], prior, risk, a_, prior_variance_basis=PVB)
        rb = compute_information_value(st["garbled"][0], prior, risk, a_, prior_variance_basis=PVB)
        interior["by_alpha"][str(a_)] = {
            "interior_operational_delta_op": ri.gross_value,
            "interior_randomized_reference_delta_R": ri.gross_value_randomized_reference,
            "interior_heuristic_min": ri.heuristic_min_of_two_policy_values,
            "interior_VT_assignment": [ri.candidate_ids[k] for k in ri.VT.assignment],
            "interior_VT_ex_ante_risk": ri.VT.ex_ante_risk,
            "original_heuristic_min": ro.heuristic_min_of_two_policy_values,
            "boundary_garbled_heuristic_min": rb.heuristic_min_of_two_policy_values,
            "hand": {k: float(v) for k, v in _interior_hand(F(str(a_))).items()}}
    g = compute_information_value(st["garbled"][0], prior, risk, ALPHA, prior_variance_basis=PVB)
    cov = BatchCoverage(100.0, 14.0, "synthetic_test_only", "synthetic_test_only", 14.0)
    inv = inventory_coverage_check(lib.Q, lib.ingredient_ids, cov, {"F": 1e9, "C": 1e9})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        be = break_even_max_cost_per_batch(g, cov, inventory_check=inv, value_definition=HEUR, scenario=True,
                                           diagnostic_without_decision_problem=True)
    try:
        per_head_day_value(g, OP, scenario=True)
        op_refusal = None
    except RandomizationChannelError as exc:
        op_refusal = str(exc)
    hand = {case: {k: _jsonable(v) for k, v in HAND[case].items() if k != "table"} for case in HAND}
    rec = {
        "title": "R3B garbling counterexample (third review §3.2), rebuilt independently from the report text",
        "evidence_level": "code_tested",
        "evidence_level_note": ("holder-run unit test on a synthetic mathematical example; it reproduces the numbers "
                                "printed in review §3.2 with this repository's code; the review's evidence/ files "
                                "were not available; this is not a third-party (independent) verification"),
        "status": "unit_passed", "value_status": "synthetic_test_only", "is_synthetic": True,
        "scope": ("synthetic mathematical counterexample; no feed, farm, price or biological claim; not a pilot, "
                  "development or official result; must not enter the manuscript"),
        "source": "docs/review_20260925_round3/RuminantSim_第三次复核报告_20260925.md §3.2",
        "unchanged": {"alpha": ALPHA, "risk_tol": 1e-8, "candidates": ["A", "B"],
                      "deterministic_policies_enumerated": 4, "bins": 2,
                      "truncation": "none (negative and near-zero values are reported as computed)"},
        "setup": {"states": {"bad": {"F_CP_fraction_DM": 0.08, "prior": 0.5},
                             "good": {"F_CP_fraction_DM": 0.12, "prior": 0.5}},
                  "constraint": "CP >= 16 % DM (the only probabilistic constraint); planned DM 20 kg/d",
                  "candidates": {"A": {"q_as_fed_kg": [42.5, 3.75], "cost": COST["A"], "violates_in": ["bad"]},
                                 "B": {"q_as_fed_kg": [37.0, 6.50], "cost": COST["B"], "violates_in": []}},
                  "L": L_MATRIX.tolist(), "G": G_MATRIX.tolist(), "L_G_exact": LG_EXACT.tolist(),
                  "L_G_float_minus_exact_max_abs": float(np.max(np.abs(L_MATRIX @ G_MATRIX - LG_EXACT))),
                  "signal_api": "SignalModel(ComponentErrorModel(F:CP, sd)) + SignalBinning(one edge), analytic "
                                "Gaussian likelihood (information.build_likelihood)",
                  "currency": "synthetic units/head/d"},
        "likelihoods": lik,
        "deterministic_policies": policies,
        "hand_values": hand,
        "values": values,
        "review_reported": REVIEW_REPORTED,
        "randomization_channel": channels,
        "policy_class_comparison": classes,
        "strong_interior_garbling": interior,
        "consequences": {
            "heuristic_rises_under_pure_garbling": {
                "original": values["original"]["heuristic_min_of_two_policy_values"],
                "garbled": values["garbled"]["heuristic_min_of_two_policy_values"],
                "randomized_reference_unchanged": [values["original"]["randomized_reference_delta_R"],
                                                   values["garbled"]["randomized_reference_delta_R"]]},
            "garbled_status_is_a_matched_benchmark_statement": channels["garbled"]["reason"],
            "money_if_the_heuristic_is_named_explicitly": {
                "head_days": cov.head_days, "gross_amount_synthetic": float(be),
                "binding": be.binding.to_dict(),
                "note": ("labelled heuristic and excluded from paper main results; not a payment bound; FIX3_BC: "
                         "converted only as an explicit development diagnostic (no complete decision problem)")},
            "operational_money_conversion_refusal": op_refusal},
        "generator": "tests/unit/test_garbling_counterexample.py --write",
    }
    if with_identity:
        root = Path(__file__).resolve().parents[2]
        src = ["src/ration_reliability/information/value.py", "src/ration_reliability/information/economics.py",
               "src/ration_reliability/information/strategies.py", "src/ration_reliability/information/evaluation.py",
               "src/ration_reliability/information/policy.py", "src/ration_reliability/information/likelihood.py",
               "tests/unit/test_garbling_counterexample.py"]
        rec["source_sha256"] = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in src}
        rec["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    return _jsonable(rec)


if __name__ == "__main__":
    import datetime

    rec = counterexample_record()
    rec["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = json.dumps(rec, ensure_ascii=False, indent=2, default=float) + "\n"
    if "--write" in sys.argv:
        out = Path(__file__).resolve().parents[2] / "reports" / "garbling_counterexample.json"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)
