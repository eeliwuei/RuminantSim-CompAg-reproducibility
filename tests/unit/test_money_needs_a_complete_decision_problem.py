"""Round-3 red team B-1 (FIX3_BC): no money value without a complete decision problem.  ALL NUMBERS ARE SYNTHETIC.

Finding (``docs/review_20260925_round3/`` red-team round, B-1): after R3B the money conversions still took the
*operational* value by default -- role ``operational``, ``heuristic=False``, not excluded from paper main results, no
warning.  The only guard was ``Δ_op <= Δ_R + tol``, i.e. ``min(Δ_op, Δ_R) = Δ_op``: the minimum of two policy
classes deciding whether money is released.  That guard is not garbling-monotone.  The red team's strong interior
counterexample (rebuilt here from the finding's text; its probe files were not shipped):

* two states, prior 0.2 / 0.8; three candidates with costs 1.1 / 2.2 / 2.7: ``k0`` violates only in state 1, ``k1``
  only in state 0, ``k2`` never; ex-ante joint risk target alpha = 0.30;
* original signal: low-bin probabilities (0.95, 0.40) per state; pure garbling ``G = [[0.85, 0.15], [0.05, 0.95]]``
  (a row-stochastic re-labelling of the observed bin; no state information added) -> (0.81, 0.37);
* original: Δ_op = 0, Δ_R = 0.301167 (``no_operational_saving``); garbled: Δ_op = 0.232800, Δ_R = 0.236294,
  B = 0 (``no_randomization_channel_measured``) -- the garbled operational value *passes* the gate, so the old code
  released 0.2328 / head / d (break-even 0 -> 325.92 at 1 400 head-days) for a worse signal.

Construction (public API): forage F (DM 1.0) with CP 8 % DM in state 0 and 12 % DM in state 1; three concentrates
C, D, P (CP 40 % DM, DM 1.0) that differ only in price; planned DM 20 kg/d; probabilistic rows CP >= 16 % and
CP <= 19.5 % DM.  k0 = (F 13, C 7) -> CP 19.2 % / 21.8 % (violates the upper row in state 1); k1 = (F 17, D 3) ->
12.8 % / 16.2 % (violates the lower row in state 0); k2 = (F 14.8, P 5.2) -> 16.32 % / 19.28 % (never).  Prices are
solved so that the costs are exactly the finding's 1.1 / 2.2 / 2.7.  The signal is ``Z = CP_F + e`` with one bin edge
(:class:`SignalModel` + :class:`SignalBinning`, analytic Gaussian likelihood).

Nothing here is a feed, farm, price or biological result; it must not enter the manuscript.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.stats import norm

from engine_test_helpers import conc, dm_offer, ing, problem
from ration_reliability.datamodel import Provenance, RationDecision, ValueStatus
from ration_reliability.errors import InvalidProblemError
from ration_reliability.information import (
    AssayBudget,
    AssayOption,
    BatchCoverage,
    CandidateLibrary,
    CompleteDecisionProblem,
    ComponentErrorModel,
    DecisionProblemElement,
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
    batch_gross_value,
    break_even_max_cost_per_batch,
    build_likelihood,
    compute_information_value,
    compute_risk_table,
    inventory_coverage_check,
    net_value_per_batch,
    per_head_day_value,
    solve_signal_policy,
    strategy_decision_value,
)

OP = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE
RND = ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE
HEUR = ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES
SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-FIX3-BC-B1")
FCP = ObservedComponent("F", "CP")
PVB = {FCP: "true_batch_state"}
ALPHA = 0.30
PRIOR = (0.2, 0.8)
COSTS = (1.1, 2.2, 2.7)
LOW_ORIGINAL = (0.95, 0.40)                       # P(low bin | state 0), P(low bin | state 1)
G_MATRIX = np.array([[0.85, 0.15], [0.05, 0.95]])
LOW_GARBLED = (0.81, 0.37)
HEAD_DAYS = 1400.0
COSTS_UNKNOWN = {k: None for k in ("sampling", "laboratory", "logistics", "waiting", "reformulation")}
DIAG = {"diagnostic_without_decision_problem": True}

#: numbers printed in the finding (6 decimals)
FINDING = {"original": {"operational": 0.0, "randomized_reference": 0.301167, "status": "no_operational_saving"},
           "garbled": {"operational": 0.232800, "randomized_reference": 0.236294, "benchmark_B": 0.0,
                       "status": "no_randomization_channel_measured", "break_even_1400": 325.92}}


# =================================================================================================
# construction
# =================================================================================================

def _prices(f: float = 0.01) -> dict[str, float]:
    """Prices per kg as fed (DM 1.0) that make the three candidate costs exactly 1.1 / 2.2 / 2.7."""
    return {"F": f, "C": (1.1 - 13 * f) / 7, "D": (2.2 - 17 * f) / 3, "P": (2.7 - 14.8 * f) / 5.2}


def build_world():
    ings = [ing("F", 1.0, {"CP": 0.10}, forage=1.0), ing("C", 1.0, {"CP": 0.40}), ing("D", 1.0, {"CP": 0.40}),
            ing("P", 1.0, {"CP": 0.40})]
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0), conc("cp_max", {"CP": 1.0}, "le", 19.5)]
    prob = problem(ings, ["CP"], cons, _prices(), problem_id="fix3_bc_b1_interior_counterexample")
    ids, dh = prob.ingredient_ids, prob.dm_estimates()
    decs = [RationDecision(ids, np.array(q, dtype=float), dh, f"synthetic_{k}")
            for k, q in (("k0", [13.0, 7.0, 0.0, 0.0]), ("k1", [17.0, 0.0, 3.0, 0.0]), ("k2", [14.8, 0.0, 0.0, 5.2]))]
    lib = CandidateLibrary.from_decisions(prob, decs, ["k0", "k1", "k2"])
    theta = np.array([[[0.08], [0.40], [0.40], [0.40]], [[0.12], [0.40], [0.40], [0.40]]])
    prior = PriorStates.from_discrete(theta, np.ones((2, 4)), list(PRIOR), ids, ["CP"], label="fix3_bc_b1",
                                      is_synthetic=True)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    return prob, lib, prior, risk


def gaussian_signal(p_low_s0: float, p_low_s1: float, sid: str) -> InformationStructure:
    """``Z = CP_F + e``: P(low | CP 0.08) = p_low_s0, P(low | CP 0.12) = p_low_s1 (one edge)."""
    sd = 0.04 / (norm.ppf(p_low_s0) - norm.ppf(p_low_s1))
    edge = 0.08 + sd * norm.ppf(p_low_s0)
    sm = SignalModel(sid, (ComponentErrorModel(FCP, sd, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True),),
                     SamplingProtocol(), is_synthetic=True)
    return InformationStructure(sid, "sample", binning=SignalBinning((FCP,), ((edge,),), "synthetic edge (FIX3_BC)"),
                                signal_model=sm)


def structures() -> dict[str, InformationStructure]:
    return {"original": gaussian_signal(*LOW_ORIGINAL, "original_L"),
            "garbled": gaussian_signal(*LOW_GARBLED, "garbled_L_G")}


def complete_problem(convention: str = "deterministic_bin_to_ration_map_no_free_randomisation", *,
                     alpha: float = ALPHA, risk_timing: str = "ex_ante_joint_risk") -> CompleteDecisionProblem:
    """A synthetic complete decision problem (every element synthetic_test_only)."""
    def el(text: str, choice=None):
        return DecisionProblemElement(text, "synthetic_test_only", choice=choice)
    return CompleteDecisionProblem(
        problem_id=f"SYN-DP-{convention}-{alpha}-{risk_timing}",
        information_processing=el("synthetic: what the decision maker may do for free with the result", convention),
        allowed_policies=el("synthetic: library {k0, k1, k2}, t0 action set, one ration per bin"),
        baseline=el("synthetic: the no-information optimum of the same class, alpha, risk event and prices"),
        risk_timing=el("synthetic: ex-ante joint risk; one assay covers 14 days", risk_timing),
        assay_cost=el("synthetic: sampling, laboratory, logistics, waiting, reformulation unknown"),
        alpha=alpha, declared_in="tests/unit/test_money_needs_a_complete_decision_problem.py (synthetic)")


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture(scope="module")
def results(world):
    prob, lib, prior, risk = world
    return {k: compute_information_value(s, prior, risk, ALPHA, prior_variance_basis=PVB)
            for k, s in structures().items()}


@pytest.fixture(scope="module")
def cov_inv(world):
    prob, lib, prior, risk = world
    cov = BatchCoverage(100.0, 14.0, "synthetic_test_only", "synthetic_test_only", 14.0)
    return cov, inventory_coverage_check(lib.Q, lib.ingredient_ids, cov, {i: 1e9 for i in lib.ingredient_ids})


# =================================================================================================
# 1  the world and the signals are the finding's
# =================================================================================================

def test_world_matches_the_finding(world):
    prob, lib, prior, risk = world
    assert lib.costs.tolist() == pytest.approx(list(COSTS), abs=1e-12)
    # rows: states 0 / 1; columns: k0, k1, k2
    assert risk.indicator().tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert prior.weights.tolist() == pytest.approx(list(PRIOR), abs=1e-15)
    L0 = build_likelihood(structures()["original"], prior).L
    L1 = build_likelihood(structures()["garbled"], prior).L
    Lm = np.array([[LOW_ORIGINAL[0], 1 - LOW_ORIGINAL[0]], [LOW_ORIGINAL[1], 1 - LOW_ORIGINAL[1]]])
    assert np.all(G_MATRIX >= 0) and np.allclose(G_MATRIX.sum(axis=1), 1.0, rtol=0, atol=0)
    assert np.max(np.abs(L0 - Lm)) <= 1e-14
    assert np.max(np.abs(L1 - Lm @ G_MATRIX)) <= 1e-14          # a pure garbling of the original signal
    assert np.max(np.abs(L1[:, 0] - np.array(LOW_GARBLED))) <= 1e-14


def _hand_deterministic(p_low: tuple[float, float]) -> tuple[Fraction, Fraction]:
    """(V0, VT) of the deterministic class by exhaustive hand enumeration (exact fractions)."""
    pri = (Fraction(1, 5), Fraction(4, 5))
    lo = tuple(Fraction(str(x)) for x in p_low)
    cost = (Fraction(11, 10), Fraction(22, 10), Fraction(27, 10))
    viol = ((0, 1, 0), (1, 0, 0))                                     # [state][candidate]
    a = Fraction(3, 10)
    joint = [[pri[s] * (lo[s] if z == 0 else 1 - lo[s]) for z in (0, 1)] for s in (0, 1)]   # P(s, z)
    pz = [joint[0][z] + joint[1][z] for z in (0, 1)]
    v0 = min(cost[k] for k in range(3) if sum(pri[s] * viol[s][k] for s in (0, 1)) <= a)
    best = None
    for k_lo in range(3):
        for k_hi in range(3):
            ks = (k_lo, k_hi)
            c = sum(pz[z] * cost[ks[z]] for z in (0, 1))
            r = sum(joint[s][z] * viol[s][ks[z]] for s in (0, 1) for z in (0, 1))
            if r <= a and (best is None or c < best):
                best = c
    return v0, best


def _independent_randomised(L: np.ndarray, risk_ind: np.ndarray) -> tuple[float, float]:
    """(V0_R, VT_R) by an independent scipy LP (not the engine's formulation)."""
    pri = np.array(PRIOR)
    c = np.array(COSTS)
    r_k = pri @ risk_ind                                                   # [K]
    v0 = linprog(c, A_ub=[r_k], b_ub=[ALPHA], A_eq=[np.ones(3)], b_eq=[1.0], bounds=[(0, 1)] * 3, method="highs")
    pz = pri @ L                                                           # [Z]
    R = np.einsum("s,sz,sk->zk", pri, L, risk_ind)                         # [Z, K]
    obj = np.concatenate([pz[z] * c for z in range(2)])
    Aeq = np.zeros((2, 6))
    Aeq[0, :3] = 1.0
    Aeq[1, 3:] = 1.0
    vt = linprog(obj, A_ub=[R.reshape(-1)], b_ub=[ALPHA], A_eq=Aeq, b_eq=[1.0, 1.0], bounds=[(0, 1)] * 6,
                 method="highs")
    assert v0.status == 0 and vt.status == 0
    return float(v0.fun), float(vt.fun)


@pytest.mark.parametrize("case", ["original", "garbled"])
def test_values_reproduce_the_finding_by_hand_and_by_an_independent_lp(world, results, case):
    prob, lib, prior, risk = world
    r = results[case]
    p_low = LOW_ORIGINAL if case == "original" else LOW_GARBLED
    v0, vt = _hand_deterministic(p_low)
    assert r.V0.expected_cost == pytest.approx(float(v0), abs=1e-12)
    assert r.VT.expected_cost == pytest.approx(float(vt), abs=1e-12)
    L = build_likelihood(structures()[case], prior).L
    v0r, vtr = _independent_randomised(L, risk.indicator())
    assert r.gross_value_randomized_reference == pytest.approx(v0r - vtr, abs=1e-9)
    for k, v in FINDING[case].items():
        if k == "operational":
            assert r.gross_value == pytest.approx(v, abs=5e-7)
        elif k == "randomized_reference":
            assert r.gross_value_randomized_reference == pytest.approx(v, abs=5e-7)
        elif k == "benchmark_B":
            assert r.randomization_benchmark_value == pytest.approx(v, abs=1e-12)
        elif k == "status":
            assert r.randomization_channel["status"] == v
    # the same deterministic value from the plain matrix structure (no Gaussian signal)
    ppm = PolicyProblem.from_arrays(prior.weights, L, risk.indicator(), risk.costs, ALPHA)
    assert solve_signal_policy(ppm).expected_cost == pytest.approx(float(vt), abs=1e-12)


def test_the_garbling_lowers_the_randomised_reference_but_raises_the_operational_value_through_the_old_gate(results):
    o, g = results["original"], results["garbled"]
    # Blackwell within the finite model: the randomised same-class reference does not grow under a garbling ...
    assert g.gross_value_randomized_reference < o.gross_value_randomized_reference
    # ... while the operational value rises from 0 and passes the old gate Δ_op <= Δ_R + tol (= min(Δ_op, Δ_R) = Δ_op)
    assert o.gross_value == pytest.approx(0.0, abs=1e-12) and g.gross_value > 0.23
    assert g.randomization_channel["operational_not_above_randomized_reference"] is True
    assert g.heuristic_min_of_two_policy_values == pytest.approx(g.gross_value, abs=1e-12)
    assert [g.candidate_ids[k] for k in g.VT.assignment] == ["k0", "k2"]


# =================================================================================================
# 2  money: refused without a complete decision problem; labelled when asked as a diagnostic
# =================================================================================================

@pytest.mark.parametrize("definition", [OP, RND])
def test_every_money_interface_refuses_without_a_decision_problem(results, cov_inv, definition):
    g = results["garbled"]
    cov, inv = cov_inv
    with pytest.raises(DecisionProblemRequiredError, match="complete decision problem"):
        per_head_day_value(g, definition, scenario=True)
    with pytest.raises(DecisionProblemRequiredError):
        batch_gross_value(g, cov, inventory_check=inv, value_definition=definition, scenario=True)
    with pytest.raises(DecisionProblemRequiredError):
        break_even_max_cost_per_batch(g, cov, inventory_check=inv, value_definition=definition, scenario=True)
    with pytest.raises(DecisionProblemRequiredError):
        net_value_per_batch(g, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=definition, scenario=True)
    # a caller-declared float is refused too
    with pytest.raises(DecisionProblemRequiredError):
        batch_gross_value(0.2328, cov, inventory_check=inv, value_definition=definition,
                          unbound_value_declaration="copied from a report")
    # the refusal comes after the more specific ones (a bare float without declaration keeps its FIX_A error)
    with pytest.raises(InvalidProblemError, match="bare float"):
        batch_gross_value(0.2328, cov, inventory_check=inv, value_definition=definition)
    with pytest.raises(InvalidProblemError, match="both"):
        per_head_day_value(g, definition, scenario=True, decision_problem=complete_problem(), **DIAG)


def test_the_rising_released_amount_is_now_labelled_as_a_diagnostic(results, cov_inv):
    """Regression of the finding: 0 -> 0.2328 per head per day (break-even 0 -> 325.92) under a pure garbling."""
    o, g = results["original"], results["garbled"]
    cov, inv = cov_inv
    cov14 = BatchCoverage(100.0, 14.0, "synthetic_test_only", "synthetic_test_only", 14.0)
    assert cov14.head_days == HEAD_DAYS
    with pytest.warns(UserWarning, match="DEVELOPMENT DIAGNOSTIC"):
        vo = per_head_day_value(o, OP, scenario=True, **DIAG)
    with pytest.warns(UserWarning, match="DEVELOPMENT DIAGNOSTIC"):
        vg = per_head_day_value(g, OP, scenario=True, **DIAG)
    assert float(vo) == pytest.approx(0.0, abs=1e-12) and float(vg) == pytest.approx(0.2328, abs=5e-7)
    for v in (vo, vg):
        b = v.binding
        assert b.value_role == "operational" and b.heuristic is False
        assert b.excluded_from_paper_main_results is True
        assert b.money_basis == "development_diagnostic_without_decision_problem" and b.decision_problem_id is None
        assert b.garbling_monotone is False and "not a payment bound" in b.exclusion_reason
    with pytest.warns(UserWarning):
        be = break_even_max_cost_per_batch(g, cov, inventory_check=inv, value_definition=OP, scenario=True, **DIAG)
    assert float(be) == pytest.approx(FINDING["garbled"]["break_even_1400"], abs=1e-6)
    assert be.binding.excluded_from_paper_main_results is True and "excluded from paper main results" in repr(be)
    with pytest.warns(UserWarning):
        nv = net_value_per_batch(g, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=OP, scenario=True,
                                 **DIAG)
    assert nv.excluded_from_paper_main_results is True and nv.money_basis.startswith("development_diagnostic")
    assert "MONEY BASIS: development diagnostic" in nv.value_interpretation
    assert "EXCLUDED FROM PAPER MAIN RESULTS" in nv.value_interpretation
    # the old FIX_A refusal stays on this path (refusal-only): an operational value above Δ_R is still refused
    with pytest.raises(RandomizationChannelError):
        per_head_day_value(dataclass_with_operational_above_reference(g), OP, scenario=True, **DIAG)


def dataclass_with_operational_above_reference(res):
    """Same result with a larger operational value (Δ_op > Δ_R): only to show the diagnostic path's refusal."""
    import dataclasses
    return dataclasses.replace(res, gross_value=float(res.gross_value_randomized_reference) + 0.1)


# =================================================================================================
# 3  with a complete decision problem: the value of its one policy class, nothing of the other class gates it
# =================================================================================================

def test_decision_problem_fixes_one_policy_class_and_labels_the_value(results, cov_inv):
    o, g = results["original"], results["garbled"]
    cov, inv = cov_inv
    det = complete_problem()
    assert det.value_definition == OP.value and det.is_synthetic and det.is_scenario
    with pytest.warns(UserWarning, match="SCENARIO decision problem"):
        v = per_head_day_value(g, OP, scenario=True, decision_problem=det)
    b = v.binding
    assert float(v) == pytest.approx(0.2328, abs=5e-7)
    assert b.money_basis == "complete_decision_problem" and b.decision_problem_id == det.problem_id
    assert b.decision_problem_fingerprint == det.fingerprint() and b.decision_problem_is_scenario is True
    assert b.garbling_monotone is False                      # the deterministic class has no such guarantee (VSD §2)
    assert b.excluded_from_paper_main_results is True and "exploratory appendix" in b.exclusion_reason
    # a bound PerHeadDayValue carries its labels (the warning was emitted when it was created)
    nv = net_value_per_batch(v, cov, COSTS_UNKNOWN, inventory_check=inv, value_definition=OP, decision_problem=det)
    assert "not garbling-monotone, not a pure information value" in nv.value_interpretation
    assert nv.decision_problem_id == det.problem_id
    # the problem's class decides: its money value is Δ_op; the randomised reference and the heuristic are refused
    with pytest.raises(InvalidProblemError, match="not randomized_same_class_information_reference"):
        per_head_day_value(g, RND, scenario=True, decision_problem=det)
    with pytest.raises(InvalidProblemError, match="not the value of any single decision problem"):
        per_head_day_value(g, HEUR, scenario=True, decision_problem=det)
    # a value bound under one basis is not converted under another
    with pytest.raises(InvalidProblemError, match="another basis"):
        batch_gross_value(v, cov, inventory_check=inv, value_definition=OP, **DIAG)
    # free randomisation: the money value is the randomised same-class value (garbling-monotone in the finite model)
    rnd = complete_problem("free_randomisation_of_the_ration_choice")
    with pytest.warns(UserWarning):
        vr_o = per_head_day_value(o, RND, scenario=True, decision_problem=rnd)
    with pytest.warns(UserWarning):
        vr_g = per_head_day_value(g, RND, scenario=True, decision_problem=rnd)
    assert float(vr_g) < float(vr_o) and vr_g.binding.garbling_monotone is True
    with pytest.raises(InvalidProblemError, match="not operational_deterministic_cost_difference"):
        per_head_day_value(g, OP, scenario=True, decision_problem=rnd)
    # the problem must be the result's: alpha and the (ex-ante) risk timing
    with pytest.raises(InvalidProblemError, match="alpha"):
        per_head_day_value(g, OP, scenario=True, decision_problem=complete_problem(alpha=0.10))
    with pytest.raises(InvalidProblemError, match="ex-ante joint risk only"):
        per_head_day_value(g, OP, scenario=True,
                           decision_problem=complete_problem(risk_timing="per_outcome_conditional_risk"))


def test_decision_problem_path_does_not_use_the_cross_class_gate_but_keeps_the_refusal(results):
    """With a complete deterministic problem an operational value above Δ_R is that problem's answer (labelled), while a
    saving that is entirely a randomisation channel is still never bought (refusal-only)."""
    import dataclasses
    g = results["garbled"]
    det = complete_problem()
    above = dataclasses.replace(g, gross_value=float(g.gross_value_randomized_reference) + 0.1)
    assert above.randomization_channel["operational_not_above_randomized_reference"] is False
    with pytest.warns(UserWarning):
        v = per_head_day_value(above, OP, scenario=True, decision_problem=det)
    assert float(v) == pytest.approx(float(g.gross_value_randomized_reference) + 0.1, abs=1e-12)
    only = dataclasses.replace(g, gross_value_randomized_reference=0.0)         # Δ_R = 0 -> randomization_only
    assert only.randomization_channel["status"] == "randomization_only"
    with pytest.raises(RandomizationChannelError, match="randomisation channel only"):
        per_head_day_value(only, OP, scenario=True, decision_problem=det)


def test_decision_problem_validation():
    with pytest.raises(InvalidProblemError, match="source"):
        DecisionProblemElement("x", "sourced")
    with pytest.raises(InvalidProblemError, match="status"):
        DecisionProblemElement("x", "guess")
    with pytest.raises(InvalidProblemError, match="statement"):
        DecisionProblemElement("  ", "synthetic_test_only")
    with pytest.raises(InvalidProblemError, match="information_processing.choice"):
        complete_problem("mix_as_convenient")
    with pytest.raises(InvalidProblemError, match="risk_timing.choice"):
        complete_problem(risk_timing="whenever")
    with pytest.raises(InvalidProblemError, match="alpha"):
        complete_problem(alpha=0.0)
    ok = complete_problem()
    with pytest.raises(InvalidProblemError, match="DecisionProblemElement"):
        CompleteDecisionProblem(ok.problem_id, ok.information_processing, "free text", ok.baseline, ok.risk_timing,
                                ok.assay_cost, ok.alpha, ok.declared_in)
    d = ok.to_dict()
    assert d["value_definition"] == OP.value and set(CompleteDecisionProblem.ELEMENTS) <= set(d)


# =================================================================================================
# 4  rankings: without a decision problem never an assay priority; with one, no cross-class screen
# =================================================================================================

def test_rankings_without_and_with_a_decision_problem(world):
    prob, lib, prior, risk = world
    st = structures()
    opts = [AssayOption(name, "F", ("CP",), s.binning, s.signal_model, 60.0)
            for name, s in (("a_original_L", st["original"]), ("b_garbled_L_G", st["garbled"]))]
    kw = dict(prior_variance_basis=PVB, min_marginal_value=1e-3, min_marginal_value_status="synthetic_test_only")
    op = strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), value_definition=OP, **kw)
    # the finding: the operational ranking picks the garbled assay and drops the original one
    assert op.selected == ("b_garbled_L_G",)
    assert op.details["assay_priority_basis"] == "none_without_complete_decision_problem"
    assert op.details["excluded_from_paper_main_results"] is True and op.details["heuristic_screen_applied"] is True
    rnd = strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), value_definition=RND, **kw)
    assert rnd.selected == ("a_original_L",)                          # Blackwell: the original is worth more here
    assert rnd.details["excluded_from_paper_main_results"] is True    # FIX3_BC: no decision problem -> no priority
    det = complete_problem()
    opd = strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), value_definition=OP,
                                  decision_problem=det, **kw)
    assert opd.details["assay_priority_basis"] == "complete_decision_problem"
    assert opd.details["heuristic_screen_applied"] is False            # no min of two policy classes
    assert "refusal-only" in opd.details["randomization_channel_rule"]
    assert opd.details["decision_problem"]["problem_id"] == det.problem_id
    assert opd.selected == ("b_garbled_L_G",) and "not garbling-monotone" in opd.notes
    assert opd.details["excluded_from_paper_main_results"] is True     # synthetic problem; RQ3 appendix
    with pytest.raises(InvalidProblemError, match="not its ranking"):
        strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=1), value_definition=RND,
                                decision_problem=det, **kw)
    with pytest.raises(InvalidProblemError, match="alpha"):
        strategy_decision_value(opts, prior, risk, 0.25, AssayBudget(max_panels=1), value_definition=OP,
                                decision_problem=det, **kw)
