"""Assay-selection strategies at equal budget on a constructed synthetic case.

Case: forage F (large inclusion, moderate CP uncertainty) and a minor ingredient X (huge CP
uncertainty but at most 0.1 kg as-fed/d).  "Largest marginal variance" points at X, while the
decision-relevant uncertainty is F's.  Every number is synthetic; the test checks the mechanics
(leakage guards, joint valuation, budgets, oracle flag), not any nutritional claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine_test_helpers import af_max, conc, dm_offer, ing, problem
from ration_reliability.errors import InvalidProblemError
from ration_reliability.information import (
    AssayBudget,
    AssayOption,
    InformationStructure,
    ObservedComponent,
    PriorStates,
    SignalBinning,
    ValueDefinition,
    compute_information_value,
    compute_risk_table,
    conditioning_from_likelihood,
    evaluate_selection_value,
    generate_conditional_library,
    perfect_partial_likelihood,
    strategy_constraint_sensitivity,
    strategy_decision_value,
    strategy_max_inclusion,
    strategy_max_marginal_variance,
    strategy_none,
    strategy_oracle_reference,
    strategy_random,
)
from ration_reliability.uncertainty import RandomStreams

FCP, XCP = ObservedComponent("F", "CP"), ObservedComponent("X", "CP")
ALPHA = 0.05


@pytest.fixture(scope="module")
def case():
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    X = ing("X", 0.90, {"CP": 0.30})
    C = ing("C", 0.80, {"CP": 0.40})
    prob = problem([F, X, C], ["CP"], [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0), af_max("X", 0.1)],
                   {"F": 0.04, "X": 0.02, "C": 0.32}, problem_id="p8_strategies")
    theta, w = [], []
    for f in (0.08, 0.10, 0.12):
        for x in (0.10, 0.50):
            theta.append([[f], [x], [0.40]])
            w.append(1 / 6)
    prior = PriorStates.from_discrete(np.array(theta), np.tile([0.40, 0.90, 0.80], (6, 1)), w, prob.ingredient_ids,
                                      ["CP"], label="strategies", is_synthetic=True)
    bF = SignalBinning((FCP,), ((0.09, 0.11),), "synthetic edges")
    bX = SignalBinning((XCP,), ((0.30,),), "synthetic edge")
    cond = conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FCP,), bF), prefix="F|") + \
        conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (XCP,), bX), prefix="X|")
    lib = generate_conditional_library(prob, prior, cond)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    opts = [AssayOption("opt_F", "F", ("CP",), bF, cost_per_panel=30.0),
            AssayOption("opt_X", "X", ("CP",), bX, cost_per_panel=10.0)]
    v0 = compute_information_value(InformationStructure("F", "perfect_partial", (FCP,), bF), prior, risk, ALPHA)
    q0 = lib.decisions[v0.V0.diagnostics["chosen_index"]]
    basis = {FCP: "true_batch_state", XCP: "true_batch_state"}
    return prob, prior, lib, risk, opts, q0, basis


#: declared materiality threshold of the decision-value strategy (required since the red-team fix; synthetic)
THR = {"min_marginal_value": 0.0, "min_marginal_value_status": "synthetic_test_only"}


def test_library_is_generated_from_development_states_and_is_structurally_feasible(case):
    prob, prior, lib, risk, *_ = case
    assert lib.size >= 3
    assert all(o["prior_stream_id"] == prior.stream_id for o in lib.origins)
    assert {o["conditioning"] for o in lib.origins} >= {"no_information"}
    assert (lib.Q[:, 1] <= 0.1 + 1e-9).all()                        # structural bound on X respected
    assert lib.is_synthetic and risk.is_synthetic


def test_simple_strategies_rank_as_constructed(case):
    prob, prior, lib, risk, opts, q0, basis = case
    one = AssayBudget(max_panels=1)
    assert strategy_none(opts, one).selected == ()
    assert strategy_max_marginal_variance(opts, prior, one).selected == ("opt_X",)
    assert strategy_max_inclusion(opts, q0, one).selected == ("opt_F",)
    sens = strategy_constraint_sensitivity(opts, prior, q0, prob.compiled, one)
    assert sens.selected == ("opt_F",)
    assert dict(sens.ranking)["opt_F"] > dict(sens.ranking)["opt_X"]
    dv = strategy_decision_value(opts, prior, risk, ALPHA, one, prior_variance_basis=basis,
                                 value_definition=ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS, **THR)
    assert dv.selected == ("opt_F",) and dv.details["value_kind"] == "net_of_randomization"
    assert dict(dv.ranking)["opt_F"] > dict(dv.ranking)["opt_X"] - 1e-12
    for s in (sens, dv):
        assert not s.is_oracle and "true" not in s.information_used.lower()


def test_decision_value_values_sets_jointly_and_may_stop(case):
    prob, prior, lib, risk, opts, q0, basis = case
    two = AssayBudget(max_panels=2)
    for kind in ("deterministic", "net_of_randomization"):
        dv = strategy_decision_value(opts, prior, risk, ALPHA, two, prior_variance_basis=basis, value_kind=kind, **THR)
        hist = dv.details["greedy_history"]
        assert hist[0]["option"] == "opt_F"
        if len(hist) > 1:
            joint = evaluate_selection_value(["opt_F", "opt_X"], opts, prior, risk, ALPHA, prior_variance_basis=basis)
            assert hist[1]["joint_value"] == pytest.approx(joint.value(kind), abs=1e-12)
            assert hist[1]["marginal_gain"] == pytest.approx(joint.value(kind) - hist[0]["joint_value"], abs=1e-12)
    rnd = strategy_decision_value(opts, prior, risk, ALPHA, two, prior_variance_basis=basis,
                                  value_kind="randomized_reference", **THR)
    assert rnd.details["value_kind"] == "randomized_reference"


def test_budgets_are_equal_and_costs_must_be_known(case):
    prob, prior, lib, risk, opts, q0, basis = case
    b = AssayBudget(max_cost=35.0)
    assert strategy_max_marginal_variance(opts, prior, b).selected == ("opt_X",)     # X (10) then F (30) too much
    assert strategy_max_inclusion(opts, q0, b).selected == ("opt_F",)
    no_cost = [AssayOption("opt_F", "F", ("CP",), opts[0].binning)]
    with pytest.raises(InvalidProblemError, match="unknown"):
        strategy_max_inclusion(no_cost, q0, b)
    with pytest.raises(InvalidProblemError):
        AssayBudget()


def test_random_strategy_is_reproducible_from_named_stream(case):
    *_, opts, q0, basis = case
    one = AssayBudget(max_panels=1)
    a = strategy_random(opts, one, RandomStreams(1103), replicate=4)
    b = strategy_random(opts, one, RandomStreams(1103), replicate=4)
    assert a.selected == b.selected and a.details["stream_id"] == "root=1103/assay_strategy_random/4"
    picks = {strategy_random(opts, one, RandomStreams(1103), replicate=r).selected for r in range(20)}
    assert picks == {("opt_F",), ("opt_X",)}


def test_strategies_refuse_non_development_states_and_oracle_is_explicit(case):
    prob, prior, lib, risk, opts, q0, basis = case
    leaked = PriorStates(prior.theta, prior.d, prior.weights, prior.ingredient_ids, prior.nutrient_ids, "test",
                         "root=1103/test", prior.source_fingerprint, True)
    with pytest.raises(InvalidProblemError, match="development"):
        strategy_decision_value(opts, leaked, risk, ALPHA, AssayBudget(max_panels=1), prior_variance_basis=basis,
                                **THR)
    true_theta = np.array([[0.08], [0.50], [0.40]])
    true_d = np.array([0.40, 0.90, 0.80])
    with pytest.raises(InvalidProblemError, match="acknowledge_oracle"):
        strategy_oracle_reference(opts, true_theta, true_d, prior, q0, prob.compiled, AssayBudget(max_panels=1))
    orc = strategy_oracle_reference(opts, true_theta, true_d, prior, q0, prob.compiled, AssayBudget(max_panels=1),
                                    acknowledge_oracle=True)
    assert orc.is_oracle and orc.information_used.startswith("ORACLE")
    assert orc.selected == ("opt_F",)
