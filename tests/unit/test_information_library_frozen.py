"""Candidate library, risk table, frozen policies and approximation checks (synthetic continuous prior).

The development prior is a synthetic independent-normal model sampled on the ``opt`` stream; the
frozen policies are evaluated on the ``test`` stream with a dedicated noise stream.  These runs are
software self-checks (``is_synthetic``), not pilot or official results.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine_test_helpers import conc, dm_offer, ing, problem
from ration_reliability.datamodel import Provenance, RationDecision, ValueStatus
from ration_reliability.errors import InvalidProblemError, LeakageError
from ration_reliability.evaluation import evaluate
from ration_reliability.information import (
    CandidateLibrary,
    ComponentErrorModel,
    InformationStructure,
    ObservedComponent,
    PriorStates,
    SignalBinning,
    SignalModel,
    ValueDefinition,
    adjustment_mask,
    calibrate_alpha_train,
    check_bin_refinement,
    check_library_enlargement,
    compute_information_value,
    compute_risk_table,
    conditioning_from_likelihood,
    double_count_guard,
    evaluate_frozen_policy,
    freeze_policy,
    generate_conditional_library,
    perfect_partial_likelihood,
    quantile_edges,
    signal_bins_for_states,
)
from ration_reliability.uncertainty import DrawSet, IndependentNormalModel, RandomStreams

SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-P8-TEST")
FCP, FDM = ObservedComponent("F", "CP"), ObservedComponent("F", "DM")
ALPHA = 0.05
OP = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE   # explicit value definition (second review R1)


@pytest.fixture(scope="module")
def case():
    F = ing("F", 0.35, {"CP": 0.10, "NDF": 0.45}, forage=1.0)
    C = ing("C", 0.88, {"CP": 0.40, "NDF": 0.20})
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            conc("fndf_min", {"G:forage:NDF": 1.0}, "ge", 18.0)]
    prob = problem([F, C], ["CP", "NDF"], cons, {"F": 0.035, "C": 0.30}, problem_id="p8_continuous")
    model = IndependentNormalModel("syn_p8", prob.ingredient_ids, prob.nutrient_ids,
                                   np.array([[0.10, 0.45], [0.40, 0.20]]), np.array([[0.012, 0.03], [0.0, 0.0]]),
                                   np.array([0.35, 0.88]), np.array([0.02, 0.0]), is_synthetic=True)
    streams = RandomStreams(1103)
    prior = PriorStates.from_drawset(model.draw(streams, "opt", 300))
    vcp = prior.component_values((FCP,))
    vdm = prior.component_values((FDM,))
    bcp = SignalBinning((FCP,), (quantile_edges(vcp, 4),), "quantiles of root=1103/opt")
    bdm = SignalBinning((FDM,), (quantile_edges(vdm, 3),), "quantiles of root=1103/opt")
    cond = conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FCP,), bcp), prefix="cp|") + \
        conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FDM,), bdm), prefix="dm|")
    lib = generate_conditional_library(prob, prior, cond)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    return prob, model, streams, prior, bcp, bdm, lib, risk


def _sample(comp, binning, sd):
    em = ComponentErrorModel(comp, sd, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True)
    sm = SignalModel(f"{comp.label()}~{sd:g}", (em,), is_synthetic=True)
    return InformationStructure(f"sample:{comp.label()}", "sample", binning=binning, signal_model=sm), sm


def test_library_candidates_use_design_dm_and_pass_structural_checks(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    dm_cands = [k for k, o in enumerate(lib.origins) if o["dm_observed"] == ["F"]]
    assert dm_cands, "DM-conditioned candidates expected"
    d_hats = {round(float(lib.decisions[k].d_hat[0]), 12) for k in dm_cands}
    assert len(d_hats) == bdm.n_bins and all(abs(d - 0.35) < 0.05 for d in d_hats)
    for k in (0, dm_cands[0]):
        dec = lib.decisions[k]
        assert abs(float(dec.x_planned_dm.sum()) - 20.0) < 1e-6      # planned DM with its own design d_hat
    assert all("status" in g for g in lib.failures)                   # failures keep their status
    with pytest.raises(InvalidProblemError, match="structurally infeasible"):
        CandidateLibrary.from_decisions(prob, [RationDecision(prob.ingredient_ids, np.array([10.0, 1.0]),
                                                              prob.dm_estimates(), "bad")])


def test_risk_table_equals_public_evaluator(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    for k in (0, lib.size // 2, lib.size - 1):
        ev = evaluate(lib.decisions[k], prior.theta, prior.d, prob.compiled, prices=prob.price_vector())
        assert np.array_equal(ev.joint_violation, risk.violated[:, k])
        assert ev.cost == pytest.approx(risk.costs[k])


def test_values_on_continuous_prior_are_consistent(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    part = compute_information_value(InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp), prior, risk, ALPHA)
    st, sm = _sample(FCP, bcp, 0.006)
    samp = compute_information_value(st, prior, risk, ALPHA,
                                     double_count_report=double_count_guard(sm, {FCP: "true_batch_state"}))
    full = compute_information_value(InformationStructure("full", "perfect_full"), prior, risk, ALPHA)
    for r in (part, samp):
        assert r.gross_value_status == "ok" and r.gross_value >= -1e-9 and r.is_synthetic
        assert r.is_information_value and r.action_space["action_space"] == "t0_fixed"
        assert r.V0.expected_cost == pytest.approx(part.V0.expected_cost)      # same V0 for every structure
    # Full perfect information also observes F's DM.  Under the decision-time DM rule (SH-DM-PLAN:
    # the observed d_i replaces d_hat_i) the no-information rations are not legal actions any more and
    # the library has no candidate designed for most single-state bins: reported, not forced
    # (red-team fix C05/F02).  The partition-refinement theorem is checked with known DM below.
    assert full.action_space["action_space"] == "information_dependent"
    assert not full.is_information_value and full.gross_value is None
    assert full.gross_value_status.startswith("action_spaces_differ:")
    assert full.VT.status == "proven_infeasible" and full.action_space["n_active_bins_without_candidate"] > 0
    assert full.V0.expected_cost == pytest.approx(part.V0.expected_cost)
    assert part.occupancy["min_effective_states_active"] >= 70                 # ~75 states per bin
    with pytest.raises(InvalidProblemError, match="double counting"):
        compute_information_value(st, prior, risk, ALPHA,
                                  double_count_report=double_count_guard(sm, {FCP: "observed_incl_sampling_and_lab"}))
    with pytest.raises(InvalidProblemError, match="double_count_report"):
        compute_information_value(st, prior, risk, ALPHA)


def test_partition_refinement_with_known_dm():
    """Same problem with deterministic DM (the DM-known controlled submodel, T2): no structure changes
    the action space, so EVPPI(CP bins) <= EVPI holds as a theorem within the library."""
    F = ing("F", 0.35, {"CP": 0.10, "NDF": 0.45}, forage=1.0)
    C = ing("C", 0.88, {"CP": 0.40, "NDF": 0.20})
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            conc("fndf_min", {"G:forage:NDF": 1.0}, "ge", 18.0)]
    prob = problem([F, C], ["CP", "NDF"], cons, {"F": 0.035, "C": 0.30}, problem_id="p8_known_dm")
    model = IndependentNormalModel("syn_p8_known_dm", prob.ingredient_ids, prob.nutrient_ids,
                                   np.array([[0.10, 0.45], [0.40, 0.20]]), np.array([[0.012, 0.03], [0.0, 0.0]]),
                                   np.array([0.35, 0.88]), np.array([0.0, 0.0]), is_synthetic=True)
    prior = PriorStates.from_drawset(model.draw(RandomStreams(1103), "opt", 300))
    bcp = SignalBinning((FCP,), (quantile_edges(prior.component_values((FCP,)), 4),), "quantiles of root=1103/opt")
    lib = generate_conditional_library(
        prob, prior, conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FCP,), bcp), prefix="cp|"))
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    part = compute_information_value(InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp), prior, risk, ALPHA)
    full = compute_information_value(InformationStructure("full", "perfect_full"), prior, risk, ALPHA)
    for r in (part, full):
        assert r.gross_value_status == "ok" and r.is_information_value and r.gross_value >= -1e-9
        assert r.action_space["action_space"] == "t0_fixed"
    assert part.gross_value <= full.gross_value + 1e-9                         # partition refinement


def test_frozen_perfect_binned_policy_reproduces_development_values_in_sample(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st = InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp)
    res = compute_information_value(st, prior, risk, ALPHA)
    pol = freeze_policy(res, st, lib, value_definition=OP)
    dev_draws = model.draw(streams, "opt", 300)                     # identical states (same stream, same size)
    ev = evaluate_frozen_policy(pol, dev_draws, prob.compiled, streams, prices=prob.price_vector())
    assert ev.in_sample
    assert ev.policy_mean_cost == pytest.approx(res.VT.expected_cost, abs=1e-12)
    assert ev.policy_joint_violation["rate_used"] == pytest.approx(res.VT.ex_ante_risk, abs=1e-12)
    assert ev.constant_mean_cost == pytest.approx(res.V0.expected_cost, abs=1e-12)
    assert ev.realised_cost_saving_per_head_day == pytest.approx(res.gross_value, abs=1e-12)


def test_frozen_sample_policy_on_independent_test_stream(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st, sm = _sample(FCP, bcp, 0.006)
    res = compute_information_value(st, prior, risk, ALPHA,
                                    double_count_report=double_count_guard(sm, {FCP: "true_batch_state"}))
    pol = freeze_policy(res, st, lib, value_definition=OP)
    test = model.draw(streams, "test", 4000)
    ev = evaluate_frozen_policy(pol, test, prob.compiled, streams, prices=prob.price_vector())
    assert not ev.in_sample and ev.n_states == 4000 and ev.is_synthetic
    assert ev.draw_stream_id == "root=1103/test" and ev.noise_stream_id == "root=1103/assay_signal_eval"
    assert ev.policy_joint_violation["interval_type"] == "MC_only_fixed_distribution"
    assert sum(ev.action_counts.values()) == 4000 and ev.bin_counts.sum() == 4000
    again = evaluate_frozen_policy(pol, test, prob.compiled, streams, prices=prob.price_vector())
    assert again.policy_mean_cost == ev.policy_mean_cost                        # reproducible streams
    with pytest.raises(InvalidProblemError):
        evaluate_frozen_policy(pol, test, prob.compiled, streams, noise_stream="test")


def test_frozen_action_ignores_unobserved_components(case):
    """Leakage check: changing hidden components of the test state cannot change the chosen ration."""
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st = InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp)
    pol = freeze_policy(compute_information_value(st, prior, risk, ALPHA), st, lib, value_definition=OP)
    test = model.draw(streams, "test", 500)
    th = np.array(test.theta)
    th[:, 0, 1] += 0.2                                               # F NDF (not observed)
    dd = np.array(test.d)
    dd[:, 0] *= 0.8                                                  # F DM (not observed)
    shifted = DrawSet(th, dd, test.stream, test.stream_id, test.model_id, test.model_fingerprint,
                      test.ingredient_ids, test.nutrient_ids, test.is_synthetic)
    z1 = signal_bins_for_states(pol, test, streams)
    z2 = signal_bins_for_states(pol, shifted, streams)
    assert np.array_equal(pol.assignment[z1], pol.assignment[z2])


def test_only_binned_structures_can_be_frozen(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    full_st = InformationStructure("full", "perfect_full")
    full = compute_information_value(full_st, prior, risk, ALPHA)
    with pytest.raises(InvalidProblemError, match="no frozen policy"):
        freeze_policy(full, full_st, lib, value_definition=OP)
    st = InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp)
    res = compute_information_value(st, prior, risk, ALPHA)
    other = lib.subset(np.arange(lib.size) < lib.size - 1)
    with pytest.raises(InvalidProblemError, match="library differs"):
        freeze_policy(res, st, other, value_definition=OP)


def test_library_enlargement_and_bin_refinement_checks(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st = InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp)
    big = compute_information_value(st, prior, risk, ALPHA)
    mask = np.array([o["conditioning"] == "no_information" for o in lib.origins])
    small_lib = lib.subset(mask)
    small = compute_information_value(st, prior, compute_risk_table(small_lib, prior, prob.compiled,
                                                                    prob.price_vector()), ALPHA)
    chk = check_library_enlargement(small, big)
    assert chk["V0"]["monotone_holds"] and chk["VT"]["monotone_holds"]
    coarse_b = SignalBinning((FCP,), ((bcp.interior_edges[0][1],),), "median only")
    coarse = compute_information_value(InformationStructure("pp:cp2", "perfect_partial", (FCP,), coarse_b),
                                       prior, risk, ALPHA)
    ref = check_bin_refinement(coarse, big)
    assert ref["monotone_holds"]


def test_adjustment_mask_restricts_both_sides(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st = InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp)
    free = compute_information_value(st, prior, risk, ALPHA)
    k0 = free.V0.diagnostics["chosen_index"]
    mask = adjustment_mask(lib, lib.Q[k0], max_rel_change=0.05)
    assert mask[k0]
    lim = compute_information_value(st, prior, risk, ALPHA, action_mask=mask)
    assert lim.V0.expected_cost >= free.V0.expected_cost - 1e-12
    assert lim.VT.expected_cost >= free.VT.expected_cost - 1e-12
    assert all(mask[k] for k in lim.VT.assignment)


def test_monte_carlo_likelihood_path_records_its_stream(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    em = ComponentErrorModel(FCP, 0.006, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True)
    sm = SignalModel("mc", (em,), is_synthetic=True)
    st = InformationStructure("mc", "sample", binning=bcp, signal_model=sm, likelihood_method="monte_carlo",
                              mc_replicates=200)
    dcr = double_count_guard(sm, {FCP: "true_batch_state"})
    with pytest.raises(InvalidProblemError, match="RandomStreams"):
        compute_information_value(st, prior, risk, ALPHA, double_count_report=dcr)
    r = compute_information_value(st, prior, risk, ALPHA, double_count_report=dcr, streams=streams)
    assert r.likelihood_kind == "sample_monte_carlo" and r.gross_value >= -1e-9
    ra, _ = _sample(FCP, bcp, 0.006)
    a = compute_information_value(ra, prior, risk, ALPHA, double_count_report=dcr)
    # same model, two likelihood approximations: values are close but not forced equal
    assert abs(r.gross_value_randomized_reference - a.gross_value_randomized_reference) < 0.05


def test_alpha_train_calibration_uses_validation_only(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st, sm = _sample(FCP, bcp, 0.006)
    dcr = double_count_guard(sm, {FCP: "true_batch_state"})
    val = model.draw(streams, "validation", 2000)
    out = calibrate_alpha_train(st, prior, risk, lib, val, prob.compiled, streams, ALPHA, [0.01, 0.02, 0.03, 0.04, 0.05],
                                prices=prob.price_vector(), double_count_report=dcr, value_definition=OP)
    assert out["validation_stream_id"] == "root=1103/validation" and len(out["table"]) == 5
    for key, col in (("alpha_train_signal_policy", "policy_validation_risk"),
                     ("alpha_train_constant_policy", "constant_validation_risk")):
        a = out[key]
        if a is not None:
            row = next(r for r in out["table"] if r["alpha_train"] == a)
            assert a <= ALPHA and row[col] <= ALPHA
    with pytest.raises(LeakageError):
        calibrate_alpha_train(st, prior, risk, lib, model.draw(streams, "test", 50), prob.compiled, streams, ALPHA,
                              [0.05], double_count_report=dcr, value_definition=OP)
    with pytest.raises(InvalidProblemError):
        calibrate_alpha_train(st, prior, risk, lib, val, prob.compiled, streams, ALPHA, [0.06],
                              double_count_report=dcr, value_definition=OP)
