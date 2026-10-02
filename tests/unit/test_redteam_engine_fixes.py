"""Regression tests for the red-team findings on the engine (2026-09-24).

Each test encodes the behaviour the contract requires and that the red team showed to be missing
(record: ``audit/_parts/FIX_engine_record.md``).  All numbers are synthetic (``is_synthetic``);
these are software checks, not pilot or official results.

Findings covered: C05/F02 (decision-time action sets), C06/F08 (no peeking), F06 (double-count
guard bound to the signal), F03 finding (randomisation channel), D09/F04 (saving at matched risk,
one alpha_train), C10 (zero tolerance), C11 (explicit MIP gap), D02 (M1 scale / DM margin),
D06 (M1 = M3a on the same box), F11 (validity period, inventory before H x T), oracle deficit (T3).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine_test_helpers import conc, dm_offer, ing, problem, supply
from ration_reliability.datamodel import Provenance, RationDecision, SolverOptions, ValueStatus
from ration_reliability.errors import InvalidProblemError
from ration_reliability.evaluation import evaluate, structural_check
from ration_reliability.information import (
    AssayBudget,
    AssayOption,
    BatchCoverage,
    ComponentErrorModel,
    DoubleCountReport,
    InformationStructure,
    InventoryCoverageResult,
    ObservedComponent,
    PolicyProblem,
    PriorStates,
    SignalBinning,
    SignalModel,
    StateMetadata,
    ValueDefinition,
    batch_gross_value,
    calibrate_alpha_train,
    check_theoretical_order,
    compute_information_value,
    compute_risk_table,
    conditioning_from_likelihood,
    double_count_guard,
    evaluate_frozen_policy,
    freeze_policy,
    generate_conditional_library,
    inventory_coverage_check,
    net_value_per_batch,
    perfect_partial_likelihood,
    quantile_edges,
    signal_bins_for_states,
    solve_no_information,
    solve_signal_policy,
    solve_signal_policy_randomized,
    strategy_decision_value,
    strategy_oracle_reference,
)
from ration_reliability.information.library import decision_time_action_masks
from ration_reliability.information.value import build_likelihood
from ration_reliability.io.run_record import check_solver_settings_for_run_type
from ration_reliability.nutrition import concentration_constraint
from ration_reliability.optimization import METHOD_EQUIVALENCES, equivalence_annotations, get_method
from ration_reliability.optimization.robust import BoxUncertaintySet
from ration_reliability.optimization.safety_margin import (
    DEVELOPMENT_GRID_RELATIVE,
    check_margin_grid,
    select_margin_on_validation,
)
from ration_reliability.uncertainty import DrawSet, IndependentNormalModel, RandomStreams

SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-FIX-ENGINE")
FCP, FDM, FNDF = ObservedComponent("F", "CP"), ObservedComponent("F", "DM"), ObservedComponent("F", "NDF")
ALPHA = 0.05
DMI = 20.0
THR = {"min_marginal_value": 1e-3, "min_marginal_value_status": "synthetic_test_only"}
OP = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE   # explicit value definition (second review R1)
CONTRAST = ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS   # diagnostic; the former engine default (R1)


def _build(dm_sd=0.02, n=300, seed=1103, dm_binning=None):
    """The red-team case: forage F with uncertain CP/NDF/DM, concentrate C; library designed for
    CP bins and DM bins (plus the no-information state)."""
    F = ing("F", 0.35, {"CP": 0.10, "NDF": 0.45}, forage=1.0)
    C = ing("C", 0.88, {"CP": 0.40, "NDF": 0.20})
    cons = [dm_offer(DMI), conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            conc("fndf_min", {"G:forage:NDF": 1.0}, "ge", 18.0)]
    prob = problem([F, C], ["CP", "NDF"], cons, {"F": 0.035, "C": 0.30}, problem_id="fix_engine")
    model = IndependentNormalModel("fix_engine", prob.ingredient_ids, prob.nutrient_ids,
                                   np.array([[0.10, 0.45], [0.40, 0.20]]), np.array([[0.012, 0.03], [0.0, 0.0]]),
                                   np.array([0.35, 0.88]), np.array([dm_sd, 0.0]), is_synthetic=True)
    streams = RandomStreams(seed)
    prior = PriorStates.from_drawset(model.draw(streams, "opt", n),
                                     metadata=StateMetadata.synthetic((FCP, FNDF, FDM), "true_batch_state",
                                                                      label="fix_engine"))
    bcp = SignalBinning((FCP,), (quantile_edges(prior.component_values((FCP,)), 4),), "q opt")
    bdm = dm_binning or SignalBinning((FDM,), (quantile_edges(prior.component_values((FDM,)), 3),), "q opt")
    cond = conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FCP,), bcp), prefix="cp|") + \
        conditioning_from_likelihood(prior, perfect_partial_likelihood(prior, (FDM,), bdm), prefix="dm|")
    lib = generate_conditional_library(prob, prior, cond)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    return prob, model, streams, prior, bcp, bdm, lib, risk


@pytest.fixture(scope="module")
def case():
    return _build()


def _planned_dm(q, d_hat):
    return float(np.asarray(q) @ np.asarray(d_hat))


def _sample_structure(sd, comp=FCP, binning=None, sid=None):
    em = ComponentErrorModel(comp, sd, 0.0, provenance={"sampling_sd": SYN}, is_synthetic=True)
    sm = SignalModel(sid or f"{comp.label()}~{sd:g}", (em,), is_synthetic=True)
    return InformationStructure(sid or f"s:{comp.label()}:{sd:g}", "sample", binning=binning, signal_model=sm), sm


# =================================================================================================
# C05 / F02: the action set follows the decision-time information
# =================================================================================================

def test_v0_and_non_dm_structures_use_only_t0_feasible_rations(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    d0 = prob.dm_estimates()
    off = [k for k, dec in enumerate(lib.decisions) if abs(_planned_dm(dec.q_as_fed, d0) - DMI) > 1e-6]
    assert off, "the library must contain DM-designed rations that are infeasible under the t0 estimate"
    assert not risk.t0_structural_ok[off].any() and risk.t0_structural_ok.sum() == lib.size - len(off)
    for st, kw in ((InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp), {}),
                   (_sample_structure(0.006, binning=bcp)[0], {"prior_variance_basis": {FCP: "true_batch_state"}})):
        res = compute_information_value(st, prior, risk, ALPHA, **kw)
        assert res.action_space["action_space"] == "t0_fixed" and res.is_information_value
        k0 = res.V0.diagnostics["chosen_index"]
        assert abs(_planned_dm(lib.decisions[k0].q_as_fed, d0) - DMI) <= 1e-6
        for k in set(int(a) for a in res.VT.assignment):
            assert abs(_planned_dm(lib.decisions[k].q_as_fed, d0) - DMI) <= 1e-6, lib.origins[k]["conditioning"]
        assert res.gross_value == pytest.approx(res.V0.expected_cost - res.VT.expected_cost)


def test_forced_instance_constant_policy_cannot_use_dm_designed_ration():
    """Red-team forced instance (DM sd 0.05): with only no-information and DM-bin rations available,
    V0 must pick a no-information ration (planned DM = DMI under t0), not a cheaper DM-bin ration."""
    prob, model, streams, prior, bcp, bdm, lib, risk = _build(dm_sd=0.05, n=400)
    d0 = prob.dm_estimates()
    mask = np.array([o["conditioning"] == "no_information" or o["dm_observed"] == ["F"] for o in lib.origins])
    res = compute_information_value(InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp), prior, risk,
                                    ALPHA, action_mask=mask)
    k0 = res.V0.diagnostics["chosen_index"]
    assert lib.origins[k0]["conditioning"] == "no_information"
    assert abs(_planned_dm(lib.decisions[k0].q_as_fed, d0) - DMI) <= 1e-6
    no_info = [k for k, o in enumerate(lib.origins) if o["conditioning"] == "no_information"]
    feasible = [k for k in no_info if risk.violated[:, k].mean() <= ALPHA]
    assert res.V0.expected_cost == pytest.approx(min(lib.costs[k] for k in feasible))


def test_dm_structure_uses_bin_specific_action_sets_and_is_not_called_an_information_value(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st = InformationStructure("pp:dm", "perfect_partial", (FDM,), bdm)
    res = compute_information_value(st, prior, risk, ALPHA)
    lik = build_likelihood(st, prior)
    acts = decision_time_action_masks(prior, lik, risk)
    assert res.action_space["action_space"] == "information_dependent" and res.action_space["dm_observed"] == ["F"]
    # every chosen ration is executable with the DM estimate of its own bin: E[d_F | z] (SH-DM-PLAN note)
    assert res.VT.has_solution
    for z in np.flatnonzero(prior.weights @ lik.L > 0):
        w = lik.L[:, z] * prior.weights
        d_z = prob.dm_estimates().copy()
        d_z[0] = float((w / w.sum()) @ prior.d[:, 0])
        assert acts.d_hat_bins[z] == pytest.approx(d_z, abs=1e-15)
        k = int(res.VT.assignment[z])
        assert abs(_planned_dm(lib.decisions[k].q_as_fed, d_z) - DMI) <= 1e-6
    # no-information rations are not legal once F's DM is known -> containment premise fails
    assert not res.action_space["constant_policies_contained"]
    assert res.information_value_semantics == "cost_risk_comparison_action_spaces_differ"
    assert res.gross_value is None and res.gross_value_randomized_reference is None
    assert res.cost_difference_V0_minus_VT == pytest.approx(res.V0.expected_cost - res.VT.expected_cost)
    assert res.gross_value_status.startswith("action_spaces_differ:")
    part = compute_information_value(InformationStructure("pp:cp", "perfect_partial", (FCP,), bcp), prior, risk,
                                     ALPHA)
    assert res.V0.expected_cost == pytest.approx(part.V0.expected_cost)      # one V0 for every structure
    checks = check_theoretical_order({"dm": res, "cp": part})
    for c in checks:
        if c.lesser == "0" and c.greater == "dm":
            assert c.holds is None                                         # no information value to check


def test_structural_check_equals_public_evaluator():
    F = ing("F", 0.35, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.88, {"CP": 0.40})
    cons = [dm_offer(DMI), conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            conc("share_F_max", {"DM:F": 1.0}, "le", 80.0, cls="structural_hard", dm_source="decision_estimate")]
    prob = problem([F, C], ["CP"], cons, {"F": 0.035, "C": 0.30})
    rng = np.random.default_rng(7)
    Q = rng.uniform(0.0, 60.0, (50, 2))
    Q[:5] = np.array([[40.0, 6.8182], [45.0, 4.8], [0.0, 22.72727], [57.142857, 0.0], [30.0, 10.795454]])
    D = rng.uniform(0.2, 0.95, (50, 2))
    m, v, nn = structural_check(Q, D, prob.compiled)
    for n in range(50):
        ev = evaluate(Q[n], prob.nominal_theta(), prob.dm_estimates(), prob.compiled, d_hat=D[n])
        np.testing.assert_array_equal(v[n], ev.structural_violated)
        np.testing.assert_allclose(m[n], ev.structural_margin, rtol=0, atol=0, equal_nan=True)
        assert bool(nn[n]) == ev.nonnegativity_ok


@pytest.mark.parametrize("seed", range(20))
def test_bin_action_sets_milp_equals_enumeration_and_never_uses_illegal_actions(seed):
    rng = np.random.default_rng(100 + seed)
    S, Z, K = 30, 3, 5
    w = np.full(S, 1 / S)
    L = rng.dirichlet(np.ones(Z), size=S)
    I = (rng.uniform(size=(S, K)) < rng.uniform(0.0, 0.3, size=K)).astype(float)
    C = rng.uniform(1, 2, size=K)
    t0 = rng.uniform(size=K) < 0.7
    t0[0] = True
    bins = rng.uniform(size=(Z, K)) < 0.6
    pp = PolicyProblem.from_arrays(w, L, I, C, 0.1, action_mask=t0, bin_action_mask=bins)
    milp, enum = solve_signal_policy(pp), solve_signal_policy(pp, method="enumeration")
    assert milp.status == enum.status
    if milp.has_solution:
        assert milp.expected_cost == pytest.approx(enum.expected_cost, abs=1e-9)
        for z in pp.active_bins:
            assert bins[z, milp.assignment[z]] and bins[z, enum.assignment[z]]
    rnd = solve_signal_policy_randomized(pp)
    if rnd.has_solution:
        assert np.all(rnd.w[:, ~bins.any(axis=0)] == 0) and np.all(rnd.w[~bins] <= 1e-12)
    v0 = solve_no_information(pp)
    if v0.has_solution:
        assert t0[v0.diagnostics["chosen_index"]]                          # V0 uses the t0 action set only
    if pp.constant_policies_contained and v0.has_solution:
        assert milp.has_solution and milp.expected_cost <= v0.expected_cost + 1e-9


def test_frozen_dm_policy_reports_unverifiable_fallback_bins():
    edges = SignalBinning((FDM,), ((0.20, 0.34, 0.36, 0.90),), "synthetic edges with empty outer bins")
    prob, model, streams, prior, bcp, bdm, lib, risk = _build(dm_binning=edges)
    st = InformationStructure("pp:dm4", "perfect_partial", (FDM,), edges)
    res = compute_information_value(st, prior, risk, ALPHA)
    assert res.VT.has_solution
    pol = freeze_policy(res, st, lib, value_definition=OP)
    unver = pol.development["fallback_bins_structurally_unverified"]
    assert unver and set(unver) <= set(np.flatnonzero(np.asarray(res.occupancy["p_bin"]) == 0).tolist())
    test = model.draw(streams, "test", 2000)
    ev = evaluate_frozen_policy(pol, test, prob.compiled, streams, prices=prob.price_vector())
    z = signal_bins_for_states(pol, test, streams)
    assert ev.extra["n_states_in_structurally_unverified_fallback_bins"] == int(np.isin(z, unver).sum())
    assert ev.extra["information_value_semantics"] == "cost_risk_comparison_action_spaces_differ"


# =================================================================================================
# C06 / F08: frozen policy never peeks at hidden components
# =================================================================================================

def test_frozen_policy_does_not_peek(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st, sm = _sample_structure(0.006, binning=bcp)
    pol = freeze_policy(compute_information_value(st, prior, risk, ALPHA,
                                                  prior_variance_basis={FCP: "true_batch_state"}), st, lib,
                        value_definition=OP)
    test = model.draw(streams, "test", 3000)
    th, d = np.array(test.theta), np.array(test.d)
    rng = np.random.default_rng(0)
    th[:, 0, 1] = rng.uniform(0.2, 0.7, size=th.shape[0])
    d[:, 0] = rng.uniform(0.2, 0.5, size=d.shape[0])
    th[:, 1, :] = rng.uniform(0.1, 0.6, size=(th.shape[0], 2))
    test2 = DrawSet(th, d, test.stream, test.stream_id, test.model_id, test.model_fingerprint,
                    test.ingredient_ids, test.nutrient_ids, test.is_synthetic)
    assert np.array_equal(signal_bins_for_states(pol, test, streams), signal_bins_for_states(pol, test2, streams))


# =================================================================================================
# F06: the double-count guard is bound to the structure's signal
# =================================================================================================

def test_double_count_report_must_belong_to_the_structure_signal(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    em = ComponentErrorModel(FCP, 0.006, 0.004, provenance={"sampling_sd": SYN, "lab_repeatability_sd": SYN},
                             is_synthetic=True)
    sm = SignalModel("s", (em,), is_synthetic=True)
    st = InformationStructure("s:cp", "sample", binning=bcp, signal_model=sm)
    real = double_count_guard(sm, {FCP: "observed_incl_sampling_and_lab"})
    assert real.status == "violation" and real.is_bound
    with pytest.raises(InvalidProblemError, match="not bound"):                  # fabricated bare "ok"
        compute_information_value(st, prior, risk, ALPHA, double_count_report=DoubleCountReport("ok", (), {}))
    other = SignalModel("o", (ComponentErrorModel(FNDF, 0.0, 0.0, is_synthetic=True),), is_synthetic=True)
    with pytest.raises(InvalidProblemError, match="another signal"):             # report of another signal
        compute_information_value(st, prior, risk, ALPHA,
                                  double_count_report=double_count_guard(other, {FNDF: "true_batch_state"}))
    forged = dataclasses.replace(real, status="ok", per_component={"F:CP": "ok"})
    with pytest.raises(InvalidProblemError, match="does not reproduce"):         # tampered verdict
        compute_information_value(st, prior, risk, ALPHA, double_count_report=forged)
    with pytest.raises(InvalidProblemError, match="double counting"):            # the real verdict blocks
        compute_information_value(st, prior, risk, ALPHA, double_count_report=real)
    with pytest.raises(InvalidProblemError, match="double counting"):
        compute_information_value(st, prior, risk, ALPHA,
                                  prior_variance_basis={FCP: "observed_incl_sampling_and_lab"})
    ok = compute_information_value(st, prior, risk, ALPHA, prior_variance_basis={FCP: "true_batch_state"})
    assert ok.error_model_identification == "identified_by_declared_sources"
    assert ok.error_model_report["source"] == "recomputed_from_prior_variance_basis"
    with pytest.raises(InvalidProblemError, match="differs"):                   # both given, inconsistent
        compute_information_value(st, prior, risk, ALPHA, prior_variance_basis={FCP: "true_batch_state"},
                                  double_count_report=real)


# =================================================================================================
# F03 finding: randomisation channel of deterministic policies
# =================================================================================================

def test_state_independent_signal_has_zero_net_value(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    un = compute_information_value(InformationStructure("noise", "uninformative", bin_probabilities=(0.25,) * 4),
                                   prior, risk, ALPHA)
    assert un.gross_value > 1e-3                        # the deterministic gross value contains randomisation
    assert un.value_net_of_randomization == 0.0         # ... which the net value removes exactly
    assert un.gross_value_randomized_reference == pytest.approx(0.0, abs=1e-9)
    st, _ = _sample_structure(10.0, binning=bcp)
    near = compute_information_value(st, prior, risk, ALPHA, prior_variance_basis={FCP: "true_batch_state"})
    good_st, _ = _sample_structure(0.006, binning=bcp)
    good = compute_information_value(good_st, prior, risk, ALPHA, prior_variance_basis={FCP: "true_batch_state"})
    assert near.value_net_of_randomization == pytest.approx(near.gross_value - near.randomization_benchmark_value,
                                                            abs=1e-12)
    assert abs(near.value_net_of_randomization) < 1e-3 < good.value_net_of_randomization
    assert near.value("net_of_randomization") == near.value_net_of_randomization


def test_decision_value_strategy_does_not_buy_a_pure_noise_assay(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    basis = {FCP: "true_batch_state", FNDF: "true_batch_state"}
    noise_cp = SignalModel("useless", (ComponentErrorModel(FCP, 50.0, 0.0, provenance={"sampling_sd": SYN},
                                                           is_synthetic=True),), is_synthetic=True)
    useless_cp = AssayOption("useless_cp", "F", ("CP",), bcp, noise_cp, cost_per_panel=None)
    sel = strategy_decision_value([useless_cp], prior, risk, ALPHA, AssayBudget(max_panels=1),
                                  prior_variance_basis=basis, value_definition=CONTRAST, **THR)
    assert sel.selected == () and sel.details["value_kind"] == "net_of_randomization"
    assert "materiality threshold" in sel.details["greedy_history"][-1]["stopped"]
    # the deterministic gross value of the same noise assay is not small: that is the randomisation channel
    det = strategy_decision_value([useless_cp], prior, risk, ALPHA, AssayBudget(max_panels=1),
                                  prior_variance_basis=basis, value_kind="deterministic", **THR)
    assert det.details["greedy_history"][0]["joint_value"] > THR["min_marginal_value"]
    # an informative CP assay is bought; a pure-noise NDF assay adds nothing on top of it
    bndf = SignalBinning((FNDF,), (quantile_edges(prior.component_values((FNDF,)), 3),), "q opt")
    noise_ndf = SignalModel("useless_ndf", (ComponentErrorModel(FNDF, 50.0, 0.0, provenance={"sampling_sd": SYN},
                                                                is_synthetic=True),), is_synthetic=True)
    good_sm = SignalModel("good", (ComponentErrorModel(FCP, 0.006, 0.0, provenance={"sampling_sd": SYN},
                                                       is_synthetic=True),), is_synthetic=True)
    opts = [AssayOption("useless_ndf", "F", ("NDF",), bndf, noise_ndf), AssayOption("good_cp", "F", ("CP",), bcp,
                                                                                    good_sm)]
    sel2 = strategy_decision_value(opts, prior, risk, ALPHA, AssayBudget(max_panels=2), prior_variance_basis=basis,
                                   value_definition=CONTRAST, **THR)
    assert sel2.selected == ("good_cp",)
    assert sel2.details["greedy_history"][-1]["stopped"].startswith("marginal joint value")
    with pytest.raises(TypeError):                                             # threshold must be declared
        strategy_decision_value([useless_cp], prior, risk, ALPHA, AssayBudget(max_panels=1),
                                prior_variance_basis=basis, value_definition=CONTRAST)
    with pytest.raises(InvalidProblemError):
        strategy_decision_value([useless_cp], prior, risk, ALPHA, AssayBudget(max_panels=1),
                                prior_variance_basis=basis, min_marginal_value=1e-3, min_marginal_value_status="guess",
                                value_definition=CONTRAST)


# =================================================================================================
# D09 / F04: saving only at matched risk; one alpha_train for V0 - VT
# =================================================================================================

def test_frozen_policy_saving_only_when_both_meet_alpha(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st, _ = _sample_structure(0.006, binning=bcp)
    res = compute_information_value(st, prior, risk, ALPHA, prior_variance_basis={FCP: "true_batch_state"})
    pol = freeze_policy(res, st, lib, value_definition=OP)
    test = model.draw(streams, "test", 5000)
    ev = evaluate_frozen_policy(pol, test, prob.compiled, streams, prices=prob.price_vector())
    both = (ev.policy_joint_violation["rate_used"] <= ALPHA + 1e-8 and
            ev.constant_joint_violation["rate_used"] <= ALPHA + 1e-8)
    assert ev.cost_difference_per_head_day == pytest.approx(ev.constant_mean_cost - ev.policy_mean_cost)
    if both:
        assert ev.realised_cost_saving_per_head_day == ev.cost_difference_per_head_day
    else:
        assert ev.realised_cost_saving_per_head_day is None
        assert ev.saving_status == "cost_difference_at_unequal_risk"
    strict = evaluate_frozen_policy(pol, test, prob.compiled, streams, prices=prob.price_vector(), target_alpha=0.001)
    assert strict.realised_cost_saving_per_head_day is None and strict.target_alpha == 0.001
    assert strict.saving_status == "cost_difference_at_unequal_risk"


def test_calibration_reports_one_common_alpha_train(case):
    prob, model, streams, prior, bcp, bdm, lib, risk = case
    st, _ = _sample_structure(0.006, binning=bcp)
    val = model.draw(streams, "validation", 1500)
    out = calibrate_alpha_train(st, prior, risk, lib, val, prob.compiled, streams, ALPHA, [0.01, 0.02, 0.03, 0.05],
                                prices=prob.price_vector(), prior_variance_basis={FCP: "true_batch_state"},
                                value_definition=OP)
    common = out["alpha_train_common"]
    if common is not None:
        row = next(r for r in out["table"] if r["alpha_train"] == common)
        assert row["policy_meets"] and row["constant_meets"]
        for a in (out["alpha_train_signal_policy"], out["alpha_train_constant_policy"]):
            assert a is not None and common <= a
    assert "never combine" in out["usage"]["alpha_train_signal_policy/alpha_train_constant_policy"]


# =================================================================================================
# C10: zero tolerance
# =================================================================================================

def _mcase(tol):
    F = ing("F", 0.35, {"CP": 0.09, "NDF": 0.48, "starch": 0.25}, forage=1.0)
    H = ing("H", 0.88, {"CP": 0.17, "NDF": 0.45, "starch": 0.03}, forage=1.0)
    G = ing("G", 0.88, {"CP": 0.09, "NDF": 0.10, "starch": 0.70})
    S = ing("S", 0.89, {"CP": 0.48, "NDF": 0.12, "starch": 0.02})
    cons = [dm_offer(DMI), conc("cp_min", {"CP": 1.0}, "ge", 16.0, tol=tol),
            conc("ndf_min", {"NDF": 1.0}, "ge", 28.0, tol=tol), conc("starch_max", {"starch": 1.0}, "le", 28.0, tol=tol),
            conc("fndf_min", {"G:forage:NDF": 1.0}, "ge", 17.0, tol=tol),
            conc("starch_minus_2fndf", {"starch": 1.0, "G:forage:NDF": -2.0}, "le", -6.0, tol=tol)]
    return problem([F, H, G, S], ["CP", "NDF", "starch"], cons, {"F": 0.04, "H": 0.20, "G": 0.25, "S": 0.45})


def test_zero_tolerance_is_rejected_and_binding_rows_are_not_false_violations():
    with pytest.raises(InvalidProblemError, match="numerical_tolerance must be finite and > 0"):
        _ = _mcase(0.0).compiled
    pr = _mcase(1e-6)
    r = get_method("M0_nominal")(pr)
    ev = evaluate(r.decision, pr.nominal_theta(), pr.dm_estimates(), pr.compiled)
    mg = ev.margin[0, ev.probabilistic_mask]
    assert np.any(np.abs(mg) < 1e-9)                    # at least one row binds at the design point
    assert not ev.joint_violation[0]
    with pytest.raises(TypeError):                      # the builder has no default tolerance any more
        concentration_constraint("cp", {"CP": 1.0}, "ge", 16.0, "%", SYN)


# =================================================================================================
# C11: explicit MIP gap for pilot / official runs
# =================================================================================================

def test_pilot_and_official_runs_need_an_explicit_mip_gap():
    for rt in ("pilot", "official"):
        for bad in ({}, {"mip_rel_gap": None}, {"mip_rel_gap": "default"}, {"mip_rel_gap": -1.0},
                    {"mip_rel_gap": float("nan")}):
            with pytest.raises(ValueError, match="mip_rel_gap"):
                check_solver_settings_for_run_type(rt, bad)
        check_solver_settings_for_run_type(rt, {"mip_rel_gap": 1e-4})
        check_solver_settings_for_run_type(rt, {"mip_rel_gap": "not_applicable"})
    check_solver_settings_for_run_type("smoke", {})
    pr = _mcase(1e-6)
    model = IndependentNormalModel("m", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(),
                                   0.03 * np.abs(pr.nominal_theta()), pr.dm_estimates(), 0.01 * pr.dm_estimates(),
                                   is_synthetic=True)
    opt = model.draw(RandomStreams(5), "opt", 30)
    m2 = get_method("M2_joint_chance_saa")
    assert m2(pr, opt_draws=opt, params={"alpha_train": 0.1}).tolerances["mip_rel_gap_source"] == \
        "solver_default_not_set_explicitly"
    r = m2(pr, opt_draws=opt, params={"alpha_train": 0.1}, solver_options=SolverOptions(mip_rel_gap=1e-4))
    assert r.tolerances["mip_rel_gap_source"] == "explicit" and r.tolerances["mip_rel_gap"] == 1e-4


# =================================================================================================
# D02: M1 scale and DM margin are declared, never defaulted; sd scale + contract grid rejected
# =================================================================================================

def test_m1_scale_and_dm_margin_have_no_defaults():
    pr = _mcase(1e-6)
    M1 = get_method("M1_safety_margin")
    r = M1(pr, params={"k": 0.05, "apply_to_dm": True})
    assert r.status.value == "invalid_input" and "margin_scale is required" in r.message
    r = M1(pr, params={"k": 0.05, "margin_scale": "relative"})
    assert r.status.value == "invalid_input" and "apply_to_dm" in r.message
    assert M1(pr, params={"k": 0.05, "margin_scale": "relative", "apply_to_dm": False}).has_solution
    with pytest.raises(ValueError, match="relative grid"):
        check_margin_grid(DEVELOPMENT_GRID_RELATIVE, "sd")
    with pytest.raises(ValueError, match="declared"):
        check_margin_grid(DEVELOPMENT_GRID_RELATIVE, None)
    check_margin_grid(DEVELOPMENT_GRID_RELATIVE, "relative")
    check_margin_grid((0.0, 0.5, 1.0, 2.0), "sd")
    model = IndependentNormalModel("m", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(),
                                   0.03 * np.abs(pr.nominal_theta()), pr.dm_estimates(), 0.01 * pr.dm_estimates(),
                                   is_synthetic=True)
    s = RandomStreams(11)
    opt, val = model.draw(s, "opt", 64), model.draw(s, "validation", 500)
    with pytest.raises(ValueError, match="relative grid"):
        select_margin_on_validation(pr, val, grid=DEVELOPMENT_GRID_RELATIVE, target_alpha=0.05,
                                    screening_rule="rate_upper_le_alpha", opt_draws=opt,
                                    params={"margin_scale": "sd", "apply_to_dm": True})
    sel = select_margin_on_validation(pr, val, grid=DEVELOPMENT_GRID_RELATIVE, target_alpha=0.05,
                                      screening_rule="rate_upper_le_alpha", opt_draws=opt,
                                      params={"margin_scale": "relative", "apply_to_dm": True})
    assert [c.value for c in sel.candidates] == list(DEVELOPMENT_GRID_RELATIVE)


# =================================================================================================
# D06: M1 and M3a on the same box are one method (annotated)
# =================================================================================================

def test_m1_and_m3a_on_the_same_box_are_one_method():
    pr = _mcase(1e-6)
    sd = 0.03 * np.abs(pr.nominal_theta())
    dsd = 0.01 * pr.dm_estimates()
    k = 1.5
    r1 = get_method("M1_safety_margin")(pr, params={"k": k, "margin_scale": "sd", "apply_to_dm": True,
                                                    "sd_source": "explicit", "theta_sd": sd.tolist(),
                                                    "d_sd": dsd.tolist()})
    box = BoxUncertaintySet.from_mean_sd("b", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), sd,
                                         pr.dm_estimates(), dsd, k, is_synthetic=True)
    r3 = get_method("M3a_box_robust")(pr, params={"uncertainty_set": box})
    assert r1.has_solution and r3.has_solution
    assert r1.objective == pytest.approx(r3.objective, rel=1e-7)
    assert "EQ-M1-M3a" in r1.diagnostics["equivalent_methods"]
    assert "EQ-M1-M3a" in r3.diagnostics["equivalent_methods"]
    ids = {rec["pair_id"] for rec in METHOD_EQUIVALENCES}
    assert {"EQ-M1-M3a", "EQ-M0-MODES", "EQ-M3c-M2-ALPHA0"} <= ids
    assert [r["pair_id"] for r in equivalence_annotations("M3c_scenario_set_robust")] == ["EQ-M3c-M2-ALPHA0"]


# =================================================================================================
# F11: validity period required; no H x T multiplier without a verified inventory
# =================================================================================================

def test_batch_values_need_validity_period_and_verified_inventory():
    with pytest.raises(TypeError):
        BatchCoverage(100, 14, "synthetic_test_only", "synthetic_test_only")
    with pytest.raises(InvalidProblemError, match="required"):
        BatchCoverage(100, 14, "synthetic_test_only", "synthetic_test_only", None)
    cov = BatchCoverage(100, 14, "synthetic_test_only", "synthetic_test_only", 14)
    assert cov.validity_status == "synthetic_test_only"
    Q = np.array([[30.0, 5.0], [40.0, 3.0]])
    short = inventory_coverage_check(Q, ("F", "C"), cov, {"F": 45000.0, "C": 10000.0})
    assert short.status == "not_covered" and not short.fixed_multiplier_valid
    with pytest.raises(InvalidProblemError, match="not_covered"):
        batch_gross_value(0.01, cov, inventory_check=short, value_definition=OP)
    with pytest.raises(InvalidProblemError, match="not_covered"):
        net_value_per_batch(0.01, cov, {c: None for c in ("sampling", "laboratory", "logistics", "waiting",
                                                          "reformulation")}, inventory_check=short,
                            value_definition=OP)
    ok = inventory_coverage_check(Q, ("F", "C"), cov, {"F": 60000.0, "C": 10000.0})
    assert batch_gross_value(0.01, cov, inventory_check=ok, value_definition=OP,       # FIX_A: bare float named
                             unbound_value_declaration="synthetic H x T arithmetic test",
                             # FIX3_BC: labelled development diagnostic (no complete decision problem)
                             diagnostic_without_decision_problem=True) == pytest.approx(14.0)
    other = BatchCoverage(50, 14, "synthetic_test_only", "synthetic_test_only", 14)
    with pytest.raises(InvalidProblemError, match="another coverage"):
        batch_gross_value(0.01, other, inventory_check=ok, value_definition=OP)
    hand = InventoryCoverageResult("covered", {}, {}, {}, (), True)           # no coverage recorded
    with pytest.raises(InvalidProblemError, match="another coverage"):
        batch_gross_value(0.01, cov, inventory_check=hand, value_definition=OP)


# =================================================================================================
# oracle reference: no sums of deficits with different units (T3)
# =================================================================================================

def test_oracle_reference_counts_constraint_events_without_cross_unit_sums():
    F = ing("F", 0.40, {"CP": 0.10, "NDF": 0.50}, forage=1.0)
    X = ing("X", 0.90, {"CP": 0.30, "NDF": 0.30})
    C = ing("C", 0.80, {"CP": 0.40, "NDF": 0.20})
    cons = [dm_offer(DMI), conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            supply("ndf_supply_min", {"NDF": 1.0}, "ge", 7.0, "kg/d")]
    prob = problem([F, X, C], ["CP", "NDF"], cons, {"F": 0.04, "X": 0.20, "C": 0.32})
    model = IndependentNormalModel("o", prob.ingredient_ids, prob.nutrient_ids, prob.nominal_theta(),
                                   np.array([[0.01, 0.03], [0.02, 0.02], [0.0, 0.0]]), prob.dm_estimates(),
                                   np.zeros(3), is_synthetic=True)
    prior = PriorStates.from_drawset(model.draw(RandomStreams(3), "opt", 200))
    q0 = RationDecision(prob.ingredient_ids, np.array([30.0, 4.0, 5.0]), prob.dm_estimates(), "syn_q0")
    b = lambda c: SignalBinning((c,), ((float(np.median(prior.component_values((c,)))),),), "median")  # noqa: E731
    XCP = ObservedComponent("X", "CP")
    opts = [AssayOption("F_cp", "F", ("CP",), b(FCP)), AssayOption("X_cp", "X", ("CP",), b(XCP)),
            AssayOption("F_ndf", "F", ("NDF",), b(FNDF))]
    true_theta = np.array([[0.07, 0.40], [0.30, 0.30], [0.40, 0.20]])
    true_d = prob.dm_estimates()
    orc = strategy_oracle_reference(opts, true_theta, true_d, prior, q0, prob.compiled, AssayBudget(max_panels=3),
                                    acknowledge_oracle=True)
    base = evaluate(q0, true_theta, true_d, prob.compiled)
    n_base = int(base.violated[0, base.probabilistic_mask].sum())
    th_bar, d_bar = prior.weighted_mean(np.ones(prior.n_states))
    for oid, score in orc.ranking:
        o = next(x for x in opts if x.option_id == oid)
        th = true_theta.copy()
        for c in o.observed:
            i, j = prior.component_index(c)
            th[i, j] = th_bar[i, j]
        ev = evaluate(q0, th, true_d, prob.compiled)
        assert score == n_base - int(ev.violated[0, ev.probabilistic_mask].sum())      # a count, not a unit sum
        per = orc.details["per_constraint_deficit_change"][oid]
        assert {v["unit"] for v in per.values()} == {"%", "kg/d"}                     # units kept per constraint
