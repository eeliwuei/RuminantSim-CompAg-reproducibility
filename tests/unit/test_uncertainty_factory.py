"""Uncertainty model factory (review R3).  All numbers are synthetic test values.

Covers: synthetic acceptance (target mean 0.02, SD 0.015 on [0, 1]: TN_MM exact, TN_NAIVE drift
flagged), the naive family gate, ``infeasible_moment_match`` handling without changing targets,
reproducibility and fingerprint sensitivity, one evaluation world for opt / validation / test /
robust sets / information prior, latent vs transformed copula correlation, the two-layer farm model
as an extension scenario only, development-data-only family selection, metadata completeness, and a
static guard against new private naive/independent-normal model entry points.
"""

from __future__ import annotations

import copy
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from ration_reliability.errors import ConfigValidationError, LeakageError
from ration_reliability.information.prior import ObservedComponent, PriorStates
from ration_reliability.information.signal import PRIOR_VARIANCE_BASES
from ration_reliability.optimization.robust import solve_box_robust
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams
from ration_reliability.uncertainty.distributions import fit_marginal, fit_truncnorm_moments, naive_truncnorm
from ration_reliability.uncertainty.factory import (
    DEVELOPMENT_SPLITS,
    FactoryBuildError,
    FactoryModel,
    FamilySelection,
    WorldMismatchError,
    build_uncertainty_model,
    calibrate_latent_rho,
    metadata_for_drawset,
    select_family_on_development_data,
    transformed_pearson,
)
from ration_reliability.uncertainty.information_states import TRUTH_SD_BASES, TwoLayerFarmModel
from ration_reliability.uncertainty.spec import (
    FAMILY_TN_NAIVE_DIAGNOSTIC,
    VARIANCE_BASES,
    UncertaintySpec,
)

from engine_test_helpers import two_ingredient_problem

_E0 = Path(__file__).resolve().parents[2] / "experiments" / "E0_verification"
if str(_E0) not in sys.path:
    sys.path.insert(0, str(_E0))
import smoke_pipeline as SP  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

RULE = {"primary_family": "TN_MM", "fallback_families": [], "on_exhausted": "error",
        "status": "synthetic_test_only", "rationale": "synthetic unit test", "selection_basis": "declared_rule"}
DEFAULTS = {"variance_basis": "true_batch_state", "data_fingerprint": "synthetic:unit", "decomposition_id": "none",
            "decomposition_source": None, "measurement_model_id": None, "provenance_status": "synthetic_test_only",
            "source_id": None, "locator": None}
NAIVE_RULE = dict(RULE, primary_family=FAMILY_TN_NAIVE_DIAGNOSTIC)


def spec_from(theta_mean, theta_sd, d_mean, d_sd, *, ids=("A",), nuts=("X",), rule=None, purpose="unit_test",
              semantics="target_marginal_moments", defaults=None, overrides=None, correlation=None, two_layer=None,
              spec_id="syn", theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0), is_synthetic=True):
    return UncertaintySpec.from_arrays(
        spec_id, list(ids), list(nuts), theta_mean, theta_sd, d_mean, d_sd, purpose=purpose,
        moment_semantics=semantics, is_synthetic=is_synthetic, family_rule=dict(rule or RULE),
        cell_defaults=dict(DEFAULTS if defaults is None else defaults), theta_bounds=theta_bounds,
        d_bounds=d_bounds, cell_overrides=overrides, correlation=correlation, two_layer=two_layer)


def one_cell(mean=0.02, sd=0.015, **kw):
    return spec_from([[mean]], [[sd]], [0.5], [0.0], **kw)


# =================================================================================================
# 1  synthetic acceptance: target mean 0.02, SD 0.015, support [0, 1]
# =================================================================================================

def test_acceptance_tn_mm_reproduces_target_moments_exactly():
    m = build_uncertainty_model(one_cell())
    c = m.metadata.cell("A", "X")
    assert c.fit_status == "matched" and c.family == "TN_MM" and not c.is_diagnostic and not c.moment_drift_flag
    assert c.target_mean == 0.02 and c.target_sd == 0.015                  # targets never modified
    assert c.achieved_mean == pytest.approx(0.02, rel=1e-12)
    assert c.achieved_sd == pytest.approx(0.015, rel=1e-12)
    ref = fit_truncnorm_moments(0.02, 0.015, 0.0, 1.0)
    assert c.params == pytest.approx(ref.params, rel=1e-12)
    assert "truncation_active" in c.flags
    # Monte Carlo check of the sampled world (independent evaluation stream)
    ds = m.draw(RandomStreams(1103), "test", 400_000)
    x = ds.theta[:, 0, 0]
    n = x.size
    se_m = 0.015 / math.sqrt(n)
    se_s = 0.015 * math.sqrt((ref.kurtosis() - 1.0) / (4.0 * n))
    assert abs(x.mean() - 0.02) < 5 * se_m
    assert abs(x.std(ddof=1) - 0.015) < 5 * se_s
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_acceptance_naive_diagnostic_drift_is_flagged_and_reported():
    spec = one_cell(rule=NAIVE_RULE, purpose="diagnostic", semantics="naive_parent_parameters")
    m = build_uncertainty_model(spec)
    c = m.metadata.cell("A", "X")
    assert c.family == FAMILY_TN_NAIVE_DIAGNOSTIC and c.fit_status == "diagnostic_drift"
    assert c.moment_drift_flag and c.is_diagnostic and m.metadata.is_diagnostic and m.is_diagnostic
    assert (c.target_mean, c.target_sd) == (0.02, 0.015)
    # the independent review's probe values (independent_probes.json, truncation_moment_probe.naive)
    assert c.achieved_mean == pytest.approx(0.022707065903033938, rel=1e-9)
    assert c.achieved_sd == pytest.approx(0.01278790350823591, rel=1e-9)
    assert c.mean_rel_shift == pytest.approx(0.13535329515169686, rel=1e-9)
    assert c.sd_rel_error == pytest.approx(-0.1474730994509393, rel=1e-9)
    ref = naive_truncnorm(0.02, 0.015, 0.0, 1.0)
    assert c.params == pytest.approx(ref.params)
    ds = m.draw(RandomStreams(1103), "test", 200_000)
    assert ds.theta[:, 0, 0].mean() - 0.02 > 20 * 0.015 / math.sqrt(ds.n_draws)     # drift is real, not noise
    assert m.metadata.summary_dict()["n_diagnostic_drift_cells"] == 1


@pytest.mark.parametrize("purpose", ["main_analysis", "sensitivity_scenario", "smoke", "unit_test", "diagnostic"])
def test_naive_family_cannot_enter_a_target_moment_model(purpose):
    with pytest.raises(ConfigValidationError, match="TN_NAIVE_DIAGNOSTIC"):
        one_cell(rule=NAIVE_RULE, purpose=purpose, semantics="target_marginal_moments")
    with pytest.raises(ConfigValidationError, match="TN_NAIVE_DIAGNOSTIC"):
        one_cell(rule=dict(RULE, fallback_families=[FAMILY_TN_NAIVE_DIAGNOSTIC]), purpose=purpose)
    with pytest.raises(ConfigValidationError, match="TN_NAIVE_DIAGNOSTIC"):
        one_cell(purpose=purpose, overrides={("A", "X"): {"family": FAMILY_TN_NAIVE_DIAGNOSTIC,
                                                          "family_choice_reason": "try"}})


def test_naive_gate_other_forms():
    with pytest.raises(ConfigValidationError, match="not requestable"):          # the bare name, even diagnostic
        one_cell(rule=dict(RULE, primary_family="TN_NAIVE"), purpose="diagnostic", semantics="naive_parent_parameters")
    with pytest.raises(ConfigValidationError, match="only with purpose 'diagnostic'"):
        one_cell(rule=NAIVE_RULE, purpose="smoke", semantics="naive_parent_parameters")
    with pytest.raises(ConfigValidationError, match="without the naive diagnostic family"):
        one_cell(purpose="diagnostic", semantics="naive_parent_parameters")
    with pytest.raises(ConfigValidationError, match="not a sampling family"):
        one_cell(rule=dict(RULE, primary_family="N_UNTRUNCATED"))
    with pytest.raises(ConfigValidationError, match="no fallback"):
        one_cell(rule=dict(NAIVE_RULE, fallback_families=["TN_MM"]), purpose="diagnostic",
                 semantics="naive_parent_parameters")


# =================================================================================================
# 2  infeasible moment match: reported, fallback in declared order, or excluded -- targets unchanged
# =================================================================================================

def test_infeasible_target_falls_back_in_declared_order_with_reason():
    spec = one_cell(0.02, 0.03, rule=dict(RULE, fallback_families=["LN_MM", "BETA_MM"]))
    c = build_uncertainty_model(spec).metadata.cell("A", "X")
    assert [a.family for a in c.attempts] == ["TN_MM", "LN_MM"]
    assert c.attempts[0].fit_status == "infeasible_moment_match" and "tn_cv_ge_1" in c.attempts[0].flags
    assert c.requested_family == "TN_MM" and c.family == "LN_MM" and c.fit_status == "matched" and c.used_fallback
    assert "fallback to LN_MM" in c.family_choice_reason and "TN_MM: infeasible_moment_match" in c.family_choice_reason
    assert (c.target_mean, c.target_sd) == (0.02, 0.03)
    assert c.achieved_mean == pytest.approx(0.02, rel=1e-8) and c.achieved_sd == pytest.approx(0.03, rel=1e-8)
    # order matters: BETA first gives BETA
    c2 = build_uncertainty_model(one_cell(0.02, 0.03, rule=dict(RULE, fallback_families=["BETA_MM", "LN_MM"]))
                                 ).metadata.cell("A", "X")
    assert c2.family == "BETA_MM" and "beta_density_unbounded_at_lower" in c2.flags


def test_exhausted_error_lists_the_cell_and_never_changes_the_target():
    with pytest.raises(FactoryBuildError) as ei:
        build_uncertainty_model(one_cell(0.02, 0.03))
    msg = str(ei.value)
    assert "A:X" in msg and "infeasible_moment_match" in msg and "targets are not changed" in msg


def test_bhatia_davis_violation_is_infeasible_for_every_family_and_can_be_excluded_as_missing():
    rule = dict(RULE, fallback_families=["BETA_MM", "LN_MM"], on_exhausted="exclude_cell_as_missing")
    spec = spec_from([[0.02, 0.30]], [[0.20, 0.05]], [0.5], [0.0], nuts=("X", "Y"), rule=rule)
    m = build_uncertainty_model(spec)
    c = m.metadata.cell("A", "X")
    assert c.fit_status == "infeasible_moment_match" and c.exclusion_action == "exclude_cell_as_missing"
    assert [a.fit_status for a in c.attempts] == ["infeasible_moment_match"] * 3
    assert all(a.distribution_status == "infeasible_any_family" for a in c.attempts)
    assert (c.target_mean, c.target_sd) == (0.02, 0.20) and math.isnan(c.achieved_mean)
    ds = m.draw(RandomStreams(3301), "opt", 50)
    assert np.all(np.isnan(ds.theta[:, 0, 0])) and np.all(np.isfinite(ds.theta[:, 0, 1]))   # NaN, never 0
    assert m.metadata.summary_dict()["excluded_cells"] == ["A:X"]
    assert ("A", "X") not in m.metadata.variance_basis_map()
    box = m.box(1.0)
    assert np.isnan(box.theta_lo[0, 0]) and np.isnan(box.theta_hi[0, 0])


def test_exclude_ingredient_reduces_the_axes_and_the_problem_must_follow():
    rule = dict(RULE, on_exhausted="exclude_ingredient")
    pr = two_ingredient_problem()
    spec = spec_from([[0.10], [0.40]], [[0.20], [0.02]], [0.40, 0.80], [0.01, 0.01], ids=("F", "C"), nuts=("CP",),
                     rule=rule)
    m = build_uncertainty_model(spec)
    assert m.ingredient_ids == ("C",) and m.metadata.excluded_ingredients[0][0] == "F"
    assert not m.metadata.cell("F", "CP").included
    fdm = m.metadata.cell("F", "DM")               # fitted fine, but leaves the model with its ingredient
    assert fdm.fit_status == "matched" and not fdm.included and fdm.exclusion_action == "ingredient_excluded"
    assert all(k[0] == "C" for k in m.metadata.variance_basis_map())
    assert m.metadata.summary_dict()["excluded_ingredients"] == ["F"]
    with pytest.raises(WorldMismatchError, match="excluded"):
        m.check_problem(pr)


def test_dm_cell_cannot_become_missing_and_missing_sd_is_not_a_point():
    rule = dict(RULE, fallback_families=["BETA_MM"], on_exhausted="exclude_cell_as_missing")
    with pytest.raises(FactoryBuildError, match="DM cell cannot be excluded"):
        build_uncertainty_model(spec_from([[0.3]], [[0.05]], [0.5], [0.6], rule=rule))
    with pytest.raises(ConfigValidationError, match="mean without SD"):
        spec_from([[0.3]], [[np.nan]], [0.5], [0.0])
    with pytest.raises(ConfigValidationError, match="DM mean is required"):
        spec_from([[0.3]], [[0.05]], [np.nan], [np.nan])


# =================================================================================================
# 3  reproducibility and fingerprints
# =================================================================================================

def _base_cfg():
    spec = spec_from([[0.30, 0.05], [0.12, 0.40]], [[0.05, 0.02], [0.02, 0.04]], [0.35, 0.88], [0.03, 0.01],
                     ids=("A", "B"), nuts=("X", "Y"),
                     correlation={"structure_id": "syn_c", "labels": [["A", "X"], ["A", "Y"]],
                                  "matrix": [[1.0, -0.4], [-0.4, 1.0]], "input_scale": "latent_gaussian",
                                  "handling": "latent_as_declared", "status": "synthetic_test_only",
                                  "provenance": "synthetic", "source_id": None, "locator": None})
    return spec.to_config()


def _mutations():
    def cell(cfg, key):
        return next(c for c in cfg["cells"] if (c["ingredient_id"], c["component"]) == key)

    def setc(key, **kv):
        def f(cfg):
            cell(cfg, key).update(kv)
        return f

    def setr(**kv):
        def f(cfg):
            cfg["family_rule"].update(kv)
        return f

    def setcorr(**kv):
        def f(cfg):
            cfg["correlation"].update(kv)
        return f

    def top(**kv):
        def f(cfg):
            cfg.update(kv)
        return f

    return {
        "mean": setc(("A", "X"), mean=0.31), "sd": setc(("A", "X"), sd=0.051),
        "lower": setc(("A", "X"), lower=0.01), "upper": setc(("A", "X"), upper=0.9),
        "dm_mean": setc(("B", "DM"), mean=0.87), "dm_sd": setc(("B", "DM"), sd=0.011),
        "family_override": setc(("A", "Y"), family="BETA_MM", family_choice_reason="synthetic"),
        "primary_family": setr(primary_family="BETA_MM"), "fallback": setr(fallback_families=["LN_MM"]),
        "on_exhausted": setr(on_exhausted="exclude_ingredient"), "rule_status": setr(status="research_scenario_assumption"),
        "rule_rationale": setr(rationale="other reason"),
        "variance_basis": setc(("A", "X"), variance_basis="unidentified"),
        "data_fingerprint": setc(("A", "X"), data_fingerprint="synthetic:other"),
        "decomposition": setc(("A", "X"), decomposition_id="syn_dec", decomposition_source="synthetic",
                              measurement_model_id="syn_lab"),
        "measurement_model": setc(("A", "X"), variance_basis="observed_incl_lab_only", measurement_model_id="syn_lab"),
        "provenance": setc(("A", "X"), provenance_status="research_scenario_assumption"),
        "source_locator": setc(("A", "X"), source_id="SYN-S", locator="p.1"),
        "correlation_value": setcorr(matrix=[[1.0, -0.41], [-0.41, 1.0]]),
        "correlation_scale": setcorr(input_scale="transformed_pearson", handling="declared_latent_scenario"),
        "correlation_status": setcorr(status="research_scenario_assumption"),
        "purpose": top(purpose="sensitivity_scenario"), "spec_id": top(spec_id="syn2"), "notes": top(notes="x"),
    }


def test_fixed_spec_and_streams_reproduce_the_draws_bitwise():
    cfg = _base_cfg()
    m1 = build_uncertainty_model(UncertaintySpec.from_config(cfg))
    m2 = build_uncertainty_model(UncertaintySpec.from_config(copy.deepcopy(cfg)))
    assert m1.fingerprint() == m2.fingerprint() and m1.spec.fingerprint() == m2.spec.fingerprint()
    assert UncertaintySpec.from_config(m1.spec.to_config()).fingerprint() == m1.spec.fingerprint()   # round trip
    for stream in ("opt", "validation", "test"):
        a, b = m1.draw(RandomStreams(1103), stream, 300), m2.draw(RandomStreams(1103), stream, 300)
        assert np.array_equal(a.theta, b.theta) and np.array_equal(a.d, b.d) and a.fingerprint == b.fingerprint
    c = m1.draw(RandomStreams(2207), "test", 300)
    assert not np.array_equal(c.theta, m1.draw(RandomStreams(1103), "test", 300).theta)


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_every_relevant_field_changes_the_model_fingerprint(name):
    base = build_uncertainty_model(UncertaintySpec.from_config(_base_cfg()))
    cfg = _base_cfg()
    _mutations()[name](cfg)
    m = build_uncertainty_model(UncertaintySpec.from_config(cfg))
    assert m.fingerprint() != base.fingerprint(), name
    assert m.spec.main_fingerprint() != base.spec.main_fingerprint(), name


# =================================================================================================
# 4  one evaluation world
# =================================================================================================

def _world_spec(sd_scale=1.0, spec_id="syn_world"):
    pr = two_ingredient_problem()
    th, dh = pr.nominal_theta(), pr.dm_estimates()
    return pr, spec_from(th, np.array([[0.01], [0.02]]) * sd_scale, dh, np.array([0.01, 0.01]) * sd_scale,
                         ids=pr.ingredient_ids, nuts=pr.nutrient_ids, spec_id=spec_id)


def test_opt_validation_test_box_and_prior_come_from_one_model():
    pr, spec = _world_spec()
    m = build_uncertainty_model(spec)
    assert m.check_problem(pr)["max_abs_theta_nominal_minus_target"] == 0.0
    w = m.draw_world(RandomStreams(1103), n_opt=64, n_validation=500, n_test=500)
    assert {d.stream for d in w.values()} == {"opt", "validation", "test"}
    assert {d.model_fingerprint for d in w.values()} == {m.fingerprint()}
    m.assert_same_world(*w.values())
    assert metadata_for_drawset(w["test"]) is m.metadata
    other = build_uncertainty_model(_world_spec(1.5, "syn_world_b")[1])
    with pytest.raises(WorldMismatchError):
        m.assert_same_world(other.draw(RandomStreams(1103), "test", 10))
    # robust set: sampled moments of the same world, tied to its fingerprint
    box = m.box(2.0)
    assert box.construction == "factory_model_moment_pm_k_sd"
    assert box.construction_params["model_fingerprint"] == m.fingerprint()
    assert box.construction_params["moment_basis"] == "achieved_moments_of_sampled_marginals"
    tm, ts, dm, ds = m.achieved_moments()
    np.testing.assert_allclose(box.theta_lo, tm - 2 * ts, rtol=0, atol=1e-15)
    np.testing.assert_allclose(box.d_hi, dm + 2 * ds, rtol=0, atol=1e-15)
    r = solve_box_robust(pr, params={"uncertainty_set": box})
    assert str(r.status) == "optimal"
    # information prior: development streams only, variance basis read from the object
    prior = m.prior_states(RandomStreams(1103), "opt", 64)
    assert isinstance(prior, PriorStates) and prior.source_fingerprint == w["opt"].fingerprint
    with pytest.raises(LeakageError):
        m.prior_states(RandomStreams(1103), "test", 64)
    vb = m.prior_variance_basis()
    assert vb[ObservedComponent("F", "CP")] == "true_batch_state" and ObservedComponent("C", "DM") in vb
    with pytest.raises(KeyError):
        m.prior_variance_basis([("F", "NDF")])
    # nominal method plans with the target means
    tn, dn = m.nominal_state()
    np.testing.assert_array_equal(tn, pr.nominal_theta())
    np.testing.assert_array_equal(dn, pr.dm_estimates())


def test_smoke_pipeline_uses_one_world_and_refuses_mixed_draws():
    pr, spec = _world_spec()
    m, w = SP.build_world(spec, RandomStreams(1103), n_opt=64, n_validation=300, n_test=300)
    assert isinstance(m, FactoryModel) and set(w) == {"opt", "validation", "test"}
    other = build_uncertainty_model(_world_spec(1.5, "syn_world_b")[1])
    specs = (SP.MethodSpec("M0_nominal", uses_opt_draws=False), SP.MethodSpec("M3c_scenario_set_robust"))
    runs = SP.run_methods(pr, specs, opt_draws=w["opt"])
    assert {r.world_fingerprint for r in runs} == {m.fingerprint()}
    SP.evaluate_on_test(runs, pr, w["test"])
    assert all(r.evaluation is not None and r.world_shift_scenario is None for r in runs)
    with pytest.raises(WorldMismatchError):
        SP.run_methods(pr, specs, opt_draws=w["opt"], validation_draws=other.draw(RandomStreams(1103), "validation", 50))
    runs2 = SP.run_methods(pr, specs, opt_draws=w["opt"])
    with pytest.raises(WorldMismatchError):
        SP.evaluate_on_test(runs2, pr, other.draw(RandomStreams(1103), "test", 300))
    SP.evaluate_on_test(runs2, pr, other.draw(RandomStreams(1103), "test", 300), world_shift_scenario="E2_syn_shift")
    assert all(r.world_shift_scenario == "E2_syn_shift" for r in runs2)
    with pytest.raises(TypeError, match="uncertainty factory"):
        SP.build_world(IndependentNormalModel("n", ("F",), ("CP",), [[0.1]], [[0.01]], [0.4], [0.01]),
                       RandomStreams(1103), n_opt=2)


# =================================================================================================
# 5  Gaussian copula: latent vs transformed correlation
# =================================================================================================

def _corr_spec(r, input_scale, handling):
    return spec_from([[0.02, 0.05]], [[0.015, 0.04]], [0.5], [0.0], nuts=("X", "Y"),
                     correlation={"structure_id": "syn_pair", "labels": [["A", "X"], ["A", "Y"]],
                                  "matrix": [[1.0, r], [r, 1.0]], "input_scale": input_scale, "handling": handling,
                                  "status": "synthetic_test_only", "provenance": "synthetic literature-like r",
                                  "source_id": None, "locator": None})


def _mc_pearson(m, n=200_000):
    ds = m.draw(RandomStreams(1103), "validation", n)
    return float(np.corrcoef(ds.theta[:, 0, 0], ds.theta[:, 0, 1])[0, 1])


def test_latent_correlation_differs_from_the_transformed_pearson_and_both_are_reported():
    m = build_uncertainty_model(_corr_spec(-0.8, "latent_gaussian", "latent_as_declared"))
    cm = m.metadata.correlation
    assert cm.latent_correlation[0][1] == -0.8 and cm.input_matrix[0][1] == -0.8
    tr = cm.transformed_correlation[0][1]
    assert abs(tr - (-0.8)) > 0.02                        # skewed marginals: not the same number
    assert tr == pytest.approx(_mc_pearson(m), abs=0.01)  # quadrature = what the draws show
    assert cm.max_abs_latent_minus_transformed == pytest.approx(abs(tr + 0.8))
    assert "latent" in cm.interpretation


def test_literature_pearson_declared_scenario_vs_calibrated_mapping():
    dec = build_uncertainty_model(_corr_spec(-0.6, "transformed_pearson", "declared_latent_scenario"))
    cal = build_uncertainty_model(_corr_spec(-0.6, "transformed_pearson", "calibrated_mapping"))
    d, c = dec.metadata.correlation, cal.metadata.correlation
    assert d.latent_correlation[0][1] == -0.6 and abs(d.transformed_correlation[0][1] + 0.6) > 0.01
    assert "by declaration" in d.interpretation
    assert c.transformed_correlation[0][1] == pytest.approx(-0.6, abs=1e-9)
    assert c.latent_correlation[0][1] < -0.6               # a stronger latent value is needed
    assert c.calibration[0][2] == -0.6 and c.calibration[0][4] == pytest.approx(-0.6, abs=1e-9)
    assert _mc_pearson(cal) == pytest.approx(-0.6, abs=0.01)
    assert dec.fingerprint() != cal.fingerprint()


def test_unattainable_pearson_and_non_psd_latent_matrix_are_refused_without_repair():
    with pytest.raises(FactoryBuildError, match="attainable range"):
        build_uncertainty_model(_corr_spec(-0.999, "transformed_pearson", "calibrated_mapping"))
    bad = [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
    spec = spec_from([[0.3, 0.2, 0.4]], [[0.05, 0.03, 0.04]], [0.5], [0.0], nuts=("X", "Y", "Z"),
                     correlation={"structure_id": "bad", "labels": [["A", "X"], ["A", "Y"], ["A", "Z"]], "matrix": bad,
                                  "input_scale": "latent_gaussian", "handling": "latent_as_declared",
                                  "status": "synthetic_test_only", "provenance": "synthetic", "source_id": None,
                                  "locator": None})
    with pytest.raises(FactoryBuildError, match="not PSD"):
        build_uncertainty_model(spec)
    with pytest.raises(ConfigValidationError, match="not a latent correlation"):
        _corr_spec(-0.6, "transformed_pearson", "latent_as_declared")
    with pytest.raises(ConfigValidationError, match="latent_as_declared"):
        _corr_spec(-0.6, "latent_gaussian", "calibrated_mapping")


def test_quadrature_helpers_agree_with_monte_carlo():
    fa = fit_marginal("TN_MM", 0.02, 0.015, 0.0, 1.0)
    fb = fit_marginal("LN_MM", 0.05, 0.04, 0.0, 1.0)
    assert transformed_pearson(fa, fb, 0.0) == pytest.approx(0.0, abs=1e-12)
    rho, (lo, hi) = calibrate_latent_rho(fa, fb, 0.5)
    assert transformed_pearson(fa, fb, rho) == pytest.approx(0.5, abs=1e-10) and lo < 0 < 0.5 < hi < 1


# =================================================================================================
# 6  two-layer farm model: extension scenario only
# =================================================================================================

def _tl(role="extension_scenario", ratio_status="research_scenario_assumption", allow=False):
    return {"role": role, "ratios": [["A", "X", 0.5], ["A", "DM", 0.5]], "ratio_basis": "true_only",
            "ratio_status": ratio_status, "ratio_source": "synthetic", "s": 1.0, "family": "TN_MM",
            "uncovered_cells": "P0", "allow_unidentified_scenario": allow}


OBS = dict(DEFAULTS, variance_basis="observed_incl_sampling_and_lab", measurement_model_id="syn_lab")


def test_unidentified_two_layer_model_is_not_a_main_dependency():
    plain = build_uncertainty_model(spec_from([[0.3]], [[0.05]], [0.4], [0.02], defaults=OBS))
    m = build_uncertainty_model(spec_from([[0.3]], [[0.05]], [0.4], [0.02], defaults=OBS, two_layer=_tl("primary")))
    ext = m.extension("two_layer_farm")
    assert ext.metadata.status == "not_built_unidentified" and not ext.is_built
    assert ext.metadata.role == "extension_scenario" and ext.metadata.requested_role == "primary"
    assert any("demoted_from_primary" in n for n in ext.metadata.notes)
    assert any("not identified" in r for r in ext.metadata.reasons)
    # main world identical with or without the extension block
    assert m.fingerprint() == plain.fingerprint()
    a, b = m.draw(RandomStreams(1103), "test", 200), plain.draw(RandomStreams(1103), "test", 200)
    assert np.array_equal(a.theta, b.theta) and np.array_equal(a.d, b.d)
    assert m.spec.fingerprint() != plain.spec.fingerprint()          # the extension config itself is tracked
    allowed = build_uncertainty_model(spec_from([[0.3]], [[0.05]], [0.4], [0.02], defaults=OBS,
                                                two_layer=_tl(allow=True)))
    e2 = allowed.extension("two_layer_farm")
    assert e2.metadata.status == "unidentified_extension_scenario" and isinstance(e2.model, TwoLayerFarmModel)
    assert e2.model.between_layer_contains_measurement_error
    assert allowed.fingerprint() == plain.fingerprint()


def test_identified_two_layer_extension_requires_true_state_basis_and_sourced_ratios():
    true_defaults = dict(DEFAULTS, decomposition_id="syn_deconvolution", decomposition_source="synthetic test",
                         measurement_model_id="syn_lab")
    m = build_uncertainty_model(spec_from([[0.3]], [[0.05]], [0.4], [0.02], defaults=true_defaults,
                                          two_layer=_tl(ratio_status="sourced")))
    e = m.extension("two_layer_farm")
    assert e.metadata.status == "identified_extension_scenario" and e.metadata.truth_variance_basis == "true_batch_state"
    assert not e.model.between_layer_contains_measurement_error


# =================================================================================================
# 7  family selection on development data only
# =================================================================================================

def test_family_selection_uses_development_data_only():
    spec = one_cell(0.05, 0.04)
    cell = spec.cell("A", "X")
    x = fit_marginal("LN_MM", 0.05, 0.04, 0.0, 1.0).sample(np.random.default_rng(7), 5000)
    s = select_family_on_development_data(cell, x, split="development", data_label="synthetic LN sample")
    assert s.chosen == "LN_MM" and s.split == "development" and s.n_obs == 5000
    assert {f for f, _, _ in s.scores} == {"TN_MM", "LN_MM", "BETA_MM"}
    for split in ("test", "official_test", "outer", None):
        with pytest.raises(LeakageError):
            select_family_on_development_data(cell, x, split=split, data_label="x")
    m = build_uncertainty_model(one_cell(0.05, 0.04))
    with pytest.raises(LeakageError):
        select_family_on_development_data(cell, m.draw(RandomStreams(1103), "test", 100), data_label="x")
    s2 = select_family_on_development_data(cell, m.draw(RandomStreams(1103), "opt", 500), data_label="opt draws")
    assert s2.split == "opt" and "opt" in DEVELOPMENT_SPLITS


def test_factory_uses_only_development_selections():
    rule = dict(RULE, selection_basis="development_data")
    spec = one_cell(0.05, 0.04, rule=rule)
    cell = spec.cell("A", "X")
    with pytest.raises(FactoryBuildError, match="no development-data selection"):
        build_uncertainty_model(spec)
    x = fit_marginal("LN_MM", 0.05, 0.04, 0.0, 1.0).sample(np.random.default_rng(7), 5000)
    s = select_family_on_development_data(cell, x, split="development", data_label="synthetic LN sample")
    m = build_uncertainty_model(spec, family_selection={("A", "X"): s})
    c = m.metadata.cell("A", "X")
    assert c.family == "LN_MM" and "development-data selection" in c.family_choice_reason
    assert m.metadata.family_selections[0].split == "development"
    forged = FamilySelection(**{**s.__dict__, "split": "test"})
    with pytest.raises(FactoryBuildError, match="not development data"):
        build_uncertainty_model(spec, family_selection={("A", "X"): forged})
    with pytest.raises(FactoryBuildError, match="declared_rule"):
        build_uncertainty_model(one_cell(0.05, 0.04), family_selection={("A", "X"): s})
    alien = FamilySelection(**{**s.__dict__, "ingredient_id": "Z"})
    with pytest.raises(FactoryBuildError, match="cell of the specification"):
        build_uncertainty_model(spec, family_selection={("A", "X"): s, ("Z", "X"): alien})


# =================================================================================================
# 8  metadata: vocabulary, completeness, no hidden defaults, public view
# =================================================================================================

def test_variance_basis_vocabulary_is_the_information_guard_vocabulary():
    assert tuple(VARIANCE_BASES) == tuple(PRIOR_VARIANCE_BASES) == tuple(TRUTH_SD_BASES)


REQUIRED_CELL_KEYS = {"variance_basis", "data_fingerprint", "decomposition_id", "decomposition_source",
                      "measurement_model_id", "is_synthetic", "family", "requested_family", "family_choice_reason",
                      "target_mean", "target_sd", "achieved_mean", "achieved_sd", "lower", "upper", "params",
                      "fit_status", "attempts", "provenance_status", "source_id", "locator"}


def test_every_cell_carries_the_required_metadata():
    m = build_uncertainty_model(UncertaintySpec.from_config(_base_cfg()))
    d = m.metadata.to_dict(include_values=True)
    assert len(d["cells"]) == 2 * 3
    for c in d["cells"]:
        assert REQUIRED_CELL_KEYS <= set(c), set(REQUIRED_CELL_KEYS) - set(c)
    assert d["metadata_fingerprint"] == m.metadata.fingerprint() and d["correlation"]["transformed_correlation"]
    pub = m.metadata.to_dict(include_values=False)
    assert not pub["values_included"]
    for c in pub["cells"]:
        assert not ({"target_mean", "target_sd", "achieved_mean", "achieved_sd", "params"} & set(c))
    assert "latent_correlation" not in pub["correlation"]


def test_no_hidden_defaults_and_consistency_rules():
    with pytest.raises(ConfigValidationError, match="no hidden defaults"):
        one_cell(defaults={k: v for k, v in DEFAULTS.items() if k != "variance_basis"})
    with pytest.raises(ConfigValidationError, match="needs a variance decomposition"):
        one_cell(is_synthetic=False, defaults=dict(DEFAULTS, provenance_status="research_scenario_assumption"),
                 rule=dict(RULE, status="research_scenario_assumption"))
    with pytest.raises(ConfigValidationError, match="measurement_model_id"):
        one_cell(defaults=dict(DEFAULTS, variance_basis="observed_incl_sampling_and_lab"))
    with pytest.raises(ConfigValidationError, match="source_id and locator"):
        one_cell(defaults=dict(DEFAULTS, provenance_status="sourced"))
    with pytest.raises(ConfigValidationError, match="synthetic_test_only value in a non-synthetic"):
        one_cell(is_synthetic=False, defaults=dict(OBS), rule=dict(RULE, status="research_scenario_assumption"))
    with pytest.raises(ConfigValidationError, match="data_fingerprint required"):
        one_cell(defaults=dict(DEFAULTS, data_fingerprint=None))
    with pytest.raises(ConfigValidationError, match="must be one of"):
        one_cell(defaults=dict(DEFAULTS, variance_basis="whatever"))
    cfg = one_cell().to_config()
    cfg["cells"] = cfg["cells"][:-1]
    with pytest.raises(ConfigValidationError, match="listed explicitly"):
        UncertaintySpec.from_config(cfg)
    cfg = one_cell().to_config()
    cfg["extra_key"] = 1
    with pytest.raises(ConfigValidationError, match="unknown key"):
        UncertaintySpec.from_config(cfg)


def test_point_and_missing_cells_are_not_given_a_variance_basis():
    spec = spec_from([[0.3, np.nan]], [[0.0, np.nan]], [0.5], [0.0], nuts=("X", "Y"))
    m = build_uncertainty_model(spec)
    assert m.metadata.cell("A", "X").fit_status == "point" and m.metadata.cell("A", "Y").fit_status == "missing"
    assert m.metadata.cell("A", "X").variance_basis == "not_applicable"
    assert m.metadata.variance_basis_map() == {}
    ds = m.draw(RandomStreams(1103), "opt", 5)
    assert np.all(ds.theta[:, 0, 0] == 0.3) and np.all(np.isnan(ds.theta[:, 0, 1]))


# =================================================================================================
# 9  static guard: no private naive / independent-normal model entry points in the study code
# =================================================================================================

#: Allowed occurrences outside tests/ (definitions, the factory's family mapping, the spec's error
#: message, and the phase-3 script's labelled drift diagnostic).  New entry points must go through
#: the factory (review R3); tests keep synthetic independent-normal fixtures.
_AUDIT = "scripts/audit_uncertainty_factory.py"      # its inventory pattern table names the patterns
_ALLOWED = {
    "IndependentNormalModel(": {"src/ration_reliability/uncertainty/models.py", _AUDIT},
    "naive_truncnorm(": {"src/ration_reliability/uncertainty/distributions.py", _AUDIT},
    '"TN_NAIVE"': {"src/ration_reliability/uncertainty/distributions.py",
                   "src/ration_reliability/uncertainty/factory.py",
                   "src/ration_reliability/uncertainty/spec.py",
                   "scripts/build_phase3_tables.py", _AUDIT},
}


def test_no_new_private_naive_model_entry_points():
    found: dict[str, set[str]] = {k: set() for k in _ALLOWED}
    for sub in ("src", "experiments", "scripts"):
        for p in (REPO / sub).rglob("*.py"):
            text = p.read_text(encoding="utf-8")
            rel = str(p.relative_to(REPO))
            for pat in _ALLOWED:
                if re.search(re.escape(pat), text):
                    found[pat].add(rel)
    for pat, allowed in _ALLOWED.items():
        assert found[pat] <= allowed, (pat, sorted(found[pat] - allowed))
