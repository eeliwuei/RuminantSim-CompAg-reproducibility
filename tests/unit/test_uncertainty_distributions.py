"""Moment-matched marginal families (B-137; configs/uncertainty.yaml b_distribution_family).

All numbers are synthetic (``synthetic_test_only``).  They test that the sampled distribution
has the requested mean and SD inside the physical range, that impossible targets are reported
instead of forced, and that the naive truncated normal drifts (smoke S-5).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from ration_reliability.uncertainty.distributions import (
    ALL_FAMILIES,
    MATCH_TOL,
    MOMENT_MATCHED_FAMILIES,
    MarginalFit,
    fit_array,
    fit_beta_moments,
    fit_lognormal_moments,
    fit_marginal,
    fit_truncnorm_moments,
    naive_truncnorm,
    untruncated_normal,
)

# synthetic targets on [0, 1]: (mean, sd) with CV from tiny to close to 1, and one near the upper bound
TARGETS = [(0.35, 0.05), (0.02, 0.015), (0.02, 0.0199), (0.006, 0.0055), (0.45, 0.2), (0.93, 0.03),
           (0.88, 0.012), (0.012, 0.006)]


def _moments_by_quadrature(fit: MarginalFit) -> tuple[float, float]:
    """Mean and SD by quadrature of ``ppf(Phi(z)) phi(z)`` over z (independent of the closed forms)."""
    def g(z, k):
        return float(fit.ppf(np.array([stats.norm.cdf(z)]))[0]) ** k * stats.norm.pdf(z)
    f1 = integrate.quad(g, -12.0, 12.0, args=(1,), limit=400, epsabs=1e-14, epsrel=1e-12)[0]
    f2 = integrate.quad(g, -12.0, 12.0, args=(2,), limit=400, epsabs=1e-14, epsrel=1e-12)[0]
    return f1, math.sqrt(max(f2 - f1 * f1, 0.0))


@pytest.mark.parametrize("m,s", TARGETS)
@pytest.mark.parametrize("family", MOMENT_MATCHED_FAMILIES)
def test_moment_matched_families_reproduce_target(family, m, s):
    fit = fit_marginal(family, m, s, 0.0, 1.0)
    assert fit.status == "matched", (family, m, s, fit.status, fit.message)
    assert abs(fit.mean_error_sd) <= MATCH_TOL and abs(fit.sd_rel_error) <= MATCH_TOL
    qm, qs = _moments_by_quadrature(fit)
    assert qm == pytest.approx(m, rel=1e-6, abs=1e-9)
    assert qs == pytest.approx(s, rel=1e-5, abs=1e-9)


def test_truncnorm_mm_independent_check_with_scipy():
    fit = fit_truncnorm_moments(0.02, 0.015, 0.0, 1.0)
    assert fit.status == "matched" and "truncation_active" in fit.flags
    mu, sigma = fit.params
    a, b = (0.0 - mu) / sigma, (1.0 - mu) / sigma
    m, v = stats.truncnorm.stats(a, b, loc=mu, scale=sigma, moments="mv")
    assert float(m) == pytest.approx(0.02, rel=1e-9)
    assert math.sqrt(float(v)) == pytest.approx(0.015, rel=1e-8)
    assert fit.outside_mass > 0.1        # the parent puts real mass below 0


def test_truncnorm_cannot_have_cv_ge_1_and_is_not_forced():
    for s in (0.02, 0.03):
        fit = fit_truncnorm_moments(0.02, s, 0.0, 1.0)
        assert fit.status == "not_matchable" and "tn_cv_ge_1" in fit.flags
        with pytest.raises(ValueError):
            fit.ppf(np.array([0.5]))
    # the other families still reproduce the same target
    assert fit_beta_moments(0.02, 0.03, 0.0, 1.0).status == "matched"
    assert fit_lognormal_moments(0.02, 0.03, 0.0, 1.0).status == "matched"


def test_naive_truncnorm_drifts_and_mm_removes_the_drift():
    naive = naive_truncnorm(0.02, 0.018, 0.0, 1.0)
    assert naive.status == "drifted"
    assert naive.mean_rel_shift > 0.01            # the smoke S-5 symptom (mean moved up by > 1 %)
    assert naive.achieved_sd < 0.018
    # same quantity as the smoke script: scipy truncnorm mean of the parent (mean, sd)
    ref = stats.truncnorm.mean((0.0 - 0.02) / 0.018, (1.0 - 0.02) / 0.018, loc=0.02, scale=0.018)
    assert naive.achieved_mean == pytest.approx(float(ref), rel=1e-12)
    mm = fit_truncnorm_moments(0.02, 0.018, 0.0, 1.0)
    assert mm.status == "matched" and abs(mm.mean_rel_shift) < 1e-9


def test_far_from_bounds_all_families_agree_with_target():
    for fam in MOMENT_MATCHED_FAMILIES:
        f = fit_marginal(fam, 0.40, 0.01, 0.0, 1.0)
        assert f.status == "matched"
    tn = fit_truncnorm_moments(0.40, 0.01, 0.0, 1.0)
    assert tn.params == pytest.approx((0.40, 0.01), rel=1e-12)   # truncation irrelevant -> parent = target
    assert naive_truncnorm(0.40, 0.01, 0.0, 1.0).mean_rel_shift == pytest.approx(0.0, abs=1e-12)


def test_lognormal_upper_truncated_branch():
    fit = fit_lognormal_moments(0.93, 0.03, 0.0, 1.0)
    assert fit.status == "matched" and "upper_truncated" in fit.flags
    x = fit.sample(np.random.default_rng(3), 50_000)
    assert x.max() <= 1.0 and x.min() > 0.0


def test_beta_closed_form_and_shape_flags():
    fit = fit_beta_moments(0.02, 0.025, 0.0, 1.0)
    assert fit.status == "matched" and "beta_density_unbounded_at_lower" in fit.flags
    a, b = fit.params
    my = 0.02
    k = my * (1 - my) / 0.025 ** 2 - 1
    assert (a, b) == pytest.approx((my * k, (1 - my) * k), rel=1e-12)
    with pytest.raises(ValueError):
        fit_beta_moments(0.5, 0.1, 0.0, np.inf)


def test_bhatia_davis_infeasible_for_every_family():
    # SD^2 >= (m - a)(b - m): no distribution on [0, 1] has these moments
    for fam in MOMENT_MATCHED_FAMILIES:
        f = fit_marginal(fam, 0.5, 0.5, 0.0, 1.0)
        assert f.status == "infeasible_any_family"
        assert not f.sampleable


def test_point_missing_and_invalid_targets():
    p = fit_marginal("TN_MM", 0.3, 0.0, 0.0, 1.0)
    assert p.status == "point"
    assert np.all(p.ppf(np.array([0.1, 0.9])) == 0.3)
    miss = fit_marginal("BETA_MM", float("nan"), 0.01, 0.0, 1.0)
    assert miss.status == "missing" and np.all(np.isnan(miss.ppf(np.array([0.5]))))
    msd = fit_marginal("LN_MM", 0.3, float("nan"), 0.0, 1.0)
    assert msd.status == "missing_sd" and not msd.sampleable      # never silently NaN or 0
    with pytest.raises(ValueError):
        msd.ppf(np.array([0.5]))
    assert fit_marginal("TN_MM", 1.2, 0.01, 0.0, 1.0).status == "invalid_target"
    assert fit_marginal("TN_MM", 0.3, -0.01, 0.0, 1.0).status == "invalid_target"
    with pytest.raises(ValueError):
        fit_marginal("NOPE", 0.3, 0.01, 0.0, 1.0)
    with pytest.raises(ValueError):
        fit_marginal("TN_MM", 0.3, 0.01, 1.0, 0.0)


@pytest.mark.parametrize("family", MOMENT_MATCHED_FAMILIES)
def test_monte_carlo_moments_within_mc_error(family):
    m, s, n = 0.03, 0.02, 400_000
    fit = fit_marginal(family, m, s, 0.0, 1.0)
    assert fit.status == "matched"
    x = fit.sample(np.random.default_rng(1103), n)
    assert x.min() >= 0.0 and x.max() <= 1.0
    se_mean = s / math.sqrt(n)
    assert abs(x.mean() - m) < 5 * se_mean
    # SD: a generous bound (5 x the normal-theory SE times a kurtosis allowance)
    assert abs(x.std(ddof=1) - s) < 5 * 3 * s / math.sqrt(2 * n)


def test_sampling_reproducible_and_monotone():
    fit = fit_marginal("TN_MM", 0.02, 0.015, 0.0, 1.0)
    a = fit.sample(np.random.default_rng(7), 1000)
    b = fit.sample(np.random.default_rng(7), 1000)
    assert np.array_equal(a, b)
    u = np.linspace(0.001, 0.999, 200)
    q = fit.ppf(u)
    assert np.all(np.diff(q) >= 0)


def test_untruncated_normal_diagnostic_mass():
    d = untruncated_normal(0.02, 0.02, 0.0, 1.0)
    assert d.status == "diagnostic_only" and not d.sampleable
    assert d.outside_mass == pytest.approx(stats.norm.cdf(-1.0) + stats.norm.sf(49.0), rel=1e-12)


def test_fit_array_shapes_and_mapping():
    mean = np.array([[0.3, 0.02], [float("nan"), 0.5]])
    sd = np.array([[0.03, 0.015], [0.01, 0.0]])
    out = fit_array("BETA_MM", mean, sd, 0.0, 1.0)
    assert out.shape == (2, 2)
    assert [o.status for o in out.ravel()] == ["matched", "matched", "missing", "point"]
    fams = {(0, 0): "TN_MM", (0, 1): "LN_MM", (1, 0): "TN_MM", (1, 1): "BETA_MM"}
    out2 = fit_array(fams, mean, sd, 0.0, 1.0)
    assert out2[0, 1].family == "LN_MM"
    assert set(ALL_FAMILIES) >= {"TN_MM", "LN_MM", "BETA_MM", "TN_NAIVE", "N_UNTRUNCATED"}
