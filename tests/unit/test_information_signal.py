"""Signal model, double-count guard, binning, likelihoods and prior-state leakage (synthetic values)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ration_reliability.datamodel import AssaySpec, Provenance, SourcedValue, ValueStatus
from ration_reliability.errors import InvalidProblemError, LeakageError
from ration_reliability.information import (
    ComponentErrorModel,
    ObservedComponent,
    PriorStates,
    SamplingProtocol,
    SignalBinning,
    SignalModel,
    bin_occupancy,
    deconvolve_true_sd,
    double_count_guard,
    exact_value_groups,
    perfect_full_likelihood,
    perfect_partial_likelihood,
    quantile_edges,
    sample_likelihood_analytic,
    sample_likelihood_monte_carlo,
    signal_model_from_assay,
)
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams

SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-P8-TEST")
C1 = ObservedComponent("F", "NDF")
C2 = ObservedComponent("F", "DM")


def _err(comp=C1, s=0.02, r=0.015, b=0.01, mean=0.0, sem="repeatability_only"):
    prov = {k: SYN for k in ("sampling_sd", "lab_repeatability_sd", "lab_bias_sd", "bias_mean")}
    return ComponentErrorModel(comp, s, r, b, mean, sem, prov, is_synthetic=True)


# ------------------------------------------------------------------ variance components (T7.5, F07)

def test_replicates_reduce_only_lab_repeatability():
    e = _err()
    for r in (1, 2, 4, 100, 10_000):
        sm = SignalModel("s", (e,), SamplingProtocol(1, True, r), is_synthetic=True)
        exp = math.sqrt(0.02 ** 2 + 0.015 ** 2 / r + 0.01 ** 2)
        assert sm.total_sd()[0] == pytest.approx(exp, rel=1e-12)
    # infinite replicates cannot remove sampling error or lab bias
    sm = SignalModel("s", (e,), SamplingProtocol(1, True, 10 ** 9), is_synthetic=True)
    assert sm.total_sd()[0] == pytest.approx(math.sqrt(0.02 ** 2 + 0.01 ** 2), rel=1e-6)


def test_field_samples_reduce_sampling_error_but_not_bias():
    e = _err()
    comp = SignalModel("s", (e,), SamplingProtocol(4, True, 1), is_synthetic=True)
    sep = SignalModel("s", (e,), SamplingProtocol(4, False, 2), is_synthetic=True)
    assert comp.total_sd()[0] == pytest.approx(math.sqrt(0.02 ** 2 / 4 + 0.015 ** 2 + 0.01 ** 2))
    assert sep.total_sd()[0] == pytest.approx(math.sqrt(0.02 ** 2 / 4 + 0.015 ** 2 / 8 + 0.01 ** 2))
    assert sep.protocol.n_analyses == 8 and comp.protocol.n_analyses == 1


def test_single_result_total_is_not_reduced_by_replicates():
    e = _err(b=0.0, sem="single_result_total")
    sm = SignalModel("s", (e,), SamplingProtocol(1, True, 50), is_synthetic=True)
    assert sm.total_sd()[0] == pytest.approx(math.sqrt(0.02 ** 2 + 0.015 ** 2))


@pytest.mark.parametrize("proto", [SamplingProtocol(1, True, 1), SamplingProtocol(3, True, 2),
                                   SamplingProtocol(3, False, 2)])
def test_simulated_signal_matches_analytic_variance(proto):
    e1, e2 = _err(C1, 0.02, 0.015, 0.01, 0.005), _err(C2, 0.01, 0.004, 0.0, -0.01)
    sm = SignalModel("s", (e1, e2), proto, is_synthetic=True)
    t = np.tile([0.45, 0.35], (200_000, 1))
    z = sm.simulate(t, np.random.default_rng(5))
    err = z - t
    assert np.all(np.abs(err.mean(axis=0) - sm.bias_means()) <= 4 * sm.total_sd() / math.sqrt(2e5))
    np.testing.assert_allclose(err.std(axis=0), sm.total_sd(), rtol=0.01)


# ------------------------------------------------------------------ double counting (F06)

def test_double_count_guard_rules():
    sm = SignalModel("s", (_err(),), is_synthetic=True)
    assert double_count_guard(sm, {C1: "true_batch_state"}).status == "ok"
    rep = double_count_guard(sm, {C1: "observed_incl_sampling_and_lab"})
    assert rep.status == "violation" and "double count" in rep.messages[0]
    assert double_count_guard(sm, {C1: "observed_incl_sampling_and_lab"}, {C1: True}).status == "ok"
    assert double_count_guard(sm, {C1: "observed_incl_lab_only"}).status == "violation"
    assert double_count_guard(sm, {C1: "unidentified"}).status == "unidentified"
    assert double_count_guard(sm, {}).status == "violation"                       # basis must be declared
    # lab-only prior basis with a signal that has sampling error only: no lab part double counted
    sm_s = SignalModel("s", (_err(r=0.0, b=0.0),), is_synthetic=True)
    assert double_count_guard(sm_s, {C1: "observed_incl_lab_only"}).status == "ok"
    # single-result total SD plus a separate lab bias double counts the between-lab part
    sm_t = SignalModel("s", (_err(sem="single_result_total"),), is_synthetic=True)
    assert double_count_guard(sm_t, {C1: "true_batch_state"}).status == "violation"


def test_deconvolve_true_sd():
    assert deconvolve_true_sd(0.05, 0.03, 0.0) == pytest.approx(0.04)
    assert deconvolve_true_sd(0.05, 0.03, 0.0, n_field_samples=9) == pytest.approx(math.sqrt(0.0025 - 0.0001))
    assert deconvolve_true_sd(0.05, 0.0, 0.05) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="inconsistent"):
        deconvolve_true_sd(0.03, 0.03, 0.01)


def test_signal_model_from_assay_requires_declared_semantics_and_no_pending_values():
    sv = lambda v: SourcedValue(v, "%", SYN, "DM")   # noqa: E731
    assay = AssaySpec("wet_chem_panel", "synthetic panel", ("NDF",), ("F",), measures_dm=True,
                      sampling_error_sd={"NDF": sv(2.0), "DM": sv(1.0)}, lab_error_sd={"NDF": sv(1.5)},
                      systematic_bias={"NDF": sv(0.5)}, is_synthetic=True)
    sm = signal_model_from_assay(assay, "F", lab_error_semantics="repeatability_only",
                                 systematic_bias_semantics="sd_of_random_lab_bias")
    assert sm.components == (ObservedComponent("F", "NDF"), ObservedComponent("F", "DM"))
    e = sm.errors[0]
    assert (e.sampling_sd, e.lab_repeatability_sd, e.lab_bias_sd, e.bias_mean) == pytest.approx((0.02, 0.015, 0.005, 0))
    sm2 = signal_model_from_assay(assay, "F", lab_error_semantics="repeatability_only",
                                  systematic_bias_semantics="known_mean_offset")
    assert sm2.errors[0].bias_mean == pytest.approx(0.005) and sm2.errors[0].lab_bias_sd == 0.0
    with pytest.raises(InvalidProblemError):
        signal_model_from_assay(assay, "F", lab_error_semantics="guess", systematic_bias_semantics="known_mean_offset")
    with pytest.raises(InvalidProblemError, match="not applicable"):
        signal_model_from_assay(assay, "C", lab_error_semantics="repeatability_only",
                                systematic_bias_semantics="known_mean_offset")
    pend = AssaySpec("p", "pending", ("NDF",), sampling_error_sd={
        "NDF": SourcedValue(None, "%", Provenance(ValueStatus.PENDING_USER_DECISION, rationale="no source yet"), "DM")})
    with pytest.raises(InvalidProblemError, match="pending"):
        signal_model_from_assay(pend, "F", lab_error_semantics="repeatability_only",
                                systematic_bias_semantics="known_mean_offset")


def test_provenance_issues_are_reported():
    e = ComponentErrorModel(C1, 0.02, 0.0)                      # non-zero value without provenance
    sm = SignalModel("s", (e,))
    assert any("without provenance" in m for m in sm.provenance_issues())
    with pytest.raises(InvalidProblemError):
        SignalModel("s", (_err(),), is_synthetic=False)          # synthetic parts in a non-synthetic model
    with pytest.raises(InvalidProblemError):
        SignalModel("s", (_err(),), error_scale="multiplicative", is_synthetic=True)


# ------------------------------------------------------------------ binning and likelihoods

def _prior(n=400, seed=1):
    m = IndependentNormalModel("syn", ("F", "C"), ("NDF", "CP"), np.array([[0.45, 0.10], [0.20, 0.40]]),
                               np.array([[0.04, 0.01], [0.0, 0.0]]), np.array([0.35, 0.88]), np.array([0.02, 0.0]),
                               is_synthetic=True)
    return m, PriorStates.from_drawset(m.draw(RandomStreams(seed), "opt", n))


def test_bin_index_half_open_convention_and_product_grid():
    b = SignalBinning((C1, C2), ((0.4, 0.5), (0.35,)), "syn")
    assert b.n_bins == 6
    v = np.array([[0.39, 0.30], [0.40, 0.30], [0.5, 0.35], [0.7, 0.9]])
    assert b.bin_index(v).tolist() == [0, 2, 5, 5]
    with pytest.raises(InvalidProblemError):
        SignalBinning((C1,), ((0.5, 0.4),), "syn")
    with pytest.raises(InvalidProblemError, match="max_bins"):
        SignalBinning((C1,), (tuple(np.linspace(0, 1, 50)),), "syn", max_bins=10)


def test_quantile_edges_and_nested_refinement():
    v = np.arange(100.0)
    assert quantile_edges(v, 4) == (25.0, 50.0, 75.0)          # 25 values below the first edge
    assert quantile_edges(np.array([1.0, 1.0, 2.0]), 3) == (2.0,)              # duplicates removed
    b = SignalBinning((C1,), ((0.45,),), "syn").refine_nested([[0.40, 0.50]])
    assert b.interior_edges == ((0.40, 0.45, 0.50),)


def test_prior_states_refuse_test_and_outer_streams():
    m, _ = _prior()
    for s in ("test", "outer"):
        with pytest.raises(LeakageError):
            PriorStates.from_drawset(m.draw(RandomStreams(1), s, 5))
    assert PriorStates.from_drawset(m.draw(RandomStreams(1), "validation", 5)).stream == "validation"


def test_perfect_likelihoods_are_partitions_and_exact_grouping_degeneracy_is_flagged():
    _, prior = _prior(200)
    full = perfect_full_likelihood(prior)
    assert full.n_bins == 200 and np.all(full.L.sum(axis=1) == 1)
    exact = perfect_partial_likelihood(prior, (C1,))
    assert exact.kind == "perfect_partial_exact" and any("degenerate" in w for w in exact.warnings)
    b = SignalBinning((C1,), (quantile_edges(prior.component_values((C1,)), 4),), "q")
    binned = perfect_partial_likelihood(prior, (C1,), b)
    occ = bin_occupancy(prior, binned)
    assert occ["n_active_bins"] == 4 and occ["effective_states"].sum() == pytest.approx(200)
    g = exact_value_groups(np.array([[1.0], [1.0], [2.0]]))
    assert g.n_groups == 2 and not g.degenerate


def test_analytic_likelihood_limits_and_monte_carlo_agreement():
    _, prior = _prior(60)
    b = SignalBinning((C1,), ((0.42, 0.48),), "syn")
    tiny = SignalModel("t", (_err(s=1e-12, r=0.0, b=0.0),), is_synthetic=True)
    Lt = sample_likelihood_analytic(prior, tiny, b).L
    Lp = perfect_partial_likelihood(prior, (C1,), b).L
    np.testing.assert_allclose(Lt, Lp, atol=1e-9)
    huge = SignalModel("h", (_err(s=1e6, r=0.0, b=0.0),), is_synthetic=True)
    Lh = sample_likelihood_analytic(prior, huge, b).L
    np.testing.assert_allclose(Lh, np.tile([0.5, 0.0, 0.5], (60, 1)), atol=1e-6)   # state-independent
    sm = SignalModel("m", (_err(C1, 0.02, 0.015, 0.01, 0.003),), is_synthetic=True)
    La = sample_likelihood_analytic(prior, sm, b).L
    Lm = sample_likelihood_monte_carlo(prior, sm, b, RandomStreams(9), 4000).L
    assert np.max(np.abs(La - Lm)) < 0.04                           # ~4.5 MC standard errors at p = 0.5
    with pytest.raises(InvalidProblemError):
        sample_likelihood_monte_carlo(prior, sm, b, RandomStreams(9), 10, stream="test")
