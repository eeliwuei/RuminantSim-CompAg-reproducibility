"""Correlation structures: PSD assertion tests (B-133), repair logging, candidates and the copula.

All matrices are synthetic.  The "old C4" matrix uses the design values +-0.9 of the superseded
C4 definition (configs/uncertainty.yaml C4_extreme.replaced_definition), not any published
correlation; its eigenvalue and Higham repair were pre-computed once in FIX_docs
(psd_precheck_20260924) and are reproduced here as an instrument check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ration_reliability.uncertainty import DrawSet, RandomStreams
from ration_reliability.uncertainty.correlation import (
    CorrelationStructure,
    GaussianCopulaModel,
    NotPSDError,
    PendingValueError,
    assert_psd,
    check_psd,
    closure_strength_for_floor,
    combine_structures,
    conservation_closure_matrix,
    conservation_closure_structure,
    extreme_scaling,
    independent_structure,
    induced_pearson,
    min_eigenvalue,
    nearest_correlation,
    pairwise_structure,
    repair_correlation,
    sampling_factor,
    validate_correlation,
    with_cross_correlation,
)
from ration_reliability.uncertainty.distributions import fit_marginal

SYN = "synthetic_test_only"
OLD_C4 = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, -0.9], [0.0, -0.9, 1.0]])   # (CP, NDF, starch) design values
CELLS3 = (("x", "CP"), ("x", "NDF"), ("x", "starch"))


# ----------------------------------------------------------------------------------------------
# validation and PSD assertions
# ----------------------------------------------------------------------------------------------

def test_validate_rejects_malformed_matrices():
    with pytest.raises(ValueError):
        validate_correlation(np.array([[1.0, 0.2], [0.3, 1.0]]))      # asymmetric
    with pytest.raises(ValueError):
        validate_correlation(np.array([[1.0, 0.2], [0.2, 0.9]]))      # diagonal != 1
    with pytest.raises(ValueError):
        validate_correlation(np.array([[1.0, 1.2], [1.2, 1.0]]))      # |rho| > 1
    with pytest.raises(ValueError):
        validate_correlation(np.ones((2, 3)))
    with pytest.raises(ValueError):
        CorrelationStructure("s", (("a", "CP"), ("a", "CP")), np.eye(2), SYN, "t")   # duplicate labels
    with pytest.raises(ValueError):
        CorrelationStructure("s", CELLS3, np.eye(3), "made_up_status", "t")


def test_assert_psd_rejects_old_c4_and_reports_lambda_min():
    lam = 1.0 - 0.9 * math.sqrt(2.0)       # analytic smallest eigenvalue
    assert min_eigenvalue(OLD_C4) == pytest.approx(lam, abs=1e-12)
    assert lam == pytest.approx(-0.2728, abs=1e-4)                 # FIX_docs precheck value
    st = CorrelationStructure("C4_old_design", CELLS3, OLD_C4, SYN, "superseded C4 definition")
    assert not st.psd.is_psd
    with pytest.raises(NotPSDError, match="lambda_min"):
        assert_psd(st)
    assert assert_psd(np.eye(3)).is_psd
    assert check_psd(np.array([[1.0, 1.0], [1.0, 1.0]])).is_psd      # singular PSD is accepted


def test_repair_refused_by_default_for_preregistered_structure():
    st = CorrelationStructure("C4_old_design", CELLS3, OLD_C4, SYN, "t")
    with pytest.raises(NotPSDError):
        repair_correlation(st)


def test_repair_log_reproduces_fix_docs_precheck():
    st = CorrelationStructure("C4_old_design", CELLS3, OLD_C4, SYN, "t")
    rep, rec = repair_correlation(st, allow=True)
    assert rec.repaired and rec.converged
    assert rec.lambda_min_before == pytest.approx(-0.2728, abs=1e-4)
    assert rec.lambda_min_after >= -1e-9
    assert rec.frobenius_diff == pytest.approx(0.3468, abs=1e-3)
    assert rep.rho(("x", "CP"), ("x", "NDF")) == pytest.approx(0.743, abs=1e-3)
    assert rep.rho(("x", "NDF"), ("x", "starch")) == pytest.approx(-0.743, abs=1e-3)
    assert rep.rho(("x", "CP"), ("x", "starch")) == pytest.approx(-0.105, abs=1e-3)   # invented by the repair
    assert len(rec.flagged_pairs) == 3 and rec.flag_threshold == 0.05
    assert any("higham_repaired" in n for n in rep.notes)
    assert rep.structure_id.endswith("+higham_repaired")


def test_repair_leaves_psd_structure_unchanged():
    st = independent_structure(CELLS3, status=SYN, provenance="t")
    rep, rec = repair_correlation(st, allow=True)
    assert rep is st and not rec.repaired and rec.frobenius_diff == 0.0


def test_nearest_correlation_higham_2002_example():
    a = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    x, _, conv = nearest_correlation(a)
    assert conv
    exp = np.array([[1.0, 0.7607, 0.1573], [0.7607, 1.0, 0.7607], [0.1573, 0.7607, 1.0]])
    assert np.allclose(x, exp, atol=1e-4)


def test_nearest_correlation_properties_on_random_indefinite_matrices():
    rng = np.random.default_rng(2207)
    for _ in range(20):
        n = int(rng.integers(3, 7))
        a = rng.uniform(-1, 1, (n, n))
        a = (a + a.T) / 2
        np.fill_diagonal(a, 1.0)
        x, _, conv = nearest_correlation(a)
        assert conv
        assert min_eigenvalue(x) >= -1e-9
        assert np.allclose(np.diag(x), 1.0) and np.allclose(x, x.T)
        # the identity is a correlation matrix, so the nearest one is at least as close
        assert np.linalg.norm(x - a) <= np.linalg.norm(np.eye(n) - a) + 1e-9
        if min_eigenvalue(a) >= 0:
            assert np.allclose(x, a, atol=1e-9)        # PSD input is a fixed point


# ----------------------------------------------------------------------------------------------
# candidate builders
# ----------------------------------------------------------------------------------------------

def test_pairwise_structure_pending_value_never_becomes_zero():
    with pytest.raises(PendingValueError):
        pairwise_structure("x", ("CP", "NDF", "starch"), {("CP", "NDF"): None}, structure_id="C1_yoder_table6",
                           status="pending_user_decision", provenance="page locator missing")
    with pytest.raises(ValueError):
        pairwise_structure("x", ("CP", "NDF"), {("NDF", "lignin"): 0.1}, structure_id="C1", status=SYN,
                           provenance="t")
    st = pairwise_structure("x", ("CP", "NDF", "starch"), {("CP", "NDF"): -0.4, ("starch", "NDF"): 0.05},
                            structure_id="C1", status=SYN, provenance="t", zero_pairs=[("starch", "NDF")])
    assert st.rho(("x", "CP"), ("x", "NDF")) == -0.4
    assert st.rho(("x", "starch"), ("x", "NDF")) == 0.0
    assert any("forced_zero" in n for n in st.notes)
    assert_psd(st)


@pytest.mark.parametrize("r1,binding", [
    (np.array([[1.0, 0.6, 0.0], [0.6, 1.0, -0.2], [0.0, -0.2, 1.0]]), "rho_cap"),
    (np.array([[1.0, -0.37, 0.0], [-0.37, 1.0, -0.26], [0.0, -0.26, 1.0]]), "lambda_floor"),
])
def test_extreme_scaling_is_psd_by_construction(r1, binding):
    r, c, b, lam = extreme_scaling(r1, rho_cap=0.9, lambda_floor=0.01)
    assert b == binding
    assert lam >= 0.01 - 1e-12
    off = r - np.eye(3)
    assert np.max(np.abs(off)) <= 0.9 + 1e-12
    assert np.allclose(off, c * (r1 - np.eye(3)))          # direction and zero pattern kept
    # closed-form c_psd agrees with the eigenvalues of R(c)
    c_psd = (1 - 0.01) / (1 - min_eigenvalue(r1))
    assert min_eigenvalue(np.eye(3) + c_psd * (r1 - np.eye(3))) == pytest.approx(0.01, abs=1e-12)
    assert_psd(r)


def test_extreme_scaling_rejects_non_psd_input_and_handles_identity():
    with pytest.raises(NotPSDError):
        extreme_scaling(OLD_C4)
    r, c, b, lam = extreme_scaling(np.eye(3))
    assert b == "identity" and c == 0.0 and np.array_equal(r, np.eye(3))


def test_conservation_closure_psd_negative_and_monotone():
    sd = np.array([0.02, 0.05, 0.06, 0.01])
    assert np.array_equal(conservation_closure_matrix(sd, 0.0), np.eye(4))
    prev = None
    for c in (0.0, 0.2, 0.5, 0.8, 0.95, 0.999):
        r = conservation_closure_matrix(sd, c)
        assert min_eigenvalue(r) > 0.0                       # positive definite for c < 1
        assert np.all(r[~np.eye(4, dtype=bool)] <= 0.0)      # closure implies non-positive correlations
        ratio = sd @ r @ sd / (sd @ sd)                      # variance of the sum relative to independence
        if prev is not None:
            assert ratio < prev
        prev = ratio
    with pytest.raises(ValueError):
        conservation_closure_matrix(sd, 1.0)
    with pytest.raises(ValueError):
        conservation_closure_matrix(np.array([0.02, float("nan")]), 0.5)
    c_max = closure_strength_for_floor(sd, 0.01)
    assert min_eigenvalue(conservation_closure_matrix(sd, c_max)) >= 0.01 - 1e-9


def test_conservation_closure_refuses_nested_components():
    with pytest.raises(ValueError, match="overlap"):
        conservation_closure_structure("x", ("ADF", "NDF"), [0.02, 0.03], 0.5)
    with pytest.raises(ValueError):
        conservation_closure_structure("x", ("TFA", "EE"), [0.01, 0.01], 0.5)
    st = conservation_closure_structure("x", ("CP", "NDF", "starch", "EE"), [0.02, 0.05, 0.06, 0.01], 0.5)
    assert any("partial_overlap" in n for n in st.notes)
    assert st.status == "research_scenario_assumption"
    assert_psd(st)


def test_cross_correlation_c5_psd_with_independent_base_and_detects_conflict():
    base = combine_structures([independent_structure((("s1", "DM"),), status=SYN, provenance="t"),
                               independent_structure((("s2", "DM"),), status=SYN, provenance="t")],
                              "C0", status=SYN, provenance="t")
    c5 = with_cross_correlation(base, {(("s1", "DM"), ("s2", "DM")): 0.3}, structure_id="C5", status=SYN,
                                provenance="t")
    assert min_eigenvalue(c5.matrix) == pytest.approx(0.7, abs=1e-12)
    with pytest.raises(PendingValueError):
        with_cross_correlation(base, {(("s1", "DM"), ("s2", "DM")): None}, structure_id="C5", status=SYN,
                               provenance="t")
    # strong within-ingredient structure + cross links can be jointly non-PSD -> must be detected
    w1 = pairwise_structure("s1", ("DM", "CP"), {("DM", "CP"): 0.9}, structure_id="w", status=SYN, provenance="t")
    w2 = pairwise_structure("s2", ("DM", "CP"), {("DM", "CP"): -0.9}, structure_id="w", status=SYN, provenance="t")
    blk = combine_structures([w1, w2], "blk", status=SYN, provenance="t")
    bad = with_cross_correlation(blk, {(("s1", "DM"), ("s2", "DM")): 0.8, (("s1", "CP"), ("s2", "CP")): 0.8},
                                 structure_id="C5_bad", status=SYN, provenance="t")
    assert not bad.psd.is_psd
    with pytest.raises(NotPSDError):
        assert_psd(bad)
    with pytest.raises(ValueError):
        combine_structures([w1, w1], "dup", status=SYN, provenance="t")


# ----------------------------------------------------------------------------------------------
# Gaussian copula
# ----------------------------------------------------------------------------------------------

IDS = ("a", "b")
NUTS = ("CP", "NDF")
MEAN = np.array([[0.10, 0.45], [0.48, 0.12]])
SD = np.array([[0.012, 0.05], [0.02, 0.015]])
DM = np.array([0.35, 0.89])
DSD = np.array([0.03, 0.008])


def _model(corr=None, family="TN_MM"):
    return GaussianCopulaModel.from_targets("syn", IDS, NUTS, MEAN, SD, DM, DSD, family=family,
                                            correlation=corr, is_synthetic=True)


def test_copula_rejects_non_psd_structure_without_repair():
    cells = (("a", "CP"), ("a", "NDF"), ("a", "DM"))
    st = CorrelationStructure("bad", cells, OLD_C4, SYN, "t")
    with pytest.raises(NotPSDError):
        _model(st)


def test_copula_identity_equals_independent_draws_crn():
    cells = (("a", "CP"), ("a", "NDF"), ("b", "DM"))
    ident = independent_structure(cells, status=SYN, provenance="t")
    s = RandomStreams(1103)
    d0 = _model(None).draw(s, "test", 500)
    d1 = _model(ident).draw(s, "test", 500)
    assert isinstance(d0, DrawSet) and d0.stream_id == "root=1103/test"
    assert np.array_equal(d0.theta, d1.theta) and np.array_equal(d0.d, d1.d)


def test_copula_preserves_marginals_and_induces_correlation():
    cells = (("a", "CP"), ("a", "NDF"))
    st = pairwise_structure("a", ("CP", "NDF"), {("CP", "NDF"): -0.7}, structure_id="C1_syn", status=SYN,
                            provenance="t")
    assert st.labels == cells
    m = _model(st)
    rng = np.random.default_rng(3301)
    th, d = m.sample(rng, 200_000)
    n = th.shape[0]
    for (i, j) in np.ndindex(2, 2):
        assert abs(th[:, i, j].mean() - MEAN[i, j]) < 5 * SD[i, j] / math.sqrt(n)
        assert abs(th[:, i, j].std() - SD[i, j]) < 0.02 * SD[i, j]
    target, est = induced_pearson(m, np.random.default_rng(1), 100_000)
    assert est[0, 1] == pytest.approx(-0.7, abs=0.02)      # near-normal marginals: Pearson close to R
    assert np.all((th >= 0) & (th <= 1)) and np.all((d > 0) & (d <= 1))


def test_copula_rejects_unusable_marginals_and_foreign_labels():
    tf = np.empty((2, 2), dtype=object)
    for idx in np.ndindex(2, 2):
        tf[idx] = fit_marginal("TN_MM", MEAN[idx], SD[idx], 0.0, 1.0)
    df = [fit_marginal("TN_MM", DM[i], DSD[i], 0.0, 1.0) for i in range(2)]
    tf_bad = tf.copy()
    tf_bad[0, 0] = fit_marginal("TN_MM", 0.02, 0.03, 0.0, 1.0)         # CV >= 1 -> not_matchable
    with pytest.raises(ValueError, match="not usable"):
        GaussianCopulaModel("m", IDS, NUTS, tf_bad, df, None, True)
    tf_sd = tf.copy()
    tf_sd[1, 1] = fit_marginal("TN_MM", 0.12, float("nan"), 0.0, 1.0)  # mean without SD
    with pytest.raises(ValueError, match="missing_sd"):
        GaussianCopulaModel("m", IDS, NUTS, tf_sd, df, None, True)
    st = independent_structure((("zz", "CP"),), status=SYN, provenance="t")
    with pytest.raises(ValueError, match="not stochastic cells"):
        GaussianCopulaModel("m", IDS, NUTS, tf, df, st, True)
    tf_miss = tf.copy()
    tf_miss[1, 0] = fit_marginal("TN_MM", float("nan"), 0.01, 0.0, 1.0)  # missing mean -> NaN, never 0
    th, _ = GaussianCopulaModel("m", IDS, NUTS, tf_miss, df, None, True).sample(np.random.default_rng(0), 10)
    assert np.all(np.isnan(th[:, 1, 0]))


def test_copula_singular_psd_and_reproducibility():
    cells = (("a", "CP"), ("b", "NDF"))
    st = CorrelationStructure("perfect", cells, np.array([[1.0, 1.0], [1.0, 1.0]]), SYN, "t")
    m = _model(st, family="BETA_MM")
    th, _ = m.sample(np.random.default_rng(5), 2000)
    # comonotone: ranks identical
    r1 = np.argsort(np.argsort(th[:, 0, 0]))
    r2 = np.argsort(np.argsort(th[:, 1, 1]))
    assert np.array_equal(r1, r2)
    lfac = sampling_factor(st.matrix)
    assert np.allclose(lfac @ lfac.T, st.matrix)
    a = m.draw(RandomStreams(1103), "validation", 50)
    b = m.draw(RandomStreams(1103), "validation", 50)
    assert np.array_equal(a.theta, b.theta) and a.model_fingerprint == b.model_fingerprint
    assert _model(None).fingerprint() != m.fingerprint()
