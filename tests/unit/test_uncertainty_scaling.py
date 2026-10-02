"""SD scaling, mean-shift scenarios and the H0/H1 information states (synthetic values).

Also checks the alignment of the information-state prior bases with
``information.signal.double_count_guard`` (contract 7.3, T7.5; acceptance F06): a prior whose SD
already contains sampling and laboratory error must not be combined with an assay signal that
adds the same errors again.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from ration_reliability.datamodel import Provenance, ValueStatus
from ration_reliability.errors import InvalidProblemError
from ration_reliability.information import ComponentErrorModel, ObservedComponent, SamplingProtocol, SignalModel
from ration_reliability.uncertainty import RandomStreams
from ration_reliability.uncertainty.distributions import fit_marginal
from ration_reliability.uncertainty.correlation import CorrelationStructure
from ration_reliability.uncertainty.information_states import (
    CORRELATION_SCOPES,
    H0_PRIOR_BASIS,
    TRUTH_SD_BASES,
    PendingDecisionError,
    RatioEntry,
    TwoLayerFarmModel,
    assay_double_count_report,
    parse_ratio,
    parse_ratio_table,
    ratio_grid,
    require_no_double_count,
    two_layer_marginal_rho,
    variance_split,
)
from ration_reliability.uncertainty.scaling import (
    build_scenario_targets,
    grid_from_config,
    mean_shift,
    scale_sd,
    scale_sd_by_group,
)

REPO = Path(__file__).resolve().parents[2]
IDS = ("sil", "conc")
NUTS = ("CP", "NDF", "Ca")
MEAN = np.array([[0.08, 0.42, 0.0025], [0.46, 0.14, 0.004]])
SD = np.array([[0.009, 0.045, 0.0008], [0.02, 0.02, 0.003]])
DM = np.array([0.35, 0.89])
DSD = np.array([0.05, 0.01])
SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-G2-TEST")


# ----------------------------------------------------------------------------------------------
# SD scaling and mean shift
# ----------------------------------------------------------------------------------------------

def test_scale_sd_multiplies_and_keeps_missing():
    sd = np.array([[0.01, np.nan], [0.02, 0.0]])
    out = scale_sd(sd, 1.25)
    assert out[0, 0] == pytest.approx(0.0125) and np.isnan(out[0, 1]) and out[1, 1] == 0.0
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            scale_sd(sd, bad)
    g = scale_sd_by_group(SD, ["forage", "concentrate"], {"forage": 1.5, "concentrate": 0.8})
    assert np.allclose(g[0], SD[0] * 1.5) and np.allclose(g[1], SD[1] * 0.8)
    with pytest.raises(ValueError, match="no SD scale factor"):
        scale_sd_by_group(SD, ["forage", "other"], {"forage": 1.5})


def test_mean_shift_only_targets_and_never_clips():
    res = mean_shift(MEAN, SD, 0.5, NUTS, targets=("CP", "Ca"))
    assert np.allclose(res.mean[:, 1], MEAN[:, 1])                   # NDF untouched
    assert np.allclose(res.mean[:, 0], MEAN[:, 0] + 0.5 * SD[:, 0])
    assert not res.infeasible.any()
    with pytest.raises(ValueError, match="outside"):
        mean_shift(MEAN, SD, -1.5, NUTS, targets=("Ca",))                # conc Ca would become negative
    res2 = mean_shift(MEAN, SD, -1.5, NUTS, targets=("Ca",), on_infeasible="nan")
    assert res2.infeasible[1, 2] and np.isnan(res2.mean[1, 2])        # flagged, not clipped to 0
    miss = mean_shift(np.array([[np.nan, 0.4, 0.01]]), np.array([[0.01, 0.02, 0.001]]), 1.0, NUTS, ("CP",))
    assert np.isnan(miss.mean[0, 0])
    with pytest.raises(ValueError):
        mean_shift(MEAN, SD, 1.0, NUTS, targets=("starch",))


def test_scenario_targets_keep_nominal_values():
    t = build_scenario_targets(MEAN, SD, DM, DSD, NUTS, s=1.5, k=1.0, shift_targets=("CP",),
                               shift_sd_basis="P0", sd_scale_applies_to_dm=True)
    assert np.array_equal(t.theta_nominal, MEAN) and np.array_equal(t.d_nominal, DM)
    assert np.allclose(t.theta_sd_true, 1.5 * SD) and np.allclose(t.d_sd_true, 1.5 * DSD)
    assert np.allclose(t.theta_mean_true[:, 0], MEAN[:, 0] + SD[:, 0])
    t2 = build_scenario_targets(MEAN, SD, DM, DSD, NUTS, s=1.5, k=1.0, shift_targets=("CP",),
                                shift_sd_basis="scaled", sd_scale_applies_to_dm=False)
    assert np.allclose(t2.theta_mean_true[:, 0], MEAN[:, 0] + 1.5 * SD[:, 0])
    assert np.array_equal(t2.d_sd_true, DSD)
    with pytest.raises(ValueError):
        build_scenario_targets(MEAN, SD, DM, DSD, NUTS, s=1.0, k=0.0, shift_targets=(), shift_sd_basis="x",
                               sd_scale_applies_to_dm=True)


def test_sd_scaling_can_make_truncated_normal_unmatchable():
    # a CV-0.7 cell is matchable at s = 1 but not after scaling by 1.5 (CV 1.05) -> reported, not forced
    assert fit_marginal("TN_MM", 0.02, 0.014, 0.0, 1.0).status == "matched"
    f = fit_marginal("TN_MM", 0.02, float(scale_sd(np.array([0.014]), 1.5)[0]), 0.0, 1.0)
    assert f.status == "not_matchable"
    assert fit_marginal("BETA_MM", 0.02, 0.021, 0.0, 1.0).status == "matched"


def test_grids_read_from_uncertainty_config():
    cfg = yaml.safe_load((REPO / "configs" / "uncertainty.yaml").read_text(encoding="utf-8"))
    sens = cfg["sensitivity_dimensions"]
    grid, status, origin = grid_from_config(sens["a_sd_scale_factor"], grid_key="grid", status_key="grid_status")
    assert origin == "grid" and status == "research_scenario_assumption"
    assert min(grid) < 1.0 < max(grid)                 # both directions (feasibility audit Q2-3(a))
    kg, kst, korig = grid_from_config(sens["g_mean_shift"], grid_key="k_grid", status_key="k_grid_status",
                                      proposal_key="k_grid_dev_start_proposal",
                                      proposal_status_key="k_grid_dev_start_status")
    assert sens["g_mean_shift"]["k_grid"] is None       # still pending: reading does not freeze it
    assert korig == "dev_start_proposal" and kst == "research_scenario_assumption"
    assert cfg["frozen"] is False


# ----------------------------------------------------------------------------------------------
# ratios and the H1 variance split
# ----------------------------------------------------------------------------------------------

def test_parse_ratio_and_entries():
    assert parse_ratio("1/4.4") == pytest.approx(1 / 4.4)
    assert parse_ratio(" 1 / 2 ") == 0.5
    for bad in ("4.4", "1/x", "2/3"):
        with pytest.raises(ValueError):
            parse_ratio(bad)
    with pytest.raises(ValueError):
        RatioEntry("a", "DM", 1.2, "true_only", "", "")
    with pytest.raises(ValueError):
        RatioEntry("a", "DM", 0.5, "made_up", "", "")


def test_parse_config_ratio_table_separates_true_only_from_observed():
    cfg = yaml.safe_load((REPO / "configs" / "uncertainty.yaml").read_text(encoding="utf-8"))
    entries = cfg["sensitivity_dimensions"]["e_information_state_history"]["states"][1]["ratio_table"]["entries"]
    imap = {e["ingredient"]: "mapped_" + str(k) for k, e in enumerate(entries)}
    parsed, notes = parse_ratio_table(entries, imap)
    true_rows = [e for e in entries if str(e["component"]).startswith("true_only")]
    n_true = sum(1 for e in true_rows for c in ("DM", "NDF", "starch", "CP") if e.get(c) is not None)
    assert sum(1 for p in parsed if p.basis == "true_only") == n_true
    assert all(0 < p.ratio < 1 for p in parsed)
    grid_true = ratio_grid(parsed, "true_only")
    grid_obs = ratio_grid(parsed, "observed_incl_sampling_and_lab")
    assert grid_true and grid_obs and grid_true == sorted(set(grid_true))
    # observed ratios contain measurement error, so the largest one is not below the largest true-only one
    assert max(grid_obs) >= max(grid_true)
    parsed2, notes2 = parse_ratio_table(entries, {})
    assert parsed2 == [] and len(notes2) == len(entries)


def test_variance_split_preserves_total_variance():
    b, w = variance_split(SD, np.full(SD.shape, 0.4), s=1.25)
    assert np.allclose(b ** 2 + w ** 2, (1.25 * SD) ** 2)
    assert np.allclose(w, 0.4 * 1.25 * SD)
    b2, w2 = variance_split(SD, np.full(SD.shape, np.nan))
    assert np.all(np.isnan(b2)) and np.all(np.isnan(w2))
    with pytest.raises(ValueError):
        variance_split(SD, np.full(SD.shape, 1.0))


def _two_layer(uncovered="P0", family="BETA_MM", s=1.0, theta_ratio=None, model_id="syn_two_layer", **kw):
    tr = np.array([[np.nan, 1 / 3.6, np.nan], [np.nan, np.nan, np.nan]]) if theta_ratio is None else theta_ratio
    dr = np.array([1 / 4.4, np.nan])
    return TwoLayerFarmModel(model_id, IDS, NUTS, MEAN, SD, DM, DSD, tr, dr, "true_only", s, family,
                             uncovered, is_synthetic=True, **kw)


def test_two_layer_rejects_observed_ratios_and_implicit_choices():
    tr = np.full(SD.shape, 0.3)
    with pytest.raises(ValueError, match="DC-03"):
        TwoLayerFarmModel("m", IDS, NUTS, MEAN, SD, DM, DSD, tr, np.array([0.3, 0.3]),
                          "observed_incl_sampling_and_lab", 1.0, "TN_MM", "P0")
    with pytest.raises(ValueError, match="uncovered_cells"):
        _two_layer(uncovered="default")
    with pytest.raises(ValueError, match="without a within ratio"):
        _two_layer(uncovered="error")


def test_farm_means_streams_are_separated_by_purpose_and_farm():
    m = _two_layer()
    s = RandomStreams(1103)
    a = m.farm_means(s, "validation", 0)
    b = m.farm_means(s, "validation", 0)
    c = m.farm_means(s, "test", 0)
    d = m.farm_means(s, "validation", 1)
    assert np.array_equal(a[0], b[0]) and a[2] == "root=1103/farm_mean/1/0"
    assert not np.array_equal(a[0][0, 1], c[0][0, 1]) and not np.array_equal(a[1][0], d[1][0])
    # uncovered cells stay at the P0 mean (no farm layer)
    assert a[0][1, 0] == MEAN[1, 0] and a[1][1] == DM[1]
    with pytest.raises(ValueError):
        m.farm_means(s, "farm", 0)


@pytest.mark.parametrize("family", ["BETA_MM", "TN_MM"])
def test_two_layer_marginal_keeps_p0_moments(family):
    m = _two_layer(family=family, s=1.25)
    s = RandomStreams(3301)
    xs, ds = [], []
    for f in range(300):
        mt, md, _ = m.farm_means(s, "validation", f)
        dr = m.within_model(mt, md, f"f{f}").draw(s, "validation", 40, f)
        xs.append(dr.theta[:, 0, 1])
        ds.append(dr.d[:, 0])
    x, d = np.concatenate(xs), np.concatenate(ds)
    # clustered draws: tolerance from the between-farm SE (300 farms), generous factor 5
    tb, tw, db, dw = m.between_within()
    se_x = math.sqrt(tb[0, 1] ** 2 / 300 + tw[0, 1] ** 2 / x.size)
    assert abs(x.mean() - MEAN[0, 1]) < 5 * se_x
    assert x.std() == pytest.approx(1.25 * SD[0, 1], rel=0.1)
    assert abs(d.mean() - DM[0]) < 5 * math.sqrt(db[0] ** 2 / 300 + dw[0] ** 2 / d.size)
    assert d.std() == pytest.approx(1.25 * DSD[0], rel=0.1)


def test_decision_priors_h0_h1_and_pending_m_hist():
    m = _two_layer()
    s = RandomStreams(1103)
    mt, md, _ = m.farm_means(s, "validation", 3)
    h0 = m.decision_prior("H0_table_only")
    assert np.array_equal(h0.theta_mean, MEAN) and np.allclose(h0.theta_sd, SD)
    assert set(h0.prior_variance_basis.values()) == {H0_PRIOR_BASIS}
    h1 = m.decision_prior("H1_perfect", mt, md)
    tb, tw, db, dw = m.between_within()
    assert h1.theta_mean[0, 1] == mt[0, 1] and h1.theta_sd[0, 1] == pytest.approx(tw[0, 1])
    assert h1.d_hat[0] == md[0] and h1.d_sd[0] == pytest.approx(dw[0])
    assert h1.theta_mean[1, 0] == MEAN[1, 0]                 # uncovered cell keeps the H0 prior
    assert h1.prior_variance_basis[("sil", "NDF")] == "true_batch_state"
    assert h1.prior_variance_basis[("conc", "CP")] == H0_PRIOR_BASIS
    with pytest.raises(PendingDecisionError):
        m.decision_prior("H1_m", mt, md, streams=s, purpose="validation", farm_index=3)
    hm = m.decision_prior("H1_m", mt, md, streams=s, purpose="validation", farm_index=3, m_hist=4)
    hm2 = m.decision_prior("H1_m", mt, md, streams=s, purpose="validation", farm_index=3, m_hist=4)
    assert np.array_equal(hm.theta_mean, hm2.theta_mean)
    assert hm.theta_sd[0, 1] == pytest.approx(tw[0, 1] * math.sqrt(1 + 1 / 4))
    assert hm.theta_mean[0, 1] != mt[0, 1] and hm.stream_ids == ("root=1103/farm_history/1/3",)


# ----------------------------------------------------------------------------------------------
# alignment with information.signal.double_count_guard
# ----------------------------------------------------------------------------------------------

def _signal(comp=ObservedComponent("sil", "NDF"), sampling=0.01, lab=0.008):
    prov = {k: SYN for k in ("sampling_sd", "lab_repeatability_sd", "lab_bias_sd", "bias_mean")}
    e = ComponentErrorModel(comp, sampling, lab, 0.0, 0.0, "repeatability_only", prov, is_synthetic=True)
    return SignalModel("syn_signal", (e,), SamplingProtocol(1, True, 1), is_synthetic=True)


def test_double_count_guard_alignment():
    m = _two_layer()
    s = RandomStreams(1103)
    mt, md, _ = m.farm_means(s, "validation", 0)
    sig = _signal()
    # H0: table SD already contains historical sampling + lab error (DC-01) -> adding them double counts
    h0 = m.decision_prior("H0_table_only")
    assert assay_double_count_report(h0, sig).status == "violation"
    with pytest.raises(InvalidProblemError, match="double counting"):
        require_no_double_count(h0, sig)
    # FIX5 (red-team finding): a bare de-convolution declaration is not evidence -- without a bound truth
    # model it is a violation (this assertion was "ok" before; tightened, the accepted case follows below)
    bare = assay_double_count_report(h0, sig, {("sil", "NDF"): True})
    assert bare.status == "violation" and any("no truth model is bound" in x for x in bare.messages)
    # bound to a truth model that still generates theta with the observed P0 SD -> still a violation
    assert assay_double_count_report(h0, sig, {("sil", "NDF"): True}, truth_model=m).status == "violation"
    # de-convolved prior declared AND the truth model's SD really declared de-convolved -> accepted
    mdec = _two_layer(truth_sd_basis="true_batch_state", truth_sd_source="synthetic: SDs taken as de-convolved")
    h0d = mdec.decision_prior("H0_table_only")
    assert assay_double_count_report(h0d, sig, {("sil", "NDF"): True}, truth_model=mdec).status == "ok"
    assert require_no_double_count(h0d, sig, truth_model=mdec).status == "ok"
    # H1 with a true_only within SD -> the assay error may be added
    h1 = m.decision_prior("H1_perfect", mt, md)
    assert require_no_double_count(h1, sig).status == "ok"
    # H1 whose prior SD uses an observed within-farm ratio -> double counting is detected
    obs_t = np.array([[np.nan, 1 / 2.1, np.nan], [np.nan, np.nan, np.nan]])
    h1o = m.decision_prior("H1_perfect", mt, md, prior_ratio_basis="observed_incl_sampling_and_lab",
                           observed_theta_ratio=obs_t, observed_d_ratio=np.array([1 / 3.0, np.nan]))
    assert h1o.prior_variance_basis[("sil", "NDF")] == "observed_incl_sampling_and_lab"
    assert h1o.theta_sd[0, 1] == pytest.approx(SD[0, 1] / 2.1)
    assert assay_double_count_report(h1o, sig).status == "violation"
    # a signal on a component the prior does not describe is not silently accepted
    assert assay_double_count_report(h1, _signal(ObservedComponent("zz", "NDF"))).status == "violation"
    # a signal without sampling/lab error adds nothing -> no double counting even for observed priors
    assert assay_double_count_report(h0, _signal(sampling=0.0, lab=0.0)).status == "ok"


# ----------------------------------------------------------------------------------------------
# FIX5: truth-side binding of the double-count guard
# ----------------------------------------------------------------------------------------------

def test_truth_sd_basis_vocabulary_and_declaration():
    from ration_reliability.information.signal import PRIOR_VARIANCE_BASES
    assert set(TRUTH_SD_BASES) == set(PRIOR_VARIANCE_BASES)
    m = _two_layer()
    assert m.truth_sd_basis == H0_PRIOR_BASIS and m.between_layer_contains_measurement_error
    with pytest.raises(ValueError, match="truth_sd_source"):
        _two_layer(truth_sd_basis="true_batch_state")                  # a de-convolution claim needs a source
    with pytest.raises(ValueError, match="truth_sd_basis"):
        _two_layer(truth_sd_basis="made_up")
    mdec = _two_layer(truth_sd_basis="true_batch_state", truth_sd_source="synthetic")
    assert not mdec.between_layer_contains_measurement_error
    # H0 sees the marginal (both layers): the P0 basis; H1 sees the true_only within layer on covered cells
    tb0 = m.truth_variance_basis("H0_table_only")
    tb1 = m.truth_variance_basis("H1_perfect")
    assert tb0[("sil", "NDF")] == H0_PRIOR_BASIS and tb1[("sil", "NDF")] == "true_batch_state"
    assert tb1[("conc", "CP")] == H0_PRIOR_BASIS                      # uncovered cell keeps the P0 basis


def test_guard_checks_truth_side_and_model_identity():
    m = _two_layer()
    s = RandomStreams(1103)
    mt, md, _ = m.farm_means(s, "validation", 0)
    sig = _signal()
    h1 = m.decision_prior("H1_perfect", mt, md)
    assert h1.model_id == m.model_id
    # H1 on a covered cell: prior and truth side both true_batch_state -> the assay error may be added
    assert assay_double_count_report(h1, sig, truth_model=m).status == "ok"
    # H0: the truth side is flagged as well (theta generated with the observed P0 SD)
    rep = assay_double_count_report(m.decision_prior("H0_table_only"), sig, truth_model=m)
    assert rep.status == "violation" and any(x.startswith("truth side:") for x in rep.messages)
    # a prior from another model cannot be checked against this truth model
    other = _two_layer(model_id="another_model")
    rep2 = assay_double_count_report(other.decision_prior("H1_perfect", mt, md), sig, truth_model=m)
    assert rep2.status == "violation" and any("truth model is" in x for x in rep2.messages)
    # signal on an uncovered cell under H1: the truth SD there is the observed P0 SD -> violation
    sig_unc = _signal(ObservedComponent("conc", "CP"))
    assert assay_double_count_report(h1, sig_unc, truth_model=m).status == "violation"
    # a truth-side-only violation is refused by the enforcing function too
    with pytest.raises(InvalidProblemError, match="double counting"):
        require_no_double_count(h1, sig_unc, truth_model=m)


def test_h1_m_notes_the_missing_history_measurement_error():
    m = _two_layer()
    s = RandomStreams(1103)
    mt, md, _ = m.farm_means(s, "validation", 3)
    hm = m.decision_prior("H1_m", mt, md, streams=s, purpose="validation", farm_index=3, m_hist=4)
    assert any("D-FIX-5" in n and "favours H1" in n for n in hm.notes)


# ----------------------------------------------------------------------------------------------
# FIX5: correlation scope in the two-layer world
# ----------------------------------------------------------------------------------------------

def test_two_layer_marginal_rho_formula():
    assert two_layer_marginal_rho(0.0, -0.8, 0.5, 0.5) == pytest.approx(-0.2)
    assert two_layer_marginal_rho(-0.8, -0.8, 0.5, 0.5) == pytest.approx(-0.8)
    # different ratios: same rho in both layers stays at or below |rho|
    v = two_layer_marginal_rho(-0.8, -0.8, 1 / 4.4, 1 / 3.6)
    assert -0.8 <= v < -0.79
    with pytest.raises(ValueError):
        two_layer_marginal_rho(0.0, 0.5, 1.0, 0.5)


_CIDS, _CNUTS = ("sil",), ("CP", "NDF")
_CMEAN, _CSD = np.array([[0.08, 0.42]]), np.array([[0.009, 0.045]])
_CDM, _CDSD = np.array([0.35]), np.array([0.0])


def _corr(rho, sid="syn_R"):
    return CorrelationStructure(sid, (("sil", "CP"), ("sil", "NDF")), np.array([[1.0, rho], [rho, 1.0]]),
                                "synthetic_test_only", "synthetic test structure")


def _corr_model(**kw):
    return TwoLayerFarmModel("syn_corr", _CIDS, _CNUTS, _CMEAN, _CSD, _CDM, _CDSD, np.array([[0.5, 0.5]]),
                             np.array([np.nan]), "true_only", 1.0, "BETA_MM", "P0", is_synthetic=True, **kw)


def test_correlation_scope_must_be_explicit_and_consistent():
    assert set(CORRELATION_SCOPES) == {"within_only", "between_only", "within_and_between"}
    with pytest.raises(ValueError, match="correlation_scope must be chosen explicitly"):
        _corr_model(within_correlation=_corr(-0.8))
    with pytest.raises(ValueError, match="does not match"):
        _corr_model(within_correlation=_corr(-0.8), correlation_scope="within_and_between")
    with pytest.raises(ValueError, match="no correlation structure"):
        _corr_model(correlation_scope="within_only")
    _corr_model(within_correlation=_corr(-0.8), correlation_scope="within_only")
    _corr_model(between_correlation=_corr(-0.8), correlation_scope="between_only")
    # a between-layer structure on a cell without a farm layer is refused
    with pytest.raises(ValueError, match="between_correlation labels"):
        _two_layer(between_correlation=CorrelationStructure(
            "x", (("conc", "CP"), ("sil", "NDF")), np.array([[1.0, 0.3], [0.3, 1.0]]), "synthetic_test_only", "t"),
            correlation_scope="between_only")


def test_within_only_dilutes_the_marginal_correlation_and_both_layers_keep_it():
    """Red-team finding: rho = -0.8 in the within layer only gives a marginal close to rho r_a r_b."""
    rho, n_farms, n_per = -0.8, 400, 25
    out = {}
    for scope, kw in (("within_only", {"within_correlation": _corr(rho)}),
                      ("within_and_between", {"within_correlation": _corr(rho),
                                              "between_correlation": _corr(rho, "syn_R_between")})):
        mdl = _corr_model(correlation_scope=scope, **kw)
        s = RandomStreams(4409)
        xs = []
        for f in range(n_farms):
            mt, md, _ = mdl.farm_means(s, "validation", f)
            xs.append(mdl.within_model(mt, md, f"f{f}").draw(s, "validation", n_per, f).theta[:, 0, :])
        x = np.concatenate(xs)
        out[scope] = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    # cluster-level SE of a correlation with 400 farms is about 0.05; tolerance 0.12
    assert out["within_only"] == pytest.approx(two_layer_marginal_rho(0.0, rho, 0.5, 0.5), abs=0.12)
    assert out["within_and_between"] == pytest.approx(rho, abs=0.12)
    assert out["within_and_between"] < out["within_only"] - 0.3


def test_farm_means_without_between_correlation_unchanged():
    """Default path (no between_correlation): the farm layer is independent across cells, as before."""
    m = _two_layer()
    a = m.farm_means(RandomStreams(1103), "validation", 0)
    b = _two_layer(truth_sd_basis="true_batch_state", truth_sd_source="synthetic").farm_means(
        RandomStreams(1103), "validation", 0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]

