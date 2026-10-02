"""Prior / truth / signal guard (second review R2).  All numbers are synthetic unless stated.

The review showed that the double-count guard only compared caller declarations: with the same
prior, risk table and signal, writing ``true_batch_state`` returned ``identified_by_declared_sources``
and writing ``observed_incl_sampling_and_lab`` refused the computation.  These tests pin the new
behaviour -- the variance basis is object metadata of the prior / truth model and travels with the
states:

1. declaration overriding object metadata (conflict raises);
2. same signal, different truth (identical states, different generating model -> different verdict);
3. historical error counted twice (object metadata alone refuses it, without any declaration);
4. negative variance decomposition (recorded and rejected, never clipped to 0);
5. replicate analyses of the same laboratory sample (only lab_random shrinks);
6. different field samples (sampling error shrinks, the prior of the batch does not);

plus the review's counterexample (no object metadata -> ``unidentified_scenario``; with object
metadata the declaration must agree), DrawSet -> PriorStates metadata retention, and traceability of
the real data chain (factory model built from the restricted NASEM core table; checked against
``reports/restricted_inputs_receipt.json``) and of error values
(``sources/error_source_locators.csv``).  The real-data test uses the restricted local table only to
recompute hashes; it asserts on hashes, ids and counts, never on values.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import norm

from engine_test_helpers import conc, dm_offer, ing, problem, two_ingredient_problem
from ration_reliability.datamodel import Provenance, RationDecision, ValueStatus
from ration_reliability.errors import InvalidProblemError, LeakageError
from ration_reliability.hashing import file_sha256, stable_hash
from ration_reliability.information import (
    CandidateLibrary,
    ComponentErrorModel,
    ComponentMetadata,
    DoubleCountReport,
    InconsistentDecompositionError,
    InformationStructure,
    MetadataConflictError,
    ObservedComponent,
    PriorStates,
    SamplingProtocol,
    SignalBinning,
    SignalModel,
    StateMetadata,
    compute_information_value,
    compute_risk_table,
    decompose_observed_variance,
    deconvolve_true_sd,
    double_count_guard,
    load_error_locators,
    resolve_truth_metadata,
    trace_error_provenance,
)
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams
from ration_reliability.uncertainty.factory import build_uncertainty_model
from ration_reliability.uncertainty.spec import UncertaintySpec

REPO = Path(__file__).resolve().parents[2]
SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-K2-GUARD")
FCP = ObservedComponent("F", "CP")
ALPHA_PROBE = 0.10
ALPHA = 0.05
HIST = "synthetic_measurement:K2-HIST-LAB"          # measurement process contained in an observed prior SD


# =================================================================================================
# fixtures: the review counterexample (independent_probes.py) and a factory-built world
# =================================================================================================

def _probe(metadata=None):
    """The review's counterexample: two CP states 8 % / 12 %, candidates A (2.90) and B (3.56),
    P(low bin | bad, good) = 0.30 / 0.10 from a Gaussian sampling error and one threshold."""
    prob = problem([ing("F", .4, {"CP": .1}, forage=1), ing("C", .8, {"CP": .4})], ["CP"],
                   [dm_offer(20), conc("cp_min", {"CP": 1}, "ge", 16)], {"F": .04, "C": .32}, "review_synthetic")
    qs = [np.array([42.5, 3.75]), np.array([37, 6.5])]
    lib = CandidateLibrary.from_decisions(prob, [RationDecision(prob.ingredient_ids, q, prob.dm_estimates(),
                                                                "synthetic") for q in qs], ["A", "B"])
    prior = PriorStates.from_discrete(np.array([[[.08], [.4]], [[.12], [.4]]]), np.array([[.4, .8], [.4, .8]]),
                                      [.5, .5], prob.ingredient_ids, ["CP"], label="review_synthetic",
                                      is_synthetic=True, metadata=metadata)
    risk = compute_risk_table(lib, prior, prob.compiled, prob.price_vector())
    sd = .04 / (norm.ppf(.3) - norm.ppf(.1))
    edge = .08 + sd * norm.ppf(.3)
    sm = SignalModel("review_signal", (ComponentErrorModel(FCP, sd, 0, provenance={"sampling_sd": SYN},
                                                           is_synthetic=True),), is_synthetic=True)
    st = InformationStructure("review_counterexample", "sample",
                              binning=SignalBinning((FCP,), ((edge,),), "synthetic edge"), signal_model=sm)
    return prior, risk, st, sm


RULE = {"primary_family": "TN_MM", "fallback_families": [], "on_exhausted": "error",
        "status": "synthetic_test_only", "rationale": "synthetic unit test (K2)", "selection_basis": "declared_rule"}


def _spec(basis="true_batch_state", *, mm=None, dec="none", dec_src=None, sd=0.01, spec_id="k2_world"):
    """Two-ingredient world; only F:CP is stochastic; every other cell is a point value."""
    pr = two_ingredient_problem()
    defaults = {"variance_basis": basis, "data_fingerprint": "synthetic:k2", "decomposition_id": dec,
                "decomposition_source": dec_src, "measurement_model_id": mm,
                "provenance_status": "synthetic_test_only", "source_id": None, "locator": None}
    spec = UncertaintySpec.from_arrays(
        spec_id, list(pr.ingredient_ids), list(pr.nutrient_ids), pr.nominal_theta(), np.array([[sd], [0.0]]),
        pr.dm_estimates(), np.array([0.0, 0.0]), purpose="unit_test", moment_semantics="target_marginal_moments",
        is_synthetic=True, family_rule=RULE, cell_defaults=defaults, theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0))
    return pr, spec


def _world(basis="true_batch_state", n=400, **kw):
    pr, spec = _spec(basis, **kw)
    model = build_uncertainty_model(spec)
    prior = model.prior_states(RandomStreams(1103), "opt", n)
    ids, dh = pr.ingredient_ids, pr.dm_estimates()
    # CP target 18 % (safe) and 16 % (nominal) at planned DM 20 kg/d
    decs = [RationDecision(ids, np.array([(20 - 20 * (c - .10) / .30) / .40, 20 * (c - .10) / .30 / .80]), dh,
                           f"synthetic_cp{c:.2f}") for c in (0.18, 0.16)]
    lib = CandidateLibrary.from_decisions(pr, decs, ["safe", "nominal"])
    risk = compute_risk_table(lib, prior, pr.compiled, pr.price_vector())
    return model, prior, risk


def _signal(samp=0.004, lab=0.003, bias=0.0, *, mm=None, protocol=SamplingProtocol(), sid="k2_signal"):
    prov = {k: SYN for k, v in (("sampling_sd", samp), ("lab_repeatability_sd", lab), ("lab_bias_sd", bias)) if v}
    em = ComponentErrorModel(FCP, samp, lab, bias, provenance=prov, is_synthetic=True, measurement_model_id=mm)
    return SignalModel(sid, (em,), protocol, is_synthetic=True)


def _structure(sm, edge=0.10):
    return InformationStructure(f"s:{sm.signal_id}", "sample", binning=SignalBinning((FCP,), ((edge,),), "syn edge"),
                                signal_model=sm)


# =================================================================================================
# 0  the review counterexample: declaration-only -> unidentified_scenario; object metadata decides
# =================================================================================================

def test_counterexample_without_object_metadata_is_an_unidentified_scenario():
    prior, risk, st, sm = _probe()
    assert prior.metadata is None
    r = compute_information_value(st, prior, risk, ALPHA_PROBE, prior_variance_basis={FCP: "true_batch_state"})
    # the computation is done (hand values of the review) ...
    assert r.V0.expected_cost == pytest.approx(3.56, abs=1e-10)
    assert r.VT.expected_cost == pytest.approx(3.56, abs=1e-10)
    # ... but a declaration is not an identification
    assert r.error_model_identification == "unidentified_scenario"
    assert r.error_model_report["object_bound"] is False
    assert r.error_model_report["basis_sources"] == {"F:CP": "caller_declaration_unbound"}
    assert any("unidentified scenario" in w for w in r.warnings)
    u = compute_information_value(st, prior, risk, ALPHA_PROBE, prior_variance_basis={FCP: "unidentified"})
    assert u.error_model_identification == "unidentified_scenario"
    # a declaration can only make the guard stricter: a declared double count is still refused
    with pytest.raises(InvalidProblemError, match="double counting"):
        compute_information_value(st, prior, risk, ALPHA_PROBE,
                                  prior_variance_basis={FCP: "observed_incl_sampling_and_lab"})
    with pytest.raises(InvalidProblemError, match="double_count_report"):     # nothing at all: refused
        compute_information_value(st, prior, risk, ALPHA_PROBE)


def test_counterexample_with_object_metadata_the_declaration_must_agree():
    md_true = StateMetadata.synthetic((FCP,), "true_batch_state", label="review_true")
    prior, risk, st, sm = _probe(md_true)
    ok = compute_information_value(st, prior, risk, ALPHA_PROBE)                     # no declaration needed
    assert ok.error_model_identification == "identified_by_declared_sources"
    assert ok.error_model_report["basis_sources"] == {"F:CP": "object_metadata:prior"}
    same = compute_information_value(st, prior, risk, ALPHA_PROBE, prior_variance_basis={FCP: "true_batch_state"})
    assert same.error_model_identification == "identified_by_declared_sources"
    assert same.VT.expected_cost == pytest.approx(ok.VT.expected_cost, abs=0)
    with pytest.raises(MetadataConflictError, match="cannot override"):
        compute_information_value(st, prior, risk, ALPHA_PROBE,
                                  prior_variance_basis={FCP: "observed_incl_sampling_and_lab"})
    with pytest.raises(MetadataConflictError, match="prior_deconvolved"):   # object is not a de-convolved SD
        compute_information_value(st, prior, risk, ALPHA_PROBE, prior_variance_basis={FCP: "true_batch_state"},
                                  prior_deconvolved={FCP: True})
    # the same states with observed-basis metadata: refused whatever is declared
    md_obs = StateMetadata.synthetic((FCP,), "observed_incl_sampling_and_lab", label="review_obs")
    prior_o, risk_o, st_o, _ = _probe(md_obs)
    np.testing.assert_array_equal(prior_o.theta, prior.theta)
    assert prior_o.fingerprint != prior.fingerprint                      # metadata are part of the object
    with pytest.raises(InvalidProblemError, match="double counting"):
        compute_information_value(st_o, prior_o, risk_o, ALPHA_PROBE)
    with pytest.raises(MetadataConflictError):
        compute_information_value(st_o, prior_o, risk_o, ALPHA_PROBE, prior_variance_basis={FCP: "true_batch_state"})
    # swapping the metadata without rebuilding the risk table is impossible (fingerprint binding)
    swapped = dataclasses.replace(prior, metadata=StateMetadata.synthetic((FCP,), "true_batch_state", label="other"))
    with pytest.raises(InvalidProblemError, match="another prior"):
        compute_information_value(st, swapped, risk, ALPHA_PROBE)


# =================================================================================================
# 1  a declaration never overrides object metadata
# =================================================================================================

def test_declarations_and_reports_cannot_override_factory_metadata():
    model, prior, risk = _world("observed_incl_sampling_and_lab", mm=HIST)
    sm = _signal()
    st = _structure(sm)
    assert prior.component_metadata(FCP).variance_basis == "observed_incl_sampling_and_lab"
    with pytest.raises(MetadataConflictError, match="cannot override"):
        compute_information_value(st, prior, risk, ALPHA, prior_variance_basis={FCP: "true_batch_state"})
    # a declaration-mode report of the wrong basis reproduces as a report ... and still conflicts with the object
    decl = double_count_guard(sm, {FCP: "true_batch_state"})
    assert decl.ok and not decl.object_mode
    with pytest.raises(MetadataConflictError):
        compute_information_value(st, prior, risk, ALPHA, double_count_report=decl)
    with pytest.raises(MetadataConflictError):                     # "de-convolved" by declaration only
        compute_information_value(st, prior, risk, ALPHA,
                                  prior_variance_basis={FCP: "observed_incl_sampling_and_lab"},
                                  prior_deconvolved={FCP: True})
    with pytest.raises(MetadataConflictError):                     # stand-alone object-mode guard
        double_count_guard(sm, {FCP: "true_batch_state"}, prior=prior)
    # explicit metadata handed to from_drawset cannot contradict the factory object either
    draws = model.draw(RandomStreams(1103), "opt", 50)
    wrong = StateMetadata.synthetic((FCP,), "true_batch_state", label="k2")
    with pytest.raises(MetadataConflictError, match="cannot override"):
        PriorStates.from_drawset(draws, metadata=wrong)
    agree = StateMetadata.from_mapping({FCP: prior.component_metadata(FCP)}, origin="explicit_prior_definition",
                                       is_synthetic=True)
    kept = PriorStates.from_drawset(draws, metadata=agree)
    assert kept.metadata.origin == "factory_model_metadata"        # the object's metadata are kept


def test_object_mode_reports_are_bound_to_prior_and_truth():
    _, prior, risk = _world("true_batch_state")
    _, prior2, _ = _world("true_batch_state", n=300)
    sm = _signal()
    st = _structure(sm)
    rep = double_count_guard(sm, prior=prior)
    assert rep.ok and rep.object_mode and rep.object_bound and rep.prior_fingerprint == prior.fingerprint
    r = compute_information_value(st, prior, risk, ALPHA, double_count_report=rep)
    assert r.error_model_identification == "identified_by_declared_sources"
    assert r.error_model_report["source"] == "recomputed_from_bound_report"
    with pytest.raises(InvalidProblemError, match="other prior states"):
        compute_information_value(st, prior2, risk, ALPHA, double_count_report=rep)
    forged = dataclasses.replace(rep, status="violation", per_component={"F:CP": "violation"})
    with pytest.raises(InvalidProblemError, match="does not reproduce"):
        compute_information_value(st, prior, risk, ALPHA, double_count_report=forged)
    bare = DoubleCountReport("ok", (), {})
    with pytest.raises(InvalidProblemError, match="not bound"):
        compute_information_value(st, prior, risk, ALPHA, double_count_report=bare)


# =================================================================================================
# 2  same signal, different truth
# =================================================================================================

def test_same_signal_same_states_different_truth_model_gives_a_different_verdict():
    m_true, p_true, r_true = _world("true_batch_state")
    m_obs, p_obs, r_obs = _world("observed_incl_sampling_and_lab", mm=HIST)
    np.testing.assert_array_equal(p_true.theta, p_obs.theta)       # bitwise the same states ...
    assert m_true.fingerprint() != m_obs.fingerprint()              # ... from two different worlds
    sm = _signal()
    st = _structure(sm)
    ok = compute_information_value(st, p_true, r_true, ALPHA)
    assert ok.error_model_identification == "identified_by_declared_sources"
    with pytest.raises(InvalidProblemError, match="double counting"):
        compute_information_value(st, p_obs, r_obs, ALPHA)
    # the truth must be the generator of the prior states
    linked = compute_information_value(st, p_true, r_true, ALPHA, truth=m_true)
    assert linked.error_model_report["truth_linked"] is True
    with pytest.raises(InvalidProblemError, match="not the model that generated"):
        compute_information_value(st, p_true, r_true, ALPHA, truth=m_obs)
    # a forged link (another model's metadata under this model's fingerprint): prior and truth disagree
    forged = dataclasses.replace(StateMetadata.from_factory_model(m_obs), model_fingerprint=m_true.fingerprint())
    with pytest.raises(InvalidProblemError, match="prior and truth metadata disagree"):
        compute_information_value(st, p_true, r_true, ALPHA, truth=forged)


def test_unlinked_truth_record_can_only_restrict():
    prior, risk, st, sm = _probe()                                  # discrete prior, no metadata
    t_obs = StateMetadata.synthetic((FCP,), "observed_incl_sampling_and_lab", label="t",
                                    origin="explicit_truth_record")
    t_true = StateMetadata.synthetic((FCP,), "true_batch_state", label="t", origin="explicit_truth_record")
    md, origin, linked = resolve_truth_metadata(t_true, prior)
    assert md is t_true and not linked and origin.endswith(":unlinked")
    with pytest.raises(InvalidProblemError, match="double counting"):
        compute_information_value(st, prior, risk, ALPHA_PROBE, truth=t_obs)
    rep = double_count_guard(sm, prior=prior, truth=t_obs)
    assert rep.status == "violation" and rep.truth_linked is False and not rep.object_bound
    # with a declared basis the unlinked record still restricts (it cannot be overruled by the declaration)
    rep2 = double_count_guard(sm, {FCP: "true_batch_state"}, prior=prior, truth=t_obs)
    assert rep2.status == "violation" and any(m.startswith("truth record (unlinked") for m in rep2.messages)
    r = compute_information_value(st, prior, risk, ALPHA_PROBE, truth=t_true)
    assert r.error_model_identification == "unidentified_scenario"     # an unlinked record cannot identify
    assert r.error_model_report["truth_linked"] is False


# =================================================================================================
# 3  historical laboratory error counted twice
# =================================================================================================

def test_historical_lab_error_in_the_prior_is_not_added_again():
    _, prior, risk = _world("observed_incl_sampling_and_lab", mm=HIST)
    sm = _signal()
    with pytest.raises(InvalidProblemError, match="prior SD already contains sampling and lab error"):
        compute_information_value(_structure(sm), prior, risk, ALPHA)          # no declaration at all
    rep = double_count_guard(sm, prior=prior)
    assert rep.status == "violation" and rep.basis_sources == (("F:CP", "object_metadata:prior"),)
    # observed with lab error only: a sampling-only signal adds nothing twice, a lab error does
    _, prior_l, risk_l = _world("observed_incl_lab_only", mm=HIST)
    r = compute_information_value(_structure(_signal(0.004, 0.0)), prior_l, risk_l, ALPHA)
    assert r.error_model_identification == "identified_by_declared_sources"
    with pytest.raises(InvalidProblemError, match="already contains laboratory error"):
        compute_information_value(_structure(_signal(0.004, 0.003)), prior_l, risk_l, ALPHA)
    # a documented de-convolution (consistent record) makes the same states a true-state prior
    rec = decompose_observed_variance(0.0125, sampling_sd=0.006, lab_sd=0.0045, component=FCP,
                                      decomposition_id="K2-SYN-DECONV", decomposition_source="synthetic unit test",
                                      measurement_model_id=HIST)
    assert rec.status == "consistent" and rec.true_sd == pytest.approx(0.01, rel=1e-12)
    cm = ComponentMetadata(FCP, "true_batch_state", "synthetic:k2", "K2-SYN-DECONV", "synthetic unit test", HIST,
                           True, "synthetic_test_only", None, None, rec)
    assert cm.deconvolved
    # FIX_A (red-team finding): the probe states have SD 0.02, the record says the de-convolved true SD is 0.01 --
    # the label is not backed by the states, so the same construction is now rejected (numerical check)
    p_bad, r_bad, st_bad, _ = _probe(StateMetadata((cm,), "explicit_prior_definition", is_synthetic=True))
    assert np.sqrt(p_bad.component_variance(FCP)) == pytest.approx(0.02, rel=1e-12)
    with pytest.raises(InvalidProblemError, match="not generated from the recorded true-state SD"):
        compute_information_value(st_bad, p_bad, r_bad, ALPHA_PROBE, prior_deconvolved={FCP: True})
    # ... and a record whose true SD equals the SD of the states (0.02) identifies, as before
    rec2 = decompose_observed_variance(float(np.sqrt(0.02 ** 2 + 0.006 ** 2 + 0.0045 ** 2)), sampling_sd=0.006,
                                       lab_sd=0.0045, component=FCP, decomposition_id="K2-SYN-DECONV",
                                       decomposition_source="synthetic unit test", measurement_model_id=HIST)
    assert rec2.status == "consistent" and rec2.true_sd == pytest.approx(0.02, rel=1e-12)
    cm2 = ComponentMetadata(FCP, "true_batch_state", "synthetic:k2", "K2-SYN-DECONV", "synthetic unit test", HIST,
                            True, "synthetic_test_only", None, None, rec2)
    p_dec, r_dec, st_dec, _ = _probe(StateMetadata((cm2,), "explicit_prior_definition", is_synthetic=True))
    ok = compute_information_value(st_dec, p_dec, r_dec, ALPHA_PROBE, prior_deconvolved={FCP: True})
    assert ok.error_model_identification == "identified_by_declared_sources"
    assert ok.error_model_report["checks"]["true_state_checks"]["F:CP"]["status"] == "verified"
    # object says "unidentified": computed as a labelled scenario, never identified
    _, prior_u, risk_u = _world("unidentified")
    u = compute_information_value(_structure(sm), prior_u, risk_u, ALPHA)
    assert u.error_model_identification == "unidentified_scenario"


def test_bare_deconvolution_declaration_and_unsourced_object_metadata_do_not_identify():
    prior, risk, st, sm = _probe()                                  # no object metadata
    # D-327 (FIX5) applied to the value path: a bare "de-convolved" declaration is not evidence
    with pytest.raises(InvalidProblemError, match="bare declaration is not evidence"):
        compute_information_value(st, prior, risk, ALPHA_PROBE,
                                  prior_variance_basis={FCP: "observed_incl_sampling_and_lab"},
                                  prior_deconvolved={FCP: True})
    # object metadata whose basis is itself a research-scenario assumption: computed, labelled scenario
    cm = ComponentMetadata(FCP, "true_batch_state", "synthetic:k2", "K2-SCENARIO-DECONV",
                           "declared scenario split (no identification data)", HIST, True,
                           "research_scenario_assumption")
    p2, r2, st2, _ = _probe(StateMetadata((cm,), "explicit_prior_definition", is_synthetic=True))
    res = compute_information_value(st2, p2, r2, ALPHA_PROBE)
    assert res.error_model_identification == "unidentified_scenario"
    assert res.error_model_report["object_bound"] is True
    assert res.error_model_report["object_metadata_without_sourced_provenance"] == ["F:CP (research_scenario_assumption)"]


# =================================================================================================
# 4  negative variance decomposition: recorded and rejected, never clipped
# =================================================================================================

def test_negative_decomposition_is_rejected_not_clipped():
    rec = decompose_observed_variance(0.01, sampling_sd=0.009, lab_sd=0.006, component=FCP)
    assert rec.status == "inconsistent_negative"
    assert rec.implied_true_variance == pytest.approx(1e-4 - 8.1e-5 - 3.6e-5, abs=1e-18)   # kept negative
    assert rec.implied_true_variance < 0 and rec.true_sd is None
    with pytest.raises(InconsistentDecompositionError, match="not clipped"):
        rec.require_consistent()
    with pytest.raises(ValueError, match="inconsistent"):
        deconvolve_true_sd(0.01, 0.009, 0.006)
    zero = decompose_observed_variance(0.05, lab_sd=0.05)
    assert zero.status == "degenerate_no_true_variation" and zero.true_sd == 0.0
    # an inconsistent decomposition cannot certify a true-state SD
    bad = decompose_observed_variance(0.01, sampling_sd=0.009, lab_sd=0.006, component=FCP,
                                      decomposition_id="K2-BAD", decomposition_source="synthetic",
                                      measurement_model_id=HIST)
    with pytest.raises(InconsistentDecompositionError):
        ComponentMetadata(FCP, "true_batch_state", "synthetic:k2", "K2-BAD", "synthetic", HIST, True,
                          "synthetic_test_only", None, None, bad)


def test_prior_variance_below_the_error_variance_of_the_same_process_is_rejected():
    # prior F:CP states 8 % / 12 %: variance 4e-4; signal single-result error variance 0.015^2 + 0.016^2 > 4e-4
    for basis in ("unidentified", "observed_incl_sampling_and_lab"):
        md = StateMetadata.synthetic((FCP,), basis, label="k2neg", measurement_model_id=HIST)
        prior, risk, _, _ = _probe(md)
        assert prior.component_variance(FCP) == pytest.approx(4e-4, rel=1e-12)
        same = _signal(0.015, 0.016, mm=HIST)
        with pytest.raises(InvalidProblemError, match="inconsistent variance decomposition"):
            compute_information_value(_structure(same, edge=0.10), prior, risk, ALPHA_PROBE)
        rep = double_count_guard(same, prior=prior)
        chk = rep.checks["decomposition"]["F:CP"]
        assert chk["status"] == "inconsistent_negative" and chk["enforced"] is True
        assert chk["implied_true_variance"] == pytest.approx(4e-4 - 0.015 ** 2 - 0.016 ** 2, rel=1e-12)
        assert "not clipped" in chk["violation"]
    # another (unlinked) laboratory: recorded as a warning, the unidentified scenario is still computed
    md = StateMetadata.synthetic((FCP,), "unidentified", label="k2neg", measurement_model_id=HIST)
    prior, risk, _, _ = _probe(md)
    other = _signal(0.015, 0.016, mm="synthetic_measurement:OTHER-LAB")
    r = compute_information_value(_structure(other, edge=0.10), prior, risk, ALPHA_PROBE)
    assert r.error_model_identification == "unidentified_scenario"
    chk = r.error_model_report["checks"]["decomposition"]["F:CP"]
    assert chk["status"] == "inconsistent_negative" and chk["enforced"] is False
    assert any("inconsistent variance decomposition" in w for w in r.warnings)
    # a consistent decomposition of the same process: recorded, not a violation
    small = _signal(0.004, 0.003, mm=HIST)
    rep = double_count_guard(small, prior=prior)
    assert rep.status == "unidentified"
    assert rep.checks["decomposition"]["F:CP"]["status"] == "consistent"


# =================================================================================================
# 5  replicate analyses of the same laboratory sample shrink only lab_random
# =================================================================================================

def test_replicates_of_one_lab_sample_reduce_only_lab_random():
    parts = {}
    for r in (1, 4, 16):
        sm = _signal(0.004, 0.006, 0.002, protocol=SamplingProtocol(1, True, r))
        (p,) = sm.variance_parts()
        parts[r] = p
        assert p["sampling"] == pytest.approx(0.004 ** 2) and p["lab_bias"] == pytest.approx(0.002 ** 2)
        assert p["lab_repeatability"] == pytest.approx(0.006 ** 2 / r)
    # Monte Carlo: r = 4 replicates of one composite
    sm4 = _signal(0.004, 0.006, 0.002, protocol=SamplingProtocol(1, True, 4))
    z = sm4.simulate(np.zeros((200_000, 1)), np.random.default_rng(20260925))
    assert np.var(z) == pytest.approx(sum(parts[4].values()), rel=0.02)
    # a single-result total SD has no identified reducible share: replicates do not reduce it
    tot = SignalModel("tot", (ComponentErrorModel(FCP, 0.004, 0.006, lab_error_semantics="single_result_total",
                                                  provenance={"sampling_sd": SYN, "lab_repeatability_sd": SYN},
                                                  is_synthetic=True),), SamplingProtocol(1, True, 16), is_synthetic=True)
    assert tot.variance_parts()[0]["lab_repeatability"] == pytest.approx(0.006 ** 2)
    # replication never removes a double count of lab error in the prior
    _, prior_l, _ = _world("observed_incl_lab_only", mm=HIST)
    _, prior_s, _ = _world("observed_incl_sampling_and_lab", mm=HIST)
    for r in (1, 16):
        sm = _signal(0.004, 0.006, 0.002, protocol=SamplingProtocol(1, True, r), mm=HIST)
        assert double_count_guard(sm, prior=prior_l).status == "violation"
        assert double_count_guard(sm, prior=prior_s).status == "violation"
    # the prior's decomposition refers to historical single results: independent of today's replicates
    e1 = double_count_guard(_signal(0.004, 0.006, 0.002, mm=HIST), prior=prior_s).checks["decomposition"]["F:CP"]
    e16 = double_count_guard(_signal(0.004, 0.006, 0.002, protocol=SamplingProtocol(1, True, 16), mm=HIST),
                             prior=prior_s).checks["decomposition"]["F:CP"]
    assert e1["single_result_error_variance"] == e16["single_result_error_variance"]
    # a report is bound to the protocol: r = 1 report is foreign to the r = 4 structure
    _, prior_t, risk_t = _world("true_batch_state")
    rep1 = double_count_guard(_signal(0.004, 0.006), prior=prior_t)
    st4 = _structure(_signal(0.004, 0.006, protocol=SamplingProtocol(1, True, 4)))
    with pytest.raises(InvalidProblemError, match="another signal"):
        compute_information_value(st4, prior_t, risk_t, ALPHA, double_count_report=rep1)


# =================================================================================================
# 6  different field samples shrink the sampling error, not the batch prior
# =================================================================================================

def test_independent_field_samples_reduce_sampling_error_but_not_the_batch_prior():
    for m in (1, 4):
        comp = _signal(0.008, 0.004, 0.002, protocol=SamplingProtocol(m, True, 2)).variance_parts()[0]
        sep = _signal(0.008, 0.004, 0.002, protocol=SamplingProtocol(m, False, 2)).variance_parts()[0]
        assert comp["sampling"] == pytest.approx(0.008 ** 2 / m) == sep["sampling"]
        assert comp["lab_repeatability"] == pytest.approx(0.004 ** 2 / 2)
        assert sep["lab_repeatability"] == pytest.approx(0.004 ** 2 / (m * 2))
        assert comp["lab_bias"] == sep["lab_bias"] == pytest.approx(0.002 ** 2)
    sm = _signal(0.008, 0.004, 0.002, protocol=SamplingProtocol(4, False, 2))
    z = sm.simulate(np.zeros((200_000, 1)), np.random.default_rng(1103))
    assert np.var(z) == pytest.approx(sum(sm.variance_parts()[0].values()), rel=0.02)
    # the batch prior does not change with the number of field samples
    _, prior_l, risk_l = _world("observed_incl_lab_only", mm=HIST)
    _, prior_s, _ = _world("observed_incl_sampling_and_lab", mm=HIST)
    v0 = prior_l.component_variance(FCP)
    results = []
    for m in (1, 8):
        s_only = _signal(0.008, 0.0, protocol=SamplingProtocol(m, True, 1))
        r = compute_information_value(_structure(s_only), prior_l, risk_l, ALPHA)
        assert r.error_model_identification == "identified_by_declared_sources"
        assert r.provenance["prior_fingerprint"] == prior_l.fingerprint
        results.append(r)
        both = _signal(0.008, 0.004, protocol=SamplingProtocol(m, True, 1), mm=HIST)
        assert double_count_guard(both, prior=prior_s).status == "violation"   # m never fixes a double count
        chk = double_count_guard(both, prior=prior_s).checks["decomposition"]["F:CP"]
        assert chk["single_result_error_variance"] == pytest.approx(0.008 ** 2 + 0.004 ** 2)  # one historical sample
    assert prior_l.component_variance(FCP) == v0                                   # the batch prior is unchanged
    assert results[0].provenance["signal_fingerprint"] != results[1].provenance["signal_fingerprint"]


# =================================================================================================
# 7  DrawSet -> PriorStates keeps the object metadata
# =================================================================================================

def test_drawset_to_prior_keeps_factory_metadata():
    pr, spec = _spec("observed_incl_sampling_and_lab", mm=HIST)
    model = build_uncertainty_model(spec)
    draws = model.draw(RandomStreams(1103), "opt", 64)
    for prior in (PriorStates.from_drawset(draws), model.prior_states(RandomStreams(1103), "opt", 64),
                  PriorStates.from_drawset(draws, model=model)):
        md = prior.metadata
        assert md.origin == "factory_model_metadata"
        assert md.model_fingerprint == model.fingerprint() == prior.generator_fingerprint
        assert md.metadata_fingerprint == model.metadata.fingerprint()
        for cell in model.metadata.cells:
            cm = md.get((cell.ingredient_id, cell.component))
            assert cm is not None and cm.data_fingerprint == cell.data_fingerprint
            assert cm.variance_basis == (cell.variance_basis if cell.is_stochastic else "not_applicable")
            assert (cm.decomposition_id, cm.measurement_model_id) == (cell.decomposition_id, cell.measurement_model_id)
    plain = dataclasses.replace(prior, metadata=None)
    assert plain.fingerprint != prior.fingerprint
    # re-labelling factory states by hand (dataclasses.replace / direct construction) is refused as well
    with pytest.raises(MetadataConflictError, match="cannot override"):
        dataclasses.replace(prior, metadata=StateMetadata.synthetic((FCP,), "true_batch_state", label="forged"))
    with pytest.raises(LeakageError):
        model.prior_states(RandomStreams(1103), "test", 10)
    other = build_uncertainty_model(_spec("true_batch_state", spec_id="k2_other")[1])
    with pytest.raises(InvalidProblemError, match="not generated by model"):
        PriorStates.from_drawset(draws, model=other)
    # a non-factory model has no object metadata unless the prior is given explicit metadata
    inm = IndependentNormalModel("k2_inm", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(),
                                 np.array([[0.01], [0.0]]), pr.dm_estimates(), np.zeros(2), is_synthetic=True)
    d2 = inm.draw(RandomStreams(1103), "opt", 64)
    assert PriorStates.from_drawset(d2).metadata is None
    ex = PriorStates.from_drawset(d2, metadata=StateMetadata.synthetic((FCP,), "true_batch_state", label="inm"))
    assert ex.metadata.model_fingerprint == inm.fingerprint() and ex.metadata.origin == "explicit_prior_definition"
    # object metadata must describe the states
    with pytest.raises(MetadataConflictError, match="not on the prior axes"):
        PriorStates.from_drawset(d2, metadata=StateMetadata.synthetic((ObservedComponent("X", "CP"),),
                                                                      "true_batch_state", label="x"))
    point = StateMetadata((ComponentMetadata(FCP, "not_applicable", None, "none", None, None, True),),
                          "explicit_prior_definition", is_synthetic=True)
    with pytest.raises(MetadataConflictError, match="states vary"):
        PriorStates.from_drawset(d2, metadata=point)


def test_component_metadata_validation_mirrors_the_spec_rules():
    with pytest.raises(InvalidProblemError, match="needs a variance decomposition"):
        ComponentMetadata(FCP, "true_batch_state", "sha:row", "none", None, None, False)
    with pytest.raises(InvalidProblemError, match="measurement_model_id"):
        ComponentMetadata(FCP, "observed_incl_sampling_and_lab", "sha:row", "none", None, None, False)
    with pytest.raises(InvalidProblemError, match="data_fingerprint"):
        ComponentMetadata(FCP, "observed_incl_lab_only", None, "none", None, "LAB", False)
    with pytest.raises(InvalidProblemError, match="decomposition_source"):
        ComponentMetadata(FCP, "true_batch_state", "sha:row", "DEC-1", None, "LAB", False)
    with pytest.raises(InvalidProblemError, match="source_id and locator"):
        ComponentMetadata(FCP, "observed_incl_lab_only", "sha:row", "none", None, "LAB", False, "sourced")
    with pytest.raises(InvalidProblemError, match="non-synthetic"):
        StateMetadata((ComponentMetadata(FCP, "true_batch_state", "synthetic:x", "none", None, None, True),),
                      "explicit_prior_definition", is_synthetic=False)


# =================================================================================================
# 8  the real data chain is traceable to the restricted-input receipt (hashes only)
# =================================================================================================

_E0 = REPO / "experiments" / "E0_verification"
CORE = REPO / "data" / "restricted_local" / "nasem_t19_1_core.csv"
RECEIPT = REPO / "reports" / "restricted_inputs_receipt.json"


def _receipt_sha(fname: str) -> str:
    rec = json.loads(RECEIPT.read_text(encoding="utf-8"))
    return next(f["sha256_actual"] for f in rec["sections"]["B_restricted_sources"]["files"] if f["file"] == fname)


@pytest.mark.skipif(not CORE.exists(), reason="restricted NASEM core table not present (data/restricted_local/ is "
                                              "not distributed); run where the legal holder keeps it")
def test_real_chain_prior_metadata_trace_to_the_receipt():
    import pandas as pd

    if str(_E0) not in sys.path:
        sys.path.insert(0, str(_E0))
    import run_smoke_nasem_dryrun as SMOKE   # noqa: E402  (sets sys.dont_write_bytecode)

    core_sha = file_sha256(CORE)
    assert core_sha == _receipt_sha("data/restricted_local/nasem_t19_1_core.csv")
    core = pd.read_csv(CORE)
    reg = yaml.safe_load((REPO / "configs" / "constraints.yaml").read_text(encoding="utf-8"))
    _, _, model_in = SMOKE.build_config(core, reg, mineral_cap_pct=SMOKE.MINERAL_CAP_PCT)
    model = build_uncertainty_model(SMOKE.build_uncertainty_spec(core, model_in, core_sha256=core_sha))
    prior = model.prior_states(RandomStreams(1103), "opt", 64)
    md = prior.metadata
    assert md is not None and md.origin == "factory_model_metadata" and md.model_fingerprint == model.fingerprint()
    registry = (REPO / "sources" / "source_registry.csv").read_text(encoding="utf-8")
    n_sourced = n_stochastic_sourced = 0
    for cm in md.components:
        if cm.provenance_status != "sourced":
            assert cm.provenance_status == "synthetic_test_only" and not cm.is_stochastic   # smoke placeholders
            continue
        n_sourced += 1
        row = core[(core.ingredient_id == cm.component.ingredient_id) & (core.nutrient_id == cm.component.component)]
        assert len(row) == 1, cm.component.label()
        r = row.iloc[0]
        expect = stable_hash("nasem_t19_1_core_row/v1", core_sha, str(r.core_id), float(r["mean"]),
                             None if pd.isna(r["sd"]) else float(r["sd"]),
                             None if pd.isna(r["n"]) else str(r["n"]), str(r.entry_agreement))
        assert cm.data_fingerprint == expect, cm.component.label()      # row -> table sha -> receipt
        assert cm.source_id in registry and cm.locator
        if cm.is_stochastic:
            n_stochastic_sourced += 1
            assert cm.variance_basis == "observed_incl_sampling_and_lab"     # DC-01: NASEM SDs are observed
            assert cm.measurement_model_id == SMOKE.MEASUREMENT_MODEL_NASEM and cm.decomposition_id == "none"
    assert n_stochastic_sourced == sum(1 for c in md.components if c.is_stochastic) > 0
    assert n_sourced >= n_stochastic_sourced
    # the guard reads this basis from the object: a sampled assay on a NASEM cell is refused without any
    # declaration, and declaring the SD a true-state SD is a conflict
    comp = next(c.component for c in md.components if c.is_stochastic and c.component.component == "CP")
    em = ComponentErrorModel(comp, 0.004, 0.003, provenance={"sampling_sd": SYN, "lab_repeatability_sd": SYN},
                             is_synthetic=True)
    sm = SignalModel("k2_real_chain", (em,), is_synthetic=True)
    assert double_count_guard(sm, prior=prior).status == "violation"
    with pytest.raises(MetadataConflictError):
        double_count_guard(sm, {comp: "true_batch_state"}, prior=prior)
    rows = md.trace_rows()
    assert rows and all(set(r) >= {"data_fingerprint", "source_id", "locator", "variance_basis"} for r in rows)
    assert not any(k in r for r in rows for k in ("mean", "sd", "target_mean", "target_sd"))


# =================================================================================================
# 9  error values trace to sources/error_source_locators.csv
# =================================================================================================

LOCATORS = REPO / "sources" / "error_source_locators.csv"


def test_error_values_trace_to_the_locator_table():
    loc = load_error_locators(LOCATORS)
    rec = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert file_sha256(LOCATORS) == rec["sections"]["G_error_parameter_evidence"]["locators_sha256"]
    entries = {e["param_id"]: e for e in
               yaml.safe_load((REPO / "configs" / "assays.yaml").read_text(encoding="utf-8"))["error_parameters"]["entries"]}

    def pick(components):
        return next(pid for pid, r in loc.items()
                    if r["error_components"] == components and r["scale"] == "additive"
                    and r["unit"] == "percentage_points" and r["value_status"] == "value_present"
                    and r["verification"] != "blocked" and entries.get(pid, {}).get("value") is not None)

    def model(pid, fname, semantics="repeatability_only", source_id=None, status=ValueStatus.SOURCED):
        r = loc[pid]
        c = ObservedComponent("ing", r["nutrient"])
        v = float(entries[pid]["value"]) / 100.0          # percentage points -> fraction (value from assays.yaml)
        kw = {"sampling_sd": 0.0, "lab_repeatability_sd": 0.0, "lab_bias_sd": 0.0}
        kw[fname] = v
        p = Provenance(status, source_id=source_id or r["registry_source_id"].split("|")[0],
                       locator=f"sources/error_source_locators.csv#{pid}")
        em = ComponentErrorModel(c, lab_error_semantics=semantics, provenance={fname: p}, is_synthetic=False,
                                 measurement_model_id=pid, **kw)
        return SignalModel(f"trace:{pid}", (em,), is_synthetic=False)

    lab = pick("lab_random")
    (row,) = trace_error_provenance(model(lab, "lab_repeatability_sd"), loc)
    assert row["status"] == "traced" and row["param_id"] == lab and not row["issues"]
    assert row["session_cache_sha256"] == loc[lab]["session_cache_sha256"] and row["persistent_locator"]
    batch = pick("batch_true")                           # true batch variation used as a sampling error
    (row,) = trace_error_provenance(model(batch, "sampling_sd"), loc)
    assert row["status"] == "issues" and any("true batch variation" in i for i in row["issues"])
    combo = pick("sampling|lab_random")                  # inseparable combined SD put into one field
    (row,) = trace_error_provenance(model(combo, "lab_repeatability_sd"), loc)
    assert any("do not match field" in i for i in row["issues"])
    (row,) = trace_error_provenance(model(lab, "lab_repeatability_sd", source_id="SRC-NOT-THIS-ONE"), loc)
    assert any("is not the registry source" in i for i in row["issues"])
    (row,) = trace_error_provenance(model(lab, "lab_bias_sd"), loc)
    assert any("do not match field" in i for i in row["issues"])
    syn = _signal(0.004, 0.0)
    (row,) = trace_error_provenance(syn, loc)
    assert row["status"] == "synthetic_not_traced"
    unknown = SignalModel("u", (ComponentErrorModel(FCP, 0.004, 0.0, provenance={
        "sampling_sd": Provenance(ValueStatus.SOURCED, source_id="SRC-X", locator="Table 9")}),))
    (row,) = trace_error_provenance(unknown, loc)
    assert any("not a param_id" in i for i in row["issues"])
