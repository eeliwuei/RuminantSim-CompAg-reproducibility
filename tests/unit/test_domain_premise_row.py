"""Reference problem v2 (D-533; official-run plan batch 1): the Table 5-1 starch-source premise as a planning row.

What is checked:

1. ``nutrition.domain.premise_planning_coefficients`` = ``(1{i in counted} - tau) * nominal starch_i``; at tau = 0.5 it
   equals the compiled nominal content of a ``DIAG-T51-DGC-STARCH-SHARE``-style row (weights +0.5 / -0.5), and the row
   built by ``experiments/E1_cost_reliability/official_v2.py`` compiles to exactly these coefficients as a
   ``structural_hard`` / ``decision_estimate`` row ``>= 0``; v2 = v1 + that one row (nothing else changes);
2. on the planned diet the row's structural verdict flips exactly at share = tau (up to the 1e-6 kg/d tolerance), for
   tau = 1/3, 1/2, 2/3, and agrees with the plan-level domain status ``plan_domain_status`` (same share definition);
3. guards: bad tau, missing counted ingredient, a second premise row, a problem that is not the one of ``cfg0``;
4. holder side (restricted dev_case_v1 inputs present and the preflight READY; skipped otherwise): v2 builds and
   validates; every arm (FULL11 = cfg0_v2, PART6P5 = ``s2_problem_cfg``, MAIN9 = ``planned_arm_cfg``) contains the
   row with the same coefficients; ``s2_consistency`` and ``planned_arm_consistency`` still pass; M0 of v2 meets the
   row.  Only booleans and counts are asserted or shown -- no restricted value is printed, even on failure.

Sections 1-3 use the SYNTHETIC toy problem ``data/synthetic_test_only/engine_toy_problem_v1.yaml`` (invented values).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ration_reliability.build import dev_case as DC
from ration_reliability.build import preflight as PF
from ration_reliability.datamodel import ConstraintClass, DMSource
from ration_reliability.evaluation import evaluate
from ration_reliability.io import build_problem
from ration_reliability.nutrition import domain as DOM

REPO = Path(__file__).resolve().parents[2]
TOY = REPO / "data" / "synthetic_test_only" / "engine_toy_problem_v1.yaml"
_OV2 = REPO / "experiments" / "E1_cost_reliability" / "official_v2.py"
_AB = REPO / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"


def _load(path: Path, name: str):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                 # module import reads no data (no main() is called)
    return mod


OV = _load(_OV2, "official_v2_under_test")
SYN_SRC = "SYN-P4A-001"
COUNTED = ("syn_grain",)


def _toy_cfg() -> dict:
    """The synthetic toy problem + a DIAG-T51-style diagnostic row (weights +0.5 for syn_grain, -0.5 otherwise)."""
    cfg = yaml.safe_load(TOY.read_text(encoding="utf-8"))
    for g in cfg["ingredients"]:
        g.setdefault("coefficients", {})["dgc_starch_share_w"] = {
            "value": 0.5 if g["ingredient_id"] in COUNTED else -0.5, "unit": "1", "status": "synthetic_test_only",
            "source_id": SYN_SRC}
    cfg["constraints"].append({
        "constraint_id": "DIAG-T51-DGC-STARCH-SHARE", "kind": "supply", "terms": {"C:dgc_starch_share_w:starch": 1.0},
        "sense": "ge", "bound": {"value": 0.0, "unit": "kg/d", "basis": "none", "status": "synthetic_test_only",
                                 "source_id": SYN_SRC},
        "constraint_class": "diagnostic_only", "numerical_tolerance": 1e-6, "dm_source": "scenario"})
    return cfg


@pytest.fixture(scope="module")
def toy():
    cfg = _toy_cfg()
    problem, rep = build_problem(cfg, mode="smoke")
    assert rep.ok, rep.errors
    return cfg, problem


def _nominal_content(problem, cid: str) -> np.ndarray:
    cc = problem.compiled
    k = list(cc.constraint_ids).index(cid)
    return np.einsum("ij,ij->i", np.asarray(cc.W[k]), problem.nominal_theta()) + np.asarray(cc.w0[k])


# =================================================================================================
# 1 coefficients
# =================================================================================================

def test_coefficients_equal_the_compiled_diag_t51_content_at_tau_one_half(toy):
    cfg, problem = toy
    th0 = problem.nominal_theta()
    coef = DOM.premise_planning_coefficients(th0, problem.ingredient_ids, problem.nutrient_ids, COUNTED, 0.5)
    # hand values (toy starch % DM: wet forage 30, dry forage 2, grain 70, protein meal 2, mineral 0)
    np.testing.assert_allclose(coef, [-0.15, -0.01, 0.35, -0.01, 0.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(coef, _nominal_content(problem, "DIAG-T51-DGC-STARCH-SHARE"), rtol=0, atol=1e-15)
    # another tau: (1{counted} - tau) * starch
    c13 = DOM.premise_planning_coefficients(th0, problem.ingredient_ids, problem.nutrient_ids, COUNTED, 1 / 3)
    np.testing.assert_allclose(c13, [-0.1, -0.02 / 3, 0.7 * 2 / 3, -0.02 / 3, 0.0], rtol=0, atol=1e-15)
    # the v2 builder: one structural row, compiled to exactly these coefficients; everything else unchanged
    cfg2, p2, rec = OV.build_v2_cfg0(cfg, problem, tau=0.5, counted_ids=COUNTED, mode="smoke")
    assert all(rec["checks"].values()) and rec["validator"]["ok"], rec["checks"]
    cc = p2.compiled
    k = list(cc.constraint_ids).index(OV.PREMISE_ROW_ID)
    assert cc.classes[k] is ConstraintClass.STRUCTURAL_HARD and cc.dm_sources[k] is DMSource.DECISION_ESTIMATE
    np.testing.assert_allclose(cc.w0[k], coef, rtol=0, atol=1e-15)
    np.testing.assert_allclose(_nominal_content(p2, OV.PREMISE_ROW_ID), _nominal_content(p2, "DIAG-T51-DGC-STARCH-SHARE"),
                               rtol=0, atol=1e-15)
    assert float(cc.tol[k]) == pytest.approx(OV.PREMISE_TOLERANCE_KG_D) and float(cc.bound[k]) == 0.0
    assert set(cc.constraint_ids) - set(problem.compiled.constraint_ids) == {OV.PREMISE_ROW_ID}
    assert p2.problem_id == "engine_toy_problem_v1|v2" and cfg["problem_id"] == "engine_toy_problem_v1"
    assert OV.PREMISE_ROW_ID not in {c["constraint_id"] for c in cfg["constraints"]}      # cfg0 not modified
    assert rec["inherited_by"][:3] == ["FULL11", "PART6P5", "MAIN9"] and rec["tau_status"] == "research_assumption"
    assert OV.PREMISE_ROW_ID == DOM.PREMISE_PLANNING_ROW_ID
    # the restricted-only list extends the dev-case driver's list by the premise row's planned margin
    assert OV.REDACT_NOMINAL_V2 == frozenset(OV.DRV.REDACT_NOMINAL) | {OV.PREMISE_ROW_ID}


# =================================================================================================
# 2 the structural verdict flips exactly at share = tau and agrees with the plan-level status
# =================================================================================================

def _plan_at_share(problem, s: float) -> np.ndarray:
    """q (kg as fed) with planned DM 20 kg: wet forage 10 (starch 3.0 kg), mineral 0.2, grain + dry forage 9.8 split so
    that the grain share of planned starch is exactly s:  0.7 x_g (1 - s) = s (3.0 + 0.02 x_d),  x_g = 9.8 - x_d."""
    x_d = (0.7 * (1 - s) * 9.8 - 3.0 * s) / (0.7 * (1 - s) + 0.02 * s)
    x = {"syn_forage_wet": 10.0, "syn_forage_dry": x_d, "syn_grain": 9.8 - x_d, "syn_protein_meal": 0.0,
         "syn_mineral": 0.2}
    assert min(x.values()) >= 0.0
    return np.array([x[i] for i in problem.ingredient_ids]) / problem.dm_estimates()


@pytest.mark.parametrize("tau", [1 / 3, 0.5, 2 / 3])
def test_structural_ok_flips_exactly_at_share_tau(toy, tau):
    cfg, problem = toy
    _, p2, _ = OV.build_v2_cfg0(cfg, problem, tau=tau, counted_ids=COUNTED, mode="smoke")
    th0, dh = p2.nominal_theta(), p2.dm_estimates()
    spec = DOM.Table51DomainSpec(f"syn-{tau:.3f}", COUNTED, tau, role="sensitivity")
    seen = []
    # planned starch is about 6 kg/d, so share - tau = -1e-8 is a margin of about -6e-8 kg/d: inside the 1e-6 tolerance
    for ds, expected_ok in [(-1e-2, False), (-1e-4, False), (-1e-8, True), (0.0, True), (1e-4, True), (1e-2, True)]:
        q = _plan_at_share(p2, tau + ds)
        ev = evaluate(q, th0[None], dh[None], p2.compiled, d_hat=dh, prices=p2.prices)
        sid = list(ev.structural_ids)
        k = sid.index(OV.PREMISE_ROW_ID)
        viol = bool(np.atleast_2d(ev.structural_violated)[0, k])
        margin = float(np.atleast_2d(ev.structural_margin)[0, k])
        # every other structural row holds on these plans: structural_ok is decided by the premise row alone
        others_ok = not np.atleast_2d(ev.structural_violated)[0, [j for j in range(len(sid)) if j != k]].any()
        assert others_ok and bool(ev.structural_ok) is expected_ok and viol is (not expected_ok), (tau, ds)
        pl = DOM.plan_domain_status(q, th0, dh, p2.ingredient_ids, p2.nutrient_ids, spec)
        assert pl.planned_counted_starch_share == pytest.approx(tau + ds, abs=1e-12)
        assert pl.planned_starch_margin_kg_d == pytest.approx(margin, abs=1e-12)          # same linear form
        assert pl.in_domain is expected_ok and (pl.status == DOM.DOMAIN_IN) is (not viol)  # same verdict
        seen.append(expected_ok)
    assert seen == [False, False, True, True, True, True]


def test_plan_status_record_has_labels_only_and_broadcasts(toy):
    cfg, problem = toy
    q = _plan_at_share(problem, 0.4)
    spec = DOM.default_table51_variants(COUNTED)[0]
    pl = DOM.plan_domain_status(q, problem.nominal_theta(), problem.dm_estimates(), problem.ingredient_ids,
                                problem.nutrient_ids, spec)
    assert pl.status == DOM.DOMAIN_OUT and not pl.in_domain
    rec = pl.to_record()
    assert "share" not in " ".join(rec) and "margin" not in " ".join(rec) and rec["status"] == DOM.DOMAIN_OUT
    dr = pl.as_domain_result(5)
    assert list(dr.status) == [DOM.DOMAIN_OUT] * 5 and not pl.state_mask(5).any()
    # no starch in the plan: the row holds (0 >= 0) but the share, and the plan status, is undefined
    q0 = np.zeros(len(problem.ingredient_ids))
    q0[list(problem.ingredient_ids).index("syn_mineral")] = 1.0
    assert DOM.plan_domain_status(q0, problem.nominal_theta(), problem.dm_estimates(), problem.ingredient_ids,
                                  problem.nutrient_ids, spec).status == DOM.DOMAIN_UNDEFINED


# =================================================================================================
# 3 guards
# =================================================================================================

def test_guards(toy):
    cfg, problem = toy
    th0, ids, nuts = problem.nominal_theta(), problem.ingredient_ids, problem.nutrient_ids
    for tau in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="tau"):
            DOM.premise_planning_coefficients(th0, ids, nuts, COUNTED, tau)
    with pytest.raises(KeyError, match="counted"):
        DOM.premise_planning_coefficients(th0, ids, nuts, ("not_a_feed",), 0.5)
    with pytest.raises(ValueError, match="unique"):
        DOM.premise_planning_coefficients(th0, ids, nuts, COUNTED * 2, 0.5)
    bad = th0.copy()
    bad[0, list(nuts).index("starch")] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DOM.premise_planning_coefficients(bad, ids, nuts, COUNTED, 0.5)
    cfg2, p2, _ = OV.build_v2_cfg0(cfg, problem, counted_ids=COUNTED, mode="smoke")
    with pytest.raises(ValueError, match="already has"):
        OV.add_domain_premise_row(cfg2, p2, counted_ids=COUNTED)
    with pytest.raises(ValueError, match="not the built problem"):
        OV.add_domain_premise_row(cfg, p2, counted_ids=COUNTED)


# =================================================================================================
# 4 holder side: dev_case_v1 (booleans and counts only)
# =================================================================================================
needs_restricted = pytest.mark.skipif(not (REPO / DC.DEV_CASE_V1.constants_file).is_file(),
                                      reason="restricted dev_case inputs not present (holder-side check only)")


@pytest.fixture(scope="module")
def dev_v2():
    AB = _load(_AB, "run_endpoint_ablation_for_v2_test")
    rep = PF.preflight_dev_case(REPO, driver="run_dev_case_v1", extra=AB.EXTRA_REQUIREMENTS)
    if rep["exit_code"] != PF.EXIT_READY:
        pytest.skip(f"dev_case inputs not READY (preflight {rep['status']}); holder-side check skipped")
    cwd = os.getcwd()
    os.chdir(REPO)                                   # the drivers' paths are relative to the project root
    try:
        from ration_reliability.evaluation.reference import load_reference_constraints
        from ration_reliability.io import load_problem
        DRV = AB.DRV
        problem, _ = load_problem(DRV.PROBLEM_YAML, mode="pilot")
        cfg0 = yaml.safe_load(DRV.PROBLEM_YAML.read_text(encoding="utf-8"))
        cfg_v2, p_v2, rec = OV.build_v2_cfg0(cfg0, problem)
        table_v2 = load_reference_constraints(OV.REFERENCE_CSV_V2)
        planned = AB.main9_planned_rows(table_v2)
        cfg_s2, _ = DRV.s2_problem_cfg(cfg_v2, p_v2)
        p_s2, rep_s2 = build_problem(cfg_s2, mode="pilot")
        cfg_m9, _ = AB.planned_arm_cfg(cfg_v2, p_v2, planned, "MAIN9")
        p_m9, rep_m9 = build_problem(cfg_m9, mode="pilot")
        yield {"AB": AB, "DRV": DRV, "problem_v1": problem, "problem_v2": p_v2, "record": rec, "planned": planned,
               "arms": {"FULL11": p_v2, "PART6P5": p_s2, "MAIN9": p_m9}, "arm_validators_ok": rep_s2.ok and rep_m9.ok}
    finally:
        os.chdir(cwd)


@needs_restricted
def test_holder_every_arm_contains_the_row_and_the_consistency_checks_still_pass(dev_v2):
    from ration_reliability.datamodel import SolverOptions
    rec = dev_v2["record"]
    ok_checks = all(rec["checks"].values())
    assert ok_checks and rec["validator"]["ok"] and rec["validator"]["n_pending"] == 0, "v2 build checks failed"
    assert dev_v2["arm_validators_ok"], "an arm built from cfg0_v2 does not validate"
    p_v2 = dev_v2["problem_v2"]
    k0 = list(p_v2.compiled.constraint_ids).index(OV.PREMISE_ROW_ID)
    w_ref = np.asarray(p_v2.compiled.w0[k0])
    for arm, p in dev_v2["arms"].items():
        cc = p.compiled
        has = OV.PREMISE_ROW_ID in cc.constraint_ids
        assert has, f"{arm} lacks the premise row"
        k = list(cc.constraint_ids).index(OV.PREMISE_ROW_ID)
        same = bool(np.array_equal(np.asarray(cc.w0[k]), w_ref))
        structural = cc.classes[k] is ConstraintClass.STRUCTURAL_HARD and cc.dm_sources[k] is DMSource.DECISION_ESTIMATE
        assert same and structural, f"{arm}: the premise row differs from the v2 FULL11 row"
        n_extra = len(set(cc.constraint_ids) - set(dev_v2["arms"]["FULL11"].compiled.constraint_ids))
        assert (arm, n_extra) in {("FULL11", 0), ("PART6P5", 5), ("MAIN9", 2)}, (arm, n_extra)   # SH-PLAN-* rows
    assert p_v2.problem_id == "dev_case_v1|v2"
    assert dev_v2["arms"]["PART6P5"].problem_id.endswith("|v2|S2") and dev_v2["arms"]["MAIN9"].problem_id.endswith(
        "|v2|MAIN9")
    opts = SolverOptions(time_limit_s=60.0, mip_rel_gap=1e-4)
    AB, DRV = dev_v2["AB"], dev_v2["DRV"]
    try:                                             # SystemExit messages carry differences; keep them out of the log
        s2_ok = bool(DRV.s2_consistency(p_v2, dev_v2["arms"]["PART6P5"], opts)["ok"])
    except SystemExit:
        s2_ok = False
    try:
        m9_ok = bool(AB.planned_arm_consistency(p_v2, dev_v2["arms"]["MAIN9"], dev_v2["planned"], opts)["ok"])
    except SystemExit:
        m9_ok = False
    assert s2_ok, "s2_consistency failed on the v2 cfg0 (details withheld)"
    assert m9_ok, "planned_arm_consistency (MAIN9) failed on the v2 cfg0 (details withheld)"
    # M0 of v2 meets every structural row, the premise row included, and is inside the plan-level primary domain
    m0 = AB.get_method("M0_nominal")(p_v2, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    has_ration = m0.decision is not None
    assert has_ration, "M0 of v2 has no ration"
    ev = evaluate(m0.decision, p_v2.nominal_theta()[None], p_v2.dm_estimates()[None], p_v2.compiled, prices=p_v2.prices)
    m0_ok = bool(ev.structural_ok)
    prim = DOM.default_table51_variants((OV.DGC_INGREDIENT,))[0]
    in_dom = DOM.plan_domain_status(m0.decision, p_v2.nominal_theta(), m0.decision.d_hat, p_v2.ingredient_ids,
                                    p_v2.nutrient_ids, prim).in_domain
    assert m0_ok and in_dom, "M0 of v2 violates a structural row or lies outside the plan-level domain"
