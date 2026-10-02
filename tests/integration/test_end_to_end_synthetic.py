"""Synthetic end-to-end integration test (synthetic_test_only; software check, not a research run).

Every core method hands over an executed ration ``q`` (kg as-fed/head/d) for the same synthetic
problem (``data/synthetic_test_only/engine_toy_problem_v1.yaml``) and the same ``opt`` draws:

* M0 ``M0_nominal`` (nominal point, no draws);
* M1 ``M1_safety_margin`` with ``k`` selected on the ``validation`` stream;
* M2 ``M2_joint_chance_saa`` with N = 128 ``opt`` scenarios (and M2b, the Bonferroni variant);
* M3a ``M3a_box_robust`` with ``k`` selected on ``validation``; M3b ``M3b_budget_robust`` with
  ``Gamma`` selected on ``validation`` at a fixed box; M3c ``M3c_scenario_set_robust`` on the same
  128 ``opt`` states.

One public evaluator scores every ``q`` on one independent ``test`` DrawSet (common random numbers).
The run goes through ``experiments/E0_verification/smoke_pipeline.py`` -- the same orchestration code
that the NASEM smoke dry-run uses -- so this test also covers that code.

Checked: structural hard constraints are never relaxed (independent recomputation from ``q`` and
``d_hat``, also with a large chance budget); failures and ``not_met`` selections stay in the tables
without cost (never 0); the ``test`` stream is never available to a method or a selection;
theory-backed cross-method identities (Gamma = 0 / full, alpha = 0 SAA = scenario-set robust,
SAA <= scenario-set robust cost, in-sample budgets).

All numbers are synthetic (``is_synthetic=True``); the uncertainty model (independent truncated
normal, CV 3 % / 2 %) is a test device, not a research assumption.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from ration_reliability.datamodel import (
    ConstraintClass,
    RationDecision,
    SolverOptions,
    SolveResult,
    SolveStatus,
)
from ration_reliability.errors import InvalidProblemError, LeakageError
from ration_reliability.evaluation import evaluate, evaluate_drawset
from ration_reliability.io import load_problem
from ration_reliability.optimization import REGISTRY, get_method
from ration_reliability.optimization.chance_saa import allowed_violations
from ration_reliability.optimization.safety_margin import select_parameter_on_validation
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams

from engine_test_helpers import conc

_E0 = Path(__file__).resolve().parents[2] / "experiments" / "E0_verification"
if str(_E0) not in sys.path:
    sys.path.insert(0, str(_E0))
import smoke_pipeline as SP  # noqa: E402

SEED = 1103
N_OPT, N_VAL, N_TEST = 128, 4000, 20000
ALPHA = 0.05
K_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)          # sd-scale test grid (not the protocol grid)
GAMMA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, "full")
K_FOR_BUDGET = 2.0
OPTS = SolverOptions(mip_rel_gap=1e-6, time_limit_s=120)
SCREEN = {"target_alpha": ALPHA, "screening_rule": "rate_upper_le_alpha"}

CORE_METHODS = ("M0_nominal", "M1_safety_margin", "M2_joint_chance_saa", "M2b_marginal_bonferroni_saa",
                "M3a_box_robust", "M3b_budget_robust", "M3c_scenario_set_robust")

SPECS = (
    SP.MethodSpec("M0_nominal", uses_opt_draws=False),
    SP.MethodSpec("M1_safety_margin", params={"margin_scale": "sd", "apply_to_dm": True},
                  selection={"param_name": "k", "grid": K_GRID, **SCREEN}),
    SP.MethodSpec("M2_joint_chance_saa", params={"alpha_train": ALPHA, "n_scenarios": N_OPT}),
    SP.MethodSpec("M2b_marginal_bonferroni_saa", params={"alpha_train": ALPHA, "n_scenarios": N_OPT}),
    SP.MethodSpec("M3a_box_robust", selection={"param_name": "k", "grid": K_GRID, **SCREEN}),
    SP.MethodSpec("M3b_budget_robust", params={"k": K_FOR_BUDGET},
                  selection={"param_name": "gamma", "grid": GAMMA_GRID, **SCREEN}),
    SP.MethodSpec("M3c_scenario_set_robust"),
)


def _synthetic_model(problem, cv_theta=0.03, cv_d=0.02):
    th = problem.nominal_theta()
    det = np.array([not g.is_stochastic for g in problem.ingredients])
    sd = np.abs(th) * cv_theta
    sd[det] = 0.0
    dh = problem.dm_estimates()
    dsd = dh * cv_d
    dsd[det] = 0.0
    upper = np.array([[1.0 if n.dimension == "mass_fraction" else np.inf for n in problem.nutrients]]
                     * len(problem.ingredients))
    return IndependentNormalModel("syn_e2e_indep_tn", problem.ingredient_ids, problem.nutrient_ids, th, sd, dh, dsd,
                                  truncate=True, theta_lower=0.0, theta_upper=upper, is_synthetic=True)


@pytest.fixture(scope="module")
def e2e(toy_yaml_path):
    problem, rep = load_problem(toy_yaml_path, mode="smoke")
    assert rep.ok and problem.is_synthetic
    model = _synthetic_model(problem)
    streams = RandomStreams(SEED)
    opt = model.draw(streams, "opt", N_OPT)
    val = model.draw(streams, "validation", N_VAL)
    test = model.draw(streams, "test", N_TEST)
    runs = SP.run_methods(problem, SPECS, opt_draws=opt, validation_draws=val, solver_options=OPTS)
    SP.evaluate_on_test(runs, problem, test)
    return {"problem": problem, "opt": opt, "val": val, "test": test, "runs": {r.spec.method_id: r for r in runs},
            "run_list": runs, "model": model, "streams": streams}


def _constraint(problem, cid):
    return next(c for c in problem.constraints if c.constraint_id == cid)


def _structural_ok_independent(problem, q, d_hat):
    """Recompute the toy structural rules from q and d_hat only (no engine code)."""
    x = q * d_hat
    dm = _constraint(problem, "dm_offer")
    share = _constraint(problem, "mineral_share_max")
    wet = _constraint(problem, "forage_wet_af_max")
    i_min = problem.ingredient_ids.index("syn_mineral")
    i_wet = problem.ingredient_ids.index("syn_forage_wet")
    ok_dm = abs(x.sum() - dm.bound) <= dm.numerical_tolerance
    ok_share = 100.0 * x[i_min] / x.sum() <= share.bound + share.numerical_tolerance
    ok_wet = q[i_wet] <= wet.bound + wet.numerical_tolerance
    ok_nonneg = bool(np.all(q >= -1e-9))
    return ok_dm and ok_share and ok_wet and ok_nonneg


# ------------------------------------------------------------------------------------------------
# 1. every core method runs end to end and hands over q
# ------------------------------------------------------------------------------------------------

def test_every_core_method_is_registered_and_hands_over_q(e2e):
    problem, runs = e2e["problem"], e2e["runs"]
    assert set(CORE_METHODS) <= set(REGISTRY), "core methods must be reachable through get_method"
    assert set(CORE_METHODS) <= set(runs)
    prices = problem.price_vector()
    for mid in CORE_METHODS:
        r = runs[mid]
        assert r.status == "optimal", (mid, r.status, None if r.result is None else r.result.message)
        res = r.result
        assert res.method_id == mid and res.is_synthetic
        dec = res.decision
        assert isinstance(dec, RationDecision) and dec.method_id == mid
        assert dec.ingredient_ids == problem.ingredient_ids
        q = np.asarray(dec.q_as_fed)
        assert q.shape == (len(problem.ingredient_ids),) and np.all(np.isfinite(q)) and np.all(q >= -1e-9)
        np.testing.assert_allclose(dec.d_hat, problem.dm_estimates(), rtol=0, atol=0)
        assert res.objective == pytest.approx(float(prices @ q), rel=1e-12, abs=1e-12)


# ------------------------------------------------------------------------------------------------
# 2. one public evaluator, one test DrawSet
# ------------------------------------------------------------------------------------------------

def test_single_public_evaluator_on_shared_test_draws(e2e):
    problem, runs, test = e2e["problem"], e2e["runs"], e2e["test"]
    cc = problem.compiled
    for mid in CORE_METHODS:
        r = runs[mid]
        ev = r.evaluation
        assert ev is not None and ev.draw_stream_id == test.stream_id == f"root={SEED}/test"
        assert ev.n_draws == N_TEST and ev.is_synthetic
        assert ev.cost == pytest.approx(r.result.objective, rel=1e-12)
        # the evaluator does not care who produced q: plain-array call gives identical output
        q = np.array(r.result.decision.q_as_fed)
        ev2 = evaluate(q, test.theta, test.d, cc, d_hat=problem.dm_estimates(), prices=problem.prices)
        assert ev2.q_hash == ev.q_hash
        np.testing.assert_array_equal(ev2.joint_violation, ev.joint_violation)
        np.testing.assert_array_equal(ev2.violated, ev.violated)
        np.testing.assert_allclose(ev2.margin, ev.margin, rtol=0, atol=0, equal_nan=True)
        # joint event = any probabilistic_nutrition violation; diagnostics never enter it
        prob = ev.probabilistic_mask
        np.testing.assert_array_equal(ev.joint_violation, ev.violated[:, prob].any(axis=1))
    diag_ids = [cc.constraint_ids[k] for k in cc.indices(ConstraintClass.DIAGNOSTIC_ONLY)]
    assert diag_ids == ["energy_supply_diag"]


# ------------------------------------------------------------------------------------------------
# 3. structural hard constraints are never relaxed
# ------------------------------------------------------------------------------------------------

def test_structural_constraints_hold_for_every_method(e2e):
    problem, runs = e2e["problem"], e2e["runs"]
    cc = problem.compiled
    struct_ids = {cc.constraint_ids[k] for k in cc.indices(ConstraintClass.STRUCTURAL_HARD)}
    diag_ids = {cc.constraint_ids[k] for k in cc.indices(ConstraintClass.DIAGNOSTIC_ONLY)}
    assert struct_ids == {"dm_offer", "mineral_share_max", "forage_wet_af_max"}
    for mid in CORE_METHODS:
        r = runs[mid]
        assert r.evaluation.structural_ok, mid
        assert not np.any(r.evaluation.structural_violated)
        dec = r.result.decision
        assert _structural_ok_independent(problem, np.array(dec.q_as_fed), np.array(dec.d_hat)), mid
        imposed = set(r.result.diagnostics["imposed_constraint_ids"])
        assert struct_ids <= imposed, (mid, struct_ids - imposed)
        assert not (diag_ids & imposed), (mid, "diagnostic_only constraints must not be imposed")


def test_large_chance_budget_does_not_relax_structural_constraints(e2e):
    problem, opt = e2e["problem"], e2e["opt"]
    m = allowed_violations(0.9, N_OPT)
    assert m == 115
    for mid in ("M2_joint_chance_saa", "M2b_marginal_bonferroni_saa"):
        res = get_method(mid)(problem, opt_draws=opt, params={"alpha_train": 0.9, "n_scenarios": N_OPT},
                              solver_options=OPTS)
        assert res.status is SolveStatus.OPTIMAL, (mid, res.message)
        dec = res.decision
        assert _structural_ok_independent(problem, np.array(dec.q_as_fed), np.array(dec.d_hat)), mid
        ev_opt = evaluate_drawset(dec, opt, problem.compiled, prices=problem.prices)
        assert ev_opt.structural_ok
        n_viol = int(ev_opt.joint_violation.sum())
        assert n_viol <= m
        if mid == "M2_joint_chance_saa":
            assert n_viol > 0, "the large budget must actually be used on probabilistic rows"
            assert res.objective <= e2e["runs"]["M2_joint_chance_saa"].result.objective * (1 + 1e-6)


# ------------------------------------------------------------------------------------------------
# 4. streams: selection only on validation, test never reachable
# ------------------------------------------------------------------------------------------------

def test_stream_separation_and_selection_provenance(e2e):
    runs, opt, val, test = e2e["runs"], e2e["opt"], e2e["val"], e2e["test"]
    for mid in CORE_METHODS:
        r = runs[mid]
        used = list(r.result.streams_used)
        if r.selection is not None:
            used += [u for c in r.selection.candidates for u in c.solve_result.streams_used]
        assert test.stream_id not in used and all(SP.stream_name(u) != "test" for u in used), mid
    assert runs["M0_nominal"].result.streams_used == ()
    for mid in ("M2_joint_chance_saa", "M2b_marginal_bonferroni_saa", "M3c_scenario_set_robust"):
        assert runs[mid].result.streams_used == (opt.stream_id,), mid
        assert runs[mid].selection is None
    for mid in ("M1_safety_margin", "M3a_box_robust", "M3b_budget_robust"):
        r = runs[mid]
        assert r.selection is not None and r.selection.status == "selected", mid
        assert r.selection.validation_stream_id == val.stream_id
        assert r.result.streams_used == (opt.stream_id, val.stream_id), (mid, r.result.streams_used)
        sel = r.result.params["selection"]
        assert sel["selected_on_stream"] == val.stream_id and sel["validation_n_draws"] == N_VAL
        assert sel["screening_rule"] == "rate_upper_le_alpha" and sel["target_alpha"] == ALPHA


def test_test_stream_is_refused_everywhere(e2e):
    problem, test, val, opt = e2e["problem"], e2e["test"], e2e["val"], e2e["opt"]
    calls = {
        "M0_nominal": {"coefficient_mode": "draw_mean"},
        "M1_safety_margin": {"k": 1.0},
        "M2_joint_chance_saa": {"alpha_train": ALPHA},
        "M2b_marginal_bonferroni_saa": {"alpha_train": ALPHA},
        "M3a_box_robust": {"k": 1.0},
        "M3b_budget_robust": {"k": 1.0, "gamma": 1.0},
        "M3c_scenario_set_robust": {},
    }
    for mid, params in calls.items():
        with pytest.raises(LeakageError):
            get_method(mid)(problem, opt_draws=test, params=params, solver_options=OPTS)
    with pytest.raises(LeakageError):
        select_parameter_on_validation(get_method("M1_safety_margin"), problem, test, param_name="k", grid=[0.0, 1.0],
                                       opt_draws=opt, **SCREEN)
    with pytest.raises(LeakageError):
        SP.run_methods(problem, SPECS[:1], opt_draws=test)
    with pytest.raises(LeakageError):
        SP.run_methods(problem, SPECS[:2], opt_draws=opt, validation_draws=test)
    with pytest.raises(LeakageError):
        SP.evaluate_on_test([], problem, val)          # final scoring only on the test stream
    # a method result that (hypothetically) used the test stream is refused before scoring
    r0 = e2e["runs"]["M0_nominal"]
    fake = SP.MethodRun(r0.spec, dataclasses.replace(r0.result, streams_used=(test.stream_id,)))
    with pytest.raises(LeakageError):
        SP.evaluate_on_test([fake], problem, test)


def test_validation_selection_is_lowest_cost_meeting_screen(e2e):
    runs = e2e["runs"]
    for mid in ("M1_safety_margin", "M3a_box_robust", "M3b_budget_robust"):
        sel = runs[mid].selection
        meets = [c for c in sel.candidates if c.meets_screen]
        assert meets, mid
        best = min(meets, key=lambda c: c.cost)
        assert sel.selected_value == best.value
        assert runs[mid].result.objective == pytest.approx(best.cost, rel=1e-12)
        for c in sel.candidates:
            if c.cost is None:                      # infeasible grid points carry no cost
                assert c.status != "optimal" and c.rate_upper is None and not c.meets_screen
                assert not c.solve_result.has_solution and c.solve_result.objective is None
            elif c.cost < best.cost - 1e-12:
                assert not c.meets_screen and c.rate_upper > ALPHA
            if c.meets_screen:
                assert c.rate_upper <= ALPHA and c.structural_ok


# ------------------------------------------------------------------------------------------------
# 5. theory-backed cross-method identities (same problem, same opt draws)
# ------------------------------------------------------------------------------------------------

def test_cross_method_identities_and_orderings(e2e):
    problem, opt, runs = e2e["problem"], e2e["opt"], e2e["runs"]
    cost = {mid: runs[mid].result.objective for mid in CORE_METHODS}
    # M1 is centred on the nominal table values: k >= 0 can only tighten the M0 rows
    assert cost["M1_safety_margin"] >= cost["M0_nominal"] - 1e-9
    # M3b: Gamma = 0 is the M0 row, Gamma = full is the M3a box row of the same box
    by_gamma = {c.value: c for c in runs["M3b_budget_robust"].selection.candidates}
    assert by_gamma[0.0].cost == pytest.approx(cost["M0_nominal"], rel=1e-9)
    box_same_k = get_method("M3a_box_robust")(problem, opt_draws=opt, params={"k": K_FOR_BUDGET}, solver_options=OPTS)
    assert box_same_k.has_solution
    assert by_gamma["full"].cost == pytest.approx(box_same_k.objective, rel=1e-9)
    feas = [c.cost for c in runs["M3b_budget_robust"].selection.candidates if c.cost is not None]
    assert all(b >= a - 1e-9 for a, b in zip(feas, feas[1:])), "budget-robust cost must not fall as Gamma grows"
    # M2 with m = floor(alpha N) >= M3c's feasible set on the same scenarios -> not more expensive
    assert cost["M2_joint_chance_saa"] <= cost["M3c_scenario_set_robust"] * (1 + 1e-6) + 1e-12
    # alpha_train = 0 turns the joint SAA into the scenario-set robust LP on the same states
    m2_zero = get_method("M2_joint_chance_saa")(problem, opt_draws=opt, params={"alpha_train": 0.0, "n_scenarios": N_OPT},
                                                solver_options=OPTS)
    assert m2_zero.objective == pytest.approx(cost["M3c_scenario_set_robust"], rel=1e-7)
    # Bonferroni budgets floor(alpha/K_p * N) are all 0 here (0.05/8*128 = 0.8), so M2b == M3c
    budgets = runs["M2b_marginal_bonferroni_saa"].result.diagnostics["row_budgets"]
    assert len(budgets) == len(SP.probabilistic_ids(problem)) == 8
    assert set(budgets.values()) == {0}
    assert cost["M2b_marginal_bonferroni_saa"] == pytest.approx(cost["M3c_scenario_set_robust"], rel=1e-7)


def test_in_sample_budgets_through_public_evaluator(e2e):
    problem, opt, runs = e2e["problem"], e2e["opt"], e2e["runs"]
    m = allowed_violations(ALPHA, N_OPT)
    assert m == 6
    ev_m2 = evaluate_drawset(runs["M2_joint_chance_saa"].result.decision, opt, problem.compiled)
    assert int(ev_m2.joint_violation.sum()) <= m
    ev_m3c = evaluate_drawset(runs["M3c_scenario_set_robust"].result.decision, opt, problem.compiled)
    assert int(ev_m3c.joint_violation.sum()) == 0
    # in-sample guarantees say nothing about the test stream: the test rate is only *reported*
    for mid in ("M2_joint_chance_saa", "M3c_scenario_set_robust"):
        s = runs[mid].evaluation.summary()
        assert 0.0 <= s["joint"]["rate_lower"] <= s["joint"]["rate_upper"] <= 1.0
        assert s["joint"]["mc_clopper_pearson"]["interval_type"] == SP.CI_TYPE


# ------------------------------------------------------------------------------------------------
# 6. tables: every method is a row; no ration -> no cost (never 0)
# ------------------------------------------------------------------------------------------------

_META = {"run_type": "unit_test", "scenario_id": "syn_toy", "animal_profile_id": "syn_cow_v1",
         "inventory_id": "syn_all", "price_id": "syn_prices_v1", "target_alpha": ALPHA, "cost_unit": "XXX/head/d",
         "independent_empirical_n": 0, "is_synthetic": True, "provenance_id": "SYN-P4A-001"}


def test_long_table_rows_and_csv_roundtrip(e2e, tmp_path):
    problem, runs, test = e2e["problem"], e2e["run_list"], e2e["test"]
    meta = dict(_META, test_stream_id=test.stream_id)
    rows = SP.long_table(runs, run_id="unit-e2e", meta=meta)
    n_eval = len(problem.compiled.indices(ConstraintClass.PROBABILISTIC_NUTRITION, ConstraintClass.DIAGNOSTIC_ONLY))
    assert n_eval == 9
    for r in runs:
        mine = [x for x in rows if x["method_label"] == r.spec.name]
        joint = [x for x in mine if x["row_type"] == "joint"]
        assert len(joint) == 1 and len(mine) == 1 + n_eval
        j = joint[0]
        assert j["cost"] == pytest.approx(r.result.objective, rel=1e-12)
        assert j["mc_draw_count"] == N_TEST and j["ci_type"] == SP.CI_TYPE and j["independent_empirical_n"] == 0
        assert j["rate_lower"] == pytest.approx(r.evaluation.joint_violation.mean())
        assert j["is_synthetic"] is True and j["structural_ok"] is True
        assert set(j) == set(SP.LONG_COLUMNS)
    p = SP.write_csv(rows, tmp_path / "evaluation_long.csv", SP.LONG_COLUMNS)
    with open(p, encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == len(rows) and tuple(back[0]) == SP.LONG_COLUMNS
    with pytest.raises(FileExistsError):
        SP.write_csv(rows, p, SP.LONG_COLUMNS)
    rations = SP.ration_table(runs, problem, run_id="unit-e2e")
    for r in runs:
        mine = [x for x in rations if x["method_label"] == r.spec.name]
        assert [x["ingredient_id"] for x in mine] == list(problem.ingredient_ids)
        assert sum(x["planned_dm_share"] for x in mine) == pytest.approx(1.0, abs=1e-12)
    prof = SP.nutrient_profile(runs[:1], problem, test, run_id="unit-e2e")
    dm = next(x for x in prof if x["quantity"] == "DM_supply")
    assert dm["planned_nominal"] == pytest.approx(20.0, abs=1e-6) and dm["n_undefined"] == 0
    assert dm["draw_p05"] < dm["draw_p50"] < dm["draw_p95"]          # realised DM is never renormalised


def test_failures_and_not_met_are_kept_without_cost(e2e, tmp_path):
    problem, opt, val, test = e2e["problem"], e2e["opt"], e2e["val"], e2e["test"]
    cp = _constraint(problem, "cp_min")
    impossible_cp = dataclasses.replace(problem, constraints=tuple(
        dataclasses.replace(c, bound=60.0) if c.constraint_id == "cp_min" else c for c in problem.constraints))
    assert cp.bound == pytest.approx(16.0) and cp.unit == "%"
    specs_same = (
        SP.MethodSpec("M1_safety_margin", params={"k": 10.0, "margin_scale": "sd", "apply_to_dm": True},
                      label="M1[k=10]"),
        SP.MethodSpec("M3a_box_robust", params={"k": 10.0}, label="M3a[k=10]"),
        SP.MethodSpec("M3b_budget_robust", params={"k": 10.0, "gamma": "full"}, label="M3b[k=10,full]"),
        SP.MethodSpec("M1_safety_margin", params={"margin_scale": "sd", "apply_to_dm": True}, label="M1[not_met]",
                      selection={"param_name": "k", "grid": (0.0, 0.5), "target_alpha": 0.0,
                                 "screening_rule": "cp_upper_le_alpha", "confidence": 0.95}),
        SP.MethodSpec("M0_nominal", uses_opt_draws=False, label="M0[ok]"),
    )
    runs = SP.run_methods(problem, specs_same, opt_draws=opt, validation_draws=val, solver_options=OPTS)
    specs_cp = (SP.MethodSpec("M0_nominal", uses_opt_draws=False, label="M0[cp60]"),
                SP.MethodSpec("M2_joint_chance_saa", params={"alpha_train": ALPHA, "n_scenarios": N_OPT},
                              label="M2[cp60]"),
                SP.MethodSpec("M3c_scenario_set_robust", label="M3c[cp60]"))
    runs_cp = SP.run_methods(impossible_cp, specs_cp, opt_draws=opt, solver_options=OPTS)
    SP.evaluate_on_test(runs, problem, test)
    SP.evaluate_on_test(runs_cp, impossible_cp, test)
    by = {r.spec.name: r for r in runs + runs_cp}
    for name in ("M1[k=10]", "M3a[k=10]", "M3b[k=10,full]", "M0[cp60]", "M2[cp60]", "M3c[cp60]"):
        r = by[name]
        assert r.status == "proven_infeasible", (name, r.status, r.result.message)
        assert r.result.decision is None and r.result.objective is None and r.evaluation is None
        assert not r.has_ration
    nm = by["M1[not_met]"]
    assert nm.status == "selection_not_met" and nm.result is None and nm.evaluation is None
    assert all(not c.meets_screen for c in nm.selection.candidates)
    assert by["M0[ok]"].status == "optimal" and by["M0[ok]"].evaluation is not None

    rows = SP.long_table(runs + runs_cp, run_id="unit-fail", meta=dict(_META, test_stream_id=test.stream_id))
    for name, r in by.items():
        mine = [x for x in rows if x["method_label"] == name]
        if r.has_ration:
            assert len(mine) > 1
            continue
        assert len(mine) == 1, name                      # kept in the denominator, one joint row
        row = mine[0]
        assert row["feasibility_status"] == r.status
        for k in ("cost", "rate_lower", "rate_upper", "n_violated", "mc_draw_count", "ci_lower", "ci_upper"):
            assert row[k] is None, (name, k)
    p = SP.write_csv(rows, tmp_path / "long.csv", SP.LONG_COLUMNS)
    with open(p, encoding="utf-8") as fh:
        back = {x["method_label"]: x for x in csv.DictReader(fh) if x["row_type"] == "joint"}
    for name in ("M1[k=10]", "M2[cp60]", "M1[not_met]"):
        assert back[name]["cost"] == "" and back[name]["rate_upper"] == ""      # empty, never "0"
    assert float(back["M0[ok]"]["cost"]) > 0
    rat = SP.ration_table(runs + runs_cp, problem, run_id="unit-fail")
    fail_rows = [x for x in rat if x["method_label"] == "M2[cp60]"]
    assert len(fail_rows) == 1 and fail_rows[0]["q_as_fed_kg_per_head_d"] is None
    # the engine itself forbids a zero-cost / empty-ration failure record
    r = by["M2[cp60]"].result
    with pytest.raises(InvalidProblemError):
        dataclasses.replace(r, objective=0.0)
    with pytest.raises(InvalidProblemError):
        dataclasses.replace(r, decision=RationDecision(problem.ingredient_ids, np.zeros(len(problem.ingredient_ids)),
                                                       problem.dm_estimates(), r.method_id))


def test_structurally_infeasible_problem_is_reported_by_every_method(e2e):
    problem, opt = e2e["problem"], e2e["opt"]
    extra = conc("mineral_share_min_contradiction", {"DM:syn_mineral": 1.0}, "ge", 5.0, "%",
                 cls="structural_hard", dm_source="decision_estimate")
    bad = dataclasses.replace(problem, constraints=problem.constraints + (extra,), problem_id="syn_struct_contradiction")
    calls = {
        "M0_nominal": ({}, None),
        "M1_safety_margin": ({"k": 1.0, "margin_scale": "sd", "apply_to_dm": True}, opt),
        "M2_joint_chance_saa": ({"alpha_train": ALPHA, "n_scenarios": N_OPT}, opt),
        "M2b_marginal_bonferroni_saa": ({"alpha_train": ALPHA, "n_scenarios": N_OPT}, opt),
        "M3a_box_robust": ({"k": 1.0}, opt),
        "M3b_budget_robust": ({"k": 1.0, "gamma": 1.0}, opt),
        "M3c_scenario_set_robust": ({}, opt),
    }
    for mid, (params, od) in calls.items():
        res = get_method(mid)(bad, opt_draws=od, params=params, solver_options=OPTS)
        assert res.status is SolveStatus.PROVEN_INFEASIBLE, (mid, res.status, res.message)
        assert res.decision is None and res.objective is None
        assert isinstance(res, SolveResult) and not math.isnan(res.wall_time_s)
