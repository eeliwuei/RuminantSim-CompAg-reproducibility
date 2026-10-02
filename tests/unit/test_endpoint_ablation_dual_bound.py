"""FIX_PKG (round-3 red team D-3): the endpoint-ablation driver records the MILP dual bound.

``ration_reliability.optimization.chance_saa`` stores HiGHS' dual bound under ``mip_dual_bound_x_space``; the driver
``experiments/E1_cost_reliability/run_endpoint_ablation.py`` looked it up as ``mip_dual_bound`` and therefore wrote
null into every dual-bound field of the development run ``pilot-20260925T055921Z-fb4f75af`` (frozen at 46c22cb; that
run and its artefacts are not changed -- the fix applies to runs of a later freeze).  Checked here, on a synthetic
problem (``is_synthetic=True``; a few-millisecond MILP of the engine's own test problem, not a research solve):

1. the driver's lookup accepts both keys, the actual key first;
2. with a feasible MILP solution the driver's solver fields and candidate fields carry a finite dual bound, equal to
   the one chance_saa stored and not above the MILP objective (minimisation); an LP method gives none;
3. the driver no longer looks the dual bound up under the legacy key alone.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from engine_test_helpers import conc, dm_offer, ing, problem
from ration_reliability.datamodel import SolveStatus, SolverOptions
from ration_reliability.optimization import get_method
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"


def _load_driver():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("run_endpoint_ablation_dual_bound_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load_driver()
EXACT = SolverOptions(mip_rel_gap=0.0, time_limit_s=60.0)     # the driver always sets a time limit


def _three_ing():
    F = ing("F", 0.40, {"CP": 0.10, "NDF": 0.55}, forage=1.0)
    G = ing("G", 0.88, {"CP": 0.09, "NDF": 0.12})
    P = ing("P", 0.89, {"CP": 0.46, "NDF": 0.14})
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0), conc("ndf_max", {"NDF": 1.0}, "le", 40.0)]
    return problem([F, G, P], ["CP", "NDF"], cons, {"F": 0.03, "G": 0.20, "P": 0.40},
                   problem_id="syn_three_ing_dual_bound")


def _opt_draws(pr, n=64):
    sd = np.array([[0.012, 0.03], [0.008, 0.015], [0.02, 0.015]])
    model = IndependentNormalModel("syn_n3_dual", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), sd,
                                   pr.dm_estimates(), np.array([0.02, 0.01, 0.01]), truncate=False, is_synthetic=True)
    return model.draw(RandomStreams(3301), "opt", n)


def _method_run(res, method_id):
    cand = SimpleNamespace(value=0.05, status="selected", meets_screen=True, rate_upper=0.01, cost=res.objective,
                           solve_result=res)
    sel = SimpleNamespace(candidates=[cand], status="selected", param_name="alpha_train", selected_value=0.05)
    return SimpleNamespace(result=res, selection=sel, has_ration=res.decision is not None,
                           spec=SimpleNamespace(method_id=method_id))


def test_the_dual_bound_lookup_prefers_the_key_chance_saa_actually_writes():
    assert AB.DUAL_BOUND_KEYS[0] == "mip_dual_bound_x_space" and "mip_dual_bound" in AB.DUAL_BOUND_KEYS
    assert AB._dual_bound({"mip_dual_bound_x_space": 1.25}) == 1.25
    assert AB._dual_bound({"mip_dual_bound": 2.5}) == 2.5                                   # legacy key still read
    assert AB._dual_bound({"mip_dual_bound_x_space": 1.25, "mip_dual_bound": 2.5}) == 1.25   # actual key first
    assert AB._dual_bound({"inner": {"mip_dual_bound_x_space": 3.0}}) == 3.0                  # nested diagnostics
    assert AB._dual_bound({"mip_dual_bound_x_space": float("-inf")}) is None                 # not finite: none
    assert AB._dual_bound(None) is None and AB._dual_bound({}) is None


def test_the_dual_bound_is_not_null_when_the_milp_has_a_feasible_solution():
    pr = _three_ing()
    res = get_method("M2_joint_chance_saa")(pr, opt_draws=_opt_draws(pr), params={"alpha_train": 0.05,
                                                                                "n_scenarios": 64},
                                            solver_options=EXACT)
    assert res.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT) and res.decision is not None
    stored = res.diagnostics["mip_dual_bound_x_space"]
    assert stored is not None and math.isfinite(float(stored))
    assert "mip_dual_bound" not in res.diagnostics             # the key the driver used to look up does not exist
    fields = AB.solve_fields(_method_run(res, "M2_joint_chance_saa"), EXACT)
    assert fields["mip_dual_bound"] is not None and math.isfinite(fields["mip_dual_bound"])
    assert fields["mip_dual_bound"] == float(stored)
    assert fields["_candidates"][0]["mip_dual_bound"] == float(stored)
    milp_obj = float(res.diagnostics["milp_objective_x_space"])
    assert fields["mip_dual_bound"] <= milp_obj + 1e-7 * max(1.0, abs(milp_obj))          # a lower bound (minimum)
    # an LP method has no MILP dual bound: the field stays empty (never a made-up value)
    m0 = get_method("M0_nominal")(pr, solver_options=EXACT)
    assert m0.status is SolveStatus.OPTIMAL
    assert AB.solve_fields(_method_run(m0, "M0_nominal"), EXACT)["mip_dual_bound"] is None


def test_the_driver_never_looks_up_the_dual_bound_under_the_legacy_key_alone():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    legacy = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_find_key"
              and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "mip_dual_bound"]
    assert legacy == []
    src = DRIVER.read_text(encoding="utf-8")
    assert src.count("_dual_bound(") >= 4        # definition + solver fields + candidate fields + frontier points
