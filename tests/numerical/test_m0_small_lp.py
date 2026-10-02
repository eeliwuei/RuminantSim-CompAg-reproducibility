"""Contract section 8 required test 3: 2-3 ingredient LPs with analytic / enumerated optima.

Also checks the M0 solution on the synthetic toy problem against an independent vertex
enumeration (numpy.linalg only, no HiGHS).
"""

import numpy as np
import pytest

from engine_test_helpers import af_max, conc, dm_offer, enumerate_lp_vertices, ing, problem, two_ingredient_problem
from ration_reliability.datamodel import SolveStatus
from ration_reliability.evaluation import evaluate
from ration_reliability.nutrition import linear_rows
from ration_reliability.optimization import available_methods, get_method
from ration_reliability.optimization.lp_builder import assemble_x_space_lp, optimization_indices

M0 = get_method("M0_nominal")


def test_registry_contains_m0():
    assert "M0_nominal" in available_methods()
    with pytest.raises(KeyError):
        get_method("M9_unknown")


@pytest.mark.parametrize("cp", [12.0, 16.0, 25.0, 40.0])
def test_two_ingredient_analytic_optimum(cp):
    r = M0(two_ingredient_problem(cp_min_pct=cp))
    assert r.status is SolveStatus.OPTIMAL
    c = cp / 100.0
    x_c = 20.0 * (c - 0.10) / 0.30
    x_f = 20.0 - x_c
    assert np.allclose(r.decision.x_planned_dm, [x_f, x_c], atol=1e-9)
    assert np.allclose(r.decision.q_as_fed, [x_f / 0.40, x_c / 0.80], atol=1e-9)
    assert r.objective == pytest.approx(0.10 * x_f + 0.40 * x_c, abs=1e-9)
    if cp == 16.0:  # worked example in docs/ENGINE_API.md
        assert np.allclose(r.decision.q_as_fed, [40.0, 5.0]) and r.objective == pytest.approx(3.2)
    assert r.solver_version.startswith("HiGHS") and r.input_hash and r.wall_time_s >= 0
    assert r.constraint_residuals["cp_min"] <= 1e-9


def test_three_ingredient_forage_rule_analytic():
    """F: NDF 0.50 forage; G: NDF 0.10; P: NDF 0.20.  Costs/kg DM 0.10 < 0.20 < 0.50 (G is cheap).

    min cost s.t. DM = 20, NDF + 2 fNDF >= 0.60, CP >= 0.14.
    """
    F = ing("F", 0.40, {"CP": 0.10, "NDF": 0.50}, forage=1.0)
    G = ing("G", 0.80, {"CP": 0.08, "NDF": 0.10})
    P = ing("P", 0.90, {"CP": 0.45, "NDF": 0.20})
    cons = [dm_offer(20.0), conc("rule", {"NDF": 1.0, "G:forage:NDF": 2.0}, "ge", 60.0),
            conc("cp", {"CP": 1.0}, "ge", 14.0)]
    prices = {"F": 0.30 * 0.40, "G": 0.10 * 0.80, "P": 0.50 * 0.90}  # per kg DM: F .30, G .10, P .50
    pr = problem([F, G, P], ["CP", "NDF"], cons, prices)
    r = M0(pr)
    assert r.status is SolveStatus.OPTIMAL
    # independent enumeration in x-space
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    rows = linear_rows(cc, pr.nominal_theta(), pr.dm_estimates(), d_hat=pr.dm_estimates())
    lp = assemble_x_space_lp(rows.A[0], rows.b, rows.is_eq, cc.constraint_ids, pr.dm_estimates(), pr.price_vector())
    best, bx = enumerate_lp_vertices(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, np.full(3, 20.0))
    assert best is not None
    assert r.objective == pytest.approx(best, abs=1e-9)
    assert np.allclose(r.decision.x_planned_dm, bx, atol=1e-7)
    # hand check of the vertex: NDF+2fNDF = 1.5 xF/20 + 0.1 xG/20 + 0.2 xP/20 >= 0.6, CP binding
    ev = evaluate(r.decision, pr.nominal_theta(), pr.dm_estimates(), pr.compiled)
    assert ev.joint_violation.sum() == 0 and ev.structural_ok


@pytest.mark.parametrize("seed", range(12))
def test_random_three_ingredient_lps_match_enumeration(seed):
    rng = np.random.default_rng(1000 + seed)  # synthetic instance generator (test only)
    dm = rng.uniform(0.3, 0.95, 3)
    cp = rng.uniform(0.05, 0.5, 3)
    ndf = rng.uniform(0.05, 0.6, 3)
    ings = [ing(f"i{k}", dm[k], {"CP": cp[k], "NDF": ndf[k]}, forage=float(k == 0)) for k in range(3)]
    cons = [dm_offer(20.0), conc("cp", {"CP": 1.0}, "ge", 100 * rng.uniform(0.08, 0.3)),
            conc("ndf", {"NDF": 1.0}, "le", 100 * rng.uniform(0.2, 0.5)),
            conc("rule", {"NDF": 1.0, "G:forage:NDF": 2.0}, "ge", 100 * rng.uniform(0.1, 0.8)),
            af_max("i1", rng.uniform(2.0, 30.0))]
    prices = {f"i{k}": rng.uniform(0.02, 0.6) for k in range(3)}
    pr = problem(ings, ["CP", "NDF"], cons, prices)
    r = M0(pr)
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    rows = linear_rows(cc, pr.nominal_theta(), pr.dm_estimates(), d_hat=pr.dm_estimates())
    lp = assemble_x_space_lp(rows.A[0], rows.b, rows.is_eq, cc.constraint_ids, pr.dm_estimates(), pr.price_vector())
    best, _ = enumerate_lp_vertices(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, np.full(3, 20.0))
    if best is None:
        assert r.status is SolveStatus.PROVEN_INFEASIBLE and r.decision is None and r.objective is None
    else:
        assert r.status is SolveStatus.OPTIMAL
        assert r.objective == pytest.approx(best, rel=1e-9, abs=1e-9)


def test_toy_problem_m0_matches_enumeration(toy_problem):
    pr = toy_problem
    r = M0(pr)
    assert r.status is SolveStatus.OPTIMAL and r.is_synthetic
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    dh = pr.dm_estimates()
    rows = linear_rows(cc, pr.nominal_theta(), dh, d_hat=dh)
    lp = assemble_x_space_lp(rows.A[0], rows.b, rows.is_eq, cc.constraint_ids, dh, pr.price_vector())
    best, bx = enumerate_lp_vertices(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, np.full(len(dh), 20.0))
    assert r.objective == pytest.approx(best, abs=1e-9)
    # diagnostic_only constraints are not imposed in optimisation
    assert "energy_supply_diag" not in r.diagnostics["imposed_constraint_ids"]
    # the executed decision is q = x / d_hat
    assert np.allclose(r.decision.q_as_fed * dh, r.decision.x_planned_dm)
