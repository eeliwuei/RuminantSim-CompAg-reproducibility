"""Contract section 8 required test 4: deliberately infeasible problems return ``proven_infeasible``
(no empty ration, no zero cost, no automatic relaxation); status mapping and invalid input."""

import numpy as np
import pytest

from engine_test_helpers import af_max, conc, dm_offer, ing, problem, two_ingredient_problem
from ration_reliability.datamodel import RationDecision, SolveResult, SolveStatus, SolverOptions
from ration_reliability.errors import InvalidProblemError
from ration_reliability.optimization import get_method
from ration_reliability.optimization import highs as H

M0 = get_method("M0_nominal")


def test_nutrient_infeasible_is_proven_infeasible():
    pr = two_ingredient_problem(cp_min_pct=45.0)  # max attainable CP is 40 %
    r = M0(pr)
    assert r.status is SolveStatus.PROVEN_INFEASIBLE
    assert r.decision is None and r.objective is None and not r.has_solution
    # no relaxation: the model actually solved contains the original constraint set
    assert r.diagnostics["imposed_constraint_ids"] == ["dm_offer", "cp_min"]
    d = r.to_dict()
    assert d["decision"] is None and d["objective"] is None and d["status"] == "proven_infeasible"


def test_structural_infeasible_is_proven_infeasible():
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    # DM 20 kg needed but at most 10 kg as-fed of each -> max DM 4 + 8 = 12
    cons = [dm_offer(20.0), af_max("F", 10.0), af_max("C", 10.0)]
    r = M0(problem([F, C], ["CP"], cons, {"F": 0.04, "C": 0.32}))
    assert r.status is SolveStatus.PROVEN_INFEASIBLE and r.decision is None


def test_solve_result_refuses_empty_ration_for_failures():
    dec = RationDecision(("a",), np.array([0.0]), np.array([0.5]), "x")
    with pytest.raises(InvalidProblemError):
        SolveResult("x", SolveStatus.PROVEN_INFEASIBLE, dec, 0.0, "XXX/head/d", "s", "v", {}, 0.0, "h")
    with pytest.raises(InvalidProblemError):
        SolveResult("x", SolveStatus.OPTIMAL, None, None, "XXX/head/d", "s", "v", {}, 0.0, "h")


def test_invalid_inputs_are_reported_not_repaired():
    pr = two_ingredient_problem()
    assert M0(pr, params={"alpha": 0.05}).status is SolveStatus.INVALID_INPUT
    assert M0(pr, params={"coefficient_mode": "median"}).status is SolveStatus.INVALID_INPUT
    assert M0(pr, d_hat=np.array([0.4, 0.0])).status is SolveStatus.INVALID_INPUT
    assert M0(pr, params={"coefficient_mode": "draw_mean"}).status is SolveStatus.INVALID_INPUT  # no draws
    # missing nominal CP for an ingredient that a constraint needs -> invalid input (never set to 0)
    F = ing("F", 0.40, {"CP": np.nan}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    r = M0(problem([F, C], ["CP"], [dm_offer(20.0), conc("cp", {"CP": 1.0}, "ge", 16.0)],
                   {"F": 0.04, "C": 0.32}))
    assert r.status is SolveStatus.INVALID_INPUT and "cp:F" in r.message and r.decision is None


def test_status_mapping_table():
    ok = np.array([1.0])
    m = H._map
    assert m(7, 0, ok, True) is SolveStatus.OPTIMAL
    assert m(7, 0, ok, False) is SolveStatus.NUMERICAL_ERROR
    assert m(13, 1, ok, True) is SolveStatus.FEASIBLE_TIME_LIMIT
    assert m(13, 1, None, False) is SolveStatus.NO_FEASIBLE_SOLUTION_FOUND
    assert m(14, 1, ok, False) is SolveStatus.NO_FEASIBLE_SOLUTION_FOUND
    assert m(8, 2, None, False) is SolveStatus.PROVEN_INFEASIBLE
    assert m(10, 3, None, False) is SolveStatus.INVALID_INPUT
    assert m(2, 2, None, False) is SolveStatus.INVALID_INPUT  # model error is not infeasibility
    assert m(9, 4, None, False) is SolveStatus.NUMERICAL_ERROR
    assert m(4, 4, None, False) is SolveStatus.NUMERICAL_ERROR


def test_linprog_wrapper_statuses_and_unbounded():
    opts = SolverOptions()
    out = H.run_linprog([1.0, 1.0], [[-1.0, 0.0]], [-5.0], None, None, [0, 0], [2, np.inf], opts)
    assert out.status is SolveStatus.PROVEN_INFEASIBLE and out.x is None
    out = H.run_linprog([-1.0, 0.0], None, None, None, None, [0, 0], [np.inf, 1], opts)
    assert out.status is SolveStatus.INVALID_INPUT  # unbounded ration model = mis-specified
    out = H.run_linprog([1.0, 2.0], [[-1.0, -1.0]], [-3.0], None, None, [0, 0], [np.inf, np.inf], opts)
    assert out.status is SolveStatus.OPTIMAL and np.allclose(out.x, [3.0, 0.0]) and out.fun == pytest.approx(3.0)


def test_milp_wrapper_gap_and_infeasible():
    opts = SolverOptions(mip_rel_gap=0.0)
    out = H.run_milp([1.0, 1.0], [[-1.0, 0.0]], [-1.5], None, None, [0, 0], [5, 1], [1, 0], opts)
    assert out.status is SolveStatus.OPTIMAL and np.allclose(out.x, [2.0, 0.0]) and out.mip_gap is not None
    out = H.run_milp([1.0], [[-1.0]], [-5.0], None, None, [0], [2], [1], opts)
    assert out.status is SolveStatus.PROVEN_INFEASIBLE


def test_primal_residual_scaled():
    ra, rr = H.primal_residual(np.array([[1.0, 1.0]]), np.array([1.0]), None, None, [0, 0], [np.inf, np.inf],
                               np.array([1.0, 0.5]))
    assert ra == pytest.approx(0.5) and rr == pytest.approx(0.5 / (1 + 1 + 1.5))


def test_zero_ration_optimum_is_flagged_not_hidden():
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    # no DM-offer rule: the zero ration satisfies CP >= 16 % trivially in linearised form
    r = M0(problem([F, C], ["CP"], [conc("cp", {"CP": 1.0}, "ge", 16.0)], {"F": 0.04, "C": 0.32}))
    assert r.status is SolveStatus.OPTIMAL and r.objective == 0.0
    assert "all-zero ration" in r.diagnostics["warning"]
    assert str(r.status) == "optimal" and f"{r.status}" == "optimal"
