"""Contract section 8 required test 6: tightening a constraint (others fixed) never lowers the
optimal cost; once infeasible, tighter versions stay infeasible."""

import dataclasses

import numpy as np

from engine_test_helpers import two_ingredient_problem
from ration_reliability.datamodel import SolveStatus
from ration_reliability.optimization import get_method

M0 = get_method("M0_nominal")


def _costs(problems):
    out = []
    for pr in problems:
        r = M0(pr)
        out.append(r.objective if r.status is SolveStatus.OPTIMAL else None)
        if r.status is not SolveStatus.OPTIMAL:
            assert r.status is SolveStatus.PROVEN_INFEASIBLE
    return out


def _check_monotone(costs):
    seen_infeasible = False
    prev = -np.inf
    for c in costs:
        if c is None:
            seen_infeasible = True
            continue
        assert not seen_infeasible, "feasible again after infeasible under tightening"
        assert c >= prev - 1e-9, (c, prev)
        prev = c


def test_cp_lower_bound_tightening_two_ingredients():
    grid = np.linspace(5.0, 45.0, 17)
    _check_monotone(_costs([two_ingredient_problem(cp_min_pct=float(v)) for v in grid]))


def _replace_bound(problem, cid, new_bound):
    cons = tuple(dataclasses.replace(c, bound=new_bound) if c.constraint_id == cid else c
                 for c in problem.constraints)
    return dataclasses.replace(problem, constraints=cons)


def test_toy_problem_tightening_each_direction(toy_problem):
    # lower bounds raised
    for cid, grid in [("cp_min", np.linspace(12, 19, 8)), ("ca_min", np.linspace(0.3, 1.5, 7)),
                      ("ndf_plus_2fndf", np.linspace(50, 110, 7))]:
        _check_monotone(_costs([_replace_bound(toy_problem, cid, float(v)) for v in grid]))
    # upper bounds lowered
    for cid, grid in [("starch_max", np.linspace(35, 5, 7)), ("cp_max", np.linspace(22, 16, 7)),
                      ("forage_wet_af_max", np.linspace(60, 10, 6))]:
        _check_monotone(_costs([_replace_bound(toy_problem, cid, float(v)) for v in grid]))
