"""Helpers to build small synthetic problems directly (bypassing YAML).

Every value produced here is synthetic test data (status ``synthetic_test_only``).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from ration_reliability.datamodel import (
    ConstraintClass,
    ConstraintKind,
    ConstraintSpec,
    DMSource,
    IngredientRecord,
    NutrientSpec,
    PriceScenario,
    Provenance,
    RationProblem,
    Sense,
    ValueStatus,
)

SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-P4A-TEST")


def ing(iid: str, dm: float, comp: Mapping[str, float], *, forage: float = 0.0,
        stochastic: bool = True) -> IngredientRecord:
    """Ingredient with composition given as *fractions* of DM (canonical)."""
    gw = {"forage": forage} if forage else {}
    return IngredientRecord(ingredient_id=iid, name_en=f"synthetic {iid}", dm_estimate=dm,
                            composition=dict(comp), category="forage" if forage else "concentrate",
                            group_weights=gw, is_stochastic=stochastic, is_synthetic=True,
                            provenance={"dm_estimate": SYN})


def conc(cid: str, terms: Mapping[str, float], sense: str, bound: float, unit: str = "%", *,
         cls: str = "probabilistic_nutrition", tol: float = 1e-6, dm_source: str = "scenario") -> ConstraintSpec:
    return ConstraintSpec(constraint_id=cid, name=cid, kind=ConstraintKind.CONCENTRATION, terms=dict(terms),
                          sense=Sense(sense), bound=bound, unit=unit, constraint_class=ConstraintClass(cls),
                          numerical_tolerance=tol, provenance=SYN, basis="DM", dm_source=DMSource(dm_source))


def supply(cid: str, terms: Mapping[str, float], sense: str, bound: float, unit: str = "kg/d", *,
           cls: str = "probabilistic_nutrition", tol: float = 1e-6, dm_source: str = "scenario",
           basis: str = "none") -> ConstraintSpec:
    return ConstraintSpec(constraint_id=cid, name=cid, kind=ConstraintKind.SUPPLY, terms=dict(terms),
                          sense=Sense(sense), bound=bound, unit=unit, constraint_class=ConstraintClass(cls),
                          numerical_tolerance=tol, provenance=SYN, basis=basis, dm_source=DMSource(dm_source))


def dm_offer(value: float, sense: str = "eq", tol: float = 1e-6) -> ConstraintSpec:
    return supply("dm_offer", {"DM": 1.0}, sense, value, "kg/d", cls="structural_hard", tol=tol,
                  dm_source="decision_estimate", basis="DM")


def af_max(iid: str, value: float, tol: float = 1e-9) -> ConstraintSpec:
    return ConstraintSpec(constraint_id=f"af_max_{iid}", name=f"af_max_{iid}", kind=ConstraintKind.AS_FED,
                          terms={f"AF:{iid}": 1.0}, sense=Sense.LE, bound=value, unit="kg/d",
                          constraint_class=ConstraintClass.STRUCTURAL_HARD, numerical_tolerance=tol,
                          provenance=SYN, basis="as_fed", dm_source=DMSource.DECISION_ESTIMATE)


def problem(ingredients: Sequence[IngredientRecord], nutrients: Iterable, constraints: Sequence[ConstraintSpec],
            prices_per_kg_as_fed: Mapping[str, float], problem_id: str = "syn_problem") -> RationProblem:
    nuts = tuple(n if isinstance(n, NutrientSpec) else NutrientSpec(*n) if isinstance(n, tuple) else NutrientSpec(n)
                 for n in nutrients)
    ps = PriceScenario(price_id="syn_prices", currency="XXX", prices_per_kg_as_fed=dict(prices_per_kg_as_fed),
                       is_scenario=True, is_synthetic=True)
    return RationProblem(problem_id=problem_id, ingredients=tuple(ingredients), nutrients=nuts,
                         constraints=tuple(constraints), prices=ps, is_synthetic=True,
                         dataset_status="synthetic_test_only")


def two_ingredient_problem(cp_min_pct: float = 16.0, dmi: float = 20.0) -> RationProblem:
    """Analytic case: forage F (CP 10 %DM, d 0.40, 0.04/kg AF) + concentrate C (CP 40 %DM, d 0.80, 0.32/kg AF).

    Per kg DM: F costs 0.10, C costs 0.40.  With planned DM = dmi and CP >= c:
    x_C = dmi (c - 0.10) / 0.30, x_F = dmi - x_C  (for 0.10 <= c <= 0.40).
    """
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    cons = [dm_offer(dmi), conc("cp_min", {"CP": 1.0}, "ge", cp_min_pct)]
    return problem([F, C], ["CP"], cons, {"F": 0.04, "C": 0.32}, problem_id="syn_two_ingredient")


def enumerate_lp_vertices(c: np.ndarray, A_ub: np.ndarray, b_ub: np.ndarray, A_eq: np.ndarray,
                          b_eq: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol: float = 1e-9):
    """Independent brute-force LP solver for tiny problems (vertex enumeration with numpy.linalg).

    Returns ``(best_value, best_x)`` or ``(None, None)`` if no feasible vertex exists.
    Assumes a bounded feasible region (finite ``ub``).
    """
    import itertools

    n = len(c)
    rows, rhs = [], []
    for a, b in zip(A_ub, b_ub):
        rows.append(np.asarray(a, float)); rhs.append(float(b))
    for i in range(n):
        e = np.zeros(n); e[i] = -1.0; rows.append(e); rhs.append(-float(lb[i]))
        e = np.zeros(n); e[i] = 1.0; rows.append(e); rhs.append(float(ub[i]))
    rows, rhs = np.array(rows), np.array(rhs)
    A_eq = np.asarray(A_eq, float).reshape(-1, n)
    b_eq = np.asarray(b_eq, float).reshape(-1)
    k = n - A_eq.shape[0]
    best, bx = None, None
    for combo in itertools.combinations(range(len(rows)), k):
        M = np.vstack([A_eq, rows[list(combo)]]) if k else A_eq
        r = np.concatenate([b_eq, rhs[list(combo)]]) if k else b_eq
        if M.shape[0] != n or abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, r)
        if np.all(rows @ x <= rhs + tol) and np.allclose(A_eq @ x, b_eq, atol=tol):
            val = float(np.asarray(c) @ x)
            if best is None or val < best - 1e-12:
                best, bx = val, x
    return best, bx
