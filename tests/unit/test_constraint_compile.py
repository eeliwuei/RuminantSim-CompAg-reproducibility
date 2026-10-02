"""Constraint grammar, unit/dimension checks and class rules (contract 6.2, T2)."""

import numpy as np
import pytest

from engine_test_helpers import SYN, af_max, conc, dm_offer, ing, problem, supply
from ration_reliability.datamodel import ConstraintClass, ConstraintKind, ConstraintSpec, DMSource, Sense
from ration_reliability.errors import InvalidProblemError
from ration_reliability.nutrition import compile_constraints, inventory_constraint

F = ing("F", 0.4, {"CP": 0.10, "NDF": 0.50, "NE": 1.4}, forage=1.0)
G = ing("G", 0.9, {"CP": 0.09, "NDF": 0.10, "NE": 2.0}, forage=0.25)
NUTS = [("CP", "mass_fraction"), ("NDF", "mass_fraction"), ("NE", "energy_density")]


def _compile(cons):
    return problem([F, G], NUTS, cons, {"F": 0.05, "G": 0.3}).compiled


def test_table51_form_coefficients():
    cc = _compile([conc("r", {"NDF": 1.0, "G:forage:NDF": 2.0}, "ge", 60.0)])
    j = cc.nutrient_ids.index("NDF")
    assert np.allclose(cc.W[0, :, j], [1 + 2 * 1.0, 1 + 2 * 0.25])
    assert cc.bound[0] == pytest.approx(0.60) and cc.unit_factor[0] == pytest.approx(0.01)


def test_energy_supply_in_mj_converted():
    cc = _compile([supply("e", {"NE": 1.0}, "ge", 125.0, "MJ/d")])
    assert cc.bound[0] == pytest.approx(125.0 / 4.184)


@pytest.mark.parametrize("spec, needle", [
    (conc("x", {"XX": 1.0}, "ge", 1.0), "unknown term"),
    (conc("x", {"G:legume:NDF": 1.0}, "ge", 1.0), "not declared on any ingredient"),
    (conc("x", {"CP": 1.0, "NE": 1.0}, "ge", 1.0), "mix dimensions"),
    (conc("x", {"CP": 1.0}, "ge", 1.0, "Mcal/kg"), "does not fit"),
    (conc("x", {"CP": 1.0}, "eq", 1.0), "equality is not allowed"),
    (conc("x", {"CP": 1.0}, "ge", 1.0, cls="structural_hard", dm_source="decision_estimate"),
     "may not depend on composition"),
    (conc("x", {"DM:F": 1.0}, "le", 50.0, cls="structural_hard"), "dm_source=decision_estimate"),
    (conc("x", {"DM:F": 1.0}, "le", 50.0, dm_source="decision_estimate"), "must use dm_source=scenario"),
    (supply("x", {"CP": 1.0}, "ge", 1.0, "%"), "does not fit"),
    (conc("x", {"AF:F": 1.0}, "le", 1.0), "AF terms require kind as_fed"),
])
def test_bad_constraints_rejected(spec, needle):
    with pytest.raises(InvalidProblemError) as ei:
        _compile([spec])
    assert needle in str(ei.value)


def test_pending_bound_cannot_compile():
    spec = ConstraintSpec("p", "p", ConstraintKind.CONCENTRATION, {"CP": 1.0}, Sense.GE, None, "%",
                          ConstraintClass.PROBABILISTIC_NUTRITION, 1e-6, SYN)
    with pytest.raises(InvalidProblemError) as ei:
        _compile([spec])
    assert "bound is missing" in str(ei.value)


def test_as_fed_probabilistic_rejected_and_structural_ok():
    bad = ConstraintSpec("a", "a", ConstraintKind.AS_FED, {"AF:F": 1.0}, Sense.LE, 10.0, "kg/d",
                         ConstraintClass.PROBABILISTIC_NUTRITION, 0.0, SYN, basis="as_fed")
    with pytest.raises(InvalidProblemError):
        _compile([bad])
    cc = _compile([af_max("F", 10.0), dm_offer(20.0)])
    assert cc.v[0].tolist() == [1.0, 0.0] and cc.dm_sources[1] is DMSource.DECISION_ESTIMATE


def test_inventory_constraint_is_bound_over_head_days():
    spec = inventory_constraint("inv", "F", batch_kg_as_fed=12000.0, heads=100, days=5, provenance=SYN)
    cc = compile_constraints([spec], [F, G], problem([F, G], NUTS, [], {"F": 1, "G": 1}).nutrients)
    assert cc.bound[0] == pytest.approx(24.0)


def test_coefficient_terms_absorbed_mineral_form():
    """C:<coef>:<nutrient> and DM terms express sum_i x_i a_i AC_i >= R0 + r1 * D (synthetic numbers)."""
    from engine_test_helpers import supply
    from ration_reliability.datamodel import IngredientRecord
    from ration_reliability.evaluation import evaluate

    A = IngredientRecord("A", "syn A", 0.40, {"Ca": 0.005}, coefficients={"AC_Ca": 0.60}, is_synthetic=True)
    B = IngredientRecord("B", "syn B", 0.90, {"Ca": 0.300}, coefficients={"AC_Ca": 0.70}, is_synthetic=True)
    # absorbed Ca (g/d) - 1.5 g per kg DM * D  >= 20 g/d   ->  DM coefficient -0.0015 kg/kg DM
    spec = supply("abs_ca", {"C:AC_Ca:Ca": 1.0, "DM": -0.0015}, "ge", 20.0, "g/d")
    pr = problem([A, B], [("Ca", "mass_fraction")], [spec], {"A": 0.05, "B": 0.3})
    cc = pr.compiled
    assert np.allclose(cc.W[0, :, 0], [0.60, 0.70]) and np.allclose(cc.w0[0], [-0.0015, -0.0015])
    q = np.array([40.0, 0.2])                      # x = [16, 0.18] kg DM, D = 16.18
    ev = evaluate(q, pr.nominal_theta(), pr.dm_estimates(), cc)
    absorbed_g = (16 * 0.005 * 0.60 + 0.18 * 0.300 * 0.70) * 1000   # 48 + 37.8 = 85.8 g/d
    expected = absorbed_g - 1.5 * 16.18 - 20.0                      # 85.8 - 24.27 - 20 = 41.53 g/d
    assert ev.margin[0, 0] == pytest.approx(expected)


def test_coefficient_term_requires_every_ingredient():
    from ration_reliability.datamodel import IngredientRecord

    A = IngredientRecord("A", "syn A", 0.40, {"Ca": 0.005}, coefficients={"AC_Ca": 0.60}, is_synthetic=True)
    B = IngredientRecord("B", "syn B", 0.90, {"Ca": 0.300}, is_synthetic=True)  # AC missing
    spec = supply("abs_ca", {"C:AC_Ca:Ca": 1.0}, "ge", 20.0, "g/d")
    with pytest.raises(InvalidProblemError) as ei:
        problem([A, B], [("Ca", "mass_fraction")], [spec], {"A": 0.05, "B": 0.3}).compiled
    assert "missing for ingredients ['B']" in str(ei.value)
