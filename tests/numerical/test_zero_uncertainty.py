"""Contract section 8 required test 5: zero uncertainty degenerates to the nominal model; restoring
parameters (sd -> 0) recovers it; M0 'draw_mean' equals 'nominal_point' on point-mass draws
(the two are one method, not two, under this condition)."""

import numpy as np
import pytest

from ration_reliability.datamodel import SolveStatus
from ration_reliability.evaluation import evaluate, evaluate_drawset
from ration_reliability.nutrition import concentrations
from ration_reliability.optimization import get_method
from ration_reliability.uncertainty import IndependentNormalModel, PointMassModel, RandomStreams

M0 = get_method("M0_nominal")


def _pm(pr):
    return PointMassModel("pm_nominal", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), pr.dm_estimates(),
                          is_synthetic=True)


def test_point_mass_evaluation_equals_nominal(toy_problem):
    pr = toy_problem
    r = M0(pr)
    assert r.status is SolveStatus.OPTIMAL
    ds = _pm(pr).draw(RandomStreams(1103), "test", 64)
    ev = evaluate_drawset(r.decision, ds, pr.compiled, prices=pr.prices)
    assert ev.joint_violation.sum() == 0 and ev.joint_unknown.sum() == 0 and ev.structural_ok
    # every draw identical to the single nominal evaluation
    ev1 = evaluate(r.decision, pr.nominal_theta(), pr.dm_estimates(), pr.compiled)
    assert np.allclose(ev.margin, ev1.margin[0][None, :], equal_nan=True)
    # realised concentrations equal nominal concentrations; planned DM equals realised DM
    c_nom = concentrations(r.decision.q_as_fed, pr.dm_estimates(), pr.nominal_theta())
    j = pr.nutrient_ids.index("CP")
    k = ev.constraint_ids.index("cp_min")
    assert ev.margin[0, k] == pytest.approx(100 * c_nom[j] - 16.0, abs=1e-9)
    assert ev.dm_supply[0] == pytest.approx(r.decision.x_planned_dm.sum())
    # binding constraints have ~0 margin at the optimum, within tolerance
    assert abs(ev.margin[0, k]) <= 1e-6
    assert ev.cost == pytest.approx(r.objective)


def test_draw_mean_mode_equals_nominal_on_point_mass(toy_problem):
    pr = toy_problem
    r0 = M0(pr)
    ds = _pm(pr).draw(RandomStreams(1103), "opt", 16)
    r1 = M0(pr, opt_draws=ds, params={"coefficient_mode": "draw_mean"})
    assert r1.status is SolveStatus.OPTIMAL
    assert r1.objective == pytest.approx(r0.objective, abs=1e-12)
    assert np.allclose(r1.decision.q_as_fed, r0.decision.q_as_fed, atol=1e-10)
    assert r1.streams_used == ("root=1103/opt",)


def test_sd_to_zero_recovers_point_mass(toy_problem):
    pr = toy_problem
    I, J = len(pr.ingredient_ids), len(pr.nutrient_ids)
    theta0, d0 = pr.nominal_theta(), pr.dm_estimates()
    s = RandomStreams(2207)
    r = M0(pr)
    base = evaluate(r.decision, theta0, d0, pr.compiled)
    for scale in (1e-3, 1e-6, 0.0):
        m = IndependentNormalModel("nm", pr.ingredient_ids, pr.nutrient_ids, theta0, np.full((I, J), scale),
                                   d0, np.full(I, scale), truncate=False, is_synthetic=True)
        ev = evaluate_drawset(r.decision, m.draw(s, "test", 200), pr.compiled)
        dev = np.nanmax(np.abs(ev.margin - base.margin[0][None, :]))
        if scale == 0.0:
            # identical draws; only BLAS summation order may differ between batch sizes
            assert dev <= 1e-12 and ev.joint_violation.sum() == 0
        else:
            assert dev < 1e4 * scale  # continuity: deviation shrinks with the scale


def test_mean_rows_use_joint_products_not_product_of_means():
    """E[d*a] != E[d]*E[a] under correlation: draw-mean rows must use the joint product."""
    from engine_test_helpers import conc, dm_offer, ing, problem
    from ration_reliability.uncertainty import ScenarioSetModel

    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    pr = problem([F, C], ["CP"], [dm_offer(20.0), conc("cp", {"CP": 1.0}, "ge", 16.0)], {"F": 0.04, "C": 0.32})
    # two equally likely states of F: (d, CP) = (0.30, 0.06) or (0.50, 0.14): means 0.40 / 0.10,
    # E[d*CP] = (0.018 + 0.070)/2 = 0.044 != 0.40*0.10 = 0.040
    th = np.array([[[0.06], [0.40]], [[0.14], [0.40]]])
    d = np.array([[0.30, 0.80], [0.50, 0.80]])
    ds = ScenarioSetModel("two_state", pr.ingredient_ids, pr.nutrient_ids, th, d, is_synthetic=True)
    from ration_reliability.uncertainty import DrawSet

    draws = DrawSet(th, d, "opt", "manual/opt", ds.model_id, ds.fingerprint(), pr.ingredient_ids, pr.nutrient_ids, True)
    from ration_reliability.nutrition import linear_rows

    cc = pr.compiled
    rows = linear_rows(cc, draws.theta, draws.d, d_hat=pr.dm_estimates()).mean()
    k = cc.constraint_ids.index("cp")
    # q-space coefficient of F in the linearised CP>=0.16 row:  -(E[d*CP] - 0.16*E[d]) = -(0.044 - 0.064)
    assert rows.A[0, k, 0] == pytest.approx(-(0.044 - 0.16 * 0.40))
    r = M0(pr, opt_draws=draws, params={"coefficient_mode": "draw_mean"})
    assert r.status is SolveStatus.OPTIMAL
