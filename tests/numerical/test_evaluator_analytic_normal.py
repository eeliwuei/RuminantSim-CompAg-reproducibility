"""Evaluator probability estimates vs a closed form (independent check of evaluator + sampling).

For fixed DM and independent normal composition (no truncation), a linear supply
``S = sum_i x_i a_i`` is normal with mean ``sum x_i mu_i`` and sd ``sqrt(sum x_i^2 sd_i^2)``, so
``P(S < K) = Phi((K - mean) / sd)``; with fixed D the concentration ``S / D`` is also normal.
The Monte Carlo violation rate from the public evaluator must agree within MC error.
(The analytic formula only verifies the implementation; it is not evidence that feed
composition is normal -- contract section 8, item 10.)
"""

import numpy as np
import pytest
from scipy import stats

from engine_test_helpers import conc, ing, problem, supply
from ration_reliability.evaluation import evaluate_drawset
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams

MU = np.array([[0.10], [0.18], [0.46]])
SD = np.array([[0.015], [0.025], [0.020]])
DM = np.array([0.35, 0.88, 0.89])
Q = np.array([30.0, 5.0, 3.0])  # kg as-fed (synthetic)


def _setup(cons):
    ings = [ing(f"i{k}", DM[k], {"CP": MU[k, 0]}) for k in range(3)]
    pr = problem(ings, ["CP"], cons, {f"i{k}": 0.1 for k in range(3)})
    m = IndependentNormalModel("syn_normal", pr.ingredient_ids, pr.nutrient_ids, MU, SD, DM, np.zeros(3),
                               truncate=False, is_synthetic=True)
    return pr, m


@pytest.mark.parametrize("n", [200_000])
def test_supply_violation_rate_matches_normal_cdf(n):
    x = Q * DM
    mean, sd = float(x @ MU[:, 0]), float(np.sqrt((x ** 2) @ SD[:, 0] ** 2))
    K = mean + 0.8 * sd  # violation prob ~ Phi(0.8) ~ 0.79 for ">= K"
    pr, m = _setup([supply("cp_sup", {"CP": 1.0}, "ge", K, "kg/d")])
    ev = evaluate_drawset(Q, m.draw(RandomStreams(1103), "test", n), pr.compiled)
    p_true = stats.norm.cdf((K - mean) / sd)
    p_hat = ev.violated[:, 0].mean()
    assert abs(p_hat - p_true) < 4 * np.sqrt(p_true * (1 - p_true) / n)


@pytest.mark.parametrize("n", [200_000])
def test_concentration_violation_rate_matches_normal_cdf(n):
    x = Q * DM
    D = x.sum()
    mean, sd = float(x @ MU[:, 0]) / D, float(np.sqrt((x ** 2) @ SD[:, 0] ** 2)) / D
    K_pct = 100 * (mean - 1.645 * sd)  # lower bound: P(conc < K) ~ 0.05
    pr, m = _setup([conc("cp_min", {"CP": 1.0}, "ge", K_pct)])
    ev = evaluate_drawset(Q, m.draw(RandomStreams(2207), "test", n), pr.compiled)
    p_true = stats.norm.cdf(-1.645)
    p_hat = ev.joint_violation.mean()
    assert abs(p_hat - p_true) < 4 * np.sqrt(p_true * (1 - p_true) / n)
    # natural-unit margins are normal with the analytic mean/sd (in % DM)
    assert ev.margin[:, 0].mean() == pytest.approx(100 * mean - K_pct, abs=4 * 100 * sd / np.sqrt(n))
    assert ev.margin[:, 0].std() == pytest.approx(100 * sd, rel=0.01)
