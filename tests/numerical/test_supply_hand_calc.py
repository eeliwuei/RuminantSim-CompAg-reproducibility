"""Contract section 8 required test 2: mass and nutrient supply against hand calculations.

Hand case (synthetic): q = [10, 5] kg as-fed; d = [0.35, 0.90]; CP = [0.08, 0.45] kg/kg DM;
NDF = [0.50, 0.20]; ingredient 1 is forage; prices [0.05, 0.40] per kg as-fed.

    D      = 10*0.35 + 5*0.90                 = 3.5 + 4.5 = 8.0 kg DM
    N_CP   = 3.5*0.08 + 4.5*0.45              = 0.28 + 2.025 = 2.305 kg
    CP %DM = 2.305 / 8.0                      = 28.8125 %
    N_NDF  = 3.5*0.50 + 4.5*0.20              = 1.75 + 0.90 = 2.65 kg  -> 33.125 %
    fNDF   = 1.75 / 8.0                       = 21.875 %
    NDF + 2 fNDF                              = 33.125 + 43.75 = 76.875 %
    C      = 10*0.05 + 5*0.40                 = 2.5
"""

import numpy as np
import pytest

from engine_test_helpers import conc, dm_offer, ing, problem, supply
from ration_reliability.evaluation import evaluate
from ration_reliability.nutrition import concentrations, dm_supply, nutrient_supply, ration_cost, realized_dm_formula

Q = np.array([10.0, 5.0])
D_TRUE = np.array([0.35, 0.90])
THETA = np.array([[0.08, 0.50], [0.45, 0.20]])  # columns CP, NDF
P = np.array([0.05, 0.40])


def _problem():
    A = ing("A", 0.35, {"CP": 0.08, "NDF": 0.50}, forage=1.0)
    B = ing("B", 0.90, {"CP": 0.45, "NDF": 0.20})
    cons = [
        dm_offer(8.0),
        conc("cp_min", {"CP": 1.0}, "ge", 30.0),                               # violated by 1.1875 %
        conc("cp_max", {"CP": 1.0}, "le", 350.0, "g/kg"),                      # slack 61.875 g/kg
        conc("rule", {"NDF": 1.0, "G:forage:NDF": 2.0}, "ge", 70.0),           # slack 6.875 %
        supply("cp_supply", {"CP": 1.0}, "ge", 2300.0, "g/d"),                 # slack 5 g/d
        supply("dm_supply", {"DM": 1.0}, "le", 7.9, "kg/d", cls="diagnostic_only"),  # over by 0.1
    ]
    return problem([A, B], ["CP", "NDF"], cons, {"A": 0.05, "B": 0.40})


def test_supply_functions_match_hand_calculation():
    assert np.allclose(realized_dm_formula(Q, D_TRUE), [3.5, 4.5])
    assert dm_supply(Q, D_TRUE) == pytest.approx(8.0)
    assert np.allclose(nutrient_supply(Q, D_TRUE, THETA), [2.305, 2.65])
    assert np.allclose(concentrations(Q, D_TRUE, THETA), [0.288125, 0.33125])
    assert ration_cost(Q, P) == pytest.approx(2.5)
    # batched form gives the same numbers
    assert np.allclose(nutrient_supply(Q, D_TRUE[None], THETA[None])[0], [2.305, 2.65])


def test_evaluator_natural_units_match_hand_calculation():
    pr = _problem()
    ev = evaluate(Q, THETA, D_TRUE, pr.compiled, d_hat=np.array([0.35, 0.90]), prices=P)
    m = dict(zip(ev.constraint_ids, ev.margin[0]))
    assert m["cp_min"] == pytest.approx(28.8125 - 30.0)          # % DM
    assert m["cp_max"] == pytest.approx(350.0 - 288.125)         # g/kg DM
    assert m["rule"] == pytest.approx(76.875 - 70.0)             # % DM
    assert m["cp_supply"] == pytest.approx(2305.0 - 2300.0)      # g/d
    assert m["dm_supply"] == pytest.approx(7.9 - 8.0)            # kg/d
    amt = dict(zip(ev.constraint_ids, ev.violation_amount[0]))
    assert amt["cp_min"] == pytest.approx(1.1875) and amt["rule"] == 0.0
    v = dict(zip(ev.constraint_ids, ev.violated[0]))
    assert v == {"cp_min": True, "cp_max": False, "rule": False, "cp_supply": False, "dm_supply": True}
    # joint indicator uses probabilistic constraints only (dm_supply is diagnostic)
    assert bool(ev.joint_violation[0]) is True
    assert ev.dm_supply[0] == pytest.approx(8.0) and np.allclose(ev.nutrient_supply[0], [2.305, 2.65])
    assert ev.cost == pytest.approx(2.5)
    assert ev.structural_ok and ev.structural_margin[0] == pytest.approx(0.0)


def test_diagnostic_violation_alone_does_not_trigger_joint_event():
    pr = _problem()
    theta = THETA.copy()
    theta[0, 0] = 0.20  # CP of A up: (3.5*0.20 + 2.025)/8 = 34.0625 %  -> 30 <= CP <= 35 holds
    ev = evaluate(Q, theta, D_TRUE, pr.compiled, d_hat=D_TRUE)
    v = dict(zip(ev.constraint_ids, ev.violated[0]))
    assert v == {"cp_min": False, "cp_max": False, "rule": False, "cp_supply": False, "dm_supply": True}
    assert dict(zip(ev.constraint_ids, ev.margin[0]))["cp_min"] == pytest.approx(34.0625 - 30.0)
    assert not ev.joint_violation[0] and not ev.joint_unknown[0]


def test_missing_data_is_flag_not_violation():
    pr = _problem()
    theta = THETA.copy()
    theta[1, 0] = np.nan  # CP of B missing; B is used
    ev = evaluate(Q, theta, D_TRUE, pr.compiled, d_hat=D_TRUE)
    k = ev.constraint_ids.index("cp_min")
    assert ev.undefined[0, k] and ev.missing_data[0, k] and not ev.violated[0, k]
    assert np.isnan(ev.margin[0, k]) and ev.violation_amount[0, k] == 0.0
    assert bool(ev.data_quality_flag[0])
    assert not ev.joint_violation[0] and ev.joint_unknown[0]
    # NDF rule does not use CP -> still defined
    assert not ev.undefined[0, ev.constraint_ids.index("rule")]
    # the same NaN on an *unused* ingredient is irrelevant
    ev2 = evaluate(np.array([10.0, 0.0]), theta, D_TRUE, pr.compiled, d_hat=D_TRUE)
    assert not ev2.undefined.any() and not ev2.data_quality_flag.any()
