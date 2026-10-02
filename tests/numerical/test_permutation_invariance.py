"""Contract section 8 required test 9: reordering ingredients / nutrients / constraints / scenarios
does not change physical results (within numerical tolerance)."""

import numpy as np
import pytest

from ration_reliability.datamodel import SolveStatus
from ration_reliability.evaluation import evaluate_drawset
from ration_reliability.optimization import get_method
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams

M0 = get_method("M0_nominal")


def _draws(pr, n=2000):
    th = pr.nominal_theta()
    m = IndependentNormalModel("syn_perm", pr.ingredient_ids, pr.nutrient_ids, th, np.abs(th) * 0.06,
                               pr.dm_estimates(), pr.dm_estimates() * 0.03, theta_upper=1.0, is_synthetic=True)
    return m.draw(RandomStreams(2207), "test", n)


def _by_id(ev):
    return {cid: ev.violated[:, k].mean() for k, cid in enumerate(ev.constraint_ids)}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ingredient_nutrient_constraint_permutation(toy_problem, seed):
    pr = toy_problem
    rng = np.random.default_rng(seed)
    ing_order = list(np.array(pr.ingredient_ids)[rng.permutation(len(pr.ingredient_ids))])
    nut_order = list(np.array(pr.nutrient_ids)[rng.permutation(len(pr.nutrient_ids))])
    con_order = [pr.constraints[k].constraint_id for k in rng.permutation(len(pr.constraints))]
    pp = pr.reordered(ing_order, nut_order, con_order)

    r, rp = M0(pr), M0(pp)
    assert r.status is SolveStatus.OPTIMAL and rp.status is SolveStatus.OPTIMAL
    assert rp.objective == pytest.approx(r.objective, rel=1e-10, abs=1e-12)
    back = rp.decision.reordered(pr.ingredient_ids)
    assert np.allclose(back.q_as_fed, r.decision.q_as_fed, atol=1e-8)  # unique optimum on the toy problem

    ds = _draws(pr)
    dsp = ds.reordered(ingredient_ids=pp.ingredient_ids, nutrient_ids=pp.nutrient_ids)
    ev = evaluate_drawset(r.decision, ds, pr.compiled, prices=pr.prices)
    evp = evaluate_drawset(rp.decision, dsp, pp.compiled, prices=pp.prices)
    assert ev.joint_violation.mean() == evp.joint_violation.mean()
    a, b = _by_id(ev), _by_id(evp)
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)
    assert np.allclose(ev.dm_supply, evp.dm_supply, atol=1e-10)
    assert evp.cost == pytest.approx(ev.cost, rel=1e-12)


def test_scenario_order_permutation(toy_problem):
    pr = toy_problem
    r = M0(pr)
    ds = _draws(pr, 1500)
    perm = np.random.default_rng(9).permutation(ds.n_draws)
    ev = evaluate_drawset(r.decision, ds, pr.compiled)
    evp = evaluate_drawset(r.decision, ds.reordered(scenario_order=perm), pr.compiled)
    assert np.array_equal(evp.margin, ev.margin[perm], equal_nan=True)
    assert evp.joint_violation.sum() == ev.joint_violation.sum()
    a, b = evp.summary()["per_constraint"], ev.summary()["per_constraint"]
    assert a.keys() == b.keys()
    for cid in a:
        for key in ("n_defined", "n_undefined", "n_violated", "violation_rate_among_defined"):
            assert a[cid][key] == b[cid][key]
        assert a[cid]["mean_deficit_given_violation"] == pytest.approx(b[cid]["mean_deficit_given_violation"],
                                                                       rel=1e-12, abs=1e-15)
