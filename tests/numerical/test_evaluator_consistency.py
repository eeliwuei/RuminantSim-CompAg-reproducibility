"""Contract section 8 required test 7: the same known q gives the same public-evaluator output no
matter which method produced it or how it is passed; evaluation is chunk-size invariant and
does not mutate its inputs."""

import numpy as np
import pytest

from ration_reliability import optimization
from ration_reliability.datamodel import RationDecision, SolveResult, SolveStatus
from ration_reliability.evaluation import evaluate, evaluate_drawset
from ration_reliability.optimization import get_method
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams


def _model(pr, scale=0.05):
    th = pr.nominal_theta()
    return IndependentNormalModel("syn_nm", pr.ingredient_ids, pr.nutrient_ids, th, np.abs(th) * scale,
                                  pr.dm_estimates(), pr.dm_estimates() * scale / 2, theta_upper=1.0,
                                  is_synthetic=True)


def _fields(ev):
    return (ev.margin, ev.violation_amount, ev.violated, ev.undefined, ev.joint_violation, ev.joint_unknown,
            ev.dm_supply, ev.nutrient_supply, ev.structural_margin, ev.structural_violated)


def _same(a, b):
    for x, y in zip(_fields(a), _fields(b)):
        assert np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)
    assert a.q_hash == b.q_hash and a.cost == b.cost


def test_same_q_same_output_across_methods(toy_problem, monkeypatch):
    pr = toy_problem
    r0 = get_method("M0_nominal")(pr)
    q = np.array(r0.decision.q_as_fed)

    # a stand-in "other method" registered temporarily: it returns the same physical q
    def fake_solve(problem, *, d_hat=None, opt_draws=None, params=None, solver_options=None):
        dec = RationDecision(problem.ingredient_ids, q.copy(), problem.dm_estimates(), "FAKE_other")
        return SolveResult("FAKE_other", SolveStatus.OPTIMAL, dec, float(problem.price_vector() @ q),
                           "XXX/head/d", "none", "none", {}, 0.0, "h", is_synthetic=True)

    import types
    mod = types.ModuleType("fake_method_module_p4a")
    mod.solve = fake_solve
    monkeypatch.setitem(__import__("sys").modules, "fake_method_module_p4a", mod)
    monkeypatch.setitem(optimization.REGISTRY, "FAKE_other", "fake_method_module_p4a:solve")
    r1 = get_method("FAKE_other")(pr)

    ds = _model(pr).draw(RandomStreams(3301), "test", 3000)
    e_arr = evaluate(q, ds.theta, ds.d, pr.compiled, d_hat=pr.dm_estimates(), prices=pr.prices)
    e_m0 = evaluate_drawset(r0.decision, ds, pr.compiled, prices=pr.prices)
    e_fk = evaluate_drawset(r1.decision, ds, pr.compiled, prices=pr.prices)
    _same(e_arr, e_m0)
    _same(e_m0, e_fk)
    assert 0 < e_m0.joint_violation.sum() < ds.n_draws  # the synthetic spread produces some violations
    s = e_m0.summary()
    assert s["joint"]["n_violated"] + s["joint"]["n_unknown"] + s["joint"]["n_satisfied"] == ds.n_draws
    assert s["joint"]["rate_lower"] == s["joint"]["rate_upper"] == pytest.approx(e_m0.joint_violation.mean())


def test_chunk_size_invariance_and_no_mutation(toy_problem):
    pr = toy_problem
    r = get_method("M0_nominal")(pr)
    ds = _model(pr).draw(RandomStreams(1103), "test", 1001)
    th, d = ds.theta.copy(), ds.d.copy()
    q = np.array(r.decision.q_as_fed)
    q_before = q.copy()
    a = evaluate(q, th, d, pr.compiled, d_hat=pr.dm_estimates(), chunk_size=1001)
    b = evaluate(q, th, d, pr.compiled, d_hat=pr.dm_estimates(), chunk_size=7)
    for x, y in zip(_fields(a), _fields(b)):
        assert np.allclose(x, y, equal_nan=True, rtol=0, atol=1e-12) if x.dtype.kind == "f" else np.array_equal(x, y)
    assert np.array_equal(q, q_before) and np.array_equal(th, ds.theta) and np.array_equal(d, ds.d)
    with pytest.raises(ValueError):
        a.margin[0, 0] = 1.0  # results are read-only


def test_structural_check_independent_of_draws(toy_problem):
    pr = toy_problem
    r = get_method("M0_nominal")(pr)
    ds1 = _model(pr, 0.01).draw(RandomStreams(1), "test", 50)
    ds2 = _model(pr, 0.20).draw(RandomStreams(2), "test", 50)
    a = evaluate_drawset(r.decision, ds1, pr.compiled)
    b = evaluate_drawset(r.decision, ds2, pr.compiled)
    assert np.array_equal(a.structural_margin, b.structural_margin) and a.structural_ok and b.structural_ok
    assert set(a.structural_ids) == {"dm_offer", "mineral_share_max", "forage_wet_af_max"}
    # negative q fails the built-in non-negativity rule
    q = np.array(r.decision.q_as_fed)
    q[2] = -0.5
    c = evaluate(q, ds1.theta, ds1.d, pr.compiled, d_hat=pr.dm_estimates())
    assert not c.nonnegativity_ok and not c.structural_ok
