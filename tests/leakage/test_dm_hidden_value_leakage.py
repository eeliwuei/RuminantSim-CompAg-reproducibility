"""Contract section 8 required test 8 (+ T1, T2.1, C05/C06): hidden true DM must not be used before
execution.

* q is fixed at decision time from d_hat; changing the hidden true DM cannot change q;
* the evaluator uses x_real = q * d_true without renormalising to the planned DM;
* probabilistic constraints ignore d_hat (only structural 'planned ration' rules use it);
* fitting steps refuse validation/test draws.
"""

import inspect

import numpy as np
import pytest

from ration_reliability.datamodel import RationDecision
from ration_reliability.errors import LeakageError
from ration_reliability.evaluation import evaluate, evaluate_drawset
from ration_reliability.nutrition import concentrations
from ration_reliability.optimization import get_method
from ration_reliability.uncertainty import PointMassModel, RandomStreams

M0 = get_method("M0_nominal")


def test_q_independent_of_hidden_true_dm(toy_problem):
    pr = toy_problem
    r = M0(pr)
    q = np.array(r.decision.q_as_fed)
    # the solver API has no argument through which test-time DM could enter
    params = inspect.signature(M0).parameters
    assert set(params) == {"problem", "d_hat", "opt_draws", "params", "solver_options"}
    # evaluating under a different hidden DM leaves the decision untouched
    d_true = pr.dm_estimates() * np.array([0.80, 1.0, 1.0, 1.0, 1.0])  # wet forage drier than assumed
    ev = evaluate(r.decision, pr.nominal_theta(), d_true, pr.compiled)
    assert np.array_equal(r.decision.q_as_fed, q)
    # realised DM supply is q . d_true (NOT renormalised to the planned 20 kg)
    planned = r.decision.x_planned_dm.sum()
    realised = float(q @ d_true)
    assert planned == pytest.approx(20.0)
    assert ev.dm_supply[0] == pytest.approx(realised) and realised < planned - 1.0
    # concentrations are computed on the realised DM mix (wet forage share falls)
    c_real = concentrations(q, d_true, pr.nominal_theta())
    k = ev.constraint_ids.index("cp_min")
    assert ev.margin[0, k] == pytest.approx(100 * c_real[pr.nutrient_ids.index("CP")] - 16.0)
    # a renormalised formula would have reported the nominal margin (~0); the evaluator must not
    ev_nom = evaluate(r.decision, pr.nominal_theta(), pr.dm_estimates(), pr.compiled)
    assert abs(ev.margin[0, k] - ev_nom.margin[0, k]) > 1e-3


def test_probabilistic_constraints_ignore_d_hat(toy_problem):
    pr = toy_problem
    r = M0(pr)
    fake = RationDecision(r.decision.ingredient_ids, r.decision.q_as_fed, r.decision.d_hat * 0.9 + 0.05,
                          "tamper")
    d_true = pr.dm_estimates()
    a = evaluate(r.decision, pr.nominal_theta(), d_true, pr.compiled)
    b = evaluate(fake, pr.nominal_theta(), d_true, pr.compiled)
    assert np.array_equal(a.margin, b.margin, equal_nan=True)            # scenario DM only
    assert not np.array_equal(a.structural_margin, b.structural_margin)  # planned-ration rules use d_hat


def test_draw_mean_refuses_non_opt_streams(toy_problem):
    pr = toy_problem
    pm = PointMassModel("pm", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), pr.dm_estimates(),
                        is_synthetic=True)
    s = RandomStreams(1103)
    for stream in ("test", "validation", "outer"):
        with pytest.raises(LeakageError):
            M0(pr, opt_draws=pm.draw(s, stream, 4), params={"coefficient_mode": "draw_mean"})
    r = M0(pr, opt_draws=pm.draw(s, "opt", 4), params={"coefficient_mode": "draw_mean"})
    assert r.has_solution and r.streams_used == ("root=1103/opt",)


def test_evaluator_rejects_misaligned_or_conflicting_inputs(toy_problem):
    pr = toy_problem
    r = M0(pr)
    rev = r.decision.reordered(tuple(reversed(pr.ingredient_ids)))
    with pytest.raises(ValueError):
        evaluate(rev, pr.nominal_theta(), pr.dm_estimates(), pr.compiled)
    with pytest.raises(ValueError):
        evaluate(r.decision, pr.nominal_theta(), pr.dm_estimates(), pr.compiled, d_hat=pr.dm_estimates() * 0.9)
    pm = PointMassModel("pm", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), pr.dm_estimates())
    ds = pm.draw(RandomStreams(1), "test", 2).reordered(nutrient_ids=tuple(reversed(pr.nutrient_ids)))
    with pytest.raises(ValueError):
        evaluate_drawset(r.decision, ds, pr.compiled)
