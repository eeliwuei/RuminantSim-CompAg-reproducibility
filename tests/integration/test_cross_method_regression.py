"""Cross-method regression: M2 joint-chance SAA with alpha_train = 0 equals M3c scenario-set robust (B-133).

``METHOD_EQUIVALENCES`` pair ``EQ-M3c-M2-ALPHA0`` (docs/ENGINE_API.md section 6.3): on the **same** ``opt``
scenario set, ``floor(alpha_train * N) = 0`` forces every ``z_s = 0``, so the SAA MILP becomes the LP
that imposes every probabilistic row in every scenario -- the scenario-set robust LP.  This file checks
the identity as a regression over a *set* of scenario sets (several seeds, several N, three Big-M
modes, alpha = 0 and 0 < alpha < 1/N) and two synthetic problems:

* the engine toy problem (``data/synthetic_test_only/engine_toy_problem_v1.yaml``: concentration rows,
  Table-5-1-form rows, structural rows), and
* a synthetic absorbed-mineral problem built with :mod:`ration_reliability.nutrition.requirements`
  (``C:AC_Ca:Ca`` / ``C:AC_P:P`` supply rows, maintenance on the scenario DM, bounds from the
  model_audit reference-cow requirement), so that the new rows are also covered by both solvers.

Checked for every case: both ``optimal``; M2 drops no scenario (``n_z_one == 0``); equal cost
(relative 1e-7) and equal executed ration ``q`` (atol 1e-6 kg as-fed); zero joint violations of both
rations on the training scenarios by the public evaluator; identical public-evaluator results on one
common ``validation`` DrawSet (used only for this comparison, never for fitting).

All numbers are synthetic (``is_synthetic=True``); the uncertainty model is a test device.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from ration_reliability.datamodel import Provenance, SolverOptions, SolveStatus, ValueStatus
from ration_reliability.evaluation import evaluate_drawset
from ration_reliability.io import load_problem
from ration_reliability.nutrition import requirements as R
from ration_reliability.nutrition.requirements import Qty
from ration_reliability.optimization import METHOD_EQUIVALENCES, get_method
from ration_reliability.optimization.chance_saa import allowed_violations
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams

from engine_test_helpers import conc, dm_offer, ing, problem

OPTS = SolverOptions(mip_rel_gap=1e-9, time_limit_s=120)
COST_REL = 1e-7
Q_ATOL = 1e-6


def _model(pr, cv_theta, cv_d, model_id):
    th = pr.nominal_theta()
    det = np.array([not g.is_stochastic for g in pr.ingredients])
    sd = np.abs(th) * cv_theta
    sd[det] = 0.0
    dh = pr.dm_estimates()
    dsd = dh * cv_d
    dsd[det] = 0.0
    upper = np.array([[1.0 if n.dimension == "mass_fraction" else np.inf for n in pr.nutrients]] * len(pr.ingredients))
    return IndependentNormalModel(model_id, pr.ingredient_ids, pr.nutrient_ids, th, sd, dh, dsd, truncate=True,
                                  theta_lower=0.0, theta_upper=upper, is_synthetic=True)


@pytest.fixture(scope="module")
def toy(toy_yaml_path):
    pr, rep = load_problem(toy_yaml_path, mode="smoke")
    assert rep.ok and pr.is_synthetic
    return pr, _model(pr, 0.03, 0.02, "syn_xmethod_toy")


# synthetic absorbed-mineral problem (composition fractions of DM, ACs, prices: invented test values)
_SYN_SRC = "SYNTHETIC-G1-XMETHOD"
_ING = {"f1": (0.35, {"Ca": 0.0060, "P": 0.0029}, 1.0, 0.05), "c1": (0.88, {"Ca": 0.0007, "P": 0.0036}, 0.0, 0.25),
        "c2": (0.90, {"Ca": 0.0042, "P": 0.0105}, 0.0, 0.45), "m1": (0.99, {"Ca": 0.3310, "P": 0.0020}, 0.0, 0.30)}
_AC = {"f1": (0.33, 0.71), "c1": (0.57, 0.66), "c2": (0.62, 0.74), "m1": (0.48, 0.79)}
_SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-P4A-TEST")


@pytest.fixture(scope="module")
def mineral(tmp_path_factory):
    path = tmp_path_factory.mktemp("ac") / "ac_synthetic.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=R.AC_TABLE_COLUMNS)
        w.writeheader()
        for iid, (ca, p) in _AC.items():
            for el, v in (("Ca", ca), ("P", p)):
                w.writerow({**{c: "" for c in R.AC_TABLE_COLUMNS}, "ingredient_id": iid, "element": el,
                            "coefficient_name": f"AC_{el}", "value": v, "unit": "1",
                            "value_status": "synthetic_test_only", "mapping_type": "chapter7_text_class",
                            "nasem_class_label": "synthetic", "locator_table": "synthetic fixture",
                            "source_id": _SYN_SRC, "rationale": "synthetic test value"})
    table = R.load_absorption_coefficient_table(path)
    ings = R.attach_absorption_coefficients(
        [ing(i, dm, comp, forage=fw, stochastic=(i != "m1")) for i, (dm, comp, fw, _) in _ING.items()],
        table, allowed_statuses=("synthetic_test_only",))
    # reference-cow requirement (model_audit section 2 inputs), option A, maintenance on the scenario DM
    nel = R.milk_net_energy_eq3_14b(Qty(3.8, "%"), Qty(3.26, "%"), Qty(4.85, "%"))
    me = R.milk_energy_output_eq20_220(nel.as_qty(), Qty(43, "kg/d"))
    dmi = R.dmi_eq2_1(Qty(1, "1"), me.as_qty(), Qty(700, "kg"), Qty(3.0, "1"), Qty(200, "d"))
    choices = R.RequirementChoices("A_An_BWgain", "text_rule", "Eq 7-3", "Eq 20-389")
    kw = dict(dmi=dmi.as_qty(), milk_yield=Qty(43, "kg/d"), milk_true_protein=Qty(3.26, "%"),
              body_weight=Qty(700, "kg"), mature_body_weight=Qty(700, "kg"), body_weight_gain=Qty(0.31, "kg/d"),
              days_pregnant=Qty(110, "d"), choices=choices, dmi_label="Eq 2-1")
    rows = [R.absorbed_mineral_row(R.ca_requirement_factorial(**kw), maintenance_basis="actual_D"),
            R.absorbed_mineral_row(R.p_requirement_factorial(**kw), maintenance_basis="actual_D")]
    specs = [dm_offer(20.0),
             conc("forage_share_min", {"DM:f1": 1.0}, "ge", 40.0, cls="structural_hard", dm_source="decision_estimate"),
             conc("mineral_share_max", {"DM:m1": 1.0}, "le", 2.0, cls="structural_hard", dm_source="decision_estimate")]
    specs += [r.to_constraint_spec(f"abs_{r.element}", provenance=_SYN, numerical_tolerance=1e-6) for r in rows]
    pr = problem(ings, ["Ca", "P"], specs, {i: v[3] for i, v in _ING.items()}, problem_id="syn_abs_mineral_xmethod")
    return pr, _model(pr, 0.05, 0.03, "syn_xmethod_mineral")


def _cases():
    out = []
    for fam in ("toy", "mineral"):
        for seed in (1103, 2207, 3301):
            for n in (16, 64, 128):
                out.append((fam, seed, n, "box_quantile"))
        for mode in ("box", "lp_tight"):
            out.append((fam, 4409, 16, mode))
    return out


def _solve_pair(pr, opt, mode, alpha):
    """Both methods on the same full ``opt`` DrawSet (N = opt.n_draws)."""
    m2 = get_method("M2_joint_chance_saa")(pr, opt_draws=opt, solver_options=OPTS,
                                           params={"alpha_train": alpha, "n_scenarios": None, "big_m_mode": mode})
    m3 = get_method("M3c_scenario_set_robust")(pr, opt_draws=opt, solver_options=OPTS)
    return m2, m3


@pytest.mark.parametrize("family, seed, n, mode", _cases())
def test_m2_alpha0_equals_m3c_ration_and_cost(request, family, seed, n, mode):
    pr, model = request.getfixturevalue(family)
    streams = RandomStreams(seed)
    opt = model.draw(streams, "opt", n)
    common = model.draw(streams, "validation", 4000)          # comparison only, never used for fitting
    for alpha in (0.0, 0.9 / n):                                # floor(alpha N) = 0 in both cases
        assert allowed_violations(alpha, n) == 0
        m2, m3 = _solve_pair(pr, opt, mode, alpha)
        assert m2.diagnostics["n_scenarios_used"] == n == opt.n_draws
        assert m2.status is SolveStatus.OPTIMAL and m3.status is SolveStatus.OPTIMAL, (m2.message, m3.message)
        assert m2.diagnostics["n_z_one"] == 0
        assert abs(m2.objective - m3.objective) <= COST_REL * max(1.0, abs(m3.objective))
        q2, q3 = np.asarray(m2.decision.q_as_fed), np.asarray(m3.decision.q_as_fed)
        assert np.max(np.abs(q2 - q3)) <= Q_ATOL, f"q differs: {q2} vs {q3}"
        # public evaluator: in-sample guarantee and identical scores on a common DrawSet
        for res in (m2, m3):
            ev_in = evaluate_drawset(res.decision, opt, pr.compiled, prices=pr.prices)
            assert ev_in.joint_violation.sum() == 0 and ev_in.structural_ok
        e2 = evaluate_drawset(m2.decision, common, pr.compiled, prices=pr.prices)
        e3 = evaluate_drawset(m3.decision, common, pr.compiled, prices=pr.prices)
        assert e2.cost == pytest.approx(e3.cost, rel=COST_REL)
        assert np.array_equal(e2.violated, e3.violated) and np.array_equal(e2.joint_violation, e3.joint_violation)
        np.testing.assert_allclose(e2.margin, e3.margin, rtol=0, atol=1e-4)


def test_equivalence_pair_is_registered_with_condition():
    pairs = [p for p in METHOD_EQUIVALENCES if p["pair_id"] == "EQ-M3c-M2-ALPHA0"]
    assert len(pairs) == 1
    assert "M3c_scenario_set_robust" in pairs[0]["methods"]
    assert any(m.startswith("M2_joint_chance_saa") for m in pairs[0]["methods"])


def test_positive_budget_breaks_the_identity_as_expected(mineral):
    """Control: with floor(alpha N) >= 1 the SAA may drop scenarios and cost no more than M3c."""
    pr, model = mineral
    opt = model.draw(RandomStreams(1103), "opt", 64)
    m2, m3 = _solve_pair(pr, opt, "box_quantile", 4 / 64)
    assert m2.status is SolveStatus.OPTIMAL and m3.status is SolveStatus.OPTIMAL
    assert m2.objective <= m3.objective * (1 + COST_REL) + 1e-12
    assert m2.diagnostics["n_z_one"] <= allowed_violations(4 / 64, 64)
