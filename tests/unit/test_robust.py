"""P4c tests for the M3 robust methods and the mean/nominal equivalence audit.

All numbers are synthetic test values (``is_synthetic=True``); they test the code, not any
nutritional claim.  Reference answers are computed by independent paths: vertex enumeration with
the engine's canonical row builder ``linear_rows``, hand calculation, brute-force subset
enumeration, and an independent joint-chance SAA MILP written in this file.

Sections
--------
A  box (M3a): worst case vs vertex enumeration, DM x composition / denominator hand case,
   evaluator check inside the box, k monotonicity, infeasibility, leakage, input checks
B  budget (M3b): Gamma = 0 -> M0, Gamma = full -> box, protection function vs enumeration and
   LP dual, monotonicity, row-wise guarantee, input checks
C  scenario-set (M3c): = SAA with floor(alpha N) = 0, cost ordering vs M0, in-sample guarantee,
   leakage, missing data
D  mean/nominal equivalence (reports/mean_nominal_equivalence.md): expected-coefficient LP =
   multi-scenario mean LP; plug-in = mean LP under exact independence; gap = sample covariance;
   correlated counterexamples E[d a] != E[d] E[a]; truncation shift; ratio of means
E  general: robust cost >= nominal cost, common evaluator, permutation invariance
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import linprog

from engine_test_helpers import af_max, conc, dm_offer, ing, problem, supply, two_ingredient_problem
from ration_reliability.datamodel import SolveStatus, SolverOptions
from ration_reliability.errors import LeakageError
from ration_reliability.evaluation import evaluate, evaluate_drawset
from ration_reliability.nutrition import linear_rows
from ration_reliability.optimization import get_method
from ration_reliability.optimization.highs import run_linprog, run_milp
from ration_reliability.optimization.lp_builder import big_m_from_box, optimization_indices
from ration_reliability.optimization.robust import (
    METHOD_ID_BOX,
    METHOD_ID_BUDGET,
    METHOD_ID_SCENARIO_SET,
    BoxUncertaintySet,
    box_coefficient_bounds,
    box_worst_case_residual,
    budget_protection,
    solve_box_robust,
    solve_budget_robust,
    solve_scenario_set_robust,
)
from ration_reliability.uncertainty import DrawSet, IndependentNormalModel, RandomStreams

M0 = get_method("M0_nominal")
OK = (SolveStatus.OPTIMAL,)


# ------------------------------------------------------------------------------------------
# synthetic problems and helpers
# ------------------------------------------------------------------------------------------

def three_problem():
    """3 ingredients x 3 nutrients; conc ge/le, Table-5-1-*form* rules, one supply row, one AF cap."""
    F = ing("F", 0.35, {"CP": 0.08, "NDF": 0.45, "starch": 0.30}, forage=1.0)
    G = ing("G", 0.88, {"CP": 0.09, "NDF": 0.10, "starch": 0.70})
    P = ing("P", 0.89, {"CP": 0.48, "NDF": 0.12, "starch": 0.02})
    cons = [dm_offer(20.0),
            conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            conc("cp_max", {"CP": 1.0}, "le", 30.0),
            conc("ndf_rule", {"NDF": 1.0, "G:forage:NDF": 2.0}, "ge", 60.0),
            conc("starch_max", {"starch": 1.0}, "le", 28.0),
            conc("starch_fndf", {"starch": 1.0, "G:forage:NDF": -2.0}, "le", -5.0),
            supply("cp_supply", {"CP": 1.0}, "ge", 3.0),
            af_max("G", 12.0)]
    return problem([F, G, P], ["CP", "NDF", "starch"], cons, {"F": 0.035, "G": 0.176, "P": 0.356},
                   problem_id="syn_three_robust")


def box_k(pr, k, rel_sd=0.10, d_rel_sd=0.04):
    th, dh = pr.nominal_theta(), pr.dm_estimates()
    return BoxUncertaintySet.from_mean_sd("syn_box", pr.ingredient_ids, pr.nutrient_ids, th, th * rel_sd,
                                          dh, dh * d_rel_sd, k, is_synthetic=True)


def manual_draws(pr, theta, d, stream="opt", label="manual"):
    theta, d = np.asarray(theta, float), np.asarray(d, float)
    return DrawSet(theta, d, stream, f"{label}/{stream}", label, f"fp-{label}", pr.ingredient_ids, pr.nutrient_ids,
                   True)


def box_vertices(box):
    """All 2^(I*(J+1)) vertices of the box as (theta [V,I,J], d [V,I]) -- independent of robust.py."""
    I, J = box.theta_lo.shape
    lo = np.concatenate([box.theta_lo, box.d_lo[:, None]], axis=1).ravel()
    hi = np.concatenate([box.theta_hi, box.d_hi[:, None]], axis=1).ravel()
    n = lo.size
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
    V = np.where(bits, hi[None, :], lo[None, :]).reshape(-1, I, J + 1)
    return V[:, :, :J].copy(), V[:, :, J].copy()


def prob_cc(pr):
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    return cc.subset([k for k, c in enumerate(cc.classes) if c.value == "probabilistic_nutrition"])


def saa_milp(pr, draws, alpha, dmi=20.0):
    """Independent joint-chance SAA MILP (T4 M2): common z_s for all rows, sum z <= floor(alpha N)."""
    dh, prices = pr.dm_estimates(), pr.price_vector()
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    ks = [k for k, c in enumerate(cc.classes) if c.value == "structural_hard"]
    kp = [k for k, c in enumerate(cc.classes) if c.value == "probabilistic_nutrition"]
    rs = linear_rows(cc.subset(ks), pr.nominal_theta(), dh, d_hat=dh)
    rp = linear_rows(cc.subset(kp), draws.theta, draws.d, d_hat=dh)
    S, K, I = rp.A.shape
    Ax = rp.A / dh[None, None, :]
    M = np.maximum(big_m_from_box(Ax, rp.b, np.zeros(I), np.full(I, dmi)), 0.0)   # x_i <= DMI (DM offer = DMI)
    n = I + S
    ub, ubr, eq, eqr = [], [], [], []
    for a, b, e in zip(rs.A[0] / dh[None, :], rs.b, rs.is_eq):
        row = np.zeros(n); row[:I] = a
        (eq if e else ub).append(row); (eqr if e else ubr).append(b)
    for s in range(S):
        for k in range(K):
            row = np.zeros(n); row[:I] = Ax[s, k]; row[I + s] = -M[s, k]
            ub.append(row); ubr.append(rp.b[k])
    card = np.zeros(n); card[I:] = 1.0
    ub.append(card); ubr.append(float(math.floor(alpha * S + 1e-12)))
    c = np.concatenate([prices / dh, np.zeros(S)])
    integ = np.concatenate([np.zeros(I), np.ones(S)])
    ubv = np.concatenate([np.full(I, np.inf), np.ones(S)])
    out = run_milp(c, np.array(ub), np.array(ubr), np.array(eq).reshape(-1, n), np.array(eqr), np.zeros(n), ubv,
                   integ, SolverOptions(mip_rel_gap=1e-9))
    assert out.status is SolveStatus.OPTIMAL, out.message
    x = out.x[:I]
    return float(prices @ (x / dh)), x / dh, np.round(out.x[I:]).astype(int)


# ==========================================================================================
# A  box (M3a)
# ==========================================================================================

@pytest.mark.parametrize("k", [0.0, 0.5, 1.0, 1.7])
def test_box_coefficients_equal_vertex_extremes(k):
    pr = three_problem()
    box = box_k(pr, k)
    cc = prob_cc(pr)
    A_max, A_min, b, miss = box_coefficient_bounds(cc, box)
    assert not miss.any()
    tv, dv = box_vertices(box)
    rows = linear_rows(cc, tv, dv, d_hat=pr.dm_estimates())
    assert np.allclose(A_max, rows.A.max(axis=0), rtol=0, atol=1e-14)
    assert np.allclose(A_min, rows.A.min(axis=0), rtol=0, atol=1e-14)
    assert np.array_equal(b, rows.b)


def test_box_worst_case_equals_vertex_enumeration():
    pr = three_problem()
    box = box_k(pr, 1.0)
    cc = prob_cc(pr)
    tv, dv = box_vertices(box)
    rows_v = linear_rows(cc, tv, dv, d_hat=pr.dm_estimates())
    r = solve_box_robust(pr, params={"uncertainty_set": box})
    assert r.status in OK
    rng = np.random.default_rng(20260924)                         # synthetic test generator
    qs = [r.decision.q_as_fed] + [rng.uniform(0, 30, 3) for _ in range(10)] + [np.array([0.0, 0.0, 5.0])]
    for q in qs:
        wc = box_worst_case_residual(cc, box, q)
        vmax = rows_v.residual(q).max(axis=0)
        assert np.allclose(wc, vmax, rtol=1e-12, atol=1e-12)
        # interior points never exceed the vertex maximum (multilinearity)
        u = rng.uniform(size=(500,) + box.theta_lo.shape)
        ud = rng.uniform(size=(500,) + box.d_lo.shape)
        ti = box.theta_lo + u * (box.theta_hi - box.theta_lo)
        di = box.d_lo + ud * (box.d_hi - box.d_lo)
        gi = linear_rows(cc, ti, di, d_hat=pr.dm_estimates()).residual(q)
        assert np.all(gi <= wc[None, :] + 1e-12)
    with pytest.raises(ValueError):
        box_worst_case_residual(cc, box, np.array([-1.0, 1.0, 1.0]))
    # the reported residuals are exactly the worst case
    wc = box_worst_case_residual(cc, box, r.decision.q_as_fed)
    for k, cid in enumerate(cc.constraint_ids):
        assert r.constraint_residuals[cid] == pytest.approx(wc[k], abs=1e-12)


def test_box_robust_equals_scenario_set_over_all_vertices():
    """Robust over the box = robust over its 4096 vertices (the row max is attained at a vertex)."""
    pr = three_problem()
    box = box_k(pr, 1.0)
    rb = solve_box_robust(pr, params={"uncertainty_set": box})
    tv, dv = box_vertices(box)
    rs = solve_scenario_set_robust(pr, opt_draws=manual_draws(pr, tv, dv, label="vertices"))
    assert rb.status in OK and rs.status in OK
    assert rb.objective == pytest.approx(rs.objective, rel=1e-9, abs=1e-12)
    assert rs.diagnostics["n_scenarios"] == 4096


def test_box_solution_never_violates_inside_box_common_evaluator():
    """Ratio constraints (N/D with the state's own D) are checked by the shared evaluator."""
    pr = three_problem()
    box = box_k(pr, 1.0)
    r = solve_box_robust(pr, params={"uncertainty_set": box})
    tv, dv = box_vertices(box)
    ev = evaluate(r.decision, tv, dv, pr.compiled, prices=pr.prices)
    assert ev.joint_violation.sum() == 0 and ev.joint_unknown.sum() == 0 and ev.structural_ok
    assert ev.cost == pytest.approx(r.objective, abs=1e-12)
    # tightness: at least one probabilistic constraint is exactly binding at some vertex
    m = ev.margin[:, [ev.constraint_classes[k] == "probabilistic_nutrition" for k in range(len(ev.constraint_ids))]]
    assert np.nanmin(m) > -1e-9 and np.nanmin(m) < 1e-7
    rng = np.random.default_rng(11)
    u = rng.uniform(size=(5000,) + box.theta_lo.shape)
    ud = rng.uniform(size=(5000,) + box.d_lo.shape)
    ev2 = evaluate(r.decision, box.theta_lo + u * (box.theta_hi - box.theta_lo),
                   box.d_lo + ud * (box.d_hi - box.d_lo), pr.compiled)
    assert ev2.joint_violation.sum() == 0
    # nominal M0 ration does violate somewhere in the same box (the box is not vacuous)
    r0 = M0(pr)
    ev0 = evaluate(r0.decision, tv, dv, pr.compiled)
    assert ev0.joint_violation.sum() > 0


def test_box_dm_times_composition_and_denominator_hand_case():
    """F: d in [0.35,0.45], CP in [0.08,0.12];  C: d in [0.75,0.85], CP in [0.36,0.44];  CP >= 16 %DM.

    Linearised row K*D - N = sum_i q_i d_i (0.16 - CP_i) <= 0.  Joint worst case per ingredient:
      F: max d (0.16 - CP) = 0.45 * 0.08 = 0.036   (low CP, *high* DM: dilutes)
      C: max d (0.16 - CP) = 0.75 * (0.16 - 0.36) = -0.15  (its worst CP 0.36, *low* DM)
    Treating D and N with separate worst DM would give 0.16*0.45 - 0.35*0.08 = 0.044 for F -- a
    state that does not exist (two different DM values for the same ingredient).
    """
    pr = two_ingredient_problem(16.0)
    box = BoxUncertaintySet.from_bounds("hand", pr.ingredient_ids, pr.nutrient_ids, [[0.08], [0.36]],
                                        [[0.12], [0.44]], [0.35, 0.75], [0.45, 0.85], is_synthetic=True)
    cc = prob_cc(pr)
    A_max, _, b, _ = box_coefficient_bounds(cc, box)
    assert A_max[0] == pytest.approx([0.036, -0.15], abs=1e-15) and b[0] == 0.0
    tv, dv = box_vertices(box)
    f_max = linear_rows(cc, tv, dv).A[:, 0, 0].max()
    assert f_max == pytest.approx(0.036, abs=1e-15) and f_max < 0.044 - 1e-3
    r = solve_box_robust(pr, params={"uncertainty_set": box})
    qC = 20.0 / (0.40 * 0.15 / 0.036 + 0.80)
    qF = 0.15 / 0.036 * qC
    assert np.allclose(r.decision.q_as_fed, [qF, qC], atol=1e-9)
    assert r.objective == pytest.approx(0.04 * qF + 0.32 * qC, abs=1e-12)        # 3.945946
    assert r.objective == pytest.approx(3.945945945945946, abs=1e-12)
    # realised concentration at the worst vertex is exactly 16 % (binding), computed as N/D
    ev = evaluate(r.decision, np.array([[0.08], [0.36]]), np.array([0.45, 0.75]), pr.compiled)
    k = ev.constraint_ids.index("cp_min")
    assert ev.margin[0, k] == pytest.approx(0.0, abs=1e-9)
    N = qF * 0.45 * 0.08 + qC * 0.75 * 0.36
    D = qF * 0.45 + qC * 0.75
    assert 100 * N / D == pytest.approx(16.0, abs=1e-9)


def test_box_k0_equals_m0_and_cost_monotone_in_k():
    pr = three_problem()
    r0 = M0(pr)
    costs = []
    for k in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        r = solve_box_robust(pr, params={"uncertainty_set": box_k(pr, k)})
        assert r.status in OK, (k, r.status)
        costs.append(r.objective)
        if k == 0.0:
            assert r.objective == pytest.approx(r0.objective, abs=1e-12)
            assert np.allclose(r.decision.q_as_fed, r0.decision.q_as_fed, atol=1e-9)
    assert all(b >= a - 1e-12 for a, b in zip(costs, costs[1:]))
    assert costs[-1] > costs[0] + 1e-6
    assert all(c >= r0.objective - 1e-12 for c in costs)            # robust cost >= nominal cost


def test_box_infeasible_is_reported_not_relaxed():
    pr = three_problem()
    r = solve_box_robust(pr, params={"uncertainty_set": box_k(pr, 3.0)})
    assert r.status is SolveStatus.PROVEN_INFEASIBLE
    assert r.decision is None and r.objective is None
    assert r.diagnostics["imposed_constraint_ids"] == list(pr.compiled.subset(
        optimization_indices(pr.compiled)).constraint_ids)
    assert r.method_id == METHOD_ID_BOX


def test_box_from_opt_draws_matches_numpy_and_refuses_other_streams():
    pr = three_problem()
    th, dh = pr.nominal_theta(), pr.dm_estimates()
    m = IndependentNormalModel("nm", pr.ingredient_ids, pr.nutrient_ids, th, th * 0.1, dh, dh * 0.04,
                               truncate=False, is_synthetic=True)
    s = RandomStreams(1103)
    opt = m.draw(s, "opt", 300)
    box = BoxUncertaintySet.from_opt_draws(opt, k=1.5)
    mu, sd = opt.theta.mean(axis=0), opt.theta.std(axis=0, ddof=1)
    assert np.allclose(box.theta_lo, np.maximum(mu - 1.5 * sd, 0.0)) and np.allclose(box.theta_hi, mu + 1.5 * sd)
    assert box.construction_params["draws_stream_id"] == "root=1103/opt"
    bq = BoxUncertaintySet.from_opt_draws(opt, quantiles=(0.05, 0.95))
    assert np.allclose(bq.d_hi, np.quantile(opt.d, 0.95, axis=0))
    r = solve_box_robust(pr, opt_draws=opt, params={"k": 1.5})
    r2 = solve_box_robust(pr, params={"uncertainty_set": box})
    assert r.status in OK and r.objective == pytest.approx(r2.objective, abs=1e-12)
    assert r.streams_used == ("root=1103/opt",) and r2.streams_used == ("root=1103/opt",)
    for bad in ("validation", "test", "outer"):
        ds = m.draw(s, bad, 50)
        with pytest.raises(LeakageError):
            BoxUncertaintySet.from_opt_draws(ds, k=1.0)
        with pytest.raises(LeakageError):
            solve_box_robust(pr, opt_draws=ds, params={"k": 1.0})
        with pytest.raises(LeakageError):
            solve_budget_robust(pr, opt_draws=ds, params={"k": 1.0, "gamma": 1})


def test_box_input_checks():
    pr = three_problem()
    box = box_k(pr, 1.0)
    bad = [
        {"uncertainty_set": box, "alpha": 0.05},              # unknown param
        {},                                                   # no set, no k
        {"k": 1.0},                                           # k without opt_draws
        {"uncertainty_set": box, "k": 1.0},                   # both
        {"uncertainty_set": "not a box"},
    ]
    for p in bad:
        r = solve_box_robust(pr, params=p)
        assert r.status is SolveStatus.INVALID_INPUT and r.decision is None, p
    # wrong label order -> invalid input (never silently reordered)
    rev = box.reordered(tuple(reversed(pr.ingredient_ids)))
    assert solve_box_robust(pr, params={"uncertainty_set": rev}).status is SolveStatus.INVALID_INPUT
    assert solve_box_robust(pr, d_hat=np.array([0.3, 0.0, 0.9]),
                            params={"uncertainty_set": box}).status is SolveStatus.INVALID_INPUT
    # missing bound for an ingredient a row needs -> invalid input listing the cell (never 0)
    lo, hi = np.array(box.theta_lo), np.array(box.theta_hi)
    lo[2, 0] = hi[2, 0] = np.nan
    miss = BoxUncertaintySet.from_bounds("m", pr.ingredient_ids, pr.nutrient_ids, lo, hi, box.d_lo, box.d_hi)
    r = solve_box_robust(pr, params={"uncertainty_set": miss})
    assert r.status is SolveStatus.INVALID_INPUT and "cp_min:P" in r.message


def test_box_set_validation_and_model_support_clip():
    ids, nids = ("a", "b"), ("CP",)
    with pytest.raises(ValueError):
        BoxUncertaintySet.from_bounds("x", ids, nids, [[0.2], [0.1]], [[0.1], [0.2]], [0.5, 0.5], [0.6, 0.6])
    with pytest.raises(ValueError):
        BoxUncertaintySet.from_bounds("x", ids, nids, [[0.1], [0.1]], [[0.2], [0.2]], [0.0, 0.5], [0.6, 0.6])
    with pytest.raises(ValueError):
        BoxUncertaintySet.from_bounds("x", ids, nids, [[np.nan], [0.1]], [[0.2], [0.2]], [0.5, 0.5], [0.6, 0.6])
    with pytest.raises(ValueError):
        BoxUncertaintySet.from_mean_sd("x", ids, nids, [[0.1], [0.1]], [[0.1], [0.1]], [0.5, 0.5], [0.1, 0.1], -1.0)
    b = BoxUncertaintySet.from_mean_sd("x", ids, nids, [[0.05], [0.10]], [[0.05], [0.01]], [0.5, 0.5],
                                       [0.3, 0.01], 2.0)
    assert b.theta_lo[0, 0] == 0.0 and b.d_lo[0] == pytest.approx(1e-6) and b.d_hi[0] == 1.0
    assert b.construction_params["n_theta_bounds_clipped"] == 1 and b.construction_params["n_d_bounds_clipped"] == 2
    m = IndependentNormalModel("tn", ids, nids, np.array([[0.05], [0.10]]), np.array([[0.05], [0.01]]),
                               np.array([0.5, 0.5]), np.array([0.01, 0.01]), truncate=True, theta_upper=0.11,
                               is_synthetic=True)
    bm = BoxUncertaintySet.from_independent_normal(m, 2.0)
    assert bm.theta_hi[1, 0] == pytest.approx(0.11) and bm.theta_lo[0, 0] == 0.0
    assert bm.contains(np.array([[0.05], [0.10]]), np.array([0.5, 0.5]))
    assert not bm.contains(np.array([[0.05], [0.13]]), np.array([0.5, 0.5]))


# ==========================================================================================
# B  budget (M3b)
# ==========================================================================================

def test_budget_gamma0_equals_m0():
    for pr in (three_problem(), two_ingredient_problem(16.0)):
        r0 = M0(pr)
        rb = solve_budget_robust(pr, params={"uncertainty_set": box_k(pr, 1.0), "gamma": 0})
        assert rb.status in OK and rb.method_id == METHOD_ID_BUDGET
        assert rb.objective == pytest.approx(r0.objective, rel=1e-10, abs=1e-12)
        assert np.allclose(rb.decision.q_as_fed, r0.decision.q_as_fed, atol=1e-7)


@pytest.mark.parametrize("full", ["full", 3, 99.0, float("inf")])
def test_budget_gamma_full_equals_box(full):
    pr = three_problem()
    box = box_k(pr, 1.0)
    rbox = solve_box_robust(pr, params={"uncertainty_set": box})
    rb = solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": full})
    assert rb.status in OK
    assert rb.objective == pytest.approx(rbox.objective, rel=1e-9, abs=1e-12)
    assert all(v == 3.0 for v in rb.diagnostics["gamma_effective"].values())
    # the budget solution is box-feasible (worst case over the whole box <= 0)
    assert np.all(box_worst_case_residual(prob_cc(pr), box, rb.decision.q_as_fed) <= 1e-9)


def test_budget_cost_monotone_between_m0_and_box():
    pr = three_problem()
    box = box_k(pr, 1.0)
    c0 = M0(pr).objective
    cbox = solve_box_robust(pr, params={"uncertainty_set": box}).objective
    costs = []
    for g in np.arange(0.0, 3.01, 0.25):
        r = solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": float(g)})
        assert r.status in OK
        costs.append(r.objective)
    assert all(b >= a - 1e-10 for a, b in zip(costs, costs[1:]))
    assert costs[0] == pytest.approx(c0, abs=1e-10) and costs[-1] == pytest.approx(cbox, abs=1e-9)
    assert all(c0 - 1e-10 <= c <= cbox + 1e-9 for c in costs)
    assert len(set(np.round(costs, 9))) > 3                          # Gamma actually moves the cost


def _beta_enumeration(v, gamma):
    n = v.size
    f = int(math.floor(gamma + 1e-12))
    frac = gamma - f
    best = 0.0
    for S in itertools.combinations(range(n), min(f, n)):
        base = v[list(S)].sum()
        rest = [t for t in range(n) if t not in S]
        extra = max((frac * v[t] for t in rest), default=0.0)
        best = max(best, base + extra)
    return best


def test_budget_protection_matches_enumeration_primal_and_dual_lp():
    rng = np.random.default_rng(5)
    for trial in range(20):
        n = int(rng.integers(1, 7))
        ahat = rng.uniform(0, 2, n) * (rng.uniform(size=n) > 0.2)
        q = rng.uniform(0, 5, n)
        v = ahat * q
        for gamma in [0.0, 0.3, 1.0, 1.5, 2.0, 2.7, float(n), n + 2.0]:
            g = min(gamma, n)
            beta = budget_protection(ahat, q, gamma)
            assert beta == pytest.approx(_beta_enumeration(v, g), abs=1e-12)
            prim = linprog(-v, A_ub=np.ones((1, n)), b_ub=[g], bounds=[(0, 1)] * n, method="highs")
            assert beta == pytest.approx(-prim.fun, abs=1e-9)
            # dual used by the method: min g z + sum p  s.t. z + p_i >= v_i, z, p >= 0
            A = np.hstack([-np.ones((n, 1)), -np.eye(n)])
            dual = linprog(np.concatenate([[g], np.ones(n)]), A_ub=A, b_ub=-v, bounds=[(0, None)] * (n + 1),
                           method="highs")
            assert beta == pytest.approx(dual.fun, abs=1e-9)


def test_budget_solution_row_wise_guarantee_gamma1():
    """Gamma = 1: every row holds when any ONE ingredient sits at its box worst case (others nominal)."""
    pr = three_problem()
    box = box_k(pr, 1.0)
    r = solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": 1})
    assert r.status in OK
    q = r.decision.q_as_fed
    cc = prob_cc(pr)
    A_max, _, b, _ = box_coefficient_bounds(cc, box)
    A_bar = linear_rows(cc, pr.nominal_theta(), pr.dm_estimates(), d_hat=pr.dm_estimates()).A[0]
    worst_single = np.max([(A_bar + np.where(np.arange(3)[None, :] == i, A_max - A_bar, 0.0)) @ q - b
                           for i in range(3)], axis=0)
    assert np.all(worst_single <= 1e-9)
    assert np.min(np.abs(worst_single)) < 1e-7                                   # some row is binding
    for k, cid in enumerate(cc.constraint_ids):
        beta = budget_protection(A_max[k] - A_bar[k], q, 1.0)
        assert r.constraint_residuals[cid] == pytest.approx(A_bar[k] @ q + beta - b[k], abs=1e-12)
        assert r.constraint_residuals[cid] == pytest.approx(worst_single[k], abs=1e-9)
    # not a joint guarantee: the full box is typically violated by a Gamma=1 solution
    assert np.max(box_worst_case_residual(cc, box, q)) > 1e-6


def test_budget_input_checks_and_per_row_gamma():
    pr = three_problem()
    box = box_k(pr, 1.0)
    for g in [None, -1.0, "half", float("nan"), {"cp_min": 1}]:
        r = solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": g})
        assert r.status is SolveStatus.INVALID_INPUT and r.decision is None, g
    ids = prob_cc(pr).constraint_ids
    mix = {cid: ("full" if cid == "cp_min" else 0) for cid in ids}
    r = solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": mix})
    assert r.status in OK
    assert r.diagnostics["gamma_effective"]["cp_min"] == 3.0 and r.diagnostics["gamma_effective"]["cp_max"] == 0.0
    c0 = M0(pr).objective
    cfull = solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": "full"}).objective
    assert c0 - 1e-10 <= r.objective <= cfull + 1e-10
    # nominal outside the box -> invalid input (Ahat would be negative)
    th, dh = pr.nominal_theta(), pr.dm_estimates()
    shifted = BoxUncertaintySet.from_bounds("shift", pr.ingredient_ids, pr.nutrient_ids, th * 1.05, th * 1.15,
                                            dh, dh, is_synthetic=True)
    r = solve_budget_robust(pr, params={"uncertainty_set": shifted, "gamma": 1})
    assert r.status is SolveStatus.INVALID_INPUT and "outside the box" in r.message
    # ... but the box centre can be declared as nominal explicitly (Gamma=0 -> LP at the centre)
    rc = solve_budget_robust(pr, params={"uncertainty_set": shifted, "gamma": 0, "nominal_mode": "box_center"})
    assert rc.status in OK
    assert solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": 1,
                                           "nominal_mode": "median"}).status is SolveStatus.INVALID_INPUT


# ==========================================================================================
# C  scenario-set (M3c)
# ==========================================================================================

def _opt_draws(pr, n, seed=2207, rel=0.06, drel=0.03, stream="opt"):
    th, dh = pr.nominal_theta(), pr.dm_estimates()
    m = IndependentNormalModel("nm_syn", pr.ingredient_ids, pr.nutrient_ids, th, th * rel, dh, dh * drel,
                               is_synthetic=True)
    return m.draw(RandomStreams(seed), stream, n)


@pytest.mark.parametrize("seed", [1103, 2207, 3301])
def test_scenario_set_equals_saa_alpha_to_zero(seed):
    pr = three_problem()
    ds = _opt_draws(pr, 60, seed)
    rs = solve_scenario_set_robust(pr, opt_draws=ds)
    assert rs.status in OK and rs.method_id == METHOD_ID_SCENARIO_SET
    for alpha in (0.0, 0.5 / 60, 0.99 / 60):                     # floor(alpha N) = 0
        c, q, z = saa_milp(pr, ds, alpha)
        assert z.sum() == 0
        assert c == pytest.approx(rs.objective, rel=1e-9, abs=1e-12)
    prev = rs.objective
    for alpha in (1 / 60, 3 / 60, 6 / 60):                       # alpha > 0: SAA may drop scenarios
        c, q, z = saa_milp(pr, ds, alpha)
        assert c <= prev + 1e-9 and z.sum() <= math.floor(alpha * 60 + 1e-12)
        ev = evaluate_drawset(q, ds, pr.compiled, d_hat=pr.dm_estimates())
        assert ev.joint_violation.sum() <= math.floor(alpha * 60 + 1e-12)
        prev = c


def test_scenario_set_cost_ordering_vs_m0():
    pr = three_problem()
    ds = _opt_draws(pr, 80)
    rs = solve_scenario_set_robust(pr, opt_draws=ds)
    rmean = M0(pr, opt_draws=ds, params={"coefficient_mode": "draw_mean"})
    assert rs.objective >= rmean.objective - 1e-12                  # always: every state => the average
    # with the nominal state included, also >= M0 nominal_point
    th = np.concatenate([ds.theta, pr.nominal_theta()[None]], axis=0)
    d = np.concatenate([ds.d, pr.dm_estimates()[None]], axis=0)
    rs2 = solve_scenario_set_robust(pr, opt_draws=manual_draws(pr, th, d, label="with_nominal"))
    assert rs2.objective >= M0(pr).objective - 1e-12 and rs2.objective >= rs.objective - 1e-12


def test_scenario_set_can_cost_less_than_nominal_when_nominal_not_in_set():
    """Documented condition, not a bug: 'robust >= nominal' needs the nominal state inside U."""
    pr = two_ingredient_problem(16.0)
    th = np.array([[[0.12], [0.40]], [[0.13], [0.40]]])            # F's CP always above its table value 0.10
    d = np.array([[0.40, 0.80], [0.40, 0.80]])
    rs = solve_scenario_set_robust(pr, opt_draws=manual_draws(pr, th, d, label="above_nominal"))
    assert rs.status in OK and rs.objective < M0(pr).objective - 1e-6


def test_scenario_set_in_sample_guarantee_and_order_invariance():
    pr = three_problem()
    ds = _opt_draws(pr, 120)
    r = solve_scenario_set_robust(pr, opt_draws=ds)
    ev = evaluate_drawset(r.decision, ds, pr.compiled, prices=pr.prices)
    assert ev.joint_violation.sum() == 0 and ev.structural_ok and ev.cost == pytest.approx(r.objective)
    assert sum(r.diagnostics["n_binding_scenarios_per_constraint"].values()) >= 1
    perm = np.random.default_rng(3).permutation(ds.n_draws)
    r2 = solve_scenario_set_robust(pr, opt_draws=ds.reordered(scenario_order=perm))
    assert r2.objective == pytest.approx(r.objective, rel=1e-10, abs=1e-12)
    # the guarantee is in-sample only: with a small set (N = 5) fresh draws of the SAME synthetic
    # distribution violate (seeded example; the fresh stream is only read by this check)
    small = solve_scenario_set_robust(pr, opt_draws=_opt_draws(pr, 5))
    assert evaluate_drawset(small.decision, _opt_draws(pr, 5), pr.compiled).joint_violation.sum() == 0
    fresh = _opt_draws(pr, 4000, seed=99, stream="validation")
    assert evaluate_drawset(small.decision, fresh, pr.compiled).joint_violation.mean() > 0.05


def test_scenario_set_leakage_and_input_checks():
    pr = three_problem()
    for bad in ("validation", "test", "outer"):
        with pytest.raises(LeakageError):
            solve_scenario_set_robust(pr, opt_draws=_opt_draws(pr, 10, stream=bad))
    assert solve_scenario_set_robust(pr).status is SolveStatus.INVALID_INPUT
    assert solve_scenario_set_robust(pr, opt_draws=_opt_draws(pr, 10),
                                     params={"gamma": 1}).status is SolveStatus.INVALID_INPUT
    ds = _opt_draws(pr, 10)
    th = np.array(ds.theta)
    th[3, 2, 0] = np.nan                                              # P's CP missing in one scenario
    r = solve_scenario_set_robust(pr, opt_draws=manual_draws(pr, th, ds.d, label="nan"))
    assert r.status is SolveStatus.INVALID_INPUT and "1 scenario" in r.message and r.decision is None
    # infeasible scenario set -> proven_infeasible, no ration
    th2 = np.array(ds.theta)
    th2[0, :, 0] = 0.05                                               # CP 5 % everywhere in one state
    r = solve_scenario_set_robust(pr, opt_draws=manual_draws(pr, th2, ds.d, label="inf"))
    assert r.status is SolveStatus.PROVEN_INFEASIBLE and r.decision is None and r.objective is None


# ==========================================================================================
# D  mean / nominal equivalence (reports/mean_nominal_equivalence.md)
# ==========================================================================================

def _lp_from_q_rows(pr, A_p, b_p):
    """Solve the LP with structural rows + given probabilistic q-space rows (x-space, HiGHS)."""
    from ration_reliability.optimization.lp_builder import assemble_x_space_lp

    dh = pr.dm_estimates()
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    ks = [k for k, c in enumerate(cc.classes) if c.value == "structural_hard"]
    rs = linear_rows(cc.subset(ks), pr.nominal_theta(), dh, d_hat=dh)
    A = np.vstack([rs.A[0], A_p])
    b = np.concatenate([rs.b, b_p])
    eq = np.concatenate([rs.is_eq, np.zeros(len(b_p), bool)])
    lp = assemble_x_space_lp(A, b, eq, [f"r{i}" for i in range(len(b))], dh, pr.price_vector())
    out = run_linprog(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, lp.ub, SolverOptions())
    assert out.status is SolveStatus.OPTIMAL
    q = out.x / dh
    return float(pr.price_vector() @ q), q


def _moment_rows(cc, Ed, Eda, dh):
    """Expected-coefficient rows computed from the moments E[d_i] and E[d_i a_ij] only."""
    K, I, J = cc.W.shape
    A = np.zeros((K, I))
    b = np.zeros(K)
    for k in range(K):
        sig = 1.0 if cc.senses[k].value == "le" else -1.0
        Edc = (cc.W[k] * Eda).sum(axis=1) + cc.w0[k] * Ed          # E[d_i c_ki]
        if cc.kinds[k].value == "concentration":
            A[k] = sig * (Edc - cc.bound[k] * Ed)
        else:
            A[k] = sig * (Edc + cc.v[k])
            b[k] = sig * cc.bound[k]
    return A, b


def _correlated_draws(pr, n, rho, seed=4):
    """Synthetic draws with DM-composition correlation rho for every ingredient (test only)."""
    rng = np.random.default_rng(seed)
    th, dh = pr.nominal_theta(), pr.dm_estimates()
    I, J = th.shape
    zd = rng.standard_normal((n, I))
    za = rho * zd[:, :, None] + math.sqrt(1 - rho ** 2) * rng.standard_normal((n, I, J))
    return manual_draws(pr, th * (1 + 0.10 * za), dh * (1 + 0.04 * zd), label=f"corr{rho}")


@pytest.mark.parametrize("rho", [0.0, 0.8, -0.8])
def test_expected_coefficient_lp_equals_multi_scenario_mean_lp(rho):
    """Always identical (linearity in q): LP with E_N[d a]-coefficients == LP 'average of the N
    scenario constraints' == M0 draw_mean.  Holds with or without correlation."""
    pr = three_problem()
    ds = _correlated_draws(pr, 400, rho)
    cc = prob_cc(pr)
    dh = pr.dm_estimates()
    rows = linear_rows(cc, ds.theta, ds.d, d_hat=dh)
    A_avg = rows.A.sum(axis=0) / rows.A.shape[0]                    # average of scenario constraints
    A_mom, b_mom = _moment_rows(cc, ds.d.mean(axis=0), (ds.d[:, :, None] * ds.theta).mean(axis=0), dh)
    assert np.allclose(A_avg, A_mom, rtol=1e-12, atol=1e-15) and np.array_equal(b_mom, rows.b)
    c_avg, q_avg = _lp_from_q_rows(pr, A_avg, rows.b)
    c_mom, q_mom = _lp_from_q_rows(pr, A_mom, b_mom)
    r_dm = M0(pr, opt_draws=manual_draws(pr, ds.theta, ds.d, label="m0"), params={"coefficient_mode": "draw_mean"})
    assert c_avg == pytest.approx(c_mom, rel=1e-12) == r_dm.objective
    assert np.allclose(q_avg, q_mom, atol=1e-9) and np.allclose(q_avg, r_dm.decision.q_as_fed, atol=1e-9)


def test_independent_factorial_design_plugin_equals_mean_lp():
    """Exact empirical independence (full factorial of every coordinate, centred on the table
    values): E_N[d a] = E_N[d] E_N[a] exactly, so M0 nominal_point == M0 draw_mean -> one method."""
    pr = three_problem()
    box = box_k(pr, 1.0)
    tv, dv = box_vertices(box)                                      # 4096 equally weighted states
    assert np.allclose(tv.mean(axis=0), pr.nominal_theta(), atol=1e-15)
    assert np.allclose(dv.mean(axis=0), pr.dm_estimates(), atol=1e-15)
    cov = (dv[:, :, None] * tv).mean(axis=0) - dv.mean(axis=0)[:, None] * tv.mean(axis=0)
    assert np.max(np.abs(cov)) < 1e-13                               # zero up to summation rounding
    r_pt = M0(pr)
    r_dm = M0(pr, opt_draws=manual_draws(pr, tv, dv, label="factorial"), params={"coefficient_mode": "draw_mean"})
    assert r_dm.objective == pytest.approx(r_pt.objective, rel=1e-12, abs=1e-13)
    assert np.allclose(r_dm.decision.q_as_fed, r_pt.decision.q_as_fed, atol=1e-9)


def test_nutrient_nutrient_correlation_alone_does_not_break_equivalence():
    """Correlation among a_ij (CP vs NDF, even perfect) does not enter the mean LP; only the DM-
    composition covariance does (the rows are linear in a for fixed d)."""
    pr = three_problem()
    th, dh = pr.nominal_theta(), pr.dm_estimates()
    states_t, states_d = [], []
    for sd_ in (-1, 1):                                             # d independent of a (factorial)
        for sa in (-1, 1):                                          # CP and NDF perfectly anti-correlated
            t = th.copy()
            t[:, 0] *= 1 + 0.1 * sa
            t[:, 1] *= 1 - 0.1 * sa
            states_t.append(t)
            states_d.append(dh * (1 + 0.04 * sd_))
    ds = manual_draws(pr, np.array(states_t), np.array(states_d), label="cp_ndf_anticorr")
    r_dm = M0(pr, opt_draws=ds, params={"coefficient_mode": "draw_mean"})
    assert r_dm.objective == pytest.approx(M0(pr).objective, rel=1e-12)


@pytest.mark.parametrize("rho", [0.9, 0.3, -0.5])
def test_mean_minus_plugin_coefficient_equals_sample_covariance(rho):
    """E_N[sigma d (c-K)] - sigma E_N[d](E_N[c]-K) = sigma cov_N(d, c)  (supply rows likewise)."""
    pr = three_problem()
    ds = _correlated_draws(pr, 300, rho, seed=12)
    cc = prob_cc(pr)
    dh = pr.dm_estimates()
    mean_rows = linear_rows(cc, ds.theta, ds.d, d_hat=dh).mean().A[0]
    plug_rows = linear_rows(cc, ds.theta.mean(axis=0), ds.d.mean(axis=0), d_hat=dh).A[0]
    c = np.einsum("sij,kij->ski", ds.theta, cc.W) + cc.w0[None]          # c_ki per scenario
    covN = (ds.d[:, None, :] * c).mean(axis=0) - ds.d.mean(axis=0)[None, :] * c.mean(axis=0)
    sig = np.array([1.0 if s.value == "le" else -1.0 for s in cc.senses])[:, None]
    assert np.allclose(mean_rows - plug_rows, sig * covN, atol=1e-13)
    if rho != 0:
        assert np.max(np.abs(covN)) > 1e-5


@pytest.mark.parametrize("states, exp_cost, exp_q", [
    # positive DM-CP correlation: (d, CP) in {(0.30, 0.06), (0.50, 0.14)} -> E[dCP] = 0.044 > 0.040
    (((0.30, 0.06), (0.50, 0.14)), 0.04 * 41.37931034482759 + 0.32 * 4.310344827586207,
     (41.37931034482759, 4.310344827586207)),
    # negative correlation: {(0.30, 0.14), (0.50, 0.06)} -> E[dCP] = 0.036 < 0.040
    (((0.30, 0.14), (0.50, 0.06)), 0.04 * 38.70967741935484 + 0.32 * 5.645161290322581,
     (38.70967741935484, 5.645161290322581)),
])
def test_correlated_dm_composition_counterexample(states, exp_cost, exp_q):
    """E[d a] != E[d] E[a]: the mean LP and the plug-in LP are different problems.

    F (planned d_hat 0.40), C fixed (d 0.80, CP 0.40), CP >= 16 %DM, planned DM 20 kg.
    Means E[d_F] = 0.40, E[CP_F] = 0.10 in both cases, so the plug-in LP is M0 nominal_point:
    q = (40, 5), cost 3.2.  Mean-LP row for F: 0.16 E[d] - E[d CP] = 0.020 (positive corr.) or
    0.028 (negative corr.); for C: 0.8 (0.16 - 0.40) = -0.192.  Hence q_F = (0.192/0.020) q_C or
    (0.192/0.028) q_C with 0.4 q_F + 0.8 q_C = 20.
    """
    pr = two_ingredient_problem(16.0)
    (d1, a1), (d2, a2) = states
    th = np.array([[[a1], [0.40]], [[a2], [0.40]]])
    d = np.array([[d1, 0.80], [d2, 0.80]])
    Ed, Ea, Eda = d[:, 0].mean(), th[:, 0, 0].mean(), (d[:, 0] * th[:, 0, 0]).mean()
    assert Ed == pytest.approx(0.40) and Ea == pytest.approx(0.10) and Eda != pytest.approx(Ed * Ea)
    ds = manual_draws(pr, th, d, label="two_state")
    r_mean = M0(pr, opt_draws=ds, params={"coefficient_mode": "draw_mean"})
    r_plug = M0(pr)
    assert r_plug.objective == pytest.approx(3.2) and np.allclose(r_plug.decision.q_as_fed, [40, 5])
    assert np.allclose(r_mean.decision.q_as_fed, exp_q, atol=1e-8)
    assert r_mean.objective == pytest.approx(exp_cost, abs=1e-9)
    assert abs(r_mean.objective - r_plug.objective) > 0.1
    # the plug-in ration does not satisfy the expected (mean) constraint when corr < 0
    g_mean = linear_rows(prob_cc(pr), th, d).mean().residual(r_plug.decision.q_as_fed)[0, 0]
    if Eda < Ed * Ea:
        assert g_mean == pytest.approx(0.028 * 40 - 0.192 * 5) and g_mean > 0
    else:
        assert g_mean < 0


def test_mean_lp_controls_ratio_of_means_not_mean_of_ratios():
    """The linearised mean LP enforces E[N] / E[D] >= K, not E[N / D] >= K."""
    pr = two_ingredient_problem(16.0)
    th = np.array([[[0.06], [0.40]], [[0.14], [0.40]]])
    d = np.array([[0.30, 0.80], [0.50, 0.80]])
    r = M0(pr, opt_draws=manual_draws(pr, th, d, label="ratio"), params={"coefficient_mode": "draw_mean"})
    ev = evaluate(r.decision, th, d, pr.compiled)
    N = ev.nutrient_supply[:, 0]
    D = ev.dm_supply
    assert N.mean() / D.mean() == pytest.approx(0.16, abs=1e-12)
    assert (N / D).mean() == pytest.approx(0.155528, abs=1e-6) and (N / D).mean() < 0.16
    assert ev.joint_violation.tolist() == [True, False]


def test_truncation_shift_breaks_nominal_point_equivalence():
    """C3 of the report: E_P[theta] must equal the table value; a truncated normal near 0 shifts
    the mean, so draw_mean and nominal_point then differ even with full independence."""
    mu, sd = 0.004, 0.004
    tmean = stats.truncnorm.mean((0 - mu) / sd, np.inf, loc=mu, scale=sd)
    assert tmean > mu * 1.2
    ids, nids = ("F", "M"), ("Ca",)
    m = IndependentNormalModel("tn", ids, nids, np.array([[mu], [0.35]]), np.array([[sd], [0.0]]),
                               np.array([0.4, 0.99]), np.zeros(2), truncate=True, is_synthetic=True)
    ds = m.draw(RandomStreams(3301), "opt", 200000)
    se = ds.theta[:, 0, 0].std(ddof=1) / math.sqrt(ds.n_draws)
    assert abs(ds.theta[:, 0, 0].mean() - tmean) < 5 * se
    assert ds.theta[:, 0, 0].mean() - mu > 20 * se


# ==========================================================================================
# E  general
# ==========================================================================================

def test_robust_results_go_through_common_evaluator():
    pr = three_problem()
    box = box_k(pr, 1.0)
    ds = _opt_draws(pr, 50)
    test_like = _opt_draws(pr, 2000, seed=4242, stream="validation")
    for r in (solve_box_robust(pr, params={"uncertainty_set": box}),
              solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": 1.5}),
              solve_scenario_set_robust(pr, opt_draws=ds)):
        assert r.status in OK and r.is_synthetic and r.solver_version.startswith("HiGHS")
        ev = evaluate_drawset(r.decision, test_like, pr.compiled, prices=pr.prices)
        assert ev.cost == pytest.approx(r.objective, abs=1e-12) and ev.structural_ok
        assert r.decision.method_id == r.method_id
        assert r.diagnostics["imposed_constraint_ids"] == ["dm_offer", "cp_min", "cp_max", "ndf_rule",
                                                           "starch_max", "starch_fndf", "cp_supply", "af_max_G"]
        d = r.to_dict()
        assert d["status"] == "optimal" and isinstance(d["params"], dict)
        assert np.allclose(r.decision.q_as_fed * pr.dm_estimates(), r.decision.x_planned_dm)


def test_permutation_invariance_box_and_budget():
    pr = three_problem()
    box = box_k(pr, 0.8)
    rb = solve_box_robust(pr, params={"uncertainty_set": box})
    rg = solve_budget_robust(pr, params={"uncertainty_set": box, "gamma": 1.3})
    order, norder = ("P", "F", "G"), ("starch", "CP", "NDF")
    pr2 = pr.reordered(order, norder, constraint_order=[c.constraint_id for c in reversed(pr.constraints)])
    box2 = box.reordered(order, norder)
    rb2 = solve_box_robust(pr2, params={"uncertainty_set": box2})
    rg2 = solve_budget_robust(pr2, params={"uncertainty_set": box2, "gamma": 1.3})
    assert rb2.objective == pytest.approx(rb.objective, rel=1e-10)
    assert rg2.objective == pytest.approx(rg.objective, rel=1e-10)
    idx = [order.index(i) for i in pr.ingredient_ids]
    assert np.allclose(rb2.decision.q_as_fed[idx], rb.decision.q_as_fed, atol=1e-8)
