"""M2 joint chance-constrained SAA and M2b marginal/Bonferroni (contract T4 M2, section 8 item 10).

* analytic normal single chance constraint  mu'x - Phi^-1(1-alpha) sqrt(x' Sigma x) >= b  vs
  independent Monte Carlo through the public evaluator (tolerance from the MC standard error);
* SAA solution = order-statistic enumeration (1-D case) and its true violation probability ~ alpha;
* joint vs marginal are not confused (each row at 95 % is not joint 95 %);
* alpha -> 0 gives the all-scenario robust LP; cost non-increasing in alpha;
* training joint violation rate <= floor(alpha N)/N on the N ladder 128 / 512 / 1024;
* Big-M validity (never cuts an SAA-feasible point) and looseness, three Big-M modes = brute force;
* status mapping: proven_infeasible / no_feasible_solution_found / feasible_time_limit / invalid.

The normal examples only verify the implementation; they are not evidence that feed composition is
normal (contract section 8 item 10).  All values are synthetic (``is_synthetic=True``).
"""

import itertools

import numpy as np
import pytest
from scipy import optimize, stats
from scipy.optimize import brentq

from engine_test_helpers import af_max, conc, dm_offer, ing, problem, supply
from ration_reliability.datamodel import SolveStatus, SolverOptions
from ration_reliability.errors import LeakageError
from ration_reliability.evaluation import evaluate, evaluate_drawset
from ration_reliability.nutrition import linear_rows
from ration_reliability.optimization import get_method
from ration_reliability.optimization import highs as H
from ration_reliability.optimization.chance_saa import (
    _run_milp_sparse,
    _solve_saa,
    PARAM_DEFAULTS_JOINT,
    allowed_violations,
    build_saa_model,
    quantile_big_m,
    select_alpha_train_on_validation,
)
from ration_reliability.optimization.lp_builder import optimization_indices
from ration_reliability.uncertainty import (
    DrawSet,
    IndependentNormalModel,
    PointMassModel,
    RandomStreams,
)

M0 = get_method("M0_nominal")
M2 = get_method("M2_joint_chance_saa")
M2b = get_method("M2b_marginal_bonferroni_saa")
EXACT = SolverOptions(mip_rel_gap=0.0)
Z95 = stats.norm.ppf(0.95)

# ------------------------------------------------------------------------------------------
# synthetic problems
# ------------------------------------------------------------------------------------------
MU2 = np.array([[0.10], [0.40]])
SD2 = np.array([[0.015], [0.03]])


def two_ing(cp_min=16.0):
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    return problem([F, C], ["CP"], [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", cp_min)],
                   {"F": 0.04, "C": 0.32}, problem_id="syn_two_ing_chance")


def two_ing_model(pr):
    return IndependentNormalModel("syn_n2", pr.ingredient_ids, pr.nutrient_ids, MU2, SD2, pr.dm_estimates(),
                                  np.zeros(2), truncate=False, is_synthetic=True)


def three_ing(extra=()):
    F = ing("F", 0.40, {"CP": 0.10, "NDF": 0.55}, forage=1.0)
    G = ing("G", 0.88, {"CP": 0.09, "NDF": 0.12})
    P = ing("P", 0.89, {"CP": 0.46, "NDF": 0.14})
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0), conc("ndf_max", {"NDF": 1.0}, "le", 40.0)]
    return problem([F, G, P], ["CP", "NDF"], cons + list(extra), {"F": 0.03, "G": 0.20, "P": 0.40},
                   problem_id="syn_three_ing_chance")


SD3 = np.array([[0.012, 0.03], [0.008, 0.015], [0.02, 0.015]])


def three_model(pr, dm_sd=(0.02, 0.01, 0.01)):
    return IndependentNormalModel("syn_n3", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), SD3,
                                  pr.dm_estimates(), np.array(dm_sd), truncate=False, is_synthetic=True)


def _m_t(t, K=0.16, D=20.0):
    """mean and sd of sum_i x_i (a_i - K) for x = (D - t, t) in the two-ingredient example."""
    m = (D - t) * (MU2[0, 0] - K) + t * (MU2[1, 0] - K)
    v = np.sqrt((D - t) ** 2 * SD2[0, 0] ** 2 + t ** 2 * SD2[1, 0] ** 2)
    return m, v


def p_viol(t):
    m, v = _m_t(t)
    return float(stats.norm.cdf(-m / v))


# ------------------------------------------------------------------------------------------
# budget arithmetic
# ------------------------------------------------------------------------------------------

def test_allowed_violations_exact_floor():
    assert allowed_violations(0.29, 100) == 29          # float 0.29 * 100 = 28.999999999999996
    assert allowed_violations(0.05, 128) == 6 and allowed_violations(0.05, 1024) == 51
    assert allowed_violations(0.0, 500) == 0 and allowed_violations(0.01, 64) == 0
    with pytest.raises(ValueError):
        allowed_violations(1.0, 10)


# ------------------------------------------------------------------------------------------
# contract section 8 item 10: analytic normal chance constraint vs independent MC
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_normal_single_chance_constraint_formula_vs_mc(seed):
    """Correlated normal CP (3 ingredients), fixed DM.  With b = mu'x - z_{1-a} sqrt(x'Sx), the
    supply S = sum x_i a_i satisfies P(S < b) = a; the evaluator's MC rate must agree."""
    rng = np.random.default_rng(40 + seed)              # synthetic generator (test only)
    mu = np.array([0.10, 0.18, 0.45])
    sd = np.array([0.015, 0.02, 0.025])
    C = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    Sigma = np.outer(sd, sd) * C
    dm = np.array([0.35, 0.88, 0.89])
    q = rng.uniform(5.0, 30.0, 3)
    x = q * dm
    alpha = [0.05, 0.10, 0.01][seed]
    b = float(mu @ x - stats.norm.ppf(1 - alpha) * np.sqrt(x @ Sigma @ x))    # kg/d
    ings = [ing(f"i{k}", dm[k], {"CP": mu[k]}) for k in range(3)]
    # tolerance must be > 0 since the red-team fix C10; 1e-12 kg/d changes P(violation) by ~density x 1e-12
    pr = problem(ings, ["CP"], [supply("cp_sup", {"CP": 1.0}, "ge", b, "kg/d", tol=1e-12)],
                 {f"i{k}": 0.1 for k in range(3)})
    n = 400_000
    gen = RandomStreams(1103).generator("analytic_check", seed)   # independent custom stream
    theta = gen.multivariate_normal(mu, Sigma, size=n)[:, :, None]
    ev = evaluate(q, theta, np.broadcast_to(dm, (n, 3)), pr.compiled)
    p_hat = ev.joint_violation.mean()
    se = np.sqrt(alpha * (1 - alpha) / n)
    assert abs(p_hat - alpha) < 4 * se


def test_analytic_chance_optimum_has_violation_alpha_by_mc():
    """Two ingredients, CP >= 16 % DM, independent normal CP.  The analytic chance-constrained
    optimum t* (smallest x_C with m(t) - z sd(t) >= 0) has violation probability exactly alpha."""
    pr = two_ing()
    t_star = brentq(lambda t: _m_t(t)[0] - Z95 * _m_t(t)[1], 0.0, 20.0, xtol=1e-12)
    x = np.array([20.0 - t_star, t_star])
    q = x / pr.dm_estimates()
    assert p_viol(t_star) == pytest.approx(0.05, abs=1e-9)
    n = 400_000
    ds = two_ing_model(pr).draw(RandomStreams(2207), "analytic_check", n)
    ev = evaluate_drawset(q, ds, pr.compiled, d_hat=pr.dm_estimates())
    assert abs(ev.joint_violation.mean() - 0.05) < 4 * np.sqrt(0.05 * 0.95 / n)


@pytest.mark.parametrize("N", [400, 1000])
def test_saa_equals_order_statistic_and_true_risk_near_alpha(N):
    pr = two_ing()
    opt = two_ing_model(pr).draw(RandomStreams(1103), "opt", N)
    r = M2(pr, opt_draws=opt, params={"alpha_train": 0.05}, solver_options=EXACT)
    assert r.status is SolveStatus.OPTIMAL and r.mip_gap is not None
    a = opt.theta[:, :, 0]
    t_s = 20.0 * (0.16 - a[:, 0]) / (a[:, 1] - a[:, 0])      # scenario s satisfied iff x_C >= t_s
    m = allowed_violations(0.05, N)
    t_saa = np.sort(t_s)[N - m - 1]
    assert r.decision.x_planned_dm[1] == pytest.approx(t_saa, abs=1e-7)
    # true (analytic) violation probability of the SAA ration is alpha up to SAA sampling error
    assert abs(p_viol(r.decision.x_planned_dm[1]) - 0.05) <= 4 * np.sqrt(0.05 * 0.95 / N) + 1.0 / N
    # training joint violation (public evaluator) within the budget
    ev = evaluate_drawset(r.decision, opt, pr.compiled)
    assert ev.joint_violation.sum() <= m


# ------------------------------------------------------------------------------------------
# joint vs marginal
# ------------------------------------------------------------------------------------------

def test_each_constraint_95_is_not_joint_95_analytic():
    """Two independent constraints each violated with probability 5 % give a joint violation
    probability 1 - 0.95^2 = 9.75 %, not 5 %."""
    pr0 = three_ing()
    q = np.array([30.0, 5.0, 4.0])
    dm = pr0.dm_estimates()
    x = q * dm
    D = x.sum()
    mu = pr0.nominal_theta()
    mc, sc = x @ mu[:, 0] / D, np.sqrt((x ** 2) @ SD3[:, 0] ** 2) / D
    mn, sn = x @ mu[:, 1] / D, np.sqrt((x ** 2) @ SD3[:, 1] ** 2) / D
    # tolerance > 0 required (red-team C10); 1e-12 %DM is negligible for these continuous MC checks
    cons = [dm_offer(D), conc("cp_min", {"CP": 1.0}, "ge", 100 * (mc - Z95 * sc), tol=1e-12),
            conc("ndf_max", {"NDF": 1.0}, "le", 100 * (mn + Z95 * sn), tol=1e-12)]
    pr = problem(pr0.ingredients, ["CP", "NDF"], cons, {"F": 0.03, "G": 0.20, "P": 0.40})
    n = 400_000
    ds = three_model(pr, dm_sd=(0.0, 0.0, 0.0)).draw(RandomStreams(3301), "analytic_check", n)
    ev = evaluate_drawset(q, ds, pr.compiled, d_hat=dm)
    per = ev.violated.mean(axis=0)
    se = lambda p: np.sqrt(p * (1 - p) / n)   # noqa: E731
    assert np.all(np.abs(per - 0.05) < 4 * se(0.05))
    p_joint = 1 - 0.95 ** 2
    assert abs(ev.joint_violation.mean() - p_joint) < 4 * se(p_joint)


def test_joint_saa_vs_marginal_each_alpha_vs_bonferroni():
    pr = three_ing()
    N, alpha = 400, 0.10
    opt = three_model(pr, dm_sd=(0.0, 0.0, 0.0)).draw(RandomStreams(2207), "opt", N)
    m = allowed_violations(alpha, N)
    r_joint = M2(pr, opt_draws=opt, params={"alpha_train": alpha}, solver_options=EXACT)
    r_bonf = M2b(pr, opt_draws=opt, params={"alpha_train": alpha}, solver_options=EXACT)
    p = dict(PARAM_DEFAULTS_JOINT, alpha_train=alpha)
    # deliberately *wrong* reading "each row at alpha" -- only reachable through the private core
    r_each = _solve_saa(pr, method_id="TEST_marginal_each_alpha_NOT_JOINT", mode="marginal", d_hat=None,
                        opt_draws=opt, p=p, opts=EXACT, row_budgets=[m, m])
    for r in (r_joint, r_bonf, r_each):
        assert r.status is SolveStatus.OPTIMAL
    ev = {k: evaluate_drawset(r.decision, opt, pr.compiled) for k, r in
          (("joint", r_joint), ("bonf", r_bonf), ("each", r_each))}
    # each row at alpha: every marginal training rate <= alpha, but the joint rate exceeds alpha
    assert np.all(ev["each"].violated.mean(axis=0) <= alpha + 1e-12)
    assert ev["each"].joint_violation.mean() > alpha
    # joint SAA: joint rate <= alpha;  Bonferroni: rows <= alpha/2, joint <= alpha
    assert ev["joint"].joint_violation.sum() <= m
    assert np.all(ev["bonf"].violated.sum(axis=0) <= allowed_violations(alpha / 2, N))
    assert ev["bonf"].joint_violation.mean() <= alpha
    # feasible-set inclusions give the cost order  each <= joint <= Bonferroni
    tol = 1e-7
    assert r_each.objective <= r_joint.objective + tol <= r_bonf.objective + 2 * tol
    assert r_bonf.diagnostics["row_budgets"] == {"cp_min": 20, "ndf_max": 20}
    assert "not the joint SAA" in r_bonf.diagnostics["naming_note"]
    assert r_joint.diagnostics["joint_event"].startswith("common z_s")


def test_bonferroni_allocation_must_not_exceed_alpha():
    pr = three_ing()
    opt = three_model(pr).draw(RandomStreams(1103), "opt", 50)
    bad = M2b(pr, opt_draws=opt, params={"alpha_train": 0.05, "alpha_allocation": {"cp_min": 0.05, "ndf_max": 0.05}})
    assert bad.status is SolveStatus.INVALID_INPUT and "sum_k alpha_k" in bad.message and bad.decision is None
    assert M2b(pr, opt_draws=opt, params={"alpha_train": 0.05, "alpha_allocation": {"cp_min": 0.05}}).status \
        is SolveStatus.INVALID_INPUT
    assert M2b(pr, opt_draws=opt, params={"alpha_train": 0.05, "alpha_allocation": {"cp_min": -0.01,
                                                                                     "ndf_max": 0.02}}).status \
        is SolveStatus.INVALID_INPUT
    ok = M2b(pr, opt_draws=opt, params={"alpha_train": 0.1, "alpha_allocation": {"cp_min": 0.06, "ndf_max": 0.04}},
             solver_options=EXACT)
    assert ok.status is SolveStatus.OPTIMAL and ok.diagnostics["row_budgets"] == {"cp_min": 3, "ndf_max": 2}


# ------------------------------------------------------------------------------------------
# degeneracies and monotonicity
# ------------------------------------------------------------------------------------------

def _robust_lp(pr, opt, N):
    """All-scenario robust LP built independently (linear_rows + linprog)."""
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    dh = pr.dm_estimates()
    pidx = [k for k, c in enumerate(cc.classes) if str(c) == "probabilistic_nutrition"]
    rows = linear_rows(cc.subset(pidx), opt.theta[:N], opt.d[:N], d_hat=dh)
    A = (rows.A / dh[None, None, :]).reshape(-1, len(dh))
    b = np.tile(rows.b, N)
    res = optimize.linprog(pr.price_vector() / dh, A_ub=A, b_ub=b, A_eq=np.ones((1, len(dh))), b_eq=[20.0],
                           bounds=[(0, None)] * len(dh), method="highs")
    return res


def test_alpha_zero_is_all_scenario_robust_lp():
    pr = three_ing()
    N = 64
    opt = three_model(pr).draw(RandomStreams(1103), "opt", N)
    ref = _robust_lp(pr, opt, N)
    assert ref.status == 0
    for a in (0.0, 0.01):               # floor(0.01 * 64) = 0 as well
        r = M2(pr, opt_draws=opt, params={"alpha_train": a}, solver_options=EXACT)
        assert r.status is SolveStatus.OPTIMAL and r.diagnostics["allowed_violations"] == 0
        assert r.objective == pytest.approx(pr.price_vector() @ (ref.x / pr.dm_estimates()), rel=1e-9, abs=1e-9)
        assert np.allclose(r.decision.x_planned_dm, ref.x, atol=1e-6)
        assert evaluate_drawset(r.decision, opt, pr.compiled).joint_violation.sum() == 0


def test_cost_non_increasing_in_alpha():
    pr = three_ing()
    opt = three_model(pr).draw(RandomStreams(2207), "opt", 200)
    costs = []
    for a in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
        r = M2(pr, opt_draws=opt, params={"alpha_train": a}, solver_options=EXACT)
        assert r.status is SolveStatus.OPTIMAL and r.mip_gap == pytest.approx(0.0, abs=1e-9)
        costs.append(r.objective)
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))
    assert costs[-1] < costs[0]


def test_zero_uncertainty_equals_m0():
    pr = three_ing()
    pm = PointMassModel("pm", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), pr.dm_estimates(),
                        is_synthetic=True)
    opt = pm.draw(RandomStreams(1103), "opt", 40)
    r0 = M0(pr)
    r2 = M2(pr, opt_draws=opt, params={"alpha_train": 0.05}, solver_options=EXACT)
    assert r2.status is SolveStatus.OPTIMAL
    assert r2.objective == pytest.approx(r0.objective, abs=1e-9)
    assert np.allclose(r2.decision.q_as_fed, r0.decision.q_as_fed, atol=1e-7)


@pytest.mark.parametrize("N", [128, 512, 1024])
def test_training_joint_violation_rate_within_budget_on_ladder(N):
    """Nested N ladder from one opt draw set (DM uncertain as well); public evaluator on the
    training scenarios: joint rate <= floor(alpha N)/N <= alpha."""
    pr = three_ing()
    opt = three_model(pr).draw(RandomStreams(3301), "opt", 1024)
    alpha = 0.05
    r = M2(pr, opt_draws=opt, params={"alpha_train": alpha, "n_scenarios": N}, solver_options=EXACT)
    assert r.status is SolveStatus.OPTIMAL
    m = allowed_violations(alpha, N)
    first_n = opt.reordered(scenario_order=range(N))
    ev = evaluate_drawset(r.decision, first_n, pr.compiled, prices=pr.prices)
    assert ev.joint_violation.sum() <= m and ev.joint_violation.mean() <= alpha
    assert ev.structural_ok and ev.cost == pytest.approx(r.objective)
    d = r.diagnostics
    assert d["n_scenarios_used"] == N and d["allowed_violations"] == m and d["n_z_one"] <= m
    assert d["n_train_scenarios_violated_linearised"] <= m and d["n_enforced_rows_violated_linearised"] == 0
    assert d["polish"]["applied"] and d["polish"]["status"] == "optimal"
    assert r.streams_used == (opt.stream_id,) and r.params["n_scenarios"] == N


# ------------------------------------------------------------------------------------------
# Big-M: validity, looseness, three modes = brute force enumeration
# ------------------------------------------------------------------------------------------

def _small():
    pr = three_ing()
    opt = three_model(pr).draw(RandomStreams(1103), "opt", 12)
    return pr, opt


def test_box_big_m_valid_and_equal_to_box_vertex_max():
    pr, opt = _small()
    mb = build_saa_model(pr, opt, big_m_mode="box")
    mt = build_saa_model(pr, opt, big_m_mode="lp_tight")
    # not too tight: box M >= max over the structural polytope (LP-tight M)
    assert np.all(mb.M >= mt.M - 1e-9)
    # not looser than the box itself: equals the maximum over the 2^I vertices of the padded box
    lo, hi = mb.padded_box
    verts = np.array(list(itertools.product(*zip(lo, hi))))                  # [8, 3]
    brute = np.einsum("ski,vi->skv", mb.A_prob, verts).max(axis=2) - mb.b_prob[None, :]
    assert np.allclose(mb.M, brute, atol=1e-12)
    # structural box: DM offer 20 kg DM -> each x_i in [0, 20]
    assert np.allclose(mb.x_lo, 0.0) and np.allclose(mb.x_hi, 20.0)


def _enumerate_joint(pr, opt, N, m):
    """Brute force: min over relaxed-scenario sets |S| <= m of the LP with all other rows."""
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    dh = pr.dm_estimates()
    pidx = [k for k, c in enumerate(cc.classes) if str(c) == "probabilistic_nutrition"]
    rows = linear_rows(cc.subset(pidx), opt.theta[:N], opt.d[:N], d_hat=dh)
    A = rows.A / dh[None, None, :]
    best, sols = np.inf, []
    for r in range(m + 1):
        for S in itertools.combinations(range(N), r):
            keep = [s for s in range(N) if s not in S]
            res = optimize.linprog(pr.price_vector() / dh, A_ub=A[keep].reshape(-1, len(dh)),
                                   b_ub=np.tile(rows.b, len(keep)), A_eq=np.ones((1, len(dh))), b_eq=[20.0],
                                   bounds=[(0, None)] * len(dh), method="highs")
            if res.status == 0:
                sols.append(res.x)
                best = min(best, res.fun)
    return best, sols, A, rows.b


@pytest.mark.parametrize("mode", ["box_quantile", "box", "lp_tight"])
def test_big_m_modes_match_brute_force_joint(mode):
    pr, opt = _small()
    N, alpha = 12, 0.2          # m = 2 -> 1 + 12 + 66 = 79 LPs
    m = allowed_violations(alpha, N)
    best, sols, A, b = _enumerate_joint(pr, opt, N, m)
    r = M2(pr, opt_draws=opt, params={"alpha_train": alpha, "big_m_mode": mode}, solver_options=EXACT)
    assert r.status is SolveStatus.OPTIMAL
    assert r.decision.x_planned_dm @ (pr.price_vector() / pr.dm_estimates()) == pytest.approx(best, abs=1e-8)
    bm = r.diagnostics["big_m"]
    assert bm["mode"] == mode and bm["flags"] == []
    assert bm["int_leak_linear_max"] == pytest.approx(1e-6 * bm["M_max"])
    if mode != "lp_tight":
        assert bm["box_looseness_ratio_max"] >= 1.0 - 1e-9
    # direct validity check of the quantile M on SAA-feasible points: every enumerated LP optimum
    # x violates at most m scenarios, so g_{s,k}(x) <= M_used[s,k] must hold for every row
    model = build_saa_model(pr, opt, big_m_mode="box_quantile")
    lo, hi = model.padded_box
    Mq = np.minimum(model.M, quantile_big_m(model.A_prob, lo, hi, [m, m]))
    for x in sols:
        g = np.einsum("ski,i->sk", A, x) - b[None, :]
        assert (g > 1e-9).any(axis=1).sum() <= m
        assert np.all(g <= Mq + 1e-7)        # also for hard rows (Mq <= 0): g <= Mq, not only <= 0


def test_quantile_m_valid_on_random_saa_feasible_points():
    pr, opt = _small()
    N, m = 12, 3
    model = build_saa_model(pr, opt, big_m_mode="box_quantile")
    lo, hi = model.padded_box
    Mq = np.minimum(model.M, quantile_big_m(model.A_prob, lo, hi, [m, m]))
    rng = np.random.default_rng(7)             # synthetic generator (test only)
    checked = 0
    for _ in range(4000):
        w = rng.dirichlet(np.ones(3))
        x = 20.0 * w                              # every point of the structural polytope (sum x = 20)
        g = np.einsum("ski,i->sk", model.A_prob, x) - model.b_prob[None, :]
        if (g > 0).any(axis=1).sum() <= m:        # SAA-feasible for budget m
            checked += 1
            assert np.all(g <= Mq + 1e-9)
    assert checked > 50


def test_big_m_marginal_matches_brute_force():
    pr, opt = _small()
    N = 10
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    dh = pr.dm_estimates()
    rows = linear_rows(cc.subset([1, 2]), opt.theta[:N], opt.d[:N], d_hat=dh)
    A = rows.A / dh[None, None, :]
    best = np.inf
    for S1 in [()] + [(s,) for s in range(N)]:
        for S2 in [()] + [(s,) for s in range(N)]:
            blocks = [A[[s for s in range(N) if s not in S1], 0, :], A[[s for s in range(N) if s not in S2], 1, :]]
            rhs = [np.full(len(blocks[0]), rows.b[0]), np.full(len(blocks[1]), rows.b[1])]
            res = optimize.linprog(pr.price_vector() / dh, A_ub=np.vstack(blocks), b_ub=np.concatenate(rhs),
                                   A_eq=np.ones((1, 3)), b_eq=[20.0], bounds=[(0, None)] * 3, method="highs")
            if res.status == 0:
                best = min(best, res.fun)
    for mode in ("box_quantile", "box"):
        r = M2b(pr, opt_draws=opt, params={"alpha_train": 0.2, "n_scenarios": N, "big_m_mode": mode},
                solver_options=EXACT)   # alpha_k = 0.1 -> one violation per row
        assert r.status is SolveStatus.OPTIMAL and r.diagnostics["row_budgets"] == {"cp_min": 1, "ndf_max": 1}
        assert r.decision.x_planned_dm @ (pr.price_vector() / dh) == pytest.approx(best, abs=1e-8)


def test_always_satisfied_rows_are_dropped_not_given_negative_m():
    pr = three_ing(extra=[conc("cp_floor", {"CP": 1.0}, "ge", 2.0)])   # every ingredient far above 2 %
    opt = three_model(pr).draw(RandomStreams(2207), "opt", 30)
    r = M2(pr, opt_draws=opt, params={"alpha_train": 0.1, "big_m_mode": "box"}, solver_options=EXACT)
    ref = M2(three_ing(), opt_draws=opt, params={"alpha_train": 0.1, "big_m_mode": "box"}, solver_options=EXACT)
    assert r.status is SolveStatus.OPTIMAL
    assert r.diagnostics["big_m"]["n_rows_dropped_always_satisfied"] >= 30
    assert r.objective == pytest.approx(ref.objective, abs=1e-9)


# ------------------------------------------------------------------------------------------
# statuses
# ------------------------------------------------------------------------------------------

def test_saa_infeasible_is_proven_infeasible():
    pr = two_ing(cp_min=60.0)                      # 60 % is ~7 sd above the richest ingredient
    opt = two_ing_model(pr).draw(RandomStreams(1103), "opt", 100)
    r = M2(pr, opt_draws=opt, params={"alpha_train": 0.05})
    assert r.status is SolveStatus.PROVEN_INFEASIBLE and r.decision is None and r.objective is None
    rb = M2b(pr, opt_draws=opt, params={"alpha_train": 0.05})
    assert rb.status is SolveStatus.PROVEN_INFEASIBLE and rb.decision is None


def test_structural_infeasible_and_unbounded():
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    cons = [dm_offer(20.0), af_max("F", 10.0), af_max("C", 10.0), conc("cp", {"CP": 1.0}, "ge", 16.0)]
    pr = problem([F, C], ["CP"], cons, {"F": 0.04, "C": 0.32})
    opt = two_ing_model(pr).draw(RandomStreams(1103), "opt", 20)
    r = M2(pr, opt_draws=opt, params={"alpha_train": 0.05})
    assert r.status is SolveStatus.PROVEN_INFEASIBLE and "structural" in r.message and r.decision is None
    pr2 = problem([F, C], ["CP"], [conc("cp", {"CP": 1.0}, "ge", 16.0)], {"F": 0.04, "C": 0.32})
    r2 = M2(pr2, opt_draws=opt, params={"alpha_train": 0.05})
    assert r2.status is SolveStatus.INVALID_INPUT and "unbounded" in r2.message


def test_limits_map_to_feasible_time_limit_or_no_solution():
    pr = three_ing()
    opt = three_model(pr).draw(RandomStreams(2207), "opt", 512)
    r0 = M2(pr, opt_draws=opt, params={"alpha_train": 0.05, "mip_node_limit": 0})
    assert r0.status is SolveStatus.NO_FEASIBLE_SOLUTION_FOUND and r0.decision is None and r0.objective is None
    r1 = M2(pr, opt_draws=opt, params={"alpha_train": 0.05, "mip_node_limit": 1}, solver_options=EXACT)
    assert r1.status is SolveStatus.FEASIBLE_TIME_LIMIT and r1.decision is not None
    assert r1.mip_gap is not None and r1.mip_gap > 0
    ev = evaluate_drawset(r1.decision, opt, pr.compiled)
    assert ev.joint_violation.sum() <= allowed_violations(0.05, 512)   # incumbent still meets the budget
    rt = M2(pr, opt_draws=opt, params={"alpha_train": 0.05}, solver_options=SolverOptions(time_limit_s=1e-9))
    assert rt.status is SolveStatus.NO_FEASIBLE_SOLUTION_FOUND and rt.decision is None
    assert rt.status is not SolveStatus.PROVEN_INFEASIBLE


def test_sparse_milp_runner_matches_dense_and_maps_limits():
    from scipy import sparse
    rng = np.random.default_rng(1)                  # synthetic market-split instance (test only)
    m, n = 4, 30
    A = rng.integers(0, 100, (m, n)).astype(float)
    b = np.floor(A.sum(1) / 2)
    Aeq = np.hstack([A, np.eye(m), -np.eye(m)])
    c = np.concatenate([rng.integers(1, 50, n) * 0.001, 1000 * np.ones(2 * m)])
    integ = np.concatenate([np.ones(n), np.zeros(2 * m)])
    lb, ub = np.zeros(n + 2 * m), np.concatenate([np.ones(n), np.full(2 * m, np.inf)])
    out = _run_milp_sparse(c, None, None, sparse.csr_matrix(Aeq), b, lb, ub, integ, SolverOptions(), 2)
    assert out.status is SolveStatus.FEASIBLE_TIME_LIMIT and out.x is not None and out.mip_gap > 0
    out0 = _run_milp_sparse(c, None, None, sparse.csr_matrix(Aeq), b, lb, ub, integ, SolverOptions(time_limit_s=0.0),
                            None)
    assert out0.status is SolveStatus.NO_FEASIBLE_SOLUTION_FOUND and out0.x is None
    # small instance: sparse runner == engine dense run_milp
    c2, A2, b2 = [1.0, 1.0, 0.5], [[-1.0, 0.0, -1.0], [0.0, -1.0, -1.0]], [-1.5, -0.5]
    d = H.run_milp(c2, A2, b2, None, None, [0, 0, 0], [5, 5, 1], [1, 0, 1], SolverOptions(mip_rel_gap=0.0))
    s = _run_milp_sparse(c2, sparse.csr_matrix(A2), b2, None, None, np.zeros(3), np.array([5, 5, 1.0]),
                         np.array([1, 0, 1]), SolverOptions(mip_rel_gap=0.0), None)
    assert d.status is s.status is SolveStatus.OPTIMAL and s.fun == pytest.approx(d.fun)


def test_invalid_inputs_and_leakage():
    pr = three_ing()
    model = three_model(pr)
    s = RandomStreams(1103)
    opt, val, test = model.draw(s, "opt", 30), model.draw(s, "validation", 30), model.draw(s, "test", 30)
    for draws in (val, test):
        with pytest.raises(LeakageError):
            M2(pr, opt_draws=draws, params={"alpha_train": 0.05})
        with pytest.raises(LeakageError):
            M2b(pr, opt_draws=draws, params={"alpha_train": 0.05})
        with pytest.raises(LeakageError):          # invalid params never hide leakage
            M2(pr, opt_draws=draws, params={"alpha_train": None})
    bad = [{}, {"alpha_train": 1.0}, {"alpha_train": -0.1}, {"alpha_train": True}, {"alpha_train": "x"},
           {"alpha_train": 0.05, "n_scenarios": 0}, {"alpha_train": 0.05, "n_scenarios": 31},
           {"alpha_train": 0.05, "big_m_mode": "1e6"}, {"alpha_train": 0.05, "k": 1.0},
           {"alpha_train": 0.05, "mip_node_limit": -1}, {"alpha_train": 0.05, "polish": "yes"}]
    for prm in bad:
        r = M2(pr, opt_draws=opt, params=prm)
        assert r.status is SolveStatus.INVALID_INPUT and r.decision is None and r.objective is None, prm
    assert M2(pr, params={"alpha_train": 0.05}).status is SolveStatus.INVALID_INPUT          # no opt draws
    # missing composition in the scenarios -> invalid, never set to 0
    th = opt.theta.copy()
    th[3, 0, 0] = np.nan
    opt_nan = DrawSet(th, opt.d, "opt", "root=1103/opt", "nan_test", "nan_test_fp", pr.ingredient_ids,
                      pr.nutrient_ids, True)
    r = M2(pr, opt_draws=opt_nan, params={"alpha_train": 0.05})
    assert r.status is SolveStatus.INVALID_INPUT and "missing" in r.message and r.decision is None


def test_select_alpha_train_on_validation_only():
    pr = three_ing()
    model = three_model(pr)
    s = RandomStreams(2207)
    opt, val = model.draw(s, "opt", 200), model.draw(s, "validation", 3000)
    grid = [0.0, 0.01, 0.02, 0.05, 0.10]
    sel = select_alpha_train_on_validation(pr, val, opt_draws=opt, grid=grid, target_alpha=0.05,
                                           screening_rule="rate_upper_le_alpha", solver_options=EXACT)
    assert sel.status == "selected" and sel.method_id == "M2_joint_chance_saa"
    meeting = [c for c in sel.candidates if c.meets_screen]
    assert sel.selected_value == min(meeting, key=lambda c: (c.cost, c.rate_upper)).value
    res = sel.selected_result
    assert res.params["selection"]["selected_on_stream"] == val.stream_id
    assert res.streams_used == (opt.stream_id, val.stream_id)
    with pytest.raises(LeakageError):
        select_alpha_train_on_validation(pr, model.draw(s, "test", 100), opt_draws=opt, grid=grid,
                                         target_alpha=0.05, screening_rule="rate_upper_le_alpha")
    sel_b = select_alpha_train_on_validation(pr, val, opt_draws=opt, grid=grid, target_alpha=0.05,
                                             screening_rule="rate_upper_le_alpha", solver_options=EXACT,
                                             marginal_bonferroni=True)
    assert sel_b.method_id == "M2b_marginal_bonferroni_saa"


def test_permutation_invariance():
    pr = three_ing()
    opt = three_model(pr).draw(RandomStreams(1103), "opt", 150)
    r = M2(pr, opt_draws=opt, params={"alpha_train": 0.05}, solver_options=EXACT)
    order = ["P", "F", "G"]
    pr2 = pr.reordered(ingredient_order=order)
    opt2 = opt.reordered(ingredient_ids=order)
    r2 = M2(pr2, opt_draws=opt2, params={"alpha_train": 0.05}, solver_options=EXACT)
    assert r2.objective == pytest.approx(r.objective, rel=1e-9, abs=1e-9)
    assert np.allclose(r2.decision.reordered(pr.ingredient_ids).q_as_fed, r.decision.q_as_fed, atol=1e-6)


def test_no_probabilistic_rows_equals_m0():
    F = ing("F", 0.40, {"CP": 0.10}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40})
    pr = problem([F, C], ["CP"], [dm_offer(20.0), conc("cp_diag", {"CP": 1.0}, "ge", 16.0, cls="diagnostic_only")],
                 {"F": 0.04, "C": 0.32})
    opt = two_ing_model(pr).draw(RandomStreams(1103), "opt", 10)
    r0 = M0(pr)
    for fn in (M2, M2b):
        r = fn(pr, opt_draws=opt, params={"alpha_train": 0.1})
        assert r.status is SolveStatus.OPTIMAL and r.objective == pytest.approx(r0.objective, abs=1e-12)
        assert r.diagnostics["probabilistic_constraint_ids"] == []


def test_toy_pipeline_streams_are_separated(toy_problem):
    """Synthetic toy: M1 k and M2 alpha_train chosen on validation only, then every ration is
    scored once by the public evaluator on an independent test stream (software self-check, not
    a research run)."""
    from ration_reliability.optimization.safety_margin import select_margin_on_validation

    pr = toy_problem
    th, d = pr.nominal_theta(), pr.dm_estimates()
    det = [i for i, g in enumerate(pr.ingredients) if not g.is_stochastic]
    sd, dsd = 0.05 * np.abs(th), 0.02 * d
    sd[det], dsd[det] = 0.0, 0.0
    model = IndependentNormalModel("syn_toy", pr.ingredient_ids, pr.nutrient_ids, th, sd, d, dsd, is_synthetic=True)
    s = RandomStreams(1103)
    opt, val, test = model.draw(s, "opt", 64), model.draw(s, "validation", 2000), model.draw(s, "test", 5000)
    sel1 = select_margin_on_validation(pr, val, grid=[0.0, 0.5, 1.0, 1.5], target_alpha=0.10,
                                       screening_rule="rate_upper_le_alpha", opt_draws=opt,
                                       params={"margin_scale": "sd", "apply_to_dm": True})  # declared (no defaults)
    sel2 = select_alpha_train_on_validation(pr, val, opt_draws=opt, grid=[0.0, 0.05, 0.10], target_alpha=0.10,
                                            screening_rule="rate_upper_le_alpha")
    rows = {"M0": M0(pr)}
    for name, sel in (("M1", sel1), ("M2", sel2)):
        assert sel.validation_stream_id == val.stream_id
        if sel.status == "selected":
            rows[name] = sel.selected_result
            assert test.stream_id not in sel.selected_result.streams_used
    for name, r in rows.items():
        ev = evaluate_drawset(r.decision, test, pr.compiled, prices=pr.prices)
        assert ev.is_synthetic and ev.draw_stream_id == test.stream_id and ev.structural_ok
        assert ev.cost == pytest.approx(r.objective)
