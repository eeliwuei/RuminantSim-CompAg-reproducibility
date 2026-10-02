"""M1 calibrated safety margin (contract T4 M1, section 9 M1): margin object, direction, analytic
cases, box worst-case equivalence, monotonicity, validation-only selection and invalid inputs.

All numbers are synthetic test values (``is_synthetic=True``); they test the code, not any
nutritional claim.
"""

import itertools

import numpy as np
import pytest

from engine_test_helpers import conc, dm_offer, ing, problem, supply, two_ingredient_problem
from ration_reliability.datamodel import NutrientSpec, SolveStatus, SolverOptions
from ration_reliability.errors import LeakageError
from ration_reliability.evaluation import evaluate, evaluate_drawset
from ration_reliability.evaluation.stats import one_sided_upper
from ration_reliability.nutrition import linear_rows
from ration_reliability.optimization import available_methods, get_method
from ration_reliability.optimization.lp_builder import optimization_indices
from ration_reliability.optimization.safety_margin import (
    DEVELOPMENT_GRID_RELATIVE,
    build_margin_rows,
    margin_spread,
    select_margin_on_validation,
    select_parameter_on_validation,
)
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams

M0 = get_method("M0_nominal")
M1 = get_method("M1_safety_margin")


#: M1 has no default scale / DM margin since the red-team fix D02: tests declare them explicitly
#: (these are the values that used to be the silent defaults).
SD = {"margin_scale": "sd", "apply_to_dm": True}


def _explicit(theta_sd, d_sd):
    return {"sd_source": "explicit", "theta_sd": np.asarray(theta_sd, float).tolist(),
            "d_sd": np.asarray(d_sd, float).tolist()}


def _toy_model(pr, cv=0.08, dm_cv=0.03, truncate=True):
    th, d = pr.nominal_theta(), pr.dm_estimates()
    sd = cv * np.abs(th)
    dsd = dm_cv * d
    det = [i for i, g in enumerate(pr.ingredients) if not g.is_stochastic]
    sd[det] = 0.0
    dsd[det] = 0.0
    return IndependentNormalModel("syn_toy_normal", pr.ingredient_ids, pr.nutrient_ids, th, sd, d, dsd,
                                  truncate=truncate, is_synthetic=True)


def test_registry_contains_m1_and_m2():
    ids = available_methods()
    for m in ("M1_safety_margin", "M2_joint_chance_saa", "M2b_marginal_bonferroni_saa"):
        assert m in ids


def test_development_grid_matches_contract():
    # contract T4 / configs/protocol.yaml methods.safety_margin_development_grid (relative scale)
    assert DEVELOPMENT_GRID_RELATIVE == (0.0, 0.025, 0.05, 0.075, 0.10)


def test_k0_equals_m0_and_evaluator_identical(toy_problem):
    pr = toy_problem
    r0 = M0(pr)
    I, J = len(pr.ingredient_ids), len(pr.nutrient_ids)
    r1 = M1(pr, params={"k": 0.0, **SD, **_explicit(np.full((I, J), 0.01), np.full(I, 0.01))})
    assert r0.status is SolveStatus.OPTIMAL and r1.status is SolveStatus.OPTIMAL
    assert np.allclose(r1.decision.q_as_fed, r0.decision.q_as_fed, atol=1e-9)
    assert r1.objective == pytest.approx(r0.objective, abs=1e-9)
    # same q -> identical public-evaluator output
    ds = _toy_model(pr).draw(RandomStreams(1103), "test", 2000)
    e0 = evaluate_drawset(r0.decision.q_as_fed, ds, pr.compiled, d_hat=pr.dm_estimates())
    e1 = evaluate_drawset(r0.decision.q_as_fed.copy(), ds, pr.compiled, d_hat=pr.dm_estimates())
    assert e0.q_hash == e1.q_hash and np.array_equal(e0.margin, e1.margin, equal_nan=True)
    # and the M1 ration scores like the M0 ration within LP tolerance
    e2 = evaluate_drawset(r1.decision, ds, pr.compiled)
    assert np.allclose(e2.margin, e0.margin, atol=1e-6, equal_nan=True)


def test_two_ingredient_analytic_content_margin():
    """F CP 10 % (sd 1), C CP 40 % (sd 2), fixed DM; CP >= 16 %.  k = 1.5 -> F 8.5 %, C 37 %."""
    pr = two_ingredient_problem(cp_min_pct=16.0)
    k = 1.5
    r = M1(pr, params={"k": k, **SD, **_explicit([[0.01], [0.02]], [0.0, 0.0])})
    assert r.status is SolveStatus.OPTIMAL
    cf, cc_ = 0.10 - k * 0.01, 0.40 - k * 0.02
    x_c = 20.0 * (0.16 - cf) / (cc_ - cf)
    x_f = 20.0 - x_c
    assert np.allclose(r.decision.x_planned_dm, [x_f, x_c], atol=1e-9)
    assert r.objective == pytest.approx(0.10 * x_f + 0.40 * x_c, abs=1e-9)
    assert r.params["k"] == k and r.diagnostics["tightened_constraint_ids"] == ["cp_min"]


def test_two_ingredient_analytic_dm_margin():
    """DM margin direction for a CP minimum: low-CP F is worst at high DM, high-CP C at low DM."""
    pr = two_ingredient_problem(cp_min_pct=16.0)
    k = 1.0
    sd_a = [[0.01], [0.02]]
    sd_d = [0.04, 0.02]
    r = M1(pr, params={"k": k, **SD, **_explicit(sd_a, sd_d)})
    assert r.status is SolveStatus.OPTIMAL
    cf, cc_ = 0.10 - 0.01, 0.40 - 0.02
    dF, dC = 0.40 + 0.04, 0.80 - 0.02
    # row: q_F dF (0.16 - cf) <= q_C dC (cc - 0.16), q = x / d_hat, x_F + x_C = 20
    a = (dF / 0.40) * (0.16 - cf)
    b = (dC / 0.80) * (cc_ - 0.16)
    x_c = 20.0 * a / (a + b)
    assert np.allclose(r.decision.x_planned_dm, [20.0 - x_c, x_c], atol=1e-9)
    mr = build_margin_rows(pr, k, np.array(sd_a), np.array(sd_d), apply_to_dm=True)
    assert np.allclose(mr.d_states["cp_min"], [dF, dC])
    assert np.allclose(mr.theta_states["cp_min"][:, 0], [cf, cc_])
    # apply_to_dm=False keeps d_hat
    mr2 = build_margin_rows(pr, k, np.array(sd_a), np.array(sd_d), apply_to_dm=False)
    assert np.allclose(mr2.d_states["cp_min"], [0.40, 0.80])


def _three(extra_cons, ndf=(0.55, 0.12, 0.14)):
    F = ing("F", 0.40, {"CP": 0.10, "NDF": ndf[0], "starch": 0.25}, forage=1.0)
    G = ing("G", 0.88, {"CP": 0.09, "NDF": ndf[1], "starch": 0.70})
    P = ing("P", 0.89, {"CP": 0.46, "NDF": ndf[2], "starch": 0.03})
    cons = [dm_offer(20.0)] + list(extra_cons)
    return problem([F, G, P], ["CP", "NDF", "starch"], cons, {"F": 0.03, "G": 0.20, "P": 0.40})


def test_direction_upper_and_mixed_sign_rows():
    cons = [conc("cp_max", {"CP": 1.0}, "le", 19.0),
            conc("st_vs_fndf", {"starch": 1.0, "G:forage:NDF": -2.0}, "le", -5.0),
            conc("ndf_rule", {"NDF": 1.0, "G:forage:NDF": 2.0}, "ge", 60.0),
            supply("cp_supply", {"CP": 1.0}, "ge", 3.0, "kg/d")]
    pr = _three(cons)
    mu = pr.nominal_theta()
    sp = np.full(mu.shape, 0.01)
    dsp = np.array([0.02, 0.01, 0.01])
    k = 2.0
    mr = build_margin_rows(pr, k, sp, dsp, apply_to_dm=True)
    j = {n: i for i, n in enumerate(pr.nutrient_ids)}
    # le on CP: content up for every ingredient; other nutrients untouched
    th = mr.theta_states["cp_max"]
    assert np.allclose(th[:, j["CP"]], mu[:, j["CP"]] + 0.02)
    assert np.allclose(th[:, j["NDF"]], mu[:, j["NDF"]]) and np.allclose(th[:, j["starch"]], mu[:, j["starch"]])
    # starch - 2 fNDF <= -5: starch up for all; forage NDF down (only forage F carries W != 0 on NDF)
    th = mr.theta_states["st_vs_fndf"]
    assert np.allclose(th[:, j["starch"]], mu[:, j["starch"]] + 0.02)
    assert th[0, j["NDF"]] == pytest.approx(mu[0, j["NDF"]] - 0.02)
    assert np.allclose(th[1:, j["NDF"]], mu[1:, j["NDF"]])
    # NDF + 2 fNDF >= 60: W = 3 for F, 1 for others -> all NDF down
    th = mr.theta_states["ndf_rule"]
    assert np.allclose(th[:, j["NDF"]], mu[:, j["NDF"]] - 0.02)
    # CP supply >= K: content down and DM down (positive contribution per kg DM)
    assert np.allclose(mr.theta_states["cp_supply"][:, j["CP"]], mu[:, j["CP"]] - 0.02)
    assert np.allclose(mr.d_states["cp_supply"], pr.dm_estimates() - k * dsp)
    # CP max concentration: an ingredient above 19 % after the margin (P) is worst at high DM,
    # the others (below 19 %) are worst at low DM
    d = mr.d_states["cp_max"]
    assert d[2] == pytest.approx(0.89 + 0.02) and d[0] == pytest.approx(0.40 - 0.04)


def test_relative_scale_states():
    pr = two_ingredient_problem(cp_min_pct=16.0)
    k = 0.10
    th, dsp, info = margin_spread(pr, margin_scale="relative")
    mr = build_margin_rows(pr, k, th, dsp, apply_to_dm=True)
    assert np.allclose(mr.theta_states["cp_min"][:, 0], [0.10 * 0.9, 0.40 * 0.9])
    assert np.allclose(mr.d_states["cp_min"], [0.40 * 1.1, 0.80 * 0.9])
    r = M1(pr, params={"k": k, "margin_scale": "relative", "apply_to_dm": True})
    assert r.status is SolveStatus.OPTIMAL and r.streams_used == ()


@pytest.mark.parametrize("seed", range(6))
def test_margin_rows_equal_box_worst_case(seed):
    """For any q >= 0 the M1 row equals the maximum of g over all vertices of the box
    {|a - mu| <= k sd, |d - d_hat| <= k sd_d} (brute force, independent of the sign logic)."""
    rng = np.random.default_rng(500 + seed)  # synthetic generator (test only)
    dm = rng.uniform(0.3, 0.85, 2)          # +/- k sd_d stays inside (0, 1]: no clipping
    a = rng.uniform(0.1, 0.5, (2, 2))       # - k sd stays >= 0: no clipping
    ings = [ing(f"i{k}", dm[k], {"CP": a[k, 0], "NDF": a[k, 1]}, forage=float(k == 0)) for k in range(2)]
    cons = [dm_offer(20.0), conc("cp", {"CP": 1.0}, "ge", 100 * rng.uniform(0.1, 0.4)),
            conc("mix", {"NDF": 1.0, "G:forage:NDF": -2.0, "CP": 0.5}, "le", 100 * rng.uniform(-0.3, 0.3)),
            supply("sup", {"NDF": 1.0, "DM": -0.2}, "ge", rng.uniform(0.0, 3.0), "kg/d")]
    pr = problem(ings, ["CP", "NDF"], cons, {"i0": 0.1, "i1": 0.3})
    sp = rng.uniform(0.005, 0.03, (2, 2))
    dsp = rng.uniform(0.01, 0.05, 2)
    k = float(rng.uniform(0.5, 2.0))
    mr = build_margin_rows(pr, k, sp, dsp, apply_to_dm=True)
    cc = pr.compiled.subset(optimization_indices(pr.compiled))
    dh = pr.dm_estimates()
    corners_a = list(itertools.product([-1, 1], repeat=4))
    corners_d = list(itertools.product([-1, 1], repeat=2))
    thetas = np.array([a + k * sp * np.array(s).reshape(2, 2) for s in corners_a])      # [16, 2, 2]
    ds = np.array([dh + k * dsp * np.array(s) for s in corners_d])                        # [4, 2]
    TH = np.repeat(thetas, len(ds), axis=0)
    DD = np.tile(ds, (len(thetas), 1))
    rows = linear_rows(cc, TH, DD, d_hat=dh)
    for _ in range(20):
        q = rng.uniform(0.0, 40.0, 2)
        g_box = rows.residual(q).max(axis=0)
        g_m1 = mr.A_q @ q - mr.b
        prob = [cc.constraint_ids.index(c) for c in mr.probabilistic_ids]
        assert np.allclose(g_m1[prob], g_box[prob], atol=1e-10)


def test_cost_monotone_in_k_and_infeasible_stays_infeasible(toy_problem):
    pr = toy_problem
    opt = _toy_model(pr).draw(RandomStreams(1103), "opt", 512)
    costs, statuses = [], []
    for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        r = M1(pr, opt_draws=opt, params={"k": k, **SD})
        statuses.append(r.status)
        costs.append(r.objective)
        assert r.streams_used == (opt.stream_id,)
    feas = [c for c, s in zip(costs, statuses) if s is SolveStatus.OPTIMAL]
    assert all(b >= a - 1e-9 for a, b in zip(feas, feas[1:]))
    first_inf = next((i for i, s in enumerate(statuses) if s is not SolveStatus.OPTIMAL), None)
    if first_inf is not None:
        assert all(s is SolveStatus.PROVEN_INFEASIBLE for s in statuses[first_inf:])
        assert all(c is None for c in costs[first_inf:])


def test_sd_from_opt_draws_close_to_declared():
    pr = two_ingredient_problem(cp_min_pct=16.0)
    sd_a, sd_d = np.array([[0.01], [0.02]]), np.array([0.0, 0.0])
    m = IndependentNormalModel("n", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), sd_a,
                               pr.dm_estimates(), sd_d, truncate=False, is_synthetic=True)
    opt = m.draw(RandomStreams(2207), "opt", 20000)
    th_sd, d_sd, info = margin_spread(pr, margin_scale="sd", opt_draws=opt)
    assert np.allclose(th_sd, sd_a, rtol=0.03) and np.all(d_sd == 0)
    assert info["n_opt_draws"] == 20000 and info["sd_ddof"] == 1
    r_draw = M1(pr, opt_draws=opt, params={"k": 1.0, **SD})
    r_expl = M1(pr, params={"k": 1.0, **SD, **_explicit(sd_a, sd_d)})
    assert r_draw.objective == pytest.approx(r_expl.objective, rel=5e-3)


def test_clipping_is_reported():
    pr = two_ingredient_problem(cp_min_pct=12.0)
    r = M1(pr, params={"k": 1.2, "margin_scale": "relative", "apply_to_dm": True})
    # mu (1 - 1.2) < 0 -> both CP cells clipped at 0; with zero CP both ingredients are below the
    # CP minimum, so both DM go up: F 0.4 * 2.2 = 0.88 (not clipped), C 0.8 * 2.2 = 1.76 -> clipped at 1
    assert r.diagnostics["n_theta_cells_clipped"] == 2
    assert r.diagnostics["n_dm_cells_clipped"] == 1
    assert r.status is SolveStatus.PROVEN_INFEASIBLE and r.decision is None


# ------------------------------------------------------------------------------------------------
# FINAL (round 3): the signed energy_density column is not clipped at 0 by the M1 margin
# ------------------------------------------------------------------------------------------------

def _signed_energy_problem():
    """SYNTHETIC numbers only.  F forage (CP 10 %, ash 8 %, EN 1.40), C concentrate (CP 40 %, ash 6 %, EN 1.90),
    M mineral-like (CP 0, ash 96 %, EN -0.35: a signed per-feed *contribution* that is negative, like the
    NEL_fixedDMI column of a mineral).  DM offer 20 kg; CP >= 16 %; ash <= 25 %; EN supply >= 30 Mcal/d;
    M >= 15 % of the planned DM (structural, forces the negative contributor into the ration)."""
    F = ing("F", 0.40, {"CP": 0.10, "ash": 0.08, "EN": 1.40}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40, "ash": 0.06, "EN": 1.90})
    M = ing("M", 0.90, {"CP": 0.0, "ash": 0.96, "EN": -0.35})
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0), conc("ash_max", {"ash": 1.0}, "le", 25.0),
            supply("en_min", {"EN": 1.0}, "ge", 30.0, "Mcal/d"),
            conc("m_min", {"DM:M": 1.0}, "ge", 15.0, cls="structural_hard", dm_source="decision_estimate")]
    return problem([F, C, M], ["CP", "ash", NutrientSpec("EN", "energy_density")], cons,
                   {"F": 0.04, "C": 0.32, "M": 0.10}, problem_id="syn_m1_signed_energy")


def test_energy_density_column_is_shifted_not_clipped_at_zero():
    """The ``ge`` energy supply row moves every EN contribution *down* by ``k |mu|``; the mineral's negative
    contribution becomes more negative and is not lifted to 0 (before the FINAL fix the clip at 0 made the
    mineral look energy-neutral at every k > 0: a row *weaker* than at k = 0).  Mass fractions still clip to
    [0, 1].  Expected values are hand arithmetic on the synthetic numbers."""
    pr = _signed_energy_problem()
    j = {n: i for i, n in enumerate(pr.nutrient_ids)}
    th_sp, d_sp, _ = margin_spread(pr, margin_scale="relative")
    k = 0.10
    mr0 = build_margin_rows(pr, 0.0, th_sp, d_sp, apply_to_dm=False)
    mr = build_margin_rows(pr, k, th_sp, d_sp, apply_to_dm=False)
    e0 = mr0.theta_states["en_min"][:, j["EN"]]
    e = mr.theta_states["en_min"][:, j["EN"]]
    assert np.allclose(e0, [1.40, 1.90, -0.35])
    # 1.40 * 0.9, 1.90 * 0.9, -0.35 - 0.1 * 0.35 = -0.385 (not 0)
    assert e == pytest.approx([1.26, 1.71, -0.385])
    assert e[2] < e0[2] < 0.0
    # q-space coefficient of the row A q <= b (supply, ge: A = -d e): mineral -0.90 * -0.385 = 0.3465 > 0.315 at k = 0
    kk = mr.constraint_ids.index("en_min")
    assert mr0.A_q[kk, 2] == pytest.approx(0.315)
    assert mr.A_q[kk, 2] == pytest.approx(0.3465)
    assert mr.A_q[kk, 2] > mr0.A_q[kk, 2]
    # mass fraction still clipped to [0, 1]: ash <= row, mineral 0.96 * 1.1 = 1.056 -> 1.0 (the only clipped cell)
    assert mr.theta_states["ash_max"][:, j["ash"]] == pytest.approx([0.088, 0.066, 1.0])
    assert mr.n_theta_clipped == 1
    # with the DM margin the negative contributor is worst at *high* DM (more DM = more negative supply):
    # d = 0.90 * 1.1 = 0.99 for M; the positive contributors go down (0.36, 0.72)
    mr_d = build_margin_rows(pr, k, th_sp, d_sp, apply_to_dm=True)
    assert mr_d.d_states["en_min"] == pytest.approx([0.36, 0.72, 0.99])
    assert mr_d.A_q[kk, 2] == pytest.approx(0.99 * 0.385)
    assert mr_d.n_theta_clipped == 1 and mr_d.n_d_clipped == 0
    r = M1(pr, params={"k": k, "margin_scale": "relative", "apply_to_dm": True})
    assert r.diagnostics["n_theta_cells_clipped"] == 1 and r.diagnostics["n_dm_cells_clipped"] == 0


@pytest.mark.parametrize("apply_to_dm", [True, False])
@pytest.mark.parametrize("k", [0.025, 0.05, 0.075, 0.10, 1.2])
def test_margin_rows_never_weaker_than_k0_with_signed_energy(k, apply_to_dm):
    """Every tightened coefficient is >= its k = 0 value (rows are A q <= b with b independent of k), for the
    development grid and a k > 1 that clips mass fractions at both ends."""
    pr = _signed_energy_problem()
    th_sp, d_sp, _ = margin_spread(pr, margin_scale="relative")
    A0 = build_margin_rows(pr, 0.0, th_sp, d_sp, apply_to_dm=apply_to_dm).A_q
    mr = build_margin_rows(pr, k, th_sp, d_sp, apply_to_dm=apply_to_dm)
    prob = [mr.constraint_ids.index(c) for c in mr.probabilistic_ids]
    assert np.all(mr.A_q[prob] >= A0[prob] - 1e-12)


def test_m1_ration_with_negative_energy_contributor_meets_nominal_rows():
    """k = 0.025, no DM margin.  Hand LP (x in kg DM, x_M = 3 forced, x_F + x_C = 17):
    k = 0: 1.40 x_F + 1.90 x_C - 0.35 * 3 >= 30 -> x_C >= 14.5.  M1 at k = 0.025 needs
    1.365 x_F + 1.8525 x_C - 0.35875 * 3 >= 30 -> x_C >= 16.15..., costlier than k = 0 and above the nominal row.
    With the old clip (mineral EN -> 0) the row was 1.365 x_F + 1.8525 x_C >= 30 -> x_C = 13.94...: cheaper than
    k = 0 and 0.28 Mcal/d short of the nominal energy row."""
    pr = _signed_energy_problem()
    j = {n: i for i, n in enumerate(pr.nutrient_ids)}
    r0 = M1(pr, params={"k": 0.0, "margin_scale": "relative", "apply_to_dm": False})
    r = M1(pr, params={"k": 0.025, "margin_scale": "relative", "apply_to_dm": False})
    assert r0.status is SolveStatus.OPTIMAL and r.status is SolveStatus.OPTIMAL
    assert r0.decision.x_planned_dm[1] == pytest.approx(14.5, abs=1e-6)
    x_c = (30.0 + 0.35875 * 3.0 - 1.365 * 17.0) / (1.8525 - 1.365)
    assert r.decision.x_planned_dm == pytest.approx([17.0 - x_c, x_c, 3.0], abs=1e-6)
    assert r.objective >= r0.objective - 1e-9
    en_nominal = float(r.decision.x_planned_dm @ pr.nominal_theta()[:, j["EN"]])
    assert en_nominal >= 30.0 - 1e-9


def _selection_setup():
    pr = _three([conc("cp_min", {"CP": 1.0}, "ge", 16.0), conc("ndf_max", {"NDF": 1.0}, "le", 40.0)])
    sd = np.array([[0.012, 0.03, 0.02], [0.008, 0.015, 0.03], [0.02, 0.015, 0.005]])
    m = IndependentNormalModel("n3", pr.ingredient_ids, pr.nutrient_ids, pr.nominal_theta(), sd, pr.dm_estimates(),
                               np.array([0.02, 0.01, 0.01]), truncate=False, is_synthetic=True)
    s = RandomStreams(3301)
    return pr, m, m.draw(s, "opt", 400), m.draw(s, "validation", 4000)


def test_selection_picks_lowest_cost_meeting_screen():
    pr, m, opt, val = _selection_setup()
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    sel = select_margin_on_validation(pr, val, grid=grid, target_alpha=0.05, screening_rule="rate_upper_le_alpha",
                                      params=SD, opt_draws=opt)
    assert sel.status == "selected"
    meeting = [c for c in sel.candidates if c.meets_screen]
    assert meeting and sel.selected_value == min(meeting, key=lambda c: (c.cost, c.rate_upper)).value
    # every grid point was solved and scored (no early stop), rates recorded
    assert [c.value for c in sel.candidates] == grid
    res = sel.selected_result
    sel_info = res.params["selection"]
    assert sel_info["candidate_grid"] == grid and sel_info["selected_on_stream"] == val.stream_id
    assert "rate_upper_le_alpha" in sel_info["criterion"] and sel_info["target_alpha"] == 0.05
    assert res.streams_used == (opt.stream_id, val.stream_id)
    # re-scoring the selected ration on the same validation draws reproduces the screen
    ev = evaluate_drawset(res.decision, val, pr.compiled, prices=pr.prices)
    assert ev.joint_violation.mean() <= 0.05 and ev.cost == pytest.approx(res.objective)
    # candidates with lower cost than the selected one all fail the screen
    for c in sel.candidates:
        if c.cost is not None and c.cost < res.objective - 1e-12:
            assert not c.meets_screen
    d = sel.to_dict()
    assert d["status"] == "selected" and len(d["candidates"]) == len(grid)


def test_selection_cp_rule_is_not_less_conservative():
    pr, m, opt, val = _selection_setup()
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    s_pt = select_margin_on_validation(pr, val, grid=grid, target_alpha=0.05, screening_rule="rate_upper_le_alpha",
                                       params=SD, opt_draws=opt)
    s_cp = select_margin_on_validation(pr, val, grid=grid, target_alpha=0.05, screening_rule="cp_upper_le_alpha",
                                       confidence=0.95, params=SD, opt_draws=opt)
    assert s_cp.status == "selected"
    assert s_cp.selected_result.objective >= s_pt.selected_result.objective - 1e-12
    for c in s_cp.candidates:
        if c.n_violated is not None:
            assert c.screen_statistic == pytest.approx(one_sided_upper(c.n_violated + c.n_unknown, c.n_draws, 0.95))


def test_selection_not_met_returns_no_ration():
    pr, m, opt, val = _selection_setup()
    sel = select_margin_on_validation(pr, val, grid=[0.0, 0.25], target_alpha=0.001,
                                      screening_rule="rate_upper_le_alpha", params=SD, opt_draws=opt)
    assert sel.status == "not_met" and sel.selected_value is None and sel.selected_result is None
    assert all(not c.meets_screen for c in sel.candidates)


def test_selection_and_method_never_accept_test_or_validation_as_opt():
    pr, m, opt, val = _selection_setup()
    test = m.draw(RandomStreams(3301), "test", 500)
    with pytest.raises(LeakageError):
        select_margin_on_validation(pr, test, grid=[0.0], target_alpha=0.05, screening_rule="rate_upper_le_alpha",
                                    params=SD, opt_draws=opt)
    with pytest.raises(LeakageError):  # validation draws passed as optimisation draws
        select_margin_on_validation(pr, val, grid=[1.0], target_alpha=0.05, screening_rule="rate_upper_le_alpha",
                                    params=SD, opt_draws=val)
    with pytest.raises(LeakageError):
        M1(pr, opt_draws=test, params={"k": 1.0, **SD})
    with pytest.raises(LeakageError):  # even when the mode would not use the draws
        M1(pr, opt_draws=test, params={"k": 0.1, "margin_scale": "relative", "apply_to_dm": True})


def test_selection_rejects_undeclared_rules():
    pr, m, opt, val = _selection_setup()
    with pytest.raises(ValueError):
        select_margin_on_validation(pr, val, grid=[0.0, 1.0], target_alpha=0.05, screening_rule="best_on_test",
                                    params=SD, opt_draws=opt)
    with pytest.raises(ValueError):
        select_margin_on_validation(pr, val, grid=[0.0, 1.0], target_alpha=0.05, screening_rule="cp_upper_le_alpha",
                                    params=SD, opt_draws=opt)  # confidence missing
    with pytest.raises(ValueError):
        select_margin_on_validation(pr, val, grid=[0.0, 1.0], target_alpha=0.05, screening_rule="rate_upper_le_alpha",
                                    params={"k": 1.0, **SD}, opt_draws=opt)


def test_generic_selection_with_m0_grid_of_modes():
    """The generic routine works for any registered method (here: M0's coefficient_mode)."""
    pr, m, opt, val = _selection_setup()
    sel = select_parameter_on_validation(M0, pr, val, param_name="coefficient_mode",
                                         grid=["nominal_point", "draw_mean"], target_alpha=0.9,
                                         screening_rule="rate_upper_le_alpha", opt_draws=opt)
    assert sel.status == "selected" and sel.method_id == "M0_nominal"


def test_invalid_inputs_are_reported_not_repaired():
    pr = two_ingredient_problem()
    ok = {**SD, **_explicit([[0.01], [0.02]], [0.0, 0.0])}
    assert M1(pr, params={"k": 1.0, **ok}).status is SolveStatus.OPTIMAL               # the reference call is valid
    assert M1(pr, params=ok).status is SolveStatus.INVALID_INPUT                       # k missing
    assert M1(pr, params={"k": -0.1, **ok}).status is SolveStatus.INVALID_INPUT
    assert M1(pr, params={"k": float("nan"), **ok}).status is SolveStatus.INVALID_INPUT
    assert M1(pr, params={"k": 1.0, "alpha": 0.05, **ok}).status is SolveStatus.INVALID_INPUT
    assert M1(pr, params={"k": 1.0, **SD, "margin_scale": "percent"}).status is SolveStatus.INVALID_INPUT
    assert M1(pr, params={"k": 1.0, **SD}).status is SolveStatus.INVALID_INPUT         # no opt draws for sd
    assert M1(pr, params={"k": 1.0, **SD, "sd_source": "explicit"}).status is SolveStatus.INVALID_INPUT
    bad_shape = {**SD, "sd_source": "explicit", "theta_sd": [[0.01, 0.0]], "d_sd": [0.0, 0.0]}
    assert M1(pr, params={"k": 1.0, **bad_shape}).status is SolveStatus.INVALID_INPUT
    r = M1(pr, params={"k": 1.0, **SD, **_explicit([[np.nan], [0.02]], [0.0, 0.0])})
    assert r.status is SolveStatus.INVALID_INPUT and "cp_min:F:CP" in r.message and r.decision is None
    r = M1(pr, params={"k": 1.0, **SD, **_explicit([[0.01], [0.02]], [np.nan, 0.0])})
    assert r.status is SolveStatus.INVALID_INPUT and "cp_min:F:DM" in r.message
    # k = 0 needs no spread at all (NaN spread is irrelevant)
    r = M1(pr, params={"k": 0.0, **SD, **_explicit([[np.nan], [np.nan]], [np.nan, np.nan])})
    assert r.status is SolveStatus.OPTIMAL
    assert M1(pr, params={**ok, "k": 1.0, "margin_scale": "relative"}).status is SolveStatus.INVALID_INPUT
    assert M1(pr, d_hat=np.array([0.4, 1.2]), params={"k": 0.0, **ok}).status is SolveStatus.INVALID_INPUT


def test_infeasible_margin_is_proven_infeasible_not_relaxed():
    pr = two_ingredient_problem(cp_min_pct=39.0)
    r = M1(pr, params={"k": 1.0, **SD, **_explicit([[0.01], [0.02]], [0.0, 0.0])})  # C at 38 % < 39 %
    assert r.status is SolveStatus.PROVEN_INFEASIBLE and r.decision is None and r.objective is None
    assert r.diagnostics["imposed_constraint_ids"] == ["dm_offer", "cp_min"]


def test_structural_rows_unchanged_and_diagnostic_not_imposed(toy_problem):
    pr = toy_problem
    I, J = len(pr.ingredient_ids), len(pr.nutrient_ids)
    r = M1(pr, params={"k": 0.5, **SD, **_explicit(np.full((I, J), 0.001), np.full(I, 0.001))},
           solver_options=SolverOptions())
    assert r.status is SolveStatus.OPTIMAL
    assert "energy_supply_diag" not in r.diagnostics["imposed_constraint_ids"]
    ev = evaluate(r.decision, pr.nominal_theta(), pr.dm_estimates(), pr.compiled)
    assert ev.structural_ok and r.decision.x_planned_dm.sum() == pytest.approx(20.0, abs=1e-6)
