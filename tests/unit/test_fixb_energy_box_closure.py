"""Review round 2, red team B (FIX_B): M3 boxes with a signed energy column, the energy box of a factory
world, the composition-closure diagnostic and the corrected conservativeness statement of the NEL row.

Every number here is SYNTHETIC (invented for the arithmetic); expected values are computed by hand in the
comments or by brute force, not by the code under test.  Nothing here says anything about any feed, cow
or nutrient requirement.

Findings covered:

* B-K7-1 / red team B #7: ``BoxUncertaintySet.from_opt_draws`` clipped every column at 0, so the
  negative ``NEL_fixedDMI`` contribution of a mineral was cut: M3a silently used an optimistic box and
  M3b returned ``invalid_input`` (nominal point outside the box).  Now the M3 methods pass the problem's
  nutrient dimensions (energy columns are not clipped) and any clip that would cut the sampled draws is
  refused.
* red team B #2: composition closure (CP + NDF + starch + EE + ash > 100 % DM, Eq 3-1 ROM < 0) is counted,
  never clipped.
* red team B #3: the linear NEL row is not a draw-wise lower bound of the Chapter 3 chain even with diet
  starch <= S_ref and D = DMI (drawn NDF below the feed mean).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ration_reliability.datamodel import NutrientSpec, SolveStatus
from ration_reliability.nutrition import energy as E
from ration_reliability.optimization.robust import (
    BoxUncertaintySet,
    solve_box_robust,
    solve_budget_robust,
)
from ration_reliability.uncertainty import UncertaintySpec, build_uncertainty_model
from ration_reliability.uncertainty.base import DrawSet

from engine_test_helpers import conc, dm_offer, ing, problem, supply

# ------------------------------------------------------------------------------------------------
# 1  M3 box on opt draws with a signed energy column (B-K7-1)
# ------------------------------------------------------------------------------------------------
NUTS = ("CP", "EN")                     # EN: synthetic energy_density column (Mcal/kg DM)


def _energy_problem():
    # F forage (CP 10 %, EN 1.40), C concentrate (CP 40 %, EN 1.90), M mineral (CP 0, EN -0.35: a per-feed
    # *contribution* that is negative, like NEL_fixedDMI of a mineral).  DM offer 20 kg; CP >= 16 %; EN supply >= 30.
    F = ing("F", 0.40, {"CP": 0.10, "EN": 1.40}, forage=1.0)
    C = ing("C", 0.80, {"CP": 0.40, "EN": 1.90})
    M = ing("M", 1.00, {"CP": 0.0, "EN": -0.35}, stochastic=False)
    cons = [dm_offer(20.0), conc("cp_min", {"CP": 1.0}, "ge", 16.0),
            supply("en_min", {"EN": 1.0}, "ge", 30.0, "Mcal/d"),
            conc("m_min", {"DM:M": 1.0}, "ge", 1.0, cls="structural_hard", dm_source="decision_estimate")]
    return problem([F, C, M], ["CP", NutrientSpec("EN", "energy_density")], cons,
                   {"F": 0.04, "C": 0.32, "M": 0.10}, problem_id="syn_fixb_energy_box")


def _opt_draws(pr, n=64, seed=11):
    """Hand-made opt draws: +/-5 % multiplicative noise on F and C, the mineral fixed (negative EN)."""
    rng = np.random.default_rng(seed)
    th0, d0 = pr.nominal_theta(), pr.dm_estimates()
    th = np.repeat(th0[None], n, axis=0)
    d = np.repeat(d0[None], n, axis=0)
    th[:, :2, :] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n, 2, 2))
    d[:, :2] *= 1.0 + rng.uniform(-0.02, 0.02, size=(n, 2))
    return DrawSet(th, d, "opt", "root=11/opt", "syn_fixb_draws", "fp-syn-fixb", pr.ingredient_ids, NUTS, True)


def test_opt_draw_box_keeps_negative_energy_column_and_refuses_silent_clip():
    pr = _energy_problem()
    opt = _opt_draws(pr)
    # without dimension information the old uniform clip at 0 would cut the mineral's EN draws -> refused
    with pytest.raises(ValueError, match="EN"):
        BoxUncertaintySet.from_opt_draws(opt, k=1.0)
    with pytest.raises(ValueError, match="EN"):
        BoxUncertaintySet.from_opt_draws(opt, quantiles=(0.05, 0.95))
    dims = {n.nutrient_id: n.dimension for n in pr.nutrients}
    box = BoxUncertaintySet.from_opt_draws(opt, k=1.0, column_dimensions=dims)
    j = NUTS.index("EN")
    assert box.theta_lo[2, j] == pytest.approx(-0.35, abs=1e-15)       # SD 0: interval is the point -0.35
    assert box.theta_hi[2, j] == pytest.approx(-0.35, abs=1e-15)
    assert box.construction_params["column_clip"] == {"CP": [0.0, None], "EN": [None, None]}
    assert box.construction_params["column_clip_basis"] == "per_column_by_dimension"
    th0, d0 = pr.nominal_theta(), pr.dm_estimates()
    mu = opt.theta.mean(axis=0)
    sd = opt.theta.std(axis=0, ddof=1)
    assert np.allclose(box.theta_lo, np.where(np.array([True, False])[None, :], np.maximum(mu - sd, 0.0), mu - sd))
    # the quantile variant with dimensions keeps the negative column too
    bq = BoxUncertaintySet.from_opt_draws(opt, quantiles=(0.05, 0.95), column_dimensions=dims)
    assert bq.theta_lo[2, j] < 0 and bq.theta_hi[2, j] < 0
    # mass-fraction behaviour unchanged: a CP box that would go below 0 is clipped at 0 (physical range)
    wide = BoxUncertaintySet.from_mean_sd("w", pr.ingredient_ids, NUTS, th0, np.abs(th0) * 5.0, d0, d0 * 0.0, 1.0,
                                          column_dimensions=dims)
    assert np.all(wide.theta_lo[:, 0] >= 0.0) and wide.theta_lo[2, j] == pytest.approx(-0.35 - 1.75)
    # a negative mean with the uniform clip is refused (the centre would lie outside its own box)
    with pytest.raises(ValueError, match="EN"):
        BoxUncertaintySet.from_mean_sd("x", pr.ingredient_ids, NUTS, th0, th0 * 0.0, d0, d0 * 0.0, 1.0)


def test_m3a_m3b_with_k_use_the_unclipped_energy_box():
    pr = _energy_problem()
    opt = _opt_draws(pr)
    dims = {n.nutrient_id: n.dimension for n in pr.nutrients}
    ra = solve_box_robust(pr, opt_draws=opt, params={"k": 1.0})
    assert ra.status is SolveStatus.OPTIMAL, ra.message
    used = ra.diagnostics["uncertainty_set"]["construction_params"]
    assert used["column_clip"]["EN"] == [None, None]
    # identical to the explicitly built per-column box: the method path does what the constructor does
    rb = solve_box_robust(pr, params={"uncertainty_set": BoxUncertaintySet.from_opt_draws(opt, k=1.0,
                                                                                         column_dimensions=dims)})
    assert ra.objective == pytest.approx(rb.objective, abs=1e-12)
    # the pre-fix behaviour, rebuilt by hand: the same box with the mineral's EN interval clipped at 0.  The
    # mineral then looks energy-neutral instead of diluting, so the box is optimistic: never more expensive
    good = BoxUncertaintySet.from_opt_draws(opt, k=1.0, column_dimensions=dims)
    clipped = BoxUncertaintySet.from_bounds("prefix_clip_at_0", pr.ingredient_ids, NUTS,
                                            np.maximum(good.theta_lo, 0.0), np.maximum(good.theta_hi, 0.0),
                                            good.d_lo, good.d_hi, source_note="synthetic: pre-fix clip rebuilt")
    assert clipped.theta_lo[2, NUTS.index("EN")] == 0.0 and good.theta_lo[2, NUTS.index("EN")] < 0.0
    rc = solve_box_robust(pr, params={"uncertainty_set": clipped})
    assert rc.status is SolveStatus.OPTIMAL
    assert rc.objective <= ra.objective + 1e-12
    # M3b with k: the nominal point lies in the box now, so Gamma = 0 reproduces the nominal row (no invalid_input)
    r0 = solve_budget_robust(pr, opt_draws=opt, params={"k": 1.0, "gamma": 0})
    assert r0.status is SolveStatus.OPTIMAL, r0.message
    rf = solve_budget_robust(pr, opt_draws=opt, params={"k": 1.0, "gamma": "full"})
    assert rf.status is SolveStatus.OPTIMAL and rf.objective == pytest.approx(ra.objective, rel=1e-9)


# ------------------------------------------------------------------------------------------------
# 2  energy box of a factory world (exact affine image, never clipped)
# ------------------------------------------------------------------------------------------------
F1 = E.FeedEnergyInputs("syn_forage", ndf=40.0, lignin=4.0, starch=20.0, fa=3.0, cp=10.0, ash=6.0,
                        rup_pct_cp=30.0, drup_pct_rup=70.0, dstarch_base=0.88, dfa=0.73)
G1 = E.FeedEnergyInputs("syn_grain", ndf=10.0, lignin=1.0, starch=70.0, fa=4.0, cp=9.0, ash=1.5,
                        rup_pct_cp=45.0, drup_pct_rup=75.0, dstarch_base=0.88, dfa=0.73)
M1 = E.FeedEnergyInputs("syn_mineral", ndf=0.0, lignin=0.0, starch=0.0, fa=0.0, cp=0.0, ash=100.0,
                        rup_pct_cp=0.0, drup_pct_rup=0.0, dstarch_base=0.95, dfa=0.73)
FEEDS = (F1, G1, M1)
SET = E.FixedDMISettings(dmi_kg_d=20.0, body_weight_kg=600.0, starch_ref_pct=24.0, milk_cp_kg_d=1.0,
                         body_gain_cp_kg_d=0.0)
NMAP = {"NDF": "ndf", "starch": "starch", "CP": "cp", "ash": "ash"}
BASE_NUTS = ("CP", "NDF", "starch", "EE", "ash")
RULE = {"primary_family": "TN_MM", "fallback_families": ["BETA_MM"], "on_exhausted": "error",
        "status": "synthetic_test_only", "rationale": "synthetic unit test", "selection_basis": "declared_rule"}
DEFAULTS = {"variance_basis": "true_batch_state", "data_fingerprint": "synthetic:fixb", "decomposition_id": "none",
            "decomposition_source": None, "measurement_model_id": None, "provenance_status": "synthetic_test_only",
            "source_id": None, "locator": None}


def _feed_means():
    fld = {"CP": "cp", "NDF": "ndf", "starch": "starch", "EE": "fa", "ash": "ash"}
    return np.array([[getattr(f, fld[n]) / 100.0 for n in BASE_NUTS] for f in FEEDS])


def _factory_world():
    tm = _feed_means()
    ts = np.where(np.arange(3)[:, None] < 2, np.abs(tm) * 0.08, 0.0)
    ts[:, 3] = np.where(np.arange(3) < 2, 0.004, 0.0)          # EE SD
    dm = np.array([0.35, 0.88, 1.0])
    ds = np.array([0.02, 0.01, 0.0])
    spec = UncertaintySpec.from_arrays("syn_fixb_world", [f.ingredient_id for f in FEEDS], list(BASE_NUTS), tm, ts,
                                       dm, ds, purpose="unit_test", moment_semantics="target_marginal_moments",
                                       is_synthetic=True, family_rule=RULE, cell_defaults=DEFAULTS,
                                       theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0))
    return build_uncertainty_model(spec)


def test_energy_box_is_the_exact_affine_image_and_is_not_clipped():
    fm = _factory_world()
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    k = 2.0
    box = E.energy_box_from_factory_model(fm, lin, k)
    assert box.nutrient_ids == BASE_NUTS + (E.ENERGY_COLUMN_ID,)
    base = BoxUncertaintySet.from_factory_model(fm, k)
    np.testing.assert_array_equal(box.theta_lo[:, :-1], base.theta_lo)
    np.testing.assert_array_equal(box.d_lo, base.d_lo)
    # brute force: evaluate the affine map at every corner of each feed's composition box
    idx = [BASE_NUTS.index(n) for n in lin.nutrient_ids]
    for i in range(3):
        corners = list(itertools.product(*[(base.theta_lo[i, j], base.theta_hi[i, j]) for j in idx]))
        vals = [lin.intercept[i] + float(np.dot(lin.slopes[i], c)) for c in corners]
        assert box.theta_lo[i, -1] == pytest.approx(min(vals), abs=1e-12)
        assert box.theta_hi[i, -1] == pytest.approx(max(vals), abs=1e-12)
    # the mineral's contribution is negative and stays negative (point value: lo == hi == nominal density)
    assert box.theta_hi[2, -1] < 0
    assert box.theta_lo[2, -1] == pytest.approx(lin.nominal_density[2], abs=1e-12)
    # the nominal state lies in the box, energy column included
    w = E.EnergyColumnModel(fm, lin)
    th0, d0 = w.nominal_state()
    assert box.contains(th0, d0)
    # the wrapper route gives the same box; a different linearisation is refused
    b2 = E.energy_box_from_factory_model(w, None, k)
    np.testing.assert_array_equal(b2.theta_lo, box.theta_lo)
    np.testing.assert_array_equal(b2.theta_hi, box.theta_hi)
    lin2 = E.linearise_nel_fixed_dmi(FEEDS, E.FixedDMISettings(20.0, 600.0, 30.0, 1.0, 0.0), nutrient_map=NMAP)
    with pytest.raises(ValueError):
        E.energy_box_from_factory_model(w, lin2, k)
    with pytest.raises(ValueError):
        E.energy_box_from_factory_model(fm, None, k)


# ------------------------------------------------------------------------------------------------
# 3  composition closure: counted, never clipped
# ------------------------------------------------------------------------------------------------

def test_composition_closure_report_counts_by_hand():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    ids = [f.ingredient_id for f in FEEDS]
    tm = _feed_means()                                           # forage sum 0.79, grain 0.945, mineral 1.00
    th = np.repeat(tm[None], 4, axis=0).copy()
    th[1, 0, BASE_NUTS.index("NDF")] = 0.62                      # forage sum 1.01 > 1
    th[2, 1, BASE_NUTS.index("starch")] = 0.75                   # grain sum 0.995 (< 1): EE 4 % not in ROM
    th[3, 1, BASE_NUTS.index("starch")] = 0.79                   # grain sum 1.035 > 1
    rep = E.composition_closure_report(th, BASE_NUTS, ids, lin=lin)
    assert rep["syn_forage"]["share_sum_gt_100pct"] == pytest.approx(0.25)
    assert rep["syn_grain"]["share_sum_gt_100pct"] == pytest.approx(0.25)
    assert rep["syn_mineral"]["share_sum_gt_100pct"] == 0.0      # exactly 100 % is not > 100 %
    # Eq 3-1 ROM = 100 - ash - NDF - starch - FA/1.06 - CP (FA held at the feed value)
    rom_f = 100 - 6.0 - 62.0 - 20.0 - 3.0 / 1.06 - 10.0          # = -0.830...
    rom_g3 = 100 - 1.5 - 10.0 - 75.0 - 4.0 / 1.06 - 9.0          # = 0.726... (>= 0)
    rom_g4 = 100 - 1.5 - 10.0 - 79.0 - 4.0 / 1.06 - 9.0          # = -3.27...
    assert rep["syn_forage"]["min_rom_pct"] == pytest.approx(rom_f, abs=1e-9)
    assert rep["syn_forage"]["share_rom_lt_0"] == pytest.approx(0.25)
    assert rom_g3 > 0 and rep["syn_grain"]["min_rom_pct"] == pytest.approx(rom_g4, abs=1e-9)
    assert rep["syn_grain"]["share_rom_lt_0"] == pytest.approx(0.25)
    assert rep["syn_mineral"]["share_rom_lt_0"] == 0.0
    # nothing was modified in place and NaN cells are left out of the counts
    assert th[1, 0, BASE_NUTS.index("NDF")] == 0.62
    th2 = th.copy()
    th2[0, 0, 0] = np.nan
    assert E.composition_closure_report(th2, BASE_NUTS, ids)["syn_forage"]["n_draws"] == 3
    assert E.composition_closure_report(th2, BASE_NUTS, ids)["syn_forage"]["share_rom_lt_0"] is None   # no lin


# ------------------------------------------------------------------------------------------------
# 4  the NEL row is exact at its reference point but not a draw-wise bound (red team B #3)
# ------------------------------------------------------------------------------------------------

def test_linear_row_can_exceed_chain_with_starch_at_or_below_sref_when_ndf_is_drawn_low():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    # D = 20 = DMI_scn; mineral 0.2 kg, grain chosen so that diet starch = S_ref = 24 % at the means:
    # starch = (x_F*20 + x_G*70)/20 = 24 with x_F + x_G = 19.8  ->  x_G = (480 - 396)/50 = 1.68
    x = np.array([19.8 - 1.68, 1.68, 0.2])
    th_mean = _feed_means()[:, [BASE_NUTS.index(n) for n in lin.nutrient_ids]]
    nl0 = E.nonlinear_diet_nel(x, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
    assert nl0["starch_pct"] == pytest.approx(24.0, abs=1e-12) and nl0["DMI"] == pytest.approx(20.0, abs=1e-12)
    assert float(lin.supply(x, th_mean, lin.nutrient_ids)) == pytest.approx(nl0["NEL"], abs=1e-9)   # exact at the means
    # draw the forage NDF 5 points below its mean (lignin fixed): starch unchanged (= S_ref), D unchanged (= DMI)
    th = th_mean.copy()
    th[0, list(lin.nutrient_ids).index("NDF")] -= 0.05
    comps = [{"ndf": 35.0}, {}, {}]
    nl = E.nonlinear_diet_nel(x, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0,
                              compositions=comps)
    assert nl["starch_pct"] == pytest.approx(24.0, abs=1e-12) and nl["DMI"] == pytest.approx(20.0, abs=1e-12)
    over = float(lin.supply(x, th, lin.nutrient_ids)) - nl["NEL"]
    assert over > 1e-4          # the linear row lies ABOVE the chain although starch <= S_ref and D = DMI
    # and it is the Eq 3-3a term: the chain's dNDF_base at NDF 35 (lignin 4) is below the one at NDF 40
    assert E.dndf_base_lignin_eq3_3a(35.0, 4.0) < E.dndf_base_lignin_eq3_3a(40.0, 4.0)
