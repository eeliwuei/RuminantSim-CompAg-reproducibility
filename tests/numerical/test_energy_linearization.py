"""Fixed-DMI NEL linearisation (NASEM 2021 ch.3) and the configured requirement layer -- hand-computed checks.

Scope (review round 2, R4 items 2 and 4; K4_nutrition_case):

* ``nutrition.energy``: Eq 3-3a, Eq 3-1, Eq 3-8 (Table 19-1 "DE base" convention), the per-feed
  fixed-DMI terms (Eq 3-5a/3-5b, 3-6a, 3-9, 3-10a/b, 3-11, 3-12), the requirement (Eq 3-13 ... 3-20),
  exactness of the linear row against the full whole-diet chain at its reference point, the declared
  conservative direction (diet starch <= S_ref), exact affine slopes, the engine glue (energy column,
  constraint spec, public evaluator) and the draw wrappers.
* ``nutrition.requirements`` calling layer: ``requirement_version_from_config``,
  ``requirements_from_config`` and ``mineral_constraint_specs`` (no defaults; configuration vs code
  mismatch raises).
* K4c (2026-09-25): ``resolve_value_refs`` and the tracked ``configs/dev_case_v1`` specification --
  derived values, constraint rows, energy bound and price conversions recomputed from the tracked files
  only (``data/restricted_local`` is never opened; restricted bounds are checked for form).

All feed and cow numbers here are SYNTHETIC (invented for the arithmetic), except the FIX5 working point
WP-M whose hand values are copied from ``reports/model_audit.md`` section 8.5.  The expected values were
computed by separate hand arithmetic (written out in the comments), not by the code under test.  No
restricted data are read.  Passing says the equations are computed as transcribed -- nothing about
nutritional adequacy (contract section 8).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ration_reliability.datamodel import (
    ConstraintClass,
    ConstraintKind,
    Provenance,
    Sense,
    ValueStatus,
)
from ration_reliability.errors import InvalidProblemError, UnitError
from ration_reliability.evaluation import evaluate
from ration_reliability.io.config import PendingRelabelError
from ration_reliability.nutrition import energy as E
from ration_reliability.nutrition import requirements as R
from ration_reliability.nutrition.requirements import Qty
from ration_reliability.uncertainty import PointMassModel, RandomStreams, ScenarioSetModel

from engine_test_helpers import dm_offer, ing, problem

TOL = 1e-9
SYN = Provenance(ValueStatus.SYNTHETIC_TEST_ONLY, source_id="SYN-K4-ENERGY-TEST")

# ------------------------------------------------------------------------------------------------
# synthetic feeds (all % of DM) -- invented values.  K4c: the base starch digestibilities were 0.89/0.91, which are
# NASEM Table 3-1 entries; they are now 0.88/0.95 (not table values) so no restricted value sits unlabelled in a
# tracked test (same concern as B-318 item 3); every expected value below was recomputed by separate hand arithmetic.
# ------------------------------------------------------------------------------------------------
F = E.FeedEnergyInputs("syn_forage", ndf=40.0, lignin=4.0, starch=20.0, fa=3.0, cp=10.0, ash=6.0,
                       rup_pct_cp=30.0, drup_pct_rup=70.0, dstarch_base=0.88, dfa=0.73)
G = E.FeedEnergyInputs("syn_grain", ndf=10.0, lignin=1.0, starch=70.0, fa=4.0, cp=9.0, ash=1.5,
                       rup_pct_cp=45.0, drup_pct_rup=75.0, dstarch_base=0.88, dfa=0.73)
P = E.FeedEnergyInputs("syn_protein", ndf=12.0, lignin=1.0, starch=2.0, fa=1.5, cp=50.0, ash=7.0,
                       rup_pct_cp=35.0, drup_pct_rup=90.0, dstarch_base=0.95, dfa=0.73)
M = E.FeedEnergyInputs("syn_mineral", ndf=0.0, lignin=0.0, starch=0.0, fa=0.0, cp=0.0, ash=100.0,
                       rup_pct_cp=0.0, drup_pct_rup=0.0, dstarch_base=0.95, dfa=0.73)
FEEDS = (F, G, P, M)
# fixed-DMI settings: DMI 20 kg/d, BW 600 kg (DMI/BW = 1/30), S_ref 24 % DM, milk CP 1.0 kg/d
SET = E.FixedDMISettings(dmi_kg_d=20.0, body_weight_kg=600.0, starch_ref_pct=24.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
NMAP = {"NDF": "ndf", "starch": "starch", "CP": "cp", "ash": "ash"}
NUT = ("NDF", "starch", "CP", "ash")


def _theta_means(feeds=FEEDS, nut=NUT):
    fld = {"NDF": "ndf", "starch": "starch", "CP": "cp", "ash": "ash"}
    return np.array([[getattr(f, fld[n]) / 100.0 for n in nut] for f in feeds])


# ------------------------------------------------------------------------------------------------
# single-equation hand checks
# ------------------------------------------------------------------------------------------------

def test_eq3_3a_lignin_ndf_digestibility_hand():
    # 0.75 * (40 - 4) * (1 - 0.1 ** 0.667) / 40 ; 0.1 ** 0.667 = 0.2152781...  -> 0.5296872329061049
    assert E.dndf_base_lignin_eq3_3a(40.0, 4.0) == pytest.approx(0.5296872329061049, abs=1e-12)
    assert E.dndf_base_lignin_eq3_3a(0.0, 0.0) == 0.0            # mineral: no NDF, no digested NDF
    with pytest.raises(ValueError):
        E.dndf_base_lignin_eq3_3a(5.0, 6.0)                     # lignin > NDF is refused, not clipped


def test_eq3_1_rom_hand():
    # 100 - 6 - 40 - 20 - 3/1.06 - 10 = 21.169811320754718
    assert E.rom_eq3_1(ndf=40, starch=20, fa=3, cp=10, ash=6) == pytest.approx(21.169811320754718, abs=1e-12)
    assert F.digested_cp_fraction == pytest.approx(0.91, abs=1e-15)          # 0.7 + 0.3 * 0.7 (Eq 3-4)


def test_eq3_8_de_base_table_convention_hand():
    # 0.042*40*0.5296872 + 0.0423*20*0.88 + 0.094*3*0.73 + 0.0565*10*0.91 + 0.040*21.1698113*0.96
    #   - 0.00565*(11.62 + 0.134*30) - 0.00565*16.5 - 0.0040*34.3            = 2.848494305999237
    r = E.feed_de_base_eq3_8(F)
    assert r.value == pytest.approx(2.848494305999237, abs=1e-12)
    assert r.unit == "Mcal/kg"
    assert {"Eq 3-8", "Eq 3-3a", "Table 3-1"} <= {x.equation for x in r.equations}
    assert all(x.locator.endswith(f"(pdf p.{x.printed_page + 20})") for x in r.equations)


def test_fixed_dmi_feed_terms_hand():
    # DMI/BW = 20/600; dNDF = 0.5296872 - 0.0059*(24 - 26) - 1.1*(1/30 - 0.035) = 0.5433205662394383
    # dSt = 0.88 - 1.0*(1/30 - 0.035) = 0.8816666666666667 ; MFCP = 11.62 + 0.134*40 = 16.98 g/kg
    # DE   = 0.042*40*dNDF + 0.0423*20*dSt + 0.20586 + 0.51415 + 0.8129207547 - 0.00565*16.98 - 0.093225 - 0.1372
    # GasE = 0.294 - 0.347*3/20 + 0.0409*40*dNDF/20 ; UE = 0.0146*160*(0.091 - 0.0165 - 0.01698)
    t = E.feed_energy_terms(F, SET)
    assert t["dNDF"] == pytest.approx(0.5433205662394383, abs=1e-12)
    assert t["dSt"] == pytest.approx(0.8816666666666667, abs=1e-12)
    assert t["DE"] == pytest.approx(2.8652373059992375, abs=1e-12)
    assert t["GasE"] == pytest.approx(0.28639362231838605, abs=1e-12)
    assert t["UE"] == pytest.approx(0.13436671999999997, abs=1e-12)
    assert t["NEL"] == pytest.approx(1.613354796029362, abs=1e-12)           # 0.66 * (DE - GasE - UE)
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    assert lin.constant_mcal_d == pytest.approx(1.54176, abs=1e-12)          # 0.66*0.0146*160*1.0 (C0)
    assert lin.nominal_density[0] == pytest.approx(1.613354796029362, abs=1e-12)


def test_mineral_carries_per_kg_dmi_losses():
    """A pure mineral (ash 100 %) has no digestible fraction but, being DMI, carries the per-kg-DMI
    endogenous losses and methane (Eq 3-8, 3-9): e = 0.66*[-(0.00565*11.62 + 0.093225 + 0.1372) - 0.294
    - 0.0146*160*(-0.0165 - 0.01162)]."""
    expect = 0.66 * (-(0.00565 * 11.62 + 0.00565 * 16.5 + 0.0040 * 34.3) - 0.294
                     - 0.0146 * 160 * (-0.0165 - 0.01162))
    assert E.feed_energy_terms(M, SET)["NEL"] == pytest.approx(expect, abs=1e-12)


# ------------------------------------------------------------------------------------------------
# linear row vs the full whole-diet chain
# ------------------------------------------------------------------------------------------------

def _x(xg):
    """Synthetic diet with D = 20 kg DM (= DMI_scn): mineral 0.2, protein 3.0, grain xg, forage the rest."""
    return np.array([20.0 - 0.2 - 3.0 - xg, xg, 3.0, 0.2])


def test_linear_row_is_exact_at_reference_point():
    # x_G = 2.76 gives starch = (14.04*20 + 2.76*70 + 3*2)/20 = 24 % = S_ref and D = 20 = DMI_scn
    x = _x(2.76)
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    nl = E.nonlinear_diet_nel(x, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
    assert nl["starch_pct"] == pytest.approx(24.0, abs=1e-12)
    assert nl["DMI"] == pytest.approx(20.0, abs=1e-12)
    lin_supply = float(lin.supply(x, _theta_means(), NUT))
    assert lin_supply == pytest.approx(nl["NEL"], abs=1e-9)
    assert lin_supply == pytest.approx(float(x @ lin.nominal_density) + 1.54176, abs=1e-12)


def test_linear_row_is_conservative_below_reference_starch_and_not_above():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    lo = _x(1.0)       # starch 19.6 % < S_ref: NDF digestibility higher in the chain -> row below chain
    hi = _x(6.0)       # starch > S_ref: the row overstates (such diets violate a starch cap set at S_ref)
    for x, sign in ((lo, -1), (hi, +1)):
        nl = E.nonlinear_diet_nel(x, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
        diff = float(lin.supply(x, _theta_means(), NUT)) - nl["NEL"]
        assert sign * diff > 1e-6
    # the gap equals the Eq 3-5a starch term only: 0.66*(0.042 - 0.0409/20)*sum_i x_i NDF_i*0.0059*(S-24)
    nl = E.nonlinear_diet_nel(lo, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
    ndf_mass = float(lo @ np.array([40.0, 10.0, 12.0, 0.0]))
    gap = 0.66 * (0.042 - 0.0409 / 20.0) * ndf_mass * 0.0059 * (nl["starch_pct"] - 24.0)
    assert float(lin.supply(lo, _theta_means(), NUT)) - nl["NEL"] == pytest.approx(gap, abs=1e-9)


def test_affine_slopes_are_exact_for_perturbed_composition():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    rng = np.random.default_rng(20260925)
    for _ in range(20):
        comp = {"ndf": 40 + rng.normal(0, 3), "starch": 20 + rng.normal(0, 3), "cp": 10 + rng.normal(0, 1),
                "ash": 6 + rng.normal(0, 1)}
        direct = E.feed_energy_terms(F, SET, comp)["NEL"]
        th = np.array([[comp["ndf"] / 100, comp["starch"] / 100, comp["cp"] / 100, comp["ash"] / 100]])
        via = lin.density(np.vstack([th, _theta_means()[1:]]), NUT)[0]
        assert via == pytest.approx(direct, abs=1e-12)


def test_unmapped_columns_stay_at_means_and_nan_propagates():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map={"CP": "cp"})
    assert lin.slopes.shape == (4, 1)
    np.testing.assert_allclose(lin.density(_theta_means(nut=("CP",)), ("CP",)), lin.nominal_density, atol=1e-12)
    th = _theta_means(nut=("CP",)).copy()
    th[1, 0] = np.nan
    e = lin.density(th, ("CP",))
    assert np.isnan(e[1]) and np.all(np.isfinite(e[[0, 2, 3]]))       # missing data is never replaced
    with pytest.raises(ValueError):
        E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map={"lignin": "lignin"})
    with pytest.raises(KeyError):
        lin.density(_theta_means(nut=("NDF",)), ("NDF",))


def test_settings_and_inputs_are_validated():
    with pytest.raises(InvalidProblemError):
        E.FixedDMISettings(dmi_kg_d=0.0, body_weight_kg=600.0, starch_ref_pct=30.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
    with pytest.raises(InvalidProblemError):
        E.FeedEnergyInputs("bad", ndf=10, lignin=11, starch=0, fa=0, cp=0, ash=0, rup_pct_cp=0, drup_pct_rup=0,
                           dstarch_base=0.9, dfa=0.73)
    with pytest.raises(InvalidProblemError):
        E.FeedEnergyInputs("bad", ndf=10, lignin=1, starch=0, fa=0, cp=0, ash=0, rup_pct_cp=0, drup_pct_rup=0,
                           dstarch_base=1.2, dfa=0.73)


# ------------------------------------------------------------------------------------------------
# requirement
# ------------------------------------------------------------------------------------------------

def _req(**kw):
    base = dict(body_weight=Qty(600, "kg"), milk_energy=Qty(25.0, "Mcal/d"), days_pregnant=Qty(250, "d"),
                calf_birth_weight=Qty(40, "kg"), days_in_milk=Qty(100, "d"), frame_gain=Qty(0.2, "kg/d"),
                reserves_gain=Qty(-0.3, "kg/d"), mature_body_weight=Qty(650, "kg"),
                empty_body_gain_per_frame_gain=Qty(0.82, "1"), activity=Qty(0.3, "Mcal/d"))
    base.update(kw)
    return E.nel_requirement_chapter3(**base)


def test_requirement_chapter3_hand():
    # maintenance 0.10*600^0.75 = 12.123093028059742 (Eq 3-13)
    # gestation t=250: GrUter(parturition) = 40*1.825 = 73; rate = 0.0243 - 0.0000245*250 = 0.018175;
    #   GrUter_Wt = 73*exp(-0.018175*30) = 42.31786939391668; gain = rate*GrUter_Wt = 0.7691272762344356;
    #   involution at DIM 100 ~ -3.7e-9; NEL = (gain + inv)*0.882/0.14*0.66 = 3.1980311992454156 (Eq 3-18)
    # frame: BW/MatBW = 600/650; Fat = (0.067+0.375r)*0.82; Prot = (0.201-0.081r)*0.82; RE = 3.759066076923077;
    #   NEL = 0.2*RE/0.61 = 1.232480680958386 (Eq 3-20)
    # reserves loss: -0.3*6.3*0.89 = -1.6821 (Eq 3-19c) ; activity 0.3 ; milk 25.0
    r = _req()
    c = r.components
    assert c["maintenance"].value == pytest.approx(12.123093028059742, abs=1e-12)
    assert c["gestation"].intermediates["GrUter_Wt"] == pytest.approx(42.31786939391668, abs=1e-11)
    assert c["gestation"].value == pytest.approx(3.1980311992454156, abs=1e-11)
    assert c["frame_gain"].intermediates["RE_FADG"] == pytest.approx(3.759066076923077, abs=1e-12)
    assert c["frame_gain"].value == pytest.approx(1.232480680958386, abs=1e-12)
    assert c["reserves"].value == pytest.approx(-1.6821, abs=1e-12)
    assert r.value == pytest.approx(40.17150490826354, abs=1e-10)
    assert {"Eq 3-13", "Eq 3-18", "Eq 3-19a", "Eq 3-20a-e"} <= {x.equation for x in r.total.equations}


def test_requirement_guards_and_open_cow():
    open_cow = _req(days_pregnant=Qty(0, "d")).components["gestation"]
    assert open_cow.intermediates["GrUter_WtGain"] == 0.0                   # not pregnant: no gravid-uterus gain
    # only the post-partum involution term remains (Eq 3-17b; ~ -3.7e-9 kg/d at DIM 100, x 4.158)
    assert open_cow.value == pytest.approx(open_cow.intermediates["involution_gain"] * 0.882 / 0.14 * 0.66, abs=1e-15)
    assert abs(open_cow.value) < 1e-7
    # early lactation, open cow (K4c: the DIM-100/200 points above cannot see the involution constants):
    # Uter_Wt = (40*0.2288 - 0.204)*exp(-0.2*5) + 0.204 = 8.948*0.36787944 + 0.204 = 3.4957852396020663 (Eq 3-15b, 3-16b)
    # gain = -0.2*(Uter_Wt - 0.204) = -0.6583570479204133 kg/d (Eq 3-17b); NEL = gain*0.882/0.14*0.66 = -2.737448605253078
    early = _req(days_pregnant=Qty(0, "d"), days_in_milk=Qty(5, "d")).components["gestation"]
    assert early.intermediates["Uter_Wt"] == pytest.approx(3.4957852396020663, abs=1e-12)
    assert early.value == pytest.approx(-2.737448605253078, abs=1e-12)      # negative: released tissue, reported as is
    with pytest.raises(ValueError):
        _req(days_pregnant=Qty(5, "d"))                          # Eq 3-16a validity 12-280 d (p.32)
    with pytest.raises(ValueError):
        _req(days_pregnant=Qty(290, "d"))
    with pytest.raises(TypeError):
        _req(body_weight=600)                                     # bare numbers refused
    with pytest.raises(UnitError):
        _req(days_in_milk=Qty(100, "kg"))


# ------------------------------------------------------------------------------------------------
# engine glue
# ------------------------------------------------------------------------------------------------

def _energy_problem(lin, req, dmi=20.0):
    comp = {}
    ings = []
    for k, (f, dm) in enumerate(zip(FEEDS, (0.35, 0.88, 0.89, 1.0))):
        c = {"NDF": f.ndf / 100, "starch": f.starch / 100, "CP": f.cp / 100, "ash": f.ash / 100,
             E.ENERGY_COLUMN_ID: float(lin.nominal_density[k])}
        comp[f.ingredient_id] = c
        ings.append(ing(f.ingredient_id, dm, c, forage=1.0 if k == 0 else 0.0, stochastic=k < 3))
    spec = E.nel_constraint_spec(lin, req, constraint_id="PN-NEL-FIXEDDMI", provenance=SYN, numerical_tolerance=1e-6)
    nuts = [("NDF",), ("starch",), ("CP",), ("ash",), (E.ENERGY_COLUMN_ID, "energy_density")]
    return problem(ings, nuts, [dm_offer(dmi), spec], {f.ingredient_id: 0.1 for f in FEEDS}), spec


def test_constraint_spec_compiles_and_public_evaluator_margin_is_hand_value():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    req = _req()
    prob, spec = _energy_problem(lin, req)
    assert spec.kind is ConstraintKind.SUPPLY and spec.sense is Sense.GE and spec.unit == "Mcal/d"
    assert spec.constraint_class is ConstraintClass.PROBABILISTIC_NUTRITION
    assert spec.bound == pytest.approx(req.value - lin.constant_mcal_d, abs=1e-12)
    x = _x(2.76)
    d = prob.dm_estimates()
    q = x / d
    ev = evaluate(q, prob.nominal_theta()[None], d[None], prob.compiled, d_hat=d)
    k = list(ev.constraint_ids).index("PN-NEL-FIXEDDMI")
    nl = E.nonlinear_diet_nel(x, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
    # margin = linear supply - (req - C0) = (sum x e + C0) - req = chain NEL - req at the reference point
    assert ev.margin[0, k] == pytest.approx(nl["NEL"] - req.value, abs=1e-9)
    # a scenario with 10 % less realised DM for every feed lowers the per-kg part exactly by 10 %
    ev2 = evaluate(q, prob.nominal_theta()[None], 0.9 * d[None], prob.compiled, d_hat=d)
    assert ev2.margin[0, k] == pytest.approx(0.9 * float(x @ lin.nominal_density) - spec.bound, abs=1e-9)


def test_append_energy_column_and_wrapper_share_one_world():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    ids = tuple(f.ingredient_id for f in FEEDS)
    mean = _theta_means()
    rng = np.random.default_rng(7)
    states = mean[None] + np.where(np.arange(4)[None, :, None] < 3, rng.normal(0, 0.01, (12, 4, 4)), 0.0)
    dstates = np.array([0.35, 0.88, 0.89, 1.0])[None] + np.where(np.arange(4) < 3, rng.normal(0, 0.01, (12, 4)), 0.0)
    base = ScenarioSetModel("syn_base", ids, NUT, states, dstates, is_synthetic=True)
    wrapped = E.EnergyColumnModel(base, lin)
    s = RandomStreams(1103)
    dw = wrapped.draw(s, "opt", 50)
    db = base.draw(s, "opt", 50)
    da = E.append_energy_column(db, lin)
    assert dw.nutrient_ids == NUT + (E.ENERGY_COLUMN_ID,) == da.nutrient_ids
    np.testing.assert_allclose(dw.theta, da.theta, atol=0, rtol=0)            # same stream -> identical draws
    np.testing.assert_allclose(dw.theta[..., -1], lin.density(db.theta, NUT), atol=1e-12)
    assert da.model_fingerprint != db.model_fingerprint
    assert dw.model_fingerprint == da.model_fingerprint == wrapped.fingerprint()   # one world, either route
    wrapped.assert_same_world(dw, da, *wrapped.draw_world(s, n_opt=5, n_test=7).values())
    with pytest.raises(ValueError):
        wrapped.assert_same_world(db)                                            # base draws lack the column
    # a different linearisation is a different world
    lin2 = E.linearise_nel_fixed_dmi(FEEDS, E.FixedDMISettings(20.0, 600.0, 30.0, 1.0, 0.0), nutrient_map=NMAP)
    assert E.EnergyColumnModel(base, lin2).fingerprint() != wrapped.fingerprint()
    # point mass at the means reproduces the nominal density exactly
    pm = PointMassModel("syn_pm", ids, NUT, mean, np.array([0.35, 0.88, 0.89, 1.0]), is_synthetic=True)
    dp = E.EnergyColumnModel(pm, lin).draw(s, "validation", 3)
    np.testing.assert_allclose(dp.theta[0, :, -1], lin.nominal_density, atol=1e-12)
    with pytest.raises(ValueError):
        E.append_energy_column(da, lin)                                        # column already present
    assert E.energy_nutrient_spec().dimension == "energy_density"


def test_wrapper_nominal_state_appends_energy_of_base_nominal():
    lin = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
    ids = tuple(f.ingredient_id for f in FEEDS)

    class _Base(PointMassModel):                      # a base with a nominal_state(), like FactoryModel
        def nominal_state(self):
            return self.theta, self.d

    base = _Base("syn_nominal", ids, NUT, _theta_means(), np.array([0.35, 0.88, 0.89, 1.0]), is_synthetic=True)
    th, d = E.EnergyColumnModel(base, lin).nominal_state()
    assert th.shape == (4, 5)
    np.testing.assert_allclose(th[:, -1], lin.nominal_density, atol=1e-12)
    np.testing.assert_allclose(d, [0.35, 0.88, 0.89, 1.0])
    with pytest.raises(AttributeError):
        pm = PointMassModel("syn_pm2", ids, NUT, _theta_means(), np.array([0.35, 0.88, 0.89, 1.0]), is_synthetic=True)
        E.EnergyColumnModel(pm, lin).nominal_state()


# ------------------------------------------------------------------------------------------------
# requirements calling layer (configuration fixes the version; no defaults)
# ------------------------------------------------------------------------------------------------

VERSION = {"dmi_function": "dmi_eq2_1", "growth_option": "A_An_BWgain", "gestation_rule": "literal_equation",
           "ca_gestation_equation": "Eq 20-375", "milk_p_equation": "Eq 20-389", "maintenance_basis": "actual_D",
           "decisions_resolved": ["PUD-P2a-01", "PUD-P2a-03", "PUD-P2a-04", "PUD-P2a-05", "PUD-P2a-06", "PUD-P2a-08"],
           "row_status": "research_scenario_assumption", "row_status_rationale": "synthetic test configuration"}
# FIX5 working point WP-M (model_audit section 8.5; synthetic inputs, hand values by mpmath there)
WPM = {"parity_eq2_1": {"value": 1, "unit": "1"}, "body_weight": {"value": 640, "unit": "kg"},
       "mature_body_weight": {"value": 720, "unit": "kg"}, "body_condition_score": {"value": 3.5, "unit": "1"},
       "days_in_milk": {"value": 290, "unit": "d"}, "milk_yield": {"value": 25, "unit": "kg/d"},
       "milk_fat": {"value": 3.9, "unit": "%"}, "milk_true_protein": {"value": 3.4, "unit": "%"},
       "milk_lactose": {"value": 4.85, "unit": "%"}, "days_pregnant": {"value": 230, "unit": "d"},
       "body_weight_gain_for_minerals": {"value": 0.20, "unit": "kg/d"}}


def test_requirement_version_requires_every_key_and_valid_values():
    v = R.requirement_version_from_config(VERSION)
    assert v.choices.gestation_rule == "literal_equation" and v.row_status is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION
    for k in VERSION:
        bad = dict(VERSION)
        del bad[k]
        with pytest.raises(InvalidProblemError):
            R.requirement_version_from_config(bad)
    for k, val in (("dmi_function", "dmi_eq99"), ("maintenance_basis", "both"), ("growth_option", "C"),
                   ("row_status", "pending_user_decision"), ("decisions_resolved", "PUD-P2a-03")):
        with pytest.raises(InvalidProblemError):
            R.requirement_version_from_config(dict(VERSION, **{k: val}))
    with pytest.raises(InvalidProblemError):
        R.requirement_version_from_config(dict(VERSION, surprise=1))
    with pytest.raises(InvalidProblemError):
        R.requirement_version_from_config(dict(VERSION, row_status_rationale=" "))


def test_requirements_from_config_working_point_wpm():
    # WP-M hand values (model_audit 8.5): DMI Eq 2-1 20.263484; Ca total A with Eq 7-3 52.707891, gestation
    # Eq 7-3 4.763146 vs Eq 20-375 4.751510 -> total with Eq 20-375 = 52.696255; row (actual_D) 34.470755
    # - 0.011636 = 34.459119.  P total A (0.48) 48.067995; P row (actual_D) 27.804511 (t = 230 > 190, so the
    # literal and text rules coincide for P).
    out = R.requirements_from_config({"reference_cow": {"inputs": WPM}, "requirement_version": VERSION})
    assert out.dmi.value == pytest.approx(20.263484, abs=1e-6)
    assert out.ca.value == pytest.approx(52.696255, abs=2e-6)
    assert out.ca_row.bound_g_per_d == pytest.approx(34.459119, abs=2e-6)
    assert out.p.value == pytest.approx(48.067995, abs=1e-6)
    assert out.p_row.bound_g_per_d == pytest.approx(27.804511, abs=1e-6)
    assert dict(out.ca_row.terms) == {"C:AC_Ca:Ca": 1.0, "DM": -0.0009}
    # derived_expected mismatch raises (configuration and code cannot silently disagree)
    with pytest.raises(InvalidProblemError):
        R.requirements_from_config({"reference_cow": {"inputs": WPM}, "requirement_version": VERSION,
                                    "derived_expected": {"DMI_kg_d": 20.0, "tolerance": 1e-6}})
    lit = R.requirements_from_config({"reference_cow": {"inputs": WPM},
                                      "requirement_version": dict(VERSION, dmi_function="dmi_eq20_21_literal")})
    assert lit.dmi.value == pytest.approx(21.985984, abs=1e-6)            # NX-06 WP-M


def test_mineral_constraint_specs_use_configured_decisions_only():
    out = R.requirements_from_config({"reference_cow": {"inputs": WPM}, "requirement_version": VERSION})
    ca, p = R.mineral_constraint_specs(out, numerical_tolerance=1e-6)
    assert ca.bound == pytest.approx(out.ca_row.bound_g_per_d) and p.bound == pytest.approx(out.p_row.bound_g_per_d)
    assert ca.provenance.status is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION and ca.provenance.rationale
    assert "pending override" not in ca.notes                                   # no override was needed
    partial = dict(VERSION, decisions_resolved=["PUD-P2a-01", "PUD-P2a-03", "PUD-P2a-06"])
    out2 = R.requirements_from_config({"reference_cow": {"inputs": WPM}, "requirement_version": partial})
    with pytest.raises(PendingRelabelError):                                    # PUD-P2a-05 (and -04) still open
        R.mineral_constraint_specs(out2, numerical_tolerance=1e-6)


def test_g1_bodies_untouched_version_choice_is_explicit():
    """The book alternatives stay available; the calling layer adds no default."""
    assert set(R.DMI_FUNCTIONS) == {"dmi_eq2_1", "dmi_eq20_21_literal"}
    with pytest.raises(TypeError):
        R.RequirementChoices()                                                  # still no defaults


# ------------------------------------------------------------------------------------------------
# dev_case_v1 tracked configuration <-> code (K4c, 2026-09-25)
# ------------------------------------------------------------------------------------------------
# These tests read only TRACKED files (configs/dev_case_v1/*.yaml and configs/animal_profile.yaml, whose
# Table 21-1 inputs were already tracked before R4); they never open data/restricted_local.  The restricted
# bounds (value_ref) are checked for form only.  They make the configuration's hand-written derived values,
# price conversions and completeness executable instead of prose (R4 acceptance: every required field present,
# bounds / units / sources / statuses on every row, prices with source, date, unit, basis, currency).

import yaml as _yaml
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
_DEV = _REPO / "configs" / "dev_case_v1"


def _y(name):
    return _yaml.safe_load((_DEV / name).read_text(encoding="utf-8"))


def test_resolve_value_refs_rules(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "restricted_local").mkdir(parents=True)
    (tmp_path / "configs" / "a.yaml").write_text("fields: {x: {value: 2.5}, s: {value: abc}}\n", encoding="utf-8")
    (tmp_path / "data" / "restricted_local" / "r.yaml").write_text("v: {value: 9}\n", encoding="utf-8")
    cfg = {"a": {"value_from": "configs/a.yaml#fields.x.value", "unit": "kg"},
           "b": [{"value_ref": "restricted_values.yaml#k"}],
           "c": {"value": 2.5, "value_from": "configs/a.yaml#fields.x.value"}}
    out, log = R.resolve_value_refs(cfg, repo_root=tmp_path)
    assert out["a"]["value"] == 2.5 and "value" not in cfg["a"]              # deep copy; input untouched
    assert "value" not in out["b"][0] and log[1]["value"] is None             # restricted: left unresolved
    out2, _ = R.resolve_value_refs(cfg, repo_root=tmp_path, restricted_values={"k": {"value": 7}})
    assert out2["b"][0]["value"] == 7.0
    for bad in ({"a": {"value_from": "data/restricted_local/r.yaml#v.value"}},     # restricted via value_from: refused
                {"a": {"value_from": "configs/a.yaml#fields.nope.value"}},         # missing key
                {"a": {"value_from": "configs/a.yaml"}},                            # no key path
                {"a": {"value_from": "../outside.yaml#x"}},                         # outside the repository
                {"a": {"value_from": "configs/a.yaml#fields.s.value"}},            # not a number
                {"a": {"value": 3.0, "value_from": "configs/a.yaml#fields.x.value"}}):  # literal disagrees
        with pytest.raises(InvalidProblemError):
            R.resolve_value_refs(bad, repo_root=tmp_path)
    with pytest.raises(InvalidProblemError):
        R.resolve_value_refs({"b": {"value_ref": "restricted_values.yaml#missing"}}, repo_root=tmp_path,
                             restricted_values={"k": 1})


def test_dev_case_animal_config_reproduces_derived_expected():
    animal, log = R.resolve_value_refs(_y("animal.yaml"), repo_root=_REPO)
    assert log and all(e["value_from"].startswith("configs/animal_profile.yaml#") for e in log)
    out = R.requirements_from_config(animal)              # raises if derived_expected and code disagree (tol 1e-6)
    exp = animal["derived_expected"]
    assert out.dmi.value == pytest.approx(exp["DMI_kg_d"], abs=1e-6)
    assert out.ca.value == pytest.approx(exp["Ca_absorbed_requirement_g_d"], abs=1e-6)
    assert out.p.value == pytest.approx(exp["P_absorbed_requirement_g_d"], abs=1e-6)
    assert out.ca_row.bound_g_per_d == pytest.approx(exp["Ca_row_bound_actual_D_g_d"], abs=1e-6)
    assert out.p_row.bound_g_per_d == pytest.approx(exp["P_row_bound_actual_D_g_d"], abs=1e-6)
    assert out.version.dmi_function == "dmi_eq2_1" and out.version.maintenance_basis == "actual_D"
    with pytest.raises(InvalidProblemError):             # the unresolved configuration is refused (no defaults)
        R.requirements_from_config(_y("animal.yaml"))


def test_dev_case_constraints_complete_and_consistent_with_code():
    animal, _ = R.resolve_value_refs(_y("animal.yaml"), repo_root=_REPO)
    req = R.requirements_from_config(animal)
    craw = _y("constraints.yaml")
    cons = {c["id"]: c for c in craw["constraints"]}
    ca, p = R.mineral_constraint_specs(req, numerical_tolerance=1e-6)
    for cid, spec in (("PN-CA-ABS", ca), ("PN-P-ABS", p)):
        assert cons[cid]["bound"]["value"] == pytest.approx(spec.bound, abs=1e-6)
        assert dict(cons[cid]["engine"]["terms"]) == dict(spec.terms)
        assert cons[cid]["engine"]["unit"] == spec.unit == "g/d"
    dmi = req.dmi.value
    assert cons["SH-DM-PLAN"]["bound"]["value"] == pytest.approx(dmi, abs=1e-6)
    assert cons["DIAG-DM-REAL-LO"]["bound"]["value"] == pytest.approx(0.9 * dmi, abs=1e-6)
    assert cons["DIAG-DM-REAL-HI"]["bound"]["value"] == pytest.approx(1.1 * dmi, abs=1e-6)
    assert cons["PN-NEL-FIXEDDMI"]["bound"]["value"] == pytest.approx(_y("energy.yaml")["constraint"]["bound_Mcal_d"], abs=1e-6)
    # the joint event is exactly the probabilistic rows; fNDF terms appear only through the forage group
    assert set(craw["joint_event"]["members"]) == {k for k, c in cons.items() if c["class"] == "probabilistic_nutrition"}
    assert cons["PN-T1"]["engine"]["terms"] == {"G:forage:NDF": 1.0}
    for cid, c in cons.items():
        assert c["class"] in ("structural_hard", "probabilistic_nutrition", "diagnostic_only"), cid
        assert c["status"] in ("sourced", "research_scenario_assumption"), cid
        assert str(c["expression"]).strip() and str(c["source"]).strip(), cid
        b = c["bound"]
        assert b.get("unit"), cid
        assert (b.get("value") is not None) != bool(b.get("value_ref")), cid          # exactly one of the two
        if b.get("value_ref"):
            assert str(b["value_ref"]).startswith("restricted_values.yaml#"), cid
        assert float(c["numerical_tolerance"]) > 0, cid
        if c["class"] == "probabilistic_nutrition":
            assert c["engine"]["dm_source"] == "scenario" and c["engine"]["sense"] in ("ge", "le"), cid
        if c["class"] == "structural_hard" and c["engine"]["kind"] != "builtin_variable_bound":
            assert c["engine"]["dm_source"] == "decision_estimate", cid


def test_dev_case_energy_requirement_and_bound_reproduced():
    animal, _ = R.resolve_value_refs(_y("animal.yaml"), repo_root=_REPO)
    energy, log = R.resolve_value_refs(_y("energy.yaml"), repo_root=_REPO)     # S_ref (value_ref) stays unresolved
    assert any(e.get("value_ref") and e["value"] is None for e in log)
    req = R.requirements_from_config(animal)
    cow = animal["reference_cow"]["inputs"]
    comp = energy["requirement"]["components"]
    nreq = E.nel_requirement_chapter3(
        body_weight=Qty(cow["body_weight"]["value"], "kg"), milk_energy=req.milk_energy.as_qty(),
        days_pregnant=Qty(cow["days_pregnant"]["value"], "d"), calf_birth_weight=Qty(cow["calf_birth_weight"]["value"], "kg"),
        days_in_milk=Qty(cow["days_in_milk"]["value"], "d"), frame_gain=Qty(cow["frame_gain"]["value"], "kg/d"),
        reserves_gain=Qty(cow["reserves_gain"]["value"], "kg/d"),
        mature_body_weight=Qty(cow["mature_body_weight"]["value"], "kg"),
        empty_body_gain_per_frame_gain=Qty(0.82, "1"), activity=Qty(comp["activity"]["value_Mcal_d"], "Mcal/d"))
    for k in ("maintenance", "activity", "lactation", "gestation", "frame_gain", "reserves"):
        assert nreq.components[k].value == pytest.approx(comp[k]["value_Mcal_d"], abs=1e-6), k
    assert nreq.value == pytest.approx(energy["requirement"]["total_Mcal_d"], abs=1e-6)
    milk_cp = cow["milk_true_protein"]["value"] / 100.0 * cow["milk_yield"]["value"] / 0.94     # p.30: 94 % true protein
    assert milk_cp == pytest.approx(energy["settings"]["milk_cp_kg_d"]["value"], abs=1e-6)
    c0 = 0.66 * 0.0146 * 160.0 * milk_cp                                                          # Eq 3-10a/b, 3-12
    assert c0 == pytest.approx(energy["constraint"]["C0_Mcal_d"], abs=1e-6)
    assert nreq.value - c0 == pytest.approx(energy["constraint"]["bound_Mcal_d"], abs=1e-6)
    assert energy["settings"]["dmi_kg_d"]["value"] == pytest.approx(req.dmi.value, abs=1e-6)
    pn_t4 = [c for c in _y("constraints.yaml")["constraints"] if c["id"] == "PN-T4"][0]
    assert energy["settings"]["starch_ref_pct"]["value_ref"] == pn_t4["bound"]["value_ref"]     # S_ref = PN-T4 bound
    assert energy["engine_column"]["nutrient_id"] == E.ENERGY_COLUMN_ID and energy["method_id"] == E.LINEARISATION_ID
    assert energy["applies_to_all_methods"] is True


def test_dev_case_prices_complete_and_derivable_from_quotes():
    pr = _y("prices.yaml")
    assert pr["currency"] == "USD" and pr["is_scenario"] is True and str(pr["price_period"]).strip()
    src = {s["source_id"]: s for s in pr["sources"]}
    for sid, s in src.items():
        assert str(s["url"]).startswith("http") and str(s["title"]).strip() and str(s["publisher"]).strip(), sid
    items = pr["items"]
    assert sorted(items) == sorted(g["ingredient_id"] for g in _y("inventory.yaml")["ingredients"])
    for iid, it in items.items():
        ev = it["engine_value"]
        assert ev["unit"] == "USD/lb" and ev["basis"] in ("DM", "as_fed") and ev["value"] > 0, iid
        if ev["basis"] == "DM":
            assert ev["dm_basis_semantics"] == "converted_with_decision_time_dm_estimate", iid
        sid = (it.get("raw_quote") or {}).get("source_id") or it.get("rule_source_id")
        assert sid in src, iid
        assert src[sid].get("report_date") or "updated" in str(src[sid]["title"]) or src[sid].get("used_for"), iid
        assert it["status"] in ("sourced", "research_scenario_assumption"), iid
        if it["status"] == "sourced":
            assert str(it.get("locator", "")).strip(), iid
        else:
            assert str(it.get("rationale", "")).strip(), iid
    # hand arithmetic (short ton = 2000 lb, cwt = 100 lb, corn bushel = 56 lb at 15.5 % moisture, silage rule 11x at 35 % DM)
    corn = 4.84
    expect = {"corn_grain_dry_ground_medium": corn / (56 * 0.845), "corn_silage_typical": 11 * corn / 0.35 / 2000,
              "legume_hay_mid": (125.0 + 150.0) / 2 / 2000, "soybean_meal_solvent_48cp": 379.08 / 2000,
              "canola_meal_solvent": 289.22 / 2000, "soybean_hulls": 134.57 / 2000,
              "limestone_ground": 11.35 / 100, "dicalcium_phosphate": 23.85 / 100}
    for iid, v in expect.items():
        assert items[iid]["engine_value"]["value"] == pytest.approx(v, rel=1e-6), iid
    assert items["corn_grain_dry_ground_medium"]["raw_quote"]["value"] == corn
    assert items["corn_silage_typical"]["inputs"]["corn_price_usd_per_bu"] == corn
    assert items["legume_hay_mid"]["raw_quote"]["value_range"] == [125.0, 150.0]
    sens = items["corn_silage_typical"]["sensitivity"]
    assert sens["multiplier_10"] == pytest.approx(10 * corn / 0.35 / 2000, rel=1e-6)
    assert sens["multiplier_12"] == pytest.approx(12 * corn / 0.35 / 2000, rel=1e-6)


def test_dev_case_inventory_fndf_rule_exclusions_and_no_nulls():
    inv = _y("inventory.yaml")
    for g in inv["ingredients"]:
        assert g["forage_weight"]["value"] == (1.0 if g["category"] == "forage" else 0.0), g["ingredient_id"]
        assert g["forage_weight"]["status"] == "sourced"
    assert {g["ingredient_id"] for g in inv["ingredients"] if g["category"] == "forage"} == {"corn_silage_typical",
                                                                                             "legume_hay_mid"}
    ex = {e["ingredient_id"] for e in inv["excluded"]}
    assert {"corn_grain_steam_flaked", "barley_grain_dry_ground", "corn_grain_high_moisture_coarse",
            "corn_grain_high_moisture_fine"} <= ex                            # Table 5-1 premise (p.63-64)
    assert not ex & {g["ingredient_id"] for g in inv["ingredients"]}
    assert all(str(e["reason"]).strip() for e in inv["excluded"])
    lime = [g for g in inv["ingredients"] if g["ingredient_id"] == "limestone_ground"][0]
    assert lime["p_content"]["status"] == "research_scenario_assumption" and lime["p_content"]["rationale"].strip()
    for name in ("animal.yaml", "inventory.yaml", "prices.yaml", "constraints.yaml", "energy.yaml"):
        d = _y(name)
        assert d["frozen"] is False and d["stage_status"] == "implemented" and d["case_id"] == "dev_case_v1"
        nulls = []

        def walk(x, path):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]")
            elif x is None:
                nulls.append(path)

        walk(d, name)
        assert not nulls, nulls
