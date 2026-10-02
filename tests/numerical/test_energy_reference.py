"""Reference energy check (review round 3, F3 / F4; next-round instruction D) -- hand-computed synthetic checks.

Scope:

* the round-3 energy probe: diet DM 20 kg, diet starch 24 % (= S_ref), only the forage NDF drawn 40 -> 35 %;
  the linear fixed-DMI row lies 0.363398 Mcal/d above the project's nonlinear Chapter 3 chain.  The
  difference is re-derived here by separate hand arithmetic (Eq 3-3a written out in this file, not taken from
  the code under test) and must agree to <= 1e-9 Mcal/d.  It is a MODEL DIFFERENCE between two
  representations of the same Chapter 3 equations, not an animal result;
* ``energy_reference.reference_energy_check``: per-state linear and reference verdicts, false pass / false
  fail, undefined states kept in the denominator, the linear margin equal to the public evaluator's, no
  re-feeding with the hidden DM, the joint event with the reference energy verdict, guards;
* F4: composition-closure classes -- "CP + NDF + starch + EE + ash > 100 % DM" and "Eq 3-1 ROM < 0" are
  ``analysis_overlap_or_measurement_anomaly`` (NDF contains part of the CP and of the ash, NASEM 2021
  p.22-23), separate from ``support_violation`` (lignin > NDF, a fraction outside [0, 1]); nothing is
  forced closed, normalised, clipped or dropped.

Every number is SYNTHETIC (invented for the arithmetic).  The requirement 35.0 Mcal/d used below is a
diagnostic threshold chosen only to place the synthetic states on both sides of it -- it is not a
nutritional recommendation.  No restricted data are read.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ration_reliability.evaluation import evaluate_drawset
from ration_reliability.nutrition import energy as E
from ration_reliability.nutrition import energy_reference as ER
from ration_reliability.uncertainty.base import DrawSet

from engine_test_helpers import dm_offer, ing, problem, supply

# ------------------------------------------------------------------------------------------------
# synthetic case (identical to tests/unit/test_fixb_energy_box_closure.py section 4)
# ------------------------------------------------------------------------------------------------
F1 = E.FeedEnergyInputs("syn_forage", ndf=40.0, lignin=4.0, starch=20.0, fa=3.0, cp=10.0, ash=6.0,
                        rup_pct_cp=30.0, drup_pct_rup=70.0, dstarch_base=0.88, dfa=0.73)
G1 = E.FeedEnergyInputs("syn_grain", ndf=10.0, lignin=1.0, starch=70.0, fa=4.0, cp=9.0, ash=1.5,
                        rup_pct_cp=45.0, drup_pct_rup=75.0, dstarch_base=0.88, dfa=0.73)
M1 = E.FeedEnergyInputs("syn_mineral", ndf=0.0, lignin=0.0, starch=0.0, fa=0.0, cp=0.0, ash=100.0,
                        rup_pct_cp=0.0, drup_pct_rup=0.0, dstarch_base=0.95, dfa=0.73)
FEEDS = (F1, G1, M1)
IDS = tuple(f.ingredient_id for f in FEEDS)
SET = E.FixedDMISettings(dmi_kg_d=20.0, body_weight_kg=600.0, starch_ref_pct=24.0, milk_cp_kg_d=1.0,
                         body_gain_cp_kg_d=0.0)
NMAP = {"NDF": "ndf", "starch": "starch", "CP": "cp", "ash": "ash"}
BASE_NUTS = ("CP", "NDF", "starch", "EE", "ash")
DM = np.array([0.35, 0.88, 1.0])
# diet DM 20 kg (= DMI_scn): mineral 0.2, grain 1.68 so that the diet starch is (18.12*20 + 1.68*70)/20 = 24 % = S_ref
X = np.array([19.8 - 1.68, 1.68, 0.2])
Q = X / DM
REQ = 35.0                  # synthetic diagnostic threshold (Mcal/d), NOT a requirement of any cow
TOL = 1e-6


def _lin():
    return E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)


def _mean_theta():
    fld = {"CP": "cp", "NDF": "ndf", "starch": "starch", "EE": "fa", "ash": "ash"}
    return np.array([[getattr(f, fld[n]) / 100.0 for n in BASE_NUTS] for f in FEEDS])


def _draws(states, d=None, stream_id="root=7/test"):
    """``states``: list of {(ingredient index, nutrient): value in % DM}; each state starts at the means."""
    th = np.repeat(_mean_theta()[None], len(states), axis=0).copy()
    for s, mods in enumerate(states):
        for (i, n), v in mods.items():
            th[s, i, BASE_NUTS.index(n)] = v / 100.0
    dd = np.repeat(DM[None], len(states), axis=0) if d is None else np.asarray(d, dtype=float)
    base = DrawSet(th, dd, "test", stream_id, "syn_ref_world", "fp-syn-ref", IDS, BASE_NUTS, True)
    return E.append_energy_column(base, _lin())


def _spec(req=REQ, lin=None):
    lin = lin or _lin()
    return ER.EnergyReferenceSpec(lin, bound_mcal_d=req - lin.constant_mcal_d, tolerance_mcal_d=TOL)


def _dndf_base_hand(ndf, lg):
    # Eq 3-3a (p.24) written out independently: 0.75 (NDF - Lg) [1 - (Lg/NDF)^0.667] / NDF
    return 0.75 * (ndf - lg) * (1.0 - (lg / ndf) ** 0.667) / ndf


# ------------------------------------------------------------------------------------------------
# 1  the round-3 energy probe, with an independent hand calculation of the difference
# ------------------------------------------------------------------------------------------------

def test_probe_linear_row_overstates_chain_by_0_363398_and_hand_difference_agrees():
    lin = _lin()
    res = ER.reference_energy_check(Q, _draws([{(0, "NDF"): 35.0}]), _spec(lin=lin))
    L, N = float(res.linear_supply_mcal_d[0]), float(res.reference_supply_mcal_d[0])
    # values of the round-3 review (probe_energy_surrogate): linear 35.198780, chain 34.835382, +0.363398 Mcal/d
    assert L == pytest.approx(35.198780, abs=5e-7)
    assert N == pytest.approx(34.835382, abs=5e-7)
    assert float(res.difference_mcal_d[0]) == pytest.approx(0.363398, abs=5e-7)
    assert res.diet_dmi_kg_d[0] == pytest.approx(20.0, abs=1e-12)
    assert res.diet_starch_pct[0] == pytest.approx(24.0, abs=1e-12)   # starch = S_ref: Eq 3-5a starch terms equal
    # Hand derivation.  D = DMI_scn = 20 and diet starch = S_ref = 24, so the Eq 3-5a starch and DMI/BW terms, the
    # Eq 3-6a MFCP (linear in NDF, x-weighted), ROM (Eq 3-1, linear), digested CP and UE are identical in the row and
    # the chain.  The only difference is Eq 3-3a: the row keeps the forage dNDF_base at its mean NDF 40 (lignin 4),
    # the chain uses the drawn NDF 35.  It enters DE through 0.042 * NDF * dNDF * x and GasE (Eq 3-9) through
    # 0.0409 * sum x NDF dNDF / DMI; NEL = 0.66 ME.  Hence
    #   L - N = 0.66 * x_F * 35 * (dNDF_base(40,4) - dNDF_base(35,4)) * (0.042 - 0.0409 / 20)
    x_f = 19.8 - 1.68
    hand = 0.66 * x_f * 35.0 * (_dndf_base_hand(40.0, 4.0) - _dndf_base_hand(35.0, 4.0)) * (0.042 - 0.0409 / 20.0)
    assert hand == pytest.approx(0.363398, abs=5e-7)
    assert abs(float(res.difference_mcal_d[0]) - hand) <= 1e-9          # review: agreement ~1.9e-13
    # the same two numbers by the energy module directly (no reference-check code involved)
    th = _mean_theta()[:, [BASE_NUTS.index(n) for n in lin.nutrient_ids]]
    th[0, list(lin.nutrient_ids).index("NDF")] = 0.35
    assert float(lin.supply(X, th, lin.nutrient_ids)) == pytest.approx(L, abs=1e-12)
    nl = E.nonlinear_diet_nel(X, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0,
                              compositions=[{"ndf": 35.0}, {}, {}])
    assert nl["NEL"] == pytest.approx(N, abs=1e-12)
    # with the synthetic diagnostic threshold 35.0 between the two, the linear row passes and the chain fails
    assert res.state_class[0] == "false_pass"
    assert res.label == ER.MODEL_DIFFERENCE_LABEL == "model_difference_not_animal_outcome"


def test_probe_is_exact_at_the_reference_point():
    res = ER.reference_energy_check(Q, _draws([{}]), _spec())
    assert abs(float(res.difference_mcal_d[0])) < 1e-9                  # means, D = DMI, starch = S_ref


# ------------------------------------------------------------------------------------------------
# 2  per-state classes, undefined states and the summary
# ------------------------------------------------------------------------------------------------
STATES = [
    {},                                              # 0 means: L = N = 34.174 < 35             -> agree_fail
    {(0, "NDF"): 35.0},                              # 1 probe: L 35.199 >= 35 > N 34.835       -> false_pass
    {(0, "starch"): 10.0},                           # 2 diet starch 14.94 < S_ref: L 34.306 < 35 <= N 35.352 -> false_fail
    {(0, "NDF"): 36.0, (0, "starch"): 10.0},         # 3 L 35.126, N 35.777                     -> agree_pass
    {(0, "NDF"): 3.0},                               # 4 NDF 3 < lignin 4: support violation    -> reference_undefined
    {(1, "NDF"): 16.0},                              # 5 grain sum 100.5 %, ROM < 0 (anomaly)   -> agree_fail (computed)
]


def test_state_classes_counts_and_denominators():
    res = ER.reference_energy_check(Q, _draws(STATES), _spec())
    assert list(res.state_class) == ["agree_fail", "false_pass", "false_fail", "agree_pass", "reference_undefined",
                                     "agree_fail"]
    assert list(res.undefined_reason) == ["", "", "", "", "support_violation", ""]
    assert list(res.support_violation) == [False, False, False, False, True, False]
    assert list(res.analysis_anomaly) == [False, False, False, False, False, True]
    # pinned regression values of state 2 (row and chain; they place the synthetic state on both sides of 35.0)
    assert float(res.reference_supply_mcal_d[2]) == pytest.approx(35.351700, abs=5e-6)
    assert float(res.linear_supply_mcal_d[2]) == pytest.approx(34.306342, abs=5e-6)
    # state 5 is an analysis anomaly (grain CP+NDF+starch+EE+ash = 9+16+70+4+1.5 = 100.5 %) but still computed
    assert np.isfinite(res.reference_supply_mcal_d[5]) and res.reference_defined[5]
    s = res.summary()
    assert s["n_states"] == 6
    assert s["class_counts"] == {"agree_pass": 1, "agree_fail": 2, "false_pass": 1, "false_fail": 1,
                                 "reference_undefined": 1, "linear_undefined": 0}
    assert (s["n_linear_pass"], s["n_linear_fail"]) == (3, 3)             # states 1, 3, 4 pass the linear row
    assert (s["n_reference_pass"], s["n_reference_fail"], s["n_reference_undefined"]) == (2, 3, 1)
    assert s["reference_undefined_reasons"]["support_violation"] == 1
    # the undefined state stays in the denominator: bounds [3/6, 4/6]
    assert s["reference_violation_rate_lower"] == pytest.approx(3 / 6)
    assert s["reference_violation_rate_upper"] == pytest.approx(4 / 6)
    assert s["false_pass_share_of_all"] == pytest.approx(1 / 6)
    assert s["false_pass_share_of_linear_pass"] == pytest.approx(1 / 2)   # linear passes with a defined reference: 1, 3
    assert s["false_fail_share_of_linear_fail"] == pytest.approx(1 / 3)   # linear fails: 0, 2, 5
    diff = res.difference_mcal_d
    assert s["difference_mcal_d"]["n"] == 5 and np.isnan(diff[4])
    assert s["difference_mcal_d"]["max"] == pytest.approx(0.363398, abs=5e-7)
    assert s["difference_mcal_d"]["share_linear_above_reference"] == pytest.approx(1 / 5)
    assert s["class_counts_in_analysis_anomaly_states"]["agree_fail"] == 1
    assert res.class_counts(np.array([True, True, False, False, False, False])) == {
        "agree_pass": 0, "agree_fail": 1, "false_pass": 1, "false_fail": 0, "reference_undefined": 0,
        "linear_undefined": 0}
    # arrays are read-only
    with pytest.raises(ValueError):
        res.reference_violated[0] = True


def test_missing_cell_of_used_feed_is_undefined_and_unused_feed_is_ignored():
    d = np.repeat(DM[None], 3, axis=0)
    d[1, 1] = np.nan                                         # grain DM missing in state 1 (grain is used)
    dr = _draws([{}, {}, {(2, "NDF"): np.nan}], d=d)          # state 2: NaN NDF in the mineral (used)
    res = ER.reference_energy_check(Q, dr, _spec())
    # a missing cell of a used feed makes the energy column NaN too, so the evaluator's row is undefined as well:
    # both states are "linear_undefined" with reason missing_data; they stay in the denominator
    assert list(res.state_class) == ["agree_fail", "linear_undefined", "linear_undefined"]
    assert res.undefined_reason[1] == "missing_data" and res.undefined_reason[2] == "missing_data"
    assert not res.linear_defined[1] and not res.reference_defined[2]
    s = res.summary()
    assert s["n_states"] == 3 and s["reference_violation_rate_upper"] == pytest.approx(3 / 3)
    # a support violation in an ingredient that is not fed does not make the state undefined
    q0 = Q.copy()
    q0[1] = 0.0
    res0 = ER.reference_energy_check(q0, _draws([{(1, "NDF"): 0.5}]), _spec())   # grain NDF 0.5 < lignin 1, unused
    assert res0.reference_defined[0] and not res0.support_violation[0]


# ------------------------------------------------------------------------------------------------
# 3  the linear verdict is the public evaluator's; no re-feeding with the hidden DM; joint event
# ------------------------------------------------------------------------------------------------

def _problem(lin, req=REQ):
    ings = []
    th0 = _mean_theta()
    for k, f in enumerate(FEEDS):
        c = {n: float(th0[k, j]) for j, n in enumerate(BASE_NUTS)}
        c[E.ENERGY_COLUMN_ID] = float(lin.nominal_density[k])
        ings.append(ing(f.ingredient_id, float(DM[k]), c, forage=1.0 if k == 0 else 0.0, stochastic=k < 2))
    row = supply("PN-NEL-FIXEDDMI", {E.ENERGY_COLUMN_ID: 1.0}, "ge", req - lin.constant_mcal_d, "Mcal/d", tol=TOL)
    starch_row = supply("PN-STARCH-SYN", {"starch": 1.0}, "le", 4.4, "kg/d", tol=TOL)   # a second synthetic member
    nuts = [n for n in BASE_NUTS] + [(E.ENERGY_COLUMN_ID, "energy_density")]
    return problem(ings, nuts, [dm_offer(20.0), row, starch_row], {i: 0.1 for i in IDS}), row


def test_linear_margins_equal_the_public_evaluator_and_spec_from_constraint():
    lin = _lin()
    pr, row = _problem(lin)
    spec = ER.EnergyReferenceSpec.from_constraint(lin, row)
    assert spec.requirement_mcal_d == pytest.approx(REQ, abs=1e-12)
    dr = _draws(STATES)
    ev = evaluate_drawset(Q, dr, pr.compiled, d_hat=DM)
    res = ER.reference_energy_check(Q, dr, spec)
    k = list(ev.constraint_ids).index("PN-NEL-FIXEDDMI")
    np.testing.assert_allclose(res.linear_margin_mcal_d, ev.margin[:, k], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(res.linear_violated, ev.violated[:, k])
    assert res.q_hash == ev.q_hash and res.stream_id == ev.draw_stream_id
    # a constraint that is not an energy supply row is refused
    with pytest.raises(ValueError):
        ER.EnergyReferenceSpec.from_constraint(lin, supply("X", {"CP": 1.0}, "ge", 1.0, "kg/d"))


def test_q_is_fixed_and_realised_dm_is_q_times_drawn_dm():
    lin = _lin()
    d = np.repeat(DM[None], 2, axis=0)
    d[1] *= 0.9                                              # 10 % less DM in every feed in state 1
    res = ER.reference_energy_check(Q, _draws([{}, {}], d=d), _spec(lin=lin))
    assert res.diet_dmi_kg_d[0] == pytest.approx(20.0, abs=1e-12)
    assert res.diet_dmi_kg_d[1] == pytest.approx(18.0, abs=1e-12)          # not re-fed to the planned 20 kg
    per_kg = float(X @ lin.nominal_density)
    assert float(res.linear_margin_mcal_d[1]) == pytest.approx(0.9 * per_kg - (REQ - lin.constant_mcal_d), abs=1e-9)
    nl = E.nonlinear_diet_nel(0.9 * X, FEEDS, body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
    assert float(res.reference_supply_mcal_d[1]) == pytest.approx(nl["NEL"], abs=1e-12)
    # the function has no d_hat / planned-DM argument through which the hidden DM could re-scale q
    assert set(inspect.signature(ER.reference_energy_check).parameters) == {"q", "draws", "spec"}


def test_joint_event_with_reference_energy_verdict():
    lin = _lin()
    pr, row = _problem(lin)
    dr = _draws(STATES)
    ev = evaluate_drawset(Q, dr, pr.compiled, d_hat=DM)
    res = ER.reference_energy_check(Q, dr, ER.EnergyReferenceSpec.from_constraint(lin, row))
    jt = ER.joint_with_reference_energy(ev, res)
    ks = list(ev.constraint_ids).index("PN-STARCH-SYN")
    other = ev.violated[:, ks]
    exp_v = other | res.reference_violated
    np.testing.assert_array_equal(jt["joint_violation"], exp_v)
    exp_u = ~exp_v & ~res.reference_defined
    np.testing.assert_array_equal(jt["joint_unknown"], exp_u)
    assert jt["energy_row_member"] and jt["n_violated"] + jt["n_unknown"] <= 6
    assert jt["rate_upper"] == pytest.approx((exp_v.sum() + exp_u.sum()) / 6)
    # the evaluator's own joint event differs exactly where the two energy verdicts differ (no other change)
    diff_states = ev.joint_violation != jt["joint_violation"]
    assert set(np.flatnonzero(diff_states)) <= set(np.flatnonzero(
        (res.state_class == "false_pass") | (res.state_class == "false_fail") | ~res.reference_defined))
    # energy not a member: the reference check does not change the event
    jt2 = ER.joint_with_reference_energy(ev, res, member_ids=["PN-STARCH-SYN"])
    np.testing.assert_array_equal(jt2["joint_violation"], other)
    # guards: another q, another stream
    res_q = ER.reference_energy_check(Q * 1.01, dr, ER.EnergyReferenceSpec.from_constraint(lin, row))
    with pytest.raises(ValueError):
        ER.joint_with_reference_energy(ev, res_q)
    res_s = ER.reference_energy_check(Q, _draws(STATES, stream_id="root=7/validation"),
                                      ER.EnergyReferenceSpec.from_constraint(lin, row))
    with pytest.raises(ValueError):
        ER.joint_with_reference_energy(ev, res_s)


def test_guards_world_order_and_q():
    lin = _lin()
    dr = _draws([{}])
    # an energy column from another linearisation is another world
    lin2 = E.linearise_nel_fixed_dmi(FEEDS, E.FixedDMISettings(20.0, 600.0, 30.0, 1.0, 0.0), nutrient_map=NMAP)
    with pytest.raises(ValueError, match="another world"):
        ER.reference_energy_check(Q, dr, _spec(lin=lin2))
    # ingredient order must match the linearisation (no reordering)
    th = dr.theta[:, ::-1, :]
    rev = DrawSet(th, dr.d[:, ::-1], "test", "root=7/test", "x", "fp", IDS[::-1], dr.nutrient_ids, True)
    with pytest.raises(ValueError, match="order"):
        ER.reference_energy_check(Q[::-1], rev, _spec(lin=lin))
    with pytest.raises(ValueError):
        ER.reference_energy_check(-Q, dr, _spec(lin=lin))
    with pytest.raises(ValueError):
        ER.reference_energy_check(Q[:2], dr, _spec(lin=lin))
    # without the energy column the linear row is computed from the linearisation (same numbers)
    base = DrawSet(dr.theta[..., :-1], dr.d, "test", "root=7/test", "b", "fpb", IDS, BASE_NUTS, True)
    r_col = ER.reference_energy_check(Q, dr, _spec(lin=lin))
    r_lin = ER.reference_energy_check(Q, base, _spec(lin=lin))
    assert r_col.energy_column_source == "draw_column" and r_lin.energy_column_source == "linearisation_density"
    np.testing.assert_allclose(r_col.linear_supply_mcal_d, r_lin.linear_supply_mcal_d, atol=1e-12)
    rec = _spec(lin=lin).to_record()
    assert rec["reference_chain_id"] == ER.REFERENCE_CHAIN_ID and "not the full NASEM" in rec["not"]


# ------------------------------------------------------------------------------------------------
# 4  F4: composition-closure classes (never forced closed, normalised, clipped or dropped)
# ------------------------------------------------------------------------------------------------

def test_sum_above_100_and_negative_rom_are_analysis_anomalies_not_support_violations():
    # A synthetic fibrous by-product analysed as NDF 60, CP 25, ash 10, starch 4, EE 3 (FA 2): the analysed sum is
    # 102 % DM and Eq 3-1 ROM = 100 - 10 - 60 - 4 - 2/1.06 - 25 = -0.887 %.  Yet the NDF residue itself contains
    # 8 % DM of neutral-detergent insoluble CP and 4 % DM of NDF ash (counted again in CP and ash, p.22-23), so the
    # overlap-free sum is 102 - 12 = 90 % DM: the state is physically possible.  Hence "analysis anomaly".
    byp = E.FeedEnergyInputs("syn_byproduct", ndf=60.0, lignin=5.0, starch=4.0, fa=2.0, cp=25.0, ash=10.0,
                             rup_pct_cp=40.0, drup_pct_rup=80.0, dstarch_base=0.9, dfa=0.73)
    lin = E.linearise_nel_fixed_dmi((byp, M1), SET, nutrient_map=NMAP)
    th = np.array([[[0.25, 0.60, 0.04, 0.03, 0.10], [0.0, 0.0, 0.0, 0.0, 1.0]],        # anomaly
                   [[0.25, 0.04, 0.04, 0.03, 0.10], [0.0, 0.0, 0.0, 0.0, 1.0]],        # NDF 4 < lignin 5: support
                   [[-0.01, 0.50, 0.04, 0.03, 0.10], [0.0, 0.0, 0.0, 0.0, 1.0]],       # CP < 0: support
                   [[0.20, 0.50, 0.04, 0.03, 0.10], [0.0, 0.0, 0.0, 0.0, 1.0]]])       # ordinary
    before = th.copy()
    cls = E.classify_composition_states(th, BASE_NUTS, ("syn_byproduct", "syn_mineral"), lin=lin)
    rom0 = 100 - 10 - 60 - 4 - 2 / 1.06 - 25
    assert rom0 == pytest.approx(-0.8867924528, abs=1e-9)
    assert list(cls.sum_gt_100pct[:, 0]) == [True, False, False, False]
    assert list(cls.rom_lt_0[:, 0]) == [True, False, False, False]
    assert list(cls.analysis_anomaly[:, 0]) == [True, False, False, False]
    assert list(cls.support_violation[:, 0]) == [False, True, True, False]
    assert list(cls.lignin_gt_ndf[:, 0]) == [False, True, False, False]
    assert list(cls.fraction_outside_unit_interval[:, 0]) == [False, False, True, False]
    # the mineral (ash exactly 100 %) is neither
    assert not cls.analysis_anomaly[:, 1].any() and not cls.support_violation[:, 1].any()
    np.testing.assert_array_equal(th, before)                                         # nothing modified
    summ = cls.summary()
    assert summ["classes"] == {"analysis_anomaly": "analysis_overlap_or_measurement_anomaly",
                               "support_violation": "support_violation"}
    assert summ["per_ingredient"]["syn_byproduct"]["n_analysis_anomaly"] == 1
    assert summ["per_ingredient"]["syn_byproduct"]["n_support_violation"] == 2
    # the chain computes the anomalous state as it is (negative ROM is not rejected, not clipped) ...
    nl = E.nonlinear_diet_nel([10.0, 0.2], (byp, M1), body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0)
    assert np.isfinite(nl["NEL"])
    assert E.rom_eq3_1(ndf=60.0, starch=4.0, fa=2.0, cp=25.0, ash=10.0) == pytest.approx(rom0, abs=1e-12)
    # ... but refuses a support violation, naming its class
    with pytest.raises(ValueError, match="support_violation"):
        E.nonlinear_diet_nel([10.0, 0.2], (byp, M1), body_weight_kg=600.0, milk_cp_kg_d=1.0, body_gain_cp_kg_d=0.0,
                             compositions=[{"ndf": 4.0}, {}])


def test_closure_report_keeps_old_fields_and_adds_classes():
    lin = _lin()
    th = np.repeat(_mean_theta()[None], 4, axis=0).copy()
    th[1, 0, BASE_NUTS.index("NDF")] = 0.62                     # forage sum 1.01 > 1 (anomaly)
    th[2, 0, BASE_NUTS.index("NDF")] = 0.03                     # forage NDF 3 % < lignin 4 % (support violation)
    rep = E.composition_closure_report(th, BASE_NUTS, IDS, lin=lin)
    f = rep["syn_forage"]
    assert f["share_sum_gt_100pct"] == pytest.approx(0.25) and f["share_rom_lt_0"] == pytest.approx(0.25)
    assert f["share_lignin_gt_ndf"] == pytest.approx(0.25) and f["share_support_violation"] == pytest.approx(0.25)
    assert f["classes"]["share_sum_gt_100pct"] == E.ANALYSIS_ANOMALY_CLASS
    assert f["classes"]["share_rom_lt_0"] == E.ANALYSIS_ANOMALY_CLASS
    assert f["classes"]["share_lignin_gt_ndf"] == E.SUPPORT_VIOLATION_CLASS
    assert rep["syn_grain"]["share_support_violation"] == 0.0
    assert E.composition_closure_report(th, BASE_NUTS, IDS)["syn_forage"]["share_lignin_gt_ndf"] is None   # no lin


def test_closure_comment_no_longer_claims_overlap_only_through_ee():
    src = inspect.getsource(E)
    assert "only through the non-FA part of EE" not in src
    i = src.index("CLOSURE_SUM_COLUMNS = (")
    # FIX3_DEF: the comment block of CLOSURE_SUM_COLUMNS (from its first line to the constant), not a fixed window
    block = src[src.rindex("#: engine columns whose drawn values are summed", 0, i):i]
    assert "NDF is itself a" in block and "analysis_overlap_or_measurement_anomaly" in block
    assert "SUPPORT_VIOLATION_CLASS" in block and "support of a declared variable" in block


def test_closure_comment_states_the_nasem_causal_order_and_subclasses():
    # red team D7/F4: NASEM calls the double subtraction incorrect and uses NDF *despite* it, for five practical
    # reasons (p.23) -- not "for that reason"; sum > 1 and ROM < 0 get distinct sub-class names
    src = inspect.getsource(E)
    i = src.index("CLOSURE_SUM_COLUMNS = (")
    block = src[src.rindex("#: engine columns whose drawn values are summed", 0, i):i]
    flat = " ".join(line.lstrip("#: ").strip() for line in block.splitlines())
    assert "for that reason" not in flat and "keeps NDF rather than" not in flat
    assert "*incorrect*" in flat and "not because of it" in flat and "five practical reasons (p.23)" in flat
    assert E.SUM_OVERLAP_SUBCLASS == "definition_overlap_or_measurement_error"
    assert E.DERIVED_QUANTITY_SUBCLASS == "derived_quantity_anomaly"
    th = np.repeat(_mean_theta()[None], 2, axis=0).copy()
    th[1, 0, BASE_NUTS.index("NDF")] = 0.62
    rep = E.composition_closure_report(th, BASE_NUTS, IDS, lin=_lin())["syn_forage"]
    assert rep["subclasses"] == {"share_sum_gt_100pct": E.SUM_OVERLAP_SUBCLASS,
                                 "share_rom_lt_0": E.DERIVED_QUANTITY_SUBCLASS}
    assert rep["classes"]["share_rom_lt_0"] == rep["classes"]["share_sum_gt_100pct"] == E.ANALYSIS_ANOMALY_CLASS
    cls = E.classify_composition_states(th, BASE_NUTS, IDS, lin=_lin()).summary()
    assert cls["subclasses"] == {"sum_gt_100pct": E.SUM_OVERLAP_SUBCLASS, "rom_lt_0": E.DERIVED_QUANTITY_SUBCLASS}


def test_reference_verdict_is_labelled_a_model_assumption_not_sourced():
    # red team D/C: the requirement is sourced, the reference verdict model is the project chain (assumptions)
    s = ER.reference_energy_check(Q, _draws([{}, {(0, "NDF"): 35.0}]), _spec()).summary()
    vs = s["verdict_status"]
    assert vs["verdict_model_status"] == ER.VERDICT_MODEL_STATUS == "research_assumption"
    assert vs["label_for_a_sourced_requirement"] == "requirement_sourced/model_assumption"
    assert vs["verdict_model"] == ER.REFERENCE_CHAIN_ID
    joined = " ".join(vs["assumptions"])
    for needle in ("16.5 g/kg DMI", "DMI = the DM supplied", "RDP adequate", "forage NDF adequate"):
        assert needle in joined
