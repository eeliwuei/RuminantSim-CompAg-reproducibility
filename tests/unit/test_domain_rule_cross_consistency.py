"""B-442: cross-consistency of the two Table 5-1 domain rules (reference problem v2, official-run plan batch 1).

``evaluation/reference.py`` (the primary rule for every reported rate) and ``nutrition/domain.py::
premise_conditioned_event`` implement the domain condition separately.  The reference evaluator uses **no** Table 5-1
verdict outside the domain (an out-of-domain state is violated only through a non-T member, otherwise unknown);
``premise_conditioned_event`` keeps an out-of-domain T violation as violated.  Given the same event read with T
verdicts in every state and the same domain, this file checks, state by state:

* the upper rates are **equal** (the violated-or-unknown sets coincide);
* the reference lower rate is **<=** the domain.py lower rate (its violated set is a subset);
* the difference is **exactly** the out-of-domain states whose only violated members are Table 5-1 rows;

for the per-state reading of every declared variant (primary, 1/3, 2/3, the weaker kernel reading), for the three
event families with T rows (main reference, H0 eleven, S2 six), and for the plan-level reading of reference problem v2
(``table51_domain = "plan"``; the ration's status at ``d_hat`` and nominal composition broadcast to all states): in
the plan domain the plan-level event equals the event with T verdicts used; outside it, it equals "violated iff a non-T
member is violated, otherwise unknown".  One hand-computed case pins the counts; randomised synthetic states (fixed
seeds) cover the rest.

Every number is SYNTHETIC (invented for the arithmetic; the energy requirement 35.0 Mcal/d is the diagnostic threshold
of tests/numerical/test_energy_reference.py, not a requirement of any cow).  No restricted data are read.
"""

from __future__ import annotations

import numpy as np
import pytest

from ration_reliability.evaluation import reference as REF
from ration_reliability.nutrition import domain as DOM
from ration_reliability.nutrition import energy as E
from ration_reliability.uncertainty.base import DrawSet

from engine_test_helpers import conc, ing, problem, supply

# ---- a synthetic reference problem (the construction of tests/unit/test_evaluate_reference.py) --------------------
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
NUTS = ("CP", "NDF", "starch", "EE", "ash", "Ca", "P")
DM = np.array([0.35, 0.88, 1.0])
REQ = 35.0
LIN = E.linearise_nel_fixed_dmi(FEEDS, SET, nutrient_map=NMAP)
#                    CP     NDF    starch  EE     ash    Ca      P      (% DM, synthetic)
MEANS = np.array([[10.0, 40.0, 20.0, 3.0, 6.0, 0.5, 0.3],
                  [9.0, 10.0, 70.0, 4.0, 1.5, 0.05, 0.3],
                  [0.0, 0.0, 0.0, 0.0, 100.0, 30.0, 0.0]]) / 100.0
#: the probe diet (DM 20 kg): forage-dominated starch -> inside the plan-level domain of every declared variant
Q_IN = np.array([19.8 - 1.68, 1.68, 0.2]) / DM
#: grain-dominated starch (forage starch 2.0 kg vs grain 6.86 kg at the means) -> outside at 1/3, 1/2 and 2/3
Q_OUT = np.array([10.0, 9.8, 0.2]) / DM
MAIN = tuple(c for c in REF.H0_ELEVEN if c not in ("PN-CP-HI", "PN-EE-HI"))
#: (domain-conditioned event, the same event with T verdicts in every state, members, energy verdict)
PAIRS = (("main_reference", REF.MAIN_EVENT_T51_IGNORED, MAIN, "reference"),
         ("h0_eleven_domain_conditioned", "h0_eleven", REF.H0_ELEVEN, "linear"),
         ("s2_six_domain_conditioned", "s2_six", REF.S2_SIX, "not_member"))


def _problem():
    ings = []
    for k, f in enumerate(FEEDS):
        c = {n: float(MEANS[k, j]) for j, n in enumerate(NUTS)}
        c[E.ENERGY_COLUMN_ID] = float(LIN.nominal_density[k])
        ings.append(ing(f.ingredient_id, float(DM[k]), c, forage=1.0 if k == 0 else 0.0, stochastic=k < 2))
    rows = [
        supply("SH-DM-PLAN", {"DM": 1.0}, "eq", 20.0, "kg/d", cls="structural_hard", dm_source="decision_estimate",
               basis="DM"),
        supply("PN-NEL-FIXEDDMI", {E.ENERGY_COLUMN_ID: 1.0}, "ge", REQ - LIN.constant_mcal_d, "Mcal/d"),
        supply("PN-CP-SUP", {"CP": 1.0}, "ge", 1.9, "kg/d"),
        conc("PN-CP-HI", {"CP": 1.0}, "le", 10.5),
        conc("PN-EE-HI", {"EE": 1.0}, "le", 7.0),
        supply("PN-CA-ABS", {"Ca": 1.0}, "ge", 0.14, "kg/d"),
        supply("PN-P-ABS", {"P": 1.0}, "ge", 0.05, "kg/d"),
        conc("PN-T1", {"G:forage:NDF": 1.0}, "ge", 30.0),
        conc("PN-T2", {"NDF": 1.0}, "ge", 32.0),
        conc("PN-T3", {"NDF": 1.0, "G:forage:NDF": 2.0}, "ge", 90.0),
        conc("PN-T4", {"starch": 1.0}, "le", 25.0),
        conc("PN-T5", {"starch": 1.0, "G:forage:NDF": -2.0}, "le", -35.0),
    ]
    nuts = list(NUTS) + [(E.ENERGY_COLUMN_ID, "energy_density")]
    return problem(ings, nuts, rows, {i: 0.1 for i in IDS}, problem_id="syn_b442_problem")


def _row(cid, status, role, verdict="public_evaluator_linear", *, h0=False, s2=False, ex=False):
    f = {c: "x" for c in REF.REQUIRED_COLUMNS}
    f.update({"constraint_id": cid, "status": status, "role": role, "verdict_model": verdict,
              "threshold_source_status": "sourced" if status == "sourced" else "research_assumption",
              "model_form_status": "not_applicable"})
    return REF.ReferenceConstraintRow(cid, status, role, verdict, h0, s2, ex, True, "u", f)


def _table():
    rows = [_row("SH-DM-PLAN", "planning_only", "optimization_only", "structural_check_d_hat"),
            _row("PN-NEL-FIXEDDMI", "sourced", "main_reference", "reference_chain_ch3", h0=True, ex=True),
            _row("PN-CP-SUP", "sourced", "main_reference", h0=True, s2=True),
            _row("PN-CP-HI", "research_assumption", "optimization_only", h0=True, ex=True),
            _row("PN-EE-HI", "research_assumption", "optimization_only", h0=True, ex=True),
            _row("PN-CA-ABS", "sourced", "main_reference", h0=True, ex=True),
            _row("PN-P-ABS", "sourced", "main_reference", h0=True, ex=True)]
    rows += [_row(c, "sourced", "main_reference", h0=True, s2=True) for c in REF.TABLE51_ROWS]
    return REF.ReferenceConstraintTable(tuple(rows), None, None)


PROBLEM = _problem()
#: variants: primary 0.50, 1/3, 2/3 (counted = syn_forage) and the weaker reading that also counts syn_grain
SPEC = REF.ReferenceSpec.from_problem(PROBLEM, LIN, _table(), dgc_ingredient_ids=("syn_forage",),
                                      corn_silage_ingredient_ids=("syn_grain",))


def _draws(theta, d, stream_id="root=7/test"):
    base = DrawSet(theta, d, "test", stream_id, "syn_b442_world", "fp-syn-b442", IDS, NUTS, True)
    return E.append_energy_column(base, LIN)


def _hand_draws(states):
    th = np.repeat(MEANS[None], len(states), axis=0).copy()
    for s, mods in enumerate(states):
        for (i, n), v in mods.items():
            th[s, i, NUTS.index(n)] = v / 100.0
    return _draws(th, np.repeat(DM[None], len(states), axis=0))


def _random_draws(seed, n=300):
    rng = np.random.default_rng(seed)
    sig = np.full(MEANS.shape, 0.25)
    sig[0, NUTS.index("starch")] = 0.9                     # wide forage starch: states on both sides of every tau
    sig[1, NUTS.index("starch")] = 0.3
    th = np.clip(MEANS[None] * np.exp(rng.normal(0.0, 1.0, (n,) + MEANS.shape) * sig[None]), 0.0, 1.0)
    th[:, 2, :] = MEANS[2]                                  # the mineral is deterministic
    miss = rng.random(n) < 0.04                             # a missing CP cell of a used feed: undefined members
    th[miss, 1, NUTS.index("CP")] = np.nan
    d = np.clip(DM[None] * np.exp(rng.normal(0.0, 0.05, (n, len(IDS)))), 0.05, 1.0)
    d[:, 2] = DM[2]
    return _draws(th, d, stream_id=f"root=7/test/{seed}")


def _split(res, members, energy):
    """(non-T member violated [S], T member violated [S]) from the public evaluator and the energy check -- computed
    here independently of both domain rules."""
    ev = res.evaluation
    ids = list(ev.constraint_ids)
    viol = np.asarray(ev.violated, dtype=bool)
    t_rows = [m for m in members if m in REF.TABLE51_ROWS]
    non_t = [m for m in members if m not in REF.TABLE51_ROWS and not (m == "PN-NEL-FIXEDDMI" and energy == "reference")]
    nt = viol[:, [ids.index(m) for m in non_t]].any(axis=1) if non_t else np.zeros(res.n_states, dtype=bool)
    if "PN-NEL-FIXEDDMI" in members and energy == "reference":
        nt = nt | np.asarray(res.energy.reference_violated, dtype=bool)
    t = viol[:, [ids.index(m) for m in t_rows]].any(axis=1)
    return nt, t


def _check_pair(ref_v, ref_u, pc, in_dom, non_t_v, t_v, where):
    """The three B-442 relations, state by state and as counts."""
    ref_v, ref_u, in_dom = np.asarray(ref_v, bool), np.asarray(ref_u, bool), np.asarray(in_dom, bool)
    np.testing.assert_array_equal(ref_v | ref_u, pc.violated | pc.unknown, err_msg=f"{where}: upper sets differ")
    assert not np.any(ref_v & ~pc.violated), f"{where}: reference violated set is not a subset"
    only_t_out = ~in_dom & t_v & ~non_t_v
    np.testing.assert_array_equal(pc.violated & ~ref_v, only_t_out, err_msg=f"{where}: difference set")
    s = pc.summary()
    lower_dom = s[DOM.PREMISE_CONDITIONED]["n_violated"]
    upper_dom = lower_dom + s[DOM.PREMISE_CONDITIONED]["n_unknown"]
    assert upper_dom == int((ref_v | ref_u).sum()) and int(ref_v.sum()) <= lower_dom, where
    assert lower_dom - int(ref_v.sum()) == int(only_t_out.sum()), where
    return int(only_t_out.sum())


# =================================================================================================
# hand-computed case (the four states of tests/unit/test_evaluate_reference.py, C-1 test)
# =================================================================================================

def test_hand_case_counts():
    # state 0: means, inside the primary domain, energy fails (34.174 < 35 Mcal/d by both verdicts) -> violated;
    # state 1: forage starch 5 % -> share 0.435 < 0.5, nothing violated; state 2: out of domain + CP supply deficit;
    # state 3: out of domain, forage NDF 30 % -> only Table 5-1 rows violated
    states = [{}, {(0, "starch"): 5.0}, {(0, "starch"): 5.0, (1, "CP"): 1.0, (0, "CP"): 2.0},
              {(0, "starch"): 5.0, (0, "NDF"): 30.0}]
    res = REF.evaluate_reference(Q_IN, _hand_draws(states), SPEC)
    prim = res.domain["T51-DGC-0.50"]
    assert list(prim.status) == [DOM.DOMAIN_IN, DOM.DOMAIN_OUT, DOM.DOMAIN_OUT, DOM.DOMAIN_OUT]
    m = res.events["main_reference"]
    pc = DOM.premise_conditioned_event(res.event_violation[REF.MAIN_EVENT_T51_IGNORED],
                                       res.event_unknown[REF.MAIN_EVENT_T51_IGNORED], prim, contains_table51_rows=True)
    s = pc.summary()[DOM.PREMISE_CONDITIONED]
    # reference.py: violated {0, 2}, unknown {1, 3}; domain.py: violated {0, 2, 3}, unknown {1}
    assert (m["n_violated"], m["n_unknown"]) == (2, 2)
    assert (s["n_violated"], s["n_unknown"]) == (3, 1)
    assert m["rate_upper"] == s["rate_upper"] == 1.0 and m["rate_lower"] == 0.5 and s["rate_lower"] == 0.75
    assert list(pc.violated & ~res.event_violation["main_reference"]) == [False, False, False, True]
    # plan level: the probe diet is inside the primary domain as formulated -> T rows judged in every state; the
    # plan-level event equals the event with T verdicts used: violated {0, 2, 3}, state 1 not violated (upper 3/4)
    assert res.plan_domain.status == DOM.DOMAIN_IN and res.summary()["headline"]["t51_plan_domain_status"] == DOM.DOMAIN_IN
    p = res.events[REF.MAIN_EVENT_PLAN_DOMAIN]
    assert (p["n_violated"], p["n_unknown"], p["rate_upper"]) == (3, 0, 0.75)
    assert p["t51_plan_domain_status"] == DOM.DOMAIN_IN and p["table51_domain"] == "plan"
    hl = res.summary()["headline"]
    assert hl["main_reference_plan_domain_rate_upper"] == 0.75 and hl["main_reference_rate_upper"] == 1.0
    assert hl["main_reference_plan_domain_unknown_only_because_out_of_domain"] == 0
    assert "F1" in hl["main_event_domain_reading"]


# =================================================================================================
# randomised synthetic states
# =================================================================================================

@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15])
@pytest.mark.parametrize("q_name", ["in", "out"])
def test_randomised_states_every_variant_every_event_and_the_plan_level(seed, q_name):
    q = Q_IN if q_name == "in" else Q_OUT
    res = REF.evaluate_reference(q, _random_draws(seed), SPEC)
    S = res.n_states
    summ = res.summary()
    n_diff_total = 0
    # (a) per-state reading, primary variant, the three event families with Table 5-1 rows
    prim = res.domain["T51-DGC-0.50"]
    in_prim = prim.status == DOM.DOMAIN_IN
    for eid, ign, members, energy in PAIRS:
        pc = DOM.premise_conditioned_event(res.event_violation[ign], res.event_unknown[ign], prim,
                                           contains_table51_rows=True)
        nt, t = _split(res, members, energy)
        n_diff_total += _check_pair(res.event_violation[eid], res.event_unknown[eid], pc, in_prim, nt, t,
                                    f"{eid} seed={seed} q={q_name}")
    # (b) every declared variant: the main event conditioned on that variant (reference.py counts in the summary)
    nt, t = _split(res, MAIN, "reference")
    ign_v, ign_u = res.event_violation[REF.MAIN_EVENT_T51_IGNORED], res.event_unknown[REF.MAIN_EVENT_T51_IGNORED]
    assert set(summ["domain"]) == {"T51-DGC-0.50", "T51-DGC-0.33", "T51-DGC-0.67", "T51-CORNKERNEL-0.50"}
    for vid, dres in res.domain.items():
        pc = DOM.premise_conditioned_event(ign_v, ign_u, dres, contains_table51_rows=True)
        blk = summ["domain"][vid]["main_reference_conditioned_on_this_variant"]
        ps = pc.summary()[DOM.PREMISE_CONDITIONED]
        only_t_out = int((~(dres.status == DOM.DOMAIN_IN) & t & ~nt).sum())
        assert blk["rate_upper"] == ps["rate_upper"], vid                             # equal upper rates
        assert blk["n_violated"] <= ps["n_violated"], vid                             # reference lower <= domain lower
        assert ps["n_violated"] - blk["n_violated"] == only_t_out, vid                # exact difference
        # plan-level reading of this variant (the ration's status at d_hat and the nominal composition)
        pv = DOM.plan_domain_status(q, SPEC.nominal_theta, SPEC.d_hat, IDS, SPEC.constraints.nutrient_ids,
                                    next(v for v in SPEC.domain_variants if v.variant_id == vid))
        assert summ["domain"][vid]["nominal_status"] == pv.status
        ppc = DOM.premise_conditioned_event(ign_v, ign_u, pv.as_domain_result(S), contains_table51_rows=True)
        pblk = summ["domain"][vid]["main_reference_plan_domain_on_this_variant"]
        pps = ppc.summary()[DOM.PREMISE_CONDITIONED]
        assert pblk["rate_upper"] == pps["rate_upper"] and pblk["n_violated"] <= pps["n_violated"], vid
        assert pps["n_violated"] - pblk["n_violated"] == (0 if pv.in_domain else int((t & ~nt).sum())), vid
    # (c) the plan-level companion events (primary variant)
    pl = res.plan_domain
    expected_status = DOM.DOMAIN_IN if q_name == "in" else DOM.DOMAIN_OUT
    assert pl.status == expected_status and summ["headline"]["t51_plan_domain_status"] == expected_status
    for eid, ign in ((REF.MAIN_EVENT_PLAN_DOMAIN, REF.MAIN_EVENT_T51_IGNORED),
                     ("main_reference_plus_cp_hi_plan_domain", None)):
        pv_, pu_ = res.event_violation[eid], res.event_unknown[eid]
        if ign is None:                                   # main + CP-HI: build the T-verdicts-used arrays here
            members = MAIN + ("PN-CP-HI",)
            nt2, t2 = _split(res, members, "reference")
            ev = res.evaluation
            ids = list(ev.constraint_ids)
            und = np.asarray(ev.undefined, bool)[:, [ids.index(m) for m in members if m != "PN-NEL-FIXEDDMI"]]
            u_any = und.any(axis=1) | ~np.asarray(res.energy.reference_defined, bool)
            iv, iu = nt2 | t2, ~(nt2 | t2) & u_any
        else:
            nt2, t2 = nt, t
            iv, iu = res.event_violation[ign], res.event_unknown[ign]
        if pl.in_domain:                                  # T rows judged in every state: = the T-verdicts-used event
            np.testing.assert_array_equal(pv_, iv, err_msg=eid)
            np.testing.assert_array_equal(pu_, iu, err_msg=eid)
        else:                                             # T rows unknown in every state
            np.testing.assert_array_equal(pv_, nt2, err_msg=eid)
            np.testing.assert_array_equal(pu_, ~nt2, err_msg=eid)
        ppc = DOM.premise_conditioned_event(iv, iu, pl.as_domain_result(S), contains_table51_rows=True)
        n_diff_total += _check_pair(pv_, pu_, ppc, pl.state_mask(S), nt2, t2, f"{eid} seed={seed} q={q_name}")
    hl = summ["headline"]
    ign_u_main = res.event_unknown[REF.MAIN_EVENT_T51_IGNORED]
    assert hl["main_reference_plan_domain_unknown_only_because_out_of_domain"] == (
        0 if pl.in_domain else int((~nt & ~ign_u_main).sum()))
    assert hl["main_reference_plan_domain_rate_upper"] == res.events[REF.MAIN_EVENT_PLAN_DOMAIN]["rate_upper"]
    # the randomised states do exercise the difference (out-of-domain states with only T rows violated)
    assert n_diff_total > 0


def test_the_plan_level_reading_is_constant_over_states_and_needs_a_nominal_composition():
    res = REF.evaluate_reference(Q_OUT, _random_draws(3, n=50), SPEC)
    assert res.events[REF.MAIN_EVENT_PLAN_DOMAIN]["table51_domain_variant"] == "T51-DGC-0.50"
    assert res.summary()["plan_domain"]["status"] == DOM.DOMAIN_OUT
    assert not any(k in res.summary()["plan_domain"] for k in ("planned_counted_starch_share",
                                                                "planned_starch_margin_kg_d"))
    # a spec with a plan-level event but no nominal composition is refused
    with pytest.raises(ValueError, match="nominal_theta"):
        REF.ReferenceSpec(constraints=SPEC.constraints, table=SPEC.table, energy=SPEC.energy,
                          domain_variants=SPEC.domain_variants, events=SPEC.events, d_hat=SPEC.d_hat,
                          nominal_theta=None)
