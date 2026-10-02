"""Table 5-1 applicability domain (review round 3, F3 section 5.2; next-round instruction D5-D6).

Hand-computed synthetic checks of ``nutrition.domain``:

* the starch-source criterion ``share = sum_counted x St / sum_all x St >= threshold`` per state, in the linear
  form of the engine's DIAG-T51 row, with realised DM ``x = q d`` (``q`` fixed);
* statuses ``in_domain_conditional`` / ``not_assessable`` / ``undefined``; every state stays in the
  denominator; the cross-tabulation of a row verdict by status;
* the threshold is a project research assumption (primary 0.50, sensitivity 1/3 and 2/3; a weaker
  corn-kernel reading only as a sensitivity); TMR and particle size are ``assumption_only`` and cannot be
  declared sourced or met;
* at 0.50 the verdict equals the public evaluator's verdict of a DIAG-T51-style row (weights +0.5 / -0.5).

Every number is SYNTHETIC; nothing here is a feed value or a statement about any diet or cow.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ration_reliability.evaluation import evaluate
from ration_reliability.nutrition import domain as DOM

from engine_test_helpers import ing, problem, supply

IDS = ("syn_dgc", "syn_silage", "syn_hulls", "syn_mineral")
NUTS = ("CP", "starch")
DM = np.array([0.88, 0.35, 0.90, 1.00])
X = np.array([5.0, 10.0, 2.0, 0.2])             # kg DM / d at the planned DM
Q = X / DM
# starch % DM at the means: dgc 70, silage 30, hulls 2, mineral 0
THETA0 = np.array([[0.09, 0.70], [0.08, 0.30], [0.12, 0.02], [0.0, 0.0]])


def _theta(mods=()):
    th = THETA0.copy()
    for (i, v) in mods:
        th[i, 1] = v
    return th


def _primary():
    return DOM.default_table51_variants(("syn_dgc",), corn_silage_ingredient_ids=("syn_silage",))


def test_hand_shares_and_statuses_for_all_declared_variants():
    # state 0 (means): starch kg/d = dgc 5*0.70 = 3.5, silage 10*0.30 = 3.0, hulls 2*0.02 = 0.04 -> total 6.54
    #   dgc share = 3.5 / 6.54 = 0.53517; corn-kernel share = 6.5 / 6.54 = 0.99388
    # state 1: silage starch drawn 40 % -> 4.0 kg, total 7.54, dgc share 3.5 / 7.54 = 0.46419
    # state 2: dgc starch drawn 80 % -> 4.0 kg, total 7.04, share 4/7.04 = 0.56818
    theta = np.stack([_theta(), _theta([(1, 0.40)]), _theta([(0, 0.80)])])
    d = np.repeat(DM[None], 3, axis=0)
    v = {s.variant_id: s for s in _primary()}
    assert set(v) == {"T51-DGC-0.50", "T51-DGC-0.33", "T51-DGC-0.67", "T51-CORNKERNEL-0.50"}
    assert v["T51-DGC-0.50"].role == "primary" and v["T51-DGC-0.50"].threshold == DOM.PRIMARY_THRESHOLD == 0.5
    r = DOM.table5_1_domain_arrays(Q, theta, d, IDS, NUTS, v["T51-DGC-0.50"])
    np.testing.assert_allclose(r.counted_starch_share, [3.5 / 6.54, 3.5 / 7.54, 4.0 / 7.04], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r.starch_margin_kg_d, [3.5 - 0.5 * 6.54, 3.5 - 0.5 * 7.54, 4.0 - 0.5 * 7.04],
                               rtol=0, atol=1e-12)
    assert list(r.status) == ["in_domain_conditional", "not_assessable", "in_domain_conditional"]
    assert list(DOM.table5_1_domain_arrays(Q, theta, d, IDS, NUTS, v["T51-DGC-0.33"]).status) == \
        ["in_domain_conditional"] * 3
    assert list(DOM.table5_1_domain_arrays(Q, theta, d, IDS, NUTS, v["T51-DGC-0.67"]).status) == \
        ["not_assessable"] * 3
    ck = DOM.table5_1_domain_arrays(Q, theta, d, IDS, NUTS, v["T51-CORNKERNEL-0.50"])
    assert ck.counted_starch_share[0] == pytest.approx(6.5 / 6.54, abs=1e-12)
    assert list(ck.status) == ["in_domain_conditional"] * 3 and ck.role == "sensitivity"
    s = r.summary()
    assert s["n_states"] == 3 and s["counts"] == {"in_domain_conditional": 2, "not_assessable": 1, "undefined": 0}
    assert s["shares"]["not_assessable"] == pytest.approx(1 / 3)
    assert s["threshold_status"] == "research_assumption"
    assert s["tmr_premise"] == s["particle_size_premise"] == "assumption_only"


def test_realised_dm_is_q_times_drawn_dm_and_q_is_fixed():
    # silage DM drawn 0.30 instead of 0.35: realised silage DM = Q*0.30 = 10*0.30/0.35 = 8.5714 kg -> starch 2.5714
    d = DM.copy()
    d[1] = 0.30
    r = DOM.table5_1_domain_arrays(Q, THETA0, d, IDS, NUTS, _primary()[0])
    silage_starch = 10.0 * 0.30 / 0.35 * 0.30
    assert r.counted_starch_share[0] == pytest.approx(3.5 / (3.5 + silage_starch + 0.04), abs=1e-12)


def test_undefined_states_stay_in_the_denominator_and_unused_feeds_are_ignored():
    theta = np.stack([_theta(), _theta([(0, np.nan)]), np.zeros((4, 2)), _theta()])
    d = np.repeat(DM[None], 4, axis=0)
    d[3, 1] = np.nan                                         # silage DM missing (used)
    r = DOM.table5_1_domain_arrays(Q, theta, d, IDS, NUTS, _primary()[0])
    assert list(r.status) == ["in_domain_conditional", "undefined", "undefined", "undefined"]
    assert np.isnan(r.counted_starch_share[1:]).all()
    assert r.summary()["n_states"] == 4 and r.summary()["shares"]["undefined"] == pytest.approx(0.75)
    # a missing cell in an ingredient that is not fed does not matter
    q = Q.copy()
    q[2] = 0.0
    th = _theta()
    th[2, 1] = np.nan
    r2 = DOM.table5_1_domain_arrays(q, th, DM, IDS, NUTS, _primary()[0])
    assert r2.status[0] == "in_domain_conditional"
    assert r2.counted_starch_share[0] == pytest.approx(3.5 / 6.5, abs=1e-12)


def test_crosstab_keeps_every_state_and_does_not_call_outside_states_safe():
    theta = np.stack([_theta(), _theta([(1, 0.40)]), _theta([(1, 0.40)]), np.zeros((4, 2))])
    r = DOM.table5_1_domain_arrays(Q, theta, np.repeat(DM[None], 4, axis=0), IDS, NUTS, _primary()[0])
    assert list(r.status) == ["in_domain_conditional", "not_assessable", "not_assessable", "undefined"]
    ct = r.crosstab(np.array([False, False, True, False]), np.array([False, False, False, True]))
    assert ct == {"in_domain_conditional": {"violated": 0, "not_violated": 1, "undefined": 0},
                  "not_assessable": {"violated": 1, "not_violated": 1, "undefined": 0},
                  "undefined": {"violated": 0, "not_violated": 0, "undefined": 1}}
    assert sum(sum(v.values()) for v in ct.values()) == 4
    meaning = r.summary()["meaning"]["not_assessable"]
    assert "not safe" in meaning and "not harmful" in meaning and "denominator" in meaning


def test_spec_refuses_sourced_threshold_or_met_premises():
    with pytest.raises(ValueError, match="research_assumption"):
        DOM.Table51DomainSpec("x", ("syn_dgc",), 0.5, threshold_status="sourced")
    with pytest.raises(ValueError, match="assumption_only"):
        DOM.Table51DomainSpec("x", ("syn_dgc",), 0.5, tmr_premise="met")
    with pytest.raises(ValueError, match="assumption_only"):
        DOM.Table51DomainSpec("x", ("syn_dgc",), 0.5, particle_size_premise="sourced")
    with pytest.raises(ValueError, match="threshold"):
        DOM.Table51DomainSpec("x", ("syn_dgc",), 1.0)
    with pytest.raises(ValueError, match="primary"):
        DOM.Table51DomainSpec("x", ("syn_dgc", "syn_silage"), 0.5, counted_reading="corn_kernel_incl_corn_silage",
                              role="primary")
    with pytest.raises(KeyError):
        DOM.table5_1_domain_arrays(Q, THETA0, DM, IDS, NUTS, DOM.Table51DomainSpec("x", ("nope",), 0.5))
    with pytest.raises(ValueError):
        DOM.table5_1_domain_arrays(-Q, THETA0, DM, IDS, NUTS, _primary()[0])
    rec = _primary()[0].to_record()
    assert rec["threshold_status"] == "research_assumption" and "Table 5-1" in rec["source"]
    assert "not attributed to the source" in rec["rationale"]
    assert DOM.SENSITIVITY_THRESHOLDS == pytest.approx((1 / 3, 1 / 2, 2 / 3))
    # fingerprints differ between variants
    fps = {s.fingerprint() for s in _primary()}
    assert len(fps) == 4


def test_primary_verdict_equals_public_evaluator_diag_t51_style_row():
    # the engine form of the configured diagnostic row: sum_i x_i St_i w_i >= 0 with w = +0.5 (dgc) / -0.5 (others)
    ings = []
    for k, iid in enumerate(IDS):
        rec = ing(iid, float(DM[k]), {"CP": float(THETA0[k, 0]), "starch": float(THETA0[k, 1])},
                  forage=1.0 if iid == "syn_silage" else 0.0)
        ings.append(dataclasses.replace(rec, coefficients={"w": 0.5 if iid == "syn_dgc" else -0.5}))
    row = supply("DIAG-T51-SYN", {"C:w:starch": 1.0}, "ge", 0.0, "kg/d", cls="diagnostic_only", tol=1e-6)
    pr = problem(ings, list(NUTS), [row], {i: 0.1 for i in IDS})
    rng = np.random.default_rng(3)
    S = 400
    theta = np.repeat(THETA0[None], S, axis=0) * (1.0 + rng.uniform(-0.35, 0.35, size=(S, 4, 2)))
    d = np.repeat(DM[None], S, axis=0) * (1.0 + rng.uniform(-0.05, 0.05, size=(S, 4)))
    d[:, 3] = 1.0
    ev = evaluate(Q, theta, d, pr.compiled, d_hat=DM)
    k = list(ev.constraint_ids).index("DIAG-T51-SYN")
    r = DOM.table5_1_domain_arrays(Q, theta, d, IDS, NUTS, _primary()[0])
    assert 0 < int(ev.violated[:, k].sum()) < S                     # both sides of the threshold occur
    np.testing.assert_array_equal(ev.violated[:, k], r.status == "not_assessable")
    np.testing.assert_allclose(ev.margin[:, k], r.starch_margin_kg_d, rtol=0, atol=1e-12)
    v, u = DOM.rows_verdict(ev, ["DIAG-T51-SYN"])
    np.testing.assert_array_equal(v, ev.violated[:, k])
    assert not u.any()


# =================================================================================================
# FIX3_DEF (review round 3 red team, D6/C and D5): an event with Table 5-1 rows read under the premise
# =================================================================================================

def test_premise_conditioned_event_hand_counts_and_the_old_reading_is_only_a_variant():
    # six synthetic states; statuses as in the red-team case: most "not violated" states lie outside the domain
    theta = np.stack([_theta(), _theta([(1, 0.40)]), _theta([(1, 0.40)]), np.zeros((4, 2)), _theta(),
                      _theta([(1, 0.40)])])
    r = DOM.table5_1_domain_arrays(Q, theta, np.repeat(DM[None], 6, axis=0), IDS, NUTS, _primary()[0])
    assert list(r.status) == ["in_domain_conditional", "not_assessable", "not_assessable", "undefined",
                              "in_domain_conditional", "not_assessable"]
    ev_v = np.array([False, False, True, False, True, False])     # state 2: a violated row outside the domain
    ev_u = np.array([False, False, False, False, False, True])    # state 5: a member undefined, nothing violated
    pc = r.premise_conditioned(ev_v, ev_u, contains_table51_rows=True)
    # violated: states 2 and 4 (a violated T row outside the domain still counts as violated)
    np.testing.assert_array_equal(pc.violated, ev_v)
    # premise adds states 1 (not_assessable) and 3 (undefined): "not violated" there rests on unreadable T rows
    np.testing.assert_array_equal(pc.premise_unknown, [False, True, False, True, False, False])
    np.testing.assert_array_equal(pc.unknown, [False, True, False, True, False, True])
    s = pc.summary()
    assert s["n_states"] == 6 and s["rule"] == DOM.PREMISE_CONDITIONED_RULE
    main = s[DOM.PREMISE_CONDITIONED]
    old = s[DOM.REGARDLESS_OF_PREMISE]
    assert (main["n_violated"], main["n_unknown"], main["n_unknown_premise_not_met"]) == (2, 3, 2)
    assert main["rate_lower"] == pytest.approx(2 / 6) and main["rate_upper"] == pytest.approx(5 / 6)
    assert (old["n_violated"], old["n_unknown"]) == (2, 1)
    assert old["rate_lower"] == main["rate_lower"] and old["rate_upper"] == pytest.approx(3 / 6)
    assert "not a headline number" in old["report_as"]
    # in-domain readable counts (states 0 and 4); no conditional rate is formed
    assert s["in_domain_readable"] == {"n_states_in_domain": 2, "n_violated": 1, "n_unknown": 0, "n_not_violated": 1,
                                       "note": s["in_domain_readable"]["note"]}
    assert not any("rate" in k for k in s["in_domain_readable"])
    assert sum(b["n_states"] for b in s["by_domain_status"].values()) == 6
    assert s["by_domain_status"]["not_assessable"] == {"n_states": 3, "violated": 1, "unknown_as_evaluated": 1,
                                                       "not_violated_as_evaluated": 1}
    # an event without a Table 5-1 row is not touched by the premise
    no_t = DOM.premise_conditioned_event(ev_v, ev_u, r, contains_table51_rows=False)
    np.testing.assert_array_equal(no_t.unknown, ev_u)
    assert not no_t.premise_unknown.any()
    # malformed inputs are refused
    with pytest.raises(ValueError):
        DOM.premise_conditioned_event(ev_v[:5], ev_u[:5], r, contains_table51_rows=True)
    with pytest.raises(ValueError):
        DOM.premise_conditioned_event(ev_v, ev_v, r, contains_table51_rows=True)


def test_premise_conditioned_event_on_evaluator_rows_matches_a_direct_count():
    # the red-team shape: every state satisfies the T rows, almost all lie outside the domain -> the event rate is
    # [0, share outside] under the premise, [0, 0] if the T rows were read regardless of the premise
    rng = np.random.default_rng(11)
    S = 300
    theta = np.repeat(THETA0[None], S, axis=0).copy()
    theta[:, 1, 1] = rng.uniform(0.20, 0.45, size=S)              # silage starch moves the dgc share around 0.5
    d = np.repeat(DM[None], S, axis=0)
    r = DOM.table5_1_domain_arrays(Q, theta, d, IDS, NUTS, _primary()[0])
    n_in = int((r.status == DOM.DOMAIN_IN).sum())
    assert 0 < n_in < S
    pc = r.premise_conditioned(np.zeros(S, bool), np.zeros(S, bool), contains_table51_rows=True)
    s = pc.summary()
    assert s[DOM.REGARDLESS_OF_PREMISE]["rate_upper"] == 0.0
    assert s[DOM.PREMISE_CONDITIONED]["rate_lower"] == 0.0
    assert s[DOM.PREMISE_CONDITIONED]["rate_upper"] == pytest.approx((S - n_in) / S)


def test_primary_threshold_is_at_least_half_exactly_one_half_is_in_domain():
    # dgc supplies exactly half of the diet starch: margin 0 -> met (">= 0.50", not a strict majority)
    ids = ("syn_dgc", "syn_silage")
    q = np.array([1.0, 1.0])
    th = np.array([[0.1, 0.5], [0.1, 0.5]])
    r = DOM.table5_1_domain_arrays(q, th, np.array([1.0, 1.0]), ids, NUTS,
                                   DOM.Table51DomainSpec("T51-DGC-0.50", ("syn_dgc",), 0.5))
    assert r.starch_margin_kg_d[0] == 0.0 and r.status[0] == DOM.DOMAIN_IN
    assert "at least half" in DOM.THRESHOLD_RATIONALE and "not a plurality" in DOM.THRESHOLD_RATIONALE
    assert "after" in DOM.THRESHOLD_RATIONALE                      # timing of the sensitivity variants disclosed
