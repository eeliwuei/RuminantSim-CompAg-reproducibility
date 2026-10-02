"""Official-run plan batch 2, tests 9 and 11b: membership by p-bar, coverage per distinct ration, the per-method report
of an empty comparable set, the frontier kept out of every set, the scored grid candidates, and the matched-cost
envelope.

All on synthetic rows (no problem, draw, solve or data file): the functions under test are pure table operations of
``experiments/E1_cost_reliability/run_endpoint_ablation.py``.

9.  ``comparable_sets(..., membership_stat="cp_upper")`` admits a ration only if its one-sided CP upper bound p-bar is
    <= alpha (a rate_upper <= alpha with p-bar > alpha is not a member; the development default ``rate_upper`` still
    admits it); coverage is reported per entry and per distinct ration (the same M0 ration in two cells counts once;
    entries without a ration are counted separately); the diagnostic frontier is never an entry or a member; the M1
    variant without the DM margin is a sensitivity line (not an entry) when ``m1_dm_off_is_entry=False``;
    ``per_method_report`` keeps every field when the set is empty (rates, p-bar, target_met, cost labelled
    ``not_a_comparable_cost``, the grid minimum and its cost, the frontier diagnostic).
    ``candidate_rations`` scores each distinct ration once, covers the whole M3b (k, Gamma) grid through the selection
    sink, skips diagnostic runs; no selection or membership function reads candidate rows.
11b. ``matched_cost_curve``: the lower envelope of main rate_upper and p-bar is non-increasing in the cost budget, uses
    only candidates within the budget, and is reported relative to the cell's M0 cost.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"


def _load_driver():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("run_endpoint_ablation_reporting_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load_driver()
PLAN = AB.MAIN_EVENT_PLAN_DOMAIN


def _row(cell, label, world, has, *, cost=None, rate=None, cp=None, lo=None, ok=True, kind="method", q=None,
         method_id=None, event="main_reference"):
    mid = method_id or ("M0_nominal" if label.endswith(":M0") else "M2_joint_chance_saa" if ":M2[" in label
                        else "M1_safety_margin" if ":M1[" in label else "M3a_box_robust")
    r = {"cell_id": cell, "label": label, "ration_kind": kind, "evaluation_world": world if has else None,
         "has_ration": has, "structural_ok": ok if has else None, "cost_usd_per_head_d": cost if has else None,
         "q_hash": (q or f"h-{cell}-{label}") if has else None, "method_id": mid, "sd_arm": "SD-H0",
         "objective_arm": "MAIN9", "selection_status": "selected" if has else "not_met",
         "target_alpha": AB._alpha_of(label), f"{AB.MAIN_EVENT_T51_IGNORED}_rate_upper": rate,
         f"{AB.MAIN_EVENT_T51_IGNORED}_cp_upper": cp, "t51_primary_not_assessable_share": 0.0 if has else None}
    for ev in ("main_reference", PLAN):
        r.update({f"{ev}_rate_upper": rate if has else None, f"{ev}_cp_upper": cp if has else None,
                  f"{ev}_rate_lower": lo if has else None})
    return r


def _rows():
    return [
        # the same M0 ration (same q_hash) in two cells of one objective arm
        _row("A", "A:M0", "SD-H0", True, cost=6.0, rate=0.9, cp=0.91, lo=0.5, q="h-M0"),
        _row("B", "B:M0", "SD-H0", True, cost=6.0, rate=0.9, cp=0.91, lo=0.5, q="h-M0"),
        # rate_upper <= alpha but p-bar > alpha: member only under the development statistic
        _row("A", "A:M1[relative,apply_to_dm=True]@alpha=0.05", "SD-H0", True, cost=6.4, rate=0.048, cp=0.052,
             lo=0.04),
        # p-bar <= alpha: member under both
        _row("A", "A:M2[N=128]@alpha=0.05", "SD-H0", True, cost=6.6, rate=0.03, cp=0.034, lo=0.02),
        # no ration: counted in the entry denominator, separately in the distinct-ration count
        _row("B", "B:M2[N=128]@alpha=0.05", "SD-H0", False),
        # M1 without the DM margin: an entry in development, a sensitivity line officially
        _row("B", "B:M1[relative,apply_to_dm=False]@alpha=0.05", "SD-H0", True, cost=6.2, rate=0.01, cp=0.012,
             lo=0.005),
        # diagnostic frontier with p-bar 0: never an entry or a member
        _row("A", "A:DIAG-frontier:M2[N=128,alpha_train=m*/N=3/128]", "SD-H0", True, cost=6.1, rate=0.0, cp=0.0003,
             lo=0.0, kind="diagnostic_frontier"),
    ]


def _set(out, world="SD-H0", a=0.05):
    return next(o for o in out if o["evaluation_world"] == world and o["target_alpha"] == a)


# =================================================================================================
# 9 membership, coverage, frontier, sensitivity line
# =================================================================================================

def test_membership_uses_p_bar_when_requested_and_rate_upper_by_default():
    dev = _set(AB.comparable_sets(_rows()))
    off = _set(AB.comparable_sets(_rows(), membership_stat="cp_upper"))
    m1 = "A:M1[relative,apply_to_dm=True]@alpha=0.05"
    assert m1 in [m["label"] for m in dev["members"]] and dev["membership_stat"] == "rate_upper"
    assert m1 not in [m["label"] for m in off["members"]] and off["membership_stat"] == "cp_upper"
    assert "Clopper-Pearson" in off["membership_statistic"]
    assert "A:M2[N=128]@alpha=0.05" in [m["label"] for m in off["members"]]
    for m in off["members"]:
        assert m["membership_cp_upper"] <= 0.05 and m["cp_upper_le_alpha"] is True
    with pytest.raises(ValueError):
        AB.comparable_sets(_rows(), membership_stat="point")


def test_membership_can_use_the_plan_level_main_event():
    rows = _rows()
    for r in rows:                       # the plan-level reading fails the M2 ration, the per-state reading passes it
        if r["label"] == "A:M2[N=128]@alpha=0.05":
            r[f"{PLAN}_cp_upper"] = 0.2
    per_state = _set(AB.comparable_sets(rows, membership_stat="cp_upper"))
    plan = _set(AB.comparable_sets(rows, membership_stat="cp_upper", main_event=PLAN))
    assert "A:M2[N=128]@alpha=0.05" in [m["label"] for m in per_state["members"]]
    assert "A:M2[N=128]@alpha=0.05" not in [m["label"] for m in plan["members"]]
    assert plan["membership_event"] == PLAN


def test_coverage_per_entry_and_per_distinct_ration_frontier_excluded():
    s = _set(AB.comparable_sets(_rows(), membership_stat="cp_upper"))
    # entries: A:M0, B:M0, A:M1(dm on), A:M2, B:M2 (no ration), B:M1(dm off) -> 6; the frontier is not an entry
    assert s["n_entries_attempted"] == 6
    labels = [m["label"] for m in s["members"]]
    assert not any(":DIAG-" in lab for lab in labels)
    assert set(labels) == {"A:M2[N=128]@alpha=0.05", "B:M1[relative,apply_to_dm=False]@alpha=0.05"}
    assert s["coverage"] == s["coverage_entries"] == pytest.approx(2 / 6)
    # distinct rations with a ration: h-M0 (counted once), A:M1, A:M2, B:M1 -> 4; members 2; one entry without ration
    assert s["n_entries_with_ration"] == 5 and s["n_entries_without_ration"] == 1
    assert s["n_distinct_rations_attempted"] == 4 and s["n_distinct_member_rations"] == 2
    assert s["coverage_distinct_rations"] == pytest.approx(2 / 4)


def test_m1_without_dm_margin_is_a_sensitivity_line_not_an_entry_when_declared():
    rows = _rows()
    s = _set(AB.comparable_sets(rows, membership_stat="cp_upper", m1_dm_off_is_entry=False))
    assert s["n_entries_attempted"] == 5 and s["m1_dm_off_is_entry"] is False
    assert [m["label"] for m in s["members"]] == ["A:M2[N=128]@alpha=0.05"]
    assert [x["label"] for x in s["sensitivity_lines"]] == ["B:M1[relative,apply_to_dm=False]@alpha=0.05"]
    assert s["sensitivity_lines"][0]["would_meet_rule"] is True
    by = {r["label"]: r for r in rows}
    assert by["B:M1[relative,apply_to_dm=False]@alpha=0.05"]["in_comparable_set"] == "sensitivity_line_not_an_entry"


def _cands():
    """Grid candidates of two entries (descriptive rows as candidate_rations returns them)."""
    def c(cell, entry, val, cost, rate, cp, q, mid="M2_joint_chance_saa", world="SD-H0", has=True):
        d = {"cell_id": cell, "entry_label": entry, "method_id": mid, "method_family": AB.method_family(mid, entry),
             "param_name": "alpha_train", "param_value": val, "fixed_params": None, "has_ration": has,
             "evaluation_world": world if has else None, "q_hash": q if has else None, "structural_ok": True if has
             else None, "cost_usd_per_head_d": cost if has else None}
        for ev in ("main_reference", PLAN):
            d.update({f"{ev}_rate_upper": rate, f"{ev}_cp_upper": cp, f"{ev}_rate_lower": rate})
        return d
    e = "B:M2[N=128]@alpha=0.05"
    return [c("B", e, 0.05, 6.5, 0.2, 0.21, "q1"), c("B", e, 0.0375, 6.9, 0.12, 0.125, "q2"),
            c("B", e, 0.025, None, None, None, None, has=False), c("B", e, 0.0125, 7.4, 0.12, 0.13, "q3"),
            c("B", "B:M0", None, 6.0, 0.9, 0.91, "h-M0", mid="M0_nominal")]


def test_per_method_report_of_an_empty_set_keeps_every_field():
    rows = [r for r in _rows() if r["label"] in ("A:M0", "B:M0", "B:M2[N=128]@alpha=0.05",
                                                 "A:DIAG-frontier:M2[N=128,alpha_train=m*/N=3/128]")]
    rows.append(_row("A", "A:M3a@alpha=0.05", "SD-H0", True, cost=7.0, rate=0.4, cp=0.41, lo=0.3))
    diag = {"A": {"min_violation": [{"N": 128, "min_violations_lower": 2, "min_violations_upper": 3}],
                  "frontier_points": [{"N": 128, "m_star_upper": 3, "m_star_lower": 2, "status": "optimal"}]},
            "B": {"min_violation": [{"N": 128, "min_violations_lower": 5, "min_violations_upper": 9}],
                  "frontier_points": []}}
    comp = _set(AB.comparable_sets([dict(r) for r in rows], membership_stat="cp_upper"))
    assert comp["n_members"] == 0 and "per_method_report" in comp["empty_set_meaning"]
    rep = AB.per_method_report(rows, _cands(), diagnostics=diag, membership_stat="cp_upper")
    sub = [x for x in rep if x["evaluation_world"] == "SD-H0" and x["target_alpha"] == 0.05]
    assert {x["label"] for x in sub} == {"A:M0", "B:M0", "B:M2[N=128]@alpha=0.05", "A:M3a@alpha=0.05"}
    assert not any(":DIAG-" in x["label"] for x in rep)                       # the frontier is never an entry
    for x in rep:
        assert set(AB.PER_METHOD_REPORT_COLUMNS) - {"run_id", "output_label"} <= set(x), sorted(
            set(AB.PER_METHOD_REPORT_COLUMNS) - set(x))
        assert x["in_comparable_set"] is False and x["comparable_set_size"] == 0
    m3a = next(x for x in sub if x["label"] == "A:M3a@alpha=0.05")
    assert m3a["rate_lower"] == 0.3 and m3a["rate_upper"] == 0.4 and m3a["cp_upper"] == 0.41
    assert m3a["target_met"] is False and m3a["cost_label"] == "not_a_comparable_cost"
    # frontier diagnostic of the cell (largest N for a non-M2 entry): m*/N bounds and the frontier ration's rates
    assert m3a["frontier_N"] == 128 and m3a["frontier_m_star_lower_over_N"] == pytest.approx(2 / 128)
    assert m3a["frontier_m_star_upper_over_N"] == pytest.approx(3 / 128)
    assert m3a["frontier_rate_upper"] == 0.0 and m3a["frontier_cp_upper"] == 0.0003
    assert "never a member" in m3a["frontier_label"]
    # the entry without a ration: status kept, no rate, no cost, grid minimum over its declared candidates
    m2 = next(x for x in sub if x["label"] == "B:M2[N=128]@alpha=0.05")
    assert m2["has_ration"] is False and m2["rate_upper"] is None and m2["cost_usd_per_head_d"] is None
    assert m2["cost_label"] is None and m2["target_met"] is None and m2["selection_status"] == "not_met"
    assert m2["grid_n_candidates"] == 4 and m2["grid_n_candidates_with_ration"] == 3
    # minimum rate_upper 0.12 reached by two candidates: the cheaper one is reported with its cost
    assert m2["grid_min_rate_upper"] == 0.12 and m2["grid_min_rate_upper_cost_usd_per_head_d"] == 6.9
    assert m2["grid_min_rate_upper_param"] == "alpha_train=0.0375" and "descriptive" in m2["grid_min_label"]
    assert m2["frontier_status"] in ("no_frontier_ration",) and m2["frontier_m_star_upper_over_N"] == pytest.approx(
        9 / 128)
    # M0 enters every alpha
    assert {x["target_alpha"] for x in rep if x["label"] == "A:M0"} == {0.05, 0.10, 0.01}


def test_per_method_report_labels_member_costs_only():
    rep = AB.per_method_report(_rows(), [], membership_stat="cp_upper")
    sub = {x["label"]: x for x in rep if x["evaluation_world"] == "SD-H0" and x["target_alpha"] == 0.05}
    assert sub["A:M2[N=128]@alpha=0.05"]["cost_label"] == "comparable_set_member"
    assert sub["A:M2[N=128]@alpha=0.05"]["target_met"] is True
    assert sub["A:M1[relative,apply_to_dm=True]@alpha=0.05"]["cost_label"] == "not_a_comparable_cost"
    assert sub["A:M1[relative,apply_to_dm=True]@alpha=0.05"]["target_met"] is False
    assert sub["A:M0"]["comparable_set_size"] == 2
    assert sub["A:M1[relative,apply_to_dm=True]@alpha=0.05"]["grid_min_label"].startswith("no scored grid candidate")


# =================================================================================================
# candidate_rations: every declared grid candidate, each distinct ration scored once; descriptive only
# =================================================================================================

def _res(q, ids=("x", "y")):
    if q is None:
        return SimpleNamespace(has_solution=False, decision=None, status="proven_infeasible")
    return SimpleNamespace(has_solution=True, status="optimal",
                           decision=SimpleNamespace(ingredient_ids=ids, q_as_fed=np.asarray(q, dtype=float)))


def _sel(param, pairs):
    cands = tuple(SimpleNamespace(value=v, status=("optimal" if q is not None else "proven_infeasible"),
                                  rate_upper=0.1, screen_statistic=0.11, meets_screen=False, solve_result=_res(q))
                  for v, q in pairs)
    return SimpleNamespace(param_name=param, candidates=cands, status="not_met")


def test_candidate_rations_scores_each_distinct_ration_once_and_covers_the_m3b_grid():
    runs = [SimpleNamespace(spec=SimpleNamespace(name="C:M0", method_id="M0_nominal"), selection=None,
                            result=_res([1.0, 2.0])),
            SimpleNamespace(spec=SimpleNamespace(name="C:M1[relative,apply_to_dm=True]@alpha=0.05",
                                                 method_id="M1_safety_margin"),
                            selection=_sel("k", [(0.0, [1.0, 2.0]), (0.05, [1.5, 2.0]), (0.1, None)]), result=None),
            SimpleNamespace(spec=SimpleNamespace(name="C:M3b[factory box]@alpha=0.05", method_id="M3b_budget_robust"),
                            selection=_sel("gamma", [(0, [9.0, 9.0])]), result=None),
            SimpleNamespace(spec=SimpleNamespace(name="C:DIAG-frontier:M2[N=128,alpha_train=m*/N=1/128]",
                                                 method_id="M2_joint_chance_saa"), selection=None,
                            result=_res([3.0, 3.0]))]
    sink = [{"entry_label": "C:M3b[factory box]@alpha=0.05", "fixed_params": {"k_box": k},
             "selection": _sel("gamma", [(0, [1.0, 2.0]), (1, [2.0, 2.0])])} for k in (1.0, 2.0)]
    seen: list = []

    def score(dec):
        seen.append(tuple(dec.q_as_fed))
        return {w: {"q_hash": f"q{tuple(dec.q_as_fed)}", "cost_usd_per_head_d": float(sum(dec.q_as_fed)),
                    "structural_ok": True, "main_reference_rate_upper": 0.5} for w in ("SD-H0", "SD-S2")}
    rows = AB.candidate_rations(runs, score, cell_id="C", extra_selections=sink)
    # distinct rations: (1,2) [M0, M1 k=0, M3b k=1/2 Gamma=0], (1.5,2), (2,2) -> scored once each; frontier skipped
    assert sorted(seen) == [(1.0, 2.0), (1.5, 2.0), (2.0, 2.0)]
    occ = {(r["entry_label"], r["param_value"], r["fixed_params"]) for r in rows}
    assert len(occ) == 1 + 3 + 4                          # M0, M1 x3 (one without ration), M3b 2 k x 2 Gamma
    assert not any(":DIAG-" in r["entry_label"] for r in rows)
    # the M3b entry's own (sb) selection is replaced by the per-k sink, not added to it
    assert not any(r["entry_label"].startswith("C:M3b") and r["fixed_params"] is None for r in rows)
    none = [r for r in rows if not r["has_ration"]]
    assert len(none) == 1 and none[0]["evaluation_world"] is None and none[0]["cost_usd_per_head_d"] is None
    assert all(sorted(r["_q"]) == ["q_x", "q_y"] for r in rows if r["has_ration"])
    assert sum(r["first_occurrence_of_q_hash"] for r in rows if r["has_ration"] and r["evaluation_world"] == "SD-H0") == 3


def test_selection_and_membership_code_never_reads_candidate_rows():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("_membership", "_coverage", "comparable_sets", "run_cell", "cell_rows"):
        names = {x.id for x in ast.walk(fns[name]) if isinstance(x, ast.Name)} | {
            x.attr for x in ast.walk(fns[name]) if isinstance(x, ast.Attribute)}
        assert not names & {"candidate_rations", "cand", "per_method_report", "matched_cost_curve"}, name
    main_src = ast.get_source_segment(DRIVER.read_text(encoding="utf-8"), fns["main"])
    assert main_src.index("comparable_sets(ends") < main_src.index("candidate_rations(")
    drv_src = (REPO / "experiments" / "E0_verification" / "run_dev_case_v1.py").read_text(encoding="utf-8")
    assert "candidate_rations" not in drv_src


# =================================================================================================
# 11b matched-cost curve
# =================================================================================================

def test_matched_cost_envelope_is_monotone_and_within_budget():
    rng = np.random.default_rng(7)
    cands = []
    for i in range(40):
        fam = ("M2_joint_chance_saa", "M1_safety_margin")[i % 2]
        entry = "E:M2[N=128]@alpha=0.05" if fam.startswith("M2") else "E:M1[relative,apply_to_dm=True]@alpha=0.05"
        cost = 6.0 * (1.0 + float(rng.uniform(0.0, 0.6)))
        rate = float(rng.uniform(0.0, 1.0))
        cands.append({"cell_id": "E", "entry_label": entry, "method_id": fam, "method_family": AB.method_family(fam, entry),
                      "has_ration": True, "evaluation_world": "SD-H0", "q_hash": f"q{i}", "structural_ok": True,
                      "cost_usd_per_head_d": cost, "main_reference_rate_upper": rate,
                      "main_reference_cp_upper": min(1.0, rate + 0.01)})
    cands.append({"cell_id": "E", "entry_label": "E:M0", "method_id": "M0_nominal", "method_family": "M0_nominal",
                  "has_ration": True, "evaluation_world": "SD-H0", "q_hash": "q-m0", "structural_ok": True,
                  "cost_usd_per_head_d": 6.0, "main_reference_rate_upper": 0.95, "main_reference_cp_upper": 0.96})
    # a structurally failing candidate and another world never enter
    cands.append({**cands[0], "q_hash": "q-bad", "structural_ok": False, "cost_usd_per_head_d": 6.0,
                  "main_reference_rate_upper": 0.0})
    cands.append({**cands[0], "q_hash": "q-other", "evaluation_world": "SD-S2", "main_reference_rate_upper": 0.0})
    curve = AB.matched_cost_curve(cands, "SD-H0")
    grid = AB.ABLATION_CONFIG["matched_cost_curve"]["relative_cost_grid"]
    assert grid == sorted(grid) and grid[0] == 0.0
    for fam in ("M2_joint_chance_saa[N=128]", "M1_safety_margin[apply_to_dm=True]", "M0_nominal"):
        pts = [r for r in curve if r["method_family"] == fam]
        assert [r["relative_cost_grid_point"] for r in pts] == grid
        ok = [r for r in pts if r["status"] == "ok"]
        rates = [r["min_rate_upper"] for r in ok]
        cps = [r["min_cp_upper"] for r in ok]
        assert all(b <= a for a, b in zip(rates, rates[1:])), fam      # non-increasing envelope
        assert all(b <= a for a, b in zip(cps, cps[1:])), fam
        assert all(r["min_rate_upper_relative_cost"] <= r["relative_cost_grid_point"] + 1e-12 for r in ok)
        n = [r["n_candidates_within_cost"] for r in pts]
        assert all(b >= a for a, b in zip(n, n[1:]))
        assert all(r["min_rate_upper"] != 0.0 for r in ok)             # the bad / other-world candidates never enter
        assert all(r["status"] == "no_candidate_within_cost" for r in pts if r["n_candidates_within_cost"] == 0)
    # brute force at every grid point
    fam_rows = [c for c in cands if c["method_family"] == "M2_joint_chance_saa[N=128]" and c["structural_ok"]
                and c["evaluation_world"] == "SD-H0"]
    for r in (x for x in curve if x["method_family"] == "M2_joint_chance_saa[N=128]" and x["status"] == "ok"):
        within = [c["main_reference_rate_upper"] for c in fam_rows
                  if c["cost_usd_per_head_d"] / 6.0 - 1.0 <= r["relative_cost_grid_point"] + 1e-12]
        assert r["min_rate_upper"] == min(within)
    m0 = [r for r in curve if r["method_family"] == "M0_nominal"]
    assert all(r["status"] == "ok" and r["min_rate_upper_relative_cost"] == 0.0 for r in m0)
    assert {k for r in curve for k in r} <= set(AB.MATCHED_COST_COLUMNS)      # every field has a column
