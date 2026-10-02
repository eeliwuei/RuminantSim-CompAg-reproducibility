#!/usr/bin/env python3
"""Evaluation-only replay of the dev_case_v1 development run ``pilot-20260924T205621Z-93d8654c`` -- SMOKE (CLOSE1).

What it shows, and what it does not
-----------------------------------
It shows that the rations **saved** by that run, scored with the **current** code (problem built from the same
restricted inputs, uncertainty factory, the same seed and stream ids, the one public evaluator), give the same cost,
joint and per-constraint violation counts, rates, Clopper-Pearson bounds and margins as the run recorded.  It does
**not** show that the solves reproduce under the current code: no LP / MILP is solved here (user rule, B-436: the Mac
only runs smoke, ``--dry-run`` and evaluation-only replays); the full re-run of dev_case_v1 waits for a server
(B-427, B-436, B-110).  Nothing here is a result; nothing may enter the manuscript.

Steps
-----
1. Preconditions: the environment matches the lock (otherwise refused, exit 2); the driver's ``check_inputs`` (build
   report passed, every registered build input unchanged); the declared run configuration of the current driver hashes
   to the value the run saved; the saved ration and residual files are the bytes the run record hashed.
2. Rebuild, without solving: the H0 problem (``load_problem``), the energy linearisation, the H0 / C1 / S2
   uncertainty specs -> factory models -> ``EnergyColumnModel`` worlds (functions of the driver itself), the S2 problem
   (``s2_problem_cfg`` + ``build_problem``); draw opt / validation / test with ``RandomStreams(1103)`` at the recorded
   sizes; stream ids, draw fingerprints and world fingerprints must equal the saved ones.
3. For every saved ration (``rations_full.csv``, ``s2_rations_full.csv``, ``c1_rations_full.csv`` in
   ``data/restricted_local/pilot/<run_id>/``: q and the decision-time d_hat), score it with
   ``evaluation.evaluate_drawset`` on the recorded test stream and compare field by field with the saved tables:
   method summaries (public ``method_summary.csv``, ``s2_method_summary.csv``, ``c1_method_summary.csv``), per-constraint
   residual tables (restricted ``constraint_residuals_full.csv``, ``s2_constraint_residuals_full.csv``) and the H0
   rations scored in the C1 world (``c1_scenario.json``).  Rows without a ration must stay without one.
4. Output (not with ``--check-only``): ``results/pilot/<replay_run_id>/replay_comparison.json`` (method labels, metric
   names, absolute differences, pass flags, code manifest fingerprints -- no ingredient-level value and no metric
   value), ``NOT_FOR_MANUSCRIPT.md`` and a run record with ``run_type = smoke`` (``RUN_TYPES`` has no ``replay``).

Comparison rule: counts, strings and booleans exactly; floats pass when ``|replay - saved| <= 1e-12 * max(1,
|saved|)`` (the saved CSV/JSON floats are Python ``repr`` values, i.e. exact doubles, so an identical code path gives
difference 0); ``exact`` records bitwise equality separately.

Usage::

    cd <project root>
    PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 experiments/E0_verification/replay_dev_case_eval.py
    ... --check-only    # same computation, prints the summary, writes nothing
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import run_dev_case_v1 as DRV  # noqa: E402  (the driver's own builders; nothing in it is solved here)
from ration_reliability.evaluation import evaluate, evaluate_drawset  # noqa: E402
from ration_reliability.evaluation.stats import clopper_pearson, mc_standard_error, one_sided_upper  # noqa: E402
from ration_reliability.hashing import file_sha256, stable_hash  # noqa: E402
from ration_reliability.io import (  # noqa: E402
    ENVIRONMENT_LOCK_FILE,
    build_problem,
    build_run_record,
    check_environment_against_lock,
    code_manifest,
    environment_fingerprint,
    load_problem,
    make_run_id,
    utc_now,
    write_run_record,
)
from ration_reliability.io.run_record import code_manifest_subset_sha256, compare_code_manifests  # noqa: E402
from ration_reliability.nutrition import energy as E  # noqa: E402
from ration_reliability.optimization.highs import solver_version_string  # noqa: E402
from ration_reliability.uncertainty import RandomStreams, build_uncertainty_model  # noqa: E402

REPLAYED_RUN = "pilot-20260924T205621Z-93d8654c"
PUB = Path("results") / "pilot" / REPLAYED_RUN
RES = Path("data") / "restricted_local" / "pilot" / REPLAYED_RUN
REL_TOL = 1e-12
SCHEMA = "ration_reliability.dev_case_replay/1"
#: method-summary fields re-computed from the evaluation (solver fields -- status, gap, wall time -- are not replayed)
SUMMARY_FLOATS = ("cost_usd_per_head_d", "joint_rate", "joint_cp_upper_one_sided_95", "joint_cp_two_sided_95_lower",
                  "joint_cp_two_sided_95_upper", "joint_mc_se")
SUMMARY_COUNTS = ("n_test", "joint_n_violated", "joint_n_unknown")
SUMMARY_EXACT = ("structural_ok", "test_stream_id")
RESIDUAL_FLOATS = ("planned_nominal_margin", "violation_rate", "cp_upper_one_sided_95", "mc_se",
                   "mean_deficit_given_violation", "max_deficit", "margin_q01", "margin_q05", "margin_q50")
RESIDUAL_COUNTS = ("n_test", "n_violated")


# ------------------------------------------------------------------------------------------------
# comparison helpers
# ------------------------------------------------------------------------------------------------
def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _num(v: Any) -> Optional[float]:
    if _missing(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        raise ValueError("boolean is not a numeric replay metric")
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("malformed numeric replay metric") from exc
    if not math.isfinite(x):
        raise ValueError("non-finite replay metric")
    return x


def _invalid_comparison(reason: str) -> dict:
    return {"abs_diff": reason, "exact": False, "pass": False}


def cmp_float(replay: Any, saved: Any) -> dict:
    try:
        a, b = _num(replay), _num(saved)
    except ValueError as exc:
        return _invalid_comparison(str(exc))
    if a is None or b is None:
        ok = a is None and b is None
        return {"abs_diff": None if ok else "presence differs", "exact": ok, "pass": ok}
    d = abs(a - b)
    # A non-finite difference is always a failure, even when a supplied bound is huge.
    return {"abs_diff": d if math.isfinite(d) else "non-finite difference", "exact": a == b,
            "pass": math.isfinite(d) and d <= REL_TOL * max(1.0, abs(b))}


def cmp_count(replay: Any, saved: Any) -> dict:
    if _missing(replay) or _missing(saved):
        ok = _missing(replay) and _missing(saved)
        return {"abs_diff": None if ok else "presence differs", "exact": ok, "pass": ok}
    try:
        if isinstance(replay, (bool, np.bool_)) or isinstance(saved, (bool, np.bool_)):
            raise ValueError("boolean count")
        a, b = Decimal(str(replay)), Decimal(str(saved))
        if not (a.is_finite() and b.is_finite() and a >= 0 and b >= 0
                and a == a.to_integral_value() and b == b.to_integral_value()):
            raise ValueError("invalid integer count")
    except (ValueError, InvalidOperation, TypeError):
        return _invalid_comparison("counts must be finite nonnegative integers")
    # Keep the comparison exact even beyond the binary64 exact-integer range.
    ia, ib = int(a), int(b)
    return {"abs_diff": abs(ia - ib), "exact": ia == ib, "pass": ia == ib}


def cmp_exact(replay: Any, saved: Any) -> dict:
    ok = str(replay) == str(saved)
    return {"abs_diff": None, "exact": ok, "pass": ok, "kind": "string/boolean equality"}


def _bool_str(v: Any) -> str:
    return "True" if v is True else ("False" if v is False else str(v))


# ------------------------------------------------------------------------------------------------
# saved tables
# ------------------------------------------------------------------------------------------------
def _csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_rations(path: Path) -> dict[str, dict]:
    """Read saved executed q without silent duplicate or status overwrite."""
    out: dict[str, dict] = {}
    for r in _csv_rows(path):
        label = r["label"]
        if not label:
            raise ValueError(f"{path.name}: missing ration label")
        e = out.setdefault(label, {"q": {}, "d_hat": {}, "status": r["status"]})
        if r["status"] != e["status"]:
            raise ValueError(f"{path.name}: inconsistent status within one ration")
        iid = r["ingredient_id"]
        if iid:
            if iid in e["q"]:
                raise ValueError(f"{path.name}: duplicate ration/ingredient row")
            q, dh = _num(r["q_as_fed_kg_per_head_d"]), _num(r["d_hat"])
            if q is None or dh is None or q < 0 or not 0 < dh <= 1:
                raise ValueError(f"{path.name}: invalid saved q or decision-time DM")
            e["q"][iid], e["d_hat"][iid] = q, dh
    return out


def _unique_rows(path: Path, keys: tuple[str, ...]) -> dict:
    out = {}
    for r in _csv_rows(path):
        key = r[keys[0]] if len(keys) == 1 else tuple(r[k] for k in keys)
        if key in out:
            raise ValueError(f"{path.name}: duplicate key in saved evidence ({', '.join(keys)})")
        out[key] = r
    return out


def read_by_label(path: Path) -> dict[str, dict]:
    return _unique_rows(path, ("label",))


def read_residuals(path: Path) -> dict[tuple[str, str], dict]:
    return _unique_rows(path, ("label", "constraint_id"))


# ------------------------------------------------------------------------------------------------
# re-evaluation (the driver's summary_rows / residual_rows formulas, from the evaluator only)
# ------------------------------------------------------------------------------------------------
def replay_ration(problem, test, entry: dict) -> tuple[dict, dict]:
    ids = list(problem.ingredient_ids)
    if set(entry["q"]) != set(ids):
        raise SystemExit(f"saved ration has other ingredients than the problem: {sorted(set(entry['q']) ^ set(ids))}")
    q = np.array([entry["q"][i] for i in ids], dtype=float)
    dh = np.array([entry["d_hat"][i] for i in ids], dtype=float)
    ev = evaluate_drawset(q, test, problem.compiled, d_hat=dh, prices=problem.prices)
    s = ev.summary()
    j = s["joint"]
    n = s["n_draws"]
    k = j["n_violated"] + j["n_unknown"]
    lo, hi = clopper_pearson(k, n, 0.95)
    summ = {"cost_usd_per_head_d": ev.cost, "n_test": n, "joint_n_violated": j["n_violated"],
            "joint_n_unknown": j["n_unknown"], "joint_rate": k / n,
            "joint_cp_upper_one_sided_95": one_sided_upper(k, n, 0.95), "joint_cp_two_sided_95_lower": lo,
            "joint_cp_two_sided_95_upper": hi, "joint_mc_se": mc_standard_error(k / n, n),
            "structural_ok": bool(s["structural_ok"]), "test_stream_id": s["draw_stream_id"]}
    th0 = problem.nominal_theta()
    evn = evaluate(q, th0[None], dh[None], problem.compiled, d_hat=dh, prices=problem.prices)
    nom = {cid: float(evn.margin[0, kk]) for kk, cid in enumerate(evn.constraint_ids)}
    smarg = np.atleast_2d(evn.structural_margin)
    for kk, cid in enumerate(evn.structural_ids):
        nom[cid] = float(smarg[0, kk])
    per: dict[str, dict] = {}
    for cid in list(evn.structural_ids) + list(evn.constraint_ids):
        row = {"planned_nominal_margin": nom.get(cid), "n_test": None, "n_violated": None, "violation_rate": None,
               "cp_upper_one_sided_95": None, "mc_se": None, "mean_deficit_given_violation": None,
               "max_deficit": None, "margin_q01": None, "margin_q05": None, "margin_q50": None}
        pc = s["per_constraint"].get(cid)
        if pc is not None:
            nd, nv = pc["n_defined"], pc["n_violated"]
            mq = pc["margin_quantiles"] or {}
            row.update({"n_test": nd, "n_violated": nv, "violation_rate": nv / nd if nd else None,
                        "cp_upper_one_sided_95": one_sided_upper(nv, nd, 0.95) if nd else None,
                        "mc_se": mc_standard_error(nv / nd, nd) if nd else None,
                        "mean_deficit_given_violation": pc["mean_deficit_given_violation"],
                        "max_deficit": pc["max_deficit"], "margin_q01": mq.get(0.01), "margin_q05": mq.get(0.05),
                        "margin_q50": mq.get(0.5)})
        per[cid] = row
    return summ, per


def compare_summary(summ: dict, saved: dict) -> dict:
    out = {m: cmp_float(summ[m], saved.get(m)) for m in SUMMARY_FLOATS}
    out.update({m: cmp_count(summ[m], saved.get(m)) for m in SUMMARY_COUNTS})
    out["structural_ok"] = cmp_exact(_bool_str(summ["structural_ok"]), saved.get("structural_ok"))
    out["test_stream_id"] = cmp_exact(summ["test_stream_id"], saved.get("test_stream_id"))
    return out


def compare_residuals(label: str, per: dict, saved: dict) -> tuple[dict, list[str]]:
    out, missing = {}, []
    saved_cids = {cid for (lab, cid) in saved if lab == label}
    for cid, row in per.items():
        s = saved.get((label, cid))
        if s is None:
            missing.append(cid)
            continue
        c = {m: cmp_float(row[m], s.get(m)) for m in RESIDUAL_FLOATS}
        c.update({m: cmp_count(row[m], s.get(m)) for m in RESIDUAL_COUNTS})
        out[cid] = c
    missing += sorted(saved_cids - set(per))
    return out, missing


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="compute and print the summary; write nothing")
    args = ap.parse_args()
    os.chdir(REPO)
    # Report missing restricted inputs before checking the environment or opening historical outputs.
    # This is read-only and deliberately does not rebuild, change the lock or manufacture missing data.
    from ration_reliability.build.preflight import Requirement, require_dev_case_inputs
    history = [PUB / n for n in ("run_record.json", "run_config.json", "method_summary.csv",
                                "s2_method_summary.csv", "c1_method_summary.csv", "c1_scenario.json",
                                "uncertainty_model.json", "s2_scenario.json")]
    history += [RES / n for n in ("rations_full.csv", "s2_rations_full.csv", "c1_rations_full.csv",
                                 "constraint_residuals_full.csv", "s2_constraint_residuals_full.csv",
                                 "s2_problem.yaml")]
    extra = [Requirement(p.as_posix(), "driver_restricted_input" if str(p).startswith("data/")
                         else "driver_tracked_config", str(p).startswith("data/"),
                         "Restore the exact historical run bytes from an authorised backup; do not regenerate "
                         "them under current code and call them the original run.") for p in history]
    require_dev_case_inputs(REPO, driver="run_dev_case_v1", extra=extra)
    started = utc_now()
    t0 = time.perf_counter()
    command = " ".join([sys.executable] + sys.argv)
    lock = check_environment_against_lock(REPO / ENVIRONMENT_LOCK_FILE, current=environment_fingerprint())
    if lock.get("status") != "matches_lock":
        print(f"environment does not match the lock ({lock.get('status')}); a bitwise replay is refused",
              file=sys.stderr)
        return 2
    man_now = code_manifest(REPO)
    pre: list[dict] = []

    def pre_check(ok, what, detail=None):
        pre.append({"check": what, "ok": bool(ok), "detail": detail})

    rec_run = json.loads((PUB / "run_record.json").read_text(encoding="utf-8"))
    run_cfg = json.loads((PUB / "run_config.json").read_text(encoding="utf-8"))
    pre_check(rec_run.get("run_id") == REPLAYED_RUN and rec_run.get("exit_status") == 0,
              "replayed run record: run id and exit status 0", rec_run.get("run_id"))
    pre_check(stable_hash(DRV.RUN_CONFIG) == run_cfg["run_config_sha256"],
              "declared run configuration of the current driver hashes to the saved run_config_sha256", None)
    saved_files = [RES / n for n in ("rations_full.csv", "s2_rations_full.csv", "c1_rations_full.csv",
                                     "constraint_residuals_full.csv", "s2_constraint_residuals_full.csv")] + \
        [PUB / n for n in ("method_summary.csv", "s2_method_summary.csv", "c1_method_summary.csv", "c1_scenario.json",
                           "uncertainty_model.json", "s2_scenario.json")]
    oh = rec_run.get("output_hashes") or {}
    changed = [str(p) for p in saved_files if oh.get(str(p)) != file_sha256(p)]
    pre_check(not changed, "saved rations / residual / summary files are the bytes hashed by the run record", changed)

    # ---- rebuild (no solve)
    inputs = DRV.check_inputs()
    pre_check(True, "dev_case_v1 build report passed and every registered build input unchanged (driver check_inputs)",
              inputs["build_summary"])
    dh_rec = rec_run.get("data_files") or {}
    pre_check(dh_rec.get(str(DRV.PROBLEM_YAML)) == file_sha256(DRV.PROBLEM_YAML),
              "problem YAML = the one the run hashed", None)
    problem, _rep = load_problem(DRV.PROBLEM_YAML, mode="pilot")
    cfg0 = yaml.safe_load(DRV.PROBLEM_YAML.read_text(encoding="utf-8"))
    lin = DRV.load_linearisation(problem)
    cells = DRV.read_cells()
    cells_sha = file_sha256(DRV.CELLS_CSV)
    sc = DRV.RUN_CONFIG["streams"]
    n_opt, n_val, n_test = sc["opt"], sc["validation"], sc["test"]
    um = json.loads((PUB / "uncertainty_model.json").read_text(encoding="utf-8"))
    s2j = json.loads((PUB / "s2_scenario.json").read_text(encoding="utf-8"))
    c1j = json.loads((PUB / "c1_scenario.json").read_text(encoding="utf-8"))
    worlds: dict[str, dict] = {}
    streams = RandomStreams(DRV.RUN_CONFIG["seed"])
    # H0
    spec = DRV.build_h0_spec(problem.ingredient_ids, cells, cells_sha)
    fm = build_uncertainty_model(spec, model_id="DEV_CASE_V1_H0_factory_TN_MM")
    w = E.EnergyColumnModel(fm, lin)
    DRV.wrapped_nominal_check(w, problem)
    wd = w.draw_world(streams, n_opt=n_opt, n_validation=n_val, n_test=n_test)
    worlds["H0"] = {"problem": problem, "test": wd["test"]}
    pre_check(w.fingerprint() == um["H0_world"]["world_fingerprint"], "H0 world fingerprint = saved", None)
    pre_check(fm.fingerprint() == um["H0_world"]["base_model_fingerprint"], "H0 base model fingerprint = saved", None)
    pre_check(spec.fingerprint() == um["H0_world"]["spec_fingerprint"], "H0 spec fingerprint = saved", None)
    for s in ("opt", "validation", "test"):
        pre_check(wd[s].stream_id == um["streams"][s] and wd[s].fingerprint == um["draw_fingerprints"][s],
                  f"H0 {s} stream id and draw fingerprint = saved", wd[s].stream_id)
    # C1 (same streams, declared correlation scenario)
    spec_c1 = DRV.build_c1_spec(problem.ingredient_ids, cells, cells_sha)
    fm_c1 = build_uncertainty_model(spec_c1, model_id="DEV_CASE_V1_C1_factory_TN_MM")
    w_c1 = E.EnergyColumnModel(fm_c1, lin)
    DRV.wrapped_nominal_check(w_c1, problem)
    wd_c1 = w_c1.draw_world(streams, n_opt=n_opt, n_validation=n_val, n_test=n_test)
    worlds["C1"] = {"problem": problem, "test": wd_c1["test"]}
    pre_check(w_c1.fingerprint() == c1j["world"]["world_fingerprint"], "C1 world fingerprint = saved", None)
    pre_check(all(wd_c1[s].stream_id == c1j["streams"][s] for s in ("opt", "validation", "test")),
              "C1 stream ids = saved (draw fingerprints were not saved for C1)", None)
    # S2 (narrowed scenario problem, same streams)
    cfg_s2, _s2_rec = DRV.s2_problem_cfg(cfg0, problem)
    problem_s2, _rep_s2 = build_problem(cfg_s2, mode="pilot")
    saved_s2 = yaml.safe_load((RES / "s2_problem.yaml").read_text(encoding="utf-8"))
    pre_check(saved_s2 == json.loads(json.dumps(cfg_s2)), "S2 problem rebuilt in memory = the saved restricted "
                                                          "s2_problem.yaml (parsed)", None)
    spec_s2 = DRV.build_s2_spec(problem.ingredient_ids, cells, cells_sha)
    fm_s2 = build_uncertainty_model(spec_s2, model_id="DEV_CASE_V1_S2_factory_TN_MM")
    w_s2 = E.EnergyColumnModel(fm_s2, lin)
    DRV.wrapped_nominal_check(w_s2, problem_s2)
    wd_s2 = w_s2.draw_world(streams, n_opt=n_opt, n_validation=n_val, n_test=n_test)
    worlds["S2"] = {"problem": problem_s2, "test": wd_s2["test"]}
    pre_check(w_s2.fingerprint() == s2j["world"]["world_fingerprint"], "S2 world fingerprint = saved", None)
    for s in ("opt", "validation", "test"):
        pre_check(wd_s2[s].stream_id == s2j["streams"][s] and wd_s2[s].fingerprint == s2j["draw_fingerprints"][s],
                  f"S2 {s} stream id and draw fingerprint = saved", wd_s2[s].stream_id)

    # ---- replay every saved ration
    blocks = [("H0", RES / "rations_full.csv", PUB / "method_summary.csv", RES / "constraint_residuals_full.csv"),
              ("S2", RES / "s2_rations_full.csv", PUB / "s2_method_summary.csv", RES / "s2_constraint_residuals_full.csv"),
              ("C1", RES / "c1_rations_full.csv", PUB / "c1_method_summary.csv", None)]
    entries: list[dict] = []
    for scen, rations_p, summary_p, resid_p in blocks:
        rations = read_rations(rations_p)
        summ_saved = read_by_label(summary_p)
        resid_saved = read_residuals(resid_p) if resid_p is not None else None
        pb, test = worlds[scen]["problem"], worlds[scen]["test"]
        labels = list(dict.fromkeys(list(rations) + list(summ_saved)))
        for lab in labels:
            e = rations.get(lab)
            srow = summ_saved.get(lab, {})
            ent: dict[str, Any] = {"scenario": scen, "label": lab, "method_id": srow.get("method_id")}
            has_q = bool(e and e["q"])
            saved_has_eval = bool(srow.get("n_test"))
            if not has_q:
                ok = not saved_has_eval and lab in summ_saved and e is not None
                ent.update({"has_ration": False, "consistency": {
                    "no_ration_in_saved_rations_and_no_evaluation_in_saved_summary": {"exact": ok, "pass": ok}}})
                entries.append(ent)
                continue
            summ, per = replay_ration(pb, test, e)
            ent["has_ration"] = True
            ent["summary_metrics"] = compare_summary(summ, srow) if lab in summ_saved else \
                {"row_present_in_saved_summary": {"exact": False, "pass": False}}
            if resid_saved is not None:
                pc, missing = compare_residuals(lab, per, resid_saved)
                ent["per_constraint"] = pc
                ent["per_constraint_rows_missing"] = missing
            entries.append(ent)
    # H0 rations scored in the C1 world (c1_scenario.json)
    h0r = read_rations(RES / "rations_full.csv")
    for lab, saved in c1j["H0_rations_in_C1_world"].items():
        summ, _per = replay_ration(problem, worlds["C1"]["test"], h0r[lab])
        entries.append({"scenario": "H0_ration_in_C1_world", "label": lab, "method_id": None, "has_ration": True,
                        "summary_metrics": {"joint_rate": cmp_float(summ["joint_rate"], saved["joint_rate"]),
                                            "cp_upper_one_sided_95": cmp_float(summ["joint_cp_upper_one_sided_95"],
                                                                               saved["cp_upper_one_sided_95"])}})

    # ---- instrument self-check (negative controls): the comparison must see a 1e-9 relative change of one ration and
    #      a change of test world; otherwise an all-pass would say nothing
    m0 = h0r["H0:M0"]
    m0_saved = read_by_label(PUB / "method_summary.csv")["H0:M0"]
    pert = {"q": {i: v * (1.0 + 1e-9) for i, v in m0["q"].items()}, "d_hat": dict(m0["d_hat"]), "status": m0["status"]}
    s_pert, _ = replay_ration(problem, worlds["H0"]["test"], pert)
    s_other, _ = replay_ration(problem, worlds["S2"]["test"], m0)
    c_pert = compare_summary(s_pert, m0_saved)
    c_other = compare_summary(s_other, m0_saved)
    self_check = {"q_relative_perturbation_1e-9_detected": not c_pert["cost_usd_per_head_d"]["pass"],
                  "other_test_world_detected": not all(c["pass"] for c in c_other.values()),
                  "meaning": "negative controls of the comparison: H0:M0 with q x (1 + 1e-9), and H0:M0 scored on the "
                             "S2 test stream instead of the H0 one, must both fail against the saved H0:M0 row"}
    pre_check(self_check["q_relative_perturbation_1e-9_detected"] and self_check["other_test_world_detected"],
              "instrument self-check: both negative controls are detected", self_check)

    # ---- summary
    flat: list[tuple[str, dict]] = []
    for ent in entries:
        for m, c in (ent.get("summary_metrics") or {}).items():
            flat.append((f"{ent['scenario']}|{ent['label']}|{m}", c))
        for m, c in (ent.get("consistency") or {}).items():
            flat.append((f"{ent['scenario']}|{ent['label']}|{m}", c))
        for cid, mm in (ent.get("per_constraint") or {}).items():
            for m, c in mm.items():
                flat.append((f"{ent['scenario']}|{ent['label']}|{cid}|{m}", c))
    diffs = [c["abs_diff"] for _k, c in flat if isinstance(c.get("abs_diff"), float)]
    fails = [k for k, c in flat if not c["pass"]]
    missing_rows = [(e["scenario"], e["label"], e["per_constraint_rows_missing"]) for e in entries
                    if e.get("per_constraint_rows_missing")]
    pre_fail = [p["check"] for p in pre if not p["ok"]]
    by_scen: dict[str, dict] = {}
    for ent in entries:
        b = by_scen.setdefault(ent["scenario"], {"rations_replayed": 0, "rows_without_ration": 0})
        b["rations_replayed" if ent["has_ration"] else "rows_without_ration"] += 1
    overall = "PASS" if not fails and not pre_fail and not missing_rows else "FAIL"
    cmp_man = compare_code_manifests(rec_run["code_manifest"], man_now)
    summary = {"n_comparisons": len(flat), "n_pass": len(flat) - len(fails), "n_fail": len(fails),
               "n_exact": sum(1 for _k, c in flat if c["exact"]),
               "max_abs_diff": max(diffs) if diffs else None, "all_exact": all(c["exact"] for _k, c in flat),
               "preconditions_failed": pre_fail, "per_constraint_rows_missing": missing_rows,
               "failures_first_20": fails[:20], "by_scenario": by_scen, "overall": overall}
    body = {
        "schema": SCHEMA, "replayed_run_id": REPLAYED_RUN, "run_type": "smoke",
        "run_type_note": "evaluation-only replay recorded as run_type smoke (ration_reliability.io.RUN_TYPES has no "
                         "'replay'); not a pilot, not official, not a re-solve",
        "what_it_shows": "the rations saved by the replayed run, scored by the current code on the recorded test "
                         "streams, reproduce the saved cost, joint / per-constraint violations, Clopper-Pearson bounds "
                         "and margins (evaluation consistency under the current code)",
        "what_it_does_not_show": "that the LP/MILP solves, the validation selections, the diagnostics (minimum "
                                 "violations, conflict map), H1 or the information value reproduce under the current "
                                 "code; the full re-run waits for a server (B-427, B-436)",
        "code_manifest": {"current_sha256": man_now["manifest_sha256"], "current_complete": man_now.get("complete"),
                          "current_n_files": len(man_now.get("files", [])),
                          "current_code_subset_sha256": code_manifest_subset_sha256(man_now),
                          "replayed_run_recorded_sha256": rec_run.get("code_manifest_sha256"),
                          "replayed_run_at_start_sha256": (rec_run.get("extra") or {}).get("code_manifest_at_start"),
                          "identical_to_replayed_run_record": cmp_man["identical"],
                          "differences_by_area": cmp_man["differences_by_area"],
                          "changed_paths": cmp_man["changed"], "only_in_current": cmp_man["only_in_second"],
                          "only_in_replayed_run": cmp_man["only_in_first"]},
        "environment_lock_status": lock.get("status"),
        "tolerance": {"floats": f"|replay - saved| <= {REL_TOL:g} * max(1, |saved|)", "counts_strings_booleans":
                      "exact", "exact_flag": "bitwise equality of the two doubles, reported separately"},
        "preconditions": pre, "summary": summary, "entries": entries,
        "restricted_values": "none: labels, metric names, absolute differences and flags only"}
    print(json.dumps({"check_only": args.check_only, "summary": summary,
                      "preconditions_ok": not pre_fail, "code_manifest_current": man_now["manifest_sha256"],
                      "elapsed_s": time.perf_counter() - t0}, indent=2, default=str))
    if args.check_only:
        return 0 if overall == "PASS" else 1

    run_id = make_run_id("smoke", started, ("dev_case_v1_eval_replay", REPLAYED_RUN, man_now["manifest_sha256"]))
    out_dir = Path("results") / "pilot" / run_id
    if out_dir.exists():
        print(f"{out_dir} exists; refusing to overwrite", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True)
    body["replay_run_id"] = run_id
    cmp_p = out_dir / "replay_comparison.json"
    with open(cmp_p, "x", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    nfm = out_dir / "NOT_FOR_MANUSCRIPT.md"
    with open(nfm, "x", encoding="utf-8") as fh:
        fh.write(f"# {run_id}\n\nEvaluation-only replay (run_type smoke) of the development run {REPLAYED_RUN}: the "
                 "saved rations are re-scored by the current code on the recorded test streams and compared with the "
                 "saved tables. It shows evaluation consistency under the current code, not that the solves "
                 "reproduce; the full re-run of dev_case_v1 waits for a server (B-427, B-436). The directory is under "
                 "results/pilot/ next to the replayed run by task instruction; its run type is smoke. No number here "
                 "may enter a formal result table, the manuscript, an abstract, a figure or a claim ledger. Report: "
                 "reports/dev_case_v1_replay.md.\n")
    data_paths = [DRV.PROBLEM_YAML, DRV.CELLS_CSV, DRV.ENERGY_JSON, DRV.BUILD_REPORT, DRV.BUILD_SCRIPT, DRV.CORE_CSV,
                  RES / "s2_problem.yaml"] + saved_files + [PUB / "run_record.json", PUB / "run_config.json"]
    rec = build_run_record(
        run_type="smoke", command=command, repo_root=REPO, started_at=started, completed_at=utc_now(),
        exit_status=0 if overall == "PASS" else 1,
        rng_streams={"H0_opt": wd["opt"].stream_id, "H0_validation": wd["validation"].stream_id,
                     "H0_test": wd["test"].stream_id, "C1_test": wd_c1["test"].stream_id,
                     "S2_test": wd_s2["test"].stream_id,
                     "note": "opt / validation are drawn only to rebuild the recorded world; only test draws are used"},
        solver_version=f"not called (evaluation-only replay); installed {solver_version_string()}",
        tolerances={"mip_rel_gap": "not_applicable", "comparison_rel_tol": REL_TOL},
        config_paths=[str(p) for p in DRV.CONFIG_PATHS], data_paths=[str(p) for p in data_paths],
        protocol_path=str(DRV.PROTOCOL), output_paths=[str(cmp_p), str(nfm)], is_synthetic=False, run_id=run_id,
        extra={"purpose": "CLOSE1 evaluation-only replay of " + REPLAYED_RUN + " (smoke; not a re-solve; not a result)",
               "replayed_run_id": REPLAYED_RUN, "overall": overall,
               "n_comparisons": summary["n_comparisons"], "n_fail": summary["n_fail"],
               "max_abs_diff": summary["max_abs_diff"], "all_exact": summary["all_exact"],
               "code_manifest_identical_to_replayed_run_record": cmp_man["identical"],
               "elapsed_s": time.perf_counter() - t0, "host_note": "local Mac; evaluation only (seconds)"})
    write_run_record(rec, out_dir / "run_record.json")
    print(json.dumps({"replay_run_id": run_id, "dir": str(out_dir), "overall": overall}, indent=2))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
