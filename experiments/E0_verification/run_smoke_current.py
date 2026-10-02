#!/usr/bin/env python3
"""Current-code re-run of the real-parameter E0 smoke (review round 2, R6 item 4) -- SMOKE ONLY.

K7_smoke_devcase, 2026-09-25.  Not a pilot, not an official run; nothing here may enter the manuscript.

What it does
------------
1. Checks the interpreter against ``environment.lock.json`` and takes the full code manifest
   (``ration_reliability.io.code_manifest``: src, experiments, scripts, tests, configs, lock files)
   *before* the runs, so that a lock mismatch or an incomplete manifest is visible up front.
2. Runs the unchanged smoke driver ``run_smoke_nasem_dryrun.py`` (K3 version: uncertainty factory,
   declared family rule ``TN_MM`` with fallback ``BETA_MM``, one world for opt and test) once per
   variant (``--mineral-caps on`` = primary smoke, ``off`` = negative control), each in its own
   process, so every run writes its own run record (new run_id; the old smoke directories are kept
   untouched as history).
3. Post-run checks of every new run (written to ``post_run_check_addendum.json`` next to the run
   record; the addendum is *not* listed in the run record, it is written after the run):
   exit status and run type; every output hash of the run record equals the file on disk; the run
   record carries the full code identity (code manifest, run identity, dirty diff) and the
   environment fingerprint with the lock check; opt and test draws come from the one factory model
   (model fingerprint); every stochastic marginal of the restricted moment table is
   ``matched`` (target moments reproduced; values stay in ``data/restricted_local``).
4. Restricted-value hygiene (DECISIONS D-226): ``rations.csv`` (q and planned DM share -> the NASEM DM
   means in one division) is moved byte-for-byte to ``data/restricted_local/smoke/<run_id>/`` and a
   ``rations_redacted.csv`` without ``planned_dm_share`` is written; ``restricted_move_addendum.json``
   records the move.  A public per-marginal fit table without target/achieved moments
   (``marginal_fit_status.csv``: family, fit status, dimensionless relative errors) is written.

Usage::

    PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 \
        experiments/E0_verification/run_smoke_current.py            # both variants, seed 1103
    ... --check-only    # in-memory run of the driver (--check-only), no file written, no run id
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ration_reliability.hashing import file_sha256  # noqa: E402
from ration_reliability.io import (  # noqa: E402
    ENVIRONMENT_LOCK_FILE,
    check_environment_against_lock,
    code_manifest,
    environment_fingerprint,
    utc_now,
)

DRIVER = REPO / "experiments" / "E0_verification" / "run_smoke_nasem_dryrun.py"
#: the previous smoke pair of each variant (kept untouched as history; CLOSE1 updated it from the W2 pair
#: 103019Z / 103021Z to the K7 pair that the CLOSE1 re-run supersedes)
HISTORICAL_SMOKE = {"on": "smoke-20260924T190623Z-ece10eb7", "off": "smoke-20260924T190625Z-03ed1cae"}
#: dimensionless columns of the restricted moment table that may be public (no target/achieved values)
PUBLIC_FIT_COLUMNS = ("ingredient_id", "item", "requested_family", "family", "fit_status", "rel_mean_shift",
                      "sd_rel_error", "flags")


def _last_json(text: str) -> dict:
    """The driver prints one JSON object last; HiGHS may print lines before it."""
    lines = text.splitlines()
    for k in range(len(lines)):
        if lines[k].startswith("{"):
            try:
                return json.loads("\n".join(lines[k:]))
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON summary found in the driver output")


def _driver_cmd(args, variant: str, check_only: bool) -> list[str]:
    cmd = [sys.executable, str(DRIVER), "--seed", str(args.seed), "--n-opt", str(args.n_opt),
           "--n-test", str(args.n_test), "--mineral-caps", variant, "--family", args.family,
           "--fallback", args.fallback]
    if check_only:
        cmd.append("--check-only")
    return cmd


def _check_run(run_id: str) -> dict:
    """Post-run checks of one new smoke run (see module docstring, step 3)."""
    out_dir = REPO / "results" / "smoke" / run_id
    res_dir = REPO / "data" / "restricted_local" / "smoke" / run_id
    rec = json.loads((out_dir / "run_record.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(ok, what, detail=None):
        checks.append({"ok": bool(ok), "check": what, "detail": detail})

    check(rec.get("exit_status") == 0, "run record exit_status == 0", rec.get("exit_status"))
    check(rec.get("run_type") == "smoke", "run_type == smoke", rec.get("run_type"))
    mism = []
    for p, h in rec.get("output_hashes", {}).items():
        pp = Path(p)
        if not pp.is_file() or file_sha256(pp) != h:
            mism.append(p)
    check(not mism, "every output hash of the run record equals the file on disk (before any move)", mism)
    for key in ("code_manifest_sha256", "run_identity_sha256", "src_tree_sha256", "code_commit_or_hash"):
        check(bool(rec.get(key)), f"run record has {key}", rec.get(key))
    man = rec.get("code_manifest") or {}
    check(bool(man.get("complete")), "code manifest complete (src, experiments, scripts, tests, configs, locks)",
          {"n_files": len(man.get("files", [])), "manifest_sha256": man.get("manifest_sha256")})
    cf = rec.get("code_fingerprint") or {}
    check("dirty_diff_hash" in rec, "dirty_diff_hash recorded (git working tree incl. untracked files)",
          {"git_head": cf.get("git_head"), "git_dirty": cf.get("git_dirty"), "dirty_diff_hash": rec.get("dirty_diff_hash"),
           "untracked_file_count": cf.get("untracked_file_count"),
           "git_manifest_crosscheck": (cf.get("git_manifest_crosscheck") or {}).get("status")})
    env = rec.get("environment") or {}
    lock = env.get("lock_check") or {}
    check(lock.get("status") == "matches_lock", "environment lock check == matches_lock",
          {"status": lock.get("status"), "matched_environment": lock.get("matched_environment"),
           "fingerprint_sha256": env.get("fingerprint_sha256")})
    ex = rec.get("extra") or {}
    um = ex.get("uncertainty_model") or {}
    check(um.get("model_fingerprint") and um.get("model_fingerprint") == ex.get("model_fingerprint"),
          "opt/test world = the factory model recorded in the run record", um.get("model_fingerprint"))
    fr = um.get("family_rule") or {}
    check(fr.get("primary_family") == "TN_MM" and list(fr.get("fallback_families") or []) == ["BETA_MM"],
          "declared family rule TN_MM with fallback BETA_MM", fr)
    check(um.get("purpose") == "smoke" and um.get("moment_semantics") == "target_marginal_moments",
          "purpose smoke, target_marginal_moments (not the naive diagnostic model)",
          {"purpose": um.get("purpose"), "moment_semantics": um.get("moment_semantics")})
    # per-marginal fit (restricted table; only dimensionless columns are copied to the public table)
    mt = res_dir / "factory_moment_diagnostics.csv"
    rows = list(csv.DictReader(mt.open(encoding="utf-8"))) if mt.is_file() else []
    status_counts: dict[str, int] = {}
    fam_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["fit_status"]] = status_counts.get(r["fit_status"], 0) + 1
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    max_shift = max((abs(float(r["rel_mean_shift"])) for r in rows), default=None)
    max_sd = max((abs(float(r["sd_rel_error"])) for r in rows), default=None)
    check(rows and set(status_counts) == {"matched"}, "every stochastic marginal matched its target moments",
          {"n_cells": len(rows), "fit_status_counts": status_counts, "family_counts": fam_counts,
           "max_abs_rel_mean_shift": max_shift, "max_abs_sd_rel_error": max_sd})
    public_fit = out_dir / "marginal_fit_status.csv"
    with open(public_fit, "x", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(PUBLIC_FIT_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in PUBLIC_FIT_COLUMNS})
    # D-226: rations.csv (q + planned DM share) -> restricted_local; redacted copy without the share
    moved = None
    src = out_dir / "rations.csv"
    if src.is_file():
        h_before = file_sha256(src)
        rec_h = rec.get("output_hashes", {}).get(str(src))
        dst = res_dir / "rations.csv"
        if dst.exists():
            raise FileExistsError(dst)
        with open(src, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = [c for c in reader.fieldnames if c != "planned_dm_share"]
            body = [{c: r[c] for c in cols} for r in reader]
        red = out_dir / "rations_redacted.csv"
        with open(red, "x", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(body)
        shutil.move(str(src), str(dst))
        moved = {"from": str(src.relative_to(REPO)), "to": str(dst.relative_to(REPO)), "sha256": h_before,
                 "sha256_after_move": file_sha256(dst), "sha256_equals_run_record_output_hash": h_before == rec_h,
                 "method": "shutil.move on the same file system (bytes unchanged)",
                 "redacted_copy": {"path": str(red.relative_to(REPO)), "sha256": file_sha256(red),
                                   "columns_kept": cols, "columns_dropped": ["planned_dm_share"],
                                   "n_rows": len(body)},
                 "why": "q + planned DM share + planned DM supply give the NASEM 2021 DM means in one division "
                        "(DECISIONS D-226); NASEM values may only live in data/restricted_local/",
                 "remaining_pathways": "BLOCKERS B-147 (nutrient_profile.csv and several q vectors still allow "
                                       "reconstruction); results/smoke is excluded from every public package"}
        (out_dir / "restricted_move_addendum.json").write_text(
            json.dumps({"run_id": run_id, "written_at_utc": utc_now(), "task": "K7_smoke_devcase (D-226 applied)",
                        "moved_file": moved, "not_in_run_record": True}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    check(moved is not None and moved["sha256_equals_run_record_output_hash"],
          "rations.csv moved to restricted_local unchanged (hash = run record output hash)", moved and moved["to"])
    summary = {"method_summary": ex.get("method_summary"), "in_sample_opt": ex.get("in_sample_opt"),
               "validation_report": ex.get("validation_report"), "n_smoke_placeholders": ex.get("n_smoke_placeholders"),
               "model_fingerprint": ex.get("model_fingerprint"), "uncertainty_model": um,
               "opt_fingerprint": ex.get("opt_fingerprint"), "test_fingerprint": ex.get("test_fingerprint"),
               "rng_streams": rec.get("rng_streams"), "tolerances": rec.get("tolerances"),
               "solver_version": rec.get("solver_version"), "elapsed_s": ex.get("elapsed_s")}
    add = {"run_id": run_id, "written_at_utc": utc_now(), "task": "K7_smoke_devcase post-run checks",
           "run_record_sha256": file_sha256(out_dir / "run_record.json"),
           "not_in_run_record": "written after the run; the run record lists the driver's own outputs",
           "all_checks_pass": all(c["ok"] for c in checks), "checks": checks,
           "identity": {"code_manifest_sha256": rec.get("code_manifest_sha256"),
                        "run_identity_sha256": rec.get("run_identity_sha256"),
                        "src_tree_sha256": rec.get("src_tree_sha256"),
                        "code_commit_or_hash": rec.get("code_commit_or_hash"),
                        "dirty_diff_hash": rec.get("dirty_diff_hash"),
                        "environment_fingerprint_sha256": env.get("fingerprint_sha256"),
                        "lock_status": lock.get("status")},
           "public_marginal_fit_table": {"path": str(public_fit.relative_to(REPO)), "sha256": file_sha256(public_fit),
                                         "columns": list(PUBLIC_FIT_COLUMNS),
                                         "note": "target/achieved moments stay in the restricted "
                                                 "factory_moment_diagnostics.csv and uncertainty_model_metadata_full.json"},
           "summary_from_run_record": summary}
    (out_dir / "post_run_check_addendum.json").write_text(json.dumps(add, indent=2, ensure_ascii=False, default=str)
                                                          + "\n", encoding="utf-8")
    return add


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1103)
    ap.add_argument("--n-opt", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--family", default="TN_MM")
    ap.add_argument("--fallback", default="BETA_MM")
    ap.add_argument("--variants", default="on,off")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    pre = {"started_at_utc": utc_now(),
           "environment_lock_check": check_environment_against_lock(REPO / ENVIRONMENT_LOCK_FILE,
                                                                    current=environment_fingerprint())}
    man = code_manifest(REPO)
    pre["code_manifest_before_runs"] = {"manifest_sha256": man["manifest_sha256"], "complete": man.get("complete"),
                                        "n_files": len(man.get("files", []))}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = {"pre": pre, "runs": {}}
    for v in variants:
        cmd = _driver_cmd(args, v, args.check_only)
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        entry = {"command": " ".join(cmd[1:]), "exit_code": p.returncode, "stderr_tail": p.stderr[-2000:]}
        if p.returncode == 0:
            s = _last_json(p.stdout)
            entry["driver_summary"] = s
            if not args.check_only:
                entry["post_run_checks"] = _check_run(s["run_id"])
                entry["historical_run_kept"] = HISTORICAL_SMOKE.get(v)
        out["runs"][v] = entry
    out["finished_at_utc"] = utc_now()
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    ok = all(e["exit_code"] == 0 and (args.check_only or e["post_run_checks"]["all_checks_pass"])
             for e in out["runs"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
