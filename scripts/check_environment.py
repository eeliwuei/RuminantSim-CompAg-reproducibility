#!/usr/bin/env python3
"""Check (or write) the environment lock and print the code content manifest.

Second-review R6 (2026-09-25).  Nothing is installed, upgraded or removed: versions are read with
``importlib.metadata`` and the imported modules' ``__version__``.

Usage (from the project root)::

    python scripts/check_environment.py                       # compare with environment.lock.json
    python scripts/check_environment.py --json-out report.json
    python scripts/check_environment.py --code-manifest-out manifest.json   # works without .git
    python scripts/check_environment.py --write-lock --label <label>        # create both lock files
    python scripts/check_environment.py --append-environment --label <label>  # add this interpreter
    python scripts/check_environment.py --require-group holder_verification   # also require PyMuPDF == lock
    python scripts/check_environment.py --update-optional-groups              # record optional groups (FIX_C)
    python scripts/check_environment.py --compare-run-record results/pilot/<run_id>/run_record.json

Optional groups (FIX_C): packages that only a holder-side script imports (``holder_verification`` =
PyMuPDF for ``scripts/verify_restricted_inputs.py``) are locked per environment entry, outside the
fingerprint core, and pinned in ``requirements-lock.txt`` as ``#[optional:<group>] name==version``
comment lines.  They are always reported; ``--require-group`` makes a mismatch fail.

``--compare-run-record`` compares the current code manifest with the ``code_manifest`` stored in a
run record, file by file and per area (src, experiments, scripts, tests, configs, root files), and
prints the ``src/experiments/scripts/tests`` subset digests of both.  Use it to see whether a run's
evidence is still about the current code (R6-5), and to compare an external package whose
``configs/`` are redacted with an internal run record (only the code subset can match; see
``docs/package_scopes.md``).  It is informational and does not change the exit status.

Exit status: 0 environment matches the lock (and every required group); 1 mismatch; 2 lock missing
or invalid; 3 refused (a file already exists, a label is taken, or bad arguments).

Every run record built by ``ration_reliability.io.build_run_record`` performs the same core check and
stores the environment fingerprint; pilot/official records are refused unless the check reports
``matches_lock``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ration_reliability.io.run_record import (  # noqa: E402
    ENVIRONMENT_LOCK_FILE,
    OPTIONAL_DISTRIBUTION_GROUPS,
    REQUIREMENTS_LOCK_FILE,
    check_environment_against_lock,
    code_fingerprint,
    code_manifest,
    code_manifest_subset_sha256,
    compare_code_manifests,
    environment_fingerprint,
    environment_lock_document,
    environment_lock_entry,
    optional_group_versions,
    requirements_lock_text,
    utc_now,
    validate_environment_lock,
)

EXIT = {"matches_lock": 0, "mismatch": 1, "lock_missing": 2, "lock_invalid": 2}
CODE_SUBSET = ("src", "experiments", "scripts", "tests")


def _write_new(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it (the old lock stays in git history)")
    path.write_text(text, encoding="utf-8")


def _update_optional_groups(lock_path: Path, req_path: Path, fp: dict, opt: dict) -> int:
    """Record the running interpreter's optional-group versions in its (core-matching) lock entry."""
    if not lock_path.is_file():
        print(f"{lock_path.name} not found; use --write-lock first", file=sys.stderr)
        return 2
    try:
        doc = json.loads(lock_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"existing lock is unreadable: {exc}", file=sys.stderr)
        return 2
    errs = validate_environment_lock(doc)
    if errs:
        print(f"existing lock is invalid: {errs}", file=sys.stderr)
        return 2
    entry = next((e for e in doc["environments"] if e["fingerprint_sha256"] == fp["fingerprint_sha256"]), None)
    if entry is None:
        print("the running interpreter matches no environment entry of the lock (core differs); optional groups are "
              "recorded only for a locked core -- use --append-environment first", file=sys.stderr)
        return 1
    new = environment_lock_entry(entry["label"], fp, optional=opt)["optional_groups"]
    if doc.get("optional_distribution_groups") is not None and entry.get("optional_groups") == new:
        print(json.dumps({"optional_groups": new, "changed": False}, ensure_ascii=False))
        return 0
    doc["optional_distribution_groups"] = {g: [{"name": n, "import_name": i, "reason": r} for n, i, r in dists]
                                           for g, dists in OPTIONAL_DISTRIBUTION_GROUPS.items()}
    old = entry.get("optional_groups")
    entry["optional_groups"] = new
    doc.setdefault("amendments", []).append({"at": utc_now(), "label": entry["label"],
                                             "optional_groups_before": old, "optional_groups_after": new,
                                             "by": "scripts/check_environment.py --update-optional-groups"})
    errs = validate_environment_lock(doc)
    if errs:
        print(f"refusing to write an invalid lock: {errs}", file=sys.stderr)
        return 2
    lock_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    req_path.write_text(requirements_lock_text(doc), encoding="utf-8")
    print(json.dumps({"optional_groups": new, "changed": True, "label": entry["label"]}, ensure_ascii=False))
    return 0


def _compare_run_record(repo: Path, record_path: Path) -> dict:
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    old = rec.get("code_manifest")
    if not isinstance(old, dict) or "files" not in old:
        return {"run_record": record_path.name, "status": "no_code_manifest_in_record",
                "note": "records written before R6 (2026-09-25) carry only src_tree_sha256; compare that field instead"}
    now = code_manifest(repo)
    cmp = compare_code_manifests(old, now)
    return {"run_record": record_path.name, "run_id": rec.get("run_id"), "status": "compared",
            "identical": cmp["identical"], "area_identical": cmp["area_identical"],
            "differences_by_area": cmp["differences_by_area"], "changed": cmp["changed"],
            "only_in_run_record": cmp["only_in_first"], "only_now": cmp["only_in_second"],
            "code_subset_sha256": {"run_record": code_manifest_subset_sha256(old, CODE_SUBSET)["subset_sha256"],
                                   "now": code_manifest_subset_sha256(now, CODE_SUBSET)["subset_sha256"],
                                   "areas": list(CODE_SUBSET)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=REPO, help="project root (default: this script's project)")
    ap.add_argument("--json-out", type=Path, help="write the check report here (refuses to overwrite)")
    ap.add_argument("--code-manifest-out", type=Path, help="write the code content manifest here (refuses to overwrite)")
    ap.add_argument("--write-lock", action="store_true", help="write environment.lock.json and requirements-lock.txt")
    ap.add_argument("--append-environment", action="store_true",
                    help="append the running interpreter to an existing environment.lock.json")
    ap.add_argument("--update-optional-groups", action="store_true",
                    help="record the running interpreter's optional-group versions in its lock entry")
    ap.add_argument("--require-group", action="append", default=[], choices=sorted(OPTIONAL_DISTRIBUTION_GROUPS),
                    help="fail (exit 1) unless this optional group matches the lock (repeatable)")
    ap.add_argument("--compare-run-record", type=Path, action="append", default=[],
                    help="compare the current code manifest with this run record's code_manifest (repeatable)")
    ap.add_argument("--label", help="environment label for --write-lock / --append-environment")
    ap.add_argument("--force", action="store_true", help="with --write-lock: replace existing lock files")
    args = ap.parse_args(argv)
    repo = args.repo.resolve()
    lock_path, req_path = repo / ENVIRONMENT_LOCK_FILE, repo / REQUIREMENTS_LOCK_FILE

    if sum([args.write_lock, args.append_environment, args.update_optional_groups]) > 1:
        print("--write-lock, --append-environment and --update-optional-groups are exclusive", file=sys.stderr)
        return 3
    fp = environment_fingerprint()
    opt = optional_group_versions()
    try:
        if args.write_lock:
            if not args.label:
                print("--write-lock needs --label", file=sys.stderr)
                return 3
            existing = [p.name for p in (lock_path, req_path) if p.exists()]
            if existing and not args.force:  # check both before writing either (no half-written lock)
                raise FileExistsError(f"{existing} exist; pass --force to replace them "
                                      "(the old lock stays in git history)")
            doc = environment_lock_document(args.label, fingerprint=fp, optional_versions=opt)
            _write_new(lock_path, json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n", args.force)
            _write_new(req_path, requirements_lock_text(doc), args.force)
        elif args.append_environment:
            if not args.label:
                print("--append-environment needs --label", file=sys.stderr)
                return 3
            if not lock_path.is_file():
                print(f"{lock_path.name} not found; use --write-lock first", file=sys.stderr)
                return 2
            try:
                doc = json.loads(lock_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                print(f"existing lock is unreadable: {exc}", file=sys.stderr)
                return 2
            errs = validate_environment_lock(doc)
            if errs:
                print(f"existing lock is invalid: {errs}", file=sys.stderr)
                return 2
            if any(e["label"] == args.label for e in doc["environments"]):
                print(f"label {args.label!r} already listed", file=sys.stderr)
                return 3
            if any(e["fingerprint_sha256"] == fp["fingerprint_sha256"] for e in doc["environments"]):
                print("this interpreter's fingerprint is already listed", file=sys.stderr)
                return 3
            doc["environments"].append(environment_lock_entry(
                args.label, fp, optional=opt if doc.get("optional_distribution_groups") is not None else None))
            doc.setdefault("amendments", []).append({"at": utc_now(), "added_label": args.label,
                                                     "by": "scripts/check_environment.py --append-environment"})
            lock_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
                                 encoding="utf-8")
        elif args.update_optional_groups:
            rc = _update_optional_groups(lock_path, req_path, fp, opt)
            if rc != 0:
                return rc
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    groups = sorted(OPTIONAL_DISTRIBUTION_GROUPS)
    check = check_environment_against_lock(lock_path, current=fp, optional_groups=groups, current_optional=opt)
    report = {
        "checked_at": utc_now(),
        "repo": repo.name,
        "environment_check": check,
        "environment_fingerprint": fp,
        "optional_group_versions": opt,
    }
    if args.compare_run_record:
        report["run_record_comparisons"] = [_compare_run_record(repo, p) for p in args.compare_run_record]
    if args.code_manifest_out:
        man = code_manifest(repo)
        report["code_manifest_summary"] = {k: man[k] for k in ("manifest_sha256", "file_count", "file_count_by_dir",
                                                               "missing_dirs", "missing_root_files", "complete")}
        report["code_fingerprint"] = code_fingerprint(repo, manifest=man)
        if args.code_manifest_out.exists():
            print(f"{args.code_manifest_out} exists; refusing to overwrite", file=sys.stderr)
            return 3
        args.code_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.code_manifest_out.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json_out:
        if args.json_out.exists():
            print(f"{args.json_out} exists; refusing to overwrite", file=sys.stderr)
            return 3
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    og = check.get("optional_groups", {})
    summary = {"status": check["status"], "matched_environment": check["matched_environment"],
               "current_fingerprint_sha256": check["current_fingerprint_sha256"],
               "mismatches": check["mismatches"], "errors": check["errors"],
               "optional_groups": {g: {k: v[k] for k in ("status", "locked", "current")} for g, v in og.items()},
               "hazards": fp["hazards"],
               "code_manifest_sha256": report.get("code_manifest_summary", {}).get("manifest_sha256")}
    if args.compare_run_record:
        summary["run_record_comparisons"] = [{k: c.get(k) for k in ("run_record", "run_id", "status", "identical",
                                                                    "differences_by_area", "code_subset_sha256")}
                                             for c in report["run_record_comparisons"]]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    rc = EXIT[check["status"]]
    if rc == 0 and any(og.get(g, {}).get("status") != "matches_lock" for g in args.require_group):
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
