#!/usr/bin/env python3
"""Verify the outputs of an official run v2 (or its rehearsal) after they were copied back (official-run plan §5, B6).

Runs on the Mac (delivery host) after ``rsync --checksum`` of the public tables ``results/official/<run_id>/`` and the
restricted mirror ``data/restricted_local/official/<run_id>/``.  Reads files, computes hashes and counts; never prints
a rate, a cost, a ratio, a restricted value or a reserved root value (only paths, statuses, booleans and counts).

Checks
------
1. **sha manifests**: the top-level ``SHA256SUMS`` of the public and of the restricted directory, and of every job
   directory, verify (no changed, missing or unlisted file); every ``DONE`` / ``MERGE_DONE`` names the sha256 of its
   ``SHA256SUMS``; every job of the run plan has a ``DONE``.
2. **run record**: ``run_record.json`` exists; ``run_type`` is ``official`` (rehearsal: ``smoke``); the output label
   is ``official`` (``official_invalidated`` is reported as a failure of this check, with the reason kept in the
   record); the end-of-run identity check says unchanged; every job exited 0; every ``output_hashes`` entry equals
   the file on disk.
3. **no pooled n**: one ``root<k>/`` directory per root of the run; every CSV in it has a ``root_k`` column equal to
   k in every row; every scored row has ``n_test`` equal to the run's test size; no table outside the root
   directories; no column or key named like ``pooled``; the record says ``pooled_across_roots: false``.
4. **diagnostics are never members**: no comparable-set member carries a ``:DIAG-`` label; every diagnostic row has
   ``ration_kind = diagnostic`` and ``comparable_set_member = False``; no endpoint row of a diagnostic ration is in a
   comparable set.
5. **no reserved root value in public files** (official): the decimal value of no reserved root (derived by the label
   rule) appears in any public file; stream ids use the alias ``root=formal-k<k>/``.
6. **training event consistency** per root (the arm's training event = its reference event).

Usage::

    python scripts/verify_official_outputs.py RUN_ID [--repo <root>] [--no-restricted]

Exit status: 0 every check passed; 1 a check failed; 2 bad arguments / not a run directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "experiments" / "E0_verification", REPO / "experiments" / "E1_cost_reliability"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SHA_MANIFEST, DONE_FILE, MERGE_DONE = "SHA256SUMS", "DONE", "MERGE_DONE"
TOP_LEVEL_META = {"OFFICIAL_RUN.json", "RUN_JOBS.json", "run_record.json", "README_OFFICIAL.md", "NOT_FOR_MANUSCRIPT.md",
                  "OFFICIAL_INVALIDATED.md", SHA_MANIFEST, MERGE_DONE}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(d: Path, top_excluded: set[str]) -> list[str]:
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            rel = p.relative_to(d).as_posix()
            if "/" not in rel and rel in top_excluded:
                continue
            out.append(rel)
    return out


def verify_manifest(d: Path) -> list[str]:
    """Problems of ``d`` against its SHA256SUMS (the same rule as ``official_v2.verify_sha_manifest``)."""
    m = d / SHA_MANIFEST
    if not m.is_file():
        return [f"{d.name}: {SHA_MANIFEST} missing"]
    listed = {}
    for ln in m.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            sha, rel = ln.split("  ", 1)
            listed[rel] = sha
    probs = [f"{d.name}: listed file missing: {rel}" for rel in listed if not (d / rel).is_file()]
    probs += [f"{d.name}: file changed: {rel}" for rel, sha in listed.items() if (d / rel).is_file() and _sha(d / rel) != sha]
    extra = [x for x in _files(d, {SHA_MANIFEST, DONE_FILE, MERGE_DONE}) if x not in listed]
    if extra:
        probs.append(f"{d.name}: unlisted files: {extra[:10]}")
    return probs


def _done_ok(d: Path, name: str) -> list[str]:
    p = d / name
    if not p.is_file():
        return [f"{d.name}: {name} missing"]
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return [f"{d.name}: {name} unreadable"]
    want = _sha(d / SHA_MANIFEST) if (d / SHA_MANIFEST).is_file() else None
    probs = []
    if doc.get("sha256sums_sha256") != want:
        probs.append(f"{d.name}: {name} does not match its {SHA_MANIFEST}")
    if name == DONE_FILE and (doc.get("job_id") != d.name or int(doc.get("exit_status", 1)) != 0):
        probs.append(f"{d.name}: DONE names another job or a nonzero exit status")
    return probs


def _csv_rows(p: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(p, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


def _walk_keys(o: Any):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _walk_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_keys(v)


def verify(run_id: str, repo: Path = REPO, *, restricted: bool = True) -> dict[str, Any]:
    import official_v2 as OV2   # noqa: E402  (data only at import)
    from ration_reliability.uncertainty import streams as ST
    route = OV2.official_output_route(run_id)
    pub, res = repo / route["public_dir"], repo / route["restricted_dir"]
    checks: dict[str, list[str]] = {k: [] for k in ("sha_manifests", "run_record", "no_pooled_n",
                                                     "diagnostics_not_members", "no_reserved_root_in_public",
                                                     "training_event_consistency")}
    if not pub.is_dir():
        return {"run_id": run_id, "error": f"{route['public_dir']} is not a directory"}
    # 1 sha manifests
    c = checks["sha_manifests"]
    c += verify_manifest(pub) + _done_ok(pub, MERGE_DONE)
    plan = json.loads((pub / "OFFICIAL_RUN.json").read_text(encoding="utf-8")) if (pub / "OFFICIAL_RUN.json").is_file() \
        else None
    if plan is None:
        c.append("OFFICIAL_RUN.json missing")
    job_ids = [j["job_id"] for j in (plan or {}).get("jobs", [])]
    dirs = [pub] + ([res] if restricted else [])
    if restricted and not res.is_dir():
        c.append(f"restricted mirror {route['restricted_dir']} missing (use --no-restricted to skip)")
        dirs = [pub]
    elif restricted:
        c += verify_manifest(res) + _done_ok(res, MERGE_DONE)
    for base in dirs:
        present = sorted(x.name for x in (base / "jobs").iterdir() if x.is_dir()) if (base / "jobs").is_dir() else []
        if sorted(job_ids) != present:
            c.append(f"{base.name}: job directories differ from the run plan")
        for jid in job_ids:
            jd = base / "jobs" / jid
            c += [f"{base.name}/{p}" for p in verify_manifest(jd) + _done_ok(jd, DONE_FILE)]
    # 2 run record
    c = checks["run_record"]
    rr_p = pub / "run_record.json"
    rec = json.loads(rr_p.read_text(encoding="utf-8")) if rr_p.is_file() else None
    ex: dict[str, Any] = {}
    if rec is None:
        c.append("run_record.json missing")
    else:
        ex = rec.get("extra") or {}
        want_type = "official" if route["mode"] == "official" else "smoke"
        if rec.get("run_type") != want_type:
            c.append(f"run_type {rec.get('run_type')!r} != {want_type!r}")
        if ex.get("output_label") != route["label"]:
            c.append(f"output label {ex.get('output_label')!r} (expected {route['label']!r}; official_invalidated is "
                     "disclosed, not accepted)")
        if not ((ex.get("identity_and_gating") or {}).get("end_of_run") or {}).get("unchanged"):
            c.append("end-of-run identity check not unchanged")
        bad_jobs = [j for j, v in (ex.get("jobs") or {}).items() if v.get("exit_status") != 0]
        if bad_jobs or len(ex.get("jobs") or {}) != len(job_ids):
            c.append(f"job exit statuses: {len(bad_jobs)} nonzero / {len(ex.get('jobs') or {})} of {len(job_ids)}")
        for path, sha in (rec.get("output_hashes") or {}).items():
            p = repo / path
            if not p.is_file() or _sha(p) != sha:
                c.append(f"output hash differs or file missing: {path}")
        if int(rec.get("exit_status", 1)) != 0:
            c.append(f"run record exit_status {rec.get('exit_status')}")
    # 3 no pooled n
    c = checks["no_pooled_n"]
    roots = [int(k) for k in (ex.get("roots_k") or (plan or {}).get("roots_k") or [])]
    n_test = int(((plan or {}).get("sizes") or {}).get("test", 0))
    if ex.get("pooled_across_roots") is not False:
        c.append("the run record does not say pooled_across_roots: false")
    for p in pub.iterdir():
        if p.is_file() and p.name not in TOP_LEVEL_META:
            c.append(f"table outside a root directory: {p.name}")
        if p.is_dir() and p.name not in {"jobs"} | {f"root{k}" for k in roots}:
            c.append(f"unexpected directory: {p.name}")
    for k in roots:
        rd = pub / f"root{k}"
        if not rd.is_dir():
            c.append(f"root{k}/ missing")
            continue
        for f in sorted(rd.glob("*.csv")):
            cols, rows = _csv_rows(f)
            if "root_k" not in cols or any(r.get("root_k") != str(k) for r in rows):
                c.append(f"root{k}/{f.name}: root_k column missing or not {k} in every row")
            if any("pooled" in col.lower() for col in cols):
                c.append(f"root{k}/{f.name}: a pooled column")
            if "n_test" in cols and n_test and any(r.get("n_test") not in ("", str(n_test)) for r in rows):
                c.append(f"root{k}/{f.name}: n_test differs from the run's test size")
        for f in sorted(rd.glob("*.json")):
            doc = json.loads(f.read_text(encoding="utf-8"))
            if any("pooled" in key.lower() for key in _walk_keys(doc)):
                c.append(f"root{k}/{f.name}: a pooled key")
            if isinstance(doc, dict) and "root_k" in doc and doc["root_k"] != k:
                c.append(f"root{k}/{f.name}: root_k {doc['root_k']} != {k}")
    # 4 diagnostics never members
    c = checks["diagnostics_not_members"]
    for k in roots:
        rd = pub / f"root{k}"
        cs = rd / "comparable_sets.json"
        if cs.is_file():
            doc = json.loads(cs.read_text(encoding="utf-8"))
            for s in doc.get("sets", []):
                for m in s.get("members", []) or []:
                    if ":DIAG-" in str(m.get("label", "")):
                        c.append(f"root{k}: a diagnostic label is a comparable-set member")
        dr = rd / "diagnostics_rows.csv"
        if dr.is_file():
            cols, rows = _csv_rows(dr)
            if any(r.get("comparable_set_member") not in ("False", "false", "0") or r.get("ration_kind") !=
                   "diagnostic" for r in rows):
                c.append(f"root{k}/diagnostics_rows.csv: a row is not a non-member diagnostic")
            if "cost_usd_per_head_d" in cols or "diag_h_canonical" in cols:
                c.append(f"root{k}/diagnostics_rows.csv: a restricted column is public")
        ep = rd / "endpoint_ablation.csv"
        if ep.is_file():
            _, rows = _csv_rows(ep)
            if any(":DIAG-" in r.get("label", "") and r.get("in_comparable_set") == "True" for r in rows):
                c.append(f"root{k}/endpoint_ablation.csv: a diagnostic ration is in a comparable set")
    # 5 no reserved root value in public files
    c = checks["no_reserved_root_in_public"]
    if route["mode"] == "official":
        needles = [str(v).encode("ascii") for v in ST.reserved_formal_roots()]
        n_hit = 0
        for rel in _files(pub, set()):
            data = (pub / rel).read_bytes()
            n_hit += sum(1 for nd in needles if nd in data)
        if n_hit:
            c.append(f"{n_hit} public file / root combinations contain a reserved root value (values withheld)")
    # 6 training event consistency
    c = checks["training_event_consistency"]
    for k in roots:
        f = pub / f"root{k}" / "training_event_consistency.json"
        if not f.is_file() or not json.loads(f.read_text(encoding="utf-8")).get("all_equal"):
            c.append(f"root{k}: training event != its reference event (or the report is missing)")
    ok = {k: not v for k, v in checks.items()}
    return {"run_id": run_id, "mode": route["mode"], "public_dir": route["public_dir"],
            "restricted_checked": bool(restricted), "roots_k": roots, "checks_passed": ok,
            "all_passed": all(ok.values()), "problems": {k: v[:20] for k, v in checks.items() if v}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--no-restricted", action="store_true", help="skip the restricted mirror (public tables only)")
    args = ap.parse_args(argv)
    try:
        out = verify(args.run_id, args.repo.resolve(), restricted=not args.no_restricted)
    except ValueError as exc:
        print(f"not a run id: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if "error" in out:
        return 2
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
