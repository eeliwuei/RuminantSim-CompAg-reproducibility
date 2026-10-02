#!/usr/bin/env python3
"""Replay a reference host's deterministic dev_case build outputs onto a run host, with a log (round-3 red team D-10).

Scripted form of ``logs/px_runs/pilot-20260925T055921Z-fb4f75af/mac_bytes_replay.md`` -- the lead's after-the-fact
record of the 2026-09-25 replay of the Mac-built restricted build outputs onto the compute host, which until now
existed only as that record (no script, no raw log; B-445).  FIX2, 2026-09-26.  The holder runs one sub-command on
each host.  This script never connects to another host, never copies between hosts (the operator moves the bundle,
e.g. with scp), never solves, and never prints or logs a restricted value (only paths, sha256, counts and statuses).

``reference`` -- on the host whose build bytes the freeze pin registered (the Mac on 2026-09-25):

1. repository: record HEAD; the work tree must be clean (``git status --porcelain`` empty);
2. back up the case directory's build outputs (``OUTPUT_FILES`` and ``build_identity.json``) into
   ``<work-dir>/reference_backup/``;
3. rebuild: ``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 <python>
   scripts/build_dev_case.py`` (exit 0);
4. determinism: every file of ``OUTPUT_FILES`` must be byte-identical to its backup -- otherwise the backup is put
   back and the replay stops (the reference bytes are not reproducible on this host);
5. the build sidecar ``build_identity.json`` exists and the work tree is still clean;
6. preflight: ``<python> scripts/preflight_dev_case.py --driver run_dev_case_v1 --run-type pilot`` (exit 0, READY);
7. dry run: ``<python> experiments/E1_cost_reliability/run_endpoint_ablation.py --dry-run`` must report
   ``checks.freeze.status = matches_frozen_anchored``;
8. bundle: ``<work-dir>/build_outputs_bundle.tgz`` with the six files and ``BUNDLE_MANIFEST.json`` (sha256 of each
   file, git HEAD, the dry run's reference-spec fingerprint, host name, UTC); its sha256 is printed and logged.

``apply`` -- on the run host, at the same commit, after the bundle was copied there:

1. repository: HEAD must equal the bundle's HEAD; the work tree must be clean;
2. bundle: exactly the six files plus the manifest, plain names (no directory, no link), sha256 = manifest --
   checked before anything on the host is touched;
3. back up the run host's own build outputs into ``--superseded-dir`` (must be new or empty; never deleted);
4. write the six files into the case directory; their sha256 must equal the manifest;
5. preflight READY (as reference step 6);
6. dry run ``matches_frozen_anchored`` (as reference step 7), its output kept as ``<work-dir>/dryrun_after_replay.json``;
7. with ``--pytest``: the full test suite on the replaced bytes (the 2026-09-25 replay did not rerun it; its
   ``pytest_px2.log`` was taken on the host-built bytes).  If step 5, 6 or 7 fails, the replaced bytes stay in place
   (the host's own bytes are in the superseded directory) and the exit status is 1.

Every step is appended to ``<work-dir>/replay_log.jsonl`` (UTC, step, ok, detail); the stdout / stderr of each
command are kept under ``<work-dir>/commands/``.  The work directory must be outside the repository or under
``data/restricted_local/`` (the bundle holds restricted files).  ``--dry-run`` prints the plan (steps, commands,
paths) after checking the arguments, and touches nothing and runs nothing (exit 0).

Usage (repository root)::

    python scripts/replay_build_outputs.py reference --work-dir <dir> [--python <py>] [--dry-run]
    python scripts/replay_build_outputs.py apply --bundle <dir>/build_outputs_bundle.tgz --work-dir <dir> \\
        --superseded-dir <dir> [--python <py>] [--pytest] [--dry-run]

Exit status: 0 every step ok (or a dry run); 1 a step failed; 3 bad arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ration_reliability.build.dev_case import OUTPUT_FILES, SIDECAR  # noqa: E402

CASE_DIR = "data/restricted_local/dev_case_v1"
RESTRICTED_ROOT = "data/restricted_local"
BUNDLE_NAME = "build_outputs_bundle.tgz"
BUNDLE_MANIFEST = "BUNDLE_MANIFEST.json"
BUNDLE_SCHEMA = "ration_reliability.build_outputs_bundle/1"
LOG_NAME = "replay_log.jsonl"
FROZEN_OK = "matches_frozen_anchored"
REPLAY_FILES = tuple(OUTPUT_FILES) + (SIDECAR,)
THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "PYTHONDONTWRITEBYTECODE": "1"}
BUILD_CMD = ("scripts/build_dev_case.py",)
PREFLIGHT_CMD = ("scripts/preflight_dev_case.py", "--driver", "run_dev_case_v1", "--run-type", "pilot")
DRYRUN_CMD = ("experiments/E1_cost_reliability/run_endpoint_ablation.py", "--dry-run")
PYTEST_CMD = ("-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs")
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 3

#: runner(argv, cwd, env) -> (exit code, stdout, stderr); the default runs a subprocess (tests pass a fake)
Runner = Callable[[list[str], Path, dict], tuple[int, str, str]]


class StepFailed(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def subprocess_runner(argv: list[str], cwd: Path, env: dict) -> tuple[int, str, str]:
    r = subprocess.run(argv, cwd=str(cwd), env={**os.environ, **env}, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def work_dir_problem(root: Path, work_dir: Path) -> str | None:
    """None when ``work_dir`` may hold restricted build files: outside the repository, or under its
    ``data/restricted_local/`` (git-ignored); else the reason."""
    root, wd = root.resolve(), work_dir.resolve()
    if wd == root or root in wd.parents:
        rl = (root / RESTRICTED_ROOT).resolve()
        if not (wd == rl or rl in wd.parents):
            return f"{wd} is inside the repository but not under {RESTRICTED_ROOT}/ (the bundle holds restricted files)"
    return None


def parse_dry_run(stdout: str) -> dict:
    """Freeze status and reference-spec fingerprint from the driver's ``--dry-run`` JSON (stdout may carry other
    lines before the JSON)."""
    doc = None
    try:
        doc = json.loads(stdout)
    except ValueError:
        k = stdout.find("{")
        if k >= 0:
            try:
                doc = json.loads(stdout[k:])
            except ValueError:
                doc = None
    if not isinstance(doc, dict):
        return {"parsed": False, "freeze_status": None, "reference_spec_fingerprint": None, "anchored": None}
    checks = doc.get("checks") or {}
    fr = checks.get("freeze") or {}
    return {"parsed": True, "freeze_status": fr.get("status"), "anchored": (fr.get("anchor") or {}).get("anchored"),
            "reference_spec_fingerprint": checks.get("reference_spec_fingerprint"),
            "preflight": checks.get("preflight")}


class Replay:
    """One replay (reference or apply) with its log; every file operation is inside ``root / CASE_DIR`` or the
    given work / superseded directories."""

    def __init__(self, root: Path, work_dir: Path, python: str, runner: Runner = subprocess_runner):
        self.root, self.work_dir, self.python, self.runner = root.resolve(), work_dir, python, runner
        self.case = self.root / CASE_DIR
        self.log_path = work_dir / LOG_NAME
        self.results: list[dict] = []

    # ---------------------------------------------------------------- log and commands
    def log(self, step: str, ok: bool, **detail) -> dict:
        rec = {"utc": utc_now(), "step": step, "ok": ok, **detail}
        self.results.append(rec)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(("OK   " if ok else "FAIL ") + step + ("" if not detail else " " + json.dumps(detail, ensure_ascii=False)))
        if not ok:
            raise StepFailed(step)
        return rec

    def run(self, name: str, argv: list[str], env: dict | None = None) -> tuple[int, str, str]:
        rc, out, err = self.runner(argv, self.root, dict(env or {}))
        d = self.work_dir / "commands"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.stdout").write_text(out or "", encoding="utf-8")
        (d / f"{name}.stderr").write_text(err or "", encoding="utf-8")
        return rc, out or "", err or ""

    def py(self, *args: str) -> list[str]:
        return [self.python, *args]

    # ---------------------------------------------------------------- shared steps
    def git_state(self) -> tuple[str, list[str]]:
        rc, head, _e = self.run("git_rev_parse", ["git", "-C", str(self.root), "rev-parse", "HEAD"])
        rc2, st, _e2 = self.run("git_status", ["git", "-C", str(self.root), "status", "--porcelain"])
        if rc or rc2:
            self.log("git", False, reason="git failed", rev_parse_exit=rc, status_exit=rc2)
        return head.strip(), [ln for ln in st.splitlines() if ln.strip()]

    def hashes(self, names=REPLAY_FILES) -> dict[str, str | None]:
        return {n: (sha256_file(self.case / n) if (self.case / n).is_file() else None) for n in names}

    def preflight(self, step: str) -> None:
        rc, out, _e = self.run(step, self.py(*PREFLIGHT_CMD))
        self.log(step, rc == 0 and "READY" in out, exit=rc, ready="READY" in out)

    def dry_run(self, step: str, keep_as: str | None = None) -> dict:
        rc, out, _e = self.run(step, self.py(*DRYRUN_CMD), THREAD_ENV)
        if keep_as:
            (self.work_dir / keep_as).write_text(out, encoding="utf-8")
        info = parse_dry_run(out)
        self.log(step, rc == 0 and info["freeze_status"] == FROZEN_OK, exit=rc, **info)
        return info

    # ---------------------------------------------------------------- reference host
    def reference(self) -> dict:
        head, dirty = self.git_state()
        self.log("r1_repository_clean", not dirty, head=head, dirty_paths=len(dirty))
        backup = self.work_dir / "reference_backup"
        if backup.exists() and any(backup.iterdir()):
            self.log("r2_backup", False, reason=f"{backup} is not empty (an earlier replay?)")
        backup.mkdir(parents=True, exist_ok=True)
        before = self.hashes()
        missing = [n for n in OUTPUT_FILES if before[n] is None]
        if missing:
            self.log("r2_backup", False, reason="build outputs missing: determinism cannot be checked", missing=missing)
        for n, h in before.items():
            if h is not None:
                shutil.copy2(self.case / n, backup / n)
        self.log("r2_backup", True, backup=str(backup), files=sorted(n for n, h in before.items() if h))
        rc, _o, _e = self.run("r3_build", self.py(*BUILD_CMD), THREAD_ENV)
        if rc != 0:
            self._restore(backup, before)
            self.log("r3_build", False, exit=rc, restored_backup=True)
        self.log("r3_build", True, exit=rc)
        after = self.hashes()
        differ = [n for n in OUTPUT_FILES if after[n] != before[n]]
        if differ:
            self._restore(backup, before)
            self.log("r4_determinism", False, differ=differ, restored_backup=True,
                     meaning="this host does not rebuild the registered bytes; nothing is bundled")
        self.log("r4_determinism", True, identical=list(OUTPUT_FILES),
                 sha256_16={n: after[n][:16] for n in OUTPUT_FILES})
        _h, dirty2 = self.git_state()
        self.log("r5_sidecar_and_clean_tree", after[SIDECAR] is not None and not dirty2,
                 sidecar=after[SIDECAR] is not None, dirty_paths=len(dirty2))
        self.preflight("r6_preflight")
        info = self.dry_run("r7_dry_run", keep_as="dryrun_reference.json")
        bundle = self._bundle(head, info, after)
        return {"bundle": str(bundle), "bundle_sha256": sha256_file(bundle), "head": head}

    def _restore(self, backup: Path, before: dict) -> None:
        for n, h in before.items():
            if h is not None:
                shutil.copy2(backup / n, self.case / n)
            elif (self.case / n).exists():
                # a file that did not exist before the rebuild (e.g. a new sidecar) is moved aside, not deleted
                (self.case / n).replace(backup / f"{n}.created_by_failed_rebuild")

    def _bundle(self, head: str, info: dict, hashes: dict) -> Path:
        man = {"schema": BUNDLE_SCHEMA, "created_utc": utc_now(), "host": socket.gethostname(), "git_head": head,
               "case_dir": CASE_DIR, "files": {n: hashes[n] for n in REPLAY_FILES},
               "reference_spec_fingerprint": info.get("reference_spec_fingerprint"),
               "dry_run_freeze_status": info.get("freeze_status"),
               "built_with": {"command": " ".join(BUILD_CMD), "env": THREAD_ENV},
               "note": "restricted build outputs (NASEM-derived): keep under data/restricted_local/ or outside any "
                       "repository; never package or publish"}
        path = self.work_dir / BUNDLE_NAME
        mb = (json.dumps(man, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        with tarfile.open(path, "w:gz") as tf:
            for n in REPLAY_FILES:
                tf.add(self.case / n, arcname=n, recursive=False)
            ti = tarfile.TarInfo(BUNDLE_MANIFEST)
            ti.size, ti.mtime = len(mb), int(datetime.now(timezone.utc).timestamp())
            tf.addfile(ti, io.BytesIO(mb))
        self.log("r8_bundle", True, bundle=str(path), bundle_sha256=sha256_file(path), git_head=head,
                 reference_spec_fingerprint=info.get("reference_spec_fingerprint"))
        return path

    # ---------------------------------------------------------------- run host
    def apply(self, bundle: Path, superseded: Path, with_pytest: bool = False) -> dict:
        members, man = read_bundle(bundle)                   # raises StepFailed via log below
        head, dirty = self.git_state()
        self.log("a1_repository", head == man.get("git_head") and not dirty, head=head,
                 bundle_head=man.get("git_head"), dirty_paths=len(dirty))
        bad = {n: {"manifest": man["files"].get(n), "bundle": hashlib.sha256(b).hexdigest()}
               for n, b in members.items() if hashlib.sha256(b).hexdigest() != man["files"].get(n)}
        self.log("a2_bundle_verified", not bad and set(members) == set(REPLAY_FILES),
                 members=sorted(members), mismatched=sorted(bad), bundle_sha256=sha256_file(bundle))
        if superseded.exists() and any(superseded.iterdir()):
            self.log("a3_backup_host_outputs", False, reason=f"{superseded} exists and is not empty")
        superseded.mkdir(parents=True, exist_ok=True)
        own = self.hashes()
        for n, h in own.items():
            if h is not None:
                shutil.copy2(self.case / n, superseded / n)
        (superseded / "SUPERSEDED.json").write_text(json.dumps(
            {"utc": utc_now(), "host": socket.gethostname(), "replaced_by_bundle": str(bundle),
             "bundle_git_head": man.get("git_head"), "files": own}, indent=1) + "\n", encoding="utf-8")
        self.log("a3_backup_host_outputs", True, superseded_dir=str(superseded),
                 host_differs_from_bundle=sorted(n for n in REPLAY_FILES if own[n] != man["files"][n]))
        self.case.mkdir(parents=True, exist_ok=True)
        for n, b in members.items():
            tmp = self.case / f".{n}.replay_tmp"
            tmp.write_bytes(b)
            os.replace(tmp, self.case / n)
        now = self.hashes()
        self.log("a4_written", all(now[n] == man["files"][n] for n in REPLAY_FILES),
                 sha256_16={n: (now[n] or "")[:16] for n in REPLAY_FILES})
        self.preflight("a5_preflight")
        info = self.dry_run("a6_dry_run", keep_as="dryrun_after_replay.json")
        out = {"dry_run": info, "superseded_dir": str(superseded)}
        if with_pytest:
            rc, o, _e = self.run("a7_pytest", self.py(*PYTEST_CMD), {"PYTHONDONTWRITEBYTECODE": "1"})
            tail = [ln for ln in o.splitlines() if ln.strip()][-1:] or [""]
            self.log("a7_pytest", rc == 0, exit=rc, summary=tail[0])
        return out


def read_bundle(bundle: Path) -> tuple[dict[str, bytes], dict]:
    """Members (name -> bytes) and manifest of a bundle; refuses anything but plain regular files with the expected
    names (no directory part, no link, no device) -- nothing is extracted to disk here."""
    members: dict[str, bytes] = {}
    man = None
    with tarfile.open(bundle, "r:gz") as tf:
        for m in tf.getmembers():
            name = m.name
            if not m.isreg() or "/" in name or "\\" in name or name in ("", ".", "..") or name.startswith("."):
                raise StepFailed(f"bundle member refused: {name!r} (not a plain regular file)")
            if name == BUNDLE_MANIFEST:
                man = json.loads(tf.extractfile(m).read().decode("utf-8"))
            elif name in REPLAY_FILES:
                members[name] = tf.extractfile(m).read()
            else:
                raise StepFailed(f"bundle member refused: {name!r} (unexpected name)")
    if not isinstance(man, dict) or man.get("schema") != BUNDLE_SCHEMA or not isinstance(man.get("files"), dict):
        raise StepFailed("bundle manifest missing or of another schema")
    return members, man


def plan(args, root: Path) -> dict:
    """The steps a real run would take (``--dry-run``); nothing is touched."""
    py = args.python
    env = " ".join(f"{k}={v}" for k, v in THREAD_ENV.items())
    common = {"root": str(root), "case_dir": CASE_DIR, "work_dir": str(args.work_dir),
              "log": str(Path(args.work_dir) / LOG_NAME), "files": list(REPLAY_FILES)}
    if args.command == "reference":
        steps = [["r1_repository_clean", "git rev-parse HEAD; git status --porcelain (must be empty)"],
                 ["r2_backup", f"copy {CASE_DIR}/{{{','.join(REPLAY_FILES)}}} -> <work-dir>/reference_backup/"],
                 ["r3_build", f"{env} {py} {' '.join(BUILD_CMD)}"],
                 ["r4_determinism", f"sha256 of {', '.join(OUTPUT_FILES)} == backup (else restore backup, stop)"],
                 ["r5_sidecar_and_clean_tree", f"{SIDECAR} present; git status --porcelain empty"],
                 ["r6_preflight", f"{py} {' '.join(PREFLIGHT_CMD)}  (exit 0, READY)"],
                 ["r7_dry_run", f"{env} {py} {' '.join(DRYRUN_CMD)}  (checks.freeze.status == {FROZEN_OK})"],
                 ["r8_bundle", f"<work-dir>/{BUNDLE_NAME} = the six files + {BUNDLE_MANIFEST}"],
                 ["(operator)", f"copy <work-dir>/{BUNDLE_NAME} to the run host (not done by this script)"]]
        return {"dry_run": True, "command": "reference", **common, "steps": steps}
    steps = [["a1_repository", "git rev-parse HEAD == bundle git_head; git status --porcelain empty"],
             ["a2_bundle_verified", f"members = the six files + {BUNDLE_MANIFEST}, plain names, sha256 = manifest"],
             ["a3_backup_host_outputs", f"copy the host's {CASE_DIR} build files -> {args.superseded_dir} (new/empty)"],
             ["a4_written", f"write the six files into {CASE_DIR}; sha256 = manifest"],
             ["a5_preflight", f"{py} {' '.join(PREFLIGHT_CMD)}  (exit 0, READY)"],
             ["a6_dry_run", f"{env} {py} {' '.join(DRYRUN_CMD)}  (checks.freeze.status == {FROZEN_OK})"]]
    if args.pytest:
        steps.append(["a7_pytest", f"PYTHONDONTWRITEBYTECODE=1 {py} {' '.join(PYTEST_CMD)}"])
    return {"dry_run": True, "command": "apply", **common, "bundle": str(args.bundle),
            "bundle_exists": Path(args.bundle).is_file(), "superseded_dir": str(args.superseded_dir),
            "steps": steps}


def main(argv: list[str] | None = None, runner: Runner = subprocess_runner) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("reference", "apply"):
        sp = sub.add_parser(name)
        sp.add_argument("--root", type=Path, default=REPO)
        sp.add_argument("--work-dir", type=Path, required=True,
                        help="log, command outputs (and, reference, the bundle); outside the repository or under "
                             "data/restricted_local/")
        sp.add_argument("--python", default=sys.executable, help="interpreter of the project environment")
        sp.add_argument("--dry-run", action="store_true", help="print the plan; touch nothing, run nothing")
        if name == "apply":
            sp.add_argument("--bundle", type=Path, required=True)
            sp.add_argument("--superseded-dir", type=Path, required=True,
                            help="where the host's own build outputs are kept (new or empty; never deleted)")
            sp.add_argument("--pytest", action="store_true", help="run the full test suite after the replacement")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK
    root = args.root.resolve()
    if not (root / BUILD_CMD[0]).is_file() or not (root / DRYRUN_CMD[0]).is_file():
        print(f"not a project root: {root}", file=sys.stderr)
        return EXIT_USAGE
    for d in [args.work_dir] + ([args.superseded_dir] if args.command == "apply" else []):
        if (why := work_dir_problem(root, d)) is not None:
            print(why, file=sys.stderr)
            return EXIT_USAGE
    if args.dry_run:
        print(json.dumps(plan(args, root), ensure_ascii=False, indent=1))
        return EXIT_OK
    rp = Replay(root, args.work_dir, args.python, runner)
    try:
        if args.command == "reference":
            res = rp.reference()
        else:
            if not args.bundle.is_file():
                print(f"bundle not found: {args.bundle}", file=sys.stderr)
                return EXIT_USAGE
            try:
                read_bundle(args.bundle)
            except StepFailed as exc:
                rp.log("a2_bundle_verified", False, reason=str(exc))
            res = rp.apply(args.bundle, args.superseded_dir, args.pytest)
    except StepFailed as exc:
        print(f"replay stopped at step {exc}; log: {rp.log_path}", file=sys.stderr)
        return EXIT_FAIL
    print(json.dumps({"ok": True, **res, "log": str(rp.log_path)}, ensure_ascii=False, indent=1))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
