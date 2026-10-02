"""Run records (contract T9; evidence_contract.yaml ``run_record_required``).

A run record is a JSON file with at least::

    run_id, run_type, started_at, completed_at, code_commit_or_hash, dirty_diff_hash,
    data_manifest_hash, config_hash, protocol_hash, rng_streams, solver_version, tolerances,
    command, exit_status, output_paths, output_hashes

plus hardware / interpreter / package versions and ``is_synthetic``.

Code identity (second-review R6, 2026-09-25)
--------------------------------------------
* ``code_manifest`` -- a content manifest of every file under ``src``, ``experiments``,
  ``scripts``, ``tests`` and ``configs`` plus the root files ``pytest.ini``,
  ``requirements-lock.txt`` and ``environment.lock.json`` (per-file sha256 and a total
  ``manifest_sha256``).  It is built by walking the file system with fixed exclusion rules
  (``__pycache__`` and other cache directories, ``*.pyc``/``*.pyo``, ``.DS_Store``), so it
  does not depend on git: a ZIP extraction without ``.git`` and a git checkout with the same
  bytes produce the same manifest.  Tracked and untracked files are treated alike.
* If ``git`` is available **and** ``repo_root`` is the top level of the work tree, the HEAD
  commit, a dirty flag and ``dirty_diff_hash`` are added.  ``dirty_diff_hash`` is the sha256 of
  ``git diff HEAD --binary`` restricted to ``_GIT_PATHS`` (the same scope as the manifest) plus
  the content hashes of untracked, non-ignored files there.  The git listing
  (tracked + untracked, non-ignored) is cross-checked against the manifest
  (``git_manifest_crosscheck``); differences are reported, never silently merged.  A
  ``repo_root`` nested inside some other repository does not borrow that repository's HEAD.
  Only read-only git commands are used.
* ``src_tree_sha256`` (every ``*.py`` under ``src/ration_reliability``) is kept for
  backward comparison with older records; it does not cover drivers, scripts or configs.

Environment identity
--------------------
``environment_fingerprint()`` records the interpreter (version, implementation), the platform
(``sys.platform``, machine) and the effective versions of the locked distributions
``LOCKED_DISTRIBUTIONS`` (read from the imported module's ``__version__`` or from
``importlib.metadata``; nothing is installed).  ``check_environment_against_lock`` compares
it with ``environment.lock.json`` / ``requirements-lock.txt``.  Every run record carries the
fingerprint and the lock check; pilot and official runs are refused unless the environment
matches the lock and the code manifest is complete.

Optional distribution groups (FIX_C, 2026-09-25): a third-party package that only a holder-side
script imports is locked in a named group of ``OPTIONAL_DISTRIBUTION_GROUPS`` instead of the core
(currently ``holder_verification`` = PyMuPDF, imported only by ``scripts/verify_restricted_inputs.py``
for the PDF text layer and page rendering).  Group versions are stored per environment entry of
the lock and as ``#[optional:<group>] name==version`` lines of ``requirements-lock.txt`` (comments to
pip, so a plain install stays the core set); they are *not* part of ``fingerprint_sha256``, so run
identities do not depend on a package the engine never imports.  ``check_environment_against_lock``
compares a group only when asked (``optional_groups=``); ``scripts/check_environment.py
--require-group holder_verification`` makes a mismatch fail.

Run identity
------------
``run_identity_sha256`` hashes the code manifest, the environment fingerprint, the config /
data / protocol hashes, the command, the random-stream ids, the solver version, the
tolerances, the run type and the synthetic flag.  Equal identities mean the two runs had the
same code, environment, inputs and settings; timestamps, exit status and outputs are not part
of it.

Build identity (review round 3, F6; R3F, 2026-09-25)
----------------------------------------------------
A run that consumes built inputs (e.g. the dev_case problem YAML) can pass ``build_identity`` (from
``build_identity()``): the build scripts (public builder code, entry point, the restricted-constants file) and
the build inputs by repository-relative path and sha256, plus the build outputs.  Files under
``data/restricted_local/`` are marked ``restricted`` and enter **by path and sha256 only** (never content,
size or values); a path outside the repository is recorded by its name only.  ``build_identity_sha256``
covers the scripts and inputs (what determined the build); when it is given, ``run_identity`` includes it,
so a changed builder or a changed build input changes the run identity even if the outputs happen to be
identical.  Without it the identity components are unchanged (older identities stay reproducible).

Build-time provenance (FIX3_DEF, review round 3 red team F-2): a run record must carry the identity **written at
build time**, not one recomputed from the files at run time.  A builder that knows this (e.g.
``build.dev_case.dev_case_build_identity``, which reads the build sidecar) adds ``usable_for_pilot_official``,
``source`` and ``sidecar``; ``build_run_record`` refuses a pilot / official record whose identity says
``usable_for_pilot_official = False`` (no build-time sidecar, or builder code / constants / inputs / outputs changed
since the build).  Smoke and debug records keep such an identity with its labels.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import importlib.metadata as _md
import json
import os
import platform
import re
import site
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..hashing import file_sha256, files_sha256, stable_hash

__all__ = [
    "RUN_TYPES",
    "CODE_MANIFEST_DIRS",
    "CODE_MANIFEST_ROOT_FILES",
    "ENVIRONMENT_LOCK_FILE",
    "REQUIREMENTS_LOCK_FILE",
    "LOCKED_DISTRIBUTIONS",
    "OPTIONAL_DISTRIBUTION_GROUPS",
    "ENVIRONMENT_CHECK_STATUSES",
    "code_manifest",
    "code_manifest_digest",
    "in_code_manifest_scope",
    "code_manifest_subset_sha256",
    "build_identity",
    "RESTRICTED_PREFIX",
    "compare_code_manifests",
    "code_fingerprint",
    "environment_fingerprint",
    "optional_group_versions",
    "environment_lock_document",
    "environment_lock_entry",
    "validate_environment_lock",
    "requirements_lock_text",
    "parse_requirements_lock",
    "parse_requirements_lock_optional",
    "check_environment_against_lock",
    "check_environment_for_run_type",
    "check_reserved_root_streams",
    "run_identity",
    "make_run_id",
    "build_run_record",
    "write_run_record",
    "utc_now",
]

RUN_TYPES = ("unit_test", "smoke", "pilot", "official")

# --- code manifest scope -------------------------------------------------------------------------
CODE_MANIFEST_DIRS = ("src", "experiments", "scripts", "tests", "configs")
ENVIRONMENT_LOCK_FILE = "environment.lock.json"
REQUIREMENTS_LOCK_FILE = "requirements-lock.txt"
CODE_MANIFEST_ROOT_FILES = ("pytest.ini", REQUIREMENTS_LOCK_FILE, ENVIRONMENT_LOCK_FILE)
_GIT_PATHS = CODE_MANIFEST_DIRS + CODE_MANIFEST_ROOT_FILES
MANIFEST_SCHEMA = "ration_reliability.code_manifest/1"
MANIFEST_EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                                         ".ipynb_checkpoints", ".venv", ".git"})
MANIFEST_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")
MANIFEST_EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})

# --- environment lock ----------------------------------------------------------------------------
ENV_FINGERPRINT_SCHEMA = "ration_reliability.environment_fingerprint/1"
ENV_LOCK_SCHEMA = "ration_reliability.environment_lock/1"
# (distribution name, import name, why it is locked).  Third-party imports of src/, experiments/,
# scripts/ and tests/ are numpy, scipy, pandas, yaml and pytest (grep of import lines, 2026-09-25),
# plus ``fitz`` (PyMuPDF) inside three functions of scripts/verify_restricted_inputs.py only -- that one
# is locked in the optional group ``holder_verification`` below, not here (FIX_C: the earlier comment
# "all third-party imports" missed it).  The transitive entries are the runtime requirements of the
# core set for CPython 3.11 on non-Windows platforms (importlib.metadata.requires with environment
# markers evaluated by hand; extras excluded).
LOCKED_DISTRIBUTIONS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "direct"),
    ("scipy", "scipy", "direct"),
    ("pandas", "pandas", "direct"),
    ("PyYAML", "yaml", "direct"),
    ("pytest", "pytest", "direct (test runner)"),
    ("python-dateutil", "dateutil", "requirement of pandas"),
    ("pytz", "pytz", "requirement of pandas"),
    ("tzdata", "tzdata", "requirement of pandas"),
    ("six", "six", "requirement of python-dateutil"),
    ("iniconfig", "iniconfig", "requirement of pytest"),
    ("packaging", "packaging", "requirement of pytest"),
    ("pluggy", "pluggy", "requirement of pytest"),
    ("Pygments", "pygments", "requirement of pytest"),
)
# Optional groups: {group: ((distribution, import name, why), ...)}.  Not imported by src/, experiments/
# or tests/; not part of the environment fingerprint core.  PyMuPDF has no runtime requirements
# (importlib.metadata.requires('PyMuPDF') is None, checked 2026-09-25).
OPTIONAL_DISTRIBUTION_GROUPS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "holder_verification": (
        ("PyMuPDF", "fitz", "scripts/verify_restricted_inputs.py sections C2, E, F (PDF text layer and page "
                            "rendering); holder-side verification only"),
    ),
}
_OPTIONAL_PIN = re.compile(r"^#\[optional:([A-Za-z0-9_]+)\]\s*([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)\s*$")
# Result of comparing the running interpreter with the lock (a check outcome, not a stage status).
ENVIRONMENT_CHECK_STATUSES = ("matches_lock", "mismatch", "lock_missing", "lock_invalid")
# Result of comparing one optional group (``not_in_lock``: the lock predates the group or omits it).
OPTIONAL_GROUP_CHECK_STATUSES = ("matches_lock", "mismatch", "not_installed", "not_in_lock")


def utc_now() -> str:
    """Current UTC time, ISO 8601 with seconds."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


# =================================================================================================
# git (read-only)
# =================================================================================================

def _git(repo: Path, *args: str) -> Optional[bytes]:
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True,
                           timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def _git_toplevel_status(root: Path) -> tuple[bool, Optional[str]]:
    """``(usable, error)``: git information is used only when ``root`` is the work-tree top level."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None:
        return False, "git unavailable or not a repository"
    try:
        same = Path(top.decode().strip()).resolve() == root.resolve()
    except OSError:
        same = False
    if not same:
        return False, "repo_root is not the top level of a git work tree (an enclosing repository was found); " \
                      "git HEAD/diff of that repository are not used"
    return True, None


# =================================================================================================
# code manifest (file-system walk; identical with and without git)
# =================================================================================================

def _is_excluded_relpath(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in MANIFEST_EXCLUDED_DIR_NAMES for p in parts[:-1]):
        return True
    name = parts[-1]
    return name in MANIFEST_EXCLUDED_FILE_NAMES or name.endswith(MANIFEST_EXCLUDED_FILE_SUFFIXES)


def _manifest_scope() -> dict[str, Any]:
    return {
        "dirs": list(CODE_MANIFEST_DIRS),
        "root_files": list(CODE_MANIFEST_ROOT_FILES),
        "excluded_dir_names": sorted(MANIFEST_EXCLUDED_DIR_NAMES),
        "excluded_file_suffixes": list(MANIFEST_EXCLUDED_FILE_SUFFIXES),
        "excluded_file_names": sorted(MANIFEST_EXCLUDED_FILE_NAMES),
        "symlinks": "recorded as links (sha256 of b'symlink\\0' + target), never followed",
    }


def _walk_paths(root: Path) -> tuple[list[Path], list[str], list[str]]:
    found: list[Path] = []
    missing_dirs: list[str] = []
    missing_files: list[str] = []
    for d in CODE_MANIFEST_DIRS:
        base = root / d
        if base.is_symlink():
            found.append(base)
            continue
        if not base.is_dir():
            missing_dirs.append(d)
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dp = Path(dirpath)
            keep = []
            for dn in dirnames:
                if dn in MANIFEST_EXCLUDED_DIR_NAMES:
                    continue
                if (dp / dn).is_symlink():
                    found.append(dp / dn)
                    continue
                keep.append(dn)
            dirnames[:] = keep
            for fn in filenames:
                p = dp / fn
                if fn in MANIFEST_EXCLUDED_FILE_NAMES or fn.endswith(MANIFEST_EXCLUDED_FILE_SUFFIXES):
                    continue
                if p.is_symlink() or p.is_file():
                    found.append(p)
    for f in CODE_MANIFEST_ROOT_FILES:
        p = root / f
        if p.is_symlink() or p.is_file():
            found.append(p)
        else:
            missing_files.append(f)
    return found, missing_dirs, missing_files


def _manifest_entry(root: Path, p: Path) -> dict[str, Any]:
    rel = p.relative_to(root).as_posix()
    if p.is_symlink():
        target = os.readlink(p)
        return {"path": rel, "kind": "symlink", "bytes": None, "symlink_target": target,
                "sha256": hashlib.sha256(b"symlink\0" + os.fsencode(target)).hexdigest()}
    try:
        size = p.stat().st_size
        sha = file_sha256(p)
    except FileNotFoundError:  # removed by a concurrent process between listing and hashing
        return {"path": rel, "kind": "vanished_during_hashing", "bytes": None, "sha256": None}
    return {"path": rel, "kind": "file", "bytes": int(size), "sha256": sha}


def _manifest_digest(scope: Mapping[str, Any], files: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(f"{MANIFEST_SCHEMA}\n".encode())
    h.update(json.dumps(scope, sort_keys=True, ensure_ascii=True).encode() + b"\n")
    for e in files:  # already sorted by path
        h.update(f"{e['path']}\0{e['kind']}\0{e['sha256'] or 'NONE'}\n".encode("utf-8", "surrogateescape"))
    return h.hexdigest()


def code_manifest(repo_root: str | Path) -> dict[str, Any]:
    """Content manifest of the code, driver, test and configuration files at ``repo_root``.

    Works without ``.git``.  ``files`` is sorted by POSIX relative path; ``manifest_sha256``
    covers the schema, the scope rules and every ``(path, kind, sha256)``.  ``complete`` is
    False when a file vanished while it was being hashed (concurrent edit): such a manifest
    identifies no stable code state and is refused for pilot/official runs.
    """
    root = Path(repo_root)
    paths, missing_dirs, missing_files = _walk_paths(root)
    files = sorted((_manifest_entry(root, p) for p in paths), key=lambda e: e["path"])
    scope = _manifest_scope()
    vanished = [e["path"] for e in files if e["kind"] == "vanished_during_hashing"]
    per_dir = {d: sum(1 for e in files if e["path"] == d or e["path"].startswith(d + "/"))
               for d in CODE_MANIFEST_DIRS}
    return {
        "schema": MANIFEST_SCHEMA,
        "method": "filesystem_walk (independent of git)",
        "scope": scope,
        "manifest_sha256": _manifest_digest(scope, files),
        "file_count": len(files),
        "file_count_by_dir": per_dir,
        "total_bytes": int(sum(e["bytes"] or 0 for e in files)),
        "missing_dirs": missing_dirs,
        "missing_root_files": missing_files,
        "complete": not vanished,
        "vanished_during_hashing": vanished,
        "files": files,
    }


def in_code_manifest_scope(rel: str) -> bool:
    """Whether a repository-relative POSIX path belongs to the code-manifest scope (same exclusion rules)."""
    if rel.startswith("./"):
        rel = rel[2:]
    if rel in CODE_MANIFEST_ROOT_FILES:
        return True
    first = rel.split("/", 1)[0]
    return first in CODE_MANIFEST_DIRS and "/" in rel and not _is_excluded_relpath(rel)


def code_manifest_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    """``manifest_sha256`` of a list of ``{"path", "kind", "sha256"}`` entries, computed exactly as ``code_manifest``
    does (same schema line and scope rules; entries are sorted by path here).  Used to compute the code-manifest
    digest of *packaged* bytes (R3F: ``scripts/package_release.py``), which a recipient recomputes with
    ``code_manifest`` on the extracted package."""
    files = sorted(({"path": e["path"], "kind": e.get("kind", "file"), "sha256": e.get("sha256")} for e in entries),
                   key=lambda e: e["path"])
    return _manifest_digest(_manifest_scope(), files)


MANIFEST_SUBSET_SCHEMA = "ration_reliability.code_manifest_subset/1"
MANIFEST_AREAS = CODE_MANIFEST_DIRS + ("root_files",)


def _manifest_area(path: str) -> str:
    first = path.split("/", 1)[0]
    return first if first in CODE_MANIFEST_DIRS else "root_files"


def code_manifest_subset_sha256(manifest: Mapping[str, Any],
                                areas: Iterable[str] = ("src", "experiments", "scripts", "tests")) -> dict[str, Any]:
    """Digest over the manifest entries whose top-level area is in ``areas`` (FIX_C, R6-2 / R5-4).

    Use: an external reproduction package has its ``configs/`` redacted (journal values, NASEM
    values), so its full ``manifest_sha256`` can never equal the one in an internal run record; the
    subset over the code areas (default ``src experiments scripts tests``) can, and says whether the
    code is byte-identical.  ``areas`` may also contain ``configs`` or ``root_files`` (the three root
    files).  The digest covers the schema, the sorted area list and every ``(path, kind, sha256)``.
    """
    ar = sorted(set(areas))
    bad = [a for a in ar if a not in MANIFEST_AREAS]
    if bad or not ar:
        raise ValueError(f"areas must be a non-empty subset of {MANIFEST_AREAS}; got {list(areas)}")
    sel = sorted((e for e in manifest["files"] if _manifest_area(e["path"]) in ar), key=lambda e: e["path"])
    h = hashlib.sha256()
    h.update(f"{MANIFEST_SUBSET_SCHEMA}\n{json.dumps(ar)}\n".encode())
    for e in sel:
        h.update(f"{e['path']}\0{e['kind']}\0{e['sha256'] or 'NONE'}\n".encode("utf-8", "surrogateescape"))
    return {"schema": MANIFEST_SUBSET_SCHEMA, "areas": ar, "file_count": len(sel), "subset_sha256": h.hexdigest()}


def compare_code_manifests(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Per-file comparison of two code manifests (e.g. a run record's ``code_manifest`` and ``code_manifest(root)``).

    Reports changed / only-in-first / only-in-second paths, their counts per area and, per area,
    whether the area subsets are identical -- so that "the run used other code" can be narrowed to
    "only configs differ" (external package) or "tests and scripts changed after the run" (R6-5).
    """
    fa = {e["path"]: e for e in first["files"]}
    fb = {e["path"]: e for e in second["files"]}
    changed = sorted(p for p in set(fa) & set(fb)
                     if (fa[p]["sha256"], fa[p]["kind"]) != (fb[p]["sha256"], fb[p]["kind"]))
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    by_area: dict[str, dict[str, int]] = {}
    for label, paths in (("changed", changed), ("only_in_first", only_a), ("only_in_second", only_b)):
        for p in paths:
            by_area.setdefault(_manifest_area(p), {"changed": 0, "only_in_first": 0, "only_in_second": 0})[label] += 1
    return {
        "identical": not (changed or only_a or only_b),
        "manifest_sha256": [first.get("manifest_sha256"), second.get("manifest_sha256")],
        "scope_equal": first.get("scope") == second.get("scope"),
        "changed": changed, "only_in_first": only_a, "only_in_second": only_b,
        "differences_by_area": by_area,
        "area_identical": {a: a not in by_area for a in MANIFEST_AREAS},
    }


def _git_listing(root: Path) -> Optional[list[str]]:
    out = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *_GIT_PATHS)
    if out is None:
        return None
    rels = {r for r in out.decode("utf-8", "surrogateescape").split("\0") if r}
    return sorted(r for r in rels if not _is_excluded_relpath(r) and os.path.lexists(root / r))


def _git_manifest_crosscheck(root: Path, manifest: Mapping[str, Any], git_usable: bool) -> dict[str, Any]:
    if not git_usable:
        return {"status": "git_not_used", "only_in_filesystem_walk": [], "only_in_git_listing": []}
    listed = _git_listing(root)
    if listed is None:
        return {"status": "git_listing_failed", "only_in_filesystem_walk": [], "only_in_git_listing": []}
    walk = {e["path"] for e in manifest["files"]}
    gl = set(listed)
    only_walk, only_git = sorted(walk - gl), sorted(gl - walk)
    return {"status": "consistent" if not only_walk and not only_git else "differs",
            "git_listed_file_count": len(gl),
            "only_in_filesystem_walk": only_walk,   # e.g. git-ignored files: still hashed (conservative)
            "only_in_git_listing": only_git,
            "note": "git listing = tracked + untracked non-ignored files under the manifest scope, with the "
                    "manifest's exclusion rules applied; the manifest itself never depends on git"}


def code_fingerprint(repo_root: str | Path, *, manifest: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Code identity of the engine at ``repo_root`` (summary; the per-file list is ``code_manifest``)."""
    root = Path(repo_root)
    src = root / "src" / "ration_reliability"
    py = sorted(p for p in src.rglob("*.py") if "__pycache__" not in p.parts) if src.is_dir() else []
    man = dict(manifest) if manifest is not None else code_manifest(root)
    out: dict[str, Any] = {
        "code_manifest_sha256": man["manifest_sha256"],
        "code_manifest_file_count": man["file_count"],
        "code_manifest_complete": man["complete"],
        "code_manifest_scope": {"dirs": list(CODE_MANIFEST_DIRS), "root_files": list(CODE_MANIFEST_ROOT_FILES)},
        "src_tree_sha256": files_sha256(py, root=root) if py else None,
        "src_file_count": len(py),
        "src_tree_note": "legacy field: *.py under src/ration_reliability only; use code_manifest_sha256",
        "git_head": None,
        "git_dirty": None,
        "dirty_diff_hash": None,
        "dirty_diff_scope": list(_GIT_PATHS),
        "untracked_file_count": None,
        "git_error": None,
    }
    usable, err = _git_toplevel_status(root)
    if usable:
        head = _git(root, "rev-parse", "HEAD")
        if head is None:
            usable, err = False, "git work tree without a HEAD commit"
    if not usable:
        out["git_error"] = err
        out["git_manifest_crosscheck"] = _git_manifest_crosscheck(root, man, False)
        return out
    out["git_head"] = head.decode().strip()
    diff = _git(root, "diff", "HEAD", "--binary", "--", *_GIT_PATHS)
    untracked_raw = _git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", *_GIT_PATHS)
    if diff is None or untracked_raw is None:
        # never report "clean" when the diff could not be computed
        out["git_error"] = "git diff / ls-files failed; dirty state unknown"
        out["git_manifest_crosscheck"] = _git_manifest_crosscheck(root, man, True)
        return out
    h = hashlib.sha256(diff)
    files = sorted(u for u in untracked_raw.decode("utf-8", "surrogateescape").split("\0") if u.strip())
    for rel in files:
        p = root / rel
        if p.is_file():
            h.update(f"{rel}\0{file_sha256(p)}\n".encode("utf-8", "surrogateescape"))
    out["git_dirty"] = bool(diff) or bool(files)
    out["dirty_diff_hash"] = h.hexdigest()
    out["untracked_file_count"] = len(files)
    out["git_manifest_crosscheck"] = _git_manifest_crosscheck(root, man, True)
    return out


# =================================================================================================
# environment fingerprint and lock
# =================================================================================================

def _norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _site_kind(path: Optional[str | Path]) -> str:
    """Classify an install location without recording absolute (user-identifying) paths."""
    if not path:
        return "unknown"
    try:
        p = str(Path(path).resolve())
    except OSError:
        return "unknown"

    def under(base: Optional[str]) -> bool:
        if not base:
            return False
        try:
            b = str(Path(base).resolve())
        except OSError:
            return False
        return p == b or p.startswith(b + os.sep)

    try:
        if under(site.getusersitepackages()):
            return "user_site"
    except Exception:  # pragma: no cover - some virtualenvs lack the helper
        pass
    try:
        if any(under(s) for s in site.getsitepackages()):
            return "site_packages"
    except Exception:  # pragma: no cover
        pass
    for key in ("purelib", "platlib"):
        if under(sysconfig.get_paths().get(key)):
            return "site_packages"
    if under(sysconfig.get_paths().get("stdlib")):
        return "stdlib"
    return "other"


def _distribution_record(dist_name: str, import_name: str) -> dict[str, Any]:
    module_version: Optional[str] = None
    module_site = "not_importable"
    try:
        mod = importlib.import_module(import_name)
        v = getattr(mod, "__version__", None)
        module_version = str(v) if v is not None else None
        module_site = _site_kind(getattr(mod, "__file__", None))
    except Exception:
        mod = None
    metas = []
    try:
        for d in _md.distributions(name=dist_name):
            if _norm_dist(d.metadata["Name"] or "") != _norm_dist(dist_name):
                continue
            metas.append({"version": d.version, "site": _site_kind(d.locate_file(""))})
    except Exception:  # pragma: no cover - broken metadata
        pass
    metas.sort(key=lambda m: (m["version"], m["site"]))
    versions = sorted({m["version"] for m in metas})
    hazards: list[str] = []
    if module_version is not None:
        effective, source = module_version, "module.__version__"
        if versions and module_version not in versions:
            hazards.append(f"{dist_name}: imported module reports {module_version} but installed metadata says "
                           f"{versions}")
    elif len(versions) == 1:
        effective, source = versions[0], "importlib.metadata"
    else:
        same_site = sorted({m["version"] for m in metas if m["site"] == module_site})
        if len(same_site) == 1:
            effective, source = same_site[0], "importlib.metadata (same site as the imported module)"
        else:
            effective, source = None, "unresolved"
    if len(versions) > 1:
        hazards.append(f"{dist_name}: {len(metas)} installed metadata records with versions {versions}; "
                       f"the effective version is taken from {source}")
    if mod is None and not metas:
        status = "not_installed"
    elif effective is None:
        status = "ambiguous"
    else:
        status = "ok"
    return {"version": effective, "version_source": source, "status": status, "import_name": import_name,
            "module_site": module_site, "metadata_records": metas, "hazards": hazards}


def _numpy_blas() -> Optional[dict[str, Any]]:
    try:
        import numpy as np

        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        return {k: {"name": deps.get(k, {}).get("name"), "version": deps.get(k, {}).get("version")}
                for k in ("blas", "lapack")}
    except Exception:  # pragma: no cover - older numpy
        return None


def _pytest_plugins() -> list[dict[str, Optional[str]]]:
    out = []
    try:
        for ep in _md.entry_points(group="pytest11"):
            dist = getattr(ep, "dist", None)
            out.append({"name": ep.name, "distribution": getattr(dist, "name", None) if dist else None,
                        "version": getattr(dist, "version", None) if dist else None})
    except Exception:  # pragma: no cover
        pass
    return sorted(out, key=lambda e: (e["name"] or "", e["distribution"] or ""))


def _fingerprint_core(python_version: str, implementation: str, sys_platform: str, machine: str,
                      distributions: Mapping[str, Optional[str]]) -> dict[str, Any]:
    return {"schema": ENV_FINGERPRINT_SCHEMA, "python_version": python_version,
            "python_implementation": implementation, "sys_platform": sys_platform, "machine": machine,
            "distributions": {k: distributions[k] for k in sorted(distributions)}}


def environment_fingerprint() -> dict[str, Any]:
    """Fingerprint of the running interpreter and of the locked distributions.

    ``fingerprint_sha256`` hashes ``core`` only (interpreter version and implementation,
    ``sys.platform``, machine, effective versions of ``LOCKED_DISTRIBUTIONS``).  ``informational``
    (OS release, BLAS, auto-loaded pytest plugins, install sites) and ``hazards`` (e.g. two
    installed metadata records for one distribution) are recorded but not hashed.
    Nothing is installed or modified; absolute paths are not recorded.
    """
    dists = {name: _distribution_record(name, imp) for name, imp, _ in LOCKED_DISTRIBUTIONS}
    core = _fingerprint_core(platform.python_version(), sys.implementation.name, sys.platform,
                             platform.machine(), {k: v["version"] for k, v in dists.items()})
    hazards = [h for v in dists.values() for h in v["hazards"]]
    for k, v in dists.items():
        if v["status"] != "ok":
            hazards.append(f"{k}: status {v['status']}")
    sites = sorted({v["module_site"] for v in dists.values()})
    if len([s for s in sites if s in ("user_site", "site_packages")]) > 1:
        hazards.append("locked distributions are imported from more than one site directory "
                       f"({', '.join(sites)}); a change in either directory changes the environment")
    return {
        "core": core,
        "fingerprint_sha256": stable_hash(core),
        "distribution_details": dists,
        "informational": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "numpy_build_dependencies": _numpy_blas(),
            "pytest_plugins_autoloaded": _pytest_plugins(),
            "python_build": list(platform.python_build()),
            "python_compiler": platform.python_compiler(),
        },
        "hazards": hazards,
    }


def optional_group_versions(groups: Optional[Iterable[str]] = None) -> dict[str, dict[str, Any]]:
    """Effective versions of the optional groups' distributions in the running interpreter (read only).

    ``{group: {"distributions": {name: version or None}, "status": {name: ok|not_installed|...},
    "hazards": [...]}}``.  Importing the module is how the effective version is read; nothing is
    installed.  A missing package gives version ``None`` and status ``not_installed``.
    """
    names = list(OPTIONAL_DISTRIBUTION_GROUPS) if groups is None else list(groups)
    out: dict[str, dict[str, Any]] = {}
    for g in names:
        if g not in OPTIONAL_DISTRIBUTION_GROUPS:
            raise ValueError(f"unknown optional distribution group {g!r}; known: {sorted(OPTIONAL_DISTRIBUTION_GROUPS)}")
        recs = {d: _distribution_record(d, imp) for d, imp, _ in OPTIONAL_DISTRIBUTION_GROUPS[g]}
        out[g] = {"distributions": {d: r["version"] for d, r in recs.items()},
                  "status": {d: r["status"] for d, r in recs.items()},
                  "hazards": [h for r in recs.values() for h in r["hazards"]]}
    return out


def _optional_groups_definition() -> dict[str, list[dict[str, str]]]:
    return {g: [{"name": n, "import_name": i, "reason": r} for n, i, r in dists]
            for g, dists in OPTIONAL_DISTRIBUTION_GROUPS.items()}


def environment_lock_document(label: str, *, fingerprint: Optional[Mapping[str, Any]] = None,
                              generated_at: Optional[str] = None,
                              generated_by: str = "scripts/check_environment.py --write-lock",
                              optional_versions: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """A lock document with one environment entry (the running one unless ``fingerprint`` is given).

    ``optional_versions`` (shape of ``optional_group_versions()``) defaults to the running
    interpreter's optional groups; they are stored in the entry, outside the fingerprint core.
    """
    if not label or not isinstance(label, str):
        raise ValueError("environment lock: a non-empty label is required")
    fp = dict(fingerprint) if fingerprint is not None else environment_fingerprint()
    opt = dict(optional_versions) if optional_versions is not None else optional_group_versions()
    return {
        "schema": ENV_LOCK_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "generated_by": generated_by,
        "method": "versions read with importlib.metadata and the imported modules' __version__; "
                  "no package was installed, upgraded or removed",
        "locked_distributions": [{"name": n, "import_name": i, "reason": r} for n, i, r in LOCKED_DISTRIBUTIONS],
        "optional_distribution_groups": _optional_groups_definition(),
        "primary_environment_label": label,
        "environments": [environment_lock_entry(label, fp, optional=opt)],
        "check_rule": "an interpreter matches the lock when its fingerprint core (python version and "
                      "implementation, sys.platform, machine, locked distribution versions) equals the core of "
                      "one listed environment; informational fields and hazards are reported, not compared; "
                      "optional groups (optional_groups of the entry) are compared only when a check asks for "
                      "them and never enter fingerprint_sha256",
    }


def environment_lock_entry(label: str, fp: Mapping[str, Any],
                           optional: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    entry = {"label": label, "core": fp["core"], "fingerprint_sha256": fp["fingerprint_sha256"],
             "informational": fp.get("informational", {}), "hazards": list(fp.get("hazards", [])),
             "distribution_details": {k: {kk: v[kk] for kk in ("version_source", "module_site", "metadata_records")
                                          if kk in v}
                                      for k, v in fp.get("distribution_details", {}).items()}}
    if optional is not None:
        entry["optional_groups"] = {g: {d: v for d, v in info.get("distributions", {}).items() if v is not None}
                                    for g, info in optional.items()}
    return entry


def requirements_lock_text(lock: Mapping[str, Any]) -> str:
    """pip-style pins for the primary environment of ``lock`` (comments carry the context)."""
    env = _primary_entry(lock)
    core = env["core"]
    lines = [
        "# requirements-lock.txt -- exact versions of the environment in which ration_reliability_study is developed",
        "# and tested.  Mirrors the primary environment of environment.lock.json (label "
        f"{env['label']!r}, fingerprint {env['fingerprint_sha256']}).",
        f"# Generated {lock.get('generated_at')} by {lock.get('generated_by')}; versions were read, nothing was "
        "installed.",
        f"# Python {core['python_version']} ({core['python_implementation']}), {core['sys_platform']} "
        f"{core['machine']}.  Check the running interpreter with: python scripts/check_environment.py",
    ]
    groups = (("# direct dependencies", lambda r: r.startswith("direct")),
              ("# runtime requirements of the direct dependencies", lambda r: not r.startswith("direct")))
    for title, pred in groups:
        lines.append(title)
        for name, _, reason in LOCKED_DISTRIBUTIONS:
            if pred(reason):
                v = core["distributions"].get(name)
                lines.append(f"{name}=={v}" if v else f"# {name}: version unresolved in the locked environment")
    groups_def = lock.get("optional_distribution_groups") or {}
    env_opt = env.get("optional_groups") or {}
    for g, dists in groups_def.items():
        lines.append(f"# optional group {g!r}: not imported by src/, experiments/ or tests/.  pip treats the next lines as "
                     "comments; install these pins by hand only to run the scripts that need them "
                     "(python scripts/check_environment.py --require-group " + g + ")")
        for d in dists:
            v = env_opt.get(g, {}).get(d["name"])
            lines.append(f"#[optional:{g}] {d['name']}=={v}" if v else
                         f"# {d['name']}: not installed in the locked environment when the group was recorded")
    return "\n".join(lines) + "\n"


def parse_requirements_lock(text: str) -> dict[str, str]:
    """``{distribution: version}`` from ``name==version`` lines; comments and blanks ignored.

    Optional-group pins are comment lines (``#[optional:<group>] name==version``) and therefore not
    returned here; see ``parse_requirements_lock_optional``.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", line)
        if not m:
            raise ValueError(f"requirements lock: unsupported line {raw!r} (only name==version pins)")
        if m.group(1) in out:
            raise ValueError(f"requirements lock: duplicate pin for {m.group(1)}")
        out[m.group(1)] = m.group(2)
    return out


def parse_requirements_lock_optional(text: str) -> dict[str, dict[str, str]]:
    """``{group: {distribution: version}}`` from ``#[optional:<group>] name==version`` lines."""
    out: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        m = _OPTIONAL_PIN.match(raw.strip())
        if not m:
            continue
        g, name, ver = m.groups()
        if name in out.setdefault(g, {}):
            raise ValueError(f"requirements lock: duplicate optional pin for {name} in group {g}")
        out[g][name] = ver
    return out


def _primary_entry(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    label = lock.get("primary_environment_label")
    for e in lock.get("environments", []):
        if e.get("label") == label:
            return e
    raise ValueError(f"environment lock: primary environment {label!r} not listed")


def validate_environment_lock(lock: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(lock, Mapping):
        return ["lock is not a JSON object"]
    if lock.get("schema") != ENV_LOCK_SCHEMA:
        errs.append(f"schema is {lock.get('schema')!r}, expected {ENV_LOCK_SCHEMA!r}")
    names = [d.get("name") for d in lock.get("locked_distributions", []) if isinstance(d, Mapping)]
    if names != [n for n, _, _ in LOCKED_DISTRIBUTIONS]:
        errs.append(f"locked_distributions {names} differ from the code's LOCKED_DISTRIBUTIONS "
                    f"{[n for n, _, _ in LOCKED_DISTRIBUTIONS]} (update both together)")
    envs = lock.get("environments")
    if not isinstance(envs, list) or not envs:
        errs.append("no environments listed")
        return errs
    labels = [e.get("label") if isinstance(e, Mapping) else None for e in envs]
    if len(set(labels)) != len(labels) or any(not lab for lab in labels):
        errs.append("environment labels must be non-empty and unique")
    for e in envs:
        if not isinstance(e, Mapping) or not isinstance(e.get("core"), Mapping):
            errs.append("environment entry without a core")
            continue
        core = e["core"]
        if sorted(core.get("distributions", {})) != sorted(names):
            errs.append(f"{e.get('label')}: core distributions do not match locked_distributions")
        if stable_hash(core) != e.get("fingerprint_sha256"):
            errs.append(f"{e.get('label')}: stored fingerprint_sha256 does not match its core (edited by hand?)")
    # optional groups (absent in locks written before FIX_C: still valid, the groups are then "not_in_lock")
    gdef = lock.get("optional_distribution_groups")
    if gdef is not None:
        if not isinstance(gdef, Mapping):
            errs.append("optional_distribution_groups is not an object")
        else:
            code_def = _optional_groups_definition()
            for g, dists in gdef.items():
                if g not in code_def:
                    errs.append(f"optional group {g!r} is not defined in the code's OPTIONAL_DISTRIBUTION_GROUPS")
                elif [d.get("name") for d in dists if isinstance(d, Mapping)] != [d["name"] for d in code_def[g]]:
                    errs.append(f"optional group {g!r}: distributions differ from OPTIONAL_DISTRIBUTION_GROUPS "
                                "(update both together)")
            for e in envs:
                if not isinstance(e, Mapping):
                    continue
                og = e.get("optional_groups", {})
                if not isinstance(og, Mapping):
                    errs.append(f"{e.get('label')}: optional_groups is not an object")
                    continue
                for g, pins in og.items():
                    allowed = {d.get("name") for d in gdef.get(g, []) if isinstance(d, Mapping)}
                    if g not in gdef or not isinstance(pins, Mapping) or set(pins) - allowed:
                        errs.append(f"{e.get('label')}: optional group {g!r} lists distributions outside its definition")
    try:
        _primary_entry(lock)
    except ValueError as exc:
        errs.append(str(exc))
    return errs


def _optional_group_check(lock: Mapping[str, Any], env_label: Optional[str], groups: Iterable[str],
                          current_optional: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    names = list(groups)
    cur = dict(current_optional) if current_optional is not None else optional_group_versions(names)
    gdef = lock.get("optional_distribution_groups") or {}
    env = next((e for e in lock.get("environments", []) if e.get("label") == env_label), None)
    if env is None:
        env = _primary_entry(lock)
    out: dict[str, Any] = {}
    for g in names:
        want = (env.get("optional_groups") or {}).get(g)
        have = dict(cur.get(g, {}).get("distributions", {}))
        rec: dict[str, Any] = {"compared_with_environment": env.get("label"), "locked": want, "current": have}
        if g not in gdef or want is None:
            rec["status"] = "not_in_lock"
        elif any(v is None for v in have.values()) or not have:
            rec["status"] = "not_installed"
        elif {k: v for k, v in have.items()} != dict(want):
            rec["status"] = "mismatch"
        else:
            rec["status"] = "matches_lock"
        out[g] = rec
    return out


def _core_differences(locked: Mapping[str, Any], current: Mapping[str, Any]) -> list[dict[str, Any]]:
    diffs = []
    for k in ("python_version", "python_implementation", "sys_platform", "machine"):
        if locked.get(k) != current.get(k):
            diffs.append({"item": k, "locked": locked.get(k), "current": current.get(k)})
    ld, cd = locked.get("distributions", {}), current.get("distributions", {})
    for name in sorted(set(ld) | set(cd)):
        if ld.get(name) != cd.get(name):
            diffs.append({"item": f"distribution:{name}", "locked": ld.get(name), "current": cd.get(name)})
    return diffs


def check_environment_against_lock(lock_path: str | Path, *, requirements_path: Optional[str | Path] = None,
                                   current: Optional[Mapping[str, Any]] = None,
                                   optional_groups: Iterable[str] = (),
                                   current_optional: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Compare the running interpreter (or ``current``) with a lock file.

    ``status`` is one of ``ENVIRONMENT_CHECK_STATUSES``.  ``requirements_path`` defaults to
    ``requirements-lock.txt`` next to the lock; it must exist and pin exactly the primary
    environment's versions (core pins and, when the lock defines optional groups, the
    ``#[optional:<group>]`` pins), otherwise the lock is ``lock_invalid``.
    ``optional_groups`` names groups to compare as well (result in ``optional_groups``, statuses
    ``OPTIONAL_GROUP_CHECK_STATUSES``; they never change ``status``, which is about the core).
    ``current_optional`` (shape of ``optional_group_versions()``) replaces the running interpreter's.
    """
    optional_groups = list(optional_groups)
    lp = Path(lock_path)
    rp = Path(requirements_path) if requirements_path is not None else lp.parent / REQUIREMENTS_LOCK_FILE
    cur = dict(current) if current is not None else environment_fingerprint()
    out: dict[str, Any] = {"lock_path": lp.name, "requirements_path": rp.name, "lock_sha256": None,
                           "requirements_sha256": None, "current_fingerprint_sha256": cur["fingerprint_sha256"],
                           "matched_environment": None, "mismatches": [], "errors": [],
                           "informational_differences": [], "current_hazards": list(cur.get("hazards", []))}
    if not lp.is_file():
        out["status"] = "lock_missing"
        out["errors"].append(f"{lp.name} not found")
        return out
    out["lock_sha256"] = file_sha256(lp)
    try:
        lock = json.loads(lp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        out["status"] = "lock_invalid"
        out["errors"].append(f"{lp.name} unreadable: {type(exc).__name__}: {exc}")
        return out
    errs = validate_environment_lock(lock)
    if not errs:
        if not rp.is_file():
            errs.append(f"{rp.name} not found (it must mirror the primary environment)")
        else:
            out["requirements_sha256"] = file_sha256(rp)
            try:
                rtext = rp.read_text(encoding="utf-8")
                pins = parse_requirements_lock(rtext)
                prim = dict(_primary_entry(lock)["core"]["distributions"])
                if pins != {k: v for k, v in prim.items() if v is not None}:
                    errs.append(f"{rp.name} pins {pins} differ from the primary environment {prim}")
                if lock.get("optional_distribution_groups") is not None:
                    opins = parse_requirements_lock_optional(rtext)
                    popt = {g: dict(v) for g, v in (_primary_entry(lock).get("optional_groups") or {}).items() if v}
                    if opins != popt:
                        errs.append(f"{rp.name} optional pins {opins} differ from the primary environment's "
                                    f"optional_groups {popt}")
            except (OSError, ValueError) as exc:
                errs.append(str(exc))
    if errs:
        out["status"] = "lock_invalid"
        out["errors"].extend(errs)
        return out
    per_env = []
    for e in lock["environments"]:
        diffs = _core_differences(e["core"], cur["core"])
        per_env.append({"label": e["label"], "differences": diffs})
        if not diffs and e["fingerprint_sha256"] == cur["fingerprint_sha256"]:
            out["matched_environment"] = e["label"]
            info = e.get("informational", {})
            cinfo = cur.get("informational", {})
            for k in sorted(set(info) | set(cinfo)):
                if info.get(k) != cinfo.get(k):
                    out["informational_differences"].append({"item": k, "locked": info.get(k),
                                                             "current": cinfo.get(k)})
    if out["matched_environment"] is not None:
        out["status"] = "matches_lock"
    else:
        out["status"] = "mismatch"
        best = min(per_env, key=lambda x: len(x["differences"]))
        out["mismatches"] = best["differences"]
        out["closest_environment"] = best["label"]
    if optional_groups:
        out["optional_groups"] = _optional_group_check(
            lock, out["matched_environment"] or out.get("closest_environment"), optional_groups, current_optional)
    return out


def check_environment_for_run_type(run_type: str, env_check: Mapping[str, Any],
                                   manifest: Optional[Mapping[str, Any]] = None) -> None:
    """R6: pilot/official runs need a locked environment and a complete code manifest."""
    if run_type not in ("pilot", "official"):
        return
    if env_check.get("status") != "matches_lock":
        raise ValueError(f"{run_type} run: the interpreter does not match the environment lock "
                         f"(status {env_check.get('status')!r}: {env_check.get('mismatches') or env_check.get('errors')}); "
                         "run scripts/check_environment.py, or add a reviewed environment entry to the lock")
    if manifest is not None and not manifest.get("complete", False):
        raise ValueError(f"{run_type} run: code files changed while the manifest was being built "
                         f"({manifest.get('vanished_during_hashing')}); no stable code identity")


# =================================================================================================
# run identity and run record
# =================================================================================================

def _canonical(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


RESTRICTED_PREFIX = "data/restricted_local"
BUILD_IDENTITY_SCHEMA = "ration_reliability.build_identity/1"


def _build_entry(root: Path, p: str | Path) -> dict[str, Any]:
    pp = Path(p)
    if not pp.is_absolute():
        pp = root / pp
    try:
        rel: Optional[str] = pp.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = None
    restricted = rel is not None and (rel == RESTRICTED_PREFIX or rel.startswith(RESTRICTED_PREFIX + "/"))
    entry: dict[str, Any] = {"path": rel if rel is not None else pp.name, "restricted": restricted,
                             "sha256": file_sha256(pp) if pp.is_file() else None}
    if rel is None:
        entry["outside_repository"] = True       # the absolute path is not recorded (it may carry a user name)
    return entry


def build_identity(repo_root: str | Path, *, build_scripts: Iterable[str | Path], build_inputs: Iterable[str | Path],
                   build_outputs: Iterable[str | Path] = ()) -> dict[str, Any]:
    """Identity of a build step that produced a run's inputs (R3F; review round 3 F6).

    Every file enters as ``{"path", "restricted", "sha256"}`` (repository-relative path; ``restricted`` for files
    under ``data/restricted_local/``, which are recorded by path and hash only).  ``build_identity_sha256`` hashes the
    scripts and the inputs (sorted by path) -- what determined the build; the outputs are recorded next to it
    (``build_outputs_sha256``).  A missing file has ``sha256 = None`` and is listed in ``missing``; the identity is
    then marked incomplete and is refused for pilot/official records by ``build_run_record``.
    """
    root = Path(repo_root)
    scripts = sorted((_build_entry(root, p) for p in build_scripts), key=lambda e: e["path"])
    inputs = sorted((_build_entry(root, p) for p in build_inputs), key=lambda e: e["path"])
    outputs = sorted((_build_entry(root, p) for p in build_outputs), key=lambda e: e["path"])
    missing = [e["path"] for e in scripts + inputs + outputs if e["sha256"] is None]

    def key(es: list[dict[str, Any]]) -> list[tuple[str, Optional[str]]]:
        return [(e["path"], e["sha256"]) for e in es]

    return {
        "schema": BUILD_IDENTITY_SCHEMA,
        "build_identity_sha256": stable_hash(BUILD_IDENTITY_SCHEMA, key(scripts), key(inputs)),
        "build_outputs_sha256": stable_hash(BUILD_IDENTITY_SCHEMA, "outputs", key(outputs)) if outputs else None,
        "build_scripts": scripts,
        "build_inputs": inputs,
        "build_outputs": outputs,
        "missing": missing,
        "complete": not missing,
        "restricted_policy": f"files under {RESTRICTED_PREFIX}/ are recorded by repository-relative path and sha256 "
                             "only (no content, no size, no value)",
    }


def run_identity(*, code_manifest_sha256: Optional[str], environment_fingerprint_sha256: Optional[str],
                 config_hash: Optional[str], data_manifest_hash: Optional[str], protocol_hash: Optional[str],
                 run_type: str, command: str, rng_streams: Mapping[str, Any], solver_version: str,
                 tolerances: Mapping[str, Any], is_synthetic: bool,
                 build_identity_sha256: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    """``(run_identity_sha256, components)``: everything that defines what a run computed.

    ``build_identity_sha256`` (R3F) enters the components only when given, so identities computed before it existed
    are reproduced unchanged."""
    comp = {
        "schema": "ration_reliability.run_identity/1",
        "code_manifest_sha256": code_manifest_sha256,
        "environment_fingerprint_sha256": environment_fingerprint_sha256,
        "config_hash": config_hash,
        "data_manifest_hash": data_manifest_hash,
        "protocol_hash": protocol_hash,
        "run_type": run_type,
        "command": command,
        "rng_streams": _canonical(dict(rng_streams)),
        "solver_version": solver_version,
        "tolerances": _canonical(dict(tolerances or {})),
        "is_synthetic": bool(is_synthetic),
    }
    if build_identity_sha256 is not None:
        comp["build_identity_sha256"] = build_identity_sha256
    return stable_hash(comp), comp


def make_run_id(run_type: str, started_at: str, salt: Any = None) -> str:
    """``<run_type>-<YYYYmmddTHHMMSSZ>-<8 hex>``."""
    ts = started_at.replace("-", "").replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    return f"{run_type}-{ts}-{stable_hash(run_type, started_at, salt)[:8]}"


def _hash_files(paths: Iterable[str | Path]) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    for p in paths:
        pp = Path(p)
        out[str(p)] = file_sha256(pp) if pp.is_file() else None
    return out


def check_solver_settings_for_run_type(run_type: str, tolerances: Mapping[str, Any]) -> None:
    """Red-team fix C11: pilot and official runs must set ``mip_rel_gap`` explicitly.

    ``tolerances["mip_rel_gap"]`` must be a finite number >= 0 (the gap passed to HiGHS), or the
    string ``"not_applicable"`` for a run without any MILP.  ``None``/missing (solver default used
    implicitly) is refused for ``pilot``/``official``; unit tests and smoke runs are not checked.
    """
    if run_type not in ("pilot", "official"):
        return
    g = dict(tolerances or {}).get("mip_rel_gap")
    if g == "not_applicable":
        return
    try:
        ok = g is not None and not isinstance(g, bool) and float(g) >= 0 and float(g) == float(g) \
            and float(g) != float("inf")
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{run_type} run: tolerances['mip_rel_gap'] must be set explicitly (a number >= 0, or "
                         "'not_applicable' without MILP); the solver default is not recorded as a decision (C11)")


_STREAM_ROOT_RE = re.compile(r"(?:^|[^0-9A-Za-z_])root=(\d+)(?:/|$)")


def check_reserved_root_streams(run_type: str, rng_streams: Mapping[str, Any]) -> None:
    """Official-run plan batch 4 (SB-05; streams policy SP-3): a run record that is not ``official`` may not name a
    stream under a reserved formal-evaluation root (``ration_reliability.uncertainty.streams.reserved_formal_roots``).
    The message withholds the value."""
    if run_type == "official":
        return
    from ..uncertainty.streams import is_reserved_root   # lazy: io must not import the uncertainty package eagerly
    for purpose, sid in (rng_streams or {}).items():
        for m in _STREAM_ROOT_RE.finditer(str(sid)):
            if is_reserved_root(int(m.group(1))):
                raise ValueError(f"{run_type} run record: stream {purpose!r} is under a reserved formal-evaluation "
                                 "root (configs/streams_policy.yaml SP-3); only an official run may use it (value "
                                 "withheld)")


def build_run_record(*, run_type: str, command: str, repo_root: str | Path, started_at: str,
                     completed_at: str, exit_status: int, rng_streams: Mapping[str, str],
                     solver_version: str, tolerances: Mapping[str, Any],
                     config_paths: Iterable[str | Path] = (), data_paths: Iterable[str | Path] = (),
                     protocol_path: Optional[str | Path] = None, output_paths: Iterable[str | Path] = (),
                     is_synthetic: bool, run_id: Optional[str] = None,
                     extra: Optional[Mapping[str, Any]] = None,
                     environment_lock_path: Optional[str | Path] = None,
                     build_identity: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Assemble a run-record dict (does not write it).

    ``rng_streams`` maps purpose -> stream id (e.g. ``{"test": "root=1103/test/0"}``).
    Hash fields are ``None`` when the corresponding input list is empty (never invented).
    ``environment_lock_path`` defaults to ``<repo_root>/environment.lock.json``.
    ``build_identity`` (R3F; from ``build_identity()``) is stored as ``record["build_identity"]`` and its
    ``build_identity_sha256`` enters the run identity; an incomplete build identity (a build script or input
    missing), or one marked ``usable_for_pilot_official = False`` (FIX3_DEF: no build-time sidecar or drift since
    the build), is refused for pilot/official runs.
    """
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}")
    check_reserved_root_streams(run_type, rng_streams)
    check_solver_settings_for_run_type(run_type, tolerances)
    if build_identity is not None:
        if build_identity.get("schema") != BUILD_IDENTITY_SCHEMA or not build_identity.get("build_identity_sha256"):
            raise ValueError("build_identity must come from run_record.build_identity()")
        if run_type in ("pilot", "official") and not build_identity.get("complete", False):
            raise ValueError(f"{run_type} run: build identity incomplete, missing {build_identity.get('missing')}")
        if run_type in ("pilot", "official") and build_identity.get("usable_for_pilot_official") is False:
            side = build_identity.get("sidecar") or {}
            drift = [f"{d.get('path')} ({d.get('change')})" for d in side.get("drift") or []][:10]
            raise ValueError(f"{run_type} run: build identity not usable (source {build_identity.get('source')}, "
                             f"sidecar {side.get('status')}{'; drift: ' + ', '.join(drift) if drift else ''}); "
                             "rebuild on the frozen code (scripts/build_dev_case.py) so that the record carries the "
                             "build-time identity")
    root = Path(repo_root)
    config_paths, data_paths, output_paths = list(config_paths), list(data_paths), list(output_paths)
    manifest = code_manifest(root)
    code = code_fingerprint(root, manifest=manifest)
    env_fp = environment_fingerprint()
    lock_path = Path(environment_lock_path) if environment_lock_path is not None else root / ENVIRONMENT_LOCK_FILE
    env_check = check_environment_against_lock(lock_path, current=env_fp)
    check_environment_for_run_type(run_type, env_check, manifest)
    cfg_h = _hash_files(config_paths)
    dat_h = _hash_files(data_paths)
    out_h = _hash_files(output_paths)
    config_hash = stable_hash(sorted(cfg_h.items())) if cfg_h else None
    data_hash = stable_hash(sorted(dat_h.items())) if dat_h else None
    protocol_hash = file_sha256(protocol_path) if protocol_path and Path(protocol_path).is_file() else None
    identity, components = run_identity(
        code_manifest_sha256=manifest["manifest_sha256"], environment_fingerprint_sha256=env_fp["fingerprint_sha256"],
        config_hash=config_hash, data_manifest_hash=data_hash, protocol_hash=protocol_hash, run_type=run_type,
        command=command, rng_streams=rng_streams, solver_version=solver_version, tolerances=tolerances,
        is_synthetic=is_synthetic,
        build_identity_sha256=None if build_identity is None else str(build_identity["build_identity_sha256"]))
    rec: dict[str, Any] = {
        "run_id": run_id or make_run_id(run_type, started_at, (command, manifest["manifest_sha256"])),
        "run_type": run_type,
        "started_at": started_at,
        "completed_at": completed_at,
        # with git: the HEAD commit (the working-tree state is in dirty_diff_hash and code_manifest);
        # without git: the full code manifest hash (src + experiments + scripts + tests + configs + locks)
        "code_commit_or_hash": code.get("git_head") or manifest["manifest_sha256"],
        "code_manifest_sha256": manifest["manifest_sha256"],
        "src_tree_sha256": code.get("src_tree_sha256"),
        "dirty_diff_hash": code.get("dirty_diff_hash"),
        "code_fingerprint": code,
        "code_manifest": manifest,
        "run_identity_sha256": identity,
        "run_identity_components": components,
        "environment": {"fingerprint_sha256": env_fp["fingerprint_sha256"], "fingerprint": env_fp,
                        "lock_check": env_check},
        "data_manifest_hash": data_hash,
        "data_files": dat_h,
        "config_hash": config_hash,
        "config_files": cfg_h,
        "protocol_hash": protocol_hash,
        "protocol_path": None if protocol_path is None else str(protocol_path),
        "rng_streams": dict(rng_streams),
        "solver_version": solver_version,
        "tolerances": dict(tolerances),
        "command": command,
        "exit_status": int(exit_status),
        "output_paths": [str(p) for p in output_paths],
        "output_hashes": out_h,
        "is_synthetic": bool(is_synthetic),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "python": platform.python_version(),
        "packages": _package_versions(),
    }
    if build_identity is not None:
        rec["build_identity"] = _canonical(dict(build_identity))
    if extra:
        rec["extra"] = dict(extra)
    return rec


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "yaml"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover
            out[name] = "not importable"
    return out


def write_run_record(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``record`` as pretty JSON (UTF-8); refuses to overwrite an existing file."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"run record {p} already exists (run records are append-only)")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return p
