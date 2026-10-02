"""Run identity, code content manifest and environment lock (second review R6, 2026-09-25).

What is checked:

1. The code identity covers ``src``, ``experiments``, ``scripts``, ``tests``, ``configs`` and the root
   files ``pytest.ini``, ``requirements-lock.txt``, ``environment.lock.json`` -- tracked or untracked,
   with or without ``.git`` (the manifest is a file-system walk; git is only cross-checked).
2. Changing only a driver under ``experiments/`` or only a script under ``scripts/`` changes the full
   run identity (the legacy ``src_tree_sha256`` does not see such a change); so does a configuration
   change, an environment change or a change of the run settings.  Byte-code and cache files do not.
3. A ZIP-style extraction without ``.git`` yields exactly the same manifest as the git checkout.
4. ``dirty_diff_hash`` covers experiments/scripts edits and the *content* of untracked files.
5. The environment lock round-trips; mismatching, hand-edited and missing locks are detected; pilot
   and official records are refused unless the interpreter matches the lock.

All projects built here are throw-away fixtures under ``tmp_path`` (the git repositories too; the
project repository is only read).  The tests do not assume that the interpreter running them equals
the committed lock: locks used for pass/fail logic are generated from the running interpreter.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ration_reliability.hashing import file_sha256, stable_hash
from ration_reliability.io import run_record as RR
from ration_reliability.io import (
    CODE_MANIFEST_DIRS,
    ENVIRONMENT_CHECK_STATUSES,
    ENVIRONMENT_LOCK_FILE,
    LOCKED_DISTRIBUTIONS,
    REQUIREMENTS_LOCK_FILE,
    build_run_record,
    check_environment_against_lock,
    code_fingerprint,
    code_manifest,
    environment_fingerprint,
    write_run_record,
)
from ration_reliability.io.run_record import (
    _GIT_PATHS,
    _manifest_digest,
    environment_lock_document,
    parse_requirements_lock,
    requirements_lock_text,
    validate_environment_lock,
)

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git executable not available on this machine")
REQUIRED_DIRS = {"src", "experiments", "scripts", "tests", "configs"}
REQUIRED_ROOT_FILES = {"pytest.ini", "requirements-lock.txt", "environment.lock.json"}

FIXTURE_FILES = {
    "src/ration_reliability/__init__.py": "VERSION = 'fixture'\n",
    "src/ration_reliability/core.py": "def f(x):\n    return 2 * x\n",
    "experiments/E0_verification/driver.py": "import sys\nprint('driver', sys.argv)\n",
    "scripts/prep.py": "def prepare(rows):\n    return sorted(rows)\n",
    "tests/test_fixture.py": "def test_ok():\n    assert True\n",
    "configs/methods.yaml": "alpha: 0.05\nmethod: M0_nominal\n",
    "pytest.ini": "[pytest]\ntestpaths = tests\n",
}


def _write_lock(root: Path, fingerprint=None, label="fixture-env") -> dict:
    doc = environment_lock_document(label, fingerprint=fingerprint or environment_fingerprint(),
                                    generated_at="2026-09-25T00:00:00+00:00", generated_by="test fixture")
    (root / ENVIRONMENT_LOCK_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (root / REQUIREMENTS_LOCK_FILE).write_text(requirements_lock_text(doc), encoding="utf-8")
    return doc


def _make_project(root: Path, *, lock: bool = True) -> Path:
    for rel, text in FIXTURE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    if lock:
        _write_lock(root)
    return root


def _record(root: Path, **over):
    kw = dict(run_type="smoke", command="python experiments/E0_verification/driver.py --seed 1103", repo_root=root,
              started_at="2026-09-25T00:00:00+00:00", completed_at="2026-09-25T00:00:01+00:00", exit_status=0,
              rng_streams={"opt": "root=1103/opt", "test": "root=1103/test"}, solver_version="HiGHS fixture",
              tolerances={"mip_rel_gap": 0.0}, config_paths=[root / "configs" / "methods.yaml"], data_paths=[],
              protocol_path=None, output_paths=[], is_synthetic=True)
    kw.update(over)
    return build_run_record(**kw)


def _git_run(root: Path, *args: str) -> None:
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    subprocess.run([GIT, "-C", str(root), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + os.devnull, *args],
                   check=True, capture_output=True, env=env)


def _git_init_commit(root: Path) -> None:
    _git_run(root, "init", "-q")
    _git_run(root, "add", "-A")
    _git_run(root, "commit", "-q", "--no-verify", "-m", "fixture")


def _touch(p: Path, suffix: str = "# edited\n") -> None:
    p.write_text(p.read_text(encoding="utf-8") + suffix, encoding="utf-8")


def _paths(man) -> set[str]:
    return {e["path"] for e in man["files"]}


# =================================================================================================
# 1. scope
# =================================================================================================

def test_scope_covers_drivers_scripts_tests_configs_and_locks():
    assert set(CODE_MANIFEST_DIRS) == REQUIRED_DIRS
    assert set(_GIT_PATHS) == REQUIRED_DIRS | REQUIRED_ROOT_FILES   # dirty-diff scope == manifest scope
    assert set(RR.CODE_MANIFEST_ROOT_FILES) == REQUIRED_ROOT_FILES


def test_repository_manifest_lists_every_code_area(repo_root):
    man = code_manifest(repo_root)
    paths = _paths(man)
    for d in CODE_MANIFEST_DIRS:
        assert man["file_count_by_dir"][d] > 0, d
    assert any(p.startswith("experiments/") and p.endswith(".py") for p in paths)
    assert {"scripts/check_environment.py", "tests/unit/test_run_identity.py",
            "src/ration_reliability/io/run_record.py"} <= paths
    assert REQUIRED_ROOT_FILES <= paths and not man["missing_root_files"]
    assert not any("__pycache__" in p or p.endswith((".pyc", ".pyo")) for p in paths)
    # the total fingerprint is reproducible from the per-file entries
    assert _manifest_digest(man["scope"], man["files"]) == man["manifest_sha256"]
    for e in man["files"]:
        if e["path"] in REQUIRED_ROOT_FILES:
            assert e["sha256"] == file_sha256(repo_root / e["path"])


def test_manifest_ignores_bytecode_and_caches_but_not_sources(tmp_path):
    root = _make_project(tmp_path / "p")
    m0 = code_manifest(root)
    (root / "src/ration_reliability/__pycache__").mkdir()
    (root / "src/ration_reliability/__pycache__/core.cpython-311.pyc").write_bytes(b"\x00bytecode")
    (root / "experiments/E0_verification/stray.pyc").write_bytes(b"\x00")
    (root / "tests/.pytest_cache").mkdir()
    (root / "tests/.pytest_cache/v").write_text("cache", encoding="utf-8")
    (root / "configs/.DS_Store").write_bytes(b"finder")
    assert code_manifest(root)["manifest_sha256"] == m0["manifest_sha256"]
    (root / "experiments/E0_verification/helper.py").write_text("X = 1\n", encoding="utf-8")   # a new source file
    m1 = code_manifest(root)
    assert m1["manifest_sha256"] != m0["manifest_sha256"]
    assert _paths(m1) - _paths(m0) == {"experiments/E0_verification/helper.py"}


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs extra privileges on Windows")
def test_symlinks_are_recorded_not_followed(tmp_path):
    root = _make_project(tmp_path / "p")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.py").write_text("SECRET = 1\n", encoding="utf-8")
    os.symlink(outside, root / "scripts" / "linked_dir")
    man = code_manifest(root)
    ent = {e["path"]: e for e in man["files"]}
    assert ent["scripts/linked_dir"]["kind"] == "symlink"
    assert "scripts/linked_dir/big.py" not in ent


# =================================================================================================
# 2. full run identity
# =================================================================================================

@pytest.mark.parametrize("rel", ["experiments/E0_verification/driver.py", "scripts/prep.py"])
def test_run_identity_changes_when_only_a_driver_or_script_changes(tmp_path, rel):
    root = _make_project(tmp_path / "p")
    r0, r0_again = _record(root), _record(root)
    assert r0["run_identity_sha256"] == r0_again["run_identity_sha256"]          # deterministic
    assert len(r0["run_identity_sha256"]) == 64
    _touch(root / rel)
    r1 = _record(root)
    assert r1["run_identity_sha256"] != r0["run_identity_sha256"]
    assert r1["code_manifest_sha256"] != r0["code_manifest_sha256"]
    assert r1["code_commit_or_hash"] != r0["code_commit_or_hash"]                # no git: manifest hash
    assert r1["src_tree_sha256"] == r0["src_tree_sha256"]                        # legacy src hash is blind to it
    before = {e["path"]: e["sha256"] for e in r0["code_manifest"]["files"]}
    after = {e["path"]: e["sha256"] for e in r1["code_manifest"]["files"]}
    assert {p for p in after if after[p] != before.get(p)} == {rel}


def test_run_identity_changes_when_configuration_changes(tmp_path):
    root = _make_project(tmp_path / "p")
    outside_cfg = tmp_path / "restricted" / "problem.yaml"         # e.g. a generated problem file outside the scope
    outside_cfg.parent.mkdir()
    outside_cfg.write_text("dmi: 20\n", encoding="utf-8")
    cfgs = [root / "configs" / "methods.yaml", outside_cfg]
    r0 = _record(root, config_paths=cfgs)
    _touch(root / "configs" / "methods.yaml")                       # tracked configuration
    r1 = _record(root, config_paths=cfgs)
    assert r1["run_identity_sha256"] != r0["run_identity_sha256"] and r1["config_hash"] != r0["config_hash"]
    outside_cfg.write_text("dmi: 21\n", encoding="utf-8")           # configuration outside the code scope
    r2 = _record(root, config_paths=cfgs)
    assert r2["code_manifest_sha256"] == r1["code_manifest_sha256"]
    assert r2["run_identity_sha256"] != r1["run_identity_sha256"]


def test_run_identity_tracks_environment_and_settings_but_not_timestamps_or_outputs(tmp_path, monkeypatch):
    root = _make_project(tmp_path / "p")
    base = _record(root)
    out = tmp_path / "out.json"
    out.write_text("{}", encoding="utf-8")
    same = _record(root, started_at="2026-09-26T00:00:00+00:00", completed_at="2026-09-26T01:00:00+00:00",
                   exit_status=1, output_paths=[out], run_id="smoke-fixture")
    assert same["run_identity_sha256"] == base["run_identity_sha256"]
    for over in ({"rng_streams": {"opt": "root=2207/opt", "test": "root=2207/test"}},
                 {"tolerances": {"mip_rel_gap": 1e-4}}, {"solver_version": "HiGHS other"},
                 {"command": "python experiments/E0_verification/driver.py --seed 2207"},
                 {"data_paths": [root / "configs" / "methods.yaml"]}):
        assert _record(root, **over)["run_identity_sha256"] != base["run_identity_sha256"], over
    real = environment_fingerprint()

    def other_numpy():
        fp = copy.deepcopy(real)
        fp["core"]["distributions"]["numpy"] = "0.0.0-other"
        fp["fingerprint_sha256"] = stable_hash(fp["core"])
        return fp

    monkeypatch.setattr(RR, "environment_fingerprint", other_numpy)
    changed = _record(root)
    assert changed["environment"]["fingerprint_sha256"] != base["environment"]["fingerprint_sha256"]
    assert changed["run_identity_sha256"] != base["run_identity_sha256"]
    assert changed["environment"]["lock_check"]["status"] == "mismatch"


def test_run_record_on_this_repository_has_identity_and_environment(tmp_path, repo_root, toy_yaml_path):
    rec = build_run_record(run_type="unit_test", command="pytest tests/unit/test_run_identity.py", repo_root=repo_root,
                           started_at=RR.utc_now(), completed_at=RR.utc_now(), exit_status=0,
                           rng_streams={"test": "root=1103/test"}, solver_version="n/a", tolerances={},
                           config_paths=[toy_yaml_path], is_synthetic=True)
    comp = rec["run_identity_components"]
    assert rec["run_identity_sha256"] == stable_hash(comp)
    assert comp["code_manifest_sha256"] == rec["code_manifest_sha256"] == rec["code_manifest"]["manifest_sha256"]
    assert comp["environment_fingerprint_sha256"] == rec["environment"]["fingerprint_sha256"]
    assert rec["environment"]["lock_check"]["status"] in ENVIRONMENT_CHECK_STATUSES
    assert rec["code_fingerprint"]["dirty_diff_scope"] == list(_GIT_PATHS)
    p = write_run_record(rec, tmp_path / "run_record.json")               # JSON-serialisable
    assert json.loads(p.read_text(encoding="utf-8"))["run_identity_sha256"] == rec["run_identity_sha256"]


def test_file_vanishing_during_hashing_is_flagged_and_refused_for_pilot(tmp_path, monkeypatch):
    root = _make_project(tmp_path / "p")
    real = RR.file_sha256

    def flaky(path, *a, **k):
        if str(path).endswith("scripts/prep.py"):
            raise FileNotFoundError(path)
        return real(path, *a, **k)

    monkeypatch.setattr(RR, "file_sha256", flaky)
    man = code_manifest(root)
    assert man["complete"] is False and man["vanished_during_hashing"] == ["scripts/prep.py"]
    with pytest.raises(ValueError, match="changed while the manifest"):
        _record(root, run_type="pilot")
    assert _record(root)["code_manifest"]["complete"] is False             # smoke: recorded, not hidden


# =================================================================================================
# 3./4. git and no-git
# =================================================================================================

@needs_git
def test_no_git_extraction_gives_the_same_manifest_as_the_git_checkout(tmp_path):
    root = _make_project(tmp_path / "repo")
    _git_init_commit(root)
    (root / "scripts" / "untracked_new.py").write_text("print('new')\n", encoding="utf-8")   # untracked
    _touch(root / "experiments/E0_verification/driver.py")                                    # modified
    fp_git = code_fingerprint(root)
    man_git = code_manifest(root)
    assert fp_git["git_error"] is None and len(fp_git["git_head"]) == 40
    assert fp_git["git_manifest_crosscheck"]["status"] == "consistent"
    assert fp_git["git_manifest_crosscheck"]["git_listed_file_count"] == man_git["file_count"]
    extract = tmp_path / "unzipped"
    shutil.copytree(root, extract, ignore=shutil.ignore_patterns(".git"))
    fp_zip = code_fingerprint(extract)
    man_zip = code_manifest(extract)
    assert fp_zip["git_head"] is None and fp_zip["git_error"]
    assert man_zip["files"] == man_git["files"]
    assert man_zip["manifest_sha256"] == man_git["manifest_sha256"] == fp_zip["code_manifest_sha256"]
    assert "scripts/untracked_new.py" in _paths(man_zip)
    rec_zip, rec_git = _record(extract), _record(root)
    assert rec_zip["code_commit_or_hash"] == man_zip["manifest_sha256"]
    assert rec_git["code_commit_or_hash"] == fp_git["git_head"]
    # same bytes -> same code and environment components, with or without .git.  (config_hash keys are the
    # paths as given, so the full run identity of the two copies differs only through their locations.)
    for k in ("code_manifest_sha256", "environment_fingerprint_sha256"):
        assert rec_zip["run_identity_components"][k] == rec_git["run_identity_components"][k]


@needs_git
def test_dirty_diff_hash_covers_experiments_scripts_root_files_and_untracked_content(tmp_path):
    root = _make_project(tmp_path / "repo")
    _git_init_commit(root)
    c0 = code_fingerprint(root)
    assert c0["git_dirty"] is False and c0["untracked_file_count"] == 0
    drv = root / "experiments/E0_verification/driver.py"
    original = drv.read_text(encoding="utf-8")
    _touch(drv)
    c1 = code_fingerprint(root)
    assert c1["git_dirty"] is True and c1["dirty_diff_hash"] != c0["dirty_diff_hash"]
    drv.write_text(original, encoding="utf-8")
    assert code_fingerprint(root)["dirty_diff_hash"] == c0["dirty_diff_hash"]
    extra = root / "scripts" / "extra.py"
    extra.write_text("A = 1\n", encoding="utf-8")
    c3 = code_fingerprint(root)
    assert c3["git_dirty"] is True and c3["untracked_file_count"] == 1
    extra.write_text("A = 2\n", encoding="utf-8")                  # same name, different content
    c4 = code_fingerprint(root)
    assert c4["dirty_diff_hash"] != c3["dirty_diff_hash"]
    extra.unlink()
    _touch(root / "pytest.ini")
    assert code_fingerprint(root)["dirty_diff_hash"] != c0["dirty_diff_hash"]


@needs_git
def test_gitignored_file_in_scope_is_reported_and_still_hashed(tmp_path):
    root = _make_project(tmp_path / "repo")
    (root / ".gitignore").write_text("configs/local_*.yaml\n", encoding="utf-8")
    _git_init_commit(root)
    (root / "configs" / "local_site.yaml").write_text("threads: 4\n", encoding="utf-8")
    fp = code_fingerprint(root)
    cc = fp["git_manifest_crosscheck"]
    assert cc["status"] == "differs"
    assert cc["only_in_filesystem_walk"] == ["configs/local_site.yaml"] and cc["only_in_git_listing"] == []
    assert "configs/local_site.yaml" in _paths(code_manifest(root))       # conservative: still in the identity
    assert fp["git_dirty"] is False                                       # git itself ignores it


@needs_git
def test_project_nested_in_another_repository_does_not_borrow_its_head(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "README.txt").write_text("outer repo\n", encoding="utf-8")
    _git_init_commit(outer)
    inner = _make_project(outer / "unzipped_project")
    fp = code_fingerprint(inner)
    assert fp["git_head"] is None and fp["dirty_diff_hash"] is None
    assert "top level" in fp["git_error"]
    assert _record(inner)["code_commit_or_hash"] == code_manifest(inner)["manifest_sha256"]


# =================================================================================================
# 5. environment lock
# =================================================================================================

def test_environment_fingerprint_core_is_complete_and_path_free():
    fp = environment_fingerprint()
    core = fp["core"]
    assert set(core["distributions"]) == {n for n, _, _ in LOCKED_DISTRIBUTIONS}
    assert core["python_version"].count(".") == 2 and core["python_implementation"]
    assert fp["fingerprint_sha256"] == stable_hash(core) == environment_fingerprint()["fingerprint_sha256"]
    dump = json.dumps(fp)
    assert str(Path.home()) not in dump and sys.prefix not in dump        # no absolute install paths


def test_environment_lock_round_trip_and_failure_modes(tmp_path):
    fp = environment_fingerprint()
    doc = _write_lock(tmp_path, fingerprint=fp, label="here")
    lock, req = tmp_path / ENVIRONMENT_LOCK_FILE, tmp_path / REQUIREMENTS_LOCK_FILE
    assert validate_environment_lock(doc) == []
    ok = check_environment_against_lock(lock, current=fp)
    assert ok["status"] == "matches_lock" and ok["matched_environment"] == "here" and ok["mismatches"] == []
    assert parse_requirements_lock(req.read_text(encoding="utf-8")) == \
        {k: v for k, v in fp["core"]["distributions"].items() if v is not None}

    other = copy.deepcopy(doc)                                             # a valid lock for another numpy
    core = other["environments"][0]["core"]
    core["distributions"]["numpy"] = "0.0.1"
    other["environments"][0]["fingerprint_sha256"] = stable_hash(core)
    lock.write_text(json.dumps(other), encoding="utf-8")
    req.write_text(requirements_lock_text(other), encoding="utf-8")
    mm = check_environment_against_lock(lock, current=fp)
    assert mm["status"] == "mismatch"
    assert [d["item"] for d in mm["mismatches"]] == ["distribution:numpy"]
    assert mm["mismatches"][0]["locked"] == "0.0.1"

    both = copy.deepcopy(other)                                            # a second listed environment matches
    second = copy.deepcopy(doc["environments"][0])
    second["label"] = "second"
    both["environments"].append(second)
    lock.write_text(json.dumps(both), encoding="utf-8")
    assert check_environment_against_lock(lock, current=fp)["matched_environment"] == "second"
    dup = copy.deepcopy(both)                                              # labels must be unique
    dup["environments"][1]["label"] = "here"
    lock.write_text(json.dumps(dup), encoding="utf-8")
    assert check_environment_against_lock(lock, current=fp)["status"] == "lock_invalid"

    edited = copy.deepcopy(doc)                                            # hand-edited: fingerprint not recomputed
    edited["environments"][0]["core"]["distributions"]["scipy"] = "9.9.9"
    lock.write_text(json.dumps(edited), encoding="utf-8")
    req.write_text(requirements_lock_text(doc), encoding="utf-8")
    bad = check_environment_against_lock(lock, current=fp)
    assert bad["status"] == "lock_invalid" and any("fingerprint" in e for e in bad["errors"])

    lock.write_text(json.dumps(doc), encoding="utf-8")                     # pins disagree with the lock
    req.write_text(requirements_lock_text(doc).replace(f"pandas=={fp['core']['distributions']['pandas']}",
                                                       "pandas==0.0.1"), encoding="utf-8")
    assert check_environment_against_lock(lock, current=fp)["status"] == "lock_invalid"
    req.unlink()
    assert check_environment_against_lock(lock, current=fp)["status"] == "lock_invalid"
    lock.unlink()
    assert check_environment_against_lock(lock, current=fp)["status"] == "lock_missing"
    fewer = copy.deepcopy(doc)                                             # lock and code must list the same set
    fewer["locked_distributions"] = fewer["locked_distributions"][:-1]
    assert any("LOCKED_DISTRIBUTIONS" in e for e in validate_environment_lock(fewer))


def test_pilot_and_official_records_need_a_matching_environment_lock(tmp_path):
    good = _make_project(tmp_path / "good")
    rec = _record(good, run_type="pilot")
    assert rec["environment"]["lock_check"]["status"] == "matches_lock"
    fp = copy.deepcopy(environment_fingerprint())
    fp["core"]["python_version"] = "0.0.0"
    fp["fingerprint_sha256"] = stable_hash(fp["core"])
    bad = _make_project(tmp_path / "bad", lock=False)
    _write_lock(bad, fingerprint=fp)
    none = _make_project(tmp_path / "none", lock=False)
    for root in (bad, none):
        for rt in ("pilot", "official"):
            with pytest.raises(ValueError, match="environment lock"):
                _record(root, run_type=rt)
        smoke = _record(root)                                              # smoke/unit_test: recorded, not refused
        assert smoke["environment"]["lock_check"]["status"] in ("mismatch", "lock_missing")
    with pytest.raises(ValueError, match="mip_rel_gap"):                  # the C11 rule is still applied first
        _record(good, run_type="pilot", tolerances={})


def test_committed_lock_files_are_valid_and_mirror_each_other(repo_root):
    lock = json.loads((repo_root / ENVIRONMENT_LOCK_FILE).read_text(encoding="utf-8"))
    assert validate_environment_lock(lock) == []
    req_text = (repo_root / REQUIREMENTS_LOCK_FILE).read_text(encoding="utf-8")
    assert req_text == requirements_lock_text(lock)
    primary = next(e for e in lock["environments"] if e["label"] == lock["primary_environment_label"])
    assert parse_requirements_lock(req_text) == primary["core"]["distributions"]
    assert {"numpy", "scipy", "pandas", "PyYAML", "pytest"} <= set(primary["core"]["distributions"])
    # environment-agnostic: another interpreter may legitimately mismatch, but the lock is never missing/invalid
    chk = check_environment_against_lock(repo_root / ENVIRONMENT_LOCK_FILE)
    assert chk["status"] in ("matches_lock", "mismatch"), chk["errors"]


# =================================================================================================
# CLI: scripts/check_environment.py
# =================================================================================================

def _cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(repo_root / "scripts" / "check_environment.py"), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_check_environment_script(tmp_path, repo_root):
    proj = _make_project(tmp_path / "p")
    out, rep = tmp_path / "manifest.json", tmp_path / "report.json"
    r = _cli(repo_root, "--repo", str(proj), "--code-manifest-out", str(out), "--json-out", str(rep))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "matches_lock"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["files"] == code_manifest(proj)["files"]
    assert written["manifest_sha256"] == code_manifest(proj)["manifest_sha256"]
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["environment_check"]["status"] == "matches_lock"
    assert _cli(repo_root, "--repo", str(proj), "--json-out", str(rep)).returncode == 3          # no overwrite
    assert _cli(repo_root, "--repo", str(proj), "--write-lock", "--label", "x").returncode == 3  # lock exists
    assert _cli(repo_root, "--repo", str(proj), "--append-environment", "--label", "x").returncode == 3  # same env

    fresh = _make_project(tmp_path / "fresh", lock=False)
    assert _cli(repo_root, "--repo", str(fresh)).returncode == 2                                  # lock missing
    w = _cli(repo_root, "--repo", str(fresh), "--write-lock", "--label", "fresh-env")
    assert w.returncode == 0, w.stderr
    assert validate_environment_lock(json.loads((fresh / ENVIRONMENT_LOCK_FILE).read_text(encoding="utf-8"))) == []

    fp = copy.deepcopy(environment_fingerprint())
    fp["core"]["distributions"]["pytest"] = "0.0.1"
    fp["fingerprint_sha256"] = stable_hash(fp["core"])
    other = _make_project(tmp_path / "other", lock=False)
    _write_lock(other, fingerprint=fp)
    assert _cli(repo_root, "--repo", str(other)).returncode == 1                                  # mismatch


# =================================================================================================
# FIX_C (2026-09-25): optional distribution groups, import coverage of the lock, manifest comparison
# =================================================================================================

def _opt(version):
    return {"holder_verification": {"distributions": {"PyMuPDF": version}}}


def test_optional_group_is_locked_outside_the_core(tmp_path):
    fp = environment_fingerprint()
    assert "PyMuPDF" not in fp["core"]["distributions"]            # the engine never imports it
    doc = environment_lock_document("here", fingerprint=fp, generated_at="2026-09-25T00:00:00+00:00",
                                    generated_by="test fixture", optional_versions=_opt("9.9.9"))
    assert validate_environment_lock(doc) == []
    assert doc["environments"][0]["fingerprint_sha256"] == fp["fingerprint_sha256"]   # core identity unchanged
    lock, req = tmp_path / ENVIRONMENT_LOCK_FILE, tmp_path / REQUIREMENTS_LOCK_FILE
    lock.write_text(json.dumps(doc), encoding="utf-8")
    text = requirements_lock_text(doc)
    req.write_text(text, encoding="utf-8")
    assert "#[optional:holder_verification] PyMuPDF==9.9.9" in text.splitlines()
    assert "PyMuPDF" not in RR.parse_requirements_lock(text)        # pip-visible pins stay the core set
    assert RR.parse_requirements_lock_optional(text) == {"holder_verification": {"PyMuPDF": "9.9.9"}}

    def chk(cur):
        return check_environment_against_lock(lock, current=fp, optional_groups=["holder_verification"],
                                              current_optional=cur)
    ok = chk(_opt("9.9.9"))
    assert ok["status"] == "matches_lock" and ok["optional_groups"]["holder_verification"]["status"] == "matches_lock"
    mm = chk(_opt("1.0.0"))
    assert mm["status"] == "matches_lock"                            # the core decision is unaffected ...
    assert mm["optional_groups"]["holder_verification"]["status"] == "mismatch"   # ... the group is reported
    assert chk(_opt(None))["optional_groups"]["holder_verification"]["status"] == "not_installed"
    assert "optional_groups" not in check_environment_against_lock(lock, current=fp)  # compared only on request

    req.write_text(text.replace("PyMuPDF==9.9.9", "PyMuPDF==9.9.8"), encoding="utf-8")   # pins disagree
    assert check_environment_against_lock(lock, current=fp)["status"] == "lock_invalid"
    req.write_text(text, encoding="utf-8")

    bad = copy.deepcopy(doc)                                         # an undefined distribution in a group
    bad["environments"][0]["optional_groups"]["holder_verification"]["numpy"] = "1"
    assert any("outside its definition" in e for e in validate_environment_lock(bad))
    old = copy.deepcopy(doc)                                         # a lock written before the groups existed
    del old["optional_distribution_groups"]
    del old["environments"][0]["optional_groups"]
    assert validate_environment_lock(old) == []
    lock.write_text(json.dumps(old), encoding="utf-8")
    req.write_text(requirements_lock_text(old), encoding="utf-8")
    old_chk = check_environment_against_lock(lock, current=fp, optional_groups=["holder_verification"],
                                             current_optional=_opt("9.9.9"))
    assert old_chk["status"] == "matches_lock"
    assert old_chk["optional_groups"]["holder_verification"]["status"] == "not_in_lock"
    with pytest.raises(ValueError, match="unknown optional distribution group"):
        RR.optional_group_versions(["no_such_group"])


def _third_party_imports(repo_root: Path) -> dict[str, set[str]]:
    import ast
    local = {"ration_reliability", "conftest"}
    files = [p for d in CODE_MANIFEST_DIRS if (repo_root / d).is_dir() for p in (repo_root / d).rglob("*.py")
             if "__pycache__" not in p.parts]
    local |= {p.stem for p in files}
    found: dict[str, set[str]] = {}
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n not in sys.stdlib_module_names and n not in local:
                    found.setdefault(n, set()).add(p.relative_to(repo_root).as_posix())
    return found


def test_every_third_party_import_is_locked(repo_root):
    """R6-3 / FIX_C: the lock must cover every third-party import (the missed ``fitz`` case)."""
    found = _third_party_imports(repo_root)
    core = {imp for _, imp, _ in LOCKED_DISTRIBUTIONS}
    optional = {imp: g for g, ds in RR.OPTIONAL_DISTRIBUTION_GROUPS.items() for _, imp, _ in ds}
    unlocked = {m: sorted(f) for m, f in found.items() if m not in core and m not in optional}
    assert not unlocked, f"third-party imports without a lock entry: {unlocked}"
    for mod, grp in optional.items():            # an optional group may only be imported outside the engine
        users = found.get(mod, set())
        assert all(u.startswith("scripts/") for u in users), (mod, grp, sorted(users))


def test_committed_lock_pins_the_holder_verification_group(repo_root):
    lock = json.loads((repo_root / ENVIRONMENT_LOCK_FILE).read_text(encoding="utf-8"))
    primary = next(e for e in lock["environments"] if e["label"] == lock["primary_environment_label"])
    pins = primary["optional_groups"]["holder_verification"]
    assert set(pins) == {"PyMuPDF"} and pins["PyMuPDF"]
    req = (repo_root / REQUIREMENTS_LOCK_FILE).read_text(encoding="utf-8")
    assert RR.parse_requirements_lock_optional(req) == {"holder_verification": pins}
    assert "import fitz" in (repo_root / "scripts" / "verify_restricted_inputs.py").read_text(encoding="utf-8")


def test_compare_code_manifests_and_code_subset(tmp_path):
    root = _make_project(tmp_path / "p")
    before = code_manifest(root)
    _touch(root / "configs" / "methods.yaml", "# redacted for an external package\n")
    after_cfg = code_manifest(root)
    cmp = RR.compare_code_manifests(before, after_cfg)
    assert not cmp["identical"] and cmp["changed"] == ["configs/methods.yaml"]
    assert cmp["area_identical"]["configs"] is False
    assert all(cmp["area_identical"][a] for a in ("src", "experiments", "scripts", "tests", "root_files"))
    sub = RR.code_manifest_subset_sha256
    assert sub(before)["subset_sha256"] == sub(after_cfg)["subset_sha256"]       # code subset unchanged
    assert sub(before, ["configs"])["subset_sha256"] != sub(after_cfg, ["configs"])["subset_sha256"]
    _touch(root / "scripts" / "prep.py")
    (root / "experiments" / "E0_verification" / "new_driver.py").write_text("print(1)\n", encoding="utf-8")
    after_code = code_manifest(root)
    cmp2 = RR.compare_code_manifests(after_cfg, after_code)
    assert cmp2["changed"] == ["scripts/prep.py"]
    assert cmp2["only_in_second"] == ["experiments/E0_verification/new_driver.py"]
    assert sub(after_cfg)["subset_sha256"] != sub(after_code)["subset_sha256"]
    assert RR.compare_code_manifests(after_code, code_manifest(root))["identical"]
    with pytest.raises(ValueError):
        sub(before, ["data"])


def test_check_environment_script_optional_group_and_run_record_comparison(tmp_path, repo_root):
    proj = _make_project(tmp_path / "p", lock=False)
    fp = environment_fingerprint()
    here = RR.optional_group_versions()
    installed = here["holder_verification"]["distributions"]["PyMuPDF"]
    doc = environment_lock_document("fixture-env", fingerprint=fp, generated_at="2026-09-25T00:00:00+00:00",
                                    generated_by="test fixture", optional_versions=_opt("0.0.1"))
    (proj / ENVIRONMENT_LOCK_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (proj / REQUIREMENTS_LOCK_FILE).write_text(requirements_lock_text(doc), encoding="utf-8")
    assert _cli(repo_root, "--repo", str(proj)).returncode == 0                  # group reported, not required
    r = _cli(repo_root, "--repo", str(proj), "--require-group", "holder_verification")
    assert r.returncode == 1                                                     # required and mismatching
    assert json.loads(r.stdout)["optional_groups"]["holder_verification"]["status"] in ("mismatch", "not_installed")
    if installed:                                                                # record the real version
        u = _cli(repo_root, "--repo", str(proj), "--update-optional-groups")
        assert u.returncode == 0, u.stderr
        assert _cli(repo_root, "--repo", str(proj), "--require-group", "holder_verification").returncode == 0
        again = _cli(repo_root, "--repo", str(proj), "--update-optional-groups")
        assert json.loads(again.stdout.splitlines()[0])["changed"] is False   # first line = the update result
    rec = _record(proj)
    rp = tmp_path / "run_record.json"
    write_run_record(rec, rp)
    _touch(proj / "scripts" / "prep.py")
    c = _cli(repo_root, "--repo", str(proj), "--compare-run-record", str(rp))
    out = json.loads(c.stdout)["run_record_comparisons"][0]
    assert out["identical"] is False and out["differences_by_area"] == {
        "scripts": {"changed": 1, "only_in_first": 0, "only_in_second": 0}}
    assert out["code_subset_sha256"]["run_record"] != out["code_subset_sha256"]["now"]
