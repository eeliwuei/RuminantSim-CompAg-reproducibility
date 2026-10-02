"""Preflight of the dev_case inputs (review round 3, instruction F-4; R3F).

The round-3 reviewer ran ``run_dev_case_v1.py --dry-run`` on the delivered package and got a ``FileNotFoundError``
traceback: the package has no restricted inputs.  What is checked here:

1. without the restricted inputs the preflight lists every missing file with its rebuild condition, says the case
   cannot be rebuilt from the package, exits 2 and writes nothing;
2. with every file present it is READY (0); a changed input, an unregistered config, a failed build check, a problem
   YAML that differs from the built one or changed restricted constants make it INCONSISTENT (1); a changed builder
   is a warning (``scripts/build_dev_case.py --check`` decides);
3. only build outputs missing -> rebuild with ``scripts/build_dev_case.py``;
4. ``require_dev_case_inputs`` prints the short list and raises ``SystemExit(code)``; the CLI never prints a traceback;
5. wiring: ``run_dev_case_v1.py`` calls the preflight in ``main`` before anything is read and at the start of
   ``check_inputs``; every driver under ``experiments/`` that reads the dev_case problem calls it (directly or through
   ``check_inputs``) -- this is how a later ablation driver is held to F-4;
6. the real driver in a package-shaped copy without ``data/restricted_local/`` stops with exit 2 and the list.

Synthetic files only.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ration_reliability.build import dev_case as DC
from ration_reliability.build import preflight as PF

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
L = DC.DEV_CASE_V1


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tree(root: Path, *, outputs: bool = True, driver_extras: bool = True) -> Path:
    """A synthetic tree with every file the preflight asks for and a consistent build report."""
    files = {f"{L.cfg_dir}/{n}": f"synthetic: {n}\n" for n in L.config_names}
    files.update({L.animal_profile: "fields: {}\n", L.core_csv: "a,b\n1,2\n", L.blayer_csv: "a,b\n3,4\n",
                  L.feed_library_csv: "UID\nX\n", L.restricted_values: "values: {}\n",
                  L.constants_file: "RESTRICTED_CONSTANTS = {}\n"})
    if driver_extras:
        for r in PF.DRIVER_PROFILES["run_dev_case_v1"]:
            files[r.path] = "synthetic\n"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    if outputs:
        case = root / L.case_dir
        prob = b"problem: synthetic\n"
        for n in DC.OUTPUT_FILES:
            (case / n).write_bytes(prob if n.endswith("problem.yaml") else b"{}\n")
        inputs = {rel: _sha((root / rel).read_bytes()) for rel in DC.build_input_paths(root)}
        (case / "build_report.json").write_text(json.dumps({"summary": {"n_checks": 3, "n_failed": 0}, "inputs": inputs,
                                                            "problem_yaml_sha256": _sha(prob)}), encoding="utf-8")
    return root


def _listing(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


# =================================================================================================
# 1-3 statuses
# =================================================================================================

def test_missing_inputs_are_listed_with_rebuild_conditions_and_nothing_is_written(tmp_path):
    root = tmp_path / "pkg"
    (root / "configs" / "dev_case_v1").mkdir(parents=True)
    for n in L.config_names:
        (root / L.cfg_dir / n).write_text("x: 1\n", encoding="utf-8")
    (root / L.animal_profile).write_text("fields: {}\n", encoding="utf-8")
    before = _listing(root)
    rep = PF.preflight_dev_case(root, driver="run_dev_case_v1")
    assert rep["exit_code"] == PF.EXIT_MISSING == 2 and rep["status"].startswith("MISSING")
    missing = {m["path"]: m for m in rep["missing"]}
    for rel in (L.core_csv, L.blayer_csv, L.feed_library_csv, L.restricted_values, L.constants_file,
                f"{L.case_dir}/build_report.json", f"{L.case_dir}/dev_case_v1_problem.yaml",
                "data/restricted_local/dev_case_v1/fixb_cp_sup_rdp_scaling.yaml"):
        assert rel in missing and missing[rel]["rebuild_condition"], rel
    # the only non-restricted misses are the driver's own tracked configs, absent from this synthetic tree
    assert {m["role"] for m in rep["missing"] if not m["restricted"]} == {"driver_tracked_config"}
    assert "先补齐" in missing[f"{L.case_dir}/build_report.json"]["rebuild_condition"]
    assert rep["rebuildable_from_package"] is False and rep["writes_nothing"]
    text = PF.format_preflight(rep)
    assert "BLOCKED" in text and "退出码 2" in text and "不是求解失败" in text and "Traceback" not in text
    assert _listing(root) == before


def test_ready_and_every_kind_of_staleness(tmp_path):
    root = _tree(tmp_path / "p")
    rep = PF.preflight_dev_case(root, driver="run_dev_case_v1")
    assert rep["exit_code"] == PF.EXIT_READY, rep
    assert any("build_identity.json" in w for w in rep["warnings"])       # legacy build: no sidecar
    case = root / L.case_dir
    br = json.loads((case / "build_report.json").read_text(encoding="utf-8"))

    def status_after(mutate):
        r = _tree(tmp_path / f"q{len(list(tmp_path.iterdir()))}")
        mutate(r)
        return PF.preflight_dev_case(r)

    s = status_after(lambda r: (r / L.core_csv).write_text("a,b\n9,9\n", encoding="utf-8"))
    assert s["exit_code"] == PF.EXIT_INCONSISTENT and any(L.core_csv in p for p in s["problems"])
    s = status_after(lambda r: (r / L.cfg_dir / "extra.yaml").write_text("k: 1\n", encoding="utf-8"))
    assert s["exit_code"] == 1 and any("extra.yaml" in p for p in s["problems"])
    s = status_after(lambda r: (r / L.case_dir / "dev_case_v1_problem.yaml").write_text("changed\n", encoding="utf-8"))
    assert s["exit_code"] == 1 and any("problem.yaml" in p for p in s["problems"])

    def failed(r):
        b = json.loads((r / L.case_dir / "build_report.json").read_text(encoding="utf-8"))
        b["summary"]["n_failed"] = 2
        (r / L.case_dir / "build_report.json").write_text(json.dumps(b), encoding="utf-8")
    s = status_after(failed)
    assert s["exit_code"] == 1 and any("检查失败" in p for p in s["problems"])
    s = status_after(lambda r: (r / L.case_dir / "build_report.json").write_text("{not json", encoding="utf-8"))
    assert s["exit_code"] == 1

    # sidecar: changed restricted constants -> inconsistent; changed builder code -> warning only
    def sidecar(r, consts_sha, code_sha):
        doc = {"build_scripts": [{"path": L.constants_file, "sha256": consts_sha},
                                 {"path": "src/ration_reliability/build/dev_case.py", "sha256": code_sha}],
               "build_inputs": [], "build_identity_sha256": "x"}
        (r / L.case_dir / DC.SIDECAR).write_text(json.dumps(doc), encoding="utf-8")
    good = _sha((root / L.constants_file).read_bytes())
    s = status_after(lambda r: sidecar(r, "0" * 64, None))
    assert s["exit_code"] == 1 and any("受限构建常数" in p for p in s["problems"])
    s = status_after(lambda r: sidecar(r, good, "0" * 64))
    assert s["exit_code"] == 0 and any("构建代码" in w for w in s["warnings"])
    assert br["summary"]["n_failed"] == 0


def test_only_outputs_missing_points_to_the_build_script(tmp_path):
    root = _tree(tmp_path / "p", outputs=False)
    rep = PF.preflight_dev_case(root)
    assert rep["exit_code"] == 2 and rep["rebuildable_from_package"] is True
    assert {m["role"] for m in rep["missing"]} == {"build_output"}
    assert all("scripts/build_dev_case.py" in m["rebuild_condition"] for m in rep["missing"])


def test_require_raises_system_exit_with_the_short_list(tmp_path):
    buf = io.StringIO()
    with pytest.raises(SystemExit) as ei:
        PF.require_dev_case_inputs(tmp_path, driver="run_dev_case_v1", stream=buf)
    assert ei.value.code == 2 and "BLOCKED" in buf.getvalue() and L.core_csv in buf.getvalue()
    root = _tree(tmp_path / "ok")
    assert PF.require_dev_case_inputs(root, driver="run_dev_case_v1", stream=io.StringIO())["exit_code"] == 0


def test_cli_exit_codes_and_no_traceback(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cli = str(REPO / "scripts" / "preflight_dev_case.py")
    r = subprocess.run([PY, cli, "--root", str(tmp_path), "--driver", "run_dev_case_v1"], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 2 and "BLOCKED" in r.stdout and "Traceback" not in r.stdout + r.stderr
    rj = subprocess.run([PY, cli, "--root", str(tmp_path), "--json"], capture_output=True, text=True, env=env)
    assert rj.returncode == 2 and json.loads(rj.stdout)["exit_code"] == 2
    ok = subprocess.run([PY, cli, "--root", str(_tree(tmp_path / "ok"))], capture_output=True, text=True, env=env)
    assert ok.returncode == 0 and "READY" in ok.stdout
    bad = subprocess.run([PY, cli, "--driver", "no_such_driver"], capture_output=True, text=True, env=env)
    assert bad.returncode == 3


# =================================================================================================
# 5 wiring of the drivers
# =================================================================================================
DRIVER = REPO / "experiments" / "E0_verification" / "run_dev_case_v1.py"


def _calls(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            out.append((node.lineno, name))
    return sorted(out)


def test_run_dev_case_v1_calls_the_preflight_before_reading_anything():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = _calls(fns["main"])
    first_pf = min(ln for ln, name in calls if name == "require_dev_case_inputs")
    for reader in ("check_inputs", "load_problem", "code_manifest", "check_environment_against_lock"):
        lines = [ln for ln, name in calls if name == reader]
        assert lines and first_pf < min(lines), reader
    ci = fns["check_inputs"]
    body = [s for s in ci.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) and \
        getattr(first.value.func, "id", None) == "require_dev_case_inputs"


def test_every_driver_that_reads_the_dev_case_problem_calls_the_preflight():
    offenders = []
    for p in sorted((REPO / "experiments").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        reads_case = ("dev_case_v1_problem.yaml" in text or "PROBLEM_YAML" in text
                      or "restricted_local/dev_case_v1" in text or '"restricted_local" / "dev_case_v1"' in text)
        if not reads_case or "def main(" not in text:
            continue
        if "require_dev_case_inputs(" not in text and "check_inputs(" not in text:
            offenders.append(p.relative_to(REPO).as_posix())
    assert not offenders, ("drivers reading dev_case inputs without the R3F preflight (call "
                           "ration_reliability.build.preflight.require_dev_case_inputs at start-up): " + ", ".join(offenders))


def test_driver_in_a_package_without_restricted_inputs_stops_with_the_list(tmp_path):
    root = tmp_path / "pkg"
    for d in ("src", "experiments", "configs"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("pytest.ini", "requirements-lock.txt", "environment.lock.json"):
        shutil.copy2(REPO / f, root / f)
    r = subprocess.run([PY, str(root / "experiments" / "E0_verification" / "run_dev_case_v1.py"), "--dry-run"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=300)
    assert r.returncode == 2, r.stderr[-2000:]
    assert "BLOCKED" in r.stderr and L.core_csv in r.stderr and "Traceback" not in r.stderr
    assert not (root / "data").exists() and not (root / "results").exists()


# =================================================================================================
# FIX3_DEF (review round 3 red team F-2): strict build-identity checks for pilot / official runs
# =================================================================================================

def _write_sidecar(root: Path) -> dict:
    """A consistent build sidecar for the synthetic tree (outputs are left as they are)."""
    ident = DC.compute_dev_case_build_identity(root, entry_point=root / "scripts" / "build_dev_case.py",
                                               constants_path=root / L.constants_file)
    (root / L.case_dir / DC.SIDECAR).write_text(json.dumps(ident), encoding="utf-8")
    return ident


def test_pilot_run_type_refuses_a_missing_sidecar_and_builder_changes_other_modes_warn(tmp_path):
    root = _tree(tmp_path / "p")
    # no sidecar: a warning without run type, INCONSISTENT for pilot / official
    assert PF.preflight_dev_case(root, driver="run_dev_case_v1")["exit_code"] == PF.EXIT_READY
    for rt in ("pilot", "official"):
        rep = PF.preflight_dev_case(root, driver="run_dev_case_v1", run_type=rt)
        assert rep["exit_code"] == PF.EXIT_INCONSISTENT and rep["build_identity_checks"] == "strict"
        assert any(DC.SIDECAR in p and "pilot/official" in p for p in rep["problems"]), rep["problems"]
        assert f"run_type={rt}" in PF.format_preflight(rep)
    assert PF.preflight_dev_case(root, run_type="smoke")["exit_code"] == PF.EXIT_READY
    # a consistent sidecar: READY in every mode
    (root / "src" / "ration_reliability" / "nutrition").mkdir(parents=True)
    (root / "src" / "ration_reliability" / "nutrition" / "syn_dep.py").write_text("X = 1\n", encoding="utf-8")
    _write_sidecar(root)
    assert PF.preflight_dev_case(root, run_type="pilot")["exit_code"] == PF.EXIT_READY
    # a builder dependency changed after the build: pilot INCONSISTENT (exit 1), otherwise a warning (exit 0)
    (root / "src" / "ration_reliability" / "nutrition" / "syn_dep.py").write_text("X = 2\n", encoding="utf-8")
    strict = PF.preflight_dev_case(root, run_type="pilot")
    assert strict["exit_code"] == PF.EXIT_INCONSISTENT
    assert any("构建代码" in p and "syn_dep.py" in p for p in strict["problems"])
    lenient = PF.preflight_dev_case(root)
    assert lenient["exit_code"] == PF.EXIT_READY and any("构建代码" in w for w in lenient["warnings"])
    # a module added to the package after the build is a builder change as well
    _write_sidecar(root)
    (root / "src" / "ration_reliability" / "nutrition" / "syn_new.py").write_text("Y = 1\n", encoding="utf-8")
    added = PF.preflight_dev_case(root, run_type="official")
    assert added["exit_code"] == 1 and any("added_since_build" in p for p in added["problems"])
    # an output replaced after the build is INCONSISTENT in every mode (the problem-YAML check alone missed it)
    _write_sidecar(root)
    (root / L.case_dir / "energy_linearisation.json").write_text('{"other": 1}\n', encoding="utf-8")
    for rt in (None, "pilot"):
        rep = PF.preflight_dev_case(root, run_type=rt)
        assert rep["exit_code"] == 1 and any("energy_linearisation.json" in p for p in rep["problems"])


def test_cli_run_type_pilot_exit_code(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cli = str(REPO / "scripts" / "preflight_dev_case.py")
    root = _tree(tmp_path / "ok")
    ok = subprocess.run([PY, cli, "--root", str(root)], capture_output=True, text=True, env=env)
    assert ok.returncode == 0 and "READY" in ok.stdout
    pilot = subprocess.run([PY, cli, "--root", str(root), "--run-type", "pilot"], capture_output=True, text=True,
                           env=env)
    assert pilot.returncode == 1 and "INCONSISTENT" in pilot.stdout and "Traceback" not in pilot.stdout + pilot.stderr


# =================================================================================================
# FIX3_DEF (review round 3 red team F-4): every experiments/ driver that reads the dev_case inputs, *executed* in a
# package-shaped copy (no data/restricted_local, no results): exit 2, the short list, no traceback, nothing written
# =================================================================================================

def _dev_case_drivers() -> list:
    params = []
    for p in sorted((REPO / "experiments").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        reads_case = ("dev_case_v1_problem.yaml" in text or "PROBLEM_YAML" in text
                      or "restricted_local/dev_case_v1" in text or '"restricted_local" / "dev_case_v1"' in text)
        if not reads_case or "def main(" not in text:
            continue
        declared = {n.args[0].value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", None) == "add_argument" and n.args
                    and isinstance(n.args[0], ast.Constant)}
        modes = [] if "add_mutually_exclusive_group(required=True)" in text else [()]
        modes += [(f,) for f in ("--dry-run", "--check-only", "--tiny") if f in declared]
        rel = p.relative_to(REPO).as_posix()
        for m in modes:
            params.append(pytest.param(rel, m, id=f"{p.stem}{''.join(m) or '(no args)'}"))
    return params


@pytest.fixture(scope="module")
def package_shaped_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("pkgshape") / "pkg"
    for d in ("src", "experiments", "configs", "scripts"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("pytest.ini", "requirements-lock.txt", "environment.lock.json"):
        shutil.copy2(REPO / f, root / f)
    return root


@pytest.mark.parametrize("driver,args", _dev_case_drivers())
def test_every_dev_case_driver_executed_in_a_package_copy_stops_with_the_list(package_shaped_copy, driver, args):
    root = package_shaped_copy
    assert not (root / "data").exists() and not (root / "results").exists()
    before = _listing(root)
    r = subprocess.run([PY, str(root / driver), *args], capture_output=True, text=True, cwd=str(root),
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=300)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, out[-3000:]
    assert r.returncode == 2, (r.returncode, out[-2000:])
    assert "BLOCKED" in r.stderr and L.core_csv in r.stderr
    assert not (root / "data").exists() and not (root / "results").exists()
    assert _listing(root) == before


def test_the_driver_list_is_not_empty_and_covers_the_known_drivers():
    ids = {p.values[0] for p in _dev_case_drivers()}
    assert {"experiments/E0_verification/run_dev_case_v1.py",
            "experiments/E0_verification/replay_dev_case_eval.py"} <= ids
