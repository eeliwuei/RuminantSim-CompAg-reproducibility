"""``run_endpoint_ablation.py --official-plan`` (official-run plan batch 3): the official cell / job matrix and compute
estimate, data only.

What is checked:

1. wiring: ``main`` reaches the official plan only after the R3F preflight; without the restricted inputs the mode
   stops with the BLOCKED list and creates nothing;
2. holder side (restricted inputs present and the preflight READY; skipped otherwise): ``--official-plan`` builds
   reference problem v2, its three arms and one world per declared SD point **without a draw, a solve, a file write or
   a directory** -- measured by the wrapped entry points of ``DryRunCounters``, not literals; it prints 9 cells, 19 cell
   jobs + 1 diagnostics job (root 0: 9 cells; roots 1-2: MAIN9 x 5; D-539: diagnostics on root 0 only), per-cell
   counts for 3 N and 5 evaluation worlds,
   and a compute estimate; the output names no reserved root value (roots appear as indices k only); the development
   ``--dry-run`` keeps its six cells.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ration_reliability.build import dev_case as DC
from ration_reliability.build import preflight as PF

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
DRIVER = REPO / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_endpoint_ablation_official_plan_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load_driver()


def _reserved_roots() -> set[int]:
    doc = yaml.safe_load((REPO / "configs" / "streams_policy.yaml").read_text(encoding="utf-8"))
    return {int(r["root_seed"]) for r in doc["reserved_formal_streams"]["roots"]}


def _contains_root(obj, roots: set[int]) -> bool:
    if isinstance(obj, dict):
        return any(_contains_root(k, roots) or _contains_root(v, roots) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_contains_root(v, roots) for v in obj)
    if isinstance(obj, bool):
        return False
    if isinstance(obj, (int, float)):
        return obj in roots
    if isinstance(obj, str):
        return any(f"root={r}/" in obj or obj == str(r) for r in roots)
    return False


def test_main_reaches_the_official_plan_only_after_the_preflight():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [(n.lineno, n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
             for n in ast.walk(fn) if isinstance(n, ast.Call)]
    first_pf = min(ln for ln, name in calls if name == "require_dev_case_inputs")
    op = [ln for ln, name in calls if name == "official_plan_main"]
    assert op and first_pf < min(op)
    # the official plan builds its context without a draw
    src = ast.get_source_segment(DRIVER.read_text(encoding="utf-8"),
                                 next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                      and n.name == "official_plan_main"))
    assert "draw=False" in src and "DryRunCounters()" in src and "RandomStreams" not in src


def test_official_plan_without_restricted_inputs_stops_with_the_list(tmp_path):
    root = tmp_path / "pkg"
    for d in ("src", "experiments", "configs"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "reports").mkdir()
    shutil.copy2(REPO / "reports" / "sd_scaling_sources.csv", root / "reports" / "sd_scaling_sources.csv")
    for f in ("pytest.ini", "requirements-lock.txt", "environment.lock.json"):
        shutil.copy2(REPO / f, root / f)
    r = subprocess.run([PY, str(root / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"),
                        "--official-plan"], capture_output=True, text=True, env=ENV, timeout=300)
    assert r.returncode == 2, r.stderr[-2000:]
    assert "BLOCKED" in r.stderr and "Traceback" not in r.stderr
    assert not (root / "data").exists() and not (root / "results").exists()


needs_restricted = pytest.mark.skipif(not (REPO / DC.DEV_CASE_V1.constants_file).is_file(),
                                      reason="restricted dev_case inputs not present (holder-side check only)")


def _ready():
    rep = PF.preflight_dev_case(REPO, driver="run_dev_case_v1", extra=AB.EXTRA_REQUIREMENTS)
    if rep["exit_code"] != PF.EXIT_READY:
        pytest.skip(f"dev_case inputs not READY (preflight {rep['status']}); holder-side check skipped")


def _listing(p: Path) -> set[str]:
    return {x.name for x in p.iterdir()} if p.is_dir() else set()


@needs_restricted
def test_official_plan_prints_the_matrix_with_measured_zero_draws_solves_and_writes():
    _ready()
    watched = (REPO / AB.DEBUG_ROOT, REPO / "results" / "pilot", REPO / "results",
               REPO / "data" / "restricted_local" / "pilot", REPO / "data" / "restricted_local")
    before = [_listing(p) for p in watched]
    r = subprocess.run([PY, str(DRIVER), "--official-plan"], capture_output=True, text=True, env=ENV, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert (out["official_plan"], out["files_written"], out["draws"], out["solves"], out["dirs_created"]) == (
        True, 0, 0, 0, 0)
    meas = out["measured_counts"]
    assert meas["method"].startswith("measured") and meas["calls"] == {}
    assert [_listing(p) for p in watched] == before                          # nothing written anywhere
    m = out["matrix"]
    assert (m["n_cells"], m["n_cell_jobs"], m["n_diagnostics_jobs"], m["n_jobs"]) == (9, 19, 1, 20)   # D-539
    assert {k: len(v) for k, v in m["cells_by_root_k"].items()} == {"0": 9, "1": 5, "2": 5}
    assert m["N_ladder"] == [128, 512, 1024] and m["alphas"] == [0.05, 0.10, 0.01]
    assert m["evaluation_worlds"] == ["SD-H0", "SD-S2", "SD-SRC14", "SD-12MO", "SD-SRC12"]
    assert len(out["jobs"]) == 20 and all(j["root_k"] in (0, 1, 2) for j in out["jobs"])
    assert m["diagnostics_roots_k"] == [0] and m["workers_declared"] == 8
    pl = out["plan_official"]
    cells = {c["cell_id"]: c for c in pl["cells"]}
    assert len(cells) == 9
    # plan() counts for 3 N and 5 worlds: 3 N x 4 alpha_train multipliers x 3 alphas M2 MILPs per cell
    assert all(c["milp_solves_methods"] == 36 and c["milp_min_violation"] == 3 and c["milp_frontier_and_check"] == 6
               for c in cells.values())
    rmax = cells["SDH0_MAIN9"]["rations_max"]
    assert pl["compute_estimate"]["reference_evaluation_states"] == 9 * rmax * 5 * 10000
    assert cells["SDH0_MAIN9"]["problem_id"] == "dev_case_v1|v2|MAIN9"
    assert cells["SDH0_FULL11"]["problem_id"] == "dev_case_v1|v2"
    assert cells["SDS2_PART6P5"]["problem_id"] == "dev_case_v1|v2|S2"
    assert len({c["world_fingerprint"] for c in cells.values()}) == 5               # one world per SD point
    assert cells["SDH0_MAIN9"]["world_fingerprint"] == cells["SDH0_FULL11"]["world_fingerprint"]
    ce = out["compute_estimate"]
    assert ce["per_cell_job_worst_case_solver_s"] == 36 * 300.0 + (60.0 + 180.0 + 360.0) + 6 * 300.0
    assert ce["serial_worst_case_s"] > 0 and set(ce["parallel_worst_case_by_workers"]) == {"4", "8"}
    for w in ce["parallel_worst_case_by_workers"].values():
        assert w["lower_bound_s"] <= w["list_scheduling_upper_bound_s"] <= ce["serial_worst_case_s"]
    chk = out["checks"]
    assert chk["sd_registry_all_points"]["ok"] and chk["reference_table"]["table_version"] == "reference_problem_v2"
    assert all(v["ok"] for v in chk["validators"].values()) and all(chk["v2_build_checks"].values())
    assert chk["import_time_sources"]["status"] == "consistent"
    leaked = _contains_root(out, _reserved_roots())
    assert not leaked, "a reserved root value appears in the --official-plan output"
