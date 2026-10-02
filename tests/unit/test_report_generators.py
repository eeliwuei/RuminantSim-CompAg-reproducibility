"""Tracked generators of the round-3 diagnostic reports (FIX3_DEF; review round 3 red team F6, E, D6/C).

The red team found that ``reports/reference_energy_audit.csv``, ``reports/domain_applicability.csv`` and
``reports/sd_scaling_sources.csv`` were produced by scripts that lived only in a session scratchpad, so the reports could
not be recomputed from the repository.  The generators are now ``scripts/r3d_energy_domain_diagnostic.py`` and
``scripts/make_sd_scaling_sources.py`` (inside the code manifest).  What is checked:

1. the SD registry is re-rendered by the tracked script byte for byte (no restricted input needed), and every S2 row
   states that the lower end of the declared range is the SD-S2 calibration point, not a sourced lower bound (E);
2. the domain CSV is internally consistent: rates = counts / states; the premise-conditioned unknown = the old unknown
   + the states added by the premise; a variant with every state in the domain adds nothing (D6/C);
3. both scripts are in the code-manifest scope; without the restricted inputs the diagnostic stops with exit 2 and the
   list (no traceback, nothing written);
4. holder side only (skipped without the restricted inputs and the saved run): the diagnostic re-computes both CSVs
   byte for byte (``--check``; evaluation only, no LP / MILP, about 5 s; development material, not a result).
"""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ration_reliability.io.run_record import in_code_manifest_scope

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
SD_SCRIPT = REPO / "scripts" / "make_sd_scaling_sources.py"
DIAG_SCRIPT = REPO / "scripts" / "r3d_energy_domain_diagnostic.py"
RUN = "pilot-20260924T205621Z-93d8654c"


def _load(path: Path, name: str):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sd_scaling_sources_csv_is_reproduced_by_the_tracked_script():
    mk = _load(SD_SCRIPT, "make_sd_scaling_sources_under_test")
    data, summ = mk.render()
    assert data == (REPO / "reports" / "sd_scaling_sources.csv").read_bytes()
    assert summ == {"n_s2": 48, "n_registered": 3, "n_borrowed": 45, "n_reg_rows": 17}
    rows = [r for r in csv.DictReader((REPO / "reports" / "sd_scaling_sources.csv").open(encoding="utf-8"))
            if r["row_kind"] == "S2_cell"]
    assert len(rows) == 48
    # red team E: the lower end of the declared range is the SD-S2 point (48/48) and every row says what that is
    assert all(r["sd_range_low"] == r["ratio_SD_S2"] for r in rows)
    assert all(mk.LOW_END_ORIGIN in r["notes"] and "不是有据的下限" in r["notes"] for r in rows)
    assert all(r["identified_variance_decomposition"] == "false" and r["variance_basis_label"] == "unidentified"
               for r in rows)


def test_domain_csv_premise_conditioned_columns_are_consistent():
    rows = list(csv.DictReader((REPO / "reports" / "domain_applicability.csv").open(encoding="utf-8")))
    assert len(rows) == 76
    for r in rows:
        n = int(r["n_states"])
        nv, nu_old = int(r["n_event_violated"]), int(r["n_event_unknown_t_rows_read_regardless_of_premise"])
        nu_pc, add = int(r["n_event_unknown_premise_conditioned"]), int(r["n_event_unknown_added_by_premise"])
        assert nu_pc == nu_old + add
        assert float(r["event_rate_lower"]) == pytest.approx(nv / n, abs=5e-7)
        assert float(r["event_rate_upper_premise_conditioned"]) == pytest.approx((nv + nu_pc) / n, abs=5e-7)
        assert float(r["event_rate_upper_t_rows_read_regardless_of_premise"]) == pytest.approx((nv + nu_old) / n,
                                                                                                abs=5e-7)
        not_in = int(r["n_not_assessable"]) + int(r["n_undefined"])
        assert add <= not_in
        if not_in == 0 or r["event_contains_t51_rows"] != "True":
            assert add == 0
        assert int(r["n_event_not_violated_in_domain"]) <= int(r["n_in_domain_conditional"])
        assert "premise_conditioned" in r["event_reading_note"]


def test_generators_are_in_the_code_manifest_scope_and_stop_cleanly_without_inputs(tmp_path):
    for p in (SD_SCRIPT, DIAG_SCRIPT):
        assert in_code_manifest_scope(p.relative_to(REPO).as_posix())
    root = tmp_path / "pkg"
    for d in ("src", "experiments", "configs", "scripts"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("pytest.ini", "requirements-lock.txt", "environment.lock.json"):
        shutil.copy2(REPO / f, root / f)
    r = subprocess.run([PY, str(root / "scripts" / "r3d_energy_domain_diagnostic.py"), "--check-only"],
                       capture_output=True, text=True, cwd=str(root), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                       timeout=300)
    assert r.returncode == 2 and "BLOCKED" in r.stderr and "Traceback" not in r.stdout + r.stderr
    assert not (root / "data").exists() and not (root / "results").exists()


_HOLDER = (REPO / "data" / "restricted_local" / "pilot" / RUN / "rations_full.csv").is_file() and \
    (REPO / "results" / "pilot" / RUN / "uncertainty_model.json").is_file() and \
    (REPO / "data" / "restricted_local" / "dev_case_v1" / "build_report.json").is_file()


@pytest.mark.skipif(not _HOLDER, reason="restricted dev_case inputs or the saved development run not present "
                                        "(holder-side check only)")
def test_diagnostic_recomputes_both_csvs_byte_for_byte():
    r = subprocess.run([PY, str(DIAG_SCRIPT), "--check", "--out-dir", "reports"], capture_output=True, text=True,
                       cwd=str(REPO), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=600)
    if r.returncode == 1 and "INCONSISTENT" in r.stderr and '"all_ok"' not in r.stdout:
        pytest.skip("dev_case inputs not consistent with their build (preflight INCONSISTENT); rebuild first")
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert '"reference_energy_audit.csv": true' in r.stdout and '"domain_applicability.csv": true' in r.stdout
