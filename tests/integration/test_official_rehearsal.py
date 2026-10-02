"""``run_endpoint_ablation.py --official-rehearsal`` (official-run plan batch 4, §3 test 13): holder side only.

The official job matrix of root index 0 (9 cells + the diagnostics job, D-539) through the same job runner (one process
per job) and the same merge as the official run, on the **development** root at tiny sizes.  Checked (labels, counts,
statuses and booleans only -- no rate, cost or restricted value is printed):

* the output is ``debug`` and lands only under ``data/restricted_local/debug/<run_id>/`` (nothing under ``results/``
  or the official restricted mirror);
* every job wrote ``DONE`` and a verifying ``SHA256SUMS``; the merge wrote per-root tables (``root0/``), the run record
  (``smoke``, label ``debug``), ``MERGE_DONE``; ``scripts/verify_official_outputs.py`` passes on it;
* the MAIN9 arm of reference problem v2 (with the premise planning row) has its own training event equal to its
  reference event (``main9_training_event``) on every own-world row, and likewise FULL11 / PART6P5;
* every arm was validated in mode ``official`` with the injected run context (plan F5);
* the diagnostics rows are never comparable-set members and carry no cost column; no reserved root value appears.
The rehearsal directory is removed at the end (it is this test's own debug output).
"""

from __future__ import annotations

import csv
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
VERIFY = REPO / "scripts" / "verify_official_outputs.py"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

needs_restricted = pytest.mark.skipif(not (REPO / DC.DEV_CASE_V1.constants_file).is_file(),
                                      reason="restricted dev_case inputs not present (holder-side check only)")


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_endpoint_ablation_rehearsal_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _listing(p: Path) -> set[str]:
    return {x.name for x in p.iterdir()} if p.is_dir() else set()


def _json_objects(text: str) -> list:
    dec, i, out = json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)
        out.append(obj)
    return out


@needs_restricted
def test_rehearsal_writes_debug_output_only_training_events_match_and_the_merge_succeeds():
    AB = _load_driver()
    rep = PF.preflight_dev_case(REPO, driver="run_dev_case_v1", extra=AB.EXTRA_REQUIREMENTS)
    if rep["exit_code"] != PF.EXIT_READY:
        pytest.skip(f"dev_case inputs not READY (preflight {rep['status']}); holder-side check skipped")
    OV = AB.OV2
    debug = REPO / OV.DEBUG_ROOT
    watched = [REPO / "results" / "official", REPO / "results" / "pilot", REPO / "results" / "smoke",
               REPO / "data" / "restricted_local" / "official", REPO / "data" / "restricted_local" / "pilot"]
    before, before_debug = [_listing(p) for p in watched], _listing(debug)
    r = subprocess.run([PY, str(DRIVER), "--official-rehearsal", "--workers", "2"], capture_output=True, text=True,
                       env=ENV, timeout=1800)
    new = sorted(_listing(debug) - before_debug)
    try:
        assert r.returncode == 0, r.stderr[-3000:]
        assert len(new) == 1 and new[0].startswith(OV.REHEARSAL_RUN_PREFIX)
        run_id = new[0]
        assert [_listing(p) for p in watched] == before                   # nothing under results/ or official/
        launch, merge = _json_objects(r.stdout)[-2:]
        assert launch["failed"] == [] and launch["n_jobs"] == 10
        assert (merge["mode"], merge["output_label"], merge["pooled_across_roots"]) == ("rehearsal", "debug", False)
        assert merge["end_of_run_identity_unchanged"] is True and merge["roots_k"] == [0]
        per = merge["per_root"]["0"]
        assert (per["n_jobs"], per["n_cells"], per["n_diagnostics_jobs"]) == (10, 9, 1)
        tec = per["training_event_consistency"]
        assert tec["all_equal"] is True
        assert tec["by_arm"]["MAIN9"]["n"] >= 5 and tec["by_arm"]["MAIN9"]["n_equal"] == tec["by_arm"]["MAIN9"]["n"]
        route = OV.official_output_route(run_id)
        pub, res = REPO / route["public_dir"], REPO / route["restricted_dir"]
        assert OV.verify_sha_manifest(pub) == [] and OV.verify_sha_manifest(res) == []
        assert (pub / OV.MERGE_DONE).is_file() and (res / OV.MERGE_DONE).is_file()
        plan = json.loads((pub / OV.RUN_PLAN_FILE).read_text(encoding="utf-8"))
        assert len(plan["jobs"]) == 10 and plan["sizes"]["test"] == 500 and plan["mode"] == "rehearsal"
        for j in plan["jobs"]:
            for base in (pub, res):
                assert OV.read_done(base / "jobs" / j["job_id"])["job_id"] == j["job_id"]
            jr = json.loads((pub / "jobs" / j["job_id"] / "job_record.json").read_text(encoding="utf-8"))
            assert jr["build_mode"] == "official" and jr["stream_registry_ok"] is True           # F5 injected
            assert jr["run_context_injected"]["primary_assumption_id"] == OV.PRIMARY_ASSUMPTION_ID
            assert jr["code_unchanged_during_job"] is True and jr["thread_env"]["OMP_NUM_THREADS"] == "1"
            assert set(jr["world_fingerprints"]) == set(OV.SD_POINTS)
        rec = json.loads((pub / "run_record.json").read_text(encoding="utf-8"))
        assert rec["run_type"] == "smoke" and rec["extra"]["output_label"] == "debug"
        assert rec["extra"]["pooled_across_roots"] is False and all(
            v["exit_status"] == 0 for v in rec["extra"]["jobs"].values())
        assert "debug / rehearsal" in (pub / "NOT_FOR_MANUSCRIPT.md").read_text(encoding="utf-8")
        rows = list(csv.DictReader((pub / "root0" / "endpoint_ablation.csv").open(encoding="utf-8")))
        assert rows and {x["root_k"] for x in rows} == {"0"}
        assert {x["cell_id"] for x in rows} == {c["cell_id"] for c in OV.OFFICIAL_CONFIG["cells"]}
        scored = [x for x in rows if x["evaluation_world"]]
        assert {x["evaluation_world"] for x in scored} == set(OV.SD_POINTS) and {x["n_test"] for x in scored} == {"500"}
        assert all(x[f"{OV.MAIN_EVENT_PLAN_DOMAIN}_cp_upper"] != "" for x in scored)          # the v2 main event
        drows = list(csv.DictReader((pub / "root0" / "diagnostics_rows.csv").open(encoding="utf-8")))
        assert drows and {x["comparable_set_member"] for x in drows} == {"False"}
        assert {x["diag_training_world"] for x in drows if x["diag_lp"] in ("A2", "E1_M2")} <= {"SD-H0", "SD-S2"}
        assert "cost_usd_per_head_d" not in drows[0] and "diag_h_canonical" not in drows[0]
        comp = json.loads((pub / "root0" / "comparable_sets.json").read_text(encoding="utf-8"))
        assert not any(":DIAG-" in m["label"] for s in comp["sets"] for m in s.get("members") or [])
        v = subprocess.run([PY, str(VERIFY), run_id], capture_output=True, text=True, env=ENV, timeout=600)
        assert v.returncode == 0, v.stdout[-2000:]
        assert json.loads(v.stdout)["all_passed"] is True
        doc = yaml.safe_load((REPO / "configs" / "streams_policy.yaml").read_text(encoding="utf-8"))
        needles = [str(int(x["root_seed"])).encode() for x in doc["reserved_formal_streams"]["roots"]]
        leaked = any(nd in p.read_bytes() for p in pub.rglob("*") if p.is_file() for nd in needles)
        assert not leaked, "a reserved root value appears in the rehearsal output"
    finally:
        for name in new:
            shutil.rmtree(debug / name, ignore_errors=True)
