"""2 x 2 (+ MAIN9) endpoint ablation driver (review round 3, instruction C; R3C; FIX3_BC): wiring, guards, --dry-run
and --tiny.

FIX3_BC (round-3 red team C-1..C-4) adds: the comparable set uses the domain-conditioned main event and reports the
R3C membership next to it; the MAIN9 arm (built, validated, consistency-checked, its training event checked against
the reference evaluator); the frozen fingerprint lives in a pin file that must be git-anchored (tracked, HEAD = pin
commit, clean tree) before ``development``; the dry run's zero counts are measured by wrapped entry points (and the
instrument itself is tested); ``--full`` needs a user authorisation file listing the host.

What is checked (no full ablation is run anywhere; instruction hard rule of round 3):

1. wiring: ``main`` calls the R3F preflight before anything is read; without the restricted inputs (package-shaped copy)
   ``--dry-run`` stops with exit 2, the short BLOCKED list and no traceback, and creates nothing;
2. guards: a reserved formal-evaluation root (``configs/streams_policy.yaml``) is refused; both SD arms equal the
   registered ratios of ``reports/sd_scaling_sources.csv`` and a changed registry stops the driver; the specification
   fingerprint detects a changed specification file; the frozen block of ``docs/reference_problem_v1.md`` parses and is
   internally consistent;
3. the comparable-set rule on synthetic rows: members only if the main reference event meets alpha on that world,
   coverage counts entries without a ration, costs are compared only inside the set;
4. holder side (restricted inputs present and the preflight READY; skipped otherwise, with the reason):
   ``--dry-run`` builds the four cells without a draw or a solve and writes nothing; ``--tiny`` (N = 16, test 500)
   runs the whole pipeline at toy size, writes only under ``data/restricted_local/debug/`` with the label ``debug``,
   scores every ration on both SD worlds with the one reference evaluator, and its own-world training event equals the
   matching reference event (H0 eleven for FULL11, S2 six for PART6P5).  The tiny output is removed afterwards.
"""

from __future__ import annotations

import ast
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
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_endpoint_ablation_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load_driver()


# =================================================================================================
# 1 wiring
# =================================================================================================

def test_main_calls_the_preflight_before_reading_anything():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            calls.append((node.lineno, f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")))
    first_pf = min(ln for ln, name in calls if name == "require_dev_case_inputs")
    for reader in ("build_context", "code_manifest", "check_environment_against_lock", "check_seed", "freeze_status"):
        lines = [ln for ln, name in calls if name == reader]
        assert lines and first_pf < min(lines), reader
    text = DRIVER.read_text(encoding="utf-8")
    assert "require_dev_case_inputs(" in text and "def main(" in text        # held to F-4 by test_preflight.py too


def test_dry_run_without_restricted_inputs_stops_with_the_list(tmp_path):
    root = tmp_path / "pkg"
    for d in ("src", "experiments", "configs"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "reports").mkdir()
    shutil.copy2(REPO / "reports" / "sd_scaling_sources.csv", root / "reports" / "sd_scaling_sources.csv")
    for f in ("pytest.ini", "requirements-lock.txt", "environment.lock.json"):
        shutil.copy2(REPO / f, root / f)
    r = subprocess.run([PY, str(root / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"), "--dry-run"],
                       capture_output=True, text=True, env=ENV, timeout=300)
    assert r.returncode == 2, r.stderr[-2000:]
    assert "BLOCKED" in r.stderr and DC.DEV_CASE_V1.core_csv in r.stderr and "Traceback" not in r.stderr
    assert "run_endpoint_ablation" in r.stderr
    assert not (root / "data").exists() and not (root / "results").exists()
    # a mode is required; no silent default run
    r2 = subprocess.run([PY, str(DRIVER)], capture_output=True, text=True, env=ENV, timeout=120)
    assert r2.returncode == 2 and "required" in r2.stderr
    # the full ablation needs an explicit statement that the host is authorised; it stops before reading anything
    r3 = subprocess.run([PY, str(root / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"), "--full"],
                        capture_output=True, text=True, env=ENV, timeout=120)
    assert r3.returncode == 2 and "--authorised-compute" in r3.stderr and "BLOCKED" not in r3.stderr
    assert not (root / "data").exists() and not (root / "results").exists()


# =================================================================================================
# 2 guards
# =================================================================================================

def test_reserved_formal_roots_are_refused_and_the_development_root_is_allowed():
    doc = yaml.safe_load((REPO / "configs" / "streams_policy.yaml").read_text(encoding="utf-8"))
    roots = [int(r["root_seed"]) for r in doc["reserved_formal_streams"]["roots"]]
    assert AB.reserved_roots(REPO) == set(roots) and len(roots) == 3
    for s in roots:
        with pytest.raises(SystemExit, match="reserved"):
            AB.check_seed(s, REPO)
    assert AB.check_seed(1103, REPO)["seed"] == 1103 == AB.ABLATION_CONFIG["seed"]


def _case_ids():
    inv = yaml.safe_load((REPO / "configs" / "dev_case_v1" / "inventory.yaml").read_text(encoding="utf-8"))
    return tuple(g["ingredient_id"] for g in inv["ingredients"])


def test_sd_arms_equal_the_registered_ratios_and_a_changed_registry_stops(tmp_path):
    ids = _case_ids()
    chk = AB.sd_registry_check(ids, REPO)
    assert chk["ok"] and chk["n_cells"] == 48
    rows = list(csv.DictReader((REPO / "reports" / "sd_scaling_sources.csv").open(encoding="utf-8")))
    cols = list(rows[0].keys())
    for r in rows:
        if r["row_kind"] == "S2_cell" and r["ingredient_id"] == "corn_silage_typical" and r["component"] == "DM":
            r["ratio_SD_S2"] = "1/3.0"                                    # a different narrowed arm
    (tmp_path / "reports").mkdir()
    with open(tmp_path / "reports" / "sd_scaling_sources.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="SD registry check failed"):
        AB.sd_registry_check(ids, tmp_path)


def _spec_copy(root: Path) -> None:
    for rel in AB.SPEC_FILES + AB.SPEC_RESTRICTED_FILES:
        if (REPO / rel).is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, root / rel)


def test_spec_fingerprint_detects_a_changed_file_and_the_frozen_pin(tmp_path):
    root = tmp_path / "r"
    _spec_copy(root)
    cur = AB.spec_fingerprint(root)
    assert set(cur["files"]) == set(AB.SPEC_FILES) and cur["restricted_files_sha256_only"]
    assert AB.freeze_status(root)["status"] == "no_frozen_fingerprint"
    AB.write_freeze_pin(root, note="unit test")
    st = AB.freeze_status(root)
    # FIX3_BC (C-3): equal content is not enough -- an unanchored pin (here: not a git repository) never allows
    # a development label
    assert st["status"] == "matches_frozen_unanchored" and st["anchor"]["anchored"] is False
    assert AB.output_route("full", True, st["status"])["label"] == "debug"
    p = root / "configs" / "dev_case_v1" / "reference_constraints.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    st = AB.freeze_status(root)
    assert st["status"] == "differs_from_frozen" and st["differs"] == ["configs/dev_case_v1/reference_constraints.csv"]
    # the old frozen block in docs/reference_problem_v1.md is not read any more (an untracked document is no anchor)
    assert AB.FREEZE_PIN.parts[0] == "experiments"                            # inside the code-manifest scope
    fr = AB.frozen_fingerprint(REPO)
    if fr is not None:                                                        # the repository pin parses and is
        from ration_reliability.hashing import stable_hash                    # internally consistent
        assert set(fr["files"]) == set(AB.SPEC_FILES)
        assert fr["digest"] == stable_hash(AB.OV2.SPEC_DIGEST_TAG, sorted(fr["files"].items()),     # v2 pin (batch 4)
                                           sorted(fr["restricted_files_sha256_only"].items()))
    r = subprocess.run([PY, str(DRIVER), "--print-spec-fingerprint"], capture_output=True, text=True, env=ENV,
                       timeout=120)
    assert r.returncode == 0 and set(json.loads(r.stdout)["files"]) == set(AB.SPEC_FILES)


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), "-c", "user.name=fix3bc-test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_freeze_pin_is_anchored_only_at_the_pin_commit_with_a_clean_tree(tmp_path):
    """FIX3_BC (round-3 red team C-3): a local throw-away repository (never the project repository, never pushed)."""
    root = tmp_path / "g"
    _spec_copy(root)
    for rel in ("experiments/E1_cost_reliability/run_endpoint_ablation.py",):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")               # restricted inputs are never tracked
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "spec")
    AB.write_freeze_pin(root, note="unit test")
    st = AB.freeze_status(root)
    assert st["status"] == "matches_frozen_unanchored" and "not tracked" in st["anchor"]["reason"]
    _git(root, "add", str(AB.FREEZE_PIN))
    _git(root, "commit", "-q", "-m", "pin")
    st = AB.freeze_status(root)
    assert st["status"] == "matches_frozen_anchored", st["anchor"]
    assert st["anchor"]["head"] == st["anchor"]["pin_commit"] == _git(root, "rev-parse", "HEAD")
    assert AB.output_route("full", True, st["status"])["label"] == "development"
    # an untracked file in the manifest scope makes the tree dirty
    (root / "configs" / "extra.yaml").write_text("x: 1\n", encoding="utf-8")
    st = AB.freeze_status(root)
    assert st["status"] == "matches_frozen_unanchored" and "not clean" in st["anchor"]["reason"]
    assert "configs/extra.yaml" in st["anchor"]["dirty_paths"]
    # a later commit (even of an unrelated file) means HEAD is no longer the frozen commit
    _git(root, "add", "configs/extra.yaml")
    _git(root, "commit", "-q", "-m", "later")
    st = AB.freeze_status(root)
    assert st["status"] == "matches_frozen_unanchored" and "HEAD is not the commit" in st["anchor"]["reason"]
    # re-pasting a fingerprint without committing it is visible as a dirty pin
    pin = root / AB.FREEZE_PIN
    pin.write_text(pin.read_text(encoding="utf-8").replace('"note": "unit test"', '"note": "edited"'), encoding="utf-8")
    assert "not clean" in AB.freeze_status(root)["anchor"]["reason"]


def test_import_time_sources_are_compared_with_the_start_manifest():
    man = {"files": [{"path": k, "sha256": v} for k, v in AB.IMPORT_TIME_SOURCE_SHA256.items()]}
    assert AB.import_time_check(man)["status"] == "consistent"
    k0 = sorted(AB.IMPORT_TIME_SOURCE_SHA256)[0]
    man["files"][0] = {"path": man["files"][0]["path"], "sha256": "0" * 64}
    chk = AB.import_time_check(man)
    assert chk["status"] == "changed_between_import_and_manifest" and len(chk["differs"]) == 1
    assert "experiments/E1_cost_reliability/run_endpoint_ablation.py" in AB.IMPORT_TIME_SOURCE_SHA256
    assert k0 in AB.IMPORT_TIME_SOURCE_SHA256


def test_dry_run_counters_measure_solves_draws_and_writes(tmp_path):
    """The instrument behind the dry run's zero counts is itself checked (FIX3_BC, C-4): every wrapped entry point
    increments its counter, and everything is restored afterwards."""
    import numpy as np
    import scipy.optimize as so
    from engine_test_helpers import two_ingredient_problem
    from ration_reliability.uncertainty import RandomStreams
    orig_open, orig_gen = open, RandomStreams.generator
    with AB.DryRunCounters() as c:
        assert c.counts == {"solves": 0, "draws": 0, "files_written": 0, "dirs_created": 0}
        with open(tmp_path / "w.txt", "w") as fh:                               # a write
            fh.write("x")
        with open(tmp_path / "w.txt") as fh:                                    # a read is not counted
            fh.read()
        (tmp_path / "p.txt").write_text("y", encoding="utf-8")                  # pathlib write
        os.makedirs(tmp_path / "d")
        RandomStreams(1).generator("opt", 0)                                     # a draw generator
        so.linprog(np.array([1.0]), bounds=[(0, 1)], method="highs")           # a direct scipy solve
        AB.get_method("M0_nominal")(two_ingredient_problem(), params={"coefficient_mode": "nominal_point"})
    rec = c.record()
    assert rec["files_written"] == 2 and rec["dirs_created"] >= 1 and rec["draws"] == 1
    assert rec["solves"] >= 2 and rec["calls"]["method:M0_nominal"] == 1 and rec["method"].startswith("measured")
    assert open is orig_open and RandomStreams.generator is orig_gen and AB.get_method.__name__ == "get_method"


def test_full_needs_a_user_authorisation_file_listing_this_host(tmp_path):
    import socket
    with pytest.raises(SystemExit, match="no authorisation file"):
        AB.check_authorisation(tmp_path / "none.json")
    f = tmp_path / "auth.json"
    f.write_text(json.dumps({"authorised_hosts": ["some-other-host"], "authorised_by": "user", "authorised_on":
                             "2026-09-25", "scope": "run_endpoint_ablation --full"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not listed"):
        AB.check_authorisation(f)
    f.write_text(json.dumps({"authorised_hosts": [socket.gethostname()], "authorised_by": "user"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="lacks"):
        AB.check_authorisation(f)
    f.write_text(json.dumps({"authorised_hosts": [socket.gethostname()], "authorised_by": "user",
                             "authorised_on": "2026-09-25", "scope": "run_endpoint_ablation --full"}), encoding="utf-8")
    ok = AB.check_authorisation(f)
    assert ok["host"] == socket.gethostname() and len(ok["sha256"]) == 64
    # the CLI stops before reading anything when the file is missing (no preflight, nothing written)
    r = subprocess.run([PY, str(DRIVER), "--full", "--authorised-compute", "--authorisation-file",
                        str(tmp_path / "missing.json")], capture_output=True, text=True, env=ENV, timeout=120)
    assert r.returncode == 2 and "no authorisation file" in r.stderr and "BLOCKED" not in r.stderr


def test_output_route_only_a_clean_frozen_full_run_is_development():
    ok = AB.output_route("full", True, "matches_frozen_anchored")
    assert ok["label"] == "development" and ok["public_dir"] == "results/pilot" and ok["run_id_prefix"] == ""
    assert ok["restricted_dir"] == "data/restricted_local/pilot"
    for args, why in [(("full", False, "matches_frozen_anchored"), "changed between import and the end"),
                      (("full", True, "matches_frozen_unanchored"), "matches_frozen_unanchored"),
                      (("full", True, "matches_frozen"), "matches_frozen"),                  # old status: refused
                      (("full", True, "differs_from_frozen"), "differs_from_frozen"),
                      (("full", True, "no_frozen_fingerprint"), "no_frozen_fingerprint"),
                      (("tiny", True, "matches_frozen_anchored"), "toy size")]:
        r = AB.output_route(*args)
        assert r["label"] == "debug" and r["public_dir"] == r["restricted_dir"] == str(AB.DEBUG_ROOT), args
        assert r["run_id_prefix"] == "debug-" and why in r["why"], (args, r["why"])
    assert AB.reproduction_vs_pilot([], "tiny")["status"] == "not_applicable"


# =================================================================================================
# 3 comparable sets (synthetic rows)
# =================================================================================================

def test_comparable_sets_compare_costs_only_inside_the_set_and_report_coverage():
    ign = AB.MAIN_EVENT_T51_IGNORED

    def row(cell, label, world, has, cost=None, rate=None, cp=None, ok=True, kind="method", rate_ign=None,
            share=0.0):
        # FIX3_BC (C-1): the membership rate is the domain-conditioned main event; the R3C reading sits next to it
        rate_ign = rate if rate_ign is None else rate_ign
        return {"cell_id": cell, "label": label, "ration_kind": kind, "evaluation_world": world, "has_ration": has,
                "structural_ok": ok if has else None, "cost_usd_per_head_d": cost, "q_hash": f"h-{cell}-{label}",
                "main_reference_rate_upper": rate, "main_reference_cp_upper": cp,
                f"{ign}_rate_upper": rate_ign, f"{ign}_cp_upper": cp,
                "t51_primary_not_assessable_share": share if has else None,
                "target_alpha": AB._alpha_of(label)}
    rows = [row("A", "A:M0", "SD-H0", True, 6.0, 0.95, 0.96),
            row("A", "A:M1[x]@alpha=0.05", "SD-H0", True, 6.6, 0.04, 0.045),
            row("A", "A:M2[N=128]@alpha=0.05", None, False),
            row("B", "B:M3a@alpha=0.05", "SD-H0", True, 6.5, 0.05, 0.056),
            row("B", "B:M3b@alpha=0.05", "SD-H0", True, 6.4, 0.049, 0.05, ok=False),
            row("B", "B:DIAG-frontier:M2[N=128,alpha_train=m*/N=0/128]", "SD-H0", True, 6.3, 0.0, 0.0003,
                kind="diagnostic_frontier"),
            row("A", "A:M1[x]@alpha=0.1", "SD-H0", True, 6.5, 0.09, 0.1),
            # out of the primary Table 5-1 domain in 97 % of states: the R3C reading met alpha, the main event does not
            row("C", "C:M2[N=128]@alpha=0.05", "SD-H0", True, 6.2, 0.97, 0.98, rate_ign=0.03, share=0.97)]
    out = AB.comparable_sets(rows)
    s = next(o for o in out if o["evaluation_world"] == "SD-H0" and o["target_alpha"] == 0.05)
    # entries at alpha 0.05: A:M0 (no alpha), A:M1, A:M2 (no ration), B:M3a, B:M3b, C:M2 -> 6; frontier excluded
    assert s["n_entries_attempted"] == 6 and s["n_members"] == 2 and s["coverage"] == pytest.approx(2 / 6)
    assert [m["label"] for m in s["members"]] == ["B:M3a@alpha=0.05", "A:M1[x]@alpha=0.05"]     # cost order
    # the out-of-domain ration is not comparable under the main (domain-conditioned) event; it would have been under
    # the R3C reading -- both memberships and the out-of-domain share are reported, the main one decides
    assert [m["label"] for m in s["members_t51_verdicts_used_out_of_domain"]][0] == "C:M2[N=128]@alpha=0.05"
    assert s["n_members_t51_verdicts_used_out_of_domain"] == 3
    assert s["headline"]["t51_primary_not_assessable_share_max"] == pytest.approx(0.97)
    assert "primary domain" in s["headline"]["membership_event"]
    assert s["members"][0]["cost_minus_cheapest_member_usd"] == 0.0
    assert s["members"][1]["cost_minus_cheapest_member_usd"] == pytest.approx(0.1)
    assert s["members"][0]["cp_upper_le_alpha"] is False and s["members"][1]["cp_upper_le_alpha"] is True
    s2 = next(o for o in out if o["evaluation_world"] == "SD-S2" and o["target_alpha"] == 0.05)
    assert s2["n_members"] == 0 and "not compared" in s2["empty_set_meaning"]
    by = {r["label"]: r for r in rows}
    assert by["A:M1[x]@alpha=0.05"]["meets_main_reference_alpha_point"] is True
    assert by["B:M3b@alpha=0.05"]["meets_main_reference_alpha_point"] is True       # rate ok, but structural fails:
    assert "alpha=0.05:False" in by["B:M3b@alpha=0.05"]["in_comparable_set"]     # not a member
    assert by["A:M2[N=128]@alpha=0.05"]["in_comparable_set"] is None
    assert by["A:M0"]["meets_main_reference_alpha_point"] is None
    assert by["C:M2[N=128]@alpha=0.05"]["in_comparable_set"] == "alpha=0.05:False"
    assert by["C:M2[N=128]@alpha=0.05"]["in_comparable_set_t51_verdicts_used_out_of_domain"] == "alpha=0.05:True"


def test_main9_arm_is_declared_from_the_adjudicated_table():
    """FIX3_BC (C-2): the MAIN9 arm trains on the 9 main-reference rows; only the two labelled research-assumption rows
    are planned; the PART6P5 description no longer claims 'sourced bounds'."""
    from ration_reliability.evaluation.reference import load_reference_constraints
    t = load_reference_constraints(REPO / "configs" / "dev_case_v1" / "reference_constraints.csv")
    assert AB.main9_planned_rows(t) == ("PN-CP-HI", "PN-EE-HI")
    arms = AB.ABLATION_CONFIG["objective_arms"]
    assert set(arms) == {"FULL11", "PART6P5", "MAIN9"}
    assert "sourced bounds" not in arms["PART6P5"]["problem"] and "declared six" in arms["PART6P5"]["training_event"]
    assert arms["MAIN9"]["reference_event_equal_to_training"] == "main9_training_event"
    cells = {c["cell_id"]: c for c in AB.ABLATION_CONFIG["cells"]}
    assert {"SDH0_MAIN9", "SDS2_MAIN9"} <= set(cells) and len(cells) == 6
    ev = {e.event_id: e for e in AB.reference_events(t)}
    assert set(ev["main9_training_event"].members) == set(t.main_reference_ids)
    assert ev["main9_training_event"].energy_verdict == "linear" and ev["main9_training_event"].table51_domain == "ignored"
    assert ev["main_reference"].table51_domain == "primary"
    for cid in t.ids:
        col = t.row(cid).fields["optimization_main9_arm"]
        if cid in t.main_reference_ids:
            assert col.startswith("probabilistic_nutrition")
        elif cid in ("PN-CP-HI", "PN-EE-HI"):
            assert "SH-PLAN-" in col


# =================================================================================================
# 4 holder side
# =================================================================================================
needs_restricted = pytest.mark.skipif(not (REPO / DC.DEV_CASE_V1.constants_file).is_file(),
                                      reason="restricted dev_case inputs not present (holder-side check only)")


def _ready():
    rep = PF.preflight_dev_case(REPO, driver="run_dev_case_v1", extra=AB.EXTRA_REQUIREMENTS)
    if rep["exit_code"] != PF.EXIT_READY:
        pytest.skip(f"dev_case inputs not READY (preflight {rep['status']}); holder-side check skipped")


def _listing(p: Path) -> set[str]:
    return {x.name for x in p.iterdir()} if p.is_dir() else set()


@needs_restricted
def test_dry_run_builds_the_four_cells_without_draws_or_solves_and_writes_nothing():
    _ready()
    debug, pilot = REPO / AB.DEBUG_ROOT, REPO / "results" / "pilot"
    before = (_listing(debug), _listing(pilot), _listing(REPO / "data" / "restricted_local" / "pilot"))
    r = subprocess.run([PY, str(DRIVER), "--dry-run"], capture_output=True, text=True, env=ENV, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert (out["dry_run"], out["files_written"], out["draws"], out["solves"]) == (True, 0, 0, 0)
    # FIX3_BC (C-4): the zeros are measured by wrapped entry points, not literals
    assert out["measured_counts"]["method"].startswith("measured") and out["dirs_created"] == 0
    assert out["checks"]["freeze"]["status"] != "matches_frozen_anchored" or out["checks"]["freeze"]["anchor"]["anchored"]
    assert out["checks"]["import_time_sources"]["status"] == "consistent"
    assert out["checks"]["validators"]["MAIN9"]["ok"] is True
    cells = {c["cell_id"]: c for c in out["plan_full"]["cells"]}
    assert set(cells) == {"SDH0_FULL11", "SDS2_FULL11", "SDH0_PART6P5", "SDS2_PART6P5", "SDH0_MAIN9", "SDS2_MAIN9"}
    assert cells["SDH0_MAIN9"]["problem_id"].endswith("|MAIN9")
    assert cells["SDH0_FULL11"]["problem_id"] == "dev_case_v1" and cells["SDS2_PART6P5"]["problem_id"].endswith("|S2")
    assert cells["SDH0_FULL11"]["world_fingerprint"] == cells["SDH0_PART6P5"]["world_fingerprint"]
    assert cells["SDH0_FULL11"]["world_fingerprint"] != cells["SDS2_FULL11"]["world_fingerprint"]
    ce = out["plan_full"]["compute_estimate"]
    assert ce["worst_case_solver_time_s"] > 0 and "authorised" in ce["host"]   # review round 4 reworded the host note
    chk = out["checks"]
    assert chk["sd_registry"]["ok"] and chk["seed"]["seed"] == 1103
    assert len(chk["reference_table"]["main_reference"]) == 9
    assert (_listing(debug), _listing(pilot), _listing(REPO / "data" / "restricted_local" / "pilot")) == before


@needs_restricted
def test_tiny_runs_the_pipeline_writes_only_debug_output_and_scores_every_ration_on_both_worlds():
    _ready()
    debug = REPO / AB.DEBUG_ROOT
    before_debug, before_pilot = _listing(debug), _listing(REPO / "results" / "pilot")
    r = subprocess.run([PY, str(DRIVER), "--tiny"], capture_output=True, text=True, env=ENV, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    new = sorted(_listing(debug) - before_debug)
    assert len(new) == 1 and new[0].startswith("debug-smoke-")
    d = debug / new[0]
    try:
        assert _listing(REPO / "results" / "pilot") == before_pilot                     # nothing under results/
        info = json.loads(r.stdout)
        assert info["output_label"] == "debug" and info["mode"] == "tiny"
        rec = json.loads((d / "run_record.json").read_text(encoding="utf-8"))
        assert rec["run_type"] == "smoke" and rec["extra"]["output_label"] == "debug"
        assert "debug / smoke" in (d / "NOT_FOR_MANUSCRIPT.md").read_text(encoding="utf-8")
        rows = list(csv.DictReader((d / "endpoint_ablation.csv").open(encoding="utf-8")))
        assert {r_["cell_id"] for r_ in rows} == {"SDH0_FULL11", "SDS2_FULL11", "SDH0_PART6P5", "SDS2_PART6P5",
                                                  "SDH0_MAIN9", "SDS2_MAIN9"}
        with_q = [r_ for r_ in rows if r_["has_ration"] == "True"]
        without = [r_ for r_ in rows if r_["has_ration"] == "False"]
        assert with_q and all(r_["output_label"] == "debug" for r_ in rows)
        assert all(r_["evaluation_world"] == "" and r_["cost_usd_per_head_d"] == "" and r_["solver_time_limit_s"]
                   for r_ in without)                                            # no ration: no cost, never 0
        by_ration: dict = {}
        for r_ in with_q:
            by_ration.setdefault((r_["cell_id"], r_["label"]), []).append(r_)
        for key, rs in by_ration.items():
            assert sorted(x["evaluation_world"] for x in rs) == ["SD-H0", "SD-S2"], key
            assert sum(x["own_world"] == "True" for x in rs) == 1 and len({x["q_hash"] for x in rs}) == 1
            assert all(int(x["n_test"]) == 500 for x in rs)
            own = next(x for x in rs if x["own_world"] == "True")
            same = {"FULL11": "h0_eleven_rate_upper", "PART6P5": "s2_six_rate_upper",
                    "MAIN9": "main9_training_event_rate_upper"}[own["objective_arm"]]
            # the cell's own training event (public evaluator on the cell problem) = the matching reference event
            assert float(own["training_event_rate_upper_own_world"]) == pytest.approx(float(own[same]), abs=1e-12), key
        # M0 uses no draws: the same objective gives bit-identical q in both SD cells; PART6P5's M0 equals FULL11's M0
        # to the S2 consistency tolerance (1e-8 kg), not necessarily bit for bit
        for arm in ("FULL11", "PART6P5", "MAIN9"):
            assert len({r_["q_hash"] for r_ in with_q if r_["label"].endswith(":M0") and r_["objective_arm"] == arm}) == 1
        qrows = {(x["cell_id"], x["label"]): x for x in csv.DictReader((d / "rations_q.csv").open(encoding="utf-8"))}
        qa, qb = qrows[("SDH0_FULL11", "SDH0_FULL11:M0")], qrows[("SDH0_PART6P5", "SDH0_PART6P5:M0")]
        assert max(abs(float(qa[k]) - float(qb[k])) for k in qa if k.startswith("q_") and k != "q_hash") < 1e-8
        sol = json.loads((d / "solve_status.json").read_text(encoding="utf-8"))
        for c in sol["cells"]:
            for mv in c["diagnostics"]["min_violation"]:
                assert mv["N"] <= 16 and "training_scenarios_only" in mv["label"] and mv["time_limit_s"] > 0
        assert sol["s2_consistency_with_full11"]["ok"] and sol["main9_consistency_with_full11"]["ok"]
        # FIX3_BC (C-1): the main event is domain-conditioned; both memberships are reported
        for r_ in with_q:
            assert r_["t51_primary_not_assessable_share"] != "" and r_[f"{AB.MAIN_EVENT_T51_IGNORED}_rate_upper"] != ""
            assert float(r_["main_reference_rate_upper"]) >= float(r_["main_reference_rate_lower"])
        comp = json.loads((d / "comparable_sets.json").read_text(encoding="utf-8"))
        assert {(c["evaluation_world"], c["target_alpha"]) for c in comp} == {
            (w, a) for w in ("SD-H0", "SD-S2") for a in (0.05, 0.10, 0.01)}
        # official-run plan batch 2: the descriptive reports are written next to the existing tables
        pmr = list(csv.DictReader((d / "per_method_report.csv").open(encoding="utf-8")))
        assert pmr and list(pmr[0]) == AB.PER_METHOD_REPORT_COLUMNS
        assert {(x["evaluation_world"], float(x["target_alpha"])) for x in pmr} == {
            (w, a) for w in ("SD-H0", "SD-S2") for a in (0.05, 0.10, 0.01)}
        assert not any(":DIAG-" in x["label"] for x in pmr)                   # the frontier is never an entry
        assert {x["cost_label"] for x in pmr} <= {"", "comparable_set_member", "not_a_comparable_cost"}
        assert all(x["grid_n_candidates"] != "" for x in pmr)
        curve = list(csv.DictReader((d / "matched_cost_curve.csv").open(encoding="utf-8")))
        assert curve and list(curve[0]) == AB.MATCHED_COST_COLUMNS
        by_fam: dict = {}
        for x in curve:
            if x["status"] == "ok":
                by_fam.setdefault((x["evaluation_world"], x["cell_id"], x["method_family"]), []).append(
                    (float(x["relative_cost_grid_point"]), float(x["min_rate_upper"])))
        for pts in by_fam.values():
            ys = [y for _, y in sorted(pts)]
            assert all(b <= a for a, b in zip(ys, ys[1:]))                    # envelope non-increasing in cost
        with (d / "candidate_rations.csv").open(encoding="utf-8") as fh:          # restricted (q): header only
            head = next(csv.reader(fh))
        assert head[:len(AB.CANDIDATE_COLUMNS)] == AB.CANDIDATE_COLUMNS and any(h.startswith("q_") for h in head)
        res = list(csv.DictReader((d / "reference_residuals.csv").open(encoding="utf-8")))
        assert {x["verdict"] for x in res if x["constraint_id"] == "PN-NEL-FIXEDDMI"} == {
            "public_evaluator_linear", "reference_chain_ch3"}
    finally:
        assert d.resolve().parent == debug.resolve() and d.name.startswith("debug-smoke-")
        shutil.rmtree(d)                                                          # the test's own debug output
