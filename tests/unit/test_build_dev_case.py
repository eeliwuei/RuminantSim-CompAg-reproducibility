"""dev_case build logic moved out of the restricted directory (review round 3 F6, instruction F-2; R3F).

What is checked:

1. restricted constants: read from the holder-side wrapper by ``ast.literal_eval`` (the file is never executed); every
   missing or malformed entry is listed; there are no defaults;
2. ``build_input_paths`` keeps the legacy order of ``build_report.json["inputs"]``;
3. ``compare_build_outputs`` reports byte- and field-level differences by path only (never values);
   ``write_build`` refuses an output directory outside ``data/restricted_local/``;
4. ``run_record.build_identity``: restricted files by path + sha256 only, the identity covers scripts and inputs
   (not outputs), and it enters the run identity only when given (older identities unchanged); an incomplete build
   identity is refused for pilot runs;
5. the CLI ``scripts/build_dev_case.py`` stops with exit 2 and a list when inputs are missing (nothing written);
6. holder side only (skipped without ``data/restricted_local/``): the in-memory build is byte-identical to the five
   outputs in the case directory, and the public builder code contains none of the restricted constants.

Synthetic values only; no restricted value is printed or asserted literally.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ration_reliability.build import dev_case as DC
from ration_reliability.build import preflight as PF
from ration_reliability.io import run_record as RR

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def _synthetic_constants() -> dict:
    # FIX3_DEF: only the restricted entries (table values, locator notes); the book's text constants are public
    # (DC.PUBLIC_TEXT_CONSTANTS) and optional in the restricted file
    return {"schema": DC.CONSTANTS_SCHEMA, "case_id": DC.CASE_ID,
            "starch_digestibility_table_3_1": {"feed_a": [0.5, "Feed A row"], "feed_b": [0.25, "Feed B row"]},
            "starch_digestibility_default": 0.5,
            "dicalcium_phosphate_ac_locators": {"AC_Ca": "synthetic locator a", "AC_P": "synthetic locator b"}}


# =================================================================================================
# 1 restricted constants
# =================================================================================================

def test_constants_are_parsed_from_the_wrapper_without_executing_it(tmp_path):
    marker = tmp_path / "executed.txt"
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(f"open({str(marker)!r}, 'w').write('x')\n"
                       f"RESTRICTED_CONSTANTS = {_synthetic_constants()!r}\n"
                       "raise SystemExit('must never run')\n", encoding="utf-8")
    c = DC.load_restricted_constants(wrapper)
    assert c["starch_digestibility_default"] == 0.5 and c["starch_digestibility_table_3_1"]["feed_b"][0] == 0.25
    assert not marker.exists()                                   # parsed, not executed


def test_constants_validation_lists_every_problem_and_has_no_defaults(tmp_path):
    bad = _synthetic_constants()
    bad["fa_digestibility_basal"] = 0.5          # a public text constant carried with another value: refused
    bad["starch_digestibility_default"] = 1.5
    bad["p_absorption_rule"] = {"inorganic": 0.5}
    bad["dicalcium_phosphate_ac_locators"] = {"AC_Ca": ""}
    bad["starch_digestibility_table_3_1"] = {"feed_a": [0.5]}
    bad["schema"] = "other"
    errs = DC.validate_restricted_constants(bad)
    for key in ("fa_digestibility_basal", "starch_digestibility_default", "p_absorption_rule",
                "dicalcium_phosphate_ac_locators", "starch_digestibility_table_3_1.feed_a", "schema"):
        assert any(key in e for e in errs), (key, errs)
    assert not DC.validate_restricted_constants(_synthetic_constants())
    # a restricted file that still carries the public text constants with their public values is accepted
    assert not DC.validate_restricted_constants({**_synthetic_constants(), **DC.PUBLIC_TEXT_CONSTANTS})
    missing_default = _synthetic_constants()
    del missing_default["starch_digestibility_default"]           # a restricted table value has no default
    assert any("starch_digestibility_default" in e for e in DC.validate_restricted_constants(missing_default))
    # a non-literal assignment, a missing variable and a missing file are refused with a listed reason
    (tmp_path / "nonliteral.py").write_text("RESTRICTED_CONSTANTS = dict(a=1)\n", encoding="utf-8")
    (tmp_path / "none.py").write_text("X = 1\n", encoding="utf-8")
    for name in ("nonliteral.py", "none.py", "absent.py"):
        with pytest.raises(DC.RestrictedConstantsError):
            DC.load_restricted_constants(tmp_path / name)
    with pytest.raises(DC.RestrictedConstantsError):
        DC.build_dev_case(tmp_path, bad)                           # the builder validates before reading anything


# =================================================================================================
# 2 inputs, comparison, writing
# =================================================================================================

def test_build_input_paths_keep_the_legacy_report_order(tmp_path):
    cfg = tmp_path / "configs" / "dev_case_v1"
    cfg.mkdir(parents=True)
    for n in ("prices.yaml", "animal.yaml", "zeta.yaml", "README.md"):
        (cfg / n).write_text("x: 1\n", encoding="utf-8")
    got = DC.build_input_paths(tmp_path)
    L = DC.DEV_CASE_V1
    assert got == [L.core_csv, L.blayer_csv, L.feed_library_csv, L.restricted_values,
                   "configs/dev_case_v1/animal.yaml", "configs/dev_case_v1/prices.yaml", "configs/dev_case_v1/zeta.yaml",
                   L.animal_profile]


def test_compare_build_outputs_reports_paths_not_values(tmp_path):
    exp = {"a.json": json.dumps({"k": {"x": 1.25, "y": 2}}).encode(), "b.csv": b"h1,h2\r\n1,2\r\n",
           "c.yaml": b"k: 1\n"}
    (tmp_path / "a.json").write_bytes(json.dumps({"k": {"x": 9.75, "y": 2}}).encode())
    (tmp_path / "b.csv").write_bytes(exp["b.csv"])
    cmp = DC.compare_build_outputs(exp, tmp_path)
    assert cmp["files"]["b.csv"]["byte_identical"] and cmp["files"]["b.csv"]["field_identical"]
    assert cmp["files"]["a.json"]["field_differences"] == ["/k/x"]
    assert not cmp["files"]["c.yaml"]["exists"] and not cmp["all_byte_identical"]
    assert "9.75" not in json.dumps(cmp) and "1.25" not in json.dumps(cmp)


def test_write_build_refuses_a_directory_outside_restricted_local(tmp_path):
    b = DC.DevCaseBuild(outputs={n: b"x" for n in DC.OUTPUT_FILES}, report={"failed": [], "summary": {}}, summary={})
    with pytest.raises(ValueError, match="restricted_local"):
        DC.write_build(tmp_path, b, tmp_path / "results" / "x", entry_point="scripts/build_dev_case.py",
                       constants_path="c.py")
    ident = DC.write_build(tmp_path, b, tmp_path / "data" / "restricted_local" / "scratch",
                           entry_point="scripts/build_dev_case.py", constants_path="c.py")
    side = json.loads((tmp_path / "data" / "restricted_local" / "scratch" / DC.SIDECAR).read_text(encoding="utf-8"))
    assert side["build_identity_sha256"] == ident["build_identity_sha256"]
    outs = {e["path"]: e for e in side["build_outputs"]}
    assert all(e["restricted"] and e["sha256"] == hashlib.sha256(b"x").hexdigest() for e in outs.values())


# =================================================================================================
# 3 build identity in run records
# =================================================================================================

def _ident_tree(root: Path) -> dict:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "data" / "restricted_local").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "build.py").write_text("print(1)\n", encoding="utf-8")
    (root / "data" / "restricted_local" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "data" / "restricted_local" / "out.yaml").write_text("k: 1\n", encoding="utf-8")
    return RR.build_identity(root, build_scripts=["scripts/build.py"],
                             build_inputs=["data/restricted_local/table.csv"],
                             build_outputs=["data/restricted_local/out.yaml"])


def test_build_identity_records_restricted_files_by_path_and_hash_only(tmp_path):
    root = tmp_path / "p"
    bi = _ident_tree(root)
    assert bi["complete"] and not bi["missing"]
    e = bi["build_inputs"][0]
    assert e == {"path": "data/restricted_local/table.csv", "restricted": True,
                 "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest()}          # no content, no size
    assert bi["build_scripts"][0]["restricted"] is False
    # outputs are recorded but do not define the build identity
    (root / "data" / "restricted_local" / "out.yaml").write_text("k: 2\n", encoding="utf-8")
    bi2 = RR.build_identity(root, build_scripts=["scripts/build.py"], build_inputs=["data/restricted_local/table.csv"],
                            build_outputs=["data/restricted_local/out.yaml"])
    assert bi2["build_identity_sha256"] == bi["build_identity_sha256"]
    assert bi2["build_outputs_sha256"] != bi["build_outputs_sha256"]
    # a changed build script or input changes it
    (root / "scripts" / "build.py").write_text("print(2)\n", encoding="utf-8")
    bi3 = RR.build_identity(root, build_scripts=["scripts/build.py"], build_inputs=["data/restricted_local/table.csv"])
    assert bi3["build_identity_sha256"] != bi["build_identity_sha256"]
    # a missing file makes it incomplete; a path outside the repository is recorded by name only
    outside = tmp_path / "elsewhere" / "consts.py"
    outside.parent.mkdir()
    outside.write_text("X = 1\n", encoding="utf-8")
    bi4 = RR.build_identity(root, build_scripts=[outside, "scripts/gone.py"], build_inputs=[])
    assert not bi4["complete"] and bi4["missing"] == ["scripts/gone.py"]
    ext = [x for x in bi4["build_scripts"] if x.get("outside_repository")][0]
    assert ext["path"] == "consts.py" and str(tmp_path) not in json.dumps(bi4)


def test_build_identity_enters_the_run_identity_only_when_given(tmp_path):
    root = tmp_path / "p"
    bi = _ident_tree(root)
    kw = dict(code_manifest_sha256="c", environment_fingerprint_sha256="e", config_hash=None, data_manifest_hash=None,
              protocol_hash=None, run_type="smoke", command="x", rng_streams={}, solver_version="s", tolerances={},
              is_synthetic=True)
    plain, comp = RR.run_identity(**kw)
    assert "build_identity_sha256" not in comp                   # identities computed before R3F are reproduced
    with_b, comp_b = RR.run_identity(**kw, build_identity_sha256=bi["build_identity_sha256"])
    assert with_b != plain and comp_b["build_identity_sha256"] == bi["build_identity_sha256"]
    other, _ = RR.run_identity(**kw, build_identity_sha256="0" * 64)
    assert other != with_b
    common = dict(command="identity check", repo_root=root, started_at="2026-09-25T00:00:00+00:00",
                  completed_at="2026-09-25T00:00:00+00:00", exit_status=0, rng_streams={}, solver_version="s",
                  is_synthetic=True)
    rec = RR.build_run_record(run_type="smoke", tolerances={}, build_identity=bi, **common)
    assert rec["build_identity"]["build_identity_sha256"] == bi["build_identity_sha256"]
    assert rec["run_identity_components"]["build_identity_sha256"] == bi["build_identity_sha256"]
    with pytest.raises(ValueError, match="build_identity"):
        RR.build_run_record(run_type="smoke", tolerances={}, build_identity={"x": 1}, **common)
    incomplete = RR.build_identity(root, build_scripts=["scripts/gone.py"], build_inputs=[])
    with pytest.raises(ValueError, match="build identity incomplete"):
        RR.build_run_record(run_type="pilot", tolerances={"mip_rel_gap": 0.0}, build_identity=incomplete, **common)


# =================================================================================================
# 3b FIX3_DEF (review round 3 red team F-2): builder scope and the build-time sidecar in run records
# =================================================================================================

def _case_tree(root: Path) -> Path:
    """Synthetic case: builder code, two dependency modules, every build input and the five outputs."""
    L = DC.DEV_CASE_V1
    files = {rel: f"# synthetic builder file {rel}\n" for rel in DC.builder_code_paths()}
    files.update({"src/ration_reliability/nutrition/syn_dep.py": "X = 1\n",
                  "src/ration_reliability/optimization/syn_method.py": "Y = 2\n",
                  L.animal_profile: "fields: {}\n", L.core_csv: "a,b\n1,2\n", L.blayer_csv: "a,b\n3,4\n",
                  L.feed_library_csv: "UID\nX\n", L.restricted_values: "values: {}\n",
                  L.constants_file: "RESTRICTED_CONSTANTS = {}\n"})
    files.update({f"{L.cfg_dir}/{n}": f"synthetic: {n}\n" for n in L.config_names})
    files.update({f"{L.case_dir}/{n}": "legacy\n" for n in DC.OUTPUT_FILES})     # outputs of an unrecorded build
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _write_synthetic_build(root: Path) -> dict:
    b = DC.DevCaseBuild(outputs={n: f"synthetic {n}\n".encode() for n in DC.OUTPUT_FILES},
                        report={"failed": [], "summary": {"n_checks": 1, "n_failed": 0}}, summary={})
    return DC.write_build(root, b, root / DC.DEV_CASE_V1.case_dir, entry_point=root / "scripts" / "build_dev_case.py",
                          constants_path=root / DC.DEV_CASE_V1.constants_file)


def test_builder_scope_is_the_whole_package_and_a_dependency_change_changes_the_identity(tmp_path):
    root = _case_tree(tmp_path / "p")
    scope = DC.builder_scope_paths(root)
    assert scope[:3] == DC.builder_code_paths()
    assert {"src/ration_reliability/nutrition/syn_dep.py", "src/ration_reliability/optimization/syn_method.py"} <= \
        set(DC.builder_dependency_paths(root))
    a = DC.compute_dev_case_build_identity(root)
    assert a["complete"] and a["builder_scope"]["n_files"] == len(scope)
    (root / "src/ration_reliability/nutrition/syn_dep.py").write_text("X = 2\n", encoding="utf-8")
    b = DC.compute_dev_case_build_identity(root)
    assert b["build_identity_sha256"] != a["build_identity_sha256"]          # a dependency is part of the build
    (root / "src/ration_reliability/optimization/syn_new.py").write_text("Z = 3\n", encoding="utf-8")
    c = DC.compute_dev_case_build_identity(root)
    assert c["build_identity_sha256"] != b["build_identity_sha256"]          # a new module too
    (root / "src/ration_reliability/optimization/__pycache__").mkdir()
    (root / "src/ration_reliability/optimization/__pycache__/x.py").write_text("junk\n", encoding="utf-8")
    assert DC.compute_dev_case_build_identity(root)["build_identity_sha256"] == c["build_identity_sha256"]


def test_run_record_identity_is_the_build_time_sidecar_and_refused_when_absent_or_stale(tmp_path):
    root = _case_tree(tmp_path / "p")
    common = dict(command="identity check", repo_root=root, started_at="2026-09-25T00:00:00+00:00",
                  completed_at="2026-09-25T00:00:00+00:00", exit_status=0, rng_streams={}, solver_version="s",
                  is_synthetic=True)
    pilot = dict(run_type="pilot", tolerances={"mip_rel_gap": 0.0})
    # 1 no sidecar (outputs of an unrecorded build): computed now, labelled, refused for pilot, kept for smoke
    none = DC.dev_case_build_identity(root)
    assert none["source"] == "computed_at_run_time" and none["sidecar"]["status"] == "absent"
    assert none["usable_for_pilot_official"] is False and "not the build" in none["provenance_note"]
    with pytest.raises(ValueError, match="not usable"):
        RR.build_run_record(build_identity=none, **pilot, **common)
    assert RR.build_run_record(run_type="smoke", tolerances={}, build_identity=none, **common)["build_identity"][
        "source"] == "computed_at_run_time"
    # 2 a build writes the sidecar: the run record carries the build-time identity (not a run-time recomputation)
    side = _write_synthetic_build(root)
    rec = DC.dev_case_build_identity(root)
    assert rec["source"] == "build_time_sidecar" and rec["sidecar"]["status"] == "consistent"
    assert rec["sidecar"]["hash_rederived_from_entries"] and rec["usable_for_pilot_official"] is True
    assert rec["build_identity_sha256"] == side["build_identity_sha256"]
    assert rec["entry_point"] == "scripts/build_dev_case.py" and rec["constants_file"] == DC.DEV_CASE_V1.constants_file
    # the build-identity guard passes; the pilot record then stops at the environment lock of this synthetic tree
    with pytest.raises(ValueError, match="environment lock"):
        RR.build_run_record(build_identity=rec, **pilot, **common)
    r = RR.build_run_record(run_type="smoke", tolerances={}, build_identity=rec, **common)
    assert r["run_identity_components"]["build_identity_sha256"] == side["build_identity_sha256"]
    assert r["build_identity"]["source"] == "build_time_sidecar"
    # 3 a builder dependency changed after the build: stale, the drift names it, pilot refused
    (root / "src/ration_reliability/nutrition/syn_dep.py").write_text("X = 3\n", encoding="utf-8")
    st = DC.dev_case_build_identity(root)
    assert st["sidecar"]["status"] == "stale" and st["usable_for_pilot_official"] is False
    assert {"path": "src/ration_reliability/nutrition/syn_dep.py", "kind": "builder_code", "change": "changed"} in \
        st["sidecar"]["drift"]
    assert st["build_identity_sha256"] == side["build_identity_sha256"]      # what the build was, flagged stale
    with pytest.raises(ValueError, match="syn_dep.py"):
        RR.build_run_record(build_identity=st, **pilot, **common)
    # 4 a rebuild clears it; an output replaced afterwards, or a tampered sidecar, is stale again
    _write_synthetic_build(root)
    assert DC.dev_case_build_identity(root)["usable_for_pilot_official"] is True
    (root / DC.DEV_CASE_V1.case_dir / "energy_linearisation.json").write_bytes(b"other\n")
    d = DC.dev_case_build_identity(root)["sidecar"]["drift"]
    assert any(x["kind"] == "build_output" and x["path"].endswith("energy_linearisation.json") for x in d)
    _write_synthetic_build(root)
    sp = root / DC.DEV_CASE_V1.case_dir / DC.SIDECAR
    doc = json.loads(sp.read_text(encoding="utf-8"))
    doc["build_identity_sha256"] = "0" * 64
    sp.write_text(json.dumps(doc), encoding="utf-8")
    tam = DC.dev_case_build_identity(root)
    assert tam["sidecar"]["hash_rederived_from_entries"] is False and tam["usable_for_pilot_official"] is False


def test_write_build_replaces_files_atomically_and_leaves_no_temporary_file(tmp_path):
    root = _case_tree(tmp_path / "p")
    _write_synthetic_build(root)
    case = root / DC.DEV_CASE_V1.case_dir
    assert sorted(p.name for p in case.iterdir() if p.name.startswith(".")) == []
    assert (case / "build_report.json").read_bytes() == b"synthetic build_report.json\n"


# =================================================================================================
# 4 CLI without inputs
# =================================================================================================

def test_cli_without_inputs_lists_them_and_writes_nothing(tmp_path):
    root = tmp_path / "pkg"
    (root / "configs").mkdir(parents=True)
    before = sorted(p.as_posix() for p in root.rglob("*"))
    r = subprocess.run([PY, str(REPO / "scripts" / "build_dev_case.py"), "--root", str(root), "--check"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 2, r.stderr
    assert "build inputs missing" in r.stderr and DC.DEV_CASE_V1.core_csv in r.stderr
    assert "Traceback" not in r.stderr
    assert sorted(p.as_posix() for p in root.rglob("*")) == before
    r2 = subprocess.run([PY, str(REPO / "scripts" / "build_dev_case.py"), "--root", str(root), "--check",
                         "--out-dir", "x"], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r2.returncode == 3


# =================================================================================================
# 5 holder side: byte identity with the existing build outputs (restricted inputs required)
# =================================================================================================
RESTRICTED = REPO / "data" / "restricted_local"
needs_restricted = pytest.mark.skipif(not (REPO / DC.DEV_CASE_V1.constants_file).is_file(),
                                      reason="restricted dev_case inputs not present (holder-side check only)")


@needs_restricted
def test_in_memory_build_is_byte_identical_to_the_case_outputs():
    rep = PF.preflight_dev_case(REPO)
    if rep["exit_code"] != PF.EXIT_READY:
        pytest.skip(f"dev_case outputs not consistent with their inputs (preflight {rep['status']}); rebuild first")
    consts = DC.load_restricted_constants(REPO / DC.DEV_CASE_V1.constants_file)
    build = DC.build_dev_case(REPO, consts)
    assert build.ok and build.report["summary"]["n_checks"] == 133 and build.report["summary"]["n_failed"] == 0
    cmp = DC.compare_build_outputs(build.outputs, REPO / DC.DEV_CASE_V1.case_dir)
    assert cmp["all_byte_identical"], {n: r.get("field_differences") for n, r in cmp["files"].items()}
    assert list(build.outputs) == list(DC.OUTPUT_FILES)


@needs_restricted
def test_public_builder_code_carries_none_of_the_restricted_constants():
    # FIX3_DEF (red team F-2): restricted = Table 3-1 values (incl. its default for unlisted feeds) and the
    # feed-library locator notes; the book's text constants are public (DC.PUBLIC_TEXT_CONSTANTS, with pages) and,
    # if the holder-side file still carries them, must equal the public values
    consts = DC.load_restricted_constants(REPO / DC.DEV_CASE_V1.constants_file)
    for k, pub in DC.PUBLIC_TEXT_CONSTANTS.items():
        if k in consts:
            assert consts[k] == pub, k                             # validated on load as well
    numbers: set[str] = {repr(float(consts["starch_digestibility_default"]))}
    numbers |= {repr(float(v[0])) for v in consts["starch_digestibility_table_3_1"].values()}
    public = {repr(float(v)) for v in DC.PUBLIC_TEXT_CONSTANTS.values() if not isinstance(v, dict)}
    public |= {repr(float(v)) for v in DC.PUBLIC_TEXT_CONSTANTS["p_absorption_rule"].values()}
    assert not (numbers & public)                                  # the two sets are disjoint
    locators = set(consts["dicalcium_phosphate_ac_locators"].values())
    import re
    for rel in DC.builder_code_paths() + ["src/ration_reliability/build/preflight.py", "scripts/preflight_dev_case.py"]:
        text = (REPO / rel).read_text(encoding="utf-8")
        tokens = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", text))
        hit = tokens & numbers
        assert not hit, f"{rel}: carries {len(hit)} restricted constant(s)"          # values are not printed
        assert not any(loc in text for loc in locators), f"{rel}: carries a restricted locator note"


@needs_restricted
def test_legacy_script_is_kept_byte_for_byte():
    legacy = REPO / DC.DEV_CASE_V1.case_dir / "legacy_builder_be942251" / "build_dev_case_v1.py"
    if not legacy.is_file():
        pytest.skip("legacy copy not present")
    assert hashlib.sha256(legacy.read_bytes()).hexdigest().startswith("be9422512755e95c")
