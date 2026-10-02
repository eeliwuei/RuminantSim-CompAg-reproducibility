"""Official-run plan batch 4 (``docs/official_run_v2_plan_20260926.md`` §0 F3 / F5, §2 "B4", §3 tests 6 / 7 / 12, §7):
the reserved-root guard, the protocol freeze gate, the output route, the job runner and the merge.

What is checked:

* **test 6 -- reserved-root guard** (``src/ration_reliability/uncertainty/streams.py``; closes SB-05): the roots
  derived from the label rule equal ``configs/streams_policy.yaml`` (integer comparison; no value printed, even on
  failure); ``RandomStreams`` refuses a reserved root without a token -- tested on a **monkeypatched synthetic reserved
  set**, never on the real roots (a real reserved root is spent on its first draw, SP-4); only
  ``official_v2.require_protocol_frozen`` can mint a token; a hand-built, copied or altered token is refused; a token
  opens only the root indices it names; a non-official run record refuses a reserved-root stream id;
* **test 7 -- official gating** in throw-away git repositories (never the project repository; synthetic placeholder
  bytes stand in for the restricted data files): the gate passes only on the frozen, anchored commit, and refuses --
  with 0 measured draws and no token -- when the protocol is unfrozen, the freeze record is missing, tampered or not
  anchored, ``test_previously_seen`` is not true, the specification differs from the v2 pin, a code file changed, or a
  root index is a seed; ``write_protocol_freeze`` refuses before the protocol says frozen; the driver refuses
  ``--official`` without ``--authorised-compute``, with an authorisation file of the wrong scope and with a
  development root as ``--root-k`` (exit 2, nothing drawn or written); ``--official --dry-run`` draws nothing; the
  output routes;
* **test 12 -- job merge**: missing, duplicate, failed or tampered jobs are refused before anything is written; the
  tables are built per root with that root's jobs only (no pooled n); ``run_jobs`` runs one process per job with one
  BLAS thread.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ration_reliability.io import run_record as RR
from ration_reliability.uncertainty import streams as ST

REPO = Path(__file__).resolve().parents[2]
_OV2 = REPO / "experiments" / "E1_cost_reliability" / "official_v2.py"
_AB = REPO / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"
PY = sys.executable
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
SYN = (11, 22, 33)                               # synthetic reserved set (never a real reserved root)


def _load(path: Path, name: str):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OV = _load(_OV2, "official_v2_batch4_under_test")
AB = _load(_AB, "run_endpoint_ablation_batch4_under_test")


@pytest.fixture
def synthetic_reserved(monkeypatch):
    monkeypatch.setattr(ST, "reserved_formal_roots", lambda: SYN)
    return SYN


# =================================================================================================
# test 6: the reserved-root guard
# =================================================================================================

def test_derived_reserved_roots_equal_the_policy_by_integer_comparison(monkeypatch):
    doc = yaml.safe_load((REPO / "configs" / "streams_policy.yaml").read_text(encoding="utf-8"))
    rfs = doc["reserved_formal_streams"]
    listed = tuple(int(r["root_seed"]) for r in rfs["roots"])
    opened = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda *a, **k: (opened.append(a[0]), real_open(*a, **k))[1])
    derived = ST.reserved_formal_roots()                          # sha256 of the label only: no file is read
    monkeypatch.undo()
    same = listed == derived
    assert same, "derived reserved roots differ from configs/streams_policy.yaml (values withheld)"
    assert opened == [] and len(derived) == ST.N_RESERVED_ROOTS == 3
    assert rfs["generation"]["label_template"] == ST.RESERVED_ROOT_LABEL_TEMPLATE
    assert [r["k"] for r in rfs["roots"]] == [0, 1, 2]
    assert all(ST.reserved_root_index(v) == k for k, v in enumerate(derived))
    assert ST.reserved_root_index(1103) is None and not ST.is_reserved_root(1103)
    for bad in (-1, 3, True, "0"):
        with pytest.raises(ValueError):
            ST.derive_reserved_root(bad)


def test_random_streams_refuse_a_reserved_root_without_a_token(synthetic_reserved):
    for r in SYN:
        with pytest.raises(ST.ReservedRootError) as exc:
            ST.RandomStreams(r)
        assert str(r) not in str(exc.value) and "reserved" in str(exc.value)
        with pytest.raises(ST.ReservedRootError):
            ST.RandomStreams(r, official_token="a string is not a token")
    assert ST.RandomStreams(1103).stream_id("opt") == "root=1103/opt"            # development root unchanged
    assert ST.RandomStreams(1103).generator("opt").random(2).shape == (2,)


def test_only_require_protocol_frozen_can_mint_a_token():
    kw = dict(head_commit="h", freeze_record_sha256="f", protocol_sha256="p", spec_digest="d", roots_k=(0,),
              issued_utc="t")
    with pytest.raises(ST.ReservedRootError):
        ST.mint_official_stream_token(**kw)

    def require_protocol_frozen():                  # the right name in the wrong file is refused too
        return ST.mint_official_stream_token(**kw)
    with pytest.raises(ST.ReservedRootError):
        require_protocol_frozen()
    for seal in ("", "0" * 64):
        with pytest.raises(ST.ReservedRootError):
            ST.OfficialStreamToken(**kw, seal=seal)


def test_run_record_of_a_non_official_run_refuses_a_reserved_root_stream(synthetic_reserved):
    for rt in ("unit_test", "smoke", "pilot"):
        with pytest.raises(ValueError) as exc:
            RR.check_reserved_root_streams(rt, {"test": "root=22/test", "opt": "root=1103/opt"})
        assert "22" not in str(exc.value)
    RR.check_reserved_root_streams("official", {"test": "root=22/test"})
    RR.check_reserved_root_streams("pilot", {"test": "root=1103/test", "x": "root=220/test", "y": "root=formal-k0/t"})


# =================================================================================================
# test 7: official gating (throw-away git repositories; synthetic placeholder data files)
# =================================================================================================

def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), "-c", "user.name=batch4-test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _set_protocol(root: Path, *, frozen: bool, seen: bool = True) -> None:
    p = root / "configs" / "protocol.yaml"
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    doc["freeze"]["is_frozen"] = bool(frozen)
    doc["freeze"]["timestamp"] = "2026-09-27T00:00:00Z" if frozen else None
    doc["splits"]["test_previously_seen"] = True if seen else None
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _skeleton(root: Path) -> None:
    for rel in OV.SPEC_FILES:
        if (REPO / rel).is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, root / rel)
    shutil.copy2(REPO / "DECISIONS.md", root / "DECISIONS.md")
    (root / "scripts").mkdir(exist_ok=True)                                      # a code file outside the spec
    shutil.copy2(REPO / "scripts" / "verify_official_outputs.py", root / "scripts" / "verify_official_outputs.py")
    for rel in sorted(set(OV.DATA_FILES) | set(OV.SPEC_RESTRICTED_FILES)):      # synthetic stand-ins, never copies
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"synthetic placeholder for {Path(rel).name}\n".encode())
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")


@pytest.fixture(scope="module")
def frozen_repo(tmp_path_factory):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    root = tmp_path_factory.mktemp("b4") / "frozen"
    _skeleton(root)
    _set_protocol(root, frozen=True)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "specification")
    OV.write_freeze_pin(root, note="batch-4 test")
    doc = OV.write_protocol_freeze("batch-4 test freeze", root)
    assert doc["declared"]["jobs"] == [j["job_id"] for j in OV.official_jobs()]
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "freeze")
    return root


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _refused(root: Path, roots=(0, 1, 2)) -> list[str]:
    with pytest.raises(OV.FreezeGateError) as exc:
        OV.require_protocol_frozen(root, roots_k=roots)
    code, rep, tok = AB.official_gate(root, roots)
    assert code == 2 and tok is None and rep["draws"] == 0 and rep["solves"] == 0 and rep["files_written"] == 0
    return exc.value.reasons


def test_the_gate_passes_on_the_frozen_anchored_commit_and_the_token_opens_only_its_roots(frozen_repo, monkeypatch):
    g = OV.require_protocol_frozen(frozen_repo, roots_k=(1,))
    tok = g["token"]
    assert g["status"] == "protocol_frozen_anchored" and isinstance(tok, ST.OfficialStreamToken) and tok.valid()
    assert g["head"] == _git(frozen_repo, "rev-parse", "HEAD") and tok.roots_k == (1,)
    code, rep, tok2 = AB.official_gate(frozen_repo, (0,))
    assert code == 0 and tok2 is not None and rep["draws"] == 0 and rep["files_written"] == 0 and "seal" not in rep["token"]
    with pytest.raises(ST.ReservedRootError):                    # an altered token loses its seal
        dataclasses.replace(tok, roots_k=(0, 1, 2))
    monkeypatch.setattr(ST, "reserved_formal_roots", lambda: SYN)     # synthetic roots from here on
    s = ST.RandomStreams(SYN[1], official_token=tok)
    assert s.generator("opt").random(1).shape == (1,)
    for other in (SYN[0], SYN[2]):
        with pytest.raises(ST.ReservedRootError):
            ST.RandomStreams(other, official_token=tok)


def test_write_protocol_freeze_refuses_before_the_protocol_is_frozen(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    root = tmp_path / "unfrozen"
    _skeleton(root)
    _set_protocol(root, frozen=False)
    OV.write_freeze_pin(root, note="t")
    with pytest.raises(OV.FreezeGateError) as exc:
        OV.write_protocol_freeze("t", root)
    assert any("is_frozen" in r for r in exc.value.reasons) and not (root / OV.PROTOCOL_FREEZE).exists()
    reasons = _refused(root)
    assert any("protocol_freeze.json missing" in r for r in reasons) and any("is_frozen" in r for r in reasons)
    _set_protocol(root, frozen=True)
    (root / OV.OFFICIAL_PUBLIC_ROOT / "official-x").mkdir(parents=True)
    OV.write_freeze_pin(root, note="t")
    with pytest.raises(OV.FreezeGateError, match="official output already exists"):
        OV.write_protocol_freeze("t", root)


def _mutate_commit(root: Path, rel: str, fn, msg: str) -> None:
    p = root / rel
    p.write_text(fn(p.read_text(encoding="utf-8")), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


@pytest.mark.parametrize("case", ["unfrozen", "freeze_missing", "freeze_tampered", "unanchored_dirty",
                                  "unanchored_untracked", "not_previously_seen", "pin_differs", "code_changed",
                                  "seed_as_root"])
def test_the_gate_refuses_with_zero_draws(frozen_repo, tmp_path, case):
    root = _copy(frozen_repo, tmp_path / case)
    roots = (0, 1, 2)
    if case == "unfrozen":
        _set_protocol(root, frozen=False)
        _git(root, "commit", "-q", "-am", "unfreeze")
        want = "freeze.is_frozen is not true"
    elif case == "freeze_missing":
        _git(root, "rm", "-q", str(OV.PROTOCOL_FREEZE))
        _git(root, "commit", "-q", "-m", "drop the freeze record")
        want = "protocol_freeze.json missing"
    elif case == "freeze_tampered":
        _mutate_commit(root, OV.PROTOCOL_FREEZE.as_posix(),
                       lambda t: t.replace('"data_sha256": {', '"data_sha256": {"data/extra_file": "0000", ', 1),
                       "tamper")
        want = "data hashes differ"
    elif case == "unanchored_dirty":
        p = root / OV.PROTOCOL_FREEZE
        p.write_text(p.read_text(encoding="utf-8").replace('"note": "batch-4 test freeze"', '"note": "edited"', 1),
                     encoding="utf-8")
        want = "not clean"
    elif case == "unanchored_untracked":
        _git(root, "rm", "-q", "--cached", str(OV.PROTOCOL_FREEZE))
        _git(root, "commit", "-q", "-m", "untrack")
        want = "not tracked"
    elif case == "not_previously_seen":
        _set_protocol(root, frozen=True, seen=False)
        _git(root, "commit", "-q", "-am", "seen null")
        want = "test_previously_seen is not true"
    elif case == "pin_differs":
        _mutate_commit(root, "docs/reference_problem_v2.md", lambda t: t + "\nedited after the freeze\n", "doc")
        want = "specification differs from the v2 pin"
    elif case == "code_changed":
        _mutate_commit(root, "scripts/verify_official_outputs.py", lambda t: t + "\n# edited\n", "code")
        want = "code-manifest subset digest differs"
    else:
        roots = (1103,)
        want = "never a seed"
    reasons = _refused(root, roots)
    assert any(want in r for r in reasons), reasons


def _listing(p: Path) -> set[str]:
    return {x.name for x in p.iterdir()} if p.is_dir() else set()


def _auth(tmp_path: Path, scope: list[str], name: str) -> Path:
    p = tmp_path / f"authorisation_{name}.json"
    p.write_text(json.dumps({"authorised_hosts": [socket.gethostname()], "authorised_by": "unit test (synthetic)",
                             "authorised_on": "2026-09-27", "scope": scope}), encoding="utf-8")
    return p


def test_driver_refuses_official_without_authorisation_or_with_the_wrong_scope(tmp_path):
    """Only refusals that happen BEFORE the root indices and the freeze gate are run as driver processes against the
    project repository -- no authorisation file of the right scope is ever handed to ``--official`` here, so the test
    is safe in any freeze state (on the frozen commit a passing gate would open a reserved root).  The gate itself is
    tested on throw-away repositories above; the root-index refusal in-process below."""
    watched = [REPO / "results" / "official", REPO / "data" / "restricted_local" / "official",
               REPO / "data" / "restricted_local" / "debug"]
    before = [_listing(p) for p in watched]
    wrong = _auth(tmp_path, ["run_endpoint_ablation", "run_official_v1"], "wrong_scope")
    cases = [(["--official", "--root-k", "0", "--workers", "1"], "--authorised-compute"),
             (["--official", "--authorised-compute", "--authorisation-file", str(wrong), "--root-k", "0",
               "--workers", "1"], "does not cover run_official_v2"),
             (["--official", "--tiny"], "not allowed with argument"),
             (["--dry-run", "--root-k", "0"], "--root-k is for --official only")]
    for argv, want in cases:
        r = subprocess.run([PY, str(_AB), *argv], capture_output=True, text=True, env=ENV, timeout=300)
        assert r.returncode == 2, (argv, r.stderr[-800:])
        assert want in r.stderr, (argv, r.stderr[-800:])
        assert "Traceback" not in r.stderr and '"gate"' not in r.stdout          # refused before the gate
    assert [_listing(p) for p in watched] == before


def test_a_development_root_or_a_seed_is_refused_as_root_index():
    for bad in ([1103], [0, 0], [3], [-1], []):
        with pytest.raises(SystemExit) as exc:
            AB._official_roots(bad)
        assert "root" in str(exc.value)
    assert AB._official_roots([0, 2]) == [0, 2]
    with pytest.raises(SystemExit, match="never a seed"):
        AB._official_roots([1103])


def test_official_dry_run_draws_nothing_and_refuses_until_frozen():
    r = subprocess.run([PY, str(_AB), "--official", "--dry-run"], capture_output=True, text=True, env=ENV, timeout=600)
    out = json.loads(r.stdout)
    assert out["official_dry_run"] is True and out["draws"] == 0 and out["solves"] == 0 and out["files_written"] == 0
    frozen = (REPO / OV.PROTOCOL_FREEZE).is_file()
    if not frozen:
        assert r.returncode == 2 and out["gate"]["status"] == "refused"
        assert any("protocol_freeze.json missing" in x for x in out["gate"]["reasons"])
    else:                                                     # after the freeze commit: ready only when anchored
        assert (r.returncode == 0) == out["status"].startswith("ready")
    doc = yaml.safe_load((REPO / "configs" / "streams_policy.yaml").read_text(encoding="utf-8"))
    leaked = any(str(int(x["root_seed"])) in r.stdout for x in doc["reserved_formal_streams"]["roots"])
    assert not leaked, "a reserved root value appears in the --official --dry-run output"


def test_check_seed_is_inverted_in_official_mode(synthetic_reserved, monkeypatch):
    monkeypatch.setattr(AB, "reserved_roots", lambda repo=REPO: set(SYN))
    assert AB.check_seed(SYN[2], official=True)["root_k"] == 2
    for bad in (1103, 12):
        with pytest.raises(SystemExit) as exc:
            AB.check_seed(bad, official=True)
        assert "not a reserved" in str(exc.value)
    with pytest.raises(SystemExit):
        AB.check_seed(SYN[0])                                   # development mode still refuses a reserved root
    assert AB.check_seed(1103)["seed"] == 1103


def test_output_routes():
    r = OV.official_output_route("official-20260927T000000Z-abcdef01")
    assert (r["label"], r["public_dir"], r["restricted_dir"]) == (
        "official", "results/official/official-20260927T000000Z-abcdef01",
        "data/restricted_local/official/official-20260927T000000Z-abcdef01")
    d = OV.official_output_route("debug-rehearsal-20260927T000000Z-abcdef01")
    assert d["label"] == "debug" and d["public_dir"].startswith("data/restricted_local/debug/")
    assert d["restricted_dir"].startswith("data/restricted_local/debug/") and "results" not in d["public_dir"]
    for bad in ("pilot-x", "../official-x", "official-x/y", ""):
        with pytest.raises(ValueError):
            OV.official_output_route(bad)
    with pytest.raises(ValueError):
        OV.official_output_route("official-x", rehearsal=True)


def test_public_files_alias_the_reserved_roots():
    obj = {"test_stream_id": "root=11/test", "other": "root=110/test", "n": 11, "k": 110, "nested": [{"root=22/opt": 1}],
           "text": "streams opt=root=22/opt (n=4)", "flag": True}
    out = OV.alias_reserved_roots(obj, {0: 11, 1: 22})
    assert out["test_stream_id"] == "root=formal-k0/test" and out["other"] == "root=110/test"
    assert out["n"] == "formal-k0" and out["k"] == 110 and out["flag"] is True
    assert out["nested"] == [{"root=formal-k1/opt": 1}] and out["text"] == "streams opt=root=formal-k1/opt (n=4)"
    assert OV.alias_reserved_roots(obj, {}) is obj


def test_d539_diagnostics_scope_in_the_configuration():
    C = OV.OFFICIAL_CONFIG
    assert C["matrix"]["diagnostics_roots_k"] == [0] and C["diagnostics"]["roots_k"] == [0]
    assert C["diagnostics"]["DIAG-T45"]["training_worlds"] == ["SD-H0", "SD-S2"]
    assert C["diagnostics"]["DIAG-E1"]["training_worlds"] == ["SD-H0", "SD-S2"]
    assert C["diagnostics"]["DIAG-T45"]["h_grid"]["a1_equal_subdivisions"] >= 1
    assert C["matrix"]["workers"] == 8 and "D-539" in C["decisions"]
    assert C["primary_assumption_id"] == OV.PRIMARY_ASSUMPTION_ID
    cfg = OV.with_run_context({"run_context": {"fit_sets": ["opt", "validation"]}},
                              {"protocol_sha256": "p", "primary_assumption_id": "a"})
    assert cfg["run_context"] == {"fit_sets": ["opt", "validation"], "protocol_sha256": "p",
                                  "primary_assumption_id": "a"}
    for extra in ("experiments/E1_cost_reliability/official_v2.py", "configs/dev_case_v1/reference_constraints_v2.csv",
                  "configs/protocol.yaml", "docs/reference_problem_v2.md", "src/ration_reliability/evaluation/stats.py",
                  "src/ration_reliability/optimization/safety_margin.py", "src/ration_reliability/uncertainty/streams.py"):
        assert extra in AB.SPEC_FILES
    assert AB.FREEZE_PIN.name == "reference_problem_v2_freeze.json" and AB.FREEZE_PIN_V1.name == (
        "reference_problem_v1_freeze.json")


# =================================================================================================
# test 12: job runner and merge
# =================================================================================================

def test_run_jobs_one_process_per_job_with_one_blas_thread(tmp_path):
    code = ("import os, sys; ok = all(os.environ.get(v) == '1' for v in "
            "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')); print(os.getpid()); sys.exit(0 if ok else 3)")
    jobs = [{"job_id": f"j{i}", "argv": [PY, "-c", code]} for i in range(3)]
    out = OV.run_jobs(jobs, 2, cwd=tmp_path, log_dir=tmp_path)
    assert [r["job_id"] for r in out] == ["j0", "j1", "j2"] and all(r["exit_status"] == 0 for r in out)
    pids = {(tmp_path / f"j{i}.log").read_text().strip() for i in range(3)}
    assert len(pids) == 3                                              # three separate processes
    seen = []
    res = OV.run_jobs([{"job_id": "a"}, {"job_id": "b"}], 1, runner=lambda j, env: seen.append(env["OMP_NUM_THREADS"])
                      or (0 if j["job_id"] == "a" else 7))
    assert [r["exit_status"] for r in res] == [0, 7] and seen == ["1", "1"]
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            OV.run_jobs(jobs, bad, log_dir=tmp_path)
    with pytest.raises(ValueError, match="twice"):
        OV.run_jobs([{"job_id": "a"}, {"job_id": "a"}], 1, runner=lambda j, e: 0)


RUN = "official-20260927T000000Z-00000000"
JOBS = [{"job_id": "root0__SDH0_MAIN9", "root_k": 0, "kind": "cell", "cell_id": "SDH0_MAIN9"},
        {"job_id": "root0__diagnostics", "root_k": 0, "kind": "diagnostics", "cell_id": None},
        {"job_id": "root1__SDH0_MAIN9", "root_k": 1, "kind": "cell", "cell_id": "SDH0_MAIN9"}]


def _synthetic_run(repo: Path, jobs=JOBS) -> tuple[Path, Path]:
    route = OV.official_output_route(RUN)
    pub, res = repo / route["public_dir"], repo / route["restricted_dir"]
    for j in jobs:
        jp, jr = pub / "jobs" / j["job_id"], res / "jobs" / j["job_id"]
        jp.mkdir(parents=True, exist_ok=True)
        jr.mkdir(parents=True, exist_ok=True)
        (jp / "job_record.json").write_text(json.dumps({"job": j, "n_test": 10}), encoding="utf-8")
        (jr / "job_payload.json").write_text(json.dumps({"rows": [{"root_k": j["root_k"], "n": 10}]}), encoding="utf-8")
        for d in (jr, jp):
            OV.write_sha_manifest(d)
            OV.write_done(d, j["job_id"])
    (pub / OV.RUN_PLAN_FILE).write_text(json.dumps({"run_id": RUN, "jobs": jobs}), encoding="utf-8")
    return pub, res


def test_merge_builds_tables_per_root_without_pooling(tmp_path):
    pub, res = _synthetic_run(tmp_path)
    calls = []

    def build(k, items, pr, rr):
        calls.append((k, sorted(it["job"]["job_id"] for it in items)))
        assert all(it["job"]["root_k"] == k and it["payload"]["rows"][0]["root_k"] == k for it in items)
        (pr / "table.csv").write_text("root_k,n\n" + "".join(f"{k},{it['payload']['rows'][0]['n']}\n" for it in items))
        return {"n_rows": len(items)}
    out = OV.merge_jobs(RUN, repo=tmp_path, build_tables=build)
    assert calls == [(0, ["root0__SDH0_MAIN9", "root0__diagnostics"]), (1, ["root1__SDH0_MAIN9"])]
    assert out["pooled_across_roots"] is False and set(out["per_root"]) == {"0", "1"}
    assert out["per_root"]["0"]["n_rows"] == 2 and out["per_root"]["1"]["n_rows"] == 1
    assert not any("pool" in k or k in ("n_total", "n_all_roots") for k in out if k != "pooled_across_roots")
    assert (pub / "root0" / "table.csv").is_file() and (pub / "root1" / "table.csv").is_file()
    assert not (pub / "table.csv").exists()


@pytest.mark.parametrize("case", ["missing", "listed_twice", "duplicate_dir", "failed", "tampered", "merged"])
def test_merge_refuses_missing_duplicate_failed_or_tampered_jobs(tmp_path, case):
    jobs = list(JOBS)
    pub, res = _synthetic_run(tmp_path)
    if case == "missing":
        (res / "jobs" / "root1__SDH0_MAIN9" / OV.DONE_FILE).unlink()
        want = "without DONE"
    elif case == "listed_twice":
        jobs = jobs + [jobs[0]]
        (pub / OV.RUN_PLAN_FILE).write_text(json.dumps({"run_id": RUN, "jobs": jobs}), encoding="utf-8")
        want = "listed twice"
    elif case == "duplicate_dir":
        shutil.copytree(pub / "jobs" / "root0__SDH0_MAIN9", pub / "jobs" / "root0__SDH0_MAIN9_copy")
        want = "holds the DONE of job"
    elif case == "failed":
        d = pub / "jobs" / "root0__diagnostics"
        (d / OV.DONE_FILE).unlink()
        OV.write_done(d, "root0__diagnostics", exit_status=1)
        want = "nonzero exit status"
    elif case == "tampered":
        (res / "jobs" / "root0__SDH0_MAIN9" / "job_payload.json").write_text('{"rows": []}', encoding="utf-8")
        want = "file changed"
    else:
        (pub / OV.MERGE_DONE).write_text("{}", encoding="utf-8")
        want = "already merged"
    calls = []
    with pytest.raises(OV.MergeError) as exc:
        OV.merge_jobs(RUN, repo=tmp_path, build_tables=lambda k, items, pr, rr: calls.append(k))
    assert any(want in r for r in exc.value.reasons), exc.value.reasons
    assert calls == [] and not (pub / "root0").exists() and not (pub / "root1").exists()


def test_sha_manifest_detects_unlisted_and_changed_files(tmp_path):
    d = tmp_path / "j"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("a")
    (d / "sub" / "b.txt").write_text("b")
    OV.write_sha_manifest(d)
    OV.write_done(d, "j")
    assert OV.verify_sha_manifest(d) == [] and OV.read_done(d)["job_id"] == "j"
    (d / "c.txt").write_text("c")
    (d / "sub" / "b.txt").write_text("B")
    probs = OV.verify_sha_manifest(d)
    assert any("unlisted" in p for p in probs) and any("changed" in p for p in probs)


# =================================================================================================
# plan §5: the compute-host scripts (static checks only -- never executed by a test: on the frozen commit the run
# script would start the official run)
# =================================================================================================

@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_px_scripts_are_valid_bash_and_carry_the_declared_steps():
    for name in ("px_official_prepare.sh", "px_official_run.sh"):
        r = subprocess.run(["bash", "-n", str(REPO / "scripts" / name)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
    prep = (REPO / "scripts" / "px_official_prepare.sh").read_text(encoding="utf-8")
    for step in ("scripts/replay_build_outputs.py apply", "-m pytest -q -p no:cacheprovider",
                 "scripts/preflight_dev_case.py --driver run_dev_case_v1 --run-type official",
                 "scripts/check_environment.py", "--official --dry-run", "matches_frozen_anchored",
                 "OPENBLAS_NUM_THREADS=1", "configs/protocol_freeze.json"):
        assert step in prep, step
    run = (REPO / "scripts" / "px_official_run.sh").read_text(encoding="utf-8")
    for step in ("OFFICIAL_STARTED", "OFFICIAL_EXIT", "OFFICIAL_ENDED", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                 "MKL_NUM_THREADS=1", "nice -n 10", "--official --authorised-compute", "--root-k $ROOTS",
                 "--workers \"$WORKERS\"", "configs/protocol_freeze.json", "--official-merge"):
        assert step in run, step
    # the replay script's interface is the one the preparation script assumes
    rb = (REPO / "scripts" / "replay_build_outputs.py").read_text(encoding="utf-8")
    for opt in ('"--work-dir"', '"--python"', '"--bundle"', '"--superseded-dir"'):
        assert opt in rb, opt
