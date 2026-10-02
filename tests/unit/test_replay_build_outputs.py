"""scripts/replay_build_outputs.py: the scripted form of the 2026-09-25 Mac-bytes replay (round-3 red team D-10; FIX2).

Everything runs on a fixture project under ``tmp_path`` with synthetic "build outputs" and a fake command runner (git,
build, preflight, dry run and pytest are simulated); nothing is built, solved or copied between hosts, and no
restricted file is read.  What is checked:

1. ``--dry-run`` prints the plan -- the steps and commands of ``mac_bytes_replay.md`` (single-thread build,
   preflight ``--driver run_dev_case_v1 --run-type pilot``, driver ``--dry-run``) -- and touches nothing;
2. ``reference``: back up -> rebuild -> byte-identical outputs -> sidecar + clean tree -> preflight READY -> dry run
   ``matches_frozen_anchored`` -> bundle with a sha256 manifest; a non-deterministic rebuild restores the backup and
   bundles nothing; a dirty tree stops before anything is touched;
3. ``apply``: the bundle is verified (names, sha256, commit) before the host is touched; the host's own outputs are
   kept in the superseded directory; the replaced bytes equal the bundle; preflight and dry run must pass
   (``differs_from_frozen`` is a failure); ``--pytest`` runs the test suite; path-traversal members are refused;
4. the work directory must be outside the repository or under ``data/restricted_local/``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("replay_build_outputs_under_test",
                                                  REPO / "scripts" / "replay_build_outputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = _load()
HEAD = "46c22cbcf384edf2e9f9b5e053e6a158420654c1"


def _project(root: Path, *, tag: str = "mac") -> Path:
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "build_dev_case.py").write_text("# placeholder\n", encoding="utf-8")
    (root / "experiments" / "E1_cost_reliability").mkdir(parents=True)
    (root / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py").write_text("# placeholder\n")
    case = root / RB.CASE_DIR
    case.mkdir(parents=True)
    for n in RB.REPLAY_FILES:
        (case / n).write_text(f"synthetic {n} built on {tag}\n", encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class FakeHost:
    """Simulated commands of one host.  ``rebuild`` rewrites the outputs (``drift`` changes one byte, as a platform
    whose build is not byte-identical); ``freeze`` is the dry run's freeze status."""

    def __init__(self, root: Path, *, head=HEAD, dirty="", drift=False, freeze=RB.FROZEN_OK, ready=True,
                 build_exit=0, pytest_exit=0):
        self.root, self.head, self.dirty, self.drift, self.freeze = root, head, dirty, drift, freeze
        self.ready, self.build_exit, self.pytest_exit = ready, build_exit, pytest_exit
        self.calls: list[list[str]] = []

    def __call__(self, argv, cwd, env):
        self.calls.append(list(argv))
        if argv[0] == "git":
            return (0, self.head + "\n", "") if "rev-parse" in argv else (0, self.dirty, "")
        if argv[1:] == list(RB.BUILD_CMD):
            assert env.get("OPENBLAS_NUM_THREADS") == "1"
            case = self.root / RB.CASE_DIR
            if self.drift:
                p = case / RB.OUTPUT_FILES[0]
                p.write_bytes(p.read_bytes() + b"~")
            (case / RB.SIDECAR).write_text("sidecar rewritten by the build\n", encoding="utf-8")
            return self.build_exit, "build summary\n", ""
        if argv[1:] == list(RB.PREFLIGHT_CMD):
            return (0, "[preflight dev_case_v1] READY\n", "") if self.ready else (1, "INCONSISTENT\n", "")
        if argv[1:] == list(RB.DRYRUN_CMD):
            doc = {"dry_run": True, "checks": {"preflight": "READY", "reference_spec_fingerprint": "e7f9" + "0" * 60,
                                                "freeze": {"status": self.freeze, "anchor": {"anchored": True}}}}
            return 0, json.dumps(doc, indent=1), ""
        if argv[1:3] == ["-m", "pytest"]:
            return self.pytest_exit, "931 passed, 2 xfailed\n", ""
        raise AssertionError(f"unexpected command {argv}")


def _reference(tmp_path: Path, **kw) -> tuple[Path, Path, int, FakeHost]:
    mac = _project(tmp_path / "mac")
    work = tmp_path / "mac_work"
    host = FakeHost(mac, **kw)
    rc = RB.main(["reference", "--root", str(mac), "--work-dir", str(work), "--python", "py"], runner=host)
    return mac, work, rc, host


def _log(work: Path) -> list[dict]:
    return [json.loads(x) for x in (work / RB.LOG_NAME).read_text(encoding="utf-8").splitlines()]


def test_dry_run_prints_the_recorded_steps_and_touches_nothing(tmp_path, capsys):
    mac = _project(tmp_path / "mac")
    before = _snapshot(tmp_path)
    work = tmp_path / "w"

    def never(*_a):
        raise AssertionError("a dry run must not run any command")
    assert RB.main(["reference", "--root", str(mac), "--work-dir", str(work), "--python", "py", "--dry-run"],
                   runner=never) == 0
    plan = json.loads(capsys.readouterr().out)
    names = [s[0] for s in plan["steps"]]
    assert names[:8] == ["r1_repository_clean", "r2_backup", "r3_build", "r4_determinism",
                         "r5_sidecar_and_clean_tree", "r6_preflight", "r7_dry_run", "r8_bundle"]
    cmds = dict((s[0], s[1]) for s in plan["steps"])
    assert "OPENBLAS_NUM_THREADS=1" in cmds["r3_build"] and "scripts/build_dev_case.py" in cmds["r3_build"]
    assert "--driver run_dev_case_v1 --run-type pilot" in cmds["r6_preflight"]
    assert "run_endpoint_ablation.py --dry-run" in cmds["r7_dry_run"] and RB.FROZEN_OK in cmds["r7_dry_run"]
    assert RB.main(["apply", "--root", str(mac), "--work-dir", str(work), "--bundle", str(tmp_path / "b.tgz"),
                    "--superseded-dir", str(tmp_path / "sup"), "--pytest", "--dry-run"], runner=never) == 0
    plan2 = json.loads(capsys.readouterr().out)
    assert [s[0] for s in plan2["steps"]][-1] == "a7_pytest" and plan2["bundle_exists"] is False
    assert _snapshot(tmp_path) == before and not work.exists()


def test_reference_rebuilds_checks_determinism_and_bundles_with_a_manifest(tmp_path):
    mac, work, rc, host = _reference(tmp_path)
    assert rc == 0
    log = _log(work)
    assert [r["step"] for r in log] == ["r1_repository_clean", "r2_backup", "r3_build", "r4_determinism",
                                        "r5_sidecar_and_clean_tree", "r6_preflight", "r7_dry_run", "r8_bundle"]
    assert all(r["ok"] for r in log)
    members, man = RB.read_bundle(work / RB.BUNDLE_NAME)
    assert set(members) == set(RB.REPLAY_FILES) and man["git_head"] == HEAD
    assert all(hashlib.sha256(b).hexdigest() == man["files"][n] for n, b in members.items())
    assert man["dry_run_freeze_status"] == RB.FROZEN_OK and man["reference_spec_fingerprint"].startswith("e7f9")
    assert members[RB.SIDECAR] == b"sidecar rewritten by the build\n"          # the new sidecar travels
    assert (work / "commands" / "r3_build.stdout").is_file() and (work / "dryrun_reference.json").is_file()


def test_a_rebuild_that_is_not_byte_identical_restores_the_backup_and_bundles_nothing(tmp_path):
    mac = _project(tmp_path / "mac")
    before = _snapshot(mac)
    work = tmp_path / "mac_work"
    rc = RB.main(["reference", "--root", str(mac), "--work-dir", str(work), "--python", "py"],
                 runner=FakeHost(mac, drift=True))
    assert rc == 1 and not (work / RB.BUNDLE_NAME).exists()
    assert _snapshot(mac) == before                                            # every output back as it was
    last = _log(work)[-1]
    assert last["step"] == "r4_determinism" and not last["ok"] and last["differ"] == [RB.OUTPUT_FILES[0]]


def test_a_dirty_tree_stops_the_reference_replay_before_anything_is_touched(tmp_path):
    mac = _project(tmp_path / "mac")
    before = _snapshot(mac)
    host = FakeHost(mac, dirty=" M src/x.py\n")
    rc = RB.main(["reference", "--root", str(mac), "--work-dir", str(tmp_path / "w"), "--python", "py"], runner=host)
    assert rc == 1 and _snapshot(mac) == before
    assert not any(c[1:] == list(RB.BUILD_CMD) for c in host.calls)


def test_apply_verifies_the_bundle_keeps_the_host_bytes_and_requires_the_anchored_freeze(tmp_path):
    _mac, mac_work, rc, _h = _reference(tmp_path)
    assert rc == 0
    px = _project(tmp_path / "px", tag="px")                                   # host-built bytes differ
    own = _snapshot(px / RB.CASE_DIR)
    sup, work = tmp_path / "px_superseded", tmp_path / "px_work"
    host = FakeHost(px)
    rc2 = RB.main(["apply", "--root", str(px), "--work-dir", str(work), "--bundle", str(mac_work / RB.BUNDLE_NAME),
                   "--superseded-dir", str(sup), "--python", "py", "--pytest"], runner=host)
    assert rc2 == 0
    members, man = RB.read_bundle(mac_work / RB.BUNDLE_NAME)
    for n in RB.REPLAY_FILES:
        assert (px / RB.CASE_DIR / n).read_bytes() == members[n]
        assert hashlib.sha256((sup / n).read_bytes()).hexdigest() == own[n]     # the host's own bytes are kept
    steps = [r["step"] for r in _log(work)]
    assert steps == ["a1_repository", "a2_bundle_verified", "a3_backup_host_outputs", "a4_written", "a5_preflight",
                     "a6_dry_run", "a7_pytest"]
    assert (work / "dryrun_after_replay.json").is_file() and json.loads((sup / "SUPERSEDED.json").read_text())
    # the same replay onto a host whose dry run is not anchored: a failure (bytes replaced, host copy kept)
    px2 = _project(tmp_path / "px2", tag="px")
    rc3 = RB.main(["apply", "--root", str(px2), "--work-dir", str(tmp_path / "w2"), "--bundle",
                   str(mac_work / RB.BUNDLE_NAME), "--superseded-dir", str(tmp_path / "s2"), "--python", "py"],
                  runner=FakeHost(px2, freeze="differs_from_frozen"))
    assert rc3 == 1 and _log(tmp_path / "w2")[-1]["step"] == "a6_dry_run"


@pytest.mark.parametrize("how", ["tampered", "other_commit", "superseded_not_empty"])
def test_apply_refuses_a_bad_bundle_or_state_before_touching_the_host(tmp_path, how):
    _mac, mac_work, _rc, _h = _reference(tmp_path)
    bundle = mac_work / RB.BUNDLE_NAME
    px = _project(tmp_path / "px", tag="px")
    before = _snapshot(px)
    sup = tmp_path / "sup"
    head = HEAD
    if how == "tampered":                          # one member changed, manifest kept
        members, man = RB.read_bundle(bundle)
        bundle = tmp_path / "tampered.tgz"
        with tarfile.open(bundle, "w:gz") as tf:
            for n, b in dict(members, **{RB.OUTPUT_FILES[0]: members[RB.OUTPUT_FILES[0]] + b"x"}).items():
                ti = tarfile.TarInfo(n)
                ti.size = len(b)
                tf.addfile(ti, io.BytesIO(b))
            mb = json.dumps(man).encode()
            ti = tarfile.TarInfo(RB.BUNDLE_MANIFEST)
            ti.size = len(mb)
            tf.addfile(ti, io.BytesIO(mb))
    elif how == "other_commit":
        head = "0" * 40
    else:
        sup.mkdir()
        (sup / "old.json").write_text("{}")
    rc = RB.main(["apply", "--root", str(px), "--work-dir", str(tmp_path / "w"), "--bundle", str(bundle),
                  "--superseded-dir", str(sup), "--python", "py"], runner=FakeHost(px, head=head))
    assert rc == 1 and _snapshot(px) == before


def test_a_member_with_a_path_is_refused(tmp_path):
    b = tmp_path / "evil.tgz"
    with tarfile.open(b, "w:gz") as tf:
        ti = tarfile.TarInfo("../escape.json")
        ti.size = 2
        tf.addfile(ti, io.BytesIO(b"{}"))
    with pytest.raises(RB.StepFailed, match="refused"):
        RB.read_bundle(b)
    px = _project(tmp_path / "px")
    before = _snapshot(px)
    rc = RB.main(["apply", "--root", str(px), "--work-dir", str(tmp_path / "w"), "--bundle", str(b),
                  "--superseded-dir", str(tmp_path / "s"), "--python", "py"], runner=FakeHost(px))
    assert rc == 1 and _snapshot(px) == before and not (tmp_path / "escape.json").exists()


def test_the_work_directory_must_be_outside_the_repository_or_restricted(tmp_path):
    root = _project(tmp_path / "p")
    assert RB.work_dir_problem(root, root / "reports" / "replay") is not None
    assert RB.work_dir_problem(root, root / RB.RESTRICTED_ROOT / "replay") is None
    assert RB.work_dir_problem(root, tmp_path / "elsewhere") is None
    assert RB.main(["reference", "--root", str(root), "--work-dir", str(root / "reports" / "x"), "--dry-run"]) == 3
    assert RB.main(["reference", "--root", str(tmp_path / "nope"), "--work-dir", str(tmp_path / "w"), "--dry-run"]) == 3


def test_the_dry_run_parser_reads_the_driver_json_even_after_other_lines():
    out = "warning: something\n" + json.dumps({"checks": {"freeze": {"status": RB.FROZEN_OK, "anchor": {
        "anchored": True}}, "reference_spec_fingerprint": "abc"}})
    info = RB.parse_dry_run(out)
    assert info["parsed"] and info["freeze_status"] == RB.FROZEN_OK and info["anchored"] is True
    assert RB.parse_dry_run("not json")["parsed"] is False
    # the recorded 2026-09-25 dry run after the replay (a public log of the run, if present) parses the same way
    p = REPO / "logs" / "px_runs" / "pilot-20260925T055921Z-fb4f75af" / "dryrun2.json"
    if p.is_file():
        assert RB.parse_dry_run(p.read_text(encoding="utf-8"))["freeze_status"] == RB.FROZEN_OK
