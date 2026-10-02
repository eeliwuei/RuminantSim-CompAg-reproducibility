"""Run records carry every field of evidence_contract.yaml ``run_record_required``."""

import json

import numpy as np
import pytest
import yaml

from ration_reliability.hashing import stable_hash
from ration_reliability.io import build_run_record, code_fingerprint, utc_now, write_run_record

CONTRACT = "docs/contract/RuminantSim_Agent_Execution_Pack/templates/evidence_contract.yaml"
# Frozen copy of ``run_record_required`` (FIX_C, 2026-09-25): an external reproduction package does not
# carry docs/contract/ (docs/package_scopes.md), so the field check must not depend on that file being
# present.  When the contract file IS present, the copy is asserted equal to it (the copy cannot drift).
RUN_RECORD_REQUIRED_FROZEN = [
    "run_id", "run_type", "started_at", "completed_at", "code_commit_or_hash", "dirty_diff_hash",
    "data_manifest_hash", "config_hash", "protocol_hash", "rng_streams", "solver_version", "tolerances",
    "command", "exit_status", "output_paths", "output_hashes",
]


def _required_run_record_fields(repo_root):
    path = repo_root / CONTRACT
    if path.is_file():
        required = yaml.safe_load(path.read_text(encoding="utf-8"))["run_record_required"]
        assert required == RUN_RECORD_REQUIRED_FROZEN, "evidence_contract.yaml changed: update the frozen copy"
        return required
    return list(RUN_RECORD_REQUIRED_FROZEN)  # external package without docs/contract/


def test_frozen_contract_copy_matches_the_contract_when_present(repo_root):
    if not (repo_root / CONTRACT).is_file():
        pytest.skip("docs/contract/ is not part of this package (external reproduction package); the frozen copy "
                    "RUN_RECORD_REQUIRED_FROZEN is used by test_run_record_has_contract_fields instead")
    assert _required_run_record_fields(repo_root) == RUN_RECORD_REQUIRED_FROZEN


def test_stable_hash_properties():
    a = np.array([1.0, 2.0])
    assert stable_hash(a) == stable_hash(np.array([1.0, 2.0]))
    assert stable_hash(a) != stable_hash(np.array([1.0, 2.0 + 1e-15]))
    assert stable_hash({"x": 1, "y": 2}) == stable_hash({"y": 2, "x": 1})
    assert stable_hash(1) != stable_hash(1.0) != stable_hash("1")
    with pytest.raises(TypeError):
        stable_hash(object())


def test_code_fingerprint_hashes_src(repo_root):
    fp = code_fingerprint(repo_root)
    assert fp["src_tree_sha256"] and len(fp["src_tree_sha256"]) == 64
    assert fp["src_file_count"] >= 10


def test_run_record_has_contract_fields(tmp_path, repo_root, toy_yaml_path):
    required = _required_run_record_fields(repo_root)
    out = tmp_path / "out.json"
    out.write_text("{}", encoding="utf-8")
    t0 = utc_now()
    rec = build_run_record(run_type="unit_test", command="pytest (record test)", repo_root=repo_root,
                           started_at=t0, completed_at=utc_now(), exit_status=0,
                           rng_streams={"test": "root=1103/test/0"}, solver_version="n/a", tolerances={},
                           config_paths=[toy_yaml_path], data_paths=[toy_yaml_path], output_paths=[out],
                           is_synthetic=True)
    missing = [k for k in required if k not in rec]
    assert not missing, missing
    assert rec["config_hash"] and rec["output_hashes"][str(out)]
    assert rec["protocol_hash"] is None  # not given -> not invented
    p = write_run_record(rec, tmp_path / "run.json")
    assert json.loads(p.read_text(encoding="utf-8"))["run_id"] == rec["run_id"]
    with pytest.raises(FileExistsError):
        write_run_record(rec, p)
    with pytest.raises(ValueError):
        build_run_record(run_type="final", command="", repo_root=repo_root, started_at=t0, completed_at=t0,
                         exit_status=0, rng_streams={}, solver_version="", tolerances={}, is_synthetic=True)
