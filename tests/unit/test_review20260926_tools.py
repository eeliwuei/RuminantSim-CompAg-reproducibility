"""Review utilities: synthetic examples and read-only stored-result checks."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_demo_computes_all_four_equivalent_problems():
    result = load_script('run_synthetic_demo').run()
    assert result['is_synthetic'] and result['n_methods'] == 4
    assert result['point_mass_equivalence_passed']
    for row in result['results']:
        assert row['evaluation']['joint']['rate_upper'] == 0
        assert row['evaluation']['structural_ok']


def test_demo_refuses_overwrite_before_running(tmp_path):
    out = tmp_path / 'kept.json'
    out.write_text('original')
    assert load_script('run_synthetic_demo').main(['--output', str(out)]) == 2
    assert out.read_text() == 'original'


@pytest.mark.parametrize('bad', ['../other', '', '..', 'a/b'])
def test_audit_refuses_path_traversal(tmp_path, bad):
    with pytest.raises(ValueError): load_script('verify_public_results').audit(tmp_path, bad)


@pytest.mark.parametrize('bad', ['nan', 'inf', '-inf'])
def test_audit_rejects_nonfinite_metrics(bad):
    with pytest.raises(ValueError): load_script('verify_public_results').finite(bad)


@pytest.mark.parametrize('bad', ['2.5', '-1', '', 'nan'])
def test_audit_rejects_invalid_saved_count(bad):
    with pytest.raises(ValueError): load_script('verify_public_results').count(bad)


def test_saved_public_audit_includes_no_ration_entries_in_denominator():
    # These are supplied historical aggregate outputs, not original restricted
    # samples or a new optimisation. The utility must state that distinction.
    result = load_script('verify_public_results').audit(ROOT, 'pilot-20260925T055921Z-fb4f75af')
    assert result['n_failed'] == 0
    assert result['endpoint_rows'] == 152
    assert result['evaluated_endpoint_rows'] == 52
    assert len(result['available_hash_inputs']) + len(result['unavailable_outputs']) == 8
    assert all(row['attempted'] == 42 for row in result['comparable_sets'])
    assert 'NOT rerun' in result['coverage']


def test_delivery_manifest_pass_and_detect_tamper(tmp_path):
    import hashlib
    tool = load_script('verify_delivery_manifest')
    f = tmp_path / 'a.txt'; f.write_bytes(b'original')
    data = {'files': {'a.txt': {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}}}
    (tmp_path / 'REVIEW_MANIFEST.json').write_text(json.dumps(data))
    assert tool.verify(tmp_path)['status'] == 'PASS'
    f.write_bytes(b'tampered')
    assert tool.verify(tmp_path)['status'] == 'FAIL'


def test_delivery_manifest_rejects_outside_path(tmp_path):
    tool = load_script('verify_delivery_manifest')
    (tmp_path / 'REVIEW_MANIFEST.json').write_text(json.dumps({'files': {'../escape': {'bytes': 0, 'sha256': ''}}}))
    result = tool.verify(tmp_path)
    assert result['status'] == 'FAIL'
    assert result['failures'][0]['reason'] == 'path outside project'
