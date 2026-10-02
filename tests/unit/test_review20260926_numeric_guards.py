"""Synthetic regression cases for the 2026-09-26 repair; no feed data are read."""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import binomtest
from ration_reliability.evaluation.stats import (
    clopper_pearson, one_sided_upper, zero_event_upper_bound, mc_standard_error,
)
from ration_reliability.evaluation.reference import rate_block

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def replay():
    spec = importlib.util.spec_from_file_location('review20260926_replay', ROOT / 'experiments/E0_verification/replay_dev_case_eval.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@pytest.mark.parametrize('left,right', [
    (5.0, float('inf')), (float('inf'), 5.0), (float('nan'), None),
    ('not-a-number', None), ('bad', 'bad'), (True, 1.0),
    (float('-inf'), float('-inf')), (1e308, -1e308),
])
def test_invalid_float_evidence_never_passes(replay, left, right):
    out = replay.cmp_float(left, right)
    assert out['pass'] is False and out['exact'] is False

@pytest.mark.parametrize('left,right', [(2.5, 2.5), (-1, -1), (True, 1), ('bad', None),
                                       (float('inf'), float('inf')),
                                       (9007199254740992, '9007199254740993')])
def test_invalid_or_different_counts_never_pass(replay, left, right):
    assert replay.cmp_count(left, right)['pass'] is False


def test_count_comparison_preserves_large_integers(replay):
    assert replay.cmp_count(9007199254740993, '9007199254740993')['pass']
    assert replay.cmp_count(3, '3.0')['pass']
    assert replay.cmp_float(None, '')['pass']

@pytest.mark.parametrize('bad', [-1, 1.2, True, '3', float('inf'), float('nan')])
def test_statistics_reject_invalid_counts_without_truncation(bad):
    with pytest.raises(ValueError): clopper_pearson(bad, 10)
    with pytest.raises(ValueError): one_sided_upper(bad, 10)
    with pytest.raises(ValueError): rate_block(0, bad, 10)

@pytest.mark.parametrize('bad', [0, 1, -0.1, 1.1, float('nan'), float('inf'), True])
def test_invalid_confidence_rejected_even_at_boundary_counts(bad):
    with pytest.raises(ValueError): clopper_pearson(0, 10, bad)
    with pytest.raises(ValueError): one_sided_upper(10, 10, bad)
    with pytest.raises(ValueError): zero_event_upper_bound(10, bad)

@pytest.mark.parametrize('v,u,n', [(2,-1,10), (2,9,10), (1,1,1), (1,0,0), (1,0,10.5)])
def test_rate_block_requires_disjoint_valid_counts(v,u,n):
    with pytest.raises(ValueError): rate_block(v,u,n)

@pytest.mark.parametrize('p', [-0.1,1.2,float('nan'),float('inf'),True])
def test_invalid_probability_is_not_zero_uncertainty(p):
    with pytest.raises(ValueError): mc_standard_error(p,10)

@pytest.mark.parametrize('k,n', [(0,10),(3,10),(10,10),(31,1000)])
def test_valid_intervals_agree_with_independent_scipy_api(k,n):
    expected=binomtest(k,n).proportion_ci(0.95,method='exact')
    assert clopper_pearson(k,n)==pytest.approx((expected.low,expected.high),abs=2e-11)
    assert clopper_pearson(np.int64(k),np.int64(n))==clopper_pearson(k,n)


def test_zero_event_large_n_does_not_cancel_to_zero():
    n=10**18
    assert zero_event_upper_bound(n)>0
    assert zero_event_upper_bound(n)==pytest.approx(-math.log(0.05)/n,rel=1e-12)

@pytest.mark.parametrize('reader,text', [
 ('read_by_label','label,value\na,1\na,2\n'),
 ('read_residuals','label,constraint_id,value\na,c1,1\na,c1,2\n'),
 ('read_rations','label,status,ingredient_id,q_as_fed_kg_per_head_d,d_hat\na,optimal,x,1,0.5\na,optimal,x,2,0.5\n'),
])
def test_duplicate_saved_keys_are_not_silently_overwritten(replay,tmp_path,reader,text):
    p=tmp_path/'synthetic.csv';p.write_text(text)
    with pytest.raises(ValueError,match='duplicate'):getattr(replay,reader)(p)

@pytest.mark.parametrize('q,dh,status2', [('nan','0.5','optimal'),('1','0','optimal'),
                                      ('-1','0.5','optimal'),('1','0.5','proven_infeasible')])
def test_invalid_saved_ration_rejected(replay,tmp_path,q,dh,status2):
    p=tmp_path/'synthetic.csv'
    p.write_text('label,status,ingredient_id,q_as_fed_kg_per_head_d,d_hat\na,optimal,x,1,0.5\n'
                 f'a,{status2},y,{q},{dh}\n')
    with pytest.raises(ValueError): replay.read_rations(p)
