from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy.special import betaincinv

BASE=Path(__file__).absolute().parent
R6=BASE.parent/'revision_round3_20261002'
OUT=BASE/'statistics';OUT.mkdir(exist_ok=False)
FAMILIES=['ablation_formal','information_formal','information_reverse_formal','coverage_formal','measurement_formal','continuous_formal']
risks=[];pairs=[];floors=[];inputs=[];cells=0;positions=0;max_error=0.
def bind(p):
    inputs.append(dict(path=str(p.relative_to(R6)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def upper(k,n):return 1. if k==n else float(betaincinv(k+1,n-k,1-5e-6))
def ci(k,n):
    return (0. if k==0 else float(betaincinv(k,n-k+1,1.25e-6)),1. if k==n else float(betaincinv(k+1,n-k,1-1.25e-6)))
for fam in FAMILIES:
    for done in sorted((R6/fam).glob('**/DONE.json')):
        cells+=1;d=done.parent;sp=d/'summary.json';s=json.loads(sp.read_text());bind(done);bind(sp)
        assert hashlib.sha256(sp.read_bytes()).hexdigest()==json.loads(done.read_text())['summary_sha256']
        losses={};identity=s['identity'];meta=d/'restricted/test_states.json';bind(meta)
        state_meta=json.loads(meta.read_text());state_sha=state_meta['sha256'];state=d/'restricted/test_states.npz'
        assert hashlib.sha256(state.read_bytes()).hexdigest()==state_sha;bind(state)
        for item in s['policies']:
            name=item['policy'];p=d/'restricted'/f'{name}_assigned_outcomes.npz';bind(p)
            with np.load(p,allow_pickle=False) as z:
                v=z['row_violated'].any(axis=1);u=(~v)&z['row_unknown'].any(axis=1);loss=v|u
                assert np.array_equal(v,z['known_failure']) and np.array_equal(u,z['unknown']) and np.array_equal(loss,z['failure'])
                n=len(loss);k=int(loss.sum());unk=int(u.sum());known=int(v.sum());positions+=n
                assert n==item['n_states'] and k==item['failure_count'] and unk==item['unknown_count'] and known==item['known_failure_count']
                up=upper(k,n);err=abs(up-item['adjusted_upper']);max_error=max(max_error,err);assert err<3e-12
                risks.append(dict(family=fam,cell=identity,policy_raw_R6=name,n=n,k=k,known_violation=known,event_unknown=unk,rate=k/n,upper_confidence_bound=up,upper_budget=5e-6,passes_01=up<=.01,passes_02=up<=.02,passes_05=up<=.05,source=str(p.relative_to(R6)),test_state_sha256=state_sha,status='PASS'))
                losses[name]=loss
        for item in s['paired']:
            if item.get('status')=='NOT_AVAILABLE':
                pairs.append(dict(family=fam,cell=identity,policy_raw_R6=item['policy'],comparator_raw_R6=item['comparator'],status='NOT_AVAILABLE_PRESERVED'));continue
            a,b=losses[item['policy']],losses[item['comparator']];n=len(a);x=int((a&~b).sum());y=int((~a&b).sum());lo,hi=ci(x,n);bl,bh=ci(y,n);interval=np.asarray([lo-bh,hi-bl])
            assert x==item['recourse_only_failure'] and y==item['constant_only_failure']
            err=float(np.max(abs(interval-item['conservative_delta_ci'])));max_error=max(max_error,err);assert err<3e-12
            pairs.append(dict(family=fam,cell=identity,policy_raw_R6=item['policy'],comparator_raw_R6=item['comparator'],n=n,n10=x,n01=y,risk_difference=(x-y)/n,difference_pp=100*(x-y)/n,ci_low=float(interval[0]),ci_high=float(interval[1]),ci_unit='probability',each_tail_budget=1.25e-6,source=str(sp.relative_to(R6)),test_state_sha256=state_sha,status='PASS'))
        if fam=='ablation_formal':
            op=d/'restricted/test_oracle_witnesses.npz';bind(op)
            with np.load(op,allow_pickle=False) as z:
                for it in s['finite_menu_coverage']:
                    flag=z[it['menu']+'_all_fail'];n=len(flag);k=int(flag.sum());lb=0. if k==0 else float(betaincinv(k,n-k+1,5e-6))
                    assert k==it['all_fail_count'] and abs(lb-it['adjusted_lower'])<3e-12
                    floors.append(dict(family=fam,cell=identity,menu=it['menu'],n=n,k=k,lower_confidence_bound=lb,source=str(op.relative_to(R6)),status='PASS',scope='Frozen finite menu only; saved all-fail flag arithmetic, not new independent canonical candidate reconstruction'))
        elif fam=='continuous_formal':
            op=d/'restricted/new_menu_candidate_flags_bitpacked.npz';bind(op)
            with np.load(op,allow_pickle=False) as z:
                it=s['finite_menu_floor'];n=it['n_states'];flag=(np.unpackbits(z['violation'],axis=0,count=n,bitorder='big')|np.unpackbits(z['unknown'],axis=0,count=n,bitorder='big')).all(axis=1);k=int(flag.sum());lb=0. if k==0 else float(betaincinv(k,n-k+1,5e-6))
                assert k==it['all_fail_count'] and abs(lb-it['adjusted_lower'])<3e-12
                floors.append(dict(family=fam,cell=identity,menu=it['menu'],n=n,k=k,lower_confidence_bound=lb,source=str(op.relative_to(R6)),status='PASS',scope='Frozen finite menu only; saved candidate flag reconstruction, not new global continuous certificate'))
assert cells==78 and len(risks)==498 and len(pairs)==576 and len(floors)==96 and positions==9960000
write('risk_recheck.csv',risks);write('paired_recheck.csv',pairs);write('floor_recheck.csv',floors)
write('claim_to_record_map.csv',[dict(family=f,formal_cells=sum(1 for p in (R6/f).glob('**/DONE.json')),risk_records=sum(r['family']==f for r in risks),paired_records=sum(r['family']==f for r in pairs),floor_records=sum(r['family']==f for r in floors),result_scope='Specific frozen protocol; no cross-protocol paired comparisons or pooled training replicates') for f in FAMILIES])
(OUT/'read_inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
r=dict(status='PASS',cells=cells,risk_records=len(risks),registered_pairs=len(pairs),fixed_menu_floors=len(floors),saved_row_event_positions=positions,max_beta_interval_error=max_error,scientific_new_draws=0,source_arrays_unchanged=True,scope='Fresh R7 saved nine-row event-mask reconstruction and independently coded beta/four-tail arithmetic. Uses saved row flags, not a new independent nutritional equation replay or independent random-stream/training/selector recreation.',exploratory_continuous_pairs='Existing six exploratory comparisons separate, not part of576; historical R6 audit retained.',overall_joint_confidence_claim=False)
(OUT/'receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False))
