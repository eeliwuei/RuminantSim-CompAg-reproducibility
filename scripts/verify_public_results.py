#!/usr/bin/env python3
"""Read-only arithmetic/manifest audit of an existing development run.

This does NOT recreate rations, draw states or solve any optimisation. Counts in
saved CSVs are the inputs. Restricted inputs are not inferred, downloaded or
reconstructed. The reported coverage distinguishes those missing files from an
arithmetic mismatch. SciPy's binomtest API is used independently of this project's
interval helper. Existing historical results are never overwritten.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN='pilot-20260925T055921Z-fb4f75af'


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


def count(v: str) -> int:
    # These CSV count columns are exported as integer strings, not rounded floats.
    text=str(v).strip()
    if not text.isdecimal(): raise ValueError('invalid nonnegative integer count')
    return int(text)


def finite(v) -> float:
    x=float(v)
    if not math.isfinite(x): raise ValueError('non-finite metric')
    return x


def audit(root: Path, run_id: str) -> dict:
    if Path(run_id).name!=run_id or run_id in ('','.','..'):
        raise ValueError('run_id must be a single directory name')
    root=root.resolve(); directory=root/'results'/'pilot'/run_id
    rec=json.loads((directory/'run_record.json').read_text(encoding='utf-8'))
    checks=[]; missing=[]; inputs=[]
    def check(ok,what,where): checks.append({'ok':bool(ok),'check':what,'location':where})
    for rel,expected in sorted((rec.get('output_hashes') or {}).items()):
        f=(root/rel).resolve()
        if not f.is_relative_to(root):
            check(False,'output path stays inside project',rel); continue
        if not f.is_file():
            missing.append({'path':rel,'expected_sha256':expected,
                            'classification':'restricted_output_unavailable' if rel.startswith('data/restricted_local/')
                                             else 'required_public_output_missing'})
            if not rel.startswith('data/restricted_local/'):check(False,'required public output present',rel)
        else:
            actual=sha256(f);inputs.append({'path':rel,'sha256':actual})
            check(actual==expected,'historical output SHA256 matches run record',rel)
    check(rec.get('run_id')==run_id,'run identity matches requested directory','run_record.json')
    ends=rows(directory/'endpoint_ablation.csv'); residuals=rows(directory/'reference_residuals.csv')
    seen=set(); evaluated=[]
    for i,r in enumerate(ends,2):
        loc=f'endpoint_ablation.csv:{i}'; key=(r['label'],r.get('evaluation_world'))
        check(key not in seen,'unique ration/world row',loc); seen.add(key)
        has=r.get('has_ration')=='True'
        check(r.get('has_ration') in ('True','False'),'explicit ration presence',loc)
        if not has:
            check(not r.get('cost_usd_per_head_d'),'absent ration has no cost, not zero',loc)
            check(not r.get('n_test'),'absent ration has no evaluation sample',loc)
            continue
        evaluated.append(r)
        try:
            n=count(r['n_test']);v=count(r['main_reference_n_violated']);u=count(r['main_reference_n_unknown'])
            check(n>0 and 0<=v<=v+u<=n,'valid disjoint reference counts',loc)
            if n<=0 or v+u>n: continue
            values={
                'main_reference_rate_lower':v/n,
                'main_reference_rate_upper':(v+u)/n,
                'main_reference_mc_se':math.sqrt((v+u)/n*(1-(v+u)/n)/n),
                'main_reference_cp_upper':binomtest(v+u,n,alternative='less').proportion_ci(.95,method='exact').high,
            }
            for field,expected in values.items():
                check(math.isclose(finite(r[field]),expected,rel_tol=2e-10,abs_tol=2e-11),field+' independently recomputed',loc)
            check(0<=finite(r['cost_usd_per_head_d']),'nonnegative finite stored scenario cost',loc)
            for field in ('energy_false_pass','energy_false_fail','energy_reference_undefined'):
                check(count(r[field])<=n,field+' in [0,n]',loc)
            if r.get('target_alpha'):
                alpha=finite(r['target_alpha'])
                expected=finite(r['main_reference_rate_upper'])<=alpha
                check((r['meets_main_reference_alpha_point']=='True')==expected,'point-risk status matches its own definition',loc)
        except (KeyError,ValueError,OverflowError) as e:
            check(False,'numeric/schema error: '+type(e).__name__,loc)
    rseen=set()
    for i,r in enumerate(residuals,2):
        loc=f'reference_residuals.csv:{i}'
        key=(r['label'],r['evaluation_world'],r['constraint_id'],r['verdict'])
        check(key not in rseen,'unique residual row',loc);rseen.add(key)
        try:
            n=count(r['n_states']);v=count(r['n_violated']);u=count(r['n_unknown'])
            check(n>0 and v+u<=n,'valid residual counts',loc)
            if not n or v+u>n:continue
            expected={'rate_lower':v/n,'rate_upper':(v+u)/n,
                      'cp_upper_one_sided':binomtest(v+u,n,alternative='less').proportion_ci(.95,method='exact').high,
                      'mc_se':math.sqrt((v+u)/n*(1-(v+u)/n)/n)}
            for field,x in expected.items():
                check(math.isclose(finite(r[field]),x,rel_tol=2e-10,abs_tol=2e-11),field+' independently recomputed',loc)
            if r.get('n_defined'):check(count(r['n_defined'])+u==n,'defined + unknown = total',loc)
        except (KeyError,ValueError,OverflowError) as e:
            check(False,'numeric/schema error: '+type(e).__name__,loc)
    sets=json.loads((directory/'comparable_sets.json').read_text(encoding='utf-8'))
    set_audits=[]
    for s in sets:
        world=s['evaluation_world'];a=float(s['target_alpha'])
        attempted=[r for r in ends if r['evaluation_world'] in ('',world) and r.get('ration_kind')=='method'
                   and (not r.get('target_alpha') or math.isclose(float(r['target_alpha']),a))]
        members=[r for r in attempted if r.get('has_ration')=='True' and r.get('structural_ok')=='True'
                 and finite(r['main_reference_rate_upper'])<=a]
        loc=f'comparable_sets.json:{world}/alpha={a}'
        check(len(attempted)==s['n_entries_attempted'],'attempted-method denominator',loc)
        check(len(members)==s['n_members'],'same-reference comparable-set size',loc)
        check(math.isclose(len(members)/len(attempted) if attempted else 0.0,s['coverage']),'coverage ratio',loc)
        set_audits.append({'world':world,'alpha':a,'attempted':len(attempted),'members':len(members)})
    failed=[c for c in checks if not c['ok']]
    return {'schema':'ration_reliability.public_result_audit/1','utc':datetime.now(timezone.utc).isoformat(),
            'run_id':run_id,'coverage':'saved-output arithmetic and available hashes only; NOT rerun, raw-data validation or official research',
            'status':'FAIL' if failed else ('PASS_WITH_UNAVAILABLE_RESTRICTED_OUTPUTS' if missing else 'PASS'),
            'n_checks':len(checks),'n_failed':len(failed),'failed_checks':failed,
            'endpoint_rows':len(ends),'evaluated_endpoint_rows':len(evaluated),'residual_rows':len(residuals),
            'available_hash_inputs':inputs,'unavailable_outputs':missing,'comparable_sets':set_audits,
            'all_checks':checks}


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--run-id',default=DEFAULT_RUN)
    ap.add_argument('--output',type=Path,help='new audit JSON; refuses an existing file')
    args=ap.parse_args(argv)
    try:
        result=audit(args.root,args.run_id)
        text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf-8') as f:f.write(text)
        print(json.dumps({k:v for k,v in result.items() if k!='all_checks'},ensure_ascii=False,indent=2,allow_nan=False))
        return 1 if result['n_failed'] else 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'BLOCKED_OR_INVALID','error':type(exc).__name__,'message':str(exc)},ensure_ascii=False))
        return 2

if __name__=='__main__': raise SystemExit(main())
