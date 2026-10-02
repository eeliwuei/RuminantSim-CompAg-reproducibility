#!/usr/bin/env python3
"""Post-hoc fixed-action fMCP sensitivity, no new states or selected policies.
Use: python reproduce_final_fmcp_sensitivity.py --core EXTRACTED_R7_CORE --output NEW_PATH
The original core is only read. Safe CSVs and private diagnostic evidence are written
separately. CP bounds are post-hoc diagnostics, not new experiment certificates.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,sys,time,traceback
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from scipy.stats import beta
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
ENERGY='PN-NEL-FIXEDDMI'; EVENT='main_reference_plan_domain'
LEVELS=(.8,.9,1.,1.1,1.2); COEF=-.00218724
PROTO_SHA='477b0375c74fca82e7ca0db75d38bb66385eb9a1943e7a663261a3ed09621c27'
FROZEN_SHA='634193881d7c69e5025b2a55d80d05546d10701e170fe6703bab54d94b4b3452'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p,rr):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def deny(*a,**kw):raise AssertionError('New random sampling/training forbidden')
def direction(x):return 'lower' if x<0 else 'higher' if x>0 else 'equal'
def pairs(family):
 if family=='coverage_formal':return [('Q2_safe_delta','Q3_delta')]
 return [('Q2_ideal_delta','Q3_ideal_delta')]+[(p,c) for noise in ('0p1','0p5') for p,c in [(f'Q3_postUA_{noise}',f'Q3_raw_noise_{noise}'),(f'Q3_postUA_{noise}',f'Q3_postmean_{noise}'),(f'Q3_postmean_{noise}',f'Q3_raw_noise_{noise}'),(f'Q2_postUA_{noise}',f'Q3_postUA_{noise}')]]
def metric(v):return float(np.mean(v)) if len(v) else None

def run(core,out):
 assert sha(HERE/'protocol_v2.json')==PROTO_SHA
 assert sha(HERE/'bound_inputs_frozen.json')==FROZEN_SHA
 frozen=read(HERE/'bound_inputs_frozen.json');protocol=read(HERE/'protocol_v2.json')
 def verify_inputs():
  for r in frozen['inputs']:
   p=core/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256'],r['path']
 verify_inputs(); assert not out.exists(),'Never overwrite completed or failed output'
 out.mkdir(parents=True);priv=out/'private';priv.mkdir();started=time.monotonic()
 spec=importlib.util.spec_from_file_location('unchanged_r7_reader',core/'study/reproducibility/replay_new_saved_results.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from ration_reliability.nutrition import energy as E
 coefficient=E._KL*(E._UE_PER_GN*E._N_PER_CP/1000-E._EN_CP)
 assert abs(coefficient-COEF)<1e-17
 allrecords=[];allpairs=[];validation=[];diagnostic=[];comparison_cache={};total_near=0
 cells=sorted({x['cell'] for x in frozen['policy_cells']})
 with ExitStack() as stack:
  stack.enter_context(patch.object(m.UncertaintyModel,'draw',deny));stack.enter_context(patch.object(m.UncertaintyModel,'sample',deny));stack.enter_context(patch.object(np.random,'default_rng',deny))
  for cellname in cells:
   cell=core/cellname;family=cellname.split('/')[1];s=read(cell/'summary.json');assert sha(cell/'summary.json')==read(cell/'DONE.json')['summary_sha256']
   p=cell/'restricted';fp=next(p.glob('frozen*.json'));f=read(fp);assert sha(fp)==read(cell/'POLICIES_FROZEN.json')['sha256'];meta=read(p/'test_states.json');assert sha(p/'test_states.npz')==meta['sha256']
   with np.load(p/'test_states.npz',allow_pickle=False) as z:theta=z['theta'].copy();d=z['d'].copy()
   draw=m.DrawSet(theta,d,meta['stream'],meta['stream_id'],meta['model_id'],meta['model_fingerprint'],tuple(meta['ingredient_ids']),tuple(meta['nutrient_ids']),bool(meta['is_synthetic']));assert draw.fingerprint==meta['fingerprint']
   parts=s['identity'].split('__');ctx=m.common.load_context(parts[0],('SD-H0',));ref=ctx['ref'];assert ref.energy.fmcp_g_per_kg_dmi==16.5
   rids=list(next(e for e in ref.events if e.event_id==EVENT).members);ej=rids.index(ENERGY);assert len(rids)==9
   qowner=f.get('parent_policy',f.get('parent',f));qset=np.asarray(qowner['q']);n=draw.n_draws;assert n==20000
   base=dict(family=family,case={'dev_case_v3a':'A','dev_case_v3c':'C'}[parts[0]],scenario=parts[2],training_rep_id=int(parts[3].replace('trainr','')))
   cellrecords={};cellarrays={}
   for item in s['policies']:
    name=item['policy'];ident=dict(**base,policy=name)
    with np.load(p/(name+'_assigned_outcomes.npz'),allow_pickle=False) as z:
     margin=z['margin'].copy();unknown=z['row_unknown'].copy();violated=z['row_violated'].copy();loss=z['failure'].copy();known=z['known_failure'].copy();eventunknown=z['unknown'].copy();dm=z['supplied_dm_kg_d'].copy();actions=z['action_id'].copy()
    assert margin.shape==(n,9) and np.array_equal(np.isfinite(margin),~unknown)
    assert np.array_equal(known,violated.any(1)) and np.array_equal(eventunknown,(~known)&unknown.any(1)) and np.array_equal(loss,known|eventunknown)
    assert loss.sum()==item['failure_count']
    # Independently recompute actual supplied DM for every saved action/state.
    dm_from_action=np.sum(qset[actions]*d,axis=1)
    assert np.allclose(dm,dm_from_action,rtol=0,atol=3e-13)
    em=margin[:,ej];ed=~unknown[:,ej];ev=violated[:,ej];tol=ref.energy.tolerance_mcal_d
    assert np.array_equal(ev,ed&(-em>tol))
    ov=np.delete(violated,ej,axis=1).any(1);ou=np.delete(unknown,ej,axis=1).any(1)
    R=ref.energy.requirement_mcal_d;assert R>0
    selected=set(np.linspace(0,n-1,33,dtype=int).tolist())|set(np.flatnonzero(~ed).tolist())|{int(np.argmin(dm)),int(np.argmax(dm))}
    leveldata={}
    for mult in LEVELS:
     shifted=em+COEF*dm*(16.5*mult-16.5)
     distance=np.where(ed,np.abs(shifted+tol),np.inf)
     selected.update(np.argsort(distance,kind='stable')[:10].tolist());close=np.flatnonzero(ed&(distance<=1e-10));selected.update(close.tolist());total_near+=len(close)
     ee=ed&(-shifted>tol);vv=ov|ee;uu=(~vv)&(ou|~ed);ll=vv|uu
     if mult==1:
      assert np.array_equal(ll,loss) and np.array_equal(vv,known) and np.array_equal(uu,eventunknown)
     dd=np.maximum(0.,-shifted[ed])/R;positive=dd>0;cond=np.maximum(0.,-shifted[ee])/R;k=math.ceil(.05*len(dd));kk=int(ll.sum());up=1. if kk==n else float(beta.ppf(1-.05/10000,kk+1,n-kk));tail=float(np.sort(dd)[-k:].mean());mean=float(dd.mean());freq=float(positive.mean());condpos=float(dd[positive].mean()) if positive.any() else 0.
     assert abs(mean-freq*condpos)<1e-16
     if mult==1:assert abs(up-item['adjusted_upper'])<3e-12
     record=dict(**ident,fMCP_multiplier=mult,n_total=n,failure_count=kk,failure_rate=kk/n,known_failure_count=int(vv.sum()),event_unknown_count=int(uu.sum()),baseline_failure_count=int(loss.sum()),sensitivity_only_failure_count=int((ll&~loss).sum()),baseline_only_failure_count=int((~ll&loss).sum()),diagnostic_upper=up,original_baseline_upper=item['adjusted_upper'],n_NEL_defined=int(ed.sum()),n_NEL_unknown=int((~ed).sum()),NEL_violation_count=int(ee.sum()),NEL_violation_rate_total=float(ee.mean()),NEL_positive_deficit_count=int(positive.sum()),NEL_positive_frequency_defined=freq,NEL_normalized_mean_defined=mean,NEL_normalized_mean_violations=metric(cond),NEL_normalized_P95_violations=float(np.quantile(cond,.95,method='linear')) if len(cond) else None,NEL_normalized_max_defined=float(dd.max()),NEL_normalized_P95_defined=float(np.quantile(dd,.95,method='linear')),NEL_top_ceil5pct_mean=tail,top_count=k,top_contains_all_positive=len(cond)<=k and int(positive.sum())<=k,top_rescaled_mean_error=abs(tail-len(dd)/k*mean),positive_frequency_times_conditional_mean_error=abs(mean-freq*condpos),scope='posthoc_fixed_action_hypothetical_fMCP_diagnostic_not_new_certificate')
     for target in (.01,.02,.05,.1,.2):
      tag=str(round(100*target)).replace('.','p')+'pct';record['empirical_le_'+tag]=kk/n<=target;record['diagnostic_upper_le_'+tag]=up<=target
     allrecords.append(record);cellrecords[(name,mult)]=record;cellarrays[(name,mult)]=(ll,vv,uu,shifted,ee)
     leveldata[mult]=(ll,vv,uu,shifted,ee)
    selected=np.array(sorted(selected));assert len(selected)>0
    # Canonical evaluation at all levels. No target, linearisation or action changes.
    for mult in LEVELS:
     rr=replace(ref,energy=replace(ref.energy,fmcp_g_per_kg_dmi=16.5*mult));assert rr.energy.requirement_mcal_d==R and rr.energy.tolerance_mcal_d==tol
     ll,vv,uu,shifted,ee=leveldata[mult];maxerr=0.;checked=0;digest=hashlib.sha256()
     for action in np.unique(actions[selected]):
      ix=selected[actions[selected]==action]
      actual=m.evaluate_reference(qset[action],draw.reordered(scenario_order=ix),rr)
      ae=actual.energy
      assert np.array_equal(ae.reference_defined,ed[ix]) and np.array_equal(ae.reference_violated,ee[ix])
      finite=ed[ix];errors=np.abs(ae.reference_margin_mcal_d[finite]-shifted[ix][finite]);maxerr=max(maxerr,float(errors.max(initial=0)))
      assert maxerr<=2e-10,(ident,mult,maxerr)
      assert np.array_equal(actual.event_violation[EVENT],vv[ix]) and np.array_equal(actual.event_unknown[EVENT],uu[ix])
      digest.update(np.ascontiguousarray(ae.reference_margin_mcal_d).tobytes());checked+=len(ix)
     validation.append(dict(**ident,fMCP_multiplier=mult,canonical_checked_policy_state_positions=checked,max_abs_energy_margin_error=maxerr,defined_mismatch=0,violation_mismatch=0,event_unknown_mismatch=0,status='PASS'))
     diagnostic.append(dict(**ident,fMCP_multiplier=mult,selected_index_sha256=ah(selected),selected_indices=selected.tolist(),shifted_margin_sha256=ah(shifted),joint_failure_sha256=ah(ll),canonical_margin_concat_sha256=digest.hexdigest(),actual_dm_max_abs_error=float(np.max(np.abs(dm-dm_from_action)))))
    # Monotonicity applies to fixed actions when fMCP alone rises.
    for lo,hi in zip(LEVELS[:-1],LEVELS[1:]):assert not np.any(leveldata[lo][0]&~leveldata[hi][0])
   for policy,comp in pairs(family):
    for mult in LEVELS:
     a,b=[cellrecords[(x,mult)] for x in (policy,comp)];la,lb=[cellarrays[(x,mult)][0] for x in (policy,comp)];n10=int((la&~lb).sum());n01=int((~la&lb).sum());delta=(n10-n01)/n
     base_delta=cellrecords[(policy,1.)]['failure_rate']-cellrecords[(comp,1.)]['failure_rate']
     pr=dict(**base,policy=policy,comparator=comp,fMCP_multiplier=mult,n_total=n,policy_failure_count=a['failure_count'],comparator_failure_count=b['failure_count'],n10_policy_only=n10,n01_comparator_only=n01,risk_delta=delta,risk_direction=direction(delta),baseline_risk_direction=direction(base_delta),direction_changed_from_baseline=direction(delta)!=direction(base_delta),NEL_violation_rate_delta=a['NEL_violation_rate_total']-b['NEL_violation_rate_total'],NEL_mean_defined_delta=a['NEL_normalized_mean_defined']-b['NEL_normalized_mean_defined'],NEL_mean_violations_delta=(a['NEL_normalized_mean_violations']-b['NEL_normalized_mean_violations']) if a['NEL_normalized_mean_violations'] is not None and b['NEL_normalized_mean_violations'] is not None else None,NEL_P95_violations_delta=(a['NEL_normalized_P95_violations']-b['NEL_normalized_P95_violations']) if a['NEL_normalized_P95_violations'] is not None and b['NEL_normalized_P95_violations'] is not None else None,NEL_max_defined_delta=a['NEL_normalized_max_defined']-b['NEL_normalized_max_defined'],scope='descriptive_same_state_comparison_no_new_significance_test')
     allpairs.append(pr)
   print(json.dumps(dict(identity=s['identity'],family=family,policy_records=len(s['policies'])*5,canonical_positions=sum(x['canonical_checked_policy_state_positions'] for x in validation if x['family']==family and x['case']==base['case'] and x['scenario']==base['scenario'] and x['training_rep_id']==base['training_rep_id']),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
 assert len(allrecords)==480 and len(allpairs)==360 and len(validation)==480
 verify_inputs()
 csvwrite(out/'final_fMCP_sensitivity_480.csv',allrecords);csvwrite(out/'final_fMCP_descriptive_pairs_360.csv',allpairs);csvwrite(priv/'canonical_validation_480.csv',validation);write(priv/'canonical_selection_and_array_hashes.json',diagnostic)
 group={}
 for r in allrecords:group.setdefault((r['family'],r['case'],r['policy'],r['fMCP_multiplier']),[]).append(r)
 ranges=[]
 for k,rr in sorted(group.items()):
  r=dict(family=k[0],case=k[1],policy=k[2],fMCP_multiplier=k[3],n_records=len(rr))
  for key in ('failure_rate','diagnostic_upper','NEL_violation_rate_total','NEL_normalized_mean_defined','NEL_normalized_mean_violations','NEL_normalized_P95_violations','NEL_normalized_max_defined'):
   vals=[x[key] for x in rr if x[key] is not None];r[key+'_min']=min(vals) if vals else None;r[key+'_max']=max(vals) if vals else None
  for target in (1,2,5,10,20):r['diagnostic_upper_le_'+str(target)+'pct_count']=sum(x['diagnostic_upper_le_'+str(target)+'pct'] for x in rr)
  r['scope']='range_across_unpooled_policy_cells';ranges.append(r)
 csvwrite(out/'final_fMCP_descriptive_ranges.csv',ranges)
 groups={}
 for r in allpairs:groups.setdefault((r['family'],r['policy'],r['comparator'],r['fMCP_multiplier']),[]).append(r)
 directions=[dict(family=k[0],policy=k[1],comparator=k[2],fMCP_multiplier=k[3],comparisons=len(rr),lower=sum(x['risk_direction']=='lower' for x in rr),equal=sum(x['risk_direction']=='equal' for x in rr),higher=sum(x['risk_direction']=='higher' for x in rr),direction_changes=sum(x['direction_changed_from_baseline'] for x in rr),min_risk_delta=min(x['risk_delta'] for x in rr),max_risk_delta=max(x['risk_delta'] for x in rr),scope='descriptive_only_no_new_paired_significance') for k,rr in sorted(groups.items())]
 csvwrite(out/'final_fMCP_pair_directions.csv',directions)
 receipt=dict(status='PASS',protocol_sha256=PROTO_SHA,frozen_input_manifest_sha256=FROZEN_SHA,core_manifest_sha256=frozen['core_manifest_sha256'],policy_cells=96,cells=24,original_saved_policy_state_positions=1920000,hypothetical_policy_state_evaluations=9600000,sensitivity_records=480,descriptive_pairs=360,input_files_before_after_unchanged=len(frozen['inputs']),canonical_checked_policy_state_positions=sum(x['canonical_checked_policy_state_positions'] for x in validation),max_abs_canonical_energy_margin_error=max(x['max_abs_energy_margin_error'] for x in validation),canonical_label_and_unknown_mismatches=0,near_threshold_1e_minus10_positions=total_near,baseline_count_upper_and_label_checks=96,new_draws=0,new_training=0,new_policy_selection=0,new_tail_or_paired_tests=0,derivative_coefficient=coefficient,elapsed_seconds=time.monotonic()-started,numpy_version=np.__version__,python_version=sys.version,scope='Posthoc hypothetical fixed-action sensitivity of final saved policies, not full microbial model, measured-range validation, prospective risk certification or source/training reproduction',outputs={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()})
 write(out/'complete_receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','sensitivity_records','descriptive_pairs','canonical_checked_policy_state_positions','max_abs_canonical_energy_margin_error','input_files_before_after_unchanged')}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--core',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.absolute()
 try:run(a.core.absolute(),out)
 except Exception:
  if out.exists():write(out/'failure_receipt.json',dict(error=traceback.format_exc(),new_draws=0,original_files_written=False))
  raise
if __name__=='__main__':main()
