#!/usr/bin/env python3
"""R7.2 descriptive readback of frozen margins/flags; imports no model/evaluator.
Core: unchanged extracted R7 reviewer bundle. No draws, retraining, selection,
new confidence intervals or significance tests. New output directory required.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,platform,time
import numpy as np
CORE_MANIFEST_SHA='65527b679a2158123b95f7f74b0ce983be29bd3f20d3255c54ce7a970e632571'
ID=['family','case','scenario','training_rep_id','policy']; ENERGY='PN-NEL-FIXEDDMI'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rr):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def key(r,fields=ID):return tuple(str(r[k]) for k in fields)
def eqnum(x,y):return abs(float(x)-float(y))<=3e-14+3e-12*abs(float(y))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--core',type=Path,required=True);ap.add_argument('--reference-tail',type=Path,required=True);ap.add_argument('--received-top5',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 core,out=a.core.absolute(),a.output.absolute();assert not out.exists(),'Never overwrite results';assert sha(core/'input_manifest.csv')==CORE_MANIFEST_SHA
 manifest={r['relative_path']:r for r in rows(core/'input_manifest.csv')};bound={};external={};start=time.monotonic()
 def bind(p):
  rel=str(p.relative_to(core));m=manifest[rel];d=sha(p);assert d==m['sha256'] and p.stat().st_size==int(m['size']),rel;bound[rel]=d;return p
 def jread(p):return read(bind(p))
 for p in [a.reference_tail,a.received_top5]:external[str(p.absolute())]=sha(p)
 existing={key(r,ID+['constraint_id']):r for r in rows(a.reference_tail)};attached={key(r,ID+['constraint_id']):r for r in rows(a.received_top5)}
 stats=[];nel=[];shared={};max_ref_error=0.;max_attached_error=0.;cells=0;policy_states=0
 for family in ['coverage_formal','measurement_formal']:
  for done in sorted((core/'study'/family).glob('**/DONE.json')):
   cell=done.parent;s=jread(cell/'summary.json');assert sha(cell/'summary.json')==jread(done)['summary_sha256'];parts=s['identity'].split('__');case={'dev_case_v3a':'A','dev_case_v3c':'C'}[parts[0]];rep=int(parts[3].replace('trainr',''))
   frozen_path=next((cell/'restricted').glob('frozen*.json'));f=jread(frozen_path);assert sha(frozen_path)==jread(cell/'POLICIES_FROZEN.json')['sha256']
   parent_path=core/'study/ablation_formal'/parts[0]/'SD-H0/C0'/('root'+str(rep))/'restricted/frozen_policies.json';parent=jread(parent_path)
   assert sha(parent_path)==f['parent_binding']['sha256']['restricted/frozen_policies.json']
   specs=parent['main_event_row_specs'];assert len(specs)==9 and specs[0]['constraint_id']==ENERGY
   if family=='measurement_formal':assert f['parent_policy']['main_event_row_specs']==specs
   for item in s['policies']:
    ident=dict(family=family,case=case,scenario=parts[2],training_rep_id=rep,policy=item['policy']);p=bind(cell/'restricted'/(item['policy']+'_assigned_outcomes.npz'))
    with np.load(p,allow_pickle=False) as z:
     m=z['margin'];u=z['row_unknown'];v=z['row_violated'];dm=z['supplied_dm_kg_d'];loss=z['failure'];known=z['known_failure'];unknown=z['unknown']
     assert m.shape==(20000,9) and np.array_equal(np.isfinite(m),~u);assert np.array_equal(v.any(1),known);assert np.array_equal((~known)&u.any(1),unknown);assert np.array_equal(known|unknown,loss);assert int(loss.sum())==item['failure_count'];policy_states+=len(loss)
     for j,spec in enumerate(specs):
      cid=spec['constraint_id'];defined=~u[:,j];tol=float(spec['tolerance_declared_unit']);assert np.array_equal(defined&(-m[:,j]>tol),v[:,j]);raw=np.maximum(0.,-m[:,j]);denom=np.full(len(dm),abs(float(spec['threshold_declared_unit'])))
      if cid in ['PN-CA-ABS','PN-P-ABS']:denom+=dm*(.9 if cid=='PN-CA-ABS' else 1.)
      d=raw[defined]/denom[defined];viol=d[v[defined,j]];nd=len(d);nv=len(viol);npos=int((d>0).sum());k=math.ceil(.05*nd);mean=float(d.mean());top=float(np.sort(d)[-k:].mean());scaled=nd/k*mean
      assert np.all(denom>0) and npos<=k
      rec=dict(**ident,constraint_id=cid,n_total=len(m),n_defined=nd,n_violation=nv,n_unknown=int(u[:,j].sum()),positive_deficit_count=npos,positive_frequency_defined=npos/nd,violation_frequency_defined=nv/nd,violation_frequency_total=nv/len(m),positive_nonviolating_within_tolerance=npos-nv,P95=float(np.quantile(d,.95,method='linear')),conditional_mean=mean,top_count=k,top_ceil5pct_mean=top,mean_scaled_to_top5=scaled,top5_mean_identity_error=abs(top-scaled),violation_conditional_mean=float(viol.mean()) if nv else '',violation_conditional_P95=float(np.quantile(viol,.95,method='linear')) if nv else '',violation_conditional_max=float(viol.max()) if nv else '',maximum=float(d.max()),positive_conditional_mean=float(d[d>0].mean()) if npos else '',positive_mean_identity_error=abs(mean-(npos/nd)*(float(d[d>0].mean()) if npos else 0.)),violation_mean_identity_remainder=float(d[(d>0)&(~v[defined,j])].sum()/nd))
      ref=existing[key(rec,ID+['constraint_id'])];refcols={'n_total':'n_total','n_defined':'n_defined','n_violation':'n_violation','n_unknown':'n_unknown','P95':'P95','conditional_mean':'conditional_mean','top_count':'top_count','top_ceil5pct_mean':'top_ceil5pct_mean','violation_conditional_mean':'violated_conditional_mean','violation_conditional_P95':'violated_conditional_P95','maximum':'maximum'}
      for c,rc in refcols.items():
       if rec[c]=='':assert ref[rc]==''
       else:assert eqnum(rec[c],ref[rc]),(ident,c,rec[c],ref[rc]);max_ref_error=max(max_ref_error,abs(float(rec[c])-float(ref[rc])))
      ar=attached[key(rec,ID+['constraint_id'])]
      for c in ar:
       if c in ID+['constraint_id']:continue
       assert eqnum(rec[c],ar[c]),(ident,c,rec[c],ar[c]);max_attached_error=max(max_attached_error,abs(float(rec[c])-float(ar[c])))
      assert rec['P95']==0 and rec['positive_nonviolating_within_tolerance']>=0;stats.append(rec)
      if cid==ENERGY:
       nr=dict(**rec,joint_failure_count=int(loss.sum()),joint_failure_rate_total=float(loss.mean()));nel.append(nr);shared[key(ident)]=(raw.copy()/denom,defined.copy(),v[:,j].copy())
   cells+=1
 assert cells==24 and len(stats)==864 and len(nel)==96 and policy_states==1920000 and len(existing)==len(attached)==864
 look={key(r):r for r in nel};comparisons=[];common=[]
 for fam,case,scenario,rep in sorted({key(r)[:4] for r in nel}):
  pairs=[('Q2_safe_delta','Q3_delta')] if fam=='coverage_formal' else [('Q3_postUA_0p5','Q3_raw_noise_0p5')]+[(f'Q2_postUA_{noise}',f'Q3_postUA_{noise}') for noise in ['0p1','0p5']]+[('Q2_ideal_delta','Q3_ideal_delta')]
  for pol,comp in pairs:
   aa,bb=[look[(fam,case,scenario,rep,p)] for p in [pol,comp]];r=dict(family=fam,case=case,scenario=scenario,training_rep_id=rep,policy=pol,comparator=comp)
   for metric in ['n_defined','n_unknown','n_violation','positive_deficit_count','positive_frequency_defined','violation_frequency_total','joint_failure_rate_total','conditional_mean','top_ceil5pct_mean','violation_conditional_mean','violation_conditional_P95','violation_conditional_max']:
    r['policy_'+metric]=aa[metric];r['comparator_'+metric]=bb[metric];r['delta_'+metric]=float(aa[metric])-float(bb[metric]);
   comparisons.append(r)
   ad,ak,av=shared[key(aa)];bd,bk,bv=shared[key(bb)];common_defined=ak&bk;both_violate=common_defined&av&bv
   common.append(dict(family=fam,case=case,scenario=scenario,training_rep_id=rep,policy=pol,comparator=comp,common_defined=int(common_defined.sum()),policy_only_defined=int((ak&~bk).sum()),comparator_only_defined=int((bk&~ak).sum()),both_violation_count=int(both_violate.sum()),policy_only_violation_count=int((common_defined&av&~bv).sum()),comparator_only_violation_count=int((common_defined&~av&bv).sum()),common_defined_policy_mean=float(ad[common_defined].mean()),common_defined_comparator_mean=float(bd[common_defined].mean()),common_defined_policy_violation_rate=float(av[common_defined].mean()),common_defined_comparator_violation_rate=float(bv[common_defined].mean())))
 compression=[];candidate_positions=0
 for case,code in [('A','dev_case_v3a'),('C','dev_case_v3c')]:
  for rep in range(3):
   cell=core/'study/ablation_formal'/code/'SD-H0/C0'/('root'+str(rep));s=jread(cell/'summary.json');assert sha(cell/'summary.json')==jread(cell/'DONE.json')['summary_sha256'];f=jread(cell/'restricted/frozen_policies.json');assert sha(cell/'restricted/frozen_policies.json')==jread(cell/'POLICIES_FROZEN.json')['sha256'];assert set(f['menus']['Q3']).issubset(f['menus']['Q2'])
   with np.load(bind(cell/'restricted/test_candidate_flags_bitpacked.npz'),allow_pickle=False) as z:
    n=int(z['test_states']);candidate_failure=np.unpackbits(z['violation'],axis=0,count=n,bitorder='big').astype(bool)|np.unpackbits(z['unknown'],axis=0,count=n,bitorder='big').astype(bool);assert candidate_failure.shape==(n,int(z['candidate_actions']));candidate_positions+=candidate_failure.size
   with np.load(bind(cell/'restricted/test_oracle_witnesses.npz'),allow_pickle=False) as z:
    rr={}
    for menu in ['Q2','Q3']:
     pol=menu+'_delta';item=next(it for it in s['policies'] if it['policy']==pol);floor=candidate_failure[:,f['menus'][menu]].all(axis=1);assert np.array_equal(floor,z[menu+'_all_fail'])
     with np.load(bind(cell/'restricted'/(pol+'_assigned_outcomes.npz')),allow_pickle=False) as a0:
      failure=a0['failure'];assert np.all(np.isin(a0['action_id'],f['menus'][menu]));assert np.array_equal(failure,candidate_failure[np.arange(n),a0['action_id']]);assert np.all(~floor|failure)
      nf=int(failure.sum());na=int(floor.sum());ne=int((failure&~floor).sum());assert nf==na+ne;assert nf==item['failure_count'] and na==item['full_information_all_fail_count'] and ne==item['selection_mistake_count'];rr[menu]=(nf,na,ne)
   row=dict(case=case,training_rep_id=rep,n_total=n,Q2_actions=len(f['menus']['Q2']),Q3_actions=len(f['menus']['Q3']))
   for j,name in enumerate(['total_failures','full_information_all_fail','oracle_excess_failures']):
    row['Q2_'+name]=rr['Q2'][j];row['Q3_'+name]=rr['Q3'][j];row['Q3_minus_Q2_'+name]=rr['Q3'][j]-rr['Q2'][j]
   assert row['Q3_minus_Q2_full_information_all_fail']>=0;compression.append(row)
 out.mkdir(parents=True);high=[r for r in comparisons if r['policy']=='Q3_postUA_0p5'];assert len(high)==6
 datasets={'top5_identity_and_occurrence_864.csv':stats,'NEL_frequency_conditional_severity_96.csv':nel,'high_noise_energy_frequency_and_conditional_severity_6.csv':high,'fullQ2_vs_Q3_energy_frequency_and_severity_36.csv':[r for r in comparisons if r['policy']!='Q3_postUA_0p5'],'common_defined_comparisons_42.csv':common,'compression_full_information_decomposition_6.csv':compression}
 for name,rr in datasets.items():write(out/name,rr)
 assert len(datasets['fullQ2_vs_Q3_energy_frequency_and_severity_36.csv'])==36 and len(common)==42
 for rel,d in bound.items():assert sha(core/rel)==d
 for p,d in external.items():assert sha(Path(p))==d
 summary=dict(status='PASS',scope='Independent descriptive readback of saved margins and candidate flags, not nutritional-algebra or stochastic regeneration',core_manifest_sha256=CORE_MANIFEST_SHA,cells=cells,policy_cell_records=len(nel),policy_state_positions=policy_states,normalized_records=len(stats),positive_deficit_fraction_NEL_min=min(r['positive_frequency_defined'] for r in nel),positive_deficit_fraction_NEL_max=max(r['positive_frequency_defined'] for r in nel),positive_deficit_fraction_all_min=min(r['positive_frequency_defined'] for r in stats),positive_deficit_fraction_all_max=max(r['positive_frequency_defined'] for r in stats),all_defined_P95_zero=all(r['P95']==0 for r in stats),max_top5_identity_error=max(r['top5_mean_identity_error'] for r in stats),max_positive_mean_identity_error=max(r['positive_mean_identity_error'] for r in stats),positive_within_tolerance_count=sum(r['positive_nonviolating_within_tolerance'] for r in stats),positive_within_tolerance_records=sum(r['positive_nonviolating_within_tolerance']>0 for r in stats),max_existing_tail_error=max_ref_error,max_received_table_error=max_attached_error,high_noise_directions={m:{'lower':sum(r['delta_'+m]<0 for r in high),'equal':sum(r['delta_'+m]==0 for r in high),'higher':sum(r['delta_'+m]>0 for r in high)} for m in ['joint_failure_rate_total','violation_frequency_total','conditional_mean','top_ceil5pct_mean','violation_conditional_mean','violation_conditional_P95','violation_conditional_max']},compression_saved_candidate_positions=candidate_positions,compression_oracle_excess_increased=sum(r['Q3_minus_Q2_oracle_excess_failures']>0 for r in compression),compression_oracle_excess_decreased=sum(r['Q3_minus_Q2_oracle_excess_failures']<0 for r in compression),bound_input_count=len(bound),new_draws=0,new_model_evaluations=0,new_training=0,new_policy_selections=0,new_inferential_tests=0,elapsed_seconds=time.monotonic()-start,python=platform.python_version(),numpy=np.__version__,outputs={n:dict(rows=len(rr),sha256=sha(out/n)) for n,rr in datasets.items()},input_sha256=bound,external_input_sha256=external,script_sha256=sha(Path(__file__).absolute()))
 (out/'descriptive_audit_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','external_input_sha256','outputs']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
