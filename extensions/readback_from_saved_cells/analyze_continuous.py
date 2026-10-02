#!/usr/bin/env python3
"""Read-only formal continuous repair summary plus separated exploratory pair.
Safe aggregate outputs omit q, needs, private random roots and feed prices.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import beta
H=Path(__file__).absolute().parent;P=H.parent.parent;F=P/'continuous_formal';PROTO=P/'continuous_repair/protocol_private.json'
CASE={'dev_case_v1':'R0','dev_case_v3b':'B'};G=1/200000;GP=1/800000
POLICIES=['oldQ3_delta','legaloldQ3_delta','new_trainingbest_static','new_delta']
EXPECTED=[('new_trainingbest_static','oldQ3_delta'),('new_delta','oldQ3_delta'),('new_delta','new_trainingbest_static'),('legaloldQ3_delta','oldQ3_delta'),('new_delta','legaloldQ3_delta')]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def csvwrite(name,rows):
 with (H/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(k,n):return (0. if k==0 else float(beta.ppf(GP,k,n-k+1)),1. if k==n else float(beta.ppf(1-GP,k+1,n-k)))
def pair(a,b):
 n=len(a);x=int((a&~b).sum());y=int((~a&b).sum());lo,hi=ci(x,n);bl,bh=ci(y,n);l,h=lo-bh,hi-bl
 return dict(n=n,policy_only_failure=x,comparator_only_failure=y,difference=(x-y)/n,delta_pp=100*(x-y)/n,adjusted_lower=l,adjusted_upper=h,adjusted_lower_pp=100*l,adjusted_upper_pp=100*h,direction='improved' if h<0 else ('worse' if l>0 else 'unresolved'))
def main():
 cfg=read(PROTO);assert list(cfg['policies'])==POLICIES and list(map(tuple,cfg['paired_comparisons']))==EXPECTED;ph=sha(PROTO);rates=[];pairs=[];explor=[];floors=[];domain=[];bindings=[];maxerr=0.
 for dp in sorted(F.glob('**/DONE.json')):
  c=dp.parent;assert sha(c/'summary.json')==read(dp)['summary_sha256'];s=read(c/'summary.json');assert s['status']=='complete' and s['test_states']==20000;frozen=c/'restricted/frozen_repair_policies.json';assert sha(frozen)==read(c/'POLICIES_FROZEN.json')['sha256'];f=read(frozen);parent=f['parent'];case=CASE[s['case']];root=s['training_root_index'];losses={};actions={};safe=dict(case=case,training_root_index=root,world='TAB',source_structure='C0')
  assert read(c/'manifest_private.json')['protocol_sha256']==ph
  for r in s['policies']:
   z=np.load(c/'restricted'/(r['policy']+'_assigned_outcomes.npz'),allow_pickle=False);v=z['row_violated'].any(axis=1);u=(~v)&z['row_unknown'].any(axis=1);l=v|u;n=len(l);k=int(l.sum());up=1. if k==n else float(beta.ppf(1-G,k+1,n-k));assert np.array_equal(v,z['known_failure']) and np.array_equal(u,z['unknown']) and np.array_equal(l,z['failure']);assert k==r['failure_count'] and int(u.sum())==r['unknown_count'];err=abs(up-r['adjusted_upper']);assert err<3e-12;maxerr=max(maxerr,err);losses[r['policy']]=l;actions[r['policy']]=z['action_id'];rates.append(dict(**safe,policy=r['policy'],n=n,failure_count=k,known_failure_count=int(v.sum()),unknown_count=int(u.sum()),failure_rate=k/n,failure_pct=100*k/n,adjusted_one_sided_upper=up,adjusted_upper_pct=100*up,qualified_targets=';'.join(map(str,r['qualified_targets'])),analysis_status='registered_formal',confidence_family='continuous_rate_alpha0p05_bound10000'))
  for r in s['paired']:
   z=pair(losses[r['policy']],losses[r['comparator']]);assert z['policy_only_failure']==r['recourse_only_failure'] and z['comparator_only_failure']==r['constant_only_failure'];err=max(abs(z['adjusted_lower']-r['conservative_delta_ci'][0]),abs(z['adjusted_upper']-r['conservative_delta_ci'][1]));assert err<3e-12;maxerr=max(maxerr,err);pairs.append(dict(**safe,policy=r['policy'],comparator=r['comparator'],**z,analysis_status='registered_formal',confidence_family='continuous_paired_alpha0p05_bound10000'))
  explor.append(dict(**safe,policy='new_trainingbest_static',comparator='legaloldQ3_delta',**pair(losses['new_trainingbest_static'],losses['legaloldQ3_delta']),analysis_status='posthoc_saved_state_exploratory',confidence_family='separate_exploratory_paired_alpha0p05_bound10000_all72_possible_ordered_pairs'))
  fl=s['finite_menu_floor'];fp=c/'restricted/new_menu_candidate_flags_bitpacked.npz';z=np.load(fp,allow_pickle=False);n=int(z['n_states']);v=np.unpackbits(z['violation'],axis=0)[:n].astype(bool);u=np.unpackbits(z['unknown'],axis=0)[:n].astype(bool);af=(v|u).all(axis=1);k=int(af.sum());lower=0. if k==0 else float(beta.ppf(G,k,n-k+1));assert k==fl['all_fail_count'];assert abs(lower-fl['adjusted_lower'])<3e-12;floors.append(dict(**safe,n=n,actions=fl['actions'],all_fail_count=k,all_fail_rate=k/n,adjusted_lower=lower,adjusted_lower_pct=100*lower,excluded_targets=';'.join(map(str,fl['excluded_targets'])),scope='only_this_frozen_training_generated_menu_under_this_model',analysis_status='registered_formal',confidence_family='continuous_floor_alpha0p05_bound10000'))
  oldn=len(parent['old_q']);legaln=len(f['legalold_indices']);newn=len(parent['new_q']);domain.append(dict(**safe,old_menu_actions=oldn,legalold_menu_actions=legaln,removed_old_actions=oldn-legaln,new_menu_actions=newn,generated_actions_added=newn-legaln,old_vs_legal_action_mismatch=int((actions['oldQ3_delta']!=actions['legaloldQ3_delta']).sum()),old_vs_legal_failure_mismatch=int((losses['oldQ3_delta']!=losses['legaloldQ3_delta']).sum()),interpretation='No old actions removed; no filtering contribution identified' if oldn==legaln else 'Filtering effect assessed by registered legalold vs old pair'))
  bindings.append(dict(cell=s['identity'],summary_sha256=sha(c/'summary.json'),frozen_sha256=sha(frozen),test_npz_sha256=read(c/'restricted/test_states.json')['sha256'],candidate_flags_sha256=sha(fp)))
 assert len(rates)==24 and len(pairs)==30 and len(explor)==6 and len(floors)==6 and len(domain)==6
 for name,rows in [('continuous_24_registered_risks_safe.csv',rates),('continuous_30_registered_pairs_safe.csv',pairs),('continuous_6_exploratory_static_vs_legalold_safe.csv',explor),('continuous_6_fixed_menu_floors_safe.csv',floors),('continuous_6_domain_and_pool_counts_safe.csv',domain)]:csvwrite(name,rows)
 ranges=[]
 for case in ['R0','B']:
  for policy in POLICIES:
   rr=[x for x in rates if x['case']==case and x['policy']==policy];ranges.append(dict(case=case,policy=policy,independent_trained_roots=3,min_failure_rate=min(x['failure_rate'] for x in rr),max_failure_rate=max(x['failure_rate'] for x in rr),min_adjusted_upper=min(x['adjusted_one_sided_upper'] for x in rr),max_adjusted_upper=max(x['adjusted_one_sided_upper'] for x in rr),all_roots_qualify20pct=all(x['adjusted_one_sided_upper']<=.2 for x in rr)))
 csvwrite('continuous_case_policy_ranges_safe.csv',ranges)
 summary=dict(status='PASS',new_draws=0,new_rules_selected_from_test=False,formal_cells=6,registered_risks=24,registered_pairs=30,registered_floors=6,additional_exploratory_pairs=6,exploratory_pairs_counted_as_formal=False,saved_event_reconstructed_policy_positions=480000,max_beta_four_tail_endpoint_difference=maxerr,protocol_sha256=ph,input_bindings=bindings,domain_screening_removed_old_actions=sum(x['removed_old_actions'] for x in domain),old_legal_action_mismatch=sum(x['old_vs_legal_action_mismatch'] for x in domain),old_legal_failure_mismatch=sum(x['old_vs_legal_failure_mismatch'] for x in domain),new_static_vs_old_improved=sum(x['direction']=='improved' for x in pairs if x['policy']=='new_trainingbest_static' and x['comparator']=='oldQ3_delta'),new_delta_vs_old_improved=sum(x['direction']=='improved' for x in pairs if x['policy']=='new_delta' and x['comparator']=='oldQ3_delta'),new_delta_vs_legalold_improved=sum(x['direction']=='improved' for x in pairs if x['policy']=='new_delta' and x['comparator']=='legaloldQ3_delta'),new_delta_vs_static_unresolved=sum(x['direction']=='unresolved' for x in pairs if x['policy']=='new_delta' and x['comparator']=='new_trainingbest_static'),exploratory_static_vs_legalold_improved=sum(x['direction']=='improved' for x in explor),all_frozen_menu_floors_exclude20pct=all(x['adjusted_lower']>.2 for x in floors),all_24_registered_risks_fail20pct=all(x['adjusted_one_sided_upper']>.2 for x in rates),confidence='rate/paired/floor/exploratory families separately allocate alpha .05, bound10000; no combined95 claim; no pooled trained-policy certificate',scope='Training-only local generated finite action menus; no global continuous optimality/impossibility, cross-protocol pairing, actual-intake/full-model/field guarantee.')
 write(H/'continuous_final_readback.json',summary);print(json.dumps({k:summary[k] for k in ['status','formal_cells','registered_risks','registered_pairs','registered_floors','additional_exploratory_pairs','domain_screening_removed_old_actions','new_static_vs_old_improved','new_delta_vs_old_improved','new_delta_vs_static_unresolved','all_frozen_menu_floors_exclude20pct']},ensure_ascii=False))
if __name__=='__main__':main()
