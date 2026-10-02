"""Independent array/count/readback audit; no draw or original helper calls."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
from scipy.stats import beta

import os
R=Path(os.environ.get('RRS_ARCHIVE_ROOT','.'))  # root of the controlled study archive
O=R/'work/revision_round3_20261002/analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cp(k,n,g):return (0. if k==0 else float(beta.ppf(g,k,n-k+1)),1. if k==n else float(beta.ppf(1-g,k+1,n-k)))
def exact_tail_goal(k,n,a,b):
 # All terms and the significance comparison are integers, not logarithmic approximations.
 term=pow(b-a,n);total=term
 for j in range(k):
  num=term*(n-j)*a;den=(j+1)*(b-a)
  term,rem=divmod(num,den);assert rem==0
  total+=term
 return 200000*total<=pow(b,n)

def main():
 p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');a=p.parse_args()
 folder=R/'work/revision_round3_20261002/ablation_formal'
 protocol=R/'work/revision_round3_20261002/ablation/protocol_private.json';cfg=json.loads(protocol.read_text())
 rate=[];pair=[];oracle=[];slots=[];bindings=[];integer=[];maxerr=0.;states=0
 done=sorted(folder.glob('dev_case_*/*/C0/root*/DONE.json'))
 if a.complete:assert len(done)==len(cfg['cells'])
 for marker in done:
  cell=marker.parent;summary=json.loads((cell/'summary.json').read_text());identity=summary['identity'];case,world,_,root=identity.split('__')
  assert json.loads(marker.read_text())['summary_sha256']==sha(cell/'summary.json')
  manifest=json.loads((cell/'manifest_private.json').read_text());assert manifest['protocol_sha256']==sha(protocol)
  freeze=cell/'restricted/frozen_policies.json';assert json.loads((cell/'POLICIES_FROZEN.json').read_text())=={'sha256':sha(freeze),'test_drawn':False}
  f=json.loads(freeze.read_text());n=summary['test_states'];assert n==cfg['test_states'];loss={};basecost=next(x['mean_feed_cost'] for x in summary['policies'] if x['policy']=='Q0_static')
  common=dict(identity=identity,case=case,world=world,root=root,n_states=n)
  for row in summary['policies']:
   z=np.load(cell/'restricted'/f"{row['policy']}_assigned_outcomes.npz");v=z['known_failure'].astype(bool);u=z['unknown'].astype(bool);l=z['failure'].astype(bool)
   assert np.array_equal(l,v|u) and not (v&u).any()
   assert np.array_equal(v,z['row_violated'].any(axis=1))
   assert np.array_equal(u,(~v)&z['row_unknown'].any(axis=1))
   k=int(l.sum());assert k==row['failure_count'];assert int(u.sum())==row['unknown_count'];assert int(v.sum())==row['known_failure_count'];loss[row['policy']]=l;states+=n
   up=cp(k,n,1/200000)[1];maxerr=max(maxerr,abs(up-row['adjusted_upper']));assert abs(up-row['adjusted_upper'])<1e-10
   assert np.isclose(z['feed_cost'].mean(),row['mean_feed_cost'],rtol=1e-12,atol=1e-12)
   target=';'.join(str(x) for x in cfg['alpha_targets'] if up<=x);assert [x for x in cfg['alpha_targets'] if up<=x]==row['qualified_targets']
   rate.append(dict(**common,policy=row['policy'],failure_count=k,known_failure_count=int(v.sum()),unknown_count=int(u.sum()),failure_rate=k/n,adjusted_upper=up,qualified_targets=target,feed_cost_ratio_to_Q0_static=float(z['feed_cost'].mean()/basecost),full_information_all_fail_count=row.get('full_information_all_fail_count',''),selection_mistake_count=row.get('selection_mistake_count',''),action_count=row.get('action_count',''),chosen_distinct_actions=row.get('chosen_distinct_actions','')))
   if world=='SD-H0' and case in ('dev_case_v3a','dev_case_v3c') and row['policy']=='Q3_delta':
    targetnum,targetden=(1,20) if case=='dev_case_v3a' else (1,50)
    ex=exact_tail_goal(k,n,targetnum,targetden);assert ex==(up<=targetnum/targetden)
    integer.append(dict(**common,policy=row['policy'],target=targetnum/targetden,integer_tail_qualified=ex))
   if world=='SD-H0' and case=='dev_case_v3c' and row['policy']=='Q2_delta':
    ex=exact_tail_goal(k,n,1,100);assert ex==(up<=.01)
    integer.append(dict(**common,policy=row['policy'],target=.01,integer_tail_qualified=ex))
  for row in summary['paired']:
   if row.get('status')=='NOT_AVAILABLE':
    pair.append(dict(**common,policy=row['policy'],comparator=row['comparator'],status='NOT_AVAILABLE',n10='',n01='',difference='',ci_low='',ci_high='',interpretation=row.get('reason','missing frozen recipe')));continue
   x,y=loss[row['policy']],loss[row['comparator']];n10=int((x&~y).sum());n01=int((~x&y).sum());assert n10==row['recourse_only_failure'] and n01==row['constant_only_failure']
   lo,hi=cp(n10,n,1/800000);bl,bh=cp(n01,n,1/800000);ci=(lo-bh,hi-bl);assert np.max(np.abs(np.asarray(ci)-row['conservative_delta_ci']))<1e-10
   pair.append(dict(**common,policy=row['policy'],comparator=row['comparator'],status='PASS',n10=n10,n01=n01,difference=(n10-n01)/n,ci_low=ci[0],ci_high=ci[1],interpretation='improved' if ci[1]<0 else 'worsened' if ci[0]>0 else 'not_resolved'))
  z=np.load(cell/'restricted/test_candidate_flags_bitpacked.npz');v=np.unpackbits(z['violation'],axis=0)[:n].astype(bool);u=np.unpackbits(z['unknown'],axis=0)[:n].astype(bool);fl=v|u
  for row in summary['finite_menu_coverage']:
   name=row['menu'];ix=np.asarray(f['menus'][name],int);allfail=fl[:,ix].all(axis=1);k=int(allfail.sum());assert k==row['all_fail_count'];lower=cp(k,n,1/200000)[0];assert abs(lower-row['adjusted_lower'])<1e-10
   oracle.append(dict(**common,menu=name,actions=len(ix),all_fail_count=k,all_fail_rate=k/n,adjusted_lower=lower))
   for policy in (name+'_static',name+'_delta'):
    mist=loss[policy]&~allfail;r=next(r for r in summary['policies'] if r['policy']==policy);assert int(mist.sum())==r['selection_mistake_count'];assert np.array_equal(loss[policy],allfail|mist)
  for row in summary['reference_slp_slots']:
   slots.append(dict(**common,slot=row.get('policy',row.get('menu','')),status=row.get('status',''),recipe_available=row.get('recipe_available',False),scope='continuous search, menu is initialization only'))
  bindings.append(dict(identity=identity,summary_sha256=sha(cell/'summary.json'),frozen_policy_sha256=sha(freeze),test_npz_sha256=sha(cell/'restricted/test_states.npz')))
 for name,rows in [('attribution_risks.csv',rate),('attribution_pairs.csv',pair),('attribution_fixed_menu_lower.csv',oracle),('attribution_SLP_slots.csv',slots),('attribution_integer_primary_goals.csv',integer)]:
  if rows:csvout(O/name,rows)
 dump(O/'attribution_independent_readback.json',dict(status='PASS_COMPLETE' if a.complete else 'INTERIM_NOT_FINAL',cells=len(done),expected_cells=len(cfg['cells']),rate_rows=len(rate),paired_slots=len(pair),oracle_rows=len(oracle),SLP_slots=len(slots),integer_primary_goals=len(integer),assigned_policy_states_readback=states,max_upper_abs_error=maxerr,protocol_sha256=sha(protocol),bindings=bindings,scope='Independent saved-array/count/decomposition and scipy.stats beta endpoint reconstruction; exact integer tails for six primary Q3_delta C0 goals and three close-boundary Q2_delta C one-percent goals. Does not independently replay every selector or every candidate-chain state.',new_draws=0))
 print(json.dumps(dict(cells=len(done),rates=len(rate),pairs=len(pair),SLP_slots=len(slots),states=states,status='PASS_COMPLETE' if a.complete else 'INTERIM')))

if __name__=='__main__':main()
