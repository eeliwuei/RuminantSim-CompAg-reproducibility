"""Read-only descriptive NEL denominator and positive-gap identity diagnostic."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
BASE=Path(__file__).absolute().parent
ROOT=BASE.parents[2]
OUT=BASE/'tail_identity_v1'
assert not OUT.exists(), 'Do not overwrite completed diagnostics'
OUT.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
norm=csvread(BASE/'candidate_normalized_latest_severity.csv')
records=csvread(BASE/'saved_risk_label_binding.csv')
lookup={(r['family'],r['case'],r['scenario'],r['training_rep_id'],r['policy']):r for r in norm if r['constraint_id']=='PN-NEL-FIXEDDMI'}
original_files={str((BASE/'TAIL_DIAGNOSTIC_SCOPE.md').relative_to(ROOT)):sha(BASE/'TAIL_DIAGNOSTIC_SCOPE.md')}
for r in records:original_files[r['saved_record_source']]=sha(ROOT/r['saved_record_source'])
(OUT/'input_bindings.json').write_text(json.dumps(original_files,indent=2)+'\n')
arrays={};out=[];errs=[]
for r in records:
 key=tuple(r[k] for k in ['family','case','scenario','training_rep_id','policy']);e=lookup[key]
 with np.load(ROOT/r['saved_record_source'],allow_pickle=False) as z:
  # Event member order is pinned by the original selected-chain spec. Locate
  # energy column through the matching normalized record's row order (9/cell).
  own=[x for x in norm if all(x[k]==r[k] for k in ['family','case','scenario','training_rep_id','policy'])]
  idx=[x['constraint_id'] for x in own].index('PN-NEL-FIXEDDMI')
  margin=np.array(z['margin'][:,idx]);defined=~np.array(z['row_unknown'][:,idx]);violation=np.array(z['row_violated'][:,idx])
  loss=np.array(z['failure'])
 d=np.maximum(0,-margin[defined]);positive=d>0;freq=float(positive.mean());cond=float(d[positive].mean()) if positive.any() else 0.;mean=float(d.mean());product=freq*cond
 err=abs(mean-product);errs.append(err);assert err<1e-12
 arrays[key]=(margin,defined)
 out.append(dict(**{k:r[k] for k in ['family','case','scenario','training_rep_id','policy']},n_total=len(margin),n_defined=int(defined.sum()),row_unknown=int((~defined).sum()),
  energy_violation_count=int(violation.sum()),energy_violation_rate_total=float(violation.mean()),joint_failure_count=int(loss.sum()),joint_failure_rate_total=float(loss.mean()),
  positive_deficit_count=int(positive.sum()),positive_deficit_frequency_defined=freq,defined_mean_deficit_Mcal_d=mean,positive_conditional_mean_deficit_Mcal_d=cond,
  identity_product=product,identity_absolute_error=err,normalization=e['normalization'],normalized_mean=float(e['conditional_mean']),normalized_top_ceil5pct_mean=float(e['top_ceil5pct_mean']),
  scope='NEL positive deficit arithmetic identity and original row/event classifications; no causal or formal tail inference'))
write(OUT/'restricted_internal/NEL_all96_physical_identity.csv',out)
safe=[{k:v for k,v in r.items() if k not in ('defined_mean_deficit_Mcal_d','positive_conditional_mean_deficit_Mcal_d','identity_product')} for r in out]
write(OUT/'NEL_all96_rule_records_safe.csv',safe)
pairs=[]
for fam,case,scenario,rep,_ in sorted(arrays):
 if fam!='measurement_formal':continue
 for noise in ['0p1','0p5']:
  policy='Q3_postUA_'+noise
  for comparator in ['Q3_raw_noise_'+noise,'Q3_postmean_'+noise]:
   k=(fam,case,scenario,rep,policy);b=(fam,case,scenario,rep,comparator)
   ma,da=arrays[k];mb,db=arrays[b];shared=da&db;n=int(shared.sum());count=math.ceil(.05*n)
   ta=float(np.sort(np.maximum(0,-ma[shared]))[-count:].mean());tb=float(np.sort(np.maximum(0,-mb[shared]))[-count:].mean())
   ea,eb=lookup[k],lookup[b]
   pa=dict(family=fam,case=case,scenario=scenario,training_rep_id=rep,policy=policy,comparator=comparator,n_total=len(da),
    policy_defined=int(da.sum()),comparator_defined=int(db.sum()),shared_defined=n,policy_only_defined=int((da&~db).sum()),comparator_only_defined=int((db&~da).sum()),
    full_normalized_tail_policy=float(ea['top_ceil5pct_mean']),full_normalized_tail_comparator=float(eb['top_ceil5pct_mean']),
    common_defined_direction='higher' if ta>tb else 'lower' if ta<tb else 'equal',
    common_defined_tail_ratio=ta/tb if tb else '',original_tail_ratio=float(ea['top_ceil5pct_mean'])/float(eb['top_ceil5pct_mean']),
    scope='same predeclared descriptive comparison on jointly defined positions; original classifications unchanged')
   if pa not in pairs:pairs.append(pa)
assert len(pairs)==24 and all(r['common_defined_direction']=='higher' for r in pairs)
write(OUT/'common_defined_NEL_tail_diagnostic.csv',pairs)
group={}
for r in safe:
 key=(r['family'],r['case'],r['scenario'],r['policy']);group.setdefault(key,[]).append(r)
ranges=[]
for (fam,case,scenario,pol),rr in sorted(group.items()):
 ranges.append(dict(family=fam,case=case,scenario=scenario,policy=pol,n_training_replicates=len(rr),
  min_normalized_top_ceil5pct_mean=min(r['normalized_top_ceil5pct_mean'] for r in rr),max_normalized_top_ceil5pct_mean=max(r['normalized_top_ceil5pct_mean'] for r in rr),
  min_energy_violation_rate=min(r['energy_violation_rate_total'] for r in rr),max_energy_violation_rate=max(r['energy_violation_rate_total'] for r in rr),
  min_n_defined=min(r['n_defined'] for r in rr),max_n_defined=max(r['n_defined'] for r in rr),scope='per-source range, no pooling'))
write(OUT/'NEL_source_policy_ranges_safe.csv',ranges)
for rel,d in original_files.items():assert sha(ROOT/rel)==d
receipt=dict(status='PASS',policy_records=96,positions=1920000,identity_max_absolute_error=max(errs),
 common_defined_pairs=24,all_24_common_defined_NEL_UA_tails_higher=True,
 max_changed_defined_positions_in_comparison=max(r['policy_only_defined']+r['comparator_only_defined'] for r in pairs),
 original_tail_ratio_min=min(r['original_tail_ratio'] for r in pairs),original_tail_ratio_max=max(r['original_tail_ratio'] for r in pairs),
 common_defined_tail_ratio_min=min(r['common_defined_tail_ratio'] for r in pairs),common_defined_tail_ratio_max=max(r['common_defined_tail_ratio'] for r in pairs),
 new_draws=0,new_training=0,formal_inference_added=False,original_input_files_unchanged=True,
 outputs={str(p.relative_to(OUT)):dict(sha256=sha(p),size=p.stat().st_size) for p in OUT.rglob('*') if p.is_file()})
(OUT/'complete_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['status','identity_max_absolute_error','common_defined_pairs','max_changed_defined_positions_in_comparison','original_tail_ratio_min','original_tail_ratio_max','common_defined_tail_ratio_min','common_defined_tail_ratio_max']}))
