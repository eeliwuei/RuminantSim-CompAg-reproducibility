#!/usr/bin/env python3
"""Read-only full-table aggregation; no draw, selection, training or pooling."""
import argparse,csv,hashlib,json,collections
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def tables(root,protocol,prefix,out):
    cfg=read(protocol);rows=[];pairs=[];clipping=[];diagnostics=[];bindings=[]
    for identity,cell in cfg['cells'].items():
        directory=root/cell['case']/'SD-H0'/cell['target_scenario']/f"trainroot{cell['training_root_index']}"
        sp=directory/'summary.json';done=read(directory/'DONE.json');s=read(sp);m=read(directory/'manifest_private.json')
        assert done['status']=='complete' and done['summary_sha256']==sha(sp)
        assert not s['qualification'] and m['protocol_sha256']==sha(protocol) and s['identity']==identity
        assert m['code_sha256']==cfg['code_sha256'] and m['parent_binding']==cfg['parent_bindings'][cell['parent_key']]
        bindings.append(dict(identity=identity,summary_sha256=sha(sp),manifest_sha256=sha(directory/'manifest_private.json')))
        common=dict(identity=identity,case=cell['case'],target=cell['target_scenario'],training_root_index=cell['training_root_index'])
        for rate in s['policies']:
            r=dict(common,**{k:v for k,v in rate.items() if k!='mean_feed_cost'})
            r['qualified_targets']='|'.join(map(str,rate['qualified_targets']));rows.append(r)
        for pair in s['paired']:
            lo,hi=pair['conservative_delta_ci']
            pairs.append(dict(common,**{k:v for k,v in pair.items() if k!='conservative_delta_ci'},
                adjusted_difference_lower=lo,adjusted_difference_upper=hi,
                adjusted_direction='improved' if hi<0 else 'worse' if lo>0 else 'includes_zero'))
        for noise in s.get('noise_audit',[]):clipping.append(dict(common,**noise))
        for diag in s.get('diagnostic',{}).get('finite_node_disagreements',[]):diagnostics.append(dict(common,**diag))
    expected_rates=len(cfg['cells'])*len(cfg['policies']);expected_pairs=len(cfg['cells'])*len(cfg['paired_comparisons'])
    assert len(rows)==expected_rates and len(pairs)==expected_pairs
    write_csv(out/(prefix+'_rates.csv'),rows);write_csv(out/(prefix+'_paired.csv'),pairs)
    write_csv(out/(prefix+'_noise_clipping.csv'),clipping);write_csv(out/(prefix+'_finite_node_diagnostics.csv'),diagnostics)
    groups=[]
    for key in sorted(set((r['case'],r['target'],r['policy']) for r in rows)):
        values=[r for r in rows if (r['case'],r['target'],r['policy'])==key]
        group=dict(case=key[0],target=key[1],policy=key[2],independent_trained_rules=len(values),
            failure_rate_min=min(r['failure_rate'] for r in values),failure_rate_max=max(r['failure_rate'] for r in values),
            adjusted_upper_min=min(r['adjusted_upper'] for r in values),adjusted_upper_max=max(r['adjusted_upper'] for r in values),
            unknown_count_min=min(r['unknown_count'] for r in values),unknown_count_max=max(r['unknown_count'] for r in values),
            interpretation='range over three separately frozen trained rules; no pooled risk certificate')
        for target in cfg['alpha_targets']:
            group[f'qualified_at_{target}']=sum(r['adjusted_upper']<=target for r in values)
        groups.append(group)
    write_csv(out/(prefix+'_case_target_policy.csv'),groups)
    pairgroups=[]
    for key in sorted(set((r['policy'],r['comparator']) for r in pairs)):
        values=[r for r in pairs if (r['policy'],r['comparator'])==key]
        pairgroups.append(dict(policy=key[0],comparator=key[1],cells=len(values),
            improved=sum(r['adjusted_direction']=='improved' for r in values),worse=sum(r['adjusted_direction']=='worse' for r in values),
            includes_zero=sum(r['adjusted_direction']=='includes_zero' for r in values),
            observed_difference_min=min(r['delta_recourse_minus_constant'] for r in values),observed_difference_max=max(r['delta_recourse_minus_constant'] for r in values)))
    write_csv(out/(prefix+'_paired_family_counts.csv'),pairgroups)
    transfer=[]
    reference_target='C0' if prefix=='information' else 'C2_Corn4D_author'
    for r in rows:
        if r['target']==reference_target:continue
        ref=next(v for v in rows if v['target']==reference_target and (v['case'],v['training_root_index'],v['policy'])==(r['case'],r['training_root_index'],r['policy']))
        transfer.append(dict(case=r['case'],training_root_index=r['training_root_index'],policy=r['policy'],
            reference_target=reference_target,target=r['target'],reference_failure_rate=ref['failure_rate'],target_failure_rate=r['failure_rate'],
            descriptive_target_minus_reference=r['failure_rate']-ref['failure_rate'],cross_target_paired=False,
            cross_target_significance_claim=False,interpretation='descriptive change across independent new target tests of the same frozen rule'))
    write_csv(out/(prefix+'_transfer_descriptive.csv'),transfer)
    return dict(protocol_sha256=sha(protocol),cells=len(cfg['cells']),rates=expected_rates,pairs=expected_pairs,
        input_summary_bindings=bindings,paired_family_counts=pairgroups,groups=groups,
        transfer_descriptive_only=True,draws_generated=0,new_policy_selected=False,
        audit_scope='saved summary/DONE/manifest/protocol readback and exact aggregation; not independent canonical full-state rerun')


def main():
    p=argparse.ArgumentParser()
    for n in ('information','information-protocol','output'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--reverse',type=Path);p.add_argument('--reverse-protocol',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('fresh aggregation output only')
    a.output.mkdir(parents=True)
    report=dict(status='complete',information=tables(a.information,a.information_protocol,'information',a.output))
    if a.reverse:
        if not a.reverse_protocol:raise ValueError('reverse protocol required')
        report['reverse']=tables(a.reverse,a.reverse_protocol,'reverse',a.output)
    report['analysis_code_sha256']=sha(__file__)
    (a.output/'完整表_汇总收据.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='complete',information_rates=report['information']['rates'],information_pairs=report['information']['pairs'],reverse_rates=report.get('reverse',{}).get('rates',0),reverse_pairs=report.get('reverse',{}).get('pairs',0))))

if __name__=='__main__':main()
