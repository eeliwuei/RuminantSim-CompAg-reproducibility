#!/usr/bin/env python3
"""Independent saved-label statistical readback, without scientific imports.

Reconstruct original event from saved canonical row labels, recompute counts
and beta-tail intervals, and preserve every registered rule/comparison.
This is not a full independent nutrition/selector rerun.
"""
import argparse,csv,hashlib,json,collections
from pathlib import Path
import numpy as np
from scipy.stats import beta


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def upper(k,n,tail):return 1. if k==n else float(beta.isf(tail,k+1,n-k))
def cp(k,n,tail):return (0. if k==0 else float(beta.ppf(tail,k,n-k+1)),1. if k==n else float(beta.isf(tail,k+1,n-k)))


def audit_saved_noise(directory,frozen,summary,common,field_rows):
    """Reconstruct observations from existing arrays; never call a generator."""
    parent=Path(frozen['parent_binding']['path']);testpath=directory/'restricted/test_states.npz'
    testmeta=read(testpath.with_suffix('.json'));trainpath=parent/'restricted/training_states.npz';trainmeta=read(trainpath.with_suffix('.json'))
    assert testmeta['sha256']==sha(testpath) and trainmeta['sha256']==sha(trainpath)
    assert testmeta['fingerprint']==summary['test_fingerprint'] and trainmeta['fingerprint']==summary['training_fingerprint']
    fields=[u['field'] for u in frozen['field_units']]
    assert fields==['CP','NDF','starch','EE','ash','Ca','P']
    assert all(u['canonical_unit']=='kg/kg DM' and u['clipping_support']==[0.,1.] and not u['fixed_energy_parameters_observed'] for u in frozen['field_units'])
    ti=[testmeta['nutrient_ids'].index(f) for f in fields];ri=[trainmeta['nutrient_ids'].index(f) for f in fields]
    with np.load(testpath,allow_pickle=False) as a:truth=a['theta'][:,:,ti].copy()
    with np.load(trainpath,allow_pickle=False) as a:training=a['theta'][:,:,ri].copy()
    noisepath=directory/'restricted/independent_nutrient_noise_normals.npz';priorpath=directory/'restricted/frozen_working_prior.npz'
    assert sha(priorpath)==frozen['working_prior_sha256']
    with np.load(noisepath,allow_pickle=False) as a:normal=a['nutrient'].copy()
    with np.load(priorpath,allow_pickle=False) as a:
        sd=a['parent_gaussian_scale'].copy();mu=a['parent_gaussian_location'].copy();count=a['finite_training_counts'].copy();point=a['exact_training_point'].copy()
    assert truth.shape==normal.shape and truth.shape[1:]==sd.shape and np.isfinite(normal).all()
    finite=np.isfinite(truth);assert not np.any(finite&((truth<0)|(truth>1))) and np.all(np.isfinite(sd)) and np.all(sd>=0)
    max_mu_difference=0.;max_sd_difference=0.
    for ix in np.ndindex(sd.shape):
        values=training[(slice(None),)+ix];values=values[np.isfinite(values)]
        assert len(values)==count[ix]
        exact=bool(len(values) and np.all(values==values[0]));assert exact==point[ix]
        expected_mu=values[0] if exact else np.mean(values) if len(values) else np.nan
        expected_sd=0. if len(values)<2 or exact else np.std(values,ddof=1)
        if np.isfinite(expected_mu):max_mu_difference=max(max_mu_difference,abs(float(expected_mu-mu[ix])))
        else:assert np.isnan(mu[ix])
        max_sd_difference=max(max_sd_difference,abs(float(expected_sd-sd[ix])))
    assert max_mu_difference<1e-12 and max_sd_difference<1e-12
    noise_rows=[]
    for recorded in summary['noise_audit']:
        multiplier=recorded['multiplier'];raw=truth+multiplier*sd*normal
        low=finite&(raw<0);high=finite&(raw>1);observed=np.where(finite,np.clip(raw,0.,1.),truth)
        values=dict(values=int(truth.size),finite_values=int(finite.sum()),clipped_low=int(low.sum()),clipped_high=int(high.sum()),missing_preserved=int((~finite).sum()))
        assert all(values[k]==recorded[k] for k in values)
        assert np.array_equal(np.isnan(observed),np.isnan(truth)) and np.array_equal(observed[:,point],truth[:,point],equal_nan=True)
        noise_rows.append(dict(common,multiplier=multiplier,**values,operation='saved truth + lambda*exact-point-aware trainSD*saved normal, observed values censored [0,1]',truth_modified=False,independently_recomputed=True))
        for i,ingredient in enumerate(testmeta['ingredient_ids']):
            for j,nutrient in enumerate(fields):
                err=(observed-truth)[:,i,j];err=err[np.isfinite(err)]
                field_rows.append(dict(common,ingredient=ingredient,nutrient=nutrient,canonical_unit='kg/kg DM',multiplier=multiplier,
                    train_sample_SD=float(sd[i,j]),declared_measurement_SD=float(multiplier*sd[i,j]),exact_training_point=bool(point[i,j]),finite_training_count=int(count[i,j]),
                    clipped_low=int(low[:,i,j].sum()),clipped_high=int(high[:,i,j].sum()),observed_error_mean=float(err.mean()) if len(err) else '',
                    observed_error_SD=float(err.std(ddof=1)) if len(err)>1 else '',interpretation='finite saved noise realization; post-censor empirical moments are descriptive'))
    receipts=frozen['censor_quadrature_receipts'];moments=frozen['censor_endpoint_moments']
    assert len(receipts)==len(moments) and all(x['success'] for x in receipts)
    assert all(0<=x['posterior_mean']<=1 and 0<=x['posterior_variance']<=.25 for x in moments)
    endpoint=dict(common,endpoint_precomputations=len(receipts),all_quad_vec_success=True,
        max_relative_vector_error_estimate=max((x['quadrature_relative_vector_error_estimate'] for x in receipts),default=0.),
        evaluations=sum(x['evaluations'] for x in receipts),negative_variance_roundoff_floored=sum(x['variance_negative_roundoff_floored'] for x in receipts),
        interpretation='saved deterministic quadrature diagnostics, not a formal global or posterior-moment error certificate')
    return noise_rows,endpoint,dict(test_sha256=sha(testpath),normal_sha256=sha(noisepath),prior_sha256=sha(priorpath),
        train_location_reconstruction_max_abs_difference=max_mu_difference,train_SD_reconstruction_max_abs_difference=max_sd_difference)


def main():
    p=argparse.ArgumentParser()
    for name in ('formal','protocol','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--qualification',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('fresh independent saved-analysis output only')
    cfg=read(a.protocol);protocolsha=sha(a.protocol)
    if cfg['status']!='frozen_before_formal_draw':raise ValueError('formal protocol not frozen')
    work=a.protocol.absolute().parents[2]
    assert {relative:sha(work/relative) for relative in cfg['code_sha256']}==cfg['code_sha256']
    # Read/verify every cell first. Partial formal results are never silently
    # promoted into the registered all-six report.
    directories={identity:a.formal/cell['case']/'SD-H0'/'C0'/f"trainroot{cell['training_root_index']}" for identity,cell in cfg['cells'].items()}
    for identity,d in directories.items():
        if not (d/'DONE.json').exists():raise FileNotFoundError('registered formal cell not done: '+identity)
    a.output.mkdir(parents=True);rates=[];pairs=[];noise=[];points=[];unknownrows=[];inputbindings=[];noise_fields=[];endpoints=[];runtime=[]
    rate_tail=cfg['families']['rates']['alpha']/cfg['families']['rates']['bound'];pair_tail=cfg['families']['paired']['alpha']/cfg['families']['paired']['bound']/4
    interval_max_abs=0.;canonical_positions=0
    for identity,directory in directories.items():
        cell=cfg['cells'][identity];s=read(directory/'summary.json');m=read(directory/'manifest_private.json');done=read(directory/'DONE.json')
        assert done['status']=='complete' and done['summary_sha256']==sha(directory/'summary.json') and s['status']=='complete'
        assert s['identity']==identity and m['protocol_sha256']==protocolsha and m['code_sha256']==cfg['code_sha256']
        assert m['parent_binding']==cfg['parent_bindings'][cell['parent_key']] and not m['no_draw_freeze_check']
        frozenpath=directory/'restricted/frozen_measurement_policies.json';frozen=read(frozenpath)
        assert read(directory/'POLICIES_FROZEN.json')['sha256']==sha(frozenpath)
        assert tuple(frozen['policies'])==tuple(cfg['policies']) and frozen['full_Q2_scores_all_candidates'] and not frozen['candidate_pruning_used']
        assert s['new_training_draws']==0 and s['full_Q2_scores_all_candidates'] and not s['candidate_pruning_used']
        fr=frozen['parent_policy'];n=s['test_states'];assert n==cfg['test_states']
        reported={r['policy']:r for r in s['policies']};assert set(reported)==set(cfg['policies'])
        common=dict(identity=identity,case=cell['case'],training_root_index=cell['training_root_index'],world='SD-H0',scenario='C0')
        for change in frozen['point_roundoff_correction_vs_prior_information']:points.append(dict(common,**change))
        binding=cfg['parent_bindings'][cell['parent_key']]
        assert {name:sha(Path(binding['path'])/name) for name in binding['sha256']}==binding['sha256']
        noise_rows,endpoint,noisebinding=audit_saved_noise(directory,frozen,s,common,noise_fields);noise.extend(noise_rows);endpoints.append(endpoint)
        assert s['zero_noise_spot']==dict(states=cfg['diagnostic_outer_states'],score_exact_equality=True,choice_exact_equality=True)
        runtime.append(dict(common,Q2_actions=s['Q2_actions'],Q3_actions=s['Q3_actions'],elapsed_s=s['elapsed_s'],prior_precompute_seconds=s['prior_precompute_seconds'],**s['selection_timing']))
        assignmentpath=directory/'restricted/test_assignments.npz';losses={};outcome_sha={}
        with np.load(assignmentpath,allow_pickle=False) as assigned:
            for policy in cfg['policies']:
                outcome=directory/'restricted'/(policy+'_assigned_outcomes.npz');outcome_sha[policy]=sha(outcome)
                with np.load(outcome,allow_pickle=False) as z:
                    action=z['action_id'];known=z['row_violated'].any(axis=1);u=(~known)&z['row_unknown'].any(axis=1);failure=known|u
                    assert len(action)==n and np.array_equal(action,assigned[policy])
                    assert np.array_equal(known,z['known_failure']) and np.array_equal(u,z['unknown']) and np.array_equal(failure,z['failure'])
                    assert not np.any(known&u)
                    menu='Q2' if policy.startswith('Q2') else 'Q3';assert set(map(int,action))<=set(fr['menus'][menu])
                    k=int(failure.sum());ku=int(u.sum());kv=int(known.sum());up=upper(k,n,rate_tail);r=reported[policy]
                    assert k==r['failure_count'] and ku==r['unknown_count'] and kv==r['known_failure_count'] and k/n==r['failure_rate']
                    error=abs(up-r['adjusted_upper']);interval_max_abs=max(interval_max_abs,error);assert error<2e-12
                    assert [t for t in cfg['alpha_targets'] if up<=t]==r['qualified_targets']
                    assert int(assigned[policy+'_fallback'].sum())==r['fallback_states']
                    assert int(len(np.unique(action)))==r['chosen_distinct_actions']
                    mean_cost=float(z['feed_cost'].mean());assert abs(mean_cost-r['mean_feed_cost'])<1e-12
                    losses[policy]=failure.copy();canonical_positions+=n
                    rates.append(dict(common,policy=policy,n_states=n,failure_count=k,known_failure_count=kv,unknown_count=ku,failure_rate=k/n,
                        adjusted_upper_recomputed=up,adjusted_upper_reported=r['adjusted_upper'],qualified_targets='|'.join(map(str,r['qualified_targets'])),
                        chosen_distinct_actions=r['chosen_distinct_actions'],fallback_states=r['fallback_states'],mean_feed_cost_private=mean_cost,
                        event_reconstruction='original violation OR unknown; known/unknown exclusive, no state dropped',candidate_pool=menu))
                    reasons=collections.Counter(z['energy_unknown_reason'][u].tolist())
                    unknownrows.append(dict(common,policy=policy,primary_unknown_count=ku,energy_reasons_on_primary_unknown=json.dumps(dict(reasons),ensure_ascii=False),
                        support_violation_all_selected_states=int(z['support_violation'].sum()),analysis_anomaly_all_selected_states=int(z['analysis_anomaly'].sum())))
        reportedpairs={(x['policy'],x['comparator']):x for x in s['paired']};assert set(reportedpairs)==set(map(tuple,cfg['paired_comparisons']))
        for policy,comparator in cfg['paired_comparisons']:
            x=losses[policy];y=losses[comparator];worse=int((x&~y).sum());better=int((~x&y).sum());wl,wu=cp(worse,n,pair_tail);bl,bu=cp(better,n,pair_tail)
            lo,hi=wl-bu,wu-bl;delta=(worse-better)/n;r=reportedpairs[(policy,comparator)]
            assert worse==r['recourse_only_failure'] and better==r['constant_only_failure'] and delta==r['delta_recourse_minus_constant']
            error=max(abs(lo-r['conservative_delta_ci'][0]),abs(hi-r['conservative_delta_ci'][1]));interval_max_abs=max(interval_max_abs,error);assert error<2e-12
            pairs.append(dict(common,policy=policy,comparator=comparator,n_states=n,policy_only_failure=worse,comparator_only_failure=better,
                delta_policy_minus_comparator=delta,adjusted_lower=lo,adjusted_upper=hi,adjusted_direction='improved' if hi<0 else 'worse' if lo>0 else 'includes_zero',
                interval_allocation='four binomial tails alpha/(4*10000); separate paired family'))
        inputbindings.append(dict(identity=identity,summary_sha256=sha(directory/'summary.json'),manifest_sha256=sha(directory/'manifest_private.json'),
            assignment_sha256=sha(assignmentpath),frozen_sha256=sha(frozenpath),outcome_sha256=outcome_sha,**noisebinding))
    assert len(rates)==len(cfg['cells'])*len(cfg['policies'])==60 and len(pairs)==len(cfg['cells'])*len(cfg['paired_comparisons'])==66
    groups=[]
    for case,policy in sorted(set((r['case'],r['policy']) for r in rates)):
        values=[r for r in rates if r['case']==case and r['policy']==policy]
        group=dict(case=case,policy=policy,independently_frozen_training_rules=len(values),failure_rate_min=min(r['failure_rate'] for r in values),failure_rate_max=max(r['failure_rate'] for r in values),
            adjusted_upper_min=min(r['adjusted_upper_recomputed'] for r in values),adjusted_upper_max=max(r['adjusted_upper_recomputed'] for r in values),unknown_count_min=min(r['unknown_count'] for r in values),unknown_count_max=max(r['unknown_count'] for r in values),
            interpretation='range of three separately frozen rules; no pooled certificate')
        for target in cfg['alpha_targets']:group['qualified_at_'+str(target)]=sum(r['adjusted_upper_recomputed']<=target for r in values)
        groups.append(group)
    directiongroups=[]
    for policy,comparator in cfg['paired_comparisons']:
        v=[r for r in pairs if r['policy']==policy and r['comparator']==comparator]
        directiongroups.append(dict(policy=policy,comparator=comparator,cells=len(v),improved=sum(r['adjusted_direction']=='improved' for r in v),worse=sum(r['adjusted_direction']=='worse' for r in v),includes_zero=sum(r['adjusted_direction']=='includes_zero' for r in v),
            observed_difference_min=min(r['delta_policy_minus_comparator'] for r in v),observed_difference_max=max(r['delta_policy_minus_comparator'] for r in v)))
    for name,rows in [('all60_rates_private.csv',rates),('all66_pairs.csv',pairs),('case_policy_ranges.csv',groups),('paired_direction_counts.csv',directiongroups),('actual_noise_clipping.csv',noise),('actual_noise_field_scales_private.csv',noise_fields),('point_roundoff_differences.csv',points),('unknown_partition.csv',unknownrows),('endpoint_quadrature_diagnostics.csv',endpoints),('runtime_and_pool_sizes.csv',runtime)]:write_csv(a.output/name,rows)
    report=dict(status='pass',new_draws=0,new_rules_selected=False,protocol_sha256=protocolsha,analysis_code_sha256=sha(__file__),cells=6,rates=60,pairs=66,
        original_event_saved_row_reconstructed_positions=canonical_positions,max_absolute_recomputed_interval_difference=interval_max_abs,input_bindings=inputbindings,
        groups=groups,paired_directions=directiongroups,all_19_current_runtime_hashes_match=True,independent_saved_noise_reconstruction=True,endpoint_diagnostics=endpoints,
        interpretation=['fullQ2 versus Q3 identified only through same-score same-lambda within-cell comparisons',
            'postmean versus raw and postUA versus postmean separate mean shrinkage and propagated-uncertainty changes; do not attribute the combined pipeline to posterior alone',
            'bounded Gaussian working prior plus correct censor likelihood differs from original TN/BETA truth; canonical event remains unchanged',
            'exact training point fields correct old information pseudoSD at rounding scale; different formal states/noise prohibit across-protocol pairs',
            'saved-label/statistical/row-event reconstruction audit, not independent canonical full-state or complete selector replay',
            'rate and paired families allocated separately; no overall95 or pooling independent trained roots'])
    if a.qualification:
        qa={}
        for case in ('A','C'):
            q=read(a.qualification/f'qualification_{case}_128_v3.json');qa[case]=dict(status=q['status'],new_draws=q['new_draws'],zero_noise_max_absolute_score_difference=q['zero_noise_score_max_absolute_difference'],gradient=q['gradient'],
                scope=q['original_energy_fixed_vs_random_scope'],profiles=q['profiles'],receipt_sha256=sha(a.qualification/f'qualification_{case}_128_v3.json'))
        report['qualification_for_candidate_SI']=qa
    dump(a.output/'独立全量统计核验与汇总.json',report)
    print(json.dumps(dict(status='pass',cells=6,rates=60,pairs=66,event_reconstruction_positions=canonical_positions,max_interval_difference=interval_max_abs)))

if __name__=='__main__':main()
