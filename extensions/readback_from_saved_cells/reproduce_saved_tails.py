#!/usr/bin/env python3
"""Postprocess frozen R7 margins. No new sampling, fitting, selection or tail test."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, sys, time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
CORE_MANIFEST_SHA = '65527b679a2158123b95f7f74b0ce983be29bd3f20d3255c54ce7a970e632571'
ENERGY = 'PN-NEL-FIXEDDMI'
ID = ['family', 'case', 'scenario', 'training_rep_id', 'policy']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, data):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def key(row, fields):
    return tuple(str(row[k]) for k in fields)


def compare(actual, expected, fields, skip=('scope', 'description')):
    a = {key(r, fields): r for r in actual}
    b = {key(r, fields): r for r in expected}
    assert len(a) == len(actual) and len(b) == len(expected) and a.keys() == b.keys()
    maximum = 0.
    for k, ref in b.items():
        for col, val in ref.items():
            if col in skip:
                continue
            assert col in a[k], col
            other = a[k][col]
            if val == '' or val is None:
                assert other == '' or other is None, (k, col)
                continue
            try:
                x, y = float(other), float(val)
            except (TypeError, ValueError):
                assert str(other) == str(val), (k, col)
            else:
                error = abs(x - y)
                maximum = max(maximum, error)
                assert error <= 3e-14 + 3e-12 * abs(y), (k, col, error)
    return maximum


def metrics(deficit, defined, violation):
    values, violated = deficit[defined], deficit[violation]
    k = math.ceil(.05 * len(values))
    def mean(a): return float(a.mean()) if len(a) else None
    def median(a): return float(np.median(a)) if len(a) else None
    def p95(a): return float(np.quantile(a, .95, method='linear')) if len(a) else None
    return dict(n_defined=int(defined.sum()), n_violation=int(violation.sum()),
                n_unknown=int((~defined).sum()), conditional_mean=mean(values),
                conditional_median=median(values), P95=p95(values),
                top_ceil5pct_mean=float(np.sort(values)[-k:].mean()) if k else None,
                top_count=k, violated_conditional_mean=mean(violated),
                violated_conditional_median=median(violated),
                violated_conditional_P95=p95(violated), maximum=float(values.max()),
                conditional_denominator='all_defined_states_including_zero_deficits',
                violated_conditional_denominator='classified_violations_only')


def deny(*args, **kwargs):
    raise AssertionError('Tail reproduction forbids new sampling or training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core', type=Path, required=True, help='Extracted unchanged R7 core review bundle')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    core, out = args.core.absolute(), args.output.absolute()
    assert not out.exists(), 'Do not overwrite any completed output'
    assert sha(core / 'input_manifest.csv') == CORE_MANIFEST_SHA
    manifest = {r['relative_path']: r for r in rows(core / 'input_manifest.csv')}
    bound = {}
    def bind(path):
        relative = str(path.relative_to(core))
        row = manifest[relative]
        digest = sha(path)
        assert path.stat().st_size == int(row['size']) and digest == row['sha256'], relative
        bound[relative] = digest
    reader = core / 'study/reproducibility/replay_new_saved_results.py'
    bind(reader)
    runtime = core / 'study/reproducibility/restricted/runtime_repo'
    source_map = read(core / 'study/reproducibility/restricted/actual_repo_complete_input_sha256_map.json')
    runtime_paths = ([r['path'] for r in source_map['entries']] if 'entries' in source_map else
                     [name for name, digest in source_map.items() if isinstance(digest, str) and len(digest) == 64])
    assert len(runtime_paths) == 151
    for name in runtime_paths:
        bind(runtime / name)
    for relative, row in manifest.items():
        if relative.startswith('study/reproducibility/restricted/helpers/') and row['role'] == 'study_custom_code':
            bind(core / relative)
    spec = importlib.util.spec_from_file_location('saved_core_reader', reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = core / 'study'
    cells = [(family, p.parent) for family in ('coverage_formal', 'measurement_formal')
             for p in sorted((study / family).glob('**/DONE.json'))]
    assert sum(f == 'coverage_formal' for f, p in cells) == 18
    assert sum(f == 'measurement_formal' for f, p in cells) == 6
    normalized, nel, arrays, identities = [], [], {}, []
    started = time.monotonic()
    with ExitStack() as stack:
        stack.enter_context(patch.object(module.UncertaintyModel, 'draw', deny))
        stack.enter_context(patch.object(module.UncertaintyModel, 'sample', deny))
        stack.enter_context(patch.object(np.random, 'default_rng', deny))
        for family, cell in cells:
            for name in ('summary.json', 'DONE.json'):
                bind(cell / name)
            summary = read(cell / 'summary.json')
            assert sha(cell / 'summary.json') == read(cell / 'DONE.json')['summary_sha256']
            parts = summary['identity'].split('__')
            ctx = module.common.load_context(parts[0], ('SD-H0',))
            ref = ctx['ref']
            rules = {r.constraint_id: r for r in ctx['problems']['FULL11'].constraints}
            rids = list(next(e for e in ref.events if e.event_id == module.EVENT).members)
            assert len(rids) == 9
            base = dict(family=family, case={'dev_case_v3a': 'A', 'dev_case_v3c': 'C'}[parts[0]],
                        scenario=parts[2], training_rep_id=int(parts[3].replace('trainr', '')))
            for item in summary['policies']:
                path = cell / 'restricted' / (item['policy'] + '_assigned_outcomes.npz')
                bind(path)
                with np.load(path, allow_pickle=False) as z:
                    margin = z['margin'].copy()
                    unknown, violated = z['row_unknown'].copy(), z['row_violated'].copy()
                    loss = z['failure'].copy()
                    dm = z['supplied_dm_kg_d'].copy()
                    known = z['known_failure'].copy()
                    event_unknown = z['unknown'].copy()
                assert margin.shape == (20000, 9) and np.array_equal(np.isfinite(margin), ~unknown)
                assert np.array_equal(known, violated.any(axis=1))
                assert np.array_equal(event_unknown, (~known) & unknown.any(axis=1))
                assert np.array_equal(loss, known | event_unknown)
                assert int(loss.sum()) == item['failure_count']
                ident = dict(**base, policy=item['policy'])
                identities.append(ident)
                for j, cid in enumerate(rids):
                    rule = rules[cid]
                    tol = ref.energy.tolerance_mcal_d if cid == ENERGY else rule.numerical_tolerance
                    assert np.array_equal((~unknown[:, j]) & (-margin[:, j] > tol), violated[:, j])
                    deficit = np.where(~unknown[:, j], np.maximum(0., -margin[:, j]), 0.)
                    if cid == ENERGY:
                        denominator = np.full(len(dm), ref.energy.requirement_mcal_d)
                        label = 'original_fixed_selected_chain_NEL_target'
                    elif cid in ('PN-CA-ABS', 'PN-P-ABS'):
                        denominator = float(rule.bound) + (.9 if cid == 'PN-CA-ABS' else 1.) * dm
                        label = 'original_statewise_factorial_' + ('Ca' if cid == 'PN-CA-ABS' else 'P') + '_target'
                    else:
                        denominator = np.full(len(dm), abs(float(rule.bound)))
                        label = ('original_fixed_CP_example_target_not_validated_protein_requirement'
                                 if cid == 'PN-CP-SUP' else 'absolute_linear_constraint_bound_not_nutrient_requirement')
                    defined = ~unknown[:, j]
                    assert np.all(np.isfinite(denominator) & (denominator > 0))
                    row = dict(**ident, constraint_id=cid, n_total=len(dm), unit='dimensionless',
                               normalization=label, classification_tolerance='original_units_only',
                               **metrics(deficit / denominator, defined, violated[:, j]))
                    normalized.append(row)
                    if cid == ENERGY:
                        d = deficit[defined]
                        positive = d > 0
                        frequency = float(positive.mean())
                        conditional = float(d[positive].mean()) if positive.any() else 0.
                        error = abs(float(d.mean()) - frequency * conditional)
                        arrays[key(ident, ID)] = (margin[:, j], defined)
                        nel.append(dict(**ident, n_total=len(dm), n_defined=int(defined.sum()),
                                        row_unknown=int((~defined).sum()), energy_violation_count=int(violated[:, j].sum()),
                                        energy_violation_rate_total=float(violated[:, j].mean()), joint_failure_count=int(loss.sum()),
                                        joint_failure_rate_total=float(loss.mean()), positive_deficit_count=int(positive.sum()),
                                        positive_deficit_frequency_defined=frequency, identity_absolute_error=error,
                                        normalization=label, normalized_mean=row['conditional_mean'],
                                        normalized_top_ceil5pct_mean=row['top_ceil5pct_mean'], scope='descriptive saved-margin reproduction'))
    assert len(nel) == 96 and len(normalized) == 864
    lookup = {key(r, ID + ['constraint_id']): r for r in normalized}
    differences = []
    for family, case, source, replicate in sorted({key(r, ID[:-1]) for r in identities}):
        pairs = [('Q2_safe_delta', 'Q3_delta')] if family == 'coverage_formal' else [
            (p, c) for noise in ('0p1', '0p5') for p, c in
            [('Q3_postUA_' + noise, 'Q3_raw_noise_' + noise),
             ('Q3_postUA_' + noise, 'Q3_postmean_' + noise),
             ('Q3_postmean_' + noise, 'Q3_raw_noise_' + noise),
             ('Q2_postUA_' + noise, 'Q3_postUA_' + noise)]] + [('Q2_ideal_delta', 'Q3_ideal_delta')]
        for policy, comparator in pairs:
            for cid in [k[-1] for k in lookup if k[:5] == (family, case, source, replicate, policy)]:
                a, b = [lookup[(family, case, source, replicate, p, cid)] for p in (policy, comparator)]
                mean_delta = a['conditional_mean'] - b['conditional_mean']
                tail_delta = a['top_ceil5pct_mean'] - b['top_ceil5pct_mean']
                direction = lambda v: 'lower' if v < 0 else 'higher' if v > 0 else 'equal'
                differences.append(dict(family=family, case=case, scenario=source, training_rep_id=replicate,
                                        policy=policy, comparator=comparator, constraint_id=cid,
                                        normalized_mean_delta=mean_delta, normalized_P95_delta=a['P95'] - b['P95'],
                                        normalized_top_ceil5pct_mean_delta=tail_delta, mean_direction=direction(mean_delta),
                                        tail_direction=direction(tail_delta), scope='descriptive saved-margin reproduction'))
    grouped = {}
    for r in differences:
        grouped.setdefault(key(r, ['family', 'policy', 'comparator', 'constraint_id']), []).append(r)
    directions = [dict(family=k[0], policy=k[1], comparator=k[2], constraint_id=k[3],
                       n_same_protocol_comparisons=len(rr), lower=sum(r['tail_direction'] == 'lower' for r in rr),
                       equal=sum(r['tail_direction'] == 'equal' for r in rr), higher=sum(r['tail_direction'] == 'higher' for r in rr),
                       min_normalized_tail_difference=min(r['normalized_top_ceil5pct_mean_delta'] for r in rr),
                       max_normalized_tail_difference=max(r['normalized_top_ceil5pct_mean_delta'] for r in rr), scope='descriptive only')
                  for k, rr in sorted(grouped.items())]
    shared_rows = []
    for family, case, source, replicate in sorted({k[:4] for k in arrays if k[0] == 'measurement_formal'}):
        for noise in ('0p1', '0p5'):
            policy = 'Q3_postUA_' + noise
            for comparator in ('Q3_raw_noise_' + noise, 'Q3_postmean_' + noise):
                ka, kb = [(family, case, source, replicate, p) for p in (policy, comparator)]
                ma, da = arrays[ka]
                mb, db = arrays[kb]
                shared = da & db
                k = math.ceil(.05 * int(shared.sum()))
                ta, tb = [float(np.sort(np.maximum(0., -m[shared]))[-k:].mean()) for m in (ma, mb)]
                a, b = [lookup[key0 + (ENERGY,)] for key0 in (ka, kb)]
                shared_rows.append(dict(family=family, case=case, scenario=source, training_rep_id=replicate,
                                        policy=policy, comparator=comparator, n_total=len(da), policy_defined=int(da.sum()),
                                        comparator_defined=int(db.sum()), shared_defined=int(shared.sum()),
                                        policy_only_defined=int((da & ~db).sum()), comparator_only_defined=int((db & ~da).sum()),
                                        full_normalized_tail_policy=a['top_ceil5pct_mean'],
                                        full_normalized_tail_comparator=b['top_ceil5pct_mean'],
                                        common_defined_direction='higher' if ta > tb else 'lower' if ta < tb else 'equal',
                                        common_defined_tail_ratio=ta / tb, original_tail_ratio=a['top_ceil5pct_mean'] / b['top_ceil5pct_mean'],
                                        scope='same defined positions; descriptive only'))
    ranges = []
    for case in ('A', 'C'):
        for source_policy, reported in [('Q3_raw_noise_0p5', 'Q3_raw_noise_0p5'),
                                        ('Q3_postmean_0p5', 'Q3_postmean_0p5'),
                                        ('Q3_postUA_0p5', 'Q3_postUA_0p5'), ('Q2_postUA_0p5', 'Q2_postUA_0p5')]:
            rr = [r for r in nel if r['family'] == 'measurement_formal' and r['case'] == case and r['policy'] == source_policy]
            assert len(rr) == 3
            ranges.append(dict(case=case, policy=reported, risk_min_pct=100 * min(r['joint_failure_rate_total'] for r in rr),
                               risk_max_pct=100 * max(r['joint_failure_rate_total'] for r in rr),
                               NEL_tail_min_pct_target=100 * min(r['normalized_top_ceil5pct_mean'] for r in rr),
                               NEL_tail_max_pct_target=100 * max(r['normalized_top_ceil5pct_mean'] for r in rr),
                               n_training_replicates=3, n_per_replicate=20000))
    datasets = [('normalized_severity.csv', normalized, ID + ['constraint_id']),
                ('descriptive_differences.csv', differences, ID + ['comparator', 'constraint_id']),
                ('tail_directions.csv', directions, ['family', 'policy', 'comparator', 'constraint_id']),
                ('NEL_records.csv', nel, ID), ('common_defined_NEL.csv', shared_rows, ID + ['comparator']),
                ('high_noise_ranges.csv', ranges, ['case', 'policy'])]
    errors = {name: compare(data, rows(HERE / 'expected' / name), fields) for name, data, fields in datasets}
    for relative, digest in bound.items():
        assert sha(core / relative) == digest
    out.mkdir(parents=True)
    for name, data, fields in datasets:
        write(out / name, data)
    receipt = dict(status='PASS', core_manifest_sha256=CORE_MANIFEST_SHA, policy_cell_records=len(nel),
                   saved_policy_state_positions=sum(r['n_total'] for r in nel), normalized_rows=len(normalized),
                   descriptive_differences=len(differences), direction_groups=len(directions), common_defined_pairs=len(shared_rows),
                   high_noise_ranges=len(ranges), input_files_before_after_unchanged=len(bound), max_errors=errors,
                   new_draws=0, training_or_policy_selection=0, canonical_evaluator_reruns=0, new_formal_tail_tests=0,
                   elapsed_seconds=time.monotonic()-started, numpy_version=np.__version__, python_version=sys.version,
                   scope='Portable readback of previously saved margins and labels; not new experiments, source regeneration, biological validation or an external reviewer execution',
                   bound_inputs=bound, outputs={n: dict(sha256=sha(out / n), rows=len(data)) for n, data, _ in datasets})
    (out / 'complete_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'policy_cell_records', 'normalized_rows', 'descriptive_differences', 'direction_groups', 'new_draws')}))


if __name__ == '__main__':
    main()
