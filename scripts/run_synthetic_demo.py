#!/usr/bin/env python3
"""Four-method software smoke on invented data. NOT a nutrition experiment.

No restricted data, network connection, prior run, or environment-lock rewrite is
needed. Point-mass uncertainty is intentional: methods should agree when the
margin and chance-violation allowance are zero. A successful run is a software
invariant, not evidence of practical ration quality or superiority of a method.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ration_reliability.datamodel import SolverOptions, SolveStatus
from ration_reliability.evaluation import evaluate_drawset
from ration_reliability.io import load_problem
from ration_reliability.optimization import get_method
from ration_reliability.uncertainty import PointMassModel, RandomStreams


def run() -> dict:
    fixture = ROOT / 'data/synthetic_test_only/engine_toy_problem_v1.yaml'
    problem, validation = load_problem(fixture, mode='smoke')
    if not validation.ok or not problem.is_synthetic:
        raise ValueError('demo requires validated, explicitly synthetic input')
    model = PointMassModel('software_demo_point_mass_v1', problem.ingredient_ids,
                           problem.nutrient_ids, problem.nominal_theta(),
                           problem.dm_estimates(), is_synthetic=True)
    streams = RandomStreams(2026092601)
    opt = model.draw(streams, 'opt', 16)
    evaluation = model.draw(streams, 'test', 128)
    options = SolverOptions(time_limit_s=30.0, mip_rel_gap=0.0)
    specs = [
        ('M0_nominal', {}, None),
        ('M1_safety_margin', {'k': 0.0, 'margin_scale': 'relative', 'apply_to_dm': False}, None),
        ('M2_joint_chance_saa', {'alpha_train': 0.0, 'n_scenarios': None}, opt),
        ('M3c_scenario_set_robust', {}, opt),
    ]
    results = []
    for method_id, params, draws in specs:
        answer = get_method(method_id)(problem, params=params, opt_draws=draws, solver_options=options)
        if answer.status is not SolveStatus.OPTIMAL or answer.decision is None:
            raise RuntimeError(f'{method_id}: expected optimum, obtained {answer.status}')
        ev = evaluate_drawset(answer.decision, evaluation, problem.compiled, prices=problem.prices)
        summary = ev.summary()
        if not summary['structural_ok'] or summary['joint']['rate_upper'] != 0.0:
            raise RuntimeError(f'{method_id}: zero-uncertainty feasibility invariant failed')
        results.append({'method_id': method_id, 'solve_status': answer.status.value,
                        'scenario_cost': float(answer.objective),
                        'cost_unit': answer.objective_unit,
                        'ration': answer.decision.to_dict(), 'evaluation': summary})
    costs = [r['scenario_cost'] for r in results]
    if not all(math.isclose(c, costs[0], rel_tol=1e-8, abs_tol=1e-8) for c in costs):
        raise RuntimeError('equivalent zero-uncertainty problems returned different optimal costs')
    return {'schema': 'ration_reliability.synthetic_smoke/1',
            'is_synthetic': True, 'run_type': 'software_smoke_only',
            'claim_boundary': 'Invented input; no biological validation, no method ranking, no official research result.',
            'fixture': str(fixture.relative_to(ROOT)),
            'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'n_methods': len(results), 'n_opt_draws': 16, 'n_eval_draws': 128,
            'seed': streams.root_seed, 'point_mass_equivalence_passed': True,
            'results': results}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, help='new JSON file; existing files are never overwritten')
    args = p.parse_args(argv)
    if args.output and args.output.exists():
        print('BLOCKED: output already exists; choose a new filename', file=sys.stderr)
        return 2
    try:
        result = run()
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as f: f.write(text)
        print(text)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
