"""End-to-end pipeline on the synthetic toy problem (synthetic_test_only; not a research run):

YAML -> validation -> M0 (HiGHS) -> named-stream draws -> public evaluator -> summary -> run record.
"""

import json

import numpy as np

from ration_reliability.datamodel import SolveStatus
from ration_reliability.evaluation import evaluate_drawset
from ration_reliability.io import build_run_record, load_problem, utc_now, write_run_record
from ration_reliability.optimization import get_method
from ration_reliability.optimization.highs import solver_version_string
from ration_reliability.uncertainty import IndependentNormalModel, RandomStreams


def test_toy_pipeline_end_to_end(tmp_path, repo_root, toy_yaml_path):
    t0 = utc_now()
    pr, rep = load_problem(toy_yaml_path, mode="smoke")
    assert rep.ok and pr.is_synthetic

    res = get_method("M0_nominal")(pr)
    assert res.status is SolveStatus.OPTIMAL and res.is_synthetic

    th = pr.nominal_theta()
    model = IndependentNormalModel("syn_toy_nm", pr.ingredient_ids, pr.nutrient_ids, th, np.abs(th) * 0.05,
                                   pr.dm_estimates(), pr.dm_estimates() * 0.03, theta_upper=1.0, is_synthetic=True)
    streams = RandomStreams(1103)
    test_draws = model.draw(streams, "test", 5000)
    ev = evaluate_drawset(res.decision, test_draws, pr.compiled, prices=pr.prices)
    s = ev.summary()
    assert s["is_synthetic"] and s["draw_stream_id"] == "root=1103/test"
    assert 0.0 <= s["joint"]["rate_lower"] <= s["joint"]["rate_upper"] <= 1.0
    assert s["structural_ok"] and s["cost_per_head_per_day"] == res.objective
    # the nominal optimum has binding CP/Ca constraints, so random composition violates them often
    assert s["joint"]["rate_among_evaluable"] > 0.3

    out = tmp_path / "summary.json"
    out.write_text(json.dumps({"solve": res.to_dict(), "evaluation": s}, default=str), encoding="utf-8")
    rec = build_run_record(run_type="smoke", command="pytest tests/integration/test_toy_pipeline_synthetic.py",
                           repo_root=repo_root, started_at=t0, completed_at=utc_now(), exit_status=0,
                           rng_streams={"test": test_draws.stream_id}, solver_version=solver_version_string(),
                           tolerances=dict(res.tolerances), config_paths=[toy_yaml_path], data_paths=[toy_yaml_path],
                           output_paths=[out], is_synthetic=True)
    p = write_run_record(rec, tmp_path / "run_record.json")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["is_synthetic"] is True and back["run_type"] == "smoke"
    assert back["output_hashes"][str(out)] and back["rng_streams"]["test"] == "root=1103/test"
