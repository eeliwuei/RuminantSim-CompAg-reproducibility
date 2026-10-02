"""Official-run plan batch 2, test 8: the validation screening rule reaches every selection of the method block.

``experiments/E0_verification/run_dev_case_v1.py::run_method_block`` selects parameters on the validation stream in
four places: M1 (k) and M2 (alpha_train) through the selection mappings that ``smoke_pipeline.run_methods`` forwards,
M3a (k_box) and every per-k M3b (Gamma) directly.  The official setting is ``cp_upper_le_alpha`` with confidence 0.95
(one-sided exact Clopper-Pearson upper bound of (n_violated + n_unknown) / S <= alpha); the default keeps the
development replay rule ``rate_upper_le_alpha``.  Checked with a spy in place of ``select_parameter_on_validation`` and
stub solvers (no problem is built, nothing is drawn or solved, no data file is read):

1. official rule requested -> all four selection kinds receive ``cp_upper_le_alpha`` and confidence 0.95, and the
   selection mappings of the M1 / M2 specs carry them too;
2. default -> all four receive ``rate_upper_le_alpha`` and no confidence (development semantics unchanged);
3. the official rule without a confidence, or an unknown rule, is refused before any selection;
4. the M3b per-k selections reach the optional ``selection_sink`` (descriptive grid report), one per k and alpha.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
E0 = REPO / "experiments" / "E0_verification"
if str(E0) not in sys.path:
    sys.path.insert(0, str(E0))
sys.dont_write_bytecode = True

import run_dev_case_v1 as DRV  # noqa: E402
import smoke_pipeline as SP  # noqa: E402

FAMILY_BY_PARAM = {"k": "M1", "alpha_train": "M2", "k_box": "M3a", "gamma": "M3b"}


def _fake_result(cost: float = 1.0):
    return SimpleNamespace(objective=cost, has_solution=True, decision=None, status="optimal", streams_used=(),
                           method_id="stub")


def _fake_selection(param_name: str, grid, a: float, rule: str, confidence):
    v = list(grid)[0]
    cand = SimpleNamespace(value=v, rate_upper=0.0, status="optimal", solve_result=_fake_result())
    return SimpleNamespace(status="selected", selected_value=v, selected_result=_fake_result(), candidates=(cand,),
                           param_name=param_name, validation_stream_id="root=0/validation",
                           to_dict=lambda: {"param_name": param_name, "screening_rule": rule, "confidence": confidence})


@pytest.fixture()
def spy(monkeypatch):
    calls: list[dict] = []

    def select(solve_fn, problem, validation_draws, *, param_name, grid, target_alpha, screening_rule,
               confidence=None, base_params=None, opt_draws=None, solver_options=None, consumer=None):
        calls.append({"family": FAMILY_BY_PARAM[param_name], "screening_rule": screening_rule,
                      "confidence": confidence, "consumer": consumer, "base_params": dict(base_params or {})})
        return _fake_selection(param_name, grid, target_alpha, screening_rule, confidence)

    stub_solver = lambda *a, **k: _fake_result()  # noqa: E731
    monkeypatch.setattr(DRV, "select_parameter_on_validation", select)
    monkeypatch.setattr(SP, "select_parameter_on_validation", select)
    monkeypatch.setattr(DRV, "get_method", lambda method_id: stub_solver)
    monkeypatch.setattr(SP, "get_method", lambda method_id: stub_solver)
    monkeypatch.setattr(DRV, "factory_energy_box", lambda fm, lin, k, tag="": ("box", float(k)))
    monkeypatch.setattr(DRV, "make_box_solver", lambda fn, boxes: stub_solver)
    return calls


def _draws(stream: str):
    return SimpleNamespace(stream=stream, stream_id=f"root=0/{stream}", model_fingerprint="w", model_id="w")


def _run(**kw):
    return DRV.run_method_block(None, None, None, _draws("opt"), _draws("validation"), [0.05, 0.01], [128, 512], None,
                                m1_variants=(True, False), tag="T", **kw)


def test_official_rule_reaches_all_four_selection_kinds(spy):
    runs, tables = _run(screening_rule=DRV.OFFICIAL_SCREENING_RULE, confidence=DRV.OFFICIAL_SCREENING_CONFIDENCE)
    assert DRV.OFFICIAL_SCREENING_RULE == "cp_upper_le_alpha" and DRV.OFFICIAL_SCREENING_CONFIDENCE == 0.95
    fams = {c["family"] for c in spy}
    assert fams == {"M1", "M2", "M3a", "M3b"}
    # 2 alphas x (2 M1 variants + 2 N + 1 M3a + 5 k of M3b)
    n_k = len(DRV.RUN_CONFIG["methods"]["M3b"]["k_grid"])
    assert len(spy) == 2 * (2 + 2 + 1 + n_k)
    assert all(c["screening_rule"] == "cp_upper_le_alpha" and c["confidence"] == 0.95 for c in spy), spy
    # the M1 / M2 selection mappings themselves carry the rule (they are recorded with the run)
    mapped = [r.spec.selection for r in runs if r.spec.selection is not None]
    assert len(mapped) == 2 * (2 + 2)
    assert all(m["screening_rule"] == "cp_upper_le_alpha" and m["confidence"] == 0.95 for m in mapped)
    # every M3b per-k call is one of the declared k values with the declared nominal mode
    ks = sorted({c["base_params"]["k_box"] for c in spy if c["family"] == "M3b"})
    assert ks == [float(k) for k in DRV.RUN_CONFIG["methods"]["M3b"]["k_grid"]]


def test_default_keeps_the_development_rule(spy):
    _run()
    assert {c["family"] for c in spy} == {"M1", "M2", "M3a", "M3b"}
    assert all(c["screening_rule"] == "rate_upper_le_alpha" and c["confidence"] is None for c in spy)
    assert DRV.DEVELOPMENT_SCREENING_RULE == "rate_upper_le_alpha"


def test_official_rule_without_confidence_or_unknown_rule_is_refused_before_any_selection(spy):
    with pytest.raises(ValueError, match="confidence"):
        _run(screening_rule="cp_upper_le_alpha")
    with pytest.raises(ValueError, match="screening_rule"):
        _run(screening_rule="point_estimate_le_alpha", confidence=0.95)
    assert spy == []


def test_m3b_per_k_selections_reach_the_sink(spy):
    sink: list = []
    runs, _ = _run(screening_rule="cp_upper_le_alpha", confidence=0.95, selection_sink=sink)
    n_k = len(DRV.RUN_CONFIG["methods"]["M3b"]["k_grid"])
    assert len(sink) == 2 * n_k
    labels = {r.spec.name for r in runs if r.spec.method_id == "M3b_budget_robust"}
    assert {e["entry_label"] for e in sink} == labels and all(e["method_id"] == "M3b_budget_robust" for e in sink)
    assert sorted({e["fixed_params"]["k_box"] for e in sink}) == [float(k) for k in
                                                                  DRV.RUN_CONFIG["methods"]["M3b"]["k_grid"]]
