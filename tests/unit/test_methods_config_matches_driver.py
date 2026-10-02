"""Official-run plan batch 2, test 4 (BLOCKERS B-453): ``configs/methods.yaml`` states what the code executes.

B-453 found six places where the method configuration and the executed drivers disagreed (screening statistic, Big-M
rule, time limits, Gamma grid, M3 box, test precision) plus the N ladder; the Methods text could not be true to both.
Batch 2 aligned the configuration to the code.  This test compares every aligned value with the executed settings:
``run_dev_case_v1.RUN_CONFIG`` and ``run_method_block`` (development replay defaults), the ablation driver's
``ABLATION_CONFIG`` and ``official_v2.OFFICIAL_SETTINGS`` (official selection, membership, N, limits).  No data file is
read; the modules are imported only.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "experiments" / "E0_verification", REPO / "experiments" / "E1_cost_reliability"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import official_v2 as OV2  # noqa: E402
import run_dev_case_v1 as DRV  # noqa: E402
from ration_reliability.optimization import robust as ROB  # noqa: E402


def _load_ablation():
    spec = importlib.util.spec_from_file_location("run_endpoint_ablation_methods_cfg_test",
                                                  REPO / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load_ablation()
TEXT = (REPO / "configs" / "methods.yaml").read_text(encoding="utf-8")
CFG = yaml.safe_load(TEXT)
M = {m["method_id"]: m for m in CFG["methods"]}
RC = DRV.RUN_CONFIG
OFF = OV2.OFFICIAL_SETTINGS


def test_screening_statistic_equals_the_official_selection_and_the_development_default_is_recorded():
    cs = CFG["common_selection"]
    assert cs["screening_statistic"] == OFF["screening"]["rule"] == DRV.OFFICIAL_SCREENING_RULE == "cp_upper_le_alpha"
    assert cs["confidence"] == OFF["screening"]["confidence"] == DRV.OFFICIAL_SCREENING_CONFIDENCE == 0.95
    assert sorted(cs["applies_to"]) == sorted(OFF["screening"]["applies_to"]) == ["M1", "M2", "M3a", "M3b"]
    assert cs["stream"] == OFF["screening"]["stream"] == "validation"
    assert "n_V + n_U" in cs["counted_states"]
    sig = inspect.signature(DRV.run_method_block).parameters
    assert sig["screening_rule"].default == DRV.DEVELOPMENT_SCREENING_RULE == RC["screening_rule"]["rule"]
    assert sig["confidence"].default is None
    assert DRV.DEVELOPMENT_SCREENING_RULE in cs["development_runs_used"]
    assert AB.ABLATION_CONFIG["endpoint_reporting"]["screening_rule"] == DRV.DEVELOPMENT_SCREENING_RULE
    for mid in ("M1",):
        assert M[mid]["selection_rule"]["screening_statistic"] == cs["screening_statistic"]
        assert M[mid]["selection_rule"]["confidence"] == cs["confidence"]
        assert M[mid]["selection_rule"]["development_runs_used"] == DRV.DEVELOPMENT_SCREENING_RULE
    for mid in ("M2", "M3"):
        rule = M[mid]["alpha_train"]["selection_rule"] if mid == "M2" else M[mid]["selection_rule"]
        assert "cp_upper_le_alpha" in rule and "0.95" in rule
    assert "r̂_val ≤ α 的最低成本" not in TEXT                   # the old point-estimate wording is gone


def test_big_m_rule_polish_and_the_min_violation_box():
    bm = M["M2"]["big_m"]
    assert bm["mode"] == RC["methods"]["M2"]["big_m_mode"] == OFF["M2"]["big_m_mode"] == "box_quantile"
    assert "min(M_box, Mq)" in bm["rule"]
    assert bm["min_violation_diagnostic_mode"] == OFF["min_violation_big_m"] == "box"
    assert 'big_m_mode="box"' in inspect.getsource(DRV.min_violations)
    assert M["M2"]["polish"] is RC["methods"]["M2"]["polish"] is OFF["M2"]["polish"] is True


def test_time_limits_equal_the_executed_limits():
    tl = CFG["solver"]["time_limits_s"]
    assert tl["methods"] == RC["solver"]["time_limit_s"] == AB.ABLATION_CONFIG["solver"]["time_limit_s"] \
        == OFF["time_limits_s"]["methods"] == 300
    assert CFG["solver"]["dev_start_settings"]["time_limit_s"] == tl["methods"]
    assert CFG["solver"]["dev_start_settings"]["mip_rel_gap"] == RC["solver"]["mip_rel_gap"] \
        == AB.ABLATION_CONFIG["solver"]["mip_rel_gap"]
    # the frontier point is solved with the method options (run_cell passes ``opts``)
    assert tl["frontier"] == OFF["time_limits_s"]["frontier"] == tl["methods"]
    assert "solver_options=opts" in inspect.getsource(AB.run_cell)
    mv = {str(k): float(v) for k, v in tl["min_violation"].items()}
    assert mv == {k: float(v) for k, v in OFF["time_limits_s"]["min_violation"].items()} == {
        "128": 60.0, "512": 180.0, "1024": 360.0}
    ab = AB.ABLATION_CONFIG["diagnostics"]["min_violation"]["time_limit_s"]
    assert all(float(ab[k]) == mv[k] for k in mv)
    assert all(float(RC["diagnostics"]["min_violation_saa"]["time_limit_s"][k]) == mv[k] for k in ("128", "512"))


def test_gamma_grid_is_one_scalar_grid_truncated_per_row():
    assert M["M3"]["gamma_grid"] == RC["methods"]["M3b"]["gamma_grid"] == [0, 1, 2, 3, 4, 5, 6]
    assert "truncated per row" in OFF["M3"]["gamma_grid"]
    assert "np.minimum(req, n_unc" in inspect.getsource(ROB._parse_gamma)          # Gamma_eff = min(Gamma, n_unc_k)
    assert M["M3"]["k_grid"] == RC["methods"]["M3a"]["k_grid"] == RC["methods"]["M3b"]["k_grid"]


def test_m3_set_kinds_and_the_primitive_box():
    us = M["M3"]["uncertainty_set"]
    assert us["kind"] == {"M3a": "box", "M3b": "budget"} == {k: OFF["M3"][k] for k in ("M3a", "M3b")}
    assert us["parameter_space"] == "primitive (a_ij, d_i)"
    assert "双线性角点" in us["construction"] and "仿射能量映射" in us["construction"]
    assert "exact image of that composition box" in RC["methods"]["M3a"]["box"]
    assert "from_factory_model" in RC["methods"]["M3a"]["box"] and "factory moments" in OFF["M3"]["box_source"]
    assert "c_ki = d_i·(K_k − a_ij)（或相应组合）的名义值" not in TEXT                # composite-coefficient wording gone


def test_test_precision_is_fixed_exact_cp_without_wilson_or_doubling():
    tp = CFG["draws"]["test_precision_rule"]
    assert tp["n_test"] == RC["streams"]["test"] == AB.ABLATION_CONFIG["streams"]["test"] \
        == OFF["test_precision"]["n_test"] == 10000
    assert tp["interval"] == OFF["test_precision"]["interval"] == "clopper_pearson_exact"
    assert tp["adaptive_doubling"] is False is OFF["test_precision"]["adaptive_doubling"]
    assert "Wilson" not in yaml.safe_dump(CFG["draws"]["mc_interval"], allow_unicode=True)
    assert "proposal" not in tp and "200,000" not in yaml.safe_dump(tp, allow_unicode=True)
    assert CFG["draws"]["validation_draws_initial"] == RC["streams"]["validation"] \
        == AB.ABLATION_CONFIG["streams"]["validation"] == OFF["validation_draws"] == 20000


def test_n_ladder_headline_and_convergence_criterion():
    m2 = M["M2"]
    assert m2["N_ladder"] == OFF["N_ladder"] == [128, 512, 1024]
    assert m2["N_final"] == OFF["N_headline"] == OFF["opt_draws"] == 1024
    assert m2["development_runs_used_N"] == RC["streams"]["N_ladder"] == AB.ABLATION_CONFIG["streams"]["N_ladder"]
    assert "2 MC SE" in m2["convergence_criterion"] and "1%" in m2["convergence_criterion"]
    assert "2 MC SE" in OFF["convergence_criterion"] and "1 %" in OFF["convergence_criterion"]
    assert m2["alpha_train"]["grid_multipliers_of_alpha"] == RC["methods"]["M2"]["alpha_train_multipliers"]


def test_m1_semantics_scale_dm_and_the_sensitivity_line():
    m1 = M["M1"]
    assert m1["margin_semantics"]["chosen"] == OFF["M1"]["margin_semantics"] == "coef_directional"
    assert m1["margin_scale"]["value"] == RC["methods"]["M1"]["margin_scale"] == OFF["M1"]["margin_scale"] == "relative"
    assert m1["grid_margin_scale"] == m1["margin_scale"]["value"]
    assert m1["grid"] == RC["methods"]["M1"]["grid"]
    dm = m1["margin_semantics"]["dm_margin"]
    assert dm["apply_to_dm"] is RC["methods"]["M1"]["apply_to_dm"] is OFF["M1"]["apply_to_dm_main"] is True
    assert dm["sensitivity_variant"]["apply_to_dm"] is RC["methods"]["M1"]["sensitivity_variant"]["apply_to_dm"] is False
    assert "不是方法条目" in dm["sensitivity_variant"]["role"]
    assert OFF["membership"]["m1_dm_off_is_entry"] is False
    assert AB.ABLATION_CONFIG["endpoint_reporting"]["m1_dm_off_is_entry"] is True            # development runs
    assert AB.M1_DM_OFF_MARKER in "M1[relative,apply_to_dm=False]"


def test_m0_mode_alphas_seed_and_membership():
    assert M["M0"]["coefficient_mode"] == RC["methods"]["M0"]["params"]["coefficient_mode"] == "nominal_point"
    rl = CFG["risk_levels"]
    assert [rl["primary_alpha"]] + rl["secondary_alphas"] == [RC["alphas"]["primary"]] + RC["alphas"]["curve"] \
        == [AB.ABLATION_CONFIG["alphas"]["primary"]] + AB.ABLATION_CONFIG["alphas"]["supplementary"]
    assert CFG["rng"]["development_seeds"][0] == RC["seed"] == AB.ABLATION_CONFIG["seed"]
    assert OFF["membership"]["membership_stat"] == "cp_upper" and OFF["membership"]["membership_stat"] in AB.MEMBERSHIP_STATS
    assert OFF["membership"]["main_event"] == AB.MAIN_EVENT_PLAN_DOMAIN                   # D-535 plan-level reading
    assert AB.ABLATION_CONFIG["endpoint_reporting"]["membership_stat"] == "rate_upper"


def test_no_aligned_item_is_left_pending_and_the_header_lists_every_change():
    assert M["M1"]["margin_semantics"]["chosen"] is not None and M["M1"]["margin_scale"]["value"] is not None
    assert M["M1"]["margin_semantics"]["dm_margin"]["apply_to_dm"] is not None
    assert M["M3"]["uncertainty_set"]["kind"] is not None and M["M2"]["N_final"] is not None
    header = TEXT.split("schema_version:", 1)[0]
    for needle in ("B-453", "cp_upper_le_alpha", "box_quantile", "300 s", "360 s", "{0..6}", "(a_ij, d_i)",
                   "10,000", "N_final", "coef_directional", "apply_to_dm", "M3a = box", "nominal_point"):
        assert needle in header, needle


def test_protocol_fields_filled_except_freeze_and_consistent_with_the_official_settings():
    """configs/protocol.yaml (batch 2): every field outside ``freeze`` is filled; ``freeze`` is consistent (unfrozen, or frozen with a record); the
    selection / membership / N / main-event entries equal the official settings."""
    proto = yaml.safe_load((REPO / "configs" / "protocol.yaml").read_text(encoding="utf-8"))

    def nulls(x, path=""):
        if isinstance(x, dict):
            return [n for k, v in x.items() for n in nulls(v, f"{path}.{k}")]
        if isinstance(x, list):
            return [n for i, v in enumerate(x) for n in nulls(v, f"{path}[{i}]")]
        return [path] if x is None else []
    assert [n for n in nulls(proto) if not n.startswith(".freeze")] == []
    fz = proto["freeze"]
    assert isinstance(fz["is_frozen"], bool)
    if fz["is_frozen"]:      # B5 (D-542): frozen -> every freeze field filled and the freeze record exists
        assert fz["timestamp"] and all(fz[k] for k in ("code_commit_or_hash", "configuration_sha256", "data_sha256",
                                                        "test_data_sha256"))
        assert (REPO / "configs" / "protocol_freeze.json").is_file()
    assert {b.split(":", 1)[0] for b in proto["freeze"]["unresolved_critical_blockers"]} == {"B-127", "B-454", "B-445"}
    assert proto["splits"]["test_previously_seen"] is True and proto["splits"]["test_seen_runs"]
    assert "no_cost_data_report_break_even_only" not in proto["information_value"]
    assert proto["units_and_decisions"]["dry_matter_mode"] == "uncertain_with_available_information"
    assert proto["risk_and_outcomes"]["membership_statistic"] == OFF["membership"]["membership_stat"]
    assert proto["constraints"]["main_event_id"] == OFF["membership"]["main_event"]
    assert proto["methods"]["screening_rule"]["statistic"] == OFF["screening"]["rule"]
    assert proto["methods"]["screening_rule"]["confidence"] == OFF["screening"]["confidence"]
    assert proto["computation"]["final_optimization_scenario_count"] == OFF["N_headline"]
    assert proto["computation"]["optimization_scenario_development_grid"] == OFF["N_ladder"]
    assert proto["methods"]["robust_uncertainty_set_kind"] == M["M3"]["uncertainty_set"]["kind"]
    assert proto["methods"]["safety_margin_development_grid"] == M["M1"]["grid"]
    assert proto["uncertainty"]["primary_evaluation_world"] == "SD-H0" and proto["methods"]["primary_training_arm"] == "MAIN9"
    assert proto["risk_and_outcomes"]["primary_target_alpha"] == CFG["risk_levels"]["primary_alpha"]
