"""Contract T9 validator: rejects undefined units, missing sources, unclear licences, pending values
in pilot/official, missing freeze hash / primary assumption, test-set fitting and smoke input
designated as empirical."""

import copy

import pytest

from ration_reliability.errors import ConfigValidationError
from ration_reliability.io import build_problem, load_problem, validate_problem_config


def _errors(cfg, mode="unit_test", path=None):
    return validate_problem_config(cfg, mode=mode, path=path).errors


def _has(errors, needle):
    return any(needle in e for e in errors), errors


def _research_cfg(toy_cfg):
    """Toy config relabelled as a non-synthetic research scenario (values become assumptions)."""
    cfg = copy.deepcopy(toy_cfg)
    cfg["dataset_status"] = "research_scenario"
    cfg["is_synthetic"] = False
    cfg["sources"] = [{"source_id": "SRC-A", "title": "declared scenario source", "is_synthetic": False,
                       "license_status": "research_use_permitted"}]

    def relabel(obj):
        if isinstance(obj, dict):
            if obj.get("status") == "synthetic_test_only":
                obj["status"] = "research_scenario_assumption"
                obj["source_id"] = "SRC-A"
                obj["rationale"] = "test-only relabelling"
            for v in obj.values():
                relabel(v)
        elif isinstance(obj, list):
            for v in obj:
                relabel(v)

    relabel(cfg)
    return cfg


def test_toy_config_is_valid_and_synthetic(toy_yaml_path):
    problem, rep = load_problem(toy_yaml_path, mode="unit_test")
    assert rep.ok and rep.n_synthetic_values > 0
    assert problem.is_synthetic and problem.dataset_status == "synthetic_test_only"
    assert len(problem.ingredients) == 5 and len(problem.nutrients) == 6
    # canonical conversion at load: 35 % DM -> 0.35; 30 g/kg EE -> 0.03; 5.44 MJ/kg -> Mcal/kg
    wet = problem.ingredients[0]
    assert wet.dm_estimate == pytest.approx(0.35)
    assert wet.composition["EE"] == pytest.approx(0.03)
    assert problem.ingredients[1].composition["ENERGY_SYN"] == pytest.approx(5.44 / 4.184)
    assert problem.prices.prices_per_kg_as_fed["syn_forage_dry"] == pytest.approx(0.25)


def test_synthetic_rejected_in_pilot_and_official(toy_yaml_path):
    for mode in ("pilot", "official"):
        with pytest.raises(ConfigValidationError) as ei:
            load_problem(toy_yaml_path, mode=mode)
        assert any("synthetic" in i for i in ei.value.issues)


def test_undefined_unit_rejected(toy_cfg):
    toy_cfg["ingredients"][0]["composition"]["CP"]["unit"] = "percent_dm"
    ok, errs = _has(_errors(toy_cfg), "undefined unit")
    assert ok, errs


def test_unit_dimension_mismatch_rejected(toy_cfg):
    toy_cfg["ingredients"][0]["composition"]["CP"]["unit"] = "Mcal/kg"
    ok, errs = _has(_errors(toy_cfg), "not allowed here")
    assert ok, errs


def test_as_fed_composition_rejected(toy_cfg):
    toy_cfg["ingredients"][0]["composition"]["CP"]["basis"] = "as_fed"
    ok, errs = _has(_errors(toy_cfg), "basis 'as_fed' not allowed")
    assert ok, errs


def test_sourced_value_needs_declared_source_and_locator(toy_cfg):
    cfg = _research_cfg(toy_cfg)
    blk = cfg["constraints"][1]["bound"]
    blk.update({"status": "sourced", "source_id": "SRC-A", "locator": None})
    ok, errs = _has(_errors(cfg), "requires a locator")
    assert ok, errs
    blk.update({"source_id": "NOT-DECLARED", "locator": "Table X"})
    ok, errs = _has(_errors(cfg), "not declared in sources")
    assert ok, errs


def test_empty_sources_rejected(toy_cfg):
    toy_cfg["sources"] = []
    ok, errs = _has(_errors(toy_cfg), "required source is empty")
    assert ok, errs


def test_pending_value_must_be_null_and_blocks_pilot(toy_cfg):
    cfg = _research_cfg(toy_cfg)
    blk = cfg["ingredients"][0]["composition"]["CP"]
    blk.update({"status": "pending_user_decision", "value": 8.0, "rationale": "needs table choice"})
    ok, errs = _has(_errors(cfg), "must have value null")
    assert ok, errs
    blk["value"] = None
    assert _errors(cfg, "unit_test") == []  # allowed in development (becomes NaN = missing)
    ok, errs = _has(_errors(cfg, "pilot"), "pending_user_decision value(s) not allowed")
    assert ok, errs


def test_assumption_requires_rationale(toy_cfg):
    cfg = _research_cfg(toy_cfg)
    cfg["constraints"][1]["bound"]["rationale"] = None
    ok, errs = _has(_errors(cfg), "requires an explicit rationale")
    assert ok, errs


def test_unclear_licence_rejected_in_pilot(toy_cfg):
    cfg = _research_cfg(toy_cfg)
    assert _errors(cfg, "pilot") == []
    cfg["sources"][0]["license_status"] = "unknown"
    ok, errs = _has(_errors(cfg, "pilot"), "licence status unclear")
    assert ok, errs


def test_official_needs_freeze_hash_and_primary_assumption(toy_cfg):
    cfg = _research_cfg(toy_cfg)
    errs = _errors(cfg, "official")
    assert _has(errs, "no frozen protocol hash")[0] and _has(errs, "primary assumption not chosen")[0], errs
    cfg["run_context"].update({"protocol_sha256": "0" * 64, "primary_assumption_id": "P_indep_baseline"})
    assert _errors(cfg, "official") == []


def test_test_set_used_for_fitting_rejected(toy_cfg):
    toy_cfg["run_context"]["fit_sets"] = ["opt", "test"]
    ok, errs = _has(_errors(toy_cfg), "test set used for fitting")
    assert ok, errs


def test_smoke_input_designated_empirical_rejected(toy_cfg):
    toy_cfg["dataset_status"] = "empirical"
    toy_cfg["is_synthetic"] = False
    errs = _errors(toy_cfg)
    assert _has(errs, "smoke input designated as empirical")[0], errs
    # the file location also betrays synthetic data
    errs2 = _errors(toy_cfg, path="/x/data/synthetic_test_only/fake.yaml")
    assert _has(errs2, "synthetic_test_only directory")[0], errs2


def test_synthetic_flag_consistency(toy_cfg):
    toy_cfg["is_synthetic"] = False
    ok, errs = _has(_errors(toy_cfg), "is_synthetic must be true iff")
    assert ok, errs


def test_unknown_key_rejected(toy_cfg):
    toy_cfg["ingredients"][0]["protein"] = 8
    ok, errs = _has(_errors(toy_cfg), "unknown keys")
    assert ok, errs


def test_duplicate_yaml_key_rejected(tmp_path, toy_yaml_path):
    text = toy_yaml_path.read_text(encoding="utf-8").replace("problem_id: engine_toy_problem_v1",
                                                             "problem_id: a\nproblem_id: b", 1)
    p = tmp_path / "dup.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigValidationError) as ei:
        load_problem(p)
    assert "duplicate YAML key" in str(ei.value)


def test_dm_basis_price_needs_semantics_and_converts(toy_cfg):
    item = toy_cfg["prices"]["items"]["syn_grain"]
    item.update({"value": 0.5, "unit": "XXX/kg", "basis": "DM"})
    ok, errs = _has(_errors(toy_cfg), "needs dm_basis_semantics")
    assert ok, errs
    # red-team C08: free text is no longer accepted -- the semantics is an enumeration
    item["dm_basis_semantics"] = "synthetic quote per kg DM at d_hat"
    ok, errs = _has(_errors(toy_cfg), "unknown dm_basis_semantics")
    assert ok, errs
    # a price settled on the measured DM of the delivered batch is a separate contract scenario
    item["dm_basis_semantics"] = "settled_on_measured_dm"
    ok, errs = _has(_errors(toy_cfg), "separate contract scenario")
    assert ok, errs
    item["dm_basis_semantics"] = "converted_with_decision_time_dm_estimate"
    problem, _ = build_problem(toy_cfg)
    assert problem.prices.prices_per_kg_as_fed["syn_grain"] == pytest.approx(0.5 * 0.88)
    assert any("kg as-fed using d_hat" in line for line in problem.prices.conversion_log)


def test_missing_price_rejected(toy_cfg):
    del toy_cfg["prices"]["items"]["syn_grain"]
    ok, errs = _has(_errors(toy_cfg), "missing prices")
    assert ok, errs


def test_constraint_class_rules_reported(toy_cfg):
    toy_cfg["constraints"][1]["constraint_class"] = "structural_hard"  # CP rule as structural
    ok, errs = _has(_errors(toy_cfg), "structural_hard constraints may not depend on composition")
    assert ok, errs


def test_zero_tolerance_is_rejected(toy_cfg):
    """Red-team C10: tolerance 0 turns rounding errors of binding rows into violations -> error (was a warning)."""
    toy_cfg["constraints"][1]["numerical_tolerance"] = 0.0
    rep = validate_problem_config(toy_cfg)
    assert not rep.ok and any("numerical_tolerance must be > 0" in e for e in rep.errors)


def test_ingredient_coefficients_loaded_with_provenance(toy_cfg):
    for ib in toy_cfg["ingredients"]:
        ib["coefficients"] = {"AC_SYN": {"value": 0.5, "unit": "1", "status": "synthetic_test_only",
                                         "source_id": "SYN-P4A-001"}}
    toy_cfg["constraints"].append({
        "constraint_id": "abs_ca_syn", "kind": "supply", "terms": {"C:AC_SYN:Ca": 1.0}, "sense": "ge",
        "bound": {"value": 10.0, "unit": "g/d", "basis": "none", "status": "synthetic_test_only",
                  "source_id": "SYN-P4A-001"},
        "constraint_class": "probabilistic_nutrition", "numerical_tolerance": 1e-6})
    problem, rep = build_problem(toy_cfg)
    assert rep.ok and problem.ingredients[0].coefficients == {"AC_SYN": 0.5}
    assert "coefficient:AC_SYN" in problem.ingredients[0].provenance
    toy_cfg["ingredients"][0]["coefficients"]["AC_SYN"]["unit"] = "%"
    ok, errs = _has(_errors(toy_cfg), "not allowed here")
    assert ok, errs
