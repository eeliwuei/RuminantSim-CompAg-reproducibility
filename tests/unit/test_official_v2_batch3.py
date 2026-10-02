"""Official-run plan batch 3 (``docs/official_run_v2_plan_20260926.md`` §1, §2 "B3", §3 tests 5 / 10 / 11a, §6 items 1-5):
``experiments/E1_cost_reliability/official_v2.py``.

What is checked:

* **test 5 -- SD-point registry** (by pattern only; no ratio value is printed, even on failure): the five declared points
  read from ``reports/sd_scaling_sources.csv`` (5 columns x 48 cells); SD-H0 = 1 everywhere; SD-S2 = the development
  S2 arrays; every point inside the declared per-cell range; the narrowing pattern of
  ``docs/uncertainty_data_layers.md`` §5.2; a changed registry stops the check with a message that names cells and
  checks, never values.  Holder side (restricted cells present): ``build_sd_point_spec`` gives the H0 specification for
  SD-H0 and the development S2 specification for SD-S2, the generic construction reproduces the S2 fingerprint, and a
  partially narrowed point keeps the H0 labels at ratio 1 and labels the narrowed cells unidentified;
* **OFFICIAL_CONFIG / official_jobs** (data only): 9 cells, 19 cell jobs + 1 diagnostics job (D-539: root 0), the matrix rule, root
  indices only (a seed is refused), the values of ``OFFICIAL_SETTINGS``, no ``RandomStreams`` is created by importing
  the module or building the jobs, no reserved root value appears in the configuration;
* **test 10 -- diagnostics** on the SYNTHETIC toy problem ``data/synthetic_test_only/engine_toy_problem_v1.yaml``
  (invented values; rows renamed to ``PN-*`` and the toy energy row made probabilistic): DIAG-T45 ``delta* = 0`` when the
  starch rows do not bind (h <= A0), ``delta*`` non-decreasing in h, infeasible above A1, the h grid from the opt stream
  only; DIAG-E1's arm has the energy row as its only probabilistic row; diagnostic rows are never comparable-set entries
  or members.  Holder side: both diagnostics on the real case at toy size from the development root (1103) -- labels,
  flags and properties only;
* **test 11a -- convergence report**: the §6.4 criterion on synthetic selection tables; a non-validation stream raises.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ration_reliability.build import dev_case as DC
from ration_reliability.build import preflight as PF
from ration_reliability.errors import LeakageError
from ration_reliability.io import build_problem
from ration_reliability.uncertainty import DrawSet

REPO = Path(__file__).resolve().parents[2]
_OV2 = REPO / "experiments" / "E1_cost_reliability" / "official_v2.py"
_AB = REPO / "experiments" / "E1_cost_reliability" / "run_endpoint_ablation.py"
TOY = REPO / "data" / "synthetic_test_only" / "engine_toy_problem_v1.yaml"
REGISTRY = REPO / "reports" / "sd_scaling_sources.csv"


def _load(path: Path, name: str):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                 # module import reads no data (no main() is called)
    return mod


OV = _load(_OV2, "official_v2_batch3_under_test")
DRV = OV.DRV


def _case_ids() -> tuple[str, ...]:
    inv = yaml.safe_load((REPO / "configs" / "dev_case_v1" / "inventory.yaml").read_text(encoding="utf-8"))
    return tuple(g["ingredient_id"] for g in inv["ingredients"])


def _registry_rows() -> tuple[list[str], list[dict]]:
    with REGISTRY.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames), list(rd)


def _ratio_strings() -> set[str]:
    """Every ratio string of the registry's point / range columns (to prove messages carry none of them)."""
    cols = [OV.sd_point_column(p) for p in OV.SD_POINTS] + ["sd_range_low", "sd_range_high"]
    _, rows = _registry_rows()
    return {r[c] for r in rows if r["row_kind"] == OV.SD_REGISTRY_ROW_KIND for c in cols if "/" in r[c]}


def _reserved_roots() -> set[int]:
    doc = yaml.safe_load((REPO / "configs" / "streams_policy.yaml").read_text(encoding="utf-8"))
    return {int(r["root_seed"]) for r in doc["reserved_formal_streams"]["roots"]}


def _contains_root(obj, roots: set[int]) -> bool:
    if isinstance(obj, dict):
        return any(_contains_root(k, roots) or _contains_root(v, roots) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return any(_contains_root(v, roots) for v in obj)
    if isinstance(obj, bool):
        return False
    if isinstance(obj, (int, float)):
        return obj in roots
    if isinstance(obj, str):
        return any(f"root={r}/" in obj or obj == str(r) for r in roots)
    return False


# =================================================================================================
# test 5: the five declared SD points (pattern only; values withheld)
# =================================================================================================

def test_registry_check_of_the_five_points_passes_with_the_declared_pattern():
    chk = OV.sd_registry_check_all(_case_ids(), repo=REPO)
    assert chk["ok"] and chk["n_cells"] == OV.N_STOCHASTIC_CELLS == 48
    below = {p: v["n_cells_below_1"] for p, v in chk["points"].items()}
    at_one = {p: v["n_cells_at_1"] for p, v in chk["points"].items()}
    # docs/uncertainty_data_layers.md §5.2: H0 none narrowed; S2 and 12MO all 48; SRC14 the 3 sourced cells; SRC12 2
    assert below == {"SD-H0": 0, "SD-S2": 48, "SD-SRC14": 3, "SD-12MO": 48, "SD-SRC12": 2}
    assert all(below[p] + at_one[p] == 48 for p in OV.SD_POINTS)
    assert all(v["within_declared_range"] for v in chk["points"].values())
    assert set(chk["points"]) == set(OV.SD_POINTS) and len(OV.SD_POINTS) == 5
    printed = json.dumps(chk)
    leaked = any(s in printed for s in _ratio_strings())
    assert not leaked, "the registry check result carries a ratio value"


def test_point_arrays_have_the_s2_layout_h0_ones_and_s2_equal_to_the_development_arrays():
    ids = _case_ids()
    _, rows = _registry_rows()
    cells = [(r["ingredient_id"], r["component"]) for r in rows if r["row_kind"] == OV.SD_REGISTRY_ROW_KIND]
    arrays = {p: OV.sd_point_ratio_arrays(p, ids, repo=REPO) for p in OV.SD_POINTS}
    shapes_ok = all(r.shape == (len(ids), len(DRV.PN)) and rd.shape == (len(ids),) for r, rd in arrays.values())
    h0_ones = bool(np.all(arrays["SD-H0"][0] == 1.0) and np.all(arrays["SD-H0"][1] == 1.0))
    r0, rd0 = DRV.s2_ratio_arrays(ids)
    s2_same = all(abs(OV._point_ratio(*arrays["SD-S2"], ids, i, c) - OV._point_ratio(r0, rd0, ids, i, c)) <= 1e-12
                  for i, c in cells)
    # cells outside the registry (the non-stochastic ingredients) keep ratio 1 at every point
    outside = [iid for iid in ids if iid not in {i for i, _ in cells}]
    outside_one = all(OV._point_ratio(*arrays[p], ids, iid, c) == 1.0 for p in OV.SD_POINTS for iid in outside
                      for c in OV.SD_POINT_COMPONENTS)
    order = all(bool(np.all(arrays[a][k] <= arrays[b][k] + 1e-12)) for a, b in
                (("SD-S2", "SD-12MO"), ("SD-12MO", "SD-SRC12"), ("SD-SRC12", "SD-H0"), ("SD-S2", "SD-SRC14"),
                 ("SD-SRC14", "SD-H0")) for k in (0, 1))
    assert shapes_ok and h0_ones and s2_same and outside and outside_one and order, \
        "SD-point arrays: layout / H0 / S2 / outside cells / order (values withheld)"
    with pytest.raises(ValueError, match="unknown SD point"):
        OV.sd_point_ratio_arrays("SD-S3", ids, repo=REPO)


def _write_registry(tmp_path: Path, mutate) -> Path:
    fields, rows = _registry_rows()
    mutate(rows)
    out = tmp_path / "reports" / "sd_scaling_sources.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return out


def _cell(rows, iid, comp):
    return next(r for r in rows if r["row_kind"] == OV.SD_REGISTRY_ROW_KIND and r["ingredient_id"] == iid
                and r["component"] == comp)


@pytest.mark.parametrize("case", ["src14_borrows", "above_range", "h0_narrowed", "s2_changed", "src12_starch",
                                  "missing_column"])
def test_a_changed_registry_stops_the_check_without_printing_a_value(tmp_path, case):
    ids = _case_ids()
    unsourced = ("soybean_hulls", "NDF")

    def mutate(rows):
        if case == "src14_borrows":                   # SRC14 must not narrow an unsourced cell
            c = _cell(rows, *unsourced)
            c["ratio_SD_SRC14"] = c["ratio_SD_S2"]
        elif case == "above_range":                   # a widening is outside [sd_range_low, sd_range_high]
            _cell(rows, DRV.CS, "NDF")["ratio_SD_12MO"] = "2"
        elif case == "h0_narrowed":
            c = _cell(rows, *unsourced)
            c["ratio_SD_H0"] = c["ratio_SD_S2"]
        elif case == "s2_changed":                    # a different narrowed arm
            c = _cell(rows, DRV.CS, "DM")
            c["ratio_SD_S2"] = c["ratio_SD_12MO"]
        elif case == "src12_starch":                  # SRC12 has no starch ratio (no 12-month value)
            c = _cell(rows, DRV.CS, "starch")
            c["ratio_SD_SRC12"] = c["ratio_SD_S2"]
        elif case == "missing_column":
            for r in rows:
                r.pop("ratio_SD_SRC12", None)
    reg = _write_registry(tmp_path, mutate)
    if case == "missing_column":
        fields, rows = _registry_rows()
        fields = [f for f in fields if f != "ratio_SD_SRC12"]
        with reg.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    with pytest.raises(SystemExit) as exc:
        OV.sd_registry_check_all(ids, repo=tmp_path, registry=Path("reports") / "sd_scaling_sources.csv")
    msg = str(exc.value)
    assert "SD registry check (five points) failed" in msg
    leaked = any(s in msg for s in _ratio_strings())
    assert not leaked, f"{case}: the failure message carries a ratio value"


def test_the_registry_path_is_one_constant():
    """B-454 may relocate the ratios: the readers default to one constant and take the path as an argument."""
    import inspect
    assert OV.SD_SCALING_SOURCES.as_posix() == "reports/sd_scaling_sources.csv"
    for fn in (OV.sd_point_ratio_arrays, OV.sd_point_cell_overrides, OV.build_sd_point_spec, OV.sd_registry_check_all):
        assert inspect.signature(fn).parameters["registry"].default == OV.SD_SCALING_SOURCES
    assert OV.OFFICIAL_CONFIG["sd_registry"] == OV.SD_SCALING_SOURCES.as_posix()


needs_restricted = pytest.mark.skipif(not (REPO / DC.DEV_CASE_V1.constants_file).is_file(),
                                      reason="restricted dev_case inputs not present (holder-side check only)")


@pytest.fixture(scope="module")
def dev_cells():
    AB = _load(_AB, "run_endpoint_ablation_for_batch3_cells")
    rep = PF.preflight_dev_case(REPO, driver="run_dev_case_v1", extra=AB.EXTRA_REQUIREMENTS)
    if rep["exit_code"] != PF.EXIT_READY:
        pytest.skip(f"dev_case inputs not READY (preflight {rep['status']}); holder-side check skipped")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        from ration_reliability.hashing import file_sha256
        cells = DRV.read_cells()                     # loader (restricted); nothing is printed
        yield {"cells": cells, "sha": file_sha256(DRV.CELLS_CSV), "ids": _case_ids()}
    finally:
        os.chdir(cwd)


@needs_restricted
def test_holder_sd_point_specs_generalise_the_h0_and_s2_builders(dev_cells):
    cells, sha, ids = dev_cells["cells"], dev_cells["sha"], dev_cells["ids"]
    h0 = OV.build_sd_point_spec("SD-H0", ids, cells, sha, repo=REPO)
    h0_same = h0.fingerprint() == DRV.build_h0_spec(ids, cells, sha).fingerprint()
    s2_dev = DRV.build_s2_spec(ids, cells, sha)
    s2 = OV.build_sd_point_spec("SD-S2", ids, cells, sha, repo=REPO)
    over, r, rd = OV.sd_point_cell_overrides("SD-S2", ids, cells, sha, repo=REPO)
    generic = OV.sd_point_spec_from_overrides("SD-S2", ids, cells, over, r, rd, spec_id=s2_dev.spec_id,
                                              notes=s2_dev.notes)
    s2_same = s2.fingerprint() == s2_dev.fingerprint()
    generic_same = generic.fingerprint() == s2_dev.fingerprint()
    assert h0_same and s2_same and generic_same, "SD-H0 / SD-S2 differ from the development builders"
    # a partially narrowed point: ratio-1 cells keep the H0 labels, narrowed cells are unidentified
    base = DRV.cell_overrides(cells, sha)
    stoch = {k for k, c in cells.items() if str(c["is_stochastic"]) == "True"}
    for point, n_narrow in (("SD-SRC14", 3), ("SD-SRC12", 2), ("SD-12MO", 48)):
        over, r, rd = OV.sd_point_cell_overrides(point, ids, cells, sha, repo=REPO)
        narrowed = {k for k in stoch if OV._point_ratio(r, rd, ids, *k) < 1.0}
        relabelled = {k for k in stoch if over[k].get("variance_basis") == "unidentified"
                      and over[k]["provenance_status"] == "research_scenario_assumption" and over[k]["source_id"] is None}
        kept = all(over[k] == base[k] for k in stoch - narrowed)
        assert len(narrowed) == n_narrow and relabelled == narrowed and kept, f"{point}: label rule"
        sp = OV.build_sd_point_spec(point, ids, cells, sha, repo=REPO)
        assert sp.purpose == "sensitivity_scenario" and point.replace("-", "") in sp.spec_id
        assert sp.fingerprint() not in {h0.fingerprint(), s2_dev.fingerprint()}


# =================================================================================================
# OFFICIAL_CONFIG and the job matrix (data only)
# =================================================================================================

def test_official_config_cells_streams_limits_and_alphas():
    C, S = OV.OFFICIAL_CONFIG, OV.OFFICIAL_SETTINGS
    cells = {c["cell_id"]: c for c in C["cells"]}
    assert len(cells) == 9
    assert {(c["sd"], c["objective"]) for c in cells.values()} == (
        {(sd, "MAIN9") for sd in OV.SD_POINTS} | {(sd, a) for a in ("FULL11", "PART6P5") for sd in ("SD-H0", "SD-S2")})
    assert C["streams"]["opt"] == 1024 and C["streams"]["N_ladder"] == [128, 512, 1024] == S["N_ladder"]
    assert C["streams"]["validation"] == 20000 and C["streams"]["test"] == 10000
    assert C["streams"]["adaptive_doubling"] is False
    assert C["solver"]["time_limit_s"] == 300.0 == C["solver"]["frontier_time_limit_s"]
    assert C["diagnostics"]["min_violation"]["time_limit_s"] == {"128": 60.0, "512": 180.0, "1024": 360.0}
    assert (C["alphas"]["primary"], C["alphas"]["supplementary"]) == (0.05, [0.10, 0.01])
    assert C["evaluation"]["evaluation_worlds"] == list(OV.SD_POINTS) and C["evaluation"]["primary_world"] == "SD-H0"
    assert C["evaluation"]["reference_constraints"] == OV.REFERENCE_CSV_V2.as_posix()
    er = C["endpoint_reporting"]
    assert (er["screening_rule"], er["screening_confidence"], er["membership_stat"], er["main_event"],
            er["m1_dm_off_is_entry"]) == ("cp_upper_le_alpha", 0.95, "cp_upper", OV.MAIN_EVENT_PLAN_DOMAIN, False)
    assert C["methods"]["grids"] == DRV.RUN_CONFIG["methods"]                       # no new method or grid
    conv = C["convergence"]
    assert conv["N_pair"] == [512, 1024] and conv["mc_se_multiplier"] == 2.0 and conv["relative_cost_tolerance"] == 0.01
    assert set(C["diagnostics"]["official_jobs"]) == {"DIAG-T45", "DIAG-E1"}
    assert C["diagnostics"]["DIAG-T45"]["relaxed_rows"] == ["PN-T4", "PN-T5"]
    assert C["diagnostics"]["DIAG-E1"]["N_ladder"] == [128, 512, 1024]


def test_official_config_matches_the_ablation_driver_shape_and_its_1024_time_limit():
    AB = _load(_AB, "run_endpoint_ablation_for_batch3_cfg")
    for arm in ("MAIN9", "FULL11", "PART6P5"):                     # same training events as the development arms
        assert OV.OFFICIAL_CONFIG["objective_arms"][arm]["training_event"] == \
            AB.ABLATION_CONFIG["objective_arms"][arm]["training_event"]
    assert AB.ABLATION_CONFIG["diagnostics"]["min_violation"]["time_limit_s"]["1024"] == 360.0
    assert AB.OV2.OFFICIAL_CONFIG["schema"] == OV.OFFICIAL_CONFIG["schema"]
    import inspect
    for fn, name in ((AB.plan, "config"), (AB.build_context, "official"), (AB.run_cell, "config"),
                     (AB.cell_rows, "config"), (AB.comparable_sets, "config"), (AB.per_method_report, "config")):
        assert inspect.signature(fn).parameters[name].default is None     # development default unchanged


def test_official_jobs_matrix_uses_root_indices_only():
    jobs = OV.official_jobs()
    cells = [j for j in jobs if j["kind"] == "cell"]
    diags = [j for j in jobs if j["kind"] == "diagnostics"]
    assert (len(cells), len(diags), len(jobs)) == (19, 1, 20)             # D-539: diagnostics on root 0 only
    by_root = {k: {j["cell_id"] for j in cells if j["root_k"] == k} for k in (0, 1, 2)}
    main9 = {f"{sd.replace('-', '')}_MAIN9" for sd in OV.SD_POINTS}
    assert len(by_root[0]) == 9 and by_root[1] == by_root[2] == main9
    assert {j["root_k"] for j in diags} == {0}
    assert all(set(j["diagnostics"]) == {"DIAG-T45", "DIAG-E1"} for j in diags)
    assert len({j["job_id"] for j in jobs}) == 20
    for bad in ([3], [0, 0], [-1], [True], [1103]):
        with pytest.raises(ValueError):
            OV.official_jobs(bad)
    assert [j["root_k"] for j in OV.official_jobs([2]) if j["kind"] == "cell"] == [2] * 5
    roots = _reserved_roots()
    assert len(roots) == 3
    leaked = _contains_root(OV.OFFICIAL_CONFIG, roots) or _contains_root(jobs, roots)
    assert not leaked, "a reserved root value appears in OFFICIAL_CONFIG or the job list"


def test_importing_official_v2_and_building_the_jobs_creates_no_random_streams(monkeypatch):
    import ration_reliability.uncertainty as U
    calls = []
    orig = U.RandomStreams.__init__

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)
    monkeypatch.setattr(U.RandomStreams, "__init__", spy)
    mod = _load(_OV2, "official_v2_batch3_streams_spy")
    mod.official_jobs()
    copy.deepcopy(mod.OFFICIAL_CONFIG)
    assert calls == []


def test_official_context_refuses_draws_without_a_development_seed(monkeypatch):
    AB = _load(_AB, "run_endpoint_ablation_for_batch3_seed")
    import ration_reliability.uncertainty as U
    calls = []
    monkeypatch.setattr(U.RandomStreams, "__init__", lambda self, *a, **k: calls.append(1))
    monkeypatch.setattr(AB, "RandomStreams", U.RandomStreams)
    with pytest.raises(SystemExit, match="explicit development seed"):
        AB.build_context({"opt": 16, "N_ladder": [16], "validation": 500, "test": 500}, draw=True,
                         official=OV.OFFICIAL_CONFIG)
    for r in sorted(_reserved_roots()):
        with pytest.raises(SystemExit) as exc:
            AB.build_context({"opt": 16, "N_ladder": [16], "validation": 500, "test": 500}, draw=True,
                             official=OV.OFFICIAL_CONFIG, seed=r)
        leaked = str(r) in str(exc.value)
        assert "reserved" in str(exc.value) and not leaked, "the refusal must withhold the root"
    assert calls == []                                  # refused before any RandomStreams exists


# =================================================================================================
# test 10: diagnostics (SYNTHETIC toy problem; invented values)
# =================================================================================================
_TOY_RENAME = {"cp_min": "PN-CP-MIN", "cp_max": "PN-CP-MAX", "ndf_min": "PN-NDF-MIN", "ndf_plus_2fndf": "PN-NDF2F",
               "starch_max": "PN-T4", "starch_vs_fndf": "PN-T5", "ee_max": "PN-EE-MAX", "ca_min": "PN-CA-MIN",
               "energy_supply_diag": "PN-ENERGY"}
TOY_ENERGY = "PN-ENERGY"


@pytest.fixture(scope="module")
def toy():
    """The synthetic toy with ``PN-*`` row ids (the planned-arm naming rule) and its energy row probabilistic."""
    cfg = yaml.safe_load(TOY.read_text(encoding="utf-8"))
    for c in cfg["constraints"]:
        if c["constraint_id"] in _TOY_RENAME:
            c["constraint_id"] = _TOY_RENAME[c["constraint_id"]]
            c["constraint_class"] = "probabilistic_nutrition"
            c.setdefault("dm_source", "scenario")
    problem, rep = build_problem(cfg, mode="smoke")
    assert rep.ok, rep.errors
    return cfg, problem


def _toy_draws(problem, stream: str, S: int, seed: int = 7) -> DrawSet:
    rng = np.random.default_rng(seed)
    th0, d0 = problem.nominal_theta(), problem.dm_estimates()
    th = np.maximum(th0[None] * (1.0 + 0.03 * rng.standard_normal((S,) + th0.shape)), 0.0)
    d = np.clip(d0[None] * (1.0 + 0.03 * rng.standard_normal((S,) + d0.shape)), 0.05, 1.0)
    return DrawSet(th, d, stream=stream, stream_id=f"synthetic/{stream}/0", model_id="SYN", model_fingerprint="syn",
                   ingredient_ids=tuple(problem.ingredient_ids), nutrient_ids=tuple(problem.nutrient_ids),
                   is_synthetic=True)


def test_t45_delta_is_zero_when_not_binding_monotone_and_infeasible_above_a1(toy):
    _, p = toy
    base = OV.diag_t45_headroom(p, None, {}, h_grid=[0.0], energy_row=TOY_ENERGY)
    H0, H1 = base["A0"]["headroom_max_canonical"], base["A1"]["headroom_max_canonical"]
    assert base["A0"]["status"] == base["A1"]["status"] == "optimal" and base["A1_headroom_ge_A0"]
    assert H1 > H0 > 0.0                                   # the toy's starch rows cap the headroom
    grid = [0.0, 0.5 * H0, H0 * (1 - 1e-6), H0 + 0.25 * (H1 - H0), H0 + 0.5 * (H1 - H0), H1 * (1 - 1e-6), H1 + 1.0]
    r = OV.diag_t45_headroom(p, None, {}, h_grid=grid, energy_row=TOY_ENERGY, tag="t10")
    w = r["worlds"]["explicit_h_grid"]
    a2 = sorted(w["A2"], key=lambda x: x["h_canonical"])
    assert [x["feasible"] for x in a2] == [True] * 6 + [False]
    assert all(x["delta_star_canonical"] <= OV.DELTA_ZERO_TOL and x["binding"] is False for x in a2[:3])
    assert all(x["binding"] is True for x in a2[3:6])
    ds = [x["delta_star_canonical"] for x in a2[:6]]
    assert all(b >= a for a, b in zip(ds, ds[1:])) and ds[3] < ds[4] < ds[5]
    assert w["delta_monotone_in_h"] and w["delta_zero_iff_h_le_A0"]
    assert a2[-1]["status"] == "proven_infeasible" and not a2[-1]["has_ration"]
    # natural unit: the relaxed rows' declared unit (% DM) -- delta in percentage points = canonical / unit factor
    assert r["delta_unit"] == "%" and a2[4]["delta_star"] == pytest.approx(ds[4] * 100.0)
    assert r["flag"] == "diagnostic" and r["comparable_set_member"] is False and r["planned_dm_row"] == "dm_offer"
    assert all(":DIAG-T45:" in x["label"] for x in r["rations"])
    # every stage-2 ration meets the headroom it was asked for (nominal energy margin >= h - tiny slack)
    for x in r["rations"]:
        if x["diag_lp"] != "A2":
            continue
        m = OV._energy_margins(p, TOY_ENERGY, x["decision"].q_as_fed, p.nominal_theta()[None], p.dm_estimates()[None])
        assert float(m[0]) >= x["diag_h_canonical"] - 1e-6 * max(1.0, x["diag_h_canonical"])


def test_t45_h_grid_comes_from_the_opt_stream_only(toy):
    _, p = toy
    opt = _toy_draws(p, "opt", 64)
    r = OV.diag_t45_headroom(p, None, {"SYN-A": {"draws": {"opt": opt}}}, energy_row=TOY_ENERGY)
    w = r["worlds"]["SYN-A"]
    spec = OV.OFFICIAL_CONFIG["diagnostics"]["DIAG-T45"]["h_grid"]
    # D-539: {0, factor x Q} united with the equal steps j x A1 / n (j = 1..n) covering [0, A1]
    assert w["n_h"] == len(spec["alphas"]) * len(spec["factors"]) + 1 + spec["a1_equal_subdivisions"]
    a1 = r["A1"]["headroom_max_canonical"]
    steps = sorted(x["h_canonical"] for x in w["A2"] if x["source"] == "A1_equal_subdivision")
    assert steps == pytest.approx([a1 * j / spec["a1_equal_subdivisions"]
                                   for j in range(1, spec["a1_equal_subdivisions"] + 1)])
    assert all("h=" in x["label"] and "e-" not in x["label"].split("h=", 1)[1] for x in w["A2"] if x.get("label"))
    assert w["shortfall"]["m0_status"] == "optimal" and w["shortfall"]["n_defined"] == 64
    q = w["shortfall_quantiles_canonical"]
    hs = {(x["alpha"], x["factor"]): x["h_canonical"] for x in w["A2"] if x["source"] == "M0_shortfall_quantile"}
    assert all(hs[(a, f)] == pytest.approx(max(0.0, f * q[str(a)])) for a in spec["alphas"] for f in spec["factors"])
    assert q[str(0.01)] >= q[str(0.05)] >= q[str(0.10)]                   # higher quantile for smaller alpha
    assert w["delta_monotone_in_h"] and w["delta_zero_iff_h_le_A0"]
    for stream in ("validation", "test"):
        with pytest.raises(LeakageError):
            OV.diag_t45_headroom(p, None, {"SYN-A": {"draws": {"opt": _toy_draws(p, stream, 16)}}},
                                 energy_row=TOY_ENERGY)
    with pytest.raises(ValueError, match="no opt draws"):
        OV.diag_t45_headroom(p, None, {"SYN-A": {"record": {}}}, energy_row=TOY_ENERGY)


def test_e1_arm_has_the_energy_row_as_its_only_probabilistic_row_and_runs_per_N(toy):
    cfg, p = toy
    AB = _load(_AB, "run_endpoint_ablation_for_batch3_e1")
    _, p_e1, arm = OV.diag_e1_arm(cfg, p, planned_arm_cfg=AB.planned_arm_cfg, energy_row=TOY_ENERGY, mode="smoke")
    from ration_reliability.datamodel import ConstraintClass
    prob = [c for c, k in zip(p_e1.compiled.constraint_ids, p_e1.compiled.classes)
            if k is ConstraintClass.PROBABILISTIC_NUTRITION]
    assert prob == [TOY_ENERGY] == arm["probabilistic_rows"]
    assert len(arm["planned_rows"]) == 8 and all(f"SH-PLAN-{c[3:]}" in p_e1.compiled.constraint_ids
                                                 for c in arm["planned_rows"])
    opt = _toy_draws(p, "opt", 32)
    scored = []

    def fake_score(dec):
        scored.append(dec.method_id)
        return {"W1": {"q_hash": f"h{len(scored)}", OV.MAIN_EVENT_PLAN_DOMAIN + "_rate_upper": 0.0,
                       OV.MAIN_EVENT_PLAN_DOMAIN + "_cp_upper": 0.0, "structural_ok": True,
                       "cost_usd_per_head_d": 1.0}}
    e1 = OV.diag_e1_energy_single_row(cfg, p, {"SYN-A": {"draws": {"opt": opt}}}, planned_arm_cfg=AB.planned_arm_cfg,
                                      Ns=[16, 32], alphas=[0.10, 0.25], time_limits={"16": 10.0, "32": 10.0},
                                      m2_time_limit_s=10.0, energy_row=TOY_ENERGY, mode="smoke", score=fake_score,
                                      tag="t10")
    per_n = e1["worlds"]["SYN-A"]["per_N"]
    assert [x["N"] for x in per_n] == [16, 32]
    assert all(x["min_violation"]["label"].startswith("training_scenarios_only") for x in per_n)
    assert all(len(x["M2"]) == 2 for x in per_n)
    assert e1["flag"] == "diagnostic" and e1["comparable_set_member"] is False
    assert e1["rows"] and all(r["ration_kind"] == "diagnostic" and r["comparable_set_member"] is False
                              and ":DIAG-E1[SYN-A]:M2[N=" in r["label"] and r["target_alpha"] is None
                              for r in e1["rows"])
    with pytest.raises(LeakageError):
        OV.diag_e1_energy_single_row(cfg, p, {"SYN-A": {"draws": {"opt": _toy_draws(p, "validation", 32)}}},
                                     planned_arm_cfg=AB.planned_arm_cfg, Ns=[16], alphas=[0.1],
                                     time_limits={"16": 5.0}, m2_time_limit_s=5.0, energy_row=TOY_ENERGY, mode="smoke")


def _method_row(cell, label, world, rate, cost, alpha=None):
    ev = OV.MAIN_EVENT_PLAN_DOMAIN
    return {"cell_id": cell, "label": label, "ration_kind": "method", "has_ration": True, "structural_ok": True,
            "evaluation_world": world, "q_hash": f"q-{cell}-{label}", "cost_usd_per_head_d": cost,
            "target_alpha": alpha, f"{ev}_rate_upper": rate, f"{ev}_cp_upper": rate,
            "main_reference_rate_upper": rate, "main_reference_cp_upper": rate,
            f"{OV.MAIN_EVENT_PLAN_DOMAIN}_rate_lower": 0.0,
            "main_reference_t51_verdicts_used_out_of_domain_rate_upper": rate,
            "main_reference_t51_verdicts_used_out_of_domain_cp_upper": rate}


def test_diagnostic_rows_are_never_comparable_set_entries_or_members():
    AB = _load(_AB, "run_endpoint_ablation_for_batch3_sets")
    C = OV.OFFICIAL_CONFIG
    worlds = C["evaluation"]["evaluation_worlds"]
    method_rows = []
    for w in worlds:
        method_rows.append(_method_row("SDH0_MAIN9", "SDH0_MAIN9:M0", w, 0.5, 10.0))
        method_rows.append(_method_row("SDH0_MAIN9", "SDH0_MAIN9:M2[N=1024]@alpha=0.05", w, 0.001, 11.0, 0.05))
    rations = [{"label": "root0:DIAG-T45:A0", "decision": object(), "diag_lp": "A0"}]

    class _Dec:
        ingredient_ids = ("a",)
        q_as_fed = np.array([1.0])
    rations[0]["decision"] = _Dec()
    diag_rows = OV.score_diagnostic_rations(
        rations, lambda dec: {w: {"q_hash": "q-diag", "cost_usd_per_head_d": 1.0, "structural_ok": True,
                                   f"{OV.MAIN_EVENT_PLAN_DOMAIN}_rate_upper": 0.0,
                                   f"{OV.MAIN_EVENT_PLAN_DOMAIN}_cp_upper": 0.0} for w in worlds},
        diagnostic_id="DIAG-T45", tag="root0")
    assert len(diag_rows) == len(worlds) and all(r["ration_kind"] == "diagnostic" for r in diag_rows)
    kw = dict(membership_stat="cp_upper", main_event=OV.MAIN_EVENT_PLAN_DOMAIN, m1_dm_off_is_entry=False, config=C)
    with_diag = AB.comparable_sets(copy.deepcopy(method_rows) + copy.deepcopy(diag_rows), **kw)
    without = AB.comparable_sets(copy.deepcopy(method_rows), **kw)
    assert len(with_diag) == len(worlds) * 3                # five official worlds x three alphas
    for a, b in zip(with_diag, without):
        assert (a["n_entries_attempted"], a["n_members"]) == (b["n_entries_attempted"], b["n_members"])
        assert all("DIAG" not in m["label"] for m in a["members"])
    assert any(s["n_members"] for s in with_diag)           # the diagnostic would have met the rule; it is not a member
    # the ablation's other descriptive readers skip :DIAG- labels as well
    assert all(":DIAG-" in r["label"] for r in diag_rows)


@pytest.fixture(scope="module")
def official_tiny():
    AB = _load(_AB, "run_endpoint_ablation_for_batch3_holder")
    rep = PF.preflight_dev_case(REPO, driver="run_dev_case_v1", extra=AB.EXTRA_REQUIREMENTS)
    if rep["exit_code"] != PF.EXIT_READY:
        pytest.skip(f"dev_case inputs not READY (preflight {rep['status']}); holder-side check skipped")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        ctx = AB.build_context({"opt": 16, "N_ladder": [16], "validation": 500, "test": 500}, draw=True,
                               official=OV.OFFICIAL_CONFIG, seed=int(AB.ABLATION_CONFIG["seed"]))   # development root
        yield AB, ctx
    finally:
        os.chdir(cwd)


@needs_restricted
def test_holder_both_diagnostics_on_the_official_context_at_toy_size(official_tiny):
    """Development root 1103, toy sizes (opt 16, test 500): a pipeline check of the wiring, not a result.  Only labels,
    flags, statuses and booleans are asserted -- no value is printed."""
    AB, ctx = official_tiny
    assert set(ctx["worlds"]) == set(OV.SD_POINTS) and ctx["registry"].check()["ok"] and ctx["official"] is True
    assert ctx["table"].is_v2 and ctx["problems"]["FULL11"].problem_id == "dev_case_v1|v2"
    score = AB.reference_scorer(ctx)
    t45 = OV.diag_t45_headroom(ctx["problems"]["FULL11"], ctx["lin"], ctx["worlds"], score=score, tag="rehearsal")
    assert t45["A0"]["status"] == "optimal" and t45["A1"]["status"] == "optimal" and t45["A1_headroom_ge_A0"]
    assert t45["planned_dm_row"] == "SH-DM-PLAN" and t45["delta_unit"] == "%"
    for sd, w in t45["worlds"].items():
        assert w["delta_monotone_in_h"] is True and w["delta_zero_iff_h_le_A0"] is True, sd
        assert w["shortfall"]["m0_status"] == "optimal"
    rows = t45["rows"]
    assert rows and {r["ration_kind"] for r in rows} == {"diagnostic"}
    assert {r["evaluation_world"] for r in rows} == set(OV.SD_POINTS)                  # scored on all five worlds
    assert all(r["n_test"] == 500 and r["q_hash"] for r in rows)
    e1 = OV.diag_e1_energy_single_row(ctx["cfg0_v2"], ctx["problems"]["FULL11"], ctx["worlds"],
                                      planned_arm_cfg=AB.planned_arm_cfg, Ns=[16], alphas=[0.05],
                                      time_limits={"16": 30.0}, m2_time_limit_s=60.0, score=score, tag="rehearsal")
    assert e1["arm"]["probabilistic_rows"] == [OV.ENERGY_ROW_ID] and len(e1["arm"]["planned_rows"]) == 10
    assert e1["arm"]["problem_id"] == "dev_case_v1|v2|E1" and e1["arm"]["validator_ok"]
    assert set(e1["worlds"]) == set(OV.SD_POINTS)
    for w in e1["worlds"].values():
        assert w["per_N"][0]["min_violation"]["status"] in ("optimal", "time_limit")
    kw = dict(membership_stat="cp_upper", main_event=OV.MAIN_EVENT_PLAN_DOMAIN, m1_dm_off_is_entry=False,
              config=OV.OFFICIAL_CONFIG)
    comp = AB.comparable_sets(rows + e1.get("rows", []), **kw)
    assert sum(c["n_entries_attempted"] for c in comp) == 0 and sum(c["n_members"] for c in comp) == 0


# =================================================================================================
# test 11a: convergence report (opt / validation only)
# =================================================================================================

def _sel(status, value, rate, cost, n=20000, stream="root=1103/validation/0"):
    cands = [{"value": v, "status": "optimal", "rate_upper": r, "cost": c} for v, r, c in
             ((value, rate, cost), (0.9, 0.5, 1.0))]
    return {"status": status, "selected_value": value if status == "selected" else None, "validation_n_draws": n,
            "validation_stream_id": stream, "candidates": cands}


def _tables(sel512, sel1024, alpha=0.05, cell="SDH0_MAIN9"):
    return [{"label": f"{cell}:M0", "method_id": "M0_nominal", "target_alpha": None, "selection": None},
            {"label": f"{cell}:M2[N=512]@alpha={alpha:g}", "method_id": "M2_joint_chance_saa", "target_alpha": alpha,
             "selection": sel512},
            {"label": f"{cell}:M2[N=1024]@alpha={alpha:g}", "method_id": "M2_joint_chance_saa", "target_alpha": alpha,
             "selection": sel1024},
            {"label": f"{cell}:M3a@alpha={alpha:g}", "method_id": "M3a_box_robust", "target_alpha": alpha,
             "selection": _sel("selected", 2.0, 0.01, 12.0)}]


def test_convergence_report_applies_the_declared_criterion():
    se = np.sqrt(0.04 * 0.96 / 20000)
    cases = {
        "converged": (_sel("selected", 0.0375, 0.040, 10.00), _sel("selected", 0.0375, 0.040 + 1.5 * se, 10.05)),
        "rate": (_sel("selected", 0.0375, 0.040, 10.00), _sel("selected", 0.0375, 0.040 + 3.0 * se, 10.00)),
        "cost": (_sel("selected", 0.0375, 0.040, 10.00), _sel("selected", 0.0375, 0.040, 10.20)),
        "status": (_sel("selected", 0.0375, 0.040, 10.00), _sel("not_met", None, None, None)),
        "none": (_sel("not_met", None, None, None), _sel("not_met", None, None, None)),
    }
    tables = {(0, f"CELL_{k}"): _tables(a, b, cell=f"CELL_{k}") for k, (a, b) in cases.items()}
    tables[(1, "CELL_missing")] = _tables(_sel("selected", 0.0375, 0.04, 10.0), None)[:2]
    rep = OV.convergence_report(tables)
    by = {r["cell_id"]: r for r in rep["rows"]}
    assert by["CELL_converged"]["verdict"] == "demonstrated" and by["CELL_converged"]["converged"] is True
    assert by["CELL_converged"]["rate_criterion_met"] and by["CELL_converged"]["cost_criterion_met"]
    assert by["CELL_rate"]["verdict"] == "not_demonstrated" and not by["CELL_rate"]["rate_criterion_met"]
    assert by["CELL_cost"]["verdict"] == "not_demonstrated" and not by["CELL_cost"]["cost_criterion_met"]
    assert by["CELL_status"]["verdict"] == "not_demonstrated" and by["CELL_status"]["status_equal"] is False
    assert by["CELL_none"]["verdict"] == "status_equal_no_selection" and by["CELL_none"]["converged"] is False
    assert by["CELL_missing"]["verdict"] == "not_applicable"
    assert rep["test_stream_read"] is False and rep["N_pair"] == [512, 1024]
    assert sum(rep["counts"].values()) == len(rep["rows"]) == 6          # M0 / M3a are not read
    # the report carries no absolute cost and no stream id (stream ids contain the root)
    text = json.dumps(rep)
    assert "root=" not in text and "10.05" not in text and "cost_usd" not in text
    # list form = mapping form
    rep2 = OV.convergence_report([{"root_k": k[0], "cell_id": k[1], "tables": t} for k, t in tables.items()])
    assert rep2["rows"] == rep["rows"]


@pytest.mark.parametrize("stream", ["root=1103/test/0", "root=1103/opt/0", "", None])
def test_convergence_report_reads_validation_selections_only(stream):
    t = _tables(_sel("selected", 0.0375, 0.04, 10.0, stream=stream), _sel("selected", 0.0375, 0.04, 10.0))
    with pytest.raises(ValueError, match="validation selections only") as exc:
        OV.convergence_report({(0, "SDH0_MAIN9"): t})
    assert "1103" not in str(exc.value)                                   # the stream id (root) is withheld
