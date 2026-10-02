"""Reference problem v2: the Table 5-1 starch-source premise as a planning row of every arm (official-run plan, batch 1),
the official configuration, the five declared SD points and the two declared diagnostics (batch 3), and the official
path (batch 4): the v2 specification pin, the protocol freeze record and its gate (:func:`require_protocol_frozen`, the
only minter of the reserved-root token), the output route, the job runner (:func:`run_jobs`) and the merge
(:func:`merge_jobs`).

Batch 1 (below): the premise planning row.  Nothing in that part draws, solves, reads a data file or writes one; the
builders take a problem configuration (``cfg0``) and its built problem as arguments.

Batch 3 (the second half of this module; ``docs/official_run_v2_plan_20260926.md`` §1, §2 "B3", §3 tests 5 / 10 / 11a,
§6 items 1-5): the SD-point arrays read from the ratio registry :data:`SD_SCALING_SOURCES` and their registry check
(:func:`sd_point_ratio_arrays`, :func:`build_sd_point_spec`, :func:`sd_registry_check_all`); :data:`OFFICIAL_CONFIG`
(data only) and the job matrix :func:`official_jobs`; the diagnostics DIAG-T45 (:func:`diag_t45_headroom`) and DIAG-E1
(:func:`diag_e1_energy_single_row`), always labelled ``diagnostic`` and never comparable-set members; the convergence
report (:func:`convergence_report`, opt / validation only).  **No function here creates a ``RandomStreams``**: a job
of the driver resolves its reserved root only after ``require_protocol_frozen()`` has minted the token (batch 4);
``official_jobs`` carries root *indices* ``k`` (0, 1, 2), never a seed.  D-539 (2026-09-27): the diagnostics job runs
on root 0 only and trains in SD-H0 and SD-S2 only; the DIAG-T45 h grid also covers [0, A1] in equal steps.  The ablation driver wires the official context and the ``--official-plan``
dry run (``run_endpoint_ablation.build_context(..., official=OFFICIAL_CONFIG)``, ``official_plan``).

Why (decision package ``manuscript/draft_methods_v0/投稿就绪评估_20260926.md`` §3.1 option B; DECISIONS.md D-533)
-------------------------------------------------------------------------------------------------------------
NASEM (2021) Table 5-1 (printed p.63) states its forage-NDF and starch limits for diets fed as a TMR, with forage of
adequate particle size and **dry ground corn as the predominant starch source**.  In reference problem v1 the premise
was only checked after the fact (per drawn state, ``nutrition/domain.py``), so the evaluation event contained a
condition that no method was asked to meet.  Reference problem v2 imposes the premise **on the plan** in every
objective arm and every method (M0 included):

    SH-PLAN-T51-DGC-SHARE:   sum_i q_i d_hat_i (1{i in dry ground corn} - tau) St_hat_i  >=  0,   tau = 0.50

with the decision-time DM estimates ``d_hat`` and the nominal (table-value) starch ``St_hat`` -- the same construction
as the ``SH-PLAN-*`` rows of ``run_endpoint_ablation.planned_arm_cfg`` and ``run_dev_case_v1.s2_problem_cfg`` (a
per-ingredient ``C:<coef>:DM`` coefficient, ``structural_hard``, ``dm_source = decision_estimate``, bound 0, tolerance
as the diagnostic row).  ``tau = 0.50`` is a project research assumption (the source gives no number for
"predominant"); it equals the primary Table 5-1 domain variant ``T51-DGC-0.50``, and at ``tau = 0.5`` the coefficients
equal the nominal content of the existing diagnostic row ``DIAG-T51-DGC-STARCH-SHARE`` (which stays a per-state
diagnostic).  The row is inserted into ``cfg0`` *before* the arms are derived, so FULL11 (``cfg0`` itself), PART6P5
(``s2_problem_cfg(cfg0_v2, ...)``) and MAIN9 (``planned_arm_cfg(cfg0_v2, ...)``) all inherit it unchanged.

The decision was taken on 2026-09-26 **after** the development results had been seen (disclosed in
``docs/reference_problem_v2.md``); it changes the problem, so v2 is a new reference problem with its own table
(``configs/dev_case_v1/reference_constraints_v2.csv``) and, later, its own freeze pin.  v1 and its results stay
development history.

Restricted values: the coefficients are table values times a weight, so they are kept in memory only.  The row's
planned margin returns restricted quantities (``d_hat`` and nominal starch), so it joins the restricted-only list
(:data:`REDACT_NOMINAL_V2`).  Nothing in this module prints or writes a value.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import yaml

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
E0 = REPO / "experiments" / "E0_verification"
for _p in (REPO / "src", E0):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_dev_case_v1 as DRV  # noqa: E402  (restricted-only list of the dev-case driver; module import reads no data)
from ration_reliability.build.dev_case import DGC_INGREDIENT  # noqa: E402
from ration_reliability.datamodel import (  # noqa: E402
    ConstraintClass,
    ConstraintKind,
    DMSource,
    RationDecision,
    Sense,
    SolverOptions,
    SolveStatus,
)
from ration_reliability.evaluation.reference import MAIN_EVENT_PLAN_DOMAIN  # noqa: E402
from ration_reliability.hashing import file_sha256, stable_hash  # noqa: E402
from ration_reliability.io import (  # noqa: E402
    CODE_MANIFEST_DIRS,
    CODE_MANIFEST_ROOT_FILES,
    build_problem,
    code_manifest,
    utc_now,
)
from ration_reliability.nutrition import domain as DOM  # noqa: E402
from ration_reliability.nutrition.constraints import linear_rows  # noqa: E402
from ration_reliability.optimization import get_method  # noqa: E402
from ration_reliability.optimization.highs import run_linprog  # noqa: E402
from ration_reliability.optimization.lp_builder import optimization_indices  # noqa: E402
from ration_reliability.uncertainty import UncertaintySpec  # noqa: E402
from ration_reliability.uncertainty import streams as ST  # noqa: E402
from ration_reliability.uncertainty.base import require_stream  # noqa: E402

#: the premise planning row (the same id as ``nutrition.domain.PREMISE_PLANNING_ROW_ID`` / ``reference.PREMISE_ROW_ID``)
PREMISE_ROW_ID = "SH-PLAN-T51-DGC-SHARE"
if PREMISE_ROW_ID != DOM.PREMISE_PLANNING_ROW_ID:  # one id across src/ and experiments/
    raise ImportError("official_v2.PREMISE_ROW_ID differs from nutrition.domain.PREMISE_PLANNING_ROW_ID")
#: per-ingredient coefficient name of the row (``C:<name>:DM``)
PREMISE_COEFFICIENT = "plan_T51_DGC_SHARE"
#: primary tau (project research assumption; = the primary Table 5-1 domain variant T51-DGC-0.50)
PRIMARY_TAU = DOM.PRIMARY_THRESHOLD
#: numerical tolerance of the row in kg starch / d (= the DIAG-T51 row and ``Table51DomainSpec.tolerance_kg_d``)
PREMISE_TOLERANCE_KG_D = 1.0e-6
REFERENCE_CSV_V2 = Path("configs") / "dev_case_v1" / "reference_constraints_v2.csv"
REFERENCE_DOC_V2 = Path("docs") / "reference_problem_v2.md"
PREMISE_SOURCE = ("NASEM (2021; DOI 10.17226/25806) Table 5-1 title and text, printed p.63: diets fed as a TMR, forage "
                  "of adequate particle size, dry ground corn as the predominant starch source (no number is given "
                  "for 'predominant')")
TAU_RATIONALE = ("tau = 0.50 is the project's operationalisation of 'predominant' (dry ground corn supplies at least "
                 "half of the planned diet starch; exactly one half counts as met): research assumption, not "
                 "attributed to the source; equals the primary Table 5-1 domain variant T51-DGC-0.50")
DECISION_REF = ("decision package (manuscript/draft_methods_v0/投稿就绪评估_20260926.md) §3.1 option B; DECISIONS.md "
                "D-533 (2026-09-26, taken after the development results had been seen; disclosed)")

#: Rows whose planned nominal margin is restricted-only in public tables: the dev-case driver's list
#: (``run_dev_case_v1.REDACT_NOMINAL``) + the premise row (its margin combines d_hat and nominal starch).
REDACT_NOMINAL_V2 = frozenset(DRV.REDACT_NOMINAL) | {PREMISE_ROW_ID}
#: Additions to ``run_dev_case_v1.RUN_CONFIG["public_redaction"]["restricted_only"]`` for v2 outputs.
RESTRICTED_ONLY_V2_EXTRA = (
    f"planned margin of {PREMISE_ROW_ID} (sum_i q_i d_hat_i (1{{dgc}} - tau) nominal starch_i: d_hat and nominal "
    "starch are restricted)",
    f"the per-ingredient coefficients {PREMISE_COEFFICIENT} and the plan-level starch share / margin "
    "(nutrition.domain.PlanDomainStatus; only the status label is public)",
)


#: Official method, selection and membership settings (official-run plan batch 2; §1, §6.3-§6.6; DECISIONS D-533,
#: D-535; BLOCKERS B-126 / B-129 / B-453).  ``configs/methods.yaml`` states exactly these values and the executed
#: ``run_dev_case_v1.RUN_CONFIG`` method grids (``tests/unit/test_methods_config_matches_driver.py``); batch 3 builds
#: ``OFFICIAL_CONFIG`` from this mapping.  The development runs made so far used ``rate_upper_le_alpha`` screening,
#: ``rate_upper`` membership on the per-state main event and both M1 variants as entries (see the ablation driver's
#: ``ABLATION_CONFIG["endpoint_reporting"]``).
OFFICIAL_SETTINGS: dict[str, Any] = {
    "screening": {"rule": DRV.OFFICIAL_SCREENING_RULE, "confidence": DRV.OFFICIAL_SCREENING_CONFIDENCE,
                  "stream": "validation", "unknown_states": "counted as violated (n_V + n_U)",
                  "applies_to": ["M1", "M2", "M3a", "M3b"]},
    "membership": {"membership_stat": "cp_upper", "confidence": 0.95, "stream": "test",
                   "main_event": MAIN_EVENT_PLAN_DOMAIN, "m1_dm_off_is_entry": False,
                   "coverage": ["entries", "distinct_rations"]},
    "N_ladder": [128, 512, 1024], "N_headline": 1024, "opt_draws": 1024,
    "N_nesting": "N < 1024 uses the first N opt draws (prefix)",
    "convergence_criterion": ("per root, SD point and alpha: M2 selection status equal at N = 512 and N = 1024, "
                              "|delta validation rate_upper| <= 2 MC SE and relative cost change <= 1 %; otherwise "
                              "reported as 'not demonstrated' (N is not extended)"),
    "time_limits_s": {"methods": 300.0, "frontier": 300.0,
                      "min_violation": {"128": 60.0, "512": 180.0, "1024": 360.0}},
    "validation_draws": 20000,
    "test_precision": {"n_test": 10000, "interval": "clopper_pearson_exact", "adaptive_doubling": False},
    "M1": {"margin_semantics": "coef_directional", "margin_scale": "relative", "apply_to_dm_main": True,
           "apply_to_dm_false": "sensitivity line, not an entry"},
    "M2": {"big_m_mode": "box_quantile", "polish": True},
    "M3": {"M3a": "box", "M3b": "budget", "box_source": "factory moments (achieved mean +/- k SD, clipped to support)",
           "gamma_grid": "scalar {0..6}, truncated per row to its number of random coefficients"},
    "min_violation_big_m": "box",
}


def add_domain_premise_row(cfg0: dict, problem: Any, tau: float = PRIMARY_TAU, *,
                           counted_ids: Sequence[str] = (DGC_INGREDIENT,), starch_column: str = "starch",
                           tag: str = "reference_problem_v2") -> tuple[dict, dict]:
    """``cfg0`` + the structural planning row :data:`PREMISE_ROW_ID` (a new configuration; ``cfg0`` is not changed).

    ``problem``: the built problem of ``cfg0`` (its nominal composition gives the coefficients).  Every ingredient
    gets the coefficient :data:`PREMISE_COEFFICIENT` = ``(1{i in counted_ids} - tau) * nominal starch_i``
    (:func:`nutrition.domain.premise_planning_coefficients`); the row ``C:<coef>:DM >= 0`` is ``structural_hard`` with
    ``dm_source = decision_estimate`` (planned DM ``q * d_hat``), bound 0 kg/d, tolerance
    :data:`PREMISE_TOLERANCE_KG_D`; its value blocks are ``research_scenario_assumption`` with the source pointer and
    the tau rationale.  No row is removed or changed; the arms built from the returned configuration inherit the row.
    Returns ``(cfg, record)``; the record carries no value.
    """
    ids = [str(g["ingredient_id"]) for g in cfg0["ingredients"]]
    if tuple(problem.ingredient_ids) != tuple(ids) or str(problem.problem_id) != str(cfg0["problem_id"]):
        raise ValueError("add_domain_premise_row: problem is not the built problem of cfg0 (ids or ingredient order)")
    if any(c["constraint_id"] == PREMISE_ROW_ID for c in cfg0["constraints"]):
        raise ValueError(f"add_domain_premise_row: cfg0 already has {PREMISE_ROW_ID}")
    if any(PREMISE_COEFFICIENT in (g.get("coefficients") or {}) for g in cfg0["ingredients"]):
        raise ValueError(f"add_domain_premise_row: coefficient {PREMISE_COEFFICIENT} already exists")
    coef = DOM.premise_planning_coefficients(problem.nominal_theta(), problem.ingredient_ids, problem.nutrient_ids,
                                             counted_ids, tau, starch_column=starch_column)
    t = float(tau)
    cfg = copy.deepcopy(cfg0)
    byid = {g["ingredient_id"]: g for g in cfg["ingredients"]}
    for i, iid in enumerate(problem.ingredient_ids):
        byid[iid].setdefault("coefficients", {})[PREMISE_COEFFICIENT] = {
            "value": float(coef[i]), "unit": "1", "status": "research_scenario_assumption",
            "rationale": (f"{tag}: weight of {PREMISE_ROW_ID} = (1{{counted}} - tau) x nominal (table-value) starch, "
                          f"canonical fraction of DM; tau = {t:g} (project research assumption); counted = "
                          f"{', '.join(counted_ids)}")}
    row = {
        "constraint_id": PREMISE_ROW_ID,
        "name": (f"{tag}: Table 5-1 premise on the plan -- the counted (dry ground corn) share of the planned diet "
                 f"starch is at least tau = {t:g} (nominal composition, decision-time d_hat)"),
        "kind": "supply",
        "terms": {f"C:{PREMISE_COEFFICIENT}:DM": 1.0},
        "sense": "ge",
        "bound": {"value": 0.0, "unit": "kg/d", "basis": "none", "status": "research_scenario_assumption",
                  "rationale": f"{PREMISE_SOURCE}; {TAU_RATIONALE}; {DECISION_REF}"},
        "constraint_class": "structural_hard",
        "numerical_tolerance": PREMISE_TOLERANCE_KG_D,
        "dm_source": "decision_estimate",
        "claim_scope": (f"planning row of {tag} in every objective arm and method (M0 included): the ration as "
                        "formulated meets the starch-source premise at table values; not a reliability statement "
                        "(the per-state share stays a diagnostic)"),
    }
    cfg["constraints"] = list(cfg["constraints"]) + [row]
    cfg["problem_id"] = f"{cfg0['problem_id']}|v2"
    cfg["description"] = (str(cfg0.get("description", "")) + f" | {tag}: + structural planning row {PREMISE_ROW_ID} "
                          f"(Table 5-1 starch-source premise on the plan, tau = {t:g}; D-533).")
    record = {"row_id": PREMISE_ROW_ID, "coefficient": PREMISE_COEFFICIENT, "tau": t,
              "tau_status": "research_assumption", "counted_ingredient_ids": list(counted_ids),
              "starch_column": starch_column, "source": PREMISE_SOURCE, "decision": DECISION_REF,
              "construction": ("structural_hard, dm_source decision_estimate, C:<coef>:DM >= 0 at the nominal "
                               "composition -- the SH-PLAN-* rule of planned_arm_cfg / s2_problem_cfg"),
              "inherited_by": ["FULL11", "PART6P5", "MAIN9", "every method including M0"],
              "unchanged": "every cfg0 row, bound and tolerance; DIAG-T51-DGC-STARCH-SHARE stays a per-state diagnostic",
              "problem_id": cfg["problem_id"]}
    return cfg, record


def build_v2_cfg0(cfg0: dict, problem: Any, *, tau: float = PRIMARY_TAU,
                  counted_ids: Sequence[str] = (DGC_INGREDIENT,), mode: str = "pilot") -> tuple[dict, Any, dict]:
    """``(cfg0_v2, problem_v2, record)``: :func:`add_domain_premise_row`, built and validated (``mode`` as in
    ``io.build_problem``), with the checks that make v2 = v1 + exactly one row:

    * the compiled premise row is ``structural_hard``, ``decision_estimate``, ``>= 0``, planned-DM terms only, and its
      coefficients equal ``premise_planning_coefficients`` at the nominal composition;
    * every v1 row keeps its compiled class, sense, arrays, bound and tolerance; nominal composition, ``d_hat`` and
      prices are unchanged.

    A failed check raises ``ValueError`` naming the check only (no value).  The record holds booleans and ids.
    """
    cfg, rec = add_domain_premise_row(cfg0, problem, tau, counted_ids=counted_ids)
    problem_v2, rep = build_problem(cfg, mode=mode)
    if not rep.ok:
        raise ValueError(f"build_v2_cfg0: the v2 configuration does not validate ({len(rep.errors)} errors)")
    c1, c2 = problem.compiled, problem_v2.compiled
    ids1, ids2 = list(c1.constraint_ids), list(c2.constraint_ids)
    k = ids2.index(PREMISE_ROW_ID) if PREMISE_ROW_ID in ids2 else None
    coef = DOM.premise_planning_coefficients(problem.nominal_theta(), problem.ingredient_ids, problem.nutrient_ids,
                                             counted_ids, tau)
    checks = {
        "one_row_added": sorted(set(ids2) - set(ids1)) == [PREMISE_ROW_ID] and set(ids1) <= set(ids2),
        "premise_row_structural_decision_estimate": k is not None
        and c2.classes[k] is ConstraintClass.STRUCTURAL_HARD and c2.dm_sources[k] is DMSource.DECISION_ESTIMATE,
        "premise_row_ge_zero_planned_dm_terms_only": k is not None and c2.senses[k] is Sense.GE
        and float(c2.bound[k]) == 0.0 and not np.any(np.asarray(c2.W[k]) != 0) and not np.any(np.asarray(c2.v[k]) != 0),
        "premise_row_coefficients_equal_premise_planning_coefficients": k is not None
        and bool(np.allclose(np.asarray(c2.w0[k], float), coef, rtol=1e-12, atol=1e-15)),
        "v1_rows_unchanged": all(
            c1.classes[a] is c2.classes[ids2.index(cid)] and c1.senses[a] is c2.senses[ids2.index(cid)]
            and np.array_equal(c1.W[a], c2.W[ids2.index(cid)]) and np.array_equal(c1.w0[a], c2.w0[ids2.index(cid)])
            and np.array_equal(c1.v[a], c2.v[ids2.index(cid)]) and float(c1.bound[a]) == float(c2.bound[ids2.index(cid)])
            and float(c1.tol[a]) == float(c2.tol[ids2.index(cid)]) for a, cid in enumerate(ids1)),
        "nominal_composition_d_hat_prices_unchanged": bool(
            np.array_equal(problem.nominal_theta(), problem_v2.nominal_theta())
            and np.array_equal(problem.dm_estimates(), problem_v2.dm_estimates())
            and tuple(problem.nutrient_ids) == tuple(problem_v2.nutrient_ids)
            and dict(problem.prices.prices_per_kg_as_fed) == dict(problem_v2.prices.prices_per_kg_as_fed)),
    }
    failed = [n for n, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"build_v2_cfg0: checks failed: {failed}")
    rec = {**rec, "validator": {"mode": mode, "ok": bool(rep.ok), "n_errors": len(rep.errors),
                                "n_pending": rep.n_pending}, "checks": checks}
    return cfg, problem_v2, rec


# =================================================================================================================
# batch 3 (official-run plan §1, §2 "B3", §3 test 5): the five declared SD points
# =================================================================================================================
#: The SD-ratio registry (docs/uncertainty_data_layers.md §4-§5).  Every reader of the per-point ratios goes through this
#: constant (and the ``registry`` argument of the readers), so BLOCKERS B-454 (where the ratio values are kept) can move
#: the file in one place.  Nothing in this module prints or writes a ratio value; the registry check reports booleans,
#: counts and cell ids only.
SD_SCALING_SOURCES = Path("reports") / "sd_scaling_sources.csv"
#: registry rows that carry the stochastic cells and their per-point ratios
SD_REGISTRY_ROW_KIND = "S2_cell"
#: stochastic (ingredient, component) cells of dev_case_v1 (6 ingredients x (DM + 7 composition columns))
N_STOCHASTIC_CELLS = 48
#: the five declared SD points (docs/uncertainty_data_layers.md §5.2, declared 2026-09-25; D-533 "all five run")
SD_POINTS = ("SD-H0", "SD-S2", "SD-SRC14", "SD-12MO", "SD-SRC12")
SD_POINT_ROLES = {
    "SD-H0": ("table (observed commercial-lab) SD used as the batch SD (H0, strong assumption); primary evaluation "
              "world"),
    "SD-S2": ("the narrowed SD of the development S2 = the lower end of the declared range, a method-calibration point "
              "selected by attainability in FIX_B screening -- not a sourced lower limit"),
    "SD-SRC14": "sensitivity: only the three corn-silage cells with a same-cell source narrowed (14 d); no borrowed ratio",
    "SD-12MO": "sensitivity: 12-month scale (corn-silage DM / NDF 12-month ratios; every other cell borrows)",
    "SD-SRC12": "sensitivity: 12-month scale, only the two sourced corn-silage cells (DM, NDF) narrowed",
}
#: components of a registry cell: DM + the base-model composition columns (``run_dev_case_v1.PN``)
SD_POINT_COMPONENTS = ("DM",) + tuple(DRV.PN)
#: the cells narrowed from a same-cell source (corn silage DM, NDF, starch; docs/uncertainty_data_layers.md §5.1)
SOURCED_CELLS = tuple((DRV.CS, c) for c in DRV.S2_CS_RATIOS)
#: the two sourced cells that have a 12-month ratio (corn silage DM, NDF; starch has none)
SOURCED_CELLS_12MO = tuple(c for c in SOURCED_CELLS if c[1] in ("DM", "NDF"))
#: exact identity of the development S2 world, so that the SD-S2 point is the development S2 specification itself
#: (checked by fingerprint in tests/unit/test_official_v2_batch3.py, holder side)
_S2_CELL_TAG = "dev_case_v1_S2_cell/v1"
_S2_SPEC_ID = "DEV_CASE_V1_S2_REFFARM_TN_MM"
_POINT_CELL_TAG = "official_v2_sd_point_cell/v1"
_RATIO_TOL = 1e-12


def sd_point_column(point: str) -> str:
    """Registry column of a declared SD point (``SD-SRC14`` -> ``ratio_SD_SRC14``)."""
    if point not in SD_POINTS:
        raise ValueError(f"unknown SD point {point!r}; declared points: {SD_POINTS}")
    return "ratio_" + point.replace("-", "_")


def _parse_ratio(s: Any) -> float:
    """``"a/b"`` or a decimal -> float (the same rule as ``run_endpoint_ablation._ratio``)."""
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def _registry_rows(repo: Path, registry: Path) -> list[dict[str, str]]:
    p = registry if Path(registry).is_absolute() else Path(repo) / registry
    with open(p, encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("row_kind") == SD_REGISTRY_ROW_KIND]


def _cell_ratio_table(point_or_column: str, rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    """{(ingredient, component): ratio} of one registry column; a missing column, a duplicate cell or a ratio that is
    not positive and finite raises (the message names the cell, never a value)."""
    col = point_or_column if point_or_column.startswith(("ratio_", "sd_range_")) else sd_point_column(point_or_column)
    out: dict[tuple[str, str], float] = {}
    for r in rows:
        if col not in r:
            raise ValueError(f"SD registry has no column {col}")
        key = (r["ingredient_id"], r["component"])
        if key[1] not in SD_POINT_COMPONENTS:
            raise ValueError(f"SD registry: {key[0]}/{key[1]} is not a component of the case")
        if key in out:
            raise ValueError(f"SD registry: cell {key[0]}/{key[1]} appears twice")
        try:
            v = _parse_ratio(r[col])
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"SD registry: {key[0]}/{key[1]} {col} is not a ratio") from None
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"SD registry: {key[0]}/{key[1]} {col} is not a positive finite ratio")
        out[key] = v
    return out


def sd_point_ratio_arrays(point: str, ids: Sequence[str], *, repo: Path = REPO,
                          registry: Path = SD_SCALING_SOURCES) -> tuple[np.ndarray, np.ndarray]:
    """``(r [I, len(PN)], rd [I])``: the SD ratios of a declared SD point, read from the registry column
    ``ratio_SD_<point>`` of :data:`SD_SCALING_SOURCES` (rows ``row_kind = S2_cell``).

    Same layout as ``run_dev_case_v1.s2_ratio_arrays`` (composition columns in ``PN`` order; DM separately).  Cells
    that are not in the registry -- the non-stochastic ingredients, whose SD is 0 -- keep ratio 1 (their SD is not
    scaled).  A registry ingredient that is not in ``ids`` raises.  No value is printed."""
    ids = tuple(ids)
    table = _cell_ratio_table(point, _registry_rows(repo, registry))
    r = np.ones((len(ids), len(DRV.PN)))
    rd = np.ones(len(ids))
    for (iid, comp), v in table.items():
        if iid not in ids:
            raise ValueError(f"SD registry: {iid} is not an ingredient of the case")
        i = ids.index(iid)
        if comp == "DM":
            rd[i] = v
        else:
            r[i, DRV.PN.index(comp)] = v
    return r, rd


def _point_ratio(r: np.ndarray, rd: np.ndarray, ids: Sequence[str], iid: str, comp: str) -> float:
    i = list(ids).index(iid)
    return float(rd[i]) if comp == "DM" else float(r[i, DRV.PN.index(comp)])


def sd_point_cell_overrides(point: str, ids: Sequence[str], cells: Mapping, cells_sha: str, *, repo: Path = REPO,
                            registry: Path = SD_SCALING_SOURCES) -> tuple[dict, np.ndarray, np.ndarray]:
    """``(overrides, r, rd)`` of a narrowed SD point: the development cell overrides (``run_dev_case_v1.cell_overrides``)
    with every stochastic cell whose ratio is below 1 relabelled ``unidentified`` / ``research_scenario_assumption``
    (no source; its data fingerprint re-hashed with the ratio); cells at ratio 1 keep their H0 labels.

    Guards: every stochastic cell of ``cells`` must be a registry cell and every registry cell stochastic; a ratio
    above 1 (a widening) is outside the declared range and raises.  Messages carry counts and ids, never values."""
    ids = tuple(ids)
    r, rd = sd_point_ratio_arrays(point, ids, repo=repo, registry=registry)
    reg_cells = set(_cell_ratio_table(point, _registry_rows(repo, registry)))
    stoch = {k for k, c in cells.items() if str(c["is_stochastic"]) == "True"}
    if stoch != reg_cells:
        raise ValueError(f"{point}: the registry cells and the stochastic cells of the case differ "
                         f"({len(reg_cells - stoch)} registry-only, {len(stoch - reg_cells)} stochastic-only)")
    if np.any(r > 1.0 + _RATIO_TOL) or np.any(rd > 1.0 + _RATIO_TOL):
        raise ValueError(f"{point}: a ratio above 1 (widening) is outside the declared range")
    tag = _S2_CELL_TAG if point == "SD-S2" else f"{_POINT_CELL_TAG}|{point}"
    over = DRV.cell_overrides(cells, cells_sha)
    for (iid, item), o in over.items():
        if (iid, item) not in stoch:
            continue
        ratio = _point_ratio(r, rd, ids, iid, item)
        if ratio == 1.0:
            continue                                          # not narrowed: the H0 labels are kept
        o.update({"variance_basis": "unidentified", "provenance_status": "research_scenario_assumption",
                  "source_id": None, "locator": None,
                  "data_fingerprint": stable_hash(tag, o["data_fingerprint"], ratio)})
    return over, r, rd


def _point_notes(point: str) -> str:
    return (f"official run v2 (D-533) declared SD point {point}: table means, SD = ratio x table SD with the ratios of "
            f"{SD_SCALING_SOURCES.as_posix()} column {sd_point_column(point)} (docs/uncertainty_data_layers.md §5.2); "
            "cells below ratio 1 are unidentified / research_scenario_assumption, cells at ratio 1 keep the H0 labels; "
            "a declared scenario, not an identified variance decomposition; not a result")


def sd_point_spec_from_overrides(point: str, ids: Sequence[str], cells: Mapping, over: dict, r: np.ndarray,
                                 rd: np.ndarray, *, spec_id: Optional[str] = None,
                                 notes: Optional[str] = None) -> UncertaintySpec:
    """The narrowed-SD specification of ``point`` (the rule of ``run_dev_case_v1.build_s2_spec`` with the point's
    arrays): table means; SD = ratio x table SD; ``sensitivity_scenario``; the factory family rule of the development
    run.  ``spec_id`` / ``notes`` default to the point's own identity (``SD-S2``: the development S2 identity)."""
    tm, ts, dm, ds = DRV.cell_arrays(cells, ids)
    if spec_id is None:
        spec_id = _S2_SPEC_ID if point == "SD-S2" else f"DEV_CASE_V1_{point.replace('-', '')}_OFFICIAL_V2_TN_MM"
    return UncertaintySpec.from_arrays(
        spec_id, tuple(ids), DRV.PN, tm, ts * r, dm, ds * rd, purpose="sensitivity_scenario",
        moment_semantics="target_marginal_moments", is_synthetic=False,
        family_rule=DRV.RUN_CONFIG["uncertainty"]["family_rule"],
        cell_defaults={"variance_basis": "observed_incl_sampling_and_lab", "decomposition_id": "none",
                       "decomposition_source": None, "measurement_model_id": DRV.MEASUREMENT_MODEL_NASEM},
        theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0), cell_overrides=over,
        notes=_point_notes(point) if notes is None else notes)


def build_sd_point_spec(point: str, ids: Sequence[str], cells: Mapping, cells_sha: str, *, repo: Path = REPO,
                        registry: Path = SD_SCALING_SOURCES) -> UncertaintySpec:
    """Uncertainty specification of a declared SD point (generalises ``run_dev_case_v1.build_s2_spec``).

    * ``SD-H0`` (ratio 1 in every cell, checked): the H0 specification itself (``run_dev_case_v1.build_h0_spec``; the H0
      labels and the H1 extension-scenario record are kept);
    * ``SD-S2``: the registry arrays must equal ``run_dev_case_v1.s2_ratio_arrays`` on every stochastic cell (checked);
      the specification is then the development S2 specification (``build_s2_spec``) -- the generic construction
      below reproduces it exactly (holder-side fingerprint test);
    * the other points: :func:`sd_point_cell_overrides` + :func:`sd_point_spec_from_overrides` (cells below 1 are
      ``unidentified`` / ``research_scenario_assumption``; cells at 1 keep the H0 labels).
    """
    ids = tuple(ids)
    if point == "SD-H0":
        r, rd = sd_point_ratio_arrays(point, ids, repo=repo, registry=registry)
        if not (np.all(r == 1.0) and np.all(rd == 1.0)):
            raise ValueError("SD-H0 must be ratio 1 in every cell (registry column ratio_SD_H0)")
        return DRV.build_h0_spec(ids, cells, cells_sha)
    over, r, rd = sd_point_cell_overrides(point, ids, cells, cells_sha, repo=repo, registry=registry)
    if point == "SD-S2":
        r0, rd0 = DRV.s2_ratio_arrays(ids)
        stoch = [k for k, c in cells.items() if str(c["is_stochastic"]) == "True"]
        same = all(abs(_point_ratio(r, rd, ids, i, c) - _point_ratio(r0, rd0, ids, i, c)) <= _RATIO_TOL
                   for i, c in stoch)
        if not same:
            raise ValueError("SD-S2: the registry column ratio_SD_S2 differs from run_dev_case_v1.s2_ratio_arrays")
        return DRV.build_s2_spec(ids, cells, cells_sha)
    return sd_point_spec_from_overrides(point, ids, cells, over, r, rd)


def sd_registry_check_all(ids: Sequence[str], *, repo: Path = REPO, registry: Path = SD_SCALING_SOURCES
                          ) -> dict[str, Any]:
    """Registry check of the five declared SD points (official-run plan B3; §3 test 5).  Raises ``SystemExit`` naming
    the failed checks and cells -- never a value -- and returns booleans and counts only.

    Checks: exactly :data:`N_STOCHASTIC_CELLS` cells, each an ingredient / component of the case; every point column is
    a positive finite ratio in every cell; SD-H0 = 1 everywhere; SD-S2 = ``run_dev_case_v1.s2_ratio_arrays``; every
    point inside the declared per-cell range [``sd_range_low``, ``sd_range_high``] with ``sd_range_low`` = SD-S2 and
    ``sd_range_high`` = 1 in every cell (§5 "下端的来历"); the narrowing pattern of §5.2:

    * SD-S2: every cell below 1; the 45 unsourced cells share one borrowed ratio ``b``; the three sourced corn-silage
      cells (DM, NDF, starch) are narrower than ``b``;
    * SD-SRC14: the three sourced cells equal SD-S2; the other 45 cells are 1;
    * SD-12MO: the 45 unsourced cells and corn-silage starch equal ``b``; corn-silage DM equals ``b``; corn-silage NDF lies
      in [SD-S2, 1) and differs from ``b``;
    * SD-SRC12: corn-silage DM and NDF equal SD-12MO; every other cell is 1;
    * element-wise order SD-S2 <= SD-12MO <= SD-SRC12 <= SD-H0 and SD-S2 <= SD-SRC14 <= SD-H0; the five points differ
      pairwise.
    """
    ids = tuple(ids)
    bad: list[str] = []
    try:
        rows = _registry_rows(repo, registry)
        tabs = {p: _cell_ratio_table(p, rows) for p in SD_POINTS}
        lo = _cell_ratio_table("sd_range_low", rows)
        hi = _cell_ratio_table("sd_range_high", rows)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"SD registry check (five points) failed: {exc}") from None
    cells = sorted(tabs["SD-H0"])
    if len(cells) != N_STOCHASTIC_CELLS:
        bad.append(f"{len(cells)} registry cells (expected {N_STOCHASTIC_CELLS})")
    foreign = sorted({i for i, _ in cells} - set(ids))
    if foreign:
        bad.append(f"registry ingredients not in the case: {foreign}")
    r_s2, rd_s2 = DRV.s2_ratio_arrays(ids) if not foreign else (None, None)

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= _RATIO_TOL

    def fail(check: str, which: list) -> None:
        if which:
            bad.append(f"{check}: {len(which)} cell(s), e.g. " + ", ".join(f"{i}/{c}" for i, c in which[:4]))

    t = tabs
    fail("SD-H0 is not 1", [k for k in cells if not close(t["SD-H0"][k], 1.0)])
    if r_s2 is not None:
        fail("SD-S2 differs from run_dev_case_v1.s2_ratio_arrays",
             [k for k in cells if not close(t["SD-S2"][k], _point_ratio(r_s2, rd_s2, ids, *k))])
    fail("sd_range_low differs from SD-S2", [k for k in cells if not close(lo[k], t["SD-S2"][k])])
    fail("sd_range_high is not 1", [k for k in cells if not close(hi[k], 1.0)])
    for p in SD_POINTS:
        fail(f"{p} outside [sd_range_low, sd_range_high]",
             [k for k in cells if not (lo[k] - _RATIO_TOL <= t[p][k] <= hi[k] + _RATIO_TOL)])
    sourced = [k for k in SOURCED_CELLS if k in t["SD-S2"]]
    if len(sourced) != len(SOURCED_CELLS):
        bad.append("the sourced corn-silage cells are not all registry cells")
    unsourced = [k for k in cells if k not in SOURCED_CELLS]
    borrowed = sorted({round(t["SD-S2"][k], 15) for k in unsourced})
    if len(borrowed) != 1:
        bad.append(f"SD-S2: the unsourced cells do not share one borrowed ratio ({len(borrowed)} distinct)")
    b = borrowed[0] if borrowed else float("nan")
    fail("SD-S2 not below 1", [k for k in cells if not t["SD-S2"][k] < 1.0])
    fail("SD-S2 sourced cell not narrower than the borrowed ratio", [k for k in sourced if not t["SD-S2"][k] < b])
    fail("SD-SRC14 sourced cell differs from SD-S2", [k for k in sourced if not close(t["SD-SRC14"][k], t["SD-S2"][k])])
    fail("SD-SRC14 unsourced cell is not 1", [k for k in unsourced if not close(t["SD-SRC14"][k], 1.0)])
    cs_starch = (DRV.CS, "starch")
    fail("SD-12MO unsourced cell / corn-silage starch is not the borrowed ratio",
         [k for k in unsourced + [cs_starch] if k in t["SD-12MO"] and not close(t["SD-12MO"][k], b)])
    cs_dm, cs_ndf = (DRV.CS, "DM"), (DRV.CS, "NDF")
    fail("SD-12MO corn-silage DM is not the borrowed ratio", [k for k in [cs_dm] if not close(t["SD-12MO"][k], b)])
    fail("SD-12MO corn-silage NDF not in [SD-S2, 1) or equal to the borrowed ratio",
         [k for k in [cs_ndf] if not (t["SD-S2"][k] - _RATIO_TOL <= t["SD-12MO"][k] < 1.0
                                      and not close(t["SD-12MO"][k], b))])
    fail("SD-SRC12 corn-silage DM / NDF differ from SD-12MO",
         [k for k in SOURCED_CELLS_12MO if not close(t["SD-SRC12"][k], t["SD-12MO"][k])])
    fail("SD-SRC12 other cell is not 1", [k for k in cells if k not in SOURCED_CELLS_12MO
                                          and not close(t["SD-SRC12"][k], 1.0)])
    for chain in (("SD-S2", "SD-12MO", "SD-SRC12", "SD-H0"), ("SD-S2", "SD-SRC14", "SD-H0")):
        for a_, b_ in zip(chain, chain[1:]):
            fail(f"order {a_} <= {b_}", [k for k in cells if not t[a_][k] <= t[b_][k] + _RATIO_TOL])
    for i, p in enumerate(SD_POINTS):
        for q in SD_POINTS[i + 1:]:
            if all(close(t[p][k], t[q][k]) for k in cells):
                bad.append(f"{p} and {q} are the same point")
    if bad:
        raise SystemExit("SD registry check (five points) failed (values withheld): " + "; ".join(bad[:12]))
    p_reg = registry if Path(registry).is_absolute() else Path(repo) / registry
    return {"registry": Path(registry).as_posix(), "registry_sha256": file_sha256(p_reg), "n_cells": len(cells),
            "points": {p: {"column": sd_point_column(p), "n_cells_below_1": sum(t[p][k] < 1.0 for k in cells),
                           "n_cells_at_1": sum(close(t[p][k], 1.0) for k in cells), "within_declared_range": True,
                           "role": SD_POINT_ROLES[p]} for p in SD_POINTS},
            "SD-H0": "ratio 1 in every stochastic cell", "SD-S2": "= run_dev_case_v1.s2_ratio_arrays (48/48)",
            "range": "sd_range_low = SD-S2 and sd_range_high = 1 in every cell; every point inside",
            "pattern": "docs/uncertainty_data_layers.md §5.2 (checked cell by cell; values withheld)", "ok": True}


# =================================================================================================================
# batch 3: the official configuration (data only) and the job matrix
# =================================================================================================================
#: the linear fixed-DMI energy row (trained in every arm; the reference chain judges energy in the main event)
ENERGY_ROW_ID = "PN-NEL-FIXEDDMI"
#: the Table 5-1 starch rows relaxed by DIAG-T45 (starch upper limit and its fNDF-dependent form; both in % DM)
T45_RELAXED_ROWS = ("PN-T4", "PN-T5")
#: role text of every diagnostic ration / row
DIAGNOSTIC_ROLE = ("diagnostic (declared before the official run); never a method, never a comparable-set entry or "
                   "member; its cost is never a comparable cost")
OBJECTIVE_ARMS = ("MAIN9", "FULL11", "PART6P5")


def _official_cell(sd: str, objective: str, role: str, note: str) -> dict[str, str]:
    return {"cell_id": f"{sd.replace('-', '')}_{objective}", "sd": sd, "objective": objective, "role": role,
            "note": note}


#: 9 cells: the main arm MAIN9 x the five SD points; the transparency arms FULL11 and PART6P5 x {SD-H0, SD-S2}
OFFICIAL_CELLS: tuple[dict[str, str], ...] = tuple(
    [_official_cell(sd, "MAIN9", "main", "main arm (D-533); every SD point shares the root's opt / validation / test "
                                         "stream ids (common random numbers)") for sd in SD_POINTS]
    + [_official_cell(sd, obj, "transparency", "transparency arm (D-533): only SD-H0 and SD-S2")
       for obj in ("FULL11", "PART6P5") for sd in ("SD-H0", "SD-S2")])
_MAIN9_CELL_IDS = [c["cell_id"] for c in OFFICIAL_CELLS if c["objective"] == "MAIN9"]
_ALL_CELL_IDS = [c["cell_id"] for c in OFFICIAL_CELLS]

#: The primary assumption of the official run (``run_context.primary_assumption_id`` of every arm in official mode,
#: plan F5; ``io/config.py`` refuses an official problem without it): primary evaluation world SD-H0, marginal family
#: TN_MM with the BETA_MM fallback, correlation C0 (independent marginals), DM-uncertain decision mode, reference
#: problem v2 with the plan-level Table 5-1 reading (configs/protocol.yaml uncertainty / units_and_decisions /
#: constraints; D-533, D-535; B-127 still open for the family and correlation, declared there).
PRIMARY_ASSUMPTION_ID = "v2|SD-H0|TN_MM+BETA_MM|C0|DM-uncertain|plan-domain"

#: The official run configuration (official-run plan §1, §5, §6 items 1-5; D-533, D-535).  DATA ONLY: nothing is drawn,
#: solved or read when this mapping is built, and no reserved root value appears in it -- the roots are the indices
#: ``k`` = 0, 1, 2 into ``configs/streams_policy.yaml`` ``reserved_formal_streams.roots`` and are resolved to seeds only
#: in batch 4, after ``require_protocol_frozen()`` (a reserved root is spent on its first draw; streams policy SP-4).
#: The keys read by the ablation driver's ``plan()`` (``alphas``, ``streams``, ``solver``, ``diagnostics.min_violation``,
#: ``cells``, ``evaluation.evaluation_worlds``, ``objective_arms.*.training_event``) have the shape of
#: ``run_endpoint_ablation.ABLATION_CONFIG``.  Items the research lead still has to confirm before the freeze (plan §6)
#: are marked ``pending_research_lead``; changing any of them after a solve is not allowed (plan §7).
OFFICIAL_CONFIG: dict[str, Any] = {
    "schema": "ration_reliability.official_v2_config/0.1",
    "case_id": "dev_case_v1",
    "reference_problem": "reference_problem_v2",
    "task": ("official-run plan batch 3 (docs/official_run_v2_plan_20260926.md §1, §2 B3, §6 items 1-5): cells, stream "
             "sizes, time limits, alpha levels, grids, evaluation worlds, diagnostics, convergence criterion, job matrix"),
    "run_role": ("official run configuration, declared before the protocol freeze (batch 4: the freeze gate "
                 "require_protocol_frozen, the reserved-root guard and the job runner are in place; batch 5 freezes: "
                 "configs/protocol_freeze.json + the v2 pin in one commit)"),
    "decisions": ["D-533", "D-535", "D-539"],
    "primary_assumption_id": PRIMARY_ASSUMPTION_ID,     # injected into every arm's run_context (plan F5)
    "roots": {
        "root_k": [0, 1, 2], "n_reserved_roots": 3,
        "source": ("configs/streams_policy.yaml reserved_formal_streams.roots, in file order; the values are never "
                   "written into this configuration, a job or a public table"),
        "resolution": ("batch 4 only: a job resolves root k to its seed after require_protocol_frozen() has issued the "
                       "token; nothing in batch 3 creates a RandomStreams from a reserved root"),
        "reporting": "tables per root; the roots are never pooled into one n (plan §7)"},
    "matrix": {
        "cells_by_root_k": {"0": list(_ALL_CELL_IDS), "1": list(_MAIN9_CELL_IDS), "2": list(_MAIN9_CELL_IDS)},
        "diagnostics_roots_k": [0],
        "rule": ("root 0: all 9 cells; roots 1 and 2: MAIN9 x the five SD points; one diagnostics job (DIAG-T45 + "
                 "DIAG-E1) on root 0 only (D-539; plan §5 had one per root) -> 19 cell jobs + 1 diagnostics job"),
        "fallback_declared_at_freeze_only": ("if the worker count makes the worst case exceed the budget, the freeze "
                                             "(not the run) may reduce roots 1 and 2 to MAIN9 x {SD-H0, SD-S2}; never "
                                             "trimmed mid-run (plan §5)"),
        "workers": 8,
        "status": ("declared (D-539: cell matrix as planned, diagnostics on root 0 only, 8 workers, BLAS "
                   "single-threaded); fixed by the freeze")},
    "sd_points": {p: {"ratio_column": sd_point_column(p), "role": SD_POINT_ROLES[p],
                      "spec_builder": "official_v2.build_sd_point_spec",
                      "labels": ("H0 labels" if p == "SD-H0" else
                                 "cells below ratio 1: unidentified / research_scenario_assumption; cells at 1: H0 labels")}
                  for p in SD_POINTS},
    "sd_registry": SD_SCALING_SOURCES.as_posix(),
    "sd_registry_check": "official_v2.sd_registry_check_all (5 columns x 48 cells; values withheld)",
    "objective_arms": {
        "MAIN9": {"problem": ("planned_arm_cfg(cfg0_v2, problem_v2, (PN-CP-HI, PN-EE-HI), 'MAIN9'): the 9 main-reference "
                              "rows probabilistic (energy trained on the linear fixed-DMI row), PN-CP-HI / PN-EE-HI "
                              "planned at table values and d_hat; + the premise planning row of v2"),
                  "training_event": "main reference nine (linear energy row, Table 5-1 verdicts in every state)",
                  "role": "main arm"},
        "FULL11": {"problem": "cfg0_v2 = the dev_case_v1 problem + SH-PLAN-T51-DGC-SHARE (11 probabilistic rows)",
                   "training_event": "H0 eleven (linear energy row)", "role": "transparency arm"},
        "PART6P5": {"problem": "run_dev_case_v1.s2_problem_cfg(cfg0_v2, problem_v2): S2 six rows + 5 planned rows",
                    "training_event": "S2 declared six", "role": "transparency arm"}},
    "cells": [dict(c) for c in OFFICIAL_CELLS],
    "streams": {"opt": int(OFFICIAL_SETTINGS["opt_draws"]), "N_ladder": list(OFFICIAL_SETTINGS["N_ladder"]),
                "N_headline": int(OFFICIAL_SETTINGS["N_headline"]), "N_nesting": OFFICIAL_SETTINGS["N_nesting"],
                "validation": int(OFFICIAL_SETTINGS["validation_draws"]),
                "test": int(OFFICIAL_SETTINGS["test_precision"]["n_test"]),
                "crn": ("every SD point of a root draws root/opt, root/validation, root/test of one RandomStreams(root): "
                        "equal stream ids, different parent distributions (common random numbers)"),
                "adaptive_doubling": False},
    "alphas": {"primary": 0.05, "supplementary": [0.10, 0.01],
               "source": "configs/methods.yaml risk_levels (research design choice, not an animal safety threshold)"},
    "solver": {"time_limit_s": float(OFFICIAL_SETTINGS["time_limits_s"]["methods"]), "mip_rel_gap": 1e-4,
               "frontier_time_limit_s": float(OFFICIAL_SETTINGS["time_limits_s"]["frontier"]),
               "source": "official-run plan §6.5; configs/methods.yaml; HiGHS via scipy"},
    "methods": {"source": ("run_dev_case_v1.RUN_CONFIG['methods'] through run_dev_case_v1.run_method_block (the "
                           "development grids; no new method or grid), N ladder of this configuration"),
                "grids": {k: copy.deepcopy(v) for k, v in DRV.RUN_CONFIG["methods"].items()},
                "N_ladder_note": ("grids['M2']['N'] is the development ladder recorded by run_dev_case_v1; the "
                                  "official N ladder is streams.N_ladder (run_method_block takes N from the run sizes)"),
                "settings": "official_v2.OFFICIAL_SETTINGS"},
    "endpoint_reporting": {
        "screening_rule": OFFICIAL_SETTINGS["screening"]["rule"],
        "screening_confidence": OFFICIAL_SETTINGS["screening"]["confidence"],
        "membership_stat": OFFICIAL_SETTINGS["membership"]["membership_stat"],
        "main_event": OFFICIAL_SETTINGS["membership"]["main_event"],
        "m1_dm_off_is_entry": OFFICIAL_SETTINGS["membership"]["m1_dm_off_is_entry"],
        "coverage": list(OFFICIAL_SETTINGS["membership"]["coverage"])},
    "comparable_set_rule": ("per evaluation world (the five SD points; SD-H0 primary) and target alpha: rations of the "
                            "declared method entries (M0-M3; M1 without the DM margin is a sensitivity line, not an "
                            "entry) with structural_ok and p-bar <= alpha, p-bar = the one-sided exact Clopper-Pearson "
                            "upper bound (confidence 0.95) of (n_violated + n_unknown) / S of the plan-level main event "
                            f"{MAIN_EVENT_PLAN_DOMAIN} (D-535) on that world's test stream; costs are compared only inside "
                            "the set; coverage per entry and per distinct ration; the frontier and the diagnostics "
                            "DIAG-T45 / DIAG-E1 are never entries or members"),
    "evaluation": {"evaluator": "ration_reliability.evaluation.reference.evaluate_reference",
                   "reference_constraints": REFERENCE_CSV_V2.as_posix(),
                   "reference_problem": "problem_v2 (cfg0_v2; every H0 row + the premise planning row compiled)",
                   "evaluation_worlds": list(SD_POINTS), "primary_world": "SD-H0", "stream": "test",
                   "main_event": MAIN_EVENT_PLAN_DOMAIN,
                   "assumption_range": ("per ration: the own-world (diagonal) value and the SD-H0 column form the "
                                        "assumption range -- never called a confidence interval")},
    "diagnostics": {
        "official_jobs": ["DIAG-T45", "DIAG-E1"],
        "roots_k": [0],
        "training_worlds": ["SD-H0", "SD-S2"],
        "evaluation_worlds": ("every diagnostic ration is scored on the test stream of all five evaluation worlds "
                              "(scoring only; the diagnostics solve in SD-H0 and SD-S2 only, D-539)"),
        "flag": "diagnostic",
        "role": DIAGNOSTIC_ROLE,
        "min_violation": {"time_limit_s": {k: float(v) for k, v in
                                           OFFICIAL_SETTINGS["time_limits_s"]["min_violation"].items()},
                          "label": ("training_scenarios_only: about these N scenarios, this world and this model; not "
                                    "a population, field or whole-space statement")},
        "frontier": ("M2 with alpha_train = m*/N per cell and N (the development rule); diagnostic, never a member; "
                     "time limit solver.frontier_time_limit_s"),
        "DIAG-T45": {
            "what": ("nominal-LP headroom test relaxing only the Table 5-1 starch rows: A0 = maximum nominal "
                     "linear-NEL headroom with every optimisation row of problem_v2; A1 = the same without PN-T4 / "
                     "PN-T5; A2(h) = the minimum common relaxation delta of the PN-T4 / PN-T5 bounds (natural unit: "
                     "percentage points of DM) such that the maximum nominal headroom is >= h; each stage-1 LP is "
                     "followed by the cheapest ration at its optimum (stage 2); every ration scored by "
                     "evaluate_reference on every evaluation world's test stream"),
            "energy_row": ENERGY_ROW_ID, "relaxed_rows": list(T45_RELAXED_ROWS),
            "delta": "one delta >= 0 added to both bounds, in their declared unit (% DM); delta* = 0 when not binding",
            "h_grid": {"alphas": [0.10, 0.05, 0.01], "factors": [0.5, 1.0, 1.5], "include_zero": True,
                       "quantile_method": "higher",
                       "a1_equal_subdivisions": 8,
                       "a1_grid": ("D-539: the grid also covers [0, A1] in equal steps, h = j x A1 / n for j = 1..n "
                                   "(n = a1_equal_subdivisions; A1 = the maximum nominal headroom without PN-T4 / "
                                   "PN-T5); j = n is A1 itself; skipped when A1 has no optimum"),
                       "shortfall": ("M0 energy-margin shortfall on the opt stream: nominal linear-NEL margin of the "
                                     "M0 ration minus its margin in each opt state; h = max(0, factor x the (1 - alpha) "
                                     "quantile)"),
                       "stream": "the whole opt stream of the job's root and world (n = streams.opt)"},
            "training_worlds": ["SD-H0", "SD-S2"],
            "lp_time_limit_s": 300.0,
            "status": ("declared (D-539: root 0 only, SD-H0 and SD-S2 only, h grid {0, 0.5 Q, Q, 1.5 Q} united with "
                       "equal steps over [0, A1], higher quantile, negative values clipped at 0)")},
        "DIAG-E1": {
            "what": ("the energy row as the only probabilistic row: planned_arm_cfg(cfg0_v2, problem_v2, <every other "
                     "probabilistic row>, 'E1') -- every other row planned at table values and d_hat (SH-PLAN-*) and "
                     "kept as a diagnostic_only scenario row; per training world and N: exact min-violation MILP and "
                     "M2 (SAA) at alpha_train = alpha; every ration scored by evaluate_reference"),
            "energy_row": ENERGY_ROW_ID,
            "N_ladder": list(OFFICIAL_SETTINGS["N_ladder"]), "alphas": [0.05, 0.10, 0.01],
            "min_violation_time_limit_s": {k: float(v) for k, v in
                                           OFFICIAL_SETTINGS["time_limits_s"]["min_violation"].items()},
            "m2_time_limit_s": float(OFFICIAL_SETTINGS["time_limits_s"]["methods"]),
            "m2_params": {"big_m_mode": OFFICIAL_SETTINGS["M2"]["big_m_mode"], "polish": OFFICIAL_SETTINGS["M2"]["polish"]},
            "training_worlds": ["SD-H0", "SD-S2"],
            "status": "declared (D-539: root 0 only, SD-H0 and SD-S2 only; N and alpha grids as above)"}},
    "convergence": {
        "criterion": OFFICIAL_SETTINGS["convergence_criterion"],
        "N_pair": [512, 1024], "mc_se_multiplier": 2.0, "relative_cost_tolerance": 0.01,
        "mc_se": ("binomial Monte Carlo SE sqrt(r (1 - r) / n_validation) of each selected ration's validation "
                  "rate_upper, one per N; the larger of the two SEs is used (D-539)"),
        "unit": "per root, cell (SD point x objective arm) and alpha; M2 entries only",
        "streams": ["opt (inside the solves)", "validation (the selection records)"],
        "both_not_selected": "reported as status_equal_no_selection; not 'demonstrated'",
        "on_failure": "not_demonstrated (N is not extended)"},
    "compute": {
        "per_state_s": 5.6e-5,
        "per_state_source": ("official-run plan §5: reference evaluation on compute host in the development run "
                             "(holder-reported; not measured by this configuration)"),
        "workers_considered": [4, 8],
        "workers_declared": 8,
        "planning_figures_plan_section_5": {
            "typical_cell_h": [0.5, 0.8], "heavy_cell_h": [1.5, 2.0], "worst_cell_h": 3.7,
            "serial_expected_h": [15, 17], "serial_worst_h": 72, "workers_8_expected_h": [2.5, 3.0],
            "workers_8_worst_h": [9, 10], "workers_4_expected_h": 5,
            "source": "docs/official_run_v2_plan_20260926.md §5 (planning agent's estimate; not a measurement)"},
        "threads": "one process per job; BLAS single-threaded (OPENBLAS / OMP / MKL_NUM_THREADS = 1)"},
}


def official_jobs(roots: Optional[Sequence[int]] = None, *, config: Optional[Mapping[str, Any]] = None
                  ) -> list[dict[str, Any]]:
    """The official job list: (root k, cell) for every cell of the root's matrix row, then (root k, diagnostics) for
    the roots of ``matrix.diagnostics_roots_k`` (D-539: root 0 only).

    ``roots``: root **indices** k (default ``config["roots"]["root_k"]`` = 0, 1, 2) into the reserved roots; a value
    that is not such an index (e.g. a seed) raises -- no seed ever enters a job.  Data only: nothing is drawn."""
    cfg = OFFICIAL_CONFIG if config is None else config
    ks = list(cfg["roots"]["root_k"] if roots is None else roots)
    n_res = int(cfg["roots"]["n_reserved_roots"])
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n_res:
            raise ValueError(f"official_jobs: a root is given by its index k in 0..{n_res - 1}, never by its seed")
    if len(set(ks)) != len(ks):
        raise ValueError("official_jobs: a root index appears twice")
    cells = {c["cell_id"]: c for c in cfg["cells"]}
    jobs: list[dict[str, Any]] = []
    diag_roots = [int(x) for x in cfg["matrix"].get("diagnostics_roots_k", [])]
    for k in ks:
        for cid in cfg["matrix"]["cells_by_root_k"][str(k)]:
            c = cells[cid]
            jobs.append({"job_id": f"root{k}__{cid}", "root_k": k, "kind": "cell", "cell_id": cid, "sd": c["sd"],
                         "objective": c["objective"], "role": c["role"]})
        if k in diag_roots:                                   # D-539: the diagnostics job runs on root 0 only
            jobs.append({"job_id": f"root{k}__diagnostics", "root_k": k, "kind": "diagnostics", "cell_id": None,
                         "diagnostics": list(cfg["diagnostics"]["official_jobs"]), "role": DIAGNOSTIC_ROLE})
    return jobs


# =================================================================================================================
# batch 3 (§1 "诊断", §3 test 10): DIAG-T45 and DIAG-E1 -- flag ``diagnostic``, never comparable-set members
# =================================================================================================================
#: delta* at or below this (canonical fraction of DM) counts as 0 (the starch rows are not binding)
DELTA_ZERO_TOL = 1e-9
_STAGE2_REL_SLACK = 1e-7


def _nominal_lp_data(problem: Any, exclude: Sequence[str] = ()) -> dict[str, Any]:
    """x-space rows (x = q d_hat, kg DM) of every optimisation row (structural + probabilistic at the nominal state and
    d_hat) of ``problem`` except ``exclude`` -- the constraint set of M0."""
    cc_all = problem.compiled
    excl = set(exclude)
    keep = [int(k) for k in optimization_indices(cc_all) if cc_all.constraint_ids[int(k)] not in excl]
    cc = cc_all.subset(keep)
    dh = np.asarray(problem.dm_estimates(), dtype=float)
    rows = linear_rows(cc, problem.nominal_theta(), dh, d_hat=dh)
    if np.any(rows.missing[0]):
        raise ValueError("DIAG-T45: missing nominal coefficients")
    return {"cc": cc, "ids": list(cc.constraint_ids), "A": np.asarray(rows.A[0], dtype=float) / dh[None, :],
            "b": np.asarray(rows.b, dtype=float), "is_eq": np.asarray(rows.is_eq, dtype=bool), "dh": dh,
            "c": np.asarray(problem.price_vector(), dtype=float) / dh}


def _planned_dm_row(lp: dict[str, Any]) -> tuple[str, float]:
    """The equality row that fixes the planned DM (equal x-space weights, e.g. SH-DM-PLAN) and its planned DM."""
    for k in np.flatnonzero(lp["is_eq"]):
        a = lp["A"][k]
        if a[0] != 0 and np.all(np.abs(a - a[0]) <= 1e-12 * abs(a[0])):
            return lp["ids"][k], float(lp["b"][k] / a[0])
    raise ValueError("DIAG-T45: A2 needs a fixed planned DM (an equality row with equal x-space weights, SH-DM-PLAN); "
                     "without it a relaxation in % DM is not linear")


def _lp_extra(lp: dict[str, Any], *, col: Mapping[str, float], shift: Mapping[str, float], c_x: np.ndarray,
              c_y: float, y_bounds: tuple[float, float], opts: SolverOptions) -> Any:
    """``min c_x x + c_y y`` s.t. ``A x + col[row] y <= b[row] - shift[row]`` (inequality rows), the equality rows,
    ``x >= 0`` and ``y`` in ``y_bounds`` (one extra scalar variable)."""
    I = lp["A"].shape[1]
    ub = ~lp["is_eq"]
    ids_ub = [lp["ids"][k] for k in np.flatnonzero(ub)]
    A_ub = np.hstack([lp["A"][ub], np.zeros((int(ub.sum()), 1))])
    b_ub = lp["b"][ub].copy()
    for cid, v in col.items():
        A_ub[ids_ub.index(cid), I] = float(v)
    for cid, v in shift.items():
        b_ub[ids_ub.index(cid)] -= float(v)
    A_eq = np.hstack([lp["A"][~ub], np.zeros((int((~ub).sum()), 1))])
    return run_linprog(np.r_[np.asarray(c_x, dtype=float), float(c_y)], A_ub, b_ub, A_eq, lp["b"][~ub],
                       np.r_[np.zeros(I), y_bounds[0]], np.r_[np.full(I, np.inf), y_bounds[1]], opts)


def _ok(out: Any) -> bool:
    return out.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT) and out.x is not None


def _decision(problem: Any, lp: dict[str, Any], x: np.ndarray, method_id: str) -> RationDecision:
    I = lp["A"].shape[1]
    q = np.maximum(np.asarray(x[:I], dtype=float), 0.0) / lp["dh"]
    return RationDecision(tuple(problem.ingredient_ids), q, lp["dh"], method_id, information_state="t0_reference_only")


def _max_headroom(problem: Any, lp: dict[str, Any], energy_row: str, opts: SolverOptions, label: str,
                  unit_factor: float) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Stage 1: max t s.t. nominal energy margin >= t and every row of ``lp``; stage 2: the cheapest ration whose
    headroom is within a relative 1e-7 of the maximum."""
    I = lp["A"].shape[1]
    s1 = _lp_extra(lp, col={energy_row: 1.0}, shift={}, c_x=np.zeros(I), c_y=-1.0, y_bounds=(-np.inf, np.inf), opts=opts)
    rec: dict[str, Any] = {"status": str(s1.status), "headroom_max": None, "stage2_status": None, "has_ration": False,
                           "label": label}
    if not _ok(s1):
        return rec, None
    t = float(s1.x[I])
    t_fix = t - _STAGE2_REL_SLACK * max(1.0, abs(t))
    s2 = _lp_extra(lp, col={energy_row: 1.0}, shift={}, c_x=lp["c"], c_y=0.0, y_bounds=(t_fix, t_fix), opts=opts)
    rec.update({"headroom_max": t / unit_factor, "headroom_max_canonical": t, "stage2_status": str(s2.status)})
    if not _ok(s2):
        return rec, None
    dec = _decision(problem, lp, s2.x, "DIAG-T45")
    rec.update({"has_ration": True, "cost_usd_per_head_d": float(problem.price_vector() @ dec.q_as_fed)})
    return rec, {"label": label, "decision": dec}


def _energy_margins(problem: Any, energy_row: str, q: np.ndarray, theta: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Linear-row margin (canonical units; >= 0 met) of the energy row for ration ``q`` in states ``(theta, d)``."""
    cc = problem.compiled
    k = list(cc.constraint_ids).index(energy_row)
    ccE = cc.subset([k])
    rows = linear_rows(ccE, theta, d, d_hat=np.asarray(problem.dm_estimates(), dtype=float))
    return -(np.asarray(rows.A[:, 0, :], dtype=float) @ np.asarray(q, dtype=float) - float(rows.b[0]))


def m0_energy_shortfall(problem: Any, opt: Any, *, energy_row: str = ENERGY_ROW_ID, opts: Optional[SolverOptions] = None,
                        alphas: Sequence[float] = (0.10, 0.05, 0.01), quantile_method: str = "higher"
                        ) -> dict[str, Any]:
    """The M0 energy-margin shortfall on the **opt** stream (DIAG-T45 h grid): M0's nominal linear-NEL margin minus its
    margin in each opt state; the (1 - alpha) quantiles (canonical units).  Any other stream raises ``LeakageError``."""
    require_stream(opt, ("opt",), "DIAG-T45 (M0 energy-margin shortfall)")
    opts = opts or SolverOptions(time_limit_s=300.0)
    m0 = get_method("M0_nominal")(problem, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    out: dict[str, Any] = {"m0_status": str(m0.status), "n_states": int(opt.n_draws), "quantiles": {}}
    if m0.decision is None:
        out.update({"n_defined": 0, "n_undefined": int(opt.n_draws)})
        return out
    q = np.asarray(m0.decision.q_as_fed, dtype=float)
    dh = np.asarray(problem.dm_estimates(), dtype=float)
    m_nom = float(_energy_margins(problem, energy_row, q, problem.nominal_theta()[None], dh[None])[0])
    m_s = _energy_margins(problem, energy_row, q, opt.theta, opt.d)
    sf = m_nom - m_s
    ok = np.isfinite(sf)
    out.update({"n_defined": int(ok.sum()), "n_undefined": int((~ok).sum())})
    if ok.any():
        out["quantiles"] = {str(a): float(np.quantile(sf[ok], 1.0 - float(a), method=quantile_method)) for a in alphas}
    return out


def _h_tag(g: Mapping[str, Any], i: int) -> str:
    """Label part of a DIAG-T45 grid point: its definition, never its value (h derives from restricted quantities)."""
    src = g.get("source")
    if src == "zero":
        return "h=0"
    if src == "A1_equal_subdivision":
        return f"h={g['j']}/{g['n']}xA1"
    if src == "M0_shortfall_quantile":
        return f"h={float(g['factor']):g}xQ(1-{float(g['alpha']):g})"
    return f"h=grid#{i}"


def diag_t45_headroom(problem_v2: Any, lin: Any, worlds: Mapping[str, Any], *, alphas: Optional[Sequence[float]] = None,
                      h_factors: Optional[Sequence[float]] = None, include_zero: Optional[bool] = None,
                      h_grid: Optional[Sequence[float]] = None, energy_row: str = ENERGY_ROW_ID,
                      relaxed_rows: Sequence[str] = T45_RELAXED_ROWS, opts: Optional[SolverOptions] = None,
                      score: Optional[Callable[[Any], dict]] = None, tag: str = "DIAG",
                      quantile_method: Optional[str] = None,
                      a1_subdivisions: Optional[int] = None) -> dict[str, Any]:
    """DIAG-T45 (official-run plan §1; declared in :data:`OFFICIAL_CONFIG` ``diagnostics.DIAG-T45``): three
    deterministic nominal LPs on ``problem_v2`` (every optimisation row at the nominal composition and ``d_hat``).

    * **A0**: the maximum nominal linear-NEL headroom (energy-row margin) with every row, and the cheapest ration at it;
    * **A1**: the same without the starch rows ``relaxed_rows`` (PN-T4, PN-T5): ``A1 >= A0``, the gap is what the starch
      rows cost in energy headroom;
    * **A2(h)**: the minimum common relaxation ``delta >= 0`` of the ``relaxed_rows`` bounds, in their natural (declared)
      unit -- percentage points of DM -- such that the nominal headroom is >= h, and the cheapest ration at ``delta*``.
      A concentration bound ``K`` relaxed to ``K + delta`` adds ``-delta * D`` to the row, linear because the planned DM
      ``D`` is fixed by SH-DM-PLAN.  ``delta*(h) = 0`` exactly when ``h <= A0`` (the rows are not binding), ``delta*`` is
      non-decreasing in h, and A2 is infeasible above A1.

    The h grid of each training world ``worlds[sd]`` (its ``draws["opt"]``, the opt stream only) is
    ``max(0, factor x Q_{1-alpha})`` of the M0 energy-margin shortfall (:func:`m0_energy_shortfall`) over the declared
    alphas and factors, plus 0, plus (D-539) the equal steps ``j x A1 / n``, j = 1..n (``a1_subdivisions`` = n, default
    ``h_grid.a1_equal_subdivisions``) that cover [0, A1]; ``h_grid`` (canonical units) overrides it for every world
    (tests).  With ``worlds``
    empty and ``h_grid`` given, one pseudo-world ``"explicit_h_grid"`` is solved.

    ``score(decision) -> {evaluation_world: row block}`` (the driver's ``reference_scorer``: ``evaluate_reference`` on
    every evaluation world's test stream) scores every distinct ration once; the rows are ``ration_kind =
    "diagnostic"``, labels ``<tag>:DIAG-T45:...``.  Nothing is written; the result keeps the decisions under
    ``rations`` (restricted: q)."""
    spec = OFFICIAL_CONFIG["diagnostics"]["DIAG-T45"]
    alphas = list(spec["h_grid"]["alphas"] if alphas is None else alphas)
    h_factors = list(spec["h_grid"]["factors"] if h_factors is None else h_factors)
    include_zero = bool(spec["h_grid"]["include_zero"] if include_zero is None else include_zero)
    quantile_method = spec["h_grid"]["quantile_method"] if quantile_method is None else quantile_method
    n_sub = int(spec["h_grid"].get("a1_equal_subdivisions", 0) if a1_subdivisions is None else a1_subdivisions)
    opts = opts or SolverOptions(time_limit_s=float(spec["lp_time_limit_s"]))
    cc = problem_v2.compiled
    cids = list(cc.constraint_ids)
    for rid in (energy_row, *relaxed_rows):
        if rid not in cids:
            raise ValueError(f"DIAG-T45: {rid} is not a row of the problem")
    kE = cids.index(energy_row)
    if cc.classes[kE] is not ConstraintClass.PROBABILISTIC_NUTRITION or cc.senses[kE] is not Sense.GE:
        raise ValueError(f"DIAG-T45: {energy_row} must be a probabilistic >= row")
    kR = [cids.index(r) for r in relaxed_rows]
    if any(cc.kinds[k] is not ConstraintKind.CONCENTRATION or cc.senses[k] is not Sense.LE for k in kR):
        raise ValueError("DIAG-T45: the relaxed rows must be <= concentration rows (% DM)")
    if len({cc.units[k] for k in kR}) != 1 or len({float(cc.unit_factor[k]) for k in kR}) != 1:
        raise ValueError("DIAG-T45: the relaxed rows must share one declared unit")
    if lin is not None and tuple(getattr(lin, "ingredient_ids", problem_v2.ingredient_ids)) != tuple(
            problem_v2.ingredient_ids):
        raise ValueError("DIAG-T45: the energy linearisation and the problem have different ingredients")
    uE, uR = float(cc.unit_factor[kE]), float(cc.unit_factor[kR[0]])
    lp_all = _nominal_lp_data(problem_v2)
    lp_no_t = _nominal_lp_data(problem_v2, exclude=relaxed_rows)
    dm_row, D = _planned_dm_row(lp_all)
    rations: list[dict[str, Any]] = []
    a0, r0 = _max_headroom(problem_v2, lp_all, energy_row, opts, f"{tag}:DIAG-T45:A0", uE)
    a1, r1 = _max_headroom(problem_v2, lp_no_t, energy_row, opts, f"{tag}:DIAG-T45:A1", uE)
    for rr, lpname in ((r0, "A0"), (r1, "A1")):
        if rr is not None:
            rations.append({**rr, "diag_lp": lpname, "diag_training_world": None, "diag_h_canonical": None})
    H0c = a0.get("headroom_max_canonical")
    I = lp_all["A"].shape[1]
    cache: dict[float, tuple[dict[str, Any], Optional[dict[str, Any]]]] = {}

    def solve_a2(h: float) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        if h in cache:
            return cache[h]
        col = {rid: -D for rid in relaxed_rows}
        s1 = _lp_extra(lp_all, col=col, shift={energy_row: h}, c_x=np.zeros(I), c_y=1.0, y_bounds=(0.0, np.inf),
                       opts=opts)
        rec: dict[str, Any] = {"status": str(s1.status), "feasible": _ok(s1), "delta_star": None,
                               "delta_star_canonical": None, "binding": None, "stage2_status": None, "has_ration": False}
        rr = None
        if _ok(s1):
            dlt = max(0.0, float(s1.x[I]))
            rec.update({"delta_star_canonical": dlt, "delta_star": dlt / uR, "binding": bool(dlt > DELTA_ZERO_TOL)})
            d_fix = dlt + (DELTA_ZERO_TOL if dlt > DELTA_ZERO_TOL else 0.0)
            s2 = _lp_extra(lp_all, col=col, shift={energy_row: h}, c_x=lp_all["c"], c_y=0.0, y_bounds=(d_fix, d_fix),
                           opts=opts)
            rec["stage2_status"] = str(s2.status)
            if _ok(s2):
                dec = _decision(problem_v2, lp_all, s2.x, "DIAG-T45")
                rec.update({"has_ration": True, "cost_usd_per_head_d": float(problem_v2.price_vector() @ dec.q_as_fed)})
                rr = {"decision": dec}
        cache[h] = (rec, rr)
        return cache[h]

    per_world: dict[str, Any] = {}
    targets: list[tuple[str, Any]] = list(worlds.items()) if worlds else []
    if not targets:
        if h_grid is None:
            raise ValueError("DIAG-T45: give training worlds (opt draws) or an explicit h_grid")
        targets = [("explicit_h_grid", None)]
    for sd, wd in targets:
        wrec: dict[str, Any] = {}
        if wd is not None:
            draws = (wd or {}).get("draws") if isinstance(wd, Mapping) else None
            if not draws or "opt" not in draws:
                raise ValueError(f"DIAG-T45: training world {sd} has no opt draws")
            sf = m0_energy_shortfall(problem_v2, draws["opt"], energy_row=energy_row, opts=opts, alphas=alphas,
                                     quantile_method=quantile_method)
            wrec["shortfall"] = {k: v for k, v in sf.items() if k != "quantiles"}
            wrec["shortfall_quantiles_canonical"] = dict(sf["quantiles"])
        if h_grid is not None:
            grid = [{"h_canonical": float(h), "alpha": None, "factor": None, "source": "explicit"} for h in h_grid]
        else:
            grid = []
            for a in alphas:
                qv = wrec["shortfall_quantiles_canonical"].get(str(a))
                for f in h_factors:
                    if qv is None:
                        grid.append({"h_canonical": None, "alpha": a, "factor": f, "source": "no_M0_shortfall"})
                    else:
                        grid.append({"h_canonical": max(0.0, float(f) * qv), "alpha": a, "factor": f,
                                     "clipped_at_zero": bool(float(f) * qv < 0.0), "source": "M0_shortfall_quantile"})
            if include_zero:
                grid.append({"h_canonical": 0.0, "alpha": None, "factor": 0.0, "source": "zero"})
            a1c = a1.get("headroom_max_canonical")
            if n_sub > 0 and a1c is not None and float(a1c) > 0.0:          # D-539: equal steps covering [0, A1]
                grid += [{"h_canonical": float(a1c) * j / n_sub, "alpha": None, "factor": None, "j": j, "n": n_sub,
                          "source": "A1_equal_subdivision"} for j in range(1, n_sub + 1)]
            elif n_sub > 0:
                wrec["a1_grid"] = "not_available (A1 has no positive optimum)"
        a2 = []
        for g in grid:
            if g["h_canonical"] is None:
                a2.append({**g, "status": "not_run", "feasible": None})
                continue
            rec, rr = solve_a2(float(g["h_canonical"]))
            label = f"{tag}:DIAG-T45:A2[{sd},{_h_tag(g, len(a2))}]"      # the label names the grid point, not h
            a2.append({**g, "h": g["h_canonical"] / uE, **rec, "label": label if rec["has_ration"] else None})
            if rr is not None:
                rations.append({"label": label, "decision": rr["decision"], "diag_lp": "A2", "diag_training_world": sd,
                                "diag_h_canonical": g["h_canonical"]})
        solved = sorted((x for x in a2 if x.get("feasible") is not None), key=lambda x: x["h_canonical"])
        feas = [x for x in solved if x["feasible"]]
        mono = all(b["delta_star_canonical"] >= a["delta_star_canonical"] - DELTA_ZERO_TOL for a, b in zip(feas, feas[1:]))
        first_inf = next((x["h_canonical"] for x in solved if not x["feasible"]), None)
        mono = mono and (first_inf is None or all(x["h_canonical"] < first_inf for x in feas))
        tol_h = _STAGE2_REL_SLACK * max(1.0, abs(H0c)) if H0c is not None else None
        zero_iff = (None if H0c is None else
                    all((x["delta_star_canonical"] <= DELTA_ZERO_TOL) == (x["h_canonical"] <= H0c + tol_h) for x in feas
                        if abs(x["h_canonical"] - H0c) > tol_h))
        per_world[sd] = {**wrec, "A2": a2, "delta_monotone_in_h": bool(mono), "delta_zero_iff_h_le_A0": zero_iff,
                         "n_h": len(grid), "n_feasible": len(feas)}
    out: dict[str, Any] = {
        "diagnostic_id": "DIAG-T45", "flag": "diagnostic", "role": DIAGNOSTIC_ROLE, "comparable_set_member": False,
        "what": spec["what"], "problem_id": str(problem_v2.problem_id), "energy_row": energy_row,
        "relaxed_rows": list(relaxed_rows), "headroom_unit": cc.units[kE], "delta_unit": cc.units[kR[0]],
        "planned_dm_row": dm_row, "lin_fingerprint": (lin.fingerprint() if hasattr(lin, "fingerprint") else None),
        "A0": a0, "A1": a1,
        "A1_headroom_ge_A0": (None if a0["headroom_max"] is None or a1["headroom_max"] is None
                              else bool(a1["headroom_max_canonical"] >= a0["headroom_max_canonical"] - 1e-9)),
        "worlds": per_world, "rations": rations, "n_distinct_lp_rations": len({r["decision"].q_as_fed.tobytes()
                                                                              for r in rations}),
        "restricted_fields": ["rations (q)", "A0/A1 headroom and cost", "A2 delta* and cost",
                              "shortfall quantiles (nominal composition and d_hat enter)"]}
    if score is not None:
        out["rows"] = score_diagnostic_rations(rations, score, diagnostic_id="DIAG-T45", tag=tag)
    return out


def diag_e1_arm(cfg0_v2: dict, problem_v2: Any, *, planned_arm_cfg: Callable[..., tuple[dict, dict]],
                energy_row: str = ENERGY_ROW_ID, mode: str = "pilot") -> tuple[dict, Any, dict[str, Any]]:
    """The DIAG-E1 arm: every probabilistic row of ``problem_v2`` except ``energy_row`` planned at table values and
    ``d_hat`` by ``planned_arm_cfg`` (the ablation driver's MAIN9 construction), so the energy row is the **only**
    probabilistic row (checked on the compiled arm).  Returns ``(cfg_e1, problem_e1, record)``."""
    cc = problem_v2.compiled
    prob = [cid for cid, cls in zip(cc.constraint_ids, cc.classes) if cls is ConstraintClass.PROBABILISTIC_NUTRITION]
    if energy_row not in prob:
        raise ValueError(f"DIAG-E1: {energy_row} is not a probabilistic row of the problem")
    planned = tuple(c for c in prob if c != energy_row)
    cfg_e1, rec = planned_arm_cfg(cfg0_v2, problem_v2, planned, "E1")
    problem_e1, rep = build_problem(cfg_e1, mode=mode)
    if not rep.ok:
        raise ValueError(f"DIAG-E1: the E1 arm does not validate ({len(rep.errors)} errors)")
    c1 = problem_e1.compiled
    prob_e1 = [cid for cid, cls in zip(c1.constraint_ids, c1.classes) if cls is ConstraintClass.PROBABILISTIC_NUTRITION]
    if prob_e1 != [energy_row]:
        raise ValueError("DIAG-E1: the E1 arm must have the energy row as its only probabilistic row")
    return cfg_e1, problem_e1, {"planned_rows": list(planned), "probabilistic_rows": prob_e1,
                                "problem_id": str(problem_e1.problem_id), "validator_ok": bool(rep.ok),
                                "construction": rec.get("construction")}


def diag_e1_energy_single_row(cfg0_v2: dict, problem_v2: Any, worlds: Mapping[str, Any], *,
                              planned_arm_cfg: Callable[..., tuple[dict, dict]], Ns: Optional[Sequence[int]] = None,
                              alphas: Optional[Sequence[float]] = None,
                              time_limits: Optional[Mapping[str, float]] = None,
                              m2_time_limit_s: Optional[float] = None, m2_params: Optional[Mapping[str, Any]] = None,
                              training_worlds: Optional[Sequence[str]] = None,
                              score: Optional[Callable[[Any], dict]] = None, tag: str = "DIAG",
                              energy_row: str = ENERGY_ROW_ID, mode: str = "pilot") -> dict[str, Any]:
    """DIAG-E1 (official-run plan §1; declared in :data:`OFFICIAL_CONFIG` ``diagnostics.DIAG-E1``): the energy row as the
    only probabilistic row (:func:`diag_e1_arm`); per training world (its opt draws only) and N: the exact
    min-violation MILP (``run_dev_case_v1.min_violations``, time limit per N) and M2 at ``alpha_train = alpha`` for
    every declared alpha (no validation selection: a diagnostic, not a method).  Every ration is scored by ``score``
    (``evaluate_reference`` on every evaluation world's test stream); rows ``ration_kind = "diagnostic"``, labels
    ``<tag>:DIAG-E1[<world>]:M2[N=..,alpha_train=..]``.  The min-violation counts are about the N training scenarios
    only."""
    spec = OFFICIAL_CONFIG["diagnostics"]["DIAG-E1"]
    Ns = [int(n) for n in (spec["N_ladder"] if Ns is None else Ns)]
    alphas = [float(a) for a in (spec["alphas"] if alphas is None else alphas)]
    tl = dict(spec["min_violation_time_limit_s"] if time_limits is None else time_limits)
    m2_tl = float(spec["m2_time_limit_s"] if m2_time_limit_s is None else m2_time_limit_s)
    m2p = dict(spec["m2_params"] if m2_params is None else m2_params)
    cfg_e1, problem_e1, arm = diag_e1_arm(cfg0_v2, problem_v2, planned_arm_cfg=planned_arm_cfg, energy_row=energy_row,
                                          mode=mode)
    opts = SolverOptions(time_limit_s=m2_tl, mip_rel_gap=float(OFFICIAL_CONFIG["solver"]["mip_rel_gap"]))
    from ration_reliability.optimization.chance_saa import allowed_violations
    rations: list[dict[str, Any]] = []
    per_world: dict[str, Any] = {}
    for sd in (list(worlds) if training_worlds is None else list(training_worlds)):
        wd = worlds[sd]
        draws = wd.get("draws") if isinstance(wd, Mapping) else None
        if not draws or "opt" not in draws:
            raise ValueError(f"DIAG-E1: training world {sd} has no opt draws")
        opt = draws["opt"]
        require_stream(opt, ("opt",), "DIAG-E1")
        recs = []
        for N in Ns:
            if N > int(opt.n_draws):
                raise ValueError(f"DIAG-E1: N = {N} exceeds the opt stream ({opt.n_draws} draws)")
            t_mv = float(tl.get(str(N), 60.0))
            mv = DRV.min_violations(problem_e1, opt, N, time_limit=t_mv)
            mrec = {k: v for k, v in mv.items() if k != "x"}
            mrec.update({"time_limit_s": t_mv, "label": OFFICIAL_CONFIG["diagnostics"]["min_violation"]["label"],
                         "attainable_training_rate_lower": (None if mv["min_violations_lower"] is None
                                                            else mv["min_violations_lower"] / N)})
            m2 = []
            for a in alphas:
                res = get_method("M2_joint_chance_saa")(problem_e1, opt_draws=opt,
                                                        params={"alpha_train": a, "n_scenarios": N, **m2p},
                                                        solver_options=opts)
                allowed = allowed_violations(a, N)
                label = f"{tag}:DIAG-E1[{sd}]:M2[N={N},alpha_train={a:g}]"
                has = bool(res.has_solution)
                m2.append({"alpha_train": a, "allowed_violations": allowed, "status": str(res.status),
                           "mip_gap": None if res.mip_gap is None else float(res.mip_gap), "has_ration": has,
                           "conflict_verdict": DRV.conflict_verdict(mv, allowed),
                           "cost_usd_per_head_d": float(res.objective) if has else None,
                           "label": label if has else None, "time_limit_s": m2_tl})
                if has:
                    rations.append({"label": label, "decision": res.decision, "diag_lp": "E1_M2",
                                    "diag_training_world": sd, "diag_N": N, "diag_alpha_train": a})
            recs.append({"N": N, "min_violation": mrec, "M2": m2})
        per_world[sd] = {"opt_n_draws": int(opt.n_draws), "per_N": recs}
    out: dict[str, Any] = {"diagnostic_id": "DIAG-E1", "flag": "diagnostic", "role": DIAGNOSTIC_ROLE,
                           "comparable_set_member": False, "what": spec["what"], "arm": arm, "energy_row": energy_row,
                           "N_ladder": Ns, "alphas": alphas, "worlds": per_world, "rations": rations,
                           "restricted_fields": ["rations (q)", "costs"]}
    if score is not None:
        out["rows"] = score_diagnostic_rations(rations, score, diagnostic_id="DIAG-E1", tag=tag)
    return out


def score_diagnostic_rations(rations: Sequence[Mapping[str, Any]], score: Callable[[Any], dict], *,
                             diagnostic_id: str, tag: str = "DIAG") -> list[dict[str, Any]]:
    """Score every distinct diagnostic ration once (dedupe by the q vector) with ``score(decision) -> {world: block}``;
    one row per (ration occurrence, evaluation world), ``ration_kind = "diagnostic"``, ``comparable_set_member =
    False``, no target alpha -- the comparable-set code reads ``ration_kind == "method"`` rows only."""
    cache: dict[tuple, dict] = {}
    rows: list[dict[str, Any]] = []
    for rr in rations:
        dec = rr["decision"]
        key = (tuple(dec.ingredient_ids), np.asarray(dec.q_as_fed, dtype=float).tobytes())
        first = key not in cache
        if first:
            cache[key] = score(dec)
        meta = {k: v for k, v in rr.items() if k != "decision"}
        for world, blk in cache[key].items():
            rows.append({**meta, "cell_id": f"{tag}:{diagnostic_id}", "diagnostic_id": diagnostic_id,
                         "ration_kind": "diagnostic", "flag": "diagnostic", "role": DIAGNOSTIC_ROLE,
                         "comparable_set_member": False, "method_id": diagnostic_id, "target_alpha": None,
                         "has_ration": True, **blk, "evaluation_world": world, "first_occurrence_of_q_hash": first})
    return rows


# =================================================================================================================
# batch 3 (§6.4, §3 test 11a): the N convergence report -- opt / validation only
# =================================================================================================================
_M2_N_RE = re.compile(r"M2\[N=(\d+)")


def _stream_kind(stream_id: Any) -> Optional[str]:
    """``validation`` / ``test`` / ``opt`` / other from a stream id such as ``root=<r>/validation/0`` (the root is never
    returned or printed)."""
    parts = [p for p in str(stream_id or "").split("/") if p]
    for kind in ("test", "validation", "opt"):
        if kind in parts:
            return kind
    return None


def _selected_candidate(sel: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if sel.get("status") != "selected":
        return None
    return next((c for c in sel.get("candidates") or [] if c.get("value") == sel.get("selected_value")), None)


def _normalise_selection_tables(selection_tables: Any) -> list[dict[str, Any]]:
    if isinstance(selection_tables, Mapping):
        out = []
        for key, tables in selection_tables.items():
            root_k, cell_id = key
            out.append({"root_k": root_k, "cell_id": cell_id, "tables": tables})
        return out
    return [dict(r) for r in selection_tables]


def convergence_report(selection_tables: Any, *, config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """N convergence per root, cell (SD point x objective arm) and alpha (official-run plan §6.4; D-533):
    **demonstrated** iff the M2 selection status is equal at N = 512 and N = 1024, both are ``selected``,
    ``|delta validation rate_upper| <= 2 MC SE`` and the relative cost change is <= 1 %; otherwise ``not_demonstrated``
    (N is not extended), ``status_equal_no_selection`` when neither N selects a value, ``not_applicable`` when the N
    pair is not in the tables.  MC SE = ``sqrt(r (1 - r) / n_validation)`` of each selected ration's validation
    rate_upper, the larger of the two.

    ``selection_tables``: ``{(root_k, cell_id): tables}`` or ``[{"root_k", "cell_id", "tables"}]`` where ``tables`` is the
    selection-table list of ``run_dev_case_v1.run_method_block`` (``{"label", "method_id", "target_alpha",
    "selection"}``).  Only the selection records are read -- they come from the opt stream (solves) and the validation
    stream (screen); a record whose stream is not a validation stream raises (the test stream is never read here).
    The report carries booleans, validation rates, the selected alpha_train and the relative cost change -- no absolute
    cost and no stream id (stream ids contain the root)."""
    cfg = (OFFICIAL_CONFIG if config is None else config)["convergence"]
    n_lo, n_hi = (int(n) for n in cfg["N_pair"])
    k_se, tol_c = float(cfg["mc_se_multiplier"]), float(cfg["relative_cost_tolerance"])
    cells = {c["cell_id"]: c for c in (OFFICIAL_CONFIG if config is None else config).get("cells", [])}
    rows: list[dict[str, Any]] = []
    for rec in _normalise_selection_tables(selection_tables):
        by: dict[tuple[float, int], Mapping[str, Any]] = {}
        for t in rec["tables"]:
            if t.get("method_id") != "M2_joint_chance_saa":
                continue
            m = _M2_N_RE.search(str(t.get("label", "")))
            sel = t.get("selection")
            if m is None or sel is None:
                continue
            if _stream_kind(sel.get("validation_stream_id")) != "validation":
                raise ValueError("convergence_report reads validation selections only; a selection record names a "
                                 "stream that is not a validation stream (stream id withheld)")
            by[(float(t["target_alpha"]), int(m.group(1)))] = sel
        cell = cells.get(rec["cell_id"], {})
        for a in sorted({a for a, _ in by}, key=lambda v: (v != 0.05, -v)):
            lo, hi = by.get((a, n_lo)), by.get((a, n_hi))
            row: dict[str, Any] = {"root_k": rec["root_k"], "cell_id": rec["cell_id"], "sd_point": cell.get("sd"),
                                   "objective": cell.get("objective"), "target_alpha": a, "N_pair": [n_lo, n_hi]}
            if lo is None or hi is None:
                rows.append({**row, "verdict": "not_applicable", "converged": False,
                             "reason": f"the tables lack M2 at N = {n_lo} or N = {n_hi}"})
                continue
            s_lo, s_hi = lo.get("status"), hi.get("status")
            row.update({f"status_N{n_lo}": s_lo, f"status_N{n_hi}": s_hi, "status_equal": s_lo == s_hi,
                        f"selected_alpha_train_N{n_lo}": lo.get("selected_value"),
                        f"selected_alpha_train_N{n_hi}": hi.get("selected_value")})
            c_lo, c_hi = _selected_candidate(lo), _selected_candidate(hi)
            if s_lo != s_hi:
                rows.append({**row, "verdict": "not_demonstrated", "converged": False,
                             "reason": "M2 selection status differs between the two N"})
                continue
            if c_lo is None or c_hi is None:
                rows.append({**row, "verdict": "status_equal_no_selection", "converged": False,
                             "reason": "no selected value at either N; the rate and cost criteria do not apply"})
                continue
            n_lo_v, n_hi_v = int(lo["validation_n_draws"]), int(hi["validation_n_draws"])
            r_lo, r_hi = float(c_lo["rate_upper"]), float(c_hi["rate_upper"])
            se = max(math.sqrt(max(r, 0.0) * max(1.0 - r, 0.0) / n) for r, n in ((r_lo, n_lo_v), (r_hi, n_hi_v)))
            d_rate = abs(r_hi - r_lo)
            rate_ok = n_lo_v == n_hi_v and d_rate <= k_se * se + 1e-15
            cost_lo, cost_hi = c_lo.get("cost"), c_hi.get("cost")
            rel = (None if cost_lo is None or cost_hi is None or float(cost_lo) == 0.0
                   else abs(float(cost_hi) - float(cost_lo)) / abs(float(cost_lo)))
            cost_ok = rel is not None and rel <= tol_c + 1e-15
            ok = bool(rate_ok and cost_ok)
            why = []
            if n_lo_v != n_hi_v:
                why.append("the two selections used validation streams of different size")
            elif not rate_ok:
                why.append(f"|delta validation rate_upper| > {k_se:g} MC SE")
            if not cost_ok:
                why.append(f"relative cost change > {tol_c:g}" if rel is not None else "cost missing")
            rows.append({**row, f"validation_rate_upper_N{n_lo}": r_lo, f"validation_rate_upper_N{n_hi}": r_hi,
                         "validation_n": n_hi_v, "mc_se": se, "abs_delta_rate_upper": d_rate,
                         "rate_criterion_met": bool(rate_ok), "relative_cost_change": rel,
                         "cost_criterion_met": bool(cost_ok),
                         "verdict": "demonstrated" if ok else "not_demonstrated", "converged": ok,
                         "reason": "all three criteria met" if ok else "; ".join(why)})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"criterion": cfg["criterion"], "N_pair": [n_lo, n_hi], "mc_se": cfg["mc_se"],
            "streams_read": "selection records only (opt inside the solves, validation for the screen)",
            "test_stream_read": False, "rows": rows, "counts": counts,
            "on_failure": cfg["on_failure"]}


# =================================================================================================================
# batch 4 (official-run plan §0 F3 / F4 / F5, §1, §2 "B4", §3 tests 6 / 7 / 12 / 13, §5, §7): specification pin v2, the
# protocol freeze record and its gate, the official output route, the job runner and the merge.
#
# Order of the official path (F3): every gate below passes BEFORE any stream of a reserved root exists -- the
# authorisation, the root indices, :func:`require_protocol_frozen` (git anchoring, protocol frozen, specification =
# v2 pin, code / data / protocol hashes = the freeze record, access scope, declared matrix), the preflight and the
# environment lock.  Only :func:`require_protocol_frozen` can mint the :class:`OfficialStreamToken` without which
# ``RandomStreams`` refuses a reserved root.  After the draws only the end-of-run identity check can fail; it labels the
# run ``official_invalidated`` (outputs kept and disclosed; a rerun uses the same roots, plan §7).
# =================================================================================================================
#: files whose bytes define the reference problem, the ablation and the official path (docs/reference_problem_v1.md §9
#: freeze + batch 4: v2 table and document, the official module, the protocol, the screen / safety-margin statistics and
#: the reserved-root guard)
SPEC_FILES: tuple[str, ...] = (
    "configs/dev_case_v1/animal.yaml", "configs/dev_case_v1/inventory.yaml", "configs/dev_case_v1/prices.yaml",
    "configs/dev_case_v1/constraints.yaml", "configs/dev_case_v1/energy.yaml", "configs/dev_case_v1/README.md",
    "configs/dev_case_v1/reference_constraints.csv", "configs/animal_profile.yaml", "configs/methods.yaml",
    "configs/uncertainty.yaml", "configs/streams_policy.yaml", "reports/sd_scaling_sources.csv",
    "docs/uncertainty_data_layers.md",
    "src/ration_reliability/evaluation/reference.py", "src/ration_reliability/evaluation/evaluator.py",
    "src/ration_reliability/evaluation/stats.py", "src/ration_reliability/nutrition/energy_reference.py",
    "src/ration_reliability/nutrition/domain.py", "src/ration_reliability/nutrition/energy.py",
    "experiments/E1_cost_reliability/run_endpoint_ablation.py", "experiments/E0_verification/run_dev_case_v1.py",
    "experiments/E0_verification/smoke_pipeline.py",
    # batch 4 (plan §2 "B4")
    "experiments/E1_cost_reliability/official_v2.py", "configs/dev_case_v1/reference_constraints_v2.csv",
    "configs/protocol.yaml", "docs/reference_problem_v2.md", "src/ration_reliability/optimization/safety_margin.py",
    "src/ration_reliability/uncertainty/streams.py",
)
#: restricted build record (pins every restricted input by its recorded sha256); entered by hash only
SPEC_RESTRICTED_FILES: tuple[str, ...] = ("data/restricted_local/dev_case_v1/build_report.json",)
#: domain tag of the specification digest of reference problem v2
SPEC_DIGEST_TAG = "reference_problem_v2/spec_fingerprint/v1"
#: the v2 specification pin (inside the code-manifest scope; the v1 pin file stays untouched as history)
FREEZE_PIN_V2 = Path("experiments") / "E1_cost_reliability" / "reference_problem_v2_freeze.json"
FREEZE_PIN_V1_HISTORY = Path("experiments") / "E1_cost_reliability" / "reference_problem_v1_freeze.json"
FREEZE_PIN_SCHEMA = "ration_reliability.reference_problem_freeze_pin/1"
#: the protocol and its freeze record (written once by the lead, committed in the one freeze commit)
PROTOCOL_YAML = Path("configs") / "protocol.yaml"
PROTOCOL_FREEZE = Path("configs") / "protocol_freeze.json"
PROTOCOL_FREEZE_SCHEMA = "ration_reliability.protocol_freeze/1"
STREAMS_POLICY = Path("configs") / "streams_policy.yaml"
#: the restricted data files of the case, entered by sha256 only (the run record's data_paths)
DATA_FILES: tuple[str, ...] = tuple(p.as_posix() for p in (DRV.PROBLEM_YAML, DRV.CELLS_CSV, DRV.ENERGY_JSON,
                                                            DRV.BUILD_REPORT, DRV.BUILD_SCRIPT))
#: the user's authorisation for the official run (restricted; D-536) and the scope it must contain
OFFICIAL_AUTHORISATION_FILE = Path("data") / "restricted_local" / "compute_authorisation_official.json"
OFFICIAL_SCOPE = "run_official_v2"
#: output routes: public tables / restricted mirror of an official run; debug for the rehearsal
OFFICIAL_PUBLIC_ROOT = Path("results") / "official"
OFFICIAL_RESTRICTED_ROOT = Path("data") / "restricted_local" / "official"
DEBUG_ROOT = Path("data") / "restricted_local" / "debug"
OFFICIAL_RUN_PREFIX = "official-"
REHEARSAL_RUN_PREFIX = "debug-rehearsal-"
#: the ablation driver (one process per job: ``<python> <driver> --official-job <run_id> <job_id>``)
DRIVER_PATH = Path("experiments") / "E1_cost_reliability" / "run_endpoint_ablation.py"
#: one BLAS / OpenMP thread per job process (plan §5; D-539)
THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "PYTHONDONTWRITEBYTECODE": "1"}
#: per-directory completion files
SHA_MANIFEST = "SHA256SUMS"
DONE_FILE = "DONE"
FAILED_FILE = "FAILED.json"
MERGE_DONE = "MERGE_DONE"
RUN_PLAN_FILE = "OFFICIAL_RUN.json"
#: the code-manifest subset of the freeze record excludes the two files the freeze commit adds last (no self-reference)
FREEZE_SUBSET_SCHEMA = "ration_reliability.protocol_freeze_code_subset/1"
FREEZE_SUBSET_EXCLUDED: tuple[str, ...] = (PROTOCOL_FREEZE.as_posix(), FREEZE_PIN_V2.as_posix())


class FreezeGateError(Exception):
    """The protocol freeze gate refused (``reasons``: short texts without any value; ``checks``: the evidence)."""

    def __init__(self, reasons: Sequence[str], checks: Optional[Mapping[str, Any]] = None):
        self.reasons = list(reasons)
        self.checks = dict(checks or {})
        super().__init__("; ".join(self.reasons))


class MergeError(Exception):
    """The merge refused (missing / duplicate / unverifiable jobs); nothing was written."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


# ---------------------------------------------------------------------------------------------------------------------
# specification identity and its git anchor (moved here from the ablation driver in batch 4; the driver re-exports)
# ---------------------------------------------------------------------------------------------------------------------
def sha_or_none(p: Path) -> Optional[str]:
    return file_sha256(p) if p.is_file() else None


def spec_fingerprint(repo: Path = REPO) -> dict[str, Any]:
    """sha256 of every specification file (restricted build record by hash only) and one combined digest."""
    repo = Path(repo)
    files = {rel: sha_or_none(repo / rel) for rel in SPEC_FILES}
    restricted = {rel: sha_or_none(repo / rel) for rel in SPEC_RESTRICTED_FILES}
    digest = stable_hash(SPEC_DIGEST_TAG, sorted(files.items()), sorted(restricted.items()))
    return {"schema": "ration_reliability.reference_problem_fingerprint/1", "digest_tag": SPEC_DIGEST_TAG,
            "files": files, "restricted_files_sha256_only": restricted, "digest": digest,
            "missing": sorted([k for k, v in {**files, **restricted}.items() if v is None])}


def frozen_fingerprint(repo: Path = REPO, pin: Path = FREEZE_PIN_V2) -> Optional[dict[str, Any]]:
    """The frozen specification fingerprint of the v2 pin file (``None`` if absent or unreadable)."""
    p = Path(repo) / pin
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict) or doc.get("pin_schema") != FREEZE_PIN_SCHEMA or not isinstance(doc.get("files"), dict):
        return None
    return doc


def _git_out(repo: Path, *args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True,
                           timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return r.returncode, r.stdout.strip()


def freeze_anchor(repo: Path = REPO, pins: Sequence[Path] = (FREEZE_PIN_V2,)) -> dict[str, Any]:
    """Git anchoring of the pin files (read-only git calls; FIX3_BC, C-3 / instruction F.1; batch 4: several pins).

    ``anchored`` only if: ``repo`` is the top level of a git work tree; every pin is tracked; HEAD is the commit that
    last changed every pin (the run's commit IS the frozen commit); and ``git status`` is empty over the code-manifest
    scope, the specification files and the pins (no modified, staged or untracked non-ignored file)."""
    repo = Path(repo)
    pins = [Path(p) for p in pins]
    out: dict[str, Any] = {"pin": pins[0].as_posix(), "pins": {p.as_posix(): None for p in pins}, "anchored": False,
                           "head": None, "pin_commit": None, "dirty_paths": None, "reason": None}
    rc, top = _git_out(repo, "rev-parse", "--show-toplevel")
    if rc != 0:
        out["reason"] = "git unavailable or not a repository"
        return out
    try:
        if Path(top).resolve() != repo.resolve():
            out["reason"] = "repository root is not the git top level"
            return out
    except OSError:
        out["reason"] = "repository root cannot be resolved"
        return out
    rc, head = _git_out(repo, "rev-parse", "HEAD")
    if rc != 0:
        out["reason"] = "no HEAD commit"
        return out
    out["head"] = head
    for p in pins:
        rc, _ = _git_out(repo, "ls-files", "--error-unmatch", "--", p.as_posix())
        if rc != 0:
            out["reason"] = (f"pin file {p.as_posix()} not tracked by git (commit it first; an untracked pin can be "
                             "rewritten silently)")
            return out
        rc, pc = _git_out(repo, "log", "-1", "--format=%H", "--", p.as_posix())
        out["pins"][p.as_posix()] = pc or None
    out["pin_commit"] = out["pins"][pins[0].as_posix()]
    paths = list(CODE_MANIFEST_DIRS) + list(CODE_MANIFEST_ROOT_FILES) + list(SPEC_FILES) + [p.as_posix() for p in pins]
    rc, st = _git_out(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths)
    if rc != 0:
        out["reason"] = "git status failed; cleanliness unknown"
        return out
    # porcelain lines are "XY path"; _git_out strips the whole output, so the first line may have lost its
    # leading status column -- parse from the right (review round 4 found the truncated first path)
    dirty = sorted({(ln[3:] if len(ln) > 3 and ln[2] == " " else ln.split(" ", 1)[-1].lstrip()) for ln in st.splitlines()
                    if ln.strip()})
    out["dirty_paths"] = dirty[:50]
    out["n_dirty_paths"] = len(dirty)
    if dirty:
        out["reason"] = f"work tree not clean over the manifest scope, specification files and pin ({len(dirty)} paths)"
        return out
    if any(not c or c != head for c in out["pins"].values()):
        out["reason"] = ("HEAD is not the commit that last changed the pin (run exactly the frozen commit: re-pin and "
                         "commit, then run without further commits)")
        return out
    out["anchored"] = True
    out["reason"] = "pin tracked; HEAD = pin commit; clean work tree over manifest scope, specification files and pin"
    return out


def freeze_status(repo: Path = REPO) -> dict[str, Any]:
    """Current specification fingerprint vs the v2 pin, plus the git anchor (FIX3_BC, C-3).

    ``status``: ``no_frozen_fingerprint`` / ``differs_from_frozen`` / ``matches_frozen_unanchored`` / ``matches_frozen_
    anchored`` (the only status that allows ``development``; the official run needs more: :func:`require_protocol_
    frozen`)."""
    cur = spec_fingerprint(repo)
    fr = frozen_fingerprint(repo)
    anchor = freeze_anchor(repo)
    if fr is None:
        return {"status": "no_frozen_fingerprint", "current_digest": cur["digest"], "differs": None,
                "pin": FREEZE_PIN_V2.as_posix(), "anchor": anchor}
    diff = sorted(k for k in set(cur["files"]) | set(fr.get("files", {}))
                  if cur["files"].get(k) != fr.get("files", {}).get(k))
    diff += sorted(k for k in set(cur["restricted_files_sha256_only"]) | set(fr.get("restricted_files_sha256_only", {}))
                   if cur["restricted_files_sha256_only"].get(k) != fr.get("restricted_files_sha256_only", {}).get(k))
    same = cur["digest"] == fr.get("digest") and not diff
    status = "differs_from_frozen" if not same else (
        "matches_frozen_anchored" if anchor["anchored"] else "matches_frozen_unanchored")
    return {"status": status, "current_digest": cur["digest"], "frozen_digest": fr.get("digest"), "differs": diff,
            "pin": FREEZE_PIN_V2.as_posix(), "anchor": anchor}


def write_freeze_pin(repo: Path = REPO, *, note: str) -> dict[str, Any]:
    """The v2 pin content for the current specification (``--write-freeze-pin``; lead-invoked, part of the freeze
    commit).  Writing is only the first step: the pin must then be committed, and the run must use exactly that
    commit (:func:`freeze_anchor`).  The v1 pin file is never written again (history)."""
    cur = spec_fingerprint(repo)
    doc = {"pin_schema": FREEZE_PIN_SCHEMA, "reference_problem": "reference_problem_v2", "note": note,
           "pinned_utc": utc_now(), **cur}
    (Path(repo) / FREEZE_PIN_V2).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


# ---------------------------------------------------------------------------------------------------------------------
# the protocol freeze record (configs/protocol_freeze.json) and its gate
# ---------------------------------------------------------------------------------------------------------------------
def read_protocol(repo: Path = REPO) -> dict[str, Any]:
    p = Path(repo) / PROTOCOL_YAML
    if not p.is_file():
        return {}
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    return doc if isinstance(doc, dict) else {}


def code_subset_for_freeze(manifest: Mapping[str, Any],
                           excluded: Sequence[str] = FREEZE_SUBSET_EXCLUDED) -> dict[str, Any]:
    """Digest over every code-manifest entry except the freeze record and the v2 pin (the two files that record the
    others; excluding them avoids a self-reference).  Same entry encoding as ``code_manifest_subset_sha256``."""
    ex = sorted(set(excluded))
    sel = sorted((e for e in manifest["files"] if e["path"] not in ex), key=lambda e: e["path"])
    h = hashlib.sha256()
    h.update(f"{FREEZE_SUBSET_SCHEMA}\n{json.dumps(ex)}\n".encode())
    for e in sel:
        h.update(f"{e['path']}\0{e['kind']}\0{e['sha256'] or 'NONE'}\n".encode("utf-8", "surrogateescape"))
    return {"schema": FREEZE_SUBSET_SCHEMA, "excluded": ex, "file_count": len(sel), "subset_sha256": h.hexdigest(),
            "manifest_complete": bool(manifest.get("complete"))}


def data_hashes(repo: Path = REPO) -> dict[str, Optional[str]]:
    """sha256 of the restricted case data files (hash only; no value is read into the record)."""
    return {rel: sha_or_none(Path(repo) / rel) for rel in DATA_FILES}


def declared_matrix(config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """What the freeze declares about the run: SD points, cells, root indices, the job matrix, the diagnostics scope,
    stream sizes, alphas and the configuration hash (no root value)."""
    C = OFFICIAL_CONFIG if config is None else config
    return {"sd_points": list(C["sd_points"]), "cells": [c["cell_id"] for c in C["cells"]],
            "roots_k": [int(k) for k in C["roots"]["root_k"]], "n_reserved_roots": int(C["roots"]["n_reserved_roots"]),
            "cells_by_root_k": {k: list(v) for k, v in C["matrix"]["cells_by_root_k"].items()},
            "diagnostics": {"jobs": list(C["diagnostics"]["official_jobs"]),
                            "roots_k": [int(k) for k in C["matrix"].get("diagnostics_roots_k", [])],
                            "training_worlds": list(C["diagnostics"].get("training_worlds", []))},
            "jobs": [j["job_id"] for j in official_jobs(config=C)],
            "streams": {k: C["streams"][k] for k in ("opt", "N_ladder", "N_headline", "validation", "test")},
            "alphas": [C["alphas"]["primary"]] + list(C["alphas"]["supplementary"]),
            "evaluation_worlds": list(C["evaluation"]["evaluation_worlds"]),
            "primary_world": C["evaluation"]["primary_world"],
            "primary_assumption_id": C.get("primary_assumption_id"),
            "workers": C["matrix"].get("workers"),
            "official_config_sha256": stable_hash(C)}


_SEEN_MARK = re.compile(r"^(是|yes)|已见|看过|已看")


def decisions_after_seeing_results(repo: Path = REPO) -> list[str]:
    """Decision ids whose DECISIONS.md row says the (development) test results had been seen, plus the ids named by
    ``configs/protocol.yaml splits.seen_before_freeze_decisions``."""
    ids: set[str] = set()
    p = Path(repo) / "DECISIONS.md"
    if p.is_file():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.startswith("| D-"):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) >= 3 and _SEEN_MARK.search(cells[-1][:40] or ""):
                ids.add(cells[0])
    txt = str((read_protocol(repo).get("splits") or {}).get("seen_before_freeze_decisions") or "")
    ids |= set(re.findall(r"D-\d+[a-z]?", txt))
    return sorted(ids, key=lambda d: (int(re.sub(r"\D", "", d) or 0), d))


def _dir_names(p: Path) -> list[str]:
    return sorted(x.name for x in p.iterdir() if x.is_dir()) if p.is_dir() else []


def official_outputs_present(repo: Path = REPO) -> dict[str, bool]:
    """Whether any official output exists (public or restricted mirror): a freeze is refused if it does."""
    repo = Path(repo)
    return {OFFICIAL_PUBLIC_ROOT.as_posix(): bool(_dir_names(repo / OFFICIAL_PUBLIC_ROOT)),
            OFFICIAL_RESTRICTED_ROOT.as_posix(): bool(_dir_names(repo / OFFICIAL_RESTRICTED_ROOT))}


def default_access_scope(repo: Path = REPO) -> dict[str, Any]:
    """Everything seen before the freeze (plan §6.12): development roots and streams (streams policy), the declared
    seen runs and the run directories on disk (names only), the decisions taken after seeing results."""
    repo = Path(repo)
    pol = yaml.safe_load((repo / STREAMS_POLICY).read_text(encoding="utf-8")) if (repo / STREAMS_POLICY).is_file() else {}
    dev = (pol or {}).get("development_material") or {}
    splits = read_protocol(repo).get("splits") or {}
    return {
        "development_roots": [{k: r.get(k) for k in ("root_seed", "root_seed_range", "scope") if r.get(k) is not None}
                              for r in dev.get("whole_roots") or []],
        "development_streams_summary": dev.get("summary"),
        "seen_runs_declared": list(splits.get("test_seen_runs") or []),
        "run_directories_seen": {"results/pilot": _dir_names(repo / "results" / "pilot"),
                                 "results/smoke": _dir_names(repo / "results" / "smoke"),
                                 "logs/px_runs": _dir_names(repo / "logs" / "px_runs")},
        "decisions_after_seeing_results": decisions_after_seeing_results(repo),
        "seen_before_freeze_decisions": splits.get("seen_before_freeze_decisions"),
        "access_log_path": splits.get("access_log_path"),
        "reserved_roots_drawn_before_freeze": False,
        "reserved_roots_evidence": ("no official output directory exists at the freeze (checked by write_protocol_"
                                    "freeze); RandomStreams refuses a reserved root without a token of "
                                    "require_protocol_frozen (streams.py)"),
    }


def _reserved_roots_consistent(repo: Path) -> bool:
    """The derived reserved roots equal configs/streams_policy.yaml (integer comparison; nothing printed)."""
    p = Path(repo) / STREAMS_POLICY
    if not p.is_file():
        return False
    try:
        pol = yaml.safe_load(p.read_text(encoding="utf-8"))
        listed = tuple(int(r["root_seed"]) for r in pol["reserved_formal_streams"]["roots"])
    except Exception:  # noqa: BLE001 -- unreadable policy = inconsistent
        return False
    return listed == ST.reserved_formal_roots()


def write_protocol_freeze(note: str, repo: Path = REPO, *, config: Optional[Mapping[str, Any]] = None,
                          access_scope_extra: Optional[Mapping[str, Any]] = None,
                          overwrite: bool = False) -> dict[str, Any]:
    """Write ``configs/protocol_freeze.json`` (lead-invoked, ``--write-protocol-freeze NOTE``; the same pattern as
    :func:`write_freeze_pin`).  Preconditions (else :class:`FreezeGateError`, nothing written): ``configs/protocol.yaml``
    already says ``freeze.is_frozen: true`` with a ``freeze.timestamp`` and ``splits.test_previously_seen: true``; the
    v2 pin exists and equals the current specification; the code manifest is complete; every data file exists; no
    official output exists; the derived reserved roots equal the streams policy.

    Records: the parent commit (HEAD at writing; the freeze commit that adds this file is its git anchor), the freeze
    time, the protocol sha256, the v2 pin digest, the code-manifest subset digest (everything but this file and the
    pin), the data sha256, the declared matrix (SD points, cells, root indices, jobs, diagnostics scope, sizes) and the
    access scope (development streams / runs seen, decisions taken after seeing results).  Next step: ``git add
    configs/protocol.yaml configs/protocol_freeze.json experiments/E1_cost_reliability/reference_problem_v2_freeze.json``
    and one commit; run exactly that commit."""
    repo = Path(repo)
    reasons: list[str] = []
    proto = read_protocol(repo)
    fz = proto.get("freeze") or {}
    if fz.get("is_frozen") is not True:
        reasons.append("configs/protocol.yaml freeze.is_frozen is not true (set it, with freeze.timestamp, first)")
    if not fz.get("timestamp"):
        reasons.append("configs/protocol.yaml freeze.timestamp is empty")
    if (proto.get("splits") or {}).get("test_previously_seen") is not True:
        reasons.append("configs/protocol.yaml splits.test_previously_seen is not true")
    fs = freeze_status(repo)
    if fs["status"] not in ("matches_frozen_unanchored", "matches_frozen_anchored"):
        reasons.append(f"the v2 pin does not match the current specification ({fs['status']}; run --write-freeze-pin "
                       "after the last specification change)")
    man = code_manifest(repo)
    if not man.get("complete"):
        reasons.append("code manifest incomplete (a file changed while hashing)")
    dh = data_hashes(repo)
    if any(v is None for v in dh.values()):
        reasons.append(f"data files missing: {sorted(k for k, v in dh.items() if v is None)}")
    if (repo / PROTOCOL_FREEZE).exists() and not overwrite:
        reasons.append(f"{PROTOCOL_FREEZE.as_posix()} exists (a freeze is written once)")
    if any(official_outputs_present(repo).values()):
        reasons.append("official output already exists; a freeze after an official draw is not allowed (plan §7)")
    if not _reserved_roots_consistent(repo):
        reasons.append("derived reserved roots differ from configs/streams_policy.yaml (values withheld)")
    if reasons:
        raise FreezeGateError(reasons, {"freeze_status": fs["status"]})
    rc, head = _git_out(repo, "rev-parse", "HEAD")
    pin = frozen_fingerprint(repo)
    doc = {
        "schema": PROTOCOL_FREEZE_SCHEMA,
        "note": note,
        "frozen_utc": utc_now(),
        "written_by": "official_v2.write_protocol_freeze (lead-invoked; --write-protocol-freeze)",
        "code_commit_parent": head if rc == 0 else None,
        "code_commit_rule": ("the freeze commit is the commit that adds this file together with configs/protocol.yaml "
                             "(is_frozen: true) and the v2 pin; require_protocol_frozen() checks HEAD = that commit, "
                             "a clean work tree and the content digests below (no self-reference: this file and the "
                             "pin are excluded from the code subset)"),
        "protocol": {"path": PROTOCOL_YAML.as_posix(), "sha256": sha_or_none(repo / PROTOCOL_YAML),
                     "freeze_timestamp": str(fz.get("timestamp"))},
        "spec": {"pin": FREEZE_PIN_V2.as_posix(), "pin_sha256": sha_or_none(repo / FREEZE_PIN_V2),
                 "digest": (pin or {}).get("digest"), "digest_tag": SPEC_DIGEST_TAG},
        "code_manifest_subset": code_subset_for_freeze(man),
        "code_manifest_sha256_at_freeze": man.get("manifest_sha256"),
        "data_sha256": dh,
        "declared": declared_matrix(config),
        "reserved_roots": {"label_rule": ST.RESERVED_ROOT_LABEL_TEMPLATE, "n": ST.N_RESERVED_ROOTS,
                           "values": "not written here (derived from the label rule; configs/streams_policy.yaml)",
                           "consistent_with_streams_policy": True, "drawn_before_freeze": False},
        "access_scope": {**default_access_scope(repo), **dict(access_scope_extra or {}), "note": note},
    }
    (repo / PROTOCOL_FREEZE).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


_ACCESS_SCOPE_KEYS = ("development_roots", "seen_runs_declared", "decisions_after_seeing_results")


def protocol_freeze_check(repo: Path = REPO, *, roots_k: Sequence[int] = (0, 1, 2),
                          config: Optional[Mapping[str, Any]] = None) -> tuple[list[str], dict[str, Any]]:
    """Every freeze-gate check without minting a token: ``(reasons, facts)``; empty ``reasons`` = frozen and anchored.
    Used by :func:`require_protocol_frozen` (before any draw) and by the end-of-run identity check (after the draws)."""
    repo = Path(repo)
    reasons: list[str] = []
    facts: dict[str, Any] = {}
    ks = list(roots_k)
    C = OFFICIAL_CONFIG if config is None else config
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k not in C["roots"]["root_k"] for k in ks) \
            or len(set(ks)) != len(ks):
        reasons.append("root indices must be distinct values of 0, 1, 2 (never a seed)")
    fz_path = repo / PROTOCOL_FREEZE
    doc: Optional[dict] = None
    if not fz_path.is_file():
        reasons.append(f"{PROTOCOL_FREEZE.as_posix()} missing (the protocol is not frozen)")
    else:
        try:
            doc = json.loads(fz_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            reasons.append(f"{PROTOCOL_FREEZE.as_posix()} is not valid JSON")
        if doc is not None and (not isinstance(doc, dict) or doc.get("schema") != PROTOCOL_FREEZE_SCHEMA):
            reasons.append(f"{PROTOCOL_FREEZE.as_posix()} has the wrong schema")
            doc = None
        facts["freeze_record_sha256"] = file_sha256(fz_path)
    anchor = freeze_anchor(repo, pins=(FREEZE_PIN_V2, PROTOCOL_FREEZE)) if fz_path.is_file() else \
        freeze_anchor(repo, pins=(FREEZE_PIN_V2,))
    facts["anchor"] = anchor
    facts["head"] = anchor.get("head")
    if not anchor["anchored"]:
        reasons.append(f"not git-anchored: {anchor['reason']}")
    proto = read_protocol(repo)
    fz = proto.get("freeze") or {}
    if fz.get("is_frozen") is not True:
        reasons.append("configs/protocol.yaml freeze.is_frozen is not true")
    if not fz.get("timestamp"):
        reasons.append("configs/protocol.yaml freeze.timestamp is empty")
    if (proto.get("splits") or {}).get("test_previously_seen") is not True:
        reasons.append("configs/protocol.yaml splits.test_previously_seen is not true")
    cur = spec_fingerprint(repo)
    pin = frozen_fingerprint(repo)
    facts["spec_digest"] = cur["digest"]
    if pin is None:
        reasons.append(f"v2 pin {FREEZE_PIN_V2.as_posix()} missing or unreadable")
    else:
        differ = sorted(k for k in set(cur["files"]) | set(pin.get("files", {}))
                        if cur["files"].get(k) != pin.get("files", {}).get(k))
        differ += sorted(k for k in set(cur["restricted_files_sha256_only"]) |
                         set(pin.get("restricted_files_sha256_only", {}))
                         if cur["restricted_files_sha256_only"].get(k) != pin.get("restricted_files_sha256_only",
                                                                                   {}).get(k))
        if cur["digest"] != pin.get("digest") or differ:
            reasons.append(f"specification differs from the v2 pin ({len(differ)} files: {differ[:8]})")
        if cur["missing"]:
            reasons.append(f"specification files missing: {cur['missing'][:8]}")
    psha = sha_or_none(repo / PROTOCOL_YAML)
    facts["protocol_sha256"] = psha
    man = code_manifest(repo)
    sub = code_subset_for_freeze(man)
    facts["code_subset_sha256"] = sub["subset_sha256"]
    facts["code_manifest_sha256"] = man.get("manifest_sha256")
    if not man.get("complete"):
        reasons.append("code manifest incomplete (a file changed while hashing)")
    dh = data_hashes(repo)
    if any(v is None for v in dh.values()):
        reasons.append(f"data files missing: {sorted(k for k, v in dh.items() if v is None)}")
    if doc is not None:
        if (doc.get("spec") or {}).get("digest") != (pin or {}).get("digest") or pin is None:
            reasons.append("the freeze record's specification digest differs from the v2 pin")
        if (doc.get("spec") or {}).get("pin_sha256") != sha_or_none(repo / FREEZE_PIN_V2):
            reasons.append("the v2 pin file changed after the freeze record was written")
        if (doc.get("code_manifest_subset") or {}).get("subset_sha256") != sub["subset_sha256"]:
            reasons.append("code-manifest subset digest differs from the freeze record (code or configuration changed)")
        rec_dh = doc.get("data_sha256") or {}
        bad = sorted(k for k in set(rec_dh) | set(dh) if rec_dh.get(k) != dh.get(k))
        if bad:
            reasons.append(f"data hashes differ from the freeze record: {bad}")
        if (doc.get("protocol") or {}).get("sha256") != psha:
            reasons.append("configs/protocol.yaml differs from the freeze record")
        acc = doc.get("access_scope")
        if not isinstance(acc, dict) or any(not acc.get(k) for k in _ACCESS_SCOPE_KEYS):
            reasons.append(f"access_scope missing or incomplete (needs {list(_ACCESS_SCOPE_KEYS)})")
        want = declared_matrix(C)
        got = doc.get("declared") or {}
        bad_keys = sorted(k for k in want if got.get(k) != want[k])
        if bad_keys:
            reasons.append(f"declared SD points / cells / roots / jobs differ from the configuration: {bad_keys}")
        if any(k not in (got.get("roots_k") or []) for k in ks if isinstance(k, int)):
            reasons.append("a requested root index is not declared in the freeze record")
    if not _reserved_roots_consistent(repo):
        reasons.append("derived reserved roots differ from configs/streams_policy.yaml (values withheld)")
    facts["n_reasons"] = len(reasons)
    return reasons, facts


def require_protocol_frozen(repo: Path = REPO, *, roots_k: Sequence[int] = (0, 1, 2),
                            config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The official freeze gate (plan §2 "B4"; F3): :func:`protocol_freeze_check` must find nothing -- the freeze
    record exists and is git-anchored (HEAD = the last commit of the record and of the v2 pin; clean tree over the
    code-manifest scope, the specification files, the pin and the record), ``freeze.is_frozen: true``,
    ``splits.test_previously_seen: true``, specification digest = v2 pin, code-manifest subset digest, data hashes and
    protocol sha256 = the record, access scope present, declared SD points / cells / roots / jobs = the configuration
    -- and only then mints the :class:`OfficialStreamToken` for ``roots_k``.  Raises :class:`FreezeGateError` (no
    token, nothing drawn) otherwise.  No root value is read into the result."""
    reasons, facts = protocol_freeze_check(repo, roots_k=roots_k, config=config)
    if reasons:
        raise FreezeGateError(reasons, facts)
    token = ST.mint_official_stream_token(head_commit=str(facts["head"]),
                                          freeze_record_sha256=str(facts["freeze_record_sha256"]),
                                          protocol_sha256=str(facts["protocol_sha256"]),
                                          spec_digest=str(facts["spec_digest"]), roots_k=tuple(int(k) for k in roots_k),
                                          issued_utc=utc_now())
    return {"ok": True, "status": "protocol_frozen_anchored", "facts": facts, "token": token,
            "protocol_sha256": facts["protocol_sha256"], "freeze_record_sha256": facts["freeze_record_sha256"],
            "head": facts["head"], "spec_digest": facts["spec_digest"], "roots_k": [int(k) for k in roots_k]}


def with_run_context(cfg: Mapping[str, Any], run_context: Mapping[str, Any]) -> dict:
    """A deep copy of ``cfg`` whose ``run_context`` carries ``protocol_sha256`` and ``primary_assumption_id`` (plan F5:
    ``io/config.py`` requires both in mode ``official``); ``fit_sets`` is kept (never ``test``)."""
    out = copy.deepcopy(dict(cfg))
    rc = dict(out.get("run_context") or {})
    rc["protocol_sha256"] = run_context["protocol_sha256"]
    rc["primary_assumption_id"] = run_context["primary_assumption_id"]
    if "fit_sets" in run_context:
        rc["fit_sets"] = list(run_context["fit_sets"])
    out["run_context"] = rc
    return out


# ---------------------------------------------------------------------------------------------------------------------
# output route, root aliases, completion files
# ---------------------------------------------------------------------------------------------------------------------
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def official_output_route(run_id: str, *, rehearsal: Optional[bool] = None) -> dict[str, Any]:
    """Where an official / rehearsal run writes.  Official: public tables ``results/official/<run_id>/``, restricted
    mirror ``data/restricted_local/official/<run_id>/`` (label ``official``).  Rehearsal (development root, tiny sizes):
    ``data/restricted_local/debug/<run_id>/{public,restricted}/`` (label ``debug``; never ``results/``)."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ValueError("run id must be a plain name (letters, digits, '.', '_', '-')")
    if rehearsal is None:
        rehearsal = run_id.startswith(REHEARSAL_RUN_PREFIX)
    if rehearsal:
        if not run_id.startswith(REHEARSAL_RUN_PREFIX):
            raise ValueError(f"a rehearsal run id starts with {REHEARSAL_RUN_PREFIX!r}")
        base = DEBUG_ROOT / run_id
        return {"label": "debug", "mode": "rehearsal", "run_id": run_id, "public_dir": str(base / "public"),
                "restricted_dir": str(base / "restricted"),
                "why": "official rehearsal: development root, tiny sizes; a pipeline check, not a result"}
    if not run_id.startswith(OFFICIAL_RUN_PREFIX):
        raise ValueError(f"an official run id starts with {OFFICIAL_RUN_PREFIX!r}")
    return {"label": "official", "mode": "official", "run_id": run_id,
            "public_dir": str(OFFICIAL_PUBLIC_ROOT / run_id), "restricted_dir": str(OFFICIAL_RESTRICTED_ROOT / run_id),
            "why": "official run v2 on reserved roots after the freeze gate"}


def root_alias(k: int) -> str:
    """Public name of reserved root k in stream ids (``root=formal-k0/test``); the value is derivable from the label
    rule of the streams policy but is never printed in a public file (plan §7)."""
    return f"formal-k{int(k)}"


def alias_reserved_roots(obj: Any, roots_by_k: Mapping[int, int]) -> Any:
    """Replace every reserved root in stream ids (``root=<seed>/...``) and every integer equal to one by its alias;
    recursive over dicts (keys and values), lists and tuples.  Used for every public official file."""
    if not roots_by_k:
        return obj
    subs = [(f"root={int(seed)}", f"root={root_alias(k)}") for k, seed in roots_by_k.items()]
    seeds = {int(seed): root_alias(k) for k, seed in roots_by_k.items()}

    def fix_str(x: str) -> str:
        for a, b in subs:
            if a in x:
                x = re.sub(re.escape(a) + r"(?![0-9])", b, x)
        return seeds.get(int(x), x) if x.isdigit() and int(x) in seeds else x

    def rec(o: Any) -> Any:
        if isinstance(o, str):
            return fix_str(o)
        if isinstance(o, bool) or o is None:
            return o
        if isinstance(o, (int, np.integer)):
            return seeds.get(int(o), o)
        if isinstance(o, Mapping):
            return {rec(k): rec(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rec(v) for v in o]
        return o
    return rec(obj)


def _iter_files(d: Path, skip_dirs: Sequence[str] = ()) -> list[Path]:
    out = []
    for dp, dn, fns in os.walk(d):
        rel_dp = Path(dp).relative_to(d)
        dn[:] = sorted(x for x in dn if (rel_dp / x).as_posix() not in skip_dirs)
        for fn in sorted(fns):
            p = Path(dp) / fn
            if rel_dp == Path(".") and fn in (SHA_MANIFEST, DONE_FILE, MERGE_DONE):
                continue
            out.append(p)
    return sorted(out)


def write_sha_manifest(d: Path, *, skip_dirs: Sequence[str] = ()) -> Path:
    """``SHA256SUMS`` of every file under ``d`` (``<sha256>  <relative path>``; the manifest and the completion files
    at the top level excluded; ``skip_dirs`` = sub-directories with their own manifest)."""
    d = Path(d)
    lines = [f"{file_sha256(p)}  {p.relative_to(d).as_posix()}" for p in _iter_files(d, skip_dirs)]
    out = d / SHA_MANIFEST
    with open(out, "x", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return out


def verify_sha_manifest(d: Path, *, skip_dirs: Sequence[str] = ()) -> list[str]:
    """Problems of ``d`` against its ``SHA256SUMS`` (missing manifest, listed file missing or changed, unlisted file)."""
    d = Path(d)
    m = d / SHA_MANIFEST
    if not m.is_file():
        return [f"{d.name}: {SHA_MANIFEST} missing"]
    listed: dict[str, str] = {}
    for ln in m.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            sha, rel = ln.split("  ", 1)
            listed[rel] = sha
    probs = []
    for rel, sha in sorted(listed.items()):
        p = d / rel
        if not p.is_file():
            probs.append(f"{d.name}: listed file missing: {rel}")
        elif file_sha256(p) != sha:
            probs.append(f"{d.name}: file changed: {rel}")
    extra = [x for x in sorted(p.relative_to(d).as_posix() for p in _iter_files(d, skip_dirs)) if x not in listed]
    if extra:
        probs.append(f"{d.name}: unlisted files: {extra[:10]}")
    return probs


def write_done(d: Path, job_id: str, *, exit_status: int = 0, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """``DONE`` (JSON: job id, exit status, completion time, sha256 of ``SHA256SUMS``); written last."""
    d = Path(d)
    doc = {"job_id": job_id, "exit_status": int(exit_status), "completed_utc": utc_now(),
           "sha256sums_sha256": sha_or_none(d / SHA_MANIFEST), **dict(extra or {})}
    out = d / DONE_FILE
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return out


def read_done(d: Path) -> Optional[dict[str, Any]]:
    p = Path(d) / DONE_FILE
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


# ---------------------------------------------------------------------------------------------------------------------
# job orchestration (one process per job) and the merge
# ---------------------------------------------------------------------------------------------------------------------
def job_argv(run_id: str, job_id: str, *, python: Optional[str] = None, repo: Path = REPO) -> list[str]:
    """The command of one job process: ``<python> <driver> --official-job <run_id> <job_id>``."""
    return [python or sys.executable, str(Path(repo) / DRIVER_PATH), "--official-job", run_id, job_id]


def run_jobs(jobs: Sequence[Mapping[str, Any]], workers: int, *, cwd: Path = REPO,
             env: Optional[Mapping[str, str]] = None, log_dir: Optional[Path] = None,
             runner: Optional[Callable[[Mapping[str, Any], dict], int]] = None) -> list[dict[str, Any]]:
    """Run every job as its own process, at most ``workers`` at a time, with ``THREAD_ENV`` (one BLAS / OpenMP thread;
    plan §5).  ``jobs``: ``{"job_id", "argv"}``; each job is deterministic and rebuilds its own streams (nothing is
    shared between processes).  Stdout and stderr of a job go to ``<log_dir>/<job_id>.log``.  ``runner(job, env) ->
    exit status`` replaces the subprocess (tests).  Returns one record per job, in the given order."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    ids = [str(j["job_id"]) for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("run_jobs: a job id appears twice")
    base_env = dict(os.environ)
    base_env.update(THREAD_ENV)
    base_env.update(dict(env or {}))
    if runner is None and log_dir is None:
        raise ValueError("run_jobs: log_dir is required for subprocess jobs")

    def one(job: Mapping[str, Any]) -> dict[str, Any]:
        started, t0 = utc_now(), time.perf_counter()
        log = None
        if runner is not None:
            rc = int(runner(job, base_env))
        else:
            log = Path(log_dir) / f"{job['job_id']}.log"
            with open(log, "x", encoding="utf-8") as fh:
                rc = subprocess.run([str(a) for a in job["argv"]], cwd=str(cwd), env=base_env, stdout=fh,
                                    stderr=subprocess.STDOUT, check=False).returncode
        return {"job_id": str(job["job_id"]), "exit_status": int(rc), "started_utc": started, "ended_utc": utc_now(),
                "wall_s": time.perf_counter() - t0, "log": None if log is None else log.name}
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=min(int(workers), len(jobs))) as ex:
        return list(ex.map(one, jobs))


def merge_jobs(run_id: str, *, repo: Path = REPO, rehearsal: Optional[bool] = None,
               expected_jobs: Optional[Sequence[Mapping[str, Any]]] = None,
               build_tables: Optional[Callable[[int, list[dict[str, Any]], Path, Path], dict[str, Any]]] = None
               ) -> dict[str, Any]:
    """Check and merge the job outputs of a run, **per root** (plan §7: roots are never pooled into one n).

    Refuses (:class:`MergeError`, nothing written) when: the run plan is missing; a job is listed twice; an expected
    job has no ``DONE`` in its public or restricted directory, or a nonzero exit status; a job directory is not in the
    plan (unexpected / duplicate); a ``DONE`` names another job; a ``SHA256SUMS`` does not verify; the run was already
    merged.  Then groups the jobs by root index and calls ``build_tables(root_k, items, public_root_dir,
    restricted_root_dir)`` once per root with that root's jobs only (``items``: ``{"job", "record", "payload"}``)."""
    repo = Path(repo)
    route = official_output_route(run_id, rehearsal=rehearsal)
    pub, res = repo / route["public_dir"], repo / route["restricted_dir"]
    reasons: list[str] = []
    if expected_jobs is None:
        plan_p = pub / RUN_PLAN_FILE
        if not plan_p.is_file():
            raise MergeError([f"{RUN_PLAN_FILE} missing: not a run directory"])
        expected_jobs = json.loads(plan_p.read_text(encoding="utf-8"))["jobs"]
    exp = [dict(j) for j in expected_jobs]
    ids = [str(j["job_id"]) for j in exp]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    if dup:
        reasons.append(f"jobs listed twice: {dup}")
    if (pub / MERGE_DONE).exists():
        reasons.append("the run is already merged")
    present = sorted(set(_dir_names(pub / "jobs")) | set(_dir_names(res / "jobs")))
    unexpected = sorted(set(present) - set(ids))
    if unexpected:
        reasons.append(f"job directories not in the run plan: {unexpected[:10]}")
    for base in (pub, res):
        for name in _dir_names(base / "jobs"):
            dn = read_done(base / "jobs" / name)
            if dn is not None and str(dn.get("job_id")) != name:
                reasons.append(f"{base.name}/jobs/{name} holds the DONE of job {dn.get('job_id')} (duplicate or "
                               "misplaced job output)")
    missing, failed = [], []
    for jid in ids:
        for base in (pub, res):
            dn = read_done(base / "jobs" / jid)
            if dn is None:
                missing.append(f"{base.name}/{jid}")
                continue
            if int(dn.get("exit_status", 1)) != 0:
                failed.append(jid)
            if dn.get("sha256sums_sha256") != sha_or_none(base / "jobs" / jid / SHA_MANIFEST):
                reasons.append(f"{base.name}/{jid}: DONE does not match its {SHA_MANIFEST}")
            reasons += [f"{base.name}/{p}" for p in verify_sha_manifest(base / "jobs" / jid)]
    if missing:
        reasons.append(f"jobs without DONE: {sorted(set(missing))[:12]}")
    if failed:
        reasons.append(f"jobs with a nonzero exit status: {sorted(set(failed))}")
    if reasons:
        raise MergeError(reasons)
    items_by_root: dict[int, list[dict[str, Any]]] = {}
    for j in exp:
        jid = str(j["job_id"])
        rec_p, pay_p = pub / "jobs" / jid / "job_record.json", res / "jobs" / jid / "job_payload.json"
        rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.is_file() else {}
        pay = json.loads(pay_p.read_text(encoding="utf-8")) if pay_p.is_file() else {}
        items_by_root.setdefault(int(j["root_k"]), []).append({"job": j, "record": rec, "payload": pay})
    per_root: dict[str, Any] = {}
    for k in sorted(items_by_root):
        items = items_by_root[k]
        summary = {"root_k": k, "n_jobs": len(items), "jobs": [it["job"]["job_id"] for it in items]}
        if build_tables is not None:
            pr, rr = pub / f"root{k}", res / f"root{k}"
            pr.mkdir(parents=True, exist_ok=False)
            rr.mkdir(parents=True, exist_ok=False)
            summary.update(build_tables(k, items, pr, rr) or {})
        per_root[str(k)] = summary
    return {"run_id": run_id, "route": route, "roots_k": sorted(items_by_root), "per_root": per_root,
            "n_jobs": len(exp), "pooled_across_roots": False,
            "rule": "tables per root; the roots are never pooled into one n (plan §7)"}
