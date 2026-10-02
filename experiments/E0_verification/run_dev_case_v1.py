#!/usr/bin/env python3
"""dev_case_v1 development run (review round 2, R4 / R8) -- run_type ``pilot`` (development), NOT OFFICIAL.

K7_smoke_devcase, 2026-09-25.  One reference cow, one inventory, one dated price scenario
(``configs/dev_case_v1/``, built by ``scripts/build_dev_case.py`` -- logic ``src/ration_reliability/build/dev_case.py``,
restricted constants in ``data/restricted_local/dev_case_v1/build_dev_case_v1.py`` (R3F); byte-identical to the K4 build);
information state H0 (composition table only) is the main scenario.  Nothing written here is a formal
result: it must not enter any formal result table, the manuscript, an abstract or a claim ledger, and
it says nothing about milk yield, health, intake or any other animal outcome.  "Reliability" means
satisfaction of the declared model constraints under the declared distribution only.

Pipeline (every number from the one public evaluator, one evaluation world)
--------------------------------------------------------------------------
0. Preflight (R3F): ``ration_reliability.build.preflight.require_dev_case_inputs`` -- a missing or stale input stops the
   driver with the list of files and their rebuild conditions (exit 2 missing / 1 stale); nothing is read or written.
1. Inputs: the K4 build report must pass (133 checks) and every input hash recorded there must equal
   the current file (configs, restricted tables); the problem YAML hash must equal the built one;
   the energy linearisation is rebuilt from its record and its fingerprint must match.
2. World (review R3): one ``UncertaintySpec`` from ``uncertainty_cells.csv`` -> uncertainty factory
   (declared family rule TN_MM, fallback BETA_MM; independent marginals C0; variance basis
   ``observed_incl_sampling_and_lab`` = NASEM Table 19-1 SDs of commercial lab results, DC-01), wrapped
   by ``nutrition.energy.EnergyColumnModel`` (NEL_fixedDMI column).  opt / validation / test streams
   of one root seed; the wrapped nominal state must equal the problem's nominal values.
3. Methods (H0): M0 nominal; M1 relative safety margin (grid on validation); M2 joint chance SAA
   (N = 128 and 512, alpha_train calibrated on validation); M3a box robust and M3b budget robust
   (box = factory moments +/- k SD, NEL interval = exact image of the composition box).  Target
   alpha 0.05 (primary), 0.10 and 0.01 (curve points).  Every ration q (kg as-fed/head/d) is scored
   on the same independent test stream (10,000 draws); a method without a ration keeps its row
   with empty cost/risk (never 0).
4. Diagnostics (not methods): minimum attainable SAA joint violation count (exact MILP, bounds kept);
   the cheapest ration at that count (M2 with alpha_train = m*/N, a frontier point, not a target);
   row-conflict map (single rows, pairs, leave-one-out); NEL headroom; NEL linearisation residual.
5. H1 scenario (declared in RUN_CONFIG before the run; research_scenario_assumption; extension
   scenario of the same specification; never replaces H0; not a result): two-layer farm world with
   the corn-silage within-farm SD ratios of ``configs/uncertainty.yaml`` (14 d, true_only), H1-perfect
   decision maker, a few test farms.
6. Information value (one component, declared): partial perfect information on corn-silage starch,
   binned (development prior, 4 quantile bins), finite library, ex-ante joint risk; operational
   (deterministic) and randomised-reference definitions (``docs/value_definition.md``), plus the
   minimum attainable ex-ante joint risk within the library as a diagnostic.
7. Outputs: public tables in ``results/pilot/<run_id>/`` (no NASEM table value, no quantity that gives
   one in a single arithmetic step: planned DM shares, per-ingredient costs, the planned margins of
   PN-CP-HI and of the three inclusion caps, the CP/NDF/starch/fNDF concentrations stay restricted);
   full tables in ``data/restricted_local/pilot/<run_id>/``; run record (``run_type = pilot``: the
   environment must match the lock, the code manifest must be complete, ``mip_rel_gap`` explicit).

FIX_B (review round 2, red team lens B, 2026-09-25) -- additions, all declared in ``RUN_CONFIG`` before any
validation/test number of the new blocks was computed:

8. Stream namespaces: H1 farm streams use sub-key ``H1_FARM_SUBKEY_OFFSET + f`` (the K7 run used ``f``, so
   farms 1 and 2 shared ``opt/1``/``opt/2`` with the information-value prior and design streams); every draw
   is registered and a stream id used by two blocks outside a declared common-random-number group stops the
   run before anything is written (``stream_registry.json``).
9. Composition closure (counted, never clipped): per ingredient the share of test draws with
   CP+NDF+starch+EE+ash > 100 % DM and with Eq 3-1 ROM < 0, per world; and M0's joint violation inside/outside
   those draws.  The NEL linearisation residual is also grouped by diet starch <= S_ref and |D - DMI| < 0.05 kg
   (the linear row is not a draw-wise bound of the chain).
10. C1: the declared corn-silage correlation scenario (Yoder 2014 Table 6 via configs/uncertainty.yaml, calibrated
    to the latent copula), same streams as H0; methods, minimum violations, H0 rations scored there.
11. S2 (red team B #1): a second development scenario in which a target alpha is attainable -- reference farm
    (table means, within-farm SD = registered true_only ratio x table SD, borrowed 1/2.1 where no ratio exists),
    joint event = the probabilistic rows with sourced bounds, the other five rows imposed as planned rows at table
    values (structural) and reported as scenario diagnostics.  A method-calibration scenario on a narrowed
    specification: not comparable with H0, not a result.  The pre-declaration screening (opt stream only) is in
    ``RUN_CONFIG["S2"]["screening_opt_only"]``.
12. PN-CP-SUP with RDP rescaled to the case DMI (E5-CP-SUP-RDP-SCALED dev diagnostic; restricted inputs in
    ``data/restricted_local/dev_case_v1/fixb_cp_sup_rdp_scaling.yaml``).
13. M3 boxes via ``nutrition.energy.energy_box_from_factory_model`` (library version of the K7 helper; the engine's
    opt-draw box no longer clips the energy column, B-K7-1).

Usage::

    cd <project root>
    PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 experiments/E0_verification/run_dev_case_v1.py
    ... --dry-run     # reduced sizes, in memory, prints statuses only, writes nothing

``<project root>`` is the repository root.  The script does not depend on it at run time: it resolves the root from
``__file__`` (``REPO``) and changes into it before reading any relative path (B-431: no machine-specific path in code).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, linprog, milp  # noqa: E402

import smoke_pipeline as SP  # noqa: E402
from ration_reliability.datamodel import ConstraintClass, SolverOptions  # noqa: E402
from ration_reliability.evaluation import evaluate, evaluate_drawset  # noqa: E402
from ration_reliability.evaluation.stats import clopper_pearson, mc_standard_error, one_sided_upper  # noqa: E402
from ration_reliability.build.dev_case import dev_case_build_identity  # noqa: E402  (R3F)
from ration_reliability.build.preflight import require_dev_case_inputs  # noqa: E402  (R3F)
from ration_reliability.hashing import file_sha256, stable_hash  # noqa: E402
from ration_reliability.information import (  # noqa: E402
    InformationStructure,
    ObservedComponent,
    PriorStates,
    SignalBinning,
    ValueDefinition,
    compute_information_value,
    compute_risk_table,
    conditioning_from_likelihood,
    generate_conditional_library,
    perfect_partial_likelihood,
    quantile_edges,
)
from ration_reliability.io import (  # noqa: E402
    ENVIRONMENT_LOCK_FILE,
    build_problem,
    build_run_record,
    check_environment_against_lock,
    code_manifest,
    environment_fingerprint,
    load_problem,
    make_run_id,
    utc_now,
    write_run_record,
)
from ration_reliability.nutrition import energy as E  # noqa: E402
from ration_reliability.nutrition.constraints import linear_rows  # noqa: E402
from ration_reliability.optimization import get_method  # noqa: E402
from ration_reliability.optimization.chance_saa import allowed_violations, build_saa_model  # noqa: E402
from ration_reliability.optimization.highs import solver_version_string  # noqa: E402
from ration_reliability.optimization.robust import (  # noqa: E402
    BoxUncertaintySet,
    solve_box_robust,
    solve_budget_robust,
)
from ration_reliability.optimization.safety_margin import select_parameter_on_validation  # noqa: E402
from ration_reliability.uncertainty import DrawSet, RandomStreams, UncertaintySpec, build_uncertainty_model  # noqa: E402

# ------------------------------------------------------------------------------------------------
# paths
# ------------------------------------------------------------------------------------------------
CASE_CFG = Path("configs") / "dev_case_v1"
CASE_RL = Path("data") / "restricted_local" / "dev_case_v1"
PROBLEM_YAML = CASE_RL / "dev_case_v1_problem.yaml"
CELLS_CSV = CASE_RL / "uncertainty_cells.csv"
ENERGY_JSON = CASE_RL / "energy_linearisation.json"
BUILD_REPORT = CASE_RL / "build_report.json"
BUILD_SCRIPT = CASE_RL / "build_dev_case_v1.py"
CORE_CSV = Path("data") / "restricted_local" / "nasem_t19_1_core.csv"
CONFIG_PATHS = [CASE_CFG / n for n in ("animal.yaml", "inventory.yaml", "prices.yaml", "constraints.yaml",
                                       "energy.yaml", "README.md")] + \
               [Path("configs") / n for n in ("animal_profile.yaml", "methods.yaml", "uncertainty.yaml")]
PROTOCOL = Path("configs") / "protocol.yaml"
MEASUREMENT_MODEL_NASEM = ("NASEM2021_T19_1:commercial_lab_results(sampling+analysis error included, not separated; "
                           "DC-01)")
PN = ("CP", "NDF", "starch", "EE", "ash", "Ca", "P")        # base-model columns (uncertainty_cells.csv order)
CS = "corn_silage_typical"

# ------------------------------------------------------------------------------------------------
# declared run configuration (fixed before the run; copied into the run record and run_config.json)
# ------------------------------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    "schema": "ration_reliability.dev_run_config/0.1",
    "case_id": "dev_case_v1",
    "run_type": "pilot",
    "run_role": ("development run of one development case (contract T6 'small pilot, separate directory, explicit "
                 "label, never a formal result'); not a protocol pilot matrix, not official, protocol not frozen"),
    "seed": 1103,
    "seed_note": "first development seed of configs/methods.yaml rng.development_seeds; one seed only (no seed "
                 "convergence study in this run)",
    "streams": {"opt": 512, "N_ladder": [128, 512], "N_nesting": "N=128 uses the first 128 opt draws",
                "validation": 20000, "test": 10000,
                "sizes_source": "configs/methods.yaml draws.validation_draws_initial / test_draws_initial; "
                                "N ladder from methods.yaml M2.N_ladder (1024 not run on this Mac)"},
    "information_state_main": "H0_table_only",
    "uncertainty": {
        "family_rule": {"primary_family": "TN_MM", "fallback_families": ["BETA_MM"], "on_exhausted": "error",
                        "status": "research_scenario_assumption", "selection_basis": "declared_rule",
                        "rationale": ("development run: TN_MM reproduces the table mean/SD inside [0, 1]; BETA_MM "
                                      "only where TN_MM cannot match (CV relative to the bound >= 1); a cell no "
                                      "declared family matches stops the run.  Same declared rule as the K3 smoke "
                                      "and the K4c factory pre-check; not the primary-family decision (D-19 "
                                      "pending)")},
        "purpose": "main_analysis",
        "purpose_note": "H0 is the main information state of this development case; not a frozen main analysis",
        "correlation": "C0_independent marginals (baseline, not a fact; D-19 pending)",
        "variance_basis": "observed_incl_sampling_and_lab (NASEM Table 19-1 SDs of commercial lab results, DC-01)",
        "measurement_model_id": MEASUREMENT_MODEL_NASEM,
        "energy_column": "nutrition.energy.EnergyColumnModel(base, lin): NEL_fixedDMI per draw from the drawn CP/NDF/"
                         "starch/ash (fixed-DMI linearisation, configs/dev_case_v1/energy.yaml)"},
    "alphas": {"primary": 0.05, "curve": [0.10, 0.01], "source": "configs/methods.yaml risk_levels (research "
                                                                "design choice, not an animal safety threshold)"},
    "screening_rule": {"rule": "rate_upper_le_alpha", "unknown_draws": "counted as violations",
                       "source": "configs/methods.yaml M1.selection_rule (r_val point estimate; one-sided CP upper "
                                 "bound recorded)", "status": "research_scenario_assumption"},
    "methods": {
        "M0": {"method_id": "M0_nominal", "params": {"coefficient_mode": "nominal_point"}},
        "M1": {"method_id": "M1_safety_margin", "margin_scale": "relative",
               "margin_scale_basis": ("configs/methods.yaml M1: margin_semantics.chosen and margin_scale are "
                                      "pending (PUD-P4b-01); the declared development start is option "
                                      "coef_directional (dev_start_proposal: true) with grid_margin_scale "
                                      "'relative' -> used here as research_scenario_assumption"),
               "grid": [0.0, 0.025, 0.05, 0.075, 0.10], "grid_source": "configs/methods.yaml M1.grid (contract T4)",
               "apply_to_dm": True,
               "apply_to_dm_basis": ("pending in methods.yaml (PUD-P4b-01, B-129); True for this development run "
                                     "because the realised DM of the forage is a declared random channel of every "
                                     "supply row -- a margin that ignores it would make M1 artificially weak "
                                     "(contract §9); research_scenario_assumption"),
               "sensitivity_variant": {"apply_to_dm": False, "role": "declared sensitivity line, not selected"}},
        "M2": {"method_id": "M2_joint_chance_saa", "N": [128, 512], "alpha_train_multipliers": [1.0, 0.75, 0.5, 0.25],
               "multipliers_source": "configs/methods.yaml M2.alpha_train.grid_multipliers_of_alpha",
               "big_m_mode": "box_quantile", "polish": True},
        "M3a": {"method_id": "M3a_box_robust", "k_grid": [1.0, 1.5, 2.0, 2.5, 3.0],
                "k_grid_source": "configs/methods.yaml M3.k_grid",
                "box": ("BoxUncertaintySet.from_factory_model (achieved moments +/- k SD, clipped to the cell "
                        "ranges) for composition and DM; NEL_fixedDMI interval = exact image of that composition "
                        "box under the affine energy map (intercept + sum_j min/max(slope_ij lo_ij, slope_ij "
                        "hi_ij)); opt-draw boxes are not used because their default clip at 0 would cut the "
                        "negative NEL contribution of the minerals"),
                "kind_status": "D-11 pending; box_mu_k_sigma is the D-11 proxy suggestion -> research_scenario_assumption"},
        "M3b": {"method_id": "M3b_budget_robust", "k_grid": [1.0, 1.5, 2.0, 2.5, 3.0], "gamma_grid": [0, 1, 2, 3, 4, 5, 6],
                "gamma_grid_source": "configs/methods.yaml M3.gamma_grid: 0..number of random coefficients per row "
                                     "(6 stochastic ingredients)",
                "nominal_mode": "nominal_point",
                "selection": "per k: lowest-cost Gamma meeting the screen (validation); across k: lowest cost, ties "
                             "-> lower rate_upper, then smaller k"},
    },
    "solver": {"time_limit_s": 300.0, "mip_rel_gap": 1e-4, "note": "HiGHS via scipy; mip_rel_gap explicit (C11)"},
    "diagnostics": {
        "min_violation_saa": {"N": [128, 512], "time_limit_s": {"128": 60.0, "512": 180.0}, "mip_rel_gap": 0.0,
                              "big_m": "box (valid for every x in the structural polytope)",
                              "frontier_point": "M2 with alpha_train = UB/N (cheapest ration at the attainable count); "
                                                "a diagnostic frontier point, NOT a target alpha"},
        "conflict_map": {"N": 128, "single_rows_time_limit_s": 30.0, "pairs_time_limit_s": 30.0,
                         "leave_one_out_time_limit_s": 60.0,
                         "rule": "a row set conflicts at alpha iff the proven lower bound on its minimum SAA "
                                 "violation count exceeds floor(alpha N)"},
        "nel_headroom": "max linear NEL supply at nominal subject to every other imposed row (energy.yaml DIAG-NEL-HEADROOM)",
        "nel_linearisation_residual": "energy.nonlinear_diet_nel per test draw (energy.yaml DIAG-NEL-LINEARISATION-RESIDUAL)"},
    "H1": {
        "status": "research_scenario_assumption", "role": "extension scenario only; never replaces H0; not a result",
        "definition": "configs/uncertainty.yaml e_information_state_history.H1_farm_history (two-layer, H1-perfect: "
                      "the decision maker knows mu_farm of the covered cells)",
        "ratios": [[CS, "DM", "1/4.4"], [CS, "NDF", "1/3.6"], [CS, "starch", "1/5.0"]],
        "ratio_row": "ratio_table entry corn_silage, horizon 14 d, component true_only (project inference from "
                     "published tables, 【推断】; DC-03: only true_only ratios may generate the truth)",
        "uncovered_cells": "P0 (every other cell keeps its H0 table mean/SD; no farm layer)",
        "s": 1.0, "family": "TN_MM", "allow_unidentified_scenario": True,
        "within_cell_variance_basis": "unidentified (ratio x observed SD; the between layer absorbs measurement "
                                      "error: between_layer_contains_measurement_error)",
        "farms": {"purpose": "test", "farm_indices": [0, 1, 2, 3, 4, 5], "stream": "farm_mean"},
        "alpha": 0.05, "N": 128,
        "per_farm_streams": ("opt/validation/test with sub-key = H1_FARM_SUBKEY_OFFSET + farm index (FIX_B: the K7 run "
                             "used sub-key = farm index, so farms 1 and 2 drew root=1103/opt/1 and opt/2 -- the same keys "
                             "as the information-value prior and design streams; red team B #4)"),
        "min_violation_time_limit_s": 30.0,
        "energy_linearisation": "the P0 linearisation is kept (farm deviations enter through the affine slopes)"},
    "information_value": {
        "structure": "perfect_partial", "component": [CS, "starch"],
        "why_this_component": ("energy channel (NEL slope) and Table 5-1 starch rows (PN-T4, PN-T5); the decision-time "
                               "action set stays the t0 set, so the value definitions are applicable (observing DM "
                               "would change the action set: cost-risk comparison only, PUD-P8-03); the double-count "
                               "guard is 'not_applicable' for perfect information (no measurement-error model)"),
        "scenario_label": ("research_scenario_assumption: ideal knowledge of a state whose prior SD is the observed "
                           "commercial-lab SD (observed_incl_sampling_and_lab); not an assay value"),
        "prior": {"stream": "opt", "sub_key": 1, "n_states": 2000},
        "binning": {"n_bins": 4, "edges": "weighted quantiles of the development prior states (restricted values)"},
        "library": {"generator": "information.generate_conditional_library (M0 per bin, tightening grid 0..3)",
                    "tightening_grid": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                    "extra_candidates": ("H0 M0 ration and the H0 frontier rations (min-violation diagnostics); per bin "
                                         "one bin-conditional frontier ration: the design sub-stream opt/2 (512 draws) "
                                         "is split by the same bin edges, the minimum violation count m_z of the "
                                         "bin's scenarios is found (MILP) and M2 is solved on them with alpha_train "
                                         "= m_z / N_z (cheapest ration at that count); development streams only"),
                    "design_stream": {"stream": "opt", "sub_key": 2, "n_draws": 512},
                    "per_bin_min_violation_time_limit_s": 60.0},
        "alphas": [0.10, 0.05, 0.01],
        "value_definition_primary": "operational_deterministic_cost_difference",
        "also_reported": ["randomized_same_class_information_reference (theoretical, never a feeding recommendation)",
                          "contrast_vs_matched_uninformative_bins (diagnostic only)",
                          "minimum attainable ex-ante joint risk within the library (diagnostic)"],
        "not_claimed": "not a continuous global EVPI/EVPPI/EVSI nor a bound of it (docs/value_definition.md §5)"},
    # ---------------------------------------------------------------------------------------------------------
    # FIX_B additions (review round 2, red team B; declared before any validation/test number of the new blocks
    # was computed -- the only pre-declaration screening used the opt stream, recorded in S2.screening_opt_only)
    # ---------------------------------------------------------------------------------------------------------
    "fixb": {
        "task": "FIX_B (review round 2, red team lens B), 2026-09-25",
        "supersedes_run": "pilot-20260924T192615Z-1fc52728 (K7): H1 stream collision, no composition-closure "
                          "diagnostic, energy conservativeness claim, no attainable development scenario",
        "unchanged_for_H0": "problem numerics, world, streams, alpha, grids, tolerances, selection rule, family rule "
                            "(price statuses of two minerals and constraint/energy rationale texts changed; "
                            "their numbers did not)"},
    "stream_namespaces": {
        "H1_farm_subkey_offset": 1000,
        "rule": ("every draw is registered (block, stream id, world); a stream id used by two blocks is a collision "
                 "unless the blocks share a declared common-random-number group (paired worlds); a collision stops "
                 "the run before any file is written"),
        "crn_groups": {"CRN-main": "H0, S2 and C1 worlds on root/opt, root/validation, root/test (paired scenarios)",
                       "CRN-iv-prior": "H0 and S2 information-value prior (root/opt/1)",
                       "CRN-iv-design": "H0 and S2 information-value design sub-stream (root/opt/2)"}},
    "diagnostics_fixb": {
        "composition_closure": ("energy.composition_closure_report on the test stream of every world: per ingredient "
                                "share of draws with CP+NDF+starch+EE+ash > 100 % DM and with Eq 3-1 ROM < 0 (FA at "
                                "the feed value); counted only, never clipped or dropped (red team B #2)"),
        "nel_residual_grouping": ("linearisation residual additionally grouped by diet starch <= S_ref and by "
                                  "|D - DMI| < 0.05 kg (red team B #3)"),
        "cp_sup_rdp_scaled": {"id": "E5-CP-SUP-RDP-SCALED (dev-run diagnostic)",
                              "definition": "PN-CP-SUP bound replaced by RUP_table + RDP_table / DMI_table x DMI_case "
                                            "(inputs in data/restricted_local/dev_case_v1/fixb_cp_sup_rdp_scaling.yaml)",
                              "reported": "M0 cost and test joint rate, N = 128 minimum violation count (H0 and S2); "
                                          "public outputs give relative changes only",
                              "status": "research_scenario_assumption (red team B #5)"}},
    "C1": {
        "status": "research_scenario_assumption", "role": "declared correlation scenario; reported next to H0; not a result",
        "structure_id": "C1_yoder_table6_corn_silage",
        "pairs": {"starch_NDF": -0.88, "CP_NDF": 0.20, "ash_NDF": 0.37},
        "source": "configs/uncertainty.yaml c_correlation_structure C1_yoder_table6 values_used.corn_silage "
                  "(Yoder et al. 2014 Table 6; SRC-C-YODER2014); NDF_lignin not used (lignin is not random here)",
        "scope": "corn silage only: the other C1 entries are other ingredients (legume silage, grass hay, cottonseed) or "
                 "flagged 'not correlation evidence' (canola meal)",
        "input_scale": "transformed_pearson", "handling": "calibrated_mapping",
        "psd_note": "star-shaped matrix [NDF; starch, CP, ash] has lambda_min = 1 - sqrt(0.88^2+0.2^2+0.37^2) = 0.025 "
                    "> 0; no repair (a non-PSD result stops the run)",
        "run": "same streams as H0 (CRN-main); M0-M3 at the three alpha (M1 with DM margin only; M2 at N = 128 only); "
               "N = 128 minimum violations; H0 rations scored in the C1 world; composition closure"},
    "S2": {
        "id": "S2_reference_farm_sourced_event",
        "status": "research_scenario_assumption",
        "role": ("second development scenario asked for by red team B #1 (R8 third lane): a declared scenario in which "
                 "at least one target alpha is attainable, so that alpha_train calibration, M1 validation selection, M3 "
                 "k/Gamma selection and same-risk comparisons are executed non-trivially on the real case data.  "
                 "Method-calibration scenario on a NARROWED specification; never replaces H0; not comparable with H0 "
                 "(different world and different risk event); not a result; no animal outcome"),
        "construction_rule": ("take the red team's listed options in their order (sourced variance decomposition; "
                              "within-farm true_only basis; joint event narrowed to rows with sourced bounds; per-row "
                              "chance constraints), use the smallest combination that makes a target alpha attainable on "
                              "the opt stream (N = 128 exact minimum-violation MILP), and within each option the most "
                              "conservative registered setting; keep every row imposed (review §5: a narrowed spec only "
                              "near a reasonable base diet); no alpha, bound, tolerance, price or candidate is changed"),
        "screening_opt_only": {
            "note": "FIX_B exploration before this declaration; opt stream (root=1103/opt) only; no validation or test "
                    "stream of any scenario was drawn; minimum SAA joint violation count over 128 (or 512) opt draws",
            "H0_all_rows": "84/128 (proven)",
            "DM_known_control (configs/uncertainty.yaml f_dm_channel DM-KNOWN)": "[84, 102]/128 (60 s)",
            "SD_scale_0.667 (sensitivity grid a, lower end)": "78/128",
            "within_farm_ratios_14d (corn silage DM 1/4.4, NDF 1/3.6, starch 1/5.0; others borrow 1/2.1)": "65/128",
            "within_farm_ratios_14d + DM_known": "[28, 46]/128 (60 s)",
            "per_row_chance_constraints alpha_k = 0.05 / 0.09 on all 11 rows (M2b allocation)": "proven_infeasible",
            "sourced_row_event, other 5 rows NOT imposed": "0/128 -- rejected: the M0 of that problem is ~10 Mcal/d "
                                                           "(~22 %) short of the energy requirement even at table "
                                                           "values (not a reasonable base diet, review §5)",
            "sourced_row_event, other 5 rows planned at table values, H0 world": "52/128 (proven)",
            "same, DM_known": "47/128",
            "same, within_farm_ratios_12mo (corn silage DM 1/2.1, NDF 1/2.5; others 1/2.1)": "15/128 and 64/512 -- "
                                                                                         "alpha = 0.10 not attainable",
            "same, within_farm_ratios_14d  <- chosen": "0/128 and 0/512",
            "no_sourced_variance_decomposition": "none registered for these cells (DC-01: NASEM SDs are observed "
                                                 "commercial-lab SDs); option 1 not available"},
        "world": {
            "definition": ("reference farm: the H1-perfect conditional world at mu_farm = mu_P0 (the table means); the "
                           "decision maker knows these means (d_hat = table DM means) and faces within-farm batch "
                           "variation SD = r x sigma_P0 around them; one declared farm, not a population"),
            "ratios": {"corn_silage_typical": {"DM": "1/4.4", "NDF": "1/3.6", "starch": "1/5.0",
                                               "other components": "1/2.1 (borrowed)"},
                       "every other stochastic ingredient, every component and DM": "1/2.1 (borrowed)"},
            "ratio_sources": ("configs/uncertainty.yaml H1_farm_history.ratio_table: corn silage 14 d true_only (the "
                              "batch horizon already declared for H1 of this case, D-K7-8); 1/2.1 = the largest "
                              "(least-reducing) registered true_only ratio (corn silage 12 mo DM), borrowed for cells "
                              "without a ratio (uncovered_ingredients rule: 'borrow and label research_scenario_assumption')"),
            "caveats": ["the replaced definition 'P0 mean + reduced SD' of H1 was retired for the IS-0 vs IS-2 "
                        "information comparison across farms; S2 uses it only as ONE declared reference farm for method "
                        "calibration and makes no information-value comparison with H0",
                        "ratios are project inferences (【推断】); borrowed ratios have no source for those ingredients",
                        "with the 12-month ratios alpha = 0.10 is not attainable (screening): attainability depends on "
                        "the within-farm horizon"],
            "variance_basis": "unidentified (ratio x observed SD; DC-03: only true_only ratios generate the state)",
            "family_rule": "same as H0 (TN_MM, fallback BETA_MM, error)", "correlation": "C0_independent"},
        "risk_event": {
            "members": ["PN-CP-SUP", "PN-T1", "PN-T2", "PN-T3", "PN-T4", "PN-T5"],
            "rule": "exactly the probabilistic rows whose bound status is 'sourced' in configs/dev_case_v1/constraints.yaml "
                    "(checked at run time)"},
        "planned_rows": {
            "members": ["PN-NEL-FIXEDDMI", "PN-CP-HI", "PN-EE-HI", "PN-CA-ABS", "PN-P-ABS"],
            "treatment": ("each is imposed as a structural_hard planning row SH-PLAN-* at the table values and d_hat "
                          "(the row M0 imposes; never relaxed by the risk budget) and its scenario form is kept as a "
                          "diagnostic_only row with the same id (per-row violation reported, not in the joint event); "
                          "bounds, senses, tolerances unchanged"),
            "encoding": ("coefficient per ingredient = the compiled row content c_ki at the nominal state; "
                         "SH-PLAN-NEL-FIXEDDMI is written in 'kg/d of 1-Mcal/kg-equivalent DM' because the engine "
                         "gives DM-term expressions a mass-rate unit; numerically identical to Mcal/d (checked: S2 M0 "
                         "= H0 M0 and every planned margin = the H0 nominal margin)")},
        "methods": "same declarations as H0 (M0; M1 relative with and without DM; M2 N = 128, 512; M3a/M3b factory "
                   "box k grid and Gamma grid; alpha 0.05, 0.10, 0.01; rate_upper_le_alpha on validation)",
        "information_value": "same declaration as H0 (corn silage starch PPI, 4 bins, same library rule) on the S2 world",
        "not_run": ["H1 (S2 is itself a within-farm reference world)", "row-conflict map and largest-feasible "
                    "bisection (every grid has feasible values)"],
        "comparison_rule": ("same-risk comparisons only among S2 methods that meet the same target alpha on validation; "
                            "costs are reported with test joint risk and CP bounds of each method; no H0-S2 cost "
                            "difference is reported as a saving")},
    "public_redaction": {
        "restricted_only": ["planned DM share per ingredient", "per-ingredient cost", "d_hat",
                            "planned margins of PN-CP-HI and of the three inclusion caps SH-INCL-LIMESTONE-MAX, SH-INCL-DCP-MAX, SH-INCL-SOYHULLS-MAX (30 - margin = planned DM share)",
                            "test-stream margin quantiles and deficits of PN-CP-HI",
                            "CP/NDF/starch/forage-NDF concentrations and forage share", "farm means (H1)",
                            "information-value bin edges", "box bounds", "target/achieved moments"],
        "rule": ("NASEM table values and any quantity that returns one in a single arithmetic step stay in "
                 "data/restricted_local (AGENT_HANDOFF forbidden_shortcuts; DECISIONS D-226; K4 D-K4-9)"),
        "known_multi_step_pathway": ("several published q vectors with the public DMI (SH-DM-PLAN) form a linear system "
                                     "in the DM means (same class as BLOCKERS B-147); results/pilot must stay out of "
                                     "public packages until the lead decides")},
}
REDACT_NOMINAL = {"PN-CP-HI", "SH-INCL-LIMESTONE-MAX", "SH-INCL-DCP-MAX", "SH-INCL-SOYHULLS-MAX", "SH-PLAN-CP-HI"}
H1_FARM_SUBKEY_OFFSET = RUN_CONFIG["stream_namespaces"]["H1_farm_subkey_offset"]
S2_EVENT = tuple(RUN_CONFIG["S2"]["risk_event"]["members"])
S2_PLANNED = tuple(RUN_CONFIG["S2"]["planned_rows"]["members"])
#: 1 Mcal per kg DM: SH-PLAN-NEL-FIXEDDMI is written in kg/d of 1-Mcal/kg-equivalent DM (numerically = Mcal/d)
E_REF_MCAL_PER_KG = 1.0
FIXB_CP_SUP_INPUTS = CASE_RL / "fixb_cp_sup_rdp_scaling.yaml"
REDACT_TEST_MARGINS = {"PN-CP-HI"}
PUBLIC_PROFILE_QUANTITIES = {"DM_supply", "NEL_fixedDMI_supply", "EE_conc", "Ca_conc", "P_conc", "ash_conc"}


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------
def _f(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return None if math.isnan(x) else x


def _rel(p: Path) -> str:
    return str(p)


class Out:
    """Collects written files (public and restricted) for the run record; writes nothing in dry-run."""

    def __init__(self, pub: Optional[Path], res: Optional[Path]):
        self.pub, self.res = pub, res
        self.files: list[Path] = []

    def json(self, where: str, name: str, obj) -> Optional[Path]:
        base = self.pub if where == "pub" else self.res
        if base is None:
            return None
        p = base / name
        with open(p, "x", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
            fh.write("\n")
        self.files.append(p)
        return p

    def csv(self, where: str, name: str, rows: list[dict], columns: list[str]) -> Optional[Path]:
        base = self.pub if where == "pub" else self.res
        if base is None:
            return None
        p = SP.write_csv(rows, base / name, columns)
        self.files.append(p)
        return p

    def text(self, where: str, name: str, text: str) -> Optional[Path]:
        base = self.pub if where == "pub" else self.res
        if base is None:
            return None
        p = base / name
        with open(p, "x", encoding="utf-8") as fh:
            fh.write(text)
        self.files.append(p)
        return p


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


class StreamRegistry:
    """Every draw of the run, by analysis block (FIX_B, red team B #4).

    A stream id (``root=<seed>/<name>/<sub keys>``) that serves two blocks is a *collision* -- the blocks would
    share their random numbers -- unless both blocks declare the same common-random-number group (paired
    scenarios on purpose).  :meth:`check` lists collisions; the run stops before writing anything if any exist.
    """

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, block: str, draws=None, *, stream_id: Optional[str] = None, world: Optional[str] = None,
            n: Optional[int] = None, crn_group: Optional[str] = None, purpose: str = "") -> None:
        if draws is not None:
            stream_id, world, n = draws.stream_id, draws.model_fingerprint, int(draws.n_draws)
        if stream_id is None:
            raise ValueError("StreamRegistry.add: stream id missing")
        self.rows.append({"block": block, "stream_id": str(stream_id), "world_fingerprint": world, "n_draws": n,
                          "crn_group": crn_group, "purpose": purpose})

    def collisions(self) -> list[dict]:
        by: dict[str, list[dict]] = {}
        for r in self.rows:
            by.setdefault(r["stream_id"], []).append(r)
        out = []
        for sid, rs in sorted(by.items()):
            blocks = sorted({r["block"] for r in rs})
            if len(blocks) < 2:
                continue
            groups = {r["crn_group"] for r in rs}
            if len(groups) == 1 and None not in groups:
                continue                                   # declared paired use
            out.append({"stream_id": sid, "blocks": blocks, "crn_groups": sorted(str(g) for g in groups)})
        return out

    def check(self) -> dict:
        col = self.collisions()
        return {"n_draw_calls": len(self.rows), "n_distinct_stream_ids": len({r["stream_id"] for r in self.rows}),
                "collisions": col, "ok": not col, "rule": RUN_CONFIG["stream_namespaces"]["rule"],
                "entries": list(self.rows)}


def h1_farm_subkey(f: int) -> int:
    """Sub-key of H1 farm ``f``'s opt/validation/test streams: fixed offset + farm index (FIX_B)."""
    return int(H1_FARM_SUBKEY_OFFSET) + int(f)


# ------------------------------------------------------------------------------------------------
# 1 inputs
# ------------------------------------------------------------------------------------------------
def check_inputs() -> dict:
    """Build report passed and every recorded input hash equals the current file (no rebuild here).

    R3F (review round 3, F-4): the preflight runs first, so a caller (this driver, replay_dev_case_eval.py, a later
    ablation driver) with missing or stale inputs stops with the short list of files and rebuild conditions (exit 2 /
    1) instead of a traceback; nothing is written."""
    require_dev_case_inputs(REPO, driver="check_inputs")
    br = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    issues = []
    if br["summary"]["n_failed"] != 0 or br["summary"]["n_checks"] != 133:
        issues.append(f"build report: {br['summary']}")
    cur = {}
    for p, h in br["inputs"].items():
        now = file_sha256(p) if Path(p).is_file() else None
        cur[p] = now
        if now != h:
            issues.append(f"input changed since the build: {p}")
    ph = file_sha256(PROBLEM_YAML)
    if ph != br["problem_yaml_sha256"]:
        issues.append("problem YAML differs from the built one")
    if issues:
        raise SystemExit("dev_case_v1 inputs are not the built ones; rerun build_dev_case_v1.py first:\n  - "
                         + "\n  - ".join(issues))
    return {"build_report_sha256": file_sha256(BUILD_REPORT), "build_script_sha256": file_sha256(BUILD_SCRIPT),
            "build_summary": br["summary"], "inputs_sha256_verified": cur, "problem_yaml_sha256": ph,
            "uncertainty_cells_sha256": file_sha256(CELLS_CSV), "energy_linearisation_sha256": file_sha256(ENERGY_JSON),
            "k4c_factory_precheck": {k: br["factory_precheck_not_a_run"].get(k)
                                     for k in ("fit_status_counts", "family_counts_matched_cells", "fallback_cells",
                                               "spec_fingerprint", "world_fingerprint")}}


def load_linearisation(problem) -> "E.NELLinearisation":
    rec = json.loads(ENERGY_JSON.read_text(encoding="utf-8"))["linearisation"]
    feeds = [E.FeedEnergyInputs(**f) for f in rec["feeds"]]
    st = {k: v for k, v in rec["settings"].items() if k != "dmi_bw"}
    lin = E.linearise_nel_fixed_dmi(feeds, E.FixedDMISettings(**st), nutrient_map=rec["nutrient_map"])
    if lin.fingerprint() != rec["fingerprint"]:
        raise SystemExit("energy linearisation rebuilt from its record has another fingerprint")
    if tuple(lin.ingredient_ids) != tuple(problem.ingredient_ids):
        raise SystemExit("energy linearisation ingredient order differs from the problem")
    k = list(problem.nutrient_ids).index(E.ENERGY_COLUMN_ID)
    d = float(np.max(np.abs(problem.nominal_theta()[:, k] - lin.nominal_density)))
    if d > 1e-12:
        raise SystemExit(f"problem NEL_fixedDMI nominal values differ from the linearisation ({d})")
    return lin


def read_cells() -> dict:
    rows = list(csv.DictReader(CELLS_CSV.open(encoding="utf-8")))
    return {(r["ingredient_id"], r["component"]): r for r in rows}


def cell_arrays(cells: dict, ids) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tm = np.array([[float(cells[(i, n)]["mean_pct"]) / 100.0 for n in PN] for i in ids])
    ts = np.array([[float(cells[(i, n)]["sd_pct"]) / 100.0 for n in PN] for i in ids])
    dm = np.array([float(cells[(i, "DM")]["mean_pct"]) / 100.0 for i in ids])
    ds = np.array([float(cells[(i, "DM")]["sd_pct"]) / 100.0 for i in ids])
    return tm, ts, dm, ds


def cell_overrides(cells: dict, cells_sha: str) -> dict:
    over = {}
    for (iid, item), c in cells.items():
        fp = stable_hash("dev_case_v1_uncertainty_cell/v1", cells_sha, iid, item, c["mean_pct"], c["sd_pct"], c["n"],
                         c["locator"])
        o = {"data_fingerprint": fp, "provenance_status": c["provenance_status"], "source_id": c["source_id"] or None,
             "locator": c["locator"] or None}
        if c["provenance_status"] == "research_scenario_assumption" and not o["source_id"]:
            o["source_id"] = None
        over[(iid, item)] = o
    return over


def build_h0_spec(ids, cells, cells_sha) -> UncertaintySpec:
    tm, ts, dm, ds = cell_arrays(cells, ids)
    h1 = RUN_CONFIG["H1"]
    two_layer = {"role": "extension_scenario",
                 "ratios": [[CS, "DM", 1 / 4.4], [CS, "NDF", 1 / 3.6], [CS, "starch", 1 / 5.0]],
                 "ratio_basis": "true_only", "ratio_status": "research_scenario_assumption",
                 "ratio_source": "configs/uncertainty.yaml e_information_state_history.H1_farm_history.ratio_table "
                                 "(corn_silage, 14 d, true_only; project inference 【推断】)",
                 "s": float(h1["s"]), "family": h1["family"], "uncovered_cells": "P0",
                 "allow_unidentified_scenario": True,
                 "notes": "K7 dev_case_v1 H1 extension scenario (research_scenario_assumption; never the main world)"}
    return UncertaintySpec.from_arrays(
        "DEV_CASE_V1_H0_TN_MM", ids, PN, tm, ts, dm, ds, purpose=RUN_CONFIG["uncertainty"]["purpose"],
        moment_semantics="target_marginal_moments", is_synthetic=False,
        family_rule=RUN_CONFIG["uncertainty"]["family_rule"],
        cell_defaults={"variance_basis": "observed_incl_sampling_and_lab", "decomposition_id": "none",
                       "decomposition_source": None, "measurement_model_id": MEASUREMENT_MODEL_NASEM},
        theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0), cell_overrides=cell_overrides(cells, cells_sha),
        two_layer=two_layer,
        notes="K7 dev_case_v1 development run, information state H0 (composition table only); independent marginals "
              "(C0 baseline); NASEM Table 19-1 SDs are observed commercial-lab SDs (DC-01). Not a formal result.")


def wrapped_nominal_check(w, problem) -> dict:
    th0, d0 = w.nominal_state()
    dth = float(np.max(np.abs(th0 - problem.nominal_theta())))
    dd = float(np.max(np.abs(d0 - problem.dm_estimates())))
    if not (dth < 1e-12 and dd < 1e-12):
        raise SystemExit(f"wrapped world nominal state differs from the problem (theta {dth}, d {dd})")
    return {"max_abs_nominal_theta_diff": dth, "max_abs_nominal_d_diff": dd}


def model_public_record(fm, w) -> dict:
    md = fm.metadata
    sm = md.summary_dict()
    return {"factory_version": md.factory_version, "base_model_id": fm.model_id, "base_model_fingerprint": fm.fingerprint(),
            "world_model_id": w.model_id, "world_fingerprint": w.fingerprint(),
            "energy_linearisation_fingerprint": w.lin.fingerprint(), "spec_id": md.spec_id,
            "spec_main_fingerprint": md.spec_main_fingerprint, "spec_fingerprint": fm.spec.fingerprint(),
            "metadata_fingerprint": md.fingerprint(), "purpose": md.purpose, "moment_semantics": md.moment_semantics,
            "family_rule": dict(md.family_rule), "fit_status_counts": sm["fit_status_counts"],
            "family_counts": sm["family_counts"], "fallback_cells": sm["fallback_cells"],
            "max_matched_moment_error_in_target_sd": sm["max_matched_moment_error_in_target_sd"],
            "variance_basis_counts": {b: sum(1 for c in md.cells if c.variance_basis == b)
                                      for b in sorted({c.variance_basis for c in md.cells})},
            "correlation": ("C0_independent (no correlation block)" if md.correlation is None else
                            f"{md.correlation.structure_id} ({md.correlation.input_scale}, {md.correlation.handling})"),
            "per_cell_fit": [{"ingredient_id": c.ingredient_id, "component": c.component, "family": c.family,
                              "fit_status": c.fit_status, "variance_basis": c.variance_basis,
                              "provenance_status": c.provenance_status,
                              "mean_rel_shift": _f(c.mean_rel_shift), "sd_rel_error": _f(c.sd_rel_error)}
                             for c in md.cells]}


def moment_rows(fm) -> list[dict]:
    out = []
    for c in fm.metadata.cells:
        out.append({"ingredient_id": c.ingredient_id, "component": c.component, "requested_family": c.requested_family,
                    "family": c.family, "fit_status": c.fit_status, "target_mean": c.target_mean,
                    "target_sd": c.target_sd, "achieved_mean": c.achieved_mean, "achieved_sd": c.achieved_sd,
                    "mean_rel_shift": c.mean_rel_shift, "sd_rel_error": c.sd_rel_error,
                    "outside_mass": c.outside_mass, "flags": ";".join(c.flags),
                    "family_choice_reason": c.family_choice_reason})
    return out


# ------------------------------------------------------------------------------------------------
# 2 boxes for M3 (factory moments + induced NEL interval)
# ------------------------------------------------------------------------------------------------
def factory_energy_box(fm, lin, k: float, tag: str = "dev_case_v1") -> BoxUncertaintySet:
    """Factory box + exact affine image of the energy column (FIX_B: moved into the library as
    ``nutrition.energy.energy_box_from_factory_model``; the K7 driver computed the same interval locally)."""
    return E.energy_box_from_factory_model(fm, lin, float(k), set_id=f"{tag}_factory_box_k{float(k):g}+NEL")


def make_box_solver(fn, boxes: dict):
    """Solve function whose parameter ``k_box`` selects a pre-built box (keeps arrays out of selection tables)."""
    def solve(problem, *, d_hat=None, opt_draws=None, params=None, solver_options=None):
        p = dict(params or {})
        k = float(p.pop("k_box"))
        p["uncertainty_set"] = boxes[k]
        return fn(problem, d_hat=d_hat, opt_draws=None, params=p, solver_options=solver_options)
    return solve


# ------------------------------------------------------------------------------------------------
# 3 diagnostics: minimum attainable SAA violation count
# ------------------------------------------------------------------------------------------------
def min_violations(problem, opt, N: int, rows: Optional[list[str]] = None, time_limit: float = 60.0) -> dict:
    """``min sum_s z_s`` over the SAA rows (exact MILP; box Big-M valid for every structurally feasible x)."""
    t0 = time.perf_counter()
    m = build_saa_model(problem, opt, n_scenarios=N, big_m_mode="box")
    A, b, M = np.asarray(m.A_prob), np.asarray(m.b_prob), np.asarray(m.M)
    ks = list(range(A.shape[1])) if rows is None else [list(m.prob_ids).index(r) for r in rows]
    I = A.shape[2]
    nv = I + N
    cons = []
    if m.A_s_ub is not None and np.size(m.A_s_ub):
        cons.append(LinearConstraint(sparse.hstack([sparse.csr_matrix(m.A_s_ub),
                                                    sparse.csr_matrix((m.A_s_ub.shape[0], N))]), -np.inf, m.b_s_ub))
    if m.A_s_eq is not None and np.size(m.A_s_eq):
        cons.append(LinearConstraint(sparse.hstack([sparse.csr_matrix(m.A_s_eq),
                                                    sparse.csr_matrix((m.A_s_eq.shape[0], N))]), m.b_s_eq, m.b_s_eq))
    data, ri, ci, ub = [], [], [], []
    r = 0
    n_hard = 0
    for s in range(N):
        for k in ks:
            for i in range(I):
                if A[s, k, i] != 0.0:
                    data.append(A[s, k, i]); ri.append(r); ci.append(i)
            if M[s, k] > 0:
                data.append(-M[s, k]); ri.append(r); ci.append(I + s)
            else:
                n_hard += 1          # holds for every x in the structural polytope (M <= 0): exact hard row
            ub.append(b[k])
            r += 1
    cons.append(LinearConstraint(sparse.csr_matrix((data, (ri, ci)), shape=(r, nv)), -np.inf, np.array(ub)))
    c = np.r_[np.zeros(I), np.ones(N)]
    integ = np.r_[np.zeros(I), np.ones(N)]
    res = milp(c, constraints=cons, integrality=integ,
               bounds=Bounds(np.zeros(nv), np.r_[np.full(I, np.inf), np.ones(N)]),
               options={"time_limit": float(time_limit), "mip_rel_gap": 0.0, "disp": False})
    ubv = None if res.x is None else int(round(float(res.fun)))
    lbv = getattr(res, "mip_dual_bound", None)
    lbv = None if lbv is None or not np.isfinite(lbv) else int(math.ceil(float(lbv) - 1e-6))
    if res.status == 0 and ubv is not None:
        lbv = ubv
    return {"N": N, "rows": "all" if rows is None else list(rows), "status_code": int(res.status),
            "status": {0: "optimal", 1: "time_limit"}.get(int(res.status), f"highs_status_{res.status}"),
            "message": str(res.message), "min_violations_upper": ubv, "min_violations_lower": lbv,
            "proven": bool(res.status == 0), "time_s": time.perf_counter() - t0, "n_rows_without_z": n_hard,
            "x": None if res.x is None else np.asarray(res.x[:I], dtype=float)}


def alpha_for_count(m: int, N: int) -> float:
    """Smallest float alpha near m/N with floor(alpha N) == m exactly (engine rule: floor from repr(alpha)).

    m/N is not representable for most N (e.g. 76/116): repr of the rounded double times N can fall just below m,
    and the engine would then allow m - 1 violations.  Step up by ulps until the engine's count equals m."""
    a = m / N
    for _ in range(64):
        if allowed_violations(a, N) >= m:
            break
        a = math.nextafter(a, 1.0)
    if allowed_violations(a, N) != m:
        raise ValueError(f"cannot represent {m}/{N} as alpha_train with floor(alpha N) == {m}")
    return a


def conflict_verdict(mv: dict, m_allowed: int) -> str:
    lb, ub = mv["min_violations_lower"], mv["min_violations_upper"]
    if lb is not None and lb > m_allowed:
        return "conflict_proven"
    if ub is not None and ub <= m_allowed:
        return "jointly_attainable"
    return "undetermined_within_time_limit"


# ------------------------------------------------------------------------------------------------
# 4 method runs (H0 or one H1 farm world)
# ------------------------------------------------------------------------------------------------
#: screening rules of the validation selection (``safety_margin.SCREENING_RULES``): the development replay default and
#: the official setting (official-run plan §1 "选参"; DECISIONS D-533; configs/methods.yaml)
DEVELOPMENT_SCREENING_RULE = "rate_upper_le_alpha"
OFFICIAL_SCREENING_RULE = "cp_upper_le_alpha"
OFFICIAL_SCREENING_CONFIDENCE = 0.95


def run_method_block(problem, fm, lin, opt, val, alphas, Ns, opts, *, m1_variants=(True, False), tag="H0",
                     box_tag: str = "dev_case_v1", screening_rule: str = DEVELOPMENT_SCREENING_RULE,
                     confidence: Optional[float] = None, selection_sink: Optional[list] = None):
    """All methods for every alpha; returns (runs, selection_tables).  Selection only on validation.

    ``screening_rule`` / ``confidence`` (official-run plan batch 2) go into **every** validation selection of the block
    (M1 and M2 through their selection mappings, M3a and every per-k M3b selection directly).  The defaults
    (``rate_upper_le_alpha``, no confidence) keep the development replay semantics of the runs made so far; the official
    setting is ``screening_rule="cp_upper_le_alpha"`` with ``confidence=0.95`` (one-sided exact Clopper-Pearson upper
    bound of (n_violated + n_unknown) / S <= alpha on the validation stream).

    ``selection_sink`` (optional list): every per-k M3b selection is appended to it as ``{"entry_label", "method_id",
    "target_alpha", "fixed_params", "selection"}`` after the M3b entry of that alpha is formed, so that a descriptive
    report can see the whole declared (k, Gamma) grid; nothing in the selection reads the sink."""
    if screening_rule not in (DEVELOPMENT_SCREENING_RULE, OFFICIAL_SCREENING_RULE):
        raise ValueError(f"run_method_block: unknown screening_rule {screening_rule!r}")
    if screening_rule == OFFICIAL_SCREENING_RULE and confidence is None:
        raise ValueError("run_method_block: cp_upper_le_alpha needs an explicit confidence (official setting 0.95)")
    runs: list[SP.MethodRun] = []
    tables: list[dict] = []
    world = opt.model_fingerprint
    cfg = RUN_CONFIG["methods"]
    m0 = get_method("M0_nominal")(problem, params=dict(cfg["M0"]["params"]), solver_options=opts)
    runs.append(SP.MethodRun(SP.MethodSpec("M0_nominal", params=cfg["M0"]["params"], label=f"{tag}:M0",
                                           uses_opt_draws=False), m0, world_fingerprint=world))
    boxes = {float(k): factory_energy_box(fm, lin, float(k), box_tag) for k in cfg["M3a"]["k_grid"]}
    m3a = make_box_solver(solve_box_robust, boxes)
    m3b = make_box_solver(solve_budget_robust, boxes)
    for a in alphas:
        specs = []
        for adm in m1_variants:
            specs.append(SP.MethodSpec("M1_safety_margin", params={"margin_scale": "relative", "apply_to_dm": adm},
                                       selection={"param_name": "k", "grid": cfg["M1"]["grid"], "target_alpha": a,
                                                  "screening_rule": screening_rule, "confidence": confidence},
                                       label=f"{tag}:M1[relative,apply_to_dm={adm}]@alpha={a:g}"))
        for N in Ns:
            grid = [float(repr_mult(mu, a)) for mu in cfg["M2"]["alpha_train_multipliers"]]
            specs.append(SP.MethodSpec("M2_joint_chance_saa",
                                       params={"n_scenarios": N, "big_m_mode": cfg["M2"]["big_m_mode"],
                                               "polish": cfg["M2"]["polish"]},
                                       selection={"param_name": "alpha_train", "grid": grid, "target_alpha": a,
                                                  "screening_rule": screening_rule, "confidence": confidence},
                                       label=f"{tag}:M2[N={N}]@alpha={a:g}"))
        rr = SP.run_methods(problem, specs, opt_draws=opt, validation_draws=val, solver_options=opts)
        for r in rr:
            tables.append({"label": r.spec.name, "method_id": r.spec.method_id, "target_alpha": a,
                           "selection": r.selection.to_dict() if r.selection else None})
        runs.extend(rr)
        # M3a: selection over k (validation)
        sel = select_parameter_on_validation(m3a, problem, val, param_name="k_box", grid=list(boxes),
                                             target_alpha=a, screening_rule=screening_rule, confidence=confidence,
                                             base_params={},
                                             opt_draws=None, solver_options=opts, consumer=f"K7:{tag}:M3a")
        runs.append(SP.MethodRun(SP.MethodSpec("M3a_box_robust", label=f"{tag}:M3a[factory box]@alpha={a:g}",
                                               uses_opt_draws=False), sel.selected_result, sel,
                                 world_fingerprint=world))
        tables.append({"label": f"{tag}:M3a@alpha={a:g}", "method_id": "M3a_box_robust", "target_alpha": a,
                       "selection": sel.to_dict()})
        # M3b: per k select Gamma, then across k
        per_k = []
        for k in boxes:
            s = select_parameter_on_validation(m3b, problem, val, param_name="gamma", grid=cfg["M3b"]["gamma_grid"],
                                               target_alpha=a, screening_rule=screening_rule, confidence=confidence,
                                               base_params={"k_box": k, "nominal_mode": cfg["M3b"]["nominal_mode"]},
                                               opt_draws=None, solver_options=opts, consumer=f"K7:{tag}:M3b")
            per_k.append((k, s))
        ok = [(s.selected_result.objective if s.selected_result else math.inf,
               next(c.rate_upper for c in s.candidates if c.value == s.selected_value), pos, k, s)
              for pos, (k, s) in enumerate(per_k) if s.status == "selected"]
        if ok:
            ok.sort(key=lambda t: (t[0], t[1], t[2]))
            _, _, _, kb, sb = ok[0]
            runs.append(SP.MethodRun(SP.MethodSpec("M3b_budget_robust",
                                                   label=f"{tag}:M3b[factory box k={kb:g}, Gamma selected]@alpha={a:g}",
                                                   uses_opt_draws=False), sb.selected_result, sb,
                                     world_fingerprint=world))
        else:
            sb = per_k[0][1]
            runs.append(SP.MethodRun(SP.MethodSpec("M3b_budget_robust", label=f"{tag}:M3b[factory box]@alpha={a:g}",
                                                   uses_opt_draws=False), None, sb, world_fingerprint=world))
        if selection_sink is not None:
            for k, s in per_k:
                selection_sink.append({"entry_label": runs[-1].spec.name, "method_id": "M3b_budget_robust",
                                       "target_alpha": a, "fixed_params": {"k_box": float(k)}, "selection": s})
        tables.append({"label": f"{tag}:M3b@alpha={a:g}", "method_id": "M3b_budget_robust", "target_alpha": a,
                       "selection_across_k": "lowest cost among per-k selections; ties lower rate_upper then smaller k",
                       "per_k": [{"k_box": k, "selection": s.to_dict()} for k, s in per_k]})
    return runs, tables


def repr_mult(mu: float, a: float) -> str:
    """alpha_train = multiplier x alpha as a short decimal (exact floor(alpha N) downstream)."""
    return repr(round(mu * a, 12))


def run_alpha(label: str) -> Optional[float]:
    if "@alpha=" in label:
        return float(label.split("@alpha=")[1])
    return None


# ------------------------------------------------------------------------------------------------
# 5 tables
# ------------------------------------------------------------------------------------------------
def summary_rows(runs, problem, run_id: str, price_meta: dict, world_fp: str) -> list[dict]:
    rows = []
    for r in runs:
        res, ev = r.result, r.evaluation
        row = {"run_id": run_id, "label": r.spec.name, "method_id": r.spec.method_id,
               "target_alpha": run_alpha(r.spec.name), "status": r.status,
               "solver_status": None if res is None else str(res.status),
               "selection_status": None if r.selection is None else r.selection.status,
               "selected_param": (None if r.selection is None or r.selection.status != "selected" else
                                  f"{r.selection.param_name}={r.selection.selected_value}"),
               "cost_usd_per_head_d": None if ev is None else _f(ev.cost),
               "currency": price_meta["currency"], "price_period": price_meta["period"],
               "price_basis": price_meta["basis"],
               "mip_gap": None if res is None else _f(res.mip_gap),
               "solve_wall_time_s": None if res is None else _f(res.wall_time_s),
               "n_test": None, "joint_n_violated": None, "joint_n_unknown": None, "joint_rate": None,
               "joint_cp_upper_one_sided_95": None, "joint_cp_two_sided_95_lower": None,
               "joint_cp_two_sided_95_upper": None, "joint_mc_se": None, "structural_ok": None,
               "world_fingerprint": r.world_fingerprint or world_fp, "test_stream_id": None,
               "world_shift_scenario": r.world_shift_scenario}
        if ev is not None:
            s = ev.summary()
            j = s["joint"]
            n = s["n_draws"]
            k = j["n_violated"] + j["n_unknown"]
            lo, hi = clopper_pearson(k, n, 0.95)
            row.update({"n_test": n, "joint_n_violated": j["n_violated"], "joint_n_unknown": j["n_unknown"],
                        "joint_rate": k / n, "joint_cp_upper_one_sided_95": one_sided_upper(k, n, 0.95),
                        "joint_cp_two_sided_95_lower": lo, "joint_cp_two_sided_95_upper": hi,
                        "joint_mc_se": mc_standard_error(k / n, n), "structural_ok": bool(s["structural_ok"]),
                        "test_stream_id": s["draw_stream_id"]})
        rows.append(row)
    return rows


SUMMARY_COLUMNS = ["run_id", "label", "method_id", "target_alpha", "status", "solver_status", "selection_status",
                   "selected_param", "cost_usd_per_head_d", "currency", "price_period", "price_basis", "mip_gap",
                   "solve_wall_time_s", "n_test", "joint_n_violated", "joint_n_unknown", "joint_rate",
                   "joint_cp_upper_one_sided_95", "joint_cp_two_sided_95_lower", "joint_cp_two_sided_95_upper",
                   "joint_mc_se", "structural_ok", "world_fingerprint", "test_stream_id", "world_shift_scenario"]


def ration_rows(runs, problem, run_id: str, *, restricted: bool) -> list[dict]:
    rows = []
    prices = problem.price_vector()
    for r in runs:
        if not r.has_ration:
            rows.append({"run_id": run_id, "label": r.spec.name, "status": r.status, "ingredient_id": None,
                         "q_as_fed_kg_per_head_d": None})
            continue
        dec = r.result.decision
        q = np.asarray(dec.q_as_fed, float)
        dh = np.asarray(dec.d_hat, float)
        x = q * dh
        for i, iid in enumerate(dec.ingredient_ids):
            row = {"run_id": run_id, "label": r.spec.name, "status": r.status, "ingredient_id": iid,
                   "q_as_fed_kg_per_head_d": float(q[i])}
            if restricted:
                row.update({"d_hat": float(dh[i]), "planned_dm_kg_per_head_d": float(x[i]),
                            "planned_dm_share": float(x[i] / x.sum()) if x.sum() > 0 else None,
                            "cost_usd_per_head_d": float(prices[i] * q[i])})
            rows.append(row)
    return rows


def constraint_meta() -> dict:
    c = yaml.safe_load((CASE_CFG / "constraints.yaml").read_text(encoding="utf-8"))
    out = {}
    for row in c["constraints"]:
        b = row.get("bound")
        if isinstance(b, dict) and "value" in b:
            bound = b["value"]
        elif isinstance(b, dict) and "value_ref" in b:
            bound = "restricted (value_ref)"
        else:
            bound = b if not isinstance(b, dict) else json.dumps(b, ensure_ascii=False)
        eng = row.get("engine") or {}
        out[row["id"]] = {"class": row["class"], "expression": row.get("expression"), "unit": eng.get("unit"),
                          "sense": eng.get("sense"), "bound_public": bound, "status": row.get("status"),
                          "source": row.get("source")}
    return out


def constraint_meta_s2(meta: dict, s2_rec: dict) -> dict:
    """Constraint metadata as used in S2: planned structural rows added, their scenario forms diagnostic."""
    out = copy.deepcopy(meta)
    for cid, rec in s2_rec["planned_rows"].items():
        base = dict(meta.get(cid, {}))
        out[cid] = dict(base, **{"class": "diagnostic_only (S2: scenario form of a planned row)"})
        out[rec["structural_id"]] = dict(base, **{"class": "structural_hard (S2 planned form, table values, d_hat)",
                                                 "unit": rec["unit_note"], "sense": base.get("sense"),
                                                 "status": base.get("status"),
                                                 "source": f"S2 planning form of {cid}; " + str(base.get("source"))})
    return out


def residual_rows(runs, problem, run_id: str, meta: dict, *, restricted: bool) -> list[dict]:
    rows = []
    th0 = problem.nominal_theta()
    for r in runs:
        if not r.has_ration:
            continue
        dec = r.result.decision
        evn = evaluate(dec, th0[None], np.asarray(dec.d_hat, float)[None], problem.compiled, prices=problem.prices)
        nom = {cid: float(evn.margin[0, k]) for k, cid in enumerate(evn.constraint_ids)}
        smarg = np.atleast_2d(evn.structural_margin)
        for k, cid in enumerate(evn.structural_ids):
            nom[cid] = float(smarg[0, k])
        s = r.evaluation.summary() if r.evaluation is not None else None
        for cid in list(evn.structural_ids) + list(evn.constraint_ids):
            m = meta.get(cid, {})
            pc = (s or {}).get("per_constraint", {}).get(cid) if s else None
            row = {"run_id": run_id, "label": r.spec.name, "constraint_id": cid, "class": m.get("class"),
                   "unit": m.get("unit"), "sense": m.get("sense"), "bound": m.get("bound_public"),
                   "value_status": m.get("status"), "source": m.get("source"),
                   "planned_nominal_margin": nom.get(cid), "n_test": None, "n_violated": None,
                   "violation_rate": None, "cp_upper_one_sided_95": None, "mc_se": None,
                   "mean_deficit_given_violation": None, "max_deficit": None, "margin_q01": None,
                   "margin_q05": None, "margin_q50": None}
            if pc is not None:
                nd, nv = pc["n_defined"], pc["n_violated"]
                mq = pc["margin_quantiles"] or {}
                row.update({"n_test": nd, "n_violated": nv, "violation_rate": nv / nd if nd else None,
                            "cp_upper_one_sided_95": one_sided_upper(nv, nd, 0.95) if nd else None,
                            "mc_se": mc_standard_error(nv / nd, nd) if nd else None,
                            "mean_deficit_given_violation": pc["mean_deficit_given_violation"],
                            "max_deficit": pc["max_deficit"], "margin_q01": mq.get(0.01), "margin_q05": mq.get(0.05),
                            "margin_q50": mq.get(0.5)})
            if not restricted:
                if cid in REDACT_NOMINAL:
                    row["planned_nominal_margin"] = "restricted"
                if cid in REDACT_TEST_MARGINS:
                    for f in ("mean_deficit_given_violation", "max_deficit", "margin_q01", "margin_q05", "margin_q50"):
                        row[f] = "restricted"
            rows.append(row)
    return rows


RESIDUAL_COLUMNS = ["run_id", "label", "constraint_id", "class", "unit", "sense", "bound", "value_status", "source",
                    "planned_nominal_margin", "n_test", "n_violated", "violation_rate", "cp_upper_one_sided_95",
                    "mc_se", "mean_deficit_given_violation", "max_deficit", "margin_q01", "margin_q05", "margin_q50"]


def profile_rows(runs, problem, test, run_id: str, lin, *, restricted: bool) -> list[dict]:
    rows = SP.nutrient_profile([r for r in runs if r.has_ration], problem, test, run_id=run_id)
    ke = list(problem.nutrient_ids).index(E.ENERGY_COLUMN_ID)
    for r in runs:
        if not r.has_ration:
            continue
        dec = r.result.decision
        q, dh = np.asarray(dec.q_as_fed, float), np.asarray(dec.d_hat, float)
        nom = float(q * dh @ problem.nominal_theta()[:, ke] + lin.constant_mcal_d)
        sup = (q[None, :] * test.d * test.theta[:, :, ke]).sum(axis=1) + lin.constant_mcal_d
        qs = np.quantile(sup, [0.05, 0.5, 0.95])
        rows.append({"run_id": run_id, "method_id": r.spec.method_id, "method_label": r.spec.name,
                     "quantity": "NEL_fixedDMI_supply", "unit": "Mcal/head/d (linear row incl. C0)",
                     "planned_nominal": nom, "draw_p05": float(qs[0]), "draw_p50": float(qs[1]),
                     "draw_p95": float(qs[2]), "n_draws": int(test.n_draws), "n_undefined": 0,
                     "draws_stream_id": test.stream_id, "is_synthetic": False})
    if restricted:
        return rows
    return [x for x in rows if x["quantity"] in PUBLIC_PROFILE_QUANTITIES]


def redact_long(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        row = dict(row)
        if row.get("constraint_id") in REDACT_TEST_MARGINS:
            for f in ("mean_deficit_given_violation", "max_deficit", "margin_q01", "margin_q05", "margin_q50"):
                row[f] = "restricted"
        out.append(row)
    return out


#: keys dropped from public solve records: they return restricted values in one step (x_box_hi of a mineral =
#: its inclusion cap x DMI; residuals of the inclusion caps; raw candidates)
PUBLIC_SOLVE_DENYLIST = {"x_box_lo", "x_box_hi", "raw_candidate_x", "constraint_residuals"}


def solve_record(res, *, restricted: bool) -> dict:
    d = json.loads(json.dumps(res.to_dict(), default=_json_default))
    if not restricted and d.get("decision"):
        d["decision"] = {"ingredient_ids": d["decision"].get("ingredient_ids"),
                         "q_as_fed_kg_per_head_per_day": d["decision"].get("q_as_fed_kg_per_head_per_day"),
                         "method_id": d["decision"].get("method_id"),
                         "d_hat_and_x_planned": "restricted (NASEM DM means); full record in restricted_local"}
    if not restricted:
        d = _deny(d)
        d["public_redaction"] = sorted(PUBLIC_SOLVE_DENYLIST)
    return d


def _deny(o):
    if isinstance(o, dict):
        return {k: ("restricted" if k in PUBLIC_SOLVE_DENYLIST else _deny(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [_deny(v) for v in o]
    return o


def largest_feasible(solve_fn, problem, base_params: dict, param: str, lo: float, hi: float, tol: float, opts,
                     opt_draws=None) -> tuple[dict, Any]:
    """Bisection for the largest feasible value of a monotone conservatism parameter (diagnostic).

    Requires a ration at ``lo`` and ``proven_infeasible`` at ``hi``; stops at ``hi - lo <= tol`` or at any
    other status (recorded, never treated as infeasible)."""
    trace = []

    def solve(v):
        r = solve_fn(problem, opt_draws=opt_draws, params={**base_params, param: v}, solver_options=opts)
        trace.append({"value": v, "status": str(r.status)})
        return r

    best = solve(lo)
    top = solve(hi)
    out = {"param": param, "base_params": {k: v for k, v in base_params.items()}, "tol": tol, "trace": trace}
    if not best.has_solution or str(top.status) != "proven_infeasible":
        out.update({"status": "bracket_not_valid", "largest_feasible": None})
        return out, None
    stop = None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        r = solve(mid)
        if r.has_solution:
            lo, best = mid, r
        elif str(r.status) == "proven_infeasible":
            hi = mid
        else:
            stop = str(r.status)
            break
    out.update({"status": "ok" if stop is None else f"stopped_at_status_{stop}", "largest_feasible": lo,
                "smallest_proven_infeasible": hi, "n_solves": len(trace)})
    return out, best


# ------------------------------------------------------------------------------------------------
# 6 energy diagnostics
# ------------------------------------------------------------------------------------------------
def nel_headroom(problem, lin) -> dict:
    cc = problem.compiled
    rows = [k for k, cid in enumerate(cc.constraint_ids)
            if cc.classes[k] is not ConstraintClass.DIAGNOSTIC_ONLY and cid != "PN-NEL-FIXEDDMI"]
    th = problem.nominal_theta()
    dh = problem.dm_estimates()
    lr = linear_rows(cc.subset(rows), th[None], dh[None], d_hat=dh)
    A, b, iseq = lr.A[0], lr.b, lr.is_eq
    ke = list(problem.nutrient_ids).index(E.ENERGY_COLUMN_ID)
    cobj = -(dh * th[:, ke])
    r = linprog(cobj, A_ub=A[~iseq], b_ub=b[~iseq], A_eq=A[iseq], b_eq=b[iseq], bounds=[(0, None)] * len(dh),
                method="highs")
    req = [c for c in problem.constraints if c.constraint_id == "PN-NEL-FIXEDDMI"][0]
    need = float(req.bound) + float(lin.constant_mcal_d)
    if r.status != 0:
        return {"status": f"linprog_status_{r.status}", "message": r.message}
    mx = -float(r.fun) + float(lin.constant_mcal_d)
    return {"status": "optimal", "max_linear_nel_supply_mcal_d": mx, "requirement_mcal_d": need,
            "headroom_pct_of_requirement": 100.0 * (mx / need - 1.0),
            "note": "nominal composition and d_hat; every other structural and probabilistic row imposed at nominal"}


def linearisation_residual(dec, test, lin, problem, n_max: int, req_mcal: Optional[float] = None) -> dict:
    ke = list(problem.nutrient_ids).index(E.ENERGY_COLUMN_ID)
    q = np.asarray(dec.q_as_fed, float)
    feeds = list(lin.feeds)
    nmap = dict(lin.nutrient_map)                     # engine id -> composition field
    jn = {f: list(problem.nutrient_ids).index(n) for n, f in nmap.items()}
    st = lin.settings
    n = min(int(test.n_draws), int(n_max))
    res = np.full(n, np.nan)
    starch = np.full(n, np.nan)
    dsum = np.full(n, np.nan)
    n_err = 0
    for s in range(n):
        x = q * test.d[s]
        comps = [{f: float(test.theta[s, i, j] * 100.0) for f, j in jn.items()} for i in range(len(feeds))]
        try:
            nl = E.nonlinear_diet_nel(x, feeds, body_weight_kg=st.body_weight_kg, milk_cp_kg_d=st.milk_cp_kg_d,
                                      body_gain_cp_kg_d=st.body_gain_cp_kg_d, compositions=comps)
        except ValueError:
            n_err += 1
            continue
        lin_sup = float(x @ test.theta[s, :, ke] + lin.constant_mcal_d)
        res[s] = lin_sup - nl["NEL"]
        starch[s], dsum[s] = nl["starch_pct"], nl["DMI"]
    ok = np.isfinite(res)
    v = res[ok]
    qs = np.quantile(v, [0.01, 0.05, 0.5, 0.95, 0.99]) if v.size else [None] * 5

    def group(mask) -> dict:
        g = res[ok & mask]
        if not g.size:
            return {"n": 0}
        return {"n": int(g.size), "share_of_defined": float(g.size / max(1, int(ok.sum()))),
                "share_linear_above_chain": float(np.mean(g > 0)), "q50": float(np.quantile(g, 0.5)),
                "q95": float(np.quantile(g, 0.95)), "max": float(g.max()),
                "max_pct_of_requirement": None if req_mcal is None else float(100.0 * g.max() / req_mcal)}

    s_ref = float(st.starch_ref_pct)
    low_starch = np.where(np.isfinite(starch), starch <= s_ref, False)
    near_dmi = np.where(np.isfinite(dsum), np.abs(dsum - float(st.dmi_kg_d)) < 0.05, False)
    return {"diagnostic_id": "DIAG-NEL-LINEARISATION-RESIDUAL", "unit": "Mcal/head/d (linear row - full chapter-3 "
            "chain with the draw's own D and starch)", "n_draws_used": int(n), "n_undefined": int(n_err),
            "stream_id": test.stream_id, "mean": float(v.mean()) if v.size else None,
            "q01": _f(qs[0]), "q05": _f(qs[1]), "q50": _f(qs[2]), "q95": _f(qs[3]), "q99": _f(qs[4]),
            "share_linear_above_chain": float(np.mean(v > 0)) if v.size else None,
            "grouped_fixb": {
                "note": ("FIX_B (red team B #3): the linear row is not above the chain only at the table-mean "
                         "composition with D = DMI and starch <= S_ref; with drawn composition (NDF below the feed "
                         "mean) it can exceed the chain even in the group 'starch <= S_ref'"),
                "starch_le_S_ref": group(low_starch), "starch_gt_S_ref": group(~low_starch),
                "starch_le_S_ref_and_abs_D_minus_DMI_lt_0.05kg": group(low_starch & near_dmi)}}


# ------------------------------------------------------------------------------------------------
# 7 H1 farm worlds
# ------------------------------------------------------------------------------------------------
def farm_problem_cfg(cfg0: dict, th0: np.ndarray, d0: np.ndarray, ids, nut_ids, farm_label: str) -> dict:
    cfg = copy.deepcopy(cfg0)
    i = list(ids).index(CS)
    why = (f"H1 scenario ({farm_label}): known farm mean of the two-layer extension scenario (research_scenario_"
           "assumption; configs/uncertainty.yaml H1_farm_history, 14 d true_only ratios)")
    for g in cfg["ingredients"]:
        if g["ingredient_id"] != CS:
            continue
        for n in ("NDF", "starch"):
            g["composition"][n] = {"value": float(th0[i, list(nut_ids).index(n)] * 100.0), "unit": "%", "basis": "DM",
                                   "status": "research_scenario_assumption", "rationale": why}
        g["composition"][E.ENERGY_COLUMN_ID] = {"value": float(th0[i, list(nut_ids).index(E.ENERGY_COLUMN_ID)]),
                                                "unit": "Mcal/kg", "basis": "DM",
                                                "status": "research_scenario_assumption",
                                                "rationale": why + "; NEL_fixedDMI at the farm composition (same "
                                                "linearisation)"}
        g["dm_estimate"] = {"value": float(d0[i] * 100.0), "unit": "%", "basis": "as_fed",
                            "status": "research_scenario_assumption", "rationale": why + "; decision-time d_hat"}
    cfg["problem_id"] = f"{cfg0['problem_id']}|H1|{farm_label}"
    return cfg


def h1_block(ext_model, fm_h0, lin, cfg0, problem_h0, h0_rations, streams, opts, cells, cells_sha, n_val, n_test,
             farms, time_limit_mv, registry: StreamRegistry) -> tuple[list[dict], dict, list]:
    """H1-perfect per test farm: farm world, decisions with the known farm mean, same public evaluator."""
    ids = problem_h0.ingredient_ids
    tb, tw, db, dw = ext_model.between_within()
    tcov, dcov = ext_model.covered()
    i = list(ids).index(CS)
    tm, ts, dm, ds = cell_arrays(cells, ids)
    over0 = cell_overrides(cells, cells_sha)
    pub_rows, restricted, all_runs = [], {"farm_means": []}, []
    for f in farms:
        t0 = time.perf_counter()
        mt, md, sid = ext_model.farm_means(streams, "test", f)
        registry.add(f"H1_farm{f}_mean", stream_id=sid, world=ext_model.fingerprint() if hasattr(ext_model, "fingerprint")
                     else None, n=1, purpose="H1 farm mean (farm_mean stream, purpose test)")
        tmf, tsf, dmf, dsf = tm.copy(), ts.copy(), dm.copy(), ds.copy()
        over = copy.deepcopy(over0)
        for j, n in enumerate(PN):
            if tcov[i, j]:
                tmf[i, j], tsf[i, j] = mt[i, j], tw[i, j]
                over[(CS, n)].update({"variance_basis": "unidentified", "provenance_status": "research_scenario_assumption",
                                      "source_id": None, "locator": None,
                                      "data_fingerprint": stable_hash("dev_case_v1_H1_cell/v1", over0[(CS, n)]["data_fingerprint"],
                                                                      sid, float(mt[i, j]), float(tw[i, j]))})
        if dcov[i]:
            dmf[i], dsf[i] = md[i], dw[i]
            over[(CS, "DM")].update({"variance_basis": "unidentified", "provenance_status": "research_scenario_assumption",
                                     "source_id": None, "locator": None,
                                     "data_fingerprint": stable_hash("dev_case_v1_H1_cell/v1", over0[(CS, "DM")]["data_fingerprint"],
                                                                     sid, float(md[i]), float(dw[i]))})
        spec_f = UncertaintySpec.from_arrays(
            f"DEV_CASE_V1_H1_farm{f}", ids, PN, tmf, tsf, dmf, dsf, purpose="sensitivity_scenario",
            moment_semantics="target_marginal_moments", is_synthetic=False,
            family_rule=RUN_CONFIG["uncertainty"]["family_rule"],
            cell_defaults={"variance_basis": "observed_incl_sampling_and_lab", "decomposition_id": "none",
                           "decomposition_source": None, "measurement_model_id": MEASUREMENT_MODEL_NASEM},
            theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0), cell_overrides=over,
            notes=f"K7 dev_case_v1 H1 farm world {f} (farm mean from {sid}); research_scenario_assumption, extension "
                  "scenario, not a result")
        fm_f = build_uncertainty_model(spec_f)
        w_f = E.EnergyColumnModel(fm_f, lin)
        th0, d0 = w_f.nominal_state()
        cfg_f = farm_problem_cfg(cfg0, th0, d0, ids, problem_h0.nutrient_ids, f"farm{f}")
        prob_f, rep_f = build_problem(cfg_f, mode="pilot")
        nomchk = wrapped_nominal_check(w_f, prob_f)
        sk = h1_farm_subkey(f)                         # FIX_B: offset namespace (K7 used sub-key f -> opt/1, opt/2)
        opt_f = w_f.draw(streams, "opt", RUN_CONFIG["H1"]["N"], sk)
        val_f = w_f.draw(streams, "validation", n_val, sk)
        test_f = w_f.draw(streams, "test", n_test, sk)
        for ds_ in (opt_f, val_f, test_f):
            registry.add(f"H1_farm{f}", ds_, purpose=f"H1 farm {f} {ds_.stream}")
        runs_f, tables_f = run_method_block(prob_f, fm_f, lin, opt_f, val_f, [RUN_CONFIG["H1"]["alpha"]],
                                            [RUN_CONFIG["H1"]["N"]], opts, m1_variants=(True,), tag=f"H1farm{f}")
        SP.evaluate_on_test(runs_f, prob_f, test_f)
        mv = min_violations(prob_f, opt_f, RUN_CONFIG["H1"]["N"], time_limit=time_limit_mv)
        def dm_diag(dec_):
            dsup = (np.asarray(dec_.q_as_fed, float)[None, :] * test_f.d).sum(axis=1)
            return {"realised_dm_supply_mean_kg_d": float(dsup.mean()),
                    "realised_dm_supply_p05_p95_kg_d": [float(np.quantile(dsup, 0.05)), float(np.quantile(dsup, 0.95))],
                    "share_above_diag_dm_real_hi": float(np.mean(dsup > 1.1 * float(lin.settings.dmi_kg_d))),
                    "share_below_diag_dm_real_lo": float(np.mean(dsup < 0.9 * float(lin.settings.dmi_kg_d)))}
        h0_eval = {}
        for lab, dec in h0_rations.items():
            ev = evaluate_drawset(dec, test_f, prob_f.compiled, prices=problem_h0.prices)
            j = ev.summary()["joint"]
            k = j["n_violated"] + j["n_unknown"]
            h0_eval[lab] = {"joint_rate": k / j["n_draws"], "cp_upper_one_sided_95": one_sided_upper(k, j["n_draws"]),
                            "cost_usd_per_head_d_at_H0_prices": _f(ev.cost), "structural_ok_under_own_d_hat":
                            bool(ev.structural_ok), "world_shift_scenario": f"H1_farm{f}_world",
                            "dm_supply_in_farm_world": dm_diag(dec),
                            "note": "the ration is executed with the table DM estimate; in this farm world the realised "
                                    "DM supply differs from the planned DMI (supply rows assume all supplied DM is eaten)"}
        own_dm = {r.spec.name: dm_diag(r.result.decision) for r in runs_f if r.has_ration}
        zs = {"DM": float((md[i] - dm[i]) / ds[i])}
        for j, n in enumerate(PN):
            if tcov[i, j]:
                zs[n] = float((mt[i, j] - tm[i, j]) / ts[i, j])
        srows = summary_rows(runs_f, prob_f, "", {"currency": "USD", "period": "see H0", "basis": "as-fed after "
                             "decision-time DM conversion (corn silage at the farm d_hat)"}, w_f.fingerprint())
        pub_rows.append({"farm_index": f, "farm_mean_stream_id": sid, "world_fingerprint": w_f.fingerprint(),
                         "problem_validator_ok": rep_f.ok, "n_research_assumptions": rep_f.n_assumptions,
                         "nominal_check": nomchk,
                         "min_violation_N128": {k: v for k, v in mv.items() if k != "x"},
                         "attainable_training_joint_risk_lower": (mv["min_violations_lower"] / mv["N"]
                                                                  if mv["min_violations_lower"] is not None else None),
                         "methods": [{k: row[k] for k in ("label", "status", "selection_status", "selected_param",
                                                          "cost_usd_per_head_d", "joint_rate",
                                                          "joint_cp_upper_one_sided_95", "joint_mc_se")}
                                     for row in srows],
                         "H0_rations_in_this_farm_world": h0_eval, "H1_rations_dm_supply": own_dm,
                         "corn_silage_farm_mean_z_vs_table": zs,
                         "z_note": "(mu_farm - table mean) / table SD; dimensionless, no table value",
                         "selection_tables": tables_f,
                         "elapsed_s": time.perf_counter() - t0})
        restricted["farm_means"].append({"farm_index": f, "stream_id": sid,
                                         "corn_silage_mu_farm": {"DM": float(md[i]),
                                                                 **{n: float(mt[i, j]) for j, n in enumerate(PN)
                                                                    if tcov[i, j]}},
                                         "corn_silage_sigma_within": {"DM": float(dw[i]),
                                                                      **{n: float(tw[i, j]) for j, n in enumerate(PN)
                                                                         if tcov[i, j]}},
                                         "rations": ration_rows(runs_f, prob_f, "", restricted=True)})
        all_runs.append((f, prob_f, runs_f, test_f))
    return pub_rows, restricted, all_runs


# ------------------------------------------------------------------------------------------------
# 7b FIX_B: composition closure, C1 correlation world, S2 scenario, CP-SUP RDP-scaled diagnostic
# ------------------------------------------------------------------------------------------------
def closure_block(label: str, draws, lin) -> tuple[dict, dict]:
    """Per-ingredient composition closure on one draw set (counted only; red team B #2)."""
    rep = E.composition_closure_report(draws.theta, draws.nutrient_ids, draws.ingredient_ids, lin=lin)
    pub = {"world": label, "stream_id": draws.stream_id, "n_draws": int(draws.n_draws),
           "world_fingerprint": draws.model_fingerprint,
           "per_ingredient": {iid: {k: r[k] for k in ("n_draws", "share_sum_gt_100pct", "share_rom_lt_0",
                                                     "sum_columns")} for iid, r in rep.items()},
           "rule": RUN_CONFIG["diagnostics_fixb"]["composition_closure"]}
    return pub, {"world": label, "stream_id": draws.stream_id, "per_ingredient": rep}


def closure_impact(label: str, dec, draws, problem, lin) -> dict:
    """Joint violation of one ration in draws where a fed ingredient has ROM < 0 vs the other draws."""
    q = np.asarray(dec.q_as_fed, float)
    th = np.asarray(draws.theta, float)
    nut = list(draws.nutrient_ids)
    field_of = {v: k for k, v in dict(lin.nutrient_map).items()}
    bad = np.zeros(draws.n_draws, dtype=bool)
    for i, f in enumerate(lin.feeds):
        if q[i] <= 0:
            continue
        comp = {fld: (th[:, i, nut.index(field_of[fld])] * 100.0 if fld in field_of else float(getattr(f, fld)))
                for fld in ("ndf", "starch", "fa", "cp", "ash")}
        rom = 100.0 - comp["ash"] - comp["ndf"] - comp["starch"] - comp["fa"] / float(f.fat_factor) - comp["cp"]
        bad |= rom < -1e-7
    ev = evaluate_drawset(dec, draws, problem.compiled, prices=problem.prices)
    j = np.asarray(ev.joint_violation, bool) | np.asarray(ev.joint_unknown, bool)
    return {"ration": label, "stream_id": draws.stream_id, "n_draws": int(draws.n_draws),
            "share_draws_with_a_fed_ingredient_rom_lt_0": float(bad.mean()),
            "joint_rate_all": float(j.mean()),
            "joint_rate_in_rom_lt_0_draws": float(j[bad].mean()) if bad.any() else None,
            "joint_rate_in_closed_draws": float(j[~bad].mean()) if (~bad).any() else None,
            "note": "the unclosed draws are kept in every rate of this run (no clipping, no dropping)"}


def build_c1_spec(ids, cells, cells_sha) -> UncertaintySpec:
    tm, ts, dm, ds = cell_arrays(cells, ids)
    c = RUN_CONFIG["C1"]
    pr = c["pairs"]
    labels = [[CS, "NDF"], [CS, "starch"], [CS, "CP"], [CS, "ash"]]
    matrix = [[1.0, pr["starch_NDF"], pr["CP_NDF"], pr["ash_NDF"]], [pr["starch_NDF"], 1.0, 0.0, 0.0],
              [pr["CP_NDF"], 0.0, 1.0, 0.0], [pr["ash_NDF"], 0.0, 0.0, 1.0]]
    corr = {"structure_id": c["structure_id"], "labels": labels, "matrix": matrix, "input_scale": c["input_scale"],
            "handling": c["handling"], "status": c["status"],
            "provenance": ("Yoder et al. 2014 Table 6 population Pearson r for corn silage (configs/uncertainty.yaml "
                           "C1_yoder_table6.values_used.corn_silage); transformed-scale r calibrated to the latent "
                           "Gaussian copula; population-level (includes between-farm) correlation"),
            "source_id": "SRC-C-YODER2014", "locator": "Yoder et al. 2014 Table 6 (corn silage)"}
    return UncertaintySpec.from_arrays(
        "DEV_CASE_V1_C1_TN_MM", ids, PN, tm, ts, dm, ds, purpose="sensitivity_scenario",
        moment_semantics="target_marginal_moments", is_synthetic=False,
        family_rule=RUN_CONFIG["uncertainty"]["family_rule"],
        cell_defaults={"variance_basis": "observed_incl_sampling_and_lab", "decomposition_id": "none",
                       "decomposition_source": None, "measurement_model_id": MEASUREMENT_MODEL_NASEM},
        theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0), cell_overrides=cell_overrides(cells, cells_sha),
        correlation=corr,
        notes="FIX_B dev_case_v1 C1 scenario: H0 marginals + declared corn-silage correlations (Yoder 2014 Table 6); "
              "research_scenario_assumption; not a result")


S2_BORROWED_RATIO = 1.0 / 2.1
S2_CS_RATIOS = {"DM": 1.0 / 4.4, "NDF": 1.0 / 3.6, "starch": 1.0 / 5.0}


def s2_ratio_arrays(ids) -> tuple[np.ndarray, np.ndarray]:
    r = np.full((len(ids), len(PN)), S2_BORROWED_RATIO)
    rd = np.full(len(ids), S2_BORROWED_RATIO)
    i = list(ids).index(CS)
    for n, v in S2_CS_RATIOS.items():
        if n == "DM":
            rd[i] = v
        else:
            r[i, PN.index(n)] = v
    return r, rd


def build_s2_spec(ids, cells, cells_sha) -> UncertaintySpec:
    tm, ts, dm, ds = cell_arrays(cells, ids)
    r, rd = s2_ratio_arrays(ids)
    over = cell_overrides(cells, cells_sha)
    for (iid, item), o in over.items():
        if str(cells[(iid, item)]["is_stochastic"]) != "True":
            continue
        ratio = float(rd[list(ids).index(iid)]) if item == "DM" else float(r[list(ids).index(iid), PN.index(item)])
        o.update({"variance_basis": "unidentified", "provenance_status": "research_scenario_assumption",
                  "source_id": None, "locator": None,
                  "data_fingerprint": stable_hash("dev_case_v1_S2_cell/v1", o["data_fingerprint"], ratio)})
    return UncertaintySpec.from_arrays(
        "DEV_CASE_V1_S2_REFFARM_TN_MM", ids, PN, tm, ts * r, dm, ds * rd, purpose="sensitivity_scenario",
        moment_semantics="target_marginal_moments", is_synthetic=False,
        family_rule=RUN_CONFIG["uncertainty"]["family_rule"],
        cell_defaults={"variance_basis": "observed_incl_sampling_and_lab", "decomposition_id": "none",
                       "decomposition_source": None, "measurement_model_id": MEASUREMENT_MODEL_NASEM},
        theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0), cell_overrides=over,
        notes="FIX_B dev_case_v1 S2 reference-farm world: table means, within-farm SD = r x table SD (corn silage 14 d "
              "true_only ratios; other cells borrow 1/2.1); research_scenario_assumption; not a result")


def s2_problem_cfg(cfg0: dict, problem) -> tuple[dict, dict]:
    """S2 problem: sourced rows stay probabilistic; the other five are imposed as planned (table-value) structural
    rows SH-PLAN-* and kept as diagnostic_only scenario rows.  Nothing is deleted; bounds/tolerances unchanged."""
    cons_yaml = yaml.safe_load((CASE_CFG / "constraints.yaml").read_text(encoding="utf-8"))
    status = {c["id"]: c["status"] for c in cons_yaml["constraints"] if c["class"] == "probabilistic_nutrition"}
    sourced = {cid for cid, st in status.items() if st == "sourced"}
    if sourced != set(S2_EVENT) or set(status) != set(S2_EVENT) | set(S2_PLANNED):
        raise SystemExit(f"S2 rule check failed: sourced probabilistic rows {sorted(sourced)} != declared {S2_EVENT}")
    cc = problem.compiled
    th0 = problem.nominal_theta()
    cfg = copy.deepcopy(cfg0)
    byid = {g["ingredient_id"]: g for g in cfg["ingredients"]}
    new_rows, record = [], {"planned_rows": {}, "rule_check": {"sourced_probabilistic_rows": sorted(sourced),
                                                             "declared_event": list(S2_EVENT), "ok": True}}
    for cid in S2_PLANNED:
        k = list(cc.constraint_ids).index(cid)
        if np.any(np.asarray(cc.v[k]) != 0):
            raise SystemExit(f"S2: planned row {cid} has as-fed terms (not supported)")
        c_nom = np.einsum("ij,ij->i", np.asarray(cc.W[k], float), th0) + np.asarray(cc.w0[k], float)
        name = "plan_" + cid[3:].replace("-", "_")
        is_energy = cid == "PN-NEL-FIXEDDMI"
        if is_energy:
            c_nom = c_nom / E_REF_MCAL_PER_KG
        for i, iid in enumerate(problem.ingredient_ids):
            byid[iid].setdefault("coefficients", {})[name] = {
                "value": float(c_nom[i]), "unit": "1", "status": "research_scenario_assumption",
                "rationale": (f"S2 planned form of {cid}: compiled row content at the nominal (table-value) state, "
                              "canonical units per kg DM" + ("; energy: Mcal per kg DM divided by 1 Mcal/kg"
                                                            if is_energy else ""))}
        orig = next(c for c in cfg["constraints"] if c["constraint_id"] == cid)
        row = copy.deepcopy(orig)
        row["constraint_id"] = "SH-PLAN-" + cid[3:]
        row["name"] = f"S2 planned form of {cid} (table values, decision-time d_hat)"
        row["terms"] = {f"C:{name}:DM": 1.0}
        row["constraint_class"] = "structural_hard"
        row["dm_source"] = "decision_estimate"
        row["claim_scope"] = (f"planning row of S2: {cid} at table values; not a reliability statement "
                              "(its scenario form is reported as a diagnostic)")
        if is_energy:
            b = dict(orig["bound"])
            b["unit"] = "kg/d"
            b["rationale"] = (str(b.get("rationale", "")) + " | S2: written in kg/d of 1-Mcal/kg-equivalent DM "
                              "(the engine gives DM-term rows a mass-rate unit); numerically identical to Mcal/d")
            row["bound"] = b
        orig["constraint_class"] = "diagnostic_only"
        new_rows.append(row)
        record["planned_rows"][cid] = {"structural_id": row["constraint_id"], "coefficient": name,
                                       "unit_note": "kg/d of 1-Mcal/kg-equivalent DM" if is_energy else orig["bound"].get("unit")}
    cfg["constraints"] += new_rows
    cfg["problem_id"] = f"{cfg0['problem_id']}|S2"
    cfg["description"] = (str(cfg0.get("description", "")) + " | FIX_B S2: joint event = sourced rows; research-"
                          "assumption rows planned at table values (structural) + scenario diagnostics.")
    return cfg, record


def s2_consistency(problem_h0, problem_s2, opts) -> dict:
    """S2 must reproduce H0's M0 exactly and every planned margin must equal the H0 nominal margin."""
    m0h = get_method("M0_nominal")(problem_h0, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    m0s = get_method("M0_nominal")(problem_s2, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    dq = float(np.max(np.abs(np.asarray(m0h.decision.q_as_fed) - np.asarray(m0s.decision.q_as_fed))))
    th0, dh = problem_h0.nominal_theta(), problem_h0.dm_estimates()
    evh = evaluate(m0h.decision, th0[None], dh[None], problem_h0.compiled, prices=problem_h0.prices)
    evs = evaluate(m0s.decision, problem_s2.nominal_theta()[None], problem_s2.dm_estimates()[None], problem_s2.compiled,
                   prices=problem_s2.prices)
    smarg = np.atleast_2d(evs.structural_margin)
    diffs = {}
    for cid in S2_PLANNED:
        mh = float(evh.margin[0, list(evh.constraint_ids).index(cid)])
        ms = float(smarg[0, list(evs.structural_ids).index("SH-PLAN-" + cid[3:])])
        diffs[cid] = abs(mh - ms)
    ok = dq < 1e-8 and abs(m0h.objective - m0s.objective) < 1e-9 and max(diffs.values()) < 1e-8
    out = {"M0_q_max_abs_diff": dq, "M0_cost_abs_diff": abs(m0h.objective - m0s.objective),
           "planned_margin_abs_diff_vs_H0_nominal_margin": diffs, "ok": bool(ok)}
    if not ok:
        raise SystemExit(f"S2 consistency check failed: {out}")
    return out


def same_risk_table(runs, alphas) -> list[dict]:
    """Per target alpha: methods selected on validation (met the screen), their cost and test risk; paired cost
    differences only among methods that met the same alpha on validation (red team B #1)."""
    out = []
    for a in alphas:
        rows = []
        for r in runs:
            if run_alpha(r.spec.name) != a:
                continue
            sel = r.selection
            met = sel is not None and sel.status == "selected"
            row = {"label": r.spec.name, "method_id": r.spec.method_id, "validation_screen_met": bool(met),
                   "selected_param": (f"{sel.param_name}={sel.selected_value}" if met else None),
                   "cost_usd_per_head_d": None, "test_joint_rate": None, "test_cp_upper_one_sided_95": None,
                   "test_rate_le_alpha": None, "test_cp_upper_le_alpha": None}
            if met and r.evaluation is not None:
                j = r.evaluation.summary()["joint"]
                k = j["n_violated"] + j["n_unknown"]
                n = j["n_draws"]
                up = one_sided_upper(k, n, 0.95)
                row.update({"cost_usd_per_head_d": _f(r.evaluation.cost), "test_joint_rate": k / n,
                            "test_cp_upper_one_sided_95": up, "test_rate_le_alpha": bool(k / n <= a),
                            "test_cp_upper_le_alpha": bool(up <= a)})
            rows.append(row)
        met_rows = [x for x in rows if x["validation_screen_met"] and x["cost_usd_per_head_d"] is not None]
        m1_met = [x for x in met_rows if x["method_id"] == "M1_safety_margin"]
        ref = min(m1_met, key=lambda x: x["cost_usd_per_head_d"]) if m1_met else None
        cheapest = min(met_rows, key=lambda x: x["cost_usd_per_head_d"]) if met_rows else None
        pairs = []
        if ref is not None:
            for x in met_rows:
                if x is ref:
                    continue
                pairs.append({"method": x["label"], "reference": ref["label"],
                              "cost_minus_reference_usd": x["cost_usd_per_head_d"] - ref["cost_usd_per_head_d"],
                              "test_joint_rate": x["test_joint_rate"], "reference_test_joint_rate": ref["test_joint_rate"]})
        out.append({"target_alpha": a, "n_methods": len(rows), "n_met_on_validation": len(met_rows),
                    "methods": rows, "cheapest_met": None if cheapest is None else cheapest["label"],
                    "reference_simple_margin": None if ref is None else ref["label"],
                    "reference_rule": "cheapest M1 variant that met the screen at this alpha (the simple calibrated "
                                      "safety margin, contract §9 M1); none -> no paired differences",
                    "paired_vs_cheapest_met_M1": pairs,
                    "rule": RUN_CONFIG["S2"]["comparison_rule"]})
    return out


def cp_sup_rdp_block(label: str, problem, cfg, opt, test, opts, mv_current: Optional[dict]) -> tuple[dict, dict]:
    """E5-CP-SUP-RDP-SCALED dev diagnostic: PN-CP-SUP bound with RDP scaled to the case DMI (red team B #5)."""
    inp = yaml.safe_load(FIXB_CP_SUP_INPUTS.read_text(encoding="utf-8"))["values"]
    row = next(c for c in cfg["constraints"] if c["constraint_id"] == "PN-CP-SUP")
    if row["bound"].get("unit") != "kg/d":
        raise SystemExit("PN-CP-SUP bound unit is not kg/d")
    dmi_case = float(next(c for c in cfg["constraints"] if c["constraint_id"] == "SH-DM-PLAN")["bound"]["value"])
    old = float(row["bound"]["value"])
    tab = float(inp["protein_intake_table_g_d"]["value"]) / 1000.0
    if abs(tab - old) > 1e-9 or abs(inp["rdp_table_g_d"]["value"] + inp["rup_table_g_d"]["value"]
                                     - inp["protein_intake_table_g_d"]["value"]) > 1e-9:
        raise SystemExit("fixb_cp_sup_rdp_scaling.yaml does not reproduce the current PN-CP-SUP bound")
    new = (float(inp["rup_table_g_d"]["value"])
           + float(inp["rdp_table_g_d"]["value"]) / float(inp["dmi_table_kg_d"]["value"]) * dmi_case) / 1000.0
    cfg_a = copy.deepcopy(cfg)
    ra = next(c for c in cfg_a["constraints"] if c["constraint_id"] == "PN-CP-SUP")
    ra["bound"] = dict(ra["bound"], value=new, status="research_scenario_assumption",
                       rationale="E5-CP-SUP-RDP-SCALED: RUP_table + RDP_table/DMI_table x DMI_case (FIX_B dev diagnostic)")
    cfg_a["problem_id"] = f"{cfg['problem_id']}|CP-SUP-RDP-scaled"
    pa, _ = build_problem(cfg_a, mode="pilot")
    base = get_method("M0_nominal")(problem, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    alt = get_method("M0_nominal")(pa, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    out = {"label": label, "bound_relative_change": new / old - 1.0, "status_M0_current": str(base.status),
           "status_M0_alt": str(alt.status)}
    res = {"label": label, "bound_kg_d_current": old, "bound_kg_d_rdp_scaled": new, "dmi_case_kg_d": dmi_case}
    if base.has_solution and alt.has_solution:
        def rates(dec, prob):
            ev = evaluate_drawset(dec, test, prob.compiled, prices=prob.prices)
            s = ev.summary()
            j = s["joint"]
            pc = s["per_constraint"].get("PN-CP-SUP") or {}
            return ((j["n_violated"] + j["n_unknown"]) / j["n_draws"],
                    (pc.get("n_violated", 0) / pc["n_defined"]) if pc.get("n_defined") else None)
        jb, cb = rates(base.decision, problem)
        ja, ca = rates(alt.decision, pa)
        out.update({"M0_cost_change_usd_per_head_d": float(alt.objective - base.objective),
                    "M0_cost_relative_change": float(alt.objective / base.objective - 1.0),
                    "M0_test_joint_rate_current": jb, "M0_test_joint_rate_alt": ja,
                    "M0_test_cp_sup_violation_rate_current": cb, "M0_test_cp_sup_violation_rate_alt": ca,
                    "test_stream_id": test.stream_id})
        res["M0_q_alt"] = np.asarray(alt.decision.q_as_fed, float).tolist()
    mva = min_violations(pa, opt, 128, time_limit=60.0)
    out["min_violations_N128_alt"] = {k: v for k, v in mva.items() if k not in ("x",)}
    out["min_violations_N128_current"] = mv_current
    out["note"] = "diagnostic only; the declared PN-CP-SUP bound of this run is unchanged"
    return out, res


# ------------------------------------------------------------------------------------------------
# 8 information value
# ------------------------------------------------------------------------------------------------
def information_value_block(problem, w, fm, streams, extra_decisions, opts, registry: StreamRegistry,
                            tag: str = "H0") -> tuple[dict, dict]:
    cfg = RUN_CONFIG["information_value"]
    pd_ = w.draw(streams, cfg["prior"]["stream"], cfg["prior"]["n_states"], cfg["prior"]["sub_key"])
    registry.add(f"{tag}_information_value_prior", pd_, crn_group="CRN-iv-prior", purpose="information-value prior")
    prior = PriorStates.from_drawset(pd_, consumer=f"K7/FIXB:{tag}:information_value", model=w)
    comp = ObservedComponent(*cfg["component"])
    vals = prior.component_values((comp,))
    edges = quantile_edges(vals, cfg["binning"]["n_bins"], prior.weights)
    binning = SignalBinning((comp,), (edges,), frozen_from=f"weighted quantiles of {pd_.stream_id}")
    st = InformationStructure("PPI_corn_silage_starch_4bins", "perfect_partial", components=(comp,), binning=binning,
                              description="partial perfect information on corn-silage starch, 4 development bins")
    lik = perfect_partial_likelihood(prior, (comp,), binning)
    cond = conditioning_from_likelihood(prior, lik)
    # bin-conditional frontier candidates on a separate development sub-stream (design only)
    dcfg = cfg["library"]["design_stream"]
    dsgn = w.draw(streams, dcfg["stream"], dcfg["n_draws"], dcfg["sub_key"])
    registry.add(f"{tag}_information_value_design", dsgn, crn_group="CRN-iv-design",
                 purpose="information-value design sub-stream (bin-conditional frontier candidates)")
    zdes = binning.bin_index(dsgn.theta[:, list(dsgn.ingredient_ids).index(comp.ingredient_id),
                                        list(dsgn.nutrient_ids).index(comp.component)][:, None])
    per_bin = []
    extra = list(extra_decisions)
    for z in range(binning.n_bins):
        idx = np.flatnonzero(zdes == z)
        sub = DrawSet(dsgn.theta[idx], dsgn.d[idx], dsgn.stream, f"{dsgn.stream_id}#bin{z}", dsgn.model_id,
                      dsgn.model_fingerprint, dsgn.ingredient_ids, dsgn.nutrient_ids, dsgn.is_synthetic)
        entry = {"bin": z, "n_design_scenarios": int(idx.size)}
        if idx.size == 0:
            entry["status"] = "no_design_scenarios"
            per_bin.append(entry)
            continue
        mv = min_violations(problem, sub, int(idx.size), time_limit=cfg["library"]["per_bin_min_violation_time_limit_s"])
        entry["min_violation"] = {k: v for k, v in mv.items() if k != "x"}
        if mv["min_violations_upper"] is not None:
            a_z = alpha_for_count(int(mv["min_violations_upper"]), int(idx.size))
            rz = get_method("M2_joint_chance_saa")(problem, opt_draws=sub, params={"alpha_train": a_z,
                                                                                 "n_scenarios": int(idx.size),
                                                                                 "big_m_mode": "box_quantile"},
                                                   solver_options=opts)
            entry.update({"alpha_train": a_z, "allowed_violations": allowed_violations(a_z, int(idx.size)),
                          "status": str(rz.status), "objective": _f(rz.objective),
                          "mip_gap": _f(rz.mip_gap)})
            if rz.has_solution:
                extra.append((f"bin{z}_frontier", rz.decision, {"method": "M2 frontier on bin design scenarios",
                                                                "bin": z, "design_stream_id": sub.stream_id,
                                                                "alpha_train": a_z}))
        per_bin.append(entry)
    lib = generate_conditional_library(problem, prior, cond, tightening_grid=cfg["library"]["tightening_grid"],
                                       extra_decisions=extra, solver_options=opts)
    risk = compute_risk_table(lib, prior, problem.compiled, prices=problem.prices)
    Iv = risk.indicator()
    pi = np.asarray(prior.weights, float)
    L = np.asarray(lik.L, float)
    rk = pi @ Iv
    Rzk = (pi[:, None] * L).T @ Iv                                     # [Z, K]
    legal = np.asarray(risk.t0_structural_ok, bool)
    r0 = float(rk[legal].min())
    k0 = [lib.candidate_ids[k] for k in np.flatnonzero(legal) if rk[k] <= r0 + 1e-12]
    rT = float(sum(Rzk[z, legal].min() for z in range(Rzk.shape[0])))
    values = {}
    for a in cfg["alphas"]:
        res = compute_information_value(st, prior, risk, float(a),
                                        value_definition=ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE,
                                        include_randomized_reference=True, include_randomization_benchmark=True,
                                        solver_options=opts)
        d = res.to_dict()
        values[str(a)] = {"to_dict": d, "definition_report": res.definition_report(),
                          "randomized_reference": res.value(ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE),
                          "primary_value": res.primary_value, "primary_value_status": res.primary_value_status,
                          "error_model_identification": res.error_model_identification}
    md_cell = [c for c in fm.metadata.cells if c.ingredient_id == comp.ingredient_id and c.component == comp.component][0]
    pub = {"structure_id": st.structure_id, "info_type": st.info_type, "component": comp.label(),
           "scenario_label": cfg["scenario_label"], "prior_stream_id": pd_.stream_id, "n_states": prior.n_states,
           "prior_fingerprint": prior.fingerprint, "observed_cell_variance_basis_in_world": md_cell.variance_basis,
           "observed_cell_provenance": md_cell.provenance_status,
           "n_bins": binning.n_bins, "bin_probabilities": (pi @ L).tolist(), "library_size": lib.size,
           "library_candidate_ids": list(lib.candidate_ids), "library_failures": len(lib.failures),
           "library_failure_statuses": _count([str(x.get("status")) for x in lib.failures]),
           "library_fingerprint": lib.fingerprint, "risk_table_fingerprint": risk.fingerprint,
           "design_stream_id": dsgn.stream_id, "bin_conditional_frontier_candidates": per_bin,
           "candidate_ex_ante_joint_risk": {cid: float(v) for cid, v in zip(lib.candidate_ids, rk)},
           "candidate_cost_usd_per_head_d": {cid: float(v) for cid, v in zip(lib.candidate_ids, lib.costs)},
           "min_attainable_ex_ante_joint_risk": {
               "no_information_deterministic": r0, "no_information_minimisers": k0,
               "with_information_deterministic": rT,
               "note": "within the finite library K and the 4 development bins; a randomised policy cannot go below "
                       "these minima (a mixture's risk is an average); diagnostic, not an information value"},
           "values_by_alpha": values}
    res_ = {"bin_edges_fraction_dm": list(edges), "library_q": {cid: dec.q_as_fed.tolist()
                                                                for cid, dec in zip(lib.candidate_ids, lib.decisions)},
            "library_origins": list(lib.origins), "library_failures": list(lib.failures)}
    return pub, res_


def _count(xs):
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="reduced sizes, in memory, no file written")
    ap.add_argument("--write-check", action="store_true",
                    help="FIX_B: reduced sizes, writes every output under data/restricted_local/_fixb_write_check_NOT_A_RUN/ "
                         "(both public and restricted files, all git-ignored) to exercise the writing code; not a run")
    args = ap.parse_args()
    os.chdir(REPO)
    started = utc_now()
    t_start = time.perf_counter()
    command = " ".join([sys.executable] + sys.argv)
    dry = bool(args.dry_run)
    wcheck = bool(args.write_check)
    if dry and wcheck:
        print("--dry-run and --write-check are exclusive", file=sys.stderr)
        return 3
    # R3F (review round 3, F-4): missing/stale inputs -> short list + exit 2/1 before anything is read or written
    require_dev_case_inputs(REPO, driver="run_dev_case_v1")
    sc = RUN_CONFIG["streams"]
    small = dry or wcheck
    n_opt, n_val, n_test = (sc["opt"], sc["validation"], sc["test"]) if not small else (512, 2000, 1000)
    Ns = sc["N_ladder"] if not small else [128]
    farms = RUN_CONFIG["H1"]["farms"]["farm_indices"] if not small else [0]
    dg = RUN_CONFIG["diagnostics"]
    pre_manifest = code_manifest(REPO)
    lock = check_environment_against_lock(REPO / ENVIRONMENT_LOCK_FILE, current=environment_fingerprint())
    if not dry and lock.get("status") != "matches_lock":
        print(f"environment does not match the lock ({lock.get('status')}); a pilot run is refused", file=sys.stderr)
        return 2

    inputs = check_inputs()
    problem, rep = load_problem(PROBLEM_YAML, mode="pilot")
    cfg0 = yaml.safe_load(PROBLEM_YAML.read_text(encoding="utf-8"))
    lin = load_linearisation(problem)
    cells = read_cells()
    cells_sha = file_sha256(CELLS_CSV)
    spec = build_h0_spec(problem.ingredient_ids, cells, cells_sha)
    fm = build_uncertainty_model(spec, model_id="DEV_CASE_V1_H0_factory_TN_MM")
    w = E.EnergyColumnModel(fm, lin)
    nomchk = wrapped_nominal_check(w, problem)
    streams = RandomStreams(RUN_CONFIG["seed"])
    world = w.draw_world(streams, n_opt=n_opt, n_validation=n_val, n_test=n_test)
    opt, val, test = world["opt"], world["validation"], world["test"]
    assert opt.model_fingerprint == val.model_fingerprint == test.model_fingerprint == w.fingerprint()
    registry = StreamRegistry()
    for ds_ in (opt, val, test):
        registry.add("H0_world", ds_, crn_group="CRN-main", purpose=f"H0 {ds_.stream}")
    req_nel = [c for c in problem.constraints if c.constraint_id == "PN-NEL-FIXEDDMI"][0]
    req_mcal = float(req_nel.bound) + float(lin.constant_mcal_d)
    opts = SolverOptions(time_limit_s=RUN_CONFIG["solver"]["time_limit_s"], mip_rel_gap=RUN_CONFIG["solver"]["mip_rel_gap"])
    price_meta = {"currency": problem.prices.currency, "period": str(cfg0["prices"].get("price_year_or_date")),
                  "basis": "USD per kg as-fed after one decision-time DM conversion of the DM-quoted items (corn, "
                           "corn silage); trading-point quotes, no freight/processing/salt/premix "
                           "(configs/dev_case_v1/prices.yaml, price_id " + str(problem.prices.price_id) + ")"}
    timings: dict[str, float] = {}

    # ---- H0 methods
    t = time.perf_counter()
    alphas = [RUN_CONFIG["alphas"]["primary"]] + list(RUN_CONFIG["alphas"]["curve"])
    runs, sel_tables = run_method_block(problem, fm, lin, opt, val, alphas, Ns, opts)
    timings["h0_methods_s"] = time.perf_counter() - t

    # ---- diagnostics: minimum attainable violation, frontier point, conflict map
    t = time.perf_counter()
    diag: dict[str, Any] = {"min_violation": [], "frontier_points": [], "conflict_map": None}
    frontier_runs = []
    for N in Ns:
        tl = float(dg["min_violation_saa"]["time_limit_s"][str(N)])
        mv = min_violations(problem, opt, N, time_limit=tl)
        diag["min_violation"].append({k: v for k, v in mv.items() if k != "x"})
        ub = mv["min_violations_upper"]
        if ub is None:
            continue
        a_front = alpha_for_count(int(ub), int(N))
        fr = get_method("M2_joint_chance_saa")(problem, opt_draws=opt, params={"alpha_train": a_front, "n_scenarios": N,
                                                                                 "big_m_mode": "box_quantile"},
                                               solver_options=opts)
        chk = None
        if mv["proven"] and ub >= 1:
            below = get_method("M2_joint_chance_saa")(problem, opt_draws=opt,
                                                      params={"alpha_train": alpha_for_count(ub - 1, N), "n_scenarios": N,
                                                              "big_m_mode": "box_quantile"}, solver_options=opts)
            chk = {"alpha_train": alpha_for_count(ub - 1, N), "allowed": allowed_violations(alpha_for_count(ub - 1, N), N),
                   "status": str(below.status), "expected": "proven_infeasible"}
        frontier_runs.append(SP.MethodRun(SP.MethodSpec("M2_joint_chance_saa",
                                                        label=f"H0:DIAG-frontier:M2[N={N},alpha_train=m*/N={ub}/{N}]",
                                                        uses_opt_draws=True), fr, world_fingerprint=opt.model_fingerprint))
        diag["frontier_points"].append({"N": N, "m_star_upper": ub, "m_star_lower": mv["min_violations_lower"],
                                        "alpha_train": a_front, "allowed_violations": allowed_violations(a_front, N),
                                        "status": str(fr.status), "objective_usd_per_head_d": _f(fr.objective),
                                        "mip_gap": _f(fr.mip_gap), "wall_time_s": _f(fr.wall_time_s),
                                        "consistency_check_one_less_violation": chk,
                                        "label": "diagnostic frontier point (not a target alpha, not a method result)"})
    diag["largest_feasible"] = []
    lf_specs = [("M1_safety_margin", {"margin_scale": "relative", "apply_to_dm": True}, "k", 0.0, 0.025, 1e-4,
                 get_method("M1_safety_margin"), opt),
                ("M1_safety_margin", {"margin_scale": "relative", "apply_to_dm": False}, "k", 0.0, 0.025, 1e-4,
                 get_method("M1_safety_margin"), opt),
                ("M3a_box_robust", {}, "k_box", 0.0, 1.0, 1e-3, None, None)]
    lf_boxes: dict = {}
    for mid_, base, prm, lo_, hi_, tol_, fn_, od_ in lf_specs:
        if fn_ is None:
            class _Boxes(dict):
                def __missing__(self, k):
                    self[k] = factory_energy_box(fm, lin, float(k))
                    return self[k]
            lf_boxes = _Boxes()
            fn_ = make_box_solver(solve_box_robust, lf_boxes)
        rec_, best_ = largest_feasible(fn_, problem, base, prm, lo_, hi_, tol_, opts, opt_draws=od_)
        rec_["method_id"] = mid_
        rec_["label"] = "diagnostic: largest feasible conservatism parameter (not a selected method)"
        diag["largest_feasible"].append(rec_)
        if best_ is not None:
            frontier_runs.append(SP.MethodRun(SP.MethodSpec(mid_, params=base, uses_opt_draws=od_ is not None,
                                                            label=f"H0:DIAG-maxfeasible:{mid_}[{prm}="
                                                                  f"{rec_['largest_feasible']:.6g},{base}]"),
                                              best_, world_fingerprint=opt.model_fingerprint))
    SP.evaluate_on_test(frontier_runs, problem, test)
    for fr_run, fp in zip(frontier_runs, diag["frontier_points"]):
        if fr_run.has_ration:
            ev_v = evaluate_drawset(fr_run.result.decision, val, problem.compiled, prices=problem.prices)
            jv = ev_v.summary()["joint"]
            fp["validation_joint_rate"] = (jv["n_violated"] + jv["n_unknown"]) / jv["n_draws"]
            fp["validation_stream_id"] = val.stream_id
    timings["min_violation_s"] = time.perf_counter() - t
    t = time.perf_counter()
    cmap = {"N": 128, "single_rows": [], "pairs": [], "leave_one_out": [], "all_rows": None}
    pids = list(SP.probabilistic_ids(problem))
    m_by_alpha = {str(a): allowed_violations(a, 128) for a in alphas}
    if not small:
        cm = dg["conflict_map"]
        for rid in pids:
            mv = min_violations(problem, opt, 128, rows=[rid], time_limit=cm["single_rows_time_limit_s"])
            cmap["single_rows"].append(_mv_public(mv, m_by_alpha))
        for a_i in range(len(pids)):
            for b_i in range(a_i + 1, len(pids)):
                mv = min_violations(problem, opt, 128, rows=[pids[a_i], pids[b_i]],
                                    time_limit=cm["pairs_time_limit_s"])
                cmap["pairs"].append(_mv_public(mv, m_by_alpha))
        for rid in pids:
            mv = min_violations(problem, opt, 128, rows=[x for x in pids if x != rid],
                                time_limit=cm["leave_one_out_time_limit_s"])
            d = _mv_public(mv, m_by_alpha)
            d["dropped_row"] = rid
            cmap["leave_one_out"].append(d)
    cmap["allowed_violations_by_alpha_N128"] = m_by_alpha
    diag["conflict_map"] = cmap
    timings["conflict_map_s"] = time.perf_counter() - t
    diag["nel_headroom"] = nel_headroom(problem, lin)

    # ---- evaluate every H0 ration on the shared test stream
    SP.evaluate_on_test(runs, problem, test)
    all_h0 = runs + frontier_runs
    t = time.perf_counter()
    diag["nel_linearisation_residual"] = {r.spec.name: linearisation_residual(r.result.decision, test, lin, problem,
                                                                              n_test, req_mcal)
                                          for r in all_h0 if r.has_ration}
    timings["linearisation_residual_s"] = time.perf_counter() - t

    # ---- H1 scenario
    t = time.perf_counter()
    ext = fm.extension("two_layer_farm")
    h1_meta = {"status": ext.metadata.status, "reasons": list(ext.metadata.reasons),
               "truth_variance_basis": ext.metadata.truth_variance_basis, "notes": list(ext.metadata.notes),
               "config_fingerprint": ext.metadata.config_fingerprint}
    h0_rations = {r.spec.name: r.result.decision for r in all_h0 if r.has_ration}
    h1_pub, h1_res, h1_runs = h1_block(ext.model, fm, lin, cfg0, problem, h0_rations, streams, opts, cells, cells_sha,
                                       n_val, n_test, farms, RUN_CONFIG["H1"]["min_violation_time_limit_s"], registry)
    timings["h1_s"] = time.perf_counter() - t

    # ---- information value
    t = time.perf_counter()
    extra = [(f"H0_{r.spec.method_id}_{k}", r.result.decision, {"method": r.spec.name, "source": "H0 run"})
             for k, r in enumerate(all_h0) if r.has_ration]
    iv_pub, iv_res = information_value_block(problem, w, fm, streams, extra, opts, registry, tag="H0")
    timings["information_value_s"] = time.perf_counter() - t

    # ---- FIX_B: composition closure (H0), C1 correlation world, S2 scenario, CP-SUP RDP-scaled diagnostic
    t = time.perf_counter()
    closure_pub, closure_res = {"worlds": [], "impact": []}, {"worlds": []}
    cp, cr = closure_block("H0", test, lin)
    closure_pub["worlds"].append(cp)
    closure_res["worlds"].append(cr)
    m0_h0 = next(r for r in runs if r.spec.method_id == "M0_nominal")
    closure_pub["impact"].append(closure_impact("H0:M0 in H0 world", m0_h0.result.decision, test, problem, lin))
    timings["closure_s"] = time.perf_counter() - t

    t = time.perf_counter()
    spec_c1 = build_c1_spec(problem.ingredient_ids, cells, cells_sha)
    fm_c1 = build_uncertainty_model(spec_c1, model_id="DEV_CASE_V1_C1_factory_TN_MM")
    w_c1 = E.EnergyColumnModel(fm_c1, lin)
    wrapped_nominal_check(w_c1, problem)
    world_c1 = w_c1.draw_world(streams, n_opt=n_opt, n_validation=n_val, n_test=n_test)
    opt_c1, val_c1, test_c1 = world_c1["opt"], world_c1["validation"], world_c1["test"]
    for ds_ in (opt_c1, val_c1, test_c1):
        registry.add("C1_world", ds_, crn_group="CRN-main", purpose=f"C1 {ds_.stream}")
    Ns_c1 = [128]
    runs_c1, sel_c1 = run_method_block(problem, fm_c1, lin, opt_c1, val_c1, alphas, Ns_c1, opts, m1_variants=(True,),
                                       tag="C1", box_tag="dev_case_v1_C1")
    SP.evaluate_on_test(runs_c1, problem, test_c1)
    mv_c1 = min_violations(problem, opt_c1, 128, time_limit=float(dg["min_violation_saa"]["time_limit_s"]["128"]))
    h0_in_c1 = {}
    for r in all_h0:
        if not r.has_ration:
            continue
        ev = evaluate_drawset(r.result.decision, test_c1, problem.compiled, prices=problem.prices)
        j = ev.summary()["joint"]
        k = j["n_violated"] + j["n_unknown"]
        h0_in_c1[r.spec.name] = {"joint_rate": k / j["n_draws"], "cp_upper_one_sided_95": one_sided_upper(k, j["n_draws"]),
                                 "world_shift_scenario": "C1_world (H0 ration scored in the declared correlation world)"}
    cp, cr = closure_block("C1", test_c1, lin)
    closure_pub["worlds"].append(cp)
    closure_res["worlds"].append(cr)
    closure_pub["impact"].append(closure_impact("H0:M0 in C1 world", m0_h0.result.decision, test_c1, problem, lin))
    cm = fm_c1.metadata.correlation
    c1_pub = {"declared": RUN_CONFIG["C1"], "world": model_public_record(fm_c1, w_c1),
              "correlation_metadata": {"latent_lambda_min": _f(cm.latent_lambda_min),
                                       "max_abs_latent_minus_transformed": _f(cm.max_abs_latent_minus_transformed),
                                       "latent_correlation": np.asarray(cm.latent_correlation).tolist(),
                                       "transformed_correlation": np.asarray(cm.transformed_correlation).tolist(),
                                       "labels": [[CS, "NDF"], [CS, "starch"], [CS, "CP"], [CS, "ash"]]},
              "streams": {"opt": opt_c1.stream_id, "validation": val_c1.stream_id, "test": test_c1.stream_id},
              "min_violation_N128": {k: v for k, v in mv_c1.items() if k != "x"},
              "selection_tables": sel_c1, "H0_rations_in_C1_world": h0_in_c1,
              "label": "research_scenario_assumption; declared correlation scenario; not a result"}
    timings["c1_s"] = time.perf_counter() - t

    t = time.perf_counter()
    cfg_s2, s2_rec = s2_problem_cfg(cfg0, problem)
    problem_s2, rep_s2 = build_problem(cfg_s2, mode="pilot")
    s2_cons = s2_consistency(problem, problem_s2, opts)
    spec_s2 = build_s2_spec(problem.ingredient_ids, cells, cells_sha)
    fm_s2 = build_uncertainty_model(spec_s2, model_id="DEV_CASE_V1_S2_factory_TN_MM")
    w_s2 = E.EnergyColumnModel(fm_s2, lin)
    wrapped_nominal_check(w_s2, problem_s2)
    world_s2 = w_s2.draw_world(streams, n_opt=n_opt, n_validation=n_val, n_test=n_test)
    opt_s2, val_s2, test_s2 = world_s2["opt"], world_s2["validation"], world_s2["test"]
    for ds_ in (opt_s2, val_s2, test_s2):
        registry.add("S2_world", ds_, crn_group="CRN-main", purpose=f"S2 {ds_.stream}")
    runs_s2, sel_s2 = run_method_block(problem_s2, fm_s2, lin, opt_s2, val_s2, alphas, Ns, opts, tag="S2",
                                       box_tag="dev_case_v1_S2")
    diag_s2: dict[str, Any] = {"min_violation": [], "frontier_points": []}
    frontier_s2 = []
    for N in Ns:
        mv = min_violations(problem_s2, opt_s2, N, time_limit=float(dg["min_violation_saa"]["time_limit_s"][str(N)]))
        diag_s2["min_violation"].append({k: v for k, v in mv.items() if k != "x"})
        if mv["min_violations_upper"] is None:
            continue
        a_front = alpha_for_count(int(mv["min_violations_upper"]), int(N))
        fr = get_method("M2_joint_chance_saa")(problem_s2, opt_draws=opt_s2,
                                               params={"alpha_train": a_front, "n_scenarios": N,
                                                       "big_m_mode": "box_quantile"}, solver_options=opts)
        frontier_s2.append(SP.MethodRun(SP.MethodSpec("M2_joint_chance_saa",
                                                      label=f"S2:DIAG-frontier:M2[N={N},alpha_train=m*/N="
                                                            f"{mv['min_violations_upper']}/{N}]", uses_opt_draws=True),
                                        fr, world_fingerprint=opt_s2.model_fingerprint))
        diag_s2["frontier_points"].append({"N": N, "m_star_upper": mv["min_violations_upper"],
                                           "m_star_lower": mv["min_violations_lower"], "alpha_train": a_front,
                                           "status": str(fr.status), "objective_usd_per_head_d": _f(fr.objective),
                                           "label": "diagnostic frontier point (alpha_train = m*/N; with m* = 0 this "
                                                    "is the M3c scenario-set ration, EQ-M3c-M2-ALPHA0)"})
    SP.evaluate_on_test(runs_s2, problem_s2, test_s2)
    SP.evaluate_on_test(frontier_s2, problem_s2, test_s2)
    all_s2 = runs_s2 + frontier_s2
    diag_s2["nel_linearisation_residual"] = {r.spec.name: linearisation_residual(r.result.decision, test_s2, lin,
                                                                                 problem_s2, n_test, req_mcal)
                                             for r in all_s2 if r.has_ration}
    s2_same_risk = same_risk_table(runs_s2, alphas)
    extra_s2 = [(f"S2_{r.spec.method_id}_{k}", r.result.decision, {"method": r.spec.name, "source": "S2 run"})
                for k, r in enumerate(all_s2) if r.has_ration]
    iv_s2_pub, iv_s2_res = information_value_block(problem_s2, w_s2, fm_s2, streams, extra_s2, opts, registry, tag="S2")
    cp, cr = closure_block("S2", test_s2, lin)
    closure_pub["worlds"].append(cp)
    closure_res["worlds"].append(cr)
    m0_s2 = next(r for r in runs_s2 if r.spec.method_id == "M0_nominal")
    closure_pub["impact"].append(closure_impact("S2:M0 in S2 world", m0_s2.result.decision, test_s2, problem_s2, lin))
    timings["s2_s"] = time.perf_counter() - t

    t = time.perf_counter()
    mv_h0_128 = next((m for m in diag["min_violation"] if m["N"] == 128), None)
    cps_h0_pub, cps_h0_res = cp_sup_rdp_block("H0", problem, cfg0, opt, test, opts, mv_h0_128)
    mv_s2_128 = next((m for m in diag_s2["min_violation"] if m["N"] == 128), None)
    cps_s2_pub, cps_s2_res = cp_sup_rdp_block("S2", problem_s2, cfg_s2, opt_s2, test_s2, opts, mv_s2_128)
    timings["cp_sup_rdp_s"] = time.perf_counter() - t

    reg_check = registry.check()
    if not reg_check["ok"]:
        print(json.dumps({"stream_collisions": reg_check["collisions"]}, indent=2), file=sys.stderr)
        print("stream collision between analysis blocks; nothing written", file=sys.stderr)
        return 4

    summary_h0 = summary_rows(all_h0, problem, "<run_id>", price_meta, w.fingerprint())
    if dry:
        print(json.dumps({"dry_run": True, "files_written": 0, "timings_s": timings,
                          "h0": [{k: r[k] for k in ("label", "status", "cost_usd_per_head_d", "joint_rate")}
                                 for r in summary_h0],
                          "min_violation": diag["min_violation"], "frontier": diag["frontier_points"],
                          "nel_headroom": diag["nel_headroom"],
                          "h1": [{"farm": p["farm_index"], "attainable_lower": p["attainable_training_joint_risk_lower"],
                                  "methods": [(m["label"], m["status"]) for m in p["methods"]]} for p in h1_pub],
                          "information_value": {"min_risk": iv_pub["min_attainable_ex_ante_joint_risk"],
                                                "status": {a: v["primary_value_status"]
                                                           for a, v in iv_pub["values_by_alpha"].items()}},
                          "fixb": {"stream_registry_ok": reg_check["ok"],
                                   "closure": [(c["world"], {k: (v["share_sum_gt_100pct"], v["share_rom_lt_0"])
                                                             for k, v in c["per_ingredient"].items()
                                                             if v["share_sum_gt_100pct"]}) for c in closure_pub["worlds"]],
                                   "closure_impact": closure_pub["impact"],
                                   "c1_min_violation": c1_pub["min_violation_N128"],
                                   "c1_methods": [(r.spec.name, r.status) for r in runs_c1],
                                   "s2_consistency": s2_cons, "s2_min_violation": diag_s2["min_violation"],
                                   "s2_methods": [{k: r[k] for k in ("label", "status", "selected_param",
                                                                     "cost_usd_per_head_d", "joint_rate")}
                                                  for r in summary_rows(all_s2, problem_s2, "<run_id>", price_meta,
                                                                        w_s2.fingerprint())],
                                   "s2_iv": {a: v["primary_value_status"] for a, v in iv_s2_pub["values_by_alpha"].items()},
                                   "cp_sup_rdp": [cps_h0_pub, cps_s2_pub],
                                   "residual_groups_H0_M0": diag["nel_linearisation_residual"].get("H0:M0", {}).get(
                                       "grouped_fixb")}},
                         indent=2, default=_json_default))
        return 0

    # ---- write outputs
    run_id = make_run_id("pilot", started, ("dev_case_v1", RUN_CONFIG["seed"], pre_manifest["manifest_sha256"]))
    pub = Path("results") / "pilot" / run_id
    res = Path("data") / "restricted_local" / "pilot" / run_id
    if wcheck:                                     # everything under a git-ignored scratch folder; not a run
        base = Path("data") / "restricted_local" / "_fixb_write_check_NOT_A_RUN"
        run_id = "WRITECHECK-" + run_id
        pub, res = base / "pub" / run_id, base / "res" / run_id
    if pub.exists() or res.exists():
        print(f"{pub} exists; refusing to overwrite", file=sys.stderr)
        return 3
    pub.mkdir(parents=True)
    res.mkdir(parents=True)
    out = Out(pub, res)
    meta = constraint_meta()
    for row in summary_h0:
        row["run_id"] = run_id
    out.text("pub", "NOT_FOR_MANUSCRIPT.md",
             f"# {run_id}\n\nDevelopment run of dev_case_v1 (run_type pilot = contract T6 small pilot; not official, "
             "protocol not frozen). No number in this directory may enter a formal result table, the manuscript, an "
             "abstract, a figure or a claim ledger. No animal outcome is claimed; 'reliability' is the satisfaction of "
             "the declared model constraints under the declared distribution. Restricted details (NASEM values) are "
             f"in data/restricted_local/pilot/{run_id}/. Report: reports/dev_case_v1_report.md.\n\n"
             "FIX_B: files s2_* belong to the declared scenario S2 (reference farm, joint event over the rows with "
             "sourced bounds, other rows planned at table values): a method-calibration development scenario on a "
             "narrowed specification, not comparable with H0 and not a result; c1_* is the declared correlation "
             "scenario C1.\n")
    out.json("pub", "run_config.json", {"run_config": RUN_CONFIG, "run_config_sha256": stable_hash(RUN_CONFIG),
                                        "inputs": inputs, "validator": {"ok": rep.ok, "n_pending": rep.n_pending,
                                                                        "n_assumptions": rep.n_assumptions,
                                                                        "n_synthetic_values": rep.n_synthetic_values,
                                                                        "n_warnings": len(rep.warnings)}})
    out.json("pub", "uncertainty_model.json", {"H0_world": model_public_record(fm, w), "nominal_check": nomchk,
                                               "streams": {"opt": opt.stream_id, "validation": val.stream_id,
                                                           "test": test.stream_id},
                                               "draw_fingerprints": {"opt": opt.fingerprint, "validation": val.fingerprint,
                                                                     "test": test.fingerprint},
                                               "H1_extension": h1_meta})
    out.csv("pub", "method_summary.csv", summary_h0, SUMMARY_COLUMNS)
    long_rows = []
    for r in all_h0:
        long_rows += SP.long_table([r], run_id=run_id, meta={
            "run_type": "pilot", "scenario_id": "dev_case_v1__H0__INV-dev_case_v1__DEV-PRICE-USMW-2026W38",
            "animal_profile_id": problem.animal.profile_id if problem.animal else None,
            "inventory_id": "dev_case_v1 (8 ingredients, stock unlimited)", "price_id": problem.prices.price_id,
            "target_alpha": run_alpha(r.spec.name), "cost_unit": "USD/head/d", "independent_empirical_n": 0,
            "is_synthetic": False, "provenance_id": "configs/dev_case_v1 + NASEM2021 T19-1/19-3 (restricted)",
            "test_stream_id": test.stream_id})
    out.csv("pub", "evaluation_long.csv", redact_long(long_rows), list(SP.LONG_COLUMNS))
    out.csv("res", "evaluation_long_full.csv", long_rows, list(SP.LONG_COLUMNS))
    out.csv("pub", "rations_q.csv", ration_rows(all_h0, problem, run_id, restricted=False),
            ["run_id", "label", "status", "ingredient_id", "q_as_fed_kg_per_head_d"])
    out.csv("res", "rations_full.csv", ration_rows(all_h0, problem, run_id, restricted=True),
            ["run_id", "label", "status", "ingredient_id", "q_as_fed_kg_per_head_d", "d_hat", "planned_dm_kg_per_head_d",
             "planned_dm_share", "cost_usd_per_head_d"])
    out.csv("pub", "constraint_residuals.csv", residual_rows(all_h0, problem, run_id, meta, restricted=False),
            RESIDUAL_COLUMNS)
    out.csv("res", "constraint_residuals_full.csv", residual_rows(all_h0, problem, run_id, meta, restricted=True),
            RESIDUAL_COLUMNS)
    out.csv("pub", "nutrient_profile_public.csv", profile_rows(all_h0, problem, test, run_id, lin, restricted=False),
            list(SP.PROFILE_COLUMNS))
    out.csv("res", "nutrient_profile_full.csv", profile_rows(all_h0, problem, test, run_id, lin, restricted=True),
            list(SP.PROFILE_COLUMNS))
    out.json("pub", "selection_tables.json", sel_tables)
    out.json("pub", "diagnostics.json", diag)
    out.json("pub", "h1_scenario.json", {"declared": RUN_CONFIG["H1"], "extension_metadata": h1_meta, "farms": h1_pub,
                                        "label": "research_scenario_assumption; extension scenario; not a result; "
                                                 "does not replace H0"})
    out.json("res", "h1_farm_details.json", h1_res)
    out.json("pub", "information_value.json", iv_pub)
    out.json("res", "information_value_restricted.json", iv_res)
    out.json("pub", "solve_results_redacted.json", [solve_record(r.result, restricted=False) for r in all_h0
                                                    if r.result is not None])
    out.json("res", "solve_results_full.json", [solve_record(r.result, restricted=True) for r in all_h0
                                                if r.result is not None])
    moments = moment_rows(fm)
    out.csv("res", "factory_moment_diagnostics.csv", moments, list(moments[0]))
    out.json("res", "uncertainty_model_metadata_full.json", fm.metadata.to_dict(include_values=True))
    out.json("res", "m3_boxes.json", {str(k): {"theta_lo": b.theta_lo, "theta_hi": b.theta_hi, "d_lo": b.d_lo,
                                               "d_hi": b.d_hi, "nutrient_ids": list(b.nutrient_ids),
                                               "fingerprint": b.fingerprint()}
                                      for k, b in ((k, factory_energy_box(fm, lin, k))
                                                   for k in RUN_CONFIG["methods"]["M3a"]["k_grid"])})
    # ---- FIX_B outputs
    out.json("pub", "stream_registry.json", reg_check)
    out.json("pub", "composition_closure.json", closure_pub)
    out.json("res", "composition_closure_full.json", closure_res)
    out.json("pub", "cp_sup_rdp_scaled.json", {"declared": RUN_CONFIG["diagnostics_fixb"]["cp_sup_rdp_scaled"],
                                               "results": [cps_h0_pub, cps_s2_pub]})
    out.json("res", "cp_sup_rdp_scaled_restricted.json", [cps_h0_res, cps_s2_res])
    out.json("pub", "c1_scenario.json", c1_pub)
    summary_c1 = summary_rows(runs_c1, problem, run_id, price_meta, w_c1.fingerprint())
    out.csv("pub", "c1_method_summary.csv", summary_c1, SUMMARY_COLUMNS)
    out.csv("res", "c1_rations_full.csv", ration_rows(runs_c1, problem, run_id, restricted=True),
            ["run_id", "label", "status", "ingredient_id", "q_as_fed_kg_per_head_d", "d_hat", "planned_dm_kg_per_head_d",
             "planned_dm_share", "cost_usd_per_head_d"])
    out.json("res", "c1_uncertainty_model_metadata_full.json", fm_c1.metadata.to_dict(include_values=True))
    meta_s2 = constraint_meta_s2(meta, s2_rec)
    summary_s2 = summary_rows(all_s2, problem_s2, run_id, price_meta, w_s2.fingerprint())
    out.json("pub", "s2_scenario.json", {
        "declared": RUN_CONFIG["S2"], "problem_record": s2_rec, "consistency_with_H0": s2_cons,
        "validator": {"ok": rep_s2.ok, "n_pending": rep_s2.n_pending, "n_assumptions": rep_s2.n_assumptions,
                      "n_synthetic_values": rep_s2.n_synthetic_values, "n_warnings": len(rep_s2.warnings)},
        "world": model_public_record(fm_s2, w_s2),
        "streams": {"opt": opt_s2.stream_id, "validation": val_s2.stream_id, "test": test_s2.stream_id},
        "draw_fingerprints": {"opt": opt_s2.fingerprint, "validation": val_s2.fingerprint, "test": test_s2.fingerprint},
        "diagnostics": diag_s2, "same_risk_comparison": s2_same_risk,
        "label": "S2 is a method-calibration development scenario on a narrowed specification; not a result; not "
                 "comparable with H0; no animal outcome"})
    out.csv("pub", "s2_method_summary.csv", summary_s2, SUMMARY_COLUMNS)
    out.json("pub", "s2_selection_tables.json", sel_s2)
    out.csv("pub", "s2_rations_q.csv", ration_rows(all_s2, problem_s2, run_id, restricted=False),
            ["run_id", "label", "status", "ingredient_id", "q_as_fed_kg_per_head_d"])
    out.csv("res", "s2_rations_full.csv", ration_rows(all_s2, problem_s2, run_id, restricted=True),
            ["run_id", "label", "status", "ingredient_id", "q_as_fed_kg_per_head_d", "d_hat", "planned_dm_kg_per_head_d",
             "planned_dm_share", "cost_usd_per_head_d"])
    out.csv("pub", "s2_constraint_residuals.csv", residual_rows(all_s2, problem_s2, run_id, meta_s2, restricted=False),
            RESIDUAL_COLUMNS)
    out.csv("res", "s2_constraint_residuals_full.csv", residual_rows(all_s2, problem_s2, run_id, meta_s2, restricted=True),
            RESIDUAL_COLUMNS)
    out.csv("pub", "s2_nutrient_profile_public.csv", profile_rows(all_s2, problem_s2, test_s2, run_id, lin,
                                                                  restricted=False), list(SP.PROFILE_COLUMNS))
    long_s2 = []
    for r in all_s2:
        long_s2 += SP.long_table([r], run_id=run_id, meta={
            "run_type": "pilot", "scenario_id": "dev_case_v1__S2_reference_farm_sourced_event__INV-dev_case_v1__"
                                               "DEV-PRICE-USMW-2026W38",
            "animal_profile_id": problem_s2.animal.profile_id if problem_s2.animal else None,
            "inventory_id": "dev_case_v1 (8 ingredients, stock unlimited)", "price_id": problem_s2.prices.price_id,
            "target_alpha": run_alpha(r.spec.name), "cost_unit": "USD/head/d", "independent_empirical_n": 0,
            "is_synthetic": False, "provenance_id": "configs/dev_case_v1 + NASEM2021 (restricted); S2 declared scenario",
            "test_stream_id": test_s2.stream_id})
    out.csv("pub", "s2_evaluation_long.csv", redact_long(long_s2), list(SP.LONG_COLUMNS))
    out.csv("res", "s2_evaluation_long_full.csv", long_s2, list(SP.LONG_COLUMNS))
    out.json("pub", "s2_information_value.json", iv_s2_pub)
    out.json("res", "s2_information_value_restricted.json", iv_s2_res)
    out.text("res", "s2_problem.yaml", "# RESTRICTED (NASEM values). FIX_B S2 problem built in memory from the dev_case_v1 "
             "problem by run_dev_case_v1.s2_problem_cfg. Not a result.\n"
             + yaml.safe_dump(cfg_s2, allow_unicode=True, sort_keys=False))
    out.json("res", "s2_uncertainty_model_metadata_full.json", fm_s2.metadata.to_dict(include_values=True))
    out.json("res", "s2_solve_results_full.json", [solve_record(r.result, restricted=True) for r in all_s2
                                                   if r.result is not None])
    out.json("pub", "s2_solve_results_redacted.json", [solve_record(r.result, restricted=False) for r in all_s2
                                                       if r.result is not None])
    post_manifest = code_manifest(REPO)
    rec = build_run_record(
        run_type="pilot", command=command, repo_root=REPO, started_at=started, completed_at=utc_now(), exit_status=0,
        rng_streams={"opt": opt.stream_id, "validation": val.stream_id, "test": test.stream_id,
                     "information_value_prior": iv_pub["prior_stream_id"],
                     "information_value_design": iv_pub["design_stream_id"],
                     "H1_farm_mean": "root=%d/farm_mean/(2, f) for f in %s" % (RUN_CONFIG["seed"], farms),
                     "H1_farm_worlds": "root=%d/{opt,validation,test}/%d+f for f in %s (FIX_B offset)"
                                       % (RUN_CONFIG["seed"], H1_FARM_SUBKEY_OFFSET, farms),
                     "C1_world": "same stream ids as H0 (CRN-main)", "S2_world": "same stream ids as H0 (CRN-main)",
                     "S2_information_value": "same stream ids as H0 information value (CRN-iv-prior/-design)",
                     "registry": "stream_registry.json (collision check ok = %s)" % reg_check["ok"]},
        solver_version=solver_version_string(), tolerances=opts.to_dict(),
        config_paths=[str(p) for p in CONFIG_PATHS],
        data_paths=[str(p) for p in (PROBLEM_YAML, CELLS_CSV, ENERGY_JSON, BUILD_REPORT, BUILD_SCRIPT, CORE_CSV,
                                     FIXB_CP_SUP_INPUTS)],
        protocol_path=str(PROTOCOL), output_paths=[str(p) for p in out.files], is_synthetic=False, run_id=run_id,
        # R3F: builder code, restricted constants file and build inputs (restricted ones by path + sha256 only) enter
        # the run identity
        build_identity=dev_case_build_identity(REPO),
        extra={"purpose": "dev_case_v1 development run (pilot); NOT a formal result; no animal outcome claimed",
               "run_config_sha256": stable_hash(RUN_CONFIG), "inputs": inputs,
               "world_fingerprint": w.fingerprint(), "base_model_fingerprint": fm.fingerprint(),
               "spec_fingerprint": spec.fingerprint(), "energy_linearisation_fingerprint": lin.fingerprint(),
               "code_manifest_at_start": pre_manifest["manifest_sha256"],
               "code_manifest_unchanged_during_run": pre_manifest["manifest_sha256"] == post_manifest["manifest_sha256"],
               "n_opt": opt.n_draws, "n_validation": val.n_draws, "n_test": test.n_draws,
               "method_summary": [{k: r[k] for k in ("label", "status", "cost_usd_per_head_d", "joint_rate")}
                                  for r in summary_h0],
               "timings_s": timings, "elapsed_s": time.perf_counter() - t_start,
               "fixb": {"task": "FIX_B (review round 2, red team B)", "supersedes": RUN_CONFIG["fixb"]["supersedes_run"],
                        "stream_registry_ok": reg_check["ok"], "n_draw_calls": reg_check["n_draw_calls"],
                        "C1_world_fingerprint": w_c1.fingerprint(), "S2_world_fingerprint": w_s2.fingerprint(),
                        "S2_problem_id": problem_s2.problem_id, "S2_consistency_ok": s2_cons["ok"],
                        "S2_method_summary": [{k: r[k] for k in ("label", "status", "cost_usd_per_head_d", "joint_rate")}
                                              for r in summary_s2],
                        "C1_method_summary": [{k: r[k] for k in ("label", "status", "cost_usd_per_head_d", "joint_rate")}
                                              for r in summary_c1]},
               "restricted_dir": str(res), "host_note": "local Mac; small development run (minutes)"})
    write_run_record(rec, pub / "run_record.json")
    print(json.dumps({"run_id": run_id, "public_dir": str(pub), "restricted_dir": str(res), "timings_s": timings,
                      "h0": [{k: r[k] for k in ("label", "status", "cost_usd_per_head_d", "joint_rate")}
                             for r in summary_h0],
                      "min_violation": diag["min_violation"], "frontier": diag["frontier_points"],
                      "nel_headroom": diag["nel_headroom"],
                      "h1": [{"farm": p["farm_index"], "attainable_lower": p["attainable_training_joint_risk_lower"]}
                             for p in h1_pub],
                      "information_value_min_risk": iv_pub["min_attainable_ex_ante_joint_risk"],
                      "code_manifest_unchanged_during_run": rec["extra"]["code_manifest_unchanged_during_run"]},
                     indent=2, default=_json_default))
    return 0


def _mv_public(mv: dict, m_by_alpha: dict) -> dict:
    d = {k: v for k, v in mv.items() if k != "x"}
    d["verdict_by_alpha"] = {a: conflict_verdict(mv, m) for a, m in m_by_alpha.items()}
    return d


if __name__ == "__main__":
    raise SystemExit(main())
