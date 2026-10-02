#!/usr/bin/env python3
"""2 x 2 (+ MAIN9) endpoint ablation of dev_case_v1 (review round 3, instruction C.4-C.8; R3C; FIX3_BC) -- DEVELOPMENT,
NOT OFFICIAL.

Why
---
The development scenario S2 changed the uncertainty (narrowed SD) and the risk event (11-row joint chance constraint ->
6 probabilistic rows + 5 rows planned at table values) at the same time, so H0 and S2 cannot be compared.  This driver
separates the two changes with the smallest design the review asks for:

==================  ======================================  ==============================================
                    objective FULL11                        objective PART6P5
                    (11-row joint chance constraint)        (S2 declared 6 rows chance + 5 nominal planned rows)
==================  ======================================  ==============================================
SD-H0 (table SD)    ``SDH0_FULL11``  (= H0 of the pilot)    ``SDH0_PART6P5`` (new)
SD-S2 (narrowed)    ``SDS2_FULL11``  (new)                  ``SDS2_PART6P5`` (= S2 of the pilot)
==================  ======================================  ==============================================

Round-3 red team FIX3_BC (C-2): a third training arm ``MAIN9`` (declared before any ablation run): the 9 main
reference rows of ``configs/dev_case_v1/reference_constraints.csv`` in the joint chance constraint (energy row trained
on the linear fixed-DMI row), the two labelled research-assumption rows ``PN-CP-HI`` / ``PN-EE-HI`` planned at table
values and ``d_hat`` (``SH-PLAN-*``) and kept as diagnostic scenario rows -- cells ``SDH0_MAIN9`` and ``SDS2_MAIN9``.
None of FULL11 / PART6P5 trains on the adjudicated main reference: FULL11 also trains on the research-assumption row
PN-CP-HI (19 % set by the project) and PART6P5 plans three sourced main-reference rows at table values.  PART6P5 is
"the S2 declared six + five planned rows"; the six are ``sourced`` only in constraints.yaml's older value-block
vocabulary, not in the adjudication of reference_constraints.csv.  Every conclusion is reported next to
``h0_eleven`` (and the main reference + CP-HI rows) so the adjudication's effect stays visible.

Every cell trains the declared methods M0 / M1 / M2 / M3a / M3b of ``experiments/E0_verification/run_dev_case_v1.py``
(``RUN_CONFIG["methods"]``; no new method, no new grid) on its own objective and world, selects parameters on its own
validation stream at alpha = 0.05 (core) and 0.10 / 0.01 (supplementary), and records its training objective and random
generation.  **Every ration of every cell is then scored by the one reference evaluator**
(``ration_reliability.evaluation.reference.evaluate_reference``) under one physical definition
(``configs/dev_case_v1/reference_constraints.csv``; ``docs/reference_problem_v1.md``) on the test stream of **both**
SD worlds (common root and stream ids; different parent distributions, so the state bytes differ -- recorded).
Moving along the SD axis changes the declared uncertainty (the problem), not the method; moving along the objective
axis changes the training event; a better 6-row rate is never reported as a better 11-row or main-reference rate.

Rules kept (instruction C.8): a cell or method without a solution keeps its status, bounds, gap and time limit (never
cost 0); the minimum-violation counts are about the N training scenarios only; a declared grid or candidate set
without a solution is not a statement about the whole decision space; costs are interpreted only inside a comparable
set (same evaluation world, same alpha, main reference event met) and its coverage is reported.

Modes
-----
``--dry-run``   build the problems, the uncertainty specifications and factory models, the reference specification and the
               checks (preflight, SD-ratio registry, reserved streams, spec fingerprint); print the plan and the compute
               estimate.  No draw, no solve, nothing written -- **measured** (FIX3_BC, C-4): the solve, draw and write
               entry points are wrapped and counted during the dry run and the counts are printed.
``--tiny``      pipeline check only (N = 16, opt 16, validation 500, test 500): every step runs at toy size; output
               labelled ``debug`` under ``data/restricted_local/debug/`` (never ``results/``); not a result.
``--full --authorised-compute``  the declared ablation.  Needs compute the user has authorised (Mac = delivery host;
               round 3 does NOT run it); without ``--authorised-compute`` it stops before reading anything; FIX3_BC (C-4):
               it also needs an authorisation file written by the user (default
               ``data/restricted_local/compute_authorisation.json``) that lists this host.  Output:
               public tables ``results/pilot/<run_id>/`` and restricted tables ``data/restricted_local/pilot/<run_id>/``;
               labelled ``development`` only if (FIX3_BC, C-3) the code/config manifest is complete and unchanged from
               import to end, the specification fingerprint equals the frozen pin (batch 4: the reference problem v2
               pin ``experiments/E1_cost_reliability/reference_problem_v2_freeze.json``; the v1 pin file is kept as
               history), that pin is tracked by git, HEAD is
               the commit that last changed the pin, and the work tree is clean over the manifest scope, the
               specification files and the pin; otherwise the whole output goes to ``data/restricted_local/debug/``
               labelled ``debug``.
``--print-spec-fingerprint``  sha256 of the reference-problem specification files (for re-pinning).
``--official-plan``  official run v2 (official-run plan batch 3; data only): build reference problem v2 (premise
               planning row), its arms, the v2 reference specification and one world per declared SD point (registry
               check of all five points), and print the official cell / job matrix (``official_v2.OFFICIAL_CONFIG``,
               ``official_v2.official_jobs``: root indices only) and the compute estimate.  No draw, no solve, nothing
               written -- measured like ``--dry-run``.  No reserved root is opened.
``--official --authorised-compute --root-k K [K ...] --workers N``  official run v2 (batch 4; plan §2 "B4", F3):
               authorisation file (default ``data/restricted_local/compute_authorisation_official.json``, scope
               ``run_official_v2``, this host) -> root indices (0, 1, 2; a seed or development root is refused) -> the
               freeze gate ``official_v2.require_protocol_frozen`` (measured: 0 draws) -> preflight -> environment lock
               -> one process per job (``official_v2.run_jobs``; BLAS single-threaded), each re-running the gate and
               only then opening its reserved root with the token -> ``--official-merge``.  Output: public tables
               ``results/official/<run_id>/`` (root aliases, per root), restricted mirror
               ``data/restricted_local/official/<run_id>/``.  Every refusal before the jobs exits 2 with nothing drawn;
               after the draws only the end-of-run identity check can fail (``official_invalidated``, disclosed).
``--official --dry-run``  the freeze gate only (measured 0 draws / solves / files); exit 2 until the protocol is frozen.
``--official-rehearsal``  the root-0 job matrix (9 cells + diagnostics) through the same runner and merge on the
               DEVELOPMENT root at tiny sizes; output ``debug`` under ``data/restricted_local/debug/<run_id>/`` only.
``--official-merge RUN_ID``  verify and merge the jobs (refuses missing / duplicate jobs; tables per root, no pooled n).
``--write-freeze-pin NOTE`` / ``--write-protocol-freeze NOTE``  lead-invoked freeze files (the one freeze commit).

Nothing here is a formal result; nothing may enter the manuscript; no animal outcome is claimed.  "Reliability" is the
satisfaction of the declared model constraints under the declared distribution.

Usage::

    cd <project root>
    PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 experiments/E1_cost_reliability/run_endpoint_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
E0 = REPO / "experiments" / "E0_verification"
E1 = Path(__file__).resolve().parent            # official_v2 (batch 3) lives next to this driver
for _p in (REPO / "src", E0, E1):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: FIX3_BC (round-3 red team C-3): sha256 of the Python sources this driver imports, taken with the standard library
#: *before* any project module is imported; compared with the start-of-run code manifest, so a file changed between
#: import and manifest cannot pass unnoticed.
IMPORT_TIME_SOURCE_DIRS = ("src/ration_reliability", "experiments/E0_verification", "experiments/E1_cost_reliability")


def _source_hashes(root: Path = REPO, dirs: tuple[str, ...] = IMPORT_TIME_SOURCE_DIRS) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in dirs:
        for dp, dn, fns in os.walk(root / d):
            dn[:] = [x for x in dn if x != "__pycache__"]
            for fn in fns:
                if fn.endswith(".py"):
                    fp = Path(dp) / fn
                    out[fp.relative_to(root).as_posix()] = hashlib.sha256(fp.read_bytes()).hexdigest()
    return out


IMPORT_TIME_SOURCE_SHA256 = _source_hashes()

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import official_v2 as OV2  # noqa: E402  (official-run plan batch 3: v2 cfg0, SD points, OFFICIAL_CONFIG; data only)
import run_dev_case_v1 as DRV  # noqa: E402  (the dev-case builders and method block; nothing new)
import smoke_pipeline as SP  # noqa: E402
from ration_reliability.build.dev_case import DGC_INGREDIENT, dev_case_build_identity  # noqa: E402
from ration_reliability.build.preflight import DRIVER_PROFILES, Requirement, require_dev_case_inputs  # noqa: E402
from ration_reliability.datamodel import SolverOptions  # noqa: E402
from ration_reliability.evaluation import evaluate  # noqa: E402
from ration_reliability.evaluation.reference import (  # noqa: E402
    H0_ELEVEN,
    MAIN_EVENT_PLAN_DOMAIN,
    MAIN_EVENT_T51_IGNORED,
    EventDefinition,
    ReferenceSpec,
    default_events,
    evaluate_reference,
    load_reference_constraints,
)
from ration_reliability.evaluation.stats import one_sided_upper  # noqa: E402
from ration_reliability.hashing import file_sha256, stable_hash  # noqa: E402
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
from ration_reliability.optimization import get_method  # noqa: E402
from ration_reliability.optimization.chance_saa import allowed_violations  # noqa: E402
from ration_reliability.optimization.highs import solver_version_string  # noqa: E402
from ration_reliability.uncertainty import RandomStreams, build_uncertainty_model  # noqa: E402
from ration_reliability.uncertainty import streams as ST  # noqa: E402

# ------------------------------------------------------------------------------------------------
# paths and declared configuration (fixed before any run; copied into the run record)
# ------------------------------------------------------------------------------------------------
REFERENCE_CSV = Path("configs") / "dev_case_v1" / "reference_constraints.csv"
STREAMS_POLICY = Path("configs") / "streams_policy.yaml"
SD_REGISTRY = Path("reports") / "sd_scaling_sources.csv"
REFERENCE_DOC = Path("docs") / "reference_problem_v1.md"
DEBUG_ROOT = Path("data") / "restricted_local" / "debug"
#: FIX3_BC (round-3 red team C-3): the frozen specification fingerprint lives in a pin file inside the code-manifest scope
#: (experiments/), not in an untracked document; it is anchored by git (tracked, HEAD = the commit that last changed it,
#: clean work tree) before a run may be labelled ``development``.  Official-run plan batch 4: the pin is the reference
#: problem v2 pin (``official_v2.FREEZE_PIN_V2``); the v1 pin file stays untouched as history (``FREEZE_PIN_V1``).
FREEZE_PIN = OV2.FREEZE_PIN_V2
FREEZE_PIN_V1 = OV2.FREEZE_PIN_V1_HISTORY
FREEZE_PIN_SCHEMA = OV2.FREEZE_PIN_SCHEMA
#: FIX3_BC (C-4): the user's authorisation for ``--full`` (restricted, written by the user, never by an agent)
AUTHORISATION_FILE = Path("data") / "restricted_local" / "compute_authorisation.json"
#: batch 4: the user's authorisation for ``--official`` (restricted; scope must contain ``run_official_v2``; D-536)
OFFICIAL_AUTHORISATION_FILE = OV2.OFFICIAL_AUTHORISATION_FILE

#: files whose bytes define the reference problem, this ablation and the official path (docs/reference_problem_v1.md §9
#: freeze; batch 4 adds official_v2.py, reference_constraints_v2.csv, configs/protocol.yaml, docs/reference_problem_v2.md,
#: optimization/safety_margin.py and uncertainty/streams.py).  Defined in ``official_v2`` (its freeze gate needs them).
SPEC_FILES = OV2.SPEC_FILES
#: restricted build record (pins every restricted input by its recorded sha256); entered by hash only
SPEC_RESTRICTED_FILES = OV2.SPEC_RESTRICTED_FILES

EXTRA_REQUIREMENTS = tuple(DRIVER_PROFILES["run_dev_case_v1"]) + (
    Requirement(REFERENCE_CSV.as_posix(), "driver_tracked_config", False),
    Requirement(STREAMS_POLICY.as_posix(), "driver_tracked_config", False),
    Requirement(SD_REGISTRY.as_posix(), "driver_tracked_config", False),
)

ABLATION_CONFIG: dict[str, Any] = {
    "schema": "ration_reliability.endpoint_ablation_config/0.2",
    "case_id": "dev_case_v1",
    "task": ("R3C_reference_ablation (review round 3, instruction C), 2026-09-25; FIX3_BC (round-3 red team C-1..C-4): "
             "domain-conditioned main event and comparable sets, MAIN9 arm, git-anchored freeze pin, measured dry run"),
    "run_role": ("development ablation of one development case (one cow, one inventory, one price); not official; "
                 "protocol not frozen; the reference problem is a development freeze (docs/reference_problem_v1.md)"),
    "seed": 1103,
    "seed_class": ("development_material: configs/streams_policy.yaml development_material.whole_roots (root 1103); the "
                   "reserved formal roots are refused by this driver"),
    "streams": {"opt": 512, "N_ladder": [128, 512], "N_nesting": "N = 128 uses the first 128 opt draws",
                "validation": 20000, "test": 10000,
                "sizes_source": "same as experiments/E0_verification/run_dev_case_v1.py RUN_CONFIG['streams']"},
    "tiny": {"opt": 16, "N_ladder": [16], "validation": 500, "test": 500,
             "role": "pipeline check only (instruction hard rule: N <= 16, test <= 500); output labelled debug"},
    "crn": ("every SD world draws root/opt, root/validation, root/test of one RandomStreams(seed): equal stream ids, "
            "different parent distributions -> different state bytes (instruction C.6); group CRN-ablation"),
    "alphas": {"primary": 0.05, "supplementary": [0.10, 0.01],
               "source": "configs/methods.yaml risk_levels (research design choice, not an animal safety threshold)"},
    "sd_arms": {
        "SD-H0": {"spec_builder": "run_dev_case_v1.build_h0_spec",
                  "definition": "NASEM Table 19-1 observed total SD used as the batch SD (H0; strong assumption)",
                  "ratio_column": "ratio_SD_H0", "data_layer": "docs/uncertainty_data_layers.md §2 H0"},
        "SD-S2": {"spec_builder": "run_dev_case_v1.build_s2_spec",
                  "definition": ("declared narrowed SD point SD-S2 (corn silage DM 1/4.4, NDF 1/3.6, starch 1/5.0; the "
                                 "other 45 stochastic cells borrow 1/2.1); unidentified / research_scenario_assumption; "
                                 "selected by attainability in FIX_B screening (docs/uncertainty_data_layers.md §4.4)"),
                  "ratio_column": "ratio_SD_S2", "data_layer": "docs/uncertainty_data_layers.md §4-§5"}},
    "objective_arms": {
        "FULL11": {"problem": "the dev_case_v1 problem (11 probabilistic_nutrition rows in the joint chance "
                              "constraint, including the research-assumption rows PN-CP-HI and PN-EE-HI)",
                   "training_event": "H0 eleven (linear energy row)", "reference_event_equal_to_training": "h0_eleven"},
        "PART6P5": {"problem": ("run_dev_case_v1.s2_problem_cfg: the S2 declared six rows probabilistic (PN-CP-SUP, "
                                "PN-T1..T5; 'sourced' only in constraints.yaml's older value-block vocabulary -- the "
                                "adjudicated reference_constraints.csv also marks PN-NEL-FIXEDDMI, PN-CA-ABS and PN-P-ABS "
                                "sourced main-reference rows, which this arm plans at table values); PN-NEL-FIXEDDMI, "
                                "PN-CP-HI, PN-EE-HI, PN-CA-ABS, PN-P-ABS planned at table values and d_hat (SH-PLAN-*) and "
                                "kept as diagnostic_only scenario rows"),
                    "training_event": "S2 declared six", "reference_event_equal_to_training": "s2_six"},
        "MAIN9": {"problem": ("planned_arm_cfg (this driver; the same construction as run_dev_case_v1.s2_problem_cfg): the 9 "
                              "main-reference rows of reference_constraints.csv probabilistic (energy row trained on the "
                              "linear fixed-DMI row); the labelled research-assumption rows PN-CP-HI, PN-EE-HI planned at "
                              "table values and d_hat (SH-PLAN-CP-HI, SH-PLAN-EE-HI) and kept as diagnostic_only scenario "
                              "rows"),
                  "training_event": "main reference nine (linear energy row, Table 5-1 verdicts in every state)",
                  "reference_event_equal_to_training": "main9_training_event",
                  "declared": ("FIX3_BC, round-3 red team C-2, before any ablation run; no new method, grid or row; the "
                               "adjudication itself was made after seeing the dev run (docs/reference_problem_v1.md §3.3)")}},
    "cells": [
        {"cell_id": "SDH0_FULL11", "sd": "SD-H0", "objective": "FULL11",
         "note": "= H0 of pilot-20260924T205621Z-93d8654c (same problem, world, streams, methods)"},
        {"cell_id": "SDS2_FULL11", "sd": "SD-S2", "objective": "FULL11", "note": "new cell"},
        {"cell_id": "SDH0_PART6P5", "sd": "SD-H0", "objective": "PART6P5", "note": "new cell"},
        {"cell_id": "SDS2_PART6P5", "sd": "SD-S2", "objective": "PART6P5",
         "note": "= S2 of pilot-20260924T205621Z-93d8654c (same problem, world, streams, methods)"},
        {"cell_id": "SDH0_MAIN9", "sd": "SD-H0", "objective": "MAIN9", "note": "new cell (FIX3_BC, C-2)"},
        {"cell_id": "SDS2_MAIN9", "sd": "SD-S2", "objective": "MAIN9", "note": "new cell (FIX3_BC, C-2)"}],
    "methods": ("run_dev_case_v1.RUN_CONFIG['methods'] through run_dev_case_v1.run_method_block: M0 nominal; M1 relative "
                "margin with and without the DM margin; M2 joint chance SAA (N ladder, alpha_train multipliers 1, 0.75, "
                "0.5, 0.25); M3a box robust (k grid); M3b budget robust (k and Gamma grids); selection on validation "
                "(rate_upper_le_alpha, unknown counted as violated); no new method or grid"),
    "diagnostics": {"min_violation": {"what": "exact MILP min sum z_s over the N opt scenarios of the cell's training "
                                              "event (run_dev_case_v1.min_violations)",
                                      "time_limit_s": {"128": 60.0, "512": 180.0, "1024": 360.0, "16": 30.0},
                                      "time_limit_note": ("1024: 360 s declared in batch 2 of the official-run plan "
                                                          "(§6.5; configs/methods.yaml); not used by the development "
                                                          "N ladder {128, 512}"),
                                      "label": "training_scenarios_only: about these N scenarios, this world and this "
                                               "model; not a population, field or whole-space statement"},
                    "frontier": ("M2 with alpha_train = m*/N (cheapest ration at the attainable count; a diagnostic "
                                 "frontier point, not a target alpha, not a method result); scored like every ration")},
    "evaluation": {"evaluator": "ration_reliability.evaluation.reference.evaluate_reference",
                   "reference_constraints": REFERENCE_CSV.as_posix(),
                   "reference_problem": "the FULL11 dev_case_v1 problem (every H0 row compiled)",
                   "evaluation_worlds": ["SD-H0", "SD-S2"], "stream": "test",
                   "also_recorded": "the cell's own training event on its own world's test stream (public evaluator)"},
    "comparable_set_rule": ("per evaluation world and target alpha: rations of the declared methods (M0-M3, not the "
                            "diagnostic frontier) with structural_ok and main-reference rate_upper <= alpha on that "
                            "world, where the main reference event judges PN-T1..T5 only inside the primary Table 5-1 "
                            "domain (unknown outside it, FIX3_BC C-1); costs are compared only inside this set; coverage "
                            "= members / method entries attempted at that alpha (entries without a ration count in the "
                            "denominator); the membership under the R3C reading (Table 5-1 verdicts used out of domain) "
                            "and each member's out-of-domain share are reported next to it, never instead of it; "
                            "batch 2 (official-run plan): coverage is also reported per distinct ration (distinct "
                            "member q_hash / distinct attempted q_hash with a ration), and the membership statistic, "
                            "event and M1-DM-off role are set by endpoint_reporting"),
    "solver": {"time_limit_s": 300.0, "mip_rel_gap": 1e-4,
               "source": "run_dev_case_v1.RUN_CONFIG['solver'] (HiGHS via scipy; mip_rel_gap explicit, C11)"},
    # official-run plan batch 2 (2026-09-26): selection, membership and the descriptive reports.  The values below are
    # the DEVELOPMENT settings of this driver (they reproduce the development ablation); the official settings are
    # experiments/E1_cost_reliability/official_v2.py OFFICIAL_SETTINGS (screening cp_upper_le_alpha with c = 0.95,
    # membership p-bar = one-sided CP upper bound <= alpha on the plan-level main event, M1 without the DM margin a
    # sensitivity line) -- batch 3 wires them into the official path.
    "endpoint_reporting": {
        "screening_rule": "rate_upper_le_alpha", "screening_confidence": None,
        "membership_stat": "rate_upper", "main_event": "main_reference", "m1_dm_off_is_entry": True,
        "coverage": ("both ways: per entry (members / entries attempted, entries without a ration in the denominator) "
                     "and per distinct ration (distinct member q_hash / distinct attempted q_hash with a ration; entries "
                     "without a ration counted separately)"),
        "official_settings": "experiments/E1_cost_reliability/official_v2.py OFFICIAL_SETTINGS"},
    "per_method_report": ("per (evaluation world, alpha, method entry): the selected ration's main-event rate pair "
                          "[rate_lower, rate_upper], one-sided CP upper bound p-bar, target_met = p-bar <= alpha, its cost "
                          "labelled not_a_comparable_cost unless it is a comparable-set member, the minimum main-event "
                          "rate_upper over the declared grid's candidates with that candidate's cost (descriptive), and "
                          "the frontier diagnostic (m*/N bounds and the frontier ration's main-event rates; never a "
                          "member)"),
    "candidate_rations": ("every declared grid candidate of every method entry (M0; M1 k grid; M2 alpha_train grid per N; "
                          "M3a k grid; M3b every (k, Gamma)) scored once per distinct q_hash by evaluate_reference on "
                          "every evaluation world's test stream AFTER all selections are made; descriptive only -- no "
                          "selection, membership or set reads it; q in the restricted table only"),
    "matched_cost_curve": {
        "what": ("descriptive 'constraint satisfaction at matched cost' curve (contract T8.1 -> descriptive): per "
                 "evaluation world, cell and method family, the lower envelope over the scored grid candidates with "
                 "cost <= c of the main-event rate_upper and of p-bar"),
        "cost_axis": "relative to the cell's M0 cost: c = cost_M0 x (1 + r) (public form relative to M0; B-420)",
        "relative_cost_grid": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50],
        "grid_status": ("declared in batch 2 before any official run (research_scenario_assumption; plan §6.7 open for "
                        "the research lead)"),
        "families": ("method_id, with M1 split by apply_to_dm and M2 split by N; the diagnostic frontier is not a "
                     "family")},
}

#: membership statistics of the comparable set (official-run plan batch 2): ``rate_upper`` = (n_violated + n_unknown)/S
#: (development runs so far); ``cp_upper`` = the one-sided exact Clopper-Pearson upper bound p-bar (official).
MEMBERSHIP_STATS = ("rate_upper", "cp_upper")
#: label marker of the M1 variant without the DM margin (a sensitivity line in the official setting, not an entry)
M1_DM_OFF_MARKER = "apply_to_dm=False"


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------
def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _json_default(o: Any) -> Any:
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _find_key(d: Any, key: str) -> Any:
    """First value of ``key`` in a nested mapping (solver diagnostics differ between methods)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


#: FIX_PKG (round-3 red team D-3): keys of the MILP dual bound in solver diagnostics, the actual key first.
#: ``ration_reliability.optimization.chance_saa`` stores HiGHS' ``mip_dual_bound`` as ``mip_dual_bound_x_space`` (a
#: bound on the MILP objective in x space, i.e. before the polish LP); this driver looked up ``mip_dual_bound`` only, so
#: every dual-bound field of run pilot-20260925T055921Z-fb4f75af (frozen at 46c22cb) is null.  That run and its
#: artefacts are not changed; the fix only affects runs of a later freeze (this driver is a specification file: any
#: new run needs a new pin).
DUAL_BOUND_KEYS = ("mip_dual_bound_x_space", "mip_dual_bound")


def _dual_bound(diagnostics: Any) -> Optional[float]:
    """The MILP dual bound recorded in ``diagnostics`` (first key of ``DUAL_BOUND_KEYS`` found); None for LP methods or
    when HiGHS reports none."""
    d = dict(diagnostics or {})
    for key in DUAL_BOUND_KEYS:
        v = _f(_find_key(d, key))
        if v is not None:
            return v
    return None


def _ratio(s: str) -> float:
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


# The specification fingerprint, the v2 pin and its git anchor live in official_v2 (batch 4: the official freeze gate
# needs them); the names below keep this driver's interface (sha_or_none, spec_fingerprint, frozen_fingerprint,
# freeze_anchor, freeze_status, write_freeze_pin).
sha_or_none = OV2.sha_or_none
spec_fingerprint = OV2.spec_fingerprint
frozen_fingerprint = OV2.frozen_fingerprint
_git_out = OV2._git_out
freeze_anchor = OV2.freeze_anchor
freeze_status = OV2.freeze_status
write_freeze_pin = OV2.write_freeze_pin


# ------------------------------------------------------------------------------------------------
# streams and SD registry guards
# ------------------------------------------------------------------------------------------------
def reserved_roots(repo: Path = REPO) -> set[int]:
    doc = yaml.safe_load((repo / STREAMS_POLICY).read_text(encoding="utf-8"))
    return {int(r["root_seed"]) for r in doc["reserved_formal_streams"]["roots"]}


def check_seed(seed: int, repo: Path = REPO, *, official: bool = False) -> dict[str, Any]:
    """Development (default): the development root is allowed; a reserved formal root is refused before any draw
    (streams policy SP-3).  ``official=True`` (batch 4, inverted): only a reserved formal root is allowed -- derived by
    the label rule (``streams.reserved_formal_roots``) and equal to the policy file -- and every other seed (a
    development root included) is refused.  Messages never contain a reserved root value."""
    res = reserved_roots(repo)
    if official:
        k = ST.reserved_root_index(seed)
        if k is None:
            raise SystemExit("official mode: the seed is not a reserved formal-evaluation root (a development or other "
                             "root is refused in official mode; value withheld)")
        if int(seed) not in res:
            raise SystemExit("official mode: the derived reserved root differs from configs/streams_policy.yaml "
                             "(values withheld); refused")
        return {"root_k": int(k), "reserved_roots_checked": len(res), "class": "reserved_formal_root"}
    if int(seed) in res:
        raise SystemExit(f"seed {seed} is a reserved formal-evaluation root (configs/streams_policy.yaml); refused")
    return {"seed": int(seed), "reserved_roots_checked": len(res), "class": ABLATION_CONFIG["seed_class"]}


def sd_registry_check(ids: tuple[str, ...], repo: Path = REPO) -> dict[str, Any]:
    """Both SD arms equal the registered ratios (reports/sd_scaling_sources.csv, S2_cell rows): SD-H0 = 1 for every
    cell, SD-S2 = run_dev_case_v1.s2_ratio_arrays.  A difference stops the run (the declared arm changed)."""
    rows = [r for r in csv.DictReader((repo / SD_REGISTRY).open(encoding="utf-8")) if r["row_kind"] == "S2_cell"]
    r_s2, rd_s2 = DRV.s2_ratio_arrays(ids)
    bad: list[str] = []
    seen = 0
    for r in rows:
        iid, comp = r["ingredient_id"], r["component"]
        if iid not in ids:
            bad.append(f"{iid}: not an ingredient of the case")
            continue
        i = list(ids).index(iid)
        want_s2 = float(rd_s2[i]) if comp == "DM" else float(r_s2[i, DRV.PN.index(comp)])
        if abs(_ratio(r["ratio_SD_S2"]) - want_s2) > 1e-9:
            bad.append(f"{iid}/{comp}: ratio_SD_S2 {r['ratio_SD_S2']} != driver {want_s2:.6f}")
        if abs(_ratio(r["ratio_SD_H0"]) - 1.0) > 1e-12:
            bad.append(f"{iid}/{comp}: ratio_SD_H0 {r['ratio_SD_H0']} != 1")
        seen += 1
    if bad or seen != 48:
        raise SystemExit("SD registry check failed (reports/sd_scaling_sources.csv vs run_dev_case_v1): "
                         + "; ".join(bad[:10]) + (f"; {seen} S2 cells (expected 48)" if seen != 48 else ""))
    return {"registry": SD_REGISTRY.as_posix(), "registry_sha256": file_sha256(repo / SD_REGISTRY), "n_cells": seen,
            "SD-H0": "ratio 1 in every stochastic cell", "SD-S2": "= run_dev_case_v1.s2_ratio_arrays (48/48)",
            "ok": True}


# ------------------------------------------------------------------------------------------------
# the MAIN9 training arm (FIX3_BC, round-3 red team C-2)
# ------------------------------------------------------------------------------------------------
def main9_planned_rows(table: Any) -> tuple[str, ...]:
    """H0 rows that are not main-reference rows (labelled research assumptions): planned in MAIN9."""
    planned = tuple(c for c in H0_ELEVEN if table.row(c).role != "main_reference")
    if set(planned) != {"PN-CP-HI", "PN-EE-HI"} or set(table.main_reference_ids) != set(H0_ELEVEN) - set(planned):
        raise SystemExit(f"MAIN9 rule check failed: planned rows {planned}, main rows {table.main_reference_ids}")
    return planned


def planned_arm_cfg(cfg0: dict, problem: Any, planned_ids: tuple[str, ...], tag: str) -> tuple[dict, dict]:
    """Problem configuration of an objective arm in which ``planned_ids`` leave the joint chance constraint: each is
    imposed as a planned (table-value, decision-time d_hat) structural row ``SH-PLAN-*`` and kept as a diagnostic_only
    scenario row.  Same construction as ``run_dev_case_v1.s2_problem_cfg`` (which PART6P5 keeps using unchanged), with
    the planned set as an argument; nothing is deleted; bounds and tolerances are unchanged."""
    cc = problem.compiled
    th0 = problem.nominal_theta()
    cfg = copy.deepcopy(cfg0)
    byid = {g["ingredient_id"]: g for g in cfg["ingredients"]}
    new_rows: list[dict] = []
    record: dict[str, Any] = {"planned_rows": {}, "arm": tag, "construction": "planned_arm_cfg (= s2_problem_cfg rule)"}
    for cid in planned_ids:
        k = list(cc.constraint_ids).index(cid)
        if np.any(np.asarray(cc.v[k]) != 0):
            raise SystemExit(f"{tag}: planned row {cid} has as-fed terms (not supported)")
        c_nom = np.einsum("ij,ij->i", np.asarray(cc.W[k], float), th0) + np.asarray(cc.w0[k], float)
        name = "plan_" + cid[3:].replace("-", "_")
        is_energy = cid == "PN-NEL-FIXEDDMI"
        if is_energy:
            c_nom = c_nom / DRV.E_REF_MCAL_PER_KG
        for i, iid in enumerate(problem.ingredient_ids):
            byid[iid].setdefault("coefficients", {})[name] = {
                "value": float(c_nom[i]), "unit": "1", "status": "research_scenario_assumption",
                "rationale": (f"{tag} planned form of {cid}: compiled row content at the nominal (table-value) state, "
                              "canonical units per kg DM")}
        orig = next(c for c in cfg["constraints"] if c["constraint_id"] == cid)
        row = copy.deepcopy(orig)
        row["constraint_id"] = "SH-PLAN-" + cid[3:]
        row["name"] = f"{tag} planned form of {cid} (table values, decision-time d_hat)"
        row["terms"] = {f"C:{name}:DM": 1.0}
        row["constraint_class"] = "structural_hard"
        row["dm_source"] = "decision_estimate"
        row["claim_scope"] = (f"planning row of {tag}: {cid} at table values; not a reliability statement "
                              "(its scenario form is reported as a diagnostic)")
        if is_energy:
            b = dict(orig["bound"])
            b["unit"] = "kg/d"
            row["bound"] = b
        orig["constraint_class"] = "diagnostic_only"
        new_rows.append(row)
        record["planned_rows"][cid] = {"structural_id": row["constraint_id"], "coefficient": name}
    cfg["constraints"] += new_rows
    cfg["problem_id"] = f"{cfg0['problem_id']}|{tag}"
    cfg["description"] = (str(cfg0.get("description", "")) + f" | {tag}: joint event without "
                          f"{', '.join(planned_ids)}; those rows planned at table values (structural) + scenario "
                          "diagnostics.")
    return cfg, record


def planned_arm_consistency(problem_full: Any, problem_arm: Any, planned_ids: tuple[str, ...],
                            opts: SolverOptions) -> dict[str, Any]:
    """The arm's M0 must equal FULL11's M0 and every planned margin must equal the FULL11 nominal margin (2 LPs)."""
    m0h = get_method("M0_nominal")(problem_full, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    m0s = get_method("M0_nominal")(problem_arm, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    dq = float(np.max(np.abs(np.asarray(m0h.decision.q_as_fed) - np.asarray(m0s.decision.q_as_fed))))
    th0, dh = problem_full.nominal_theta(), problem_full.dm_estimates()
    evh = evaluate(m0h.decision, th0[None], dh[None], problem_full.compiled, prices=problem_full.prices)
    evs = evaluate(m0s.decision, problem_arm.nominal_theta()[None], problem_arm.dm_estimates()[None],
                       problem_arm.compiled, prices=problem_arm.prices)
    smarg = np.atleast_2d(evs.structural_margin)
    diffs = {cid: abs(float(evh.margin[0, list(evh.constraint_ids).index(cid)])
                      - float(smarg[0, list(evs.structural_ids).index("SH-PLAN-" + cid[3:])])) for cid in planned_ids}
    ok = dq < 1e-8 and abs(m0h.objective - m0s.objective) < 1e-9 and max(diffs.values()) < 1e-8
    out = {"M0_q_max_abs_diff": dq, "M0_cost_abs_diff": abs(m0h.objective - m0s.objective),
           "planned_margin_abs_diff_vs_full_nominal_margin": diffs, "ok": bool(ok)}
    if not ok:
        raise SystemExit(f"planned-arm consistency check failed: {out}")
    return out


def reference_events(table: Any) -> tuple[EventDefinition, ...]:
    """Default reference events + the MAIN9 training event (main rows, linear energy row, Table 5-1 verdicts in every
    state) so that the MAIN9 cell's own training event can be checked against the one reference evaluator."""
    main = table.main_reference_ids
    return tuple(default_events(table)) + (EventDefinition(
        "main9_training_event", main, "linear" if table.energy_row_id in main else "not_member",
        "the MAIN9 training event (main rows; energy by the linear row; Table 5-1 verdicts used in every state) -- "
        "a training-event check, not an endpoint", table51_domain="ignored"),)


# ------------------------------------------------------------------------------------------------
# context: problems, worlds, reference specification
# ------------------------------------------------------------------------------------------------
def build_context(sizes: dict[str, Any], *, draw: bool, official: Optional[dict[str, Any]] = None,
                  seed: Optional[int] = None, root_k: Optional[int] = None, official_token: Any = None,
                  run_context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Problems, worlds and the reference specification of the run.

    Development (``official`` is None; unchanged): the v1 problems, the two SD arms and ``ABLATION_CONFIG``'s root.
    Official (official-run plan batch 3): ``official`` is ``official_v2.OFFICIAL_CONFIG`` (or a mapping of its shape)
    and :func:`build_official_context` builds reference problem v2 and one world per declared SD point."""
    if official is not None:
        return build_official_context(sizes, draw=draw, config=official, seed=seed, root_k=root_k,
                                      official_token=official_token, run_context=run_context)
    inputs = DRV.check_inputs()                                        # preflight again + build report hashes
    problem, rep = load_problem(DRV.PROBLEM_YAML, mode="pilot")
    cfg0 = yaml.safe_load(DRV.PROBLEM_YAML.read_text(encoding="utf-8"))
    lin = DRV.load_linearisation(problem)
    cells = DRV.read_cells()
    cells_sha = file_sha256(DRV.CELLS_CSV)
    cfg_s2, s2_rec = DRV.s2_problem_cfg(cfg0, problem)
    problem_s2, rep_s2 = build_problem(cfg_s2, mode="pilot")
    table = load_reference_constraints(REFERENCE_CSV)
    main9_planned = main9_planned_rows(table)
    cfg_m9, m9_rec = planned_arm_cfg(cfg0, problem, main9_planned, "MAIN9")
    problem_m9, rep_m9 = build_problem(cfg_m9, mode="pilot")
    ids = tuple(problem.ingredient_ids)
    sdreg = sd_registry_check(ids)
    specs = {"SD-H0": DRV.build_h0_spec(ids, cells, cells_sha), "SD-S2": DRV.build_s2_spec(ids, cells, cells_sha)}
    worlds: dict[str, dict[str, Any]] = {}
    for sd, sp in specs.items():
        fm = build_uncertainty_model(sp, model_id=f"R3C_ABLATION_{sd.replace('-', '')}_factory_TN_MM")
        w = E.EnergyColumnModel(fm, lin)
        DRV.wrapped_nominal_check(w, problem)
        worlds[sd] = {"spec": sp, "fm": fm, "w": w, "record": DRV.model_public_record(fm, w)}
    ref = ReferenceSpec.from_problem(problem, lin, table, dgc_ingredient_ids=(DGC_INGREDIENT,),
                                     corn_silage_ingredient_ids=(DRV.CS,), events=reference_events(table),
                                     confidence=0.95)
    ctx = {"inputs": inputs, "problems": {"FULL11": problem, "PART6P5": problem_s2, "MAIN9": problem_m9},
           "validators": {"FULL11": rep, "PART6P5": rep_s2, "MAIN9": rep_m9}, "cfg0": cfg0, "s2_record": s2_rec,
           "main9_record": m9_rec, "main9_planned": main9_planned, "lin": lin,
           "worlds": worlds, "sd_registry": sdreg, "table": table, "ref": ref, "streams": None, "registry": None}
    if draw:
        streams = RandomStreams(int(ABLATION_CONFIG["seed"]))
        registry = DRV.StreamRegistry()
        for sd, wd in worlds.items():
            wd["draws"] = wd["w"].draw_world(streams, n_opt=sizes["opt"], n_validation=sizes["validation"],
                                             n_test=sizes["test"])
            for ds in wd["draws"].values():
                registry.add(f"ablation_world_{sd}", ds, crn_group="CRN-ablation", purpose=f"{sd} {ds.stream}")
        ctx["streams"], ctx["registry"] = streams, registry
    return ctx


def build_official_context(sizes: dict[str, Any], *, draw: bool, config: dict[str, Any],
                           seed: Optional[int] = None, root_k: Optional[int] = None, official_token: Any = None,
                           run_context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Context of the official run v2 (official-run plan batch 3; D-533, D-535): reference problem v2
    (``official_v2.build_v2_cfg0``: dev_case_v1 + the premise planning row) and its arms FULL11 (= cfg0_v2), PART6P5
    (``s2_problem_cfg(cfg0_v2, ...)``) and MAIN9 (``planned_arm_cfg(cfg0_v2, ...)``); the v2 reference table
    (``official_v2.REFERENCE_CSV_V2``) and ``ReferenceSpec`` of problem_v2; one world per declared SD point of
    ``config["sd_points"]`` (``official_v2.build_sd_point_spec``), checked against the registry
    (``official_v2.sd_registry_check_all``: 5 columns x 48 cells).

    ``draw=False`` (the ``--official-plan`` / ``--official --dry-run`` dry runs) reads the inputs and builds everything
    without a draw.  ``draw=True`` either (rehearsal / pipeline check) with an explicit **development** ``seed`` -- a
    reserved formal root is refused before any ``RandomStreams`` exists (the message withholds it) -- or (batch 4,
    official) with ``root_k`` and the ``official_token`` minted by ``official_v2.require_protocol_frozen()``: the
    reserved root k is derived by the label rule (``check_seed(..., official=True)``) and opened with the token;
    ``RandomStreams`` itself refuses it without a valid token.  ``run_context`` (plan F5): ``protocol_sha256`` and
    ``primary_assumption_id`` injected in memory into cfg0_v2 and every arm, which are then validated in mode
    ``official`` (``io/config.py`` refuses an official problem without them)."""
    seed_chk: dict[str, Any] = {}
    if draw:
        if official_token is not None:
            if root_k is None:
                raise SystemExit("official context: an official draw needs the root index k; nothing was drawn")
            seed_chk = check_seed(ST.reserved_formal_roots()[int(root_k)], official=True)
        elif seed is None:
            raise SystemExit("official context: draws need an explicit development seed (rehearsal) or the official "
                             "token of the freeze gate with a root index; nothing was drawn")
        else:
            try:
                seed_chk = check_seed(int(seed))
            except SystemExit:
                raise SystemExit("official context: the seed is a reserved formal-evaluation root; refused before any "
                                 "draw (value withheld)") from None
    inputs = DRV.check_inputs()                                        # preflight again + build report hashes
    problem_v1, rep_v1 = load_problem(DRV.PROBLEM_YAML, mode="pilot")
    cfg0 = yaml.safe_load(DRV.PROBLEM_YAML.read_text(encoding="utf-8"))
    cfg0_v2, problem_chk, v2_rec = OV2.build_v2_cfg0(cfg0, problem_v1)
    build_mode = "pilot"
    if run_context is not None:                                        # plan F5: official-mode validation of every arm
        cfg0_v2 = OV2.with_run_context(cfg0_v2, run_context)
        build_mode = "official"
    problem, rep = build_problem(cfg0_v2, mode=build_mode)
    if not rep.ok or problem is None or problem.compiled.fingerprint != problem_chk.compiled.fingerprint:
        raise SystemExit(f"official context: problem_v2 does not rebuild identically / does not validate in mode "
                         f"{build_mode} ({len(rep.errors)} errors: {[e[:120] for e in rep.errors[:3]]})")
    lin = DRV.load_linearisation(problem)
    cells = DRV.read_cells()
    cells_sha = file_sha256(DRV.CELLS_CSV)
    cfg_s2, s2_rec = DRV.s2_problem_cfg(cfg0_v2, problem)
    if run_context is not None:
        cfg_s2 = OV2.with_run_context(cfg_s2, run_context)
    problem_s2, rep_s2 = build_problem(cfg_s2, mode=build_mode)
    table = load_reference_constraints(OV2.REFERENCE_CSV_V2)
    if not table.is_v2:
        raise SystemExit(f"official context: {OV2.REFERENCE_CSV_V2.as_posix()} is not a v2 reference table")
    main9_planned = main9_planned_rows(table)
    cfg_m9, m9_rec = planned_arm_cfg(cfg0_v2, problem, main9_planned, "MAIN9")
    if run_context is not None:
        cfg_m9 = OV2.with_run_context(cfg_m9, run_context)
    problem_m9, rep_m9 = build_problem(cfg_m9, mode=build_mode)
    bad_arms = [k for k, r_ in (("PART6P5", rep_s2), ("MAIN9", rep_m9)) if not r_.ok]
    if bad_arms:
        raise SystemExit(f"official context: arms {bad_arms} do not validate in mode {build_mode}")
    ids = tuple(problem.ingredient_ids)
    sdreg = OV2.sd_registry_check_all(ids)
    points = list(config["sd_points"])
    worlds: dict[str, dict[str, Any]] = {}
    for sd in points:
        sp = OV2.build_sd_point_spec(sd, ids, cells, cells_sha)
        fm = build_uncertainty_model(sp, model_id=f"OFFICIAL_V2_{sd.replace('-', '')}_factory_TN_MM")
        w = E.EnergyColumnModel(fm, lin)
        DRV.wrapped_nominal_check(w, problem)
        worlds[sd] = {"spec": sp, "fm": fm, "w": w, "record": DRV.model_public_record(fm, w)}
    ref = ReferenceSpec.from_problem(problem, lin, table, dgc_ingredient_ids=(DGC_INGREDIENT,),
                                     corn_silage_ingredient_ids=(DRV.CS,), events=reference_events(table),
                                     confidence=0.95)
    missing = [w_ for w_ in config["evaluation"]["evaluation_worlds"] if w_ not in worlds]
    if missing:
        raise SystemExit(f"official context: evaluation worlds without an SD point: {missing}")
    ctx = {"inputs": inputs, "problems": {"FULL11": problem, "PART6P5": problem_s2, "MAIN9": problem_m9},
           "validators": {"FULL11": rep, "PART6P5": rep_s2, "MAIN9": rep_m9}, "cfg0": cfg0, "cfg0_v2": cfg0_v2,
           "v2_record": v2_rec, "s2_record": s2_rec, "main9_record": m9_rec, "main9_planned": main9_planned,
           "lin": lin, "worlds": worlds, "sd_registry": sdreg, "table": table, "ref": ref, "streams": None,
           "registry": None, "official": True, "config": config,
           "evaluation_worlds": list(config["evaluation"]["evaluation_worlds"]),
           "build_mode": build_mode,
           "run_context_injected": (None if run_context is None else
                                    {"protocol_sha256": run_context["protocol_sha256"],
                                     "primary_assumption_id": run_context["primary_assumption_id"],
                                     "arms": ["FULL11", "PART6P5", "MAIN9"], "mode": build_mode}),
           "root_k": None if official_token is None else int(root_k),
           "seed_class": (None if not draw else
                          "reserved formal root k (official; opened with the freeze-gate token)"
                          if official_token is not None else
                          "development root (rehearsal / pipeline check; not an official draw)")}
    if draw:
        if official_token is not None:                                 # the only place a reserved root is opened
            streams = RandomStreams(ST.reserved_formal_roots()[int(root_k)], official_token=official_token)
        else:
            streams = RandomStreams(int(seed_chk["seed"]))
        registry = DRV.StreamRegistry()
        for sd, wd in worlds.items():
            wd["draws"] = wd["w"].draw_world(streams, n_opt=sizes["opt"], n_validation=sizes["validation"],
                                             n_test=sizes["test"])
            for ds in wd["draws"].values():
                registry.add(f"official_v2_world_{sd}", ds, crn_group="CRN-official-v2", purpose=f"{sd} {ds.stream}")
        ctx["streams"], ctx["registry"] = streams, registry
    return ctx


# ------------------------------------------------------------------------------------------------
# plan and compute estimate
# ------------------------------------------------------------------------------------------------
def method_counts(Ns: list[int], n_alpha: int) -> dict[str, int]:
    m = DRV.RUN_CONFIG["methods"]
    lp = 1 + 2 * len(m["M1"]["grid"]) * n_alpha + len(m["M3a"]["k_grid"]) * n_alpha \
        + len(m["M3b"]["k_grid"]) * len(m["M3b"]["gamma_grid"]) * n_alpha
    milp = len(Ns) * len(m["M2"]["alpha_train_multipliers"]) * n_alpha
    return {"lp_solves": lp, "milp_solves_methods": milp, "milp_min_violation": len(Ns),
            "milp_frontier_and_check": 2 * len(Ns),
            "validation_candidate_evaluations": lp - 1 + milp,
            "rations_max": 1 + 2 * n_alpha + len(Ns) * n_alpha + 2 * n_alpha + len(Ns)}


def plan(ctx: dict[str, Any], sizes: dict[str, Any], mode: str, per_state_s: Optional[float] = None, *,
         config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Cells, per-cell counts and the compute estimate.  ``config`` (official-run plan batch 3): a configuration of the
    shape of ``ABLATION_CONFIG`` for the keys read here -- ``official_v2.OFFICIAL_CONFIG`` gives 9 cells, 3 N and 5
    evaluation worlds; ``None`` = the development ablation (unchanged)."""
    C = ABLATION_CONFIG if config is None else config
    alphas = [C["alphas"]["primary"]] + list(C["alphas"]["supplementary"])
    Ns = list(sizes["N_ladder"])
    mc = method_counts(Ns, len(alphas))
    tl = float(C["solver"]["time_limit_s"])
    mv_tl = sum(float(C["diagnostics"]["min_violation"]["time_limit_s"].get(str(N), 60.0)) for N in Ns)
    worst_cell = mc["milp_solves_methods"] * tl + mv_tl + mc["milp_frontier_and_check"] * tl
    n_cells = len(C["cells"])
    n_eval_worlds = len(C["evaluation"]["evaluation_worlds"])
    ref_states = n_cells * mc["rations_max"] * n_eval_worlds * int(sizes["test"])
    # batch 2: every grid candidate scored once per distinct ration (upper bound: no two candidates share a ration)
    cand_states = n_cells * (mc["validation_candidate_evaluations"] + 1) * n_eval_worlds * int(sizes["test"])
    per_state = per_state_s if per_state_s is not None else 2.5e-5
    prior = None
    rr = REPO / "results" / "pilot" / "pilot-20260924T205621Z-93d8654c" / "run_record.json"
    if rr.is_file():
        try:
            t = json.loads(rr.read_text(encoding="utf-8"))["extra"]["timings_s"]
            prior = {"source": "pilot-20260924T205621Z-93d8654c run_record extra.timings_s (holder_reported; local Mac)",
                     "h0_methods_s": _f(t.get("h0_methods_s")), "h0_min_violation_and_diagnostics_s":
                     _f(t.get("min_violation_s")), "s2_block_incl_information_value_s": _f(t.get("s2_s"))}
        except Exception:  # noqa: BLE001 -- the estimate stays without it
            prior = None
    cells = []
    for c in C["cells"]:
        p = ctx["problems"][c["objective"]]
        w = ctx["worlds"][c["sd"]]
        cells.append({"cell_id": c["cell_id"], "sd_arm": c["sd"], "objective_arm": c["objective"],
                      "problem_id": p.problem_id, "training_event": C["objective_arms"][c["objective"]]
                      ["training_event"], "world_fingerprint": w["w"].fingerprint(),
                      "spec_fingerprint": w["spec"].fingerprint(), "note": c["note"], **mc})
    return {"mode": mode, "sizes": sizes, "alphas": alphas, "cells": cells,
            "totals": {k: v * n_cells for k, v in mc.items()},
            "compute_estimate": {
                "worst_case_solver_time_s": worst_cell * n_cells,
                "worst_case_note": "every MILP reaching its time limit (upper bound; LPs are milliseconds each)",
                "reference_evaluation_states": ref_states,
                "reference_evaluation_s_estimate": ref_states * per_state,
                "candidate_scoring_states_max": cand_states,
                "candidate_scoring_s_estimate_max": cand_states * per_state,
                "per_state_s_used": per_state,
                "per_state_source": ("measured in this run (--tiny, smoke)" if per_state_s is not None else
                                     "default 2.5e-5 s/state (R3C smoke on the Mac: about 2e-5 s/state; smoke, not a "
                                     "benchmark)"),
                "prior_dev_run_timings": prior,
                "memory": ("draws of two worlds: 2 x (opt + validation + test) x 8 x 8 doubles; well under 1 GB"
                           if config is None else
                           f"draws of {n_eval_worlds} worlds per job: {n_eval_worlds} x (opt + validation + test) x 8 x "
                           "8 doubles; well under 1 GB"),
                "host": "host explicitly authorised for this invocation; consult this run record (not a historical status label)"}}


def official_plan(ctx: dict[str, Any], config: Optional[dict[str, Any]] = None,
                  per_state_s: Optional[float] = None) -> dict[str, Any]:
    """The official cell / job matrix and its compute estimate (official-run plan batch 3, §5; ``--official-plan``).

    Data only: :func:`plan` with the official configuration (9 cells, N ladder {128, 512, 1024}, 5 evaluation worlds),
    ``official_v2.official_jobs`` (root indices k -- never a seed) and a worst-case / planning-figure estimate per job,
    serial and with the considered worker counts.  Worst case = every MILP at its time limit (LPs are milliseconds);
    the reference-evaluation seconds use ``per_state_s`` (default: the configuration's plan-§5 figure)."""
    C = OV2.OFFICIAL_CONFIG if config is None else config
    per_state = float(C["compute"]["per_state_s"] if per_state_s is None else per_state_s)
    sizes = dict(C["streams"])
    pl = plan(ctx, sizes, "official (planned; not run)", per_state_s=per_state, config=C)
    jobs = OV2.official_jobs(config=C)
    n_cells = len(C["cells"])
    n_worlds = len(C["evaluation"]["evaluation_worlds"])
    n_test = int(sizes["test"])
    ce = pl["compute_estimate"]
    cell_solver_s = ce["worst_case_solver_time_s"] / n_cells
    cell_eval_s = (ce["reference_evaluation_s_estimate"] + ce["candidate_scoring_s_estimate_max"]) / n_cells
    t45, e1 = C["diagnostics"]["DIAG-T45"], C["diagnostics"]["DIAG-E1"]
    n_h = len(t45["h_grid"]["alphas"]) * len(t45["h_grid"]["factors"]) + int(bool(t45["h_grid"]["include_zero"])) \
        + int(t45["h_grid"].get("a1_equal_subdivisions", 0))                   # D-539: equal steps over [0, A1]
    n_t45_worlds = len(t45["training_worlds"])
    t45_rations = 2 + n_t45_worlds * n_h
    e1_mv = sum(float(e1["min_violation_time_limit_s"].get(str(N), 60.0)) for N in e1["N_ladder"])
    e1_solver_s = len(e1["training_worlds"]) * (e1_mv + len(e1["N_ladder"]) * len(e1["alphas"]) * float(e1["m2_time_limit_s"]))
    e1_rations = len(e1["training_worlds"]) * len(e1["N_ladder"]) * len(e1["alphas"])
    diag_eval_s = (t45_rations + e1_rations) * n_worlds * n_test * per_state
    diag_s = e1_solver_s + diag_eval_s
    n_cell_jobs = sum(j["kind"] == "cell" for j in jobs)
    n_diag_jobs = sum(j["kind"] == "diagnostics" for j in jobs)
    job_s = [cell_solver_s + cell_eval_s if j["kind"] == "cell" else diag_s for j in jobs]
    serial = float(sum(job_s))
    longest = float(max(job_s)) if job_s else 0.0
    par = {str(w): {"lower_bound_s": max(longest, serial / w), "list_scheduling_upper_bound_s": serial / w + longest}
           for w in C["compute"]["workers_considered"]}
    matrix = {str(k): [j["cell_id"] for j in jobs if j["root_k"] == k and j["kind"] == "cell"]
              for k in C["roots"]["root_k"]}
    return {"plan": pl, "jobs": jobs,
            "matrix": {"n_cells": n_cells, "n_cell_jobs": n_cell_jobs, "n_diagnostics_jobs": n_diag_jobs,
                       "n_jobs": len(jobs), "cells_by_root_k": matrix, "rule": C["matrix"]["rule"],
                       "roots": ("root indices k = 0, 1, 2 (a job resolves its reserved root only after the freeze "
                                 "gate, batch 4)"),
                       "diagnostics_roots_k": list(C["matrix"].get("diagnostics_roots_k", [])),
                       "workers_declared": C["matrix"].get("workers"),
                       "sd_points": list(C["sd_points"]), "evaluation_worlds": list(C["evaluation"]["evaluation_worlds"]),
                       "N_ladder": list(sizes["N_ladder"]), "alphas": pl["alphas"]},
            "compute_estimate": {
                "per_cell_job_worst_case_solver_s": cell_solver_s,
                "per_cell_job_reference_and_candidate_scoring_s": cell_eval_s,
                "per_diagnostics_job_worst_case_s": diag_s,
                "per_diagnostics_job": {"DIAG-E1_worst_case_solver_s": e1_solver_s,
                                        "DIAG-T45": f"{2 + n_t45_worlds} LP pairs + {n_t45_worlds * n_h} A2 LP pairs "
                                                    "(milliseconds each)",
                                        "scoring_s": diag_eval_s, "max_rations": t45_rations + e1_rations},
                "serial_worst_case_s": serial, "serial_worst_case_h": serial / 3600.0,
                "parallel_worst_case_by_workers": par,
                "planning_figures_plan_section_5": C["compute"]["planning_figures_plan_section_5"],
                "per_state_s_used": per_state, "per_state_source": C["compute"]["per_state_source"],
                "worst_case_note": ("every MILP at its time limit (methods, min-violation, frontier, DIAG-E1); LPs "
                                    "are milliseconds; validation screening of candidates is not included"),
                "threads": C["compute"]["threads"]}}


# ------------------------------------------------------------------------------------------------
# one cell
# ------------------------------------------------------------------------------------------------
def run_cell(ctx: dict[str, Any], cell: dict[str, Any], sizes: dict[str, Any], opts: SolverOptions, *,
             config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One cell: the method block, the min-violation diagnostics and the frontier points.  ``config`` (batch 3): the
    official configuration (alphas, screening rule, min-violation time limits incl. N = 1024, training events);
    ``None`` = ``ABLATION_CONFIG`` (development, unchanged)."""
    C = ABLATION_CONFIG if config is None else config
    problem = ctx["problems"][cell["objective"]]
    wd = ctx["worlds"][cell["sd"]]
    opt, val = wd["draws"]["opt"], wd["draws"]["validation"]
    alphas = [C["alphas"]["primary"]] + list(C["alphas"]["supplementary"])
    Ns = list(sizes["N_ladder"])
    tag = cell["cell_id"]
    t0 = time.perf_counter()
    rep_cfg = C["endpoint_reporting"]
    m3b_per_k: list[dict[str, Any]] = []
    runs, tables = DRV.run_method_block(problem, wd["fm"], ctx["lin"], opt, val, alphas, Ns, opts,
                                        m1_variants=(True, False), tag=tag,
                                        box_tag=f"dev_case_v1_{cell['sd'].replace('-', '')}",
                                        screening_rule=rep_cfg["screening_rule"],
                                        confidence=rep_cfg["screening_confidence"], selection_sink=m3b_per_k)
    t_methods = time.perf_counter() - t0
    diag = {"min_violation": [], "frontier_points": []}
    frontier_runs = []
    tlmap = C["diagnostics"]["min_violation"]["time_limit_s"]
    for N in Ns:
        mv = DRV.min_violations(problem, opt, N, time_limit=float(tlmap.get(str(N), 60.0)))
        rec = {k: v for k, v in mv.items() if k != "x"}
        rec.update({"time_limit_s": float(tlmap.get(str(N), 60.0)),
                    "label": C["diagnostics"]["min_violation"]["label"],
                    "training_event": C["objective_arms"][cell["objective"]]["training_event"],
                    "attainable_training_rate_lower": (None if mv["min_violations_lower"] is None
                                                       else mv["min_violations_lower"] / N),
                    "allowed_by_alpha": {str(a): allowed_violations(a, N) for a in alphas}})
        diag["min_violation"].append(rec)
        ub = mv["min_violations_upper"]
        if ub is None:
            continue
        a_front = DRV.alpha_for_count(int(ub), int(N))
        fr = get_method("M2_joint_chance_saa")(problem, opt_draws=opt, params={"alpha_train": a_front, "n_scenarios": N,
                                                                              "big_m_mode": "box_quantile"},
                                               solver_options=opts)
        chk = None
        if mv["proven"] and ub >= 1:
            a_b = DRV.alpha_for_count(ub - 1, N)
            below = get_method("M2_joint_chance_saa")(problem, opt_draws=opt, params={"alpha_train": a_b, "n_scenarios": N,
                                                                                     "big_m_mode": "box_quantile"},
                                                      solver_options=opts)
            chk = {"alpha_train": a_b, "allowed": allowed_violations(a_b, N), "status": str(below.status),
                   "expected": "proven_infeasible"}
        frontier_runs.append(SP.MethodRun(SP.MethodSpec("M2_joint_chance_saa", uses_opt_draws=True,
                                                        label=f"{tag}:DIAG-frontier:M2[N={N},alpha_train=m*/N={ub}/{N}]"),
                                          fr, world_fingerprint=opt.model_fingerprint))
        diag["frontier_points"].append({"N": N, "m_star_upper": ub, "m_star_lower": mv["min_violations_lower"],
                                        "alpha_train": a_front, "status": str(fr.status), "mip_gap": _f(fr.mip_gap),
                                        "mip_dual_bound": _dual_bound(fr.diagnostics),
                                        "objective_usd_per_head_d": _f(fr.objective),
                                        "consistency_check_one_less_violation": chk,
                                        "label": "diagnostic frontier point; training scenarios only; not a method"})
    return {"cell": cell, "problem": problem, "runs": runs, "frontier_runs": frontier_runs, "tables": tables,
            "diag": diag, "m3b_per_k": m3b_per_k,
            "timings_s": {"methods": t_methods, "total": time.perf_counter() - t0}}


# ------------------------------------------------------------------------------------------------
# rows
# ------------------------------------------------------------------------------------------------
def _alpha_of(label: str) -> Optional[float]:
    return DRV.run_alpha(label)


def solve_fields(r: Any, opts: SolverOptions) -> dict[str, Any]:
    res = r.result
    sel = r.selection
    cand = []
    if sel is not None:
        for c in sel.candidates:
            sr = c.solve_result
            cand.append({"value": c.value, "status": c.status, "meets_screen": c.meets_screen,
                         "rate_upper_validation": c.rate_upper, "cost": c.cost, "mip_gap": _f(sr.mip_gap),
                         "mip_dual_bound": _dual_bound(sr.diagnostics),
                         "wall_time_s": _f(sr.wall_time_s)})
    statuses = sorted({c["status"] for c in cand})
    if r.has_ration:
        scope = None
    elif sel is not None and sel.status != "selected":
        if r.spec.method_id == "M2_joint_chance_saa" and statuses == ["proven_infeasible"]:
            scope = ("SAA over the N training scenarios proven infeasible for every alpha_train of the declared grid "
                     "(training scenarios only; not a population, field or whole-space statement)")
        else:
            scope = ("no value of the declared grid met the validation screen (candidate statuses: "
                     + ", ".join(statuses) + "); a grid without a solution is not a statement about the whole "
                     "decision space")
    else:
        scope = f"solver status {None if res is None else res.status}; not a statement about the whole decision space"
    return {"solver_status": None if res is None else str(res.status),
            "selection_status": None if sel is None else sel.status,
            "selected_param": (None if sel is None or sel.status != "selected" else
                               f"{sel.param_name}={sel.selected_value}"),
            "mip_gap": None if res is None else _f(res.mip_gap),
            "mip_dual_bound": None if res is None else _dual_bound(res.diagnostics),
            "solver_time_limit_s": float(opts.time_limit_s), "solve_wall_time_s": None if res is None else
            _f(res.wall_time_s), "candidate_statuses": ";".join(statuses) if statuses else None,
            "infeasibility_scope": scope, "_candidates": cand}


REFERENCE_BLOCK_EVENTS = ("main_reference", MAIN_EVENT_PLAN_DOMAIN, "main_reference_linear_energy",
                          MAIN_EVENT_T51_IGNORED, "main_reference_plus_cp_hi", "main_reference_plus_cp_hi_plan_domain",
                          "h0_eleven",
                          "h0_eleven_reference_energy", "h0_eleven_domain_conditioned", "s2_six",
                          "s2_six_domain_conditioned", "excluded_five", "main9_training_event")


def reference_block(res: Any) -> dict[str, Any]:
    s = res.summary()
    ev = s["events"]
    out: dict[str, Any] = {}
    for eid in REFERENCE_BLOCK_EVENTS:
        e = ev.get(eid)
        if e is None:
            continue
        out.update({f"{eid}_n_violated": e["n_violated"], f"{eid}_n_unknown": e["n_unknown"],
                    f"{eid}_rate_lower": e["rate_lower"], f"{eid}_rate_upper": e["rate_upper"],
                    f"{eid}_cp_upper": e["cp_upper_one_sided"], f"{eid}_mc_se": e["mc_se"]})
    er = s["energy_reference"]
    prim = next(v for v in s["domain"].values() if v["role"] == "primary")
    out.update({"energy_linear_violation_rate": er["linear_violation_rate"],
                "energy_reference_violation_rate_lower": er["reference_violation_rate_lower"],
                "energy_reference_violation_rate_upper": er["reference_violation_rate_upper"],
                "energy_false_pass": er["class_counts"]["false_pass"],
                "energy_false_fail": er["class_counts"]["false_fail"],
                "energy_reference_undefined": er["n_reference_undefined"],
                "t51_primary_variant": prim["variant_id"],
                "t51_primary_not_assessable_share": prim["shares"]["not_assessable"],
                "t51_primary_undefined_share": prim["shares"]["undefined"],
                "t51_primary_nominal_status": prim.get("nominal_status"),
                "main_reference_unknown_only_because_out_of_domain": s["headline"].get(
                    "main_reference_unknown_only_because_out_of_domain"),
                "main_reference_rate_upper_by_t51_variant": ";".join(
                    f"{vid}={blk['main_reference_conditioned_on_this_variant']['rate_upper']:.6g}"
                    for vid, blk in s["domain"].items() if "main_reference_conditioned_on_this_variant" in blk),
                "n_analysis_anomaly_states": s["composition"]["n_analysis_anomaly_states"],
                "n_support_violation_states": s["composition"]["n_support_violation_states"],
                "structural_ok": s["structural"]["structural_ok"]})
    return out


ENDPOINT_COLUMNS = [
    "run_id", "output_label", "cell_id", "sd_arm", "objective_arm", "training_event", "training_world_fingerprint",
    "random_generation", "label", "method_id", "ration_kind", "target_alpha", "selection_status", "selected_param",
    "solver_status", "mip_gap", "mip_dual_bound", "solver_time_limit_s", "solve_wall_time_s", "candidate_statuses",
    "infeasibility_scope", "has_ration", "q_hash", "cost_usd_per_head_d", "evaluation_world",
    "evaluation_world_fingerprint", "test_stream_id", "own_world", "n_test", "training_event_rate_upper_own_world",
    "training_event_cp_upper_own_world",
    "main_reference_n_violated", "main_reference_n_unknown", "main_reference_rate_lower", "main_reference_rate_upper",
    "main_reference_cp_upper", "main_reference_mc_se", "t51_primary_not_assessable_share",
    "main_reference_unknown_only_because_out_of_domain",
    "main_reference_linear_energy_rate_upper", "main_reference_linear_energy_cp_upper",
    f"{MAIN_EVENT_T51_IGNORED}_rate_lower", f"{MAIN_EVENT_T51_IGNORED}_rate_upper", f"{MAIN_EVENT_T51_IGNORED}_cp_upper",
    "main_reference_plus_cp_hi_rate_upper", "main_reference_plus_cp_hi_cp_upper",
    "h0_eleven_rate_upper", "h0_eleven_cp_upper", "h0_eleven_reference_energy_rate_upper",
    "h0_eleven_domain_conditioned_rate_upper",
    "s2_six_rate_upper", "s2_six_cp_upper", "s2_six_domain_conditioned_rate_upper",
    "excluded_five_rate_upper", "excluded_five_cp_upper", "main9_training_event_rate_upper",
    "energy_linear_violation_rate", "energy_reference_violation_rate_lower", "energy_reference_violation_rate_upper",
    "energy_false_pass", "energy_false_fail", "energy_reference_undefined", "t51_primary_variant",
    "t51_primary_undefined_share", "t51_primary_nominal_status", "main_reference_rate_upper_by_t51_variant",
    "n_analysis_anomaly_states", "n_support_violation_states", "structural_ok", "meets_main_reference_alpha_point",
    "meets_main_reference_alpha_cp_upper", "in_comparable_set", "in_comparable_set_t51_verdicts_used_out_of_domain"]

RESIDUAL_COLUMNS = ["run_id", "output_label", "cell_id", "label", "q_hash", "evaluation_world", "test_stream_id",
                    "constraint_id", "verdict", "class_in_reference_problem", "status", "role", "unit",
                    "in_main_reference", "in_h0_eleven", "in_s2_six", "in_excluded_five", "n_states", "n_defined",
                    "n_violated", "n_unknown", "rate_lower", "rate_upper", "cp_upper_one_sided", "mc_se",
                    "mean_deficit_given_violation", "max_deficit", "margin_q01", "margin_q05", "margin_q50"]


def cell_rows(ctx: dict[str, Any], cr: dict[str, Any], opts: SolverOptions, run_id: str, label_out: str, *,
              config: Optional[dict[str, Any]] = None) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """(endpoint rows, residual rows, ration rows, solve-status rows) of one cell; every ration is scored on every
    evaluation world (development: both SD arms; ``config`` = the official configuration: the five SD points) with the
    one reference evaluator."""
    C = ABLATION_CONFIG if config is None else config
    cell = cr["cell"]
    problem = cr["problem"]
    wd_own = ctx["worlds"][cell["sd"]]
    gen = (f"{wd_own['spec'].spec_id} via uncertainty factory (TN_MM, fallback BETA_MM; C0 independent) + "
           f"EnergyColumnModel; streams opt={wd_own['draws']['opt'].stream_id} (n={wd_own['draws']['opt'].n_draws}), "
           f"validation={wd_own['draws']['validation'].stream_id} (n={wd_own['draws']['validation'].n_draws})")
    all_runs = list(cr["runs"]) + list(cr["frontier_runs"])
    # the cell's own training event on its own test stream (public evaluator; leakage check: no test stream used)
    SP.evaluate_on_test(all_runs, problem, wd_own["draws"]["test"])
    ends, resid, rations, status = [], [], [], []
    for r in all_runs:
        kind = "diagnostic_frontier" if ":DIAG-frontier:" in r.spec.name else "method"
        sf = solve_fields(r, opts)
        base = {"run_id": run_id, "output_label": label_out, "cell_id": cell["cell_id"], "sd_arm": cell["sd"],
                "objective_arm": cell["objective"],
                "training_event": C["objective_arms"][cell["objective"]]["training_event"],
                "training_world_fingerprint": wd_own["w"].fingerprint(), "random_generation": gen,
                "label": r.spec.name, "method_id": r.spec.method_id, "ration_kind": kind,
                "target_alpha": _alpha_of(r.spec.name), "has_ration": bool(r.has_ration),
                **{k: v for k, v in sf.items() if not k.startswith("_")}}
        status.append({**{k: base[k] for k in ("cell_id", "label", "method_id", "ration_kind", "target_alpha")},
                       **{k: v for k, v in sf.items() if not k.startswith("_")}, "candidates": sf["_candidates"]})
        if not r.has_ration:
            ends.append({**base, "q_hash": None, "cost_usd_per_head_d": None, "evaluation_world": None})
            continue
        dec = r.result.decision
        own_ev = r.evaluation.summary()["joint"]
        k_own = own_ev["n_violated"] + own_ev["n_unknown"]
        rations.append({"run_id": run_id, "cell_id": cell["cell_id"], "label": r.spec.name,
                        "q_hash": None, **{f"q_{iid}": float(q) for iid, q in zip(dec.ingredient_ids, dec.q_as_fed)}})
        for sd_eval in C["evaluation"]["evaluation_worlds"]:
            wd = ctx["worlds"][sd_eval]
            test = wd["draws"]["test"]
            res = evaluate_reference(dec, test, ctx["ref"])
            own = sd_eval == cell["sd"]
            row = {**base, "q_hash": res.identity["q_hash"], "cost_usd_per_head_d": res.summary()["cost"],
                   "evaluation_world": sd_eval, "evaluation_world_fingerprint": wd["w"].fingerprint(),
                   "test_stream_id": test.stream_id, "own_world": own, "n_test": res.n_states,
                   "training_event_rate_upper_own_world": (k_own / own_ev["n_draws"]) if own else None,
                   "training_event_cp_upper_own_world": (one_sided_upper(k_own, own_ev["n_draws"], 0.95)
                                                         if own else None),
                   **reference_block(res)}
            ends.append(row)
            rations[-1]["q_hash"] = res.identity["q_hash"]
            for pr in res.per_constraint_rows():
                resid.append({"run_id": run_id, "output_label": label_out, "cell_id": cell["cell_id"],
                              "label": r.spec.name, "q_hash": res.identity["q_hash"], "evaluation_world": sd_eval,
                              "test_stream_id": test.stream_id,
                              **{k: pr.get(k) for k in RESIDUAL_COLUMNS if k in pr}})
    return ends, resid, rations, status


def _stat_key(rate_key: str, cp_key: str, membership_stat: str) -> str:
    if membership_stat not in MEMBERSHIP_STATS:
        raise ValueError(f"membership_stat must be one of {MEMBERSHIP_STATS}, got {membership_stat!r}")
    return rate_key if membership_stat == "rate_upper" else cp_key


def _meets(r: dict[str, Any], a: float, stat_key: str) -> bool:
    """A ration meets the membership rule at ``a``: it exists, its planning rows hold and its statistic is <= a."""
    return bool(r.get("has_ration") and r.get("structural_ok") and r.get(stat_key) is not None and r[stat_key] <= a)


def is_sensitivity_line(label: str) -> bool:
    """The M1 variant without the DM margin (declared a sensitivity line in the official setting)."""
    return ":M1[" in str(label) and M1_DM_OFF_MARKER in str(label)


def _membership(rows: list[dict[str, Any]], att: list, sd_eval: str, a: float, rate_key: str, cp_key: str,
                tag: str, membership_stat: str = "rate_upper") -> list[dict[str, Any]]:
    """Members of the comparable set among the entries ``att`` on ``sd_eval`` at ``a``.

    ``membership_stat`` (official-run plan batch 2): ``rate_upper`` -- the event's rate_upper (n_violated + n_unknown)
    / S <= a (development runs so far); ``cp_upper`` -- the one-sided exact Clopper-Pearson upper bound p-bar <= a
    (official).  Both statistics are carried by every member whichever decides."""
    stat_key = _stat_key(rate_key, cp_key, membership_stat)
    att_set = set(att)
    members = []
    for r in rows:
        if (r["cell_id"], r["label"]) not in att_set or r.get("evaluation_world") != sd_eval:
            continue
        ok = _meets(r, a, stat_key)
        r.setdefault(tag, {})[str(a)] = ok
        if ok:
            members.append({"cell_id": r["cell_id"], "label": r["label"], "q_hash": r["q_hash"],
                            "cost_usd_per_head_d": r["cost_usd_per_head_d"],
                            "main_reference_rate_upper": r.get("main_reference_rate_upper"),
                            "main_reference_cp_upper": r.get("main_reference_cp_upper"),
                            f"{MAIN_EVENT_T51_IGNORED}_rate_upper": r.get(f"{MAIN_EVENT_T51_IGNORED}_rate_upper"),
                            "t51_primary_not_assessable_share": r.get("t51_primary_not_assessable_share"),
                            "h0_eleven_rate_upper": r.get("h0_eleven_rate_upper"),
                            "main_reference_plus_cp_hi_rate_upper": r.get("main_reference_plus_cp_hi_rate_upper"),
                            "membership_stat": membership_stat,
                            "membership_rate_upper": r.get(rate_key), "membership_cp_upper": r.get(cp_key),
                            "rate_upper_le_alpha": bool(r.get(rate_key) is not None and r[rate_key] <= a),
                            "cp_upper_le_alpha": bool(r.get(cp_key) is not None and r[cp_key] <= a)})
    cheapest = min((m["cost_usd_per_head_d"] for m in members), default=None)
    for m in members:
        m["cost_minus_cheapest_member_usd"] = m["cost_usd_per_head_d"] - cheapest
    return sorted(members, key=lambda m: m["cost_usd_per_head_d"])


def _coverage(rows: list[dict[str, Any]], att: list, sd_eval: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage both ways (official-run plan §1 "成员"; D-533): per entry and per distinct ration.

    Per entry: members / entries attempted (an entry without a ration stays in the denominator).  Per distinct ration:
    distinct member q_hash / distinct q_hash among the attempted entries that have a ration on ``sd_eval`` -- the same
    ration reached by several entries (M0 in every cell of one objective arm, one M1 ration at every alpha) counts once;
    entries without a ration are counted separately (``n_entries_without_ration``)."""
    att_set = set(att)
    with_ration = {(r["cell_id"], r["label"]): r["q_hash"] for r in rows
                   if (r["cell_id"], r["label"]) in att_set and r.get("has_ration")
                   and r.get("evaluation_world") == sd_eval}
    no_ration = sorted(k for k in att_set if k not in with_ration)
    distinct_att = set(with_ration.values())
    distinct_mem = {m["q_hash"] for m in members}
    return {"coverage_entries": (len(members) / len(att)) if att else None,
            "n_entries_with_ration": len(with_ration), "n_entries_without_ration": len(no_ration),
            "n_distinct_rations_attempted": len(distinct_att), "n_distinct_member_rations": len(distinct_mem),
            "coverage_distinct_rations": (len(distinct_mem) / len(distinct_att)) if distinct_att else None}


def comparable_sets(rows: list[dict[str, Any]], *, membership_stat: str = "rate_upper",
                    main_event: str = "main_reference", m1_dm_off_is_entry: bool = True,
                    config: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Per evaluation world and target alpha: comparable feasible set, coverage, costs inside the set only.

    FIX3_BC (round-3 red team C-1): membership uses the **domain-conditioned** main reference event (Table 5-1 verdicts
    only inside the primary domain; out-of-domain states enter rate_upper).  The membership under the R3C reading (T
    verdicts used out of domain) is reported next to it (``members_t51_verdicts_used_out_of_domain``), and every member
    carries its out-of-domain share and the H0-eleven rate; the headline of each set gives the out-of-domain share
    range of the rations considered.

    Official-run plan batch 2: ``membership_stat`` (``rate_upper`` development default; ``cp_upper`` official = p-bar,
    the one-sided CP upper bound, <= alpha), ``main_event`` (the event whose ``<event>_rate_upper`` / ``_cp_upper``
    columns decide; development ``main_reference``, official the plan-level reading ``main_reference_plan_domain``,
    D-535) and ``m1_dm_off_is_entry`` (development True; official False: the M1 variant without the DM margin is a
    sensitivity line, reported under ``sensitivity_lines`` and never an entry or a member).  Coverage is reported per
    entry (``coverage`` = ``coverage_entries``) and per distinct ration (``coverage_distinct_rations``).  The diagnostic
    frontier is never an entry.  Only ``ration_kind == "method"`` rows are read: the frontier and the declared
    diagnostics (``ration_kind = "diagnostic"``, batch 3) are never entries or members.  ``config`` (batch 3): the
    official configuration (alphas, the five evaluation worlds); ``None`` = ``ABLATION_CONFIG``."""
    C = ABLATION_CONFIG if config is None else config
    alphas = [C["alphas"]["primary"]] + list(C["alphas"]["supplementary"])
    rate_key, cp_key = f"{main_event}_rate_upper", f"{main_event}_cp_upper"
    stat_key = _stat_key(rate_key, cp_key, membership_stat)
    out = []
    method_rows = [r for r in rows if r["ration_kind"] == "method"]
    all_entries = {(r["cell_id"], r["label"]) for r in method_rows}
    entries = {k for k in all_entries if m1_dm_off_is_entry or not is_sensitivity_line(k[1])}
    sens = all_entries - entries
    for sd_eval in C["evaluation"]["evaluation_worlds"]:
        for a in alphas:
            att = sorted({(c, lab) for (c, lab) in entries
                          if _alpha_of(lab) in (None, a)})              # M0 (no alpha) enters every alpha
            members = _membership(method_rows, att, sd_eval, a, rate_key, cp_key, "_member", membership_stat)
            alt = _membership(method_rows, att, sd_eval, a, f"{MAIN_EVENT_T51_IGNORED}_rate_upper",
                              f"{MAIN_EVENT_T51_IGNORED}_cp_upper", "_member_alt", membership_stat)
            cov = _coverage(method_rows, att, sd_eval, members)
            shares = [r.get("t51_primary_not_assessable_share") for r in method_rows
                      if (r["cell_id"], r["label"]) in att and r.get("evaluation_world") == sd_eval
                      and r.get("t51_primary_not_assessable_share") is not None]
            sens_rows = [{"cell_id": r["cell_id"], "label": r["label"], "q_hash": r.get("q_hash"),
                          "rate_upper": r.get(rate_key), "cp_upper": r.get(cp_key),
                          "would_meet_rule": _meets(r, a, stat_key),
                          "role": "sensitivity line (M1 without the DM margin); not an entry, never a member"}
                         for r in method_rows if (r["cell_id"], r["label"]) in sens
                         and _alpha_of(r["label"]) == a and r.get("evaluation_world") == sd_eval]
            out.append({"evaluation_world": sd_eval, "target_alpha": a, "n_entries_attempted": len(att),
                        "n_members": len(members), "coverage": cov["coverage_entries"], **cov,
                        "membership_event": main_event, "membership_stat": membership_stat,
                        "membership_statistic": ("one-sided exact Clopper-Pearson upper bound (confidence 0.95) of "
                                                 "(n_violated + n_unknown) / S <= alpha" if membership_stat == "cp_upper"
                                                 else "(n_violated + n_unknown) / S <= alpha"),
                        "m1_dm_off_is_entry": bool(m1_dm_off_is_entry),
                        "members": members,
                        "headline": {
                            "membership_event": ("main_reference (Table 5-1 rows judged only inside the primary domain)"
                                                 if main_event == "main_reference" else main_event),
                            "t51_primary_not_assessable_share_min": min(shares) if shares else None,
                            "t51_primary_not_assessable_share_max": max(shares) if shares else None,
                            "n_members_t51_verdicts_used_out_of_domain": len(alt),
                            "note": ("out-of-domain states cannot be judged by Table 5-1; they enter the main event's "
                                     "rate_upper unless another member is violated (FIX3_BC, C-1)")},
                        "n_members_t51_verdicts_used_out_of_domain": len(alt),
                        "coverage_t51_verdicts_used_out_of_domain": (len(alt) / len(att)) if att else None,
                        "members_t51_verdicts_used_out_of_domain": alt,
                        "sensitivity_lines": sens_rows,
                        "rule": C.get("comparable_set_rule", ABLATION_CONFIG["comparable_set_rule"]),
                        "empty_set_meaning": ("no declared method in any cell met the main reference event at this alpha "
                                              "on this world; costs are not compared (not a whole-space statement); "
                                              "see per_method_report.csv for each method's minimum attainable "
                                              "main-event risk and the frontier diagnostic")})
    for r in rows:
        mem = r.pop("_member", None)
        alt_mem = r.pop("_member_alt", None)
        a = r.get("target_alpha")
        r["meets_main_reference_alpha_point"] = (None if a is None or r.get(rate_key) is None
                                                 else bool(r[rate_key] <= a))
        r["meets_main_reference_alpha_cp_upper"] = (None if a is None or r.get(cp_key) is None
                                                    else bool(r[cp_key] <= a))
        r["in_comparable_set"] = None if mem is None else ";".join(f"alpha={k}:{v}" for k, v in sorted(mem.items()))
        if mem is None and (r.get("cell_id"), r.get("label")) in sens and r.get("has_ration"):
            r["in_comparable_set"] = "sensitivity_line_not_an_entry"
        r["in_comparable_set_t51_verdicts_used_out_of_domain"] = (
            None if alt_mem is None else ";".join(f"alpha={k}:{v}" for k, v in sorted(alt_mem.items())))
    return out


# ------------------------------------------------------------------------------------------------
# descriptive reports (official-run plan batch 2): per-method report, scored grid candidates, matched-cost curve
# ------------------------------------------------------------------------------------------------
PER_METHOD_REPORT_COLUMNS = [
    "run_id", "output_label", "evaluation_world", "target_alpha", "cell_id", "sd_arm", "objective_arm", "label",
    "method_id", "method_family", "entry_role", "has_ration", "selection_status", "selected_param", "solver_status",
    "infeasibility_scope", "membership_event", "membership_stat", "q_hash", "rate_lower", "rate_upper", "cp_upper",
    "target_met", "in_comparable_set", "comparable_set_size", "cost_usd_per_head_d", "cost_label",
    "grid_n_candidates", "grid_n_candidates_with_ration", "grid_min_rate_upper", "grid_min_rate_upper_cp_upper",
    "grid_min_rate_upper_cost_usd_per_head_d", "grid_min_rate_upper_param", "grid_min_rate_upper_q_hash",
    "grid_min_label", "frontier_N", "frontier_status", "frontier_m_star_lower_over_N", "frontier_m_star_upper_over_N",
    "frontier_rate_lower", "frontier_rate_upper", "frontier_cp_upper", "frontier_label"]

CANDIDATE_COLUMNS = [
    "run_id", "output_label", "cell_id", "entry_label", "method_id", "method_family", "target_alpha", "param_name",
    "param_value", "fixed_params", "candidate_status", "validation_rate_upper", "validation_screen_statistic",
    "validation_meets_screen", "has_ration", "q_hash", "first_occurrence_of_q_hash", "cost_usd_per_head_d",
    "evaluation_world", "test_stream_id", "n_test", "structural_ok",
    "main_reference_rate_lower", "main_reference_rate_upper", "main_reference_cp_upper",
    f"{MAIN_EVENT_PLAN_DOMAIN}_rate_lower", f"{MAIN_EVENT_PLAN_DOMAIN}_rate_upper", f"{MAIN_EVENT_PLAN_DOMAIN}_cp_upper",
    f"{MAIN_EVENT_T51_IGNORED}_rate_upper", "h0_eleven_rate_upper", "t51_primary_not_assessable_share"]

MATCHED_COST_COLUMNS = [
    "run_id", "output_label", "evaluation_world", "cell_id", "method_family", "membership_event", "status",
    "relative_cost_grid_point", "n_distinct_candidates", "n_candidates_within_cost", "min_rate_upper",
    "min_rate_upper_relative_cost", "min_rate_upper_q_hash", "min_cp_upper", "min_cp_upper_relative_cost", "label"]


def method_family(method_id: str, label: str) -> str:
    """Family of a method entry for the descriptive curves: M1 split by apply_to_dm, M2 split by N."""
    lab = str(label)
    if method_id == "M1_safety_margin":
        return f"{method_id}[{M1_DM_OFF_MARKER if M1_DM_OFF_MARKER in lab else 'apply_to_dm=True'}]"
    if method_id == "M2_joint_chance_saa" and "[N=" in lab:
        return f"{method_id}[N={lab.split('[N=', 1)[1].split(']', 1)[0].split(',', 1)[0]}]"
    return str(method_id)


def _label_N(label: str) -> Optional[int]:
    lab = str(label)
    if "M2[N=" not in lab:
        return None
    try:
        return int(lab.split("M2[N=", 1)[1].split(",", 1)[0].split("]", 1)[0])
    except ValueError:
        return None


def candidate_rations(runs: list, score: Any, *, cell_id: str,
                      extra_selections: Any = ()) -> list[dict[str, Any]]:
    """Every declared grid candidate of the method entries in ``runs``, each distinct ration scored once.

    Official-run plan batch 2 (descriptive only): the candidates are read from the finished selections (M0 = its single
    ration; M1 / M2 / M3a = ``selection.candidates``; M3b = every per-k selection of ``extra_selections`` -- the
    ``selection_sink`` of ``run_dev_case_v1.run_method_block`` -- so the whole (k, Gamma) grid is covered).  Diagnostic runs
    (``:DIAG-``) are not candidates.  ``score(decision)`` returns ``{evaluation_world: {"q_hash", "cost_usd_per_head_d",
    ...reference columns}}`` (the driver's :func:`reference_scorer`); it is called once per distinct ration (dedupe by
    the q vector, i.e. by q_hash) and the result is shared by every occurrence.  One row per (occurrence, evaluation
    world); occurrences without a ration keep their status and have no world.  The private key ``_q`` carries the
    ration for the restricted table.  Nothing in the selection or membership code reads these rows."""
    extra_by_entry: dict[str, list] = {}
    for e in extra_selections or ():
        extra_by_entry.setdefault(e["entry_label"], []).append(e)
    occ: list[dict[str, Any]] = []
    for r in runs:
        name = r.spec.name
        if ":DIAG-" in name:
            continue
        a = _alpha_of(name)
        base = {"cell_id": cell_id, "entry_label": name, "method_id": r.spec.method_id,
                "method_family": method_family(r.spec.method_id, name), "target_alpha": a}
        if r.selection is None:
            occ.append({**base, "param_name": None, "param_value": None, "fixed_params": None,
                        "candidate_status": None if r.result is None else str(r.result.status),
                        "validation_rate_upper": None, "validation_screen_statistic": None,
                        "validation_meets_screen": None, "_result": r.result})
            continue
        sels = [(e["fixed_params"], e["selection"]) for e in extra_by_entry.get(name, [])] or [(None, r.selection)]
        for fixed, sel in sels:
            for c in sel.candidates:
                occ.append({**base, "param_name": sel.param_name, "param_value": c.value,
                            "fixed_params": None if not fixed else json.dumps(fixed, sort_keys=True),
                            "candidate_status": str(c.status), "validation_rate_upper": c.rate_upper,
                            "validation_screen_statistic": c.screen_statistic, "validation_meets_screen": c.meets_screen,
                            "_result": c.solve_result})
    cache: dict[tuple, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for o in occ:
        res = o.pop("_result")
        dec = getattr(res, "decision", None) if res is not None else None
        has = bool(res is not None and getattr(res, "has_solution", False) and dec is not None)
        o["has_ration"] = has
        if not has:
            rows.append({**o, "q_hash": None, "cost_usd_per_head_d": None, "evaluation_world": None})
            continue
        q = np.asarray(dec.q_as_fed, dtype=float)
        key = (tuple(dec.ingredient_ids), q.tobytes())
        first = key not in cache
        if first:
            cache[key] = score(dec)
        for world, blk in cache[key].items():
            rows.append({**o, **blk, "evaluation_world": world, "first_occurrence_of_q_hash": first,
                         "_q": {f"q_{iid}": float(v) for iid, v in zip(dec.ingredient_ids, q)}})
    return rows


def reference_scorer(ctx: dict[str, Any]) -> Any:
    """``score(decision)`` for :func:`candidate_rations`: the one reference evaluator on every evaluation world's test
    stream (the same call as the endpoint rows).  An official context (batch 3) names its evaluation worlds (the five SD
    points) in ``ctx["evaluation_worlds"]``; otherwise the development worlds of ``ABLATION_CONFIG``."""
    worlds = list(ctx.get("evaluation_worlds") or ABLATION_CONFIG["evaluation"]["evaluation_worlds"])

    def score(dec: Any) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for sd in worlds:
            test = ctx["worlds"][sd]["draws"]["test"]
            res = evaluate_reference(dec, test, ctx["ref"])
            out[sd] = {"q_hash": res.identity["q_hash"], "cost_usd_per_head_d": res.summary()["cost"],
                       "test_stream_id": test.stream_id, "n_test": res.n_states, **reference_block(res)}
        return out
    return score


def _frontier_for(label: str, cell_id: str, world: str, frontier_rows: list[dict[str, Any]],
                  diagnostics: dict[str, Any], rate_key: str, cp_key: str, lo_key: str) -> dict[str, Any]:
    pts = list((diagnostics.get(cell_id) or {}).get("frontier_points") or [])
    mv = list((diagnostics.get(cell_id) or {}).get("min_violation") or [])
    n_lab = _label_N(label)
    ns = sorted({int(p["N"]) for p in pts} | {int(m["N"]) for m in mv})
    N = n_lab if n_lab is not None else (ns[-1] if ns else None)
    out: dict[str, Any] = {"frontier_N": N, "frontier_status": None, "frontier_m_star_lower_over_N": None,
                           "frontier_m_star_upper_over_N": None, "frontier_rate_lower": None, "frontier_rate_upper": None,
                           "frontier_cp_upper": None, "frontier_label": None}
    if N is None:
        out["frontier_status"] = "no_min_violation_diagnostic"
        return out
    m = next((x for x in mv if int(x["N"]) == N), None)
    p = next((x for x in pts if int(x["N"]) == N), None)
    lo = (m or {}).get("min_violations_lower", (p or {}).get("m_star_lower"))
    up = (m or {}).get("min_violations_upper", (p or {}).get("m_star_upper"))
    out["frontier_m_star_lower_over_N"] = None if lo is None else lo / N
    out["frontier_m_star_upper_over_N"] = None if up is None else up / N
    fr = next((r for r in frontier_rows if r["cell_id"] == cell_id and _label_N(r["label"]) == N
               and r.get("evaluation_world") == world and r.get("has_ration")), None)
    if fr is None:
        out["frontier_status"] = "no_frontier_ration" if p is None else f"frontier_{p.get('status')}_no_ration"
        return out
    out.update({"frontier_status": p.get("status") if p else "scored", "frontier_rate_lower": fr.get(lo_key),
                "frontier_rate_upper": fr.get(rate_key), "frontier_cp_upper": fr.get(cp_key),
                "frontier_label": "diagnostic frontier point (training scenarios only); not a method, never a member"})
    return out


def per_method_report(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], *,
                      diagnostics: Optional[dict[str, Any]] = None, membership_stat: str = "rate_upper",
                      main_event: str = "main_reference", m1_dm_off_is_entry: bool = True,
                      config: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """One row per (evaluation world, alpha, method entry) -- the report that stays informative when the comparable set
    is empty (official-run plan §1 and batch 2; D-533 "空集合时报告最低可达风险与前沿").

    Per entry: the selected ration's main-event rate pair [rate_lower, rate_upper], p-bar (one-sided CP upper bound),
    ``target_met`` = p-bar <= alpha, membership by the declared rule (``membership_stat`` / ``main_event`` as in
    :func:`comparable_sets`), its cost with ``cost_label`` = ``comparable_set_member`` or ``not_a_comparable_cost``;
    the minimum main-event rate_upper over the entry's declared grid candidates (``candidate_rows``; lowest cost on
    ties) with that candidate's cost (labelled descriptive); and the frontier diagnostic of the entry's cell (for M2
    the entry's N, otherwise the largest N): m*/N bounds of the minimum-violation MILP and the frontier ration's
    main-event rates.  The frontier is never an entry.  M0 (no alpha) is reported at every alpha.  ``config`` (batch 3):
    the official configuration (alphas, evaluation worlds); ``None`` = ``ABLATION_CONFIG``."""
    C = ABLATION_CONFIG if config is None else config
    alphas = [C["alphas"]["primary"]] + list(C["alphas"]["supplementary"])
    worlds = list(C["evaluation"]["evaluation_worlds"])
    rate_key, cp_key, lo_key = f"{main_event}_rate_upper", f"{main_event}_cp_upper", f"{main_event}_rate_lower"
    stat_key = _stat_key(rate_key, cp_key, membership_stat)
    diagnostics = diagnostics or {}
    method_rows = [r for r in rows if r["ration_kind"] == "method"]
    frontier_rows = [r for r in rows if r["ration_kind"] == "diagnostic_frontier"]
    entries: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in method_rows:
        entries.setdefault((r["cell_id"], r["label"]), []).append(r)
    cands: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in candidate_rows:
        cands.setdefault((c["cell_id"], c["entry_label"]), []).append(c)
    out: list[dict[str, Any]] = []
    for world in worlds:
        for a in alphas:
            keys = sorted(k for k in entries if _alpha_of(k[1]) in (None, a))
            rep: list[dict[str, Any]] = []
            for key in keys:
                rs = entries[key]
                r = next((x for x in rs if x.get("evaluation_world") == world), None) or rs[0]
                role = "sensitivity_line" if (not m1_dm_off_is_entry and is_sensitivity_line(key[1])) else "entry"
                has = bool(r.get("has_ration")) and r.get("evaluation_world") == world
                p_bar = r.get(cp_key) if has else None
                member = bool(role == "entry" and has and _meets(r, a, stat_key))
                row = {"evaluation_world": world, "target_alpha": a, "cell_id": key[0], "label": key[1],
                       "sd_arm": r.get("sd_arm"), "objective_arm": r.get("objective_arm"), "method_id": r.get("method_id"),
                       "method_family": method_family(r.get("method_id"), key[1]), "entry_role": role,
                       "has_ration": has, "selection_status": r.get("selection_status"),
                       "selected_param": r.get("selected_param"), "solver_status": r.get("solver_status"),
                       "infeasibility_scope": r.get("infeasibility_scope"), "membership_event": main_event,
                       "membership_stat": membership_stat, "q_hash": r.get("q_hash") if has else None,
                       "rate_lower": r.get(lo_key) if has else None, "rate_upper": r.get(rate_key) if has else None,
                       "cp_upper": p_bar, "target_met": None if p_bar is None else bool(p_bar <= a),
                       "in_comparable_set": member,
                       "cost_usd_per_head_d": r.get("cost_usd_per_head_d") if has else None,
                       "cost_label": (None if not has else "comparable_set_member" if member
                                      else "not_a_comparable_cost")}
                gc = [c for c in cands.get(key, []) if c.get("evaluation_world") in (world, None)]
                scored = [c for c in gc if c.get("has_ration") and c.get("evaluation_world") == world
                          and c.get("structural_ok") is not False and c.get(rate_key) is not None]
                n_occ = len({(c.get("param_name"), repr(c.get("param_value")), c.get("fixed_params")) for c in gc})
                row.update({"grid_n_candidates": n_occ,
                            "grid_n_candidates_with_ration": len({(c.get("param_name"), repr(c.get("param_value")),
                                                                   c.get("fixed_params")) for c in scored})})
                if scored:
                    best = min(scored, key=lambda c: (c[rate_key], c.get("cost_usd_per_head_d") or math.inf))
                    prm = None if best.get("param_name") is None else f"{best['param_name']}={best['param_value']}"
                    if best.get("fixed_params"):
                        prm = f"{prm} {best['fixed_params']}"
                    row.update({"grid_min_rate_upper": best[rate_key], "grid_min_rate_upper_cp_upper": best.get(cp_key),
                                "grid_min_rate_upper_cost_usd_per_head_d": best.get("cost_usd_per_head_d"),
                                "grid_min_rate_upper_param": prm, "grid_min_rate_upper_q_hash": best.get("q_hash"),
                                "grid_min_label": ("minimum attainable main-event rate_upper over the declared grid "
                                                   "(descriptive; its cost is not a comparable cost)")})
                else:
                    row.update({"grid_min_rate_upper": None, "grid_min_rate_upper_cp_upper": None,
                                "grid_min_rate_upper_cost_usd_per_head_d": None, "grid_min_rate_upper_param": None,
                                "grid_min_rate_upper_q_hash": None,
                                "grid_min_label": "no scored grid candidate with a ration (not a whole-space statement)"})
                row.update(_frontier_for(key[1], key[0], world, frontier_rows, diagnostics, rate_key, cp_key, lo_key))
                rep.append(row)
            n_set = sum(1 for x in rep if x["in_comparable_set"])
            for x in rep:
                x["comparable_set_size"] = n_set
            out += rep
    return out


def matched_cost_curve(candidate_rows: list[dict[str, Any]], world: str, *, main_event: str = "main_reference",
                       relative_grid: Optional[list[float]] = None) -> list[dict[str, Any]]:
    """Descriptive "constraint satisfaction at matched cost" curve (official-run plan batch 2; contract T8.1 ->
    descriptive): per cell and method family, on the declared relative cost grid ``c = cost_M0 x (1 + r)`` (the cell's
    M0 ration), the lower envelope over the family's scored grid candidates (distinct q_hash, planning rows met) with
    cost <= c of the main-event rate_upper and, separately, of p-bar.  Non-increasing in c by construction.  Not a
    selection and not a comparable-set cost comparison; costs appear only relative to M0."""
    grid = sorted(float(x) for x in (relative_grid if relative_grid is not None
                                     else ABLATION_CONFIG["matched_cost_curve"]["relative_cost_grid"]))
    rate_key, cp_key = f"{main_event}_rate_upper", f"{main_event}_cp_upper"
    scored = [c for c in candidate_rows if c.get("evaluation_world") == world and c.get("has_ration")
              and c.get("structural_ok") is not False and c.get(rate_key) is not None
              and c.get("cost_usd_per_head_d") is not None]
    out: list[dict[str, Any]] = []
    label = ("descriptive matched-cost curve (T8.1 -> descriptive): lower envelope over declared grid candidates; not a "
             "selection, not a comparable-set comparison; cost relative to the cell's M0 ration")
    for cell in sorted({c["cell_id"] for c in scored}):
        mine = [c for c in scored if c["cell_id"] == cell]
        m0 = next((c for c in mine if c["method_id"] == "M0_nominal"), None)
        fams = sorted({c["method_family"] for c in mine})
        for fam in fams:
            uniq: dict[str, dict[str, Any]] = {}
            for c in mine:
                if c["method_family"] == fam:
                    uniq.setdefault(c["q_hash"], c)
            base = {"evaluation_world": world, "cell_id": cell, "method_family": fam, "membership_event": main_event,
                    "n_distinct_candidates": len(uniq), "label": label}
            if m0 is None or not m0["cost_usd_per_head_d"]:
                out.append({**base, "status": "no_M0_cost_in_cell"})
                continue
            c0 = float(m0["cost_usd_per_head_d"])
            rel = {h: float(c["cost_usd_per_head_d"]) / c0 - 1.0 for h, c in uniq.items()}
            for g in grid:
                ok = [h for h in uniq if rel[h] <= g + 1e-12]
                row = {**base, "status": "ok" if ok else "no_candidate_within_cost", "relative_cost_grid_point": g,
                       "n_candidates_within_cost": len(ok)}
                if ok:
                    hr = min(ok, key=lambda h: (uniq[h][rate_key], rel[h]))
                    hp = min((h for h in ok if uniq[h].get(cp_key) is not None),
                             key=lambda h: (uniq[h][cp_key], rel[h]), default=None)
                    row.update({"min_rate_upper": uniq[hr][rate_key], "min_rate_upper_relative_cost": rel[hr],
                                "min_rate_upper_q_hash": hr,
                                "min_cp_upper": None if hp is None else uniq[hp][cp_key],
                                "min_cp_upper_relative_cost": None if hp is None else rel[hp]})
                out.append(row)
    return out


# ------------------------------------------------------------------------------------------------
# writing
# ------------------------------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> Path:
    with open(path, "x", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    return path


def _write_json(path: Path, obj: Any) -> Path:
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    return path


def redact_residuals(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        if r.get("constraint_id") in DRV.REDACT_TEST_MARGINS:
            for f in ("mean_deficit_given_violation", "max_deficit", "margin_q01", "margin_q05", "margin_q50"):
                r[f] = "restricted"
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------
# output routing and the reproduction check
# ------------------------------------------------------------------------------------------------
PENDING = "<pending>"
PILOT_RUN = "pilot-20260924T205621Z-93d8654c"
#: cells that repeat a declared block of the pilot run: (cell id, pilot tag, public method summary file)
PILOT_EQUIVALENT = (("SDH0_FULL11", "H0", "method_summary.csv"), ("SDS2_PART6P5", "S2", "s2_method_summary.csv"))


def output_route(mode: str, manifest_same: bool, freeze: str) -> dict[str, Any]:
    """Where the output goes and how it is labelled.  Only a full run whose code/config manifest was complete and did not
    change from import to end and whose specification equals the git-anchored frozen pin (``matches_frozen_anchored``:
    pin tracked, HEAD = pin commit, clean work tree; FIX3_BC, C-3) is ``development`` (public tables in results/pilot/,
    restricted in data/restricted_local/pilot/); everything else is ``debug`` under data/restricted_local/debug/ (F.1)."""
    if mode == "full" and manifest_same and freeze == "matches_frozen_anchored":
        return {"label": "development", "public_dir": str(Path("results") / "pilot"),
                "restricted_dir": str(Path("data") / "restricted_local" / "pilot"), "run_id_prefix": "",
                "why": ("full run; code/config manifest complete and unchanged from import to end; specification = "
                        "git-anchored frozen pin (HEAD = pin commit, clean work tree)")}
    why = []
    if mode != "full":
        why.append(f"mode {mode} (pipeline check at toy size)" if mode == "tiny" else f"mode {mode}")
    if not manifest_same:
        why.append("code/config manifest incomplete, or code/specification changed between import and the end of the run")
    if freeze != "matches_frozen_anchored":
        why.append(f"specification vs the frozen pin {FREEZE_PIN.as_posix()}: {freeze}")
    return {"label": "debug", "public_dir": str(DEBUG_ROOT), "restricted_dir": str(DEBUG_ROOT), "run_id_prefix": "debug-",
            "why": "; ".join(why)}


def import_time_check(pre_manifest: dict[str, Any]) -> dict[str, Any]:
    """The Python sources hashed before the project imports equal the start-of-run manifest (FIX3_BC, C-3)."""
    man = {e["path"]: e.get("sha256") for e in pre_manifest.get("files", [])}
    differs = sorted(k for k, v in IMPORT_TIME_SOURCE_SHA256.items() if man.get(k) != v)
    return {"status": "consistent" if not differs else "changed_between_import_and_manifest",
            "n_files": len(IMPORT_TIME_SOURCE_SHA256), "differs": differs[:50],
            "dirs": list(IMPORT_TIME_SOURCE_DIRS)}


def check_authorisation(path: Path, repo: Path = REPO, *, scope: str = "run_endpoint_ablation",
                        flag: str = "--full") -> dict[str, Any]:
    """``--full`` (``--official``: batch 4) needs a user-written authorisation file that lists this host (FIX3_BC, C-4).

    Required keys: ``authorised_hosts`` (list of host names, compared with ``socket.gethostname()``),
    ``authorised_by``, ``authorised_on`` and ``scope`` containing ``scope`` (``run_endpoint_ablation`` for ``--full``,
    ``run_official_v2`` for ``--official``).  A missing or non-matching file stops the run before anything else is
    read.  The file is restricted (never written by an agent); the run record keeps its sha256 and the matched host
    only."""
    p = path if path.is_absolute() else repo / path
    if not p.is_file():
        raise SystemExit(f"{flag}: no authorisation file {path} (the user writes it for an authorised compute host; "
                         "nothing was read or written)")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"{flag}: authorisation file {path} is not valid JSON ({exc})")
    if not isinstance(doc, dict):
        raise SystemExit(f"{flag}: authorisation file {path} is not a JSON object")
    host = socket.gethostname()
    missing = [k for k in ("authorised_hosts", "authorised_by", "authorised_on", "scope") if not doc.get(k)]
    if missing:
        raise SystemExit(f"{flag}: authorisation file {path} lacks {missing}")
    if scope not in str(doc["scope"]):
        raise SystemExit(f"{flag}: authorisation file {path} does not cover {scope}")
    if host not in [str(h) for h in doc["authorised_hosts"]]:
        raise SystemExit(f"{flag}: host {host!r} is not listed in {path}; the run is refused on this host")
    return {"file": path.as_posix(), "sha256": file_sha256(p), "host": host, "platform": platform.system(),
            "authorised_by": doc["authorised_by"], "authorised_on": doc["authorised_on"], "scope_checked": scope}


class DryRunCounters:
    """Measured counts of solves, draws and writes while the dry run executes (FIX3_BC, C-4): the solve, sampling and
    file-writing entry points are wrapped for the duration of the ``with`` block and restored afterwards.

    Wrapped: optimisation methods returned by ``get_method`` (this driver, run_dev_case_v1 and official_v2),
    ``run_method_block``,
    ``min_violations``, ``s2_consistency``, scipy ``linprog`` / ``milp`` (the module attributes and the names bound in
    the engine's HiGHS wrappers), ``RandomStreams.generator``, ``EnergyColumnModel.draw_world``, ``open`` in a writing
    mode (``builtins`` and ``io``) and ``os.mkdir`` / ``os.makedirs``."""

    def __init__(self) -> None:
        self.counts = {"solves": 0, "draws": 0, "files_written": 0, "dirs_created": 0}
        self.calls: dict[str, int] = {}
        self._undo: list[tuple[Any, str, Any]] = []

    def _patch(self, obj: Any, name: str, kind: str) -> None:
        if not hasattr(obj, name):
            return
        orig = getattr(obj, name)
        counters = self
        label = f"{getattr(obj, '__name__', type(obj).__name__)}.{name}"

        def wrapper(*a: Any, **k: Any) -> Any:
            counters.counts[kind] += 1
            counters.calls[label] = counters.calls.get(label, 0) + 1
            return orig(*a, **k)
        self._undo.append((obj, name, orig))
        setattr(obj, name, wrapper)

    def _patch_open(self, obj: Any, name: str) -> None:
        orig = getattr(obj, name)
        counters = self

        def wrapper(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
            if any(c in str(mode) for c in "wxa+"):
                counters.counts["files_written"] += 1
                counters.calls[f"{name}:{mode}"] = counters.calls.get(f"{name}:{mode}", 0) + 1
            return orig(file, mode, *a, **k)
        self._undo.append((obj, name, orig))
        setattr(obj, name, wrapper)

    def _wrap_get_method(self, namespace: dict) -> None:
        orig = namespace["get_method"]
        counters = self

        def get_method_counted(method_id: str) -> Any:
            fn = orig(method_id)

            def run(*a: Any, **k: Any) -> Any:
                counters.counts["solves"] += 1
                counters.calls[f"method:{method_id}"] = counters.calls.get(f"method:{method_id}", 0) + 1
                return fn(*a, **k)
            return run
        self._undo.append((namespace, "get_method", orig))
        namespace["get_method"] = get_method_counted

    def __enter__(self) -> "DryRunCounters":
        import scipy.optimize as _so
        from ration_reliability.optimization import chance_saa as _saa, highs as _hs
        self._wrap_get_method(globals())
        self._wrap_get_method(vars(DRV))
        self._wrap_get_method(vars(OV2))                  # batch 3: the official diagnostics solve through official_v2
        for obj, name in ((DRV, "run_method_block"), (DRV, "min_violations"), (DRV, "s2_consistency"),
                          (_so, "linprog"), (_so, "milp"), (_hs, "linprog"), (_hs, "milp"), (_saa, "milp"),
                          (DRV, "linprog"), (DRV, "milp")):
            self._patch(obj, name, "solves")
        self._patch(RandomStreams, "generator", "draws")
        self._patch(E.EnergyColumnModel, "draw_world", "draws")
        self._patch_open(builtins, "open")
        self._patch_open(io, "open")
        self._patch(os, "mkdir", "dirs_created")
        self._patch(os, "makedirs", "dirs_created")
        return self

    def __exit__(self, *exc: Any) -> None:
        for obj, name, orig in reversed(self._undo):
            if isinstance(obj, dict):
                obj[name] = orig
            else:
                setattr(obj, name, orig)
        self._undo.clear()

    def record(self) -> dict[str, Any]:
        return {**self.counts, "calls": dict(sorted(self.calls.items())), "method": "measured (wrapped entry points)"}


def reproduction_vs_pilot(ends: list[dict[str, Any]], mode: str, repo: Path = REPO) -> dict[str, Any]:
    """Full run only: the two cells that repeat a pilot block (H0, S2) against the pilot's public method summaries --
    status and cost per method label.  A record, not a gate (the pilot ran on earlier code; a difference is reported)."""
    if mode != "full":
        return {"status": "not_applicable", "reason": f"mode {mode}: sizes differ from the pilot run"}
    out: dict[str, Any] = {"pilot_run": PILOT_RUN, "cells": {}}
    for cell, tag, fname in PILOT_EQUIVALENT:
        p = repo / "results" / "pilot" / PILOT_RUN / fname
        if not p.is_file():
            out["cells"][cell] = {"status": "pilot_summary_missing", "file": str(p.relative_to(repo))}
            continue
        saved = {r["label"].split(":", 1)[1]: r for r in csv.DictReader(p.open(encoding="utf-8"))
                 if r["label"].startswith(tag + ":") and ":DIAG-" not in r["label"]}
        mine = {r["label"].split(":", 1)[1]: r for r in ends if r["cell_id"] == cell and r["ration_kind"] == "method"
                and (not r.get("evaluation_world") or r.get("own_world"))}
        rows = []
        for key in sorted(set(saved) | set(mine)):
            a, b = mine.get(key), saved.get(key)
            if a is None or b is None:
                rows.append({"method": key, "present_in": "ablation" if b is None else "pilot"})
                continue
            st_a = a.get("solver_status") if a.get("has_ration") else f"selection_{a.get('selection_status')}"
            ca, cb = _f(a.get("cost_usd_per_head_d")), _f(b.get("cost_usd_per_head_d"))
            rows.append({"method": key, "status_ablation": st_a, "status_pilot": b.get("status"),
                         "status_equal": st_a == b.get("status"),
                         "cost_abs_diff": None if ca is None or cb is None else abs(ca - cb)})
        out["cells"][cell] = {"status": "compared", "rows": rows,
                              "all_equal": all(r_.get("status_equal") and (r_.get("cost_abs_diff") in (None, 0.0) or
                                                                          r_["cost_abs_diff"] <= 1e-9)
                                               for r_ in rows if "status_equal" in r_)}
    return out


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def official_plan_main() -> int:
    """``--official-plan`` (official-run plan batch 3): the official context without a draw (reference problem v2, the
    five SD-point worlds, the v2 reference specification, the registry check of the five points) and
    :func:`official_plan`, inside :class:`DryRunCounters` -- solves, draws, files written and directories created are
    measured and printed.  No root value, ratio or restricted value is printed (fingerprints, counts, ids, labels)."""
    t_start = time.perf_counter()
    cfg = OV2.OFFICIAL_CONFIG
    pre_manifest = code_manifest(REPO)
    imp = import_time_check(pre_manifest)
    lock = check_environment_against_lock(REPO / ENVIRONMENT_LOCK_FILE, current=environment_fingerprint())
    counters = DryRunCounters()
    with counters:
        ctx = build_context(dict(cfg["streams"]), draw=False, official=cfg)
        op = official_plan(ctx, cfg)
    meas = counters.record()
    checks = {"preflight": "READY", "environment_lock": lock.get("status"), "import_time_sources": imp,
              "code_manifest_complete_at_start": bool(pre_manifest.get("complete")),
              "sd_registry_all_points": ctx["sd_registry"],
              "reference_table": {"path": OV2.REFERENCE_CSV_V2.as_posix(), "table_version": ctx["table"].table_version,
                                  "premise_row": ctx["table"].premise_row_id,
                                  "main_reference": list(ctx["table"].main_reference_ids)},
              "reference_spec_fingerprint": ctx["ref"].fingerprint(),
              "v2_build_checks": ctx["v2_record"]["checks"],
              "validators": {k: {"ok": v.ok, "n_pending": v.n_pending, "n_assumptions": v.n_assumptions}
                             for k, v in ctx["validators"].items()},
              "worlds": {sd: {"spec_id": wd["spec"].spec_id, "spec_fingerprint": wd["spec"].fingerprint(),
                              "world_fingerprint": wd["w"].fingerprint()} for sd, wd in ctx["worlds"].items()},
              "reserved_roots": ("not opened (--official-plan never draws; a reserved root is opened only by an "
                                 "official job after require_protocol_frozen(); roots appear only as indices k)")}
    print(json.dumps({"official_plan": True, "files_written": meas["files_written"], "draws": meas["draws"],
                      "solves": meas["solves"], "dirs_created": meas["dirs_created"], "measured_counts": meas,
                      "checks": checks, "matrix": op["matrix"], "jobs": op["jobs"],
                      "compute_estimate": op["compute_estimate"], "plan_official": op["plan"],
                      "status": ("declared; " + ("protocol freeze record present (check --official --dry-run)" if
                                                 (REPO / OV2.PROTOCOL_FREEZE).is_file() else
                                                 "not frozen (protocol freeze = batch 5)")),
                      "elapsed_s": time.perf_counter() - t_start}, indent=1, ensure_ascii=False,
                     default=_json_default))
    return 0


# ------------------------------------------------------------------------------------------------
# official run v2 (official-run plan batch 4: §0 F3 / F5, §2 "B4", §5, §7)
# ------------------------------------------------------------------------------------------------
#: rehearsal sizes of the diagnostics (tiny opt stream: N = 16 only)
REHEARSAL_DIAGNOSTICS = {"Ns": [16], "alphas": [0.05], "time_limits": {"16": 30.0}, "m2_time_limit_s": 60.0}
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
#: the reference event that equals each arm's own training event (checked on every own-world row at the merge)
TRAINING_EVENT_COLUMN = {"FULL11": "h0_eleven_rate_upper", "PART6P5": "s2_six_rate_upper",
                         "MAIN9": "main9_training_event_rate_upper"}
_EVENT_STATS = ("n_violated", "n_unknown", "rate_lower", "rate_upper", "cp_upper", "mc_se")
#: the public endpoint table of an official run: the development columns + the plan-level main event (D-535) and its
#: companion, per root
OFFICIAL_ENDPOINT_COLUMNS = ["root_k"] + ENDPOINT_COLUMNS + [
    f"{ev}_{st}" for ev in (MAIN_EVENT_PLAN_DOMAIN, "main_reference_plus_cp_hi_plan_domain") for st in _EVENT_STATS
    if f"{ev}_{st}" not in ENDPOINT_COLUMNS]
#: public columns of the scored diagnostic rations (no cost, no h value: restricted, official_v2 restricted_fields)
DIAG_PUBLIC_COLUMNS = ["root_k", "run_id", "output_label", "label", "diagnostic_id", "diag_lp", "diag_training_world",
                       "diag_N", "diag_alpha_train", "ration_kind", "flag", "comparable_set_member", "q_hash",
                       "first_occurrence_of_q_hash", "evaluation_world", "test_stream_id", "n_test"] + [
    c for c in OFFICIAL_ENDPOINT_COLUMNS if c.startswith(("main_reference", "h0_eleven", "s2_six", "excluded_five",
                                                          "main9_training_event", MAIN_EVENT_T51_IGNORED, "energy_",
                                                          "t51_", "n_analysis", "n_support", "structural_ok"))]
DIAG_RESTRICTED_KEYS = ("cost_usd_per_head_d", "diag_h_canonical")


def _official_roots(values: Optional[list[int]]) -> list[int]:
    """``--root-k``: distinct root indices 0, 1, 2 -- never a seed (a development root is refused)."""
    n = int(OV2.OFFICIAL_CONFIG["roots"]["n_reserved_roots"])
    if not values:
        raise SystemExit("--official needs --root-k with one or more root indices 0, 1, 2")
    ks = [int(v) for v in values]
    if any(not 0 <= k < n for k in ks) or len(set(ks)) != len(ks):
        raise SystemExit(f"--root-k takes distinct root indices 0..{n - 1} (never a seed; a development root is "
                         "refused in official mode); nothing was read or drawn")
    return ks


def official_gate(repo: Path = REPO, roots_k: Any = (0, 1, 2)) -> tuple[int, dict[str, Any], Any]:
    """``official_v2.require_protocol_frozen`` inside :class:`DryRunCounters` (draws / solves / writes measured):
    ``(0, report, token)`` when the protocol is frozen and anchored, ``(2, report, None)`` otherwise.  The report has no
    token and no root value."""
    counters = DryRunCounters()
    token = None
    with counters:
        try:
            g = OV2.require_protocol_frozen(repo, roots_k=tuple(roots_k))
            token = g.pop("token")
            rep_ = {**g, "token": token.public_record()}
            code = 0
        except OV2.FreezeGateError as exc:
            rep_ = {"ok": False, "status": "refused", "reasons": exc.reasons, "facts": exc.checks}
            code = 2
    meas = counters.record()
    rep_.update({"measured_counts": meas, "draws": meas["draws"], "solves": meas["solves"],
                 "files_written": meas["files_written"], "dirs_created": meas["dirs_created"]})
    return code, rep_, token


def _aliases(roots_k: Any, mode: str) -> dict[int, int]:
    """Reserved root k -> value, for aliasing public files (official only; the rehearsal uses a development root)."""
    if mode != "official":
        return {}
    roots = ST.reserved_formal_roots()
    return {int(k): roots[int(k)] for k in roots_k}


def _pub(obj: Any, aliases: dict[int, int]) -> Any:
    return OV2.alias_reserved_roots(obj, aliases) if aliases else obj


def redact_official_residuals(rows: list[dict]) -> list[dict]:
    """Public residual rows of an official run: the development redaction plus the premise planning row's margins."""
    out = []
    for r in redact_residuals(rows):
        if r.get("constraint_id") == OV2.PREMISE_ROW_ID:
            r = dict(r)
            for f in ("mean_deficit_given_violation", "max_deficit", "margin_q01", "margin_q05", "margin_q50"):
                r[f] = "restricted"
        out.append(r)
    return out


def training_event_consistency(ends: list[dict[str, Any]]) -> dict[str, Any]:
    """Every own-world method row: the cell's training event on its own test stream (public evaluator on the cell
    problem) equals the matching reference event (MAIN9: ``main9_training_event``, v2 premise row included in the
    arm; FULL11: ``h0_eleven``; PART6P5: ``s2_six``), to 1e-12."""
    rows = []
    for r in ends:
        if not r.get("own_world") or r.get("ration_kind") != "method" or r.get("objective_arm") not in \
                TRAINING_EVENT_COLUMN:
            continue
        a, b = _f(r.get("training_event_rate_upper_own_world")), _f(r.get(TRAINING_EVENT_COLUMN[r["objective_arm"]]))
        rows.append({"cell_id": r.get("cell_id"), "label": r.get("label"), "objective_arm": r["objective_arm"],
                     "equal": a is not None and b is not None and abs(a - b) <= 1e-12})
    by_arm: dict[str, dict[str, int]] = {}
    for x in rows:
        d = by_arm.setdefault(x["objective_arm"], {"n": 0, "n_equal": 0})
        d["n"] += 1
        d["n_equal"] += int(x["equal"])
    return {"rule": "training event (own world, public evaluator) == matching reference event, |diff| <= 1e-12",
            "by_arm": by_arm, "all_equal": all(x["equal"] for x in rows), "n_rows": len(rows),
            "not_equal": [x for x in rows if not x["equal"]][:20]}


def _strip_decisions(rations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rr in rations:
        d = {k: v for k, v in rr.items() if k != "decision"}
        dec = rr.get("decision")
        if dec is not None:
            d["q_as_fed"] = {f"q_{i}": float(q) for i, q in zip(dec.ingredient_ids, dec.q_as_fed)}
        out.append(d)
    return out


def _public_t45(t45: dict[str, Any]) -> dict[str, Any]:
    """DIAG-T45 without its restricted fields (headroom, delta*, h values, costs, shortfall quantiles, q)."""
    keep_a2 = ("source", "alpha", "factor", "j", "n", "status", "feasible", "binding", "has_ration", "stage2_status",
               "label", "clipped_at_zero")
    worlds = {sd: {"n_h": w.get("n_h"), "n_feasible": w.get("n_feasible"),
                   "delta_monotone_in_h": w.get("delta_monotone_in_h"),
                   "delta_zero_iff_h_le_A0": w.get("delta_zero_iff_h_le_A0"), "a1_grid": w.get("a1_grid"),
                   "m0_shortfall_status": (w.get("shortfall") or {}).get("m0_status"),
                   "A2": [{k: x.get(k) for k in keep_a2 if k in x} for x in w.get("A2", [])]}
              for sd, w in (t45.get("worlds") or {}).items()}
    return {k: t45.get(k) for k in ("diagnostic_id", "flag", "role", "comparable_set_member", "what", "problem_id",
                                    "energy_row", "relaxed_rows", "headroom_unit", "delta_unit", "planned_dm_row",
                                    "A1_headroom_ge_A0", "n_distinct_lp_rations", "restricted_fields")} | {
        "A0": {"status": (t45.get("A0") or {}).get("status")}, "A1": {"status": (t45.get("A1") or {}).get("status")},
        "worlds": worlds}


def _public_e1(e1: dict[str, Any]) -> dict[str, Any]:
    """DIAG-E1 without costs and q (min-violation counts, statuses and verdicts stay; as in development)."""
    worlds = {}
    for sd, w in (e1.get("worlds") or {}).items():
        per_n = []
        for rec in w.get("per_N", []):
            per_n.append({"N": rec["N"], "min_violation": rec["min_violation"],
                          "M2": [{k: v for k, v in m.items() if k != "cost_usd_per_head_d"} for m in rec["M2"]]})
        worlds[sd] = {"opt_n_draws": w.get("opt_n_draws"), "per_N": per_n}
    return {k: e1.get(k) for k in ("diagnostic_id", "flag", "role", "comparable_set_member", "what", "arm", "energy_row",
                                   "N_ladder", "alphas")} | {"worlds": worlds,
                                                              "restricted_fields": e1.get("restricted_fields")}


def _official_cell_job(ctx: dict[str, Any], job: dict[str, Any], sizes: dict[str, Any], opts: SolverOptions,
                       cfg: dict[str, Any], run_id: str, label: str) -> dict[str, Any]:
    cell = next(c for c in cfg["cells"] if c["cell_id"] == job["cell_id"])
    t0 = time.perf_counter()
    cr = run_cell(ctx, cell, sizes, opts, config=cfg)
    t_eval = time.perf_counter()
    ends, resid, rations, status = cell_rows(ctx, cr, opts, run_id, label, config=cfg)
    t_eval = time.perf_counter() - t_eval
    t_cand = time.perf_counter()
    cand = candidate_rations(cr["runs"], reference_scorer(ctx), cell_id=cell["cell_id"],
                             extra_selections=cr["m3b_per_k"])
    for r in cand:
        r["run_id"], r["output_label"] = run_id, label
    t_cand = time.perf_counter() - t_cand
    return {"kind": "cell", "cell": cell, "problem_id": cr["problem"].problem_id, "ends": ends, "resid": resid,
            "rations": rations, "status": status, "cand": cand, "diag": cr["diag"], "tables": cr["tables"],
            "timings_s": {**cr["timings_s"], "reference_evaluation": t_eval, "candidate_scoring": t_cand,
                          "job_compute": time.perf_counter() - t0}}


def _official_diagnostics_job(ctx: dict[str, Any], job: dict[str, Any], cfg: dict[str, Any], *, rehearsal: bool,
                              run_id: str, label: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    tw = list(cfg["diagnostics"]["training_worlds"])
    tag = f"root{int(job['root_k'])}"
    score = reference_scorer(ctx)
    problem_v2 = ctx["problems"]["FULL11"]
    t45 = OV2.diag_t45_headroom(problem_v2, ctx["lin"], {w: ctx["worlds"][w] for w in tw}, score=score, tag=tag)
    e1_kw = dict(REHEARSAL_DIAGNOSTICS) if rehearsal else {}
    e1 = OV2.diag_e1_energy_single_row(ctx["cfg0_v2"], problem_v2, ctx["worlds"], planned_arm_cfg=planned_arm_cfg,
                                       training_worlds=tw, score=score, tag=tag, mode=ctx["build_mode"], **e1_kw)
    rows = list(t45.get("rows", [])) + list(e1.get("rows", []))
    for r in rows:
        r["run_id"], r["output_label"] = run_id, label
    t45_full = {k: v for k, v in t45.items() if k not in ("rations", "rows")}
    e1_full = {k: v for k, v in e1.items() if k not in ("rations", "rows")}
    t45_full["rations"], e1_full["rations"] = _strip_decisions(t45["rations"]), _strip_decisions(e1["rations"])
    return {"kind": "diagnostics", "training_worlds": tw, "rows": rows, "t45": t45_full, "e1": e1_full,
            "public_summary": {"DIAG-T45": _public_t45(t45), "DIAG-E1": _public_e1(e1),
                               "rule": OV2.DIAGNOSTIC_ROLE, "training_worlds": tw,
                               "evaluation_worlds": cfg["diagnostics"]["evaluation_worlds"],
                               "rehearsal_overrides": e1_kw or None},
            "timings_s": {"job_compute": time.perf_counter() - t0}}


def _job_failed(jr: Path, job_id: str, stage: str, exc: BaseException) -> None:
    """A failed job leaves FAILED.json in its restricted directory (type, stage, short message; no DONE)."""
    try:
        with open(jr / OV2.FAILED_FILE, "x", encoding="utf-8") as fh:
            json.dump({"job_id": job_id, "stage": stage, "error_type": type(exc).__name__,
                       "message": str(exc)[:500], "utc": utc_now()}, fh, indent=1)
    except OSError:
        pass


def official_job_main(run_id: str, job_id: str) -> int:
    """One job process (``--official-job RUN_ID JOB_ID``, launched by :func:`official_v2.run_jobs`): re-runs the freeze
    gate (official) and mints its own token, rebuilds the context and its streams (deterministic: the same root, stream
    names and sizes give the same draws), runs one cell or the diagnostics, writes ``jobs/<job_id>/`` in the public
    and the restricted directory, each with ``SHA256SUMS`` and, last, ``DONE``."""
    rehearsal = run_id.startswith(OV2.REHEARSAL_RUN_PREFIX)
    try:
        route = OV2.official_output_route(run_id, rehearsal=rehearsal)
    except ValueError as exc:
        print(f"--official-job: {exc}", file=sys.stderr)
        return 2
    jp, jr = REPO / route["public_dir"] / "jobs" / job_id, REPO / route["restricted_dir"] / "jobs" / job_id
    spec_p = jr / "job_spec.json"
    if not spec_p.is_file() or not jp.is_dir():
        print(f"--official-job: no job directory / job_spec.json for {job_id} (the parent creates them)", file=sys.stderr)
        return 2
    if (jp / OV2.DONE_FILE).exists() or (jr / OV2.DONE_FILE).exists() or (jr / OV2.FAILED_FILE).exists():
        print(f"--official-job: {job_id} already ran (DONE / FAILED present); a rerun uses a new run id", file=sys.stderr)
        return 3
    spec = json.loads(spec_p.read_text(encoding="utf-8"))
    job, sizes, cfg = spec["job"], dict(spec["sizes"]), OV2.OFFICIAL_CONFIG
    k = int(job["root_k"])
    mode = "rehearsal" if rehearsal else "official"
    started, t0 = utc_now(), time.perf_counter()
    pre_manifest = code_manifest(REPO)
    imp = import_time_check(pre_manifest)
    threads = {v: os.environ.get(v) for v in THREAD_VARS}
    stage = "gate"
    try:
        if rehearsal:
            gate_pub: dict[str, Any] = {"status": "rehearsal: development root; the protocol freeze is not required"}
            rc = {"protocol_sha256": sha_or_none(REPO / DRV.PROTOCOL), "primary_assumption_id":
                  OV2.PRIMARY_ASSUMPTION_ID, "note": "rehearsal: current (unfrozen) protocol.yaml"}
            stage = "context"
            ctx = build_context(sizes, draw=True, official=cfg, seed=int(ABLATION_CONFIG["seed"]), run_context=rc)
        else:
            try:
                g = OV2.require_protocol_frozen(REPO, roots_k=(k,))
            except OV2.FreezeGateError as exc:
                _job_failed(jr, job_id, "gate (before any draw)", exc)
                print(json.dumps({"job_id": job_id, "status": "refused_by_freeze_gate", "reasons": exc.reasons}),
                      file=sys.stderr)
                return 2
            token = g.pop("token")
            gate_pub = {**g, "token": token.public_record()}
            rc = {"protocol_sha256": g["protocol_sha256"], "primary_assumption_id": OV2.PRIMARY_ASSUMPTION_ID}
            stage = "context"
            ctx = build_context(sizes, draw=True, official=cfg, root_k=k, official_token=token, run_context=rc)
        aliases = _aliases([k], mode)
        opts = SolverOptions(time_limit_s=float(cfg["solver"]["time_limit_s"]),
                             mip_rel_gap=float(cfg["solver"]["mip_rel_gap"]))
        stage = "compute"
        if job["kind"] == "cell":
            payload = _official_cell_job(ctx, job, sizes, opts, cfg, run_id, route["label"])
        else:
            payload = _official_diagnostics_job(ctx, job, cfg, rehearsal=rehearsal, run_id=run_id,
                                                label=route["label"])
        reg = ctx["registry"].check()
        if not reg["ok"]:
            raise RuntimeError(f"stream collisions: {len(reg['collisions'])}")
        stage = "write"
        post_manifest = code_manifest(REPO)
        unchanged = bool(pre_manifest.get("complete")) and bool(post_manifest.get("complete")) and \
            pre_manifest["manifest_sha256"] == post_manifest["manifest_sha256"] and imp["status"] == "consistent"
        worlds = {sd: {"spec_id": wd["spec"].spec_id, "spec_fingerprint": wd["spec"].fingerprint(),
                       "world_fingerprint": wd["w"].fingerprint(), "record": wd["record"]}
                  for sd, wd in ctx["worlds"].items()}
        record = {
            "schema": "ration_reliability.official_job_record/1", "run_id": run_id, "job": job, "mode": mode,
            "output_label": route["label"], "started_utc": started, "completed_utc": utc_now(),
            "wall_s": time.perf_counter() - t0, "exit_status": 0, "gate": gate_pub,
            "root": {"root_k": k, "alias": OV2.root_alias(k) if not rehearsal else None,
                     "class": ctx["seed_class"]},
            "sizes": sizes, "thread_env": threads, "pid": os.getpid(), "host": socket.gethostname(),
            "cpu_count": os.cpu_count(), "solver_version": solver_version_string(), "tolerances": opts.to_dict(),
            "code_manifest_at_start": pre_manifest["manifest_sha256"],
            "code_manifest_at_end": post_manifest["manifest_sha256"], "code_unchanged_during_job": unchanged,
            "import_time_sources": imp["status"], "world_fingerprints": worlds,
            "sd_registry_check": ctx["sd_registry"],
            "run_context_injected": ctx["run_context_injected"], "build_mode": ctx["build_mode"],
            "stream_registry": reg, "stream_registry_ok": reg["ok"],
            "rng_streams": {f"{sd}/{kk}": ds.stream_id for sd, wd in ctx["worlds"].items()
                            for kk, ds in wd["draws"].items()},
            "timings_s": payload["timings_s"],
            "n_rows": {key: len(payload.get(key) or []) for key in ("ends", "resid", "rations", "cand", "rows")}}
        for key in ("ends", "resid", "rations", "status", "cand", "rows"):
            for r in payload.get(key) or []:
                r["root_k"] = k
        _write_json(jr / "job_payload.json", payload)
        _write_json(jr / "stream_registry_full.json", {"stream_registry": reg, "rng_streams": record["rng_streams"]})
        _write_json(jp / "job_record.json", _pub(record, aliases))
        if job["kind"] == "cell":
            _write_csv(jp / "endpoint_rows.csv", _pub(payload["ends"], aliases), OFFICIAL_ENDPOINT_COLUMNS)
            _write_json(jp / "solve_status.json", _pub({"cell": payload["cell"], "problem_id": payload["problem_id"],
                                                        "diagnostics": payload["diag"],
                                                        "selection_tables": payload["tables"],
                                                        "per_method": payload["status"],
                                                        "timings_s": payload["timings_s"]}, aliases))
        else:
            _write_json(jp / "diagnostics_summary.json", _pub(payload["public_summary"], aliases))
        OV2.write_sha_manifest(jr)
        OV2.write_done(jr, job_id)
        OV2.write_sha_manifest(jp)
        OV2.write_done(jp, job_id)
    except Exception as exc:  # noqa: BLE001 -- recorded; no DONE; the merge refuses the run
        _job_failed(jr, job_id, stage, exc)
        print(f"job {job_id} failed at stage {stage}: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"job_id": job_id, "status": "done", "mode": mode, "wall_s": time.perf_counter() - t0,
                      "code_unchanged_during_job": unchanged}))
    return 0


def _write_run_dirs(run_id: str, route: dict[str, Any], jobs: list[dict[str, Any]], sizes: dict[str, Any],
                    plan_doc: dict[str, Any], aliases: dict[int, int]) -> tuple[Path, Path]:
    pub, res = REPO / route["public_dir"], REPO / route["restricted_dir"]
    (pub / "jobs").mkdir(parents=True)
    (res / "jobs").mkdir(parents=True)
    (res / "job_logs").mkdir()
    for j in jobs:
        (pub / "jobs" / j["job_id"]).mkdir()
        (res / "jobs" / j["job_id"]).mkdir()
        _write_json(res / "jobs" / j["job_id"] / "job_spec.json",
                    {"run_id": run_id, "job": j, "sizes": sizes, "mode": route["mode"],
                     "official_config_sha256": stable_hash(OV2.OFFICIAL_CONFIG)})
    _write_json(pub / OV2.RUN_PLAN_FILE, _pub(plan_doc, aliases))
    _write_json(res / OV2.RUN_PLAN_FILE, plan_doc)
    return pub, res


def _launch_official_run(*, run_id: str, mode: str, roots: list[int], jobs: list[dict[str, Any]],
                         sizes: dict[str, Any], workers: int, gate: dict[str, Any], authorisation: Any,
                         started: str, pre_manifest: dict[str, Any], lock: dict[str, Any], imp: dict[str, Any],
                         command: str) -> int:
    route = OV2.official_output_route(run_id, rehearsal=(mode == "rehearsal"))
    pub, res = REPO / route["public_dir"], REPO / route["restricted_dir"]
    if pub.exists() or res.exists():
        print(f"{route['public_dir']} or its restricted mirror exists; refusing to overwrite", file=sys.stderr)
        return 3
    aliases = _aliases(roots, mode)
    plan_doc = {
        "schema": "ration_reliability.official_run_plan/1", "run_id": run_id, "mode": mode,
        "output_label": route["label"], "route": route, "started_utc": started, "command": command,
        "roots_k": roots, "root_aliases": {str(k): OV2.root_alias(k) for k in roots} if mode == "official" else None,
        "root_resolution": ("reserved roots k by the label rule, opened in each job after its own freeze gate" if
                            mode == "official" else
                            f"development root {ABLATION_CONFIG['seed']} for the matrix row of root index 0 "
                            "(rehearsal; never a reserved root)"),
        "jobs": jobs, "n_jobs": len(jobs), "sizes": sizes, "workers": workers, "thread_env": dict(OV2.THREAD_ENV),
        "gate": gate, "authorisation": authorisation, "environment_lock": lock.get("status"),
        "code_manifest_at_start": pre_manifest["manifest_sha256"],
        "code_manifest_complete": bool(pre_manifest.get("complete")), "import_time_sources": imp["status"],
        "spec_digest_at_start": spec_fingerprint(REPO)["digest"],
        "freeze_record_sha256": sha_or_none(REPO / OV2.PROTOCOL_FREEZE),
        "official_config_sha256": stable_hash(OV2.OFFICIAL_CONFIG), "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "rules": {"per_root": "tables per root; never pooled (plan §7)",
                  "rerun": "a bug-driven rerun uses the same roots under a new run id; this output is kept (plan §7)",
                  "diagnostics": OV2.DIAGNOSTIC_ROLE}}
    pub, res = _write_run_dirs(run_id, route, jobs, sizes, plan_doc, aliases)
    specs = [{"job_id": j["job_id"], "argv": OV2.job_argv(run_id, j["job_id"])} for j in jobs]
    t0 = time.perf_counter()
    results = OV2.run_jobs(specs, workers, cwd=REPO, log_dir=res / "job_logs")
    failed = [r["job_id"] for r in results if r["exit_status"] != 0]
    _write_json(pub / "RUN_JOBS.json", {"run_id": run_id, "workers": workers, "wall_s": time.perf_counter() - t0,
                                        "results": results, "failed": failed})
    print(json.dumps({"run_id": run_id, "mode": mode, "n_jobs": len(jobs), "failed": failed,
                      "jobs_wall_s": time.perf_counter() - t0}, indent=1))
    if failed:
        print(f"{len(failed)} job(s) failed; no merge (outputs kept; see the restricted job logs / FAILED.json)",
              file=sys.stderr)
        return 4
    return official_merge_main(run_id)


def official_main(args: argparse.Namespace) -> int:
    """``--official --authorised-compute --root-k K [K ...] --workers N`` (plan §2 "B4", F3).  Order: authorisation file
    (scope ``run_official_v2``, this host) -> root indices -> the freeze gate (measured: 0 draws) -> preflight ->
    environment lock -> code manifest -> output directories -> jobs (each job re-runs the gate and only then opens its
    reserved root) -> merge.  Any refusal before the jobs exits 2 with nothing drawn."""
    if not args.authorised_compute:
        print("--official needs --authorised-compute: the official run is for the host the user authorised; nothing "
              "was read or written.", file=sys.stderr)
        return 2
    try:
        authorisation = check_authorisation(Path(args.authorisation_file or OFFICIAL_AUTHORISATION_FILE),
                                            scope=OV2.OFFICIAL_SCOPE, flag="--official")
        roots = _official_roots(args.root_k)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.workers is None or not 1 <= int(args.workers) <= 64:
        print("--official needs --workers N (1..64; plan §5: min(8, nproc - 2))", file=sys.stderr)
        return 2
    started = utc_now()
    command = " ".join([sys.executable] + sys.argv)
    code, gate, _token = official_gate(REPO, roots)
    del _token                                            # the parent never draws; every job mints its own token
    if code != 0:
        print(json.dumps({"official": True, "status": "refused_by_freeze_gate", "gate": gate}, indent=1,
                         default=_json_default))
        return 2
    require_dev_case_inputs(REPO, driver="run_endpoint_ablation", extra=EXTRA_REQUIREMENTS)
    lock = check_environment_against_lock(REPO / ENVIRONMENT_LOCK_FILE, current=environment_fingerprint())
    if lock.get("status") != "matches_lock":
        print(f"environment does not match the lock ({lock.get('status')}); the official run is refused (nothing drawn)",
              file=sys.stderr)
        return 2
    pre_manifest = code_manifest(REPO)
    imp = import_time_check(pre_manifest)
    if not pre_manifest.get("complete") or imp["status"] != "consistent":
        print("code manifest incomplete or sources changed since import; refused (nothing drawn)", file=sys.stderr)
        return 2
    jobs = OV2.official_jobs(roots)
    run_id = make_run_id("official", started, ("dev_case_v1", "official_v2", tuple(roots),
                                               pre_manifest["manifest_sha256"]))
    return _launch_official_run(run_id=run_id, mode="official", roots=roots, jobs=jobs,
                                sizes=dict(OV2.OFFICIAL_CONFIG["streams"]), workers=int(args.workers), gate=gate,
                                authorisation=authorisation, started=started, pre_manifest=pre_manifest, lock=lock,
                                imp=imp, command=command)


def official_rehearsal_main(workers: Optional[int]) -> int:
    """``--official-rehearsal``: the official job matrix of root index 0 (9 cells + the diagnostics job) run through
    the same job runner and merge, on the **development** root (never a reserved root) at tiny sizes; output labelled
    ``debug`` under ``data/restricted_local/debug/<run_id>/`` only.  A pipeline check, not a result."""
    started = utc_now()
    command = " ".join([sys.executable] + sys.argv)
    pre_manifest = code_manifest(REPO)
    imp = import_time_check(pre_manifest)
    lock = check_environment_against_lock(REPO / ENVIRONMENT_LOCK_FILE, current=environment_fingerprint())
    check_seed(int(ABLATION_CONFIG["seed"]))                   # the development root; a reserved root is refused
    sizes = dict(ABLATION_CONFIG["tiny"])
    assert max(sizes["N_ladder"]) <= 16 and sizes["opt"] <= 16 and sizes["test"] <= 500
    jobs = OV2.official_jobs([0])
    run_id = "debug-" + make_run_id("rehearsal", started, ("dev_case_v1", "official_v2_rehearsal",
                                                           pre_manifest["manifest_sha256"]))
    w = int(workers) if workers else max(1, min(4, (os.cpu_count() or 2) - 1))
    gate = {"status": "rehearsal: development root; the protocol freeze is not required and was not checked"}
    return _launch_official_run(run_id=run_id, mode="rehearsal", roots=[0], jobs=jobs, sizes=sizes, workers=w,
                                gate=gate, authorisation=None, started=started, pre_manifest=pre_manifest, lock=lock,
                                imp=imp, command=command)


def _official_root_tables(k: int, items: list[dict[str, Any]], pr: Path, rr: Path, *, cfg: dict[str, Any],
                          aliases: dict[int, int]) -> dict[str, Any]:
    """The tables of one root (plan §7: per root, never pooled): public (aliased) and restricted (raw)."""
    cells = [it for it in items if it["job"]["kind"] == "cell"]
    diags = [it for it in items if it["job"]["kind"] == "diagnostics"]
    ends, resid, rations, status, cand = [], [], [], [], []
    diag_by_cell, tables, timings = {}, [], {}
    for it in cells:
        p, cid = it["payload"], it["job"]["cell_id"]
        ends += p["ends"]
        resid += p["resid"]
        rations += p["rations"]
        status += p["status"]
        cand += p["cand"]
        diag_by_cell[cid] = p["diag"]
        tables.append({"root_k": k, "cell_id": cid, "tables": p["tables"]})
        timings[cid] = p["timings_s"]
    for rows_ in (ends, resid, rations, status, cand):
        for r in rows_:
            r["root_k"] = k
    rep_cfg = cfg["endpoint_reporting"]
    rep_kw = {"membership_stat": rep_cfg["membership_stat"], "main_event": rep_cfg["main_event"],
              "m1_dm_off_is_entry": rep_cfg["m1_dm_off_is_entry"]}
    comp = comparable_sets(ends, config=cfg, **rep_kw)
    pmr = per_method_report(ends, cand, diagnostics=diag_by_cell, config=cfg, **rep_kw)
    curve = [row for w in cfg["evaluation"]["evaluation_worlds"]
             for row in matched_cost_curve(cand, w, main_event=rep_kw["main_event"])]
    for rows_ in (pmr, curve):
        for r in rows_:
            r["root_k"] = k
    conv = OV2.convergence_report(tables, config=cfg)
    tec = training_event_consistency(ends)
    diag_rows: list[dict[str, Any]] = []
    diag_pub, diag_full = None, None
    for it in diags:
        p = it["payload"]
        diag_rows += p["rows"]
        diag_pub = p["public_summary"]
        diag_full = {"DIAG-T45": p["t45"], "DIAG-E1": p["e1"], "training_worlds": p["training_worlds"]}
    for r in diag_rows:
        r["root_k"] = k
    pub_files = [
        _write_csv(pr / "endpoint_ablation.csv", _pub(ends, aliases), OFFICIAL_ENDPOINT_COLUMNS),
        _write_csv(pr / "reference_residuals.csv", _pub(redact_official_residuals(resid), aliases),
                   ["root_k"] + RESIDUAL_COLUMNS),
        _write_json(pr / "comparable_sets.json", _pub({"root_k": k, "rule": cfg["comparable_set_rule"],
                                                       "sets": comp}, aliases)),
        _write_csv(pr / "per_method_report.csv", _pub(pmr, aliases), ["root_k"] + PER_METHOD_REPORT_COLUMNS),
        _write_csv(pr / "matched_cost_curve.csv", _pub(curve, aliases), ["root_k"] + MATCHED_COST_COLUMNS),
        _write_json(pr / "convergence_report.json", _pub({"root_k": k, **conv}, aliases)),
        _write_json(pr / "training_event_consistency.json", {"root_k": k, **tec}),
        _write_json(pr / "solve_status.json", _pub({"root_k": k, "cells": [
            {"cell_id": t["cell_id"], "diagnostics": diag_by_cell[t["cell_id"]], "selection_tables": t["tables"],
             "timings_s": timings[t["cell_id"]]} for t in tables], "per_method": status,
            "rules": {"no_solution": "status, bounds, gap and time limit kept; cost empty, never 0",
                      "min_violation": cfg["diagnostics"]["min_violation"]["label"]}}, aliases))]
    if diags:
        pub_files += [_write_json(pr / "diagnostics_summary.json", _pub({"root_k": k, **diag_pub}, aliases)),
                      _write_csv(pr / "diagnostics_rows.csv",
                                 _pub([{kk: v for kk, v in r.items() if kk not in DIAG_RESTRICTED_KEYS}
                                       for r in diag_rows], aliases), DIAG_PUBLIC_COLUMNS)]
    ids = sorted({kk[2:] for r in rations for kk in r if kk.startswith("q_") and kk != "q_hash"})
    qcols = ["root_k", "run_id", "cell_id", "label", "q_hash"] + [f"q_{i}" for i in ids]
    cand_q = [{**{kk: v for kk, v in c.items() if kk != "_q"}, **(c.get("_q") or {})} for c in cand]
    res_files = [
        _write_csv(rr / "rations_q.csv", rations, qcols),
        _write_csv(rr / "candidate_rations.csv", cand_q, ["root_k"] + CANDIDATE_COLUMNS + [f"q_{i}" for i in ids]),
        _write_csv(rr / "reference_residuals_full.csv", resid, ["root_k"] + RESIDUAL_COLUMNS),
        _write_csv(rr / "endpoint_ablation_raw_ids.csv", ends, OFFICIAL_ENDPOINT_COLUMNS)]
    if diag_full is not None:
        res_files += [_write_json(rr / "diagnostics_full.json", diag_full),
                      _write_csv(rr / "diagnostics_rows_full.csv", diag_rows,
                                 DIAG_PUBLIC_COLUMNS + list(DIAG_RESTRICTED_KEYS))]
    return {"files_public": [str(p) for p in pub_files], "files_restricted": [str(p) for p in res_files],
            "n_cells": len(cells), "n_diagnostics_jobs": len(diags), "n_endpoint_rows": len(ends),
            "n_candidate_rows": len(cand), "n_diagnostic_rows": len(diag_rows),
            "n_comparable_sets": len(comp), "training_event_consistency": {"all_equal": tec["all_equal"],
                                                                            "by_arm": tec["by_arm"]},
            "convergence_counts": conv["counts"]}


def official_merge_main(run_id: str) -> int:
    """``--official-merge RUN_ID`` (also called by ``--official`` / ``--official-rehearsal`` after the jobs): verify
    and merge the jobs per root (``official_v2.merge_jobs``: refuses missing / duplicate / unverifiable jobs), then the
    end-of-run identity check (code manifest and specification unchanged from the run start and every job; official:
    the freeze record still verifies) -> ``official`` or ``official_invalidated`` (outputs kept, disclosed), the run
    record (``run_type = official``; rehearsal: ``smoke``), a README and the top-level ``SHA256SUMS`` + ``MERGE_DONE``."""
    t0 = time.perf_counter()
    rehearsal = run_id.startswith(OV2.REHEARSAL_RUN_PREFIX)
    try:
        route = OV2.official_output_route(run_id, rehearsal=rehearsal)
    except ValueError as exc:
        print(f"--official-merge: {exc}", file=sys.stderr)
        return 2
    pub, res = REPO / route["public_dir"], REPO / route["restricted_dir"]
    plan_p = res / OV2.RUN_PLAN_FILE
    if not plan_p.is_file():
        print(f"--official-merge: {route['restricted_dir']}/{OV2.RUN_PLAN_FILE} missing (not a run)", file=sys.stderr)
        return 2
    plan_doc = json.loads(plan_p.read_text(encoding="utf-8"))
    mode = plan_doc["mode"]
    roots = [int(k) for k in plan_doc["roots_k"]]
    aliases = _aliases(roots, mode)
    cfg = OV2.OFFICIAL_CONFIG
    opts = SolverOptions(time_limit_s=float(cfg["solver"]["time_limit_s"]), mip_rel_gap=float(cfg["solver"]["mip_rel_gap"]))
    run_type = "official" if mode == "official" else "smoke"
    if mode == "official":
        rng = {f"root{k}/{sd}/{s_}": f"root={OV2.root_alias(k)}/{s_}" for k in roots
               for sd in cfg["evaluation"]["evaluation_worlds"] for s_ in ("opt", "validation", "test")}
    else:
        rng = {f"{sd}/{s_}": f"root={int(ABLATION_CONFIG['seed'])}/{s_}"
               for sd in cfg["evaluation"]["evaluation_worlds"] for s_ in ("opt", "validation", "test")}
    rec_kw = dict(run_type=run_type, command=str(plan_doc.get("command")) + " ; merge: " + " ".join(
        [sys.executable] + sys.argv), repo_root=REPO, started_at=plan_doc["started_utc"], rng_streams=rng,
        solver_version=solver_version_string(), tolerances=opts.to_dict(),
        config_paths=[p for p in SPEC_FILES if p.startswith("configs/")],
        data_paths=[str(p) for p in (DRV.PROBLEM_YAML, DRV.CELLS_CSV, DRV.ENERGY_JSON, DRV.BUILD_REPORT,
                                     DRV.BUILD_SCRIPT)],
        protocol_path=str(DRV.PROTOCOL), is_synthetic=False, run_id=run_id, build_identity=dev_case_build_identity(REPO))
    try:                                         # validate the record's gates before anything is written
        build_run_record(completed_at=utc_now(), exit_status=0, output_paths=[], **rec_kw)
    except ValueError as exc:
        print(f"--official-merge: the run record would be refused ({str(exc)[:300]}); nothing written", file=sys.stderr)
        return 2
    try:
        merged = OV2.merge_jobs(run_id, repo=REPO, rehearsal=rehearsal,
                                build_tables=lambda k, items, pr, rr: _official_root_tables(k, items, pr, rr, cfg=cfg,
                                                                                            aliases=aliases))
    except OV2.MergeError as exc:
        print(json.dumps({"merge": "refused", "reasons": exc.reasons[:30]}, indent=1), file=sys.stderr)
        return 2
    # end-of-run identity (F3: the only check that may fail after the draws)
    post_manifest = code_manifest(REPO)
    post_spec = spec_fingerprint(REPO)
    job_recs = {}
    for j in plan_doc["jobs"]:
        p = pub / "jobs" / j["job_id"] / "job_record.json"
        job_recs[j["job_id"]] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    start_sha = plan_doc["code_manifest_at_start"]
    jobs_same = all(r.get("code_manifest_at_start") == start_sha and r.get("code_manifest_at_end") == start_sha
                    and r.get("code_unchanged_during_job") for r in job_recs.values())
    identity = {"code_manifest_at_start": start_sha, "code_manifest_at_end": post_manifest["manifest_sha256"],
                "code_manifest_complete_at_end": bool(post_manifest.get("complete")),
                "spec_digest_at_start": plan_doc["spec_digest_at_start"], "spec_digest_at_end": post_spec["digest"],
                "every_job_saw_the_start_manifest": jobs_same}
    unchanged = (post_manifest["manifest_sha256"] == start_sha and bool(post_manifest.get("complete")) and jobs_same
                 and post_spec["digest"] == plan_doc["spec_digest_at_start"])
    if mode == "official":
        reasons, facts = OV2.protocol_freeze_check(REPO, roots_k=tuple(roots))
        identity["freeze_check_at_end"] = {"reasons": reasons, "freeze_record_sha256": facts.get(
            "freeze_record_sha256"), "freeze_record_sha256_at_start": plan_doc.get("freeze_record_sha256")}
        unchanged = unchanged and not reasons and facts.get("freeze_record_sha256") == plan_doc.get(
            "freeze_record_sha256")
    identity["unchanged"] = bool(unchanged)
    label = route["label"] if (unchanged or mode != "official") else "official_invalidated"
    tec = {k: v["training_event_consistency"] for k, v in merged["per_root"].items()}
    extra = {
        "purpose": ("official run v2 (reference problem v2; D-533, D-535, D-539); per root; no animal outcome" if mode ==
                    "official" else "official rehearsal (development root, tiny sizes): a pipeline check, not a result"),
        "mode": mode, "output_label": label, "output_route": route,
        "identity_and_gating": {"gate_at_start": plan_doc.get("gate"), "authorisation": plan_doc.get("authorisation"),
                                "environment_lock": plan_doc.get("environment_lock"),
                                "import_time_sources": plan_doc.get("import_time_sources"),
                                "end_of_run": identity, "official_config_sha256": plan_doc.get(
                                    "official_config_sha256"), "freeze_pin": FREEZE_PIN.as_posix(),
                                "protocol_freeze_record": OV2.PROTOCOL_FREEZE.as_posix()},
        "roots_k": roots, "root_aliases": plan_doc.get("root_aliases"), "root_resolution": plan_doc.get(
            "root_resolution"),
        "world_fingerprints": (next(iter(job_recs.values()), {}) or {}).get("world_fingerprints"),
        "world_fingerprints_equal_across_jobs": len({json.dumps({sd: w.get("world_fingerprint") for sd, w in (
            r.get("world_fingerprints") or {}).items()}, sort_keys=True) for r in job_recs.values()}) == 1,
        "stream_registry": {jid: {"ok": r.get("stream_registry_ok"), "rng_streams": r.get("rng_streams")}
                            for jid, r in job_recs.items()},
        "jobs": {jid: {"exit_status": r.get("exit_status"), "wall_s": r.get("wall_s"), "timings_s": r.get(
            "timings_s"), "host": r.get("host"), "pid": r.get("pid"), "thread_env": r.get("thread_env"),
            "build_mode": r.get("build_mode"), "run_context_injected": r.get("run_context_injected")}
                 for jid, r in job_recs.items()},
        "per_root": merged["per_root"], "pooled_across_roots": False, "training_event_consistency": tec,
        "sd_registry_check": (next(iter(job_recs.values()), {}) or {}).get("sd_registry_check"),
        "sd_registry_check_ok_in_every_job": all((r.get("sd_registry_check") or {}).get("ok") for r in
                                                 job_recs.values()),
        "diagnostic_definitions": cfg["diagnostics"], "convergence": cfg["convergence"],
        "convergence_reports": {str(k): {"file": f"root{k}/convergence_report.json",
                                         "counts": merged["per_root"][str(k)].get("convergence_counts")}
                                for k in merged["roots_k"]},
        "solver": {"version": solver_version_string(), "time_limit_s": cfg["solver"]["time_limit_s"],
                   "mip_rel_gap": cfg["solver"]["mip_rel_gap"]},
        "threads": {"declared": dict(OV2.THREAD_ENV), "workers": plan_doc.get("workers"),
                    "cpu_count": plan_doc.get("cpu_count")},
        "sizes": plan_doc.get("sizes"),
        "merge_wall_s": time.perf_counter() - t0}
    readme = pub / ("NOT_FOR_MANUSCRIPT.md" if mode != "official" else "README_OFFICIAL.md")
    txt = (f"# {run_id}\n\nOfficial run v2 of dev_case_v1 (reference problem v2; label {label}).\n\n"
           "Tables are per reserved root (root<k>/); the roots are never pooled into one n. Reliability = satisfaction "
           "of the declared model constraints under the declared distribution (no animal outcome). The main event is "
           f"{cfg['endpoint_reporting']['main_event']} (plan-level Table 5-1 reading, D-535); membership by the one-sided "
           "exact Clopper-Pearson upper bound on the evaluation world's test stream; costs only inside comparable_sets. "
           "The own-world value and the SD-H0 column form an assumption range, never a confidence interval. DIAG-T45 "
           "and DIAG-E1 (root 0; SD-H0 and SD-S2) are diagnostics, never methods or set members. Stream ids name the "
           "reserved roots by alias (root=formal-k<k>/...).\n")
    if mode != "official":
        txt += ("\n**debug / rehearsal**: development root, toy sizes; a pipeline check, not a result, not an estimate "
                "of any rate or cost.\n")
    if label == "official_invalidated":
        txt += ("\n**official_invalidated**: the end-of-run identity check failed (see run_record.json extra."
                "identity_and_gating.end_of_run); the reserved roots are spent -- a rerun uses the same roots under a "
                "new run id and this output is kept and disclosed (plan §7).\n")
        (pub / "OFFICIAL_INVALIDATED.md").write_text(txt, encoding="utf-8")
    readme.write_text(txt, encoding="utf-8")
    outputs = sorted(str(p.relative_to(REPO)) for p in pub.rglob("*") if p.is_file())
    exit_status = 0 if label != "official_invalidated" else 5
    rec = build_run_record(completed_at=utc_now(), exit_status=exit_status, output_paths=outputs, extra=extra,
                           **rec_kw)
    write_run_record(_pub(rec, aliases), pub / "run_record.json")
    _write_json(res / "run_record_restricted_note.json",
                {"run_id": run_id, "public_run_record": f"{route['public_dir']}/run_record.json",
                 "true_stream_ids": "jobs/<job_id>/stream_registry_full.json (restricted; the public files use "
                                    "root aliases)", "jobs": sorted(job_recs)})
    OV2.write_sha_manifest(res)
    OV2.write_sha_manifest(pub)
    for d in (res, pub):
        with open(d / OV2.MERGE_DONE, "x", encoding="utf-8") as fh:
            json.dump({"run_id": run_id, "label": label, "utc": utc_now(),
                       "sha256sums_sha256": sha_or_none(d / OV2.SHA_MANIFEST)}, fh, indent=1)
    print(json.dumps({"run_id": run_id, "mode": mode, "output_label": label, "public_dir": route["public_dir"],
                      "restricted_dir": route["restricted_dir"], "roots_k": roots,
                      "per_root": {k: {kk: v[kk] for kk in ("n_jobs", "n_cells", "n_diagnostics_jobs",
                                                            "n_endpoint_rows", "n_diagnostic_rows",
                                                            "training_event_consistency", "convergence_counts")}
                                   for k, v in merged["per_root"].items()},
                      "pooled_across_roots": False, "end_of_run_identity_unchanged": identity["unchanged"],
                      "merge_wall_s": time.perf_counter() - t0}, indent=1, default=_json_default))
    return exit_status


def official_dry_run_main(roots_k: Optional[list[int]]) -> int:
    """``--official --dry-run`` (plan §2 "B4"; B5 gate on the Mac): the freeze gate, measured (0 draws, 0 solves, 0
    files); refused -> exit 2.  When frozen: the preflight, the v2 freeze status and the official context without a
    draw in mode ``official`` (every arm validated with the injected run context, F5), measured again."""
    t_start = time.perf_counter()
    try:
        roots = _official_roots(roots_k or [0, 1, 2])
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    code, gate, _token = official_gate(REPO, roots)
    del _token                                            # a dry run never draws
    out: dict[str, Any] = {"official_dry_run": True, "roots_k": roots, "gate": gate}
    if code != 0:
        out.update({"status": "refused: the protocol is not frozen / not anchored (gate.reasons); nothing was drawn",
                    "draws": gate["draws"], "solves": gate["solves"], "files_written": gate["files_written"],
                    "dirs_created": gate["dirs_created"], "elapsed_s": time.perf_counter() - t_start})
        print(json.dumps(out, indent=1, ensure_ascii=False, default=_json_default))
        return 2
    require_dev_case_inputs(REPO, driver="run_endpoint_ablation", extra=EXTRA_REQUIREMENTS)
    cfg = OV2.OFFICIAL_CONFIG
    rc = {"protocol_sha256": gate["protocol_sha256"], "primary_assumption_id": OV2.PRIMARY_ASSUMPTION_ID}
    counters = DryRunCounters()
    with counters:
        fz = freeze_status(REPO)
        ctx = build_context(dict(cfg["streams"]), draw=False, official=cfg, run_context=rc)
        op = official_plan(ctx, cfg)
    meas = counters.record()
    ok = fz["status"] == "matches_frozen_anchored" and meas["draws"] == 0 and meas["solves"] == 0 and \
        meas["files_written"] == 0 and gate["draws"] == 0
    out.update({"status": ("ready: protocol frozen, specification = v2 pin (matches_frozen_anchored), 0 draws" if ok
                           else "not ready"), "freeze": fz["status"], "draws": meas["draws"] + gate["draws"],
                "solves": meas["solves"] + gate["solves"], "files_written": meas["files_written"],
                "dirs_created": meas["dirs_created"], "measured_counts": meas, "build_mode": ctx["build_mode"],
                "validators": {k: v.ok for k, v in ctx["validators"].items()},
                "jobs": OV2.official_jobs(roots), "matrix": op["matrix"], "elapsed_s": time.perf_counter() - t_start})
    print(json.dumps(out, indent=1, ensure_ascii=False, default=_json_default))
    return 0 if ok else 2


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # ``--official --dry-run`` is the one pair of modes that combines (the official dry run, plan §2 "B4"); it is read
    # before the mutually exclusive modes are parsed, so every other pair stays an argparse error
    raw = list(sys.argv[1:] if argv is None else argv)
    official_dry_run = "--official" in raw and "--dry-run" in raw
    if official_dry_run:
        raw = [a for a in raw if a != "--dry-run"]
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="build and check only; print the plan; write nothing "
                                                          "(with --official: the official freeze gate, measured)")
    g.add_argument("--tiny", action="store_true", help="toy-size pipeline check; debug output in restricted_local/debug")
    g.add_argument("--full", action="store_true", help="the declared ablation (authorised compute only)")
    g.add_argument("--print-spec-fingerprint", action="store_true", help="print the specification fingerprint")
    g.add_argument("--official-plan", action="store_true",
                   help=("official run v2 plan (batch 3; data only): build reference problem v2 and the five SD-point "
                         "worlds, print the official cell / job matrix and the compute estimate; no draw, no solve, "
                         "nothing written (measured)"))
    g.add_argument("--write-freeze-pin", metavar="NOTE",
                   help=f"write the current specification fingerprint to {FREEZE_PIN.as_posix()} (the v2 pin; part of "
                        "the one freeze commit; commit it and run exactly that commit)")
    g.add_argument("--write-protocol-freeze", metavar="NOTE",
                   help=(f"write {OV2.PROTOCOL_FREEZE.as_posix()} (lead-invoked, after protocol.yaml is_frozen: true "
                         "and the v2 pin; then one freeze commit)"))
    g.add_argument("--official-rehearsal", action="store_true",
                   help=("the official job matrix of root index 0 (9 cells + diagnostics) through the job runner and "
                         "merge, on the DEVELOPMENT root at tiny sizes; debug output only"))
    g.add_argument("--official-merge", metavar="RUN_ID", help="verify and merge the jobs of an official run, per root")
    g.add_argument("--official-job", nargs=2, metavar=("RUN_ID", "JOB_ID"),
                   help="internal: one job process of an official run or rehearsal (launched by the job runner)")
    g.add_argument("--official", action="store_true",
                   help=("official run v2 on the reserved roots (with --authorised-compute, --root-k, --workers); "
                         "with --dry-run: the freeze gate only, measured"))
    ap.add_argument("--authorised-compute", action="store_true",
                    help="required with --full / --official: the user has authorised this host")
    ap.add_argument("--authorisation-file", default=None,
                    help=(f"with --full (default {AUTHORISATION_FILE.as_posix()}, scope run_endpoint_ablation) or "
                          f"--official (default {OFFICIAL_AUTHORISATION_FILE.as_posix()}, scope "
                          f"{OV2.OFFICIAL_SCOPE}): user-written JSON listing the authorised hosts"))
    ap.add_argument("--root-k", type=int, nargs="+", default=None,
                    help="with --official: reserved root INDICES (0, 1, 2), never a seed")
    ap.add_argument("--workers", type=int, default=None,
                    help="with --official / --official-rehearsal: parallel job processes (plan §5; D-539: 8)")
    args = ap.parse_args(raw)
    args.dry_run = bool(args.dry_run or official_dry_run)
    if (args.root_k is not None) and not args.official:
        ap.error("--root-k is for --official only")
    os.chdir(REPO)
    if args.print_spec_fingerprint:
        print(json.dumps(spec_fingerprint(REPO), indent=1, ensure_ascii=False))
        return 0
    if args.write_freeze_pin is not None:
        doc = write_freeze_pin(REPO, note=str(args.write_freeze_pin))
        print(json.dumps({"pin": FREEZE_PIN.as_posix(), "digest": doc["digest"], "missing": doc["missing"],
                          "next": ("the freeze commit: configs/protocol.yaml (is_frozen: true), this pin and "
                                   "configs/protocol_freeze.json (--write-protocol-freeze) in one commit")},
                         indent=1))
        return 0
    if args.write_protocol_freeze is not None:
        try:
            doc = OV2.write_protocol_freeze(str(args.write_protocol_freeze), REPO)
        except OV2.FreezeGateError as exc:
            print(json.dumps({"write_protocol_freeze": "refused", "reasons": exc.reasons}, indent=1), file=sys.stderr)
            return 2
        print(json.dumps({"written": OV2.PROTOCOL_FREEZE.as_posix(), "spec_digest": doc["spec"]["digest"],
                          "code_subset_sha256": doc["code_manifest_subset"]["subset_sha256"],
                          "declared_jobs": len(doc["declared"]["jobs"]),
                          "next": ("git add configs/protocol.yaml configs/protocol_freeze.json "
                                   f"{FREEZE_PIN.as_posix()} && one commit; then --official --dry-run must report "
                                   "ready at that commit")}, indent=1))
        return 0
    if args.official and args.dry_run:
        return official_dry_run_main(args.root_k)
    if args.official:
        return official_main(args)
    if args.official_merge is not None:
        return official_merge_main(str(args.official_merge))
    if args.full and not args.authorised_compute:
        print("--full needs --authorised-compute: the full ablation runs only on a host the user has authorised (the Mac "
              "is a delivery host; round 3 runs --dry-run and --tiny only). Nothing was read or written.", file=sys.stderr)
        return 2
    authorisation = None
    if args.full:
        try:
            authorisation = check_authorisation(Path(args.authorisation_file or AUTHORISATION_FILE))
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 2
    # R3F / instruction F-4: missing or stale inputs -> short list and exit 2 / 1 before anything is read or written
    require_dev_case_inputs(REPO, driver="run_endpoint_ablation", extra=EXTRA_REQUIREMENTS)
    if args.official_plan:
        return official_plan_main()
    if args.official_job is not None:
        return official_job_main(str(args.official_job[0]), str(args.official_job[1]))
    if args.official_rehearsal:
        return official_rehearsal_main(args.workers)
    mode = "dry_run" if args.dry_run else ("tiny" if args.tiny else "full")
    started = utc_now()
    t_start = time.perf_counter()
    command = " ".join([sys.executable] + sys.argv)
    pre_manifest = code_manifest(REPO)
    pre_spec = spec_fingerprint(REPO)
    imp = import_time_check(pre_manifest)
    frz = freeze_status(REPO)
    lock = check_environment_against_lock(REPO / ENVIRONMENT_LOCK_FILE, current=environment_fingerprint())
    if mode == "full" and lock.get("status") != "matches_lock":
        print(f"environment does not match the lock ({lock.get('status')}); the full ablation is refused", file=sys.stderr)
        return 2
    seed_chk = check_seed(int(ABLATION_CONFIG["seed"]))
    sizes = dict(ABLATION_CONFIG["tiny"] if mode == "tiny" else ABLATION_CONFIG["streams"])
    if mode == "tiny":
        assert max(sizes["N_ladder"]) <= 16 and sizes["opt"] <= 16 and sizes["test"] <= 500
    counters = DryRunCounters() if mode == "dry_run" else None
    with (counters if counters is not None else contextlib.nullcontext()):
        ctx = build_context(sizes, draw=mode != "dry_run")
        pl_dry = plan(ctx, dict(ABLATION_CONFIG["streams"]), "full (planned)") if mode == "dry_run" else None
    checks = {"preflight": "READY", "seed": seed_chk, "sd_registry": ctx["sd_registry"],
              "environment_lock": lock.get("status"), "freeze": frz, "import_time_sources": imp,
              "code_manifest_complete_at_start": bool(pre_manifest.get("complete")),
              "reference_spec_fingerprint": ctx["ref"].fingerprint(),
              "reference_table": ctx["table"].to_record(),
              "validators": {k: {"ok": v.ok, "n_pending": v.n_pending, "n_assumptions": v.n_assumptions}
                             for k, v in ctx["validators"].items()}}
    if mode == "dry_run":
        meas = counters.record()
        print(json.dumps({"dry_run": True, "files_written": meas["files_written"], "draws": meas["draws"],
                          "solves": meas["solves"], "dirs_created": meas["dirs_created"], "measured_counts": meas,
                          "checks": checks, "plan_full": pl_dry, "plan_tiny_sizes": ABLATION_CONFIG["tiny"],
                          "full_command": "PYTHONDONTWRITEBYTECODE=1 <python3.11> "
                                          "experiments/E1_cost_reliability/run_endpoint_ablation.py --full "
                                          "--authorised-compute",
                          "elapsed_s": time.perf_counter() - t_start}, indent=1, ensure_ascii=False,
                         default=_json_default))
        return 0

    s2_cons = DRV.s2_consistency(ctx["problems"]["FULL11"], ctx["problems"]["PART6P5"],
                                 SolverOptions(time_limit_s=60.0, mip_rel_gap=1e-4))
    m9_cons = planned_arm_consistency(ctx["problems"]["FULL11"], ctx["problems"]["MAIN9"], ctx["main9_planned"],
                                      SolverOptions(time_limit_s=60.0, mip_rel_gap=1e-4))
    opts = SolverOptions(time_limit_s=float(ABLATION_CONFIG["solver"]["time_limit_s"]),
                         mip_rel_gap=float(ABLATION_CONFIG["solver"]["mip_rel_gap"]))
    results = [run_cell(ctx, c, sizes, opts) for c in ABLATION_CONFIG["cells"]]
    reg = ctx["registry"].check()
    if not reg["ok"]:
        print(json.dumps({"stream_collisions": reg["collisions"]}, indent=1), file=sys.stderr)
        return 4
    t_eval = time.perf_counter()
    ends, resid, rations, status = [], [], [], []
    for cr in results:
        e_, r_, q_, s_ = cell_rows(ctx, cr, opts, PENDING, PENDING)
        ends += e_
        resid += r_
        rations += q_
        status += s_
    t_eval = time.perf_counter() - t_eval
    rep_cfg = ABLATION_CONFIG["endpoint_reporting"]
    rep_kw = {"membership_stat": rep_cfg["membership_stat"], "main_event": rep_cfg["main_event"],
              "m1_dm_off_is_entry": rep_cfg["m1_dm_off_is_entry"]}
    comp = comparable_sets(ends, **rep_kw)
    # official-run plan batch 2: descriptive reports, built only after every selection and the comparable sets
    t_cand = time.perf_counter()
    score = reference_scorer(ctx)
    cand: list[dict[str, Any]] = []
    for cr in results:
        cand += candidate_rations(cr["runs"], score, cell_id=cr["cell"]["cell_id"], extra_selections=cr["m3b_per_k"])
    t_cand = time.perf_counter() - t_cand
    pmr = per_method_report(ends, cand, diagnostics={cr["cell"]["cell_id"]: cr["diag"] for cr in results}, **rep_kw)
    curve = [row for w in ABLATION_CONFIG["evaluation"]["evaluation_worlds"]
             for row in matched_cost_curve(cand, w, main_event=rep_cfg["main_event"])]
    # code / specification identity at the end of every computation (instruction F.1: changed -> debug)
    post_manifest = code_manifest(REPO)
    post_spec = spec_fingerprint(REPO)
    # FIX3_BC (C-3): complete manifests at start and end, unchanged, and equal to the sources hashed before the imports
    manifest_same = bool(pre_manifest.get("complete")) and bool(post_manifest.get("complete")) and \
        pre_manifest["manifest_sha256"] == post_manifest["manifest_sha256"] and \
        pre_spec["digest"] == post_spec["digest"] and imp["status"] == "consistent"
    route = output_route(mode, manifest_same, frz["status"])
    run_type = "pilot" if mode == "full" else "smoke"
    run_id = route["run_id_prefix"] + make_run_id(run_type, started, ("dev_case_v1", "endpoint_ablation",
                                                                      ABLATION_CONFIG["seed"],
                                                                      pre_manifest["manifest_sha256"]))
    for rows_ in (ends, resid, rations):
        for r in rows_:
            r["run_id"] = run_id
            if "output_label" in r:
                r["output_label"] = route["label"]
    for rows_ in (pmr, curve, cand):
        for r in rows_:
            r["run_id"], r["output_label"] = run_id, route["label"]
    per_state = None
    n_states_eval = sum(1 for r in ends if r.get("evaluation_world")) * int(sizes["test"])
    if n_states_eval:
        per_state = t_eval / n_states_eval
    pl = plan(ctx, sizes, mode, per_state_s=per_state)
    pl_full = plan(ctx, dict(ABLATION_CONFIG["streams"]), "full (planned)", per_state_s=per_state)
    solve_status = {"cells": [{"cell_id": cr["cell"]["cell_id"], "sd_arm": cr["cell"]["sd"],
                               "objective_arm": cr["cell"]["objective"], "problem_id": cr["problem"].problem_id,
                               "diagnostics": cr["diag"], "selection_tables": cr["tables"],
                               "timings_s": cr["timings_s"]} for cr in results],
                    "per_method": status, "s2_consistency_with_full11": s2_cons,
                    "main9_consistency_with_full11": m9_cons, "main9_record": ctx["main9_record"],
                    "reproduction_vs_pilot": reproduction_vs_pilot(ends, mode),
                    "rules": {"no_solution": "status, bounds, gap and time limit kept; cost empty, never 0",
                              "min_violation": ABLATION_CONFIG["diagnostics"]["min_violation"]["label"],
                              "grid": "a declared grid or candidate set without a solution is not a whole-space "
                                      "statement"}}
    development = route["label"] == "development"
    pub = Path(route["public_dir"]) / run_id
    res_dir = Path(route["restricted_dir"]) / run_id
    if pub.exists() or res_dir.exists():
        print(f"{pub} exists; refusing to overwrite", file=sys.stderr)
        return 3
    pub.mkdir(parents=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    files = []
    note = (f"# {run_id}\n\n2 x 3 endpoint ablation of dev_case_v1 (including MAIN9; mode {mode}; output label {route['label']}).\n\n"
            "Development material. No number here may enter a formal result table, the manuscript, an abstract, a "
            "figure or a claim ledger. No animal outcome is claimed; 'reliability' = satisfaction of the declared model "
            "constraints under the declared distribution. Every ration is scored by evaluate_reference under one "
            "physical definition (configs/dev_case_v1/reference_constraints.csv) on the test stream of both SD worlds; "
            "the SD axis changes the declared problem, not the method; 6-row rates are not 11-row or main-reference "
            "rates. The main reference event judges the Table 5-1 rows only inside the primary Table 5-1 domain "
            "(unknown outside it; the out-of-domain share is reported next to every main rate); h0_eleven and the main "
            "reference + PN-CP-HI are reported next to it. Minimum-violation counts are about the N training scenarios "
            "only. Costs are interpreted only inside comparable_sets.json. per_method_report.csv reports every "
            "method's rates, target_met, minimum attainable main-event risk over its declared grid and the frontier "
            "diagnostic (costs outside the comparable set are labelled not_a_comparable_cost); matched_cost_curve.csv is "
            "descriptive (T8.1).\n")
    if mode == "tiny":
        note += ("\n**debug / smoke**: toy sizes (N <= 16, test <= 500); a pipeline check, not a result, not an "
                 "estimate of any rate or cost.\n")
    if mode == "full" and not development:
        note += f"\n**downgraded to debug**: {route['why']}\n"
    files.append(pub / "NOT_FOR_MANUSCRIPT.md")
    files[-1].write_text(note, encoding="utf-8")
    files.append(_write_csv(pub / "endpoint_ablation.csv", ends, ENDPOINT_COLUMNS))
    files.append(_write_csv(pub / "reference_residuals.csv", redact_residuals(resid) if development else resid,
                            RESIDUAL_COLUMNS))
    files.append(_write_json(pub / "comparable_sets.json", comp))
    files.append(_write_csv(pub / "per_method_report.csv", pmr, PER_METHOD_REPORT_COLUMNS))
    files.append(_write_csv(pub / "matched_cost_curve.csv", curve, MATCHED_COST_COLUMNS))
    files.append(_write_json(pub / "solve_status.json", solve_status))
    files.append(_write_json(pub / "ablation_config.json", {"ablation_config": ABLATION_CONFIG,
                                                           "ablation_config_sha256": stable_hash(ABLATION_CONFIG),
                                                           "dev_run_config_sha256": stable_hash(DRV.RUN_CONFIG),
                                                           "checks": checks, "plan": pl, "plan_full": pl_full,
                                                           "reference_spec": ctx["ref"].to_record(),
                                                           "worlds": {k: v["record"] for k, v in ctx["worlds"].items()},
                                                           "stream_registry": reg, "output_route": route}))
    qcols = ["run_id", "cell_id", "label", "q_hash"] + [f"q_{i}" for i in ctx["problems"]["FULL11"].ingredient_ids]
    files.append(_write_csv(res_dir / "rations_q.csv", rations, qcols))
    # grid candidates (descriptive): scores per (occurrence, world) with the ration q -> restricted only
    cand_q = [{**{k: v for k, v in c.items() if k != "_q"}, **(c.get("_q") or {})} for c in cand]
    files.append(_write_csv(res_dir / "candidate_rations.csv", cand_q,
                            CANDIDATE_COLUMNS + [f"q_{i}" for i in ctx["problems"]["FULL11"].ingredient_ids]))
    if development:
        files.append(_write_csv(res_dir / "reference_residuals_full.csv", resid, RESIDUAL_COLUMNS))
    rec = build_run_record(
        run_type=run_type, command=command, repo_root=REPO, started_at=started, completed_at=utc_now(), exit_status=0,
        rng_streams={f"{sd}_{k}": ds.stream_id for sd, wd in ctx["worlds"].items() for k, ds in wd["draws"].items()},
        solver_version=solver_version_string(), tolerances=opts.to_dict(),
        config_paths=[p for p in SPEC_FILES if p.startswith("configs/")],
        data_paths=[str(p) for p in (DRV.PROBLEM_YAML, DRV.CELLS_CSV, DRV.ENERGY_JSON, DRV.BUILD_REPORT, DRV.BUILD_SCRIPT)],
        protocol_path=str(DRV.PROTOCOL), output_paths=[str(p) for p in files], is_synthetic=False, run_id=run_id,
        build_identity=dev_case_build_identity(REPO),
        extra={"purpose": "2 x 3 endpoint ablation of dev_case_v1 (including MAIN9); development only; no animal outcome",
               "mode": mode, "output_label": route["label"], "output_route": route,
               "ablation_config_sha256": stable_hash(ABLATION_CONFIG),
               "code_manifest_at_start": pre_manifest["manifest_sha256"],
               "code_manifest_at_end": post_manifest["manifest_sha256"],
               "spec_fingerprint_at_start": pre_spec["digest"], "spec_fingerprint_at_end": post_spec["digest"],
               "code_and_spec_unchanged_during_run": manifest_same, "freeze": frz,
               "import_time_sources": imp, "code_manifest_complete": [bool(pre_manifest.get("complete")),
                                                                      bool(post_manifest.get("complete"))],
               "authorisation": authorisation,
               "reference_spec_fingerprint": ctx["ref"].fingerprint(), "timings_s": {
                   "cells": {cr["cell"]["cell_id"]: cr["timings_s"] for cr in results}, "reference_evaluation": t_eval,
                   "candidate_scoring": t_cand,
                   "total": time.perf_counter() - t_start}, "stream_registry_ok": reg["ok"],
               "host_note": "local Mac smoke (tiny)" if mode == "tiny" else "see run command host"})
    write_run_record(rec, pub / "run_record.json")
    print(json.dumps({"run_id": run_id, "mode": mode, "output_label": route["label"], "public_dir": str(pub),
                      "restricted_dir": str(res_dir), "n_endpoint_rows": len(ends), "n_residual_rows": len(resid),
                      "n_rations": len(rations), "n_candidate_rows": len(cand), "n_per_method_rows": len(pmr),
                      "n_matched_cost_rows": len(curve), "code_and_spec_unchanged_during_run": manifest_same,
                      "freeze": frz["status"], "timings_s": rec["extra"]["timings_s"],
                      "compute_estimate_full": pl_full["compute_estimate"]}, indent=1, default=_json_default))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
