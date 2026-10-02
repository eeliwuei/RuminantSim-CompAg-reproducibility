"""Unified reference evaluation of an executed ration: ``evaluate_reference(q, draws, spec)``.

Review round 3 (F2; next-round instruction C.2-C.3; R3C, 2026-09-25).  Status (contract section 2.3):
``implemented``, ``unit_passed`` (``tests/unit/test_evaluate_reference.py``, synthetic hand-computed cases);
evidence level ``code_tested``.  Passing the tests shows that the counts below are computed as documented --
not that any ration meets any cow's requirements, and not any animal outcome.

Why
---
The development scenario S2 changed two things at once: the uncertainty (narrowed SD) and the risk event (the
11-row joint event of H0 became 6 probabilistic rows + 5 planned rows).  A cost or risk difference between them is
therefore not a like-for-like comparison.  This module scores **every** ration -- whatever objective, world or
method produced it -- against one fixed physical definition, so that the change of problem and the change of
method can be told apart:

* the **main reference event**: union of the rows whose role is ``main_reference`` in
  ``configs/dev_case_v1/reference_constraints.csv`` (adjudicated by source only; see
  ``docs/reference_problem_v1.md`` §3), with the energy row judged by the project's Chapter 3 reference chain
  (:mod:`..nutrition.energy_reference`) and, next to it, by the linear fixed-DMI row;
* the historical **H0 event** (11 rows, linear energy row) and the same 11 rows with the reference energy
  verdict; the **S2 subset** (6 rows); the **excluded five** rows (``PN-NEL-FIXEDDMI, PN-CP-HI, PN-EE-HI,
  PN-CA-ABS, PN-P-ABS``), as a union and row by row;
* every evaluated row: violation counts, undefined counts and the size of the violation in the row's natural
  (declared) unit;
* the NASEM (2021) Table 5-1 applicability domain of every state (:mod:`..nutrition.domain`): the declared
  variants side by side; ``not_assessable`` states are a separate column, never removed from a denominator;
* round-3 red team C-1 (FIX3_BC): the **main reference event is domain-conditioned** -- in a state outside the
  primary Table 5-1 domain (``not_assessable`` or ``undefined``) the verdicts of PN-T1..T5 are not used: the T rows
  count as ``unknown`` there (the state enters the upper value of the rate pair unless another member is violated),
  as ``docs/reference_problem_v1.md`` §6 requires ("cannot judge or be extrapolated").  The R3C reading that used the
  T verdicts in every state is kept, clearly named, as ``main_reference_t51_verdicts_used_out_of_domain``; the H0 and
  S2 training events stay as declared (``h0_eleven``, ``s2_six``) and get domain-conditioned companions
  (``h0_eleven_domain_conditioned``, ``s2_six_domain_conditioned``).  Every declared domain variant also reports the
  main event conditioned on that variant, and ``summary()["headline"]`` puts the out-of-domain share next to the
  main rate;
* reference problem v2 (D-533; ``docs/reference_problem_v2.md``; official-run plan batch 1): the table may be the v2
  table ``configs/dev_case_v1/reference_constraints_v2.csv`` (every row carries ``table_version =
  reference_problem_v2``; exactly one premise planning row ``SH-PLAN-T51-DGC-SHARE``; see :func:`_validate_rows`).
  A third Table 5-1 reading, ``table51_domain = "plan"``, judges the T rows by the **plan-level** domain status of
  the ration (primary variant, planned diet at the decision-time ``d_hat`` and the nominal composition; computed once
  per ration): in the domain the T rows are judged in every state, outside it they are unknown in every state.  The
  companion events ``main_reference_plan_domain`` and ``main_reference_plus_cp_hi_plan_domain`` stand next to the
  per-state events; which reading the main endpoint uses is decision F1 (to be confirmed at the protocol freeze), so
  ``summary()["headline"]`` reports both readings and the plan-level status;
* the reference energy check: linear vs reference verdicts, false passes / false fails, the difference;
* composition classes of the used feeds (analysis anomaly; support violation) as counts;
* the deterministic structural (planning) rows at the decision-time ``d_hat``;
* exact Clopper-Pearson bounds and the Monte Carlo standard error of every rate (simulation error for a fixed
  declared distribution only, contract T8.2 -- not data uncertainty).

Decision information (contract T2.1): ``q`` (kg as fed / head / d) is fixed.  In every state the realised DM is
``x_s = q * d_s``; it is never renormalised and ``q`` is never re-derived from the hidden ``d_s`` (the function
takes no planned-DM argument; the decision-time ``d_hat`` is used only for the structural planning rows).

Denominators: every rate has the full number of states ``S`` as denominator.  A state in which a member row is
undefined (missing cell, no DM, a reference-chain support violation) and no member is violated is ``unknown``: the
rate is reported as the pair ``[n_violated / S, (n_violated + n_unknown) / S]``; the screening convention of the
project (unknown counted as violated) is the upper value, and the Clopper-Pearson bounds are computed on it.

API
---
::

    table = load_reference_constraints("configs/dev_case_v1/reference_constraints.csv")
    spec  = ReferenceSpec.from_problem(problem, lin, table, dgc_ingredient_ids=("corn_grain_dry_ground_medium",),
                                       corn_silage_ingredient_ids=("corn_silage_typical",))
    res   = evaluate_reference(q, draws, spec)          # ReferenceEvaluation
    res.summary()                                       # JSON-safe dict (events, domain, energy, composition, ...)
    res.per_constraint_rows()                           # one dict per row and verdict (natural units)

Nothing here solves an optimisation problem.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..datamodel import ConstraintClass, DMSource, RationDecision, Sense
from ..hashing import stable_hash
from ..nutrition import domain as DOM
from ..nutrition.energy_reference import (
    MODEL_DIFFERENCE_LABEL,
    REFERENCE_CHAIN_ID,
    EnergyReferenceResult,
    EnergyReferenceSpec,
    joint_with_reference_energy,
    reference_energy_check,
)
from .evaluator import evaluate
from .stats import _count, clopper_pearson, mc_standard_error, one_sided_upper

__all__ = [
    "REFERENCE_SCHEMA",
    "TABLE51_DOMAIN_MODES",
    "MAIN_EVENT_T51_IGNORED",
    "MAIN_EVENT_PLAN_DOMAIN",
    "PREMISE_ROW_ID",
    "TABLE_VERSION_COLUMN",
    "TABLE_VERSION_V2",
    "V2_REQUIRED_COLUMNS",
    "H0_ELEVEN",
    "S2_SIX",
    "EXCLUDED_FIVE",
    "TABLE51_ROWS",
    "STATUSES",
    "ROLES",
    "VERDICT_MODELS",
    "THRESHOLD_SOURCE_STATUSES",
    "REQUIRED_COLUMNS",
    "ReferenceConstraintRow",
    "ReferenceConstraintTable",
    "EventDefinition",
    "ReferenceSpec",
    "ReferenceEvaluation",
    "load_reference_constraints",
    "default_events",
    "rate_block",
    "evaluate_reference",
]

#: /2 (FIX3_BC, round-3 red team C-1): main event domain-conditioned; ``EventDefinition.table51_domain``; headline.
#: /3 (reference problem v2, batch 1): ``table51_domain = "plan"``, the plan-level companion events, the plan-level
#: status in the headline and the domain block, v2 tables (premise planning row).
REFERENCE_SCHEMA = "ration_reliability.reference_evaluation/3"
#: The 11 probabilistic rows of the H0 joint event (configs/dev_case_v1/constraints.yaml joint_event.members).
H0_ELEVEN = ("PN-NEL-FIXEDDMI", "PN-CP-SUP", "PN-CP-HI", "PN-EE-HI", "PN-CA-ABS", "PN-P-ABS",
             "PN-T1", "PN-T2", "PN-T3", "PN-T4", "PN-T5")
#: The S2 event as declared by FIX_B: the probabilistic rows whose *value-block* status in constraints.yaml is
#: ``sourced``.  That older status vocabulary differs from ``reference_constraints.csv`` (which adjudicates the source
#: of the threshold only and marks PN-NEL-FIXEDDMI, PN-CA-ABS and PN-P-ABS ``sourced`` / ``main_reference`` too); the
#: six are therefore "the S2 declared six", not "the rows with sourced bounds" (round-3 red team C-2).
S2_SIX = ("PN-CP-SUP", "PN-T1", "PN-T2", "PN-T3", "PN-T4", "PN-T5")
#: The five rows S2 moved out of its event (planned rows + scenario diagnostics); round-3 review F2.
EXCLUDED_FIVE = ("PN-NEL-FIXEDDMI", "PN-CP-HI", "PN-EE-HI", "PN-CA-ABS", "PN-P-ABS")
TABLE51_ROWS = ("PN-T1", "PN-T2", "PN-T3", "PN-T4", "PN-T5")
#: Adjudicated nature of a row (instruction C.1).
STATUSES = ("sourced", "research_assumption", "planning_only", "diagnostic")
#: Role of a row in the reference problem (instruction C.1).
ROLES = ("main_reference", "optimization_only", "diagnostic")
#: How the row's verdict is obtained in the reference evaluation.
VERDICT_MODELS = ("public_evaluator_linear", "reference_chain_ch3", "structural_check_d_hat", "post_hoc_report")
THRESHOLD_SOURCE_STATUSES = ("sourced", "sourced_other_quantity", "research_assumption", "not_applicable")
MODEL_FORM_STATUSES = ("sourced", "research_assumption", "not_applicable")
REQUIRED_COLUMNS = ("constraint_id", "case_class", "variable", "unit", "expression", "sense", "threshold_public",
                    "threshold_source", "threshold_source_status", "model_form_status", "applicability_domain",
                    "status", "role", "verdict_model", "adjudication_rule", "assumption_rationale", "in_H0_11",
                    "in_S2_6", "in_excluded_5", "optimization_full_arm", "optimization_partial_arm",
                    "optimization_main9_arm",          # FIX3_BC (round-3 red team C-2): the MAIN9 training arm
                    "in_reference_problem", "case_yaml_status", "evidence_level", "decided_after_seeing_dev_results")
_YES_NO = ("yes", "no")
_ENERGY_VERDICTS = ("reference", "linear", "not_member")
#: How an event treats the Table 5-1 rows (FIX3_BC, C-1): ``primary`` = T-row verdicts used only in states inside the
#: primary Table 5-1 domain (``unknown`` elsewhere); ``ignored`` = T-row verdicts used in every state (as declared for
#: the H0 / S2 training events, and the R3C reading kept for comparison); ``plan`` (reference problem v2) = the domain
#: status is computed once from the ration's planned diet (``q``, decision-time ``d_hat``, nominal composition; primary
#: variant): in the domain the T rows are judged in every state, outside it they are ``unknown`` in every state.
TABLE51_DOMAIN_MODES = ("primary", "ignored", "plan")
#: Main event id of the R3C reading (T-row verdicts used out of domain; comparison only).
MAIN_EVENT_T51_IGNORED = "main_reference_t51_verdicts_used_out_of_domain"
#: Companion of ``main_reference`` with the plan-level Table 5-1 reading (reference problem v2; decision F1 pending).
MAIN_EVENT_PLAN_DOMAIN = "main_reference_plan_domain"
#: The premise planning row of reference problem v2 (D-533): structural, decision-time d_hat, nominal composition.
PREMISE_ROW_ID = DOM.PREMISE_PLANNING_ROW_ID
#: v2 marker (the detection rule of :func:`_validate_rows`): a table is v2 iff it has this column; every row must then
#: carry :data:`TABLE_VERSION_V2`.  A table without the column is v1 and may not contain the premise row.
TABLE_VERSION_COLUMN = "table_version"
TABLE_VERSION_V2 = "reference_problem_v2"
#: Columns a v2 table has in addition to :data:`REQUIRED_COLUMNS`.  ``optimization_all_arms``: the row's class common
#: to the three objective arms (the first word of the three arm columns when they agree, else ``arm_specific``).
V2_REQUIRED_COLUMNS = (TABLE_VERSION_COLUMN, "optimization_all_arms")
_ARM_COLUMNS = ("optimization_full_arm", "optimization_partial_arm", "optimization_main9_arm")


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    x = float(v)
    return x if math.isfinite(x) else None


# =====================================================================================================
# the reference constraint table
# =====================================================================================================

@dataclass(frozen=True)
class ReferenceConstraintRow:
    """One row of ``reference_constraints.csv`` (all columns kept as text in ``fields``)."""

    constraint_id: str
    status: str
    role: str
    verdict_model: str
    in_h0_11: bool
    in_s2_6: bool
    in_excluded_5: bool
    in_reference_problem: bool
    unit: str
    fields: Mapping[str, str]


@dataclass(frozen=True)
class ReferenceConstraintTable:
    """The adjudicated reference constraint table (validated on load; see :func:`load_reference_constraints`)."""

    rows: tuple[ReferenceConstraintRow, ...]
    source_path: Optional[str]
    file_sha256: Optional[str]

    def __post_init__(self) -> None:
        errs = _validate_rows(self.rows)
        if errs:
            raise ValueError("reference constraint table: " + "; ".join(errs))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(r.constraint_id for r in self.rows)

    def row(self, cid: str) -> ReferenceConstraintRow:
        for r in self.rows:
            if r.constraint_id == cid:
                return r
        raise KeyError(cid)

    def ids_where(self, **cond: Any) -> tuple[str, ...]:
        return tuple(r.constraint_id for r in self.rows if all(getattr(r, k) == v for k, v in cond.items()))

    @property
    def main_reference_ids(self) -> tuple[str, ...]:
        return self.ids_where(role="main_reference")

    @property
    def energy_row_id(self) -> Optional[str]:
        e = self.ids_where(verdict_model="reference_chain_ch3")
        return e[0] if e else None

    @property
    def table_version(self) -> str:
        """``reference_problem_v2`` for a v2 table (validated), ``reference_problem_v1`` for a table without the
        :data:`TABLE_VERSION_COLUMN` column."""
        return _table_version(self.rows) or "reference_problem_v1"

    @property
    def is_v2(self) -> bool:
        return self.table_version == TABLE_VERSION_V2

    @property
    def premise_row_id(self) -> Optional[str]:
        """:data:`PREMISE_ROW_ID` for a v2 table, ``None`` for v1."""
        return PREMISE_ROW_ID if self.is_v2 else None

    def fingerprint(self) -> str:
        return stable_hash("ReferenceConstraintTable/v1",
                           [(r.constraint_id, sorted(r.fields.items())) for r in self.rows])

    def to_record(self) -> dict[str, Any]:
        return {"source_path": self.source_path, "file_sha256": self.file_sha256, "fingerprint": self.fingerprint(),
                "table_version": self.table_version, "premise_row": self.premise_row_id,
                "n_rows": len(self.rows), "main_reference": list(self.main_reference_ids),
                "energy_row": self.energy_row_id,
                "h0_eleven": list(self.ids_where(in_h0_11=True)), "s2_six": list(self.ids_where(in_s2_6=True)),
                "excluded_five": list(self.ids_where(in_excluded_5=True)),
                "by_status": {s: list(self.ids_where(status=s)) for s in STATUSES},
                "by_role": {r: list(self.ids_where(role=r)) for r in ROLES}}


def _yes(v: str, col: str, cid: str, errs: list[str]) -> bool:
    s = str(v).strip()
    if s not in _YES_NO:
        errs.append(f"{cid}: {col} must be yes/no (got {s!r})")
    return s == "yes"


def _validate_rows(rows: Sequence[ReferenceConstraintRow]) -> list[str]:
    errs: list[str] = []
    ids = [r.constraint_id for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("duplicate constraint ids")
    for r in rows:
        if r.status not in STATUSES:
            errs.append(f"{r.constraint_id}: status {r.status!r} not in {STATUSES}")
        if r.role not in ROLES:
            errs.append(f"{r.constraint_id}: role {r.role!r} not in {ROLES}")
        if r.verdict_model not in VERDICT_MODELS:
            errs.append(f"{r.constraint_id}: verdict_model {r.verdict_model!r} not in {VERDICT_MODELS}")
        tss = r.fields.get("threshold_source_status")
        if tss is not None and tss not in THRESHOLD_SOURCE_STATUSES:
            errs.append(f"{r.constraint_id}: threshold_source_status {tss!r} not in {THRESHOLD_SOURCE_STATUSES}")
        mfs = r.fields.get("model_form_status")
        if mfs is not None and mfs not in MODEL_FORM_STATUSES:
            errs.append(f"{r.constraint_id}: model_form_status {mfs!r} not in {MODEL_FORM_STATUSES}")
        # the adjudication rules (docs/reference_problem_v1.md section 3)
        if r.role == "main_reference":
            if r.status != "sourced":
                errs.append(f"{r.constraint_id}: a main_reference row must be status 'sourced'")
            if tss is not None and tss != "sourced":
                errs.append(f"{r.constraint_id}: a main_reference row needs a threshold sourced for its own quantity")
            if not r.in_h0_11:
                errs.append(f"{r.constraint_id}: main_reference rows are chosen among the 11 H0 rows only (no new rows)")
            if not r.in_reference_problem:
                errs.append(f"{r.constraint_id}: a main_reference row must be a compiled row of the reference problem")
        if r.status == "research_assumption" and r.role == "main_reference":
            errs.append(f"{r.constraint_id}: a research_assumption row cannot be main_reference")
        if r.status == "planning_only" and r.role != "optimization_only":
            errs.append(f"{r.constraint_id}: planning_only rows have role optimization_only")
        if r.status == "diagnostic" and r.role != "diagnostic":
            errs.append(f"{r.constraint_id}: diagnostic rows have role diagnostic")
        if r.status == "planning_only" and r.verdict_model != "structural_check_d_hat":
            errs.append(f"{r.constraint_id}: planning_only rows are checked on the plan (structural_check_d_hat)")
        if r.in_h0_11 and r.verdict_model not in ("public_evaluator_linear", "reference_chain_ch3"):
            errs.append(f"{r.constraint_id}: an H0 row is judged by the evaluator or the reference chain")
    h0 = {r.constraint_id for r in rows if r.in_h0_11}
    s2 = {r.constraint_id for r in rows if r.in_s2_6}
    ex = {r.constraint_id for r in rows if r.in_excluded_5}
    if h0 != set(H0_ELEVEN):
        errs.append(f"in_H0_11 must mark exactly {sorted(H0_ELEVEN)} (got {sorted(h0)})")
    if s2 != set(S2_SIX):
        errs.append(f"in_S2_6 must mark exactly {sorted(S2_SIX)} (got {sorted(s2)})")
    if ex != set(EXCLUDED_FIVE):
        errs.append(f"in_excluded_5 must mark exactly {sorted(EXCLUDED_FIVE)} (got {sorted(ex)})")
    if s2 & ex or (s2 | ex) != h0:
        errs.append("the S2 six and the excluded five must partition the H0 eleven")
    energy = [r.constraint_id for r in rows if r.verdict_model == "reference_chain_ch3"]
    if len(energy) > 1:
        errs.append(f"at most one row is judged by the reference chain (got {energy})")
    if not any(r.role == "main_reference" for r in rows):
        errs.append("no main_reference row")
    _validate_version(rows, errs)
    return errs


def _table_version(rows: Sequence[ReferenceConstraintRow]) -> Optional[str]:
    """Value of :data:`TABLE_VERSION_COLUMN` (first row that has the column), ``None`` if no row has it (v1)."""
    present = [r.fields.get(TABLE_VERSION_COLUMN) for r in rows if r.fields.get(TABLE_VERSION_COLUMN) is not None]
    return str(present[0]).strip() if present else None


def _first_word(s: Any) -> str:
    """Leading token of an arm column (``probabilistic_nutrition（...）`` -> ``probabilistic_nutrition``)."""
    return re.split(r"[（(\s]", str(s).strip(), maxsplit=1)[0]


def _validate_version(rows: Sequence[ReferenceConstraintRow], errs: list[str]) -> None:
    """v1 / v2 rule (reference problem v2, D-533; ``docs/reference_problem_v2.md`` §5).

    Detection: a table is **v2 iff it has the column** :data:`TABLE_VERSION_COLUMN`.  Then every row must say
    :data:`TABLE_VERSION_V2`, every row needs a non-empty ``optimization_all_arms``, and there must be **exactly one**
    premise planning row :data:`PREMISE_ROW_ID` (planning_only / optimization_only / structural_check_d_hat, in no
    event, a compiled row of the reference problem, ``ge 0``, threshold a research assumption, structural_hard in every
    arm).  A table without the column is v1 and may not contain the premise row.  Dropping the row from a v2 table,
    or pasting it into a v1 table, is therefore refused; the marker column is data, not a file name."""
    has_col = any(r.fields.get(TABLE_VERSION_COLUMN) is not None for r in rows)
    premise = [r for r in rows if r.constraint_id == PREMISE_ROW_ID]
    if not has_col:
        if premise:
            errs.append(f"{PREMISE_ROW_ID} belongs to reference problem v2; a table without the {TABLE_VERSION_COLUMN} "
                        f"column is v1 and may not contain it")
        return
    bad = sorted({str(r.fields.get(TABLE_VERSION_COLUMN)) for r in rows
                  if str(r.fields.get(TABLE_VERSION_COLUMN, "")).strip() != TABLE_VERSION_V2})
    if bad:
        errs.append(f"{TABLE_VERSION_COLUMN} must be {TABLE_VERSION_V2!r} in every row (got {bad})")
    for col in V2_REQUIRED_COLUMNS:
        empty = [r.constraint_id for r in rows if not str(r.fields.get(col) or "").strip()]
        if empty:
            errs.append(f"a v2 table needs a non-empty {col} in every row (missing in {empty})")
    if len(premise) != 1:
        errs.append(f"a v2 table needs exactly one premise planning row {PREMISE_ROW_ID} (got {len(premise)})")
    for r in premise:
        f = r.fields
        if (r.status, r.role, r.verdict_model) != ("planning_only", "optimization_only", "structural_check_d_hat"):
            errs.append(f"{PREMISE_ROW_ID}: must be planning_only / optimization_only / structural_check_d_hat")
        if r.in_h0_11 or r.in_s2_6 or r.in_excluded_5:
            errs.append(f"{PREMISE_ROW_ID}: a planning row is a member of no event (H0 / S2 / excluded five)")
        if not r.in_reference_problem:
            errs.append(f"{PREMISE_ROW_ID}: a compiled row of the v2 reference problem (in_reference_problem = yes)")
        try:
            zero = float(f.get("threshold_public", "")) == 0.0
        except ValueError:
            zero = False
        if str(f.get("sense", "")).strip() != "ge" or not zero:
            errs.append(f"{PREMISE_ROW_ID}: the planning row reads sum_i q_i d_hat_i coef_i >= 0 (sense ge, bound 0)")
        if f.get("threshold_source_status") not in (None, "research_assumption"):
            errs.append(f"{PREMISE_ROW_ID}: tau is a project research assumption (threshold_source_status)")
        arms = [_first_word(f.get(c, "")) for c in _ARM_COLUMNS + ("optimization_all_arms",)]
        if any(a != "structural_hard" for a in arms):
            errs.append(f"{PREMISE_ROW_ID}: structural_hard in every objective arm and in optimization_all_arms")
    for r in rows:
        oa = str(r.fields.get("optimization_all_arms") or "").strip()
        if not oa:
            continue
        words = {_first_word(r.fields.get(c, "")) for c in _ARM_COLUMNS}
        expected = next(iter(words)) if len(words) == 1 else "arm_specific"
        if _first_word(oa) != expected:
            errs.append(f"{r.constraint_id}: optimization_all_arms {_first_word(oa)!r} disagrees with the three arm "
                        f"columns (expected {expected!r})")


def load_reference_constraints(path: str | Path) -> ReferenceConstraintTable:
    """Read and validate ``reference_constraints.csv`` or ``reference_constraints_v2.csv`` (UTF-8).  Every column of
    :data:`REQUIRED_COLUMNS` must be present (a v2 table -- one with :data:`TABLE_VERSION_COLUMN` -- also every column
    of :data:`V2_REQUIRED_COLUMNS`); enumerations, yes/no flags, the 11 / 6 / 5 partition, the adjudication rules and
    the v1 / v2 rule of :func:`_validate_version` are checked."""
    p = Path(path)
    raw = p.read_bytes()
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    cols = tuple(reader.fieldnames or ())
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if TABLE_VERSION_COLUMN in cols:
        missing += [c for c in V2_REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}")
    rows: list[ReferenceConstraintRow] = []
    errs: list[str] = []
    for rec in reader:
        cid = str(rec["constraint_id"]).strip()
        rows.append(ReferenceConstraintRow(
            constraint_id=cid, status=str(rec["status"]).strip(), role=str(rec["role"]).strip(),
            verdict_model=str(rec["verdict_model"]).strip(),
            in_h0_11=_yes(rec["in_H0_11"], "in_H0_11", cid, errs), in_s2_6=_yes(rec["in_S2_6"], "in_S2_6", cid, errs),
            in_excluded_5=_yes(rec["in_excluded_5"], "in_excluded_5", cid, errs),
            in_reference_problem=_yes(rec["in_reference_problem"], "in_reference_problem", cid, errs),
            unit=str(rec["unit"]).strip(), fields={k: str(v) for k, v in rec.items()}))
    if errs:
        raise ValueError(f"{p}: " + "; ".join(errs))
    return ReferenceConstraintTable(tuple(rows), str(path), hashlib.sha256(raw).hexdigest())


# =====================================================================================================
# events and the specification
# =====================================================================================================

@dataclass(frozen=True)
class EventDefinition:
    """A joint violation event: ``1{any member violated}``; the energy row (if a member) is judged by
    ``energy_verdict`` (``reference`` chain or ``linear`` row).

    ``table51_domain`` (FIX3_BC, C-1): ``primary`` -- the Table 5-1 rows among the members are judged only in states
    inside the primary Table 5-1 domain; outside it (``not_assessable`` / ``undefined``) they are ``unknown``, so the
    state is violated only if another member is violated and is otherwise ``unknown`` (upper value of the rate pair);
    ``ignored`` -- T-row verdicts are used in every state; ``plan`` (reference problem v2) -- the primary variant's
    domain status of the ration as formulated (planned diet at the decision-time ``d_hat`` and the nominal composition,
    one status per ration) decides for every state: in the domain the T rows are judged in every state (the event then
    equals the ``ignored`` reading), outside it they are ``unknown`` in every state."""

    event_id: str
    members: tuple[str, ...]
    energy_verdict: str
    description: str
    table51_domain: str = "ignored"

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError(f"event {self.event_id}: members must be non-empty and unique")
        if self.energy_verdict not in _ENERGY_VERDICTS:
            raise ValueError(f"event {self.event_id}: energy_verdict must be one of {_ENERGY_VERDICTS}")
        if self.table51_domain not in TABLE51_DOMAIN_MODES:
            raise ValueError(f"event {self.event_id}: table51_domain must be one of {TABLE51_DOMAIN_MODES}")


def default_events(table: ReferenceConstraintTable) -> tuple[EventDefinition, ...]:
    """The events every ration is scored on (instruction C.3; FIX3_BC C-1):

    * ``main_reference`` -- main rows, energy by the reference chain, **Table 5-1 rows judged only inside the primary
      domain** (unknown outside it); ``main_reference_linear_energy`` -- the same with the linear energy row;
      ``main_reference_t51_verdicts_used_out_of_domain`` -- the R3C reading (T verdicts used in every state;
      comparison only, never the main endpoint); ``main_reference_plus_cp_hi`` -- main rows + PN-CP-HI (C-5: shows
      the effect of adjudication A2);
    * ``h0_eleven`` (the H0 joint event as declared = the FULL11 training event, linear energy row, T verdicts in
      every state), ``h0_eleven_reference_energy`` and ``h0_eleven_domain_conditioned``;
    * ``s2_six`` (the S2 declared six = the PART6P5 training event) and ``s2_six_domain_conditioned``;
    * ``excluded_five`` (energy by the reference chain; no Table 5-1 row);
    * reference problem v2 (batch 1): the plan-level companions ``main_reference_plan_domain`` (next to
      ``main_reference``) and ``main_reference_plus_cp_hi_plan_domain`` (next to ``main_reference_plus_cp_hi``): same
      rows and energy verdict, Table 5-1 rows read at the plan level (``table51_domain = "plan"``).  Which reading is
      the main endpoint is decision F1, to be confirmed at the protocol freeze; both are always reported.  Every
      earlier event id is unchanged."""
    e = table.energy_row_id
    main = table.main_reference_ids
    main_energy = "reference" if e in main else "not_member"
    ev = [EventDefinition("main_reference", main, main_energy,
                          "union of the main_reference rows (adjudicated by source); energy row judged by the Chapter 3 "
                          "reference chain; Table 5-1 rows judged only inside the primary Table 5-1 domain (unknown "
                          "outside it)", table51_domain="primary"),
          EventDefinition(MAIN_EVENT_PLAN_DOMAIN, main, main_energy,
                          "main_reference with the plan-level Table 5-1 reading: the ration's primary-variant domain "
                          "status at the decision-time d_hat and nominal composition decides for every state (in the "
                          "domain: T rows judged in every state; outside: unknown in every state); reference problem "
                          "v2 companion, reported next to main_reference (decision F1 pending)", table51_domain="plan")]
    if e in main:
        ev.append(EventDefinition("main_reference_linear_energy", main, "linear",
                                  "same rows and domain rule; energy row judged by the linear fixed-DMI row (comparison "
                                  "only)", table51_domain="primary"))
    ev.append(EventDefinition(MAIN_EVENT_T51_IGNORED, main, main_energy,
                              "R3C reading of the main event: Table 5-1 verdicts used also outside their domain "
                              "(comparison only; not the main endpoint, round-3 red team C-1)", table51_domain="ignored"))
    if "PN-CP-HI" in table.ids and "PN-CP-HI" not in main:
        # round-3 red team C-5: the adjudication A2 (made after seeing the dev run) moved PN-CP-HI out of the main
        # event; the main event + CP-HI is reported next to it in every table, so the adjudication's effect stays visible
        ev.append(EventDefinition("main_reference_plus_cp_hi", tuple(main) + ("PN-CP-HI",), main_energy,
                                  "main reference rows + the research-assumption row PN-CP-HI (same energy and domain "
                                  "rules); reported next to the main event, never instead of it",
                                  table51_domain="primary"))
        ev.append(EventDefinition("main_reference_plus_cp_hi_plan_domain", tuple(main) + ("PN-CP-HI",), main_energy,
                                  "main_reference_plus_cp_hi with the plan-level Table 5-1 reading (reference problem "
                                  "v2 companion; keeps the effect of adjudication A2 visible under either reading)",
                                  table51_domain="plan"))
    ev += [EventDefinition("h0_eleven", H0_ELEVEN, "linear", "the H0 joint event as declared (11 rows, linear energy row)"),
           EventDefinition("h0_eleven_reference_energy", H0_ELEVEN, "reference",
                           "the 11 H0 rows with the energy row judged by the reference chain"),
           EventDefinition("h0_eleven_domain_conditioned", H0_ELEVEN, "linear",
                           "the H0 event with Table 5-1 rows judged only inside the primary domain",
                           table51_domain="primary"),
           EventDefinition("s2_six", S2_SIX, "not_member",
                           "the S2 declared six (FIX_B; 'sourced' in constraints.yaml's value-block vocabulary)"),
           EventDefinition("s2_six_domain_conditioned", S2_SIX, "not_member",
                           "the S2 declared six with Table 5-1 rows judged only inside the primary domain",
                           table51_domain="primary"),
           EventDefinition("excluded_five", EXCLUDED_FIVE, "reference",
                           "union of the five rows S2 moved out of its event; energy row judged by the reference chain")]
    return tuple(ev)


def _premise_row_errors(cc: Any, nominal_theta: Optional[np.ndarray], primary: DOM.Table51DomainSpec) -> list[str]:
    """v2: the compiled premise planning row must be the structural, decision-time-``d_hat`` form of the **primary**
    Table 5-1 variant (same counted ingredients, same tau, same tolerance), so that a ration meeting it is in the domain
    at the plan level.  Messages carry no value (the coefficients derive from restricted table values)."""
    errs: list[str] = []
    ids = list(cc.constraint_ids)
    if PREMISE_ROW_ID not in ids:
        return [f"v2 table: the reference problem has no compiled row {PREMISE_ROW_ID}"]
    k = ids.index(PREMISE_ROW_ID)
    if cc.classes[k] is not ConstraintClass.STRUCTURAL_HARD or cc.dm_sources[k] is not DMSource.DECISION_ESTIMATE:
        errs.append(f"{PREMISE_ROW_ID}: must be structural_hard with dm_source decision_estimate")
    if cc.senses[k] is not Sense.GE or float(cc.bound[k]) != 0.0:
        errs.append(f"{PREMISE_ROW_ID}: must read >= 0")
    if np.any(np.asarray(cc.W[k]) != 0.0) or np.any(np.asarray(cc.v[k]) != 0.0):
        errs.append(f"{PREMISE_ROW_ID}: only planned-DM (C:<coef>:DM) terms are allowed")
    tol = float(cc.tol[k]) / float(cc.unit_factor[k])
    if not math.isclose(tol, float(primary.tolerance_kg_d), rel_tol=1e-12, abs_tol=0.0):
        errs.append(f"{PREMISE_ROW_ID}: numerical tolerance differs from the primary variant's tolerance_kg_d")
    if nominal_theta is not None:
        try:
            coef = DOM.premise_planning_coefficients(nominal_theta, cc.ingredient_ids, cc.nutrient_ids,
                                                     primary.counted_ingredient_ids, primary.threshold,
                                                     starch_column=primary.starch_column)
        except (KeyError, ValueError) as exc:
            return errs + [f"{PREMISE_ROW_ID}: cannot form the primary variant's coefficients ({type(exc).__name__})"]
        if not np.allclose(np.asarray(cc.w0[k], dtype=float), coef, rtol=1e-12, atol=1e-15):
            errs.append(f"{PREMISE_ROW_ID}: compiled coefficients are not (1{{counted}} - tau) * nominal starch of the "
                        f"primary Table 5-1 variant {primary.variant_id} (counted ingredients or tau differ)")
    return errs


@dataclass(frozen=True)
class ReferenceSpec:
    """Everything :func:`evaluate_reference` holds fixed (one physical definition for every ration).

    ``constraints``: compiled constraints of the reference problem (every H0 row compiled; the full problem, not an
    objective arm).  ``d_hat``: decision-time DM estimates of the reference problem, used only for the structural
    planning rows when ``q`` is a plain array (a :class:`RationDecision` carries its own).  ``nominal_theta``:
    nominal composition (per-ration nominal domain status); optional, but required by events with
    ``table51_domain = "plan"`` (the default events include two).  With a v2 table (reference problem v2) the compiled
    premise planning row must encode the primary Table 5-1 variant (checked here; see :func:`_premise_row_errors`).
    """

    constraints: Any
    table: ReferenceConstraintTable
    energy: EnergyReferenceSpec
    domain_variants: tuple[DOM.Table51DomainSpec, ...]
    events: tuple[EventDefinition, ...]
    d_hat: np.ndarray
    prices: Optional[Any] = None
    nominal_theta: Optional[np.ndarray] = None
    confidence: float = 0.95
    reference_problem_id: str = ""

    def __post_init__(self) -> None:
        cc = self.constraints
        errs: list[str] = []
        compiled = list(cc.constraint_ids)
        classes = {cid: cc.classes[k] for k, cid in enumerate(compiled)}
        table_ref = set(self.table.ids_where(in_reference_problem=True))
        if set(compiled) != table_ref:
            errs.append(f"table rows with in_reference_problem=yes {sorted(table_ref)} must equal the compiled rows "
                        f"{sorted(compiled)} (every row of the reference problem is listed exactly once)")
        for cid in self.table.main_reference_ids:
            if classes.get(cid) is not ConstraintClass.PROBABILISTIC_NUTRITION:
                errs.append(f"{cid}: main_reference rows must be probabilistic_nutrition in the reference problem")
        evaluated = {cid for cid, c in classes.items() if c is not ConstraintClass.STRUCTURAL_HARD}
        for e in self.events:
            bad = [m for m in e.members if m not in evaluated]
            if bad:
                errs.append(f"event {e.event_id}: members {bad} are not evaluated rows of the reference problem")
            if e.energy_verdict != "not_member" and self.energy.constraint_id not in e.members:
                errs.append(f"event {e.event_id}: energy_verdict {e.energy_verdict} but the energy row is not a member")
            if e.energy_verdict == "not_member" and self.energy.constraint_id in e.members:
                errs.append(f"event {e.event_id}: the energy row is a member; say 'reference' or 'linear'")
        if len({e.event_id for e in self.events}) != len(self.events):
            errs.append("duplicate event ids")
        if self.table.energy_row_id is not None and self.table.energy_row_id != self.energy.constraint_id:
            errs.append(f"the table's reference-chain row {self.table.energy_row_id} differs from the energy spec "
                        f"{self.energy.constraint_id}")
        if tuple(self.energy.lin.ingredient_ids) != tuple(cc.ingredient_ids):
            errs.append("energy linearisation and reference problem have different ingredient order")
        dh = np.asarray(self.d_hat, dtype=float)
        if dh.shape != (len(cc.ingredient_ids),) or not np.all(np.isfinite(dh)) or np.any(dh <= 0) or np.any(dh > 1):
            errs.append("d_hat must be finite in (0, 1] with one entry per ingredient")
        object.__setattr__(self, "d_hat", _ro(dh.copy()))
        if self.nominal_theta is not None:
            nt = np.asarray(self.nominal_theta, dtype=float)
            if nt.shape != (len(cc.ingredient_ids), len(cc.nutrient_ids)):
                errs.append("nominal_theta must be [I, J] of the reference problem")
            object.__setattr__(self, "nominal_theta", _ro(nt.copy()))
        if not (0.0 < float(self.confidence) < 1.0):
            errs.append("confidence must be in (0, 1)")
        if not self.domain_variants or sum(v.role == "primary" for v in self.domain_variants) != 1:
            errs.append("exactly one primary Table 5-1 domain variant is required")
        elif self.table.is_v2:
            primary = next(v for v in self.domain_variants if v.role == "primary")
            errs += _premise_row_errors(cc, self.nominal_theta, primary)
        if any(e.table51_domain == "plan" for e in self.events) and self.nominal_theta is None:
            errs.append("events with table51_domain='plan' need nominal_theta (plan-level Table 5-1 status)")
        object.__setattr__(self, "domain_variants", tuple(self.domain_variants))
        object.__setattr__(self, "events", tuple(self.events))
        if errs:
            raise ValueError("ReferenceSpec: " + "; ".join(errs))

    @classmethod
    def from_problem(cls, problem: Any, lin: Any, table: ReferenceConstraintTable, *,
                     dgc_ingredient_ids: Sequence[str], corn_silage_ingredient_ids: Sequence[str] = (),
                     events: Optional[Sequence[EventDefinition]] = None,
                     domain_variants: Optional[Sequence[DOM.Table51DomainSpec]] = None,
                     confidence: float = 0.95) -> "ReferenceSpec":
        """Spec of the reference problem ``problem`` (a :class:`RationProblem` with every H0 row compiled)."""
        e_id = table.energy_row_id
        if e_id is None:
            raise ValueError("the table names no reference-chain energy row")
        row = [c for c in problem.constraints if c.constraint_id == e_id]
        if len(row) != 1:
            raise ValueError(f"the reference problem has no single constraint {e_id}")
        energy = EnergyReferenceSpec.from_constraint(lin, row[0])
        dv = tuple(domain_variants) if domain_variants is not None else DOM.default_table51_variants(
            dgc_ingredient_ids, corn_silage_ingredient_ids=tuple(corn_silage_ingredient_ids))
        return cls(constraints=problem.compiled, table=table, energy=energy, domain_variants=dv,
                   events=tuple(events) if events is not None else default_events(table),
                   d_hat=np.asarray(problem.dm_estimates(), dtype=float), prices=problem.prices,
                   nominal_theta=np.asarray(problem.nominal_theta(), dtype=float), confidence=float(confidence),
                   reference_problem_id=str(problem.problem_id))

    def fingerprint(self) -> str:
        return stable_hash("ReferenceSpec/v1", REFERENCE_SCHEMA, self.constraints.fingerprint, self.table.fingerprint(),
                           self.energy.fingerprint(), [v.fingerprint() for v in self.domain_variants],
                           [(e.event_id, e.members, e.energy_verdict, e.table51_domain) for e in self.events], self.d_hat,
                           float(self.confidence), self.reference_problem_id)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe description (no ingredient-level values)."""
        return {"schema": REFERENCE_SCHEMA, "fingerprint": self.fingerprint(),
                "reference_problem_id": self.reference_problem_id,
                "compiled_constraints_fingerprint": self.constraints.fingerprint,
                "table": self.table.to_record(), "energy": self.energy.to_record(),
                "domain_variants": [v.to_record() for v in self.domain_variants],
                "events": [{"event_id": e.event_id, "members": list(e.members), "energy_verdict": e.energy_verdict,
                            "table51_domain": e.table51_domain, "description": e.description} for e in self.events],
                "confidence": float(self.confidence),
                "rate_convention": ("denominator = all states; rate_lower = n_violated / S; rate_upper = (n_violated + "
                                    "n_unknown) / S (unknown counted as violated, the screening convention); "
                                    "Clopper-Pearson bounds and MC standard error on the upper count; MC error of a "
                                    "fixed declared distribution only (T8.2)"),
                "table51_domain_rule": ("events with table51_domain='primary' judge PN-T1..T5 only in states inside the "
                                        "primary Table 5-1 domain; outside it (not_assessable / undefined) the T rows "
                                        "are unknown -- never counted as passed or failed (FIX3_BC, C-1)"),
                "table51_plan_domain_rule": ("events with table51_domain='plan' (reference problem v2) take the primary "
                                             "variant's status of the ration as formulated (q, decision-time d_hat, "
                                             "nominal composition; one status per ration): in the domain PN-T1..T5 are "
                                             "judged in every state, outside it they are unknown in every state; which "
                                             "reading is the main endpoint is decision F1 (to be confirmed at freeze)"),
                "table_version": self.table.table_version,
                "decision_information": "q fixed; x = q * d_state, never renormalised; q never re-derived from the "
                                        "hidden DM; d_hat only for the structural planning rows"}


# =====================================================================================================
# result
# =====================================================================================================

def rate_block(n_violated: int, n_unknown: int, n: int, confidence: float = 0.95) -> dict[str, Any]:
    """Counts, the rate pair and the Monte Carlo statistics of one event or row (denominator ``n`` = all states)."""
    n = _count(n, "n", positive=True)
    v = _count(n_violated, "n_violated")
    u = _count(n_unknown, "n_unknown")
    if v + u > n:
        raise ValueError("rate_block: disjoint violated and unknown counts cannot exceed n")
    k = v + u
    lo, hi = clopper_pearson(k, n, confidence)
    return {"n_states": n, "n_violated": v, "n_unknown": u, "rate_lower": v / n, "rate_upper": k / n,
            "cp_upper_one_sided": one_sided_upper(k, n, confidence), "cp_two_sided_lower": lo,
            "cp_two_sided_upper": hi, "mc_se": mc_standard_error(k / n, n), "confidence": float(confidence)}


def _deficit_stats(margin: np.ndarray, violated: np.ndarray) -> dict[str, Any]:
    """Size of the violation in the row's natural unit: deficit = -margin in violated states; margin quantiles over
    defined states."""
    m = np.asarray(margin, dtype=float)
    fin = np.isfinite(m)
    dv = -m[violated & fin]
    qs = np.quantile(m[fin], [0.01, 0.05, 0.5]) if fin.any() else [None, None, None]
    return {"mean_deficit_given_violation": _f(dv.mean()) if dv.size else None,
            "max_deficit": _f(dv.max()) if dv.size else None,
            "margin_q01": _f(qs[0]), "margin_q05": _f(qs[1]), "margin_q50": _f(qs[2])}


@dataclass(frozen=True)
class ReferenceEvaluation:
    """Outcome of :func:`evaluate_reference` for one ration on one set of states.

    ``event_violation`` / ``event_unknown``: per-state arrays ``[S]`` of every event (read-only).  ``evaluation``:
    the public evaluator's :class:`EvaluationResult` of the reference problem.  ``energy``: the per-state reference
    energy check.  ``domain``: per-variant Table 5-1 domain results.  ``plan_domain``: the primary variant's
    plan-level status of the ration (reference problem v2; ``None`` without a nominal composition in the spec).
    """

    spec_fingerprint: str
    identity: Mapping[str, Any]
    event_violation: Mapping[str, np.ndarray]
    event_unknown: Mapping[str, np.ndarray]
    events: Mapping[str, dict]
    evaluation: Any
    energy: EnergyReferenceResult
    domain: Mapping[str, DOM.Table51DomainResult]
    plan_domain: Optional[DOM.PlanDomainStatus] = None
    _summary: Mapping[str, Any] = field(repr=False, default_factory=dict)
    _rows: tuple = field(repr=False, default=())

    @property
    def n_states(self) -> int:
        return int(self.identity["n_states"])

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary: identity, events, per-row results, domain, energy, composition, structural, cost."""
        return dict(self._summary)

    def per_constraint_rows(self) -> list[dict[str, Any]]:
        """One dict per evaluated row and verdict (the energy row twice: linear and reference chain)."""
        return [dict(r) for r in self._rows]


# =====================================================================================================
# the evaluation
# =====================================================================================================

def evaluate_reference(q: Any, draws: Any, spec: ReferenceSpec) -> ReferenceEvaluation:
    """Score the executed ration ``q`` on the states ``draws`` against the fixed reference definition ``spec``.

    Parameters
    ----------
    q : :class:`RationDecision` (its ``d_hat`` is used for the structural planning rows) or ``[I]`` kg as fed /
        head / d in the order of the reference problem (``spec.d_hat`` is then used for the planning rows).  Fixed.
    draws : :class:`~ration_reliability.uncertainty.base.DrawSet` with the reference problem's ingredient and
        nutrient labels (including the energy column); any world -- the result records its stream id and model
        fingerprint.
    spec : :class:`ReferenceSpec`.
    """
    cc = spec.constraints
    if tuple(draws.ingredient_ids) != tuple(cc.ingredient_ids) or tuple(draws.nutrient_ids) != tuple(cc.nutrient_ids):
        raise ValueError("evaluate_reference: draw labels differ from the reference problem (no reordering)")
    conf = float(spec.confidence)
    is_dec = isinstance(q, RationDecision)
    ev = evaluate(q, draws.theta, draws.d, cc, d_hat=None if is_dec else spec.d_hat, prices=spec.prices,
                  draw_stream_id=getattr(draws, "stream_id", None),
                  is_synthetic=bool(getattr(draws, "is_synthetic", False)))
    en = reference_energy_check(q, draws, spec.energy)
    if en.q_hash != ev.q_hash:
        raise ValueError("evaluate_reference: evaluator and energy check saw different q")
    S = int(ev.n_draws)
    ids = list(ev.constraint_ids)
    viol = np.asarray(ev.violated, dtype=bool)
    und = np.asarray(ev.undefined, dtype=bool)
    e_id = spec.energy.constraint_id

    # ---- Table 5-1 domain of every declared variant (computed first: the main event is domain-conditioned, C-1)
    dom: dict[str, DOM.Table51DomainResult] = {v.variant_id: DOM.table5_1_domain(q, draws, v)
                                               for v in spec.domain_variants}
    primary_variant = next(v for v in spec.domain_variants if v.role == "primary")
    in_primary = np.asarray(dom[primary_variant.variant_id].status == DOM.DOMAIN_IN, dtype=bool)
    # plan-level status (reference problem v2): computed ONCE from the planned diet -- q, the decision-time d_hat (the
    # decision's own for a RationDecision) and the nominal composition -- with the primary variant's share definition
    dh_plan = np.asarray(q.d_hat if is_dec else spec.d_hat, dtype=float)

    def _plan(v: DOM.Table51DomainSpec) -> Optional[DOM.PlanDomainStatus]:
        if spec.nominal_theta is None:
            return None
        return DOM.plan_domain_status(q, spec.nominal_theta, dh_plan, cc.ingredient_ids, cc.nutrient_ids, v)

    plan_primary = _plan(primary_variant)
    in_plan = plan_primary.state_mask(S) if plan_primary is not None else None

    def _mask(mode: str) -> Optional[np.ndarray]:
        """In-domain mask of a Table 5-1 reading (``None`` = T verdicts used in every state)."""
        if mode == "primary":
            return in_primary
        if mode == "plan":
            if in_plan is None:
                raise ValueError("evaluate_reference: a plan-level event needs spec.nominal_theta")
            return in_plan
        return None

    def _joint(e: EventDefinition, in_domain: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(violated [S], unknown [S]) of event ``e``; with ``in_domain`` the T rows are unknown outside it."""
        t_members = [m for m in e.members if m in TABLE51_ROWS] if in_domain is not None else []
        others = [m for m in e.members if m not in t_members]
        if others:
            if e.energy_verdict == "reference":
                jt = joint_with_reference_energy(ev, en, member_ids=others)   # checks q, stream and margins
                jv, ju = np.asarray(jt["joint_violation"], bool), np.asarray(jt["joint_unknown"], bool)
            else:
                k = [ids.index(m) for m in others]
                jv = viol[:, k].any(axis=1)
                ju = ~jv & und[:, k].any(axis=1)
        else:
            jv, ju = np.zeros(S, dtype=bool), np.zeros(S, dtype=bool)
        if t_members:
            kt = [ids.index(m) for m in t_members]
            tv = viol[:, kt].any(axis=1) & in_domain                  # T verdicts only inside the domain
            tu = (und[:, kt].any(axis=1) & in_domain) | ~in_domain    # outside: unknown, never passed or failed
            jv = jv | tv
            ju = ~jv & (ju | tu)
        return jv, ju

    # ---- events
    ev_v: dict[str, np.ndarray] = {}
    ev_u: dict[str, np.ndarray] = {}
    events: dict[str, dict] = {}
    for e in spec.events:
        jv, ju = _joint(e, _mask(e.table51_domain))
        ev_v[e.event_id], ev_u[e.event_id] = _ro(jv), _ro(ju)
        events[e.event_id] = {"members": list(e.members), "energy_verdict": e.energy_verdict,
                              "table51_domain": e.table51_domain,
                              "table51_domain_variant": primary_variant.variant_id
                              if e.table51_domain in ("primary", "plan") else None,
                              "description": e.description, **rate_block(int(jv.sum()), int(ju.sum()), S, conf)}
        if e.table51_domain == "plan":
            events[e.event_id]["t51_plan_domain_status"] = plan_primary.status   # label only (no share / margin)

    # ---- per-row results (natural = declared units)
    trow = {r.constraint_id: r for r in spec.table.rows}
    groups = {"main_reference": set(spec.table.main_reference_ids), "h0_eleven": set(H0_ELEVEN),
              "s2_six": set(S2_SIX), "excluded_five": set(EXCLUDED_FIVE)}
    rows: list[dict[str, Any]] = []
    for k, cid in enumerate(ids):
        tr = trow.get(cid)
        v_k, u_k = viol[:, k], und[:, k] & ~viol[:, k]
        base = {"constraint_id": cid, "verdict": "public_evaluator_linear", "class_in_reference_problem":
                ev.constraint_classes[k], "status": tr.status if tr else None, "role": tr.role if tr else None,
                "unit": ev.constraint_units[k], "n_defined": int(np.sum(~und[:, k]))}
        base.update({f"in_{g}": cid in s for g, s in groups.items()})
        base.update(rate_block(int(v_k.sum()), int(u_k.sum()), S, conf))
        base.update(_deficit_stats(ev.margin[:, k], v_k))
        if cid in TABLE51_ROWS:
            # FIX3_BC (C-1): where the row's verdicts lie relative to the primary Table 5-1 domain
            base.update({"n_violated_in_primary_domain": int((v_k & in_primary).sum()),
                         "n_violated_outside_primary_domain": int((v_k & ~in_primary).sum()),
                         "n_states_outside_primary_domain": int((~in_primary).sum())})
        rows.append(base)
    tr = trow.get(e_id)
    rv = np.asarray(en.reference_violated, bool)
    ru = ~np.asarray(en.reference_defined, bool)
    er = {"constraint_id": e_id, "verdict": "reference_chain_ch3", "class_in_reference_problem":
          "probabilistic_nutrition" if e_id in ids and ev.constraint_classes[ids.index(e_id)] ==
          ConstraintClass.PROBABILISTIC_NUTRITION.value else "reference_check",
          "status": tr.status if tr else None, "role": tr.role if tr else None, "unit": "Mcal/d",
          "n_defined": int(np.sum(~ru))}
    er.update({f"in_{g}": e_id in s for g, s in groups.items()})
    er.update(rate_block(int(rv.sum()), int((ru & ~rv).sum()), S, conf))
    er.update(_deficit_stats(en.reference_margin_mcal_d, rv))
    rows.append(er)

    # ---- Table 5-1 domain (all declared variants; nothing removed from any denominator)
    dom_sum: dict[str, Any] = {}
    t_rows = [r for r in TABLE51_ROWS if r in ids]
    tv, tu = DOM.rows_verdict(ev, t_rows) if t_rows else (np.zeros(S, bool), np.zeros(S, bool))
    energy_status_by_domain: dict[str, Any] = {}
    main_def = next((e for e in spec.events if e.event_id == "main_reference"), None)
    for v in spec.domain_variants:
        r = dom[v.variant_id]
        s = r.summary()
        s["table51_rows_crosstab"] = r.crosstab(tv, tu)
        s["event_crosstab"] = {eid: r.crosstab(ev_v[eid], ev_u[eid])
                               for eid in ("main_reference", MAIN_EVENT_PLAN_DOMAIN, MAIN_EVENT_T51_IGNORED,
                                           "h0_eleven", "s2_six")
                               if eid in ev_v}
        if main_def is not None:
            # the main event conditioned on THIS variant's domain (the threshold is a research assumption; C-1)
            mv_, mu_ = _joint(main_def, np.asarray(r.status == DOM.DOMAIN_IN, dtype=bool))
            s["main_reference_conditioned_on_this_variant"] = rate_block(int(mv_.sum()), int(mu_.sum()), S, conf)
        pv = _plan(v)
        if pv is not None:
            # the plan-level status of this variant (label only; = the former per-ration "nominal_status")
            s["nominal_status"] = pv.status
            if main_def is not None:
                # reference problem v2: the main rows read at the plan level of THIS variant
                pmv, pmu = _joint(main_def, pv.state_mask(S))
                s["main_reference_plan_domain_on_this_variant"] = rate_block(int(pmv.sum()), int(pmu.sum()), S, conf)
        dom_sum[v.variant_id] = s
        if v.role == "primary":
            energy_status_by_domain = {st: en.class_counts(r.status == st) for st in DOM.DOMAIN_STATUSES}

    # ---- composition classes of the used feeds (labels only)
    an = np.asarray(en.analysis_anomaly, bool)
    sv = np.asarray(en.support_violation, bool)
    mv = ev_v.get("main_reference", np.zeros(S, dtype=bool))
    mu = ev_u.get("main_reference", np.zeros(S, dtype=bool))

    def _in(mask: np.ndarray) -> dict[str, int]:
        return {"n_states": int(mask.sum()), "main_reference_violated": int((mv & mask).sum()),
                "main_reference_unknown": int((mu & mask).sum())}

    composition = {"analysis_anomaly_class": "analysis_overlap_or_measurement_anomaly",
                   "n_analysis_anomaly_states": int(an.sum()), "n_support_violation_states": int(sv.sum()),
                   "main_reference_in_analysis_anomaly_states": _in(an),
                   "main_reference_in_support_violation_states": _in(sv),
                   "main_reference_in_other_states": _in(~an & ~sv),
                   "rule": "labels of the used feeds only; no state is clipped, normalised or dropped"}

    # ---- structural planning rows (decision-time d_hat), cost
    smarg = np.atleast_1d(np.asarray(ev.structural_margin, dtype=float))
    sviol = np.atleast_1d(np.asarray(ev.structural_violated, dtype=bool))
    structural = {"structural_ok": bool(ev.structural_ok), "nonnegativity_ok": bool(ev.nonnegativity_ok),
                  "rows": {sid: {"violated": bool(sviol[i]), "planned_margin": _f(smarg[i])}
                           for i, sid in enumerate(ev.structural_ids)},
                  "basis": "decision-time d_hat (planning rows; not random events)"}

    identity = {"schema": REFERENCE_SCHEMA, "spec_fingerprint": spec.fingerprint(), "q_hash": ev.q_hash,
                "stream_id": getattr(draws, "stream_id", None),
                "model_fingerprint": getattr(draws, "model_fingerprint", None),
                "draws_fingerprint": getattr(draws, "fingerprint", None), "n_states": S,
                "is_synthetic": bool(ev.is_synthetic), "reference_problem_id": spec.reference_problem_id,
                "energy_reference_chain": REFERENCE_CHAIN_ID, "label": MODEL_DIFFERENCE_LABEL + " (energy verdicts)",
                "claim_scope": "model-constraint reliability under the declared distribution and model; no animal "
                               "outcome, no field risk"}
    esum = en.summary()
    esum["class_counts_by_primary_domain_status"] = energy_status_by_domain
    prim_sum = dom_sum[primary_variant.variant_id]
    headline: dict[str, Any] = {
        "primary_domain_variant": primary_variant.variant_id,
        "t51_primary_not_assessable_share": prim_sum["shares"][DOM.DOMAIN_OUT],
        "t51_primary_undefined_share": prim_sum["shares"][DOM.DOMAIN_UNDEFINED],
        "t51_primary_in_domain_share": prim_sum["shares"][DOM.DOMAIN_IN],
        "note": ("main_reference judges PN-T1..T5 only inside the primary Table 5-1 domain; outside it they are "
                 "unknown and enter rate_upper unless another member is violated; the out-of-domain share is reported "
                 "next to every main rate (FIX3_BC, C-1)")}
    if "main_reference" in events:
        m = events["main_reference"]
        headline.update({"main_reference_rate_lower": m["rate_lower"], "main_reference_rate_upper": m["rate_upper"],
                         "main_reference_cp_upper": m["cp_upper_one_sided"]})
        if main_def is not None and any(x in TABLE51_ROWS for x in main_def.members):
            _, mu0 = _joint(main_def, None)
            # unknown in the main event but not in the R3C reading (T verdicts used): unknown because of the domain rule
            headline["main_reference_unknown_only_because_out_of_domain"] = int(
                (ev_u["main_reference"] & ~in_primary & ~mu0).sum())
    if MAIN_EVENT_T51_IGNORED in events:
        headline["main_reference_t51_verdicts_used_out_of_domain_rate_upper"] = events[MAIN_EVENT_T51_IGNORED][
            "rate_upper"]
    # reference problem v2: the plan-level reading next to the per-state one (decision F1 pending; both reported)
    headline["t51_plan_domain_status"] = plan_primary.status if plan_primary is not None else None
    headline["main_event_domain_reading"] = (
        "to be confirmed at the protocol freeze (decision F1): main_reference reads the Table 5-1 domain per state, "
        "main_reference_plan_domain at the plan level (ration as formulated at d_hat and nominal composition); both "
        "are reported, neither replaces the other")
    plan_def = next((e for e in spec.events if e.event_id == MAIN_EVENT_PLAN_DOMAIN), None)
    if plan_def is not None:
        m = events[MAIN_EVENT_PLAN_DOMAIN]
        headline.update({"main_reference_plan_domain_rate_lower": m["rate_lower"],
                         "main_reference_plan_domain_rate_upper": m["rate_upper"],
                         "main_reference_plan_domain_cp_upper": m["cp_upper_one_sided"]})
        if any(x in TABLE51_ROWS for x in plan_def.members):
            _, mu0p = _joint(plan_def, None)
            # unknown in the plan-level event but not with T verdicts used: unknown because the PLAN is out of domain
            headline["main_reference_plan_domain_unknown_only_because_out_of_domain"] = int(
                (ev_u[MAIN_EVENT_PLAN_DOMAIN] & ~in_plan & ~mu0p).sum())
    summary = {"identity": identity, "cost": _f(ev.cost) if ev.cost is not None else None, "headline": headline,
               "events": events, "per_constraint": rows, "domain": dom_sum,
               "plan_domain": plan_primary.to_record() if plan_primary is not None else None,
               "energy_reference": esum, "composition": composition, "structural": structural}
    return ReferenceEvaluation(spec_fingerprint=identity["spec_fingerprint"], identity=identity,
                               event_violation=ev_v, event_unknown=ev_u, events=events, evaluation=ev, energy=en,
                               domain=dom, plan_domain=plan_primary, _summary=summary, _rows=tuple(rows))
