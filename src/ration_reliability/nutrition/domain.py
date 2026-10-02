"""NASEM (2021) Table 5-1 applicability domain, per drawn state and per ration.

Review round 3 (F3 §5.2; next-round instruction D5-D6).  Status (contract section 2.3): ``implemented``,
``unit_passed`` (``tests/unit/test_domain_applicability.py``; synthetic hand-computed cases only); evidence
level ``code_tested``.  The operational threshold is ``assumption_only`` (a project research assumption).

What the source says (paraphrased; printed page p.63, PDF page = printed + 20)
-----------------------------------------------------------------------------
Table 5-1 (minimum forage and total NDF, maximum starch for lactating cows) is stated for diets **fed as a
TMR**, with **forage of adequate particle size**, and with **dry ground corn as the predominant starch
source**; the text on p.63 repeats these caveats for the minimum total NDF and explains (p.63-64) that
diets with rapidly fermentable starch sources (high-moisture corn, wheat, barley) need more forage NDF,
and that feeding method changes the requirement.  The source gives **no number** for "predominant".

How the project operationalises the premises (research assumptions, not the source)
----------------------------------------------------------------------------------
* TMR: a property of the feeding system, not of composition.  In ``dev_case_v1`` it is a declared profile
  setting (``research_scenario_assumption``); here it is ``assumption_only`` for every state.
* Forage particle size: not modelled (no particle-size or peNDF variable exists in the engine); it cannot
  be checked from chemical composition.  ``assumption_only`` for every state.
* "Dry ground corn is the predominant starch source": the share of diet starch supplied by the counted
  ingredients, ``share_s = sum_{i in counted} x_si St_si / sum_i x_si St_si`` (realised DM ``x = q d``,
  drawn starch), must be ``>= threshold``.  Evaluated in the linear form of the engine's
  ``DIAG-T51-DGC-STARCH-SHARE`` row: ``margin_s = sum_counted x St - threshold * sum_all x St`` (kg/d),
  met iff ``margin_s >= -tolerance_kg_d``.  **The threshold is a project research assumption**
  (``threshold_status = research_assumption``): primary ``share >= 0.50`` -- *at least half* of the diet
  starch comes from dry ground corn; a share of exactly one half counts as met, so this is "at least half",
  not a strict majority.  Declared sensitivity thresholds: 1/3 (a weaker reading: dry ground corn supplies
  at least one third of the diet starch; this is **not** the definition of a plurality, which would depend
  on the shares of the other sources) and 2/3 (a strong-majority reading).  A second sensitivity reading
  also counts corn silage kernel starch (the project's earlier justification: Table 3-1 gives 32-37 % DM
  corn silage and dry medium-ground corn the same base starch digestibility, p.25); it is weaker than the
  text, which names dry ground corn.
* Timing (disclosure; review round 3 red team, D5): the primary 0.50 is the operationalisation of the
  diagnostic row ``DIAG-T51-DGC-STARCH-SHARE`` that already existed in the round-2 build; the sensitivity
  variants (1/3, 2/3, corn-kernel reading) were added in round 3 **after** the round-2 development result
  (96-100 % of the S2 states outside the primary domain) had been seen.  ``configs/dev_case_v1/
  reference_constraints.csv`` discloses this (``decided_after_seeing_dev_results = yes``).  The variants are
  reported side by side; none is chosen, or dropped, because it makes a result look better, and the diet
  range / reading must be decided before any full run (R3C open item 3).

Per-state status
----------------
* ``in_domain_conditional``: the starch-source criterion is met; the verdicts of Table 5-1 rows may be read
  against the source, **conditional on** the two ``assumption_only`` premises (TMR, particle size);
* ``not_assessable``: the criterion is not met; Table 5-1 rows cannot judge this state and must not be
  extrapolated to it.  The state stays in every denominator; it is not called safe (a satisfied row is not
  "within Table 5-1") and not called harmful (an outside-domain state is not a disease prediction);
* ``undefined``: no starch is supplied, or a used ingredient has a missing DM or starch cell.

Nothing is removed from any denominator; :meth:`Table51DomainResult.crosstab` splits any row verdict by status.

Reading an event that contains Table 5-1 rows (review round 3 red team, D6/C; adjudication rule A7)
------------------------------------------------------------------------------------------------
Keeping ``not_assessable`` states in the denominator is not enough if their Table 5-1 row verdicts are then read
as ordinary verdicts: a state outside the premise whose T rows happen to be satisfied would count as "not
violated" in the event rate, i.e. as judged safe by a table that cannot judge it.  :func:`premise_conditioned_event`
therefore gives, for any event (union of rows) and one declared variant, the pair

* ``premise_conditioned`` (the reading to report as the event rate): a state is **violated** when any member row is
  violated -- including a violated T row outside the domain (a violated stated limit is kept as a violation, the
  conservative direction); a state with no violated member is **unknown** when a member is undefined *or* the event
  contains a Table 5-1 row and the state is not ``in_domain_conditional`` (its "not violated" verdict rests on a
  T row that cannot be read there).  Rates ``[n_violated / S, (n_violated + n_unknown) / S]``, denominator = all S;
* ``t_rows_read_regardless_of_premise``: the same event with every T row read whatever the domain status -- the
  old reading, kept only as an annotated variant (never a headline number).

The lower rate is the same under both readings; only the upper rate moves.  In-domain counts (``in_domain_readable``)
are given as counts; no conditional rate on the in-domain states is formed (the in-domain subset is selected by the
ration's own composition).  Which variant is primary, and which diet range the case uses, must be fixed before any
full run -- never chosen from these numbers.

Relation to the reference evaluator (BLOCKERS B-442)
----------------------------------------------------
``evaluation/reference.py`` is the **primary rule** for every reported event rate.  Its domain-conditioned events do
not use a Table 5-1 verdict outside the domain at all (neither pass nor fail): an out-of-domain state is violated only
through a non-T member and is otherwise unknown.  :func:`premise_conditioned_event` keeps an out-of-domain T violation
as violated.  For the same event, states and domain the two therefore have **equal upper rates**, the lower rate of
this module is **>=** the reference evaluator's, and the difference is exactly the number of out-of-domain states whose
only violated members are Table 5-1 rows (``tests/unit/test_domain_rule_cross_consistency.py``).

Reference problem v2 (D-533; ``docs/reference_problem_v2.md``): premise planning row and plan-level status
-------------------------------------------------------------------------------------------------------
* :func:`premise_planning_coefficients` gives the per-ingredient coefficients ``(1{i in counted} - tau) * St_i`` at the
  nominal (table) composition; ``sum_i q_i d_hat_i coef_i >= 0`` is the linear form of "the counted share of the
  *planned* diet starch is at least tau".  Reference problem v2 imposes it as the structural planning row
  :data:`PREMISE_PLANNING_ROW_ID` in every objective arm and method (decision package section 3.1 option B); at
  tau = 0.50 its coefficients equal the nominal content of ``DIAG-T51-DGC-STARCH-SHARE``.
* :func:`plan_domain_status` is the plan-level domain status of a ration: the same share definition evaluated once on
  the planned diet (``q``, decision-time ``d_hat``, nominal composition) -- a property of the ration as formulated, not
  of a drawn state.  Whether the main event reads the domain per state or at the plan is decision F1 of
  ``docs/official_run_v2_plan_20260926.md`` (to be confirmed at the protocol freeze); both readings are computed.
* The share, margin and coefficients are functions of restricted table values: they stay in memory (holder side) and
  are never written to public output; only the status label is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..hashing import stable_hash

__all__ = [
    "DOMAIN_IN",
    "DOMAIN_OUT",
    "DOMAIN_UNDEFINED",
    "DOMAIN_STATUSES",
    "PREMISE_ASSUMPTION_ONLY",
    "PRIMARY_THRESHOLD",
    "SENSITIVITY_THRESHOLDS",
    "THRESHOLD_STATUS",
    "THRESHOLD_RATIONALE",
    "TABLE51_SOURCE",
    "Table51DomainSpec",
    "Table51DomainResult",
    "default_table51_variants",
    "table5_1_domain",
    "table5_1_domain_arrays",
    "rows_verdict",
    "PREMISE_CONDITIONED",
    "REGARDLESS_OF_PREMISE",
    "PREMISE_CONDITIONED_RULE",
    "PremiseConditionedEvent",
    "premise_conditioned_event",
    "PREMISE_PLANNING_ROW_ID",
    "PLAN_LEVEL_BASIS",
    "premise_planning_coefficients",
    "PlanDomainStatus",
    "plan_domain_status",
]

DOMAIN_IN = "in_domain_conditional"
DOMAIN_OUT = "not_assessable"
DOMAIN_UNDEFINED = "undefined"
DOMAIN_STATUSES = (DOMAIN_IN, DOMAIN_OUT, DOMAIN_UNDEFINED)
#: Status of a premise the engine cannot observe (TMR feeding, forage particle size).
PREMISE_ASSUMPTION_ONLY = "assumption_only"
#: Primary operationalisation of "predominant" (research assumption; the source gives no number).
PRIMARY_THRESHOLD = 0.5
#: Declared sensitivity range of the threshold (research assumption).
SENSITIVITY_THRESHOLDS = (1.0 / 3.0, 0.5, 2.0 / 3.0)
THRESHOLD_STATUS = "research_assumption"
THRESHOLD_RATIONALE = (
    "NASEM 2021 Table 5-1 title and p.63 name dry ground corn as the predominant starch source without a number; "
    "the project reads 'predominant' as dry ground corn supplying at least half of diet starch (share >= 0.50; "
    "exactly one half counts as met) and reports 1/3 (weaker reading: at least one third; not a plurality "
    "definition) and 2/3 (strong-majority reading) as declared sensitivity thresholds, added in round 3 after the "
    "round-2 result had been seen (disclosed); project research assumption, not attributed to the source")
#: Label of the event reading that keeps a Table 5-1 row verdict only where the premise holds (report this one).
PREMISE_CONDITIONED = "premise_conditioned"
#: Label of the old reading (T rows read whatever the domain status): annotated variant, never a headline number.
REGARDLESS_OF_PREMISE = "t_rows_read_regardless_of_premise"
PREMISE_CONDITIONED_RULE = (
    "event containing Table 5-1 rows: violated if any member row is violated (a violated T row outside the domain "
    "still counts as violated); with no violated member, unknown if a member is undefined or the state is not "
    "in_domain_conditional (its 'not violated' rests on a T row the table cannot judge there); denominator = all "
    "states; rates [n_violated/S, (n_violated + n_unknown)/S]; events without a Table 5-1 row are unchanged")
TABLE51_SOURCE = "NASEM (2021) Table 5-1 title and text, printed p.63-64 (PDF page = printed + 20)"
#: Structural planning row of reference problem v2 that imposes the starch-source premise on the planned diet
#: (D-533; docs/reference_problem_v2.md).  Built by experiments/E1_cost_reliability/official_v2.py.
PREMISE_PLANNING_ROW_ID = "SH-PLAN-T51-DGC-SHARE"
#: Basis of the plan-level domain status (one status per ration, not per state).
PLAN_LEVEL_BASIS = ("planned diet: fixed q, decision-time d_hat (never a drawn DM) and the nominal (table-value) "
                    "composition; one status per ration")
_READINGS = ("dry_ground_corn_only", "corn_kernel_incl_corn_silage")
_ROLES = ("primary", "sensitivity")


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class Table51DomainSpec:
    """One operationalisation of the Table 5-1 premises (see the module docstring).

    ``counted_ingredient_ids``: ingredients whose starch counts as the named source (primary reading: the
    dry ground corn ingredients only).  ``threshold`` in (0, 1).  ``tolerance_kg_d``: numerical tolerance of
    the linear-form margin (kg starch/d; the ``DIAG-T51`` row uses 1e-6).  ``threshold_status`` must be
    ``research_assumption`` and both unobservable premises ``assumption_only`` -- the spec refuses to call
    either sourced or met.
    """

    variant_id: str
    counted_ingredient_ids: tuple[str, ...]
    threshold: float
    counted_reading: str = "dry_ground_corn_only"
    role: str = "primary"
    starch_column: str = "starch"
    tolerance_kg_d: float = 1e-6
    tmr_premise: str = PREMISE_ASSUMPTION_ONLY
    particle_size_premise: str = PREMISE_ASSUMPTION_ONLY
    threshold_status: str = THRESHOLD_STATUS
    rationale: str = THRESHOLD_RATIONALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "counted_ingredient_ids", tuple(str(i) for i in self.counted_ingredient_ids))
        errs = []
        if not self.variant_id:
            errs.append("variant_id is required")
        if not self.counted_ingredient_ids or len(set(self.counted_ingredient_ids)) != len(self.counted_ingredient_ids):
            errs.append("counted_ingredient_ids must be non-empty and unique")
        t = float(self.threshold)
        if not (math.isfinite(t) and 0.0 < t < 1.0):
            errs.append("threshold must be in (0, 1)")
        object.__setattr__(self, "threshold", t)
        if self.counted_reading not in _READINGS:
            errs.append(f"counted_reading must be one of {_READINGS}")
        if self.role not in _ROLES:
            errs.append(f"role must be one of {_ROLES}")
        if not (math.isfinite(float(self.tolerance_kg_d)) and float(self.tolerance_kg_d) >= 0):
            errs.append("tolerance_kg_d must be >= 0")
        if self.threshold_status != THRESHOLD_STATUS:
            errs.append("threshold_status must be 'research_assumption' (the source gives no number)")
        for f in ("tmr_premise", "particle_size_premise"):
            if getattr(self, f) != PREMISE_ASSUMPTION_ONLY:
                errs.append(f"{f} must be 'assumption_only': the engine cannot observe it")
        if self.role == "primary" and self.counted_reading != "dry_ground_corn_only":
            errs.append("the primary variant counts dry ground corn only (the named source)")
        if errs:
            raise ValueError(f"Table51DomainSpec {self.variant_id!r}: " + "; ".join(errs))

    def fingerprint(self) -> str:
        return stable_hash("Table51DomainSpec/v1", self.variant_id, self.counted_ingredient_ids, self.threshold,
                           self.counted_reading, self.role, self.starch_column, float(self.tolerance_kg_d),
                           self.tmr_premise, self.particle_size_premise, self.threshold_status)

    def to_record(self) -> dict[str, Any]:
        return {"variant_id": self.variant_id, "role": self.role, "counted_reading": self.counted_reading,
                "counted_ingredient_ids": list(self.counted_ingredient_ids), "threshold": self.threshold,
                "threshold_status": self.threshold_status, "tolerance_kg_d": float(self.tolerance_kg_d),
                "tmr_premise": self.tmr_premise, "particle_size_premise": self.particle_size_premise,
                "rationale": self.rationale, "source": TABLE51_SOURCE, "fingerprint": self.fingerprint()}


def default_table51_variants(dgc_ingredient_ids: Sequence[str], *,
                             corn_silage_ingredient_ids: Sequence[str] = ()) -> tuple[Table51DomainSpec, ...]:
    """The declared variants: primary (dry ground corn, 0.50), sensitivity 1/3 and 2/3, and -- if corn
    silage ids are given -- the weaker corn-kernel reading at 0.50 (sensitivity only)."""
    dgc = tuple(dgc_ingredient_ids)
    out = [Table51DomainSpec("T51-DGC-0.50", dgc, PRIMARY_THRESHOLD, role="primary"),
           Table51DomainSpec("T51-DGC-0.33", dgc, SENSITIVITY_THRESHOLDS[0], role="sensitivity"),
           Table51DomainSpec("T51-DGC-0.67", dgc, SENSITIVITY_THRESHOLDS[2], role="sensitivity")]
    if corn_silage_ingredient_ids:
        out.append(Table51DomainSpec("T51-CORNKERNEL-0.50", dgc + tuple(corn_silage_ingredient_ids),
                                     PRIMARY_THRESHOLD, counted_reading="corn_kernel_incl_corn_silage",
                                     role="sensitivity"))
    return tuple(out)


@dataclass(frozen=True)
class Table51DomainResult:
    """Per-state Table 5-1 domain status of one executed ration (arrays ``[S]``, read-only)."""

    variant_id: str
    spec_fingerprint: str
    threshold: float
    role: str
    counted_reading: str
    stream_id: Optional[str]
    status: np.ndarray
    starch_margin_kg_d: np.ndarray
    counted_starch_share: np.ndarray
    tmr_premise: str = PREMISE_ASSUMPTION_ONLY
    particle_size_premise: str = PREMISE_ASSUMPTION_ONLY

    @property
    def n_states(self) -> int:
        return int(self.status.shape[0])

    def counts(self) -> dict[str, int]:
        return {s: int(np.sum(self.status == s)) for s in DOMAIN_STATUSES}

    def summary(self) -> dict[str, Any]:
        """Status counts and shares; the denominator is every state (nothing removed)."""
        S = self.n_states
        c = self.counts()
        return {"variant_id": self.variant_id, "role": self.role, "threshold": self.threshold,
                "threshold_status": THRESHOLD_STATUS, "counted_reading": self.counted_reading,
                "tmr_premise": self.tmr_premise, "particle_size_premise": self.particle_size_premise,
                "stream_id": self.stream_id, "n_states": S, "counts": c,
                "shares": {k: (v / S if S else None) for k, v in c.items()},
                "meaning": {DOMAIN_IN: "starch-source criterion met; Table 5-1 readable conditional on TMR and "
                                       "particle size (assumption_only)",
                            DOMAIN_OUT: "outside the premise: Table 5-1 cannot judge or be extrapolated to this "
                                        "state; kept in the denominator; not safe, not harmful",
                            DOMAIN_UNDEFINED: "no starch supplied or a missing cell of a used ingredient"}}

    def crosstab(self, violated: np.ndarray, undefined: Optional[np.ndarray] = None) -> dict[str, dict[str, int]]:
        """Counts of a row verdict (``violated [S]``, optional ``undefined [S]``) within each domain status."""
        v = np.asarray(violated, dtype=bool)
        u = np.zeros(self.n_states, dtype=bool) if undefined is None else np.asarray(undefined, dtype=bool)
        if v.shape != (self.n_states,) or u.shape != (self.n_states,):
            raise ValueError("crosstab: verdict arrays must have shape [S]")
        out = {}
        for s in DOMAIN_STATUSES:
            m = self.status == s
            out[s] = {"violated": int(np.sum(m & v)), "not_violated": int(np.sum(m & ~v & ~u)),
                      "undefined": int(np.sum(m & ~v & u))}
        return out

    def premise_conditioned(self, event_violated: np.ndarray, event_unknown: np.ndarray, *,
                            contains_table51_rows: bool) -> "PremiseConditionedEvent":
        """:func:`premise_conditioned_event` for this variant."""
        return premise_conditioned_event(event_violated, event_unknown, self,
                                         contains_table51_rows=contains_table51_rows)


@dataclass(frozen=True)
class PremiseConditionedEvent:
    """One event (union of rows) read under one Table 5-1 variant (arrays ``[S]``, read-only; module docstring).

    ``violated`` is the event's violation as evaluated (the same under both readings); ``unknown`` is the
    premise-conditioned unknown; ``unknown_regardless_of_premise`` the event's own unknown (T rows read whatever the
    domain status); ``premise_unknown`` the states added by the premise (no violated member, every member defined,
    state not ``in_domain_conditional``, event contains a Table 5-1 row).
    """

    variant_id: str
    role: str
    threshold: float
    contains_table51_rows: bool
    status: np.ndarray
    violated: np.ndarray
    unknown: np.ndarray
    unknown_regardless_of_premise: np.ndarray
    premise_unknown: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.violated.shape[0])

    def summary(self) -> dict[str, Any]:
        """Counts and rate pairs of both readings; denominators are every state (nothing removed)."""
        S = self.n_states
        nv = int(self.violated.sum())
        nu_pc = int(self.unknown.sum())
        nu_rg = int(self.unknown_regardless_of_premise.sum())

        def pair(nu: int) -> dict[str, Any]:
            return {"n_violated": nv, "n_unknown": nu, "rate_lower": (nv / S) if S else None,
                    "rate_upper": ((nv + nu) / S) if S else None}

        ind = self.status == DOMAIN_IN
        by_status = {}
        for st in DOMAIN_STATUSES:
            m = self.status == st
            by_status[st] = {"n_states": int(m.sum()), "violated": int((m & self.violated).sum()),
                             "unknown_as_evaluated": int((m & self.unknown_regardless_of_premise).sum()),
                             "not_violated_as_evaluated": int((m & ~self.violated &
                                                               ~self.unknown_regardless_of_premise).sum())}
        return {"variant_id": self.variant_id, "role": self.role, "threshold": self.threshold,
                "threshold_status": THRESHOLD_STATUS, "contains_table51_rows": self.contains_table51_rows,
                "n_states": S, "rule": PREMISE_CONDITIONED_RULE,
                PREMISE_CONDITIONED: {**pair(nu_pc), "n_unknown_premise_not_met": int(self.premise_unknown.sum()),
                                      "report_as": "event rate (denominator = all states)"},
                REGARDLESS_OF_PREMISE: {**pair(nu_rg),
                                        "report_as": "annotated variant only: Table 5-1 rows read whatever the "
                                                     "domain status; not a headline number"},
                "in_domain_readable": {"n_states_in_domain": int(ind.sum()),
                                       "n_violated": int((ind & self.violated).sum()),
                                       "n_unknown": int((ind & self.unknown).sum()),
                                       "n_not_violated": int((ind & ~self.violated & ~self.unknown).sum()),
                                       "note": "counts only; no rate conditional on the in-domain subset is formed"},
                "by_domain_status": by_status}


def premise_conditioned_event(event_violated: np.ndarray, event_unknown: np.ndarray, domain: "Table51DomainResult",
                              *, contains_table51_rows: bool) -> PremiseConditionedEvent:
    """Read an event's verdicts under the Table 5-1 premise of ``domain`` (see the module docstring).

    ``event_violated [S]``: any member row violated; ``event_unknown [S]``: no member violated and some member
    undefined (the evaluator's convention).  ``contains_table51_rows``: whether any member is a Table 5-1 row (PN-T1
    ... PN-T5); if not, the premise does not enter and both readings coincide.  Nothing is removed; ``q`` and the
    draws are not touched (the arrays come from an evaluation of a fixed ``q``).

    B-442 (cross-consistency with ``evaluation/reference.py``, which is the primary rule for reported rates): this
    function **keeps an out-of-domain Table 5-1 violation as violated** (the conservative direction), whereas the
    reference evaluator's domain-conditioned events (``table51_domain`` ``"primary"`` or ``"plan"``) use no T verdict
    outside the domain and record such a state as unknown unless a non-T member is violated.  Given the same event
    arrays (the event read with T verdicts in every state) and the same domain, therefore:

    * the **upper** rates are equal -- every out-of-domain state is violated or unknown under both rules;
    * this function's **lower** rate is **>=** the reference evaluator's; the difference equals the number of
      out-of-domain states whose only violated members are Table 5-1 rows.

    For the plan-level reading pass :meth:`PlanDomainStatus.as_domain_result` as ``domain``.  Checked on hand-built and
    randomised synthetic states in ``tests/unit/test_domain_rule_cross_consistency.py``.
    """
    v = np.asarray(event_violated, dtype=bool)
    u = np.asarray(event_unknown, dtype=bool)
    S = domain.n_states
    if v.shape != (S,) or u.shape != (S,):
        raise ValueError("premise_conditioned_event: event arrays must have shape [S] of the domain result")
    if np.any(v & u):
        raise ValueError("premise_conditioned_event: a state cannot be both violated and unknown")
    not_in = domain.status != DOMAIN_IN
    add = (~v & ~u & not_in) if contains_table51_rows else np.zeros(S, dtype=bool)
    return PremiseConditionedEvent(domain.variant_id, domain.role, domain.threshold, bool(contains_table51_rows),
                                   domain.status, _ro(v.copy()), _ro(u | add), _ro(u.copy()), _ro(add))


def table5_1_domain_arrays(q: Any, theta: np.ndarray, d: np.ndarray, ingredient_ids: Sequence[str],
                           nutrient_ids: Sequence[str], spec: Table51DomainSpec, *,
                           stream_id: Optional[str] = None) -> Table51DomainResult:
    """Per-state status for explicit arrays: ``theta [S, I, J]`` or ``[I, J]`` (canonical fractions, DM basis),
    ``d [S, I]`` or ``[I]`` (DM fractions; for the per-ration nominal status pass the decision-time ``d_hat``
    and the nominal composition).  ``q`` is fixed: ``x = q * d``, never renormalised."""
    ids = [str(i) for i in ingredient_ids]
    nut = list(nutrient_ids)
    th = np.asarray(theta, dtype=float)
    dd = np.asarray(d, dtype=float)
    if th.ndim == 2:
        th = th[None]
    if dd.ndim == 1:
        dd = dd[None]
    if th.ndim != 3 or th.shape[1:] != (len(ids), len(nut)) or dd.shape != (th.shape[0], len(ids)):
        raise ValueError("table5_1_domain: theta [S,I,J] and d [S,I] must match the labels")
    missing = [i for i in spec.counted_ingredient_ids if i not in ids]
    if missing:
        raise KeyError(f"table5_1_domain: counted ingredients {missing} are not in the ration")
    if spec.starch_column not in nut:
        raise KeyError(f"table5_1_domain: starch column {spec.starch_column!r} is not in the draws")
    qv = np.array(getattr(q, "q_as_fed", q), dtype=float)
    if qv.shape != (len(ids),) or not np.all(np.isfinite(qv)) or np.any(qv < 0):
        raise ValueError("table5_1_domain: q must be finite, >= 0, with one entry per ingredient")
    used = qv != 0.0
    st = th[:, :, nut.index(spec.starch_column)]
    x = dd * qv[None, :]
    miss = ~(np.all(np.isfinite(x[:, used]), axis=1) & np.all(np.isfinite(st[:, used]), axis=1))
    mass = np.where(used[None, :], np.where(np.isfinite(x * st), x * st, 0.0), 0.0)   # kg starch / d per feed
    counted = np.array([i in spec.counted_ingredient_ids for i in ids], dtype=bool)
    s_tot = mass.sum(axis=1)
    s_cnt = mass[:, counted].sum(axis=1)
    margin = s_cnt - spec.threshold * s_tot
    undefined = miss | ~(s_tot > 0.0)
    met = ~undefined & (margin >= -float(spec.tolerance_kg_d))
    status = np.where(undefined, DOMAIN_UNDEFINED, np.where(met, DOMAIN_IN, DOMAIN_OUT)).astype("<U24")
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(undefined, np.nan, s_cnt / np.where(s_tot > 0, s_tot, np.nan))
    return Table51DomainResult(spec.variant_id, spec.fingerprint(), spec.threshold, spec.role, spec.counted_reading,
                               stream_id, _ro(status), _ro(np.where(undefined, np.nan, margin)), _ro(share),
                               spec.tmr_premise, spec.particle_size_premise)


def table5_1_domain(q: Any, draws: Any, spec: Table51DomainSpec) -> Table51DomainResult:
    """Per-state status of ``q`` on a :class:`~ration_reliability.uncertainty.base.DrawSet`."""
    return table5_1_domain_arrays(q, draws.theta, draws.d, draws.ingredient_ids, draws.nutrient_ids, spec,
                                  stream_id=getattr(draws, "stream_id", None))


def rows_verdict(eval_result: Any, row_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(any_violated [S], any_undefined_without_violation [S])`` of the evaluator rows ``row_ids``
    (e.g. the Table 5-1 rows of a problem), for :meth:`Table51DomainResult.crosstab`."""
    ids = list(eval_result.constraint_ids)
    missing = [r for r in row_ids if r not in ids]
    if missing:
        raise KeyError(f"rows_verdict: rows {missing} are not in the evaluator result")
    k = [ids.index(r) for r in row_ids]
    v = np.asarray(eval_result.violated, dtype=bool)[:, k].any(axis=1)
    u = np.asarray(eval_result.undefined, dtype=bool)[:, k].any(axis=1) & ~v
    return v, u


# =====================================================================================================
# reference problem v2 (D-533): the premise as a planning row, and the plan-level domain status
# =====================================================================================================

def premise_planning_coefficients(nominal_theta: np.ndarray, ingredient_ids: Sequence[str],
                                  nutrient_ids: Sequence[str], counted_ids: Sequence[str], tau: float, *,
                                  starch_column: str = "starch") -> np.ndarray:
    """Per-ingredient coefficients ``coef_i = (1{i in counted_ids} - tau) * St_i`` at the nominal composition.

    ``nominal_theta [I, J]``: nominal (table-value) composition, canonical fractions of DM; ``St_i`` is its
    ``starch_column``.  With the planned DM ``x_hat_i = q_i d_hat_i`` (decision-time ``d_hat``; never a drawn DM)::

        sum_i q_i d_hat_i coef_i = sum_{i in counted} x_hat_i St_i - tau * sum_i x_hat_i St_i  >= 0

    is the linear form of "the counted ingredients supply at least the share ``tau`` of the planned diet starch" --
    the share definition of :func:`table5_1_domain_arrays`, evaluated on the plan.  At ``tau = 0.5`` with the dry
    ground corn ingredients counted, ``coef`` equals the compiled content of ``DIAG-T51-DGC-STARCH-SHARE`` (weights
    +0.5 / -0.5 times starch) at the nominal state.  If the planned diet contains no starch the row holds (0 >= 0)
    while the share, and so :func:`plan_domain_status`, is ``undefined``.

    Pure function: no I/O.  The returned values derive from table values (restricted in ``dev_case_v1``): callers
    keep them in memory and never write them to public output.  ``tau`` is a project research assumption.
    """
    ids = [str(i) for i in ingredient_ids]
    nut = [str(n) for n in nutrient_ids]
    th = np.asarray(nominal_theta, dtype=float)
    if th.shape != (len(ids), len(nut)):
        raise ValueError("premise_planning_coefficients: nominal_theta must be [I, J] of the labels")
    if len(set(ids)) != len(ids):
        raise ValueError("premise_planning_coefficients: ingredient ids must be unique")
    cnt = tuple(str(i) for i in counted_ids)
    if not cnt or len(set(cnt)) != len(cnt):
        raise ValueError("premise_planning_coefficients: counted_ids must be non-empty and unique")
    missing = [i for i in cnt if i not in ids]
    if missing:
        raise KeyError(f"premise_planning_coefficients: counted ingredients {missing} are not in the problem")
    if starch_column not in nut:
        raise KeyError(f"premise_planning_coefficients: starch column {starch_column!r} is not a nutrient")
    t = float(tau)
    if not (math.isfinite(t) and 0.0 < t < 1.0):
        raise ValueError("premise_planning_coefficients: tau must be in (0, 1)")
    st = th[:, nut.index(starch_column)]
    if not (np.all(np.isfinite(st)) and np.all(st >= 0.0)):
        raise ValueError("premise_planning_coefficients: every nominal starch value must be finite and >= 0 "
                         "(a planning row cannot carry a missing cell)")
    counted = np.array([i in cnt for i in ids], dtype=float)
    return (counted - t) * st


@dataclass(frozen=True)
class PlanDomainStatus:
    """Plan-level Table 5-1 domain status of one ration (reference problem v2; see the module docstring).

    ``status``: one of :data:`DOMAIN_STATUSES`, from the planned diet (``q``, decision-time ``d_hat``, nominal
    composition) with the share definition of :func:`table5_1_domain_arrays`.  ``planned_starch_margin_kg_d`` and
    ``planned_counted_starch_share`` are derived from restricted table values: kept for holder-side checks, left out
    of :meth:`to_record`.
    """

    variant_id: str
    spec_fingerprint: str
    threshold: float
    role: str
    counted_reading: str
    status: str
    planned_starch_margin_kg_d: Optional[float]
    planned_counted_starch_share: Optional[float]
    tolerance_kg_d: float
    tmr_premise: str = PREMISE_ASSUMPTION_ONLY
    particle_size_premise: str = PREMISE_ASSUMPTION_ONLY

    @property
    def in_domain(self) -> bool:
        return self.status == DOMAIN_IN

    def state_mask(self, n_states: int) -> np.ndarray:
        """``[S]`` read-only mask: every state carries the plan's status (``True`` iff the plan is in the domain)."""
        n = int(n_states)
        if n < 1:
            raise ValueError("state_mask: n_states must be >= 1")
        return _ro(np.full(n, self.in_domain, dtype=bool))

    def as_domain_result(self, n_states: int, *, stream_id: Optional[str] = None) -> Table51DomainResult:
        """The plan status broadcast to ``n_states`` states, so that :func:`premise_conditioned_event` and
        :meth:`Table51DomainResult.crosstab` can be applied to the plan-level reading (the per-state share and margin
        columns carry the plan's values)."""
        n = int(n_states)
        if n < 1:
            raise ValueError("as_domain_result: n_states must be >= 1")
        m = np.nan if self.planned_starch_margin_kg_d is None else float(self.planned_starch_margin_kg_d)
        sh = np.nan if self.planned_counted_starch_share is None else float(self.planned_counted_starch_share)
        return Table51DomainResult(self.variant_id, self.spec_fingerprint, self.threshold, self.role,
                                   self.counted_reading, stream_id, _ro(np.full(n, self.status).astype("<U24")),
                                   _ro(np.full(n, m)), _ro(np.full(n, sh)), self.tmr_premise,
                                   self.particle_size_premise)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe labels only (no share, no margin: they derive from restricted table values)."""
        return {"variant_id": self.variant_id, "role": self.role, "threshold": self.threshold,
                "threshold_status": THRESHOLD_STATUS, "counted_reading": self.counted_reading, "status": self.status,
                "basis": PLAN_LEVEL_BASIS, "tmr_premise": self.tmr_premise,
                "particle_size_premise": self.particle_size_premise, "spec_fingerprint": self.spec_fingerprint}


def plan_domain_status(q: Any, nominal_theta: np.ndarray, d_hat: np.ndarray, ingredient_ids: Sequence[str],
                       nutrient_ids: Sequence[str], spec: Table51DomainSpec) -> PlanDomainStatus:
    """Plan-level status of ``q`` (a :class:`RationDecision` or ``[I]`` kg as fed / head / d): the same share and
    tolerance as :func:`table5_1_domain_arrays`, evaluated once with ``nominal_theta [I, J]`` and the decision-time
    ``d_hat [I]``.  A ration that meets the premise planning row with ``tau = spec.threshold`` and the same counted
    ingredients and tolerance is ``in_domain_conditional`` here unless its planned diet contains no starch."""
    th = np.asarray(nominal_theta, dtype=float)
    dh = np.asarray(d_hat, dtype=float)
    if th.ndim != 2 or dh.ndim != 1:
        raise ValueError("plan_domain_status: nominal_theta must be [I, J] and d_hat [I] (one planned diet)")
    r = table5_1_domain_arrays(q, th, dh, ingredient_ids, nutrient_ids, spec)
    status = str(r.status[0])
    margin = float(r.starch_margin_kg_d[0])
    share = float(r.counted_starch_share[0])
    return PlanDomainStatus(spec.variant_id, spec.fingerprint(), spec.threshold, spec.role, spec.counted_reading,
                            status, margin if math.isfinite(margin) else None, share if math.isfinite(share) else None,
                            float(spec.tolerance_kg_d), spec.tmr_premise, spec.particle_size_premise)
