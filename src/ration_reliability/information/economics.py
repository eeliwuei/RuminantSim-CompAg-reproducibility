"""Economic conversion of per-head-per-day values to one assayed batch (contract §12.5, T7.7, F11, F12).

An assay is paid per sample / panel / batch, while the ration saving is per head per day.  The two
are linked only through the verifiable coverage of the assayed batch: ``H`` heads fed from that
batch for ``T`` days (``T`` also bounded by the validity period of the assay result, e.g. the
3/14/30-day scenarios of the feasibility audit §4.5).  With a fixed coverage and inventory::

    gross value per batch  =  H * T * dc          (dc = V_0 - V_T, currency/head/d)

No annualisation (no arbitrary x365), no health, milk or disease values are attached.  When the
assay cost is unknown only the **break-even maximum cost per batch** is reported; a net value is
returned only when every declared cost component is known -- unknown components are listed, never
set to 0.  If the with-information policy changes the amounts fed so that the batch no longer
covers ``H x T`` (inventory ``B_i``), the fixed multiplier is invalid (``inventory_coverage_check``).

Red-team fix F11 (2026-09-24): the validity period of one assay result is a required field of
:class:`BatchCoverage` (``T <= validity``), and every per-batch conversion
(:func:`batch_gross_value`, :func:`break_even_max_cost_per_batch`, :func:`net_value_per_batch`,
:func:`per_head_day_from_batch_cost`) requires an :class:`InventoryCoverageResult` for the same
coverage with ``fixed_multiplier_valid=True``; otherwise it raises (no H x T multiplier without a
verified inventory coverage, contract §12.5, T2.3).

Second review R1 (2026-09-25; ``docs/value_definition.md``): every conversion of a *value*
(:func:`batch_gross_value`, :func:`break_even_max_cost_per_batch`, :func:`net_value_per_batch`) needs
an explicit ``value_definition``; there is no default difference.

* ``operational_deterministic_cost_difference`` -- the executable basis (deterministic policies);
* ``randomized_same_class_information_reference`` -- accepted but labelled a *theoretical* break-even
  (it assumes randomised feeding policies, which are never recommended);
* ``contrast_vs_matched_uninformative_bins`` -- refused: a diagnostic contrast against a hypothetical
  randomisation device is not the value of buying the assay and cannot bound a payment.

:func:`per_head_day_value` extracts ``dc`` from an :class:`~.value.InformationValueResult` with these
checks (and refuses non-nested action spaces, T7.2).  :func:`per_head_day_from_batch_cost` converts a
*cost*, not a value, and needs no definition.

Red-team FIX_A (2026-09-25; ``docs/value_definition.md`` §8.2):

* a value is **bound** to the result it comes from: :func:`per_head_day_value` returns a
  :class:`PerHeadDayValue` (a ``float`` carrying a :class:`ValueBinding`: definition, role, result
  fingerprint, identification label, randomisation-channel assessment, scenario flag); the per-batch
  conversions accept that value or the result itself and copy the binding into their outputs
  (:class:`BatchAmount`, :class:`NetValueResult`).  The definition named by the caller must be the
  bound one (a randomised reference cannot be relabelled operational, a contrast cannot enter).
* a bare ``float`` is refused unless the caller names it with ``unbound_value_declaration=...``; the
  outputs then say ``caller_declared_unbound_float`` and role ``unbound_caller_declaration`` -- never
  "operational".
* the operational value is refused unless it does not exceed the randomised same-class reference
  (``Δ_op <= Δ_R``, or no saving): :class:`~.value.RandomizationChannelError`.  The
  randomisation-channel assessment (status, benchmark, diagnostic ratio) is carried in every output.
* ``unidentified_scenario`` / ``observed_state_scenario`` / ``error_model_unsourced`` results are
  converted only with an explicit ``scenario=True`` and keep their label in every output.

Third review R3B (2026-09-25; ``docs/value_semantics_decision.md``):

* the FIX_A recommendation "use ``executable_information_supported_value`` as the money basis" is
  withdrawn.  The definition is renamed ``heuristic_min_of_two_policy_values`` (deprecated alias kept):
  a heuristic of two policy classes that is not garbling-monotone (a pure garbling raises it from 0 to
  0.018857; at ``H x T = 1400`` head-days that is a synthetic gross amount of 26.4 for a *worse* signal).
  It is not a default of anything: a conversion accepts it only when the caller names it explicitly,
  and every output then carries ``value_role="heuristic"``, ``heuristic=True`` and
  ``excluded_from_paper_main_results=True`` -- never a payment bound, net value or assay priority for
  paper main results.
* the refusal of an operational value above the randomised reference has no recommended substitute:
  an economic value needs a complete decision problem (freely implementable information processing,
  allowed policies, baseline, risk timing, assay cost) that is not yet written.
* ``NetValueResult.randomization_share_of_operational`` is a deprecated alias of
  ``matched_uninformative_contrast_ratio`` (a diagnostic algebraic ratio, not a randomisation share).

Round-3 red team FIX3_BC (2026-09-25; finding B-1; ``docs/value_semantics_decision.md`` §2):
the R3B version still converted the *operational* value (and the randomised reference) to money by
default, unlabelled, and its only gate ``Δ_op <= Δ_R + tol`` is ``min(Δ_op, Δ_R) = Δ_op`` -- the minimum
of two policy classes deciding whether money is released.  That gate is not garbling-monotone: in the
red team's interior counterexample (``tests/unit/test_money_needs_a_complete_decision_problem.py``) a
pure garbling raises the released operational value from 0 to 0.2328 (break-even 0 -> 325.92 at
1 400 head-days) while passing the gate.  Now:

* **every** money conversion (:func:`per_head_day_value`, :func:`batch_gross_value`,
  :func:`break_even_max_cost_per_batch`, :func:`net_value_per_batch`; bound values and caller-declared
  floats alike, every definition) is refused by default (:class:`DecisionProblemRequiredError`) unless
  the caller passes either
  - ``decision_problem=`` a :class:`CompleteDecisionProblem` -- the five elements of VSD §2 ((a) freely
    implementable information processing, (b) allowed policies, (c) baseline, (d) risk timing, (e) assay
    cost), each with a status.  Element (a) fixes the **one** policy class whose ``V0 - VT`` is the money
    value (deterministic bin -> ration map without a free randomisation device: the operational
    difference; free randomisation of the ration choice: the randomised same-class value).  Only that
    definition is accepted (the heuristic minimum is refused: it is not the value of any single decision
    problem), the problem's ``alpha`` must equal the result's and its risk timing must be the ex-ante
    joint risk the engine computes.  **No quantity of the other policy class gates the money** -- the
    ``Δ_op <= Δ_R`` gate is not applied on this path; the cross-class comparison travels with the output
    as a diagnostic.  The only refusal kept is refusal-only: a saving that is entirely reachable without
    information (``randomization_only``) or cannot be assessed is never bought.  A deterministic-class
    value is labelled ``garbling_monotone=False`` (VSD §2: that class need not be garbling-monotone; the
    value is not a pure information value); or
  - ``diagnostic_without_decision_problem=True`` -- a development diagnostic: the FIX_A refusals are kept
    (refusal-only; passing them releases nothing), and the output is labelled
    ``money_basis="development_diagnostic_without_decision_problem"`` and
    ``excluded_from_paper_main_results=True`` with a :class:`UserWarning`.
* every money output carries ``excluded_from_paper_main_results=True`` in this version: RQ3 is an
  exploratory appendix (VSD §1; :data:`RQ3_MONEY_IN_PAPER_MAIN_RESULTS`), whatever the basis; the reason
  is in ``exclusion_reason``.
* the check order keeps every earlier, more specific refusal (definition, inventory, binding,
  identification, randomisation channel) before the decision-problem requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

import warnings as _warnings

from ..errors import InvalidProblemError
from ..hashing import stable_hash
from .value import (
    HEURISTIC_VALUE_DEFINITIONS,
    IDENTIFIED_LABELS,
    SCENARIO_IDENTIFICATION_LABELS,
    VALUE_DEFINITION_NOTES,
    VALUE_DEFINITION_ROLES,
    DefinitionLike,
    InformationValueResult,
    ValueDefinition,
    randomization_channel_assessment,
    require_executable_operational_value,
    require_operational_not_above_randomized_reference,
    resolve_value_definition,
    with_deprecated_assessment_keys,
)

__all__ = [
    "COST_COMPONENTS",
    "ECONOMIC_VALUE_DEFINITIONS",
    "VALUE_BINDINGS",
    "MONEY_BASES",
    "RQ3_MONEY_IN_PAPER_MAIN_RESULTS",
    "DECISION_PROBLEM_ELEMENT_STATUSES",
    "INFORMATION_PROCESSING_CONVENTIONS",
    "RISK_TIMINGS",
    "ENGINE_RISK_TIMING",
    "DecisionProblemRequiredError",
    "DecisionProblemElement",
    "CompleteDecisionProblem",
    "economic_value_definition",
    "ValueBinding",
    "PerHeadDayValue",
    "BatchAmount",
    "per_head_day_value",
    "BatchCoverage",
    "batch_gross_value",
    "break_even_max_cost_per_batch",
    "NetValueResult",
    "net_value_per_batch",
    "per_head_day_from_batch_cost",
    "InventoryCoverageResult",
    "inventory_coverage_check",
]

#: Cost components that must be declared (value or ``None`` = unknown) for a net value.
COST_COMPONENTS = ("sampling", "laboratory", "logistics", "waiting", "reformulation")

#: Definitions accepted as a per-head-day value basis for money conversions when named explicitly (R1);
#: the diagnostic contrast is refused.  R3B: the heuristic minimum is accepted only as an explicitly named
#: heuristic (outputs labelled ``heuristic``, excluded from paper main results); none is a default.
ECONOMIC_VALUE_DEFINITIONS = (ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value,
                              ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE.value,
                              ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value)

_ECON_INTERPRETATION = {
    ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value:
        "operational: executable deterministic policies at the same ex-ante joint risk (within the deterministic "
        "policy class; money only inside a complete decision problem, docs/value_semantics_decision.md)",
    ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE.value:
        "theoretical reference: assumes randomised feeding policies (never recommended); not an operational "
        "willingness to pay",
    ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value:
        "HEURISTIC (named explicitly by the caller): min(operational deterministic difference, randomised same-class "
        "reference) of two different policy classes; not garbling-monotone; not an established willingness to pay, "
        "payment bound, net value or assay priority; excluded from paper main results (R3B, "
        "docs/value_semantics_decision.md)",
}


def _is_heuristic(definition: str) -> bool:
    return definition in HEURISTIC_VALUE_DEFINITIONS


def economic_value_definition(value_definition: Optional[DefinitionLike], context: str) -> ValueDefinition:
    """Explicit definition allowed for a money conversion, else :class:`InvalidProblemError` (R1)."""
    d = resolve_value_definition(value_definition, context=context, allow_legacy=False)
    if d.value not in ECONOMIC_VALUE_DEFINITIONS:
        raise InvalidProblemError(
            f"{context}: {d.value} is a diagnostic contrast against a state-independent device with the same bin "
            "probabilities; it is not the value of buying the assay and cannot be converted to a payment bound, "
            "an H x T value or a net value (docs/value_definition.md)")
    return d


#: How a per-head-day value entered a money conversion (FIX_A).
VALUE_BINDINGS = ("bound_to_information_value_result", "caller_declared_unbound_float")

# ---------------------------------------------------------------------------------------------
# complete decision problem (round-3 red team FIX3_BC, finding B-1; VSD §2)
# ---------------------------------------------------------------------------------------------

#: Basis on which a money conversion was allowed (FIX3_BC).  There is no default basis.
MONEY_BASES = ("complete_decision_problem", "development_diagnostic_without_decision_problem")

#: Study decision (``docs/value_semantics_decision.md`` §0-§1): RQ3 is an exploratory appendix, so no money
#: output of this module is a paper main result in this version, whatever its basis.
RQ3_MONEY_IN_PAPER_MAIN_RESULTS = False

#: Status of each element of a :class:`CompleteDecisionProblem`.
DECISION_PROBLEM_ELEMENT_STATUSES = ("sourced", "research_scenario_assumption", "synthetic_test_only")

#: VSD §2 (a): what the decision maker may do for free with an assay result -> the one policy class whose
#: ``V0 - VT`` is the money value of that problem.  Both sides (with and without the assay) use the same class.
INFORMATION_PROCESSING_CONVENTIONS = {
    "deterministic_bin_to_ration_map_no_free_randomisation":
        ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value,
    "free_randomisation_of_the_ration_choice":
        ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE.value,
}

#: VSD §2 (d): risk timing.  The engine computes the ex-ante joint risk only; a per-outcome conditional risk
#: rule is another decision problem (T7.2) and is not converted here.
RISK_TIMINGS = ("ex_ante_joint_risk", "per_outcome_conditional_risk")
ENGINE_RISK_TIMING = "ex_ante_joint_risk"


class DecisionProblemRequiredError(InvalidProblemError):
    """A money conversion was asked for without a :class:`CompleteDecisionProblem` and without an explicit
    ``diagnostic_without_decision_problem=True`` (FIX3_BC, finding B-1)."""


@dataclass(frozen=True)
class DecisionProblemElement:
    """One element of a complete decision problem (VSD §2): a statement, its status and, if sourced, the source.

    ``choice`` is the machine-readable choice of elements (a) (a key of
    :data:`INFORMATION_PROCESSING_CONVENTIONS`) and (d) (one of :data:`RISK_TIMINGS`); ``None`` otherwise.
    """

    statement: str
    status: str
    source: Optional[str] = None
    choice: Optional[str] = None

    def __post_init__(self) -> None:
        if not (isinstance(self.statement, str) and self.statement.strip()):
            raise InvalidProblemError("DecisionProblemElement: a non-empty statement is required")
        if self.status not in DECISION_PROBLEM_ELEMENT_STATUSES:
            raise InvalidProblemError(f"DecisionProblemElement: status must be one of {DECISION_PROBLEM_ELEMENT_STATUSES}")
        if self.status == "sourced" and not (isinstance(self.source, str) and self.source.strip()):
            raise InvalidProblemError("DecisionProblemElement: a sourced element needs its source")

    def to_dict(self) -> dict:
        return {"statement": self.statement, "status": self.status, "source": self.source, "choice": self.choice}


@dataclass(frozen=True)
class CompleteDecisionProblem:
    """The five elements a money value must come from (``docs/value_semantics_decision.md`` §2; FIX3_BC).

    * ``information_processing`` (a): what the decision maker may do for free with the assay result;
      ``choice`` in :data:`INFORMATION_PROCESSING_CONVENTIONS` -- it fixes the one policy class (and hence
      the one value definition) of the problem, on both sides;
    * ``allowed_policies`` (b): action set, decision-time DM rule, library or full space, one ration per bin
      or mixtures, per-outcome rules;
    * ``baseline`` (c): the no-information optimum of the same class, alpha, risk event and prices;
    * ``risk_timing`` (d): ``choice`` in :data:`RISK_TIMINGS` (only ``ex_ante_joint_risk`` is computed by the
      engine), plus the assay's coverage / validity statement (numbers go to :class:`BatchCoverage`);
    * ``assay_cost`` (e): the cost components and their sources (numbers go to :func:`net_value_per_batch`).

    ``alpha`` must equal the ex-ante joint risk target of the result that is converted; ``declared_in`` names
    the document where the problem is written (protocol section, decision note).  A problem whose elements
    are not all ``sourced`` is a scenario (``is_scenario``); a problem with a ``synthetic_test_only`` element
    is synthetic.
    """

    problem_id: str
    information_processing: DecisionProblemElement
    allowed_policies: DecisionProblemElement
    baseline: DecisionProblemElement
    risk_timing: DecisionProblemElement
    assay_cost: DecisionProblemElement
    alpha: float
    declared_in: str

    ELEMENTS = ("information_processing", "allowed_policies", "baseline", "risk_timing", "assay_cost")

    def __post_init__(self) -> None:
        if not (isinstance(self.problem_id, str) and self.problem_id.strip()):
            raise InvalidProblemError("CompleteDecisionProblem: problem_id is required")
        if not (isinstance(self.declared_in, str) and self.declared_in.strip()):
            raise InvalidProblemError("CompleteDecisionProblem: declared_in (where the problem is written) is required")
        for n in self.ELEMENTS:
            if not isinstance(getattr(self, n), DecisionProblemElement):
                raise InvalidProblemError(f"CompleteDecisionProblem: {n} must be a DecisionProblemElement "
                                          "(all five elements of docs/value_semantics_decision.md §2 are required)")
        if self.information_processing.choice not in INFORMATION_PROCESSING_CONVENTIONS:
            raise InvalidProblemError("CompleteDecisionProblem: information_processing.choice must be one of "
                                      f"{tuple(INFORMATION_PROCESSING_CONVENTIONS)}")
        if self.risk_timing.choice not in RISK_TIMINGS:
            raise InvalidProblemError(f"CompleteDecisionProblem: risk_timing.choice must be one of {RISK_TIMINGS}")
        a = float(self.alpha)
        if not (math.isfinite(a) and 0.0 < a < 1.0):
            raise InvalidProblemError("CompleteDecisionProblem: alpha must be in (0, 1)")

    @property
    def value_definition(self) -> str:
        """The one definition whose number is the money value of this problem (element (a))."""
        return INFORMATION_PROCESSING_CONVENTIONS[str(self.information_processing.choice)]

    @property
    def is_scenario(self) -> bool:
        return any(getattr(self, n).status != "sourced" for n in self.ELEMENTS)

    @property
    def is_synthetic(self) -> bool:
        return any(getattr(self, n).status == "synthetic_test_only" for n in self.ELEMENTS)

    @property
    def excluded_from_paper_main_results(self) -> bool:
        return bool(self.is_scenario or not RQ3_MONEY_IN_PAPER_MAIN_RESULTS)

    def exclusion_reason(self) -> str:
        why = []
        if not RQ3_MONEY_IN_PAPER_MAIN_RESULTS:
            why.append("RQ3 is an exploratory appendix (docs/value_semantics_decision.md §1)")
        if self.is_scenario:
            why.append("decision-problem elements not all sourced: "
                       + ", ".join(f"{n}={getattr(self, n).status}" for n in self.ELEMENTS
                                   if getattr(self, n).status != "sourced"))
        return "; ".join(why)

    def fingerprint(self) -> str:
        return stable_hash("CompleteDecisionProblem/v1", self.problem_id,
                           [(n, sorted(getattr(self, n).to_dict().items(), key=lambda kv: kv[0])) for n in self.ELEMENTS],
                           float(self.alpha), self.declared_in)

    def to_dict(self) -> dict:
        return {"problem_id": self.problem_id, "alpha": float(self.alpha), "declared_in": self.declared_in,
                "value_definition": self.value_definition, "is_scenario": self.is_scenario,
                "is_synthetic": self.is_synthetic, "fingerprint": self.fingerprint(),
                **{n: getattr(self, n).to_dict() for n in self.ELEMENTS}}

    def check_result(self, result: InformationValueResult, context: str) -> None:
        """The result must belong to this problem: same alpha, ex-ante joint risk (the engine's risk timing)."""
        if self.risk_timing.choice != ENGINE_RISK_TIMING:
            raise InvalidProblemError(
                f"{context}: decision problem {self.problem_id} uses risk timing {self.risk_timing.choice!r}; the engine "
                "computes the ex-ante joint risk only (a per-outcome conditional risk rule is another problem, T7.2)")
        if abs(float(result.alpha) - float(self.alpha)) > 1e-12:
            raise InvalidProblemError(
                f"{context}: decision problem {self.problem_id} is declared for alpha {self.alpha:g}, the result "
                f"{result.structure_id} for alpha {result.alpha:g}")


def _money_basis_args(decision_problem: Optional[CompleteDecisionProblem], diagnostic: bool, context: str) -> None:
    if decision_problem is not None and not isinstance(decision_problem, CompleteDecisionProblem):
        raise InvalidProblemError(f"{context}: decision_problem must be a CompleteDecisionProblem")
    if not isinstance(diagnostic, (bool, np.bool_)):
        raise InvalidProblemError(f"{context}: diagnostic_without_decision_problem must be True or False")
    if decision_problem is not None and diagnostic:
        raise InvalidProblemError(f"{context}: give either decision_problem or diagnostic_without_decision_problem=True, "
                                  "not both")


def _require_money_basis(decision_problem: Optional[CompleteDecisionProblem], diagnostic: bool, context: str,
                         what: str) -> None:
    """Last check of every money conversion (after all value-level refusals): no basis -> refused (FIX3_BC)."""
    if decision_problem is None and not diagnostic:
        raise DecisionProblemRequiredError(
            f"{context}: {what}: no money value without a complete decision problem -- pass decision_problem= a "
            "CompleteDecisionProblem (VSD §2: free information processing, allowed policies, baseline, risk timing, "
            "assay cost; the money value is V0 - VT of that one policy class), or diagnostic_without_decision_problem="
            "True for a labelled development diagnostic (excluded from paper main results). Passing the operational <= "
            "randomised-reference check is not a basis: it compares two policy classes and is not garbling-monotone "
            "(round-3 red team B-1; docs/value_semantics_decision.md)")


_DIAGNOSTIC_EXCLUSION = ("development diagnostic without a complete decision problem (not a payment bound, willingness to "
                         "pay, net value or assay priority); RQ3 is an exploratory appendix (docs/value_semantics_decision.md "
                         "§1-§2)")


@dataclass(frozen=True)
class ValueBinding:
    """Where a per-head-day value comes from and what it may be used for (FIX_A).

    ``randomization_channel`` is the :func:`~.value.randomization_channel_assessment` of the source
    result (empty for an unbound float); ``identification_scenario`` is True when the result was not
    identified and the conversion was explicitly allowed as a scenario (``scenario=True``).
    ``heuristic`` / ``excluded_from_paper_main_results`` (R3B) are True for the explicitly named
    heuristic ``heuristic_min_of_two_policy_values`` (the latter also for the diagnostic contrast).

    FIX3_BC: ``money_basis`` (:data:`MONEY_BASES`) says why the conversion was allowed; with a complete
    decision problem its id and fingerprint are recorded.  ``excluded_from_paper_main_results`` is True for
    every money output of this version (``exclusion_reason``); ``garbling_monotone`` is ``False`` for a value
    of the deterministic policy class (no such guarantee), ``True`` for the randomised same-class value
    (finite-model theorem), ``None`` otherwise.
    """

    binding: str
    value_definition: str
    value_role: str
    structure_id: Optional[str] = None
    result_fingerprint: Optional[str] = None
    error_model_identification: Optional[str] = None
    identification_scenario: bool = False
    randomization_channel: Mapping[str, Any] = field(default_factory=dict)
    declaration: str = ""
    is_synthetic: Optional[bool] = None
    heuristic: bool = False
    excluded_from_paper_main_results: bool = False
    money_basis: Optional[str] = None
    decision_problem_id: Optional[str] = None
    decision_problem_fingerprint: Optional[str] = None
    decision_problem_is_scenario: Optional[bool] = None
    exclusion_reason: str = ""
    garbling_monotone: Optional[bool] = None

    def to_dict(self) -> dict:
        return {"binding": self.binding, "value_definition": self.value_definition, "value_role": self.value_role,
                "structure_id": self.structure_id, "result_fingerprint": self.result_fingerprint,
                "error_model_identification": self.error_model_identification,
                "identification_scenario": self.identification_scenario,
                "randomization_channel": dict(self.randomization_channel), "declaration": self.declaration,
                "is_synthetic": self.is_synthetic, "heuristic": self.heuristic,
                "excluded_from_paper_main_results": self.excluded_from_paper_main_results,
                "money_basis": self.money_basis, "decision_problem_id": self.decision_problem_id,
                "decision_problem_fingerprint": self.decision_problem_fingerprint,
                "decision_problem_is_scenario": self.decision_problem_is_scenario,
                "exclusion_reason": self.exclusion_reason, "garbling_monotone": self.garbling_monotone}


def _garbling_monotone(definition: str) -> Optional[bool]:
    if definition == ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value:
        return False
    if definition == ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE.value:
        return True
    return None


def _basis_fields(definition: str, decision_problem: Optional[CompleteDecisionProblem]) -> dict[str, Any]:
    """Money-basis fields of a binding (FIX3_BC); every money output is excluded from paper main results here."""
    if decision_problem is None:
        return {"money_basis": "development_diagnostic_without_decision_problem", "decision_problem_id": None,
                "decision_problem_fingerprint": None, "decision_problem_is_scenario": None,
                "exclusion_reason": _DIAGNOSTIC_EXCLUSION, "garbling_monotone": _garbling_monotone(definition),
                "excluded_from_paper_main_results": True}
    return {"money_basis": "complete_decision_problem", "decision_problem_id": decision_problem.problem_id,
            "decision_problem_fingerprint": decision_problem.fingerprint(),
            "decision_problem_is_scenario": decision_problem.is_scenario,
            "exclusion_reason": decision_problem.exclusion_reason(), "garbling_monotone": _garbling_monotone(definition),
            "excluded_from_paper_main_results": decision_problem.excluded_from_paper_main_results}


def _warn_money(context: str, definition: str, decision_problem: Optional[CompleteDecisionProblem]) -> None:
    if decision_problem is None:
        _warnings.warn(f"{context}: {definition} converted as a DEVELOPMENT DIAGNOSTIC without a complete decision "
                       "problem: labelled excluded_from_paper_main_results=True; not a payment bound, willingness to pay, "
                       "net value or assay priority (docs/value_semantics_decision.md §2)", UserWarning, stacklevel=3)
    elif decision_problem.is_scenario:
        _warnings.warn(f"{context}: money value of the SCENARIO decision problem {decision_problem.problem_id} "
                       f"({decision_problem.exclusion_reason()}); labelled excluded_from_paper_main_results=True",
                       UserWarning, stacklevel=3)


class PerHeadDayValue(float):
    """``dc`` (currency/head/d) that remembers its :class:`ValueBinding` (FIX_A).  Behaves as a ``float``."""

    binding: ValueBinding

    def __new__(cls, value: float, binding: ValueBinding) -> "PerHeadDayValue":
        obj = super().__new__(cls, float(value))
        obj.binding = binding
        return obj

    def __repr__(self) -> str:
        b = self.binding
        return (f"PerHeadDayValue({float(self)!r}, {b.value_definition}, role={b.value_role}, "
                f"identification={b.error_model_identification}, "
                f"randomization_channel={dict(b.randomization_channel).get('status')}, money_basis={b.money_basis}"
                + (", HEURISTIC" if b.heuristic else "")
                + (", excluded from paper main results" if b.excluded_from_paper_main_results else "") + ")")


class BatchAmount(float):
    """A per-batch amount (currency/batch) that remembers its :class:`ValueBinding` and ``H * T`` (FIX_A)."""

    binding: ValueBinding
    head_days: float

    def __new__(cls, value: float, binding: ValueBinding, head_days: float) -> "BatchAmount":
        obj = super().__new__(cls, float(value))
        obj.binding = binding
        obj.head_days = float(head_days)
        return obj

    def __repr__(self) -> str:
        b = self.binding
        return (f"BatchAmount({float(self)!r}, {b.value_definition}, role={b.value_role}, "
                f"H*T={self.head_days:g}, money_basis={b.money_basis}"
                + (", HEURISTIC" if b.heuristic else "")
                + (", excluded from paper main results" if b.excluded_from_paper_main_results else "") + ")")


def per_head_day_value(result: InformationValueResult, value_definition: Optional[DefinitionLike], *,
                       scenario: bool = False, decision_problem: Optional[CompleteDecisionProblem] = None,
                       diagnostic_without_decision_problem: bool = False) -> PerHeadDayValue:
    """``dc`` (currency/head/d) of ``result`` under an explicit, economically admissible definition.

    FIX3_BC (round-3 red team B-1): refused (:class:`DecisionProblemRequiredError`) unless
    ``decision_problem`` (a :class:`CompleteDecisionProblem`; only its one definition is accepted, its alpha
    must match, the cross-class ``Δ_op <= Δ_R`` gate is not applied, ``randomization_only`` is still refused)
    or ``diagnostic_without_decision_problem=True`` (FIX_A refusals kept; output labelled a development
    diagnostic, excluded from paper main results, :class:`UserWarning`).  Every other refusal below comes first.

    Raises if no definition is given, if it is the diagnostic contrast, if the two sides compare
    different action spaces (T7.2: not an information value) or if the value is undefined.  The
    value may be negative; it is not truncated.

    FIX_A: the operational value is refused unless it does not exceed the randomised same-class
    reference (``Δ_op <= Δ_R + tol`` or no saving) -- :class:`~.value.RandomizationChannelError`.  Results
    labelled ``unidentified_scenario`` / ``observed_state_scenario`` / ``error_model_unsourced`` need
    ``scenario=True``.  The returned :class:`PerHeadDayValue` carries the binding (definition, result
    fingerprint, identification label, randomisation-channel assessment and diagnostic ratio).

    R3B: ``heuristic_min_of_two_policy_values`` (deprecated name ``executable_information_supported_value``)
    is converted only because the caller named it; the binding says ``heuristic=True`` and
    ``excluded_from_paper_main_results=True`` and a :class:`UserWarning` restates that it is not a
    payment bound (``docs/value_semantics_decision.md``).
    """
    d = economic_value_definition(value_definition, "per_head_day_value")
    _money_basis_args(decision_problem, diagnostic_without_decision_problem, "per_head_day_value")
    if not isinstance(result, InformationValueResult):
        raise InvalidProblemError("per_head_day_value: result must be an InformationValueResult")
    if not result.is_information_value:
        raise InvalidProblemError(f"{result.structure_id}: {result.information_value_semantics}; V0 - VT is not an "
                                  "information value and has no economic value basis (T7.2)")
    v = result.value(d)
    if v is None:
        raise InvalidProblemError(f"{result.structure_id}: {d.value} is undefined here "
                                  f"({result.definition_report(d)['values'][d.value]['status']})")
    ident = result.error_model_identification
    if ident not in IDENTIFIED_LABELS:
        if not scenario:
            raise InvalidProblemError(
                f"{result.structure_id}: error_model_identification={ident!r} is not an identified value; a money "
                "conversion of a scenario needs scenario=True and keeps the label in every output (review R2-5, "
                "B-K2-2)")
        if ident not in SCENARIO_IDENTIFICATION_LABELS:
            raise InvalidProblemError(f"{result.structure_id}: unknown identification label {ident!r}")
    if decision_problem is not None:
        # FIX3_BC: the money value is V0 - VT of the problem's one policy class; nothing of the other class gates it
        if d.value != decision_problem.value_definition:
            raise InvalidProblemError(
                f"per_head_day_value: decision problem {decision_problem.problem_id} fixes the policy class by its "
                f"information processing ({decision_problem.information_processing.choice}); its money value is "
                f"{decision_problem.value_definition}, not {d.value}"
                + (" (the heuristic minimum of two policy classes is not the value of any single decision problem)"
                   if _is_heuristic(d.value) else ""))
        decision_problem.check_result(result, "per_head_day_value")
        if d is ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE:
            # refusal-only: a saving entirely reachable without information (or not assessable) is never bought
            rc = require_executable_operational_value(result, "per_head_day_value")
        else:
            rc = randomization_channel_assessment(result)
    else:
        if d is ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE:
            # FIX_A refusal kept on the diagnostic path (refusal-only: passing it releases nothing, B-1)
            rc = require_operational_not_above_randomized_reference(result, "per_head_day_value")
        else:
            rc = randomization_channel_assessment(result)
    _require_money_basis(decision_problem, diagnostic_without_decision_problem, "per_head_day_value",
                         f"{result.structure_id} / {d.value}")
    heur = _is_heuristic(d.value)
    if heur:
        _warnings.warn(f"per_head_day_value: {result.structure_id}: converting the explicitly named HEURISTIC "
                       f"{d.value} (min of two policy classes; not garbling-monotone): the output is labelled heuristic "
                       "and is not a payment bound, net value or assay priority for paper main results "
                       "(docs/value_semantics_decision.md)", UserWarning, stacklevel=2)
    _warn_money("per_head_day_value", d.value, decision_problem)
    b = ValueBinding("bound_to_information_value_result", d.value, VALUE_DEFINITION_ROLES[d.value],
                     result.structure_id, result.fingerprint, ident, ident not in IDENTIFIED_LABELS,
                     with_deprecated_assessment_keys(rc), "", bool(result.is_synthetic), heur,
                     **_basis_fields(d.value, decision_problem))
    return PerHeadDayValue(float(v), b)


def _bind(delta_c: Any, value_definition: Optional[DefinitionLike], *, scenario: bool,
          unbound_value_declaration: Optional[str], context: str,
          decision_problem: Optional[CompleteDecisionProblem] = None,
          diagnostic_without_decision_problem: bool = False) -> tuple[float, ValueBinding]:
    """``(dc, binding)`` for a money conversion; the named definition must be the bound one (FIX_A); the money
    basis (decision problem or explicit diagnostic) must be given and must match a bound value's (FIX3_BC)."""
    d = economic_value_definition(value_definition, context)
    _money_basis_args(decision_problem, diagnostic_without_decision_problem, context)
    if isinstance(delta_c, InformationValueResult):
        v = per_head_day_value(delta_c, d, scenario=scenario, decision_problem=decision_problem,
                               diagnostic_without_decision_problem=diagnostic_without_decision_problem)
        return float(v), v.binding
    if isinstance(delta_c, PerHeadDayValue):
        b = delta_c.binding
        if b.value_definition != d.value:
            raise InvalidProblemError(f"{context}: the value is bound to {b.value_definition}, not {d.value}; a value "
                                      "cannot be relabelled (FIX_A)")
        _require_money_basis(decision_problem, diagnostic_without_decision_problem, context,
                             f"value bound to {b.structure_id} / {b.value_definition}")
        want_fp = None if decision_problem is None else decision_problem.fingerprint()
        if b.decision_problem_fingerprint != want_fp or (decision_problem is None) != \
                (b.money_basis == "development_diagnostic_without_decision_problem"):
            raise InvalidProblemError(f"{context}: the value was bound under money basis {b.money_basis} "
                                      f"({b.decision_problem_id}); it cannot be converted under another basis (FIX3_BC)")
        return float(delta_c), b
    if isinstance(delta_c, (bool, np.bool_)) or not isinstance(delta_c, (int, float, np.integer, np.floating)):
        raise InvalidProblemError(f"{context}: delta_c must be an InformationValueResult, a PerHeadDayValue or a float")
    if not (isinstance(unbound_value_declaration, str) and unbound_value_declaration.strip()):
        raise InvalidProblemError(
            f"{context}: a bare float is not bound to any InformationValueResult, so nothing checks that it is the "
            f"{d.value} it is labelled as (a diagnostic contrast or a randomised reference could be passed as "
            "'operational'). Pass the result or per_head_day_value(result, ...), or name the float with "
            "unbound_value_declaration='...' (outputs are then labelled caller_declared_unbound_float) (FIX_A)")
    if decision_problem is not None and d.value != decision_problem.value_definition:
        raise InvalidProblemError(f"{context}: decision problem {decision_problem.problem_id} values "
                                  f"{decision_problem.value_definition}, not {d.value}")
    _require_money_basis(decision_problem, diagnostic_without_decision_problem, context,
                         f"caller-declared float labelled {d.value}")
    _warn_money(context, d.value, decision_problem)
    b = ValueBinding("caller_declared_unbound_float", d.value, "unbound_caller_declaration", None, None, None, False,
                     {}, unbound_value_declaration.strip(), None, _is_heuristic(d.value),
                     **_basis_fields(d.value, decision_problem))
    return float(delta_c), b


@dataclass(frozen=True)
class BatchCoverage:
    """Heads ``H`` and days ``T`` covered by one assayed batch, with provenance notes.

    ``heads`` / ``days`` must come from the inventory scenario (sourced or declared assumption).
    ``validity_period_days`` (required) is the period for which one assay result is used (e.g. the
    3/14/30-day scenarios of the feasibility audit §4.5); ``days`` may not exceed it.
    ``validity_status`` defaults to ``days_status`` (the same declaration covers both).
    """

    heads: float
    days: float
    heads_status: str
    days_status: str
    validity_period_days: float
    notes: str = ""
    validity_status: Optional[str] = None

    def __post_init__(self) -> None:
        for n in ("heads", "days", "validity_period_days"):
            raw = getattr(self, n)
            if raw is None:
                raise InvalidProblemError(f"BatchCoverage.{n} is required (no default coverage or validity period)")
            v = float(raw)
            if not np.isfinite(v) or v <= 0:
                raise InvalidProblemError(f"BatchCoverage.{n} must be finite and > 0")
        if self.validity_status is None:
            object.__setattr__(self, "validity_status", self.days_status)
        for n in ("heads_status", "days_status", "validity_status"):
            if getattr(self, n) not in ("sourced", "research_scenario_assumption", "synthetic_test_only"):
                raise InvalidProblemError(f"BatchCoverage.{n} must be sourced / research_scenario_assumption / "
                                          "synthetic_test_only")
        if float(self.days) > float(self.validity_period_days) + 1e-12:
            raise InvalidProblemError("BatchCoverage: days exceed the validity period of one assay result; split the "
                                      "period into several assay events")

    @property
    def head_days(self) -> float:
        """``H * T``."""
        return float(self.heads) * float(self.days)


def _require_fixed_multiplier(coverage: BatchCoverage, inventory_check: "InventoryCoverageResult") -> None:
    if not isinstance(inventory_check, InventoryCoverageResult):
        raise InvalidProblemError("an InventoryCoverageResult (inventory_coverage_check) is required before any H x T "
                                  "conversion (contract §12.5, T2.3)")
    if not (abs(float(inventory_check.heads) - float(coverage.heads)) <= 1e-12 and
            abs(float(inventory_check.days) - float(coverage.days)) <= 1e-12):
        raise InvalidProblemError("inventory check was made for another coverage (H, T differ or not recorded)")
    if not inventory_check.fixed_multiplier_valid:
        raise InvalidProblemError(f"inventory coverage is {inventory_check.status!r}: the fixed H x T multiplier is not "
                                  "valid for this policy (re-check coverage; no per-batch value)")


def batch_gross_value(delta_c_per_head_day: Union[InformationValueResult, PerHeadDayValue, float],
                      coverage: BatchCoverage, *,
                      inventory_check: "InventoryCoverageResult",
                      value_definition: Optional[DefinitionLike] = None, scenario: bool = False,
                      unbound_value_declaration: Optional[str] = None,
                      decision_problem: Optional[CompleteDecisionProblem] = None,
                      diagnostic_without_decision_problem: bool = False) -> BatchAmount:
    """``H * T * dc`` (currency per assayed batch).  ``dc`` may be negative; it is not truncated.

    Requires ``inventory_check`` (same ``H``, ``T``) with ``fixed_multiplier_valid=True`` and an
    explicit ``value_definition`` naming what ``dc`` is (R1: operational deterministic difference or
    the theoretical randomised reference; the diagnostic contrast is refused; R3B: the heuristic minimum
    only when named, labelled ``heuristic`` and excluded from paper main results).  FIX_A: ``dc`` is an
    :class:`~.value.InformationValueResult` or a :class:`PerHeadDayValue` from
    :func:`per_head_day_value` (whose checks apply: randomisation channel, identification with
    ``scenario``); a bare float needs ``unbound_value_declaration``.  The returned
    :class:`BatchAmount` carries the binding.  FIX3_BC: a money basis is required -- ``decision_problem``
    (:class:`CompleteDecisionProblem`) or ``diagnostic_without_decision_problem=True`` (see
    :func:`per_head_day_value`); otherwise :class:`DecisionProblemRequiredError`.
    """
    economic_value_definition(value_definition, "batch_gross_value")
    _money_basis_args(decision_problem, diagnostic_without_decision_problem, "batch_gross_value")
    _require_fixed_multiplier(coverage, inventory_check)
    dc, b = _bind(delta_c_per_head_day, value_definition, scenario=scenario,
                  unbound_value_declaration=unbound_value_declaration, context="batch_gross_value",
                  decision_problem=decision_problem,
                  diagnostic_without_decision_problem=diagnostic_without_decision_problem)
    if not np.isfinite(dc):
        raise InvalidProblemError("delta_c must be finite")
    return BatchAmount(coverage.head_days * dc, b, coverage.head_days)


def break_even_max_cost_per_batch(delta_c_per_head_day: Union[InformationValueResult, PerHeadDayValue, float],
                                  coverage: BatchCoverage, *,
                                  inventory_check: "InventoryCoverageResult",
                                  value_definition: Optional[DefinitionLike] = None, scenario: bool = False,
                                  unbound_value_declaration: Optional[str] = None,
                                  decision_problem: Optional[CompleteDecisionProblem] = None,
                                  diagnostic_without_decision_problem: bool = False) -> BatchAmount:
    """Highest total information cost per batch at which the net value is still >= 0.

    Equals ``H * T * dc``; a value <= 0 means that no positive cost is justified in the declared
    model (reported as is).  Same inventory, explicit-definition, binding and money-basis requirements as
    :func:`batch_gross_value`; with ``randomized_same_class_information_reference`` the result is a
    theoretical break-even (randomised feeding policies), not an operational willingness to pay, unless a
    complete decision problem declares free randomisation.  FIX_A: an operational value that is a
    randomisation channel only has no break-even (refused).  FIX3_BC: without a complete decision problem
    this is at most a labelled development diagnostic.
    """
    economic_value_definition(value_definition, "break_even_max_cost_per_batch")
    return batch_gross_value(delta_c_per_head_day, coverage, inventory_check=inventory_check,
                             value_definition=value_definition, scenario=scenario,
                             unbound_value_declaration=unbound_value_declaration, decision_problem=decision_problem,
                             diagnostic_without_decision_problem=diagnostic_without_decision_problem)


def per_head_day_from_batch_cost(cost_per_batch: float, coverage: BatchCoverage, *,
                                 inventory_check: "InventoryCoverageResult") -> float:
    """Batch cost spread over the covered head-days (``cost / (H T)``); same inventory requirement.

    Converts an assay *cost*, not an information value, so no ``value_definition`` is involved (R1 audit).
    """
    _require_fixed_multiplier(coverage, inventory_check)
    return float(cost_per_batch) / coverage.head_days


@dataclass(frozen=True)
class NetValueResult:
    """Net value per batch, or ``None`` with the list of unknown cost components.

    FIX_A: the binding of the per-head-day value travels with the result -- where it comes from
    (``value_binding``, ``structure_id``, ``result_fingerprint``), its identification label (kept for
    scenarios, ``identification_scenario``) and the randomisation-channel assessment of the operational
    value (``randomization_channel``: status, benchmark, diagnostic ratio).  R3B: ``heuristic`` and
    ``excluded_from_paper_main_results`` are True for the explicitly named heuristic minimum.
    """

    gross_value_per_batch: float
    break_even_max_cost_per_batch: float
    known_cost_total: float
    unknown_components: tuple[str, ...]
    net_value_per_batch: Optional[float]
    status: str
    notes: str = field(default="")
    #: explicit definition of the per-head-day value (R1) and its role / interpretation
    value_definition: Optional[str] = None
    value_role: Optional[str] = None
    value_interpretation: str = ""
    #: FIX_A binding of the per-head-day value
    value_binding: Optional[str] = None
    structure_id: Optional[str] = None
    result_fingerprint: Optional[str] = None
    error_model_identification: Optional[str] = None
    identification_scenario: bool = False
    randomization_channel: Mapping[str, Any] = field(default_factory=dict)
    unbound_value_declaration: str = ""
    #: R3B: the per-head-day value is the explicitly named heuristic minimum
    heuristic: bool = False
    excluded_from_paper_main_results: bool = False
    #: FIX3_BC: money basis (complete decision problem or labelled development diagnostic)
    money_basis: Optional[str] = None
    decision_problem_id: Optional[str] = None
    decision_problem_fingerprint: Optional[str] = None
    decision_problem_is_scenario: Optional[bool] = None
    exclusion_reason: str = ""
    garbling_monotone: Optional[bool] = None

    @property
    def randomization_channel_status(self) -> Optional[str]:
        return dict(self.randomization_channel).get("status")

    @property
    def randomization_benchmark_value(self) -> Optional[float]:
        return dict(self.randomization_channel).get("randomization_benchmark_value")

    @property
    def matched_uninformative_contrast_ratio(self) -> Optional[float]:
        """``B / operational`` for the matched state-independent benchmark (diagnostic ratio, R3B)."""
        return dict(self.randomization_channel).get("matched_uninformative_contrast_ratio")

    @property
    def randomization_share_of_operational(self) -> Optional[float]:
        """DEPRECATED alias of :attr:`matched_uninformative_contrast_ratio` (R3B): an algebraic ratio for one
        benchmark, not an identified share of the saving caused by randomisation."""
        _warnings.warn("randomization_share_of_operational is deprecated; use matched_uninformative_contrast_ratio (a "
                       "diagnostic ratio, not a randomisation share; docs/value_semantics_decision.md)",
                       DeprecationWarning, stacklevel=2)
        return self.matched_uninformative_contrast_ratio


def net_value_per_batch(delta_c_per_head_day: Union[InformationValueResult, PerHeadDayValue, float],
                        coverage: BatchCoverage,
                        cost_components: Mapping[str, Optional[float]], *,
                        inventory_check: "InventoryCoverageResult",
                        value_definition: Optional[DefinitionLike] = None, scenario: bool = False,
                        unbound_value_declaration: Optional[str] = None,
                        decision_problem: Optional[CompleteDecisionProblem] = None,
                        diagnostic_without_decision_problem: bool = False) -> NetValueResult:
    """Net value ``H T dc - sum(costs)``; ``None`` if any declared component is unknown.

    All of :data:`COST_COMPONENTS` must be present as keys (value or ``None``) so that forgetting a
    component cannot silently mean "free".  ``value_definition`` is required (R1); the result records
    it with its role (``operational``, ``theoretical_reference`` or, R3B, ``heuristic``) and, FIX_A, the
    binding of the value (source result, identification label, randomisation-channel status and
    diagnostic ratio).  R3B: a net value of the explicitly named heuristic minimum carries
    ``heuristic=True`` and ``excluded_from_paper_main_results=True``.  FIX3_BC: a money basis is required
    (``decision_problem`` or ``diagnostic_without_decision_problem=True``); the result records it
    (``money_basis``, ``decision_problem_id`` ...) and is excluded from paper main results in this version.
    """
    d = economic_value_definition(value_definition, "net_value_per_batch")
    _money_basis_args(decision_problem, diagnostic_without_decision_problem, "net_value_per_batch")
    missing = [c for c in COST_COMPONENTS if c not in cost_components]
    if missing:
        raise InvalidProblemError(f"cost components not declared (use None if unknown): {missing}")
    gross = batch_gross_value(delta_c_per_head_day, coverage, inventory_check=inventory_check, value_definition=d,
                              scenario=scenario, unbound_value_declaration=unbound_value_declaration,
                              decision_problem=decision_problem,
                              diagnostic_without_decision_problem=diagnostic_without_decision_problem)
    b = gross.binding
    unknown = tuple(sorted(k for k, v in cost_components.items() if v is None))
    for k, v in cost_components.items():
        if v is not None and (not np.isfinite(float(v)) or float(v) < 0):
            raise InvalidProblemError(f"cost component {k} must be finite and >= 0")
    known = float(sum(float(v) for v in cost_components.values() if v is not None))
    interp = _ECON_INTERPRETATION[d.value] + "; " + VALUE_DEFINITION_NOTES[d.value]
    if b.binding == "caller_declared_unbound_float":
        interp = ("caller-declared float, not bound to an InformationValueResult (no randomisation or identification "
                  f"check): {b.declaration}; labelled {d.value} by the caller only"
                  + ("; HEURISTIC: not for paper main results" if b.heuristic else ""))
    elif b.identification_scenario:
        interp += f"; SCENARIO: error_model_identification={b.error_model_identification}"
    rc = with_deprecated_assessment_keys(b.randomization_channel)
    if rc.get("status") == "partly_randomization_channel":
        interp += (f"; matched state-independent benchmark reaches "
                   f"{rc.get('randomization_benchmark_value'):.6g} of the operational saving "
                   f"(matched_uninformative_contrast_ratio {rc.get('matched_uninformative_contrast_ratio'):.3g}: a "
                   "diagnostic ratio, not a randomisation share)")
    if b.money_basis == "complete_decision_problem":
        interp += (f"; MONEY BASIS: complete decision problem {b.decision_problem_id} (V0 - VT of its one policy class; "
                   "no quantity of the other policy class gates it)")
        if b.garbling_monotone is False:
            interp += ("; deterministic policy class: not garbling-monotone, not a pure information value "
                       "(docs/value_semantics_decision.md §2)")
    else:
        interp += "; MONEY BASIS: development diagnostic without a complete decision problem (FIX3_BC)"
    if b.excluded_from_paper_main_results:
        interp += f"; EXCLUDED FROM PAPER MAIN RESULTS: {b.exclusion_reason}"
    lab = {"value_definition": d.value, "value_role": b.value_role, "value_interpretation": interp,
           "value_binding": b.binding, "structure_id": b.structure_id, "result_fingerprint": b.result_fingerprint,
           "error_model_identification": b.error_model_identification,
           "identification_scenario": b.identification_scenario, "randomization_channel": rc,
           "unbound_value_declaration": b.declaration, "heuristic": b.heuristic,
           "excluded_from_paper_main_results": b.excluded_from_paper_main_results,
           "money_basis": b.money_basis, "decision_problem_id": b.decision_problem_id,
           "decision_problem_fingerprint": b.decision_problem_fingerprint,
           "decision_problem_is_scenario": b.decision_problem_is_scenario,
           "exclusion_reason": b.exclusion_reason, "garbling_monotone": b.garbling_monotone}
    g = float(gross)
    if unknown:
        return NetValueResult(g, g, known, unknown, None, "net_value_unavailable_unknown_costs",
                              "report the break-even maximum cost; unknown components are not zero", **lab)
    return NetValueResult(g, g, known, (), g - known, "net_value_all_components_declared", **lab)


@dataclass(frozen=True)
class InventoryCoverageResult:
    """Whether the batch inventory covers ``H x T`` for every ration the policy may feed."""

    status: str                               # covered / not_covered / unverified
    required_kg_as_fed: Mapping[str, float]
    inventory_kg_as_fed: Mapping[str, Optional[float]]
    shortfalls: Mapping[str, float]
    unverified_ingredients: tuple[str, ...]
    fixed_multiplier_valid: bool
    heads: float = float("nan")               # coverage the check was made for (binds the result)
    days: float = float("nan")


def inventory_coverage_check(rations_q: np.ndarray, ingredient_ids: Sequence[str], coverage: BatchCoverage,
                             inventory_kg_as_fed: Mapping[str, Optional[float]],
                             bin_probabilities: Optional[np.ndarray] = None) -> InventoryCoverageResult:
    """Check ``H T max_z q_{k(z), i} <= B_i`` over the rations used in bins with positive probability.

    ``rations_q`` is ``[n_rations, I]`` (e.g. ``library.Q[assignment]``).  If a bin's ration needs
    more than the inventory, the fixed ``H x T`` multiplier is invalid for that policy (T2.3,
    §12.5).  Missing inventory entries make the check ``unverified`` (never assumed sufficient).
    """
    Q = np.asarray(rations_q, dtype=float)
    ids = tuple(ingredient_ids)
    if Q.ndim != 2 or Q.shape[1] != len(ids):
        raise InvalidProblemError("inventory_coverage_check: rations must be [n, I]")
    if bin_probabilities is not None:
        p = np.asarray(bin_probabilities, dtype=float)
        if p.shape != (Q.shape[0],):
            raise InvalidProblemError("inventory_coverage_check: bin_probabilities must match rations")
        Q = Q[p > 0]
    need = coverage.head_days * Q.max(axis=0)
    req = {i: float(n) for i, n in zip(ids, need) if n > 0}
    short: dict[str, float] = {}
    unver: list[str] = []
    for i, n in req.items():
        b = inventory_kg_as_fed.get(i)
        if b is None:
            unver.append(i)
        elif n > float(b) + 1e-9:
            short[i] = n - float(b)
    status = "not_covered" if short else ("unverified" if unver else "covered")
    return InventoryCoverageResult(status, req, dict(inventory_kg_as_fed), short, tuple(unver), status == "covered",
                                   float(coverage.heads), float(coverage.days))
