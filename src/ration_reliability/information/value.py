"""Gross information value ``V_0 - V_T`` in one declared decision model (contract T7.1-T7.4, §12).

Every value computed here is a **value within a finite candidate library K and fixed development
signal bins** under an ex-ante joint risk constraint.  It is neither a lower nor an upper bound of
a continuous global EVPI/EVPPI/EVSI unless separately proven (T7.3); approximation error is checked
by K enlargement, bin refinement and exhaustive enumeration on small problems (design note §6).

Kinds of information are kept apart (§12.1, F01):

* ``perfect_full``      -- ideal knowledge of every modelled unknown (EVPI within K);
* ``perfect_partial``   -- ideal knowledge of listed components, exact or binned (EVPPI within K);
* ``sample``            -- a signal with sampling / laboratory / bias error (EVSI within K; only
  with a source-backed or explicitly scenario-labelled error model and a passed
  double-count guard);
* ``uninformative``     -- a state-independent signal (randomisation benchmark, not information).

The theoretical ordering ``0 <= EVSI <= EVPPI <= EVPI`` is only *checked* (diagnostic); violations
are reported with their size and cause class, never truncated or re-ordered (T7.4, F10).

Red-team fixes (2026-09-24)
---------------------------
* C05/F02 -- action sets follow the decision-time information
  (:func:`~.library.decision_time_action_masks`): V0 and structures without DM information may only
  use rations that are structurally feasible under the t0 DM estimate; a DM-observing structure
  uses, in each bin, the rations feasible under that bin's ``E[d | z]``.  If the constant
  no-information policies are not all legal in every active bin, ``V0 - VT`` compares two action
  spaces: it is reported as ``cost_difference_V0_minus_VT`` with
  ``information_value_semantics = "cost_risk_comparison_action_spaces_differ"`` and
  ``gross_value = None`` (T7.2: not an information value; open decision PUD-P8-03).
* F03 finding -- ``VT(state-independent signal, same P_z) - VT`` is reported (a state-independent
  signal gets exactly 0).  Second review R1 (2026-09-25): this is **not** "the information value net
  of randomisation" -- it is a contrast against one restricted no-information randomisation device
  and can be negative (``tests/unit/test_value_definition_counterexample.py``: -0.132 while the
  randomised same-class reference is +0.018857).  It is now the diagnostic field
  ``contrast_vs_matched_uninformative_bins`` (deprecated alias ``value_net_of_randomization``) and is
  no longer a default of anything.
* F06 -- the double-count guard is re-run inside :func:`compute_information_value` on the
  structure's own signal (from ``prior_variance_basis`` or from the declared inputs recorded in a
  bound :class:`~.signal.DoubleCountReport`); unbound or foreign reports are rejected.
* Second review R2 (2026-09-25) -- prior / truth / signal: the guard also runs in *object mode*
  against the actual prior states and truth record (``truth=``; default: the generator of the prior
  states).  The object metadata of the prior (:class:`~.prior.StateMetadata`) decide the variance
  basis; a declaration that contradicts them raises :class:`~.prior.MetadataConflictError`; prior and
  linked truth must agree.  ``error_model_identification`` is ``identified_by_declared_sources`` only
  when every signal component is bound to object metadata (the "declared sources" are the object's
  metadata, not a caller string) and the verdict is ``ok``; a basis that only comes from a caller
  declaration gives ``unidentified_scenario`` (computed and labelled, never an identified value); a
  declaration can make the guard stricter (a declared double count is still refused), never looser.

Value definitions (second review R1; ``docs/value_definition.md``)
------------------------------------------------------------------
Every result carries all the differences; none of them is a default.  A caller that wants one
number must name it with :class:`ValueDefinition`:

* ``operational_deterministic_cost_difference`` = ``V0 - VT`` over deterministic policies (T7.3
  default class; executable; >= 0 under constant-policy containment; contains the randomisation
  channel of deterministic bin -> ration maps);
* ``randomized_same_class_information_reference`` = ``V0_rand - VT_rand`` over randomised policies on
  both sides (theoretical reference only -- a randomised policy is never a feeding recommendation;
  >= 0 under containment, 0 for state-independent signals);
* ``contrast_vs_matched_uninformative_bins`` = ``VT(state-independent signal, same P_z) - VT``
  (diagnostic only; any sign; never a willingness-to-pay, an H x T value or a frozen-policy value);
* ``heuristic_min_of_two_policy_values`` = ``min(operational, randomized reference)`` (third review
  R3B; formerly ``executable_information_supported_value``, kept as a deprecated alias): a HEURISTIC --
  the smaller of two values computed in two different policy classes.  It does not remove the
  randomisation channel, is not free of a policy-class convention, is not garbling-monotone (a pure
  garbling raises it from 0 to 0.018857 in ``tests/unit/test_garbling_counterexample.py``) and is
  not an established payment bound, net value or assay priority; outputs are labelled ``heuristic``
  and never enter paper main results (``docs/value_semantics_decision.md``).

``InformationValueResult.primary_value`` is ``None`` unless a definition was chosen
(``compute_information_value(..., value_definition=...)`` or ``result.with_value_definition(...)``);
``require_primary_value()`` raises instead.  None of these is a continuous global EVPI/EVPPI/EVSI or a
bound of it (T7.3).

Red-team FIX_A (2026-09-25)
---------------------------
* Randomisation channel of the operational value (:func:`randomization_channel_assessment`,
  ``InformationValueResult.randomization_channel``): a deterministic bin -> ration map can use a
  signal as a randomisation device, so a pure-noise or heavily garbled assay gets a positive
  operational value.  Executable conclusions based on the operational value (money conversions,
  assay priority, frozen policies) are refused when the whole saving is randomisation
  (``randomization_only``: the randomised same-class reference finds no decision-relevant
  information, or a state-independent device with the same bin probabilities already reaches the
  saving); when part of it is (``partly_randomization_channel``) the benchmark and its share of the
  operational value are carried with every executable output.  Nothing is truncated.
* ``definition_report`` no longer lists the matched uninformative-bins policy among the executable
  deterministic policies: it is a hypothetical randomisation device (diagnostic block).
* Identification labels (:data:`IDENTIFICATION_LABELS`) also cover perfect information:
  ``perfect_information_on_true_state`` only when every observed component is a verified
  ``true_batch_state``; ``observed_state_scenario`` when the prior SD is an observed SD (historical
  sampling/lab error valued as batch variation, contract 7.3); ``unidentified_scenario`` otherwise.
* Sample information identifies only with *verified* object metadata (factory registry, or the
  definition of a synthetic world), a numerically checked de-convolution, traced sourced error values
  and no research-scenario error values (see :mod:`.signal`, :mod:`.prior`).

Third review R3B (2026-09-25; ``docs/value_semantics_decision.md``)
-------------------------------------------------------------------
* ``executable_information_supported_value`` is renamed ``heuristic_min_of_two_policy_values`` (role
  ``heuristic``); the old enum member, string, property and assessment key still work with a
  :class:`DeprecationWarning`.  The FIX_A readings "removes randomisation", "needs no policy-class
  convention" and "established payment bound / assay priority" are withdrawn: the review's garbling
  counterexample (a pure garbling ``L' = L G`` of a signal) leaves the randomised reference at
  0.018857 and raises the minimum from 0 to 0.018857.
* ``randomization_share_of_operational`` is renamed ``matched_uninformative_contrast_ratio``: the
  algebraic ratio ``B / Δ_op`` for one specific state-independent benchmark, a diagnostic -- not an
  identified share of the saving caused by randomisation.  ``no_randomization_channel_measured``
  means "this matched benchmark detected nothing", not "no randomisation mechanism exists".
* :meth:`InformationValueResult.policy_class_comparison` shows the deterministic class and the
  randomised same-class reference side by side (cost, joint risk, actions), never combined.
* An undefined cost value (e.g. no feasible no-information *and* with-information policy) stays
  ``None`` with its reason; ``min_attainable_ex_ante_joint_risk`` reports, as a risk diagnostic within
  the library and bins, the lowest reachable ex-ante joint risk with and without the signal (never
  converted to money; not a whole-decision-space statement).

Round-3 red team FIX3_BC (2026-09-25; finding B-1)
--------------------------------------------------
``operational_not_above_randomized_reference`` (``Δ_op <= Δ_R + tol``) is no longer a permission for money:
it equals ``min(Δ_op, Δ_R) = Δ_op``, compares two policy classes and is not garbling-monotone.  Money needs
an :class:`~.economics.CompleteDecisionProblem` (VSD §2; the value is ``V0 - VT`` of its one policy class) or
an explicit, labelled development diagnostic (:mod:`.economics`).
"""

from __future__ import annotations

import dataclasses
import warnings as _warnings
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ..datamodel import SolverOptions
from ..errors import InvalidProblemError
from ..hashing import stable_hash
from ..uncertainty.streams import RandomStreams
from .binning import SignalBinning
from .library import RiskTable, decision_time_action_masks
from .likelihood import (
    SignalLikelihood,
    bin_occupancy,
    perfect_full_likelihood,
    perfect_partial_likelihood,
    sample_likelihood_analytic,
    sample_likelihood_monte_carlo,
    uninformative_likelihood,
)
from .policy import (
    RISK_TOL_DEFAULT,
    PolicyProblem,
    PolicyResult,
    solve_no_information,
    solve_no_information_randomized,
    solve_signal_policy,
    solve_signal_policy_randomized,
)
from .prior import (
    IDENTIFYING_VERIFICATION_STATUSES,
    MetadataConflictError,
    ObservedComponent,
    PriorStates,
    resolve_truth_link,
    verify_state_metadata,
)
from .signal import DoubleCountReport, SignalModel, _true_state_variance_check, double_count_guard

__all__ = [
    "INFO_TYPES",
    "SCOPE_LABEL",
    "VALUE_SEMANTICS",
    "ValueDefinition",
    "VALUE_DEFINITIONS",
    "VALUE_DEFINITION_ROLES",
    "LEGACY_VALUE_KIND_ALIASES",
    "DEPRECATED_VALUE_DEFINITION_NAMES",
    "HEURISTIC_VALUE_DEFINITIONS",
    "EXCLUDED_FROM_PAPER_MAIN_RESULTS",
    "resolve_value_definition",
    "IDENTIFICATION_LABELS",
    "IDENTIFIED_LABELS",
    "SCENARIO_IDENTIFICATION_LABELS",
    "RANDOMIZATION_CHANNEL_STATUSES",
    "EXECUTABLE_RANDOMIZATION_STATUSES",
    "DEPRECATED_ASSESSMENT_KEYS",
    "with_deprecated_assessment_keys",
    "RandomizationChannelError",
    "randomization_channel_assessment",
    "require_executable_operational_value",
    "require_operational_not_above_randomized_reference",
    "require_information_supported_operational_value",
    "InformationStructure",
    "InformationValueResult",
    "build_likelihood",
    "compute_information_value",
    "OrderCheck",
    "check_theoretical_order",
    "standard_order_relations",
    "check_library_enlargement",
    "check_bin_refinement",
]

INFO_TYPES = ("perfect_full", "perfect_partial", "sample", "uninformative")

#: ``information_value_semantics`` values.
VALUE_SEMANTICS = ("information_value_within_library",              # constant policies contained (T7.1)
                   "cost_risk_comparison_action_spaces_differ")     # T7.2: not an information value

#: Mandatory scope statement attached to every value (T7.3, T10).
SCOPE_LABEL = ("gross value within a finite candidate library and fixed development signal bins under the declared "
               "prior, ex-ante joint risk and action space; not a continuous global EVPI/EVPPI/EVSI and not a "
               "bound of it; excludes sampling, laboratory, logistics, waiting and reformulation costs")


#: Renamed definition strings (R3B): deprecated name -> current name.  Still accepted everywhere with a
#: :class:`DeprecationWarning` (``resolve_value_definition``, ``ValueDefinition(...)``).
DEPRECATED_VALUE_DEFINITION_NAMES = {
    "executable_information_supported_value": "heuristic_min_of_two_policy_values",
}
#: Renamed enum members (R3B): deprecated member name -> current member name.
_DEPRECATED_DEFINITION_MEMBERS = {
    "EXECUTABLE_INFORMATION_SUPPORTED_VALUE": "HEURISTIC_MIN_OF_TWO_POLICY_VALUES",
}


class _ValueDefinitionMeta(EnumMeta):
    """Enum metaclass that keeps renamed members readable, with a :class:`DeprecationWarning` (R3B)."""

    def __getattr__(cls, name: str):
        new = _DEPRECATED_DEFINITION_MEMBERS.get(name)
        if new is not None:
            _warnings.warn(f"ValueDefinition.{name} is deprecated; use ValueDefinition.{new} -- a heuristic, not an "
                           "executable information-supported value (docs/value_semantics_decision.md)",
                           DeprecationWarning, stacklevel=2)
            return cls[new]
        parent = getattr(super(), "__getattr__", None)
        if parent is None:
            raise AttributeError(name)
        return parent(name)


class ValueDefinition(str, Enum, metaclass=_ValueDefinitionMeta):
    """Explicit definition of the single number taken from an :class:`InformationValueResult` (R1).

    All of them live in the same declared finite model (prior states, library K, fixed bins, cost,
    risk event, ``alpha``, ``risk_tol`` and decision-time action masks); see
    ``docs/value_definition.md`` for the formulas and when each is non-negative, and
    ``docs/value_semantics_decision.md`` for what each may be used for.
    """

    #: ``V0 - VT`` over deterministic policies (contract T7.3 default class; executable).
    OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE = "operational_deterministic_cost_difference"
    #: ``V0_rand - VT_rand`` over randomised policies on both sides (theoretical reference only).
    RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE = "randomized_same_class_information_reference"
    #: ``VT(state-independent signal with the same active-bin probabilities) - VT`` (diagnostic only).
    CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS = "contrast_vs_matched_uninformative_bins"
    #: ``min(operational, randomized reference)`` -- HEURISTIC (R3B; formerly
    #: ``executable_information_supported_value``): the smaller of two values from two different policy
    #: classes; not garbling-monotone; not a payment bound, net value or assay priority; never a paper
    #: main result.  The deprecated member name ``EXECUTABLE_INFORMATION_SUPPORTED_VALUE`` still resolves here.
    HEURISTIC_MIN_OF_TWO_POLICY_VALUES = "heuristic_min_of_two_policy_values"

    @classmethod
    def _missing_(cls, value: Any) -> Optional["ValueDefinition"]:
        new = DEPRECATED_VALUE_DEFINITION_NAMES.get(value) if isinstance(value, str) else None
        if new is None:
            return None
        _warnings.warn(f"value definition {value!r} is deprecated; use {new!r} -- a heuristic, not an executable "
                       "information-supported value (docs/value_semantics_decision.md)", DeprecationWarning,
                       stacklevel=2)
        return cls(new)


VALUE_DEFINITIONS = tuple(d.value for d in ValueDefinition)

#: Role of each definition: what the number may be used for.
VALUE_DEFINITION_ROLES = {
    ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value: "operational",
    ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE.value: "theoretical_reference",
    ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS.value: "diagnostic",
    ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value: "heuristic",
}

#: Heuristic definitions (R3B): usable only when named explicitly; every output says ``heuristic``.
HEURISTIC_VALUE_DEFINITIONS = (ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value,)

#: Definitions whose numbers never enter paper main results (R3B; ``docs/value_semantics_decision.md``).
#: ``False`` for the other definitions only means "not excluded by the definition itself"; the study-level
#: decision (RQ3 as an exploratory appendix, money only from a complete decision problem) still applies.
EXCLUDED_FROM_PAPER_MAIN_RESULTS = (ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS.value,
                                    ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value)

#: One-line interpretation attached to every reported number.
VALUE_DEFINITION_NOTES = {
    ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value:
        "cost difference of executable deterministic policies (constant vs bin -> ration map) at the same ex-ante "
        "joint risk; >= 0 under constant-policy containment; includes the randomisation channel of deterministic "
        "maps, so it is not a pure information value",
    ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE.value:
        "theoretical reference: randomised policies on both sides (LP relaxations); >= 0 under containment and 0 for "
        "state-independent signals; the randomised policies are never feeding recommendations",
    ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS.value:
        "diagnostic contrast against a state-independent signal with the same active-bin probabilities; any sign; "
        "not an information value, not a willingness to pay, not an assay priority",
    ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value:
        "HEURISTIC: min(operational deterministic difference, randomised same-class reference) -- the smaller of two "
        "values computed in two different policy classes; 0 for state-independent signals and >= 0 under "
        "containment, but it does not remove the randomisation channel, is not free of a policy-class convention and "
        "is not garbling-monotone (a pure garbling raised it from 0 to 0.018857 in the review counterexample); not "
        "a payment bound, net value or assay priority; never a paper main result (R3B, "
        "docs/value_semantics_decision.md)",
}

#: Deprecated ``value_kind`` strings (engine before the second review) -> definition.
LEGACY_VALUE_KIND_ALIASES = {
    "deterministic": ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE,
    "randomized_reference": ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE,
    "net_of_randomization": ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS,
}
_LEGACY_OF = {v.value: k for k, v in LEGACY_VALUE_KIND_ALIASES.items()}

DefinitionLike = Union[ValueDefinition, str]


def resolve_value_definition(value_definition: Optional[DefinitionLike] = None, *,
                             legacy_value_kind: Optional[str] = None, context: str = "value",
                             allow_legacy: bool = True) -> ValueDefinition:
    """Explicit :class:`ValueDefinition` or :class:`InvalidProblemError` -- there is no default (R1).

    ``value_definition`` accepts a member or its canonical string.  ``legacy_value_kind`` (and, if
    ``allow_legacy``, a legacy string passed as ``value_definition``) maps the deprecated names
    ``deterministic`` / ``randomized_reference`` / ``net_of_randomization`` with a
    :class:`DeprecationWarning`; both given and different raises.  A renamed definition string
    (:data:`DEPRECATED_VALUE_DEFINITION_NAMES`, R3B: ``executable_information_supported_value``) is
    accepted in every context with a :class:`DeprecationWarning` and resolves to its current name.
    """
    legacy = None
    if legacy_value_kind is not None:
        if legacy_value_kind not in LEGACY_VALUE_KIND_ALIASES:
            raise InvalidProblemError(f"{context}: unknown value_kind {legacy_value_kind!r}; use value_definition= one "
                                      f"of {VALUE_DEFINITIONS}")
        legacy = LEGACY_VALUE_KIND_ALIASES[legacy_value_kind]
        _warnings.warn(f"value_kind={legacy_value_kind!r} is deprecated; use value_definition={legacy.value!r} "
                       "(docs/value_definition.md)", DeprecationWarning, stacklevel=3)
    if value_definition is None:
        if legacy is None:
            raise InvalidProblemError(
                f"{context}: an explicit value_definition is required (one of {VALUE_DEFINITIONS}); no difference "
                "is used by default (second review R1, docs/value_definition.md)")
        return legacy
    if isinstance(value_definition, ValueDefinition):
        out = value_definition
    elif isinstance(value_definition, str) and value_definition in VALUE_DEFINITIONS:
        out = ValueDefinition(value_definition)
    elif isinstance(value_definition, str) and value_definition in DEPRECATED_VALUE_DEFINITION_NAMES:
        out = ValueDefinition(DEPRECATED_VALUE_DEFINITION_NAMES[value_definition])
        _warnings.warn(f"value definition {value_definition!r} is deprecated; use {out.value!r} -- a heuristic, not an "
                       "executable information-supported value (docs/value_semantics_decision.md)",
                       DeprecationWarning, stacklevel=3)
    elif allow_legacy and isinstance(value_definition, str) and value_definition in LEGACY_VALUE_KIND_ALIASES:
        out = LEGACY_VALUE_KIND_ALIASES[value_definition]
        _warnings.warn(f"{value_definition!r} is a deprecated value kind; use {out.value!r} (docs/value_definition.md)",
                       DeprecationWarning, stacklevel=3)
    else:
        raise InvalidProblemError(f"{context}: unknown value_definition {value_definition!r}; expected one of "
                                  f"{VALUE_DEFINITIONS}")
    if legacy is not None and legacy is not out:
        raise InvalidProblemError(f"{context}: value_definition={out.value!r} and value_kind={legacy_value_kind!r} "
                                  "disagree")
    return out


# ---------------------------------------------------------------------------------------------
# identification labels and the randomisation channel of the operational value (FIX_A)
# ---------------------------------------------------------------------------------------------

#: ``error_model_identification`` values.
IDENTIFICATION_LABELS = (
    "identified_by_declared_sources",      # sample information; every basis from verified object metadata
    "perfect_information_on_true_state",   # perfect information; every observed component a verified true state
    "observed_state_scenario",             # perfect information on an observed-basis SD (contract 7.3): scenario
    "unidentified_scenario",               # computed and labelled; never an identified value
    "error_model_unsourced",               # the signal's error values lack provenance / cannot be traced
    "not_applicable",                      # uninformative structure (no information about the state)
)
#: Labels a money conversion may use without an explicit ``scenario=True`` (economics).
IDENTIFIED_LABELS = ("identified_by_declared_sources", "perfect_information_on_true_state", "not_applicable")
#: Labels that need an explicit scenario declaration before any money conversion (and keep the label).
SCENARIO_IDENTIFICATION_LABELS = ("unidentified_scenario", "observed_state_scenario", "error_model_unsourced")

#: ``randomization_channel_assessment(...)["status"]``.  The statuses describe one matched state-independent
#: benchmark and the randomised same-class reference; they do not identify the causal source of a saving (R3B).
RANDOMIZATION_CHANNEL_STATUSES = (
    "no_operational_saving",               # operational value <= tolerance: nothing is bought on this basis
    "no_randomization_channel_measured",   # the matched benchmark detects no gain (benchmark <= tol); this benchmark
                                           # did not detect a channel -- NOT "no randomisation mechanism exists"
    "partly_randomization_channel",        # the matched benchmark reaches part of it (diagnostic ratio B/op given);
                                           # the rest is not reached by this benchmark, which does not make it information
    "randomization_only",                  # the whole operational saving is reachable without information
    "not_assessable",                      # references not computed, or not an information value
)
#: Statuses under which the operational (deterministic) value is not refused as a randomisation channel only
#: (a frozen deterministic policy may then be fed; money and ranking have their own, stricter rules).
EXECUTABLE_RANDOMIZATION_STATUSES = ("no_operational_saving", "no_randomization_channel_measured",
                                     "partly_randomization_channel")

#: Renamed keys of the assessment (R3B): deprecated key -> current key.  Reading a deprecated key from the
#: mapping returned by :func:`randomization_channel_assessment` still works, with a :class:`DeprecationWarning`.
DEPRECATED_ASSESSMENT_KEYS = {
    "randomization_share_of_operational": "matched_uninformative_contrast_ratio",
    "executable_information_supported_value": "heuristic_min_of_two_policy_values",
    "operational_information_supported": "operational_not_above_randomized_reference",
}


class _AssessmentDict(dict):
    """``dict`` whose deprecated keys (:data:`DEPRECATED_ASSESSMENT_KEYS`) stay readable with a warning (R3B).

    Only the current keys are stored, so ``dict(x)``, iteration and JSON output carry the current names.
    """

    def _renamed(self, key: Any) -> Optional[str]:
        new = DEPRECATED_ASSESSMENT_KEYS.get(key) if isinstance(key, str) else None
        if new is not None:
            _warnings.warn(f"randomization-channel key {key!r} is deprecated; use {new!r} "
                           "(docs/value_semantics_decision.md)", DeprecationWarning, stacklevel=3)
        return new

    def __missing__(self, key: Any) -> Any:
        new = self._renamed(key)
        if new is None:
            raise KeyError(key)
        return self[new]

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        new = self._renamed(key)
        return default if new is None else dict.get(self, new, default)


def with_deprecated_assessment_keys(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of an assessment mapping that keeps the deprecated-key aliases readable (R3B)."""
    return _AssessmentDict(assessment)


class RandomizationChannelError(InvalidProblemError):
    """The operational (deterministic) value is entirely a randomisation channel (or cannot be checked),
    or -- on the development-diagnostic money path -- exceeds the randomised same-class reference (FIX_A, R3B,
    FIX3_BC)."""


def randomization_channel_assessment(res: "InformationValueResult") -> dict[str, Any]:
    """How much of the operational value a state-independent randomisation device already reaches (FIX_A).

    Uses only numbers the result already carries -- operational ``Δ_op = V0 - VT``, the randomised
    same-class reference ``Δ_R``, the matched-device benchmark ``B = V0 - VT(U_P)`` and the contrast
    ``Δ_C = VT(U_P) - VT = Δ_op - B`` -- with the cost tolerance ``cost_tol * max(1, |V0|)``:

    * ``no_operational_saving``: ``Δ_op <= tol`` (nothing to buy on the operational basis; the value is
      reported as is, also when negative in floating point);
    * ``randomization_only``: ``Δ_op > tol`` and either ``Δ_R <= tol`` (no decision-relevant
      information in the same policy class -- theorem: 0 for state-independent signals) or
      ``Δ_C <= tol`` (a state-independent device with the same bin probabilities reaches the whole
      saving: the signal's bins do not allow any cheaper deterministic assignment than a coin);
      uninformative structures are this case by definition;
    * ``no_randomization_channel_measured``: ``B <= tol`` -- this matched benchmark detects no gain.  It
      does **not** show that no randomisation mechanism exists: a pure garbling of an informative signal
      can create a randomisation channel that the matched benchmark misses (R3B garbling counterexample:
      ``B = 0`` while ``Δ_op = 0.150857 > Δ_R = 0.018857``);
    * ``partly_randomization_channel``: otherwise; ``matched_uninformative_contrast_ratio = B / Δ_op`` is
      reported -- the algebraic ratio for this one benchmark, a diagnostic, not an identified share of the
      saving caused by randomisation.  The part not reached by this benchmark is not thereby identified
      as information, and nothing is subtracted;
    * ``not_assessable``: not an information value, or ``Δ_R`` / ``B`` not computed
      (``include_randomized_reference`` / ``include_randomization_benchmark`` false).

    Independently of the status, ``heuristic_min_of_two_policy_values = min(Δ_op, Δ_R)`` (R3B: a
    heuristic, not garbling-monotone) and ``operational_not_above_randomized_reference`` (``Δ_op <= tol``
    or ``Δ_op <= Δ_R + tol``; FIX3_BC: only the refusal check of the labelled development-diagnostic money
    path -- never a permission to convert the operational value to money) are reported.  Deprecated keys (:data:`DEPRECATED_ASSESSMENT_KEYS`) remain readable with a warning.
    """
    op, rr = res.gross_value, res.gross_value_randomized_reference
    b, c = res.randomization_benchmark_value, res.contrast_vs_matched_uninformative_bins
    cost_tol = float(res.provenance.get("cost_tol", 1e-9)) if isinstance(res.provenance, Mapping) else 1e-9
    tol = cost_tol * max(1.0, abs(res.V0.expected_cost)) if res.V0.has_solution else cost_tol
    out: dict[str, Any] = _AssessmentDict({
        "operational_deterministic_cost_difference": op,
        "randomized_same_class_information_reference": rr,
        "randomization_benchmark_value": b, "contrast_vs_matched_uninformative_bins": c,
        "tolerance": tol, "matched_uninformative_contrast_ratio": None,
        "matched_uninformative_contrast_ratio_note": (
            "B / operational for one specific state-independent benchmark with the same active-bin probabilities: "
            "an algebraic ratio (diagnostic); not an identified share of the saving caused by randomisation and not "
            "a decomposition of the saving by source (R3B)")})
    if op is not None and op > tol and b is not None:
        out["matched_uninformative_contrast_ratio"] = float(b / op)
    out["heuristic_min_of_two_policy_values"] = res.heuristic_min_of_two_policy_values
    out["operational_excess_over_randomized_reference"] = (None if op is None or rr is None else float(op - rr))
    out["operational_not_above_randomized_reference"] = (None if op is None else
                                                          bool(op <= tol or (rr is not None and op <= rr + tol)))
    if not res.is_information_value or op is None:
        status, reason = "not_assessable", ("not an information value (" + res.information_value_semantics + ")"
                                            if not res.is_information_value else
                                            f"operational value undefined ({res.gross_value_status})")
    elif op <= tol:
        status, reason = "no_operational_saving", "operational value <= tolerance: no executable saving to buy"
    elif res.info_type == "uninformative":
        status, reason = "randomization_only", "the structure is a state-independent device (no information)"
    elif rr is None or b is None or c is None:
        status, reason = "not_assessable", ("randomised same-class reference or matched-device benchmark not computed "
                                            "(compute with include_randomized_reference=True and "
                                            "include_randomization_benchmark=True)")
    elif rr <= tol:
        status, reason = "randomization_only", (f"the randomised same-class reference is {rr:.6g} <= {tol:.3g}: no "
                                                "decision-relevant information; the operational saving "
                                                f"{op:.6g} is a randomisation channel of the deterministic bin map")
    elif c <= tol:
        status, reason = "randomization_only", (f"a state-independent device with the same bin probabilities reaches "
                                                f"{b:.6g} >= the operational saving {op:.6g} (contrast {c:.6g}): the "
                                                "signal allows no cheaper deterministic assignment than a coin")
    elif b <= tol:
        status, reason = "no_randomization_channel_measured", (
            f"the matched state-independent benchmark (same active-bin probabilities) detects no gain (benchmark "
            f"{b:.6g}): this benchmark did not detect a randomisation channel; that does not show that no "
            "randomisation mechanism exists (a garbling can create one the matched benchmark misses, R3B)")
    else:
        status, reason = "partly_randomization_channel", (
            f"the matched state-independent benchmark reaches {b:.6g} of the operational saving {op:.6g} "
            f"(matched_uninformative_contrast_ratio {b / op:.3g}: an algebraic ratio for this one benchmark, not an "
            "identified randomisation share); the rest is not reached by this benchmark, which does not identify it "
            "as information; nothing is subtracted -- report the ratio as a diagnostic next to every executable "
            "output")
    out.update({"status": status, "reason": reason,
                "executable_use_allowed": status in EXECUTABLE_RANDOMIZATION_STATUSES,
                "status_scope": ("statuses describe one matched state-independent benchmark and the randomised "
                                 "same-class reference; they do not identify the causal source of a saving (R3B)"),
                "rule": "FIX_A: randomization_only if op > tol and (randomized reference <= tol or contrast <= tol); "
                        "the other statuses describe the matched benchmark only (docs/value_definition.md §2.5)"})
    return out


def require_executable_operational_value(res: "InformationValueResult", context: str) -> dict[str, Any]:
    """Assessment of ``res``; :class:`RandomizationChannelError` unless the operational value may support an
    executable conclusion (``randomization_only`` or, with a positive operational value, ``not_assessable``)."""
    a = randomization_channel_assessment(res)
    if a["status"] == "randomization_only":
        raise RandomizationChannelError(
            f"{context}: {res.structure_id}: the operational value is a randomisation channel only ({a['reason']}); it "
            "cannot justify buying, ranking or freezing this assay (randomised feeding is never recommended; "
            "docs/value_definition.md §8.2)")
    if a["status"] == "not_assessable" and res.gross_value is not None and res.gross_value > a["tolerance"]:
        raise RandomizationChannelError(
            f"{context}: {res.structure_id}: the randomisation channel of the positive operational value cannot be "
            f"assessed ({a['reason']})")
    return a


def require_operational_not_above_randomized_reference(res: "InformationValueResult", context: str
                                                       ) -> dict[str, Any]:
    """Assessment of ``res``; :class:`RandomizationChannelError` unless the operational value does not
    exceed the randomised same-class reference (``Δ_op <= tol`` or ``Δ_op <= Δ_R + tol``) -- the FIX_A rule, kept
    only as a refusal on the development-diagnostic money path (FIX3_BC: passing it permits nothing).

    R3B: the refusal is not followed by a recommended substitute.  ``heuristic_min_of_two_policy_values``
    is a heuristic of two policy classes (not garbling-monotone) and can be converted only when named
    explicitly (outputs labelled ``heuristic``, never paper main results); an economic value needs a
    complete decision problem (``docs/value_semantics_decision.md``).

    Round-3 red team FIX3_BC (B-1): this is a **refusal only**, never a permission.  ``Δ_op <= Δ_R + tol`` is
    ``min(Δ_op, Δ_R) = Δ_op`` -- a comparison of two policy classes that is not garbling-monotone (a pure
    garbling can raise ``Δ_op`` from 0 to 0.2328 while passing it).  It is used only on the labelled
    development-diagnostic path of the money conversions (``diagnostic_without_decision_problem=True``);
    passing it releases nothing.  With a :class:`~.economics.CompleteDecisionProblem` it is not applied: the
    money value is ``V0 - VT`` of that problem's one policy class."""
    a = require_executable_operational_value(res, context)
    if a["operational_not_above_randomized_reference"] is False:
        h = a["heuristic_min_of_two_policy_values"]
        raise RandomizationChannelError(
            f"{context}: {res.structure_id}: the operational (deterministic) value {res.gross_value:.6g} exceeds the "
            f"randomised same-class reference {res.gross_value_randomized_reference:.6g}; the development-diagnostic "
            "money path refuses an operational value above the randomised reference (the excess cannot be split into "
            f"information and randomisation within this engine; matched-benchmark status {a['status']}, "
            f"matched_uninformative_contrast_ratio {a['matched_uninformative_contrast_ratio']}). "
            f"'{ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES.value}' (formerly "
            f"'executable_information_supported_value'; = min(operational, randomised reference) = {h!r}) is not a "
            "substitute basis: it is a heuristic, not garbling-monotone, convertible only when named explicitly "
            "(labelled heuristic, never a paper main result); a money value needs a complete decision problem "
            "(economics.CompleteDecisionProblem; docs/value_semantics_decision.md §2). Passing this check is not a "
            "permission either (FIX3_BC, round-3 red team B-1)")
    return a


def require_information_supported_operational_value(res: "InformationValueResult", context: str) -> dict[str, Any]:
    """DEPRECATED name of :func:`require_operational_not_above_randomized_reference` (R3B): the condition
    ``Δ_op <= Δ_R`` compares two policy classes; it does not show that information "supports" a saving."""
    _warnings.warn("require_information_supported_operational_value is deprecated; use "
                   "require_operational_not_above_randomized_reference (docs/value_semantics_decision.md)",
                   DeprecationWarning, stacklevel=2)
    return require_operational_not_above_randomized_reference(res, context)


@dataclass(frozen=True)
class InformationStructure:
    """What is observed at t2 before the ration is chosen at t3 (T1).

    * ``perfect_full``: no further fields.
    * ``perfect_partial``: ``components``; ``binning`` (``None`` = exact values, discrete priors only).
    * ``sample``: ``signal_model`` and ``binning`` (components = signal components);
      ``likelihood_method`` ``analytic_gaussian`` or ``monte_carlo`` (``mc_replicates``).
    * ``uninformative``: ``bin_probabilities``.
    """

    structure_id: str
    info_type: str
    components: tuple[ObservedComponent, ...] = ()
    binning: Optional[SignalBinning] = None
    signal_model: Optional[SignalModel] = None
    likelihood_method: str = "analytic_gaussian"
    mc_replicates: int = 0
    bin_probabilities: tuple[float, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if self.info_type not in INFO_TYPES:
            raise InvalidProblemError(f"info_type must be one of {INFO_TYPES}")
        object.__setattr__(self, "components", tuple(self.components))
        if self.info_type == "perfect_partial" and not self.components:
            raise InvalidProblemError("perfect_partial needs components")
        if self.info_type == "sample":
            if self.signal_model is None or self.binning is None:
                raise InvalidProblemError("sample information needs a signal_model and a binning")
            if self.components and tuple(self.components) != self.signal_model.components:
                raise InvalidProblemError("sample: components must equal the signal model components")
            object.__setattr__(self, "components", self.signal_model.components)
            if self.likelihood_method not in ("analytic_gaussian", "monte_carlo"):
                raise InvalidProblemError("likelihood_method must be analytic_gaussian or monte_carlo")
        if self.info_type == "uninformative" and not self.bin_probabilities:
            raise InvalidProblemError("uninformative needs bin_probabilities")


def build_likelihood(structure: InformationStructure, prior: PriorStates,
                     streams: Optional[RandomStreams] = None) -> SignalLikelihood:
    """Likelihood matrix of a structure on the given (development) prior states."""
    t = structure.info_type
    if t == "perfect_full":
        return perfect_full_likelihood(prior)
    if t == "perfect_partial":
        return perfect_partial_likelihood(prior, structure.components, structure.binning)
    if t == "uninformative":
        return uninformative_likelihood(prior, structure.bin_probabilities)
    if structure.likelihood_method == "analytic_gaussian":
        return sample_likelihood_analytic(prior, structure.signal_model, structure.binning)
    if streams is None:
        raise InvalidProblemError("monte_carlo likelihood needs RandomStreams")
    return sample_likelihood_monte_carlo(prior, structure.signal_model, structure.binning, streams,
                                         structure.mc_replicates)


def _policy_block(p: Optional[PolicyResult], candidate_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Cost, ex-ante joint risk and feasibility of one policy (reported next to every value, R1)."""
    if p is None:
        return {"computed": False}
    out: dict[str, Any] = {
        "computed": True, "method": p.method, "status": p.status, "has_solution": p.has_solution,
        "expected_cost": p.expected_cost, "ex_ante_joint_risk": p.ex_ante_risk, "alpha": p.alpha,
        "risk_tol": p.risk_tol,
        "meets_risk_target": None if p.ex_ante_risk is None else bool(p.ex_ante_risk <= p.alpha + p.risk_tol)}
    if p.assignment is not None:
        a = [int(k) for k in np.asarray(p.assignment, dtype=int)]
        out["assignment"] = a
        if candidate_ids:
            out["assignment_candidate_ids"] = [candidate_ids[k] if 0 <= k < len(candidate_ids) else None for k in a]
    if p.mixture is not None:
        out["mixture"] = np.asarray(p.mixture, dtype=float).tolist()
    if p.w is not None:
        out["w"] = np.asarray(p.w, dtype=float).tolist()
    return out


@dataclass(frozen=True, repr=False)
class InformationValueResult:
    """Values of one information structure under every definition; no default definition (R1).

    * ``gross_value`` -- operational deterministic cost difference ``V0 - VT`` (``None`` if the action
      spaces are not nested);
    * ``gross_value_randomized_reference`` -- randomised same-class reference ``V0_rand - VT_rand``;
    * ``randomization_benchmark_value`` -- ``V0 - VT(U)`` with ``U`` a state-independent signal with
      the same active-bin probabilities (``VT_uninformative_benchmark`` holds that policy);
    * ``contrast_vs_matched_uninformative_bins`` -- ``VT(U) - VT`` (DIAGNOSTIC; any sign; deprecated
      alias ``value_net_of_randomization``).  ``gross_value = benchmark + contrast`` (identity).
    * ``heuristic_min_of_two_policy_values`` (property) -- ``min(gross_value, randomised reference)``
      (HEURISTIC, R3B; deprecated alias ``executable_information_supported_value``).

    ``value_definition`` is ``None`` unless chosen; ``primary_value`` is then ``None`` and
    :meth:`require_primary_value` raises.  :meth:`definition_report` always pairs a value with the cost,
    ex-ante joint risk and feasibility of the deterministic policies; :meth:`policy_class_comparison`
    shows the deterministic class and the randomised reference side by side (R3B).  An undefined value
    is ``None`` with a reason, never 0; ``min_attainable_ex_ante_joint_risk`` is a risk diagnostic within
    the library and bins (never money).
    """

    structure_id: str
    info_type: str
    likelihood_kind: str
    alpha: float
    unknown_policy: str
    V0: PolicyResult
    VT: PolicyResult
    gross_value: Optional[float]
    gross_value_status: str
    V0_randomized: Optional[PolicyResult]
    VT_randomized: Optional[PolicyResult]
    gross_value_randomized_reference: Optional[float]
    randomization_benchmark_value: Optional[float]
    containment_check: Mapping[str, Any]
    occupancy: Mapping[str, Any]
    error_model_identification: str
    warnings: tuple[str, ...]
    n_states: int
    n_bins: int
    n_candidates: int
    provenance: Mapping[str, Any]
    is_synthetic: bool
    information_value_semantics: str = "information_value_within_library"
    cost_difference_V0_minus_VT: Optional[float] = None
    cost_difference_randomized_reference: Optional[float] = None
    #: DIAGNOSTIC (R1): ``VT(state-independent signal, same active-bin P_z) - VT``; any sign; never a default.
    contrast_vs_matched_uninformative_bins: Optional[float] = None
    action_space: Mapping[str, Any] = field(default_factory=dict)
    error_model_report: Mapping[str, Any] = field(default_factory=dict)
    scope_label: str = SCOPE_LABEL
    value_unit: str = "currency/head/d"
    #: deterministic policy of the matched state-independent signal (``None`` if not computed)
    VT_uninformative_benchmark: Optional[PolicyResult] = None
    candidate_ids: tuple[str, ...] = ()
    #: explicitly chosen definition (``None`` = none chosen; nothing is chosen by default)
    value_definition: Optional[str] = None
    #: R3B risk diagnostic: lowest ex-ante joint risk reachable within the library and bins, without and with
    #: the signal (same for deterministic and randomised policies); never converted to money
    min_attainable_ex_ante_joint_risk: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        if self.value_definition is not None:
            d = resolve_value_definition(self.value_definition, context="InformationValueResult",
                                         allow_legacy=False)
            object.__setattr__(self, "value_definition", d.value)

    # ------------------------------------------------------------------ named values
    @property
    def operational_deterministic_cost_difference(self) -> Optional[float]:
        """``V0 - VT`` over deterministic policies (= ``gross_value``)."""
        return self.gross_value

    @property
    def randomized_same_class_information_reference(self) -> Optional[float]:
        """``V0_rand - VT_rand`` (= ``gross_value_randomized_reference``); theoretical reference only."""
        return self.gross_value_randomized_reference

    @property
    def heuristic_min_of_two_policy_values(self) -> Optional[float]:
        """HEURISTIC ``min(operational, randomised reference)`` (R3B); ``None`` if either is undefined.

        The smaller of two values from two different policy classes: not garbling-monotone, not free of
        a policy-class convention, not a payment bound or assay priority, never a paper main result.
        """
        a, b = self.gross_value, self.gross_value_randomized_reference
        if a is None or b is None:
            return None
        return float(min(a, b))

    @property
    def executable_information_supported_value(self) -> Optional[float]:
        """DEPRECATED alias of :attr:`heuristic_min_of_two_policy_values` (R3B)."""
        _warnings.warn("executable_information_supported_value is a deprecated alias of the heuristic "
                       "heuristic_min_of_two_policy_values; it is not an executable information-supported value "
                       "(docs/value_semantics_decision.md)", DeprecationWarning, stacklevel=2)
        return self.heuristic_min_of_two_policy_values

    @property
    def value_net_of_randomization(self) -> Optional[float]:
        """DEPRECATED alias of the diagnostic :attr:`contrast_vs_matched_uninformative_bins` (R1).

        The old name suggested a pure information value net of randomisation; it is not one (it can
        be negative while the randomised same-class reference is positive).
        """
        _warnings.warn("value_net_of_randomization is a deprecated alias of the diagnostic "
                       "contrast_vs_matched_uninformative_bins; it is not an information value "
                       "(docs/value_definition.md)", DeprecationWarning, stacklevel=2)
        return self.contrast_vs_matched_uninformative_bins

    # ------------------------------------------------------------------ primary value (explicit only)
    @property
    def primary_value(self) -> Optional[float]:
        """Value under the explicitly chosen :attr:`value_definition`; ``None`` if none was chosen.

        There is no engine default (second review R1): ``None`` here means "no definition chosen" or
        "not defined under the chosen definition" -- see :attr:`primary_value_status`.  Use
        :meth:`require_primary_value` to fail loudly instead.
        """
        if self.value_definition is None:
            return None
        return self.value(self.value_definition)

    @property
    def primary_value_status(self) -> str:
        """``no_value_definition_selected`` / ``ok`` / ``undefined_under_definition:<reason>``."""
        if self.value_definition is None:
            return "no_value_definition_selected"
        if self.value(self.value_definition) is None:
            return f"undefined_under_definition:{self._undefined_reason(self.value_definition)}"
        return "ok"

    @property
    def primary_value_role(self) -> Optional[str]:
        """``operational`` / ``theoretical_reference`` / ``diagnostic`` / ``heuristic`` (``None`` if no
        definition)."""
        return None if self.value_definition is None else VALUE_DEFINITION_ROLES[self.value_definition]

    def require_primary_value(self) -> float:
        """Primary value or :class:`InvalidProblemError` (no definition chosen, or value undefined)."""
        if self.value_definition is None:
            raise InvalidProblemError(
                f"{self.structure_id}: no value_definition was chosen, so there is no primary value; pass "
                f"value_definition= one of {VALUE_DEFINITIONS} (docs/value_definition.md)")
        v = self.value(self.value_definition)
        if v is None:
            raise InvalidProblemError(f"{self.structure_id}: {self.value_definition} is undefined here ("
                                      f"{self._undefined_reason(self.value_definition)})")
        return float(v)

    def with_value_definition(self, value_definition: DefinitionLike) -> "InformationValueResult":
        """Copy of this result with an explicitly chosen definition (all diagnostics kept)."""
        d = resolve_value_definition(value_definition, context="with_value_definition", allow_legacy=False)
        self._check_definition_available(d)
        return dataclasses.replace(self, value_definition=d.value)

    def _check_definition_available(self, d: ValueDefinition) -> None:
        prov = self.provenance
        if d is ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE and \
                prov.get("include_randomized_reference") is False:
            raise InvalidProblemError("randomized_same_class_information_reference needs "
                                      "include_randomized_reference=True")
        if d is ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS and self.info_type != "uninformative" and \
                prov.get("include_randomization_benchmark") is False:
            raise InvalidProblemError("contrast_vs_matched_uninformative_bins needs include_randomization_benchmark=True")
        if d is ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES and \
                prov.get("include_randomized_reference") is False:
            raise InvalidProblemError("heuristic_min_of_two_policy_values needs include_randomized_reference=True")

    def _undefined_reason(self, definition: DefinitionLike) -> str:
        d = resolve_value_definition(definition, context="value", allow_legacy=False)
        if not self.is_information_value:
            return self.information_value_semantics
        if d is ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE:
            return self.gross_value_status
        if d is ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE or \
                (d is ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES and self.gross_value is not None):
            if self.V0_randomized is None:
                return "randomized_reference_not_computed"
            return f"V0_rand:{self.V0_randomized.status}/VT_rand:" + \
                ("none" if self.VT_randomized is None else self.VT_randomized.status)
        if d is ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES:
            return self.gross_value_status
        if self.VT_uninformative_benchmark is None:
            return "uninformative_benchmark_not_computed_or_V0_infeasible"
        return f"VT:{self.VT.status}/VT_uninformative:{self.VT_uninformative_benchmark.status}"

    @property
    def is_information_value(self) -> bool:
        """True if ``V0`` and ``VT`` share the action space premise of T7.1 (constant policies contained)."""
        return self.information_value_semantics == "information_value_within_library"

    @property
    def randomization_channel(self) -> dict[str, Any]:
        """:func:`randomization_channel_assessment` of this result (FIX_A)."""
        return randomization_channel_assessment(self)

    @property
    def fingerprint(self) -> str:
        """Content hash of what this result says (model fingerprints, values, statuses, identification).

        The explicitly chosen ``value_definition`` is not part of it (choosing a definition does not
        change the result); money conversions record it separately (FIX_A)."""
        def pol(p: Optional[PolicyResult]):
            if p is None:
                return None
            return (p.method, p.status, p.expected_cost, p.ex_ante_risk,
                    None if p.assignment is None else tuple(int(k) for k in np.asarray(p.assignment).ravel()))
        prov = self.provenance if isinstance(self.provenance, Mapping) else {}
        keys = ("prior_fingerprint", "risk_table_fingerprint", "library_fingerprint", "policy_problem_fingerprint",
                "signal_fingerprint", "binning_fingerprint", "action_mask_fingerprint", "bin_action_mask_fingerprint")
        return stable_hash("InformationValueResult/v1", self.structure_id, self.info_type, self.likelihood_kind,
                           float(self.alpha), self.unknown_policy, pol(self.V0), pol(self.VT), pol(self.V0_randomized),
                           pol(self.VT_randomized), pol(self.VT_uninformative_benchmark), self.gross_value,
                           self.gross_value_status, self.gross_value_randomized_reference,
                           self.randomization_benchmark_value, self.contrast_vs_matched_uninformative_bins,
                           self.information_value_semantics, self.error_model_identification, bool(self.is_synthetic),
                           tuple((k, None if prov.get(k) is None else str(prov.get(k))) for k in keys))

    def value(self, kind: DefinitionLike) -> Optional[float]:
        """Value under one definition (:class:`ValueDefinition` or its string); ``None`` if not defined
        (never 0 as a placeholder).  The deprecated names ``deterministic`` / ``randomized_reference`` /
        ``net_of_randomization`` are still accepted with a :class:`DeprecationWarning`."""
        d = resolve_value_definition(kind, context="InformationValueResult.value")
        if d is ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE:
            return self.gross_value
        if d is ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE:
            return self.gross_value_randomized_reference
        if d is ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES:
            return self.heuristic_min_of_two_policy_values
        return self.contrast_vs_matched_uninformative_bins

    def values_by_definition(self) -> dict[str, Optional[float]]:
        """All values keyed by (current) definition name (no selection)."""
        return {d.value: self.value(d) for d in ValueDefinition}

    def policy_class_comparison(self) -> dict[str, Any]:
        """Deterministic policy class and randomised same-class reference side by side, never combined (R3B).

        Each class reports its own no-information and with-information cost, ex-ante joint risk and
        actions (deterministic assignment / randomised mixture and bin weights) and its own ``V0 - VT``.
        No number is formed across the two classes here (no minimum, maximum or difference).  An
        undefined difference is ``None`` with its reason -- never 0 (e.g. no feasible no-information and
        with-information policy under H0).  The minimum attainable ex-ante joint risk within the library
        and bins is added as a risk diagnostic; it is never converted to money.
        """
        cids = self.candidate_ids

        def block(p0: Optional[PolicyResult], pt: Optional[PolicyResult], diff: Optional[float],
                  definition: ValueDefinition, policy_class: str, note: str) -> dict[str, Any]:
            return {"policy_class": policy_class, "note": note,
                    "no_information_policy": _policy_block(p0, cids),
                    "with_information_policy": _policy_block(pt, cids),
                    "cost_difference_V0_minus_VT": diff,
                    "cost_difference_status": "ok" if diff is not None else
                    f"undefined:{self._undefined_reason(definition)}",
                    "value_definition": definition.value, "role": VALUE_DEFINITION_ROLES[definition.value],
                    "unit": self.value_unit}

        out = {
            "structure_id": self.structure_id, "alpha": self.alpha, "is_synthetic": self.is_synthetic,
            "information_value_semantics": self.information_value_semantics,
            "deterministic_policy_class": block(
                self.V0, self.VT, self.gross_value, ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE,
                "deterministic (contract T7.3 default class; one ration per bin; executable)",
                "cost difference of executable deterministic policies at the same ex-ante joint risk; a deterministic "
                "bin -> ration map cannot randomise by itself, so the signal's bin frequencies can act as a "
                "randomisation device -- this number is not a pure information value"),
            "randomized_same_class_reference": block(
                self.V0_randomized, self.VT_randomized, self.gross_value_randomized_reference,
                ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE,
                "randomised on both sides (LP relaxations; theoretical reference)",
                "randomised policies are never feeding recommendations; 0 for state-independent signals and "
                "garbling-monotone within this finite model"),
            "min_attainable_ex_ante_joint_risk": dict(self.min_attainable_ex_ante_joint_risk),
            "not_combined": ("the two classes are reported separately; no minimum, maximum or difference across the "
                             "classes is formed here (heuristic_min_of_two_policy_values is a separate, explicitly "
                             "named heuristic; docs/value_semantics_decision.md)"),
        }
        if not self.is_information_value:
            out["raw_cost_difference_action_spaces_differ"] = {
                "deterministic": self.cost_difference_V0_minus_VT,
                "randomized": self.cost_difference_randomized_reference,
                "note": "V0 and VT use different action sets (T7.2): a cost-risk comparison, not an information value"}
        return out

    def definition_report(self, value_definition: Optional[DefinitionLike] = None) -> dict[str, Any]:
        """Every value with its role, next to the deterministic policies' cost, joint risk and feasibility.

        The randomised reference policies are included only as a theoretical reference
        (``not_a_feeding_recommendation=True``); the executable policies are the deterministic ones.
        ``value_definition`` (optional) marks the selected number; otherwise ``selected`` is ``None``.

        FIX_A: the deterministic policy of the matched state-independent signal is a hypothetical
        randomisation device (a coin with the signal's bin probabilities), not an executable feeding
        policy; it is reported in ``diagnostic_randomisation_devices``.  The signal policy is marked
        ``feeding_candidate=False`` when its operational saving is a randomisation channel only (or
        cannot be assessed); ``randomization_channel`` carries the assessment and the diagnostic ratio
        ``matched_uninformative_contrast_ratio`` (R3B).  Each value carries ``excluded_from_paper_main_results``
        (R3B: the diagnostic contrast and the heuristic minimum); ``policy_class_comparison`` shows the two
        policy classes side by side.
        """
        sel = None if value_definition is None else resolve_value_definition(
            value_definition, context="definition_report", allow_legacy=False).value
        if sel is None and self.value_definition is not None:
            sel = self.value_definition
        cids = self.candidate_ids
        rc = self.randomization_channel
        vt_block = _policy_block(self.VT, cids)
        if vt_block.get("computed"):
            cand = rc["status"] in EXECUTABLE_RANDOMIZATION_STATUSES or not self.is_information_value
            vt_block["feeding_candidate"] = bool(cand and self.VT.has_solution)
            vt_block["feeding_candidate_reason"] = (
                "operational saving not attributable to information: " + rc["reason"] if not cand else
                ("executable deterministic bin -> ration map; randomisation channel: " + rc["status"]))
        return {
            "structure_id": self.structure_id, "info_type": self.info_type, "alpha": self.alpha,
            "value_unit": self.value_unit, "scope_label": self.scope_label, "is_synthetic": self.is_synthetic,
            "information_value_semantics": self.information_value_semantics,
            "selected_value_definition": sel,
            "selected_value": None if sel is None else self.value(sel),
            "values": {d.value: {"value": self.value(d), "role": VALUE_DEFINITION_ROLES[d.value],
                                 "note": VALUE_DEFINITION_NOTES[d.value],
                                 "excluded_from_paper_main_results": d.value in EXCLUDED_FROM_PAPER_MAIN_RESULTS,
                                 "status": "ok" if self.value(d) is not None else
                                 f"undefined:{self._undefined_reason(d)}"}
                       for d in ValueDefinition},
            "randomization_benchmark_value_V0_minus_VT_uninformative": self.randomization_benchmark_value,
            "randomization_channel": dict(rc),
            "error_model_identification": self.error_model_identification,
            "policy_class_comparison": self.policy_class_comparison(),
            "min_attainable_ex_ante_joint_risk": dict(self.min_attainable_ex_ante_joint_risk),
            "deterministic_policies": {
                "role": ("executable deterministic policies: the constant no-information ration and the signal's bin -> "
                         "ration map; the signal policy is a feeding candidate only if feeding_candidate is true "
                         "(its saving is not a randomisation channel only)"),
                "V0_constant": _policy_block(self.V0, cids),
                "VT_signal_policy": vt_block,
            },
            "diagnostic_randomisation_devices": {
                "role": "hypothetical_randomisation_device",
                "not_a_feeding_recommendation": True,
                "note": ("deterministic policy of a state-independent signal with the same active-bin probabilities: "
                         "a coin with the signal's bin probabilities, used only to measure the randomisation channel "
                         "(benchmark B = V0 - VT(U)); never a feeding policy"),
                "VT_matched_uninformative_bins_policy": _policy_block(self.VT_uninformative_benchmark, cids),
            },
            "randomized_reference_policies": {
                "role": "theoretical reference only", "not_a_feeding_recommendation": True,
                "V0_randomized": _policy_block(self.V0_randomized, cids),
                "VT_randomized": _policy_block(self.VT_randomized, cids),
            },
        }

    def __repr__(self) -> str:
        def f(v):
            return "None" if v is None else f"{v:.6g}"
        sel = self.value_definition
        return (f"InformationValueResult(structure_id={self.structure_id!r}, info_type={self.info_type!r}, "
                f"alpha={self.alpha:g}, semantics={self.information_value_semantics!r}, "
                f"V0={f(self.V0.expected_cost)}, VT={f(self.VT.expected_cost)}, "
                f"value_definition={sel!r}, primary_value={f(self.primary_value)} [{self.primary_value_status}], "
                f"operational_deterministic_cost_difference={f(self.gross_value)}, "
                f"randomized_same_class_information_reference[theoretical]={f(self.gross_value_randomized_reference)}, "
                f"heuristic_min_of_two_policy_values[heuristic]={f(self.heuristic_min_of_two_policy_values)}, "
                f"contrast_vs_matched_uninformative_bins[diagnostic, any sign, not an information value]="
                f"{f(self.contrast_vs_matched_uninformative_bins)}, "
                f"randomization_benchmark_value={f(self.randomization_benchmark_value)}, "
                f"is_synthetic={self.is_synthetic})")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (policy details included)."""
        def c(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, Mapping):
                return {str(k): c(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [c(x) for x in v]
            return v
        return {
            "structure_id": self.structure_id, "info_type": self.info_type, "likelihood_kind": self.likelihood_kind,
            "alpha": self.alpha, "unknown_policy": self.unknown_policy,
            "V0": self.V0.to_dict(), "VT": self.VT.to_dict(), "gross_value": self.gross_value,
            "gross_value_status": self.gross_value_status,
            "V0_randomized": None if self.V0_randomized is None else self.V0_randomized.to_dict(),
            "VT_randomized": None if self.VT_randomized is None else self.VT_randomized.to_dict(),
            "gross_value_randomized_reference": self.gross_value_randomized_reference,
            "randomization_benchmark_value": self.randomization_benchmark_value,
            "containment_check": c(self.containment_check), "occupancy": c(self.occupancy),
            "error_model_identification": self.error_model_identification, "warnings": list(self.warnings),
            "n_states": self.n_states, "n_bins": self.n_bins, "n_candidates": self.n_candidates,
            "provenance": c(self.provenance), "is_synthetic": self.is_synthetic, "scope_label": self.scope_label,
            "value_unit": self.value_unit, "information_value_semantics": self.information_value_semantics,
            "cost_difference_V0_minus_VT": self.cost_difference_V0_minus_VT,
            "cost_difference_randomized_reference": self.cost_difference_randomized_reference,
            "contrast_vs_matched_uninformative_bins": self.contrast_vs_matched_uninformative_bins,
            "VT_uninformative_benchmark": (None if self.VT_uninformative_benchmark is None
                                           else self.VT_uninformative_benchmark.to_dict()),
            "value_definition": self.value_definition, "primary_value": self.primary_value,
            "primary_value_status": self.primary_value_status, "primary_value_role": self.primary_value_role,
            "values_by_definition": self.values_by_definition(),
            "value_definition_roles": dict(VALUE_DEFINITION_ROLES),
            "excluded_from_paper_main_results": list(EXCLUDED_FROM_PAPER_MAIN_RESULTS),
            "deprecated_aliases": {"value_net_of_randomization": "contrast_vs_matched_uninformative_bins "
                                                                  "(diagnostic; not an information value)",
                                   "executable_information_supported_value": "heuristic_min_of_two_policy_values "
                                                                             "(heuristic; R3B)",
                                   **{f"randomization_channel.{k}": f"randomization_channel.{v}"
                                      for k, v in DEPRECATED_ASSESSMENT_KEYS.items()}},
            "candidate_ids": list(self.candidate_ids),
            "action_space": c(self.action_space), "error_model_report": c(self.error_model_report),
            "randomization_channel": c(self.randomization_channel), "result_fingerprint": self.fingerprint,
            "policy_class_comparison": c(self.policy_class_comparison()),
            "min_attainable_ex_ante_joint_risk": c(self.min_attainable_ex_ante_joint_risk),
        }


def _diff(a: PolicyResult, b: PolicyResult) -> Optional[float]:
    if a.has_solution and b.has_solution:
        return float(a.expected_cost - b.expected_cost)
    return None


_OBSERVED_BASES = ("observed_incl_sampling_and_lab", "observed_incl_lab_only")


def _perfect_information_identification(structure: InformationStructure, prior: PriorStates, truth: Any
                                        ) -> tuple[str, dict[str, Any]]:
    """Identification label of *perfect* information (FIX_A; review finding: perfect structures returned
    ``not_applicable`` whatever the prior SD represented).

    Perfect information on a component values knowing the component's state as the prior models it.
    If the prior SD is an observed SD (historical results that contain sampling/laboratory error), the
    "state" includes that measurement error, so the value treats historical measurement error as batch
    variation -- exactly what contract 7.3 forbids for an identified value.  Labels:

    * ``perfect_information_on_true_state``: every observed component is a ``true_batch_state`` from
      verified object metadata (factory registry or synthetic world definition), with a numerically
      verified record when de-convolved, and sourced/synthetic provenance;
    * ``observed_state_scenario``: some observed component has an observed basis (an upper-end
      scenario, not the value of knowing the true batch state);
    * ``unidentified_scenario``: otherwise (no object metadata, ``unidentified`` basis, unverified
      metadata, unverified de-convolution, research-scenario provenance).
    """
    link = resolve_truth_link(truth, prior)                   # raises when the truth is another world
    if prior.metadata is not None:
        md, end = prior.metadata, "prior"
    elif link.metadata is not None and link.verified:
        md, end = link.metadata, "truth"
    else:
        md, end = None, None
    ver, ver_issues = (None, []) if md is None else verify_state_metadata(md)
    if ver == "factory_claim_contradicted":
        raise MetadataConflictError("the object metadata claim factory origin but contradict the registered factory "
                                    "model: " + "; ".join(ver_issues))
    comps = structure.components if structure.info_type == "perfect_partial" else prior.stochastic_components()
    per: dict[str, Any] = {}
    kinds: dict[str, list[str]] = {}
    for c in comps:
        lab = c.label()
        m = None if md is None else md.get(c)
        if m is None:
            per[lab] = {"status": "no_object_metadata"}
            kinds.setdefault("no_object_metadata", []).append(lab)
            continue
        entry: dict[str, Any] = {"variance_basis": m.variance_basis, "provenance_status": m.provenance_status,
                                 "basis_source": f"object_metadata:{end}"}
        if not m.is_stochastic:
            entry["status"] = "point_value"
        elif m.variance_basis in _OBSERVED_BASES:
            entry["status"] = "observed_basis"
            kinds.setdefault("observed", []).append(lab)
        elif m.variance_basis == "unidentified":
            entry["status"] = "unidentified_basis"
            kinds.setdefault("unidentified", []).append(lab)
        else:                                                   # true_batch_state
            chk = _true_state_variance_check(c, m, prior)
            entry["true_state_check"] = chk
            if chk["status"] == "rejected":
                raise MetadataConflictError(chk["violation"])
            if not chk["identifies"]:
                entry["status"] = "true_state_unverified"
                kinds.setdefault("unverified_true_state", []).append(lab)
            elif ver not in IDENTIFYING_VERIFICATION_STATUSES:
                entry["status"] = "metadata_unverified"
                kinds.setdefault("unverified_metadata", []).append(lab)
            elif m.provenance_status not in ("sourced", "synthetic_test_only"):
                entry["status"] = "research_scenario_basis"
                kinds.setdefault("research_scenario_basis", []).append(lab)
            else:
                entry["status"] = "verified_true_state"
        per[lab] = entry
    warnings: list[str] = []
    if "observed" in kinds:
        label = "observed_state_scenario"
        warnings.append(f"perfect information on {kinds['observed']}: the prior SD is an observed SD that contains "
                        "historical sampling/laboratory error, so knowing the 'state' values knowing that measurement "
                        "error as if it were batch variation (contract 7.3); labelled observed_state_scenario -- an "
                        "upper-end scenario, not the value of knowing the true batch state")
    elif set(kinds) - {"observed"}:
        label = "unidentified_scenario"
        warnings.append("perfect information whose prior SD is not a verified true-batch-state SD ("
                        + "; ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) + "): unidentified_scenario")
    else:
        label = "perfect_information_on_true_state"
    return label, {"status": label, "source": "object_metadata" if md is not None else "none",
                   "per_component": per, "prior_metadata_verification": ver,
                   "prior_metadata_verification_issues": list(ver_issues), "metadata_end": end,
                   "truth_origin": link.origin, "truth_linked": link.linked, "truth_verified": link.verified,
                   "warnings": warnings}


def _error_model_identification(structure: InformationStructure, report: Optional[DoubleCountReport],
                                prior_variance_basis: Optional[Mapping[ObservedComponent, str]],
                                prior_deconvolved: Optional[Mapping[ObservedComponent, bool]], *,
                                prior: Optional[PriorStates] = None, truth: Any = None,
                                error_locators: Optional[Mapping[str, Mapping[str, str]]] = None
                                ) -> tuple[str, dict[str, Any]]:
    """F06 + R2: re-run the double-count guard on the structure's own signal *and* on the actual prior
    states and truth record; never trust a bare status or a bare declaration.

    Order: (1) report binding (bound, same signal, same prior, reproduces / agrees with
    ``prior_variance_basis``); (2) object-mode guard against ``prior`` and ``truth`` (declaration vs
    object conflict raises :class:`~.prior.MetadataConflictError`); (3) untraceable sourced error values
    raise, a violation raises; (4) label (:data:`IDENTIFICATION_LABELS`):
    ``identified_by_declared_sources`` only if every component is bound to *verified* object metadata
    (factory registry or synthetic world definition, FIX_A), the verdict is ``ok``, a de-convolution
    is numerically verified, the metadata provenance is sourced/synthetic and no error value is a
    research-scenario assumption; otherwise ``unidentified_scenario``; ``error_model_unsourced`` if the
    signal's error values lack provenance or cannot be traced (no locator table).

    Perfect information is labelled by :func:`_perfect_information_identification`; uninformative
    structures are ``not_applicable``.
    """
    if structure.info_type == "uninformative":
        return "not_applicable", {}
    if structure.info_type in ("perfect_full", "perfect_partial"):
        if prior is None:
            raise InvalidProblemError("perfect information: the identification label needs the actual prior states")
        return _perfect_information_identification(structure, prior, truth)
    sm = structure.signal_model
    decl_basis: Optional[Mapping[ObservedComponent, str]] = None
    decl_dec: Optional[Mapping[ObservedComponent, bool]] = None
    if report is not None and report.object_mode and prior is not None and report.prior_fingerprint != prior.fingerprint:
        raise InvalidProblemError("double_count_report was produced for other prior states (prior fingerprint "
                                  "mismatch)")
    if prior_variance_basis is not None:
        rep_decl = double_count_guard(sm, prior_variance_basis, prior_deconvolved)
        if report is not None and not report.object_mode and not report.same_verdict(rep_decl):
            raise InvalidProblemError("double_count_report differs from the guard recomputed from prior_variance_basis "
                                      "for this signal")
        decl_basis, decl_dec = prior_variance_basis, prior_deconvolved
        source = "recomputed_from_prior_variance_basis"
    elif report is not None:
        if not report.is_bound:
            raise InvalidProblemError("double_count_report is not bound to a signal (no signal fingerprint / declared "
                                      "prior-variance basis); pass prior_variance_basis=... or a report produced by "
                                      "double_count_guard for this signal")
        if report.signal_fingerprint != sm.error_fingerprint():
            raise InvalidProblemError("double_count_report was produced for another signal model (fingerprint "
                                      "mismatch)")
        decl_basis, decl_dec = report.declared_inputs()
        if not report.object_mode:
            rep_decl = double_count_guard(sm, decl_basis, decl_dec)
            if not report.same_verdict(rep_decl):
                raise InvalidProblemError("double_count_report does not reproduce: the guard recomputed from its "
                                          "declared inputs gives another verdict")
        source = "recomputed_from_bound_report"
    else:
        source = "object_metadata"
    if prior is None:
        raise InvalidProblemError("sample information: the double-count guard needs the actual prior states")
    has_object = prior.metadata is not None or truth is not None
    if source == "object_metadata" and not has_object:
        raise InvalidProblemError("sample information needs a double_count_report (double_count_guard) or "
                                  "prior_variance_basis, or prior states / a truth record carrying object metadata "
                                  "(review R2)")
    rep = double_count_guard(sm, decl_basis, decl_dec, prior=prior, truth=truth, error_locators=error_locators)
    if report is not None and report.object_mode:
        if report.truth_fingerprint != rep.truth_fingerprint:
            raise InvalidProblemError("double_count_report was produced with another truth record (truth fingerprint "
                                      "mismatch)")
        if not report.same_verdict(rep):
            raise InvalidProblemError("double_count_report does not reproduce: the guard recomputed on the prior "
                                      "states and truth record gives another verdict" if prior_variance_basis is None
                                      else "double_count_report differs from the guard recomputed from "
                                           "prior_variance_basis on the prior states")
    checks = dict(rep.checks)
    trace = dict(checks.get("error_provenance") or {})
    trace_issues = [i for v in (trace.get("issues") or {}).values() for i in v]
    if trace_issues:
        raise InvalidProblemError("error-model provenance does not trace to sources/error_source_locators.csv "
                                  "(a sourced error value must be the measurement error it is used as; review R2 item 6, "
                                  "FIX_A): " + "; ".join(trace_issues))
    if rep.status == "violation":
        raise InvalidProblemError("double counting of batch variation and measurement error: "
                                  + "; ".join(rep.messages))
    bound = rep.object_bound
    warnings = list(checks.get("warnings", ()))
    # object metadata identify only with a provenance chain (sourced; synthetic for unit tests); a basis that
    # is itself a research-scenario assumption, or metadata without provenance, gives a labelled scenario
    comp_checks = checks.get("components", {}) or {}
    not_sourced = []
    unverified = []
    verification_by_end = {"prior": (checks.get("prior_metadata") or {}).get("verification"),
                           "truth": (checks.get("truth") or {}).get("verification")}
    used_verifications = set()
    for lab, src in rep.basis_sources:
        if not src.startswith("object_metadata"):
            continue
        end = "prior" if src.endswith(":prior") else "truth"
        md = comp_checks.get(lab, {}).get("prior_metadata" if end == "prior" else "truth_metadata") or {}
        if md.get("provenance_status") not in ("sourced", "synthetic_test_only"):
            not_sourced.append(f"{lab} ({md.get('provenance_status')})")
        v = verification_by_end[end]
        used_verifications.add(v)
        if v not in IDENTIFYING_VERIFICATION_STATUSES:
            unverified.append(f"{lab} ({end}: {v})")
    if not_sourced:
        warnings.append(f"object metadata of {not_sourced} have no sourced provenance (research_scenario_assumption or "
                        "none): the value is a labelled scenario, not an identified value")
    if unverified:
        warnings.append(f"object metadata of {unverified} cannot be verified against a generating object (factory "
                        "registry) and are not the definition of a synthetic world: the value is a labelled scenario, "
                        "not an identified value (FIX_A)")
    ts = checks.get("true_state_checks", {}) or {}
    ts_unverified = sorted(lab for lab, chk in ts.items() if chk.get("identifies") is False)
    scen_err = list(trace.get("scenario_assumption_components") or ())
    if scen_err:
        warnings.append(f"error values of {scen_err} are research-scenario assumptions: the sample-information value is "
                        "a labelled scenario, not an identified value (contract F05)")
    ident = ("identified_by_declared_sources"
             if (rep.ok and bound and not not_sourced and not unverified and not ts_unverified and not scen_err)
             else "unidentified_scenario")
    if sm.provenance_issues() or trace.get("status") == "locator_table_unavailable":
        ident = "error_model_unsourced"
    if not bound:
        unbound = [lab for lab, src in rep.basis_sources if not src.startswith("object_metadata")]
        warnings.append(f"prior variance basis of {unbound} comes only from a caller declaration (no object metadata on "
                        "the prior states or a verified truth model): the value is an unidentified scenario, not an "
                        "identified sample-information value (review R2)")
    pm = checks.get("prior_metadata") or {}
    if pm.get("is_diagnostic"):
        warnings.append("the prior states come from a diagnostic model (TN_NAIVE_DIAGNOSTIC, moments drifted): not the "
                        "declared target-moment world")
    rejected_members = sorted(lab for lab, chk in (checks.get("decomposition") or {}).items()
                              if chk.get("deconvolved_member") == "rejected_inconsistent_negative")
    scenario_member = ("all_true_variation_end_only (de-convolved member rejected: inconsistent decomposition for "
                       f"{rejected_members})" if rejected_members else None)
    if ident == "identified_by_declared_sources":
        ibasis = ("factory_registry_verified" if used_verifications == {"factory_registry_verified"}
                  else "synthetic_world_definition" if used_verifications == {"synthetic_world_definition"}
                  else "mixed:" + ",".join(sorted(str(v) for v in used_verifications)))
    else:
        ibasis = None
    return ident, {"status": rep.status, "per_component": dict(rep.per_component), "source": source,
                   "signal_fingerprint": rep.signal_fingerprint, "object_bound": bound,
                   "basis_sources": dict(rep.basis_sources), "prior_fingerprint": rep.prior_fingerprint,
                   "truth_fingerprint": rep.truth_fingerprint, "truth_origin": rep.truth_origin,
                   "truth_linked": rep.truth_linked, "truth_verified": rep.truth_verified,
                   "declared_basis": [list(x) for x in (rep.declared_basis or ())],
                   "object_metadata_without_sourced_provenance": not_sourced,
                   "object_metadata_unverified": unverified,
                   "true_state_unverified": ts_unverified,
                   "error_values_research_scenario": scen_err,
                   "identification_basis": ibasis,
                   "scenario_member": scenario_member,
                   "checks": checks, "warnings": warnings}


def compute_information_value(structure: InformationStructure, prior: PriorStates, risk: RiskTable, alpha: float, *,
                              unknown_policy: str = "error_if_any", action_mask: Optional[np.ndarray] = None,
                              double_count_report: Optional[DoubleCountReport] = None,
                              prior_variance_basis: Optional[Mapping[ObservedComponent, str]] = None,
                              prior_deconvolved: Optional[Mapping[ObservedComponent, bool]] = None,
                              streams: Optional[RandomStreams] = None,
                              include_randomized_reference: bool = True,
                              include_randomization_benchmark: bool = True,
                              policy_method: str = "milp",
                              solver_options: Optional[SolverOptions] = None,
                              risk_tol: float = RISK_TOL_DEFAULT,
                              cost_tol: float = 1e-9,
                              value_definition: Optional[DefinitionLike] = None,
                              truth: Any = None,
                              error_locators: Optional[Mapping[str, Mapping[str, str]]] = None
                              ) -> InformationValueResult:
    """``V_0``, ``V_T`` and the gross value ``V_0 - V_T`` of one structure (same prior/library/risk/mask).

    Value definition (second review R1): every difference is returned; ``value_definition``
    (optional, :class:`ValueDefinition`) only marks which one is the ``primary_value``.  Without it
    ``primary_value`` is ``None`` -- no difference is promoted by default.  Choosing the randomised
    reference needs ``include_randomized_reference=True``; choosing the diagnostic contrast needs
    ``include_randomization_benchmark=True``.

    Action sets: V0 uses the candidates structurally feasible under the t0 DM estimate (``&``
    ``action_mask``); VT uses, in every bin, the candidates feasible under that bin's
    decision-time DM estimate (``&`` ``action_mask``).  ``gross_value`` is reported only when every
    V0 action is legal in every active bin (T7.1); otherwise ``cost_difference_V0_minus_VT`` carries
    the raw difference and ``information_value_semantics`` says it is a cost-risk comparison.

    Sample information requires ``prior_variance_basis`` (the guard is run here on the structure's
    own signal) or a bound ``double_count_report`` from :func:`~.signal.double_count_guard` for this
    signal: ``violation`` raises; ``unidentified`` is allowed but the result is labelled
    ``unidentified_scenario`` (report as one member of a scenario range).  A signal model with
    provenance issues is labelled ``error_model_unsourced``.

    Prior / truth / signal (review R2): the guard also runs against the actual ``prior`` states and
    the ``truth`` record (``StateMetadata``, ``PriorStates`` or ``FactoryModel``; ``None`` = the
    generator of the prior states, which is the truth of this finite model).  Object metadata of the
    prior decide the basis and may replace ``prior_variance_basis``; a contradicting declaration
    raises :class:`~.prior.MetadataConflictError`; a truth model that did not generate the prior states
    raises.  Without object metadata a declared basis is used but the result is at best
    ``unidentified_scenario``.

    FIX_A: object metadata identify only when verified (factory registry or synthetic world
    definition); a truth record helps only through a verified link; a de-convolved true state must
    match its numeric record; ``sourced`` error values are traced to the locator table
    (``error_locators``; default ``sources/error_source_locators.csv``) and untraceable ones raise.
    Perfect information carries ``perfect_information_on_true_state`` / ``observed_state_scenario`` /
    ``unidentified_scenario``.  ``randomization_channel`` (a property of the result) says whether
    the operational value is a randomisation channel.
    """
    vdef = None if value_definition is None else resolve_value_definition(
        value_definition, context="compute_information_value", allow_legacy=False)
    if vdef is ValueDefinition.RANDOMIZED_SAME_CLASS_INFORMATION_REFERENCE and not include_randomized_reference:
        raise InvalidProblemError("value_definition=randomized_same_class_information_reference needs "
                                  "include_randomized_reference=True")
    if vdef is ValueDefinition.CONTRAST_VS_MATCHED_UNINFORMATIVE_BINS and not include_randomization_benchmark and \
            structure.info_type != "uninformative":
        raise InvalidProblemError("value_definition=contrast_vs_matched_uninformative_bins needs "
                                  "include_randomization_benchmark=True")
    if vdef is ValueDefinition.HEURISTIC_MIN_OF_TWO_POLICY_VALUES and not include_randomized_reference:
        raise InvalidProblemError("value_definition=heuristic_min_of_two_policy_values needs "
                                  "include_randomized_reference=True")
    ident, em_report = _error_model_identification(structure, double_count_report, prior_variance_basis,
                                                   prior_deconvolved, prior=prior, truth=truth,
                                                   error_locators=error_locators)
    lik = build_likelihood(structure, prior, streams)
    acts = decision_time_action_masks(prior, lik, risk, user_mask=action_mask)
    if not acts.t0_mask.any():
        raise InvalidProblemError("no candidate is structurally feasible under the t0 (no-information) DM estimate "
                                  "within the action mask; the library must contain no-information candidates (T7.1)")
    bin_mask = None if acts.action_space == "t0_fixed" else acts.bin_masks
    pp = PolicyProblem.build(prior, lik, risk, alpha, unknown_policy=unknown_policy, action_mask=acts.t0_mask,
                             risk_tol=risk_tol, bin_action_mask=bin_mask)
    contained = pp.constant_policies_contained
    semantics = "information_value_within_library" if contained else "cost_risk_comparison_action_spaces_differ"
    v0 = solve_no_information(pp)
    vt = solve_signal_policy(pp, method=policy_method, solver_options=solver_options)
    raw = _diff(v0, vt)
    if not v0.has_solution and vt.has_solution:
        status = "V0_infeasible_within_library"      # information makes the target reachable; no finite value
    elif not v0.has_solution:
        status = "V0_and_VT_infeasible_within_library"
    elif not vt.has_solution:
        status = f"VT_failed:{vt.status}"
    else:
        status = "ok"
    containment = {"holds": None, "excess": None, "constant_policies_contained": contained}
    tol_eff = cost_tol * max(1.0, abs(v0.expected_cost)) if v0.has_solution else cost_tol
    if raw is not None and contained:
        containment.update({"holds": bool(raw >= -tol_eff), "excess": float(-raw) if raw < 0 else 0.0,
                            "tolerance": tol_eff,
                            "note": "VT <= V0 is guaranteed by constant-policy containment; a violation is a "
                                    "numerical or solver issue and is reported, not truncated"})
        if raw < -tol_eff:
            status = "containment_violated_numerical"
    if not contained:
        containment["note"] = ("some no-information actions are not legal after the assay (decision-time DM rule): "
                               "V0 - VT compares different action spaces; reported as a cost-risk comparison, not "
                               "an information value (T7.2; PUD-P8-03)")
        status = f"action_spaces_differ:{status}"
    gross = raw if contained else None
    v0r = vtr = None
    rawr = None
    if include_randomized_reference:
        v0r = solve_no_information_randomized(pp, solver_options)
        vtr = solve_signal_policy_randomized(pp, solver_options)
        rawr = _diff(v0r, vtr)
    grossr = rawr if contained else None
    bench = None
    vtu = None
    if include_randomization_benchmark and v0.has_solution and structure.info_type != "uninformative":
        # value of a state-independent signal with the same bin probabilities (pure randomisation device);
        # a state-independent signal carries no DM information: every bin uses the t0 action set
        pz = pp.Pz[pp.active_bins]
        ul = uninformative_likelihood(prior, pz / pz.sum())
        ppu = PolicyProblem.build(prior, ul, risk, alpha, unknown_policy=unknown_policy, action_mask=acts.t0_mask,
                                  risk_tol=risk_tol)
        vtu = solve_signal_policy(ppu, method=policy_method, solver_options=solver_options)
        bench = _diff(v0, vtu)
    # DIAGNOSTIC contrast (R1): VT(matched state-independent bins) - VT; any sign, never truncated, never a default
    contrast = None
    if contained and vt.has_solution:
        if structure.info_type == "uninformative":
            contrast = 0.0                           # the structure is its own randomisation benchmark
        elif vtu is not None and vtu.has_solution:
            contrast = float(vtu.expected_cost - vt.expected_cost)
    occ = bin_occupancy(prior, lik)
    warnings = list(lik.warnings) + list(em_report.get("warnings", ()))
    if bench is not None and bench > tol_eff:
        warnings.append(f"a state-independent signal with the same bin probabilities already lowers the deterministic "
                        f"V_T by {bench:.6g}: the deterministic gross value includes randomisation value; the "
                        f"randomised same-class reference removes this channel, while "
                        f"contrast_vs_matched_uninformative_bins is only a diagnostic contrast (not a pure information "
                        f"value; docs/value_definition.md, design note §4)")
    if contrast is not None and contrast < -tol_eff:
        warnings.append(f"diagnostic contrast_vs_matched_uninformative_bins = {contrast:.6g} < 0: the signal's bins are "
                        f"worse for deterministic mixing than a state-independent device with the same bin "
                        f"probabilities; kept as is (not truncated); it is not an information value")
    if not contained:
        warnings.append("information-dependent action space: V0 - VT is not an information value (see "
                        "information_value_semantics)")
    if acts.n_active_bins_without_candidate:
        warnings.append(f"{acts.n_active_bins_without_candidate} active bin(s) have no structurally feasible candidate "
                        "under their decision-time DM estimate: extend the library for this structure")
    min_risk = _min_attainable_risk(pp, contained)
    if not v0.has_solution and not vt.has_solution:
        warnings.append(
            "no feasible no-information and no feasible with-information policy at this alpha within the library: every "
            "cost value is undefined (None, reason in gross_value_status), not 0; the risk diagnostic "
            f"min_attainable_ex_ante_joint_risk ({min_risk.get('no_information')} without, "
            f"{min_risk.get('with_information')} with the signal) is library-limited and never converted to money (R3B)")
    prov = dict(pp.provenance)
    prov.update({"structure": structure.structure_id, "description": structure.description,
                 "observed_components": tuple(c.label() for c in lik.components),
                 "binning_fingerprint": None if structure.binning is None else structure.binning.fingerprint,
                 "signal_fingerprint": None if structure.signal_model is None else structure.signal_model.fingerprint(),
                 "policy_problem_fingerprint": pp.fingerprint,
                 "action_mask_fingerprint": stable_hash("action_mask", pp.action_mask),
                 "bin_action_mask_fingerprint": stable_hash("bin_action_mask", pp.bin_action_mask),
                 "action_space": acts.action_space,
                 "include_randomized_reference": bool(include_randomized_reference),
                 "include_randomization_benchmark": bool(include_randomization_benchmark),
                 "cost_tol": float(cost_tol)})
    res = InformationValueResult(
        structure.structure_id, structure.info_type, lik.kind, float(alpha), unknown_policy, v0, vt, gross, status,
        v0r, vtr, grossr, bench, containment,
        {"p_bin": occ["p_bin"], "effective_states": occ["effective_states"], "n_active_bins": occ["n_active_bins"],
         "min_effective_states_active": occ["min_effective_states_active"]},
        ident, tuple(warnings), prior.n_states, lik.n_bins, pp.n_candidates, prov,
        bool(risk.is_synthetic or prior.is_synthetic),
        information_value_semantics=semantics, cost_difference_V0_minus_VT=raw,
        cost_difference_randomized_reference=rawr, contrast_vs_matched_uninformative_bins=contrast,
        action_space=acts.summary(risk.candidate_ids), error_model_report=em_report,
        VT_uninformative_benchmark=vtu, candidate_ids=tuple(risk.candidate_ids),
        value_definition=None if vdef is None else vdef.value, min_attainable_ex_ante_joint_risk=min_risk)
    rc = randomization_channel_assessment(res)
    if rc["status"] in ("randomization_only", "partly_randomization_channel"):
        # FIX_A: the executable consequence of the randomisation channel is stated on the result itself
        extra = (f"randomization channel ({rc['status']}): {rc['reason']}; "
                 + ("the operational value cannot justify buying, ranking or freezing this assay"
                    if rc["status"] == "randomization_only" else
                    "report matched_uninformative_contrast_ratio (a diagnostic ratio, not a randomisation share) next "
                    "to every executable output")
                 + " (docs/value_definition.md §2.5)")
        res = dataclasses.replace(res, warnings=tuple(res.warnings) + (extra,))
    return res


def _min_attainable_risk(pp: PolicyProblem, contained: bool) -> dict[str, Any]:
    """Lowest ex-ante joint risk reachable within the library and bins, without and with the signal (R3B).

    ``no_information = min_{k in A_0} r_k``; ``with_information = sum_{active z} min_{k in A_z} R[z, k]``
    (``None`` if an active bin has no legal candidate).  A randomised policy cannot go lower (a mixture's
    risk is an average of its members' risks), so the numbers hold for both policy classes.  A risk
    diagnostic in probability units: never converted to money, and limited to the finite library K and
    the fixed bins -- not a statement about the whole decision space (a whole-space minimum needs a
    separate minimum-violation solve over the ration space, not done here).
    """
    r0 = float(np.min(pp.r_marginal[pp.action_mask]))
    allowed = pp.vt_allowed()
    rt: Optional[float] = 0.0
    for z in pp.active_bins:
        m = allowed[z]
        if not m.any():
            rt = None
            break
        rt += float(np.min(pp.R[z, m]))
    tol = float(pp.risk_tol)
    return {"no_information": r0, "with_information": rt,
            "reduction_with_information": None if rt is None else float(r0 - rt),
            "alpha": float(pp.alpha), "risk_tol": tol,
            "no_information_reaches_alpha": bool(r0 <= pp.alpha + tol),
            "with_information_reaches_alpha": None if rt is None else bool(rt <= pp.alpha + tol),
            "unit": "probability (ex-ante joint violation of the declared risk event)",
            "scope": "finite candidate library K, fixed development bins, declared prior and decision-time action sets",
            "whole_decision_space": "not computed here (needs a separate whole-space minimum-violation solve)",
            "policy_classes": "same for deterministic and randomised policies (a mixture's risk is an average)",
            "action_spaces_nested": bool(contained),
            "money": "never converted to money (R3B; docs/value_semantics_decision.md)",
            "role": "risk_diagnostic"}


# ---------------------------------------------------------------------------------------------
# theoretical-order diagnostics (T7.4, F10)
# ---------------------------------------------------------------------------------------------

GUARANTEES = (
    "theorem",                                  # holds in the declared model (up to solver tolerance)
    "not_guaranteed_randomization_channel",     # deterministic policies: noise/bins can act as randomisation
    "not_guaranteed_not_a_garbling",            # the lesser signal is not a garbling of the greater one
    "not_guaranteed_information_dependent_action_space",  # DM information changes the legal action sets
)


@dataclass(frozen=True)
class OrderCheck:
    """One ordering diagnostic ``value(lesser) <= value(greater)`` (or ``0 <= value``)."""

    relation: str
    lesser: str
    greater: str
    policy_class: str
    lesser_value: Optional[float]
    greater_value: Optional[float]
    holds: Optional[bool]
    violation_size: Optional[float]
    guarantee: str
    note: str = ""


def _val(res: InformationValueResult, policy_class: str) -> Optional[float]:
    # explicit mapping (R1): deterministic -> operational_deterministic_cost_difference,
    # randomized -> randomized_same_class_information_reference; the diagnostic contrast is never order-checked
    return res.gross_value if policy_class == "deterministic" else res.gross_value_randomized_reference


def standard_order_relations(results: Mapping[str, InformationValueResult]) -> list[tuple[str, str, str, str]]:
    """Relations ``(lesser, greater, policy_class, guarantee)`` for a set of results.

    * sample on C  <= perfect partial on C: theorem only for randomised policies and an *exact*
      partial structure (the sample is then a garbling); binned partial structures are not
      guaranteed (not a garbling); deterministic policies are not guaranteed (randomisation).
    * perfect partial on C <= perfect full: theorem (partition refinement) for both classes.
    * sample on C <= perfect full: theorem for randomised policies; not guaranteed for deterministic.
    """
    rel: list[tuple[str, str, str, str]] = []
    full = [k for k, r in results.items() if r.info_type == "perfect_full"]
    part = {k: r for k, r in results.items() if r.info_type == "perfect_partial"}
    samp = {k: r for k, r in results.items() if r.info_type == "sample"}
    for f in full:
        for p in part:
            rel.append((p, f, "deterministic", "theorem"))
            rel.append((p, f, "randomized", "theorem"))
        for s in samp:
            rel.append((s, f, "deterministic", "not_guaranteed_randomization_channel"))
            rel.append((s, f, "randomized", "theorem"))
    for s, rs in samp.items():
        comps_s = set(rs.provenance.get("observed_components", ()))
        for p, rp in part.items():
            comps_p = set(rp.provenance.get("observed_components", ()))
            if comps_s and comps_s == comps_p:
                exact = rp.likelihood_kind == "perfect_partial_exact"
                rel.append((s, p, "deterministic", "not_guaranteed_randomization_channel"))
                rel.append((s, p, "randomized", "theorem" if exact else "not_guaranteed_not_a_garbling"))
    return rel


def check_theoretical_order(results: Mapping[str, InformationValueResult],
                            relations: Optional[Sequence[tuple[str, str, str, str]]] = None,
                            tol: float = 1e-9) -> list[OrderCheck]:
    """Diagnostics for ``0 <= value`` of every result and for the listed relations.

    Nothing is modified: violations are reported with their size and guarantee class.  A violated
    ``theorem`` points to a numerical/solver problem or an inconsistent setup (different prior,
    library, risk or mask) and must be investigated; a violated ``not_guaranteed_*`` relation is an
    expected possibility that must be explained in the report.  ``tol`` is relative to the cost
    scale ``max(1, |V_0|)``.
    """
    out: list[OrderCheck] = []
    scale = max([1.0] + [abs(r.V0.expected_cost) for r in results.values() if r.V0.has_solution])
    tol = float(tol) * scale
    for k, r in results.items():
        for pc in ("deterministic", "randomized"):
            v = _val(r, pc)
            holds = None if v is None else bool(v >= -tol)
            out.append(OrderCheck("nonnegative", "0", k, pc, 0.0, v, holds,
                                  None if v is None or v >= -tol else float(-v), "theorem",
                                  "constant-policy containment"))
    fps = {r.provenance.get("risk_table_fingerprint") for r in results.values()}
    pri = {r.provenance.get("prior_fingerprint") for r in results.values()}
    alphas = {r.alpha for r in results.values()}
    masks = {r.provenance.get("action_mask_fingerprint") for r in results.values()}
    unk = {r.unknown_policy for r in results.values()}
    same_model = len(fps) == 1 and len(pri) == 1 and len(alphas) == 1 and len(masks) == 1 and len(unk) == 1
    rels = list(relations) if relations is not None else standard_order_relations(results)
    for lesser, greater, pc, guarantee in rels:
        a, b = _val(results[lesser], pc), _val(results[greater], pc)
        if not same_model:
            guarantee = "not_guaranteed_not_a_garbling"
        elif any(results[n].provenance.get("action_space") == "information_dependent" for n in (lesser, greater)):
            guarantee = "not_guaranteed_information_dependent_action_space"
        if a is None or b is None:
            out.append(OrderCheck("order", lesser, greater, pc, a, b, None, None, guarantee,
                                  "one side has no finite value"))
            continue
        holds = bool(a <= b + tol)
        note = "" if same_model else ("results do not share prior / risk table / alpha / action mask / unknown "
                                      "policy: no theoretical order")
        out.append(OrderCheck("order", lesser, greater, pc, a, b, holds, None if holds else float(a - b), guarantee,
                              note))
    return out


# ---------------------------------------------------------------------------------------------
# approximation-error checks (T7.3 "K 扩大、分箱精化、小问题完全枚举", F09)
# ---------------------------------------------------------------------------------------------

def check_library_enlargement(small: InformationValueResult, large: InformationValueResult,
                              tol: float = 1e-9) -> dict[str, Any]:
    """K ⊂ K': both ``V_0`` and ``V_T`` must not increase (theorem); the gross value may move either way."""
    out: dict[str, Any] = {}
    for name in ("V0", "VT"):
        a, b = getattr(small, name), getattr(large, name)
        if a.has_solution and b.has_solution:
            out[name] = {"small": a.expected_cost, "large": b.expected_cost,
                         "monotone_holds": bool(b.expected_cost <= a.expected_cost + tol),
                         "change": float(b.expected_cost - a.expected_cost)}
        else:
            out[name] = {"small": a.status, "large": b.status, "monotone_holds": None}
    out["gross_value_change"] = (None if small.gross_value is None or large.gross_value is None
                                 else float(large.gross_value - small.gross_value))
    out["gross_value_change_definition"] = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value
    out["value_change_by_definition"] = _value_changes(small, large)
    out["note"] = ("values are not monotone in K; their change (per explicit definition) is the library-approximation "
                   "signal")
    return out


def _value_changes(a: InformationValueResult, b: InformationValueResult) -> dict[str, Optional[float]]:
    """``b - a`` under every definition (``None`` where either side is undefined; never 0 as a placeholder)."""
    out: dict[str, Optional[float]] = {}
    for d in ValueDefinition:
        va, vb = a.value(d), b.value(d)
        out[d.value] = None if va is None or vb is None else float(vb - va)
    return out


def check_bin_refinement(coarse: InformationValueResult, fine: InformationValueResult,
                         tol: float = 1e-9) -> dict[str, Any]:
    """Nested refinement of *perfect-information* bins: ``V_T(fine) <= V_T(coarse)`` (theorem).

    For noisy signals a finer binning is also a refinement (the coarse bin is a function of the fine
    bin), so the same monotonicity holds; the change size is the binning-approximation signal.
    """
    a, b = coarse.VT, fine.VT
    if not (a.has_solution and b.has_solution):
        return {"monotone_holds": None, "coarse": a.status, "fine": b.status}
    # R1: no ``or 0.0`` placeholder for an undefined value; the change is given per explicit definition
    return {"coarse_VT": a.expected_cost, "fine_VT": b.expected_cost,
            "monotone_holds": bool(b.expected_cost <= a.expected_cost + tol),
            "gross_value_change": (None if fine.gross_value is None or coarse.gross_value is None
                                   else float(fine.gross_value - coarse.gross_value)),
            "gross_value_change_definition": ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE.value,
            "value_change_by_definition": _value_changes(coarse, fine)}
