"""Information-value module (contract §12, T1, T7; stage 8 prototype).

Status (contract §2.3): ``implemented``, ``unit_passed`` on synthetic data only
(``reports/engine_test_log_P8.md``).  No pilot, no official run, no research result.

Pipeline (all development objects from ``opt``/``validation`` states only):

1. :class:`PriorStates` -- finite weighted representation of the prior over the current batch state.
2. :class:`SignalModel` -- ``Z = theta_current + e_sampling + e_lab + b_systematic`` with separate
   components, :class:`SamplingProtocol` (replicates reduce only lab repeatability) and
   :func:`double_count_guard` (batch variation vs. measurement error).
3. :class:`SignalBinning` -- fixed development bins; likelihoods ``L[s, z]`` for perfect (full /
   partial) and sample information (:mod:`.likelihood`).
4. :class:`CandidateLibrary` (+ :func:`generate_conditional_library`) and :func:`compute_risk_table`
   (public evaluator).
5. :class:`PolicyProblem` -> :func:`solve_no_information` (V0, constant policy) and
   :func:`solve_signal_policy` (VT, binary MILP / exhaustive enumeration); randomised LP references;
   :func:`per_outcome_conditional_risk_rule` (T7.2, separate, never EVSI).
6. :func:`compute_information_value` (gross value V0 - VT within K and bins) and
   :func:`check_theoretical_order` (diagnostic only; never truncates).
7. :mod:`.economics` -- H x T batch conversion, break-even cost, net value only with all costs known.
8. :mod:`.strategies` -- assay-selection strategies at equal budget (oracle only as explicit reference).
9. :func:`freeze_policy` / :func:`evaluate_frozen_policy` -- frozen policy on independent states;
   :func:`calibrate_alpha_train` -- validation-stream calibration of the training risk level.

The containers :class:`AssaySpec` and :class:`InformationPolicy` remain defined in
:mod:`ration_reliability.datamodel`.

Red-team fixes (2026-09-24; ``audit/_parts/FIX_engine_record.md``): decision-time action sets
(:func:`decision_time_action_masks`; V0 and non-DM structures use the t0 DM estimate, DM structures
the bin's ``E[d | z]``; non-nested action spaces are reported as cost-risk comparisons, not as
information values), ``value_net_of_randomization``, double-count guard bound to the structure's
signal, saving only at matched risk, required validity period and inventory check before H x T.

Second review R1 (2026-09-25; ``docs/value_definition.md``): explicit :class:`ValueDefinition`
(``operational_deterministic_cost_difference`` / ``randomized_same_class_information_reference`` /
``contrast_vs_matched_uninformative_bins``); no default primary value; ``value_net_of_randomization``
is a deprecated alias of the diagnostic contrast; strategy ranking, break-even / H x T conversions and
frozen-policy evaluation require an explicit definition.

Second review R2 (2026-09-25; ``audit/_parts/round2/K2_prior_truth_signal_record.md``): prior and truth
carry per-component object metadata (:class:`StateMetadata` / :class:`ComponentMetadata`: variance
basis, data fingerprint, decomposition, measurement model, synthetic flag, provenance), bound from the
uncertainty factory when draws become :class:`PriorStates`; :func:`double_count_guard` has an object
mode (prior / truth / signal), declarations never override object metadata
(:class:`MetadataConflictError`), a caller declaration alone never yields an identified value
(``unidentified_scenario``), variance decompositions are recorded and never clipped
(:class:`VarianceDecomposition`), and :func:`trace_error_provenance` ties sourced error values to
``sources/error_source_locators.csv``.

Red-team FIX_A (2026-09-25; ``audit/_parts/round2/FIX_A_record.md``): the operational value's
randomisation channel is assessed (:func:`randomization_channel_assessment`) and a pure randomisation
channel is never converted to money, ranked or frozen (:class:`RandomizationChannelError`); money
conversions take values bound to their result (:class:`PerHeadDayValue`, :class:`BatchAmount`,
:class:`ValueBinding`) and keep scenario labels; perfect information has identification labels; object
metadata identify only when verified (:func:`verify_state_metadata`, :func:`resolve_truth_link`);
de-convolutions are checked numerically; sourced error values are traced
(:func:`default_error_locators`); :meth:`PriorStates.from_drawset` resolves the metadata of
``EnergyColumnModel`` worlds and refuses unresolved non-synthetic draws.

Third review R3B (2026-09-25; ``docs/value_semantics_decision.md``): ``executable_information_supported_value``
is renamed ``heuristic_min_of_two_policy_values`` (role ``heuristic``; deprecated aliases kept) -- a heuristic
of two policy classes, not garbling-monotone, never a default payment bound, net value or ranking basis and
never a paper main result; ``randomization_share_of_operational`` is renamed
``matched_uninformative_contrast_ratio`` (a diagnostic ratio); ``InformationValueResult.policy_class_comparison``
reports the deterministic class and the randomised reference side by side; undefined cost values stay
``None`` with a reason and ``min_attainable_ex_ante_joint_risk`` is a library-limited risk diagnostic (never money).

Round-3 red team FIX3_BC (2026-09-25; finding B-1): every money conversion is refused by default
(:class:`DecisionProblemRequiredError`) unless it comes from a :class:`CompleteDecisionProblem` (the five
elements of ``docs/value_semantics_decision.md`` §2; the money value is ``V0 - VT`` of that problem's one policy
class; nothing of the other class gates it) or is an explicitly requested, labelled development diagnostic
(``diagnostic_without_decision_problem=True``); every money output is excluded from paper main results in this
version.  Decision-value rankings without a decision problem are labelled accordingly; undefined or screened
rankings are ``None`` with a reason (B-4).
"""

from ..datamodel import AssaySpec, InformationPolicy  # noqa: F401
from .binning import ExactGrouping, SignalBinning, exact_value_groups, quantile_edges  # noqa: F401
from .economics import (  # noqa: F401
    COST_COMPONENTS,
    DECISION_PROBLEM_ELEMENT_STATUSES,
    ECONOMIC_VALUE_DEFINITIONS,
    ENGINE_RISK_TIMING,
    INFORMATION_PROCESSING_CONVENTIONS,
    MONEY_BASES,
    RISK_TIMINGS,
    RQ3_MONEY_IN_PAPER_MAIN_RESULTS,
    VALUE_BINDINGS,
    CompleteDecisionProblem,
    DecisionProblemElement,
    DecisionProblemRequiredError,
    BatchAmount,
    PerHeadDayValue,
    ValueBinding,
    economic_value_definition,
    per_head_day_value,
    BatchCoverage,
    InventoryCoverageResult,
    NetValueResult,
    batch_gross_value,
    break_even_max_cost_per_batch,
    inventory_coverage_check,
    net_value_per_batch,
    per_head_day_from_batch_cost,
)
from .evaluation import (  # noqa: F401
    FrozenPolicyEvaluation,
    FrozenSignalPolicy,
    calibrate_alpha_train,
    evaluate_frozen_policy,
    freeze_policy,
    signal_bins_for_states,
)
from .library import (  # noqa: F401
    UNKNOWN_POLICIES,
    CandidateLibrary,
    ConditioningSpec,
    DecisionTimeActions,
    RiskTable,
    adjustment_mask,
    decision_time_action_masks,
    compute_risk_table,
    conditioning_from_likelihood,
    generate_conditional_library,
)
from .likelihood import (  # noqa: F401
    SignalLikelihood,
    bin_occupancy,
    perfect_full_likelihood,
    perfect_partial_likelihood,
    sample_likelihood_analytic,
    sample_likelihood_monte_carlo,
    uninformative_likelihood,
)
from .policy import (  # noqa: F401
    RISK_TOL_DEFAULT,
    ConditionalRiskRuleResult,
    PolicyProblem,
    PolicyResult,
    per_outcome_conditional_risk_rule,
    solve_no_information,
    solve_no_information_randomized,
    solve_signal_policy,
    solve_signal_policy_randomized,
)
from .prior import DEVELOPMENT_STREAMS, DM_COMPONENT, ObservedComponent, PriorStates  # noqa: F401
from .prior import (  # noqa: F401  (second review R2: prior / truth object metadata)
    COMPONENT_VARIANCE_BASES,
    DECOMPOSITION_STATUSES,
    METADATA_ORIGINS,
    ComponentMetadata,
    InconsistentDecompositionError,
    MetadataConflictError,
    StateMetadata,
    VarianceDecomposition,
    decompose_observed_variance,
    resolve_truth_metadata,
)
from .prior import (  # noqa: F401  (FIX_A: verification of object metadata and of the truth link)
    IDENTIFYING_VERIFICATION_STATUSES,
    METADATA_VERIFICATION_STATUSES,
    TruthLink,
    resolve_truth_link,
    state_metadata_of_model,
    verify_state_metadata,
)
from .signal import (  # noqa: F401
    LAB_ERROR_SEMANTICS,
    PRIOR_VARIANCE_BASES,
    ComponentErrorModel,
    DoubleCountReport,
    SamplingProtocol,
    SignalModel,
    deconvolve_true_sd,
    double_count_guard,
    signal_model_from_assay,
)
from .signal import (  # noqa: F401  (second review R2: guard object mode and error provenance tracing)
    BASIS_SOURCES,
    ERROR_FIELD_COMPONENTS,
    TRUE_STATE_SD_MC_Z,
    default_error_locators,
    load_error_locators,
    trace_error_provenance,
)
from .strategies import (  # noqa: F401
    STRATEGY_IDS,
    AssayBudget,
    AssayOption,
    StrategySelection,
    combined_structure,
    evaluate_selection_value,
    strategy_constraint_sensitivity,
    strategy_decision_value,
    strategy_max_inclusion,
    strategy_max_marginal_variance,
    strategy_none,
    strategy_oracle_reference,
    strategy_random,
)
from .value import (  # noqa: F401
    INFO_TYPES,
    LEGACY_VALUE_KIND_ALIASES,
    DEPRECATED_VALUE_DEFINITION_NAMES,
    DEPRECATED_ASSESSMENT_KEYS,
    HEURISTIC_VALUE_DEFINITIONS,
    EXCLUDED_FROM_PAPER_MAIN_RESULTS,
    require_operational_not_above_randomized_reference,
    SCOPE_LABEL,
    VALUE_DEFINITION_ROLES,
    VALUE_DEFINITIONS,
    VALUE_SEMANTICS,
    ValueDefinition,
    resolve_value_definition,
    EXECUTABLE_RANDOMIZATION_STATUSES,
    IDENTIFICATION_LABELS,
    IDENTIFIED_LABELS,
    RANDOMIZATION_CHANNEL_STATUSES,
    SCENARIO_IDENTIFICATION_LABELS,
    RandomizationChannelError,
    randomization_channel_assessment,
    require_executable_operational_value,
    InformationStructure,
    InformationValueResult,
    OrderCheck,
    build_likelihood,
    check_bin_refinement,
    check_library_enlargement,
    check_theoretical_order,
    compute_information_value,
    standard_order_relations,
)

__all__ = [
    "AssaySpec", "InformationPolicy",
    "ExactGrouping", "SignalBinning", "exact_value_groups", "quantile_edges",
    "COST_COMPONENTS", "ECONOMIC_VALUE_DEFINITIONS", "economic_value_definition", "per_head_day_value",
    "BatchCoverage", "InventoryCoverageResult", "NetValueResult", "batch_gross_value",
    "break_even_max_cost_per_batch", "inventory_coverage_check", "net_value_per_batch", "per_head_day_from_batch_cost",
    "FrozenPolicyEvaluation", "FrozenSignalPolicy", "calibrate_alpha_train", "evaluate_frozen_policy",
    "freeze_policy",
    "signal_bins_for_states",
    "UNKNOWN_POLICIES", "CandidateLibrary", "ConditioningSpec", "RiskTable", "adjustment_mask", "compute_risk_table",
    "conditioning_from_likelihood", "generate_conditional_library", "DecisionTimeActions", "decision_time_action_masks",
    "SignalLikelihood", "bin_occupancy", "perfect_full_likelihood", "perfect_partial_likelihood",
    "sample_likelihood_analytic", "sample_likelihood_monte_carlo", "uninformative_likelihood",
    "RISK_TOL_DEFAULT", "ConditionalRiskRuleResult", "PolicyProblem", "PolicyResult",
    "per_outcome_conditional_risk_rule", "solve_no_information", "solve_no_information_randomized",
    "solve_signal_policy", "solve_signal_policy_randomized",
    "DEVELOPMENT_STREAMS", "DM_COMPONENT", "ObservedComponent", "PriorStates",
    "LAB_ERROR_SEMANTICS", "PRIOR_VARIANCE_BASES", "ComponentErrorModel", "DoubleCountReport", "SamplingProtocol",
    "SignalModel", "deconvolve_true_sd", "double_count_guard", "signal_model_from_assay",
    "STRATEGY_IDS", "AssayBudget", "AssayOption", "StrategySelection", "combined_structure",
    "evaluate_selection_value", "strategy_constraint_sensitivity", "strategy_decision_value",
    "strategy_max_inclusion", "strategy_max_marginal_variance", "strategy_none", "strategy_oracle_reference",
    "strategy_random",
    "INFO_TYPES", "SCOPE_LABEL", "VALUE_SEMANTICS", "InformationStructure", "InformationValueResult", "OrderCheck", "build_likelihood",
    "ValueDefinition", "VALUE_DEFINITIONS", "VALUE_DEFINITION_ROLES", "LEGACY_VALUE_KIND_ALIASES",
    "resolve_value_definition",
    "COMPONENT_VARIANCE_BASES", "DECOMPOSITION_STATUSES", "METADATA_ORIGINS", "ComponentMetadata",
    "InconsistentDecompositionError", "MetadataConflictError", "StateMetadata", "VarianceDecomposition",
    "decompose_observed_variance", "resolve_truth_metadata",
    "BASIS_SOURCES", "ERROR_FIELD_COMPONENTS", "load_error_locators", "trace_error_provenance",
    "check_bin_refinement", "check_library_enlargement", "check_theoretical_order", "compute_information_value",
    "standard_order_relations",
    # FIX_A
    "VALUE_BINDINGS", "BatchAmount", "PerHeadDayValue", "ValueBinding",
    "IDENTIFYING_VERIFICATION_STATUSES", "METADATA_VERIFICATION_STATUSES", "TruthLink", "resolve_truth_link",
    "state_metadata_of_model", "verify_state_metadata", "TRUE_STATE_SD_MC_Z", "default_error_locators",
    "EXECUTABLE_RANDOMIZATION_STATUSES", "IDENTIFICATION_LABELS", "IDENTIFIED_LABELS",
    "RANDOMIZATION_CHANNEL_STATUSES", "SCENARIO_IDENTIFICATION_LABELS", "RandomizationChannelError",
    "randomization_channel_assessment", "require_executable_operational_value",
    # R3B
    "DEPRECATED_VALUE_DEFINITION_NAMES", "DEPRECATED_ASSESSMENT_KEYS", "HEURISTIC_VALUE_DEFINITIONS",
    "EXCLUDED_FROM_PAPER_MAIN_RESULTS", "require_operational_not_above_randomized_reference",
    # FIX3_BC (round-3 red team B-1)
    "MONEY_BASES", "RQ3_MONEY_IN_PAPER_MAIN_RESULTS", "DECISION_PROBLEM_ELEMENT_STATUSES",
    "INFORMATION_PROCESSING_CONVENTIONS", "RISK_TIMINGS", "ENGINE_RISK_TIMING", "CompleteDecisionProblem",
    "DecisionProblemElement", "DecisionProblemRequiredError",
]
