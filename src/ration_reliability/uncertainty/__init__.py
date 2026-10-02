"""Uncertainty models, draw sets and named random streams."""

from .base import DrawSet, UncertaintyModel, require_stream  # noqa: F401
from .models import IndependentNormalModel, PointMassModel, ScenarioSetModel  # noqa: F401
from .streams import (  # noqa: F401
    CORE_STREAMS,
    OfficialStreamToken,
    RandomStreams,
    ReservedRootError,
    is_reserved_root,
    reserved_formal_roots,
    stream_index,
)

__all__ = [
    "DrawSet",
    "UncertaintyModel",
    "require_stream",
    "IndependentNormalModel",
    "PointMassModel",
    "ScenarioSetModel",
    "CORE_STREAMS",
    "RandomStreams",
    "stream_index",
    "OfficialStreamToken",
    "ReservedRootError",
    "is_reserved_root",
    "reserved_formal_roots",
]

# --- appended by G2_phase3_uncertainty (2026-09-24): phase-3 candidate mechanisms (no parameter values) ---
from .distributions import (  # noqa: F401,E402
    ALL_FAMILIES,
    MOMENT_MATCHED_FAMILIES,
    MarginalFit,
    fit_array,
    fit_marginal,
)
from .correlation import (  # noqa: F401,E402
    CorrelationStructure,
    GaussianCopulaModel,
    NotPSDError,
    PendingValueError,
    assert_psd,
    check_psd,
    conservation_closure_structure,
    extreme_scaling,
    independent_structure,
    nearest_correlation,
    pairwise_structure,
    repair_correlation,
)
from .scaling import ScenarioTargets, build_scenario_targets, mean_shift, scale_sd  # noqa: F401,E402
from .information_states import (  # noqa: F401,E402
    INFORMATION_STATES,
    DecisionPrior,
    PendingDecisionError,
    TwoLayerFarmModel,
    assay_double_count_report,
    require_no_double_count,
)

__all__ += [
    "ALL_FAMILIES", "MOMENT_MATCHED_FAMILIES", "MarginalFit", "fit_array", "fit_marginal",
    "CorrelationStructure", "GaussianCopulaModel", "NotPSDError", "PendingValueError", "assert_psd", "check_psd",
    "conservation_closure_structure", "extreme_scaling", "independent_structure", "nearest_correlation",
    "pairwise_structure", "repair_correlation",
    "ScenarioTargets", "build_scenario_targets", "mean_shift", "scale_sd",
    "INFORMATION_STATES", "DecisionPrior", "PendingDecisionError", "TwoLayerFarmModel",
    "assay_double_count_report", "require_no_double_count",
]

# --- appended by K3_uncertainty_factory (round 2, review R3): the single model factory (exports only) ---
from .spec import (  # noqa: F401,E402
    FAMILY_TN_NAIVE_DIAGNOSTIC,
    REQUESTABLE_FAMILIES,
    SPEC_SCHEMA,
    CellSpec,
    CorrelationSpec,
    FamilyRule,
    TwoLayerSpec,
    UncertaintySpec,
)
from .factory import (  # noqa: F401,E402
    FACTORY_VERSION,
    CellModelMetadata,
    CorrelationMetadata,
    ExtensionScenario,
    FactoryBuildError,
    FactoryModel,
    FamilySelection,
    ModelMetadata,
    WorldMismatchError,
    build_uncertainty_model,
    calibrate_latent_rho,
    metadata_for_drawset,
    metadata_for_fingerprint,
    select_family_on_development_data,
    transformed_pearson,
)

__all__ += [
    "FAMILY_TN_NAIVE_DIAGNOSTIC", "REQUESTABLE_FAMILIES", "SPEC_SCHEMA", "CellSpec", "CorrelationSpec", "FamilyRule",
    "TwoLayerSpec", "UncertaintySpec",
    "FACTORY_VERSION", "CellModelMetadata", "CorrelationMetadata", "ExtensionScenario", "FactoryBuildError",
    "FactoryModel", "FamilySelection", "ModelMetadata", "WorldMismatchError", "build_uncertainty_model",
    "calibrate_latent_rho", "metadata_for_drawset", "metadata_for_fingerprint", "select_family_on_development_data",
    "transformed_pearson",
]
