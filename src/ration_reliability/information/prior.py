"""Weighted prior states for the finite-scenario information-value model (contract T7.1, T7.3).

The information-value calculations work on a *finite* representation of the prior p(theta):
``S`` joint states ``(theta_s [I, J], d_s [I])`` with weights ``pi_s`` (sum 1).  Two sources are
allowed:

* :meth:`PriorStates.from_drawset` -- Monte Carlo draws from an
  :class:`~ration_reliability.uncertainty.UncertaintyModel`.  During development only the ``opt``
  and ``validation`` streams are accepted (T5: candidate library, signal bins and policy
  construction use development data only).  ``test`` or ``outer`` draws raise
  :class:`~ration_reliability.errors.LeakageError`.
* :meth:`PriorStates.from_discrete` -- an explicitly enumerated discrete prior (used for
  analytically solvable unit tests and for exact small-problem checks).

``theta_s`` is the state of the *current batch* that will be fed during the covered period
(T1, T7.5).  Its variance must be the true batch-to-batch variance, not an observed variance that
already contains sampling/laboratory error (see :func:`~.signal.double_count_guard`).

Object metadata (second review R2, 2026-09-25)
----------------------------------------------
What the SD behind the states *represents* is part of the object, not a string handed to the
information-value functions later.  Per component (:class:`ComponentMetadata`):
``variance_basis`` (``true_batch_state`` / ``observed_incl_sampling_and_lab`` /
``observed_incl_lab_only`` / ``unidentified``; ``not_applicable`` for point and missing cells),
``data_fingerprint`` (hash of the source row), ``decomposition_id`` / ``decomposition_source`` (the
variance decomposition applied, ``"none"`` if none), ``measurement_model_id`` (which measurement
process the SD contains, or which one a decomposition removed), ``is_synthetic`` and the provenance
(``provenance_status``, ``source_id``, ``locator``).  The vocabulary and the validation rules are
those of :mod:`ration_reliability.uncertainty.spec` (review R3), so factory metadata converts
without translation.

* :meth:`PriorStates.from_drawset` binds the metadata of the generating
  :class:`~ration_reliability.uncertainty.factory.FactoryModel` automatically (registry lookup by the
  draws' ``model_fingerprint``, or ``model=``).  An explicit ``metadata=`` that contradicts the
  factory metadata raises :class:`MetadataConflictError`; for a non-factory model an explicit
  ``metadata=`` is accepted and becomes part of the object.
* :meth:`PriorStates.from_discrete` takes ``metadata=`` as part of the discrete definition.
* The metadata enter :attr:`PriorStates.fingerprint`, hence the fingerprints of every likelihood
  and risk table built on the prior: changing the metadata means a different prior.

:class:`StateMetadata` is also the *truth record* (the model that generates the true state that
the value is evaluated against); :func:`resolve_truth_metadata` links it to a prior.  In the
finite information-value model the risk table is computed on the prior states, so the truth is the
generator of the prior states: a truth record is *linked* only when it is the prior's own metadata
or carries the same generating-model fingerprint; a different generating model raises.

:class:`VarianceDecomposition` records ``observed = true + error`` and is never clipped: a negative
implied true variance has status ``inconsistent_negative`` and ``true_sd = None``.

Verification of object metadata (red-team FIX_A, 2026-09-25)
------------------------------------------------------------
Object metadata only *identify* a value when they can be checked against an object that the caller
cannot copy by hand (:func:`verify_state_metadata`):

* ``factory_model_metadata`` is re-derived from the factory registry of the generating model (for a
  column-appending wrapper such as :class:`~ration_reliability.nutrition.energy.EnergyColumnModel`,
  from its base model, ``derived_from_model_fingerprint``) and must be identical ->
  ``factory_registry_verified``; a contradiction raises; a registry miss is
  ``factory_claim_unverifiable_in_this_process`` (never identifies);
* explicit metadata (``explicit_prior_definition`` / ``explicit_truth_record``) of a *synthetic* world
  are the definition of that test world -> ``synthetic_world_definition``; of a non-synthetic world they
  cannot be verified -> ``explicit_non_factory_unverifiable`` (at most ``unidentified_scenario``).

A truth record is *linked* to the prior when it claims the same world; it may serve as the basis
source of the guard only when the link is *verified* (:func:`resolve_truth_link`): the prior's own
states or metadata, a model object whose fingerprint is computed (not copied), or factory metadata
verified against the registry.  A copied ``model_fingerprint`` is a claim, not a verification.  A truth
given as :class:`PriorStates` must be the prior states themselves or states verifiably drawn from the
same generating model; different states are another world and raise.

:meth:`PriorStates.from_drawset` resolves the metadata of non-synthetic draws from the generating
factory model (registry, ``model=``, or ``model=`` a column-appending wrapper whose ``base`` is a
factory model); if they cannot be resolved it raises instead of returning ``metadata=None``, and
explicit metadata are never accepted for non-synthetic draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from ..errors import InvalidProblemError, LeakageError
from ..hashing import stable_hash
from ..uncertainty.base import DrawSet
from ..uncertainty.spec import NO_DECOMPOSITION, NOT_APPLICABLE_BASIS, PROVENANCE_STATUSES, VARIANCE_BASES

__all__ = [
    "DM_COMPONENT",
    "DEVELOPMENT_STREAMS",
    "METADATA_ORIGINS",
    "COMPONENT_VARIANCE_BASES",
    "DECOMPOSITION_STATUSES",
    "MetadataConflictError",
    "InconsistentDecompositionError",
    "ObservedComponent",
    "VarianceDecomposition",
    "decompose_observed_variance",
    "ComponentMetadata",
    "StateMetadata",
    "PriorStates",
    "METADATA_VERIFICATION_STATUSES",
    "IDENTIFYING_VERIFICATION_STATUSES",
    "verify_state_metadata",
    "state_metadata_of_model",
    "TruthLink",
    "resolve_truth_link",
    "resolve_truth_metadata",
]

#: Component name used for the DM fraction ``d_i`` of an ingredient.
DM_COMPONENT = "DM"

#: Streams on which the information-value *development* objects may be built (T5, T7.3).
DEVELOPMENT_STREAMS: tuple[str, ...] = ("opt", "validation")

#: Stream label of an explicitly enumerated (exact) discrete prior.
DISCRETE_PRIOR_STREAM = "discrete_exact_prior"

#: Where a :class:`StateMetadata` comes from.
METADATA_ORIGINS = (
    "factory_model_metadata",       # read from a FactoryModel (UncertaintySpec cells; part of the world id)
    "explicit_prior_definition",    # given when the prior object is constructed (bound into its fingerprint)
    "explicit_truth_record",        # a stand-alone truth record (linked only through a model fingerprint)
)
#: Variance bases a component may carry (``not_applicable`` for point and missing cells).
COMPONENT_VARIANCE_BASES: tuple[str, ...] = tuple(VARIANCE_BASES) + (NOT_APPLICABLE_BASIS,)
#: Status of a :class:`VarianceDecomposition`.
DECOMPOSITION_STATUSES = ("consistent", "degenerate_no_true_variation", "inconsistent_negative")
#: Result of :func:`verify_state_metadata` (FIX_A).
METADATA_VERIFICATION_STATUSES = (
    "factory_registry_verified",                  # re-derived from the generating factory model: identical
    "factory_claim_unverifiable_in_this_process",  # claims factory origin, registry has no such model here
    "factory_claim_contradicted",                  # claims factory origin, registry says otherwise (raises)
    "synthetic_world_definition",                  # explicit metadata of a synthetic test world (its definition)
    "explicit_non_factory_unverifiable",           # explicit metadata of a non-synthetic world: never identify
)
#: Verification statuses under which object metadata may identify a value.
IDENTIFYING_VERIFICATION_STATUSES = ("factory_registry_verified", "synthetic_world_definition")


class MetadataConflictError(InvalidProblemError):
    """An external declaration (string mapping, report or explicit metadata) contradicts the object
    metadata of the prior or of the linked truth model (review R2: declarations never override objects)."""


class InconsistentDecompositionError(InvalidProblemError):
    """``observed variance - error variance < 0``: the decomposition is rejected, never clipped to 0."""


@dataclass(frozen=True, order=True)
class ObservedComponent:
    """One scalar unknown of the batch state: ``(ingredient_id, component)``.

    ``component`` is a nutrient id (value ``theta[i, j]``, canonical unit, DM basis) or
    ``"DM"`` (value ``d[i]``, kg DM / kg as-fed).
    """

    ingredient_id: str
    component: str

    def label(self) -> str:
        """Printable id, e.g. ``corn_silage:NDF``."""
        return f"{self.ingredient_id}:{self.component}"


def _comp(key: Any) -> ObservedComponent:
    """``ObservedComponent`` from an ``ObservedComponent``, a ``(ingredient, component)`` pair or ``"i:c"``."""
    if isinstance(key, ObservedComponent):
        return key
    if isinstance(key, str) and ":" in key:
        i, c = key.split(":", 1)
        return ObservedComponent(i, c)
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return ObservedComponent(str(key[0]), str(key[1]))
    if hasattr(key, "ingredient_id") and hasattr(key, "component"):
        return ObservedComponent(str(key.ingredient_id), str(key.component))
    raise InvalidProblemError(f"cannot interpret {key!r} as an observed component")


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _txt(x: Any) -> str:
    return str(x or "").strip()


# ------------------------------------------------------------------------------------------------
# variance decomposition (never clipped)
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VarianceDecomposition:
    """``observed_sd^2 = true_variance + sampling_sd^2 / n_field_samples + lab_sd^2`` (T7.5).

    ``implied_true_variance`` is the raw difference and may be negative; it is never replaced by 0.
    ``status``: ``consistent`` (> tolerance), ``degenerate_no_true_variation`` (|v| <= tolerance: all
    observed variation is measurement error) or ``inconsistent_negative`` (the error model claims more
    variance than was observed; :attr:`true_sd` is ``None`` and :meth:`require_consistent` raises).
    """

    observed_sd: float
    sampling_sd: float
    lab_sd: float
    n_field_samples: int
    error_variance: float
    implied_true_variance: float
    status: str
    tolerance: float
    component: Optional[ObservedComponent] = None
    decomposition_id: Optional[str] = None
    decomposition_source: Optional[str] = None
    measurement_model_id: Optional[str] = None

    @property
    def true_sd(self) -> Optional[float]:
        """De-convolved SD, ``None`` if the decomposition is inconsistent (never a clipped 0)."""
        if self.status == "inconsistent_negative":
            return None
        v = self.implied_true_variance
        if self.status == "degenerate_no_true_variation":
            # |v| within the round-off tolerance (e.g. 0.05**2 - 0.05**2); flagged by the status
            return math.sqrt(v) if v > 0 else 0.0
        return math.sqrt(v)

    def require_consistent(self) -> float:
        """The de-convolved SD; raises :class:`InconsistentDecompositionError` if inconsistent."""
        if self.status == "inconsistent_negative":
            where = "" if self.component is None else f"{self.component.label()}: "
            raise InconsistentDecompositionError(
                f"{where}inconsistent decomposition: error variance {self.error_variance:g} exceeds observed variance "
                f"{self.observed_sd ** 2:g} (implied true variance {self.implied_true_variance:g}); rejected, not "
                "clipped to 0 -- report the component as unidentified or revisit the error model")
        return float(self.true_sd)

    def fingerprint(self) -> str:
        return stable_hash("VarianceDecomposition/v1", self)

    def to_dict(self) -> dict:
        return {"component": None if self.component is None else self.component.label(),
                "observed_sd": self.observed_sd, "sampling_sd": self.sampling_sd, "lab_sd": self.lab_sd,
                "n_field_samples": self.n_field_samples, "error_variance": self.error_variance,
                "implied_true_variance": self.implied_true_variance, "status": self.status,
                "true_sd": self.true_sd, "tolerance": self.tolerance, "decomposition_id": self.decomposition_id,
                "decomposition_source": self.decomposition_source,
                "measurement_model_id": self.measurement_model_id}


def decompose_observed_variance(observed_sd: float, *, sampling_sd: float = 0.0, lab_sd: float = 0.0,
                                n_field_samples: int = 1, component: Any = None,
                                decomposition_id: Optional[str] = None, decomposition_source: Optional[str] = None,
                                measurement_model_id: Optional[str] = None,
                                tolerance: float = 1e-15) -> VarianceDecomposition:
    """Record the decomposition of an observed SD of historical single results (never raises on a
    negative result, never clips; see :class:`VarianceDecomposition`).

    ``observed_sd`` is the SD of results that each came from ``n_field_samples`` composited field
    samples analysed once; ``lab_sd`` is the total SD of one analysis.
    """
    vals = {}
    for n, v in (("observed_sd", observed_sd), ("sampling_sd", sampling_sd), ("lab_sd", lab_sd),
                 ("tolerance", tolerance)):
        v = float(v)
        if not math.isfinite(v) or v < 0:
            raise InvalidProblemError(f"decompose_observed_variance: {n} must be finite and >= 0, got {v}")
        vals[n] = v
    if int(n_field_samples) < 1:
        raise InvalidProblemError("decompose_observed_variance: n_field_samples must be >= 1")
    o, s, l, tol = vals["observed_sd"], vals["sampling_sd"], vals["lab_sd"], vals["tolerance"]
    err = s * s / int(n_field_samples) + l * l
    v = o * o - err
    status = "consistent" if v > tol else ("degenerate_no_true_variation" if v >= -tol else "inconsistent_negative")
    return VarianceDecomposition(o, s, l, int(n_field_samples), err, v, status, tol,
                                 None if component is None else _comp(component), decomposition_id,
                                 decomposition_source, measurement_model_id)


# ------------------------------------------------------------------------------------------------
# per-component and per-model metadata
# ------------------------------------------------------------------------------------------------

#: Fields that must agree between the prior end and the truth end of one component.
_BINDING_FIELDS = ("variance_basis", "data_fingerprint", "decomposition_id", "decomposition_source",
                   "measurement_model_id", "is_synthetic")


@dataclass(frozen=True)
class ComponentMetadata:
    """Object metadata of one component of a state-generating model (prior or truth).

    Validation follows :mod:`ration_reliability.uncertainty.spec` (review R3): an observed basis names
    its ``measurement_model_id``; a non-synthetic ``true_batch_state`` needs a variance decomposition
    (``decomposition_id`` + ``decomposition_source`` + the removed ``measurement_model_id``); a
    decomposition needs a source; a stochastic basis needs a ``data_fingerprint``.  An attached
    :class:`VarianceDecomposition` must be consistent (a negative decomposition cannot certify a
    true-state SD).
    """

    component: ObservedComponent
    variance_basis: str
    data_fingerprint: Optional[str]
    decomposition_id: str
    decomposition_source: Optional[str]
    measurement_model_id: Optional[str]
    is_synthetic: bool
    provenance_status: Optional[str] = None
    source_id: Optional[str] = None
    locator: Optional[str] = None
    decomposition: Optional[VarianceDecomposition] = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _comp(self.component))
        object.__setattr__(self, "is_synthetic", bool(self.is_synthetic))
        where = f"metadata {self.component.label()}"
        vb = self.variance_basis
        issues: list[str] = []
        if vb not in COMPONENT_VARIANCE_BASES:
            issues.append(f"variance_basis {vb!r} must be one of {COMPONENT_VARIANCE_BASES}")
        dec = _txt(self.decomposition_id)
        if not dec:
            issues.append(f"decomposition_id required ('{NO_DECOMPOSITION}' when no decomposition was applied)")
        if dec and dec != NO_DECOMPOSITION and not _txt(self.decomposition_source):
            issues.append(f"decomposition {dec!r} needs decomposition_source")
        stochastic = vb in VARIANCE_BASES
        if stochastic and not _txt(self.data_fingerprint):
            issues.append("data_fingerprint required for a stochastic component (hash of the source row, or "
                          "'synthetic:<label>')")
        if vb in ("observed_incl_sampling_and_lab", "observed_incl_lab_only") and not _txt(self.measurement_model_id):
            issues.append(f"variance_basis {vb!r} needs measurement_model_id (which measurement process the SD "
                          "contains)")
        if vb == "true_batch_state" and not self.is_synthetic and dec == NO_DECOMPOSITION:
            issues.append("true_batch_state for non-synthetic data needs a variance decomposition (decomposition_id + "
                          "decomposition_source); an observed SD is not a true-state SD by declaration")
        if vb == "true_batch_state" and dec and dec != NO_DECOMPOSITION and not _txt(self.measurement_model_id):
            issues.append("a decomposition to true_batch_state must name the removed measurement model "
                          "(measurement_model_id)")
        ps = self.provenance_status
        if ps is not None:
            if ps not in PROVENANCE_STATUSES:
                issues.append(f"provenance_status must be one of {PROVENANCE_STATUSES}")
            elif ps == "sourced" and not (_txt(self.source_id) and _txt(self.locator)):
                issues.append("sourced values need source_id and locator")
            elif ps == "synthetic_test_only" and not self.is_synthetic:
                issues.append("synthetic_test_only provenance on non-synthetic metadata")
        rec = self.decomposition
        if rec is not None:
            if vb != "true_batch_state":
                issues.append(f"a decomposition record belongs to a true_batch_state component, not {vb!r}")
            if rec.component is not None and rec.component != self.component:
                issues.append(f"decomposition record is for {rec.component.label()}")
            if rec.decomposition_id is not None and rec.decomposition_id != dec:
                issues.append(f"decomposition record id {rec.decomposition_id!r} != decomposition_id {dec!r}")
            if rec.measurement_model_id is not None and rec.measurement_model_id != self.measurement_model_id:
                issues.append(f"decomposition record removed measurement model {rec.measurement_model_id!r}, metadata "
                              f"says {self.measurement_model_id!r}")
        if issues:
            raise InvalidProblemError(f"{where}: " + "; ".join(issues))
        if rec is not None and rec.status == "inconsistent_negative":
            raise InconsistentDecompositionError(
                f"{where}: the attached variance decomposition is inconsistent (implied true variance "
                f"{rec.implied_true_variance:g} < 0); a true_batch_state SD cannot be certified by it -- rejected, "
                "not clipped to 0")

    # ---- accessors -----------------------------------------------------------------------
    @property
    def is_stochastic(self) -> bool:
        return self.variance_basis in VARIANCE_BASES

    @property
    def deconvolved(self) -> bool:
        """True when the SD is a true-state SD obtained by an explicit decomposition."""
        return self.variance_basis == "true_batch_state" and self.decomposition_id != NO_DECOMPOSITION

    def binding_key(self) -> tuple:
        """Fields that must agree between the prior end and the truth end."""
        return tuple(getattr(self, f) for f in _BINDING_FIELDS)

    def fingerprint(self) -> str:
        return stable_hash("ComponentMetadata/v1", self)

    def to_dict(self) -> dict:
        return {"component": self.component.label(), "variance_basis": self.variance_basis,
                "data_fingerprint": self.data_fingerprint, "decomposition_id": self.decomposition_id,
                "decomposition_source": self.decomposition_source,
                "measurement_model_id": self.measurement_model_id, "is_synthetic": self.is_synthetic,
                "provenance_status": self.provenance_status, "source_id": self.source_id, "locator": self.locator,
                "decomposition": None if self.decomposition is None else self.decomposition.to_dict(),
                "note": self.note}

    @classmethod
    def from_mapping(cls, component: Any, fields: Mapping[str, Any]) -> "ComponentMetadata":
        """From a mapping of the field names (unknown keys raise; no hidden defaults for the core fields)."""
        allowed = {"variance_basis", "data_fingerprint", "decomposition_id", "decomposition_source",
                   "measurement_model_id", "is_synthetic", "provenance_status", "source_id", "locator",
                   "decomposition", "note"}
        extra = sorted(set(fields) - allowed)
        if extra:
            raise InvalidProblemError(f"metadata {_comp(component).label()}: unknown key(s) {extra}")
        req = ("variance_basis", "data_fingerprint", "decomposition_id", "decomposition_source",
               "measurement_model_id", "is_synthetic")
        miss = [k for k in req if k not in fields]
        if miss:
            raise InvalidProblemError(f"metadata {_comp(component).label()}: missing field(s) {miss} (no hidden "
                                      "defaults)")
        return cls(_comp(component), **dict(fields))

    @classmethod
    def from_cell(cls, cell: Any) -> "ComponentMetadata":
        """From a factory :class:`~ration_reliability.uncertainty.factory.CellModelMetadata` (object metadata).

        A cell that is not sampled (point, missing, excluded) gets ``variance_basis='not_applicable'``;
        the spec basis and fit status are kept in ``note``.
        """
        stochastic = bool(getattr(cell, "is_stochastic", False))
        vb = cell.variance_basis if stochastic else NOT_APPLICABLE_BASIS
        note = "" if stochastic else (f"not sampled (fit_status={getattr(cell, 'fit_status', None)}, "
                                      f"exclusion={getattr(cell, 'exclusion_action', None)}, "
                                      f"spec variance_basis={cell.variance_basis})")
        return cls(ObservedComponent(cell.ingredient_id, cell.component), vb, cell.data_fingerprint,
                   cell.decomposition_id, cell.decomposition_source, cell.measurement_model_id,
                   bool(cell.is_synthetic), cell.provenance_status, cell.source_id, cell.locator, None, note)


@dataclass(frozen=True)
class StateMetadata:
    """Object metadata of one state-generating model: the prior's generator or the truth model.

    ``model_fingerprint`` is the world id of the generating model (``DrawSet.model_fingerprint``) when
    there is one; ``metadata_fingerprint`` the fingerprint of the factory :class:`ModelMetadata`.
    ``derived_from_model_fingerprint`` (FIX_A) is the world id of the factory model whose cells these
    are when the states come from a column-appending wrapper of it (e.g. ``EnergyColumnModel``); the
    appended columns carry no metadata.
    """

    components: tuple[ComponentMetadata, ...]
    origin: str
    model_id: Optional[str] = None
    model_fingerprint: Optional[str] = None
    metadata_fingerprint: Optional[str] = None
    spec_id: Optional[str] = None
    spec_main_fingerprint: Optional[str] = None
    is_synthetic: bool = False
    is_diagnostic: bool = False
    derived_from_model_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        comps = tuple(self.components)
        object.__setattr__(self, "components", comps)
        if self.origin not in METADATA_ORIGINS:
            raise InvalidProblemError(f"StateMetadata.origin must be one of {METADATA_ORIGINS}")
        seen = [c.component for c in comps]
        if len(set(seen)) != len(seen):
            raise InvalidProblemError("StateMetadata: duplicate component")
        if any(not isinstance(c, ComponentMetadata) for c in comps):
            raise InvalidProblemError("StateMetadata.components must be ComponentMetadata")
        if any(c.is_synthetic for c in comps) and not self.is_synthetic:
            raise InvalidProblemError("StateMetadata: synthetic component metadata in non-synthetic metadata")

    # ---- access --------------------------------------------------------------------------
    def get(self, comp: Any) -> Optional[ComponentMetadata]:
        c = _comp(comp)
        for m in self.components:
            if m.component == c:
                return m
        return None

    def basis_map(self, stochastic_only: bool = True) -> dict[ObservedComponent, str]:
        """``ObservedComponent -> variance_basis`` (the object's, not a declaration)."""
        return {m.component: m.variance_basis for m in self.components if m.is_stochastic or not stochastic_only}

    def fingerprint(self) -> str:
        return stable_hash("StateMetadata/v1", self)

    def to_dict(self) -> dict:
        return {"origin": self.origin, "model_id": self.model_id, "model_fingerprint": self.model_fingerprint,
                "metadata_fingerprint": self.metadata_fingerprint, "spec_id": self.spec_id,
                "spec_main_fingerprint": self.spec_main_fingerprint, "is_synthetic": self.is_synthetic,
                "is_diagnostic": self.is_diagnostic,
                "derived_from_model_fingerprint": self.derived_from_model_fingerprint,
                "state_metadata_fingerprint": self.fingerprint(),
                "components": [c.to_dict() for c in self.components]}

    def trace_rows(self, stochastic_only: bool = True) -> list[dict]:
        """One row per component: basis, decomposition, measurement model and the provenance chain
        (data fingerprint -> source id -> locator) for run records (no numeric values)."""
        rows = []
        for m in self.components:
            if stochastic_only and not m.is_stochastic:
                continue
            rows.append({"component": m.component.label(), "variance_basis": m.variance_basis,
                         "data_fingerprint": m.data_fingerprint, "provenance_status": m.provenance_status,
                         "source_id": m.source_id, "locator": m.locator, "decomposition_id": m.decomposition_id,
                         "decomposition_source": m.decomposition_source,
                         "measurement_model_id": m.measurement_model_id, "is_synthetic": m.is_synthetic,
                         "model_fingerprint": self.model_fingerprint, "origin": self.origin,
                         "derived_from_model_fingerprint": self.derived_from_model_fingerprint})
        return rows

    def restricted_to(self, ingredient_ids: Sequence[str], nutrient_ids: Sequence[str]) -> "StateMetadata":
        """Only the components on the given axes."""
        ing, nut = set(ingredient_ids), set(nutrient_ids) | {DM_COMPONENT}
        keep = tuple(m for m in self.components
                     if m.component.ingredient_id in ing and m.component.component in nut)
        return replace(self, components=keep)

    # ---- constructors --------------------------------------------------------------------
    @classmethod
    def from_model_metadata(cls, md: Any, *, model_fingerprint: Optional[str],
                            model_id: Optional[str] = None) -> "StateMetadata":
        """From a factory :class:`~ration_reliability.uncertainty.factory.ModelMetadata` (cells on the model
        axes only; excluded ingredients are not part of the world)."""
        axes = set(md.ingredient_ids)
        comps = tuple(ComponentMetadata.from_cell(c) for c in md.cells if c.ingredient_id in axes)
        return cls(comps, "factory_model_metadata", model_id or md.model_id, model_fingerprint, md.fingerprint(),
                   md.spec_id, md.spec_main_fingerprint, bool(md.is_synthetic), bool(md.is_diagnostic))

    @classmethod
    def from_factory_model(cls, model: Any) -> "StateMetadata":
        """From a :class:`~ration_reliability.uncertainty.factory.FactoryModel` (its world id is the fingerprint)."""
        return cls.from_model_metadata(model.metadata, model_fingerprint=model.fingerprint(), model_id=model.model_id)

    @classmethod
    def from_mapping(cls, mapping: Mapping[Any, Union[ComponentMetadata, Mapping[str, Any]]], *, origin: str,
                     is_synthetic: bool, model_id: Optional[str] = None,
                     model_fingerprint: Optional[str] = None) -> "StateMetadata":
        """From ``{component: ComponentMetadata | field mapping}``."""
        comps = []
        for k, v in mapping.items():
            if isinstance(v, ComponentMetadata):
                if v.component != _comp(k):
                    raise InvalidProblemError(f"metadata key {k!r} holds metadata of {v.component.label()}")
                comps.append(v)
            else:
                comps.append(ComponentMetadata.from_mapping(k, v))
        return cls(tuple(comps), origin, model_id, model_fingerprint, None, None, None, bool(is_synthetic))

    @classmethod
    def synthetic(cls, components: Iterable[Any], variance_basis: str, *, label: str,
                  measurement_model_id: Optional[str] = None,
                  origin: str = "explicit_prior_definition") -> "StateMetadata":
        """Synthetic unit-test metadata (``data_fingerprint = 'synthetic:<label>'``, no decomposition,
        provenance ``synthetic_test_only``).  Not for real data."""
        mm = measurement_model_id
        if mm is None and variance_basis in ("observed_incl_sampling_and_lab", "observed_incl_lab_only"):
            mm = f"synthetic_measurement:{label}"
        comps = tuple(ComponentMetadata(_comp(c), variance_basis, f"synthetic:{label}", NO_DECOMPOSITION, None, mm,
                                        True, "synthetic_test_only", f"SYNTHETIC:{label}", None)
                      for c in components)
        return cls(comps, origin, None, None, None, None, None, True)


def _is_factory_model(obj: Any) -> bool:
    md = getattr(obj, "metadata", None)
    return md is not None and hasattr(md, "cells") and callable(getattr(obj, "fingerprint", None))


def _is_column_appending_wrapper(obj: Any) -> bool:
    """Known wrappers that append derived columns and leave the base columns untouched (FIX_A).

    Only :class:`~ration_reliability.nutrition.energy.EnergyColumnModel` qualifies: its ``sample``
    returns the base draws unchanged plus the ``NEL_fixedDMI`` column.  Unknown wrappers are not
    resolved (their base metadata might not describe the transformed columns)."""
    try:
        from ..nutrition.energy import EnergyColumnModel   # lazy: information does not import nutrition eagerly
    except Exception:                                        # pragma: no cover - nutrition always present
        return False
    base = getattr(obj, "base", None)
    return (isinstance(obj, EnergyColumnModel) and base is not None
            and tuple(obj.ingredient_ids) == tuple(base.ingredient_ids)
            and tuple(obj.nutrient_ids[:len(base.nutrient_ids)]) == tuple(base.nutrient_ids))


def state_metadata_of_model(model: Any) -> Optional[StateMetadata]:
    """Object metadata of a model *object* (verifiable: the fingerprint is computed, not copied).

    * a :class:`~ration_reliability.uncertainty.factory.FactoryModel` -> its cells;
    * a column-appending wrapper of one (``EnergyColumnModel``) -> the base model's cells with
      ``model_fingerprint`` = the wrapper's world id and ``derived_from_model_fingerprint`` = the base
      world id; the appended columns carry no metadata;
    * anything else -> ``None``.
    """
    if _is_factory_model(model):
        return StateMetadata.from_factory_model(model)
    if _is_column_appending_wrapper(model):
        base_md = state_metadata_of_model(model.base)
        if base_md is None:
            return None
        return replace(base_md, model_id=getattr(model, "model_id", base_md.model_id),
                       model_fingerprint=model.fingerprint(),
                       derived_from_model_fingerprint=base_md.derived_from_model_fingerprint
                       or base_md.model_fingerprint)
    return None


def verify_state_metadata(md: StateMetadata) -> tuple[str, list[str]]:
    """``(status, issues)``: can these object metadata be checked against an object (FIX_A)?

    See :data:`METADATA_VERIFICATION_STATUSES`.  Factory metadata are re-derived from the factory
    registry of this process (by ``derived_from_model_fingerprint`` or ``model_fingerprint``) and must
    be identical component by component (every field, including provenance)."""
    if md.origin == "factory_model_metadata":
        key = md.derived_from_model_fingerprint or md.model_fingerprint
        if not key:
            return ("factory_claim_unverifiable_in_this_process",
                    ["factory-origin metadata without a generating-model fingerprint"])
        from ..uncertainty.factory import metadata_for_fingerprint   # lazy: keep the import graph acyclic

        fmd = metadata_for_fingerprint(key)
        if fmd is None:
            return ("factory_claim_unverifiable_in_this_process",
                    [f"no factory model {key[:12]} is registered in this process; the metadata cannot be re-derived "
                     "(pass the model object, or rebuild the model from its spec)"])
        issues = []
        if md.metadata_fingerprint is not None and md.metadata_fingerprint != fmd.fingerprint():
            issues.append("metadata_fingerprint differs from the registered factory model")
        obj = StateMetadata.from_model_metadata(fmd, model_fingerprint=key)
        for m in md.components:
            b = obj.get(m.component)
            if b is None:
                issues.append(f"{m.component.label()}: not a cell of the registered factory model")
            elif b.fingerprint() != m.fingerprint():
                issues.append(f"{m.component.label()}: differs from the registered factory model's cell metadata")
        if issues:
            return "factory_claim_contradicted", issues
        return "factory_registry_verified", []
    if md.is_synthetic:
        return "synthetic_world_definition", []
    return ("explicit_non_factory_unverifiable",
            [f"explicit ({md.origin}) metadata of a non-synthetic world cannot be checked against a generating object; "
             "only factory-model metadata can identify a value for real data"])


def _as_state_metadata(obj: Any) -> StateMetadata:
    """``StateMetadata`` of a truth argument (``StateMetadata``, ``PriorStates``, a ``FactoryModel`` or a
    column-appending wrapper of one)."""
    if isinstance(obj, StateMetadata):
        return obj
    if isinstance(obj, PriorStates):
        if obj.metadata is None:
            raise InvalidProblemError("truth: the given PriorStates carry no object metadata")
        return obj.metadata
    md = state_metadata_of_model(obj)
    if md is not None:
        return md
    raise InvalidProblemError(f"truth must be StateMetadata, PriorStates, a FactoryModel or an EnergyColumnModel of "
                              f"one, got {type(obj).__name__}")


def _factory_metadata_for(draws: DrawSet, model: Any = None) -> Optional[StateMetadata]:
    if model is not None:
        if not callable(getattr(model, "fingerprint", None)) or model.fingerprint() != draws.model_fingerprint:
            raise InvalidProblemError(f"PriorStates.from_drawset: draws {draws.stream_id} were not generated by model "
                                      f"{getattr(model, 'model_id', model)!r} (model fingerprint mismatch)")
        return state_metadata_of_model(model)
    from ..uncertainty.factory import metadata_for_drawset   # lazy: keep the import graph acyclic

    md = metadata_for_drawset(draws)
    if md is None:
        return None
    return StateMetadata.from_model_metadata(md, model_fingerprint=draws.model_fingerprint)


# ------------------------------------------------------------------------------------------------
# the prior
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorStates:
    """Finite weighted representation of the prior over the batch state.

    Attributes
    ----------
    theta : ``[S, I, J]`` composition (canonical units, DM basis; NaN = missing).
    d : ``[S, I]`` DM fraction.
    weights : ``[S]`` non-negative, summing to 1.
    stream, stream_id : provenance of the states (``opt`` / ``validation`` /
        ``discrete_exact_prior`` ...).
    source_fingerprint : fingerprint of the originating draw set or discrete definition.
    is_synthetic : True for test-only parameters.
    metadata : object metadata per component (review R2; ``None`` = none bound -- the double-count
        guard can then only run on caller declarations and never reports an identified value).
    source_model_fingerprint : world id of the generating model (draw sets only).
    """

    theta: np.ndarray
    d: np.ndarray
    weights: np.ndarray
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    stream: str
    stream_id: str
    source_fingerprint: str
    is_synthetic: bool
    metadata: Optional[StateMetadata] = None
    source_model_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        th, d, w = _ro(self.theta), _ro(self.d), _ro(self.weights)
        object.__setattr__(self, "theta", th)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))
        if th.ndim != 3 or d.ndim != 2 or th.shape[:2] != d.shape:
            raise InvalidProblemError("PriorStates: theta must be [S,I,J] and d [S,I]")
        if th.shape[1:] != (len(self.ingredient_ids), len(self.nutrient_ids)):
            raise InvalidProblemError("PriorStates: axis labels do not match array shapes")
        if w.shape != (th.shape[0],):
            raise InvalidProblemError("PriorStates: weights must have shape [S]")
        if th.shape[0] < 1:
            raise InvalidProblemError("PriorStates: at least one state is required")
        if np.any(~np.isfinite(w)) or np.any(w < 0) or abs(float(w.sum()) - 1.0) > 1e-9:
            raise InvalidProblemError("PriorStates: weights must be finite, >= 0 and sum to 1")
        if self.metadata is not None:
            self._check_metadata(self.metadata)

    def _check_metadata(self, md: StateMetadata) -> None:
        if not isinstance(md, StateMetadata):
            raise InvalidProblemError("PriorStates.metadata must be StateMetadata")
        issues = []
        for m in md.components:
            c = m.component
            if c.ingredient_id not in self.ingredient_ids or (c.component != DM_COMPONENT
                                                              and c.component not in self.nutrient_ids):
                issues.append(f"{c.label()} is not on the prior axes")
                continue
            if not m.is_stochastic:
                col = self._column(c)
                fin = col[np.isfinite(col)]
                if fin.size and float(np.max(fin) - np.min(fin)) > 0.0:
                    issues.append(f"{c.label()}: metadata says '{m.variance_basis}' (point/missing) but the states vary")
        if md.is_synthetic and not self.is_synthetic:
            issues.append("synthetic metadata on a non-synthetic prior")
        if (md.model_fingerprint is not None and self.source_model_fingerprint is not None
                and md.model_fingerprint != self.source_model_fingerprint):
            issues.append("metadata belong to another generating model (model fingerprint mismatch)")
        if md.origin == "factory_model_metadata":
            # FIX_A: factory-origin metadata must describe states drawn from that model and must not
            # contradict the registered factory object (direct construction cannot forge them)
            if self.source_model_fingerprint is None:
                issues.append("factory-model metadata on states that were not drawn from a model (discrete prior): "
                              "use explicit_prior_definition metadata")
            else:
                vstat, vissues = verify_state_metadata(md)
                if vstat == "factory_claim_contradicted":
                    issues.append("metadata claim factory origin but contradict the registered factory model ("
                                  + "; ".join(vissues) + "); an external declaration cannot override the object")
        if self.source_model_fingerprint is not None and md.origin != "factory_model_metadata":
            # states drawn from a factory model: metadata attached any other way (direct construction,
            # dataclasses.replace) must not contradict the factory object
            from ..uncertainty.factory import metadata_for_fingerprint   # lazy: keep the import graph acyclic

            fmd = metadata_for_fingerprint(self.source_model_fingerprint)
            if fmd is not None:
                obj = StateMetadata.from_model_metadata(fmd, model_fingerprint=self.source_model_fingerprint)
                for m in md.components:
                    b = obj.get(m.component)
                    if b is None or b.binding_key() != m.binding_key():
                        issues.append(f"{m.component.label()}: metadata contradict the generating factory model's "
                                      "object metadata (an external declaration cannot override the object)")
        if issues:
            raise MetadataConflictError("PriorStates metadata: " + "; ".join(issues))

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_drawset(cls, draws: DrawSet, *, allowed_streams: Collection[str] = DEVELOPMENT_STREAMS,
                     consumer: str = "information.development", model: Any = None,
                     metadata: Optional[StateMetadata] = None) -> "PriorStates":
        """Uniformly weighted states from a :class:`DrawSet`.

        Raises :class:`LeakageError` if the draws come from a stream the consumer may not use
        (default: only ``opt`` / ``validation``).  Evaluation of a *frozen* policy on ``test``
        draws goes through :func:`ration_reliability.information.evaluate_frozen_policy`, which
        takes the :class:`DrawSet` directly.

        Metadata (R2): the object metadata of the generating factory model are bound automatically
        (``model=`` or the factory registry by ``draws.model_fingerprint``).  ``metadata=`` is accepted
        for a non-factory model (it becomes part of this object) and must equal the factory metadata
        otherwise (:class:`MetadataConflictError`); it may not name another generating model.

        FIX_A: ``model=`` may also be a column-appending wrapper of a factory model
        (``EnergyColumnModel``): the base model's cells are bound (the appended column has none).
        For **non-synthetic** draws the metadata must resolve from the generating factory model; if
        they do not (registry miss in another process, non-factory model, wrapper without ``model=``)
        this raises instead of returning ``metadata=None``, and explicit ``metadata=`` is refused
        (it could not be verified against any object).  Explicit metadata remain available for
        synthetic test worlds, where they are the definition of the world.
        """
        if draws.stream not in set(allowed_streams):
            raise LeakageError(f"{consumer} may only use streams {sorted(allowed_streams)}, got "
                               f"{draws.stream!r} ({draws.stream_id})")
        bound = _factory_metadata_for(draws, model)
        if bound is not None:
            bound = bound.restricted_to(draws.ingredient_ids, draws.nutrient_ids)
        elif not draws.is_synthetic:
            raise InvalidProblemError(
                f"{consumer}: the object metadata of non-synthetic draws {draws.stream_id} (model {draws.model_id!r}, "
                f"{draws.model_fingerprint[:12]}) could not be resolved from a factory model"
                + (" (explicit metadata of a non-synthetic world cannot be verified and are refused)"
                   if metadata is not None else "")
                + "; pass model=<the generating FactoryModel, or its EnergyColumnModel wrapper> (review R2 / FIX_A: "
                  "no silent metadata=None for real data)")
        md = bound
        if metadata is not None:
            if not isinstance(metadata, StateMetadata):
                raise InvalidProblemError("PriorStates.from_drawset: metadata must be StateMetadata")
            if metadata.model_fingerprint is not None and metadata.model_fingerprint != draws.model_fingerprint:
                raise MetadataConflictError("PriorStates.from_drawset: explicit metadata name another generating model "
                                            "(model fingerprint mismatch)")
            if bound is not None:
                diffs = []
                for m in metadata.components:
                    b = bound.get(m.component)
                    if b is None:
                        diffs.append(f"{m.component.label()}: not a cell of the generating model")
                    elif b.binding_key() != m.binding_key():
                        diffs.append(f"{m.component.label()}: explicit "
                                     f"{dict(zip(_BINDING_FIELDS, m.binding_key()))} vs object "
                                     f"{dict(zip(_BINDING_FIELDS, b.binding_key()))}")
                if diffs:
                    raise MetadataConflictError(
                        "PriorStates.from_drawset: explicit metadata contradict the object metadata of the generating "
                        f"factory model {bound.model_id!r}; an external declaration cannot override the object: "
                        + "; ".join(diffs))
                # consistent: the factory metadata (the object) are kept
            else:
                md = replace(metadata, model_fingerprint=draws.model_fingerprint)
        S = draws.n_draws
        return cls(draws.theta, draws.d, np.full(S, 1.0 / S), draws.ingredient_ids, draws.nutrient_ids,
                   draws.stream, draws.stream_id, draws.fingerprint, bool(draws.is_synthetic), md,
                   draws.model_fingerprint)

    @classmethod
    def from_discrete(cls, theta: np.ndarray, d: np.ndarray, weights: Sequence[float],
                      ingredient_ids: Sequence[str], nutrient_ids: Sequence[str], *, label: str,
                      is_synthetic: bool, metadata: Optional[Union[StateMetadata, Mapping[Any, Any]]] = None
                      ) -> "PriorStates":
        """Explicit discrete prior (exact enumeration; e.g. analytic test cases).

        ``metadata`` (optional, R2) is part of the discrete definition: a :class:`StateMetadata` or a
        mapping ``{component: ComponentMetadata | field mapping}``; it enters the fingerprint.
        """
        th = np.asarray(theta, dtype=float)
        dd = np.asarray(d, dtype=float)
        w = np.asarray(weights, dtype=float)
        md = None
        if metadata is not None:
            md = metadata if isinstance(metadata, StateMetadata) else StateMetadata.from_mapping(
                metadata, origin="explicit_prior_definition", is_synthetic=bool(is_synthetic))
        if md is None:
            fp = stable_hash("PriorStates/discrete/v1", label, tuple(ingredient_ids), tuple(nutrient_ids), th, dd, w,
                             bool(is_synthetic))
        else:
            fp = stable_hash("PriorStates/discrete/v2", label, tuple(ingredient_ids), tuple(nutrient_ids), th, dd, w,
                             bool(is_synthetic), md.fingerprint())
        return cls(th, dd, w, tuple(ingredient_ids), tuple(nutrient_ids), DISCRETE_PRIOR_STREAM,
                   f"{DISCRETE_PRIOR_STREAM}/{label}", fp, bool(is_synthetic), md)

    # ------------------------------------------------------------------ accessors
    @property
    def n_states(self) -> int:
        """Number of states ``S``."""
        return int(self.weights.shape[0])

    @property
    def fingerprint(self) -> str:
        """Content hash of states, weights, labels and (when present) object metadata."""
        if self.metadata is None and self.source_model_fingerprint is None:
            return stable_hash("PriorStates/v1", self.stream, self.stream_id, self.source_fingerprint,
                               self.ingredient_ids, self.nutrient_ids, self.theta, self.d, self.weights,
                               self.is_synthetic)
        return stable_hash("PriorStates/v2", self.stream, self.stream_id, self.source_fingerprint,
                           self.ingredient_ids, self.nutrient_ids, self.theta, self.d, self.weights,
                           self.is_synthetic, None if self.metadata is None else self.metadata.fingerprint(),
                           self.source_model_fingerprint)

    @property
    def generator_fingerprint(self) -> Optional[str]:
        """World id of the model that generated the states (``None`` for a discrete prior)."""
        if self.metadata is not None and self.metadata.model_fingerprint is not None:
            return self.metadata.model_fingerprint
        return self.source_model_fingerprint

    def component_metadata(self, comp: Any) -> Optional[ComponentMetadata]:
        """Object metadata of one component (``None`` if not bound)."""
        return None if self.metadata is None else self.metadata.get(comp)

    def component_index(self, comp: ObservedComponent) -> tuple[int, int | None]:
        """``(i, j)`` of a component (``j is None`` for DM)."""
        try:
            i = self.ingredient_ids.index(comp.ingredient_id)
        except ValueError:
            raise InvalidProblemError(f"unknown ingredient {comp.ingredient_id!r}") from None
        if comp.component == DM_COMPONENT:
            return i, None
        try:
            j = self.nutrient_ids.index(comp.component)
        except ValueError:
            raise InvalidProblemError(f"unknown nutrient {comp.component!r}") from None
        return i, j

    def _column(self, comp: ObservedComponent) -> np.ndarray:
        i, j = self.component_index(comp)
        return self.d[:, i] if j is None else self.theta[:, i, j]

    def component_values(self, comps: Iterable[ObservedComponent]) -> np.ndarray:
        """True values ``[S, m]`` of the listed components in every state (canonical units)."""
        cols = []
        for c in comps:
            cols.append(self._column(c))
        if not cols:
            return np.zeros((self.n_states, 0))
        out = np.stack(cols, axis=1)
        if np.any(np.isnan(out)):
            raise InvalidProblemError("observed component has missing (NaN) prior values; an assay of an "
                                      "unmodelled quantity cannot be simulated")
        return out

    def component_variance(self, comp: ObservedComponent) -> Optional[float]:
        """Weighted variance of one component over the states (``None`` if missing)."""
        col = self._column(comp)
        if np.any(np.isnan(col)):
            return None
        w = self.weights
        m = float(w @ col)
        return float(w @ (col - m) ** 2)

    def weighted_mean(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conditional means ``(theta_bar [I, J], d_bar [I])`` under non-negative state weights.

        NaN entries stay NaN (missing stays missing, never 0).
        """
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.n_states,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise InvalidProblemError("weighted_mean: weights must be [S], >= 0, finite, positive sum")
        w = w / w.sum()
        th = np.einsum("s,sij->ij", w, np.where(np.isnan(self.theta), 0.0, self.theta))
        th = np.where(np.isnan(self.theta).any(axis=0), np.nan, th)
        dd = w @ self.d
        return th, dd

    def stochastic_components(self, tol: float = 0.0) -> tuple[ObservedComponent, ...]:
        """Components whose value varies across states (used for full perfect information)."""
        out: list[ObservedComponent] = []
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        for i in range(I):
            for j in range(J):
                col = self.theta[:, i, j]
                if np.all(np.isnan(col)):
                    continue
                if np.nanmax(col) - np.nanmin(col) > tol:
                    out.append(ObservedComponent(self.ingredient_ids[i], self.nutrient_ids[j]))
            col = self.d[:, i]
            if np.max(col) - np.min(col) > tol:
                out.append(ObservedComponent(self.ingredient_ids[i], DM_COMPONENT))
        return tuple(out)


@dataclass(frozen=True)
class TruthLink:
    """Truth end of the guard (FIX_A).

    ``linked``: the record claims the world of the prior states (it may restrict the verdict and must
    agree with the prior's metadata).  ``verified``: the claim was checked against an object the
    caller cannot copy by hand -- only then may the record serve as the *basis source* of a
    component (and so help identify a value).  ``verification`` is the
    :func:`verify_state_metadata` status of the metadata (``None`` when there are none).
    """

    metadata: Optional[StateMetadata]
    origin: str
    linked: bool
    verified: bool
    verification: Optional[str]
    reason: str = ""


def resolve_truth_link(truth: Any, prior: PriorStates) -> TruthLink:
    """Truth end of the guard with an explicit verification flag (FIX_A).

    * ``truth=None``: the finite model evaluates risk on the prior states, so the truth is the
      generator of the prior states; its metadata are the prior's own (linked, verified as the prior's
      own record) or unknown.
    * ``PriorStates``: linked and verified if it *is* the prior (same fingerprint) or its states come
      from the same generating model with registry-verified factory metadata; otherwise the truth
      states are another world and this raises (same metadata on different states is not a link).
    * a model object (``FactoryModel`` or ``EnergyColumnModel`` of one): its metadata are computed from
      the object; linked and verified iff its fingerprint is the prior's generator; another generator
      raises.
    * ``StateMetadata``: the prior's own metadata -> linked, verified; a ``model_fingerprint`` claim
      equal to the prior's generator -> linked, verified only if the metadata are factory metadata
      re-derived identically from the registry (a copied fingerprint is a claim, not a verification);
      another generator raises; no fingerprint -> unlinked (may only restrict).
    """
    if truth is None:
        if prior.metadata is not None:
            return TruthLink(prior.metadata, "prior_states_generate_the_finite_model_truth", True, True,
                             verify_state_metadata(prior.metadata)[0])
        return TruthLink(None, "unbound", False, False, None, "no object metadata on the prior and no truth record")
    gen = prior.generator_fingerprint
    if isinstance(truth, PriorStates):
        if truth.fingerprint == prior.fingerprint:
            if truth.metadata is None:
                return TruthLink(None, "same_prior_states:no_metadata", False, False, None,
                                 "the truth states are the prior states but carry no object metadata")
            return TruthLink(truth.metadata, f"{truth.metadata.origin}:same_prior_states", True, True,
                             verify_state_metadata(truth.metadata)[0])
        tg = truth.generator_fingerprint
        if tg is not None and gen is not None and tg == gen and truth.metadata is not None \
                and truth.metadata.origin == "factory_model_metadata":
            v, iss = verify_state_metadata(truth.metadata)
            if v == "factory_registry_verified":
                return TruthLink(truth.metadata, "factory_model_metadata:same_verified_generating_model", True, True, v)
        raise InvalidProblemError(
            "truth states are not the prior states and are not verifiably drawn from the prior's generating model "
            "(registry-verified factory metadata of the same world id): they describe another world. The finite "
            "information-value model evaluates risk on the prior states, so the truth must be their generator "
            "(double counting guard, review R2 / FIX_A)")
    from_object = not isinstance(truth, StateMetadata)
    md = _as_state_metadata(truth)
    v, iss = verify_state_metadata(md)
    if prior.metadata is not None and md.fingerprint() == prior.metadata.fingerprint():
        return TruthLink(md, f"{md.origin}:same_as_prior_metadata", True, True, v)
    if md.model_fingerprint is not None and gen is not None:
        if md.model_fingerprint != gen:
            raise InvalidProblemError(
                f"truth model {md.model_id!r} ({md.model_fingerprint[:12]}) is not the model that generated the prior "
                f"states ({gen[:12]}); the finite information-value model evaluates risk on the prior states, so the "
                "truth must be their generator (double counting guard, review R2)")
        if from_object:
            return TruthLink(md, f"{md.origin}:linked_by_model_object", True, True, v)
        if v == "factory_registry_verified":
            return TruthLink(md, f"{md.origin}:linked_by_verified_factory_metadata", True, True, v)
        return TruthLink(md, f"{md.origin}:model_fingerprint_claim_unverified", True, False, v,
                         "the record names the prior's generating model but cannot be re-derived from a model object "
                         f"({v}): it may restrict the verdict, never identify a value; " + "; ".join(iss))
    return TruthLink(md, f"{md.origin}:unlinked", False, False, v,
                     "no generating-model fingerprint to compare: the record may only restrict the verdict")


def resolve_truth_metadata(truth: Any, prior: PriorStates) -> tuple[Optional[StateMetadata], str, bool]:
    """Truth end of the guard: ``(truth_metadata, truth_origin, linked)`` (see :func:`resolve_truth_link`,
    which also says whether the link is *verified*; only a verified link may identify a value)."""
    link = resolve_truth_link(truth, prior)
    return link.metadata, link.origin, link.linked
