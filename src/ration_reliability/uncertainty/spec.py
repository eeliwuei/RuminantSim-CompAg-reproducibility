"""Declarative uncertainty specification -- the single input of the model factory (review R3).

Every stochastic model used in a study run is built by :func:`.factory.build_uncertainty_model`
from one :class:`UncertaintySpec`.  Optimisation draws, validation/test draws, robust sets and the
information-value prior are then all taken from the *same* built model (one evaluation world); a
nominal method differs only in that it ignores the uncertainty while optimising.

What the specification records, per cell ``(ingredient_id, component)`` (``component`` is a
nutrient id or ``"DM"``):

* the target moments ``mean``, ``sd`` and the physical range ``[lower, upper]`` -- never modified
  by the factory (a target that cannot be matched is reported, not changed);
* the provenance of the numbers: ``provenance_status`` (``sourced`` / ``research_scenario_assumption``
  / ``synthetic_test_only``), ``source_id``, ``locator`` and a ``data_fingerprint`` of the source row;
* what the SD represents (``variance_basis``, same vocabulary as
  ``information.signal.PRIOR_VARIANCE_BASES``), which variance decomposition was applied
  (``decomposition_id`` / ``decomposition_source``) and which measurement model the SD refers to
  (``measurement_model_id``).  These travel with the built model so that the prior / truth / signal
  guard (review R2) can read them from the object instead of from a caller's declaration;
* an optional per-cell family override with its ``family_choice_reason``.

Family rule (``family_rule``): a primary family, an ordered fallback list tried only when a family
cannot match the target moments (``fit_status = "infeasible_moment_match"``), and what to do when the
list is exhausted (``error`` / ``exclude_cell_as_missing`` / ``exclude_ingredient``).  Families are
the moment-matched ones of :mod:`.distributions` (``TN_MM``, ``LN_MM``, ``BETA_MM``).  The naive
truncated normal is available **only** as ``TN_NAIVE_DIAGNOSTIC`` in a specification whose
``purpose`` is ``diagnostic`` and whose ``moment_semantics`` is ``naive_parent_parameters``; a
specification that declares ``target_marginal_moments`` ("the given mean and SD are the moments of
the sampled distribution") cannot request it anywhere.

Correlation (``correlation``, optional): a Gaussian-copula correlation over stochastic cells.  The
input scale is explicit: ``latent_gaussian`` (the matrix *is* the latent correlation) or
``transformed_pearson`` (the matrix is a Pearson correlation on the original scale, e.g. from the
literature).  A transformed-scale Pearson ``r`` is used either as a *declared latent scenario*
(``declared_latent_scenario``; the resulting transformed correlation is reported and differs) or
through a calibrated mapping (``calibrated_mapping``: the latent value is solved so that the
transformed Pearson equals ``r``).  Both matrices are written to the model metadata.

Two-layer farm model (``two_layer``, optional): only ever an *extension scenario*.  It is built next
to the main model and never changes it; without identification evidence it is labelled
``unidentified_extension_scenario`` (if explicitly allowed) or not built at all.

No hidden defaults: every metadata field of every cell must be given, either in the cell or in the
specification's ``cell_defaults`` block.  All problems are collected and raised together as
:class:`~ration_reliability.errors.ConfigValidationError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..errors import ConfigValidationError
from ..hashing import stable_hash
from .distributions import MOMENT_MATCHED_FAMILIES

__all__ = [
    "SPEC_SCHEMA",
    "DM_COMPONENT",
    "FAMILY_TN_NAIVE_DIAGNOSTIC",
    "REQUESTABLE_FAMILIES",
    "PURPOSES",
    "MOMENT_SEMANTICS",
    "ON_EXHAUSTED",
    "SELECTION_BASES",
    "PROVENANCE_STATUSES",
    "VARIANCE_BASES",
    "NOT_APPLICABLE_BASIS",
    "NO_DECOMPOSITION",
    "CORRELATION_INPUT_SCALES",
    "CORRELATION_HANDLING",
    "TWO_LAYER_ROLES",
    "CELL_METADATA_FIELDS",
    "CellSpec",
    "FamilyRule",
    "CorrelationSpec",
    "TwoLayerSpec",
    "UncertaintySpec",
]

SPEC_SCHEMA = "ration_reliability.uncertainty_spec/0.1"
DM_COMPONENT = "DM"

#: The naive truncated normal (parent = target, then truncated) -- diagnostic only.
FAMILY_TN_NAIVE_DIAGNOSTIC = "TN_NAIVE_DIAGNOSTIC"
#: Families a specification may request.
REQUESTABLE_FAMILIES: tuple[str, ...] = tuple(MOMENT_MATCHED_FAMILIES) + (FAMILY_TN_NAIVE_DIAGNOSTIC,)
#: What the model is built for.  ``diagnostic`` is the only purpose that may use the naive family.
PURPOSES = ("main_analysis", "sensitivity_scenario", "smoke", "diagnostic", "unit_test")
#: ``target_marginal_moments``: the given mean/SD are the moments of the sampled marginal.
#: ``naive_parent_parameters``: the given mean/SD are the parent normal's parameters (diagnostic).
MOMENT_SEMANTICS = ("target_marginal_moments", "naive_parent_parameters")
ON_EXHAUSTED = ("error", "exclude_cell_as_missing", "exclude_ingredient")
#: ``declared_rule``: family fixed by the rule; ``development_data``: chosen per cell on
#: development data only (:func:`.factory.select_family_on_development_data`).
SELECTION_BASES = ("declared_rule", "development_data")
PROVENANCE_STATUSES = ("sourced", "research_scenario_assumption", "synthetic_test_only")
#: Same vocabulary as ``information.signal.PRIOR_VARIANCE_BASES`` and
#: ``information_states.TRUTH_SD_BASES`` (equality is unit-tested).
VARIANCE_BASES = ("true_batch_state", "observed_incl_sampling_and_lab", "observed_incl_lab_only", "unidentified")
#: Recorded instead of a variance basis for point (SD = 0) and missing cells.
NOT_APPLICABLE_BASIS = "not_applicable"
#: ``decomposition_id`` value meaning "no variance decomposition was applied" (must be written out).
NO_DECOMPOSITION = "none"
CORRELATION_INPUT_SCALES = ("latent_gaussian", "transformed_pearson")
CORRELATION_HANDLING = ("latent_as_declared", "declared_latent_scenario", "calibrated_mapping")
TWO_LAYER_ROLES = ("extension_scenario", "primary")

#: Metadata fields every cell must resolve (from the cell or from ``cell_defaults``).
CELL_METADATA_FIELDS = ("lower", "upper", "variance_basis", "data_fingerprint", "decomposition_id",
                        "decomposition_source", "measurement_model_id", "provenance_status", "source_id",
                        "locator")
_CELL_KEYS = ("ingredient_id", "component", "mean", "sd", "family", "family_choice_reason") + CELL_METADATA_FIELDS
_TOP_KEYS = ("schema", "spec_id", "purpose", "moment_semantics", "is_synthetic", "ingredient_ids", "nutrient_ids",
             "family_rule", "cell_defaults", "cells", "correlation", "two_layer", "notes")
_RULE_KEYS = ("primary_family", "fallback_families", "on_exhausted", "status", "rationale", "selection_basis")
_CORR_KEYS = ("structure_id", "labels", "matrix", "input_scale", "handling", "status", "provenance", "source_id",
              "locator")
_TL_KEYS = ("role", "ratios", "ratio_basis", "ratio_status", "ratio_source", "s", "family", "uncovered_cells",
            "allow_unidentified_scenario", "notes")


def _f(x: Any) -> float:
    """``None`` -> NaN (missing); anything else -> float."""
    return float("nan") if x is None else float(x)


def _num_or_none(x: float) -> Optional[float]:
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


# ------------------------------------------------------------------------------------------------
# parts
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CellSpec:
    """One cell ``(ingredient_id, component)`` with target moments, range and metadata.

    ``mean`` NaN = missing (sampled as NaN, never 0); ``sd`` 0 = point value.  ``family`` (optional)
    overrides the rule's primary family and then needs ``family_choice_reason``.
    """

    ingredient_id: str
    component: str
    mean: float
    sd: float
    lower: float
    upper: float
    variance_basis: str
    data_fingerprint: Optional[str]
    decomposition_id: str
    decomposition_source: Optional[str]
    measurement_model_id: Optional[str]
    provenance_status: str
    source_id: Optional[str]
    locator: Optional[str]
    family: Optional[str] = None
    family_choice_reason: Optional[str] = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.ingredient_id, self.component)

    @property
    def label(self) -> str:
        return f"{self.ingredient_id}:{self.component}"

    @property
    def is_missing(self) -> bool:
        return math.isnan(self.mean)

    @property
    def is_point(self) -> bool:
        return (not self.is_missing) and self.sd == 0.0

    @property
    def is_stochastic_target(self) -> bool:
        return (not self.is_missing) and (not math.isnan(self.sd)) and self.sd > 0.0

    def to_config(self) -> dict:
        return {"ingredient_id": self.ingredient_id, "component": self.component,
                "mean": _num_or_none(self.mean), "sd": _num_or_none(self.sd),
                "lower": float(self.lower), "upper": float(self.upper), "family": self.family,
                "family_choice_reason": self.family_choice_reason, "variance_basis": self.variance_basis,
                "data_fingerprint": self.data_fingerprint, "decomposition_id": self.decomposition_id,
                "decomposition_source": self.decomposition_source,
                "measurement_model_id": self.measurement_model_id,
                "provenance_status": self.provenance_status, "source_id": self.source_id, "locator": self.locator}


@dataclass(frozen=True)
class FamilyRule:
    """Primary family, ordered fallback list, action when exhausted, status and rationale."""

    primary_family: str
    fallback_families: tuple[str, ...]
    on_exhausted: str
    status: str
    rationale: str
    selection_basis: str

    def to_config(self) -> dict:
        return {"primary_family": self.primary_family, "fallback_families": list(self.fallback_families),
                "on_exhausted": self.on_exhausted, "status": self.status, "rationale": self.rationale,
                "selection_basis": self.selection_basis}


@dataclass(frozen=True)
class CorrelationSpec:
    """Copula correlation over labelled stochastic cells, with an explicit input scale."""

    structure_id: str
    labels: tuple[tuple[str, str], ...]
    matrix: tuple[tuple[float, ...], ...]
    input_scale: str
    handling: str
    status: str
    provenance: str
    source_id: Optional[str]
    locator: Optional[str]

    def matrix_array(self) -> np.ndarray:
        return np.array(self.matrix, dtype=float)

    def to_config(self) -> dict:
        return {"structure_id": self.structure_id, "labels": [list(c) for c in self.labels],
                "matrix": [list(r) for r in self.matrix], "input_scale": self.input_scale,
                "handling": self.handling, "status": self.status, "provenance": self.provenance,
                "source_id": self.source_id, "locator": self.locator}


@dataclass(frozen=True)
class TwoLayerSpec:
    """Two-layer (between / within farm) extension scenario (never a main-model dependency).

    ``ratios`` maps ``(ingredient_id, component)`` to the within/total SD ratio ``r``.  The basis of
    the SD that generates the truth is **not** declared here: it is read from the cells'
    ``variance_basis`` (object metadata; a separate declaration could contradict it).
    """

    role: str
    ratios: tuple[tuple[str, str, float], ...]
    ratio_basis: str
    ratio_status: str
    ratio_source: str
    s: float
    family: str
    uncovered_cells: str
    allow_unidentified_scenario: bool
    notes: str = ""

    def ratio_map(self) -> dict[tuple[str, str], float]:
        return {(a, b): float(r) for a, b, r in self.ratios}

    def to_config(self) -> dict:
        return {"role": self.role, "ratios": [[a, b, float(r)] for a, b, r in self.ratios],
                "ratio_basis": self.ratio_basis, "ratio_status": self.ratio_status,
                "ratio_source": self.ratio_source, "s": float(self.s), "family": self.family,
                "uncovered_cells": self.uncovered_cells,
                "allow_unidentified_scenario": bool(self.allow_unidentified_scenario), "notes": self.notes}


# ------------------------------------------------------------------------------------------------
# the specification
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UncertaintySpec:
    """Validated, canonical uncertainty specification.  Build with :meth:`from_config` or
    :meth:`from_arrays` (both run the same validation)."""

    spec_id: str
    purpose: str
    moment_semantics: str
    is_synthetic: bool
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    family_rule: FamilyRule
    cells: tuple[CellSpec, ...]
    correlation: Optional[CorrelationSpec] = None
    two_layer: Optional[TwoLayerSpec] = None
    notes: str = ""
    schema: str = SPEC_SCHEMA

    # ---- access --------------------------------------------------------------------------
    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.nutrient_ids) + (DM_COMPONENT,)

    def cell(self, ingredient_id: str, component: str) -> CellSpec:
        for c in self.cells:
            if c.ingredient_id == ingredient_id and c.component == component:
                return c
        raise KeyError(f"no cell {ingredient_id}:{component}")

    def uses_naive_family(self) -> bool:
        fams = {self.family_rule.primary_family, *self.family_rule.fallback_families}
        fams |= {c.family for c in self.cells if c.family}
        return FAMILY_TN_NAIVE_DIAGNOSTIC in fams

    # ---- canonical form and fingerprints -----------------------------------------------
    def to_config(self) -> dict:
        """Canonical configuration (round-trips through :meth:`from_config`)."""
        return {"schema": self.schema, "spec_id": self.spec_id, "purpose": self.purpose,
                "moment_semantics": self.moment_semantics, "is_synthetic": bool(self.is_synthetic),
                "ingredient_ids": list(self.ingredient_ids), "nutrient_ids": list(self.nutrient_ids),
                "family_rule": self.family_rule.to_config(), "cell_defaults": {},
                "cells": [c.to_config() for c in self.cells],
                "correlation": None if self.correlation is None else self.correlation.to_config(),
                "two_layer": None if self.two_layer is None else self.two_layer.to_config(),
                "notes": self.notes}

    def main_fingerprint(self) -> str:
        """Hash of every field that defines the main model (everything except ``two_layer``)."""
        cfg = self.to_config()
        cfg.pop("two_layer")
        return stable_hash("UncertaintySpec/main/v1", cfg)

    def fingerprint(self) -> str:
        """Hash of the whole specification (main model + extension scenarios)."""
        return stable_hash("UncertaintySpec/full/v1", self.to_config())

    # ---- constructors --------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "UncertaintySpec":
        """Validate a configuration mapping; every issue is collected and raised together."""
        issues: list[str] = []
        spec = _parse(cfg, issues)
        if issues or spec is None:
            raise ConfigValidationError(issues or ["uncertainty spec could not be parsed"])
        return spec

    @classmethod
    def from_arrays(cls, spec_id: str, ingredient_ids: Sequence[str], nutrient_ids: Sequence[str],
                    theta_mean, theta_sd, d_mean, d_sd, *, purpose: str, moment_semantics: str,
                    is_synthetic: bool, family_rule: Mapping[str, Any], cell_defaults: Mapping[str, Any],
                    theta_bounds: tuple[float, float], d_bounds: tuple[float, float],
                    cell_overrides: Optional[Mapping[tuple[str, str], Mapping[str, Any]]] = None,
                    correlation: Optional[Mapping[str, Any]] = None, two_layer: Optional[Mapping[str, Any]] = None,
                    notes: str = "") -> "UncertaintySpec":
        """Build the configuration from ``[I, J]`` / ``[I]`` arrays (NaN = missing) and validate it.

        ``cell_defaults`` must give every metadata field that is not overridden per cell; the
        bounds are explicit (no default range).
        """
        tm, ts = np.asarray(theta_mean, float), np.asarray(theta_sd, float)
        dm, ds = np.asarray(d_mean, float), np.asarray(d_sd, float)
        ing, nut = list(ingredient_ids), list(nutrient_ids)
        if tm.shape != (len(ing), len(nut)) or ts.shape != tm.shape or dm.shape != (len(ing),) or ds.shape != dm.shape:
            raise ConfigValidationError(["from_arrays: array shapes do not match the labels"])
        over = dict(cell_overrides or {})
        cells = []
        for i, iid in enumerate(ing):
            for j, nid in enumerate(nut):
                c = {"ingredient_id": iid, "component": nid, "mean": _num_or_none(tm[i, j]),
                     "sd": _num_or_none(ts[i, j]), "lower": float(theta_bounds[0]), "upper": float(theta_bounds[1])}
                c.update(dict(over.pop((iid, nid), {})))
                cells.append(c)
            c = {"ingredient_id": iid, "component": DM_COMPONENT, "mean": _num_or_none(dm[i]),
                 "sd": _num_or_none(ds[i]), "lower": float(d_bounds[0]), "upper": float(d_bounds[1])}
            c.update(dict(over.pop((iid, DM_COMPONENT), {})))
            cells.append(c)
        if over:
            raise ConfigValidationError([f"cell_overrides for unknown cells: {sorted(over)}"])
        cfg = {"schema": SPEC_SCHEMA, "spec_id": spec_id, "purpose": purpose, "moment_semantics": moment_semantics,
               "is_synthetic": is_synthetic, "ingredient_ids": ing, "nutrient_ids": nut,
               "family_rule": dict(family_rule), "cell_defaults": dict(cell_defaults), "cells": cells,
               "correlation": correlation, "two_layer": two_layer, "notes": notes}
        return cls.from_config(cfg)

    # ---- arrays ----------------------------------------------------------------------------
    def target_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(theta_mean [I, J], theta_sd [I, J], d_mean [I], d_sd [I])`` of the targets (NaN kept)."""
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        tm, ts = np.full((I, J), np.nan), np.full((I, J), np.nan)
        dm, ds = np.full(I, np.nan), np.full(I, np.nan)
        for c in self.cells:
            i = self.ingredient_ids.index(c.ingredient_id)
            if c.component == DM_COMPONENT:
                dm[i], ds[i] = c.mean, c.sd
            else:
                j = self.nutrient_ids.index(c.component)
                tm[i, j], ts[i, j] = c.mean, c.sd
        return tm, ts, dm, ds


# ------------------------------------------------------------------------------------------------
# parsing / validation
# ------------------------------------------------------------------------------------------------

def _unknown(where: str, d: Mapping, allowed: Sequence[str], issues: list[str]) -> None:
    extra = sorted(set(d) - set(allowed))
    if extra:
        issues.append(f"{where}: unknown key(s) {extra}")


def _check_family_name(where: str, fam: Any, issues: list[str]) -> None:
    if fam == "TN_NAIVE":
        issues.append(f"{where}: family 'TN_NAIVE' is not requestable; the naive truncated normal exists only as "
                      f"'{FAMILY_TN_NAIVE_DIAGNOSTIC}' in a diagnostic specification")
    elif fam == "N_UNTRUNCATED":
        issues.append(f"{where}: 'N_UNTRUNCATED' is a diagnostic report of out-of-range mass, not a sampling family")
    elif fam not in REQUESTABLE_FAMILIES:
        issues.append(f"{where}: unknown family {fam!r}; allowed {REQUESTABLE_FAMILIES}")


def _parse_rule(r: Any, issues: list[str]) -> Optional[FamilyRule]:
    if not isinstance(r, Mapping):
        issues.append("family_rule: must be a mapping")
        return None
    _unknown("family_rule", r, _RULE_KEYS, issues)
    miss = [k for k in _RULE_KEYS if k not in r]
    if miss:
        issues.append(f"family_rule: missing key(s) {miss} (no defaults)")
        return None
    _check_family_name("family_rule.primary_family", r["primary_family"], issues)
    fb = r["fallback_families"]
    if not isinstance(fb, (list, tuple)):
        issues.append("family_rule.fallback_families: must be a list (may be empty)")
        fb = []
    for k, f in enumerate(fb):
        _check_family_name(f"family_rule.fallback_families[{k}]", f, issues)
    if len(set(fb)) != len(fb) or r["primary_family"] in fb:
        issues.append("family_rule.fallback_families: duplicates or repeats the primary family")
    if r["on_exhausted"] not in ON_EXHAUSTED:
        issues.append(f"family_rule.on_exhausted: must be one of {ON_EXHAUSTED}")
    if r["status"] not in PROVENANCE_STATUSES:
        issues.append(f"family_rule.status: must be one of {PROVENANCE_STATUSES} (a pending family cannot build a model)")
    if not str(r["rationale"] or "").strip():
        issues.append("family_rule.rationale: required (why this family and fallback order)")
    if r["selection_basis"] not in SELECTION_BASES:
        issues.append(f"family_rule.selection_basis: must be one of {SELECTION_BASES}")
    return FamilyRule(str(r["primary_family"]), tuple(str(x) for x in fb), str(r["on_exhausted"]), str(r["status"]),
                      str(r["rationale"] or ""), str(r["selection_basis"]))


def _parse_cell(k: int, raw: Any, defaults: Mapping, is_synthetic: bool, issues: list[str]) -> Optional[CellSpec]:
    where = f"cells[{k}]"
    if not isinstance(raw, Mapping):
        issues.append(f"{where}: must be a mapping")
        return None
    _unknown(where, raw, _CELL_KEYS, issues)
    for req in ("ingredient_id", "component", "mean", "sd"):
        if req not in raw:
            issues.append(f"{where}: missing key {req!r}")
            return None
    where = f"cell {raw['ingredient_id']}:{raw['component']}"
    merged = {f: (raw[f] if f in raw else defaults.get(f, _MISSING)) for f in CELL_METADATA_FIELDS}
    miss = [f for f, v in merged.items() if v is _MISSING]
    if miss:
        issues.append(f"{where}: metadata field(s) {miss} neither in the cell nor in cell_defaults (no hidden defaults)")
        return None
    try:
        mean, sd = _f(raw["mean"]), _f(raw["sd"])
        lo, hi = float(merged["lower"]), float(merged["upper"])
    except (TypeError, ValueError):
        issues.append(f"{where}: mean/sd/lower/upper must be numbers (null = missing)")
        return None
    if not (math.isfinite(lo) and lo < hi):
        issues.append(f"{where}: need finite lower < upper, got [{lo}, {hi}]")
    if not math.isnan(sd) and sd < 0:
        issues.append(f"{where}: sd < 0")
    if math.isnan(mean) and not math.isnan(sd):
        issues.append(f"{where}: sd given without a mean")
    if not math.isnan(mean) and math.isnan(sd):
        issues.append(f"{where}: mean without SD: not sampleable; the point-value policy is pending "
                      "(configs/uncertainty.yaml point_value_policy) -- pass sd = 0 explicitly for a declared point value")
    is_missing = math.isnan(mean)
    is_point = (not is_missing) and (not math.isnan(sd)) and sd == 0.0
    vb = merged["variance_basis"]
    if is_missing or is_point:
        vb = NOT_APPLICABLE_BASIS
    elif vb not in VARIANCE_BASES:
        issues.append(f"{where}: variance_basis {vb!r} must be one of {VARIANCE_BASES}")
    fp = merged["data_fingerprint"]
    if not is_missing and not str(fp or "").strip():
        issues.append(f"{where}: data_fingerprint required for every cell with a value (hash of the source row, or "
                      "'synthetic:<label>' for synthetic values)")
    dec = merged["decomposition_id"]
    if not str(dec or "").strip():
        issues.append(f"{where}: decomposition_id required ('{NO_DECOMPOSITION}' when no decomposition was applied)")
    dsrc = merged["decomposition_source"]
    if dec != NO_DECOMPOSITION and not str(dsrc or "").strip():
        issues.append(f"{where}: decomposition {dec!r} needs decomposition_source")
    mm = merged["measurement_model_id"]
    if vb in ("observed_incl_sampling_and_lab", "observed_incl_lab_only") and not str(mm or "").strip():
        issues.append(f"{where}: variance_basis {vb!r} needs measurement_model_id (which measurement process the SD "
                      "contains)")
    if vb == "true_batch_state" and not is_synthetic and dec == NO_DECOMPOSITION:
        issues.append(f"{where}: true_batch_state for non-synthetic data needs a variance decomposition "
                      "(decomposition_id + decomposition_source); an observed SD is not a true-state SD by declaration")
    if vb == "true_batch_state" and dec != NO_DECOMPOSITION and not str(mm or "").strip():
        issues.append(f"{where}: a decomposition to true_batch_state must name the removed measurement model "
                      "(measurement_model_id)")
    ps = merged["provenance_status"]
    if ps not in PROVENANCE_STATUSES:
        issues.append(f"{where}: provenance_status must be one of {PROVENANCE_STATUSES} (pending values cannot build a model)")
    if ps == "sourced" and not (str(merged["source_id"] or "").strip() and str(merged["locator"] or "").strip()):
        issues.append(f"{where}: sourced values need source_id and locator")
    if ps == "synthetic_test_only" and not is_synthetic:
        issues.append(f"{where}: synthetic_test_only value in a non-synthetic specification")
    fam = raw.get("family")
    reason = raw.get("family_choice_reason")
    if fam is not None:
        _check_family_name(f"{where}.family", fam, issues)
        if not str(reason or "").strip():
            issues.append(f"{where}: a per-cell family override needs family_choice_reason")
    return CellSpec(str(raw["ingredient_id"]), str(raw["component"]), mean, sd, lo, hi, str(vb),
                    None if fp is None else str(fp), str(dec), None if dsrc is None else str(dsrc),
                    None if mm is None else str(mm), str(ps),
                    None if merged["source_id"] is None else str(merged["source_id"]),
                    None if merged["locator"] is None else str(merged["locator"]),
                    None if fam is None else str(fam), None if reason is None else str(reason))


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


def _parse_corr(r: Any, cells: Mapping[tuple[str, str], CellSpec], issues: list[str]) -> Optional[CorrelationSpec]:
    if not isinstance(r, Mapping):
        issues.append("correlation: must be a mapping or null")
        return None
    _unknown("correlation", r, _CORR_KEYS, issues)
    miss = [k for k in _CORR_KEYS if k not in r]
    if miss:
        issues.append(f"correlation: missing key(s) {miss}")
        return None
    labels = tuple(tuple(str(x) for x in c) for c in r["labels"])
    if any(len(c) != 2 for c in labels) or len(set(labels)) != len(labels):
        issues.append("correlation.labels: must be distinct [ingredient_id, component] pairs")
        return None
    for c in labels:
        if c not in cells:
            issues.append(f"correlation.labels: {c[0]}:{c[1]} is not a cell of the specification")
        elif not cells[c].is_stochastic_target:
            issues.append(f"correlation.labels: {c[0]}:{c[1]} is not a stochastic cell (point or missing)")
    try:
        m = np.array(r["matrix"], dtype=float)
    except (TypeError, ValueError):
        issues.append("correlation.matrix: must be numeric")
        return None
    if m.shape != (len(labels), len(labels)):
        issues.append("correlation.matrix: shape does not match labels")
        return None
    if not np.all(np.isfinite(m)) or not np.allclose(m, m.T, atol=1e-12, rtol=0) or \
            not np.allclose(np.diag(m), 1.0, atol=1e-12, rtol=0) or np.any(np.abs(m) > 1.0 + 1e-12):
        issues.append("correlation.matrix: must be finite, symmetric, unit diagonal, |rho| <= 1")
    if r["input_scale"] not in CORRELATION_INPUT_SCALES:
        issues.append(f"correlation.input_scale: must be one of {CORRELATION_INPUT_SCALES}")
    if r["handling"] not in CORRELATION_HANDLING:
        issues.append(f"correlation.handling: must be one of {CORRELATION_HANDLING}")
    if r["input_scale"] == "latent_gaussian" and r["handling"] != "latent_as_declared":
        issues.append("correlation: input_scale 'latent_gaussian' requires handling 'latent_as_declared'")
    if r["input_scale"] == "transformed_pearson" and r["handling"] == "latent_as_declared":
        issues.append("correlation: a transformed-scale Pearson matrix must be handled as "
                      "'declared_latent_scenario' or 'calibrated_mapping' (it is not a latent correlation)")
    if r["status"] not in PROVENANCE_STATUSES:
        issues.append(f"correlation.status: must be one of {PROVENANCE_STATUSES} (pending values cannot build a model)")
    if r["status"] == "sourced" and not (str(r["source_id"] or "").strip() and str(r["locator"] or "").strip()):
        issues.append("correlation: sourced values need source_id and locator")
    if not str(r["provenance"] or "").strip():
        issues.append("correlation.provenance: required")
    return CorrelationSpec(str(r["structure_id"]), labels, tuple(tuple(float(x) for x in row) for row in m),
                           str(r["input_scale"]), str(r["handling"]), str(r["status"]), str(r["provenance"] or ""),
                           None if r["source_id"] is None else str(r["source_id"]),
                           None if r["locator"] is None else str(r["locator"]))


def _parse_two_layer(r: Any, cells: Mapping[tuple[str, str], CellSpec], issues: list[str]) -> Optional[TwoLayerSpec]:
    if not isinstance(r, Mapping):
        issues.append("two_layer: must be a mapping or null")
        return None
    _unknown("two_layer", r, _TL_KEYS, issues)
    miss = [k for k in _TL_KEYS if k not in r and k != "notes"]
    if miss:
        issues.append(f"two_layer: missing key(s) {miss}")
        return None
    if r["role"] not in TWO_LAYER_ROLES:
        issues.append(f"two_layer.role: must be one of {TWO_LAYER_ROLES}")
    ratios = []
    for k, e in enumerate(r["ratios"] or []):
        try:
            a, b, v = str(e[0]), str(e[1]), float(e[2])
        except (TypeError, ValueError, IndexError):
            issues.append(f"two_layer.ratios[{k}]: must be [ingredient_id, component, ratio]")
            continue
        if (a, b) not in cells:
            issues.append(f"two_layer.ratios[{k}]: {a}:{b} is not a cell of the specification")
        ratios.append((a, b, v))
    if r["ratio_status"] not in PROVENANCE_STATUSES:
        issues.append(f"two_layer.ratio_status: must be one of {PROVENANCE_STATUSES}")
    _check_family_name("two_layer.family", r["family"], issues)
    if r["family"] == FAMILY_TN_NAIVE_DIAGNOSTIC:
        issues.append("two_layer.family: the naive family cannot generate a two-layer world")
    try:
        s = float(r["s"])
    except (TypeError, ValueError):
        issues.append("two_layer.s: must be a number")
        s = float("nan")
    return TwoLayerSpec(str(r["role"]), tuple(ratios), str(r["ratio_basis"]), str(r["ratio_status"]),
                        str(r["ratio_source"] or ""), s, str(r["family"]), str(r["uncovered_cells"]),
                        bool(r["allow_unidentified_scenario"]), str(r.get("notes") or ""))


def _parse(cfg: Any, issues: list[str]) -> Optional[UncertaintySpec]:
    if not isinstance(cfg, Mapping):
        issues.append("uncertainty spec: must be a mapping")
        return None
    _unknown("spec", cfg, _TOP_KEYS, issues)
    miss = [k for k in _TOP_KEYS if k not in cfg and k not in ("correlation", "two_layer", "notes", "cell_defaults")]
    if miss:
        issues.append(f"spec: missing key(s) {miss}")
        return None
    if cfg["schema"] != SPEC_SCHEMA:
        issues.append(f"spec.schema: expected {SPEC_SCHEMA!r}")
    if not str(cfg["spec_id"] or "").strip():
        issues.append("spec.spec_id: required")
    purpose, sem = cfg["purpose"], cfg["moment_semantics"]
    if purpose not in PURPOSES:
        issues.append(f"spec.purpose: must be one of {PURPOSES}")
    if sem not in MOMENT_SEMANTICS:
        issues.append(f"spec.moment_semantics: must be one of {MOMENT_SEMANTICS}")
    if sem == "naive_parent_parameters" and purpose != "diagnostic":
        issues.append("spec: moment_semantics 'naive_parent_parameters' is allowed only with purpose 'diagnostic'")
    if not isinstance(cfg["is_synthetic"], bool):
        issues.append("spec.is_synthetic: must be true or false")
    is_syn = bool(cfg["is_synthetic"])
    ing = tuple(str(x) for x in cfg["ingredient_ids"])
    nut = tuple(str(x) for x in cfg["nutrient_ids"])
    if len(set(ing)) != len(ing) or len(set(nut)) != len(nut) or DM_COMPONENT in nut:
        issues.append("spec: ingredient_ids / nutrient_ids must be distinct and 'DM' is not a nutrient column")
    rule = _parse_rule(cfg["family_rule"], issues)
    if rule is not None and rule.status == "synthetic_test_only" and not is_syn:
        issues.append("family_rule.status synthetic_test_only in a non-synthetic specification")
    defaults = cfg.get("cell_defaults") or {}
    if not isinstance(defaults, Mapping):
        issues.append("cell_defaults: must be a mapping")
        defaults = {}
    _unknown("cell_defaults", defaults, CELL_METADATA_FIELDS, issues)
    cells: dict[tuple[str, str], CellSpec] = {}
    for k, raw in enumerate(cfg["cells"] or []):
        c = _parse_cell(k, raw, defaults, is_syn, issues)
        if c is None:
            continue
        if c.key in cells:
            issues.append(f"cell {c.label}: listed twice")
        cells[c.key] = c
    expected = [(i, n) for i in ing for n in nut + (DM_COMPONENT,)]
    missing_cells = [f"{i}:{n}" for i, n in expected if (i, n) not in cells]
    extra_cells = [f"{i}:{n}" for (i, n) in cells if (i, n) not in set(expected)]
    if missing_cells:
        issues.append(f"cells: every (ingredient, nutrient) and (ingredient, DM) cell must be listed explicitly "
                      f"(mean null = missing); absent: {missing_cells}")
    if extra_cells:
        issues.append(f"cells: unknown ingredient/component: {extra_cells}")
    for c in cells.values():
        if c.component == DM_COMPONENT and c.is_missing:
            issues.append(f"cell {c.label}: DM mean is required (the executed q needs a DM fraction)")
    if rule is not None:
        naive_used = rule.primary_family == FAMILY_TN_NAIVE_DIAGNOSTIC or \
            FAMILY_TN_NAIVE_DIAGNOSTIC in rule.fallback_families or \
            any(c.family == FAMILY_TN_NAIVE_DIAGNOSTIC for c in cells.values())
        if naive_used and not (purpose == "diagnostic" and sem == "naive_parent_parameters"):
            issues.append(f"spec: '{FAMILY_TN_NAIVE_DIAGNOSTIC}' requires purpose 'diagnostic' and moment_semantics "
                          "'naive_parent_parameters'; a model declared on target mean/SD must not use the naive "
                          "truncated normal")
        if sem == "naive_parent_parameters" and not naive_used:
            issues.append("spec: moment_semantics 'naive_parent_parameters' without the naive diagnostic family")
        if naive_used and rule.fallback_families:
            issues.append("spec: a diagnostic naive specification has no fallback families (its drift is the point)")
    corr = None if cfg.get("correlation") is None else _parse_corr(cfg["correlation"], cells, issues)
    tl = None if cfg.get("two_layer") is None else _parse_two_layer(cfg["two_layer"], cells, issues)
    if issues or rule is None:
        return None
    ordered = tuple(cells[k] for k in expected)
    return UncertaintySpec(str(cfg["spec_id"]), str(purpose), str(sem), is_syn, ing, nut, rule, ordered, corr, tl,
                           str(cfg.get("notes") or ""), SPEC_SCHEMA)
