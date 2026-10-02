"""Uncertainty model factory (review R3): one specification -> one model -> one evaluation world.

:func:`build_uncertainty_model` turns a validated :class:`~.spec.UncertaintySpec` into a
:class:`FactoryModel`.  Every consumer takes its random quantities from that one object:

* optimisation draws (``opt``), development selection (``validation``) and the frozen hold-out
  simulation (``test``): :meth:`FactoryModel.draw` / :meth:`FactoryModel.draw_world`;
* robust uncertainty sets: :meth:`FactoryModel.box` (moments of the *sampled* marginals, so the box
  and the evaluation world describe the same distribution);
* the information-value prior: :meth:`FactoryModel.prior_states` (development streams only) and the
  per-cell variance basis :meth:`FactoryModel.prior_variance_basis` read from the model metadata.

A nominal method uses :meth:`FactoryModel.nominal_state` (the target means) and simply ignores the
uncertainty while optimising; it is scored in the same world as every other method.  All draws carry
``DrawSet.model_fingerprint == FactoryModel.fingerprint()`` (the world id), which
``experiments/E0_verification/smoke_pipeline.py`` checks before scoring.

Rules implemented here (review R3 items 1-8):

1. Families are moment matched (``TN_MM`` / ``LN_MM`` / ``BETA_MM``).  A family that cannot
   reproduce a target gets ``fit_status = "infeasible_moment_match"`` in the cell's attempt trail;
   the factory then tries the declared fallback families *in order* and finally applies
   ``family_rule.on_exhausted`` (``error`` / ``exclude_cell_as_missing`` / ``exclude_ingredient``).
   The target mean and SD are never changed and nothing is clipped to make a fit succeed.
2. ``TN_NAIVE_DIAGNOSTIC`` (parent = target, then truncated) is available only in a diagnostic
   specification (checked by :mod:`.spec`); its cells get ``fit_status = "diagnostic_drift"``, a
   ``moment_drift_flag`` and ``is_diagnostic = True`` in the metadata.
3. Every cell's metadata records variance basis, data fingerprint, decomposition id/source,
   measurement model id, synthetic flag, family, target and achieved moments, bounds, parameters,
   fit status and the reason for the family choice (:class:`CellModelMetadata`).
4. Gaussian copula: the metadata keeps the *latent* correlation (what the copula samples with) and
   the *transformed-scale* Pearson correlation of the marginals (Gauss-Hermite quadrature,
   deterministic), plus the handling of a literature Pearson ``r`` (declared latent scenario, or a
   calibrated mapping solved pair by pair).  A non-PSD latent matrix raises; nothing is repaired.
5. The two-layer farm model is only an *extension scenario* (:class:`ExtensionScenario`): it never
   changes the main model, its draws or its fingerprint; without identification it is either an
   explicitly allowed ``unidentified_extension_scenario`` or not built.
6. The model fingerprint hashes the main specification, the fitted marginals, the correlation and the
   metadata; fixed specification + fixed :class:`~.streams.RandomStreams` reproduce the draws exactly.
7. Synthetic acceptance (target mean 0.02, SD 0.015 on [0, 1]) is in
   ``tests/unit/test_uncertainty_factory.py`` and ``reports/uncertainty_factory_audit.json``.
8. A data-driven family choice (:func:`select_family_on_development_data`) accepts development data
   only (``development`` / ``validation`` / ``opt``); a test split raises
   :class:`~ration_reliability.errors.LeakageError`.

No parameter value is embedded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import optimize, special, stats

from ..errors import ConfigValidationError, LeakageError
from ..hashing import stable_hash
from .base import DrawSet, UncertaintyModel, require_stream
from .correlation import CorrelationStructure, GaussianCopulaModel, NotPSDError, check_psd
from .distributions import MOMENT_MATCHED_FAMILIES, MarginalFit, fit_marginal
from .information_states import TwoLayerFarmModel
from .spec import DM_COMPONENT, FAMILY_TN_NAIVE_DIAGNOSTIC, CellSpec, UncertaintySpec
from .streams import RandomStreams

__all__ = [
    "FACTORY_VERSION",
    "FACTORY_FIT_STATUSES",
    "DEVELOPMENT_SPLITS",
    "EXTENSION_STATUSES",
    "FactoryBuildError",
    "WorldMismatchError",
    "FitAttempt",
    "CellModelMetadata",
    "CorrelationMetadata",
    "ExtensionMetadata",
    "ExtensionScenario",
    "ModelMetadata",
    "FamilySelection",
    "FactoryModel",
    "build_uncertainty_model",
    "select_family_on_development_data",
    "transformed_pearson",
    "calibrate_latent_rho",
    "metadata_for_fingerprint",
    "metadata_for_drawset",
]

FACTORY_VERSION = "uncertainty-factory/0.1"
#: Factory-level fit statuses (the attempt trail keeps the raw :mod:`.distributions` status).
FACTORY_FIT_STATUSES = ("matched", "point", "missing", "diagnostic_drift", "infeasible_moment_match")
#: Splits on which a data-driven family choice may be made (review R3 item 8).
DEVELOPMENT_SPLITS = ("development", "validation", "opt")
EXTENSION_STATUSES = ("identified_extension_scenario", "unidentified_extension_scenario", "not_built_unidentified",
                      "invalid_extension_config")

#: Gauss-Hermite nodes for the transformed-scale Pearson correlation (probabilists' Hermite).
_GH_N = 96
_GH_X, _GH_W = hermegauss(_GH_N)
_GH_W = _GH_W / math.sqrt(2.0 * math.pi)
#: Latent correlations are searched in ``[-_RHO_EDGE, _RHO_EDGE]``.
_RHO_EDGE = 1.0 - 1e-9

_REGISTRY: dict[str, "ModelMetadata"] = {}


class FactoryBuildError(ConfigValidationError):
    """The specification is valid but no model can be built (every issue listed)."""


class WorldMismatchError(ValueError):
    """Draws from different models were combined where one evaluation world is required."""


# ------------------------------------------------------------------------------------------------
# metadata containers
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FitAttempt:
    """One family tried for a cell: raw distribution status and factory fit status."""

    family: str
    distribution_status: str
    fit_status: str
    flags: tuple[str, ...]
    message: str

    def to_dict(self) -> dict:
        return {"family": self.family, "distribution_status": self.distribution_status,
                "fit_status": self.fit_status, "flags": list(self.flags), "message": self.message}


@dataclass(frozen=True)
class CellModelMetadata:
    """Everything recorded about one cell of a built model (review R3 item 4 / R2 bindings)."""

    ingredient_id: str
    component: str
    included: bool
    exclusion_action: Optional[str]
    variance_basis: str
    data_fingerprint: Optional[str]
    decomposition_id: str
    decomposition_source: Optional[str]
    measurement_model_id: Optional[str]
    provenance_status: str
    source_id: Optional[str]
    locator: Optional[str]
    is_synthetic: bool
    requested_family: Optional[str]
    family: Optional[str]
    family_choice_reason: str
    attempts: tuple[FitAttempt, ...]
    fit_status: str
    is_diagnostic: bool
    moment_drift_flag: bool
    target_mean: float
    target_sd: float
    achieved_mean: float
    achieved_sd: float
    lower: float
    upper: float
    params: tuple[float, ...]
    mean_rel_shift: float
    sd_rel_error: float
    outside_mass: float
    flags: tuple[str, ...]
    message: str

    @property
    def label(self) -> str:
        return f"{self.ingredient_id}:{self.component}"

    @property
    def is_stochastic(self) -> bool:
        return self.included and self.fit_status in ("matched", "diagnostic_drift")

    @property
    def used_fallback(self) -> bool:
        return self.family is not None and self.requested_family is not None and self.family != self.requested_family

    def to_dict(self, include_values: bool = True) -> dict:
        d = {"ingredient_id": self.ingredient_id, "component": self.component, "included": self.included,
             "exclusion_action": self.exclusion_action, "variance_basis": self.variance_basis,
             "data_fingerprint": self.data_fingerprint, "decomposition_id": self.decomposition_id,
             "decomposition_source": self.decomposition_source, "measurement_model_id": self.measurement_model_id,
             "provenance_status": self.provenance_status, "source_id": self.source_id, "locator": self.locator,
             "is_synthetic": self.is_synthetic, "requested_family": self.requested_family, "family": self.family,
             "family_choice_reason": self.family_choice_reason, "attempts": [a.to_dict() for a in self.attempts],
             "fit_status": self.fit_status, "is_diagnostic": self.is_diagnostic,
             "moment_drift_flag": self.moment_drift_flag, "lower": self.lower, "upper": self.upper,
             "flags": list(self.flags), "message": self.message}
        if include_values:
            d.update({"target_mean": _jf(self.target_mean), "target_sd": _jf(self.target_sd),
                      "achieved_mean": _jf(self.achieved_mean), "achieved_sd": _jf(self.achieved_sd),
                      "params": [float(p) for p in self.params], "mean_rel_shift": _jf(self.mean_rel_shift),
                      "sd_rel_error": _jf(self.sd_rel_error), "outside_mass": _jf(self.outside_mass)})
        else:
            if self.attempts:
                d["attempts"] = [{k: v for k, v in a.to_dict().items() if k != "message"} for a in self.attempts]
            d["message"] = "(values withheld)" if self.message else ""
        return d


@dataclass(frozen=True)
class CorrelationMetadata:
    """Latent (copula) and transformed-scale (Pearson of the sampled values) correlations."""

    structure_id: str
    labels: tuple[tuple[str, str], ...]
    input_scale: str
    handling: str
    status: str
    provenance: str
    source_id: Optional[str]
    locator: Optional[str]
    input_matrix: tuple[tuple[float, ...], ...]
    latent_correlation: tuple[tuple[float, ...], ...]
    transformed_correlation: tuple[tuple[float, ...], ...]
    latent_lambda_min: float
    max_abs_latent_minus_transformed: float
    calibration: tuple[tuple[str, str, float, float, float], ...]   # (a, b, target r, latent rho, achieved r)
    interpretation: str

    def to_dict(self, include_values: bool = True) -> dict:
        d = {"structure_id": self.structure_id, "labels": [list(c) for c in self.labels],
             "input_scale": self.input_scale, "handling": self.handling, "status": self.status,
             "provenance": self.provenance, "source_id": self.source_id, "locator": self.locator,
             "interpretation": self.interpretation, "n_cells": len(self.labels)}
        if include_values:
            d.update({"input_matrix": [list(r) for r in self.input_matrix],
                      "latent_correlation": [list(r) for r in self.latent_correlation],
                      "transformed_correlation": [list(r) for r in self.transformed_correlation],
                      "latent_lambda_min": self.latent_lambda_min,
                      "max_abs_latent_minus_transformed": self.max_abs_latent_minus_transformed,
                      "calibration": [list(c) for c in self.calibration]})
        return d


@dataclass(frozen=True)
class ExtensionMetadata:
    """Status of an extension scenario (never a dependency of the main model)."""

    name: str
    requested_role: str
    role: str
    status: str
    reasons: tuple[str, ...]
    truth_variance_basis: Optional[str]
    notes: tuple[str, ...]
    config_fingerprint: str

    def to_dict(self) -> dict:
        return {"name": self.name, "requested_role": self.requested_role, "role": self.role, "status": self.status,
                "reasons": list(self.reasons), "truth_variance_basis": self.truth_variance_basis,
                "notes": list(self.notes), "config_fingerprint": self.config_fingerprint}


@dataclass(frozen=True, eq=False)
class ExtensionScenario:
    """An extension scenario next to the main model (e.g. the two-layer farm world)."""

    metadata: ExtensionMetadata
    model: Optional[TwoLayerFarmModel]

    @property
    def is_built(self) -> bool:
        return self.model is not None


@dataclass(frozen=True)
class FamilySelection:
    """A per-cell family choice made on development data (review R3 item 8)."""

    ingredient_id: str
    component: str
    split: str
    data_label: str
    data_fingerprint: str
    n_obs: int
    target_mean: float
    target_sd: float
    candidates: tuple[str, ...]
    scores: tuple[tuple[str, float, str], ...]      # (family, neg. log-likelihood or inf, fit status)
    chosen: str
    criterion: str
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.ingredient_id, self.component)

    def fingerprint(self) -> str:
        return stable_hash("FamilySelection/v1", self)

    def to_dict(self) -> dict:
        return {"ingredient_id": self.ingredient_id, "component": self.component, "split": self.split,
                "data_label": self.data_label, "data_fingerprint": self.data_fingerprint, "n_obs": self.n_obs,
                "candidates": list(self.candidates), "scores": [list(s) for s in self.scores],
                "chosen": self.chosen, "criterion": self.criterion, "reason": self.reason}


@dataclass(frozen=True)
class ModelMetadata:
    """Model-level metadata of a built model (the evaluation world)."""

    factory_version: str
    spec_id: str
    spec_main_fingerprint: str
    model_id: str
    purpose: str
    moment_semantics: str
    is_diagnostic: bool
    is_synthetic: bool
    spec_ingredient_ids: tuple[str, ...]
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    excluded_ingredients: tuple[tuple[str, str], ...]
    cells: tuple[CellModelMetadata, ...]
    correlation: Optional[CorrelationMetadata]
    family_rule: tuple[tuple[str, Any], ...]
    family_selections: tuple[FamilySelection, ...]
    summary: tuple[tuple[str, Any], ...]

    def fingerprint(self) -> str:
        return stable_hash("ModelMetadata/v1", self)

    def cell(self, ingredient_id: str, component: str) -> CellModelMetadata:
        for c in self.cells:
            if c.ingredient_id == ingredient_id and c.component == component:
                return c
        raise KeyError(f"no cell {ingredient_id}:{component}")

    def summary_dict(self) -> dict:
        return {k: v for k, v in self.summary}

    def variance_basis_map(self, stochastic_only: bool = True) -> dict[tuple[str, str], str]:
        """``(ingredient_id, component) -> variance_basis`` taken from the object (not a declaration)."""
        return {(c.ingredient_id, c.component): c.variance_basis for c in self.cells
                if c.included and (c.is_stochastic or not stochastic_only)}

    def to_dict(self, include_values: bool = True) -> dict:
        return {"factory_version": self.factory_version, "spec_id": self.spec_id,
                "spec_main_fingerprint": self.spec_main_fingerprint, "model_id": self.model_id,
                "metadata_fingerprint": self.fingerprint(), "purpose": self.purpose,
                "moment_semantics": self.moment_semantics, "is_diagnostic": self.is_diagnostic,
                "is_synthetic": self.is_synthetic, "spec_ingredient_ids": list(self.spec_ingredient_ids),
                "ingredient_ids": list(self.ingredient_ids), "nutrient_ids": list(self.nutrient_ids),
                "excluded_ingredients": [list(x) for x in self.excluded_ingredients],
                "family_rule": dict(self.family_rule),
                "family_selections": [s.to_dict() for s in self.family_selections],
                "summary": self.summary_dict(),
                "correlation": None if self.correlation is None else self.correlation.to_dict(include_values),
                "cells": [c.to_dict(include_values) for c in self.cells],
                "values_included": bool(include_values)}


def _jf(x: float) -> Optional[float]:
    return None if (x is None or not math.isfinite(float(x))) else float(x)


# ------------------------------------------------------------------------------------------------
# the model
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FactoryModel(UncertaintyModel):
    """A model built by the factory: sampling mechanism + metadata + extension scenarios.

    Sampling is delegated to a :class:`~.correlation.GaussianCopulaModel` over the moment-matched
    (or, in a diagnostic specification, naive) marginals; with no correlation the draws equal
    independent draws from those marginals.  :meth:`fingerprint` is the world id.
    """

    model_id: str
    ingredient_ids: tuple
    nutrient_ids: tuple
    is_synthetic: bool
    inner: GaussianCopulaModel
    metadata: ModelMetadata
    spec: UncertaintySpec
    extensions: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))
        object.__setattr__(self, "extensions", tuple(self.extensions))
        if self.inner.ingredient_ids != self.ingredient_ids or self.inner.nutrient_ids != self.nutrient_ids:
            raise ValueError("FactoryModel: inner model axes differ")

    # ---- UncertaintyModel ----------------------------------------------------------------
    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        return self.inner.sample(rng, n_draws)

    def params_for_fingerprint(self) -> dict:
        return {"factory_version": FACTORY_VERSION, "spec_main_fingerprint": self.spec.main_fingerprint(),
                "inner_fingerprint": self.inner.fingerprint(), "metadata_fingerprint": self.metadata.fingerprint()}

    def fingerprint(self) -> str:
        fp = self.__dict__.get("_fp_cache")
        if fp is None:
            fp = super().fingerprint()
            object.__setattr__(self, "_fp_cache", fp)
        return fp

    # ---- one world for every consumer -------------------------------------------------------
    @property
    def is_diagnostic(self) -> bool:
        return self.metadata.is_diagnostic

    def draw_world(self, streams: RandomStreams, *, n_opt: Optional[int] = None, n_validation: Optional[int] = None,
                   n_test: Optional[int] = None) -> dict[str, DrawSet]:
        """Draw the requested purpose streams from this one model (same world, named streams)."""
        out: dict[str, DrawSet] = {}
        for name, n in (("opt", n_opt), ("validation", n_validation), ("test", n_test)):
            if n is not None:
                out[name] = self.draw(streams, name, int(n))
        return out

    def assert_same_world(self, *draw_sets: DrawSet) -> None:
        """Raise :class:`WorldMismatchError` unless every draw set comes from this model."""
        fp = self.fingerprint()
        bad = [f"{d.stream_id} (model {d.model_id})" for d in draw_sets if d.model_fingerprint != fp]
        if bad:
            raise WorldMismatchError(f"draws not generated by model {self.model_id} ({fp[:12]}): {bad}")

    def nominal_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Target means ``(theta [I, J], d [I])`` on the model axes (what a nominal method plans with)."""
        tm, _, dm, _ = self._arrays("target")
        return tm, dm

    def achieved_moments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analytic moments of the sampled marginals ``(theta_mean, theta_sd, d_mean, d_sd)``."""
        return self._arrays("achieved")

    def bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Physical ranges ``(theta_lower, theta_upper, d_lower, d_upper)`` on the model axes."""
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        tl, tu, dl, du = np.zeros((I, J)), np.zeros((I, J)), np.zeros(I), np.zeros(I)
        for c in self.metadata.cells:
            if c.ingredient_id not in self.ingredient_ids:
                continue
            i = self.ingredient_ids.index(c.ingredient_id)
            if c.component == DM_COMPONENT:
                dl[i], du[i] = c.lower, c.upper
            else:
                j = self.nutrient_ids.index(c.component)
                tl[i, j], tu[i, j] = c.lower, c.upper
        return tl, tu, dl, du

    def _arrays(self, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        tm, ts = np.full((I, J), np.nan), np.full((I, J), np.nan)
        dm, ds = np.full(I, np.nan), np.full(I, np.nan)
        for c in self.metadata.cells:
            if c.ingredient_id not in self.ingredient_ids:
                continue
            i = self.ingredient_ids.index(c.ingredient_id)
            if which == "target":
                m, s = c.target_mean, c.target_sd
            else:
                m, s = (c.achieved_mean, c.achieved_sd) if c.included else (float("nan"), float("nan"))
            if c.component == DM_COMPONENT:
                dm[i], ds[i] = m, s
            else:
                j = self.nutrient_ids.index(c.component)
                tm[i, j], ts[i, j] = m, s
        return tm, ts, dm, ds

    def box(self, k: float, *, set_id: Optional[str] = None, d_min: float = 1e-6):
        """Robust box ``achieved mean -/+ k achieved SD`` clipped to the ranges (same world).

        See :meth:`ration_reliability.optimization.robust.BoxUncertaintySet.from_factory_model`."""
        from ..optimization.robust import BoxUncertaintySet   # lazy: optimization imports uncertainty

        return BoxUncertaintySet.from_factory_model(self, k, set_id=set_id, d_min=d_min)

    def prior_states(self, streams: RandomStreams, stream: str, n_draws: int, *sub: int):
        """Information-value prior from this model's draws (development streams only)."""
        from ..information.prior import PriorStates   # lazy: information imports uncertainty

        return PriorStates.from_drawset(self.draw(streams, stream, n_draws, *sub),
                                        consumer=f"FactoryModel.prior_states[{self.model_id}]")

    def prior_variance_basis(self, components: Optional[Sequence] = None) -> dict:
        """``ObservedComponent -> variance_basis`` from the metadata (not from a caller declaration).

        ``components`` (``ObservedComponent`` or ``(ingredient_id, component)``) restricts the
        result; a requested component that is not a stochastic cell of the model raises."""
        from ..information.prior import ObservedComponent

        vb = self.metadata.variance_basis_map(stochastic_only=True)
        if components is None:
            return {ObservedComponent(i, c): b for (i, c), b in vb.items()}
        out = {}
        for comp in components:
            key = (comp.ingredient_id, comp.component) if hasattr(comp, "ingredient_id") else tuple(comp)
            if key not in vb:
                raise KeyError(f"{key[0]}:{key[1]} is not a stochastic cell of model {self.model_id}")
            out[ObservedComponent(*key)] = vb[key]
        return out

    def check_problem(self, problem, *, require_nominal_equal: bool = True, atol: float = 1e-15) -> dict:
        """Check that ``problem`` uses exactly this model's axes (excluded ingredients absent) and,
        optionally, that its nominal composition / d_hat equal the target means."""
        excluded = {i for i, _ in self.metadata.excluded_ingredients}
        offered = sorted(excluded & set(problem.ingredient_ids))
        if offered:
            raise WorldMismatchError(f"problem still offers ingredient(s) excluded by the factory: {offered}")
        if tuple(problem.ingredient_ids) != self.ingredient_ids or tuple(problem.nutrient_ids) != self.nutrient_ids:
            raise WorldMismatchError("problem axes differ from the model axes (order matters)")
        tm, dm = self.nominal_state()
        th0 = np.asarray(problem.nominal_theta(), float)
        d0 = np.asarray(problem.dm_estimates(), float)
        both = ~np.isnan(tm) & ~np.isnan(th0)
        dt = float(np.max(np.abs(tm[both] - th0[both]))) if both.any() else 0.0
        dd = float(np.max(np.abs(dm - d0))) if dm.size else 0.0
        nan_mismatch = int(np.sum(np.isnan(tm) != np.isnan(th0)))
        rep = {"max_abs_theta_nominal_minus_target": dt, "max_abs_d_hat_minus_target": dd,
               "n_missing_pattern_mismatch": nan_mismatch}
        if require_nominal_equal and (dt > atol or dd > atol or nan_mismatch):
            raise WorldMismatchError(f"problem nominal values differ from the model's target means: {rep}")
        return rep

    def extension(self, name: str) -> ExtensionScenario:
        for e in self.extensions:
            if e.metadata.name == name:
                return e
        raise KeyError(f"no extension scenario {name!r}")


# ------------------------------------------------------------------------------------------------
# fitting helpers
# ------------------------------------------------------------------------------------------------

_STATUS_MAP = {"matched": "matched", "point": "point", "missing": "missing", "drifted": "diagnostic_drift",
               "not_matchable": "infeasible_moment_match", "infeasible_any_family": "infeasible_moment_match",
               "invalid_target": "infeasible_moment_match"}


def _fit(family: str, cell: CellSpec) -> tuple[Optional[MarginalFit], FitAttempt]:
    fam = "TN_NAIVE" if family == FAMILY_TN_NAIVE_DIAGNOSTIC else family
    try:
        f = fit_marginal(fam, cell.mean, cell.sd, cell.lower, cell.upper)
    except ValueError as exc:
        return None, FitAttempt(family, "error", "infeasible_moment_match", ("family_not_applicable",), str(exc))
    fs = _STATUS_MAP.get(f.status)
    if fs is None:   # missing_sd / diagnostic_only -- excluded by the spec validation
        return None, FitAttempt(family, f.status, "infeasible_moment_match", tuple(f.flags), f.message)
    return f, FitAttempt(family, f.status, fs, tuple(f.flags), f.message)


def _missing_fit(cell: CellSpec, note: str) -> MarginalFit:
    return MarginalFit("TN_MM", cell.mean, cell.sd, cell.lower, cell.upper, "missing", message=note)


def _cell_meta(cell: CellSpec, spec: UncertaintySpec, *, included: bool, exclusion_action: Optional[str],
               requested: Optional[str], family: Optional[str], reason: str, attempts: Sequence[FitAttempt],
               fit_status: str, fit: Optional[MarginalFit]) -> CellModelMetadata:
    diag = family == FAMILY_TN_NAIVE_DIAGNOSTIC
    nan = float("nan")
    return CellModelMetadata(
        cell.ingredient_id, cell.component, included, exclusion_action, cell.variance_basis, cell.data_fingerprint,
        cell.decomposition_id, cell.decomposition_source, cell.measurement_model_id, cell.provenance_status,
        cell.source_id, cell.locator, cell.provenance_status == "synthetic_test_only", requested, family, reason,
        tuple(attempts), fit_status, bool(diag or spec.purpose == "diagnostic"),
        bool(diag and fit is not None and fit.status == "drifted"),
        float(cell.mean), float(cell.sd),
        nan if fit is None else float(fit.achieved_mean), nan if fit is None else float(fit.achieved_sd),
        float(cell.lower), float(cell.upper), () if fit is None else tuple(float(p) for p in fit.params),
        nan if fit is None else float(fit.mean_rel_shift), nan if fit is None else float(fit.sd_rel_error),
        nan if fit is None else float(fit.outside_mass), () if fit is None else tuple(fit.flags),
        "" if fit is None else str(fit.message))


# ------------------------------------------------------------------------------------------------
# transformed-scale correlation
# ------------------------------------------------------------------------------------------------

#: Uniform scores are kept in [_U_EPS, 1 - _U_EPS] (|z| <= ~8.2) before the inverse CDF: beyond it the
#: quadrature weights are below 1e-15 and some inverse CDFs (beta) stop converging in double precision.
_U_EPS = 1e-16


def _quad_values(fit: MarginalFit, z: np.ndarray) -> np.ndarray:
    u = np.clip(special.ndtr(z), _U_EPS, 1.0 - _U_EPS)
    return np.asarray(fit.ppf(u), dtype=float)


def transformed_pearson(fit_a: MarginalFit, fit_b: MarginalFit, rho_latent: float) -> float:
    """Pearson correlation of ``(F_a^-1(Phi(Z1)), F_b^-1(Phi(Z2)))`` for latent ``corr(Z1, Z2) = rho``.

    Deterministic 2-D Gauss-Hermite quadrature (``_GH_N`` nodes per axis).  For moment-matched
    marginals the result differs from ``rho`` unless both marginals are normal (review R3 item 5).
    """
    if not (fit_a.is_stochastic and fit_b.is_stochastic):
        raise ValueError("transformed_pearson needs two stochastic marginals")
    r = float(np.clip(rho_latent, -1.0, 1.0))
    x = _GH_X
    W = _GH_W[:, None] * _GH_W[None, :]
    va = _quad_values(fit_a, x)[:, None] * np.ones((1, _GH_N))
    z2 = r * x[:, None] + math.sqrt(max(1.0 - r * r, 0.0)) * x[None, :]
    vb = _quad_values(fit_b, z2)
    ma, mb = float((W * va).sum()), float((W * vb).sum())
    cov = float((W * (va - ma) * (vb - mb)).sum())
    sa = math.sqrt(float((W * (va - ma) ** 2).sum()))
    sb = math.sqrt(float((W * (vb - mb) ** 2).sum()))
    return cov / (sa * sb)


def calibrate_latent_rho(fit_a: MarginalFit, fit_b: MarginalFit, target_pearson: float
                         ) -> tuple[Optional[float], tuple[float, float]]:
    """Latent ``rho`` whose transformed-scale Pearson equals ``target_pearson``.

    Returns ``(rho, (attainable_min, attainable_max))``; ``rho`` is ``None`` when the target lies
    outside the attainable range of the two marginals (never forced to the boundary).
    """
    t = float(target_pearson)
    lo, hi = transformed_pearson(fit_a, fit_b, -_RHO_EDGE), transformed_pearson(fit_a, fit_b, _RHO_EDGE)
    if not (lo - 1e-10 <= t <= hi + 1e-10):
        return None, (lo, hi)
    if t == 0.0:
        return 0.0, (lo, hi)
    if t <= lo:
        return -_RHO_EDGE, (lo, hi)
    if t >= hi:
        return _RHO_EDGE, (lo, hi)
    rho = optimize.brentq(lambda r: transformed_pearson(fit_a, fit_b, r) - t, -_RHO_EDGE, _RHO_EDGE,
                          xtol=1e-13, rtol=4 * np.finfo(float).eps, maxiter=300)
    return float(rho), (lo, hi)


# ------------------------------------------------------------------------------------------------
# development-data family selection
# ------------------------------------------------------------------------------------------------

def _logpdf(fit: MarginalFit, x: np.ndarray) -> np.ndarray:
    lo, hi = float(fit.lower), float(fit.upper)
    inside = (x >= lo) & (x <= hi)
    out = np.full(x.shape, -np.inf)
    if fit.family in ("TN_MM", "TN_NAIVE"):
        mu, sig = fit.params
        out[inside] = stats.truncnorm.logpdf(x[inside], (lo - mu) / sig, (hi - mu) / sig, loc=mu, scale=sig)
    elif fit.family == "BETA_MM":
        a, b = fit.params
        w = hi - lo
        out[inside] = stats.beta.logpdf((x[inside] - lo) / w, a, b) - math.log(w)
    elif fit.family == "LN_MM":
        mu, sig = fit.params
        norm = 0.0 if not math.isfinite(hi) else float(special.log_ndtr((math.log(hi - lo) - mu) / sig))
        pos = inside & (x > lo)
        out[pos] = stats.lognorm.logpdf(x[pos] - lo, s=sig, scale=math.exp(mu)) - norm
    return out


def select_family_on_development_data(cell: CellSpec, observations, *, split: Optional[str] = None,
                                      data_label: str, candidates: Sequence[str] = MOMENT_MATCHED_FAMILIES
                                      ) -> FamilySelection:
    """Choose a family for ``cell`` by the negative log-likelihood of development observations
    under each candidate fitted to the cell's *target* moments (the targets are not re-estimated).

    ``observations`` is a 1-D array with an explicit ``split`` (one of :data:`DEVELOPMENT_SPLITS`)
    or a :class:`DrawSet` from the ``opt`` / ``validation`` stream.  Test data raise
    :class:`LeakageError` (review R3 item 8).  Candidates that cannot match the target score ``inf``.
    """
    if isinstance(observations, DrawSet):
        require_stream(observations, ("opt", "validation"), "select_family_on_development_data")
        i = observations.ingredient_ids.index(cell.ingredient_id)
        x = observations.d[:, i] if cell.component == DM_COMPONENT else \
            observations.theta[:, i, observations.nutrient_ids.index(cell.component)]
        split = observations.stream
    else:
        if split not in DEVELOPMENT_SPLITS:
            raise LeakageError(f"family selection may use development data only {DEVELOPMENT_SPLITS}; got split "
                               f"{split!r} (the formal test set is never used to choose a family)")
        x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("observations must be a finite 1-D array with at least 2 values")
    if not cell.is_stochastic_target:
        raise ValueError(f"{cell.label}: family selection needs a stochastic target (SD > 0)")
    scores = []
    for fam in candidates:
        if fam not in MOMENT_MATCHED_FAMILIES:
            raise ValueError(f"candidate {fam!r} is not a moment-matched family")
        f, att = _fit(fam, cell)
        if f is None or att.fit_status != "matched":
            scores.append((fam, float("inf"), att.fit_status))
            continue
        nll = float(-np.sum(_logpdf(f, x)))
        scores.append((fam, nll if math.isfinite(nll) else float("inf"), "matched"))
    finite = [s for s in scores if math.isfinite(s[1])]
    if not finite:
        raise ValueError(f"{cell.label}: no candidate family has a finite likelihood on the development data")
    chosen = min(finite, key=lambda s: (s[1], candidates.index(s[0])))[0]
    fp = stable_hash("dev_obs/v1", x, str(split), str(data_label))
    reason = (f"development-data selection on split {split!r} ({data_label}, n={x.size}): lowest negative "
              f"log-likelihood at the target moments among {list(candidates)}")
    return FamilySelection(cell.ingredient_id, cell.component, str(split), str(data_label), fp, int(x.size),
                           float(cell.mean), float(cell.sd), tuple(candidates), tuple(scores), chosen,
                           "neg_log_likelihood_at_target_moments", reason)


# ------------------------------------------------------------------------------------------------
# two-layer extension scenario
# ------------------------------------------------------------------------------------------------

def _build_two_layer(spec: UncertaintySpec) -> ExtensionScenario:
    tl = spec.two_layer
    assert tl is not None
    cfg_fp = stable_hash("TwoLayerSpec/v1", tl.to_config(), spec.main_fingerprint())
    notes: list[str] = ["extension scenario only: never a dependency of the main model (review R3 item 7)"]
    if tl.role == "primary":
        notes.append("demoted_from_primary: the two-layer farm model cannot be the primary world; it is built (if at "
                     "all) as an extension scenario next to the unchanged main model")
    ratio_map = tl.ratio_map()
    covered = [spec.cell(i, c) for (i, c) in ratio_map]
    bases = sorted({c.variance_basis for c in covered if c.is_stochastic_target})

    def done(status, reasons, basis=None, model=None):
        return ExtensionScenario(ExtensionMetadata("two_layer_farm", tl.role, "extension_scenario", status,
                                                   tuple(reasons), basis, tuple(notes), cfg_fp), model)

    if not bases:
        return done("invalid_extension_config", ["no covered stochastic cell (ratios only on point/missing cells)"])
    if len(bases) > 1:
        return done("invalid_extension_config", [f"covered cells have mixed variance bases {bases}; the truth-side "
                                                 "basis must be one object property, not a declaration"])
    truth_basis = bases[0]
    reasons = []
    if tl.ratio_status != "sourced":
        reasons.append(f"within/between ratios are {tl.ratio_status!r}, not sourced: the split of the variance into "
                       "farm and batch layers is a scenario, not identified")
    if truth_basis != "true_batch_state":
        reasons.append(f"cell SDs have variance basis {truth_basis!r}: sigma_between^2 = SD^2 - sigma_within^2 absorbs "
                       "sampling/lab error (between_layer_contains_measurement_error), the farm layer is not identified")
    if reasons and not tl.allow_unidentified_scenario:
        return done("not_built_unidentified", reasons, truth_basis)
    tm, ts, dm, ds = spec.target_arrays()
    I, J = len(spec.ingredient_ids), len(spec.nutrient_ids)
    tr, dr = np.full((I, J), np.nan), np.full(I, np.nan)
    for (iid, comp), r in ratio_map.items():
        i = spec.ingredient_ids.index(iid)
        if comp == DM_COMPONENT:
            dr[i] = r
        else:
            tr[i, spec.nutrient_ids.index(comp)] = r
    th_lo = {c.lower for c in spec.cells if c.component != DM_COMPONENT}
    th_hi = {c.upper for c in spec.cells if c.component != DM_COMPONENT}
    d_lo = {c.lower for c in spec.cells if c.component == DM_COMPONENT}
    d_hi = {c.upper for c in spec.cells if c.component == DM_COMPONENT}
    if len(th_lo) != 1 or len(th_hi) != 1 or len(d_lo) != 1 or len(d_hi) != 1:
        return done("invalid_extension_config", ["the two-layer model needs one composition range and one DM range"],
                    truth_basis)
    src = "; ".join(sorted({c.decomposition_source for c in covered if c.decomposition_source}))
    try:
        model = TwoLayerFarmModel(
            f"{spec.spec_id}/two_layer", spec.ingredient_ids, spec.nutrient_ids, tm, ts, dm, np.nan_to_num(ds, nan=0.0),
            tr, dr, tl.ratio_basis, tl.s, tl.family, tl.uncovered_cells, theta_lower=th_lo.pop(),
            theta_upper=th_hi.pop(), d_lower=d_lo.pop(), d_upper=d_hi.pop(), is_synthetic=spec.is_synthetic,
            truth_sd_basis=truth_basis, truth_sd_source=src)
    except ValueError as exc:
        return done("invalid_extension_config", reasons + [f"TwoLayerFarmModel rejected the configuration: {exc}"],
                    truth_basis)
    return done("identified_extension_scenario" if not reasons else "unidentified_extension_scenario", reasons,
                truth_basis, model)


# ------------------------------------------------------------------------------------------------
# the factory
# ------------------------------------------------------------------------------------------------

def build_uncertainty_model(spec: UncertaintySpec, *, model_id: Optional[str] = None,
                            family_selection: Optional[Mapping[tuple[str, str], FamilySelection]] = None
                            ) -> FactoryModel:
    """Build the one model of ``spec`` (see the module documentation for the rules).

    Raises :class:`FactoryBuildError` with every issue (a cell that no declared family can match under
    ``on_exhausted='error'``, a DM cell that would have to become missing, a correlation label on an
    excluded or non-stochastic cell, a non-PSD or unattainable correlation, an incomplete
    development-data family selection, ...).
    """
    if not isinstance(spec, UncertaintySpec):
        raise TypeError("build_uncertainty_model needs an UncertaintySpec (UncertaintySpec.from_config)")
    rule = spec.family_rule
    issues: list[str] = []
    sel = dict(family_selection or {})
    if rule.selection_basis == "declared_rule" and sel:
        issues.append("family_selection given but family_rule.selection_basis is 'declared_rule'")
    if rule.selection_basis == "development_data":
        cell_keys = {c.key for c in spec.cells}
        for key, s in sel.items():
            if not isinstance(s, FamilySelection) or s.key != tuple(key) or tuple(key) not in cell_keys:
                issues.append(f"family_selection[{key}]: must be the FamilySelection of a cell of the specification")
                continue
            if s.split not in DEVELOPMENT_SPLITS:
                issues.append(f"family_selection[{key}]: split {s.split!r} is not development data (LeakageError rule)")
            c = spec.cell(*key)
            if (s.target_mean, s.target_sd) != (float(c.mean), float(c.sd)):
                issues.append(f"family_selection[{key}]: selected at different target moments than the specification")
        need = [c.label for c in spec.cells if c.is_stochastic_target and c.family is None and c.key not in sel]
        if need:
            issues.append(f"selection_basis 'development_data' but no development-data selection for {need}")

    metas: dict[tuple[str, str], CellModelMetadata] = {}
    fits: dict[tuple[str, str], MarginalFit] = {}
    excluded: dict[str, str] = {}
    for cell in spec.cells:
        if cell.is_missing:
            f = _missing_fit(cell, "mean missing (null is never 0)")
            metas[cell.key] = _cell_meta(cell, spec, included=True, exclusion_action=None, requested=None, family=None,
                                         reason="not applicable (missing value)", attempts=(), fit_status="missing",
                                         fit=f)
            fits[cell.key] = f
            continue
        if cell.is_point:
            f, att = _fit("TN_MM", cell)
            if f is None or f.status != "point":
                issues.append(f"cell {cell.label}: point value outside [{cell.lower}, {cell.upper}] ({att.message})")
                continue
            metas[cell.key] = _cell_meta(cell, spec, included=True, exclusion_action=None, requested=None, family=None,
                                         reason="not applicable (SD = 0, point value)", attempts=(), fit_status="point",
                                         fit=f)
            fits[cell.key] = f
            continue
        if cell.family is not None:
            requested, reason = cell.family, f"per-cell override: {cell.family_choice_reason}"
        elif cell.key in sel:
            requested, reason = sel[cell.key].chosen, sel[cell.key].reason
        else:
            requested, reason = rule.primary_family, f"family_rule ({rule.status}): {rule.rationale}"
        chain = [requested] + [f for f in rule.fallback_families if f != requested]
        attempts: list[FitAttempt] = []
        chosen: Optional[MarginalFit] = None
        chosen_family = None
        for fam in chain:
            f, att = _fit(fam, cell)
            attempts.append(att)
            if f is not None and att.fit_status in ("matched", "diagnostic_drift"):
                chosen, chosen_family = f, fam
                break
        if chosen is not None:
            if chosen_family != requested:
                failed = "; ".join(f"{a.family}: {a.fit_status} ({','.join(a.flags) or a.distribution_status})"
                                   for a in attempts[:-1])
                reason = (f"{reason} | fallback to {chosen_family} (next in the declared fallback list "
                          f"{list(rule.fallback_families)}) after {failed}")
            metas[cell.key] = _cell_meta(cell, spec, included=True, exclusion_action=None, requested=requested,
                                         family=chosen_family, reason=reason, attempts=attempts,
                                         fit_status=attempts[-1].fit_status, fit=chosen)
            fits[cell.key] = chosen
            continue
        # exhausted: every declared family reported infeasible_moment_match
        action = rule.on_exhausted
        trail = "; ".join(f"{a.family}: {a.distribution_status} {','.join(a.flags)}".strip() for a in attempts)
        if action == "error":
            issues.append(f"cell {cell.label}: infeasible_moment_match for every declared family ({trail}); "
                          "on_exhausted='error' (targets are not changed)")
            continue
        if action == "exclude_cell_as_missing" and cell.component == DM_COMPONENT:
            issues.append(f"cell {cell.label}: a DM cell cannot be excluded as missing (q needs a DM fraction); "
                          "use on_exhausted='exclude_ingredient' or 'error'")
            continue
        f = _missing_fit(cell, f"excluded after infeasible_moment_match ({trail}); sampled as NaN, never 0")
        metas[cell.key] = _cell_meta(cell, spec, included=(action == "exclude_cell_as_missing"),
                                     exclusion_action=action, requested=requested, family=None,
                                     reason=f"{reason} | exhausted: {action}", attempts=attempts,
                                     fit_status="infeasible_moment_match", fit=None)
        fits[cell.key] = f
        if action == "exclude_ingredient":
            excluded.setdefault(cell.ingredient_id, f"{cell.label}: infeasible_moment_match for every declared family")

    for key, meta in list(metas.items()):     # every cell of an excluded ingredient leaves the model
        if meta.ingredient_id in excluded and meta.exclusion_action is None:
            metas[key] = replace(meta, included=False, exclusion_action="ingredient_excluded",
                                 family_choice_reason=f"{meta.family_choice_reason} | ingredient excluded: "
                                                      f"{excluded[meta.ingredient_id]}")
    ing = tuple(i for i in spec.ingredient_ids if i not in excluded)
    nut = tuple(spec.nutrient_ids)
    if not ing:
        issues.append("every ingredient was excluded")
    corr_struct, corr_meta = None, None
    if spec.correlation is not None and not issues:
        corr_struct, corr_meta = _build_correlation(spec, fits, metas, excluded, issues)
    if issues:
        raise FactoryBuildError(issues)

    I, J = len(ing), len(nut)
    tf = np.empty((I, J), dtype=object)
    df = []
    for a, iid in enumerate(ing):
        for b, nid in enumerate(nut):
            tf[a, b] = fits[(iid, nid)]
        df.append(fits[(iid, DM_COMPONENT)])
    mid = model_id or f"factory[{spec.spec_id}]"
    try:
        inner = GaussianCopulaModel(mid + "/copula", ing, nut, tf, df, corr_struct, spec.is_synthetic,
                                    allow_drifted=spec.uses_naive_family())
    except (ValueError, NotPSDError) as exc:
        raise FactoryBuildError([f"copula model rejected the fitted marginals: {exc}"]) from None
    cells_meta = tuple(metas[c.key] for c in spec.cells)
    summary = _summary(cells_meta, excluded)
    meta = ModelMetadata(FACTORY_VERSION, spec.spec_id, spec.main_fingerprint(), mid, spec.purpose,
                         spec.moment_semantics, bool(spec.purpose == "diagnostic" or spec.uses_naive_family()),
                         bool(spec.is_synthetic), tuple(spec.ingredient_ids), ing, nut,
                         tuple(sorted(excluded.items())), cells_meta, corr_meta,
                         tuple(sorted(rule.to_config().items(), key=lambda kv: kv[0])),
                         tuple(sel[k] for k in sorted(sel)), tuple(sorted(summary.items())))
    exts = (_build_two_layer(spec),) if spec.two_layer is not None else ()
    model = FactoryModel(mid, ing, nut, bool(spec.is_synthetic), inner, meta, spec, exts)
    _REGISTRY[model.fingerprint()] = meta
    return model


def _build_correlation(spec, fits, metas, excluded, issues):
    cs = spec.correlation
    for lab in cs.labels:
        if lab[0] in excluded:
            issues.append(f"correlation label {lab[0]}:{lab[1]}: ingredient excluded by the factory")
        elif not metas[lab].is_stochastic:
            issues.append(f"correlation label {lab[0]}:{lab[1]}: cell is not stochastic after fitting "
                          f"({metas[lab].fit_status})")
    if issues:
        return None, None
    m_in = cs.matrix_array()
    n = m_in.shape[0]
    latent = np.eye(n)
    calib = []
    for a in range(n):
        for b in range(a + 1, n):
            r = float(m_in[a, b])
            fa, fb = fits[cs.labels[a]], fits[cs.labels[b]]
            if cs.handling == "calibrated_mapping":
                rho, (lo, hi) = calibrate_latent_rho(fa, fb, r)
                if rho is None:
                    issues.append(f"correlation {cs.labels[a]}~{cs.labels[b]}: transformed-scale Pearson {r} is "
                                  f"outside the attainable range [{lo:.4f}, {hi:.4f}] of these marginals (not forced)")
                    continue
                latent[a, b] = latent[b, a] = rho
            else:   # latent_as_declared / declared_latent_scenario
                latent[a, b] = latent[b, a] = r
    if issues:
        return None, None
    chk = check_psd(latent)
    if not chk.is_psd:
        issues.append(f"correlation {cs.structure_id}: latent matrix not PSD (lambda_min {chk.lambda_min:.6g}); "
                      "a structure that needs repair is a definition error (no silent Higham repair)")
        return None, None
    trans = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            p = transformed_pearson(fits[cs.labels[a]], fits[cs.labels[b]], latent[a, b])
            trans[a, b] = trans[b, a] = p
            if cs.handling == "calibrated_mapping":
                calib.append((f"{cs.labels[a][0]}:{cs.labels[a][1]}", f"{cs.labels[b][0]}:{cs.labels[b][1]}",
                              float(m_in[a, b]), float(latent[a, b]), float(p)))
    interp = {
        "latent_as_declared": "matrix is the latent Gaussian-copula correlation; the transformed-scale Pearson "
                              "correlation of the sampled values is reported separately and differs",
        "declared_latent_scenario": "a transformed-scale (e.g. literature) Pearson r is used AS the latent correlation "
                                    "by declaration (research scenario); the sampled values then have the reported "
                                    "transformed correlation, which is not r",
        "calibrated_mapping": "latent correlation solved pair by pair so that the transformed-scale Pearson equals the "
                              "input r (quadrature); the joint matrix is then checked for PSD without repair",
    }[cs.handling]
    label = cs.structure_id + ("" if cs.handling != "calibrated_mapping" else "+latent_calibrated")
    struct = CorrelationStructure(label, cs.labels, latent, cs.status, cs.provenance,
                                  (f"input_scale={cs.input_scale}", f"handling={cs.handling}"))
    off = ~np.eye(n, dtype=bool)
    meta = CorrelationMetadata(cs.structure_id, cs.labels, cs.input_scale, cs.handling, cs.status, cs.provenance,
                               cs.source_id, cs.locator, tuple(tuple(float(x) for x in r) for r in m_in),
                               tuple(tuple(float(x) for x in r) for r in latent),
                               tuple(tuple(float(x) for x in r) for r in trans), float(chk.lambda_min),
                               float(np.max(np.abs(latent - trans)[off])) if n > 1 else 0.0, tuple(calib), interp)
    return struct, meta


def _summary(cells: Sequence[CellModelMetadata], excluded: Mapping[str, str]) -> dict:
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for c in cells:
        by_status[c.fit_status] = by_status.get(c.fit_status, 0) + 1
        if c.family is not None:
            by_family[c.family] = by_family.get(c.family, 0) + 1
    matched = [c for c in cells if c.fit_status == "matched"]
    worst = max((max(abs((c.achieved_mean - c.target_mean) / c.target_sd), abs(c.achieved_sd / c.target_sd - 1.0))
                 for c in matched), default=0.0)
    return {"n_cells": len(cells), "fit_status_counts": by_status, "family_counts": by_family,
            "n_fallback_cells": sum(1 for c in cells if c.used_fallback),
            "fallback_cells": [c.label for c in cells if c.used_fallback],
            "n_infeasible_moment_match_attempts": sum(1 for c in cells for a in c.attempts
                                                      if a.fit_status == "infeasible_moment_match"),
            "excluded_cells": [c.label for c in cells if c.exclusion_action is not None],
            "excluded_ingredients": sorted(excluded),
            "n_diagnostic_drift_cells": by_status.get("diagnostic_drift", 0),
            "max_matched_moment_error_in_target_sd": float(worst)}


# ------------------------------------------------------------------------------------------------
# registry (convenience lookup from a DrawSet to the metadata of its model)
# ------------------------------------------------------------------------------------------------

def metadata_for_fingerprint(model_fingerprint: str) -> Optional[ModelMetadata]:
    """Metadata of a model built in this process, by its fingerprint (``None`` if unknown).

    The authoritative copy is :attr:`FactoryModel.metadata`, which run records serialise; this
    lookup lets code that only holds a :class:`DrawSet` (e.g. ``PriorStates.from_drawset``) bind the
    object metadata instead of trusting a caller's declaration."""
    return _REGISTRY.get(str(model_fingerprint))


def metadata_for_drawset(draws: DrawSet) -> Optional[ModelMetadata]:
    """:func:`metadata_for_fingerprint` of ``draws.model_fingerprint``."""
    return metadata_for_fingerprint(draws.model_fingerprint)
