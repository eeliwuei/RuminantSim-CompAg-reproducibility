"""Assay signal model ``Z = theta_current + e_sampling + e_lab + b_systematic`` (contract T7.5).

Components are kept separate (never collapsed into one "assay SD" before the protocol is known):

``sampling_sd``
    Representativeness of one field sample for the *current batch* (grab / probe / face sample).
    Independent field samples reduce its variance as ``1/m`` (assumed independent).
``lab_repeatability_sd``
    Within-laboratory random analytical error of **one** analysis.  Replicate analyses of the same
    ground (composite) sample reduce only this part (variance / r).
``lab_bias_sd``
    Standard deviation of the unknown between-laboratory / method bias of the laboratory used
    (e.g. an SDLab-type quantity).  It is common to all replicates and samples sent to the same
    laboratory, so it is **not** reduced by replication.
``bias_mean``
    A *known* systematic offset (e.g. a documented instrument calibration offset).  It enters the
    likelihood and is therefore accounted for by the policy optimisation.

``lab_error_semantics`` must be declared explicitly for every component:

* ``"repeatability_only"`` -- ``lab_repeatability_sd`` is a within-lab repeatability SD;
  ``lab_bias_sd`` may be added separately.
* ``"single_result_total"`` -- ``lab_repeatability_sd`` is the total SD of a single reported result
  (already containing the between-lab part).  Replicates are then assumed **not** to reduce it
  (the reducible share is unknown) and a separate ``lab_bias_sd`` would double count -> rejected by
  :func:`double_count_guard`.

Additive Gaussian errors in canonical units are the only implemented error scale; bounded
compositions are *not* truncated in the signal (the signal is only binned; see the design note).

**Double counting (contract 7.3, F06)**: if the prior SD of theta was estimated from historical
laboratory results, it already contains sampling and analytical error.  Using it as the true
batch-to-batch variance *and* adding the same errors to the signal double counts.  The prior basis
must be declared per component and checked with :func:`double_count_guard`;
:func:`deconvolve_true_sd` gives the de-convolved SD when the error components are identified.

No numerical values of sampling or laboratory error are provided in this module.  Values must
come from registered sources (e.g. the variance-component studies listed in
``audit/phase1_survey_20260924/DATA_FEASIBILITY_AUDIT.md`` section 4) or be declared scenario
assumptions; unit tests use synthetic values only.

Prior / truth / signal guard (second review R2, 2026-09-25)
-----------------------------------------------------------
:func:`double_count_guard` has two modes.

* Declaration mode (``prior=None``, ``truth=None``; unchanged): checks that caller *declarations*
  (``prior_variance_basis`` / ``prior_deconvolved``) are consistent with the signal.  It says
  nothing about the object the states were generated from.
* Object mode (``prior=...`` and optionally ``truth=...``): the basis of each signal component is
  read from the object metadata of the prior (:class:`~.prior.StateMetadata`) or of the linked truth
  model; a declaration that differs raises :class:`~.prior.MetadataConflictError` (declarations never
  override objects); prior and truth metadata must agree; an *unlinked* truth record and bare
  declarations can only make the verdict stricter.  Components without object metadata are recorded
  as ``caller_declaration_unbound`` -- :func:`~.value.compute_information_value` then labels the
  value ``unidentified_scenario`` (computed, never reported as identified).  For observed/unidentified
  bases the report records ``prior variance - single-result error variance``; when the signal's
  ``measurement_model_id`` is the prior's measurement model and this difference is negative, the
  decomposition is inconsistent and the component is rejected (never clipped to 0).

:func:`trace_error_provenance` ties every sourced error value of a signal to a row of
``sources/error_source_locators.csv`` (parameter id, registry source, error component class) and
refuses parameters that contain true batch variation or do not match the error field.

Red-team FIX_A (2026-09-25) -- the object mode additionally

* uses a truth record as the basis source only when its link is *verified*
  (:func:`~.prior.resolve_truth_link`; a copied model fingerprint is only a claim);
* checks a de-convolved ``true_batch_state`` numerically: the attached
  :class:`~.prior.VarianceDecomposition` must exist (non-synthetic: required; synthetic: otherwise the
  value cannot be identified), its ``true_sd`` must equal the SD of the prior states (exact for a
  discrete prior, a declared Monte Carlo tolerance for draws), and for the *same* measurement process
  the signal's single-result error parameters must equal the removed ones;
* requires the signal to declare ``measurement_model_id`` whenever the prior basis contains a named
  measurement process and the decomposition check subtracts a non-zero error; for an unlinked
  process a negative difference rejects the de-convolved member of the scenario range (recorded; the
  computed value is only the "all true variation" member);
* traces every ``sourced`` error value to the locator table (:func:`trace_error_provenance`; default
  table :func:`default_error_locators`); any issue -- true batch variation used as measurement error, a
  total single-result SD declared as repeatability, another nutrient, a blocked value, a registry-source
  mismatch, or a non-synthetic ``measurement_model_id`` that is not the traced ``param_id`` -- is a
  violation.  Without a locator table the report says so and the value is ``error_model_unsourced``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from ..datamodel import AssaySpec, Provenance, SourcedValue, ValueStatus
from ..errors import InvalidProblemError
from ..hashing import stable_hash
from .prior import (
    DISCRETE_PRIOR_STREAM,
    DM_COMPONENT,
    NOT_APPLICABLE_BASIS,
    MetadataConflictError,
    ObservedComponent,
    PriorStates,
    _comp,
    decompose_observed_variance,
    resolve_truth_link,
    verify_state_metadata,
)

__all__ = [
    "LAB_ERROR_SEMANTICS",
    "PRIOR_VARIANCE_BASES",
    "BASIS_SOURCES",
    "ERROR_FIELD_COMPONENTS",
    "ComponentErrorModel",
    "SamplingProtocol",
    "SignalModel",
    "signal_model_from_assay",
    "DoubleCountReport",
    "double_count_guard",
    "deconvolve_true_sd",
    "load_error_locators",
    "default_error_locators",
    "trace_error_provenance",
    "TRUE_STATE_SD_MC_Z",
]

#: Width (in normal-approximation standard errors of a sample variance) of the tolerance used when the
#: SD of *drawn* prior states is compared with a recorded de-convolved true SD (FIX_A); a discrete
#: (exact) prior is compared to relative 1e-9.  A declared numerical tolerance, not a fitted value.
TRUE_STATE_SD_MC_Z = 6.0

#: Allowed declarations of what ``lab_repeatability_sd`` means.
LAB_ERROR_SEMANTICS = ("repeatability_only", "single_result_total")

#: Allowed declarations of what the prior SD of a component represents.
PRIOR_VARIANCE_BASES = (
    "true_batch_state",                      # variance of the true batch state (no measurement error)
    "observed_incl_sampling_and_lab",        # SD of single historical lab results of field samples
    "observed_incl_lab_only",                # SD of lab results of perfectly representative samples
    "unidentified",                          # unknown mixture -> scenario range only
)

#: Where the basis used for a component came from (object mode of :func:`double_count_guard`).
BASIS_SOURCES = ("object_metadata:prior", "object_metadata:truth", "caller_declaration_unbound")

#: ``sources/error_source_locators.csv`` ``error_components`` admissible for each error field, by
#: ``lab_error_semantics`` (review R2 item 6).  A class containing ``batch_true`` is never admissible:
#: it is (partly) true batch variation, not measurement error of the signal.
ERROR_FIELD_COMPONENTS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "sampling_sd": {"repeatability_only": ("sampling",), "single_result_total": ("sampling",)},
    "lab_repeatability_sd": {"repeatability_only": ("lab_random",),
                             "single_result_total": ("lab_random", "lab_random|lab_bias")},
    "lab_bias_sd": {"repeatability_only": ("lab_bias",), "single_result_total": ("lab_bias",)},
    "bias_mean": {"repeatability_only": ("lab_bias",), "single_result_total": ("lab_bias",)},
}


def _nonneg(name: str, v: float) -> float:
    v = float(v)
    if not np.isfinite(v) or v < 0:
        raise InvalidProblemError(f"{name} must be finite and >= 0, got {v}")
    return v


@dataclass(frozen=True)
class ComponentErrorModel:
    """Error components (canonical units, e.g. kg/kg DM) for one observed component.

    ``provenance`` maps field name (``sampling_sd``, ``lab_repeatability_sd``, ``lab_bias_sd``,
    ``bias_mean``) to a :class:`Provenance`; every non-zero value needs one.

    ``measurement_model_id`` (optional, R2) names the measurement process of this error model; when
    it equals the prior's ``measurement_model_id`` the guard can check the variance decomposition
    (prior variance - error variance must not be negative).
    """

    component: ObservedComponent
    sampling_sd: float
    lab_repeatability_sd: float
    lab_bias_sd: float = 0.0
    bias_mean: float = 0.0
    lab_error_semantics: str = "repeatability_only"
    provenance: Mapping[str, Provenance] = field(default_factory=dict)
    is_synthetic: bool = False
    measurement_model_id: Optional[str] = None

    def __post_init__(self) -> None:
        for n in ("sampling_sd", "lab_repeatability_sd", "lab_bias_sd"):
            object.__setattr__(self, n, _nonneg(n, getattr(self, n)))
        bm = float(self.bias_mean)
        if not np.isfinite(bm):
            raise InvalidProblemError("bias_mean must be finite")
        object.__setattr__(self, "bias_mean", bm)
        if self.lab_error_semantics not in LAB_ERROR_SEMANTICS:
            raise InvalidProblemError(f"lab_error_semantics must be one of {LAB_ERROR_SEMANTICS}")

    def provenance_issues(self) -> list[str]:
        """Missing or invalid provenance of non-zero error values."""
        out: list[str] = []
        for n in ("sampling_sd", "lab_repeatability_sd", "lab_bias_sd", "bias_mean"):
            if getattr(self, n) != 0.0:
                p = self.provenance.get(n)
                where = f"{self.component.label()}.{n}"
                if p is None:
                    out.append(f"{where}: non-zero value without provenance")
                else:
                    out.extend(p.issues(where))
                    if p.status is ValueStatus.SYNTHETIC_TEST_ONLY and not self.is_synthetic:
                        out.append(f"{where}: synthetic provenance in a non-synthetic error model")
        return out


@dataclass(frozen=True)
class SamplingProtocol:
    """How the current batch is sampled and analysed (fixed at t1, contract T1).

    ``n_field_samples`` independent field samples; if ``composite`` they are mixed and analysed
    as one ground sample ``n_lab_replicates`` times; otherwise each field sample is analysed
    ``n_lab_replicates`` times and all results are averaged.  All analyses go to one laboratory
    (so ``lab_bias_sd`` is shared).
    """

    n_field_samples: int = 1
    composite: bool = True
    n_lab_replicates: int = 1

    def __post_init__(self) -> None:
        if int(self.n_field_samples) < 1 or int(self.n_lab_replicates) < 1:
            raise InvalidProblemError("SamplingProtocol: counts must be >= 1")

    @property
    def n_analyses(self) -> int:
        """Number of laboratory analyses (for per-analysis costing)."""
        m, r = int(self.n_field_samples), int(self.n_lab_replicates)
        return r if self.composite else m * r


@dataclass(frozen=True)
class SignalModel:
    """Signal of one assay/panel applied to one ingredient batch at t1.

    ``errors`` lists one :class:`ComponentErrorModel` per observed component (panel = several
    components on the same physical sample).  Errors of different components are treated as
    independent (a declared assumption; correlated panel errors are not implemented).
    """

    signal_id: str
    errors: tuple[ComponentErrorModel, ...]
    protocol: SamplingProtocol = SamplingProtocol()
    assay_id: Optional[str] = None
    error_scale: str = "additive"
    is_synthetic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        errs = tuple(self.errors)
        object.__setattr__(self, "errors", errs)
        if not errs:
            raise InvalidProblemError("SignalModel: at least one observed component is required")
        comps = [e.component for e in errs]
        if len(set(comps)) != len(comps):
            raise InvalidProblemError("SignalModel: duplicate observed component")
        if self.error_scale != "additive":
            raise InvalidProblemError("SignalModel: only error_scale='additive' is implemented "
                                      "(multiplicative/log scale: planned)")
        if any(e.is_synthetic for e in errs) and not self.is_synthetic:
            raise InvalidProblemError("SignalModel: synthetic error components in a non-synthetic model")

    @property
    def components(self) -> tuple[ObservedComponent, ...]:
        """Observed components in signal order."""
        return tuple(e.component for e in self.errors)

    def variance_parts(self) -> list[dict[str, float]]:
        """Per component: variance contributed by sampling, lab repeatability and lab bias."""
        m, r = int(self.protocol.n_field_samples), int(self.protocol.n_lab_replicates)
        out = []
        for e in self.errors:
            v_samp = e.sampling_sd ** 2 / m
            if e.lab_error_semantics == "single_result_total":
                # reducible share unknown -> replicates assumed not to reduce it (conservative)
                v_rep = e.lab_repeatability_sd ** 2
            elif self.protocol.composite:
                v_rep = e.lab_repeatability_sd ** 2 / r
            else:
                v_rep = e.lab_repeatability_sd ** 2 / (m * r)
            out.append({"sampling": v_samp, "lab_repeatability": v_rep, "lab_bias": e.lab_bias_sd ** 2})
        return out

    def total_sd(self) -> np.ndarray:
        """SD of ``Z - theta - bias_mean`` per component ``[m]``."""
        return np.array([math.sqrt(sum(p.values())) for p in self.variance_parts()])

    def bias_means(self) -> np.ndarray:
        """Known systematic offsets ``[m]``."""
        return np.array([e.bias_mean for e in self.errors])

    def with_protocol(self, protocol: SamplingProtocol) -> "SignalModel":
        """Same errors under another sampling protocol."""
        return SignalModel(self.signal_id, self.errors, protocol, self.assay_id, self.error_scale,
                           self.is_synthetic, self.notes)

    def scaled(self, factor: float, signal_id: Optional[str] = None) -> "SignalModel":
        """All error SDs multiplied by ``factor`` (for noise-limit checks); offsets unchanged."""
        f = _nonneg("factor", factor)
        errs = tuple(ComponentErrorModel(e.component, e.sampling_sd * f, e.lab_repeatability_sd * f,
                                         e.lab_bias_sd * f, e.bias_mean, e.lab_error_semantics,
                                         e.provenance, e.is_synthetic, e.measurement_model_id) for e in self.errors)
        return SignalModel(signal_id or f"{self.signal_id}*{f:g}", errs, self.protocol, self.assay_id,
                           self.error_scale, self.is_synthetic, self.notes)

    def simulate(self, true_values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw signals ``[S, m]`` for true values ``[S, m]`` component by component.

        Sampling errors are drawn per field sample, repeatability errors per analysis and the lab
        bias once per assay event, then combined exactly as the protocol prescribes.  Only the
        passed ``rng`` is used.
        """
        t = np.asarray(true_values, dtype=float)
        if t.ndim != 2 or t.shape[1] != len(self.errors):
            raise ValueError("simulate: true_values must be [S, m]")
        S, mm = t.shape
        m, r = int(self.protocol.n_field_samples), int(self.protocol.n_lab_replicates)
        samp_sd = np.array([e.sampling_sd for e in self.errors])
        rep_sd = np.array([e.lab_repeatability_sd for e in self.errors])
        bias_sd = np.array([e.lab_bias_sd for e in self.errors])
        total_rep = np.array([e.lab_error_semantics == "single_result_total" for e in self.errors])
        # sampling error: one draw per field sample, averaged (composite or averaged results)
        samp_err = (rng.standard_normal((S, m, mm)) * samp_sd[None, None, :]).mean(axis=1)
        if self.protocol.composite:
            e_rep = rng.standard_normal((S, r, mm)) * rep_sd[None, None, :]      # r analyses of 1 composite
            rep_mean, rep_single = e_rep.mean(axis=1), e_rep[:, 0, :]
        else:
            e_rep = rng.standard_normal((S, m, r, mm)) * rep_sd[None, None, None, :]   # m*r analyses
            rep_mean, rep_single = e_rep.mean(axis=(1, 2)), e_rep[:, 0, 0, :]
        # a 'single_result_total' SD has no identified reducible share -> not reduced (conservative)
        rep_err = np.where(total_rep[None, :], rep_single, rep_mean)
        # lab bias: one draw per assay event and component, shared by all its samples and replicates
        bias = rng.standard_normal((S, mm)) * bias_sd[None, :]
        return t + self.bias_means()[None, :] + samp_err + rep_err + bias

    def _measurement_ids(self) -> tuple:
        """Per-component ``measurement_model_id`` (hashed only when one is set, so that existing
        fingerprints of signals without it are unchanged)."""
        ids = tuple(e.measurement_model_id for e in self.errors)
        return () if all(i is None for i in ids) else (("measurement_model_ids", ids),)

    def fingerprint(self) -> str:
        """Content hash (error values, semantics, protocol)."""
        return stable_hash("SignalModel/v1", self.signal_id, self.assay_id, self.error_scale,
                           tuple((e.component.ingredient_id, e.component.component, e.sampling_sd,
                                  e.lab_repeatability_sd, e.lab_bias_sd, e.bias_mean, e.lab_error_semantics)
                                 for e in self.errors),
                           (self.protocol.n_field_samples, self.protocol.composite,
                            self.protocol.n_lab_replicates), self.is_synthetic, *self._measurement_ids())

    def error_fingerprint(self) -> str:
        """Content hash of what the double-count guard depends on: observed components, error values
        and semantics, sampling protocol, error scale and (when set) measurement model ids (labels
        ``signal_id``/``assay_id`` excluded)."""
        return stable_hash("SignalModel.errors/v1", self.error_scale,
                           tuple((e.component.ingredient_id, e.component.component, e.sampling_sd,
                                  e.lab_repeatability_sd, e.lab_bias_sd, e.bias_mean, e.lab_error_semantics)
                                 for e in self.errors),
                           (self.protocol.n_field_samples, self.protocol.composite,
                            self.protocol.n_lab_replicates), *self._measurement_ids())

    def provenance_issues(self) -> list[str]:
        """All provenance issues of the error components."""
        out: list[str] = []
        for e in self.errors:
            out.extend(e.provenance_issues())
        return out


def _sv(v: Optional[SourcedValue], where: str) -> tuple[float, Optional[Provenance]]:
    if v is None:
        return 0.0, None
    if v.value is None:
        raise InvalidProblemError(f"{where}: value pending ({v.provenance.rationale}); an error model "
                                  "with pending values cannot be simulated (report ideal information only)")
    c = v.canonical()
    return float(c), v.provenance


def signal_model_from_assay(assay: AssaySpec, ingredient_id: str, *, lab_error_semantics: str,
                            systematic_bias_semantics: str, protocol: SamplingProtocol = SamplingProtocol(),
                            components: Optional[Sequence[str]] = None,
                            measurement_model_id: Optional[str] = None) -> SignalModel:
    """Build a :class:`SignalModel` from an :class:`~ration_reliability.datamodel.AssaySpec`.

    The meaning of the container fields must be declared, never guessed:

    * ``lab_error_semantics``: meaning of ``assay.lab_error_sd`` (see module doc);
    * ``systematic_bias_semantics``: ``"sd_of_random_lab_bias"`` (``assay.systematic_bias`` is the
      SD of an unknown lab bias) or ``"known_mean_offset"`` (it is a known offset).

    Missing error entries are treated as 0 **only** if the key is absent; a present entry with a
    pending value raises.  ``components`` defaults to ``assay.nutrient_ids`` (+ ``DM`` if
    ``assay.measures_dm``).  ``measurement_model_id`` (optional, R2) is copied to every component.
    """
    if lab_error_semantics not in LAB_ERROR_SEMANTICS:
        raise InvalidProblemError(f"lab_error_semantics must be one of {LAB_ERROR_SEMANTICS}")
    if systematic_bias_semantics not in ("sd_of_random_lab_bias", "known_mean_offset"):
        raise InvalidProblemError("systematic_bias_semantics must be 'sd_of_random_lab_bias' or "
                                  "'known_mean_offset'")
    if assay.applicable_ingredient_ids and ingredient_id not in assay.applicable_ingredient_ids:
        raise InvalidProblemError(f"assay {assay.assay_id} is not applicable to {ingredient_id}")
    comps = list(components) if components is not None else \
        list(assay.nutrient_ids) + ([DM_COMPONENT] if assay.measures_dm else [])
    if not comps:
        raise InvalidProblemError(f"assay {assay.assay_id}: no components")
    errs = []
    for c in comps:
        where = f"assay {assay.assay_id}/{ingredient_id}:{c}"
        s, ps = _sv(assay.sampling_error_sd.get(c), where + ".sampling_error_sd")
        lr, pl = _sv(assay.lab_error_sd.get(c), where + ".lab_error_sd")
        b, pb = _sv(assay.systematic_bias.get(c), where + ".systematic_bias")
        prov = {k: p for k, p in (("sampling_sd", ps), ("lab_repeatability_sd", pl)) if p is not None}
        if systematic_bias_semantics == "sd_of_random_lab_bias":
            bias_sd, bias_mean = b, 0.0
            if pb is not None:
                prov["lab_bias_sd"] = pb
        else:
            bias_sd, bias_mean = 0.0, b
            if pb is not None:
                prov["bias_mean"] = pb
        errs.append(ComponentErrorModel(ObservedComponent(ingredient_id, c), s, lr, bias_sd, bias_mean,
                                        lab_error_semantics, prov, bool(assay.is_synthetic), measurement_model_id))
    return SignalModel(f"{assay.assay_id}@{ingredient_id}", tuple(errs), protocol, assay.assay_id,
                       assay.error_scale, bool(assay.is_synthetic), assay.notes)


# ---------------------------------------------------------------------------------------------
# double counting of batch variation and measurement error (contract 7.3, T7.5, F06, F07)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DoubleCountReport:
    """Result of :func:`double_count_guard`.

    ``status``: ``ok`` / ``violation`` / ``unidentified``.  ``violation`` blocks the sample-
    information computation; ``unidentified`` allows it only as a labelled scenario-range member.

    Binding (red-team fix F06): a report produced by :func:`double_count_guard` records the error
    fingerprint (:meth:`SignalModel.error_fingerprint`) of the signal it was computed for and the
    declared inputs (prior-variance basis and
    de-convolution flags of the signal's components).  :func:`~.value.compute_information_value`
    re-runs the guard from these inputs on the structure's own signal and rejects a report that is
    unbound, belongs to another signal or does not reproduce.

    Object mode (review R2): ``prior_fingerprint`` / ``truth_fingerprint`` bind the report to the
    prior states and the truth record it was computed with; ``basis_sources`` says per component
    whether the basis came from object metadata or from an unbound caller declaration; ``checks``
    holds the prior/truth binding and the variance-decomposition records.
    """

    status: str
    messages: tuple[str, ...]
    per_component: Mapping[str, str]
    signal_fingerprint: Optional[str] = None
    declared_basis: Optional[tuple[tuple[str, str, Optional[str]], ...]] = None
    declared_deconvolved: tuple[tuple[str, str, bool], ...] = ()
    prior_fingerprint: Optional[str] = None
    truth_fingerprint: Optional[str] = None
    truth_origin: Optional[str] = None
    truth_linked: Optional[bool] = None
    basis_sources: tuple[tuple[str, str], ...] = ()
    checks: Mapping[str, Any] = field(default_factory=dict)
    #: FIX_A: the truth link was verified against an object (only then may the truth be a basis source)
    truth_verified: Optional[bool] = None

    @property
    def ok(self) -> bool:
        """True if no violation and no unidentified component."""
        return self.status == "ok"

    @property
    def is_bound(self) -> bool:
        """True if the report carries the signal fingerprint and its declared inputs."""
        return self.signal_fingerprint is not None and self.declared_basis is not None

    @property
    def object_mode(self) -> bool:
        """True if the report was computed against a prior object (R2 object mode)."""
        return self.prior_fingerprint is not None

    @property
    def object_bound(self) -> bool:
        """True if every signal component's basis came from object metadata (prior or linked truth)."""
        return bool(self.basis_sources) and all(s.startswith("object_metadata") for _, s in self.basis_sources)

    def declared_inputs(self) -> tuple[dict[ObservedComponent, str], dict[ObservedComponent, bool]]:
        """``(prior_variance_basis, prior_deconvolved)`` as declared when the report was produced."""
        if self.declared_basis is None:
            raise InvalidProblemError("DoubleCountReport: not bound to a signal (no declared inputs)")
        basis = {ObservedComponent(i, c): b for i, c, b in self.declared_basis if b is not None}
        dec = {ObservedComponent(i, c): bool(v) for i, c, v in self.declared_deconvolved}
        return basis, dec

    def same_verdict(self, other: "DoubleCountReport") -> bool:
        """Status and per-component verdicts identical."""
        return self.status == other.status and dict(self.per_component) == dict(other.per_component)


_RANK = {"ok": 0, "unidentified": 1, "violation": 2}


def _basis_rule(e: ComponentErrorModel, basis: Optional[str], deconvolved: bool) -> tuple[str, list[str]]:
    """The double-counting rules of one component for one (basis, de-convolved) pair (unchanged F06 rules)."""
    c = e.component
    lab = e.lab_repeatability_sd > 0 or e.lab_bias_sd > 0
    samp = e.sampling_sd > 0
    msgs: list[str] = []
    st = "ok"
    if basis is None:
        st = "violation"
        msgs.append(f"{c.label()}: prior variance basis not declared (one of {PRIOR_VARIANCE_BASES})")
    elif basis not in PRIOR_VARIANCE_BASES:
        st = "violation"
        msgs.append(f"{c.label()}: unknown prior variance basis {basis!r}")
    elif basis == "observed_incl_sampling_and_lab" and not deconvolved and (samp or lab):
        st = "violation"
        msgs.append(f"{c.label()}: prior SD already contains sampling and lab error; adding them to the "
                    "signal double counts. De-convolve the prior (deconvolve_true_sd) or declare the prior "
                    "basis with a source.")
    elif basis == "observed_incl_lab_only" and not deconvolved and lab:
        st = "violation"
        msgs.append(f"{c.label()}: prior SD already contains laboratory error; adding lab error to the signal "
                    "double counts.")
    elif basis == "unidentified":
        st = "unidentified"
        msgs.append(f"{c.label()}: prior variance decomposition unidentified; report the bounding treatments "
                    "(all true variation vs. de-convolved) as a scenario range, not as one EVSI.")
    if e.lab_error_semantics == "single_result_total" and e.lab_bias_sd > 0:
        st = "violation"
        msgs.append(f"{c.label()}: lab_error_semantics='single_result_total' already includes the between-lab "
                    "part; a separate lab_bias_sd double counts it.")
    return st, msgs


def _decomposition_check(e: ComponentErrorModel, obj: Any, basis: Optional[str], prior: PriorStates
                         ) -> dict[str, Any]:
    """``prior variance - single-result error variance`` for an observed / unidentified basis (R2 item 3).

    The historical single result that produced an observed prior SD contains one field sample and one
    analysis (sampling + repeatability + lab bias; lab parts only for ``observed_incl_lab_only``).  The
    difference is recorded as is.  It is *enforced* (negative -> violation) only when the signal's
    ``measurement_model_id`` equals the prior's (same measurement process); otherwise a negative
    difference is a recorded warning, because another laboratory may simply be noisier.
    """
    c = e.component
    out: dict[str, Any] = {"component": c.label(), "basis": basis, "checked": False}
    if basis not in ("observed_incl_sampling_and_lab", "observed_incl_lab_only", "unidentified"):
        out["reason"] = "basis is not observed/unidentified: no decomposition implied"
        return out
    pv = prior.component_variance(c)
    if pv is None:
        out["reason"] = "prior values missing"
        return out
    lab_var = e.lab_repeatability_sd ** 2 + e.lab_bias_sd ** 2
    err = lab_var if basis == "observed_incl_lab_only" else e.sampling_sd ** 2 + lab_var
    implied = pv - err                       # raw difference, never clipped
    tol = 1e-12 * max(pv, err, 1e-300)
    status = "consistent" if implied > tol else ("degenerate_no_true_variation" if implied >= -tol
                                                  else "inconsistent_negative")
    prior_mm = None if obj is None else obj.measurement_model_id
    sig_mm = e.measurement_model_id
    same = bool(prior_mm) and bool(sig_mm) and prior_mm == sig_mm
    out.update({"checked": True, "prior_variance": pv, "single_result_error_variance": err,
                "implied_true_variance": implied, "status": status, "tolerance": tol, "enforced": same,
                "prior_measurement_model_id": prior_mm, "signal_measurement_model_id": sig_mm,
                "deconvolved_member": ("exists" if status != "inconsistent_negative"
                                       else "rejected_inconsistent_negative"),
                "note": ("same measurement process: a negative difference rejects the decomposition" if same else
                         "measurement processes not linked (ids differ or are missing): a negative difference rejects "
                         "only the de-convolved member of the scenario range (recorded)")})
    if prior_mm and err > 0 and not sig_mm:
        # FIX_A: without a declared process the check below could never be enforced (the review's
        # "missing / renamed id" bypass); the prior names the process its SD contains
        out["violation"] = (f"{c.label()}: the prior basis {basis!r} contains measurement process {prior_mm!r}; the "
                            "signal must declare measurement_model_id so that the variance decomposition can be "
                            "checked against it (an undeclared process would silently skip the check)")
    if status == "inconsistent_negative":
        msg = (f"{c.label()}: prior variance {pv:.6g} minus the single-result error variance {err:.6g} of the "
               f"signal is {implied:.6g} < 0: inconsistent variance decomposition")
        if same:
            out["violation"] = (msg + f" for the same measurement process {sig_mm!r}; rejected (not clipped to 0)")
        elif "violation" not in out:
            out["warning"] = (msg + " if the prior contained this error; the measurement processes are not linked "
                                    f"(prior {prior_mm!r}, signal {sig_mm!r}), so the computation is not refused, but the "
                                    "de-convolved bounding member does not exist (rejected, recorded): the computed "
                                    "value is only the 'all observed variation is true variation' member of the "
                                    "scenario range")
    return out


def _true_state_variance_check(c: ObservedComponent, obj: Any, prior: PriorStates) -> dict[str, Any]:
    """Record and state-variance part of :func:`_true_state_check` (also used for perfect information)."""
    out: dict[str, Any] = {"component": c.label(), "checked": False}
    if obj is None or obj.variance_basis != "true_batch_state" or not obj.deconvolved:
        out.update({"status": "not_a_deconvolved_true_state", "identifies": True})
        return out
    rec = obj.decomposition
    if rec is None:
        if obj.is_synthetic:
            out.update({"status": "record_missing_synthetic", "identifies": False,
                        "warning": f"{c.label()}: de-convolved true_batch_state ({obj.decomposition_id!r}) without a "
                                   "numeric VarianceDecomposition record: the de-convolution cannot be checked against "
                                   "the states; computed as a labelled scenario, not identified"})
        else:
            out.update({"status": "record_missing", "identifies": False,
                        "violation": f"{c.label()}: a non-synthetic de-convolved true_batch_state "
                                     f"({obj.decomposition_id!r}) must carry its numeric VarianceDecomposition record "
                                     "(observed SD, removed sampling/lab SD, true SD); strings are not a de-convolution"})
        return out
    pv = prior.component_variance(c)
    tsd = rec.true_sd
    out.update({"checked": True, "decomposition": rec.to_dict(), "prior_variance": pv})
    if pv is None or tsd is None:
        out.update({"status": "rejected", "identifies": False,
                    "violation": f"{c.label()}: prior values missing or decomposition without a true SD"})
        return out
    tv = float(tsd) ** 2
    if prior.stream == DISCRETE_PRIOR_STREAM:
        tol, kind = 1e-9 * max(pv, tv) + 1e-300, "exact_discrete_prior_relative_1e-9"
    else:
        S = max(int(prior.n_states), 2)
        tol = TRUE_STATE_SD_MC_Z * math.sqrt(2.0 / (S - 1)) * tv + 1e-300
        kind = f"monte_carlo_{TRUE_STATE_SD_MC_Z:g}_se_of_sample_variance_normal_approx_S={S}"
    out.update({"recorded_true_variance": tv, "difference": pv - tv, "tolerance": tol, "tolerance_kind": kind})
    if abs(pv - tv) > tol:
        out.update({"status": "rejected", "identifies": False,
                    "violation": f"{c.label()}: the prior states have variance {pv:.6g} (SD {math.sqrt(pv):.6g}) but "
                                 f"the attached decomposition records a de-convolved true variance {tv:.6g} (SD "
                                 f"{tsd:.6g}); |difference| {abs(pv - tv):.3g} > tolerance {tol:.3g}: the states were "
                                 "not generated from the recorded true-state SD (a label is not a de-convolution) -- "
                                 "rejected"})
        return out
    out.update({"status": "verified", "identifies": True})
    return out


def _true_state_check(e: ComponentErrorModel, obj: Any, prior: PriorStates) -> dict[str, Any]:
    """Numeric check of a de-convolved ``true_batch_state`` component (FIX_A; review finding "the
    de-convolution is only a few strings").

    * the component must carry a :class:`~.prior.VarianceDecomposition` record (non-synthetic:
      violation without it; synthetic: recorded, the value cannot be identified);
    * the recorded true variance must equal the variance of the prior states (discrete prior: relative
      1e-9; drawn states: :data:`TRUE_STATE_SD_MC_Z` normal-approximation standard errors of a sample
      variance, ``sqrt(2/(S-1))`` relative);
    * the signal must name its measurement process when the metadata are non-synthetic; for the *same*
      process, the signal's single-result error (``sampling_sd``, ``sqrt(lab_repeatability^2 +
      lab_bias^2)``) must equal the error the decomposition removed.
    """
    c = e.component
    out = _true_state_variance_check(c, obj, prior)
    if out["status"] in ("not_a_deconvolved_true_state", "record_missing", "record_missing_synthetic"):
        return out
    rec = obj.decomposition
    issues: list[str] = [out.pop("violation")] if "violation" in out else []
    rec_mm, sig_mm = obj.measurement_model_id, e.measurement_model_id
    out.update({"removed_measurement_model_id": rec_mm, "signal_measurement_model_id": sig_mm})
    if not sig_mm:
        if not (obj.is_synthetic and e.is_synthetic):
            issues.append(f"{c.label()}: the prior was de-convolved by removing measurement process {rec_mm!r}; a "
                          "non-synthetic signal must declare its measurement_model_id so that the removed and the added "
                          "errors can be compared")
        out["measurement_link"] = "signal_measurement_model_not_declared"
    elif sig_mm == rec_mm:
        lab_single = math.sqrt(e.lab_repeatability_sd ** 2 + e.lab_bias_sd ** 2)
        same_s = math.isclose(e.sampling_sd, rec.sampling_sd, rel_tol=1e-9, abs_tol=1e-15)
        same_l = math.isclose(lab_single, rec.lab_sd, rel_tol=1e-9, abs_tol=1e-15)
        out["measurement_link"] = "same_process"
        if not (same_s and same_l):
            issues.append(f"{c.label()}: same measurement process {sig_mm!r}, but the signal's single-result error "
                          f"(sampling {e.sampling_sd:.6g}, lab {lab_single:.6g}) differs from the error the "
                          f"de-convolution removed (sampling {rec.sampling_sd:.6g}, lab {rec.lab_sd:.6g})")
    else:
        out["measurement_link"] = "different_process"
    if issues:
        out.update({"status": "rejected", "identifies": False, "violation": "; ".join(issues)})
    else:
        out.update({"status": "verified", "identifies": True})
    return out


def double_count_guard(signal: SignalModel, prior_variance_basis: Optional[Mapping[ObservedComponent, str]] = None,
                       prior_deconvolved: Optional[Mapping[ObservedComponent, bool]] = None, *,
                       prior: Optional[PriorStates] = None, truth: Any = None,
                       error_locators: Optional[Mapping[str, Mapping[str, str]]] = None) -> DoubleCountReport:
    """Check that batch variation and measurement error are not counted twice.

    Parameters
    ----------
    signal : the assay signal model.
    prior_variance_basis : for every observed component, what the prior SD used to generate the
        states represents (one of :data:`PRIOR_VARIANCE_BASES`).  A missing entry is an error
        (declaration mode); in object mode it may be omitted for components with object metadata.
    prior_deconvolved : True for components whose prior SD was already reduced to the true-state SD
        (e.g. with :func:`deconvolve_true_sd`) before drawing the states.
    prior : (object mode, R2) the :class:`~.prior.PriorStates`; its metadata are authoritative.
    truth : (object mode) truth record -- ``StateMetadata``, ``PriorStates`` or ``FactoryModel``;
        ``None`` = the generator of the prior states (see :func:`~.prior.resolve_truth_link`).
    error_locators : (object mode, FIX_A) ``param_id -> row`` of the error-parameter locator table used
        to trace ``sourced`` error values; ``None`` = :func:`default_error_locators`.

    Rules
    -----
    * ``true_batch_state``: OK.
    * ``observed_incl_sampling_and_lab`` (not de-convolved) with any sampling or lab error in the
      signal -> violation.
    * ``observed_incl_lab_only`` (not de-convolved) with lab repeatability or lab bias in the signal
      -> violation.
    * ``unidentified`` -> ``unidentified`` (report the bounding treatments as a scenario range; do
      not present a single EVSI).
    * ``lab_error_semantics='single_result_total'`` together with ``lab_bias_sd > 0`` -> violation
      (the single-result total already contains the between-lab part).

    Object mode adds: declaration vs object conflict -> :class:`~.prior.MetadataConflictError`;
    prior vs linked-truth disagreement -> violation; an unlinked or unverified truth record can only
    make the verdict stricter; ``not_applicable`` (point value in the prior) -> ok (nothing to double
    count); an inconsistent variance decomposition for the same measurement process -> violation;
    FIX_A: a de-convolved true state that does not match its numeric record, a missing signal
    measurement process where the prior names one, and sourced error values that do not trace to the
    locator table -> violation.
    """
    if prior is None and truth is None:
        return _declaration_guard(signal, prior_variance_basis or {}, prior_deconvolved)
    if prior is None:
        raise InvalidProblemError("double_count_guard: a truth record needs the prior states (prior=...)")
    return _object_guard(signal, prior_variance_basis, prior_deconvolved, prior, truth, error_locators)


def _declaration_guard(signal: SignalModel, prior_variance_basis: Mapping[ObservedComponent, str],
                       prior_deconvolved: Optional[Mapping[ObservedComponent, bool]]) -> DoubleCountReport:
    dec = dict(prior_deconvolved or {})
    msgs: list[str] = []
    per: dict[str, str] = {}
    worst = "ok"
    for e in signal.errors:
        c = e.component
        st, m = _basis_rule(e, prior_variance_basis.get(c), dec.get(c, False))
        msgs.extend(m)
        per[c.label()] = st
        if _RANK[st] > _RANK[worst]:
            worst = st
    comps = signal.components
    declared = tuple((c.ingredient_id, c.component, prior_variance_basis.get(c)) for c in comps)
    declared_dec = tuple((c.ingredient_id, c.component, bool(dec[c])) for c in comps if c in dec)
    return DoubleCountReport(worst, tuple(msgs), per, signal.error_fingerprint(), declared, declared_dec)


def _trace_block(signal: SignalModel, error_locators: Optional[Mapping[str, Mapping[str, str]]]) -> dict[str, Any]:
    """Forced trace of ``sourced`` error values (FIX_A; review finding: ``trace_error_provenance`` was never
    called).  Returns ``status`` (``no_sourced_error_values`` / ``traced`` / ``issues`` /
    ``locator_table_unavailable``), ``rows`` and per-component ``issues``."""
    sourced = [(e, f) for e in signal.errors for f in ("sampling_sd", "lab_repeatability_sd", "lab_bias_sd", "bias_mean")
               if getattr(e, f) != 0.0 and e.provenance.get(f) is not None
               and e.provenance[f].status is ValueStatus.SOURCED]
    scen = sorted({e.component.label() for e in signal.errors
                   for f in ("sampling_sd", "lab_repeatability_sd", "lab_bias_sd", "bias_mean")
                   if getattr(e, f) != 0.0 and e.provenance.get(f) is not None
                   and e.provenance[f].status is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION})
    out: dict[str, Any] = {"status": "no_sourced_error_values", "rows": [], "issues": {},
                           "scenario_assumption_components": scen, "locator_table": None}
    if not sourced:
        return out
    if error_locators is None:
        locs, where = default_error_locators(), _default_locator_path()
        where = None if locs is None else {"path": "sources/error_source_locators.csv", "sha256": _file_sha(where)}
    else:
        locs, where = error_locators, {"path": "caller_supplied", "sha256": None}
    out["locator_table"] = where
    if locs is None:
        out["status"] = "locator_table_unavailable"
        return out
    rows = trace_error_provenance(signal, locs)
    issues: dict[str, list[str]] = {}
    for r in rows:
        for i in r["issues"]:
            issues.setdefault(r["component"], []).append(f"{r['component']}.{r['field']}: {i}")
    for e in signal.errors:
        pids = {r["param_id"] for r in rows if r["component"] == e.component.label() and r.get("param_id")
                and r.get("provenance_status") == ValueStatus.SOURCED.value}
        if pids and not e.is_synthetic and e.measurement_model_id is not None and e.measurement_model_id not in pids:
            issues.setdefault(e.component.label(), []).append(
                f"{e.component.label()}: measurement_model_id {e.measurement_model_id!r} is not the locator param_id of "
                f"its sourced error values {sorted(pids)}; an arbitrary string cannot link or unlink measurement "
                "processes")
    out.update({"status": "issues" if issues else "traced", "rows": rows, "issues": issues})
    return out


def _object_guard(signal: SignalModel, prior_variance_basis: Optional[Mapping[Any, str]],
                  prior_deconvolved: Optional[Mapping[Any, bool]], prior: PriorStates, truth: Any,
                  error_locators: Optional[Mapping[str, Mapping[str, str]]] = None) -> DoubleCountReport:
    if not isinstance(prior, PriorStates):
        raise InvalidProblemError("double_count_guard: prior must be PriorStates")
    decl = {_comp(k): v for k, v in (prior_variance_basis or {}).items()}
    ddec = {_comp(k): bool(v) for k, v in (prior_deconvolved or {}).items()}
    link = resolve_truth_link(truth, prior)
    truth_md, truth_origin, linked, verified = link.metadata, link.origin, link.linked, link.verified
    prior_md = prior.metadata
    prior_ver, prior_ver_issues = (None, []) if prior_md is None else verify_state_metadata(prior_md)
    if prior_ver == "factory_claim_contradicted":
        raise MetadataConflictError("the prior's metadata claim factory origin but contradict the registered factory "
                                    "model: " + "; ".join(prior_ver_issues))
    trace = _trace_block(signal, error_locators)
    msgs: list[str] = []
    per: dict[str, str] = {}
    sources: list[tuple[str, str]] = []
    decomp: dict[str, Any] = {}
    true_state: dict[str, Any] = {}
    warns: list[str] = []
    worst = "ok"
    for e in signal.errors:
        c = e.component
        lab = c.label()
        pm = None if prior_md is None else prior_md.get(c)
        tm = None if truth_md is None else truth_md.get(c)
        st = "ok"
        if pm is not None and tm is not None and linked and pm.binding_key() != tm.binding_key():
            st = "violation"
            msgs.append(f"{lab}: prior and truth metadata disagree (prior {pm.to_dict()} vs truth {tm.to_dict()}); the "
                        "states must be generated by the model whose metadata they carry")
        # FIX_A: the truth end is a basis source only when its link is verified against an object
        obj, end = (pm, "prior") if pm is not None else ((tm, "truth") if (tm is not None and verified) else (None, None))
        d_b, d_d = decl.get(c), ddec.get(c)
        basis_from_unlinked_truth = False
        if obj is not None:
            conflicts = []
            if d_b is not None and d_b != obj.variance_basis:
                conflicts.append(f"declared prior variance basis {d_b!r} vs object {obj.variance_basis!r}")
            if d_d is not None and d_d != obj.deconvolved:
                conflicts.append(f"declared prior_deconvolved={d_d} vs object de-convolved={obj.deconvolved} "
                                 f"(variance_basis {obj.variance_basis!r}, decomposition_id {obj.decomposition_id!r})")
            if conflicts:
                raise MetadataConflictError(
                    f"{lab}: an external declaration conflicts with the object metadata of the {end} "
                    f"({'; '.join(conflicts)}); declarations cannot override object metadata (double counting guard "
                    "of batch variation and measurement error, review R2)")
            basis, dflag, src = obj.variance_basis, obj.deconvolved, f"object_metadata:{end}"
        elif d_b is None and tm is not None:        # unlinked / unverified truth record used as the (unbound) declaration
            basis = tm.variance_basis if tm.is_stochastic else NOT_APPLICABLE_BASIS
            dflag = tm.deconvolved if d_d is None else bool(d_d)
            src = "caller_declaration_unbound"
            basis_from_unlinked_truth = True
        else:
            basis, dflag, src = d_b, bool(d_d) if d_d is not None else False, "caller_declaration_unbound"
        if basis == NOT_APPLICABLE_BASIS:
            note = (obj.note if obj is not None else "") or "point/missing cell"
            rst, rmsgs = ("ok", [f"{lab}: the prior carries no variation for this component ({note}); nothing to "
                                 "double count"])
            if e.lab_error_semantics == "single_result_total" and e.lab_bias_sd > 0:
                rst, rmsgs = _basis_rule(e, "true_batch_state", False)
        else:
            rst, rmsgs = _basis_rule(e, basis, dflag)
        if src == "caller_declaration_unbound" and d_d is True and basis != "true_batch_state":
            # D-327 rule (FIX5), same as information_states.assay_double_count_report: a bare de-convolution
            # declaration is not evidence that the states were generated from a de-convolved SD
            rst = "violation"
            rmsgs.append(f"{lab}: prior_deconvolved is declared but no object metadata (decomposition_id / "
                         "decomposition_source / measurement_model_id on the prior or a linked truth model) records the "
                         "de-convolution; a bare declaration is not evidence (D-327)")
        if src == "caller_declaration_unbound" and rst != "violation":
            rmsgs.append(f"{lab}: basis {basis!r} is an unbound caller declaration (no object metadata on the prior or "
                         "a verified truth model): not an identification")
        msgs.extend(rmsgs)
        if _RANK[rst] > _RANK[st]:
            st = rst
        if tm is not None and not verified and not basis_from_unlinked_truth:   # unverified truth: may only restrict
            tst, tmsgs = _basis_rule(e, tm.variance_basis if tm.is_stochastic else "true_batch_state", tm.deconvolved)
            if _RANK[tst] > _RANK["ok"]:
                msgs.extend(f"truth record (unverified, {truth_origin}): {m}" if linked else
                            f"truth record (unlinked, {truth_origin}): {m}" for m in tmsgs)
            if _RANK[tst] > _RANK[st]:
                st = tst
        chk = _decomposition_check(e, obj, basis, prior)
        decomp[lab] = chk
        if "violation" in chk:
            st = "violation"
            msgs.append(chk["violation"])
        elif "warning" in chk:
            warns.append(chk["warning"])
        tchk = _true_state_check(e, obj, prior)
        true_state[lab] = tchk
        if "violation" in tchk:
            st = "violation"
            msgs.append(tchk["violation"])
        elif "warning" in tchk:
            warns.append(tchk["warning"])
        for i in trace["issues"].get(lab, ()):
            st = "violation"
            msgs.append(f"error provenance does not trace: {i}")
        per[lab] = st
        sources.append((lab, src))
        if _RANK[st] > _RANK[worst]:
            worst = st
    if trace["status"] == "locator_table_unavailable":
        warns.append("sourced error values could not be traced: no locator table (sources/error_source_locators.csv) is "
                     "available; the error model is treated as unsourced")
    comps = signal.components
    declared = tuple((c.ingredient_id, c.component, decl.get(c)) for c in comps)
    declared_dec = tuple((c.ingredient_id, c.component, bool(ddec[c])) for c in comps if c in ddec)
    checks = {"prior_metadata": None if prior_md is None else
              {"origin": prior_md.origin, "model_id": prior_md.model_id, "model_fingerprint": prior_md.model_fingerprint,
               "metadata_fingerprint": prior_md.metadata_fingerprint, "state_metadata_fingerprint": prior_md.fingerprint(),
               "derived_from_model_fingerprint": prior_md.derived_from_model_fingerprint,
               "is_diagnostic": prior_md.is_diagnostic, "is_synthetic": prior_md.is_synthetic,
               "verification": prior_ver, "verification_issues": list(prior_ver_issues)},
              "prior_generator_fingerprint": prior.generator_fingerprint,
              "truth": {"origin": truth_origin, "linked": linked, "verified": verified,
                        "verification": link.verification, "reason": link.reason,
                        "state_metadata_fingerprint": None if truth_md is None else truth_md.fingerprint(),
                        "model_fingerprint": None if truth_md is None else truth_md.model_fingerprint},
              "components": {c.label(): {"basis_source": src,
                                         "prior_metadata": None if (prior_md is None or prior_md.get(c) is None)
                                         else prior_md.get(c).to_dict(),
                                         "truth_metadata": None if (truth_md is None or truth_md.get(c) is None)
                                         else truth_md.get(c).to_dict()}
                             for c, (_, src) in zip(signal.components, sources)},
              "decomposition": decomp, "true_state_checks": true_state,
              "error_provenance": {k: v for k, v in trace.items()}, "warnings": warns}
    return DoubleCountReport(worst, tuple(msgs), per, signal.error_fingerprint(), declared, declared_dec,
                             prior.fingerprint, None if truth_md is None else truth_md.fingerprint(), truth_origin,
                             linked, tuple(sources), checks, verified)


def deconvolve_true_sd(observed_sd: float, sampling_sd: float = 0.0, lab_sd: float = 0.0,
                       n_field_samples: int = 1) -> float:
    """True batch-to-batch SD from the SD of historical single results.

    ``sqrt(observed^2 - sampling^2 / n_field_samples - lab^2)``, where ``observed_sd`` is the SD of
    results that each came from ``n_field_samples`` composited field samples analysed once, and
    ``lab_sd`` is the total SD of one analysis.  Raises
    :class:`~.prior.InconsistentDecompositionError` (a ``ValueError``) if the error variance exceeds
    the observed variance (the decomposition is then inconsistent and must be reported as
    unidentified, never clipped to 0 silently).  :func:`~.prior.decompose_observed_variance` returns
    the full record instead of raising.
    """
    for n, v in (("observed_sd", observed_sd), ("sampling_sd", sampling_sd), ("lab_sd", lab_sd)):
        _nonneg(n, v)
    if int(n_field_samples) < 1:
        raise ValueError("n_field_samples must be >= 1")
    return decompose_observed_variance(observed_sd, sampling_sd=sampling_sd, lab_sd=lab_sd,
                                       n_field_samples=n_field_samples).require_consistent()


# ---------------------------------------------------------------------------------------------
# provenance of error values -> sources/error_source_locators.csv (review R2 item 6)
# ---------------------------------------------------------------------------------------------

_LOCATOR_PREFIXES = ("sources/error_source_locators.csv#", "error_source_locators.csv#", "param_id=")


def load_error_locators(path: Union[str, Path]) -> dict[str, dict[str, str]]:
    """``param_id -> row`` of ``sources/error_source_locators.csv`` (strings as written; no values)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        pid = (r.get("param_id") or "").strip()
        if not pid:
            raise InvalidProblemError(f"{path}: row without param_id")
        if pid in out:
            raise InvalidProblemError(f"{path}: duplicate param_id {pid!r}")
        out[pid] = dict(r)
    return out


_LOCATOR_CACHE: dict[tuple, dict[str, dict[str, str]]] = {}


def _default_locator_path() -> Path:
    """``<project root>/sources/error_source_locators.csv`` (source checkout layout ``src/ration_reliability``)."""
    return Path(__file__).resolve().parents[3] / "sources" / "error_source_locators.csv"


def _file_sha(path: Optional[Path]) -> Optional[str]:
    if path is None or not Path(path).is_file():
        return None
    from ..hashing import file_sha256

    return file_sha256(path)


def default_error_locators() -> Optional[dict[str, dict[str, str]]]:
    """The project's error-parameter locator table, or ``None`` if it is not present (e.g. an installed
    package without the ``sources/`` tree).  Cached per file size and modification time (FIX_A)."""
    p = _default_locator_path()
    if not p.is_file():
        return None
    stt = p.stat()
    key = (str(p), stt.st_size, stt.st_mtime_ns)
    if key not in _LOCATOR_CACHE:
        _LOCATOR_CACHE.clear()
        _LOCATOR_CACHE[key] = load_error_locators(p)
    return _LOCATOR_CACHE[key]


def _param_id(locator: Optional[str]) -> Optional[str]:
    s = str(locator or "").strip()
    for p in _LOCATOR_PREFIXES:
        if s.startswith(p):
            return s[len(p):].strip()
    return s or None


def trace_error_provenance(signal: SignalModel, locators: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    """Trace every non-zero error value of ``signal`` to ``sources/error_source_locators.csv``.

    A ``sourced`` provenance must carry the parameter id in ``locator`` (``<param_id>`` or
    ``sources/error_source_locators.csv#<param_id>``) and the registry id in ``source_id``.  Checks:
    the row exists; ``source_id`` is one of its ``registry_source_id``; its ``error_components`` are
    admissible for the error field (:data:`ERROR_FIELD_COMPONENTS`; anything with ``batch_true`` is
    refused: true batch variation used as measurement error double counts); the nutrient matches the
    observed component; the scale is additive (a CV needs a declared conversion); the value and
    verification are not ``blocked``.  Synthetic and scenario provenance is reported as not traced.
    Returns one row per error value with ``status`` (``traced`` / ``issues`` / ``synthetic_not_traced`` /
    ``scenario_assumption_not_traced``) and ``issues``.
    """
    out: list[dict[str, Any]] = []
    for e in signal.errors:
        for fname in ("sampling_sd", "lab_repeatability_sd", "lab_bias_sd", "bias_mean"):
            if getattr(e, fname) == 0.0:
                continue
            p = e.provenance.get(fname)
            row: dict[str, Any] = {"component": e.component.label(), "field": fname,
                                   "lab_error_semantics": e.lab_error_semantics,
                                   "measurement_model_id": e.measurement_model_id,
                                   "provenance_status": None if p is None else getattr(p.status, "value", p.status),
                                   "source_id": None if p is None else p.source_id,
                                   "locator": None if p is None else p.locator, "param_id": None, "issues": []}
            if p is None:
                row["issues"].append("non-zero value without provenance")
                row["status"] = "issues"
                out.append(row)
                continue
            if p.status is ValueStatus.SYNTHETIC_TEST_ONLY:
                row["status"] = "synthetic_not_traced"
                out.append(row)
                continue
            if p.status is not ValueStatus.SOURCED:
                row["status"] = "scenario_assumption_not_traced"
                out.append(row)
                continue
            pid = _param_id(p.locator)
            row["param_id"] = pid
            loc = locators.get(pid) if pid else None
            if loc is None:
                row["issues"].append(f"locator {p.locator!r} is not a param_id of sources/error_source_locators.csv")
            else:
                regs = [s.strip() for s in str(loc.get("registry_source_id", "")).split("|") if s.strip()]
                if p.source_id not in regs:
                    row["issues"].append(f"source_id {p.source_id!r} is not the registry source of {pid} ({regs})")
                comps = str(loc.get("error_components", "")).strip()
                if "batch_true" in comps.split("|"):
                    row["issues"].append(f"{pid} has error components {comps!r}: it contains true batch variation and "
                                         "cannot be a measurement error of the signal (double counting)")
                elif comps not in ERROR_FIELD_COMPONENTS[fname][e.lab_error_semantics]:
                    row["issues"].append(f"{pid} error components {comps!r} do not match field {fname} "
                                         f"(lab_error_semantics {e.lab_error_semantics!r}; admissible "
                                         f"{ERROR_FIELD_COMPONENTS[fname][e.lab_error_semantics]})")
                nut = str(loc.get("nutrient", "")).strip()
                if nut.lower() != e.component.component.lower():
                    row["issues"].append(f"{pid} is for {nut!r}, the signal observes {e.component.component!r}")
                if str(loc.get("scale", "")).strip() != "additive":
                    row["issues"].append(f"{pid} scale {loc.get('scale')!r}: only additive errors are implemented "
                                         "(a CV needs a declared conversion)")
                for k in ("value_status", "verification"):
                    if str(loc.get(k, "")).strip() == "blocked":
                        row["issues"].append(f"{pid} {k} is 'blocked'")
                row.update({"registry_source_id": loc.get("registry_source_id"), "error_components": comps,
                            "error_layer_class": loc.get("error_layer_class"), "doi": loc.get("doi"),
                            "table_or_locator": loc.get("table_or_locator"), "evidence_form": loc.get("evidence_form"),
                            "session_cache_sha256": loc.get("session_cache_sha256"),
                            "persistent_locator": loc.get("persistent_locator"),
                            "verification": loc.get("verification")})
            row["status"] = "issues" if row["issues"] else "traced"
            out.append(row)
    return out


def components_of(models: Iterable[SignalModel]) -> tuple[ObservedComponent, ...]:
    """Union of observed components (in order of first appearance)."""
    seen: list[ObservedComponent] = []
    for m in models:
        for c in m.components:
            if c not in seen:
                seen.append(c)
    return tuple(seen)
