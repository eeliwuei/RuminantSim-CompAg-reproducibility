"""Information states H0 / H1 (``configs/uncertainty.yaml`` ``e_information_state_history``).

H0 ``H0_table_only``
    The decision maker knows only P0: prior ``(mu_P0, s * sigma_P0)``, ``d_hat = mu_P0(DM)``.
H1 two-layer world (same outer world for H0 and H1, common random numbers)
    (1) farm mean ``mu_farm ~ F(mu_P0, sigma_between)`` from the named stream ``farm_mean``;
    (2) current batch ``theta | mu_farm ~ F(mu_farm, sigma_within)`` from the purpose stream
        (``validation`` / ``test`` / ...), sub-key = farm index;
    ``sigma_within = r * s * sigma_P0`` and ``sigma_between^2 = (s sigma_P0)^2 - sigma_within^2``
    (``0 < r < 1``), so that the marginal of ``theta`` keeps mean ``mu_P0`` and SD ``s sigma_P0``
    (law of total variance; each layer is moment matched with a family of
    :mod:`.distributions`).
    ``H1_perfect`` knows ``mu_farm``; ``H1_m`` knows ``mu_hat = mu_farm + e``,
    ``e ~ N(0, sigma_within^2 / m_hist)`` (stream ``farm_history``), prior SD
    ``sigma_within * sqrt(1 + 1 / m_hist)``.  ``m_hist`` is ``pending_user_decision`` and has no
    default.

Stream keys: ``farm_mean`` and ``farm_history`` use sub-keys ``(purpose_index, farm_index)`` with
``purpose_index = CORE_STREAMS[purpose]``, so validation farms and test farms are different
worlds (T5).  Within-farm draws use ``model.draw(streams, purpose, n, farm_index)``.

Ratios ``r`` (DC-03): only ``true_only`` ratios may generate the true batch state; ratios derived
from observed/residual SDs already contain sampling and laboratory error.  They may describe a
*decision prior* (what a farm knows from its own lab history), and then the prior-variance basis
is ``observed_incl_sampling_and_lab``: adding the assay's sampling/lab error to that prior double
counts, which :func:`assay_double_count_report` detects through
:func:`ration_reliability.information.signal.double_count_guard` (contract 7.3, T7.5; F06).
H0 priors (NASEM SD, historical lab results) have the same observed basis (DC-01).

Cells without a ratio: ``uncovered_cells`` must be chosen explicitly -- ``"P0"`` (no farm layer:
``mu_farm = mu_P0``, within SD ``= s sigma_P0``, prior basis as H0) or ``"error"``.

Correlation in the two-layer world (FIX5, red-team finding)
    A candidate correlation structure (C1, C4, C_CONS, ...) describes the *population* joint
    distribution.  Put only into the within layer, the marginal correlation of ``theta`` is diluted
    to about ``rho r_a r_b`` (Gaussian scale; :func:`two_layer_marginal_rho`), so IS-1 (one-layer
    copula) and IS-2 (two layers) would be different joint worlds under the "same" candidate.  The
    scope is therefore an explicit choice whenever a correlation is given:
    ``correlation_scope`` = ``"within_only"`` | ``"between_only"`` | ``"within_and_between"``
    (``between_correlation`` is drawn with a Gaussian copula in :meth:`TwoLayerFarmModel.farm_means`).
    Which scope E3 uses is ``pending_user_decision`` (D-19 / D-10); there is no default.

Truth-side basis of the double-count guard (FIX5, red-team finding)
    The guard of :mod:`ration_reliability.information.signal` only sees the *decision prior's*
    basis.  The SD that generates the true state here is ``s sigma_P0`` (marginal, H0) or
    ``sigma_within`` (H1, covered cells); ``truth_sd_basis`` declares what the supplied P0 SDs are
    (default ``observed_incl_sampling_and_lab``: NASEM SDs contain historical sampling and lab error,
    DC-01; ``true_batch_state`` only for SDs that the caller has actually de-convolved).  With
    ``truth_model=...`` :func:`assay_double_count_report` also checks the truth side, and a
    ``prior_deconvolved`` declaration is accepted only when the truth model's generating SD for that
    component is ``true_batch_state``.  Limitations kept visible (not solved here):
    ``sigma_between^2 = (s sigma_P0)^2 - sigma_within^2`` absorbs the measurement error of an
    observed-basis P0 SD into the farm layer (:attr:`TwoLayerFarmModel.between_layer_contains_measurement_error`);
    and the H1_m history error ``e`` uses ``sigma_within`` only (no sampling/lab error of the
    historical assays), which favours H1 -- whether to add it is part of D-FIX-5.

No parameter values are embedded; ratios come from the caller (for example the ``ratio_table`` of
``configs/uncertainty.yaml``, parsed by :func:`parse_ratio_table`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .correlation import CorrelationStructure, GaussianCopulaModel
from .distributions import fit_array
from .streams import CORE_STREAMS, RandomStreams

__all__ = [
    "INFORMATION_STATES",
    "RATIO_BASES",
    "PRIOR_BASIS_OF_RATIO",
    "H0_PRIOR_BASIS",
    "PendingDecisionError",
    "RatioEntry",
    "parse_ratio",
    "parse_ratio_table",
    "ratio_grid",
    "variance_split",
    "two_layer_marginal_rho",
    "CORRELATION_SCOPES",
    "TRUTH_SD_BASES",
    "DecisionPrior",
    "TwoLayerFarmModel",
    "assay_double_count_report",
    "require_no_double_count",
]

#: Explicit scope of a correlation structure in the two-layer world (FIX5; no default).
CORRELATION_SCOPES = ("within_only", "between_only", "within_and_between")
#: What the supplied P0 SDs represent (same vocabulary as ``information.signal.PRIOR_VARIANCE_BASES``;
#: equality is checked by a unit test -- the information package imports this one, so no import here).
TRUTH_SD_BASES = ("true_batch_state", "observed_incl_sampling_and_lab", "observed_incl_lab_only", "unidentified")

INFORMATION_STATES = ("H0_table_only", "H1_perfect", "H1_m")
#: Ratio component bases: ``true_only`` (true batch variation only) or observed (incl. sampling+lab).
RATIO_BASES = ("true_only", "observed_incl_sampling_and_lab")
#: Mapping to :data:`ration_reliability.information.signal.PRIOR_VARIANCE_BASES`.
PRIOR_BASIS_OF_RATIO = {"true_only": "true_batch_state",
                        "observed_incl_sampling_and_lab": "observed_incl_sampling_and_lab"}
#: Prior-variance basis of a P0 (table) SD: historical lab results of field samples (DC-01).
H0_PRIOR_BASIS = "observed_incl_sampling_and_lab"


class PendingDecisionError(ValueError):
    """A required research decision (e.g. ``m_hist``) is still ``pending_user_decision``."""


# ----------------------------------------------------------------------------------------------
# ratio table
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RatioEntry:
    """One within-farm / population SD ratio ``r`` for ``(ingredient_id, component)``."""

    ingredient_id: str
    component: str
    ratio: float
    basis: str
    horizon: str
    source: str
    status: str = "research_scenario_assumption"
    raw: str = ""

    def __post_init__(self) -> None:
        if self.basis not in RATIO_BASES:
            raise ValueError(f"ratio basis must be one of {RATIO_BASES}")
        r = float(self.ratio)
        if not (0.0 < r < 1.0):
            raise ValueError(f"ratio must lie in (0, 1), got {r}")
        object.__setattr__(self, "ratio", r)


_RATIO_RE = re.compile(r"^\s*1\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def parse_ratio(text) -> float:
    """``"1/4.4" -> 1/4.4``; a plain number is accepted; anything else raises."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    m = _RATIO_RE.match(str(text))
    if not m:
        raise ValueError(f"cannot parse ratio {text!r}")
    return 1.0 / float(m.group(1))


def _component_basis(label: str) -> str:
    lab = str(label)
    if lab.strip().startswith("true_only"):
        return "true_only"
    return "observed_incl_sampling_and_lab"


def parse_ratio_table(entries: Sequence[Mapping], ingredient_map: Mapping[str, Optional[str]],
                      components: Sequence[str] = ("DM", "NDF", "starch", "CP"), source: str = ""
                      ) -> tuple[list[RatioEntry], list[str]]:
    """Parse ``ratio_table.entries`` of ``configs/uncertainty.yaml``.

    ``ingredient_map`` maps the configuration's ingredient label to a model ingredient id (``None``
    = not mapped, recorded).  The component basis is ``true_only`` only when the entry's
    ``component`` starts with ``true_only``; every other entry (observed, residual, within-farm
    total, single farm vs population) is treated as ``observed_incl_sampling_and_lab``.
    Returns ``(entries, notes)``.
    """
    out: list[RatioEntry] = []
    notes: list[str] = []
    for e in entries:
        label = str(e.get("ingredient"))
        iid = ingredient_map.get(label)
        if iid is None:
            notes.append(f"ratio entry for {label!r} not mapped to a model ingredient (skipped)")
            continue
        basis = _component_basis(e.get("component", ""))
        for c in components:
            if c in e and e[c] is not None:
                out.append(RatioEntry(iid, c, parse_ratio(e[c]), basis, str(e.get("horizon", "")), source,
                                      "research_scenario_assumption", str(e[c])))
    return out, notes


def ratio_grid(entries: Sequence[RatioEntry], basis: str = "true_only") -> list[float]:
    """Sorted distinct ratios of one basis (candidate grid; not frozen)."""
    if basis not in RATIO_BASES:
        raise ValueError(f"basis must be one of {RATIO_BASES}")
    return sorted({round(e.ratio, 12) for e in entries if e.basis == basis})


def variance_split(sd_p0: np.ndarray, r: np.ndarray, s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(sigma_between, sigma_within)`` for SD ``s * sd_p0`` and ratio ``r`` (NaN ratio -> NaN)."""
    sd = np.asarray(sd_p0, dtype=float) * float(s)
    r = np.asarray(r, dtype=float)
    fin = np.isfinite(r)
    if np.any(fin & ~((r > 0) & (r < 1))):
        raise ValueError("ratios must lie in (0, 1)")
    within = np.where(fin, r * sd, np.nan)
    between = np.where(fin, np.sqrt(np.maximum(sd * sd - within * within, 0.0)), np.nan)
    return between, within


def two_layer_marginal_rho(rho_between: float, rho_within: float, r_a: float, r_b: float) -> float:
    """Marginal correlation (Gaussian scale) of two cells in the two-layer world.

    With independent layers, ``sigma_within = r S`` and ``sigma_between = sqrt(1 - r^2) S``::

        rho_marginal = rho_between sqrt((1 - r_a^2)(1 - r_b^2)) + rho_within r_a r_b

    ``within_only`` (``rho_between = 0``) dilutes a population candidate ``rho`` to
    ``rho r_a r_b``; the same ``rho`` in both layers keeps it exactly when ``r_a = r_b`` and
    slightly below otherwise.  The copula's Pearson correlation after the marginal transforms
    differs again (``correlation.induced_pearson``).
    """
    for r in (r_a, r_b):
        if not (0.0 < float(r) < 1.0):
            raise ValueError("ratios must lie in (0, 1)")
    ra, rb = float(r_a), float(r_b)
    return float(rho_between) * np.sqrt((1.0 - ra * ra) * (1.0 - rb * rb)) + float(rho_within) * ra * rb


# ----------------------------------------------------------------------------------------------
# decision priors
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionPrior:
    """What the decision maker knows before any assay (means, SDs and prior-variance bases).

    ``prior_variance_basis`` maps ``(ingredient_id, component)`` to a value of
    ``information.signal.PRIOR_VARIANCE_BASES`` for every stochastic cell; it is the input
    :func:`assay_double_count_report` passes to ``double_count_guard``.
    """

    state_id: str
    theta_mean: np.ndarray
    theta_sd: np.ndarray
    d_hat: np.ndarray
    d_sd: np.ndarray
    prior_variance_basis: Mapping[tuple, str]
    farm_index: Optional[int] = None
    stream_ids: tuple = field(default_factory=tuple)
    notes: tuple = field(default_factory=tuple)
    #: id of the :class:`TwoLayerFarmModel` that produced this prior (checked against ``truth_model``).
    model_id: Optional[str] = None


def _ro(a) -> np.ndarray:
    arr = np.array(a, dtype=float)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class TwoLayerFarmModel:
    """Two-layer (between-farm / within-farm) world for the H0 / H1 information states.

    Parameters are P0 target moments (canonical units), the within ratios ``theta_ratio [I, J]``
    and ``d_ratio [I]`` (NaN = no ratio for that cell), the SD scale ``s``, the marginal family
    (both layers) and the explicit policy for uncovered cells.  All ratios must be ``true_only``
    (DC-03); :meth:`decision_prior` may additionally use observed ratios for the decision prior.

    FIX5 fields (defaults keep the previous behaviour byte for byte):
    ``between_correlation`` -- correlation of the farm-mean layer (Gaussian copula, labels must be
    covered cells); ``correlation_scope`` -- required whenever ``within_correlation`` or
    ``between_correlation`` is given, one of :data:`CORRELATION_SCOPES` and consistent with which
    structures are present; ``truth_sd_basis`` -- what ``theta_sd_p0`` / ``d_sd_p0`` represent
    (:data:`TRUTH_SD_BASES`; default ``observed_incl_sampling_and_lab`` = the P0 SD as printed, DC-01).
    """

    model_id: str
    ingredient_ids: tuple
    nutrient_ids: tuple
    theta_mean_p0: np.ndarray
    theta_sd_p0: np.ndarray
    d_mean_p0: np.ndarray
    d_sd_p0: np.ndarray
    theta_ratio: np.ndarray
    d_ratio: np.ndarray
    ratio_basis: str
    s: float
    family: str
    uncovered_cells: str
    theta_lower: float = 0.0
    theta_upper: float = 1.0
    d_lower: float = 0.0
    d_upper: float = 1.0
    within_correlation: Optional[CorrelationStructure] = None
    is_synthetic: bool = False
    between_correlation: Optional[CorrelationStructure] = None
    correlation_scope: Optional[str] = None
    truth_sd_basis: str = H0_PRIOR_BASIS
    truth_sd_source: str = ""

    def __post_init__(self) -> None:
        for n in ("theta_mean_p0", "theta_sd_p0", "d_mean_p0", "d_sd_p0", "theta_ratio", "d_ratio"):
            object.__setattr__(self, n, _ro(getattr(self, n)))
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        for n, shp in (("theta_mean_p0", (I, J)), ("theta_sd_p0", (I, J)), ("theta_ratio", (I, J)),
                       ("d_mean_p0", (I,)), ("d_sd_p0", (I,)), ("d_ratio", (I,))):
            if getattr(self, n).shape != shp:
                raise ValueError(f"{n} must have shape {shp}")
        if self.ratio_basis != "true_only":
            raise ValueError("DC-03: only true_only ratios may generate the true batch state; observed/residual "
                             "ratios already contain sampling and laboratory error (use them only for "
                             "decision_prior(..., prior_ratio_basis='observed_incl_sampling_and_lab'))")
        if self.uncovered_cells not in ("P0", "error"):
            raise ValueError("uncovered_cells must be 'P0' or 'error' (explicit choice, no default)")
        s = float(self.s)
        if not np.isfinite(s) or s <= 0:
            raise ValueError("s must be finite and > 0")
        for r in (self.theta_ratio, self.d_ratio):
            fin = np.isfinite(r)
            if np.any(fin & ~((r > 0) & (r < 1))):
                raise ValueError("ratios must lie in (0, 1)")
        stoch_t = np.isfinite(self.theta_mean_p0) & np.isfinite(self.theta_sd_p0) & (self.theta_sd_p0 > 0)
        stoch_d = np.isfinite(self.d_sd_p0) & (self.d_sd_p0 > 0)
        if self.uncovered_cells == "error":
            if np.any(stoch_t & ~np.isfinite(self.theta_ratio)) or np.any(stoch_d & ~np.isfinite(self.d_ratio)):
                raise ValueError("stochastic cell(s) without a within ratio and uncovered_cells='error'")
        # FIX5: truth-side basis and explicit correlation scope
        if self.truth_sd_basis not in TRUTH_SD_BASES:
            raise ValueError(f"truth_sd_basis must be one of {TRUTH_SD_BASES}")
        if self.truth_sd_basis != H0_PRIOR_BASIS and not str(self.truth_sd_source).strip():
            raise ValueError("truth_sd_basis other than the P0 default needs truth_sd_source (how the SDs were "
                             "de-convolved, e.g. information.signal.deconvolve_true_sd with sourced error parameters)")
        has_w, has_b = self.within_correlation is not None, self.between_correlation is not None
        if not (has_w or has_b):
            if self.correlation_scope is not None:
                raise ValueError("correlation_scope given but no correlation structure")
        else:
            expected = {(True, False): "within_only", (False, True): "between_only",
                        (True, True): "within_and_between"}[(has_w, has_b)]
            if self.correlation_scope is None:
                raise ValueError("a correlation structure is given: correlation_scope must be chosen explicitly "
                                 f"({CORRELATION_SCOPES}); a population candidate put only into the within layer is "
                                 "diluted to about rho*r_a*r_b in the marginal (pending_user_decision, D-19/D-10)")
            if self.correlation_scope not in CORRELATION_SCOPES:
                raise ValueError(f"correlation_scope must be one of {CORRELATION_SCOPES}")
            if self.correlation_scope != expected:
                raise ValueError(f"correlation_scope {self.correlation_scope!r} does not match the structures given "
                                 f"(within: {has_w}, between: {has_b} -> {expected!r})")
        if has_b:
            tcov, dcov = np.isfinite(self.theta_ratio), np.isfinite(self.d_ratio)
            bad = []
            for (iid, comp) in self.between_correlation.labels:
                if iid not in self.ingredient_ids:
                    bad.append(f"{iid}:{comp}")
                    continue
                i = self.ingredient_ids.index(iid)
                if comp == "DM":
                    ok = bool(dcov[i] and stoch_d[i])
                elif comp in self.nutrient_ids:
                    j = self.nutrient_ids.index(comp)
                    ok = bool(tcov[i, j] and stoch_t[i, j])
                else:
                    ok = False
                if not ok:
                    bad.append(f"{iid}:{comp}")
            if bad:
                raise ValueError("between_correlation labels must be stochastic cells with a within ratio "
                                 "(covered cells have a farm layer): " + ", ".join(bad))

    @property
    def between_layer_contains_measurement_error(self) -> bool:
        """True unless the P0 SDs were declared de-convolved (``truth_sd_basis='true_batch_state'``):
        ``sigma_between^2 = (s sigma_P0)^2 - sigma_within^2`` then absorbs the sampling/lab error of
        the observed SD, so the marginal ``theta`` still contains it (``configs/uncertainty.yaml``
        H1_farm_history.limitation)."""
        return self.truth_sd_basis != "true_batch_state"

    def truth_variance_basis(self, state_id: str) -> dict:
        """Basis of the SD that generates the true state, as seen from information state ``state_id``.

        ``H0_table_only``: the marginal ``theta`` (both layers, SD ``s sigma_P0``) -> ``truth_sd_basis``
        for every stochastic cell.  ``H1_perfect`` / ``H1_m``: covered cells are generated around the
        known farm mean with ``sigma_within`` from a ``true_only`` ratio (DC-03) -> ``true_batch_state``;
        uncovered cells -> ``truth_sd_basis``.  Keys ``(ingredient_id, component)``.
        """
        if state_id not in INFORMATION_STATES:
            raise ValueError(f"state_id must be one of {INFORMATION_STATES}")
        out = {}
        for c, covered in self._cells():
            out[c] = "true_batch_state" if (covered and state_id != "H0_table_only") else self.truth_sd_basis
        return out

    # ------------------------------------------------------------------------------------------
    def between_within(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(theta_between, theta_within, d_between, d_within)`` SDs; uncovered cells get
        between = 0 and within = ``s * sigma_P0``."""
        s = float(self.s)
        tb, tw = variance_split(self.theta_sd_p0, self.theta_ratio, s)
        db, dw = variance_split(self.d_sd_p0, self.d_ratio, s)
        tcov, dcov = np.isfinite(self.theta_ratio), np.isfinite(self.d_ratio)
        tb = np.where(tcov, tb, 0.0)
        tw = np.where(tcov, tw, s * self.theta_sd_p0)
        db = np.where(dcov, db, 0.0)
        dw = np.where(dcov, dw, s * self.d_sd_p0)
        return tb, tw, db, dw

    def covered(self) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks of cells with a within ratio."""
        return np.isfinite(self.theta_ratio), np.isfinite(self.d_ratio)

    @staticmethod
    def _purpose_index(purpose: str) -> int:
        if purpose not in CORE_STREAMS:
            raise ValueError(f"purpose must be one of {tuple(CORE_STREAMS)}")
        return CORE_STREAMS[purpose]

    def farm_means(self, streams: RandomStreams, purpose: str, farm_index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """Draw ``mu_farm`` (theta ``[I, J]``, DM ``[I]``) for one farm world.

        Stream ``farm_mean`` with sub-keys ``(purpose_index, farm_index)``.  Moment matched with
        ``family`` (mean ``mu_P0``, SD ``sigma_between``); uncovered cells return ``mu_P0``.
        Without ``between_correlation`` every cell gets its own uniform (farm layer independent across
        components and ingredients -- the previous behaviour, unchanged); with it, the farm means are
        drawn from a Gaussian copula (:class:`~.correlation.GaussianCopulaModel`, one draw) on the same
        stream (FIX5).
        """
        pidx = self._purpose_index(purpose)
        rng = streams.generator("farm_mean", pidx, int(farm_index))
        tb, _, db, _ = self.between_within()
        tf = fit_array(self.family, self.theta_mean_p0, tb, self.theta_lower, self.theta_upper)
        df = fit_array(self.family, self.d_mean_p0, db, self.d_lower, self.d_upper)
        bad = [f"{self.ingredient_ids[i]}:{self.nutrient_ids[j]}" for i, j in np.ndindex(tf.shape)
               if tf[i, j].status not in ("matched", "point", "missing")]
        bad += [f"{self.ingredient_ids[i]}:DM" for i in range(df.shape[0]) if df[i].status not in ("matched", "point")]
        if bad:
            raise ValueError("between-farm layer not moment-matchable with family "
                             f"{self.family}: {', '.join(bad)}")
        if self.between_correlation is not None:
            cop = GaussianCopulaModel(f"{self.model_id}/between", self.ingredient_ids, self.nutrient_ids, tf,
                                      list(df), self.between_correlation, self.is_synthetic)
            th, dd = cop.sample(rng, 1)
            return th[0], dd[0], streams.stream_id("farm_mean", pidx, int(farm_index))
        I, J = tf.shape
        u = rng.random(I * J + I)
        mt = np.array([tf[i, j].ppf(np.array([u[i * J + j]]))[0] for i, j in np.ndindex(I, J)]).reshape(I, J)
        md = np.array([df[i].ppf(np.array([u[I * J + i]]))[0] for i in range(I)])
        return mt, md, streams.stream_id("farm_mean", pidx, int(farm_index))

    def within_model(self, theta_mu_farm: np.ndarray, d_mu_farm: np.ndarray, farm_label: str = "") -> GaussianCopulaModel:
        """True-batch model inside one farm world: mean ``mu_farm``, SD ``sigma_within``."""
        _, tw, _, dw = self.between_within()
        return GaussianCopulaModel.from_targets(
            f"{self.model_id}/within{('/' + farm_label) if farm_label else ''}", self.ingredient_ids,
            self.nutrient_ids, np.asarray(theta_mu_farm, dtype=float), tw, np.asarray(d_mu_farm, dtype=float), dw,
            family=self.family, theta_lower=self.theta_lower, theta_upper=self.theta_upper,
            d_lower=self.d_lower, d_upper=self.d_upper, correlation=self.within_correlation,
            is_synthetic=self.is_synthetic)

    def _cells(self) -> list[tuple]:
        out = []
        for i, iid in enumerate(self.ingredient_ids):
            for j, nid in enumerate(self.nutrient_ids):
                if np.isfinite(self.theta_sd_p0[i, j]) and self.theta_sd_p0[i, j] > 0:
                    out.append(((iid, nid), bool(np.isfinite(self.theta_ratio[i, j]))))
            if np.isfinite(self.d_sd_p0[i]) and self.d_sd_p0[i] > 0:
                out.append(((iid, "DM"), bool(np.isfinite(self.d_ratio[i]))))
        return out

    def decision_prior(self, state_id: str, theta_mu_farm: Optional[np.ndarray] = None,
                       d_mu_farm: Optional[np.ndarray] = None, *, streams: Optional[RandomStreams] = None,
                       purpose: Optional[str] = None, farm_index: Optional[int] = None,
                       m_hist: Optional[int] = None, prior_ratio_basis: str = "true_only",
                       observed_theta_ratio: Optional[np.ndarray] = None,
                       observed_d_ratio: Optional[np.ndarray] = None) -> DecisionPrior:
        """Decision-time prior of one information state.

        ``H0_table_only``: ``(mu_P0, s sigma_P0)``, basis ``observed_incl_sampling_and_lab`` (DC-01).
        ``H1_perfect``: ``(mu_farm, sigma_within)`` on covered cells.
        ``H1_m``: ``mu_hat = mu_farm + e`` (stream ``farm_history``), SD
        ``sigma_within sqrt(1 + 1/m_hist)``; ``m_hist=None`` raises :class:`PendingDecisionError`.
        ``prior_ratio_basis="observed_incl_sampling_and_lab"`` uses the observed ratios for the
        prior SD of covered cells and marks their basis accordingly (so adding assay sampling/lab
        error is caught by the double-count guard).  Uncovered cells always carry the H0 prior.
        A ``mu_hat`` outside the physical range raises (never clipped).
        """
        if state_id not in INFORMATION_STATES:
            raise ValueError(f"state_id must be one of {INFORMATION_STATES}")
        if prior_ratio_basis not in RATIO_BASES:
            raise ValueError(f"prior_ratio_basis must be one of {RATIO_BASES}")
        s = float(self.s)
        h0_t, h0_d = self.theta_mean_p0.copy(), self.d_mean_p0.copy()
        h0_ts, h0_ds = s * self.theta_sd_p0, s * self.d_sd_p0
        cells = self._cells()
        if state_id == "H0_table_only":
            basis = {c: self.truth_sd_basis for c, _ in cells}   # = H0_PRIOR_BASIS unless declared (FIX5)
            return DecisionPrior(state_id, _ro(h0_t), _ro(h0_ts), _ro(h0_d), _ro(h0_ds), basis,
                                 notes=("H0: P0 mean and s*SD; prior SD is an observed SD (DC-01)",),
                                 model_id=self.model_id)
        if theta_mu_farm is None or d_mu_farm is None:
            raise ValueError("H1 states need the farm means mu_farm (from farm_means)")
        tcov, dcov = self.covered()
        _, tw, _, dw = self.between_within()
        if prior_ratio_basis == "observed_incl_sampling_and_lab":
            if observed_theta_ratio is None or observed_d_ratio is None:
                raise ValueError("observed ratios required for prior_ratio_basis='observed_incl_sampling_and_lab'")
            _, tw = variance_split(self.theta_sd_p0, observed_theta_ratio, s)
            _, dw = variance_split(self.d_sd_p0, observed_d_ratio, s)
            tcov = tcov & np.isfinite(tw)
            dcov = dcov & np.isfinite(dw)
        mt = np.where(tcov, np.asarray(theta_mu_farm, dtype=float), h0_t)
        md = np.where(dcov, np.asarray(d_mu_farm, dtype=float), h0_d)
        sd_t = np.where(tcov, tw, h0_ts)
        sd_d = np.where(dcov, dw, h0_ds)
        sids: tuple = ()
        notes = [f"{state_id}: covered cells use the farm layer; uncovered cells keep the H0 prior"]
        if state_id == "H1_m":
            if m_hist is None:
                raise PendingDecisionError("m_hist is pending_user_decision (configs/uncertainty.yaml "
                                           "e_information_state_history.H1_farm_history.m_hist); no default")
            m = int(m_hist)
            if m < 1:
                raise ValueError("m_hist must be >= 1")
            if streams is None or purpose is None or farm_index is None:
                raise ValueError("H1_m needs streams, purpose and farm_index for the farm_history stream")
            pidx = self._purpose_index(purpose)
            rng = streams.generator("farm_history", pidx, int(farm_index))
            et = rng.standard_normal(mt.shape) * np.where(tcov, sd_t, 0.0) / np.sqrt(m)
            ed = rng.standard_normal(md.shape) * np.where(dcov, sd_d, 0.0) / np.sqrt(m)
            mt = mt + et
            md = md + ed
            infl = np.sqrt(1.0 + 1.0 / m)
            sd_t = np.where(tcov, sd_t * infl, sd_t)
            sd_d = np.where(dcov, sd_d * infl, sd_d)
            sids = (streams.stream_id("farm_history", pidx, int(farm_index)),)
            live_t = np.isfinite(mt)
            if np.any(live_t & ~((mt > self.theta_lower) & (mt < self.theta_upper))) or \
                    np.any(~((md > self.d_lower) & (md <= self.d_upper))):
                raise ValueError("H1_m: estimated farm mean outside the physical range (not clipped); "
                                 "report this farm world as invalid for the chosen m_hist")
            notes.append(f"H1_m with m_hist={m}")
            notes.append("H1_m: the history error e uses the true within-farm SD only; sampling and laboratory "
                         "error of the m_hist historical assays is not included, which favours H1 (whether to add "
                         "it is part of D-FIX-5, pending_user_decision)")
        pb = PRIOR_BASIS_OF_RATIO[prior_ratio_basis]
        basis = {}
        for c, _ in cells:
            iid, comp = c
            i = self.ingredient_ids.index(iid)
            cov = bool(dcov[i]) if comp == "DM" else bool(tcov[i, self.nutrient_ids.index(comp)])
            basis[c] = pb if cov else self.truth_sd_basis
        return DecisionPrior(state_id, _ro(mt), _ro(sd_t), _ro(md), _ro(sd_d), basis, farm_index, sids, tuple(notes),
                             model_id=self.model_id)


# ----------------------------------------------------------------------------------------------
# double-count guard (aligned with information.signal.double_count_guard)
# ----------------------------------------------------------------------------------------------

def assay_double_count_report(prior: DecisionPrior, signal, prior_deconvolved: Optional[Mapping] = None, *,
                              truth_model: Optional[TwoLayerFarmModel] = None):
    """Run ``information.signal.double_count_guard`` for ``signal`` with this prior's bases.

    The mapping keys are converted to ``information.prior.ObservedComponent``; a component of the
    signal that the prior does not know is passed without a basis (the guard reports it as a
    violation).  Imported lazily (the information package imports this package).

    Truth side (FIX5): a ``prior_deconvolved`` declaration for a signal component is accepted only
    when ``truth_model`` is given and generates that component with a ``true_batch_state`` SD
    (:meth:`TwoLayerFarmModel.truth_variance_basis` for ``prior.state_id``); otherwise the report is
    a violation (a bare declaration is not evidence that the true state was de-convolved).  With
    ``truth_model`` the guard is also run on the truth-side bases (true state generated with an SD
    that already contains sampling/lab error + the same errors in the signal = double counting), and
    the prior must come from the same model (``model_id``).  The returned report keeps the prior-side
    declared inputs; a violation found only on the truth side therefore does not reproduce in
    ``information.value`` either way it is refused.
    """
    from ..information.prior import ObservedComponent
    from ..information.signal import DoubleCountReport, double_count_guard

    basis = {ObservedComponent(i, c): b for (i, c), b in prior.prior_variance_basis.items()}
    dec = None
    if prior_deconvolved is not None:
        dec = {ObservedComponent(*k) if isinstance(k, tuple) else k: bool(v) for k, v in prior_deconvolved.items()}
    rep = double_count_guard(signal, basis, dec)
    msgs = list(rep.messages)
    per = dict(rep.per_component)
    status = rep.status
    rank = {"ok": 0, "unidentified": 1, "violation": 2}

    def _raise(label: str, st: str, msg: str) -> None:
        nonlocal status
        msgs.append(msg)
        if rank[st] > rank.get(per.get(label, "ok"), 0):
            per[label] = st
        if rank[st] > rank[status]:
            status = st

    truth_basis = None
    if truth_model is not None:
        if prior.model_id is not None and prior.model_id != truth_model.model_id:
            _raise("_model", "violation", f"prior comes from model {prior.model_id!r}, truth model is "
                                          f"{truth_model.model_id!r}: the guard must see the model that generates theta")
        truth_basis = {ObservedComponent(i, c): b for (i, c), b in truth_model.truth_variance_basis(prior.state_id).items()}
        trep = double_count_guard(signal, truth_basis, None)
        for m in trep.messages:
            msgs.append("truth side: " + m)
        for lab, st in trep.per_component.items():
            if rank[st] > rank.get(per.get(lab, "ok"), 0):
                per[lab] = st
            if rank[st] > rank[status]:
                status = st
    for c, flag in (dec or {}).items():
        if not flag or c not in signal.components:
            continue
        if truth_basis is None:
            _raise(c.label(), "violation", f"{c.label()}: prior_deconvolved is declared but no truth model is bound; "
                                           "the declaration must be tied to the model that generates theta "
                                           "(pass truth_model=...)")
        elif truth_basis.get(c) != "true_batch_state":
            _raise(c.label(), "violation", f"{c.label()}: prior declared de-convolved, but the truth model generates "
                                           f"theta with an SD of basis {truth_basis.get(c)!r} (not de-convolved)")
    if status == rep.status and msgs == list(rep.messages):
        return rep
    return DoubleCountReport(status, tuple(msgs), per, rep.signal_fingerprint, rep.declared_basis,
                             rep.declared_deconvolved)


def require_no_double_count(prior: DecisionPrior, signal, prior_deconvolved: Optional[Mapping] = None, *,
                            truth_model: Optional[TwoLayerFarmModel] = None):
    """Like :func:`assay_double_count_report` but raise ``InvalidProblemError`` on a violation."""
    from ..errors import InvalidProblemError

    rep = assay_double_count_report(prior, signal, prior_deconvolved, truth_model=truth_model)
    if rep.status == "violation":
        raise InvalidProblemError("double counting of batch variation and assay error: " + " | ".join(rep.messages))
    return rep
