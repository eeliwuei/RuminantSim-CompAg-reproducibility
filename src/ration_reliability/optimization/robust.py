"""M3 robust ration formulation (contract section 9 M3, T4 M3): box, budget and scenario-set.

Three *different* uncertainty models are implemented and named after the set they actually use
(contract T4: "有限情景集合、连续区间、预算不确定集是不同模型。实现哪一种就报告哪一种"):

==============================  ==================================  ===================================
method id / function            uncertainty set U                   what the solution guarantees
==============================  ==================================  ===================================
``M3a_box_robust``              box (interval) on the parameters    every imposed probabilistic_nutrition
:func:`solve_box_robust`        ``d_i in [d_lo, d_hi]``,            constraint holds for **every** state
                                ``a_ij in [lo_ij, hi_ij]``          in the box (jointly, exact robust
                                (e.g. ``mu -/+ k sigma``)           counterpart, LP tolerance only)
``M3b_budget_robust``           Bertsimas & Sim (2004) budget set   row-wise: constraint k holds whenever
:func:`solve_budget_robust`     on the q-coefficients derived from  at most ``floor(Gamma_k)`` of its
                                the box, ``Gamma_k`` per row        coefficients sit at their box worst
                                                                    case and one more moves a fraction
``M3c_scenario_set_robust``     a finite list of joint states       every imposed constraint holds in the
:func:`solve_scenario_set_robust` ``(theta_s, d_s)``, s = 1..N,     N listed states only ("scenario-set
                                taken from the ``opt`` stream       robust"); no continuous-set or
                                                                    out-of-sample guarantee
==============================  ==================================  ===================================

None of the three makes a probability statement by itself.  A reliability number for any
returned ``q`` comes only from the shared evaluator (:func:`ration_reliability.evaluation.evaluate`)
under a declared distribution.  The size of U (``k``, quantiles, ``Gamma``) is a research setting
to be selected on development/validation data (contract T8.1), never on the test stream.

Model and notation (contract T2)
--------------------------------
Executed decision ``q_i`` (kg as-fed/head/d, ``q >= 0``); scenario DM ``d_i`` (kg DM/kg as-fed);
composition ``a_ij`` (kg/kg DM or Mcal/kg DM).  For compiled constraint ``k`` (see
:mod:`ration_reliability.nutrition.constraints`) the per-ingredient content is
``c_ki(a_i) = sum_j W[k,i,j] a_ij + w0[k,i]`` and, with ``sigma_k = +1`` for ``le`` and ``-1`` for
``ge``, the linearised row is ``g_k(q, theta) = sum_i A_ki(theta_i) q_i - b_k <= 0`` with

* concentration (``E = S / D``, ``D = sum_i q_i d_i``):
  ``A_ki = sigma_k d_i (c_ki - K_k)``, ``b_k = 0``  (i.e. ``K D - S <= 0`` or ``S - K D <= 0``);
* supply (``E = S``): ``A_ki = sigma_k (d_i c_ki + v_ki)``, ``b_k = sigma_k K_k``.

``A_ki`` depends only on ingredient i's own parameters ``theta_i = (d_i, a_i1..a_iJ)``.

Box (M3a): derivation of the coefficient intervals
--------------------------------------------------
Let ``U = prod_i U_i`` with ``U_i = [d_lo_i, d_hi_i] x prod_j [lo_ij, hi_ij]`` and ``d_lo_i > 0``.

1. *Separability.*  For ``q >= 0`` and a product set,
   ``max_{theta in U} sum_i A_ki(theta_i) q_i = sum_i q_i max_{theta_i in U_i} A_ki(theta_i)``,
   because each summand depends on its own block ``theta_i`` only and ``q_i >= 0``.  The robust
   counterpart is therefore the LP row ``sum_i Amax_ki q_i <= b_k`` -- exact, no approximation.
2. *Composition part.*  ``c_ki`` is affine in ``a_i``; over the box it ranges in
   ``[cmin_ki, cmax_ki]`` with ``cmax = w0 + sum_j max(W lo, W hi)`` and
   ``cmin = w0 + sum_j min(W lo, W hi)``.  ``W[k,i,j]`` already *aggregates* every term that uses
   the same ``a_ij`` (e.g. ``NDF + 2 fNDF`` gives ``W = 1 + 2 w_forage``), so one physical
   quantity is never treated as two independent ones.  Different nutrients ``j`` of the same
   ingredient are extremised independently -- that is what a box means (see caveat below).
3. *DM x composition product.*  With ``h_k(c) = sigma_k (c - K_k)`` (concentration) or
   ``sigma_k c`` (supply), ``A_ki = d_i h_k(c_ki) (+ sigma_k v_ki)`` is **bilinear** in
   ``(d_i, c_ki)`` on the rectangle ``[d_lo, d_hi] x [cmin, cmax]``; its maximum is attained at
   one of the four corners:  ``Amax_ki = max{d h : d in {d_lo, d_hi}, h in {h(cmin), h(cmax)}}``.
   Since ``d > 0`` this is ``d_hi * hmax`` if ``hmax >= 0`` and ``d_lo * hmax`` otherwise.
4. *The denominator D.*  ``D = sum_i q_i d_i`` enters the linearised concentration row through
   the same ``d_i`` that multiplies the numerator: the per-ingredient coefficient is
   ``d_i (K - c_i)`` (``ge``).  The worst case is taken **jointly** over that single ``d_i``.
   Taking a separate worst ``d`` for the numerator ``N`` and for the denominator ``D`` would
   correspond to a physically inconsistent state and would be over-conservative.  Reading of the
   result: an ingredient whose worst-case content is on the wrong side of ``K`` (``h >= 0``,
   e.g. a low-CP forage under a CP minimum) is worst at ``d_hi`` (more of its DM dilutes the
   ration); an ingredient that improves the ratio even at its worst content (``h < 0``) is worst
   at ``d_lo``.
5. *Equivalence with the ratio constraint.*  For every ``theta in U``, ``D(q, theta) > 0``
   whenever ``q != 0`` (because ``d_lo > 0``), and ``E = S/D >= K  <=>  K D - S <= 0``.  Hence
   "linearised row holds for all theta in U" is equivalent to "concentration holds for all theta
   in U": the box robust counterpart is exact for the ratio constraint as well.
6. *Joint event.*  One ``q`` is feasible for every row for every ``theta in U`` simultaneously,
   so the whole joint event ``I(q, theta) = 0`` holds on U (not merely each row separately).

Caveat (contract T4 M3): a box built from marginal intervals may contain states that are jointly
implausible (e.g. all ingredients at their worst simultaneously, or nutrient pairs against a
known correlation).  It is a conservative comparison, not a known joint range.  ``mu -/+ k sigma``
is a set definition, not a probability statement; the fraction of a distribution inside the box
is not ``1 - alpha`` in general.

Budget (M3b): Bertsimas, D. & Sim, M. (2004) The Price of Robustness.  Operations Research
52(1):35-53, doi:10.1287/opre.1030.0065 (metadata checked via api.crossref.org, 2026-09-24).
------------------------------------------------------------------------------------------------
Per probabilistic row ``k`` the coefficients vary in ``[Abar_ki - dn_ki, Abar_ki + Ahat_ki]``
where ``Abar`` is the *nominal* coefficient (by default exactly the M0 ``nominal_point`` row:
``theta = composition table values``, ``d = d_hat``) and ``Ahat_ki = Amax_ki - Abar_ki >= 0``
comes from the box of M3a.  Because ``q >= 0`` and the row is ``<= b``, only upward deviations
can hurt, so the asymmetric interval reduces to the upward half.  The protection function

    beta_k(q, Gamma_k) = max { sum_{i in J_k} Ahat_ki q_i u_i :  sum_i u_i <= Gamma_k, 0 <= u_i <= 1 }

(J_k = coefficients with ``Ahat_ki > 0``) is replaced by its LP dual, which gives the linear
robust counterpart derived in B&S 2004 (theorem numbers not re-checked here).  Variable names
follow B&S (``z_k`` per row, ``p_ki`` per entry; their ``y`` equals ``q`` because ``q >= 0``)::

    sum_i Abar_ki q_i + Gamma_k z_k + sum_{i in J_k} p_ki <= b_k
    z_k + p_ki >= Ahat_ki q_i                  for i in J_k
    z_k >= 0, p_ki >= 0

``Gamma_k = 0`` gives the nominal row (M0); ``Gamma_k >= |J_k|`` gives
``sum_i (Abar + Ahat) q = sum_i Amax q``, i.e. the box row (M3a).  Values above ``|J_k|`` are
clipped to ``|J_k|`` and reported.  Here a "coefficient" is one ingredient's whole contribution to
row k, so ``Gamma_k`` counts *ingredients* whose parameters are allowed to sit at their worst case
in that row.  The guarantee is **row-wise** (B&S protect each row separately); the joint event is
not guaranteed for the same adversarial state across rows.  The probability bounds of B&S
2004 require independent, symmetric coefficient perturbations; here coefficients are
bilinear functions of ``(d, a)``, distributions may be truncated or correlated, so **those bounds
are not claimed**; ``Gamma`` is a conservatism parameter to be chosen on development data.

Scenario set (M3c)
------------------
``g_k(q, theta_s) <= 0`` for every listed state ``s`` and every imposed probabilistic row.  This
is the joint chance-constrained SAA with ``floor(alpha N) = 0`` (every ``z_s = 0``), i.e. its
``alpha -> 0`` limit on the same scenarios; it gives no guarantee outside the N states.  Its cost
is always ``>=`` the multi-scenario *mean* LP on the same states (M0 ``draw_mean``) because every
state constraint implies the averaged one; it is ``>=`` the M0 ``nominal_point`` cost only if the
nominal state is one of the scenarios (otherwise it can be lower -- see the tests).

Common rules (ENGINE_API section 6)
-----------------------------------
* same :class:`RationProblem`; only ``structural_hard`` and ``probabilistic_nutrition``
  constraints are imposed; structural rows are deterministic (``d_hat``) and never robustified
  nor relaxed; ``diagnostic_only`` rows are never imposed;
* LP in x-space (kg DM/head/d) with ``q = x / d_hat`` -- a positive column scaling that preserves
  every statement above; ``d_hat`` is the decision-time estimate, never a hidden draw;
* sets built from draws accept the ``opt`` stream only (:func:`require_stream`,
  :class:`~ration_reliability.errors.LeakageError` otherwise); ``validation``/``test`` are never
  read here;
* infeasible / failed problems return no ration and no cost; nothing is relaxed;
* scoring is left to the public evaluator.

Not registered in :data:`ration_reliability.optimization.REGISTRY` by this module (task rule: new
files only); the three functions already follow the :class:`SolveFn` signature.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..datamodel import (
    NUTRIENT_DIMENSIONS,
    ConstraintClass,
    ConstraintKind,
    RationDecision,
    RationProblem,
    Sense,
    SolverOptions,
    SolveResult,
    SolveStatus,
)
from ..errors import InvalidProblemError
from ..hashing import stable_hash
from ..nutrition.constraints import CompiledConstraints, linear_rows
from ..uncertainty.base import DrawSet, require_stream
from ..uncertainty.models import IndependentNormalModel
from .highs import run_linprog, solver_version_string
from .lp_builder import assemble_x_space_lp, optimization_indices

__all__ = [
    "METHOD_ID_BOX",
    "METHOD_ID_BUDGET",
    "METHOD_ID_SCENARIO_SET",
    "BoxUncertaintySet",
    "box_coefficient_bounds",
    "box_worst_case_residual",
    "budget_protection",
    "solve_box_robust",
    "solve_budget_robust",
    "solve_scenario_set_robust",
]

METHOD_ID_BOX = "M3a_box_robust"
METHOD_ID_BUDGET = "M3b_budget_robust"
METHOD_ID_SCENARIO_SET = "M3c_scenario_set_robust"

_SOLVER = "scipy.optimize.linprog(method='highs')"
_OPT_ONLY = ("opt",)
#: relative tolerance when checking that the nominal coefficient lies inside the box interval
_NOMINAL_IN_BOX_RTOL = 1e-9

#: Nutrient dimensions whose per-feed values carry no physical sign restriction (review round 2, FIX_B /
#: B-K7-1).  An ``energy_density`` column such as ``NEL_fixedDMI`` (nutrition.energy) is a per-feed
#: *contribution* that is negative for minerals; clipping its box at 0 would silently cut the evaluation
#: world.  With ``column_dimensions`` given, such columns are never clipped; ``mass_fraction`` columns get
#: the caller's ``theta_clip`` (physical range).
_UNCLIPPED_DIMENSIONS = frozenset({"energy_density"})

_BOX_DEFAULTS: dict[str, Any] = {"uncertainty_set": None, "k": None, "quantiles": None,
                                 "information_state": "t0_reference_only"}
_BUDGET_DEFAULTS: dict[str, Any] = {**_BOX_DEFAULTS, "gamma": None, "nominal_mode": "nominal_point"}
_SCEN_DEFAULTS: dict[str, Any] = {"information_state": "t0_reference_only"}


def _ro(a) -> np.ndarray:
    arr = np.array(a, dtype=float)
    arr.setflags(write=False)
    return arr


# ================================================================================================
# uncertainty set
# ================================================================================================

@dataclass(frozen=True, eq=False)
class BoxUncertaintySet:
    """Box (interval) uncertainty set on ``theta = (a [I, J], d [I])``, canonical units, DM basis.

    ``theta_lo/theta_hi`` may be NaN (missing) -- then both must be NaN at that cell; a robust
    method fails with ``invalid_input`` if a used row needs it (never set to 0).  ``d_lo > 0`` is
    required (needed for ``D > 0`` in every state, docstring point 5).

    Construct with :meth:`from_mean_sd`, :meth:`from_bounds`, :meth:`from_factory_model` (the research
    path: moments of the uncertainty-factory model that also generates the evaluation draws, review
    R3), :meth:`from_opt_draws` or the legacy :meth:`from_independent_normal`; each records
    ``construction`` and ``construction_params``.
    """

    set_id: str
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    theta_lo: np.ndarray
    theta_hi: np.ndarray
    d_lo: np.ndarray
    d_hi: np.ndarray
    construction: str
    construction_params: Mapping[str, Any] = field(default_factory=dict)
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        for name in ("theta_lo", "theta_hi", "d_lo", "d_hi"):
            object.__setattr__(self, name, _ro(getattr(self, name)))
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))
        object.__setattr__(self, "construction_params", dict(self.construction_params))
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        if self.theta_lo.shape != (I, J) or self.theta_hi.shape != (I, J) or \
                self.d_lo.shape != (I,) or self.d_hi.shape != (I,):
            raise ValueError("BoxUncertaintySet: shapes do not match labels")
        nl, nh = np.isnan(self.theta_lo), np.isnan(self.theta_hi)
        if not np.array_equal(nl, nh):
            raise ValueError("BoxUncertaintySet: theta_lo/theta_hi must be NaN at the same cells")
        fin = ~nl
        if np.any(~np.isfinite(self.theta_lo[fin])) or np.any(~np.isfinite(self.theta_hi[fin])):
            raise ValueError("BoxUncertaintySet: bounds must be finite (NaN = missing only)")
        if np.any(self.theta_lo[fin] > self.theta_hi[fin]):
            raise ValueError("BoxUncertaintySet: theta_lo > theta_hi")
        if not (np.all(np.isfinite(self.d_lo)) and np.all(np.isfinite(self.d_hi))):
            raise ValueError("BoxUncertaintySet: DM bounds must be finite")
        if np.any(self.d_lo <= 0) or np.any(self.d_hi > 1) or np.any(self.d_lo > self.d_hi):
            raise ValueError("BoxUncertaintySet: need 0 < d_lo <= d_hi <= 1")

    # ---- constructors ---------------------------------------------------------------------
    @classmethod
    def from_bounds(cls, set_id: str, ingredient_ids: Sequence[str], nutrient_ids: Sequence[str],
                    theta_lo, theta_hi, d_lo, d_hi, *, construction: str = "explicit_bounds",
                    source_note: str = "", is_synthetic: bool = False) -> "BoxUncertaintySet":
        """Explicit bounds, e.g. empirical percentile ranges or table Min/Max (record the source)."""
        return cls(set_id, tuple(ingredient_ids), tuple(nutrient_ids), theta_lo, theta_hi, d_lo, d_hi,
                   construction, {"source_note": source_note}, is_synthetic)

    @classmethod
    def from_mean_sd(cls, set_id: str, ingredient_ids: Sequence[str], nutrient_ids: Sequence[str],
                     theta_mean, theta_sd, d_mean, d_sd, k: float, *,
                     theta_clip: tuple[Optional[float], Optional[float]] = (0.0, None),
                     d_clip: tuple[float, float] = (1e-6, 1.0), source_note: str = "",
                     is_synthetic: bool = False,
                     column_dimensions: Optional[Mapping[str, str]] = None) -> "BoxUncertaintySet":
        """``mu -/+ k sigma`` per coordinate, clipped to the physical range (clips are counted).

        ``k`` is a set size chosen on development data; it is not a nutritional safety factor.
        ``column_dimensions`` (``{nutrient_id: NutrientSpec.dimension}``) makes the clip per column:
        ``mass_fraction`` columns get ``theta_clip``, ``energy_density`` columns are not clipped (review
        round 2, FIX_B).  A finite mean outside a column's clip range raises ``ValueError`` -- the box
        centre would lie outside the box (e.g. a negative energy contribution with the default clip at 0).
        """
        k = float(k)
        if not (np.isfinite(k) and k >= 0):
            raise ValueError("from_mean_sd: k must be finite and >= 0")
        tm, ts = np.asarray(theta_mean, float), np.asarray(theta_sd, float)
        dm, dsd = np.asarray(d_mean, float), np.asarray(d_sd, float)
        if np.any(ts[~np.isnan(tm)] < 0) or np.any(np.isnan(ts[~np.isnan(tm)])) or np.any(dsd < 0):
            raise ValueError("from_mean_sd: sd must be >= 0 (and given wherever the mean is)")
        c_lo, c_hi, c_basis = _column_clips(nutrient_ids, theta_clip, column_dimensions)
        bad = _columns_outside_clip(tm, c_lo, c_hi, nutrient_ids)
        if bad:
            raise ValueError(f"from_mean_sd: finite means outside the clip range in column(s) {bad} (a signed column "
                             "such as an energy_density contribution? pass column_dimensions or theta_clip=(None, None)); "
                             "the box is never clipped past its own centre")
        lo, hi = tm - k * ts, tm + k * ts
        lo = np.where(np.isnan(tm), np.nan, lo)
        hi = np.where(np.isnan(tm), np.nan, hi)
        (lo, hi, n_th), (dlo, dhi, n_d) = _clip_theta_cols(lo, hi, c_lo, c_hi), _clip_d(dm - k * dsd, dm + k * dsd, d_clip)
        params = {"k": k, "theta_clip": list(theta_clip), "d_clip": list(d_clip), "n_theta_bounds_clipped": n_th,
                  "n_d_bounds_clipped": n_d, "source_note": source_note}
        if column_dimensions is not None:
            params.update(_column_clip_record(nutrient_ids, c_lo, c_hi, c_basis))
        return cls(set_id, tuple(ingredient_ids), tuple(nutrient_ids), lo, hi, dlo, dhi,
                   "mean_pm_k_sd", params, is_synthetic)

    @classmethod
    def from_factory_model(cls, model, k: float, *, set_id: Optional[str] = None,
                           d_min: float = 1e-6) -> "BoxUncertaintySet":
        """``m -/+ k s`` from the *sampled* marginals of a factory model (review R3: same world).

        ``model`` is a :class:`ration_reliability.uncertainty.factory.FactoryModel`; ``m``/``s`` are
        the analytic moments of the marginals it samples (for ``matched`` cells equal to the target
        moments; for a diagnostic naive cell the drifted moments, i.e. the world actually evaluated),
        clipped to each cell's physical range; DM lower bounds below ``d_min`` are raised to ``d_min``
        (``d_lo > 0`` is required, point 5 of the module doc).  Missing / excluded cells stay NaN.  The
        model, spec and metadata fingerprints are recorded, so the set is tied to the evaluation world.
        """
        from ..uncertainty.factory import FactoryModel

        if not isinstance(model, FactoryModel):
            raise TypeError("from_factory_model needs a FactoryModel (ration_reliability.uncertainty.factory)")
        k = float(k)
        if not (np.isfinite(k) and k >= 0):
            raise ValueError("from_factory_model: k must be finite and >= 0")
        tm, ts, dm, dsd = model.achieved_moments()
        tl, tu, dl, du = model.bounds()
        ts = np.where(np.isnan(tm), np.nan, np.where(np.isnan(ts), 0.0, ts))
        lo, hi = tm - k * ts, tm + k * ts
        n_th = int(np.sum(~np.isnan(lo) & (lo < tl)) + np.sum(~np.isnan(hi) & (hi > tu)))
        lo = np.where(np.isnan(lo), np.nan, np.maximum(lo, tl))
        hi = np.where(np.isnan(hi), np.nan, np.minimum(hi, tu))
        dsd = np.where(np.isnan(dsd), 0.0, dsd)
        dlo_raw, dhi_raw = dm - k * dsd, dm + k * dsd
        d_floor = np.maximum(dl, float(d_min))
        n_d = int(np.sum(dlo_raw < d_floor) + np.sum(dhi_raw > du))
        dlo = np.minimum(np.maximum(dlo_raw, d_floor), du)
        dhi = np.maximum(np.minimum(dhi_raw, du), dlo)
        meta = model.metadata
        params = {"k": k, "moment_basis": "achieved_moments_of_sampled_marginals",
                  "clipped_to_cell_ranges": True, "d_min": float(d_min), "n_theta_bounds_clipped": n_th,
                  "n_d_bounds_clipped": n_d, "model_id": model.model_id, "model_fingerprint": model.fingerprint(),
                  "spec_main_fingerprint": meta.spec_main_fingerprint, "metadata_fingerprint": meta.fingerprint(),
                  "is_diagnostic_model": bool(meta.is_diagnostic),
                  "source_note": f"uncertainty factory model {model.model_id}"}
        return cls(set_id or f"box[{model.model_id}]", model.ingredient_ids, model.nutrient_ids, lo, hi, dlo, dhi,
                   "factory_model_moment_pm_k_sd", params, bool(model.is_synthetic))

    @classmethod
    def from_independent_normal(cls, model: IndependentNormalModel, k: float,
                                set_id: Optional[str] = None) -> "BoxUncertaintySet":
        """``mu -/+ k sigma`` from a model's *declared* parameters (not its truncated moments),
        clipped to the model's support when it is truncated.

        Legacy adapter (review R3): for a truncated :class:`IndependentNormalModel` the declared
        ``mu``/``sigma`` are the parent normal's parameters, not the moments of the sampled world
        (naive-truncation drift).  Research entry points build models through the uncertainty
        factory and use :meth:`from_factory_model`; this adapter stays for the engine's synthetic
        tests and records ``parameter_basis = declared_parent_parameters`` in the construction
        parameters."""
        legacy = {"parameter_basis": "declared_parent_parameters_not_sampled_moments",
                  "entry_status": "legacy_adapter_not_uncertainty_factory"}
        if model.truncate:
            th_clip = (None, None)
            box = cls.from_mean_sd(set_id or f"box[{model.model_id}]", model.ingredient_ids, model.nutrient_ids,
                                   model.theta_mean, model.theta_sd, model.d_mean, model.d_sd, k,
                                   theta_clip=th_clip, d_clip=(max(float(model.d_lower), 1e-12), float(model.d_upper)),
                                   is_synthetic=model.is_synthetic,
                                   source_note=f"declared parameters of {model.model_id}")
            lo = np.maximum(box.theta_lo, model.theta_lower)
            hi = np.minimum(box.theta_hi, model.theta_upper)
            n = int(np.sum((lo != box.theta_lo) & ~np.isnan(lo)) + np.sum((hi != box.theta_hi) & ~np.isnan(hi)))
            lo = np.minimum(lo, hi)
            params = dict(box.construction_params)
            params.update({"model_fingerprint": model.fingerprint(), "clipped_to_model_support": True,
                           "n_theta_bounds_clipped": int(params["n_theta_bounds_clipped"]) + n, **legacy})
            return cls(box.set_id, box.ingredient_ids, box.nutrient_ids, lo, hi, box.d_lo, box.d_hi,
                       "model_mean_pm_k_sd", params, box.is_synthetic)
        box = cls.from_mean_sd(set_id or f"box[{model.model_id}]", model.ingredient_ids, model.nutrient_ids,
                               model.theta_mean, model.theta_sd, model.d_mean, model.d_sd, k,
                               is_synthetic=model.is_synthetic, source_note=f"declared parameters of {model.model_id}")
        params = dict(box.construction_params)
        params.update({"model_fingerprint": model.fingerprint(), "clipped_to_model_support": False, **legacy})
        return cls(box.set_id, box.ingredient_ids, box.nutrient_ids, box.theta_lo, box.theta_hi, box.d_lo, box.d_hi,
                   "model_mean_pm_k_sd", params, box.is_synthetic)

    @classmethod
    def from_opt_draws(cls, draws: DrawSet, *, k: Optional[float] = None,
                       quantiles: Optional[Sequence[float]] = None, set_id: Optional[str] = None,
                       theta_clip: tuple[Optional[float], Optional[float]] = (0.0, None),
                       d_clip: tuple[float, float] = (1e-6, 1.0),
                       column_dimensions: Optional[Mapping[str, str]] = None) -> "BoxUncertaintySet":
        """Box estimated from ``opt`` draws only: sample mean -/+ k sample SD (ddof=1), or the
        per-coordinate sample quantiles ``(q_lo, q_hi)``.  Any NaN in a coordinate keeps it
        missing.  Raises :class:`LeakageError` for any other stream.

        Clipping (review round 2, FIX_B / B-K7-1): ``theta_clip`` is a *physical-range* clip.  With
        ``column_dimensions`` (``{nutrient_id: dimension}``; the M3 methods pass the problem's nutrient
        dimensions) it applies to ``mass_fraction`` columns only; ``energy_density`` columns (e.g. the
        ``NEL_fixedDMI`` contribution, negative for minerals) are not clipped.  In every case a column
        with a finite draw outside its clip range raises ``ValueError``: the box is never clipped silently
        past the sampled world (before this fix the default clip at 0 cut the negative energy column,
        giving M3a an optimistic box and M3b ``invalid_input``)."""
        require_stream(draws, _OPT_ONLY, "BoxUncertaintySet.from_opt_draws")
        if (k is None) == (quantiles is None):
            raise ValueError("from_opt_draws: give exactly one of k or quantiles")
        th, d = draws.theta, draws.d
        c_lo, c_hi, c_basis = _column_clips(draws.nutrient_ids, theta_clip, column_dimensions)
        bad = _columns_outside_clip(th, c_lo, c_hi, draws.nutrient_ids)
        if bad:
            raise ValueError(f"from_opt_draws: opt draws outside the clip range in column(s) {bad} (a signed column "
                             "such as an energy_density contribution? pass column_dimensions or theta_clip=(None, None)); "
                             "clipping would cut the evaluation world")
        clip_rec = _column_clip_record(draws.nutrient_ids, c_lo, c_hi, c_basis)
        if k is not None:
            if draws.n_draws < 2:
                raise ValueError("from_opt_draws: need >= 2 draws for a sample SD")
            box = cls.from_mean_sd(set_id or f"box[{draws.stream_id}]", draws.ingredient_ids, draws.nutrient_ids,
                                   th.mean(axis=0), th.std(axis=0, ddof=1), d.mean(axis=0), d.std(axis=0, ddof=1),
                                   k, theta_clip=theta_clip, d_clip=d_clip, is_synthetic=draws.is_synthetic,
                                   source_note="opt-stream sample moments", column_dimensions=column_dimensions)
            construction = "opt_draws_mean_pm_k_sd"
        else:
            ql, qh = (float(x) for x in quantiles)
            if not (0.0 <= ql < qh <= 1.0):
                raise ValueError("from_opt_draws: quantiles must satisfy 0 <= q_lo < q_hi <= 1")
            lo, hi = np.quantile(th, ql, axis=0), np.quantile(th, qh, axis=0)
            nan_cell = np.isnan(th).any(axis=0)
            lo, hi = np.where(nan_cell, np.nan, lo), np.where(nan_cell, np.nan, hi)
            (lo, hi, n_th) = _clip_theta_cols(lo, hi, c_lo, c_hi)
            dlo, dhi, n_d = _clip_d(np.quantile(d, ql, axis=0), np.quantile(d, qh, axis=0), d_clip)
            box = cls(set_id or f"box[{draws.stream_id}]", draws.ingredient_ids, draws.nutrient_ids, lo, hi, dlo, dhi,
                      "tmp", {"quantiles": [ql, qh], "theta_clip": list(theta_clip), "d_clip": list(d_clip),
                              "n_theta_bounds_clipped": n_th, "n_d_bounds_clipped": n_d},
                      draws.is_synthetic)
            construction = "opt_draws_quantiles"
        params = dict(box.construction_params)
        params.update({"draws_stream_id": draws.stream_id, "draws_fingerprint": draws.fingerprint,
                       "n_opt_draws": draws.n_draws, **clip_rec})
        return cls(box.set_id, box.ingredient_ids, box.nutrient_ids, box.theta_lo, box.theta_hi, box.d_lo, box.d_hi,
                   construction, params, box.is_synthetic)

    # ---- utilities -------------------------------------------------------------------------
    def center(self) -> tuple[np.ndarray, np.ndarray]:
        """Box centre ``((lo + hi)/2, (d_lo + d_hi)/2)`` (NaN stays NaN)."""
        return (self.theta_lo + self.theta_hi) / 2.0, (self.d_lo + self.d_hi) / 2.0

    def contains(self, theta: np.ndarray, d: np.ndarray, atol: float = 1e-12) -> bool:
        """True if the state ``(theta [I, J], d [I])`` lies in the box (NaN cells ignored)."""
        theta, d = np.asarray(theta, float), np.asarray(d, float)
        m = ~np.isnan(self.theta_lo) & ~np.isnan(theta)
        ok_t = np.all(theta[m] >= self.theta_lo[m] - atol) and np.all(theta[m] <= self.theta_hi[m] + atol)
        ok_d = np.all(d >= self.d_lo - atol) and np.all(d <= self.d_hi + atol)
        return bool(ok_t and ok_d)

    def reordered(self, ingredient_ids: Sequence[str], nutrient_ids: Optional[Sequence[str]] = None) -> "BoxUncertaintySet":
        """Same set with permuted ingredient / nutrient axes."""
        ii = [self.ingredient_ids.index(i) for i in ingredient_ids]
        nut = self.nutrient_ids if nutrient_ids is None else tuple(nutrient_ids)
        jj = [self.nutrient_ids.index(j) for j in nut]
        return BoxUncertaintySet(self.set_id, tuple(ingredient_ids), tuple(nut),
                                 self.theta_lo[ii][:, jj], self.theta_hi[ii][:, jj], self.d_lo[ii], self.d_hi[ii],
                                 self.construction, self.construction_params, self.is_synthetic)

    def fingerprint(self) -> str:
        """Content hash of the set."""
        return stable_hash("BoxUncertaintySet/v1", self.set_id, self.ingredient_ids, self.nutrient_ids,
                           self.theta_lo, self.theta_hi, self.d_lo, self.d_hi, self.construction,
                           dict(self.construction_params), self.is_synthetic)

    def describe(self) -> dict[str, Any]:
        """JSON-able summary recorded in :attr:`SolveResult.params` (arrays via fingerprint)."""
        return {"set_type": "box", "set_id": self.set_id, "construction": self.construction,
                "construction_params": dict(self.construction_params), "fingerprint": self.fingerprint(),
                "is_synthetic": self.is_synthetic}


def _column_clips(nutrient_ids: Sequence[str], theta_clip, column_dimensions: Optional[Mapping[str, str]]
                  ) -> tuple[np.ndarray, np.ndarray, str]:
    """Per-column clip ``(lo_clip [J], hi_clip [J], basis)``; +/-inf = no clip (review round 2, FIX_B)."""
    a, b = theta_clip
    J = len(nutrient_ids)
    lo = np.full(J, -np.inf if a is None else float(a))
    hi = np.full(J, np.inf if b is None else float(b))
    if column_dimensions is None:
        return lo, hi, "uniform_theta_clip"
    missing = [n for n in nutrient_ids if n not in column_dimensions]
    unknown = sorted({str(column_dimensions[n]) for n in nutrient_ids
                      if n in column_dimensions and column_dimensions[n] not in NUTRIENT_DIMENSIONS})
    if missing or unknown:
        raise ValueError(f"column_dimensions: missing columns {missing}, unknown dimensions {unknown}")
    for j, n in enumerate(nutrient_ids):
        if column_dimensions[n] in _UNCLIPPED_DIMENSIONS:
            lo[j], hi[j] = -np.inf, np.inf
    return lo, hi, "per_column_by_dimension"


def _columns_outside_clip(values, c_lo, c_hi, nutrient_ids: Sequence[str]) -> list[str]:
    """Columns with a finite value (``[..., J]``) strictly outside their clip range."""
    v = np.asarray(values, dtype=float)
    fin = np.isfinite(v)
    below = fin & (v < c_lo)
    above = fin & (v > c_hi)
    axes = tuple(range(v.ndim - 1))
    bad = np.any(below | above, axis=axes) if axes else (below | above)
    return [str(nutrient_ids[j]) for j in np.flatnonzero(bad)]


def _column_clip_record(nutrient_ids: Sequence[str], c_lo, c_hi, basis: str) -> dict[str, Any]:
    def f(x):
        return None if not np.isfinite(x) else float(x)
    return {"column_clip_basis": basis,
            "column_clip": {str(n): [f(c_lo[j]), f(c_hi[j])] for j, n in enumerate(nutrient_ids)}}


def _clip_theta_cols(lo, hi, c_lo, c_hi):
    """Clip ``lo``/``hi`` ``[I, J]`` column-wise to ``[c_lo_j, c_hi_j]`` (NaN kept); count moved bounds."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    fin = ~np.isnan(lo)
    cl = np.broadcast_to(np.asarray(c_lo, float), lo.shape)
    ch = np.broadcast_to(np.asarray(c_hi, float), lo.shape)
    n = int(np.sum(fin & (lo < cl)) + np.sum(fin & (hi > ch)))
    lo2 = np.where(fin, np.minimum(np.maximum(lo, cl), ch), lo)
    hi2 = np.where(fin, np.maximum(np.minimum(hi, ch), cl), hi)
    return lo2, hi2, n


def _clip_theta(lo, hi, clip):
    lo, hi = np.array(lo, float), np.array(hi, float)
    a, b = clip
    n = 0
    fin = ~np.isnan(lo)
    if a is not None:
        n += int(np.sum(fin & (lo < a)))
        lo = np.where(fin & (lo < a), a, lo)
        hi = np.where(fin & (hi < a), a, hi)
    if b is not None:
        n += int(np.sum(fin & (hi > b)))
        hi = np.where(fin & (hi > b), b, hi)
        lo = np.where(fin & (lo > b), b, lo)
    return lo, hi, n


def _clip_d(lo, hi, clip):
    lo, hi = np.array(lo, float), np.array(hi, float)
    a, b = float(clip[0]), float(clip[1])
    if not (0.0 < a <= b <= 1.0):
        raise ValueError("d_clip must satisfy 0 < d_min <= d_max <= 1")
    n = int(np.sum(lo < a) + np.sum(hi > b))
    lo, hi = np.clip(lo, a, b), np.clip(hi, a, b)
    return lo, hi, n


# ================================================================================================
# robust rows
# ================================================================================================

def box_coefficient_bounds(cc: CompiledConstraints, box: BoxUncertaintySet
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact range of the q-space coefficients ``A_ki(theta_i)`` over the box.

    Only scenario-DM rows (``probabilistic_nutrition`` / ``diagnostic_only``) are accepted.

    Returns
    -------
    A_max, A_min : ``[K, I]`` maximum / minimum of ``A_ki`` over ``U_i`` (module doc, points 2-3).
    b : ``[K]`` right-hand side (``g = A q - b``).
    missing : ``[K, I]`` True where a needed bound is NaN (``A`` is NaN there).
    """
    if box.ingredient_ids != cc.ingredient_ids or box.nutrient_ids != cc.nutrient_ids:
        raise ValueError("box_coefficient_bounds: box labels differ from the constraints; use box.reordered()")
    if any(c is ConstraintClass.STRUCTURAL_HARD for c in cc.classes):
        raise ValueError("box_coefficient_bounds: structural rows are deterministic; pass scenario-DM rows only")
    K = cc.n_constraints
    W = cc.W
    lo, hi = box.theta_lo[None, :, :], box.theta_hi[None, :, :]
    t_hi = np.where(W > 0, W * hi, np.where(W < 0, W * lo, 0.0))
    t_lo = np.where(W > 0, W * lo, np.where(W < 0, W * hi, 0.0))
    c_max = t_hi.sum(axis=2) + cc.w0
    c_min = t_lo.sum(axis=2) + cc.w0
    sigma = np.array([1.0 if s is Sense.LE else -1.0 for s in cc.senses]).reshape(K, 1)
    conc = np.array([k is ConstraintKind.CONCENTRATION for k in cc.kinds], dtype=bool)
    shift = np.where(conc, cc.bound, 0.0).reshape(K, 1)
    h1, h2 = sigma * (c_min - shift), sigma * (c_max - shift)
    h_lo, h_hi = np.minimum(h1, h2), np.maximum(h1, h2)
    dlo, dhi = box.d_lo[None, :], box.d_hi[None, :]
    corners = np.stack([dlo * h_lo, dlo * h_hi, dhi * h_lo, dhi * h_hi])
    offset = np.where(conc.reshape(K, 1), 0.0, sigma * cc.v)
    A_max = corners.max(axis=0) + offset
    A_min = corners.min(axis=0) + offset
    b = np.where(conc, 0.0, sigma.ravel() * cc.bound)
    missing = np.isnan(A_max)
    return A_max, A_min, b, missing


def box_worst_case_residual(cc: CompiledConstraints, box: BoxUncertaintySet, q: np.ndarray) -> np.ndarray:
    """``max_{theta in box} g_k(q, theta)`` for every row (valid for ``q >= 0`` only)."""
    q = np.asarray(q, dtype=float)
    if np.any(q < 0):
        raise ValueError("box_worst_case_residual: the separable worst case requires q >= 0")
    A_max, _, b, _ = box_coefficient_bounds(cc, box)
    return A_max @ q - b


def budget_protection(ahat: np.ndarray, q: np.ndarray, gamma: float) -> float:
    """Exact Bertsimas-Sim protection ``beta(q, Gamma)`` of one row (``ahat >= 0``, ``q >= 0``).

    ``beta = sum of the floor(Gamma) largest ahat_i q_i + (Gamma - floor(Gamma)) * next largest``,
    with ``Gamma`` clipped to ``[0, number of entries]``.  Used to report residuals and to check the
    LP-dual reformulation independently.
    """
    v = np.sort(np.asarray(ahat, float) * np.asarray(q, float))[::-1]
    if np.any(v < -1e-15):
        raise ValueError("budget_protection: needs ahat >= 0 and q >= 0")
    g = min(max(float(gamma), 0.0), float(v.size))
    f = int(math.floor(g))
    frac = g - f
    return float(v[:f].sum() + (frac * v[f] if f < v.size else 0.0))


# ================================================================================================
# shared plumbing
# ================================================================================================

class _Fail(Exception):
    def __init__(self, status: SolveStatus, message: str, diagnostics: Optional[dict] = None):
        super().__init__(message)
        self.status, self.message, self.diagnostics = status, message, dict(diagnostics or {})


class _Run:
    """Common set-up for the three methods (validation, split of constraint classes, results)."""

    def __init__(self, method_id: str, problem: RationProblem, params, defaults, solver_options, opt_draws):
        self.t0 = time.perf_counter()
        self.method_id = method_id
        self.problem = problem
        self.opts = solver_options or SolverOptions()
        self.p = dict(defaults)
        self.p.update(dict(params or {}))
        self.defaults = defaults
        self.is_synth = bool(problem.is_synthetic or (opt_draws is not None and opt_draws.is_synthetic))
        self.currency = problem.prices.currency
        self.streams: tuple[str, ...] = ()
        self.input_hash = stable_hash(method_id, "pre", problem.problem_id, problem.ingredient_ids,
                                      self.hashable_params(), self.opts.to_dict())
        self.diag: dict[str, Any] = {}

    def hashable_params(self) -> dict:
        out = {}
        for k, v in self.p.items():
            if isinstance(v, BoxUncertaintySet):
                v = v.fingerprint()
            elif isinstance(v, Mapping):
                v = {str(a): b for a, b in v.items()}
            elif isinstance(v, (list, tuple)):
                v = list(v)
            try:
                stable_hash(v)
            except TypeError:  # unsupported object: hash its repr (the params check rejects it later)
                v = f"repr:{v!r}"
            out[k] = v
        return out

    def recorded_params(self) -> dict:
        out = {}
        for k, v in self.p.items():
            if isinstance(v, BoxUncertaintySet):
                v = v.describe()
            elif isinstance(v, float) and math.isinf(v):
                v = "full"  # JSON-safe spelling of Gamma = inf
            out[k] = v
        return out

    def check_params(self) -> None:
        unknown = sorted(set(self.p) - set(self.defaults))
        if unknown:
            raise _Fail(SolveStatus.INVALID_INPUT, f"unknown params {unknown}")

    def setup(self, d_hat) -> None:
        pr = self.problem
        try:
            cc_all = pr.compiled
        except InvalidProblemError as exc:
            raise _Fail(SolveStatus.INVALID_INPUT, str(exc)) from None
        self.ids = pr.ingredient_ids
        if len(self.ids) == 0:
            raise _Fail(SolveStatus.INVALID_INPUT, "no ingredients")
        self.prices = pr.price_vector()
        dh = pr.dm_estimates() if d_hat is None else np.asarray(d_hat, dtype=float)
        if dh.shape != (len(self.ids),) or np.any(~np.isfinite(dh)) or np.any(dh <= 0) or np.any(dh > 1):
            raise _Fail(SolveStatus.INVALID_INPUT, "d_hat must be finite in (0, 1] with shape [I]")
        self.dh = dh
        self.cc = cc_all.subset(optimization_indices(cc_all))
        cls = self.cc.classes
        self.idx_s = np.array([k for k, c in enumerate(cls) if c is ConstraintClass.STRUCTURAL_HARD], dtype=int)
        self.idx_p = np.array([k for k, c in enumerate(cls) if c is ConstraintClass.PROBABILISTIC_NUTRITION], dtype=int)
        self.cc_s = self.cc.subset(self.idx_s)
        self.cc_p = self.cc.subset(self.idx_p)
        rows_s = linear_rows(self.cc_s, pr.nominal_theta(), dh, d_hat=dh)
        self.A_s, self.b_s, self.eq_s = rows_s.A[0], rows_s.b, rows_s.is_eq
        self.diag.update({"imposed_constraint_ids": list(self.cc.constraint_ids),
                          "robustified_constraint_ids": list(self.cc_p.constraint_ids),
                          "structural_constraint_ids_not_robustified": list(self.cc_s.constraint_ids),
                          "representation": "x_space_kg_dm; q = x / d_hat"})

    def resolve_box(self, opt_draws: Optional[DrawSet]) -> BoxUncertaintySet:
        box = self.p.get("uncertainty_set")
        k, qs = self.p.get("k"), self.p.get("quantiles")
        if box is not None:
            if not isinstance(box, BoxUncertaintySet):
                raise _Fail(SolveStatus.INVALID_INPUT, "params['uncertainty_set'] must be a BoxUncertaintySet")
            if k is not None or qs is not None:
                raise _Fail(SolveStatus.INVALID_INPUT, "give either uncertainty_set or k/quantiles, not both")
        else:
            if (k is None) == (qs is None):
                raise _Fail(SolveStatus.INVALID_INPUT,
                            "no uncertainty set: pass params['uncertainty_set'] or exactly one of k / quantiles "
                            "(box estimated from opt_draws)")
            if opt_draws is None:
                raise _Fail(SolveStatus.INVALID_INPUT, "k/quantiles need opt_draws (opt stream)")
            require_stream(opt_draws, _OPT_ONLY, self.method_id)  # LeakageError is never downgraded
            if opt_draws.ingredient_ids != self.ids or opt_draws.nutrient_ids != self.problem.nutrient_ids:
                raise _Fail(SolveStatus.INVALID_INPUT, "opt_draws labels differ from the problem")
            try:
                box = BoxUncertaintySet.from_opt_draws(
                    opt_draws, k=k, quantiles=qs,
                    column_dimensions={n.nutrient_id: n.dimension for n in self.problem.nutrients})
            except ValueError as exc:
                raise _Fail(SolveStatus.INVALID_INPUT, f"cannot build box from opt_draws: {exc}") from None
            self.streams = (opt_draws.stream_id,)
        if box.ingredient_ids != self.ids or box.nutrient_ids != self.problem.nutrient_ids:
            raise _Fail(SolveStatus.INVALID_INPUT, "uncertainty set labels differ from the problem (use reordered())")
        self.is_synth = self.is_synth or box.is_synthetic
        if not self.streams and box.construction_params.get("draws_stream_id"):
            self.streams = (str(box.construction_params["draws_stream_id"]),)
        self.diag["uncertainty_set"] = box.describe()
        return box

    def box_rows(self, box: BoxUncertaintySet):
        A_max, A_min, b, miss = box_coefficient_bounds(self.cc_p, box)
        if np.any(miss):
            bad = sorted({f"{self.cc_p.constraint_ids[k]}:{self.ids[i]}" for k, i in np.argwhere(miss)})
            raise _Fail(SolveStatus.INVALID_INPUT, "missing box bounds (constraint:ingredient) " + ", ".join(bad))
        return A_max, A_min, b

    def fail(self, f: _Fail) -> SolveResult:
        d = dict(self.diag)
        d.update(f.diagnostics)
        return SolveResult(method_id=self.method_id, status=f.status, decision=None, objective=None,
                           objective_unit=f"{self.currency}/head/d", solver=_SOLVER,
                           solver_version=solver_version_string(), tolerances=self.opts.to_dict(),
                           wall_time_s=time.perf_counter() - self.t0, input_hash=self.input_hash, message=f.message,
                           params=self.recorded_params(), streams_used=self.streams, diagnostics=d,
                           is_synthetic=self.is_synth)

    def success(self, out, x_full: np.ndarray, residuals_p: np.ndarray, extra_diag: dict) -> SolveResult:
        q = np.asarray(x_full[: len(self.ids)], float) / self.dh
        dec = RationDecision(self.ids, q, self.dh, self.method_id, information_state=str(self.p["information_state"]))
        g_s = self.A_s @ q - self.b_s
        g_s = np.where(self.eq_s, np.abs(g_s), g_s)
        res_by_id = {cid: float(v) for cid, v in zip(self.cc_s.constraint_ids, g_s)}
        res_by_id.update({cid: float(v) for cid, v in zip(self.cc_p.constraint_ids, residuals_p)})
        residuals = {cid: res_by_id[cid] for cid in self.cc.constraint_ids}
        d = dict(self.diag)
        d.update(extra_diag)
        d.update({"highs_model_status": out.highs_model_status, "resolved_without_presolve": out.resolved_without_presolve,
                  "max_rel_residual_lp": out.max_rel_residual, "max_abs_residual_lp": out.max_abs_residual,
                  "solver_wall_time_s": out.wall_time_s, "lp_objective_x_space": out.fun,
                  "total_q_as_fed_kg": float(q.sum()),
                  "residual_semantics": "structural: nominal row; probabilistic: worst case over the method's "
                                        "uncertainty set (<= 0 satisfied)"})
        if not np.any(q > 0):
            d["warning"] = "all-zero ration is optimal: check that a DM-offer/intake rule is declared"
        return SolveResult(method_id=self.method_id, status=out.status, decision=dec, objective=float(self.prices @ q),
                           objective_unit=f"{self.currency}/head/d", solver=_SOLVER,
                           solver_version=solver_version_string(), tolerances=self.opts.to_dict(),
                           wall_time_s=time.perf_counter() - self.t0, input_hash=self.input_hash, mip_gap=None,
                           iterations=out.iterations, n_evaluations=None, message=out.message,
                           constraint_residuals=residuals, params=self.recorded_params(), streams_used=self.streams,
                           diagnostics=d, is_synthetic=self.is_synth)

    def solve_rows(self, A_p: np.ndarray, b_p: np.ndarray, ids_p: Sequence[str]):
        """LP with structural rows + the given (already robust) probabilistic rows, x-space."""
        A = np.vstack([self.A_s, A_p]) if A_p.size else self.A_s
        b = np.concatenate([self.b_s, b_p])
        eq = np.concatenate([self.eq_s, np.zeros(len(b_p), dtype=bool)])
        ids = tuple(self.cc_s.constraint_ids) + tuple(ids_p)
        lp = assemble_x_space_lp(A, b, eq, ids, self.dh, self.prices)
        out = run_linprog(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, lp.ub, self.opts)
        self.diag.update({"n_rows_ub": int(lp.A_ub.shape[0]), "n_rows_eq": int(lp.A_eq.shape[0])})
        return out


def _raise_if_failed(out) -> None:
    if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT):
        raise _Fail(out.status, out.message, {"raw_candidate_x": out.extra.get("raw_x"),
                                              "highs_model_status": out.highs_model_status})


# ================================================================================================
# M3a  box
# ================================================================================================

def solve_box_robust(problem: RationProblem, *, d_hat: Optional[np.ndarray] = None,
                     opt_draws: Optional[DrawSet] = None, params: Optional[Mapping[str, Any]] = None,
                     solver_options: Optional[SolverOptions] = None) -> SolveResult:
    """M3a box (interval) robust LP: every probabilistic row holds for every state in the box.

    Parameters (``params``)
    -----------------------
    uncertainty_set : :class:`BoxUncertaintySet` (source recorded), **or**
    k : build the box as opt-draw sample mean -/+ k SD (needs ``opt_draws``), **or**
    quantiles : ``(q_lo, q_hi)`` of the opt draws per coordinate (needs ``opt_draws``);
    information_state : label copied to the decision.
    """
    run = _Run(METHOD_ID_BOX, problem, params, _BOX_DEFAULTS, solver_options, opt_draws)
    try:
        run.check_params()
        run.setup(d_hat)
        box = run.resolve_box(opt_draws)
        A_max, _, b_p = run.box_rows(box)
        run.input_hash = stable_hash(METHOD_ID_BOX, "v1", problem.problem_id, run.ids, problem.nutrient_ids,
                                     run.cc.fingerprint, box.fingerprint(), run.dh, run.prices,
                                     run.hashable_params(), run.opts.to_dict())
        out = run.solve_rows(A_max, b_p, run.cc_p.constraint_ids)
        _raise_if_failed(out)
        q = out.x / run.dh
        worst = A_max @ q - b_p
        from . import equivalence_annotations   # local import (package imports modules lazily)
        diag = {"guarantee": "joint: every imposed probabilistic_nutrition row holds for every state in the box "
                             "(exact robust counterpart; LP feasibility tolerance only); no probability statement",
                "equivalent_methods": [r["pair_id"] for r in equivalence_annotations(METHOD_ID_BOX)],
                "robust_coefficients_q_space": {cid: A_max[k].tolist() for k, cid in enumerate(run.cc_p.constraint_ids)}}
        return run.success(out, out.x, worst, diag)
    except _Fail as f:
        return run.fail(f)


# ================================================================================================
# M3b  budget (Bertsimas & Sim 2004)
# ================================================================================================

def _parse_gamma(gamma, prob_ids: Sequence[str], n_unc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K = len(prob_ids)

    def one(v, where: str) -> float:
        if isinstance(v, str):
            if v == "full":
                return math.inf
            raise _Fail(SolveStatus.INVALID_INPUT, f"gamma{where}: unknown string {v!r} (use 'full' or a number)")
        try:
            g = float(v)
        except (TypeError, ValueError):
            raise _Fail(SolveStatus.INVALID_INPUT, f"gamma{where}: not a number: {v!r}") from None
        if math.isnan(g) or g < 0:
            raise _Fail(SolveStatus.INVALID_INPUT, f"gamma{where}: must be >= 0, got {v!r}")
        return g

    if gamma is None:
        raise _Fail(SolveStatus.INVALID_INPUT, "params['gamma'] is required (choose it on development data; "
                                               "0 = nominal, 'full' = box)")
    if isinstance(gamma, Mapping):
        extra = sorted(set(map(str, gamma)) - set(prob_ids))
        lacking = sorted(set(prob_ids) - set(map(str, gamma)))
        if extra or lacking:
            raise _Fail(SolveStatus.INVALID_INPUT, f"gamma mapping must cover exactly the probabilistic constraints; "
                                                   f"unknown {extra}, missing {lacking}")
        req = np.array([one(gamma[c], f"[{c}]") for c in prob_ids], dtype=float)
    else:
        req = np.full(K, one(gamma, ""), dtype=float)
    eff = np.minimum(req, n_unc.astype(float))
    return req, eff


def solve_budget_robust(problem: RationProblem, *, d_hat: Optional[np.ndarray] = None,
                        opt_draws: Optional[DrawSet] = None, params: Optional[Mapping[str, Any]] = None,
                        solver_options: Optional[SolverOptions] = None) -> SolveResult:
    """M3b budget robust LP (Bertsimas & Sim 2004, doi:10.1287/opre.1030.0065), LP-dual form.

    Parameters (``params``)
    -----------------------
    gamma : number ``>= 0``, ``"full"``, or a mapping ``{probabilistic constraint id: value}``
        (required; no default).  ``0`` -> M0 nominal row, ``>= |J_k|`` -> box row.
    uncertainty_set / k / quantiles : the box that defines ``Amax`` (as in M3a).
    nominal_mode : ``"nominal_point"`` (default; ``Abar`` = M0 row at table composition and
        ``d_hat``, so ``Gamma = 0`` reproduces M0) or ``"box_center"``.  The nominal state must lie
        in the box (checked on the coefficients; otherwise ``invalid_input``).
    information_state : label copied to the decision.
    """
    run = _Run(METHOD_ID_BUDGET, problem, params, _BUDGET_DEFAULTS, solver_options, opt_draws)
    try:
        run.check_params()
        run.setup(d_hat)
        box = run.resolve_box(opt_draws)
        A_max, _, b_p = run.box_rows(box)
        mode = run.p["nominal_mode"]
        if mode == "nominal_point":
            th_n, d_n = problem.nominal_theta(), run.dh
        elif mode == "box_center":
            th_n, d_n = box.center()
        else:
            raise _Fail(SolveStatus.INVALID_INPUT, f"unknown nominal_mode {mode!r}")
        rows_n = linear_rows(run.cc_p, th_n, d_n, d_hat=run.dh)
        A_bar = rows_n.A[0]
        if np.any(rows_n.missing[0]):
            bad = sorted({f"{run.cc_p.constraint_ids[k]}:{run.ids[i]}" for k, i in np.argwhere(np.isnan(A_bar))})
            raise _Fail(SolveStatus.INVALID_INPUT, "missing nominal coefficients (constraint:ingredient) " + ", ".join(bad))
        A_hat = A_max - A_bar
        tol = _NOMINAL_IN_BOX_RTOL * (1.0 + np.abs(A_max) + np.abs(A_bar))
        if np.any(A_hat < -tol):
            bad = sorted({f"{run.cc_p.constraint_ids[k]}:{run.ids[i]}" for k, i in np.argwhere(A_hat < -tol)})
            raise _Fail(SolveStatus.INVALID_INPUT,
                        f"nominal state ({mode}) is outside the box for (constraint:ingredient) " + ", ".join(bad))
        A_hat = np.where(A_hat > tol, A_hat, 0.0)          # |J_k| counts genuinely uncertain entries only
        unc = A_hat > 0
        n_unc = unc.sum(axis=1)
        g_req, g_eff = _parse_gamma(run.p["gamma"], run.cc_p.constraint_ids, n_unc)
        run.input_hash = stable_hash(METHOD_ID_BUDGET, "v1", problem.problem_id, run.ids, problem.nutrient_ids,
                                     run.cc.fingerprint, box.fingerprint(), run.dh, run.prices, A_bar, g_eff,
                                     run.hashable_params(), run.opts.to_dict())

        # ---- LP in x-space: variables [x (I) | z (Kp) | p (nJ)] -----------------------------
        I, Kp = len(run.ids), len(b_p)
        pairs = [(k, i) for k in range(Kp) for i in range(I) if unc[k, i]]
        nJ = len(pairs)
        n = I + Kp + nJ
        Abar_x = A_bar / run.dh[None, :]
        Ahat_x = A_hat / run.dh[None, :]
        As_x = run.A_s / run.dh[None, :]
        ub_rows, ub_rhs, eq_rows, eq_rhs = [], [], [], []
        for r, (a, bb, e) in enumerate(zip(As_x, run.b_s, run.eq_s)):
            row = np.zeros(n)
            row[:I] = a
            (eq_rows if e else ub_rows).append(row)
            (eq_rhs if e else ub_rhs).append(bb)
        col_p = {pk: I + Kp + m for m, pk in enumerate(pairs)}
        for k in range(Kp):
            row = np.zeros(n)
            row[:I] = Abar_x[k]
            row[I + k] = g_eff[k]
            for i in range(I):
                if unc[k, i]:
                    row[col_p[(k, i)]] = 1.0
            ub_rows.append(row)
            ub_rhs.append(b_p[k])
        for (k, i) in pairs:
            row = np.zeros(n)
            row[i] = Ahat_x[k, i]
            row[I + k] = -1.0
            row[col_p[(k, i)]] = -1.0
            ub_rows.append(row)
            ub_rhs.append(0.0)
        c = np.concatenate([run.prices / run.dh, np.zeros(Kp + nJ)])
        A_ub = np.array(ub_rows).reshape(-1, n)
        A_eq = np.array(eq_rows).reshape(-1, n)
        out = run_linprog(c, A_ub, np.array(ub_rhs, float), A_eq, np.array(eq_rhs, float),
                          np.zeros(n), np.full(n, np.inf), run.opts)
        run.diag.update({"n_rows_ub": int(A_ub.shape[0]), "n_rows_eq": int(A_eq.shape[0]),
                         "n_aux_variables": int(Kp + nJ)})
        _raise_if_failed(out)
        q = out.x[:I] / run.dh
        beta = np.array([budget_protection(A_hat[k], q, g_eff[k]) for k in range(Kp)])
        resid = A_bar @ q + beta - b_p
        pid = run.cc_p.constraint_ids
        diag = {"guarantee": "row-wise: row k holds whenever at most floor(Gamma_k) of its ingredient coefficients sit "
                             "at their box worst case and one more moves a fraction; not a joint guarantee; "
                             "Bertsimas-Sim probability bounds NOT claimed (non-symmetric/bilinear/possibly "
                             "correlated perturbations)",
                "nominal_mode": mode,
                "gamma_requested": {cid: (None if math.isinf(g) else float(g)) for cid, g in zip(pid, g_req)},
                "gamma_requested_is_full": {cid: bool(math.isinf(g)) for cid, g in zip(pid, g_req)},
                "gamma_effective": {cid: float(g) for cid, g in zip(pid, g_eff)},
                "n_uncertain_coefficients": {cid: int(v) for cid, v in zip(pid, n_unc)},
                "protection_q_space": {cid: float(v) for cid, v in zip(pid, beta)},
                "nominal_residual_q_space": {cid: float(v) for cid, v in zip(pid, A_bar @ q - b_p)},
                "citation": "Bertsimas D, Sim M (2004) The Price of Robustness. Oper Res 52(1):35-53. "
                            "doi:10.1287/opre.1030.0065"}
        return run.success(out, out.x, resid, diag)
    except _Fail as f:
        return run.fail(f)


# ================================================================================================
# M3c  scenario-set
# ================================================================================================

def solve_scenario_set_robust(problem: RationProblem, *, d_hat: Optional[np.ndarray] = None,
                              opt_draws: Optional[DrawSet] = None, params: Optional[Mapping[str, Any]] = None,
                              solver_options: Optional[SolverOptions] = None) -> SolveResult:
    """M3c scenario-set robust LP: every probabilistic row holds in each of the N ``opt`` states.

    ``opt_draws`` (``opt`` stream only) is the finite scenario set.  The guarantee is in-sample
    only (no continuous interval, no out-of-sample claim).  Equals joint-chance SAA with
    ``floor(alpha N) = 0`` on the same states.
    """
    run = _Run(METHOD_ID_SCENARIO_SET, problem, params, _SCEN_DEFAULTS, solver_options, opt_draws)
    try:
        run.check_params()
        run.setup(d_hat)
        if opt_draws is None:
            raise _Fail(SolveStatus.INVALID_INPUT, "scenario-set robust needs opt_draws (the finite scenario set)")
        require_stream(opt_draws, _OPT_ONLY, METHOD_ID_SCENARIO_SET)  # LeakageError, never a status
        if opt_draws.ingredient_ids != run.ids or opt_draws.nutrient_ids != problem.nutrient_ids:
            raise _Fail(SolveStatus.INVALID_INPUT, "opt_draws labels differ from the problem")
        run.streams = (opt_draws.stream_id,)
        rows = linear_rows(run.cc_p, opt_draws.theta, opt_draws.d, d_hat=run.dh)
        if np.any(rows.missing):
            ks = sorted({run.cc_p.constraint_ids[k] for k in np.argwhere(rows.missing)[:, 1]})
            n_bad = int(rows.missing.any(axis=1).sum())
            raise _Fail(SolveStatus.INVALID_INPUT,
                        f"missing composition/DM in {n_bad} scenario(s) for constraints {ks}")
        S, Kp, I = rows.A.shape
        run.input_hash = stable_hash(METHOD_ID_SCENARIO_SET, "v1", problem.problem_id, run.ids, problem.nutrient_ids,
                                     run.cc.fingerprint, opt_draws.fingerprint, run.dh, run.prices,
                                     run.hashable_params(), run.opts.to_dict())
        A_p = rows.A.reshape(S * Kp, I)
        b_p = np.tile(rows.b, S)
        ids_p = [f"{cid}@s{s}" for s in range(S) for cid in run.cc_p.constraint_ids]
        out = run.solve_rows(A_p, b_p, ids_p)
        _raise_if_failed(out)
        q = out.x / run.dh
        g = rows.residual(q)                              # [S, Kp]
        worst = g.max(axis=0) if S else np.zeros(Kp)
        from . import equivalence_annotations   # local import (package imports modules lazily)
        diag = {"guarantee": "in-sample only: every imposed probabilistic_nutrition row holds in each of the N listed "
                             "opt states; no continuous-set or out-of-sample guarantee",
                "equivalent_methods": [r["pair_id"] for r in equivalence_annotations(METHOD_ID_SCENARIO_SET)],
                "n_scenarios": int(S), "scenario_stream_id": opt_draws.stream_id,
                "scenario_fingerprint": opt_draws.fingerprint,
                "n_binding_scenarios_per_constraint": {
                    cid: int(np.sum(g[:, k] > -1e-9 * (1.0 + np.abs(rows.A[:, k, :]) @ np.abs(q))))
                    for k, cid in enumerate(run.cc_p.constraint_ids)}}
        return run.success(out, out.x, worst, diag)
    except _Fail as f:
        return run.fail(f)
