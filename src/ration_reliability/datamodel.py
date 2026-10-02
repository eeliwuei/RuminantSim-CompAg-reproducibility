"""Core data structures of the engine (contract section 8; evidence_contract.yaml).

All numeric fields that enter computation are stored in *canonical* units
(see :mod:`ration_reliability.normalization.units`):

* composition ``a_ij``: kg/kg DM (``fraction``) for mass nutrients, Mcal/kg DM for energy;
* DM fraction ``d_i``: kg DM / kg as-fed;
* decision ``q_i``: kg as-fed / head / day;  internal DM formula ``x_i = q_i * d_hat_i``;
* price ``p_i``: currency / kg as-fed.

Provenance is carried by :class:`Provenance`.  A value without a source must not be invented:
it is either ``None`` with status ``pending_user_decision`` or an explicit
``research_scenario_assumption`` with a written rationale (task hard rule 3).
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .errors import InvalidProblemError
from .normalization import units as U

__all__ = [
    "ValueStatus",
    "ConstraintClass",
    "ConstraintKind",
    "Sense",
    "DMSource",
    "SolveStatus",
    "NUTRIENT_DIMENSIONS",
    "Provenance",
    "SourcedValue",
    "NutrientSpec",
    "IngredientRecord",
    "ObservationRecord",
    "AnimalProfile",
    "ConstraintSpec",
    "PriceScenario",
    "RationDecision",
    "SolverOptions",
    "SolveResult",
    "EvaluationResult",
    "AssaySpec",
    "InformationPolicy",
    "RationProblem",
]


# --------------------------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------------------------

class _StrEnum(str, enum.Enum):
    """String enum whose ``str()``/``format()`` is the plain value (stable in logs and JSON)."""

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, spec: str) -> str:
        return format(str(self.value), spec)


class ValueStatus(_StrEnum):
    """Provenance status of a value."""

    SOURCED = "sourced"
    RESEARCH_SCENARIO_ASSUMPTION = "research_scenario_assumption"
    PENDING_USER_DECISION = "pending_user_decision"
    SYNTHETIC_TEST_ONLY = "synthetic_test_only"


class ConstraintClass(_StrEnum):
    """Constraint category (contract section 6.2)."""

    STRUCTURAL_HARD = "structural_hard"
    PROBABILISTIC_NUTRITION = "probabilistic_nutrition"
    DIAGNOSTIC_ONLY = "diagnostic_only"


class ConstraintKind(_StrEnum):
    """How the constraint expression is compared with its bound.

    * ``concentration``: ``E = S(q, theta) / D(q, theta)`` (per kg DM), linearised as
      ``K D - S <= 0`` (ge) or ``S - K D <= 0`` (le) -- contract T2.
    * ``supply``: ``E = S(q, theta)`` per head per day (kg/d or Mcal/d).
    * ``as_fed``: ``E = sum_i v_i q_i`` (kg as-fed/head/d); deterministic in q.
    """

    CONCENTRATION = "concentration"
    SUPPLY = "supply"
    AS_FED = "as_fed"


class Sense(_StrEnum):
    """Comparison sense ``E (sense) K``.  ``eq`` is only allowed for structural constraints."""

    GE = "ge"
    LE = "le"
    EQ = "eq"


class DMSource(_StrEnum):
    """Which DM fraction a constraint uses.

    * ``scenario``: the (uncertain) scenario DM ``d_s`` -- required for probabilistic and
      diagnostic constraints (T2.3: never check a random ration with nominal inclusion rates).
    * ``decision_estimate``: the decision-time estimate ``d_hat`` recorded in the
      :class:`RationDecision`; only for structural constraints that describe the *planned* ration
      (e.g. planned DM offered, planned DM inclusion share).  ``d_hat`` is known at decision time,
      it is not the hidden true DM.
    """

    SCENARIO = "scenario"
    DECISION_ESTIMATE = "decision_estimate"


class SolveStatus(_StrEnum):
    """Solver outcome (contract section 8).  A time-out without a solution is *not* infeasibility."""

    OPTIMAL = "optimal"
    FEASIBLE_TIME_LIMIT = "feasible_time_limit"
    PROVEN_INFEASIBLE = "proven_infeasible"
    NO_FEASIBLE_SOLUTION_FOUND = "no_feasible_solution_found"
    NUMERICAL_ERROR = "numerical_error"
    INVALID_INPUT = "invalid_input"


#: Allowed nutrient dimensions and their canonical units.
NUTRIENT_DIMENSIONS: dict[str, str] = {
    "mass_fraction": "fraction",
    "energy_density": "Mcal/kg",
}


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """Where a value comes from.

    Attributes
    ----------
    status:
        One of :class:`ValueStatus`.
    source_id:
        Id in the source registry (required for ``sourced`` and ``synthetic_test_only``).
    locator:
        Table / page / section / row (required for ``sourced``).
    rationale:
        Written justification (required for ``research_scenario_assumption`` and
        ``pending_user_decision``).
    """

    status: ValueStatus
    source_id: Optional[str] = None
    locator: Optional[str] = None
    rationale: Optional[str] = None

    def issues(self, where: str) -> list[str]:
        """Return provenance rule violations (empty list if OK)."""
        out: list[str] = []
        st = self.status
        if not isinstance(st, ValueStatus):
            return [f"{where}: provenance status {st!r} is not a ValueStatus"]
        if st is ValueStatus.SOURCED:
            if not self.source_id:
                out.append(f"{where}: status 'sourced' requires source_id")
            if not self.locator:
                out.append(f"{where}: status 'sourced' requires a locator (table/page/section)")
        elif st is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION:
            if not self.rationale:
                out.append(f"{where}: research_scenario_assumption requires an explicit rationale")
        elif st is ValueStatus.PENDING_USER_DECISION:
            if not self.rationale:
                out.append(f"{where}: pending_user_decision requires a rationale (what must be decided)")
        elif st is ValueStatus.SYNTHETIC_TEST_ONLY:
            if not self.source_id:
                out.append(f"{where}: synthetic_test_only requires the synthetic source_id")
        return out


@dataclass(frozen=True)
class SourcedValue:
    """A scalar with unit, basis and provenance.  ``value is None`` only if pending."""

    value: Optional[float]
    unit: str
    provenance: Provenance
    basis: str = "none"

    def issues(self, where: str) -> list[str]:
        """Unit / basis / provenance / null-value rule violations."""
        out: list[str] = []
        try:
            U.get_unit(self.unit)
        except Exception as exc:  # UnitError
            out.append(f"{where}: {exc}")
        if self.basis not in U.BASES:
            out.append(f"{where}: undefined basis {self.basis!r}")
        out.extend(self.provenance.issues(where))
        pending = self.provenance.status is ValueStatus.PENDING_USER_DECISION
        if self.value is None and not pending:
            out.append(f"{where}: value is null but status is not pending_user_decision")
        if self.value is not None and pending:
            out.append(f"{where}: pending_user_decision must have value null (no placeholder numbers)")
        if self.value is not None and not np.isfinite(float(self.value)):
            out.append(f"{where}: value must be finite")
        return out

    def canonical(self) -> Optional[float]:
        """Value in the canonical unit of its dimension (``None`` if pending)."""
        if self.value is None:
            return None
        v, _ = U.to_canonical(float(self.value), self.unit)
        return float(v)


# --------------------------------------------------------------------------------------------
# problem data
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NutrientSpec:
    """A nutrient (composition column ``j``).

    ``dimension`` is ``mass_fraction`` (canonical kg/kg DM) or ``energy_density``
    (canonical Mcal/kg DM).  Nutrient ids must not contain ``':'`` and must not be ``DM``/``AF``.
    """

    nutrient_id: str
    dimension: str = "mass_fraction"
    name: str = ""

    def __post_init__(self) -> None:
        if self.dimension not in NUTRIENT_DIMENSIONS:
            raise InvalidProblemError(f"nutrient {self.nutrient_id}: unknown dimension {self.dimension!r}")
        if not self.nutrient_id or ":" in self.nutrient_id or self.nutrient_id in ("DM", "AF"):
            raise InvalidProblemError(f"illegal nutrient id {self.nutrient_id!r}")

    @property
    def canonical_unit(self) -> str:
        """Canonical per-kg-DM unit of this nutrient."""
        return NUTRIENT_DIMENSIONS[self.dimension]


@dataclass(frozen=True)
class IngredientRecord:
    """One ingredient (row ``i``) with decision-time nominal information.

    Attributes
    ----------
    ingredient_id:
        Stable id (no ``':'``).
    name_en, name_zh, original_label:
        Names are kept separate; ``original_label`` is the label in the source (contract 7.1).
    category:
        Free label, e.g. ``forage`` / ``energy_concentrate`` / ``protein_concentrate`` /
        ``mineral`` / ``additive``.
    group_weights:
        Membership weights in [0, 1] for grouped terms, e.g. ``{"forage": 1.0}`` so that
        ``G:forage:NDF`` gives forage NDF.  Must be explicit in the input (no name-based guess).
    coefficients:
        Named per-ingredient real coefficients used by ``C:<name>:<nutrient>`` terms, e.g. an
        absorption coefficient ``{"AC_Ca": 0.6}`` for an absorbed-Ca supply.  Each needs its own
        provenance (key ``"coefficient:<name>"``); a term referring to a coefficient that an
        ingredient lacks is a compile error (never a silent 0).
    dm_estimate:
        Decision-time DM estimate ``d_hat_i`` (kg DM/kg as-fed) used to convert x -> q.
    composition:
        Nominal composition on DM basis, canonical units; ``NaN`` = missing (never 0).
    is_stochastic:
        False for deterministic inputs (e.g. minerals given as point values).
    is_synthetic:
        True for test-only data.
    lineage_id, external_ids, provenance, notes:
        Traceability fields (``provenance`` keys: ``"dm_estimate"``, ``"composition:<j>"``...).
    """

    ingredient_id: str
    name_en: str
    dm_estimate: float
    composition: Mapping[str, float]
    category: str = "unspecified"
    group_weights: Mapping[str, float] = field(default_factory=dict)
    coefficients: Mapping[str, float] = field(default_factory=dict)
    name_zh: Optional[str] = None
    original_label: Optional[str] = None
    is_stochastic: bool = True
    is_synthetic: bool = False
    lineage_id: Optional[str] = None
    external_ids: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Provenance] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.ingredient_id or ":" in self.ingredient_id:
            raise InvalidProblemError(f"illegal ingredient id {self.ingredient_id!r}")
        d = float(self.dm_estimate)
        if not (np.isfinite(d) and 0.0 < d <= 1.0):
            raise InvalidProblemError(f"{self.ingredient_id}: dm_estimate must be in (0, 1], got {d}")
        for g, w in self.group_weights.items():
            if ":" in g or not (0.0 <= float(w) <= 1.0):
                raise InvalidProblemError(f"{self.ingredient_id}: group weight {g}={w} must be in [0,1]")
        for c, w in self.coefficients.items():
            if ":" in c or not np.isfinite(float(w)):
                raise InvalidProblemError(f"{self.ingredient_id}: coefficient {c}={w} must be finite, no ':'")


@dataclass(frozen=True)
class ObservationRecord:
    """One observation / summary statistic from a source (evidence_contract.yaml).

    Optional fields must stay ``None`` unless present in the source (``do_not_fabricate``).
    ``statistic_type`` examples: ``single_sample``, ``mean``, ``sd``, ``n``, ``min``, ``max``,
    ``p10``, ``p90``.
    """

    observation_id: str
    ingredient_id: str
    source_id: str
    original_label: str
    nutrient_id: str
    value: float
    unit: str
    basis: str
    statistic_type: str
    is_measured_or_derived: str
    is_synthetic: bool
    batch_id: Optional[str] = None
    sampled_at: Optional[str] = None
    analytical_method: Optional[str] = None
    lab_id: Optional[str] = None
    original_sample_id: Optional[str] = None
    n: Optional[int] = None
    sd: Optional[float] = None
    covariance_group: Optional[str] = None

    def issues(self) -> list[str]:
        """Unit/basis/enum checks."""
        out: list[str] = []
        where = f"observation {self.observation_id}"
        try:
            U.get_unit(self.unit)
        except Exception as exc:
            out.append(f"{where}: {exc}")
        if self.basis not in U.BASES:
            out.append(f"{where}: undefined basis {self.basis!r}")
        if self.is_measured_or_derived not in ("measured", "derived", "equation", "unknown"):
            out.append(f"{where}: is_measured_or_derived must be measured/derived/equation/unknown")
        return out


@dataclass(frozen=True)
class AnimalProfile:
    """Reference animal / scenario profile (contract 6.1).

    ``attributes`` holds sourced values such as ``body_weight``, ``days_in_milk``,
    ``milk_yield``, ``dmi`` (a fixed scenario intake, e.g. from an equation named in
    ``dmi_equation``).  A set production level is *not* a predicted milk yield of the optimised
    ration.
    """

    profile_id: str
    species: str
    animal_stage: str
    attributes: Mapping[str, SourcedValue] = field(default_factory=dict)
    nutrition_standard_and_version: Optional[str] = None
    dmi_equation: Optional[str] = None
    is_synthetic: bool = False
    notes: str = ""

    def get(self, name: str) -> SourcedValue:
        """Return attribute ``name`` or raise ``KeyError``."""
        return self.attributes[name]

    @property
    def dmi_kg_per_day(self) -> Optional[float]:
        """Scenario DMI in kg DM/head/d (canonical), ``None`` if absent or pending."""
        sv = self.attributes.get("dmi")
        if sv is None:
            return None
        U.check_unit(sv.unit, "mass_rate")
        return sv.canonical()


@dataclass(frozen=True)
class ConstraintSpec:
    """One constraint, in natural units, with provenance (contract 6.2; evidence_contract).

    Expression grammar (``terms`` maps term key -> coefficient):

    ============================  ===========================================================
    key                           contribution to S(q, theta)
    ============================  ===========================================================
    ``<nutrient>``                ``sum_i q_i d_i * a_ij``          (all ingredients)
    ``G:<group>:<nutrient>``      ``sum_i q_i d_i * w_ig * a_ij``   (group-weighted, e.g. forage NDF)
    ``DM``                        ``sum_i q_i d_i``                 (DM mass)
    ``DM:<ingredient>``           ``q_k d_k``
    ``G:<group>:DM``              ``sum_i q_i d_i * w_ig``
    ``C:<coef>:<nutrient>``       ``sum_i q_i d_i * c_i * a_ij``   (named per-ingredient coefficient)
    ``C:<coef>:DM``               ``sum_i q_i d_i * c_i``
    ``AF``                        ``sum_i q_i``                     (as-fed mass; kind ``as_fed`` only)
    ``AF:<ingredient>``           ``q_k``
    ============================  ===========================================================

    Examples (thresholds must come from a cited source or be flagged as assumptions):

    * CP lower concentration: ``kind=concentration, terms={"CP": 1}, sense=ge, unit="%", basis=DM``.
    * NASEM-2021-Table-5-1-*form* rule ``NDF + 2 fNDF >= K``:
      ``terms={"NDF": 1, "G:forage:NDF": 2}``, ``sense=ge``.
    * starch <= 2 fNDF - K2: ``terms={"starch": 1, "G:forage:NDF": -2}``, ``sense=le``, bound ``-K2``.
    * planned DM share of ingredient k <= u (structural, uses d_hat):
      ``kind=concentration, terms={"DM:k": 1}, dm_source=decision_estimate``.
    * inventory ``H T q_k <= B_k``: ``kind=as_fed, terms={"AF:k": 1}, sense=le, bound=B_k/(H T)``.
    * absorbed mineral supply ``sum_i x_i a_i AC_i >= R0 + r1 * D`` (``r1`` in kg per kg DM):
      ``kind=supply, terms={"C:AC_Ca:Ca": 1, "DM": -r1}, sense=ge, bound=R0`` (unit e.g. ``g/d``).

    Coefficients multiply canonical quantities (kg/d or Mcal/d), so a ``DM`` coefficient in a
    supply expression must itself be in kg (or Mcal) per kg DM.

    ``numerical_tolerance`` is in ``unit`` and must be > 0 (compile rejects 0).  A constraint is *violated* in a
    scenario when its deficit in natural units exceeds the tolerance.
    """

    constraint_id: str
    name: str
    kind: ConstraintKind
    terms: Mapping[str, float]
    sense: Sense
    bound: Optional[float]
    unit: str
    constraint_class: ConstraintClass
    numerical_tolerance: float
    provenance: Provenance
    basis: str = "DM"
    dm_source: DMSource = DMSource.SCENARIO
    animal_stage: Optional[str] = None
    standard_version: Optional[str] = None
    claim_scope: Optional[str] = None
    quantity_type: str = "model_quantity"
    notes: str = ""

    @property
    def lower_or_upper(self) -> str:
        """``lower`` / ``upper`` / ``equality`` (evidence_contract field)."""
        return {Sense.GE: "lower", Sense.LE: "upper", Sense.EQ: "equality"}[Sense(self.sense)]

    def expression_str(self) -> str:
        """Human-readable expression, e.g. ``conc[NDF + 2*G:forage:NDF] >= 60 %``."""
        parts = []
        for k, c in self.terms.items():
            c = float(c)
            parts.append(k if c == 1.0 else f"{c:g}*{k}")
        op = {Sense.GE: ">=", Sense.LE: "<=", Sense.EQ: "=="}[Sense(self.sense)]
        return f"{ConstraintKind(self.kind).value}[{' + '.join(parts)}] {op} {self.bound} {self.unit}"


@dataclass(frozen=True)
class PriceScenario:
    """Ingredient prices per kg **as-fed** in one currency (T2.2).

    ``is_scenario=True`` means the prices are a declared research scenario, not a dated market
    series; results must then be phrased as "cost difference under the scenario prices".
    ``conversion_log`` records every basis/unit conversion applied at load time (e.g. per tonne
    -> per kg; per kg DM -> per kg as-fed with the stated DM semantics).  Prices never change
    with test scenarios.
    """

    price_id: str
    currency: str
    prices_per_kg_as_fed: Mapping[str, float]
    is_scenario: bool
    is_synthetic: bool
    price_year_or_date: Optional[str] = None
    provenance: Mapping[str, Provenance] = field(default_factory=dict)
    conversion_log: tuple[str, ...] = ()

    def vector(self, ingredient_ids: Sequence[str]) -> np.ndarray:
        """Price vector aligned with ``ingredient_ids`` (raises if a price is missing)."""
        missing = [i for i in ingredient_ids if i not in self.prices_per_kg_as_fed]
        if missing:
            raise InvalidProblemError(f"price scenario {self.price_id}: missing prices for {missing}")
        p = np.array([float(self.prices_per_kg_as_fed[i]) for i in ingredient_ids], dtype=float)
        if not np.all(np.isfinite(p)):
            raise InvalidProblemError(f"price scenario {self.price_id}: non-finite price")
        return p


@dataclass(frozen=True)
class RationDecision:
    """An executable ration: as-fed amounts plus the DM estimates used to derive them.

    ``q_as_fed`` (kg as-fed/head/d) is the executed decision.  ``x_planned_dm = q * d_hat`` is
    the *planned* DM formula.  The realised DM formula in a scenario is ``q * d_true`` and is
    never renormalised by the evaluator (T2.1).
    """

    ingredient_ids: tuple[str, ...]
    q_as_fed: np.ndarray
    d_hat: np.ndarray
    method_id: str
    information_state: str = "t0_reference_only"
    x_planned_dm: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        q = np.array(self.q_as_fed, dtype=float).copy()
        dh = np.array(self.d_hat, dtype=float).copy()
        ids = tuple(self.ingredient_ids)
        if q.shape != (len(ids),) or dh.shape != (len(ids),):
            raise InvalidProblemError("RationDecision: q_as_fed and d_hat must have shape (I,)")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dh))):
            raise InvalidProblemError("RationDecision: non-finite q or d_hat")
        if np.any(dh <= 0) or np.any(dh > 1):
            raise InvalidProblemError("RationDecision: d_hat must be in (0, 1]")
        x = q * dh
        for a in (q, dh, x):
            a.setflags(write=False)
        object.__setattr__(self, "ingredient_ids", ids)
        object.__setattr__(self, "q_as_fed", q)
        object.__setattr__(self, "d_hat", dh)
        object.__setattr__(self, "x_planned_dm", x)

    def reordered(self, ingredient_ids: Sequence[str]) -> "RationDecision":
        """Same decision with ingredients in another order."""
        idx = [self.ingredient_ids.index(i) for i in ingredient_ids]
        return RationDecision(tuple(ingredient_ids), self.q_as_fed[idx], self.d_hat[idx],
                              self.method_id, self.information_state)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict."""
        return {
            "ingredient_ids": list(self.ingredient_ids),
            "q_as_fed_kg_per_head_per_day": self.q_as_fed.tolist(),
            "x_planned_kg_dm_per_head_per_day": self.x_planned_dm.tolist(),
            "d_hat_kg_dm_per_kg_as_fed": self.d_hat.tolist(),
            "method_id": self.method_id,
            "information_state": self.information_state,
        }


@dataclass(frozen=True)
class SolverOptions:
    """Solver settings recorded in every :class:`SolveResult`.

    Defaults equal the HiGHS defaults for feasibility tolerances (1e-7); ``residual_check_rel_tol``
    is the engine's own post-solve check on the returned candidate.
    """

    time_limit_s: Optional[float] = None
    presolve: bool = True
    primal_feasibility_tolerance: float = 1e-7
    dual_feasibility_tolerance: float = 1e-7
    mip_rel_gap: Optional[float] = None
    residual_check_rel_tol: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SolveResult:
    """Outcome of one method call (contract section 8).

    Invariants
    ----------
    * ``decision`` and ``objective`` are set **only** for ``optimal`` / ``feasible_time_limit``.
      An infeasible or failed solve never carries an empty ration or a zero cost.
    * No automatic relaxation: the solved problem is exactly the input problem.
    * ``constraint_residuals`` maps constraint id -> linearised residual ``g(q)`` of the model the
      solver used (``<= 0`` satisfied for ge/le; ``abs`` value for eq).
    """

    method_id: str
    status: SolveStatus
    decision: Optional[RationDecision]
    objective: Optional[float]
    objective_unit: str
    solver: str
    solver_version: str
    tolerances: Mapping[str, Any]
    wall_time_s: float
    input_hash: str
    mip_gap: Optional[float] = None
    iterations: Optional[int] = None
    n_evaluations: Optional[int] = None
    message: str = ""
    constraint_residuals: Mapping[str, float] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    streams_used: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        st = SolveStatus(self.status)
        object.__setattr__(self, "status", st)
        has = st in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT)
        if has and (self.decision is None or self.objective is None):
            raise InvalidProblemError(f"SolveResult: status {st.value} requires decision and objective")
        if not has and (self.decision is not None or self.objective is not None):
            raise InvalidProblemError(
                f"SolveResult: status {st.value} must not carry a decision/objective (no empty ration, no zero cost)")

    @property
    def has_solution(self) -> bool:
        """True for ``optimal`` and ``feasible_time_limit``."""
        return self.decision is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict."""
        def _clean(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, enum.Enum):
                return v.value
            if isinstance(v, Mapping):
                return {str(k): _clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return v

        return {
            "method_id": self.method_id,
            "status": self.status.value,
            "decision": None if self.decision is None else self.decision.to_dict(),
            "objective": self.objective,
            "objective_unit": self.objective_unit,
            "solver": self.solver,
            "solver_version": self.solver_version,
            "tolerances": _clean(dict(self.tolerances)),
            "wall_time_s": self.wall_time_s,
            "mip_gap": self.mip_gap,
            "iterations": self.iterations,
            "n_evaluations": self.n_evaluations,
            "message": self.message,
            "constraint_residuals": _clean(dict(self.constraint_residuals)),
            "input_hash": self.input_hash,
            "params": _clean(dict(self.params)),
            "streams_used": list(self.streams_used),
            "diagnostics": _clean(dict(self.diagnostics)),
            "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Output of the single public evaluator (T3).  Arrays are read-only.

    Shapes: ``S`` = number of draws, ``K`` = evaluated (probabilistic + diagnostic) constraints,
    ``Ks`` = structural constraints, ``J`` = nutrients.

    * ``margin[s, k]``: signed slack in the constraint's declared unit (``> 0`` satisfied;
      for ``eq`` it is ``-|E - K|``); ``NaN`` where undefined.
    * ``violation_amount = max(0, -margin)`` in declared units (0 where undefined).
    * ``violated[s, k] = (-margin > tolerance)``; False where undefined.
    * ``undefined[s, k]``: missing data used (``missing_data``) or ``D <= 0`` for a ratio.
    * ``joint_violation[s]``: at least one ``probabilistic_nutrition`` constraint violated (I).
    * ``joint_unknown[s]``: no probabilistic constraint violated but at least one undefined.
      Missing data is a data-quality flag, never counted as a nutrient failure (T3).
    * structural constraints are checked deterministically, independent of the risk budget.
    """

    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    q_as_fed: np.ndarray
    q_hash: str
    constraint_ids: tuple[str, ...]
    constraint_classes: tuple[str, ...]
    constraint_units: tuple[str, ...]
    tolerances: np.ndarray
    margin: np.ndarray
    violation_amount: np.ndarray
    violated: np.ndarray
    undefined: np.ndarray
    missing_data: np.ndarray
    joint_violation: np.ndarray
    joint_unknown: np.ndarray
    data_quality_flag: np.ndarray
    dm_supply: np.ndarray
    nutrient_supply: np.ndarray
    structural_ids: tuple[str, ...]
    structural_margin: np.ndarray
    structural_violated: np.ndarray
    nonnegativity_ok: bool
    cost: Optional[float]
    n_draws: int
    draw_stream_id: Optional[str] = None
    is_synthetic: bool = False

    @property
    def structural_ok(self) -> bool:
        """All structural constraints (and q >= 0) satisfied within tolerance."""
        return bool(self.nonnegativity_ok and not np.any(self.structural_violated))

    @property
    def probabilistic_mask(self) -> np.ndarray:
        """Boolean mask of ``probabilistic_nutrition`` columns."""
        return np.array([c == ConstraintClass.PROBABILISTIC_NUTRITION.value for c in self.constraint_classes],
                        dtype=bool)

    def joint_counts(self) -> dict[str, int]:
        """Counts of violated / unknown / satisfied draws for the joint event I."""
        nv = int(self.joint_violation.sum())
        nu = int(self.joint_unknown.sum())
        return {"n_draws": self.n_draws, "n_violated": nv, "n_unknown": nu,
                "n_satisfied": self.n_draws - nv - nu}

    def summary(self, confidence: float = 0.95) -> dict[str, Any]:
        """Per-constraint and joint summaries (natural units).

        The Clopper-Pearson interval is the Monte Carlo interval for the *fixed* declared
        distribution only (T8.2/T8.3); it is not a data-uncertainty interval.
        """
        from .evaluation.stats import clopper_pearson  # local import (avoid cycle)

        per: dict[str, Any] = {}
        for k, cid in enumerate(self.constraint_ids):
            defined = ~self.undefined[:, k]
            nd = int(defined.sum())
            viol = self.violated[:, k]
            nv = int(viol.sum())
            m = self.margin[defined, k]
            amt = self.violation_amount[viol, k]
            per[cid] = {
                "class": self.constraint_classes[k],
                "unit": self.constraint_units[k],
                "tolerance": float(self.tolerances[k]),
                "n_defined": nd,
                "n_undefined": int(self.n_draws - nd),
                "n_violated": nv,
                "violation_rate_among_defined": (nv / nd) if nd else None,
                "mean_deficit_given_violation": float(amt.mean()) if nv else 0.0,
                "max_deficit": float(amt.max()) if nv else 0.0,
                "margin_quantiles": ({q: float(np.quantile(m, q)) for q in (0.01, 0.05, 0.5)} if nd else None),
            }
        jc = self.joint_counts()
        n_eval = jc["n_violated"] + jc["n_satisfied"]
        joint = dict(jc)
        joint["rate_lower"] = jc["n_violated"] / self.n_draws if self.n_draws else None
        joint["rate_upper"] = (jc["n_violated"] + jc["n_unknown"]) / self.n_draws if self.n_draws else None
        joint["rate_among_evaluable"] = (jc["n_violated"] / n_eval) if n_eval else None
        if n_eval:
            lo, hi = clopper_pearson(jc["n_violated"], n_eval, confidence)
            joint["mc_clopper_pearson"] = {"confidence": confidence, "lower": lo, "upper": hi,
                                           "interval_type": "MC_only_fixed_distribution"}
        return {
            "n_draws": self.n_draws,
            "draw_stream_id": self.draw_stream_id,
            "cost_per_head_per_day": self.cost,
            "structural_ok": self.structural_ok,
            "structural_violations": [c for c, v in zip(self.structural_ids, self.structural_violated) if v],
            "joint": joint,
            "per_constraint": per,
            "n_data_quality_flagged": int(self.data_quality_flag.sum()),
            "is_synthetic": self.is_synthetic,
        }


# --------------------------------------------------------------------------------------------
# information containers (algorithms are planned for a later stage)
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AssaySpec:
    """A purchasable assay / panel.  Unknown numbers stay ``None`` (no invented lab errors/costs).

    ``cost_per_sample`` is charged per sample/panel (contract 12.3), not per nutrient.
    Error components follow T7.5: ``Z = theta_batch + e_sampling + e_lab + b_systematic``.
    """

    assay_id: str
    name: str
    nutrient_ids: tuple[str, ...]
    applicable_ingredient_ids: tuple[str, ...] = ()
    measures_dm: bool = False
    cost_per_sample: Optional[SourcedValue] = None
    turnaround_days: Optional[SourcedValue] = None
    sampling_error_sd: Mapping[str, SourcedValue] = field(default_factory=dict)
    lab_error_sd: Mapping[str, SourcedValue] = field(default_factory=dict)
    systematic_bias: Mapping[str, SourcedValue] = field(default_factory=dict)
    error_scale: str = "additive"
    is_synthetic: bool = False
    notes: str = ""


@dataclass(frozen=True)
class InformationPolicy:
    """Which assays are bought at t1 and how the result may change the ration at t3 (T1, T7).

    ``is_oracle=True`` marks an explicit oracle reference (e.g. perfect information); such a
    policy may never be reported as a feasible strategy.
    """

    policy_id: str
    description: str
    assays: tuple[tuple[str, str], ...] = ()  # (ingredient_id, assay_id)
    decision_rule: str = "constant"  # constant | reoptimize_within_adjustment_limits | library_lookup
    information_available_at_decision: str = "t0"
    is_oracle: bool = False
    adjustment_limits: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""


# --------------------------------------------------------------------------------------------
# problem container
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RationProblem:
    """A complete formulation problem: ingredients, nutrients, constraints and prices.

    Ingredient order defines index ``i``; nutrient order defines index ``j``.  All methods
    receive the same :class:`RationProblem`, and all decisions are scored by the same evaluator.
    """

    problem_id: str
    ingredients: tuple[IngredientRecord, ...]
    nutrients: tuple[NutrientSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    prices: PriceScenario
    animal: Optional[AnimalProfile] = None
    is_synthetic: bool = False
    dataset_status: str = "unspecified"

    def __post_init__(self) -> None:
        ids = [g.ingredient_id for g in self.ingredients]
        if len(set(ids)) != len(ids):
            raise InvalidProblemError("duplicate ingredient ids")
        nids = [n.nutrient_id for n in self.nutrients]
        if len(set(nids)) != len(nids):
            raise InvalidProblemError("duplicate nutrient ids")
        cids = [c.constraint_id for c in self.constraints]
        if len(set(cids)) != len(cids):
            raise InvalidProblemError("duplicate constraint ids")

    @property
    def ingredient_ids(self) -> tuple[str, ...]:
        """Ingredient ids in index order."""
        return tuple(g.ingredient_id for g in self.ingredients)

    @property
    def nutrient_ids(self) -> tuple[str, ...]:
        """Nutrient ids in index order."""
        return tuple(n.nutrient_id for n in self.nutrients)

    def nominal_theta(self) -> np.ndarray:
        """Nominal composition ``[I, J]`` (canonical, DM basis); NaN = missing."""
        out = np.full((len(self.ingredients), len(self.nutrients)), np.nan)
        for i, g in enumerate(self.ingredients):
            for j, n in enumerate(self.nutrients):
                if n.nutrient_id in g.composition:
                    out[i, j] = float(g.composition[n.nutrient_id])
        return out

    def dm_estimates(self) -> np.ndarray:
        """Decision-time DM estimates ``d_hat`` ``[I]``."""
        return np.array([float(g.dm_estimate) for g in self.ingredients])

    def price_vector(self) -> np.ndarray:
        """Prices per kg as-fed ``[I]``."""
        return self.prices.vector(self.ingredient_ids)

    @cached_property
    def compiled(self):  # -> CompiledConstraints
        """Compiled constraint arrays (see :func:`ration_reliability.nutrition.compile_constraints`)."""
        from .nutrition.constraints import compile_constraints

        return compile_constraints(self.constraints, self.ingredients, self.nutrients)

    def reordered(self, ingredient_order: Optional[Sequence[str]] = None,
                  nutrient_order: Optional[Sequence[str]] = None,
                  constraint_order: Optional[Sequence[str]] = None) -> "RationProblem":
        """Copy with permuted ingredients / nutrients / constraints (for invariance tests)."""
        ing = self.ingredients
        if ingredient_order is not None:
            m = {g.ingredient_id: g for g in self.ingredients}
            ing = tuple(m[i] for i in ingredient_order)
        nut = self.nutrients
        if nutrient_order is not None:
            m2 = {n.nutrient_id: n for n in self.nutrients}
            nut = tuple(m2[j] for j in nutrient_order)
        con = self.constraints
        if constraint_order is not None:
            m3 = {c.constraint_id: c for c in self.constraints}
            con = tuple(m3[c] for c in constraint_order)
        return dataclasses.replace(self, ingredients=ing, nutrients=nut, constraints=con)
