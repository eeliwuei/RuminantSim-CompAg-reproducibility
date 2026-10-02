"""Constraint compilation and linearisation (contract T2, section 6.2).

Every constraint ``k`` is compiled into coefficient arrays so that, for one state
``(theta, d)``,

::

    c_ki(theta) = sum_j W[k,i,j] * a_ij + w0[k,i]          # per kg realised DM of ingredient i
    S_k(q, theta) = sum_i q_i d_i c_ki(theta) + sum_i v[k,i] q_i

and the constraint reads, by kind,

* ``concentration``:  ``E = S / D``  (D = sum_i q_i d_i), linearised (D > 0) as
  ``K D - S <= 0`` (ge) or ``S - K D <= 0`` (le)  -- T2;
* ``supply``:         ``E = S``,  ``K - S <= 0`` (ge) or ``S - K <= 0`` (le);
* ``as_fed``:         ``E = sum_i v_ki q_i`` (deterministic).

The *linear rows* used by optimisers are, per state ``s``, ``g_k(q) = A[s,k,:] @ q - b[k]`` with
``g <= 0`` for ge/le and ``g == 0`` for eq.  Because every ``g_k`` is linear in ``q`` for fixed
``theta``, the expectation ``E_theta[g_k(q)] = mean_s(A[s,k,:]) @ q - b[k]`` is exact for a
sample (contract T4 M0: products ``d_i a_ij`` are averaged jointly, never as ``E[d] E[a]``).

Rules enforced at compile time
------------------------------
* units must be registered and dimensionally consistent with the kind;
* ``structural_hard`` constraints may not depend on composition (no nutrient terms) and use the
  decision-time DM estimate ``d_hat`` for DM terms;
* ``probabilistic_nutrition`` / ``diagnostic_only`` use the scenario DM and may not be ``eq``;
* ``bound`` must be present (pending thresholds cannot be compiled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..datamodel import (
    ConstraintClass,
    ConstraintKind,
    ConstraintSpec,
    DMSource,
    IngredientRecord,
    NutrientSpec,
    Provenance,
    Sense,
    ValueStatus,
)
from ..errors import InvalidProblemError, UnitError
from ..hashing import stable_hash
from ..normalization import units as U

__all__ = [
    "CompiledConstraints",
    "LinearRows",
    "compile_constraints",
    "linear_rows",
    "concentration_constraint",
    "dm_offer_constraint",
    "inventory_constraint",
    "as_fed_upper_bound",
]

_KIND_EXPR_DIMS = {
    # kind -> {expression dimension -> allowed unit dimension}
    ConstraintKind.CONCENTRATION: {"mass_fraction": "mass_fraction", "energy_density": "energy_density"},
    ConstraintKind.SUPPLY: {"mass_fraction": "mass_rate", "energy_density": "energy_rate"},
    ConstraintKind.AS_FED: {"as_fed": "mass_rate"},
}


@dataclass(frozen=True)
class CompiledConstraints:
    """Array form of a constraint set for a fixed ingredient and nutrient order.

    Attributes (``K`` constraints, ``I`` ingredients, ``J`` nutrients)
    ----------------------------------------------------------------
    W : ``[K, I, J]`` composition weights.
    w0 : ``[K, I]`` DM-mass weights.
    v : ``[K, I]`` as-fed weights.
    bound : ``[K]`` bound in canonical units.
    tol : ``[K]`` tolerance in canonical units.
    unit_factor : ``[K]`` ``canonical = declared * unit_factor``.
    """

    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    specs: tuple[ConstraintSpec, ...]
    constraint_ids: tuple[str, ...]
    kinds: tuple[ConstraintKind, ...]
    senses: tuple[Sense, ...]
    classes: tuple[ConstraintClass, ...]
    dm_sources: tuple[DMSource, ...]
    units: tuple[str, ...]
    W: np.ndarray
    w0: np.ndarray
    v: np.ndarray
    bound: np.ndarray
    tol: np.ndarray
    unit_factor: np.ndarray
    fingerprint: str

    @property
    def n_constraints(self) -> int:
        """Number of constraints ``K``."""
        return len(self.constraint_ids)

    def indices(self, *classes: ConstraintClass | str) -> np.ndarray:
        """Indices of constraints whose class is in ``classes`` (all if empty)."""
        if not classes:
            return np.arange(self.n_constraints)
        want = {ConstraintClass(c) for c in classes}
        return np.array([k for k, c in enumerate(self.classes) if c in want], dtype=int)

    def subset(self, idx: Sequence[int]) -> "CompiledConstraints":
        """Constraint subset (keeps order of ``idx``)."""
        idx = [int(k) for k in idx]
        specs = tuple(self.specs[k] for k in idx)
        return _build(specs, self.ingredient_ids, self.nutrient_ids,
                      self.W[idx], self.w0[idx], self.v[idx], self.bound[idx], self.tol[idx],
                      self.unit_factor[idx])


@dataclass(frozen=True)
class LinearRows:
    """Linear constraint rows in q-space: ``g_k(q; s) = A[s,k,:] @ q - b[k]``.

    ``is_eq[k]`` marks equality rows (``g == 0``); the others are ``g <= 0``.
    ``missing[s,k]`` is True where a NaN composition/DM of a potentially used ingredient entered
    the row (the corresponding ``A`` entries are NaN).
    """

    constraint_ids: tuple[str, ...]
    A: np.ndarray
    b: np.ndarray
    is_eq: np.ndarray
    missing: np.ndarray

    def mean(self) -> "LinearRows":
        """Scenario average of the rows (exact expectation of the linear ``g`` over the sample)."""
        return LinearRows(self.constraint_ids, self.A.mean(axis=0, keepdims=True), self.b.copy(),
                          self.is_eq.copy(), self.missing.any(axis=0, keepdims=True))

    def residual(self, q: np.ndarray) -> np.ndarray:
        """``g(q)`` per scenario and constraint, ``[S, K]`` (abs value for eq rows)."""
        g = np.einsum("ski,i->sk", self.A, np.asarray(q, dtype=float)) - self.b[None, :]
        return np.where(self.is_eq[None, :], np.abs(g), g)


def _build(specs, ing_ids, nut_ids, W, w0, v, bound, tol, factor) -> CompiledConstraints:
    kinds = tuple(ConstraintKind(s.kind) for s in specs)
    senses = tuple(Sense(s.sense) for s in specs)
    classes = tuple(ConstraintClass(s.constraint_class) for s in specs)
    dms = tuple(DMSource(s.dm_source) for s in specs)
    units = tuple(s.unit for s in specs)
    arrays = [np.array(a, dtype=float) for a in (W, w0, v, bound, tol, factor)]
    for a in arrays:
        a.setflags(write=False)
    W, w0, v, bound, tol, factor = arrays
    fp = stable_hash("CompiledConstraints/v1", tuple(ing_ids), tuple(nut_ids),
                     tuple(s.constraint_id for s in specs),
                     tuple(k.value for k in kinds), tuple(x.value for x in senses),
                     tuple(c.value for c in classes), tuple(d.value for d in dms), units,
                     W, w0, v, bound, tol, factor)
    return CompiledConstraints(tuple(ing_ids), tuple(nut_ids), tuple(specs),
                               tuple(s.constraint_id for s in specs), kinds, senses, classes, dms, units,
                               W, w0, v, bound, tol, factor, fp)


def compile_constraints(specs: Sequence[ConstraintSpec],
                        ingredients: Sequence[IngredientRecord],
                        nutrients: Sequence[NutrientSpec]) -> CompiledConstraints:
    """Compile ``specs`` against the given ingredient and nutrient order.

    Raises
    ------
    InvalidProblemError
        Listing *all* problems found (grammar, units, class rules, missing bounds).
    """
    ing_ids = [g.ingredient_id for g in ingredients]
    nut_ids = [n.nutrient_id for n in nutrients]
    nut_dim = {n.nutrient_id: n.dimension for n in nutrients}
    ing_index = {g: i for i, g in enumerate(ing_ids)}
    nut_index = {n: j for j, n in enumerate(nut_ids)}
    groups: set[str] = set()
    for g in ingredients:
        groups.update(g.group_weights.keys())

    I, J, K = len(ing_ids), len(nut_ids), len(specs)
    W = np.zeros((K, I, J))
    w0 = np.zeros((K, I))
    v = np.zeros((K, I))
    bound = np.zeros(K)
    tol = np.zeros(K)
    factor = np.ones(K)
    errors: list[str] = []

    def gw(group: str) -> np.ndarray:
        return np.array([float(g.group_weights.get(group, 0.0)) for g in ingredients])

    coef_names: set[str] = set()
    for g in ingredients:
        coef_names.update(g.coefficients.keys())

    for k, s in enumerate(specs):
        where = f"constraint {s.constraint_id}"
        try:
            kind = ConstraintKind(s.kind)
            sense = Sense(s.sense)
            cls = ConstraintClass(s.constraint_class)
            dms = DMSource(s.dm_source)
        except ValueError as exc:
            errors.append(f"{where}: {exc}")
            continue
        if not s.terms:
            errors.append(f"{where}: empty expression")
            continue
        expr_dims: set[str] = set()
        has_nut = has_dm = has_af = False
        for key, coef in s.terms.items():
            try:
                c = float(coef)
            except (TypeError, ValueError):
                errors.append(f"{where}: non-numeric coefficient for {key!r}")
                continue
            if not np.isfinite(c):
                errors.append(f"{where}: non-finite coefficient for {key!r}")
                continue
            parts = str(key).split(":")
            if parts[0] == "AF":
                has_af = True
                if len(parts) == 1:
                    v[k] += c
                elif len(parts) == 2 and parts[1] in ing_index:
                    v[k, ing_index[parts[1]]] += c
                else:
                    errors.append(f"{where}: bad as-fed term {key!r} (unknown ingredient?)")
            elif parts[0] == "DM":
                has_dm = True
                expr_dims.add("mass_fraction")
                if len(parts) == 1:
                    w0[k] += c
                elif len(parts) == 2 and parts[1] in ing_index:
                    w0[k, ing_index[parts[1]]] += c
                else:
                    errors.append(f"{where}: bad DM term {key!r} (unknown ingredient?)")
            elif parts[0] == "G":
                if len(parts) != 3:
                    errors.append(f"{where}: group term must be 'G:<group>:<nutrient|DM>', got {key!r}")
                    continue
                grp, tgt = parts[1], parts[2]
                if grp not in groups:
                    errors.append(f"{where}: group {grp!r} is not declared on any ingredient")
                    continue
                if tgt == "DM":
                    has_dm = True
                    expr_dims.add("mass_fraction")
                    w0[k] += c * gw(grp)
                elif tgt in nut_index:
                    has_nut = True
                    expr_dims.add(nut_dim[tgt])
                    W[k, :, nut_index[tgt]] += c * gw(grp)
                else:
                    errors.append(f"{where}: unknown nutrient {tgt!r} in {key!r}")
            elif parts[0] == "C":
                if len(parts) != 3:
                    errors.append(f"{where}: coefficient term must be 'C:<coef>:<nutrient|DM>', got {key!r}")
                    continue
                cname, tgt = parts[1], parts[2]
                lacking = [g.ingredient_id for g in ingredients if cname not in g.coefficients]
                if cname not in coef_names or lacking:
                    errors.append(f"{where}: coefficient {cname!r} missing for ingredients {lacking}")
                    continue
                cv = np.array([float(g.coefficients[cname]) for g in ingredients])
                if tgt == "DM":
                    has_dm = True
                    expr_dims.add("mass_fraction")
                    w0[k] += c * cv
                elif tgt in nut_index:
                    has_nut = True
                    expr_dims.add(nut_dim[tgt])
                    W[k, :, nut_index[tgt]] += c * cv
                else:
                    errors.append(f"{where}: unknown nutrient {tgt!r} in {key!r}")
            elif len(parts) == 1 and parts[0] in nut_index:
                has_nut = True
                expr_dims.add(nut_dim[parts[0]])
                W[k, :, nut_index[parts[0]]] += c
            else:
                errors.append(f"{where}: unknown term {key!r}")

        # --- kind / dimension / unit rules
        if has_af and (has_nut or has_dm):
            errors.append(f"{where}: as-fed terms cannot be mixed with DM/nutrient terms")
        if kind is ConstraintKind.AS_FED:
            if not has_af:
                errors.append(f"{where}: kind as_fed requires AF terms")
            expr_dim = "as_fed"
        else:
            if has_af:
                errors.append(f"{where}: AF terms require kind as_fed")
            if len(expr_dims) > 1:
                errors.append(f"{where}: terms mix dimensions {sorted(expr_dims)}")
            expr_dim = next(iter(expr_dims)) if expr_dims else "mass_fraction"
        allowed = _KIND_EXPR_DIMS[kind]
        try:
            u = U.get_unit(s.unit)
            if expr_dim in allowed and u.dimension != allowed[expr_dim]:
                errors.append(f"{where}: unit {s.unit!r} ({u.dimension}) does not fit a {kind.value} "
                              f"of a {expr_dim} expression (needs {allowed[expr_dim]})")
            factor[k] = u.factor
        except UnitError as exc:
            errors.append(f"{where}: {exc}")
        if kind is ConstraintKind.CONCENTRATION and s.basis != "DM":
            errors.append(f"{where}: concentration constraints must be on DM basis")
        if kind is ConstraintKind.AS_FED and s.basis != "as_fed":
            errors.append(f"{where}: as_fed constraints must have basis 'as_fed'")
        if kind is ConstraintKind.SUPPLY and s.basis not in ("none", "DM"):
            errors.append(f"{where}: supply constraints must have basis 'none' or 'DM'")

        # --- class rules
        if cls is ConstraintClass.STRUCTURAL_HARD:
            if has_nut:
                errors.append(f"{where}: structural_hard constraints may not depend on composition "
                              f"(model it as probabilistic_nutrition or diagnostic_only)")
            if has_dm and dms is not DMSource.DECISION_ESTIMATE:
                errors.append(f"{where}: structural DM terms must use dm_source=decision_estimate")
        else:
            if dms is not DMSource.SCENARIO:
                errors.append(f"{where}: {cls.value} constraints must use dm_source=scenario")
            if sense is Sense.EQ:
                errors.append(f"{where}: equality is not allowed for {cls.value} (cannot hold in all states)")
            if kind is ConstraintKind.AS_FED:
                errors.append(f"{where}: as_fed constraints are deterministic; classify as structural_hard")

        # --- bound / tolerance
        if s.bound is None:
            errors.append(f"{where}: bound is missing (pending thresholds cannot be compiled)")
        elif not np.isfinite(float(s.bound)):
            errors.append(f"{where}: bound must be finite")
        else:
            bound[k] = float(s.bound) * factor[k]
        t = float(s.numerical_tolerance)
        if not (np.isfinite(t) and t > 0):
            # red-team fix C10: with tolerance 0 a binding row evaluated at its own design point is
            # counted as violated because of floating-point rounding (margins of -3e-15 observed)
            errors.append(f"{where}: numerical_tolerance must be finite and > 0 (0 turns rounding errors of binding "
                          "rows into violations)")
        else:
            tol[k] = t * factor[k]
        prov = s.provenance
        if not isinstance(prov, Provenance):
            errors.append(f"{where}: provenance missing")
        else:
            errors.extend(prov.issues(where))
            if prov.status is ValueStatus.PENDING_USER_DECISION:
                errors.append(f"{where}: pending_user_decision constraint cannot be compiled")

    if errors:
        raise InvalidProblemError("constraint compilation failed:\n  - " + "\n  - ".join(errors))
    return _build(tuple(specs), ing_ids, nut_ids, W, w0, v, bound, tol, factor)


def linear_rows(cc: CompiledConstraints, theta: np.ndarray, d: np.ndarray,
                d_hat: Optional[np.ndarray] = None) -> LinearRows:
    """Linearised rows of all constraints in ``cc`` for states ``(theta, d)``.

    Parameters
    ----------
    theta : ``[S, I, J]`` or ``[I, J]`` composition (canonical, DM basis).
    d : ``[S, I]`` or ``[I]`` scenario DM fractions (used by ``dm_source=scenario``).
    d_hat : ``[I]`` decision-time DM estimates, required if any constraint uses
        ``dm_source=decision_estimate`` with DM terms.
    """
    theta = np.asarray(theta, dtype=float)
    d = np.asarray(d, dtype=float)
    if theta.ndim == 2:
        theta = theta[None]
    if d.ndim == 1:
        d = d[None]
    S, I, J = theta.shape
    K = cc.n_constraints
    if (I, J) != (len(cc.ingredient_ids), len(cc.nutrient_ids)) or d.shape != (S, I):
        raise ValueError("linear_rows: theta/d shapes do not match the compiled constraints")

    uses_dhat = np.array([dm is DMSource.DECISION_ESTIMATE and np.any(cc.w0[k] != 0)
                          for k, dm in enumerate(cc.dm_sources)], dtype=bool)
    if uses_dhat.any():
        if d_hat is None:
            raise ValueError("linear_rows: d_hat required for structural DM terms")
        d_hat = np.asarray(d_hat, dtype=float)
        if d_hat.shape != (I,):
            raise ValueError("linear_rows: d_hat must have shape [I]")

    nan_t = np.isnan(theta)
    th0 = np.where(nan_t, 0.0, theta)
    c = np.einsum("sij,kij->ski", th0, cc.W) + cc.w0[None, :, :]          # [S,K,I]
    miss_c = np.einsum("sij,kij->ski", nan_t.astype(float), (cc.W != 0).astype(float)) > 0

    D = np.broadcast_to(d[:, None, :], (S, K, I)).copy()
    if uses_dhat.any():
        D[:, uses_dhat, :] = d_hat[None, None, :]
    nan_d = np.isnan(D)
    D0 = np.where(nan_d, 0.0, D)

    A = np.empty((S, K, I))
    b = np.zeros(K)
    is_eq = np.array([s is Sense.EQ for s in cc.senses], dtype=bool)
    for k in range(K):
        kind, sense, Kk = cc.kinds[k], cc.senses[k], cc.bound[k]
        if kind is ConstraintKind.CONCENTRATION:
            row = D0[:, k, :] * (c[:, k, :] - Kk)          # S - K D
            A[:, k, :] = -row if sense is Sense.GE else row
            b[k] = 0.0
        elif kind is ConstraintKind.SUPPLY:
            row = D0[:, k, :] * c[:, k, :] + cc.v[k][None, :]
            A[:, k, :] = -row if sense is Sense.GE else row
            b[k] = -Kk if sense is Sense.GE else Kk
        else:  # AS_FED
            row = np.broadcast_to(cc.v[k][None, :], (S, I))
            A[:, k, :] = -row if sense is Sense.GE else row
            b[k] = -Kk if sense is Sense.GE else Kk
    d_needed = np.array([kind is not ConstraintKind.AS_FED for kind in cc.kinds], dtype=bool)
    missing = miss_c | (nan_d & d_needed[None, :, None])
    A = np.where(missing, np.nan, A)
    return LinearRows(cc.constraint_ids, A, b, is_eq, missing.any(axis=2))


# ---------------------------------------------------------------------------------------------
# convenience builders (thresholds must be supplied by the caller with provenance)
# ---------------------------------------------------------------------------------------------

def concentration_constraint(constraint_id: str, terms: dict[str, float], sense: str, bound: float,
                             unit: str, provenance: Provenance, *, tolerance: float,
                             constraint_class: str = "probabilistic_nutrition",
                             name: str = "", **kw) -> ConstraintSpec:
    """Build a DM-basis concentration constraint (e.g. ``{"CP": 1}``, ``ge``, ``16``, ``"%"``).

    ``tolerance`` (in ``unit``, > 0) is required: it is a declared research setting, not a default.
    """
    return ConstraintSpec(constraint_id=constraint_id, name=name or constraint_id,
                          kind=ConstraintKind.CONCENTRATION, terms=dict(terms), sense=Sense(sense),
                          bound=bound, unit=unit, constraint_class=ConstraintClass(constraint_class),
                          numerical_tolerance=tolerance, provenance=provenance, basis="DM", **kw)


def dm_offer_constraint(constraint_id: str, dmi_kg_per_day: float, provenance: Provenance, *,
                        sense: str = "eq", tolerance: float = 1e-6, name: str = "planned DM offered",
                        **kw) -> ConstraintSpec:
    """Structural rule: planned DM offered ``sum_i q_i d_hat_i (sense) DMI`` (kg DM/head/d).

    Uses the decision-time estimate ``d_hat``; the realised DM supply under uncertain DM is
    reported by the evaluator and can be constrained separately as a probabilistic interval.
    """
    return ConstraintSpec(constraint_id=constraint_id, name=name, kind=ConstraintKind.SUPPLY,
                          terms={"DM": 1.0}, sense=Sense(sense), bound=dmi_kg_per_day, unit="kg/d",
                          constraint_class=ConstraintClass.STRUCTURAL_HARD, numerical_tolerance=tolerance,
                          provenance=provenance, basis="DM", dm_source=DMSource.DECISION_ESTIMATE, **kw)


def as_fed_upper_bound(constraint_id: str, ingredient_id: str, q_max_kg_per_day: float,
                       provenance: Provenance, *, tolerance: float = 1e-9, name: str = "", **kw) -> ConstraintSpec:
    """Structural ``q_k <= q_max`` (kg as-fed/head/d)."""
    return ConstraintSpec(constraint_id=constraint_id, name=name or f"q[{ingredient_id}] upper",
                          kind=ConstraintKind.AS_FED, terms={f"AF:{ingredient_id}": 1.0}, sense=Sense.LE,
                          bound=q_max_kg_per_day, unit="kg/d",
                          constraint_class=ConstraintClass.STRUCTURAL_HARD, numerical_tolerance=tolerance,
                          provenance=provenance, basis="as_fed", dm_source=DMSource.DECISION_ESTIMATE, **kw)


def inventory_constraint(constraint_id: str, ingredient_id: str, batch_kg_as_fed: float, heads: float,
                         days: float, provenance: Provenance, *, tolerance: float = 1e-9) -> ConstraintSpec:
    """Structural inventory rule ``H * T * q_k <= B_k`` (T2.3), i.e. ``q_k <= B_k / (H T)``."""
    if heads <= 0 or days <= 0:
        raise InvalidProblemError("inventory_constraint: heads and days must be > 0")
    return as_fed_upper_bound(constraint_id, ingredient_id, batch_kg_as_fed / (heads * days), provenance,
                              tolerance=tolerance, name=f"inventory {ingredient_id} (H={heads:g}, T={days:g})",
                              notes=f"B={batch_kg_as_fed:g} kg as-fed; H={heads:g}; T={days:g}")
