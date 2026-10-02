"""Correlation-structure candidates, PSD checks, nearest-correlation repair and a Gaussian copula.

Contract 01 §7.4 / §7.5 and ``configs/uncertainty.yaml`` (``c_correlation_structure``): with only
marginal summaries (NASEM Table 19-1 mean/SD/N), the joint structure is unknown; independence is
a *baseline*, not a fact.  This module implements the candidate structures as mechanisms; no
correlation value is embedded here.

Labels
    A *cell* is ``(ingredient_id, component)`` with ``component`` a nutrient id or ``"DM"``.

Candidates (``structure_id`` prefixes follow ``configs/uncertainty.yaml``)
    ``C0_independent``     identity (:func:`independent_structure`).
    ``C1_yoder_table6``    published population correlations, other pairs 0
                           (:func:`pairwise_structure`); every value needs a source.  A ``None``
                           value raises :class:`PendingValueError` (never replaced by 0).
    ``C_CONS_closure``     negative correlations implied by a partial mass balance of
                           non-overlapping components (:func:`conservation_closure_structure`).
    ``C4_extreme``         ``R(c) = I + c (R1 - I)``, ``c* = min(c_cap, c_psd)``
                           (:func:`extreme_scaling`), PSD by construction.
    ``C5_cross``           same-component correlation between two ingredients
                           (:func:`with_cross_correlation`).

PSD policy (``configs/uncertainty.yaml`` ``psd_repair.rule``): a pre-registered structure must be
PSD by construction.  :func:`assert_psd` raises :class:`NotPSDError`;
:func:`repair_correlation` refuses to repair unless ``allow=True`` (diagnostic use), and then
returns a :class:`RepairRecord` with the Frobenius difference, the eigenvalues before/after and
every coefficient that moved by more than ``flag_threshold`` (0.05).

Joint sampling (:class:`GaussianCopulaModel`): ``z ~ N(0, R)``, ``u = Phi(z)``,
``x = F^{-1}(u)`` with the moment-matched marginals of :mod:`.distributions`.  The copula keeps
every marginal exactly; the Pearson correlation of the transformed values differs from ``R``
(reported by :func:`induced_pearson`).  With ``R = I`` the draws equal the independent draws
(common random numbers across correlation scenarios).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import special

from ..hashing import stable_hash
from .base import UncertaintyModel
from .distributions import MarginalFit, fit_array

__all__ = [
    "Cell",
    "PARAMETER_STATUSES",
    "CLOSURE_FORBIDDEN_PAIRS",
    "PARTIAL_COMPONENT_OVERLAPS",
    "NotPSDError",
    "PendingValueError",
    "PSDCheck",
    "RepairRecord",
    "CorrelationStructure",
    "validate_correlation",
    "min_eigenvalue",
    "check_psd",
    "assert_psd",
    "nearest_correlation",
    "repair_correlation",
    "independent_structure",
    "pairwise_structure",
    "conservation_closure_matrix",
    "conservation_closure_structure",
    "closure_strength_for_floor",
    "extreme_scaling",
    "combine_structures",
    "with_cross_correlation",
    "sampling_factor",
    "GaussianCopulaModel",
    "induced_pearson",
]

Cell = tuple  # (ingredient_id, component)

#: Provenance statuses of correlation parameters (same vocabulary as ``datamodel.ValueStatus``).
PARAMETER_STATUSES = ("sourced", "research_scenario_assumption", "pending_user_decision",
                      "synthetic_test_only")

#: Component pairs whose definitions nest (one is part of the other): a mass balance over them is
#: meaningless (contract 01 §7.4: indicators must not be forced to sum to 100 %).
CLOSURE_FORBIDDEN_PAIRS: frozenset = frozenset({
    frozenset({"ADF", "NDF"}), frozenset({"lignin", "ADF"}), frozenset({"lignin", "NDF"}),
    frozenset({"TFA", "EE"}), frozenset({"Ca", "ash"}), frozenset({"P", "ash"}), frozenset({"DM", "CP"}),
    frozenset({"DM", "NDF"}), frozenset({"DM", "starch"}), frozenset({"DM", "EE"}), frozenset({"DM", "ash"}),
})
#: Pairs that overlap only partly (CP contains NDICP, which is also inside NDF): allowed in the
#: closure with a recorded caveat.
PARTIAL_COMPONENT_OVERLAPS: frozenset = frozenset({frozenset({"CP", "NDF"})})


class NotPSDError(ValueError):
    """A correlation structure is not positive semi-definite (or would need repair)."""


class PendingValueError(ValueError):
    """A correlation value needed by a structure is ``None`` (pending source / user decision)."""


# ----------------------------------------------------------------------------------------------
# basic checks
# ----------------------------------------------------------------------------------------------

def validate_correlation(r: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Return ``r`` as float array after checking square, finite, symmetric, unit diagonal, |rho|<=1."""
    r = np.array(r, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("correlation matrix must be square")
    if not np.all(np.isfinite(r)):
        raise ValueError("correlation matrix has non-finite entries")
    if not np.allclose(r, r.T, atol=atol, rtol=0):
        raise ValueError("correlation matrix is not symmetric")
    if not np.allclose(np.diag(r), 1.0, atol=atol, rtol=0):
        raise ValueError("correlation matrix must have a unit diagonal")
    if np.any(np.abs(r) > 1.0 + atol):
        raise ValueError("correlation coefficients must lie in [-1, 1]")
    return r


def min_eigenvalue(r: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return 1.0
    return float(np.linalg.eigvalsh((r + r.T) / 2.0)[0])


@dataclass(frozen=True)
class PSDCheck:
    """Result of a PSD check: ``lambda_min`` and ``is_psd = lambda_min >= -tol``."""

    lambda_min: float
    is_psd: bool
    tol: float


def check_psd(r: np.ndarray, tol: float = 1e-10) -> PSDCheck:
    """PSD check with numerical tolerance ``tol`` on the smallest eigenvalue."""
    lam = min_eigenvalue(r)
    return PSDCheck(lam, bool(lam >= -float(tol)), float(tol))


def assert_psd(obj, name: str = "", tol: float = 1e-10) -> PSDCheck:
    """Raise :class:`NotPSDError` unless ``obj`` (matrix or :class:`CorrelationStructure`) is PSD."""
    mat = obj.matrix if isinstance(obj, CorrelationStructure) else np.asarray(obj, dtype=float)
    label = name or (obj.structure_id if isinstance(obj, CorrelationStructure) else "correlation matrix")
    chk = check_psd(mat, tol)
    if not chk.is_psd:
        raise NotPSDError(f"{label}: not positive semi-definite (lambda_min = {chk.lambda_min:.6g} < -{tol:g}); "
                          "a pre-registered structure that needs repair is a definition error "
                          "(configs/uncertainty.yaml psd_repair.rule)")
    return chk


# ----------------------------------------------------------------------------------------------
# nearest correlation matrix (Higham 2002, alternating projections with Dykstra correction)
# ----------------------------------------------------------------------------------------------

def _proj_psd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((a + a.T) / 2.0)
    return (v * np.maximum(w, 0.0)) @ v.T


def nearest_correlation(a: np.ndarray, tol: float = 1e-12, max_iter: int = 20000
                        ) -> tuple[np.ndarray, int, bool]:
    """Nearest correlation matrix in the Frobenius norm (Higham 2002, IMA J Numer Anal 22:329-343).

    Alternating projections onto the PSD cone (with Dykstra's correction) and the unit-diagonal
    set.  Returns ``(X, iterations, converged)``; ``X`` is symmetric with a unit diagonal.
    """
    a = np.array(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    y = (a + a.T) / 2.0
    ds = np.zeros_like(y)
    x = y.copy()
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        r = y - ds
        x = _proj_psd(r)
        ds = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        diff = np.linalg.norm(y_new - y, "fro") / max(np.linalg.norm(y_new, "fro"), 1e-300)
        y = y_new
        if diff < tol and np.linalg.norm(y - x, "fro") / max(np.linalg.norm(y, "fro"), 1e-300) < max(tol * 1e3, 1e-10):
            converged = True
            break
    out = (y + y.T) / 2.0
    np.fill_diagonal(out, 1.0)
    return out, it, converged


@dataclass(frozen=True)
class RepairRecord:
    """What a nearest-correlation repair changed (contract 01 §7.4: report the modification)."""

    structure_id: str
    repaired: bool
    lambda_min_before: float
    lambda_min_after: float
    frobenius_diff: float
    max_abs_change: float
    changed_pairs: tuple[tuple[str, str, float, float, float], ...]  # (label_a, label_b, before, after, delta)
    flagged_pairs: tuple[tuple[str, str, float, float, float], ...]  # |delta| > flag_threshold
    flag_threshold: float
    iterations: int
    converged: bool

    def as_dict(self) -> dict:
        return {"structure_id": self.structure_id, "repaired": self.repaired,
                "lambda_min_before": self.lambda_min_before, "lambda_min_after": self.lambda_min_after,
                "frobenius_diff": self.frobenius_diff, "max_abs_change": self.max_abs_change,
                "n_changed_pairs": len(self.changed_pairs), "n_flagged_pairs": len(self.flagged_pairs),
                "flag_threshold": self.flag_threshold, "iterations": self.iterations,
                "converged": self.converged}


def _label(c: Cell) -> str:
    return f"{c[0]}:{c[1]}"


def repair_correlation(structure: "CorrelationStructure", *, allow: bool = False, flag_threshold: float = 0.05,
                       tol: float = 1e-10, change_atol: float = 1e-9
                       ) -> tuple["CorrelationStructure", RepairRecord]:
    """Repair a non-PSD structure with :func:`nearest_correlation` -- only when ``allow=True``.

    A PSD structure is returned unchanged with ``repaired=False``.  For a non-PSD structure and
    ``allow=False`` :class:`NotPSDError` is raised (pre-registered structures must be PSD by
    construction).  With ``allow=True`` (diagnostics, e.g. demonstrating what a repair would
    silently change) the repaired structure carries the note ``higham_repaired`` and the record
    lists every coefficient that moved.
    """
    before = structure.matrix
    lam0 = min_eigenvalue(before)
    if lam0 >= -tol:
        rec = RepairRecord(structure.structure_id, False, lam0, lam0, 0.0, 0.0, (), (), float(flag_threshold), 0, True)
        return structure, rec
    if not allow:
        assert_psd(structure, tol=tol)
    after, it, conv = nearest_correlation(before)
    lam1 = min_eigenvalue(after)
    delta = after - before
    changed, flagged = [], []
    n = before.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(delta[i, j])
            if abs(d) > change_atol:
                row = (_label(structure.labels[i]), _label(structure.labels[j]), float(before[i, j]),
                       float(after[i, j]), d)
                changed.append(row)
                if abs(d) > flag_threshold:
                    flagged.append(row)
    rec = RepairRecord(structure.structure_id, True, lam0, lam1, float(np.linalg.norm(delta, "fro")),
                       float(np.max(np.abs(delta))), tuple(changed), tuple(flagged), float(flag_threshold), it, conv)
    rep = CorrelationStructure(structure.structure_id + "+higham_repaired", structure.labels, after,
                               structure.status, structure.provenance,
                               structure.notes + ("higham_repaired (diagnostic only; not a pre-registered structure)",),
                               psd_tol=max(structure.psd_tol, 1e-9))
    return rep, rec


# ----------------------------------------------------------------------------------------------
# structure container
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class CorrelationStructure:
    """A correlation matrix over labelled cells with provenance.

    The constructor validates shape, symmetry, unit diagonal and |rho| <= 1; PSD is *not*
    enforced here so that a failing candidate can be reported (use :func:`assert_psd` or the
    copula model, which enforces it).
    """

    structure_id: str
    labels: tuple
    matrix: np.ndarray
    status: str
    provenance: str
    notes: tuple = field(default_factory=tuple)
    psd_tol: float = 1e-10

    def __post_init__(self) -> None:
        labels = tuple(tuple(c) for c in self.labels)
        if any(len(c) != 2 for c in labels):
            raise ValueError("labels must be (ingredient_id, component) pairs")
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate labels in correlation structure")
        m = validate_correlation(self.matrix)
        if m.shape[0] != len(labels):
            raise ValueError("matrix size does not match labels")
        if self.status not in PARAMETER_STATUSES:
            raise ValueError(f"status must be one of {PARAMETER_STATUSES}")
        m = (m + m.T) / 2.0
        m.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def psd(self) -> PSDCheck:
        """PSD check of the matrix."""
        return check_psd(self.matrix, self.psd_tol)

    def index(self, cell: Cell) -> int:
        return self.labels.index(tuple(cell))

    def rho(self, a: Cell, b: Cell) -> float:
        """Correlation between two labelled cells."""
        return float(self.matrix[self.index(a), self.index(b)])

    def submatrix(self, cells: Sequence[Cell]) -> np.ndarray:
        idx = [self.index(c) for c in cells]
        return self.matrix[np.ix_(idx, idx)].copy()

    def fingerprint(self) -> str:
        return stable_hash("CorrelationStructure/v1", self.structure_id, self.labels, self.matrix, self.status)


# ----------------------------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------------------------

def independent_structure(cells: Sequence[Cell], structure_id: str = "C0_independent",
                          status: str = "research_scenario_assumption",
                          provenance: str = "configs/uncertainty.yaml C0_independent: baseline, not a fact "
                                            "(contract 01 §7.4)") -> CorrelationStructure:
    """``C0``: identity over ``cells``."""
    return CorrelationStructure(structure_id, tuple(cells), np.eye(len(cells)), status, provenance)


def pairwise_structure(ingredient_id: str, components: Sequence[str],
                       pairs: Mapping[tuple[str, str], Optional[float]], *, structure_id: str, status: str,
                       provenance: str, zero_pairs: Iterable[tuple[str, str]] = ()) -> CorrelationStructure:
    """Within-ingredient structure from listed pairs; unlisted pairs are 0.

    ``pairs`` maps ``(component_a, component_b) -> rho``.  A ``None`` value raises
    :class:`PendingValueError` (a missing source is never read as 0).  Pairs whose component is not
    in ``components`` raise ``ValueError`` (the caller must restrict the pairs explicitly, e.g. to
    the stochastic components).  ``zero_pairs`` are forced to 0 and recorded (e.g. pairs "not
    treated as correlation evidence").  PSD is not enforced here.
    """
    comps = tuple(components)
    if len(set(comps)) != len(comps):
        raise ValueError("duplicate components")
    r = np.eye(len(comps))
    zero = {frozenset(p) for p in zero_pairs}
    pending = []
    for (a, b), v in pairs.items():
        if a not in comps or b not in comps:
            raise ValueError(f"pair ({a}, {b}) is outside the component set {comps}")
        if a == b:
            raise ValueError("diagonal pairs are not allowed")
        if frozenset((a, b)) in zero:
            continue
        if v is None:
            pending.append(f"{a}_{b}")
            continue
        i, j = comps.index(a), comps.index(b)
        r[i, j] = r[j, i] = float(v)
    if pending:
        raise PendingValueError(f"{structure_id} [{ingredient_id}]: correlation value(s) pending: "
                                f"{', '.join(sorted(pending))}")
    notes = tuple(f"forced_zero:{'_'.join(sorted(p))}" for p in sorted(zero, key=sorted))
    return CorrelationStructure(structure_id, tuple((ingredient_id, c) for c in comps), r, status, provenance, notes)


def _check_disjoint(components: Sequence[str]) -> list[str]:
    notes = []
    comps = list(components)
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            p = frozenset((comps[i], comps[j]))
            if p in CLOSURE_FORBIDDEN_PAIRS:
                raise ValueError(f"components {sorted(p)} overlap by definition; a mass balance over them is "
                                 "not defined (contract 01 §7.4)")
            if p in PARTIAL_COMPONENT_OVERLAPS:
                notes.append(f"partial_overlap:{'_'.join(sorted(p))} (NDICP is inside both CP and NDF)")
    return notes


def conservation_closure_matrix(sds: Sequence[float], strength: float) -> np.ndarray:
    """Correlation matrix of the *renormalised closure projection*.

    With ``D = diag(sd^2)``, ``s = sum(sd^2)`` and strength ``c`` in ``[0, 1)``:
    ``Sigma_c = D - c D 1 1' D / s`` (the variance of the component sum is reduced from ``s`` to
    ``(1 - c) s`` while every component keeps a share), and ``R_c`` is ``Sigma_c`` rescaled to a
    unit diagonal:  ``rho_kl = -c sd_k sd_l / sqrt((s - c sd_k^2)(s - c sd_l^2))``.
    ``R_c`` is positive definite for ``0 <= c < 1`` (congruent to ``I - c u u'`` with ``|u| = 1``)
    and ``R_0 = I``.
    """
    sd = np.asarray(sds, dtype=float)
    if sd.ndim != 1 or sd.size < 2:
        raise ValueError("need at least two components")
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("closure needs finite SD > 0 for every component (a missing SD is never 0)")
    c = float(strength)
    if not (0.0 <= c < 1.0):
        raise ValueError("closure strength must lie in [0, 1)")
    v = sd * sd
    s = float(v.sum())
    denom = np.sqrt(s - c * v)
    r = -c * np.outer(sd, sd) / np.outer(denom, denom)
    np.fill_diagonal(r, 1.0)
    return r


def closure_strength_for_floor(sds: Sequence[float], lambda_floor: float = 0.01, tol: float = 1e-12) -> float:
    """Largest closure strength ``c`` with ``lambda_min(R_c) >= lambda_floor`` (bisection)."""
    if not (0.0 < lambda_floor < 1.0):
        raise ValueError("lambda_floor must lie in (0, 1)")
    lo, hi = 0.0, 1.0 - 1e-15
    if min_eigenvalue(conservation_closure_matrix(sds, hi)) >= lambda_floor:
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if min_eigenvalue(conservation_closure_matrix(sds, mid)) >= lambda_floor:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def conservation_closure_structure(ingredient_id: str, components: Sequence[str], sds: Sequence[float],
                                   strength: float, *, structure_id: str = "C_CONS_closure",
                                   status: str = "research_scenario_assumption",
                                   provenance: str = "partial mass balance of non-overlapping components; "
                                                     "strength is a scenario parameter without a source"
                                   ) -> CorrelationStructure:
    """``C_CONS``: closure-implied negative correlations among non-overlapping ``components``.

    Only definitionally disjoint components are allowed (nested pairs such as ADF/NDF, TFA/EE,
    Ca/ash raise ``ValueError``; CP/NDF is accepted with a recorded partial-overlap caveat).  The
    remainder of the dry matter (ash, sugars, organic acids, pectins, ... minus overlaps) absorbs
    the rest of the balance, so no 100 % sum is imposed (contract 01 §7.4).

    Scenario construction, not an estimate (FIX5, acceptance B07): the mass-balance argument
    presumes that the components are measured on the same samples, but the SDs passed here come
    from different sample sets (NASEM Table 19-1 prints a different N for each component; in the
    17 stochastic entries no entry has one common N over {CP, NDF, starch, EE}).  Building a
    covariance from them is the "pseudo-pairing" of B07; the result is labelled
    ``research_scenario_assumption`` and must not be reported as a correlation estimated from data.
    Nor is ``C_CONS`` at its maximal strength a stand-in for the pre-registered extreme structure C4.
    """
    notes = _check_disjoint(components)
    r = conservation_closure_matrix(sds, strength)
    notes.append(f"closure_strength={float(strength)!r}")
    return CorrelationStructure(structure_id, tuple((ingredient_id, c) for c in components), r, status,
                                provenance, tuple(notes))


def extreme_scaling(r1: np.ndarray, *, rho_cap: float = 0.9, lambda_floor: float = 0.01
                    ) -> tuple[np.ndarray, float, str, float]:
    """``C4``: ``R(c) = I + c (R1 - I)`` with ``c* = min(c_cap, c_psd)``.

    ``c_cap = rho_cap / max|rho_offdiag(R1)|``; ``c_psd`` is the largest ``c`` with
    ``lambda_min(R(c)) >= lambda_floor``: since ``eig(R(c)) = 1 + c (eig(R1) - 1)``,
    ``c_psd = (1 - lambda_floor) / (1 - lambda_min(R1))`` when ``lambda_min(R1) < 1``.
    Pairs that must stay 0 have to be 0 in ``R1`` already.  ``R1`` must be PSD.
    Returns ``(R(c*), c*, binding, lambda_min(R(c*)))`` with ``binding`` in
    ``{"rho_cap", "lambda_floor", "identity"}``.
    """
    r1 = validate_correlation(r1)
    assert_psd(r1, "extreme_scaling input R1")
    off = r1 - np.eye(r1.shape[0])
    mx = float(np.max(np.abs(off))) if off.size else 0.0
    if mx == 0.0:
        return np.eye(r1.shape[0]), 0.0, "identity", 1.0
    c_cap = float(rho_cap) / mx
    lam1 = min_eigenvalue(r1)
    c_psd = (1.0 - float(lambda_floor)) / (1.0 - lam1) if lam1 < 1.0 else np.inf
    if c_cap <= c_psd:
        c, binding = c_cap, "rho_cap"
    else:
        c, binding = c_psd, "lambda_floor"
    r = np.eye(r1.shape[0]) + c * off
    return r, float(c), binding, min_eigenvalue(r)


def combine_structures(structures: Sequence[CorrelationStructure], structure_id: str, *, status: str,
                       provenance: str) -> CorrelationStructure:
    """Block-diagonal union of structures over disjoint label sets (cross blocks = 0)."""
    labels: list = []
    for s in structures:
        labels.extend(s.labels)
    if len(set(labels)) != len(labels):
        raise ValueError("structures to combine share labels")
    n = len(labels)
    r = np.eye(n)
    k = 0
    notes: list[str] = []
    for s in structures:
        m = s.matrix.shape[0]
        r[k:k + m, k:k + m] = s.matrix
        k += m
        notes.extend(f"{s.structure_id}:{x}" for x in s.notes)
    return CorrelationStructure(structure_id, tuple(labels), r, status, provenance, tuple(notes))


def with_cross_correlation(base: CorrelationStructure, pairs: Mapping[tuple[Cell, Cell], Optional[float]], *,
                           structure_id: str, status: str, provenance: str) -> CorrelationStructure:
    """Copy of ``base`` with extra entries between labelled cells (e.g. ``C5``: DM of two silages).

    Missing labels are added with an identity row first.  ``None`` raises :class:`PendingValueError`.
    PSD is not enforced here (check with :func:`assert_psd`).
    """
    labels = list(base.labels)
    for (a, b) in pairs:
        for c in (tuple(a), tuple(b)):
            if c not in labels:
                labels.append(c)
    n = len(labels)
    r = np.eye(n)
    k = base.matrix.shape[0]
    r[:k, :k] = base.matrix
    for (a, b), v in pairs.items():
        if v is None:
            raise PendingValueError(f"{structure_id}: cross correlation {a}-{b} pending")
        i, j = labels.index(tuple(a)), labels.index(tuple(b))
        if i == j:
            raise ValueError("cross pair must join two different cells")
        r[i, j] = r[j, i] = float(v)
    return CorrelationStructure(structure_id, tuple(labels), r, status, provenance,
                                base.notes + (f"cross_pairs={len(pairs)}",))


# ----------------------------------------------------------------------------------------------
# Gaussian copula
# ----------------------------------------------------------------------------------------------

def sampling_factor(r: np.ndarray) -> np.ndarray:
    """``L`` with ``L L' = R`` via the symmetric eigendecomposition (works for singular PSD ``R``)."""
    r = np.asarray(r, dtype=float)
    w, v = np.linalg.eigh((r + r.T) / 2.0)
    if w[0] < -1e-8:
        raise NotPSDError(f"cannot factor a non-PSD matrix (lambda_min = {w[0]:.3g})")
    return v * np.sqrt(np.maximum(w, 0.0))


@dataclass(frozen=True, eq=False)
class GaussianCopulaModel(UncertaintyModel):
    """Joint draws of ``theta [S, I, J]`` and ``d [S, I]`` from moment-matched marginals + copula.

    ``theta_fits`` is an ``[I, J]`` object array (or nested sequence) of :class:`MarginalFit`,
    ``d_fits`` a length-``I`` sequence.  ``correlation`` (optional) covers a subset of the
    stochastic cells; all other stochastic cells are independent.  Every marginal must be
    ``matched``, ``point`` or ``missing`` (``drifted`` only with ``allow_drifted=True``, for
    reproducing the naive smoke behaviour); ``not_matchable`` or ``infeasible`` fits raise.  A
    non-PSD correlation raises :class:`NotPSDError` (no silent repair).

    Random numbers: one ``standard_normal((n, K))`` block for the ``K`` stochastic cells in the
    fixed order (ingredient-major; nutrients in ``nutrient_ids`` order, then ``DM``), multiplied by
    the correlation factor on the correlated subset.  Identity correlation reproduces the
    independent draws exactly.
    """

    model_id: str
    ingredient_ids: tuple
    nutrient_ids: tuple
    theta_fits: object
    d_fits: object
    correlation: Optional[CorrelationStructure] = None
    is_synthetic: bool = False
    allow_drifted: bool = False

    def __post_init__(self) -> None:
        ing, nut = tuple(self.ingredient_ids), tuple(self.nutrient_ids)
        object.__setattr__(self, "ingredient_ids", ing)
        object.__setattr__(self, "nutrient_ids", nut)
        I, J = len(ing), len(nut)
        tf = np.empty((I, J), dtype=object)
        src = np.asarray(self.theta_fits, dtype=object)
        if src.shape != (I, J):
            raise ValueError(f"theta_fits must have shape {(I, J)}, got {src.shape}")
        tf[...] = src
        df = np.empty(I, dtype=object)
        dsrc = list(self.d_fits)
        if len(dsrc) != I:
            raise ValueError("d_fits must have one fit per ingredient")
        for i, f in enumerate(dsrc):
            df[i] = f
        allowed = {"matched", "point", "missing"} | ({"drifted"} if self.allow_drifted else set())
        bad = []
        for idx in np.ndindex(I, J):
            f = tf[idx]
            if not isinstance(f, MarginalFit):
                raise TypeError("theta_fits must contain MarginalFit objects")
            if f.status not in allowed:
                bad.append(f"{ing[idx[0]]}:{nut[idx[1]]}={f.family}/{f.status}")
        for i in range(I):
            f = df[i]
            if not isinstance(f, MarginalFit):
                raise TypeError("d_fits must contain MarginalFit objects")
            if f.status not in allowed or f.status == "missing":
                bad.append(f"{ing[i]}:DM={f.family}/{f.status}")
        if bad:
            raise ValueError("GaussianCopulaModel: marginals not usable for sampling: " + "; ".join(bad))
        tf.setflags(write=False)
        df.setflags(write=False)
        object.__setattr__(self, "theta_fits", tf)
        object.__setattr__(self, "d_fits", df)
        cells = []
        for i in range(I):
            for j in range(J):
                if tf[i, j].is_stochastic:
                    cells.append((ing[i], nut[j]))
            if df[i].is_stochastic:
                cells.append((ing[i], "DM"))
        object.__setattr__(self, "_cells", tuple(cells))
        if self.correlation is not None:
            corr = self.correlation
            missing = [c for c in corr.labels if c not in cells]
            if missing:
                raise ValueError("correlation labels are not stochastic cells of the model: "
                                 + ", ".join(_label(c) for c in missing))
            assert_psd(corr)
            idx = np.array([cells.index(c) for c in corr.labels], dtype=int)
            object.__setattr__(self, "_corr_idx", idx)
            object.__setattr__(self, "_factor", sampling_factor(corr.matrix))
        else:
            object.__setattr__(self, "_corr_idx", np.zeros(0, dtype=int))
            object.__setattr__(self, "_factor", np.zeros((0, 0)))

    # ------------------------------------------------------------------------------------------
    @classmethod
    def from_targets(cls, model_id: str, ingredient_ids: Sequence[str], nutrient_ids: Sequence[str],
                     theta_mean: np.ndarray, theta_sd: np.ndarray, d_mean: np.ndarray, d_sd: np.ndarray, *,
                     family, theta_lower=0.0, theta_upper=1.0, d_lower=0.0, d_upper=1.0,
                     correlation: Optional[CorrelationStructure] = None, is_synthetic: bool = False,
                     allow_drifted: bool = False) -> "GaussianCopulaModel":
        """Fit every marginal with ``family`` (name or index mapping) and build the model."""
        tf = fit_array(family if isinstance(family, str) else family["theta"], theta_mean, theta_sd,
                       theta_lower, theta_upper)
        df = fit_array(family if isinstance(family, str) else family["d"], d_mean, d_sd, d_lower, d_upper)
        return cls(model_id, tuple(ingredient_ids), tuple(nutrient_ids), tf, list(df), correlation,
                   is_synthetic, allow_drifted)

    @property
    def stochastic_cells(self) -> tuple:
        """Labels of the stochastic cells in random-number order."""
        return self._cells

    def normal_scores(self, rng: np.random.Generator, n_draws: int) -> np.ndarray:
        """Correlated standard normal scores ``[S, K]`` for the stochastic cells."""
        n = int(n_draws)
        z = rng.standard_normal((n, len(self._cells)))
        if self._corr_idx.size:
            z[:, self._corr_idx] = z[:, self._corr_idx] @ self._factor.T
        return z

    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(n_draws)
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        z = self.normal_scores(rng, n)
        u = special.ndtr(z)
        theta = np.empty((n, I, J))
        d = np.empty((n, I))
        k = 0
        for i in range(I):
            for j in range(J):
                f = self.theta_fits[i, j]
                if f.is_stochastic:
                    theta[:, i, j] = f.ppf(u[:, k])
                    k += 1
                else:
                    theta[:, i, j] = f.ppf(np.zeros(n) + 0.5)
            f = self.d_fits[i]
            if f.is_stochastic:
                d[:, i] = f.ppf(u[:, k])
                k += 1
            else:
                d[:, i] = f.ppf(np.zeros(n) + 0.5)
        return theta, d

    def params_for_fingerprint(self) -> dict:
        return {"theta_fits": [f.fingerprint_params() for f in self.theta_fits.ravel()],
                "d_fits": [f.fingerprint_params() for f in self.d_fits],
                "correlation": None if self.correlation is None else self.correlation.fingerprint(),
                "allow_drifted": bool(self.allow_drifted)}

    def diagnostics(self) -> list[dict]:
        """One record per cell (family, status, moment errors, outside mass)."""
        out = []
        for i, iid in enumerate(self.ingredient_ids):
            for j, nid in enumerate(self.nutrient_ids):
                out.append({"ingredient_id": iid, "component": nid, **self.theta_fits[i, j].as_record()})
            out.append({"ingredient_id": iid, "component": "DM", **self.d_fits[i].as_record()})
        return out


def induced_pearson(model: GaussianCopulaModel, rng: np.random.Generator, n_draws: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """MC estimate of the Pearson correlation of the transformed values over the correlated cells.

    Returns ``(target R, estimated Pearson)`` for ``model.correlation.labels`` (diagnostic: the
    copula preserves the marginals, not the Pearson coefficients).
    """
    if model.correlation is None:
        raise ValueError("model has no correlation structure")
    theta, d = model.sample(rng, n_draws)
    cols = []
    for (iid, comp) in model.correlation.labels:
        i = model.ingredient_ids.index(iid)
        cols.append(d[:, i] if comp == "DM" else theta[:, i, model.nutrient_ids.index(comp)])
    x = np.column_stack(cols)
    return model.correlation.matrix.copy(), np.corrcoef(x, rowvar=False)
