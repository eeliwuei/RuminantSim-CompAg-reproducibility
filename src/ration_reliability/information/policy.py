"""Policy optimisation on a finite candidate library and fixed signal bins (contract T7.1-T7.3).

Notation (all arrays share one prior, one library, one risk event):

* ``pi_s`` prior state weights, ``L[s, z]`` signal likelihood, ``I[s, k]`` joint violation of
  candidate ``k`` in state ``s`` (public evaluator), ``C_k`` cost per head per day.
* ``P_z = sum_s pi_s L[s, z]``; ``R[z, k] = sum_s pi_s L[s, z] I[s, k] = P_z r_zk``.

With-information problem (T7.3, binary deterministic policies, ex-ante joint risk)::

    V_T = min  sum_z sum_k P_z C_k w_zk
          s.t. sum_k w_zk = 1                 for every bin z with P_z > 0
               sum_z sum_k R[z, k] w_zk <= alpha
               w_zk in {0, 1}

No-information problem: the same library and risk, one action for all bins (the constant
policy), ``V_0 = min_k {C_k : r_k <= alpha}`` with ``r_k = sum_s pi_s I[s, k]``.

Action sets (red-team fix C05/F02).  ``action_mask [K]`` is the action set of the no-information
decision (structurally feasible under the t0 DM estimate).  ``bin_action_mask [Z, K]`` (optional)
is the action set of the with-information policy in each bin (structurally feasible under that
bin's decision-time DM estimate, see :func:`~.library.decision_time_action_masks`); when omitted,
every bin uses ``action_mask``.  If every V0 action is legal in every active bin, every constant
policy is feasible in the with-information problem and ``V_T <= V_0`` (checked, never forced);
otherwise the two problems have different action spaces (:attr:`PolicyProblem.constant_policies_contained`).

Randomised references (``*_randomized_reference``) solve the LP relaxations (``w_zk in [0, 1]``,
mixtures of constant candidates for ``V_0``).  They are **not** the default policy class (T7.3:
"如研究允许随机政策，另行说明，默认不使用"); they are reported because under an ex-ante
(expectation) risk constraint, a deterministic signal-dependent policy can gain from the signal
acting as a randomisation device even when the signal carries no information about the risk
(see ``docs/INFORMATION_VALUE_DESIGN.md`` §4).

The per-outcome conditional-risk rule of T7.2 (``r_zk <= alpha`` in every bin) is a *different*
problem with a different feasible set; it is implemented separately in
:func:`per_outcome_conditional_risk_rule` and never reported as EVSI.

Bins with ``P_z = 0`` are not part of the optimisation; the frozen policy assigns them a
pre-declared fallback (the no-information choice) so that a later evaluation on independent
states always has an action.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..datamodel import SolverOptions, SolveStatus
from ..errors import InvalidProblemError
from ..hashing import stable_hash
from ..optimization.highs import run_linprog, run_milp, solver_version_string
from .library import RiskTable
from .likelihood import SignalLikelihood
from .prior import PriorStates

__all__ = [
    "RISK_TOL_DEFAULT",
    "PolicyProblem",
    "PolicyResult",
    "ConditionalRiskRuleResult",
    "solve_no_information",
    "solve_no_information_randomized",
    "solve_signal_policy",
    "solve_signal_policy_randomized",
    "per_outcome_conditional_risk_rule",
]

#: Absolute tolerance on the ex-ante risk (probability units), identical for V0 and VT.
RISK_TOL_DEFAULT = 1e-8


def _ro(a, dtype=float) -> np.ndarray:
    a = np.array(a, dtype=dtype)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class PolicyProblem:
    """Aggregated policy problem ``(P_z, R[z, k], C_k, r_k, alpha, action mask)``."""

    Pz: np.ndarray
    R: np.ndarray
    costs: np.ndarray
    r_marginal: np.ndarray
    alpha: float
    action_mask: np.ndarray
    risk_tol: float
    bin_labels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    provenance: dict = field(default_factory=dict)
    bin_action_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        Pz, R, C, r = _ro(self.Pz), _ro(self.R), _ro(self.costs), _ro(self.r_marginal)
        m = _ro(self.action_mask, bool)
        for n, a in (("Pz", Pz), ("R", R), ("costs", C), ("r_marginal", r), ("action_mask", m)):
            object.__setattr__(self, n, a)
        Z, K = R.shape if R.ndim == 2 else (-1, -1)
        if Pz.shape != (Z,) or C.shape != (K,) or r.shape != (K,) or m.shape != (K,):
            raise InvalidProblemError("PolicyProblem: inconsistent shapes")
        if self.bin_action_mask is not None:
            bm = _ro(self.bin_action_mask, bool)
            if bm.shape != (Z, K):
                raise InvalidProblemError("PolicyProblem: bin_action_mask must be [Z, K]")
            object.__setattr__(self, "bin_action_mask", bm)
        if len(self.bin_labels) != Z or len(self.candidate_ids) != K:
            raise InvalidProblemError("PolicyProblem: label lengths differ from shapes")
        if not (0.0 <= float(self.alpha) <= 1.0):
            raise InvalidProblemError("PolicyProblem: alpha must be in [0, 1]")
        if not m.any():
            raise InvalidProblemError("PolicyProblem: action mask excludes every candidate")
        if abs(float(Pz.sum()) - 1.0) > 1e-9 or np.any(Pz < 0):
            raise InvalidProblemError("PolicyProblem: P_z must be a probability vector")
        if np.any(R < -1e-15) or np.any(R > Pz[:, None] + 1e-12):
            raise InvalidProblemError("PolicyProblem: need 0 <= R[z,k] <= P_z")
        if np.any(np.abs(R.sum(axis=0) - r) > 1e-9):
            raise InvalidProblemError("PolicyProblem: sum_z R[z,k] must equal the marginal risk r_k")
        if not np.all(np.isfinite(C)):
            raise InvalidProblemError("PolicyProblem: non-finite costs")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_arrays(cls, weights: np.ndarray, L: np.ndarray, I: np.ndarray, costs: np.ndarray, alpha: float, *,
                    action_mask: Optional[np.ndarray] = None, risk_tol: float = RISK_TOL_DEFAULT,
                    bin_labels: Optional[Sequence[str]] = None, candidate_ids: Optional[Sequence[str]] = None,
                    provenance: Optional[dict] = None,
                    bin_action_mask: Optional[np.ndarray] = None) -> "PolicyProblem":
        """Aggregate state-level arrays ``pi [S]``, ``L [S, Z]``, ``I [S, K]``, ``C [K]``."""
        w = np.asarray(weights, dtype=float)
        L = np.asarray(L, dtype=float)
        I = np.asarray(I, dtype=float)
        C = np.asarray(costs, dtype=float)
        if L.ndim != 2 or I.ndim != 2 or w.shape != (L.shape[0],) or I.shape[0] != L.shape[0] or \
                C.shape != (I.shape[1],):
            raise InvalidProblemError("from_arrays: shapes must be pi[S], L[S,Z], I[S,K], C[K]")
        if np.any((I != 0) & (I != 1)):
            raise InvalidProblemError("from_arrays: I must be 0/1")
        if np.any(np.abs(L.sum(axis=1) - 1.0) > 1e-9) or np.any(L < 0):
            raise InvalidProblemError("from_arrays: L rows must be probability vectors")
        WL = w[:, None] * L
        Pz = WL.sum(axis=0)
        R = WL.T @ I
        r = w @ I
        Z, K = L.shape[1], I.shape[1]
        mask = np.ones(K, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        return cls(Pz, np.clip(R, 0.0, None), C, r, float(alpha), mask, float(risk_tol),
                   tuple(bin_labels) if bin_labels is not None else tuple(f"z{z}" for z in range(Z)),
                   tuple(candidate_ids) if candidate_ids is not None else tuple(f"k{k}" for k in range(K)),
                   dict(provenance or {}), bin_action_mask)

    @classmethod
    def build(cls, prior: PriorStates, likelihood: SignalLikelihood, risk: RiskTable, alpha: float, *,
              unknown_policy: str = "error_if_any", action_mask: Optional[np.ndarray] = None,
              risk_tol: float = RISK_TOL_DEFAULT, bin_action_mask: Optional[np.ndarray] = None) -> "PolicyProblem":
        """Policy problem for one information structure (same prior/library/risk for every structure)."""
        if likelihood.prior_fingerprint != prior.fingerprint or risk.prior_fingerprint != prior.fingerprint:
            raise InvalidProblemError("PolicyProblem.build: likelihood / risk table were built on another prior")
        I = risk.indicator(unknown_policy)
        prov = {"prior_fingerprint": prior.fingerprint, "prior_stream_id": prior.stream_id,
                "likelihood_kind": likelihood.kind, "likelihood_fingerprint": likelihood.fingerprint,
                "risk_table_fingerprint": risk.fingerprint, "library_fingerprint": risk.library_fingerprint,
                "unknown_policy": unknown_policy, "n_states": prior.n_states, "is_synthetic": risk.is_synthetic}
        return cls.from_arrays(prior.weights, likelihood.L, I, risk.costs, alpha, action_mask=action_mask,
                               risk_tol=risk_tol, bin_labels=likelihood.bin_labels, candidate_ids=risk.candidate_ids,
                               provenance=prov, bin_action_mask=bin_action_mask)

    # ------------------------------------------------------------------ helpers
    @property
    def n_bins(self) -> int:
        """Number of bins ``Z``."""
        return int(self.Pz.shape[0])

    @property
    def n_candidates(self) -> int:
        """Library size ``K``."""
        return int(self.costs.shape[0])

    @property
    def active_bins(self) -> np.ndarray:
        """Indices of bins with positive probability."""
        return np.flatnonzero(self.Pz > 0)

    @property
    def allowed(self) -> np.ndarray:
        """Indices of candidates allowed by the no-information action mask (V0's action set)."""
        return np.flatnonzero(self.action_mask)

    def vt_allowed(self) -> np.ndarray:
        """``[Z, K]`` action set of the with-information policy in every bin."""
        if self.bin_action_mask is None:
            return np.broadcast_to(self.action_mask, (self.n_bins, self.n_candidates))
        return self.bin_action_mask

    @property
    def constant_policies_contained(self) -> bool:
        """True if every V0 action is legal in every active bin (T7.1 containment premise)."""
        za = self.active_bins
        ka = self.allowed
        if za.size == 0 or ka.size == 0:
            return True
        return bool(np.all(self.vt_allowed()[np.ix_(za, ka)]))

    def conditional_risk(self) -> np.ndarray:
        """``r_zk = R[z, k] / P_z`` (NaN for bins with ``P_z = 0``)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.Pz[:, None] > 0, self.R / self.Pz[:, None], np.nan)

    def policy_cost_risk(self, assignment: np.ndarray) -> tuple[float, float]:
        """Exact expected cost and ex-ante risk of a deterministic assignment ``[Z]``.

        Inactive bins (``P_z = 0``) contribute nothing whatever their assignment.
        """
        a = np.asarray(assignment, dtype=int)
        za = self.active_bins
        if np.any(a[za] < 0):
            raise InvalidProblemError("policy_cost_risk: active bin without action")
        cost = float(np.sum(self.Pz[za] * self.costs[a[za]]))
        risk = float(np.sum(self.R[za, a[za]]))
        return cost, risk

    @property
    def fingerprint(self) -> str:
        """Content hash."""
        return stable_hash("PolicyProblem/v2", self.Pz, self.R, self.costs, self.r_marginal, float(self.alpha),
                           self.action_mask, float(self.risk_tol), self.bin_labels, self.candidate_ids,
                           self.bin_action_mask)


@dataclass(frozen=True)
class PolicyResult:
    """Outcome of one policy optimisation.

    ``expected_cost`` (currency/head/d) and ``ex_ante_risk`` are recomputed exactly from the
    returned policy; they are ``None`` for failed statuses (no zero-cost placeholder).
    """

    method: str
    status: str
    expected_cost: Optional[float]
    ex_ante_risk: Optional[float]
    alpha: float
    risk_tol: float
    assignment: Optional[np.ndarray] = None       # deterministic: candidate index per bin (-1 = none)
    w: Optional[np.ndarray] = None                # randomised signal policy [Z, K]
    mixture: Optional[np.ndarray] = None          # randomised constant policy [K]
    mip_gap: Optional[float] = None
    solver: str = ""
    solver_version: str = ""
    wall_time_s: float = 0.0
    message: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = self.status in (SolveStatus.OPTIMAL.value, SolveStatus.FEASIBLE_TIME_LIMIT.value)
        if ok and (self.expected_cost is None or self.ex_ante_risk is None):
            raise InvalidProblemError(f"PolicyResult: status {self.status} needs cost and risk")
        if not ok and self.expected_cost is not None:
            raise InvalidProblemError(f"PolicyResult: status {self.status} must not carry a cost")

    @property
    def has_solution(self) -> bool:
        """True for optimal / feasible_time_limit."""
        return self.expected_cost is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        def c(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, dict):
                return {str(k): c(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [c(x) for x in v]
            return v
        return {"method": self.method, "status": self.status, "expected_cost": self.expected_cost,
                "ex_ante_risk": self.ex_ante_risk, "alpha": self.alpha, "risk_tol": self.risk_tol,
                "assignment": c(self.assignment), "w": c(self.w), "mixture": c(self.mixture),
                "mip_gap": self.mip_gap, "solver": self.solver, "solver_version": self.solver_version,
                "wall_time_s": self.wall_time_s, "message": self.message, "diagnostics": c(self.diagnostics)}


def _fail(method: str, status: SolveStatus, pp: PolicyProblem, msg: str, t0: float, **kw) -> PolicyResult:
    return PolicyResult(method, status.value, None, None, float(pp.alpha), float(pp.risk_tol), message=msg,
                        wall_time_s=time.perf_counter() - t0, **kw)


# ---------------------------------------------------------------------------------------------
# no-information (constant) policies
# ---------------------------------------------------------------------------------------------

def solve_no_information(pp: PolicyProblem) -> PolicyResult:
    """``V_0``: cheapest allowed candidate with ``r_k <= alpha + risk_tol`` (exact enumeration).

    Ties: lower risk, then lower index.  ``proven_infeasible`` (within the library) if no candidate
    meets the risk target -- never an empty ration or zero cost.
    """
    t0 = time.perf_counter()
    ks = pp.allowed
    feas = ks[pp.r_marginal[ks] <= pp.alpha + pp.risk_tol]
    if feas.size == 0:
        return _fail("constant_deterministic", SolveStatus.PROVEN_INFEASIBLE, pp,
                     "no allowed candidate meets the ex-ante risk target (within the library)", t0,
                     diagnostics={"min_risk_in_library": float(pp.r_marginal[ks].min())})
    order = np.lexsort((feas, pp.r_marginal[feas], pp.costs[feas]))
    k = int(feas[order[0]])
    a = np.full(pp.n_bins, k, dtype=int)
    return PolicyResult("constant_deterministic", SolveStatus.OPTIMAL.value, float(pp.costs[k]),
                        float(pp.r_marginal[k]), float(pp.alpha), float(pp.risk_tol), assignment=a,
                        solver="exact enumeration over K", wall_time_s=time.perf_counter() - t0,
                        diagnostics={"chosen_candidate": pp.candidate_ids[k], "chosen_index": k,
                                     "n_feasible_candidates": int(feas.size)})


#: Tighter LP tolerances for the randomised references (costs are compared to ~1e-9).
LP_REFERENCE_OPTIONS = SolverOptions(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10)


def solve_no_information_randomized(pp: PolicyProblem, solver_options: Optional[SolverOptions] = None) -> PolicyResult:
    """Reference: best *mixture* of constant candidates (LP).  Not the default policy class."""
    t0 = time.perf_counter()
    ks = pp.allowed
    n = ks.size
    scale = float(pp.alpha) if pp.alpha > 0 else 1.0
    out = run_linprog(pp.costs[ks], pp.r_marginal[ks][None, :] / scale, np.array([pp.alpha / scale]),
                      np.ones((1, n)), np.array([1.0]), np.zeros(n), np.ones(n), solver_options or LP_REFERENCE_OPTIONS)
    common = dict(solver="scipy.optimize.linprog(method='highs')", solver_version=solver_version_string())
    if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT):
        return _fail("constant_randomized_reference", out.status, pp, out.message, t0, **common)
    lam = np.zeros(pp.n_candidates)
    lam[ks] = np.clip(out.x, 0.0, None)
    lam /= lam.sum()
    risk = float(pp.r_marginal @ lam)
    if risk > pp.alpha + pp.risk_tol:
        return _fail("constant_randomized_reference", SolveStatus.NUMERICAL_ERROR, pp,
                     f"LP mixture violates the risk target by {risk - pp.alpha:.3g}; not repaired", t0,
                     diagnostics={"raw_mixture": lam.tolist()}, **common)
    return PolicyResult("constant_randomized_reference", out.status.value, float(pp.costs @ lam), risk,
                        float(pp.alpha), float(pp.risk_tol), mixture=lam, wall_time_s=time.perf_counter() - t0,
                        message=out.message, **common)


# ---------------------------------------------------------------------------------------------
# signal-dependent policies
# ---------------------------------------------------------------------------------------------

def _fallback_assignment(pp: PolicyProblem, active_choice: dict[int, int], fallback: Optional[int]) -> np.ndarray:
    a = np.full(pp.n_bins, -1 if fallback is None else int(fallback), dtype=int)
    for z, k in active_choice.items():
        a[z] = k
    return a


def _bins_without_candidate(pp: PolicyProblem) -> list[int]:
    M = pp.vt_allowed()
    return [int(z) for z in pp.active_bins if not M[z].any()]


def _check_assignment_legal(pp: PolicyProblem, a: np.ndarray) -> list[int]:
    M = pp.vt_allowed()
    return [int(z) for z in pp.active_bins if not M[z, int(a[z])]]


def solve_signal_policy(pp: PolicyProblem, *, method: str = "milp", solver_options: Optional[SolverOptions] = None,
                        max_combinations: int = 200_000) -> PolicyResult:
    """``V_T``: optimal deterministic bin -> candidate policy (T7.3 binary problem).

    ``method="milp"`` uses HiGHS (``mip_rel_gap`` defaults to 0 here; the achieved gap is
    recorded); ``method="enumeration"`` enumerates all policies over the per-bin action sets
    (small problems only; used as the exact check).  Bins with ``P_z = 0`` receive the
    no-information choice as a pre-declared fallback (irrelevant to the value).  An active bin
    whose action set is empty makes the problem ``proven_infeasible`` within the library.
    """
    t0 = time.perf_counter()
    za = pp.active_bins
    M = pp.vt_allowed()
    v0 = solve_no_information(pp)
    fallback = v0.diagnostics.get("chosen_index") if v0.has_solution else None
    tag = f"signal_deterministic_{method}"
    if method not in ("milp", "enumeration"):
        raise InvalidProblemError(f"unknown method {method!r} (milp | enumeration)")
    empty = _bins_without_candidate(pp)
    if empty:
        return _fail(tag, SolveStatus.PROVEN_INFEASIBLE, pp,
                     f"{len(empty)} active bin(s) have no structurally feasible candidate under their decision-time "
                     "information (within the library)", t0,
                     diagnostics={"bins_without_candidate": [pp.bin_labels[z] for z in empty],
                                  "p_bins_without_candidate": float(np.sum(pp.Pz[empty]))})
    if method == "enumeration":
        lists = [np.flatnonzero(M[z]) for z in za]
        n_comb = float(np.prod([float(len(x)) for x in lists])) if lists else 1.0
        if n_comb > max_combinations:
            raise InvalidProblemError(f"enumeration of {n_comb:.3g} policies exceeds max_combinations={max_combinations}")
        best = None
        for combo in itertools.product(*lists):
            ks = np.asarray(combo, dtype=int)
            cost = float(np.sum(pp.Pz[za] * pp.costs[ks]))
            risk = float(np.sum(pp.R[za, ks]))
            if risk <= pp.alpha + pp.risk_tol:
                key = (cost, risk)
                if best is None or key < best[0]:
                    best = (key, ks.copy())
        if best is None:
            return _fail(tag, SolveStatus.PROVEN_INFEASIBLE, pp, "no feasible policy (exhaustive enumeration)", t0,
                         solver="exhaustive enumeration")
        a = _fallback_assignment(pp, {int(z): int(k) for z, k in zip(za, best[1])}, fallback)
        cost, risk = pp.policy_cost_risk(a)
        return PolicyResult(tag, SolveStatus.OPTIMAL.value, cost, risk, float(pp.alpha), float(pp.risk_tol),
                            assignment=a, solver="exhaustive enumeration", wall_time_s=time.perf_counter() - t0,
                            diagnostics={"n_policies_enumerated": int(n_comb), "fallback_candidate": fallback})
    ka = np.flatnonzero(M[za].any(axis=0))
    Zn, Kn = za.size, ka.size
    c = (pp.Pz[za][:, None] * pp.costs[ka][None, :]).ravel()
    scale = float(pp.alpha) if pp.alpha > 0 else 1.0
    A_ub = (pp.R[np.ix_(za, ka)] / scale).reshape(1, -1)
    b_ub = np.array([pp.alpha / scale])
    A_eq = np.zeros((Zn, Zn * Kn))
    for i in range(Zn):
        A_eq[i, i * Kn:(i + 1) * Kn] = 1.0
    ub = M[np.ix_(za, ka)].astype(float).ravel()          # 0 = not a legal action in that bin
    opts = solver_options or SolverOptions(mip_rel_gap=0.0)
    out = run_milp(c, A_ub, b_ub, A_eq, np.ones(Zn), np.zeros(Zn * Kn), ub, np.ones(Zn * Kn), opts)
    diag = {"n_active_bins": int(Zn), "n_allowed_candidates": int(Kn), "n_binaries": int(Zn * Kn),
            "n_binaries_fixed_to_zero_by_action_sets": int(np.sum(ub == 0)),
            "highs_model_status": out.highs_model_status, "mip_node_count": out.extra.get("mip_node_count"),
            "fallback_candidate": fallback, "risk_row_scale": scale}
    common = dict(solver="scipy.optimize.milp (HiGHS)", solver_version=solver_version_string(), mip_gap=out.mip_gap)
    if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT):
        diag["raw_candidate_x"] = out.extra.get("raw_x")
        return _fail(tag, out.status, pp, out.message, t0, diagnostics=diag, **common)
    W = out.x.reshape(Zn, Kn)
    choice = {int(za[i]): int(ka[int(np.argmax(W[i]))]) for i in range(Zn)}
    a = _fallback_assignment(pp, choice, fallback)
    cost, risk = pp.policy_cost_risk(a)
    diag["lp_objective"] = out.fun
    illegal = _check_assignment_legal(pp, a)
    if illegal:
        return _fail(tag, SolveStatus.NUMERICAL_ERROR, pp,
                     f"returned policy uses an illegal action in {len(illegal)} bin(s); not repaired", t0,
                     diagnostics=diag, **common)
    if risk > pp.alpha + pp.risk_tol:
        diag["exact_risk_of_returned_policy"] = risk
        return _fail(tag, SolveStatus.NUMERICAL_ERROR, pp,
                     f"returned policy violates the risk target by {risk - pp.alpha:.3g} (> risk_tol); not repaired",
                     t0, diagnostics=diag, **common)
    return PolicyResult(tag, out.status.value, cost, risk, float(pp.alpha), float(pp.risk_tol), assignment=a,
                        wall_time_s=time.perf_counter() - t0, message=out.message, diagnostics=diag, **common)


def solve_signal_policy_randomized(pp: PolicyProblem, solver_options: Optional[SolverOptions] = None) -> PolicyResult:
    """Reference: LP relaxation (randomised bin -> candidate policies).  Not the default class."""
    t0 = time.perf_counter()
    za = pp.active_bins
    M = pp.vt_allowed()
    common = dict(solver="scipy.optimize.linprog(method='highs')", solver_version=solver_version_string())
    empty = _bins_without_candidate(pp)
    if empty:
        return _fail("signal_randomized_reference", SolveStatus.PROVEN_INFEASIBLE, pp,
                     f"{len(empty)} active bin(s) have no structurally feasible candidate", t0, **common)
    ka = np.flatnonzero(M[za].any(axis=0))
    Zn, Kn = za.size, ka.size
    c = (pp.Pz[za][:, None] * pp.costs[ka][None, :]).ravel()
    scale = float(pp.alpha) if pp.alpha > 0 else 1.0
    A_ub = (pp.R[np.ix_(za, ka)] / scale).reshape(1, -1)
    A_eq = np.zeros((Zn, Zn * Kn))
    for i in range(Zn):
        A_eq[i, i * Kn:(i + 1) * Kn] = 1.0
    ub = M[np.ix_(za, ka)].astype(float).ravel()
    out = run_linprog(c, A_ub, np.array([pp.alpha / scale]), A_eq, np.ones(Zn), np.zeros(Zn * Kn),
                      ub, solver_options or LP_REFERENCE_OPTIONS)
    if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT):
        return _fail("signal_randomized_reference", out.status, pp, out.message, t0, **common)
    W = np.zeros((pp.n_bins, pp.n_candidates))
    W[np.ix_(za, ka)] = np.clip(out.x.reshape(Zn, Kn), 0.0, None) * M[np.ix_(za, ka)]
    W[za] /= W[za].sum(axis=1, keepdims=True)
    cost = float(np.sum(pp.Pz[:, None] * pp.costs[None, :] * W))
    risk = float(np.sum(pp.R * W))
    if risk > pp.alpha + pp.risk_tol:
        return _fail("signal_randomized_reference", SolveStatus.NUMERICAL_ERROR, pp,
                     f"LP policy violates the risk target by {risk - pp.alpha:.3g}; not repaired", t0, **common)
    return PolicyResult("signal_randomized_reference", out.status.value, cost, risk, float(pp.alpha),
                        float(pp.risk_tol), w=W, wall_time_s=time.perf_counter() - t0, message=out.message,
                        diagnostics={"n_fractional_bins": int(np.sum(np.max(W[za], axis=1) < 1 - 1e-9))}, **common)


# ---------------------------------------------------------------------------------------------
# T7.2: per-outcome conditional risk rule (a different problem; never an EVSI)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionalRiskRuleResult:
    """Per-outcome rule: in every signal bin choose the cheapest candidate with ``r_zk <= alpha``.

    This changes the feasible set compared with the ex-ante problem (a constant ration that is
    feasible ex ante need not be feasible in every bin), so the classical non-negativity of
    ``V_0 - V_T`` does not apply.  ``cost_difference_vs_no_information_rule`` may be negative and is
    **not** an information value (T7.2, F04).
    """

    rule: str
    status: str
    assignment: np.ndarray
    conditional_risk_of_choice: np.ndarray
    p_bins_without_feasible_candidate: float
    expected_cost: Optional[float]
    ex_ante_risk_of_rule: float
    no_information_rule_cost: Optional[float]
    cost_difference_vs_no_information_rule: Optional[float]
    is_information_value: bool = False
    notes: str = ("per-outcome conditional-risk rule (contract T7.2); feasible set differs from the ex-ante problem; "
                  "not an EVSI and not to be placed in the EVSI table")


def per_outcome_conditional_risk_rule(pp: PolicyProblem) -> ConditionalRiskRuleResult:
    """Apply the T7.2 per-outcome rule to a :class:`PolicyProblem` (same library and bins)."""
    rz = pp.conditional_risk()
    a = np.full(pp.n_bins, -1, dtype=int)
    rc = np.full(pp.n_bins, np.nan)
    p_inf = 0.0
    M = pp.vt_allowed()
    for z in pp.active_bins:
        legal = np.flatnonzero(M[z])
        ks = legal[rz[z, legal] <= pp.alpha + pp.risk_tol]
        if ks.size == 0:
            p_inf += float(pp.Pz[z])
            continue
        order = np.lexsort((ks, rz[z, ks], pp.costs[ks]))
        k = int(ks[order[0]])
        a[z] = k
        rc[z] = rz[z, k]
    act = pp.active_bins
    feas = act[a[act] >= 0]
    cost = float(np.sum(pp.Pz[feas] * pp.costs[a[feas]])) if p_inf == 0.0 else None
    risk = float(np.sum(pp.R[feas, a[feas]]))
    v0 = solve_no_information(pp)
    base = v0.expected_cost
    diff = (base - cost) if (base is not None and cost is not None) else None
    status = "all_active_bins_feasible" if p_inf == 0.0 else "some_bins_without_feasible_candidate"
    return ConditionalRiskRuleResult("per_outcome_conditional_risk_rule_T7_2", status, a, rc, p_inf, cost, risk,
                                     base, diff)
