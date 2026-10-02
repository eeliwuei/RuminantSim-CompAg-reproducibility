"""Policy-model properties on random small instances (synthetic arrays; no nutrition data).

* MILP optimum = exhaustive enumeration (small problems);
* constant-policy containment: V_T <= V_0, hence gross value >= 0 (deterministic and randomised);
* randomised relaxation is never more expensive than the deterministic policy;
* partition refinement (perfect information): finer partitions never increase V_T;
* garbling (randomised class): a noisy signal of an exactly observed partition never beats it;
* infeasible libraries, action masks, unknown outcomes and result invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from ration_reliability.errors import InvalidProblemError
from ration_reliability.information import (
    PolicyProblem,
    PolicyResult,
    RiskTable,
    per_outcome_conditional_risk_rule,
    solve_no_information,
    solve_no_information_randomized,
    solve_signal_policy,
    solve_signal_policy_randomized,
)

TOL = 1e-9


def _instance(rng, S=8, Z=3, K=4, alpha=None, noisy=True):
    w = rng.dirichlet(np.ones(S))
    if noisy:
        L = rng.dirichlet(np.ones(Z) * 0.7, size=S)
    else:
        L = np.eye(Z)[rng.integers(0, Z, size=S)]
    I = (rng.random((S, K)) < rng.uniform(0.1, 0.7, size=K)[None, :]).astype(float)
    I[:, 0] = 0.0                                   # one always-safe (expensive) candidate
    C = np.sort(rng.uniform(1.0, 3.0, size=K))[::-1]  # safe candidate most expensive
    a = float(rng.uniform(0.02, 0.4)) if alpha is None else alpha
    return w, L, I, C, a


@pytest.mark.parametrize("seed", range(25))
def test_milp_equals_enumeration_and_containment(seed):
    rng = np.random.default_rng(1000 + seed)
    w, L, I, C, a = _instance(rng, S=int(rng.integers(3, 10)), Z=int(rng.integers(2, 4)), K=int(rng.integers(2, 5)))
    pp = PolicyProblem.from_arrays(w, L, I, C, a)
    v0 = solve_no_information(pp)
    m = solve_signal_policy(pp, method="milp")
    e = solve_signal_policy(pp, method="enumeration")
    assert v0.has_solution                                         # candidate 0 is always safe
    assert m.status == e.status == "optimal"
    assert m.expected_cost == pytest.approx(e.expected_cost, abs=1e-9)
    assert m.ex_ante_risk <= a + pp.risk_tol and e.ex_ante_risk <= a + pp.risk_tol
    assert m.expected_cost <= v0.expected_cost + TOL                # containment -> gross value >= 0
    r0 = solve_no_information_randomized(pp)
    rt = solve_signal_policy_randomized(pp)
    assert rt.expected_cost <= r0.expected_cost + 1e-8              # randomised containment
    assert rt.expected_cost <= m.expected_cost + 1e-8               # relaxation <= binary optimum
    assert r0.expected_cost <= v0.expected_cost + 1e-8


@pytest.mark.parametrize("seed", range(15))
def test_partition_refinement_never_increases_vt(seed):
    """Perfect information: a finer partition (full state identity) is at least as valuable."""
    rng = np.random.default_rng(2000 + seed)
    S, K = int(rng.integers(4, 9)), int(rng.integers(2, 5))
    w, _, I, C, a = _instance(rng, S=S, Z=2, K=K)
    coarse = rng.integers(0, 3, size=S)
    Lc = np.eye(3)[coarse]
    Lf = np.eye(S)                                                  # each state its own bin
    for L1, L2 in ((Lc, Lf),):
        vc = solve_signal_policy(PolicyProblem.from_arrays(w, L1, I, C, a))
        vf = solve_signal_policy(PolicyProblem.from_arrays(w, L2, I, C, a))
        assert vf.expected_cost <= vc.expected_cost + TOL
        rc = solve_signal_policy_randomized(PolicyProblem.from_arrays(w, L1, I, C, a))
        rf = solve_signal_policy_randomized(PolicyProblem.from_arrays(w, L2, I, C, a))
        assert rf.expected_cost <= rc.expected_cost + 1e-8


@pytest.mark.parametrize("seed", range(15))
def test_garbling_never_beats_exact_partition_randomized_class(seed):
    """L_noisy = L_exact @ M (M row-stochastic): randomised V_T(noisy) >= V_T(exact)."""
    rng = np.random.default_rng(3000 + seed)
    S, G, Z, K = int(rng.integers(4, 9)), 3, 3, int(rng.integers(2, 5))
    w, _, I, C, a = _instance(rng, S=S, Z=G, K=K)
    Le = np.eye(G)[rng.integers(0, G, size=S)]
    M = rng.dirichlet(np.ones(Z), size=G)
    Ln = Le @ M
    ve = solve_signal_policy_randomized(PolicyProblem.from_arrays(w, Le, I, C, a))
    vn = solve_signal_policy_randomized(PolicyProblem.from_arrays(w, Ln, I, C, a))
    assert vn.expected_cost >= ve.expected_cost - 1e-8


def test_uninformative_signal_randomized_value_is_exactly_zero():
    rng = np.random.default_rng(7)
    for _ in range(10):
        w, _, I, C, a = _instance(rng, S=6, Z=3, K=4)
        p = rng.dirichlet(np.ones(3))
        L = np.tile(p, (6, 1))
        pp = PolicyProblem.from_arrays(w, L, I, C, a)
        assert solve_signal_policy_randomized(pp).expected_cost == \
            pytest.approx(solve_no_information_randomized(pp).expected_cost, abs=1e-8)


def test_no_feasible_candidate_is_reported_not_zero_cost():
    w = np.array([0.5, 0.5])
    I = np.array([[1.0, 0.0], [0.0, 1.0]])                         # each candidate fails in one state
    C = np.array([1.0, 2.0])
    pp = PolicyProblem.from_arrays(w, np.array([[1.0], [1.0]]), I, C, 0.1)
    v0 = solve_no_information(pp)
    assert v0.status == "proven_infeasible" and v0.expected_cost is None and v0.assignment is None
    assert solve_signal_policy(pp).status == "proven_infeasible"
    pp2 = PolicyProblem.from_arrays(w, np.eye(2), I, C, 0.5)
    assert solve_no_information(pp2).status == "optimal"          # r_0 = 0.5 <= 0.5
    # perfect information makes the target reachable (state 0 -> 1, state 1 -> 0); V0 has no finite value
    pp3 = PolicyProblem.from_arrays(w, np.eye(2), I, C, 0.1)
    assert solve_no_information(pp3).status == "proven_infeasible"
    vt = solve_signal_policy(pp3)
    assert vt.status == "optimal" and vt.expected_cost == pytest.approx(0.5 * 2.0 + 0.5 * 1.0)
    assert vt.ex_ante_risk == pytest.approx(0.0)
    with pytest.raises(InvalidProblemError):
        PolicyResult("x", "proven_infeasible", 1.0, 0.0, 0.1, 1e-8)   # failure cannot carry a cost


def test_action_mask_applies_to_both_sides():
    rng = np.random.default_rng(11)
    w, L, I, C, a = _instance(rng, S=6, Z=2, K=4, alpha=0.3)
    mask = np.array([True, False, True, True])
    pp = PolicyProblem.from_arrays(w, L, I, C, a, action_mask=mask)
    v0 = solve_no_information(pp)
    vt = solve_signal_policy(pp)
    assert v0.diagnostics["chosen_index"] != 1
    assert 1 not in set(vt.assignment[pp.active_bins].tolist())
    assert vt.expected_cost <= v0.expected_cost + TOL
    with pytest.raises(InvalidProblemError):
        PolicyProblem.from_arrays(w, L, I, C, a, action_mask=np.zeros(4, dtype=bool))


def test_inactive_bins_get_the_declared_fallback():
    w = np.array([0.5, 0.5])
    L = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])               # bin 2 never occurs
    I = np.array([[1.0, 0.0], [0.0, 0.0]])
    pp = PolicyProblem.from_arrays(w, L, I, np.array([1.0, 2.0]), 0.05)
    vt = solve_signal_policy(pp)
    v0 = solve_no_information(pp)
    assert vt.assignment[2] == v0.diagnostics["chosen_index"] == 1
    assert vt.diagnostics["fallback_candidate"] == 1


def test_unknown_outcomes_require_an_explicit_policy():
    viol = np.array([[False, True], [False, False]])
    unk = np.array([[True, False], [False, False]])
    rt = RiskTable(viol, unk, np.array([1.0, 2.0]), ("a", "b"), "p", "l", "c", True)
    with pytest.raises(InvalidProblemError, match="undefined"):
        rt.indicator()
    assert rt.indicator("count_as_violation").tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert rt.indicator("count_as_satisfied").tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_policy_problem_validates_inputs():
    w = np.array([0.5, 0.5])
    with pytest.raises(InvalidProblemError):
        PolicyProblem.from_arrays(w, np.array([[0.7, 0.2], [0.5, 0.5]]), np.zeros((2, 2)), np.ones(2), 0.1)
    with pytest.raises(InvalidProblemError):
        PolicyProblem.from_arrays(w, np.eye(2), np.full((2, 2), 0.5), np.ones(2), 0.1)
    with pytest.raises(InvalidProblemError):
        PolicyProblem.from_arrays(w, np.eye(2), np.zeros((2, 2)), np.ones(2), 1.5)
    with pytest.raises(InvalidProblemError, match="max_combinations"):
        solve_signal_policy(PolicyProblem.from_arrays(np.full(40, 1 / 40), np.eye(40), np.zeros((40, 3)), np.ones(3),
                                                      0.1), method="enumeration", max_combinations=1000)


def test_conditional_rule_bins_without_feasible_candidate_are_reported():
    w = np.array([0.5, 0.5])
    I = np.array([[1.0, 1.0], [0.0, 0.0]])                          # state 0: every candidate violates
    pp = PolicyProblem.from_arrays(w, np.eye(2), I, np.array([1.0, 2.0]), 0.6)
    rule = per_outcome_conditional_risk_rule(pp)
    assert rule.status == "some_bins_without_feasible_candidate"
    assert rule.p_bins_without_feasible_candidate == pytest.approx(0.5)
    assert rule.expected_cost is None and rule.cost_difference_vs_no_information_rule is None
    assert rule.assignment[0] == -1
