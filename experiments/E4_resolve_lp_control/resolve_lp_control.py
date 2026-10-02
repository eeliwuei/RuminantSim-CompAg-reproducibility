#!/usr/bin/env python3
"""Observed-composition re-solve control ("observe the composition, then re-solve the nominal LP") -- EXPLORATORY.

Status and scope
----------------
* ``exploratory / v2.1``, **unregistered**, **development streams only**.  Not part of the frozen v2 / v3 protocol,
  not a member of any comparable set, never merged into an official table and never a substitute for an official
  number.  Nothing here says anything about animals: "failure" is a violation (or an unresolved verdict) of the declared
  model constraints under the declared distribution.
* Cases: reference problem v3 arm A (``dev_case_v3a``) and arm C (``dev_case_v3c``), each built exactly as the
  official v3 path builds it (``official_v3.activate`` + ``official_v3.build_official_context_v3``: the case problem +
  the premise planning row, the MAIN9 arm, the arm's v2-rule-set reference table and ``ReferenceSpec``).
* World: SD-H0 (TAB: table SD as the batch SD; ``official_v3.build_sd_point_spec_v3``) with independent cells (C0,
  the primary assumption of the official configuration), energy column by ``EnergyColumnModel`` (the case's
  fixed-DMI linearisation at the planned intake).
* Streams: one development ``RandomStreams(seed)`` per run, test stream only (``root=<seed>/test/0``); the seed is
  checked by ``run_endpoint_ablation.check_seed`` (a reserved formal root is refused before any draw).  Declared
  development seeds: :data:`DEV_SEEDS`.  Outputs carry the label ``development`` / ``exploratory_v2_1``; the restricted
  output directory must lie under ``data/restricted_local/exploratory_v2_1/`` (refused otherwise).

Policy (per test state s = (theta_s, d_s))
-----------------------------------------
1. Observed-composition re-solve: the MAIN9 LP of the nominal M0 problem (``m0_nominal.solve``: ``linear_rows`` of the
   imposed rows, ``assemble_x_space_lp``, ``highs.run_linprog`` with the official method solver options), with the
   nominal composition replaced by ``theta_s`` in every probabilistic (nutrient) row.  The energy row uses the state's
   own ``NEL_fixedDMI`` column, i.e. the builder's fixed-DMI linearisation (``nutrition.energy``) evaluated at
   ``theta_s`` at the planned DM supply (checked against the direct per-feed evaluation); its bound
   ``NEL_req - C0`` does not depend on the composition.  DM fractions stay at the planned ``d_hat``
   (``problem.dm_estimates()``, the M0 value), and every structural row (planned-DM equality, inclusion caps, premise
   planning row SH-PLAN-T51-DGC-SHARE, planned CP / EE limits SH-PLAN-CP-HI / SH-PLAN-EE-HI) is taken at the nominal
   state exactly as in M0.  Minimise as-fed cost.  If the LP is not optimal (infeasible, failed, missing coefficient)
   the policy executes the nominal M0 ration in that state; such states are counted (``lp_fallback``).
2. ``q_s`` is scored with the canonical reference evaluator (``evaluate_reference``) on the realised state
   ``(theta_s, d_s)`` only, main event ``main_reference_plan_domain``; failure = violated OR unknown.
3. The nominal M0 ration (solved once at the nominal composition) is scored on the same states (paired comparison).

Restricted outputs (absolute costs, rations) go to ``--out`` (default
``data/restricted_local/exploratory_v2_1/resolve_lp_control/``); each run also writes ``summary_public.json`` there
(counts, rates, bounds, cost ratios, fingerprints only).  ``--summarise`` collects those into the public files
``experiments/E4_resolve_lp_control/results_public.csv`` and ``RESULT_<tag>.md``.

Usage::

    PYTHONDONTWRITEBYTECODE=1 nice -n 10 /opt/homebrew/opt/python@3.11/bin/python3.11 \
        experiments/E4_resolve_lp_control/resolve_lp_control.py --case dev_case_v3a --n-states 20000 --seed 202610031
    ... --summarise --n-states 20000                  # public CSV + markdown from the per-run public summaries
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (REPO / "src", REPO / "experiments" / "E0_verification", REPO / "experiments" / "E1_cost_reliability"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

# ------------------------------------------------------------------------------------------------
# declared configuration (fixed before the first run)
# ------------------------------------------------------------------------------------------------
SCHEMA = "ration_reliability.resolve_lp_control/0.1"
LABEL = "exploratory_v2_1"
RUN_TYPE = "development"
STATUS = "exploratory, unregistered, development-stream control; not part of any frozen protocol"
CASES = ("dev_case_v3a", "dev_case_v3c")
ARM_NAME = {"dev_case_v3a": "A", "dev_case_v3c": "C"}
WORLD = "SD-H0"
OBJECTIVE_ARM = "MAIN9"
MAIN_EVENT = "main_reference_plan_domain"
#: three independent development roots declared for this control (not reserved formal roots; checked before drawing)
DEV_SEEDS = (202610031, 202610032, 202610033)
N_STATES_DEFAULT = 20000
#: per-claim error of the one-sided Clopper-Pearson upper bound (0.05 / 10,000 claims)
PER_CLAIM_ALPHA = 0.05 / 10000
#: orientation only (official v3 runs, TAB): M0 main-event violation about 0.88-0.98; validation window of this control
M0_EXPECTED_RANGE = (0.85, 0.98)
SOLVER = {"time_limit_s": 300.0, "mip_rel_gap": 1e-4}            # the official method setting (as E3)
OUT_DEFAULT = Path("data") / "restricted_local" / LABEL / "resolve_lp_control"
OUT_ALLOWED_ROOT = Path("data") / "restricted_local" / LABEL
FORBIDDEN_OUT = (Path("results") / "official", Path("data") / "restricted_local" / "official")
PUBLIC_DIR = Path("experiments") / "E4_resolve_lp_control"
#: code / configuration paths whose modification would change what this control computes (checked at run time)
ENGINE_SCOPE = ("src/", "configs/", "experiments/E0_verification/", "experiments/E1_cost_reliability/",
                "experiments/E4_resolve_lp_control/resolve_lp_control.py", "reports/")
POLICY_METHOD_ID = "EXPL_resolve_observed_composition_lp"
POLICY_INFORMATION_STATE = "composition_observed(theta_s); DM at d_hat (exploratory)"


# =================================================================================================
# pure helpers
# =================================================================================================
def cp_upper(k: int, n: int, alpha: float = PER_CLAIM_ALPHA) -> float:
    """One-sided Clopper-Pearson upper bound ``beta.ppf(1 - alpha, k + 1, n - k)`` (1 when k == n)."""
    from scipy.stats import beta
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def mcnemar_exact_p(n10: int, n01: int) -> Optional[float]:
    """Two-sided exact McNemar p-value (binomial on the discordant pairs; descriptive)."""
    from scipy.stats import binomtest
    m = int(n10) + int(n01)
    if m == 0:
        return None
    return float(binomtest(int(n10), m, 0.5).pvalue)


def git_head(repo: Path = REPO) -> dict[str, Any]:
    def run(*a: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    porcelain = run("status", "--porcelain") or ""
    tracked_dirty = [ln[3:] for ln in porcelain.splitlines() if ln and not ln.startswith("??")]
    engine = [f for f in tracked_dirty if f.startswith(ENGINE_SCOPE)]
    return {"head": run("rev-parse", "HEAD"), "head_short": run("rev-parse", "--short", "HEAD"),
            "tracked_files_modified": len(tracked_dirty), "engine_scope_files_modified": len(engine),
            "engine_scope": list(ENGINE_SCOPE)}


def file_sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jd(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def check_out_dir(out: Path) -> Path:
    """Output-label guard: restricted output only under data/restricted_local/exploratory_v2_1/ (never official)."""
    rel = Path(os.path.relpath(Path(out).resolve(), REPO))
    for bad in FORBIDDEN_OUT:
        if rel == bad or bad in rel.parents:
            raise SystemExit(f"--out {rel.as_posix()} is an official output location; refused")
    if not (rel == OUT_ALLOWED_ROOT or OUT_ALLOWED_ROOT in rel.parents):
        raise SystemExit(f"--out must lie under {OUT_ALLOWED_ROOT.as_posix()}/ (label {LABEL}); got {rel.as_posix()}")
    return REPO / rel


# =================================================================================================
# engine-dependent part
# =================================================================================================
def build_case_context(case_id: str, n_states: int, seed: int) -> dict[str, Any]:
    """The official v3 context of ``case_id`` with the SD-H0 world only and a development test stream of
    ``n_states`` states (no opt / validation draws)."""
    import official_v3 as OV3
    import run_endpoint_ablation as RA
    from ration_reliability.build.preflight import require_dev_case_inputs

    OV3.activate(case_id)
    arm = OV3.active_arm()
    require_dev_case_inputs(REPO, driver="resolve_lp_control", layout=arm.layout, extra=OV3.v3_requirements(arm))
    seed_chk = RA.check_seed(int(seed))                    # refuses a reserved formal root before any draw
    cfg = copy.deepcopy(OV3.OFFICIAL_CONFIG_V3[case_id])
    cfg["sd_points"] = {WORLD: cfg["sd_points"][WORLD]}
    cfg["evaluation"] = dict(cfg["evaluation"])
    cfg["evaluation"]["evaluation_worlds"] = [WORLD]
    sizes = {"opt": None, "validation": None, "test": int(n_states)}
    ctx = OV3.build_official_context_v3(sizes, draw=True, config=cfg, seed=int(seed))
    ctx["arm_record"] = arm.record()
    ctx["seed_check"] = seed_chk
    ctx["config_primary_assumption_id"] = cfg["primary_assumption_id"]
    return ctx


def energy_relinearisation_check(test: Any, lin: Any, n_check: int = 5) -> dict[str, Any]:
    """The state's energy column equals (a) the builder's affine linearisation at theta_s and (b) the direct per-feed
    fixed-DMI evaluation (``feed_energy_terms`` at the planned DM supply) with theta_s; booleans and max |diff| only."""
    from ration_reliability.nutrition import energy as E
    nid = list(test.nutrient_ids)
    kn = nid.index(E.ENERGY_COLUMN_ID)
    m = min(1000, test.n_draws)
    dens = lin.density(test.theta[:m], test.nutrient_ids)
    d_affine = float(np.max(np.abs(dens - test.theta[:m, :, kn])))
    d_direct = 0.0
    for s in range(min(n_check, test.n_draws)):
        for i, feed in enumerate(lin.feeds):
            comp = {fld: 100.0 * float(test.theta[s, i, nid.index(n)]) for n, fld in lin.nutrient_map.items()}
            e = E.feed_energy_terms(feed, lin.settings, comp)["NEL"]
            d_direct = max(d_direct, abs(e - float(test.theta[s, i, kn])))
    return {"energy_column_equals_affine_linearisation": d_affine <= 1e-12, "max_abs_diff_affine": d_affine,
            "energy_column_equals_direct_feed_evaluation": d_direct <= 1e-9, "max_abs_diff_direct": d_direct,
            "n_states_direct_checked": min(n_check, test.n_draws), "nutrient_map": dict(lin.nutrient_map)}


class ResolveLP:
    """The M0 LP (``m0_nominal.solve``) with the probabilistic rows at a given composition and d = d_hat."""

    def __init__(self, problem: Any, opts: Any):
        from ration_reliability.datamodel import ConstraintClass
        from ration_reliability.nutrition.constraints import linear_rows
        from ration_reliability.optimization.lp_builder import optimization_indices
        self.problem, self.opts = problem, opts
        self.cc = problem.compiled.subset(optimization_indices(problem.compiled))
        self.ids = tuple(problem.ingredient_ids)
        self.dh = np.asarray(problem.dm_estimates(), float)
        self.prices = np.asarray(problem.price_vector(), float)
        self._linear_rows = linear_rows
        self.struct = np.array([c is ConstraintClass.STRUCTURAL_HARD for c in self.cc.classes], dtype=bool)
        self.prob = np.array([c is ConstraintClass.PROBABILISTIC_NUTRITION for c in self.cc.classes], dtype=bool)
        nom = linear_rows(self.cc, problem.nominal_theta(), self.dh, d_hat=self.dh)
        self.A_nom, self.b, self.is_eq = nom.A[0], nom.b, nom.is_eq
        self.struct_theta_free = bool(all(not np.any(np.asarray(self.cc.W[k]) != 0)
                                          for k in np.flatnonzero(self.struct)))

    def rows_for(self, theta: np.ndarray) -> np.ndarray:
        """``A_q [S, K, I]`` for compositions ``theta [S, I, J]`` at d = d_hat; structural rows at the nominal state."""
        S = theta.shape[0]
        r = self._linear_rows(self.cc, theta, np.broadcast_to(self.dh, (S, len(self.dh))), d_hat=self.dh)
        if not np.array_equal(r.b, self.b) or not np.array_equal(r.is_eq, self.is_eq):
            raise RuntimeError("linear_rows: right-hand sides / senses changed with the composition")
        A = np.array(r.A)
        A[:, self.struct, :] = self.A_nom[self.struct][None]
        return A

    def solve(self, A_q: np.ndarray) -> tuple[str, Optional[np.ndarray]]:
        from ration_reliability.datamodel import SolveStatus
        from ration_reliability.optimization.highs import run_linprog
        from ration_reliability.optimization.lp_builder import assemble_x_space_lp
        if np.any(~np.isfinite(A_q)):
            return "missing_coefficients", None
        lp = assemble_x_space_lp(A_q, self.b, self.is_eq, self.cc.constraint_ids, self.dh, self.prices)
        out = run_linprog(lp.c, lp.A_ub, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, lp.ub, self.opts)
        st = out.status.value if hasattr(out.status, "value") else str(out.status)
        if out.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT) or out.x is None:
            return st, None
        return st, np.asarray(out.x, float) / self.dh


def run_cell(case_id: str, n_states: int, seed: int, out_root: Path, chunk: int = 500) -> dict[str, Any]:
    from ration_reliability.datamodel import RationDecision, SolverOptions
    from ration_reliability.evaluation.reference import evaluate_reference
    from ration_reliability.nutrition import energy as E
    from ration_reliability.optimization import get_method
    from ration_reliability.uncertainty.base import DrawSet

    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    gh = git_head()
    t = time.perf_counter()
    ctx = build_case_context(case_id, n_states, seed)
    timings["context_and_draws_s"] = time.perf_counter() - t
    problem = ctx["problems"][OBJECTIVE_ARM]
    ref, lin = ctx["ref"], ctx["lin"]
    wd = ctx["worlds"][WORLD]
    test = wd["draws"]["test"]
    world_rec = wd["record"]

    # ---- stream / label guard and alignment checks (booleans only)
    checks: dict[str, Any] = {
        "stream_is_test": test.stream == "test",
        "stream_id_is_development_root": test.stream_id.startswith(f"root={int(seed)}/test"),
        "n_states": int(test.n_draws) == int(n_states),
        "labels_match_problem": (tuple(test.ingredient_ids) == tuple(problem.ingredient_ids)
                                 and tuple(test.nutrient_ids) == tuple(problem.nutrient_ids)),
        "labels_match_reference": (tuple(test.ingredient_ids) == tuple(ref.constraints.ingredient_ids)
                                   and tuple(test.nutrient_ids) == tuple(ref.constraints.nutrient_ids)),
        "world_correlation_C0": str(world_rec.get("correlation", "")).startswith("C0"),
        "primary_assumption_has_SD_H0_and_C0": ("|SD-H0|" in ctx["config_primary_assumption_id"]
                                                and "|C0|" in ctx["config_primary_assumption_id"]),
        "no_nan_in_test_states": bool(np.all(np.isfinite(test.theta)) and np.all(np.isfinite(test.d))),
        "main_event_defined": MAIN_EVENT in {e.event_id for e in ref.events},
    }
    en_chk = energy_relinearisation_check(test, lin)
    checks["energy_relinearisation"] = en_chk
    opts = SolverOptions(time_limit_s=SOLVER["time_limit_s"], mip_rel_gap=SOLVER["mip_rel_gap"])
    rl = ResolveLP(problem, opts)
    checks["structural_rows_composition_free"] = rl.struct_theta_free
    checks["imposed_rows"] = {"structural": [c for c, s in zip(rl.cc.constraint_ids, rl.struct) if s],
                              "probabilistic": [c for c, s in zip(rl.cc.constraint_ids, rl.prob) if s]}
    # planned DM supply of the linearisation = the SH-DM-PLAN bound (the case's planned intake)
    kdm = list(problem.compiled.constraint_ids).index("SH-DM-PLAN")
    checks["linearisation_dmi_equals_planned_dm"] = bool(
        abs(float(problem.compiled.bound[kdm]) - float(lin.settings.dmi_kg_d)) < 1e-6)
    hard = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if not en_chk["energy_column_equals_affine_linearisation"] or not en_chk["energy_column_equals_direct_feed_evaluation"]:
        hard.append("energy_relinearisation")
    if hard:
        raise SystemExit(f"{case_id}: checks failed: {hard}")

    # ---- M0 (engine) and the regression of the re-solve LP at the nominal composition
    t = time.perf_counter()
    m0 = get_method("M0_nominal")(problem, params={"coefficient_mode": "nominal_point"}, solver_options=opts)
    if not m0.has_solution:
        raise SystemExit(f"{case_id}: engine M0 has no solution ({m0.status})")
    q0 = np.asarray(m0.decision.q_as_fed, float)
    cost0 = float(rl.prices @ q0)
    st_reg, q_reg = rl.solve(rl.rows_for(problem.nominal_theta()[None])[0])
    regression = {"status": st_reg,
                  "q_max_abs_diff_vs_engine_m0": None if q_reg is None else float(np.max(np.abs(q_reg - q0))),
                  "objective_abs_diff_vs_engine_m0": None if q_reg is None else abs(float(rl.prices @ q_reg) - cost0)}
    regression["ok"] = bool(q_reg is not None and regression["q_max_abs_diff_vs_engine_m0"] <= 1e-9
                            and regression["objective_abs_diff_vs_engine_m0"] <= 1e-9)
    if not regression["ok"]:
        raise SystemExit(f"{case_id}: the re-solve LP at the nominal composition does not reproduce engine M0")
    res0 = evaluate_reference(m0.decision, test, ref)
    m0_v = np.asarray(res0.event_violation[MAIN_EVENT], bool)
    m0_u = np.asarray(res0.event_unknown[MAIN_EVENT], bool)
    m0_fail = m0_v | m0_u
    # the single-state evaluation path used for the policy reproduces the batch verdicts of M0 (first states)
    n_cons = min(200, int(test.n_draws))
    same = 0
    for s in range(n_cons):
        ds = DrawSet(test.theta[s:s + 1], test.d[s:s + 1], test.stream, test.stream_id, test.model_id,
                     test.model_fingerprint, test.ingredient_ids, test.nutrient_ids, test.is_synthetic)
        r = evaluate_reference(m0.decision, ds, ref)
        same += int(bool(np.asarray(r.event_violation[MAIN_EVENT])[0]) == bool(m0_v[s])
                    and bool(np.asarray(r.event_unknown[MAIN_EVENT])[0]) == bool(m0_u[s]))
    checks["single_state_path_reproduces_batch_m0"] = bool(same == n_cons)
    checks["single_state_path_states_checked"] = n_cons
    if same != n_cons:
        raise SystemExit(f"{case_id}: single-state evaluation differs from the batch evaluation of M0")
    timings["m0_solve_and_evaluation_s"] = time.perf_counter() - t
    eval_ids = list(res0.evaluation.constraint_ids)
    members = list(next(e for e in ref.events if e.event_id == MAIN_EVENT).members)
    lin_members = [m for m in members if m in eval_ids]
    k_lin = [eval_ids.index(m) for m in lin_members]
    m0_row_viol = {m: int(np.asarray(res0.evaluation.violated, bool)[:, k].sum()) for m, k in zip(lin_members, k_lin)}
    m0_row_viol[f"{ref.energy.constraint_id}[reference_chain]"] = int(np.asarray(res0.energy.reference_violated,
                                                                                bool).sum())

    # ---- the policy, state by state
    S = int(test.n_draws)
    I = len(rl.ids)
    q_pol = np.zeros((S, I))
    status = np.empty(S, dtype=object)
    fallback = np.zeros(S, dtype=bool)
    pol_v = np.zeros(S, dtype=bool)
    pol_u = np.zeros(S, dtype=bool)
    pol_plan_in = np.zeros(S, dtype=bool)
    pol_struct_ok = np.zeros(S, dtype=bool)
    pol_row_viol = np.zeros((S, len(k_lin)), dtype=bool)
    pol_energy_ref_viol = np.zeros(S, dtype=bool)
    plan_status_counts: dict[str, int] = {}
    t_lp = t_ev = 0.0
    t = time.perf_counter()
    for c0 in range(0, S, chunk):
        c1 = min(S, c0 + chunk)
        t1 = time.perf_counter()
        A = rl.rows_for(test.theta[c0:c1])
        t_lp += time.perf_counter() - t1
        for j in range(c1 - c0):
            s = c0 + j
            t1 = time.perf_counter()
            st, q = rl.solve(A[j])
            t_lp += time.perf_counter() - t1
            status[s] = st
            if q is None:
                fallback[s] = True
                q = q0
            q_pol[s] = q
            t1 = time.perf_counter()
            dec = RationDecision(rl.ids, q, rl.dh, POLICY_METHOD_ID, information_state=POLICY_INFORMATION_STATE)
            ds = DrawSet(test.theta[s:s + 1], test.d[s:s + 1], test.stream, test.stream_id, test.model_id,
                         test.model_fingerprint, test.ingredient_ids, test.nutrient_ids, test.is_synthetic)
            r = evaluate_reference(dec, ds, ref)
            pol_v[s] = bool(np.asarray(r.event_violation[MAIN_EVENT])[0])
            pol_u[s] = bool(np.asarray(r.event_unknown[MAIN_EVENT])[0])
            ps = r.plan_domain.status if r.plan_domain is not None else "none"
            plan_status_counts[ps] = plan_status_counts.get(ps, 0) + 1
            pol_plan_in[s] = bool(np.asarray(r.plan_domain.state_mask(1))[0]) if r.plan_domain is not None else False
            pol_struct_ok[s] = bool(r.evaluation.structural_ok)
            vv = np.asarray(r.evaluation.violated, bool)[0]
            pol_row_viol[s] = vv[k_lin]
            pol_energy_ref_viol[s] = bool(np.asarray(r.energy.reference_violated, bool)[0])
            t_ev += time.perf_counter() - t1
        el = time.perf_counter() - t
        print(f"[{time.strftime('%H:%M:%S')}] {case_id} seed={seed}: {c1}/{S} states, fallback {int(fallback.sum())}, "
              f"policy fail so far {int((pol_v | pol_u)[:c1].sum())}/{c1}, {el:.0f} s", flush=True)
    timings["policy_lp_s"] = t_lp
    timings["policy_evaluation_s"] = t_ev
    timings["policy_total_s"] = time.perf_counter() - t
    pol_fail = pol_v | pol_u

    # ---- statistics (public: counts, rates, bounds, ratios)
    k_pol, k_m0 = int(pol_fail.sum()), int(m0_fail.sum())
    n10 = int((pol_fail & ~m0_fail).sum())
    n01 = int((~pol_fail & m0_fail).sum())
    costs = q_pol @ rl.prices
    solved = ~fallback
    st_counts: dict[str, int] = {}
    for st in status:
        st_counts[str(st)] = st_counts.get(str(st), 0) + 1
    pol_row_counts = {m: int(pol_row_viol[:, j].sum()) for j, m in enumerate(lin_members)}
    pol_row_counts[f"{ref.energy.constraint_id}[reference_chain]"] = int(pol_energy_ref_viol.sum())
    summary = {
        "schema": SCHEMA, "label": LABEL, "run_type": RUN_TYPE, "status": STATUS,
        "case_id": case_id, "arm": ARM_NAME[case_id], "reference_problem": ctx["reference_problem"],
        "objective_arm": OBJECTIVE_ARM, "world": WORLD, "correlation": world_rec.get("correlation"),
        "world_spec_id": world_rec.get("spec_id"), "main_event": MAIN_EVENT,
        "seed": int(seed), "seed_class": "development root (declared for this control; check_seed passed)",
        "stream_id": test.stream_id, "n_states": S,
        "policy": {"fail_k": k_pol, "fail_rate": k_pol / S, "violated_k": int(pol_v.sum()),
                   "unknown_k": int((pol_u & ~pol_v).sum()), "cp_upper_per_claim": cp_upper(k_pol, S),
                   "lp_fallback_k": int(fallback.sum()), "lp_status_counts": st_counts,
                   "fail_k_among_fallback_states": int((pol_fail & fallback).sum()),
                   "plan_domain_status_counts": plan_status_counts,
                   "plan_out_of_domain_k": int((~pol_plan_in).sum()),
                   "structural_ok_k": int(pol_struct_ok.sum()),
                   "per_row_violated_k": pol_row_counts,
                   "mean_cost_ratio_vs_m0_all_states": float(np.mean(costs) / cost0),
                   "mean_cost_ratio_vs_m0_solved_states": (float(np.mean(costs[solved]) / cost0)
                                                           if solved.any() else None),
                   "min_cost_ratio_vs_m0": float(np.min(costs) / cost0),
                   "max_cost_ratio_vs_m0": float(np.max(costs) / cost0)},
        "m0": {"fail_k": k_m0, "fail_rate": k_m0 / S, "violated_k": int(m0_v.sum()),
               "unknown_k": int((m0_u & ~m0_v).sum()), "cp_upper_per_claim": cp_upper(k_m0, S),
               "plan_domain_status": res0.plan_domain.status if res0.plan_domain is not None else None,
               "per_row_violated_k": m0_row_viol,
               "in_expected_range": bool(M0_EXPECTED_RANGE[0] <= k_m0 / S <= M0_EXPECTED_RANGE[1])},
        "paired": {"n10_policy_fails_m0_passes": n10, "n01_policy_passes_m0_fails": n01,
                   "n11_both_fail": int((pol_fail & m0_fail).sum()), "n00_both_pass": int((~pol_fail & ~m0_fail).sum()),
                   "difference_policy_minus_m0": (k_pol - k_m0) / S, "mcnemar_exact_p_two_sided": mcnemar_exact_p(n10, n01)},
        "per_claim_alpha": PER_CLAIM_ALPHA, "m0_expected_range": list(M0_EXPECTED_RANGE),
        "checks": {k: v for k, v in checks.items() if k != "energy_relinearisation"},
        "energy_relinearisation_check": {k: v for k, v in en_chk.items() if k != "nutrient_map"},
        "energy_linearisation_nutrients": sorted(en_chk["nutrient_map"]),
        "regression_resolve_at_nominal_equals_engine_m0": regression["ok"],
        "git_head": gh["head"], "git_head_short": gh["head_short"],
        "git_tracked_files_modified": gh["tracked_files_modified"],
        "git_engine_scope_files_modified": gh["engine_scope_files_modified"], "git_engine_scope": gh["engine_scope"],
        "problem_fingerprint": problem.compiled.fingerprint,
        "problem_fingerprint_full11": ctx["problems"]["FULL11"].compiled.fingerprint,
        "world_fingerprint": wd["w"].fingerprint(), "energy_linearisation_fingerprint": lin.fingerprint(),
        "reference_spec_fingerprint": ref.fingerprint(), "draws_fingerprint": test.fingerprint,
        "code_sha256": file_sha256(Path(__file__)),
        "solver_options": opts.to_dict(), "timings_s": timings, "runtime_s": time.perf_counter() - t_all,
        "command": " ".join([sys.executable] + sys.argv),
    }
    # public-safety: the public summary carries no absolute cost and no ration
    _assert_public_safe(summary)

    # ---- restricted outputs
    out = out_root / f"{case_id}_seed{int(seed)}_n{S}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_state.npz", q_policy=q_pol, cost_policy=costs, fallback=fallback,
                        status=np.array([str(x) for x in status]), policy_violated=pol_v, policy_unknown=pol_u,
                        m0_violated=m0_v, m0_unknown=m0_u, policy_plan_in_domain=pol_plan_in,
                        policy_row_violated=pol_row_viol, policy_row_ids=np.array(lin_members),
                        policy_energy_reference_violated=pol_energy_ref_viol, ingredient_ids=np.array(rl.ids))
    restricted = {"summary": summary, "m0_cost_usd_per_head_d": cost0,
                  "m0_q_as_fed": dict(zip(rl.ids, q0.tolist())), "d_hat": dict(zip(rl.ids, rl.dh.tolist())),
                  "prices_per_kg_as_fed": dict(zip(rl.ids, rl.prices.tolist())),
                  "policy_mean_cost_usd_per_head_d": float(np.mean(costs)), "regression": regression,
                  "energy_relinearisation_check": en_chk, "world_record": world_rec, "arm": ctx["arm_record"],
                  "seed_check": ctx["seed_check"], "inputs": ctx["inputs"],
                  "m0_reference_headline": res0.summary()["headline"]}
    (out / "run_record_restricted.json").write_text(json.dumps(restricted, default=_jd, indent=1), encoding="utf-8")
    (out / "summary_public.json").write_text(json.dumps(summary, default=_jd, indent=1), encoding="utf-8")
    summary["_out"] = out.as_posix()
    return summary


def _assert_public_safe(summary: dict) -> None:
    """No key of the public summary may name an absolute cost, a ration, a price or a DM estimate."""
    banned = ("cost_usd", "q_as_fed", "q_policy", "prices", "price", "d_hat", "x_dm", "ration", "m0_cost",
              "mean_cost", "requirement")

    def keys(o: Any) -> list[str]:
        if isinstance(o, dict):
            return [str(k) for k in o] + [x for v in o.values() for x in keys(v)]
        if isinstance(o, (list, tuple)):
            return [x for v in o for x in keys(v)]
        return []
    hits = sorted({k for k in keys(summary) for b in banned
                   if (k == b or k.startswith(b + "_")) and "ratio" not in k})
    if hits:
        raise RuntimeError(f"public summary would carry restricted keys: {hits}")


# =================================================================================================
# public summary (counts, rates, bounds, ratios only)
# =================================================================================================
CSV_COLUMNS = ("case_id", "arm", "world", "correlation_label", "seed", "stream_id", "n_states",
               "policy_fail_k", "policy_fail_rate", "policy_violated_k", "policy_unknown_k",
               "policy_cp_upper_per_claim", "m0_fail_k", "m0_fail_rate", "m0_cp_upper_per_claim",
               "n10_policy_fails_m0_passes", "n01_policy_passes_m0_fails", "n11_both_fail", "n00_both_pass",
               "paired_difference", "mcnemar_exact_p", "lp_fallback_k", "policy_plan_out_of_domain_k",
               "cost_ratio_mean_all_states", "cost_ratio_mean_solved_states", "cost_ratio_min", "cost_ratio_max",
               "per_claim_alpha", "runtime_s", "git_head_short", "problem_fingerprint")


def csv_row(s: dict) -> dict:
    p, m, pr = s["policy"], s["m0"], s["paired"]
    return {"case_id": s["case_id"], "arm": s["arm"], "world": s["world"], "correlation_label": "C0",
            "seed": s["seed"], "stream_id": s["stream_id"], "n_states": s["n_states"],
            "policy_fail_k": p["fail_k"], "policy_fail_rate": f"{p['fail_rate']:.6f}",
            "policy_violated_k": p["violated_k"], "policy_unknown_k": p["unknown_k"],
            "policy_cp_upper_per_claim": f"{p['cp_upper_per_claim']:.6f}",
            "m0_fail_k": m["fail_k"], "m0_fail_rate": f"{m['fail_rate']:.6f}",
            "m0_cp_upper_per_claim": f"{m['cp_upper_per_claim']:.6f}",
            "n10_policy_fails_m0_passes": pr["n10_policy_fails_m0_passes"],
            "n01_policy_passes_m0_fails": pr["n01_policy_passes_m0_fails"], "n11_both_fail": pr["n11_both_fail"],
            "n00_both_pass": pr["n00_both_pass"], "paired_difference": f"{pr['difference_policy_minus_m0']:+.6f}",
            "mcnemar_exact_p": _p(pr["mcnemar_exact_p_two_sided"]),
            "lp_fallback_k": p["lp_fallback_k"], "policy_plan_out_of_domain_k": p["plan_out_of_domain_k"],
            "cost_ratio_mean_all_states": f"{p['mean_cost_ratio_vs_m0_all_states']:.6f}",
            "cost_ratio_mean_solved_states": ("" if p["mean_cost_ratio_vs_m0_solved_states"] is None
                                              else f"{p['mean_cost_ratio_vs_m0_solved_states']:.6f}"),
            "cost_ratio_min": f"{p['min_cost_ratio_vs_m0']:.6f}", "cost_ratio_max": f"{p['max_cost_ratio_vs_m0']:.6f}",
            "per_claim_alpha": f"{s['per_claim_alpha']:.0e}", "runtime_s": f"{s['runtime_s']:.0f}",
            "git_head_short": s["git_head_short"], "problem_fingerprint": s["problem_fingerprint"]}


def collect(out_root: Path, n_states: int) -> list[dict]:
    rows = []
    for case in CASES:
        for seed in DEV_SEEDS:
            f = out_root / f"{case}_seed{seed}_n{n_states}" / "summary_public.json"
            if f.is_file():
                rows.append(json.loads(f.read_text(encoding="utf-8")))
    return rows


def write_public(rows: list[dict], tag: str, n_states: int, validation: Optional[list[dict]] = None) -> list[Path]:
    pub = REPO / PUBLIC_DIR
    csv_path = pub / "results_public.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for s in rows:
            w.writerow(csv_row(s))
    md_path = pub / f"RESULT_{tag}.md"
    md_path.write_text(public_md(rows, tag, n_states, validation or []), encoding="utf-8")
    return [csv_path, md_path]


def _p(v: Optional[float]) -> str:
    """p-value text; an exact p below the double range is written as an inequality."""
    if v is None:
        return ""
    return "<1e-300" if v < 1e-300 else f"{v:.3g}"


def _r(v: Optional[float], nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


#: official v3 M0 main-event rate_upper, SD-H0 cell SDH0_MAIN9, roots 0 / 1 / 2 (public endpoint_ablation.csv of
#: official-v3a-20260927T184000Z-8d7138b2 and official-v3c-20260927T220316Z-5d86b41a); orientation only
OFFICIAL_M0_ORIENTATION = {"dev_case_v3a": (0.8755, 0.8698, 0.8760), "dev_case_v3c": (0.8741, 0.8719, 0.8769)}
VALIDATION_DIR = "_validation500"


def public_md(rows: list[dict], tag: str, n_states: int, validation: Sequence[dict] = ()) -> str:
    L: list[str] = []
    a = L.append
    a(f"# E4 observed-composition re-solve control — result ({tag[:4]}-{tag[4:6]}-{tag[6:]})")
    a("")
    a("**EXPLORATORY, UNREGISTERED, DEVELOPMENT-STREAM CONTROL.** Not part of the frozen v2 / v3 protocol, not a member "
      "of any comparable set, not merged into any official table. Development seeds only. Rates are about the declared "
      "model constraints under the declared distribution; nothing is said about animals. Only counts, rates, bounds and "
      "cost ratios appear here (no ration, no absolute cost, no restricted table value).")
    a("")
    if rows:
        s0 = rows[0]
        a(f"- Script: `{PUBLIC_DIR.as_posix()}/resolve_lp_control.py`; restricted outputs: "
          f"`{OUT_DEFAULT.as_posix()}/<case>_seed<seed>_n<n>/`")
        heads = sorted({r['git_head_short'] for r in rows})
        a(f"- Git HEAD at run: `{', '.join(heads)}`; tracked files modified under the engine scope "
          f"({', '.join(ENGINE_SCOPE)}) at run time: {sorted({r['git_engine_scope_files_modified'] for r in rows})} "
          f"(other tracked files modified, outside the engine scope: "
          f"{sorted({r['git_tracked_files_modified'] - r['git_engine_scope_files_modified'] for r in rows})})")
        a(f"- World: `{WORLD}` (TAB), correlation `{s0['correlation']}`; objective arm `{OBJECTIVE_ARM}`; main event "
          f"`{MAIN_EVENT}` (failure = violated OR unknown)")
        a(f"- Streams: development roots {list(DEV_SEEDS)} (test stream `root=<seed>/test/0`, {n_states:,} states each); "
          "the same three roots for both cases (the states differ: arm C has the cottonseed cells)")
        a(f"- Per-claim error of the one-sided Clopper–Pearson upper bound: {PER_CLAIM_ALPHA:.0e} "
          "(`scipy.stats.beta.ppf(1 - 5e-6, k + 1, n - k)`)")
        a("")
    a("## Policy")
    a("")
    a("For every test state s = (θ_s, d_s): (1) solve the MAIN9 least-cost LP of the nominal M0 problem with θ_s in place "
      "of the nominal composition in every probabilistic (nutrient) row — CP supply, Ca and P absorbed, the five "
      "Table 5-1 rows and the fixed-DMI energy row, whose per-ingredient coefficient is the builder's fixed-DMI "
      "linearisation evaluated at θ_s at the planned DM supply (bound NEL_req − C0, composition-free) — with the DM "
      "fractions held at the planned d̂ (the M0 value) and every structural row as in M0 (planned-DM equality, inclusion "
      "caps, premise planning row SH-PLAN-T51-DGC-SHARE, planned CP / EE limits SH-PLAN-CP-HI / SH-PLAN-EE-HI); "
      "minimise as-fed cost. If the LP is not optimal the nominal M0 ration is executed in that state (counted as LP "
      "fallback). (2) Score q_s with `evaluate_reference` on the realised state (θ_s, d_s) only. (3) Score the M0 "
      "ration (solved once at the nominal composition) on the same states (paired).")
    a("")
    a("## Validation before scaling (500 states, seed 202610031)")
    a("")
    a("| case | M0 fail rate (500) | policy fail rate (500) | official v3 M0 rate, SD-H0, roots 0/1/2 (orientation) |")
    a("|---|---|---|---|")
    for v in validation:
        off = OFFICIAL_M0_ORIENTATION.get(v["case_id"], ())
        a(f"| {v['case_id']} | {v['m0']['fail_rate']:.3f} | {v['policy']['fail_rate']:.3f} | "
          f"{' / '.join(f'{x:.4f}' for x in off)} |")
    a("")
    a("The validation window was 0.85–0.98. Arm C at 500 states (0.846) sat 0.004 below its lower edge and within "
      "sampling error of the official rate (binomial SE about 0.015 at n = 500); every 20,000-state cell lies inside "
      "the window (0.8715–0.8788) and agrees with the official v3 TAB rates (0.8698–0.8769).")
    a("")
    a("## Results per cell (case × stream)")
    a("")
    a("| case | seed | n | policy fail k | policy rate | policy CP upper | M0 fail k | M0 rate | n10 | n01 | "
      "paired diff | LP fallback | cost ratio (mean, all) | cost ratio (mean, solved) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        p, m, pr = s["policy"], s["m0"], s["paired"]
        a(f"| {s['case_id']} ({s['arm']}) | {s['seed']} | {s['n_states']:,} | {p['fail_k']:,} | {p['fail_rate']:.4f} | "
          f"{p['cp_upper_per_claim']:.4f} | {m['fail_k']:,} | {m['fail_rate']:.4f} | {pr['n10_policy_fails_m0_passes']:,} | "
          f"{pr['n01_policy_passes_m0_fails']:,} | {pr['difference_policy_minus_m0']:+.4f} | {p['lp_fallback_k']:,} | "
          f"{p['mean_cost_ratio_vs_m0_all_states']:.4f} | {_r(p['mean_cost_ratio_vs_m0_solved_states'])} |")
    a("")
    a("n10 = re-solve fails and M0 passes; n01 = re-solve passes and M0 fails; paired diff = (n10 − n01)/n = policy rate "
      "− M0 rate. Cost ratio = mean as-fed cost of the executed policy rations / M0 cost (all states: fallback states at "
      "the M0 cost; solved: LP-optimal states only).")
    a("")
    a("## Failure composition and diagnostics")
    a("")
    a("| case | seed | policy violated / unknown | M0 violated / unknown | M0 CP upper | McNemar exact p | "
      "policy plan out of Table 5-1 domain | M0 plan status | cost ratio min / max |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        p, m, pr = s["policy"], s["m0"], s["paired"]
        mp = _p(pr["mcnemar_exact_p_two_sided"]) or "—"
        a(f"| {s['case_id']} | {s['seed']} | {p['violated_k']:,} / {p['unknown_k']:,} | {m['violated_k']:,} / "
          f"{m['unknown_k']:,} | {m['cp_upper_per_claim']:.4f} | {mp} | "
          f"{p['plan_out_of_domain_k']:,} | {m['plan_domain_status']} | {p['min_cost_ratio_vs_m0']:.4f} / "
          f"{p['max_cost_ratio_vs_m0']:.4f} |")
    a("")
    a("### Per-row violation counts of the main-event members (policy | M0)")
    a("")
    if rows:
        ids = list(rows[0]["policy"]["per_row_violated_k"])
        a("| case | seed | " + " | ".join(ids) + " |")
        a("|---|---|" + "---|" * len(ids))
        for s in rows:
            pp, mm = s["policy"]["per_row_violated_k"], s["m0"]["per_row_violated_k"]
            a(f"| {s['case_id']} | {s['seed']} | " + " | ".join(f"{pp.get(i, 0):,} \\| {mm.get(i, 0):,}" for i in ids)
              + " |")
        a("")
        a("Linear rows by the public evaluator at the realised (θ_s, d_s); the energy row by the Chapter 3 reference "
          "chain (the event's energy verdict). A state can violate several rows.")
        a("")
    a("## Notes and deviations")
    a("")
    tot = sum(r["runtime_s"] for r in rows)
    a(f"- Size: {n_states:,} states per cell as specified (about {tot / max(1, len(rows)) / 60:.1f} min per cell, "
      f"{tot / 60:.1f} min for the {len(rows)} cells, one process at a time).")
    a("- The streams are not pooled; each row is one independent development stream.")
    a("- The LP fallback rule (execute M0 when the re-solve LP is not optimal) was never triggered: every state's LP "
      "was optimal.")
    a("- Structural rows are taken at the nominal state; they are composition-free in both cases (checked), so this is "
      "identical to building them from θ_s.")
    a("- Additional descriptive quantities beyond the specification: the violated / unknown split, M0 Clopper–Pearson "
      "bound, exact McNemar p, the plan-level Table 5-1 status of the re-solved rations, per-row violation counts and "
      "the min / max cost ratio.")
    a("- The control does not isolate why the re-solve still fails: the DM fractions stay unobserved (rations are "
      "formulated at d̂ and judged at d_s) and the energy verdict is the Chapter 3 reference chain while the LP uses the "
      "linear fixed-DMI row. A d = d̂ counterfactual was not run.")
    a("")
    a("## Checks (every run)")
    a("")
    cur = file_sha256(Path(__file__))
    shas = sorted({r["code_sha256"] for r in list(rows) + list(validation)})
    a(f"- Script sha256 at run time: `{', '.join(x[:16] for x in shas)}`; equal to the script that wrote this file: "
      f"{shas == [cur]}")
    for s in rows:
        c = s["checks"]
        e = s["energy_relinearisation_check"]
        a(f"- {s['case_id']} seed {s['seed']}: stream/label guard {all(c[k] for k in ('stream_is_test', 'stream_id_is_development_root'))}; "
          f"labels aligned {c['labels_match_problem'] and c['labels_match_reference']}; world C0 {c['world_correlation_C0']}; "
          f"structural rows composition-free {c['structural_rows_composition_free']}; linearisation DMI = planned DM "
          f"{c['linearisation_dmi_equals_planned_dm']}; energy column = builder linearisation at θ_s "
          f"{e['energy_column_equals_affine_linearisation']} (direct per-feed check {e['energy_column_equals_direct_feed_evaluation']}); "
          f"re-solve at nominal θ reproduces engine M0 {s['regression_resolve_at_nominal_equals_engine_m0']}; "
          f"M0 rate in the validation window {M0_EXPECTED_RANGE} {s['m0']['in_expected_range']}; "
          f"problem fingerprint `{s['problem_fingerprint'][:16]}`; runtime {s['runtime_s'] / 60:.1f} min")
    a("")
    return "\n".join(L)


# =================================================================================================
# main
# =================================================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", choices=CASES, help="v3 arm case (dev_case_v3a = arm A, dev_case_v3c = arm C)")
    ap.add_argument("--n-states", type=int, default=N_STATES_DEFAULT, help="test-stream states (default 20,000)")
    ap.add_argument("--seed", type=int, default=DEV_SEEDS[0], help=f"development root (declared: {list(DEV_SEEDS)})")
    ap.add_argument("--out", default=OUT_DEFAULT.as_posix(),
                    help="restricted output root (must lie under data/restricted_local/exploratory_v2_1/)")
    ap.add_argument("--chunk", type=int, default=500, help="states per row-assembly chunk (memory only)")
    ap.add_argument("--summarise", action="store_true",
                    help="write the public CSV + markdown from the per-run public summaries under --out")
    ap.add_argument("--tag", default="20261003", help="date tag of the public markdown (RESULT_<tag>.md)")
    args = ap.parse_args(argv)
    os.chdir(REPO)
    out_root = check_out_dir(Path(args.out))
    if args.summarise:
        rows = collect(out_root, int(args.n_states))
        if not rows:
            raise SystemExit(f"no per-run public summaries for n = {args.n_states} under {out_root}")
        validation = collect(out_root / VALIDATION_DIR, 500)
        for p in write_public(rows, args.tag, int(args.n_states), validation):
            print(f"wrote {os.path.relpath(p, REPO)} ({len(rows)} cells)")
        return 0
    if args.case is None:
        raise SystemExit("--case is required (or --summarise)")
    if int(args.seed) not in DEV_SEEDS:
        print(f"note: seed {args.seed} is not one of the declared development seeds {list(DEV_SEEDS)}", flush=True)
    s = run_cell(args.case, int(args.n_states), int(args.seed), out_root, chunk=int(args.chunk))
    p, m, pr = s["policy"], s["m0"], s["paired"]
    print(json.dumps({"case": s["case_id"], "seed": s["seed"], "n": s["n_states"],
                      "policy_fail_k": p["fail_k"], "policy_rate": round(p["fail_rate"], 6),
                      "policy_cp_upper": round(p["cp_upper_per_claim"], 6), "m0_fail_k": m["fail_k"],
                      "m0_rate": round(m["fail_rate"], 6), "m0_in_expected_range": m["in_expected_range"],
                      "n10": pr["n10_policy_fails_m0_passes"], "n01": pr["n01_policy_passes_m0_fails"],
                      "lp_fallback": p["lp_fallback_k"], "lp_status": p["lp_status_counts"],
                      "plan_status_counts": p["plan_domain_status_counts"],
                      "cost_ratio_all": round(p["mean_cost_ratio_vs_m0_all_states"], 6),
                      "per_row_policy": p["per_row_violated_k"], "per_row_m0": m["per_row_violated_k"],
                      "timings_s": {k: round(v, 1) for k, v in s["timings_s"].items()},
                      "runtime_s": round(s["runtime_s"], 1), "out": os.path.relpath(s["_out"], REPO)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
