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

DM-margin re-solves (``--dm-margin KAPPA``; default 0 = the policy above, unchanged)
-----------------------------------------------------------------------------------
With KAPPA > 0 step 1 solves the same LP at theta_s with the original study's M1 safety margin switched on for DM.  The
engine M1 (``optimization/safety_margin.solve``, ``M1_safety_margin``; official setting
``official_v2.OFFICIAL_SETTINGS["M1"]`` = coef_directional, margin_scale "relative", apply_to_dm True) is called on an
:class:`ObservedCompositionView` of the MAIN9 problem: ``nominal_theta()`` returns theta_s, every other attribute
(compiled rows, d_hat, prices, nutrients) is the problem's own.  M1 reads the composition only through
``nominal_theta()``, so this is M1 with its margin centred at the observed composition.  ``--margin-scope``:

* ``dm_only`` (default; the fair comparator: once theta is observed the only remaining uncertainty is DM).  Engine M1
  through its explicit-spread interface with composition spread 0 and DM spread d_hat (= the DM spread of M1's
  relative scale), apply_to_dm True.  In every probabilistic row k and for every ingredient i the content stays at
  theta_s and the DM fraction becomes ``d_i = clip(d_hat_i (1 + KAPPA sgn_ki), 1e-6, 1)`` with
  ``sgn_ki = sign(s_k (c_ki(theta_s) - K_k))`` (concentration rows) or ``sign(s_k c_ki(theta_s))`` (supply rows),
  ``s_k = -1`` (ge) / ``+1`` (le): M1's relative DM margin with its composition margin switched off.
* ``all``: engine M1 exactly in the official setting (relative scale, apply_to_dm True) centred at theta_s:
  ``a_ij -> theta_s,ij (1 + KAPPA sign(s_k W_kij))`` (mass fractions clipped to [0, 1]) plus the DM margin above,
  evaluated at the shifted content.

Structural rows are unchanged, the ration is executed as ``q = x / d_hat`` (as M1), and a non-optimal LP falls back to
the nominal M0 ration (counted in ``lp_fallback``; ``lp_infeasible`` = proven infeasible).

Restricted outputs (absolute costs, rations) go to ``--out`` (default
``data/restricted_local/exploratory_v2_1/resolve_lp_control/``); each run also writes ``summary_public.json`` there
(counts, rates, bounds, cost ratios, fingerprints only).  ``--summarise`` collects those into the public files
``experiments/E4_resolve_lp_control/results_public.csv`` and ``RESULT_<tag>.md``.

Usage::

    PYTHONDONTWRITEBYTECODE=1 nice -n 10 /opt/homebrew/opt/python@3.11/bin/python3.11 \
        experiments/E4_resolve_lp_control/resolve_lp_control.py --case dev_case_v3a --n-states 20000 --seed 202610031
    ... --case dev_case_v3a --n-states 20000 --seed 202610031 --dm-margin 0.05   # DM-margin re-solve (dm_only)
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
#: DM-margin re-solves (``--dm-margin``): declared grid (the contract T4 relative grid 0.025-0.10 plus 0.15) and scopes
DM_MARGIN_GRID = (0.025, 0.05, 0.10, 0.15)
MARGIN_SCOPES = ("dm_only", "all")
M1_METHOD_ID = "M1_safety_margin"
#: the official M1 setting the margin follows (``official_v2.OFFICIAL_SETTINGS["M1"]``; checked at run time)
M1_EXPECTED_SETTING = {"margin_semantics": "coef_directional", "margin_scale": "relative", "apply_to_dm_main": True}
POLICY_METHOD_ID_MARGIN = {"dm_only": "EXPL_resolve_observed_composition_lp_dm_margin",
                           "all": "EXPL_resolve_observed_composition_lp_m1_margin"}
POLICY_INFORMATION_STATE_MARGIN = "composition_observed(theta_s); DM at d_hat with an M1 margin (exploratory)"
MARGIN_DEFINITION = {
    "none": "no margin (the zero-margin observed-composition re-solve)",
    "dm_only": ("engine M1 (safety_margin.solve) on the observed-composition view, explicit spreads: composition spread "
                "0, DM spread d_hat, apply_to_dm True; per probabilistic row k and ingredient i: a_ij = theta_s,ij, "
                "d_i = clip(d_hat_i * (1 + kappa * sign(s_k * (c_ki(theta_s) - K_k))), 1e-6, 1) for concentration rows "
                "and sign(s_k * c_ki(theta_s)) for supply rows, s_k = -1 (ge) / +1 (le); structural rows unchanged; "
                "q = x / d_hat"),
    "all": ("engine M1 (safety_margin.solve) on the observed-composition view in the official setting (margin_scale "
            "relative, apply_to_dm True): a_ij = clip(theta_s,ij * (1 + kappa * sign(s_k * W_kij))) and the DM margin "
            "of dm_only evaluated at the shifted content; structural rows unchanged; q = x / d_hat"),
}


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
            # rstrip only: a leading space is part of the first porcelain status column (" M path"); strip() used to
            # cut it and the first path character with it (fixed with the --dm-margin option)
            return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.rstrip()
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


def run_dir_name(case_id: str, seed: int, n_states: int, kappa: float = 0.0, scope: str = "dm_only") -> str:
    """Per-run directory: the zero-margin runs keep their original name; margin runs add ``_<scope>_k<kappa>``."""
    base = f"{case_id}_seed{int(seed)}_n{int(n_states)}"
    return base if float(kappa) == 0.0 else f"{base}_{scope}_k{float(kappa):g}"


def margin_record(s: dict) -> dict:
    """The margin block of a public summary (runs written before ``--dm-margin`` existed have kappa 0, no margin)."""
    m = s.get("margin") or {}
    return {"kappa": float(m.get("kappa", 0.0)), "scope": str(m.get("scope", "none"))}


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


class ObservedCompositionView:
    """Read-only view of a problem whose nominal composition is the observed ``theta_s``.

    ``nominal_theta()`` returns ``theta_s``; every other attribute (compiled rows, ingredients, nutrients, d_hat,
    prices, ids, problem id) is the wrapped problem's own.  The engine M1 reads the composition only through
    ``nominal_theta()`` (``safety_margin.margin_spread`` and ``build_margin_rows``), so M1 called on this view is M1
    with its margin centred at the observed composition (checked: at k = 0 it reproduces the zero-margin re-solve).
    """

    def __init__(self, problem: Any, theta_s: np.ndarray):
        th = np.array(theta_s, dtype=float)
        if th.shape != (len(problem.ingredient_ids), len(problem.nutrient_ids)):
            raise ValueError("theta_s shape does not match the problem")
        th.setflags(write=False)
        self._problem = problem
        self._theta = th

    def nominal_theta(self) -> np.ndarray:
        return np.array(self._theta)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


class MarginResolve:
    """Engine M1 (``safety_margin.solve``) at margin ``kappa`` centred at the observed composition."""

    def __init__(self, problem: Any, opts: Any, kappa: float, scope: str):
        from ration_reliability.optimization import get_method
        if scope not in MARGIN_SCOPES:
            raise ValueError(f"unknown margin scope {scope!r}")
        if not (math.isfinite(float(kappa)) and float(kappa) >= 0.0):
            raise ValueError("kappa must be finite and >= 0")
        self.problem, self.opts, self.kappa, self.scope = problem, opts, float(kappa), scope
        self.m1 = get_method(M1_METHOD_ID)
        self.dh = np.asarray(problem.dm_estimates(), float)
        self.shape = (len(problem.ingredient_ids), len(problem.nutrient_ids))
        self.params = self.params_for(self.kappa)

    def params_for(self, k: float) -> dict[str, Any]:
        if self.scope == "dm_only":
            return {"k": float(k), "margin_scale": "sd", "sd_source": "explicit", "theta_sd": np.zeros(self.shape),
                    "d_sd": self.dh.copy(), "apply_to_dm": True, "information_state": POLICY_INFORMATION_STATE_MARGIN}
        return {"k": float(k), "margin_scale": "relative", "apply_to_dm": True,
                "information_state": POLICY_INFORMATION_STATE_MARGIN}

    def public_params(self) -> dict[str, Any]:
        """The engine parameters without arrays (public record)."""
        if self.scope == "dm_only":
            return {"margin_scale": "sd", "sd_source": "explicit", "theta_sd": "zeros [I, J]",
                    "d_sd": "d_hat (the DM spread of M1's relative scale)", "apply_to_dm": True, "k": self.kappa}
        return {"margin_scale": "relative", "apply_to_dm": True, "k": self.kappa}

    def spreads(self, view: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(theta_spread, d_spread)`` exactly as the engine M1 computes them for these parameters."""
        from ration_reliability.optimization.safety_margin import margin_spread
        p = self.params
        th, d, _ = margin_spread(view, margin_scale=p["margin_scale"], sd_source=p.get("sd_source", "opt_draws"),
                                 theta_sd=p.get("theta_sd"), d_sd=p.get("d_sd"), d_hat=self.dh)
        return th, d

    def solve(self, theta_s: np.ndarray, k: Optional[float] = None) -> tuple[str, Optional[np.ndarray], tuple[int, int]]:
        from ration_reliability.datamodel import SolveStatus
        params = self.params if k is None else self.params_for(k)
        r = self.m1(ObservedCompositionView(self.problem, theta_s), params=params, solver_options=self.opts)
        st = r.status.value if hasattr(r.status, "value") else str(r.status)
        diag = (int(r.diagnostics.get("n_theta_cells_clipped", 0) or 0),
                int(r.diagnostics.get("n_dm_cells_clipped", 0) or 0))
        if r.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE_TIME_LIMIT) or r.decision is None:
            return st, None, diag
        return st, np.asarray(r.decision.q_as_fed, float), diag


def margin_definition_check(mres: MarginResolve, rl: ResolveLP, theta: np.ndarray) -> dict[str, Any]:
    """On the given states: (a) engine M1 on the view at k = 0 reproduces the zero-margin re-solve (rows and ration);
    (b) the engine's margin rows at kappa follow the declared definition, re-derived independently here from the
    compiled rows (content and DM fraction of every probabilistic row).  Booleans and max |diff| only."""
    from ration_reliability.datamodel import ConstraintKind, Sense
    from ration_reliability.optimization.robust import _UNCLIPPED_DIMENSIONS
    from ration_reliability.optimization.safety_margin import DM_FLOOR, build_margin_rows
    k = mres.kappa
    cc = rl.cc
    A = rl.rows_for(theta)
    nut = rl.problem.nutrients
    lower = np.array([-np.inf if n.dimension in _UNCLIPPED_DIMENSIONS else 0.0 for n in nut])
    upper = np.array([1.0 if n.dimension == "mass_fraction" else np.inf for n in nut])
    out = {"n_states_checked": int(theta.shape[0]), "k0_status_equal": True, "k0_max_abs_dq": 0.0,
           "k0_max_abs_dA": 0.0, "content_max_abs_diff": 0.0, "dm_max_abs_diff": 0.0,
           "content_max_abs_diff_vs_theta_s": 0.0, "rows_tightened": None, "states_with_dm_moved": 0}
    for s in range(theta.shape[0]):
        view = ObservedCompositionView(rl.problem, theta[s])
        st_m, q_m, _ = mres.solve(theta[s], k=0.0)
        st_r, q_r = rl.solve(A[s])
        out["k0_status_equal"] &= (st_m == st_r)
        if q_m is not None and q_r is not None:
            out["k0_max_abs_dq"] = max(out["k0_max_abs_dq"], float(np.max(np.abs(q_m - q_r))))
        elif (q_m is None) != (q_r is None):
            out["k0_max_abs_dq"] = float("inf")
        th_sp, d_sp = mres.spreads(view)
        mr0 = build_margin_rows(view, 0.0, th_sp, d_sp, apply_to_dm=True, d_hat=rl.dh)
        out["k0_max_abs_dA"] = max(out["k0_max_abs_dA"], float(np.max(np.abs(np.asarray(mr0.A_q) - A[s]))))
        mr = build_margin_rows(view, k, th_sp, d_sp, apply_to_dm=True, d_hat=rl.dh)
        out["rows_tightened"] = list(mr.probabilistic_ids)
        moved = False
        for kk, cid in enumerate(mr.constraint_ids):
            if cid not in mr.theta_states:
                continue
            sk = -1.0 if cc.senses[kk] is Sense.GE else 1.0
            Wk = np.asarray(cc.W[kk])
            need = Wk != 0
            th_exp = theta[s] + np.where(need, k * th_sp * np.sign(sk * Wk), 0.0)
            th_exp = np.clip(th_exp, lower[None, :], upper[None, :])
            out["content_max_abs_diff"] = max(out["content_max_abs_diff"],
                                              float(np.max(np.abs(mr.theta_states[cid] - th_exp))))
            out["content_max_abs_diff_vs_theta_s"] = max(out["content_max_abs_diff_vs_theta_s"],
                                                         float(np.max(np.abs(mr.theta_states[cid] - theta[s]))))
            c = (Wk * th_exp).sum(axis=1) + np.asarray(cc.w0[kk])
            slope = sk * (c - cc.bound[kk]) if cc.kinds[kk] is ConstraintKind.CONCENTRATION else sk * c
            d_exp = np.clip(rl.dh + k * d_sp * np.sign(slope), DM_FLOOR, 1.0)
            out["dm_max_abs_diff"] = max(out["dm_max_abs_diff"], float(np.max(np.abs(mr.d_states[cid] - d_exp))))
            moved |= bool(np.any(mr.d_states[cid] != rl.dh))
        out["states_with_dm_moved"] += int(moved)
    out["k0_reproduces_resolve"] = bool(out["k0_status_equal"] and out["k0_max_abs_dq"] <= 1e-9
                                        and out["k0_max_abs_dA"] <= 1e-12)
    out["rows_follow_definition"] = bool(out["content_max_abs_diff"] <= 1e-12 and out["dm_max_abs_diff"] <= 1e-12)
    if mres.scope == "dm_only":
        out["content_unchanged_at_theta_s"] = bool(out["content_max_abs_diff_vs_theta_s"] <= 0.0)
    return out


def m1_setting_check() -> dict[str, Any]:
    """The official M1 setting is the one this control follows (relative scale, DM margin on in the main setting)."""
    import official_v2 as OV2
    import run_dev_case_v1 as DRV
    s = OV2.OFFICIAL_SETTINGS["M1"]
    d = DRV.RUN_CONFIG["methods"]["M1"]
    return {"official_settings_M1": {k: s.get(k) for k in M1_EXPECTED_SETTING},
            "matches": bool(all(s.get(k) == v for k, v in M1_EXPECTED_SETTING.items())
                            and d.get("margin_scale") == "relative" and d.get("apply_to_dm") is True),
            "driver_grid": list(d.get("grid", []))}


def run_cell(case_id: str, n_states: int, seed: int, out_root: Path, chunk: int = 500, kappa: float = 0.0,
             scope: str = "dm_only") -> dict[str, Any]:
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

    # ---- DM-margin variant: engine M1 centred at theta_s (kappa > 0 only; kappa = 0 is the unchanged policy)
    kappa = float(kappa)
    mres: Optional[MarginResolve] = None
    margin_chk: Optional[dict[str, Any]] = None
    m1_set: Optional[dict[str, Any]] = None
    if kappa > 0.0:
        t = time.perf_counter()
        m1_set = m1_setting_check()
        mres = MarginResolve(problem, opts, kappa, scope)
        margin_chk = margin_definition_check(mres, rl, test.theta[:min(50, int(test.n_draws))])
        checks["m1_official_setting_relative_with_dm_margin"] = m1_set["matches"]
        checks["margin_k0_on_observed_view_reproduces_resolve"] = margin_chk["k0_reproduces_resolve"]
        checks["margin_rows_follow_definition"] = margin_chk["rows_follow_definition"]
        bad_m = [k for k in ("m1_official_setting_relative_with_dm_margin",
                             "margin_k0_on_observed_view_reproduces_resolve", "margin_rows_follow_definition")
                 if not checks[k]]
        if scope == "dm_only" and not margin_chk.get("content_unchanged_at_theta_s", False):
            bad_m.append("content_unchanged_at_theta_s")
        if bad_m:
            raise SystemExit(f"{case_id}: margin checks failed: {bad_m}")
        timings["margin_checks_s"] = time.perf_counter() - t
    method_id = POLICY_METHOD_ID if mres is None else POLICY_METHOD_ID_MARGIN[scope]
    info_state = POLICY_INFORMATION_STATE if mres is None else POLICY_INFORMATION_STATE_MARGIN
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
    theta_clip = np.zeros(S, dtype=np.int64)
    dm_clip = np.zeros(S, dtype=np.int64)
    plan_status_counts: dict[str, int] = {}
    t_lp = t_ev = 0.0
    t = time.perf_counter()
    for c0 in range(0, S, chunk):
        c1 = min(S, c0 + chunk)
        if mres is None:
            t1 = time.perf_counter()
            A = rl.rows_for(test.theta[c0:c1])
            t_lp += time.perf_counter() - t1
        for j in range(c1 - c0):
            s = c0 + j
            t1 = time.perf_counter()
            if mres is None:
                st, q = rl.solve(A[j])
            else:
                st, q, (theta_clip[s], dm_clip[s]) = mres.solve(test.theta[s])
            t_lp += time.perf_counter() - t1
            status[s] = st
            if q is None:
                fallback[s] = True
                q = q0
            q_pol[s] = q
            t1 = time.perf_counter()
            dec = RationDecision(rl.ids, q, rl.dh, method_id, information_state=info_state)
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
        print(f"[{time.strftime('%H:%M:%S')}] {case_id} seed={seed} kappa={kappa:g}"
              f"{'' if mres is None else '/' + scope}: {c1}/{S} states, fallback {int(fallback.sum())}, "
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
                   "lp_infeasible_k": int(st_counts.get("proven_infeasible", 0)),
                   "lp_solved_k": int(solved.sum()),
                   "fail_k_among_fallback_states": int((pol_fail & fallback).sum()),
                   "fail_k_solved_states": int((pol_fail & solved).sum()),
                   "fail_rate_solved_states": (float((pol_fail & solved).sum() / solved.sum())
                                               if solved.any() else None),
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
        "margin": ({"kappa": 0.0, "scope": "none", "definition": MARGIN_DEFINITION["none"]} if mres is None else
                   {"kappa": kappa, "scope": scope, "definition": MARGIN_DEFINITION[scope],
                    "engine_method": M1_METHOD_ID, "engine_function": "ration_reliability.optimization.safety_margin.solve",
                    "centre": "ObservedCompositionView(theta_s): nominal_theta() = theta_s, everything else the MAIN9 "
                              "problem's own", "engine_params": mres.public_params(),
                    "official_m1_setting": m1_set, "definition_check": margin_chk,
                    "n_theta_cells_clipped_total": int(theta_clip.sum()),
                    "states_with_theta_clip": int((theta_clip > 0).sum()),
                    "states_with_dm_clip": int((dm_clip > 0).sum()),
                    "dm_cells_clipped_per_state_max": int(dm_clip.max()) if S else 0}),
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
    out = out_root / run_dir_name(case_id, seed, S, kappa, scope)
    out.mkdir(parents=True, exist_ok=True)
    extra = {} if mres is None else {"theta_cells_clipped": theta_clip, "dm_cells_clipped": dm_clip}
    np.savez_compressed(out / "per_state.npz", q_policy=q_pol, cost_policy=costs, fallback=fallback,
                        status=np.array([str(x) for x in status]), policy_violated=pol_v, policy_unknown=pol_u,
                        m0_violated=m0_v, m0_unknown=m0_u, policy_plan_in_domain=pol_plan_in,
                        policy_row_violated=pol_row_viol, policy_row_ids=np.array(lin_members),
                        policy_energy_reference_violated=pol_energy_ref_viol, ingredient_ids=np.array(rl.ids),
                        **extra)
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
CSV_COLUMNS = ("case_id", "arm", "world", "correlation_label", "margin_scope", "kappa", "seed", "stream_id",
               "n_states", "policy_fail_k", "policy_fail_rate", "policy_violated_k", "policy_unknown_k",
               "policy_cp_upper_per_claim", "m0_fail_k", "m0_fail_rate", "m0_cp_upper_per_claim",
               "n10_policy_fails_m0_passes", "n01_policy_passes_m0_fails", "n11_both_fail", "n00_both_pass",
               "paired_difference", "mcnemar_exact_p", "lp_fallback_k", "lp_infeasible_k", "lp_solved_k",
               "policy_fail_k_solved_states", "policy_fail_rate_solved_states", "policy_plan_out_of_domain_k",
               "cost_ratio_mean_all_states", "cost_ratio_mean_solved_states", "cost_ratio_min", "cost_ratio_max",
               "per_claim_alpha", "runtime_s", "git_head_short", "code_sha256_16", "problem_fingerprint")


def policy_extras(s: dict) -> dict:
    """LP infeasible / solved counts and the failure count among LP-solved states (derived for older summaries)."""
    p, n = s["policy"], int(s["n_states"])
    inf = p.get("lp_infeasible_k", int(p.get("lp_status_counts", {}).get("proven_infeasible", 0)))
    solved = p.get("lp_solved_k", n - int(p["lp_fallback_k"]))
    fs = p.get("fail_k_solved_states", int(p["fail_k"]) - int(p["fail_k_among_fallback_states"]))
    return {"lp_infeasible_k": int(inf), "lp_solved_k": int(solved), "fail_k_solved": int(fs),
            "fail_rate_solved": (fs / solved) if solved else None}


def csv_row(s: dict) -> dict:
    p, m, pr = s["policy"], s["m0"], s["paired"]
    mg, ex = margin_record(s), policy_extras(s)
    return {"case_id": s["case_id"], "arm": s["arm"], "world": s["world"], "correlation_label": "C0",
            "margin_scope": mg["scope"], "kappa": f"{mg['kappa']:g}",
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
            "lp_fallback_k": p["lp_fallback_k"], "lp_infeasible_k": ex["lp_infeasible_k"],
            "lp_solved_k": ex["lp_solved_k"], "policy_fail_k_solved_states": ex["fail_k_solved"],
            "policy_fail_rate_solved_states": "" if ex["fail_rate_solved"] is None else f"{ex['fail_rate_solved']:.6f}",
            "policy_plan_out_of_domain_k": p["plan_out_of_domain_k"],
            "cost_ratio_mean_all_states": f"{p['mean_cost_ratio_vs_m0_all_states']:.6f}",
            "cost_ratio_mean_solved_states": ("" if p["mean_cost_ratio_vs_m0_solved_states"] is None
                                              else f"{p['mean_cost_ratio_vs_m0_solved_states']:.6f}"),
            "cost_ratio_min": f"{p['min_cost_ratio_vs_m0']:.6f}", "cost_ratio_max": f"{p['max_cost_ratio_vs_m0']:.6f}",
            "per_claim_alpha": f"{s['per_claim_alpha']:.0e}", "runtime_s": f"{s['runtime_s']:.0f}",
            "git_head_short": s["git_head_short"], "code_sha256_16": s["code_sha256"][:16],
            "problem_fingerprint": s["problem_fingerprint"]}


def collect(out_root: Path, n_states: int) -> list[dict]:
    """Per-run public summaries: the zero-margin runs first, then every scope and kappa of the grid present."""
    keys = [(0.0, "none")] + [(k, sc) for sc in MARGIN_SCOPES for k in DM_MARGIN_GRID]
    rows = []
    for kappa, scope in keys:
        for case in CASES:
            for seed in DEV_SEEDS:
                f = out_root / run_dir_name(case, seed, n_states, kappa, scope) / "summary_public.json"
                if not f.is_file():
                    continue
                s = json.loads(f.read_text(encoding="utf-8"))
                mg = margin_record(s)
                if mg["kappa"] != kappa or (kappa > 0 and mg["scope"] != scope) or s["n_states"] != n_states:
                    raise SystemExit(f"{f}: summary does not match its directory (kappa / scope / n)")
                rows.append(s)
    return rows


def k0_regression_check(out_root: Path) -> list[dict]:
    """The zero-margin path of the current script against the original 500-state validation runs: per-state arrays
    (rations, costs, statuses, verdicts, row violations) byte-identical.  Booleans only."""
    keys = ("q_policy", "cost_policy", "fallback", "status", "policy_violated", "policy_unknown", "m0_violated",
            "m0_unknown", "policy_plan_in_domain", "policy_row_violated", "policy_row_ids",
            "policy_energy_reference_violated", "ingredient_ids")
    res = []
    for case in CASES:
        name = run_dir_name(case, DEV_SEEDS[0], 500)
        a, b = out_root / VALIDATION_DIR / name, out_root / REGRESSION_K0_DIR / name
        if not ((a / "per_state.npz").is_file() and (b / "per_state.npz").is_file()):
            continue
        za, zb = np.load(a / "per_state.npz"), np.load(b / "per_state.npz")
        same = {k: bool(np.array_equal(za[k], zb[k])) for k in keys}
        sa = json.loads((a / "summary_public.json").read_text(encoding="utf-8"))
        sb = json.loads((b / "summary_public.json").read_text(encoding="utf-8"))
        res.append({"case_id": case, "all_identical": all(same.values()), "arrays": same,
                    "original_sha16": sa["code_sha256"][:16], "current_sha16": sb["code_sha256"][:16],
                    "draws_fingerprint_equal": sa["draws_fingerprint"] == sb["draws_fingerprint"]})
    return res


def write_public(rows: list[dict], tag: str, n_states: int, validation: Optional[list[dict]] = None,
                 k0_regression: Sequence[dict] = ()) -> list[Path]:
    pub = REPO / PUBLIC_DIR
    csv_path = pub / "results_public.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for s in rows:
            w.writerow(csv_row(s))
    md_path = pub / f"RESULT_{tag}.md"
    md_path.write_text(public_md(rows, tag, n_states, validation or [], k0_regression), encoding="utf-8")
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
#: re-run of the zero-margin path with the current script (500 states, first seed) compared with VALIDATION_DIR
REGRESSION_K0_DIR = "_regression_k0_500"


def _rng(vals: Sequence[float], fmt: str = "{:.4f}") -> str:
    """'lo–hi' of the values (one value when all equal; '—' when empty)."""
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else f"{lo}–{hi}"


def _per_row_table(rows: Sequence[dict]) -> list[str]:
    L: list[str] = []
    if not rows:
        return L
    ids = list(rows[0]["policy"]["per_row_violated_k"])
    L.append("| case | κ | seed | " + " | ".join(ids) + " |")
    L.append("|---|---|---|" + "---|" * len(ids))
    for s in rows:
        pp, mm = s["policy"]["per_row_violated_k"], s["m0"]["per_row_violated_k"]
        L.append(f"| {s['case_id']} | {margin_record(s)['kappa']:g} | {s['seed']} | "
                 + " | ".join(f"{pp.get(i, 0):,} \\| {mm.get(i, 0):,}" for i in ids) + " |")
    return L


def margin_md(base: Sequence[dict], mrows: Sequence[dict], vbase: Sequence[dict], mval: Sequence[dict],
              n_states: int) -> list[str]:
    L: list[str] = []
    a = L.append
    a("## DM-margin re-solves")
    a("")
    a("The zero-margin re-solve above was called a straw man: it re-solves at θ_s but carries no protection against the "
      "DM fractions, which stay hidden (the ration is formulated at d̂ and judged at d_s). This section re-solves the "
      "same LP at θ_s with a safety margin against the hidden DM, following the original study's M1 coefficient-shift "
      "safety margin with its DM margin switched on (`--dm-margin KAPPA`; κ = 0 is the zero-margin policy above, "
      "unchanged). Same cases, world, objective arm, main event, three development streams and 20,000 states per "
      "stream as κ = 0 (same seeds; the draws fingerprints are checked equal below).")
    a("")
    a("### Margin definition")
    a("")
    a("- **Engine M1.** `src/ration_reliability/optimization/safety_margin.py` (`M1_safety_margin`, `build_margin_rows`) "
      "tightens every probabilistic row k to the row-wise worst case of the box centre ± κ·spread: content "
      "a_ij → μ_ij + κ·spread_ij·sign(s_k W_kij) and, with `apply_to_dm`, DM fraction "
      "d_i → d̂_i + κ·spread_d,i·sign(s_k (c_ki − K_k)) in concentration rows or sign(s_k c_ki) in supply rows "
      "(c_ki at the already-shifted content; s_k = −1 for ≥ rows, +1 for ≤ rows); mass fractions are clipped to [0, 1], "
      "DM to [1e−6, 1]; structural rows are unchanged; the ration is executed as q = x / d̂. The official setting "
      "(`official_v2.OFFICIAL_SETTINGS[\"M1\"]`, checked at run time) is coef_directional with `margin_scale` relative "
      "(spread = |μ| for the composition, d̂ for DM) and `apply_to_dm` True in the main setting (DM margin off is the "
      "declared sensitivity line); development grid κ ∈ {0, 0.025, 0.05, 0.075, 0.10} (`run_dev_case_v1.RUN_CONFIG`).")
    a("- **M1's margin is defined on all coefficients.** The composition margin is always on; the DM margin is an "
      "add-on switch. There is no switch that turns the composition margin off. The DM-only variant used here calls "
      "the same engine function through its explicit-spread interface (composition spread 0, DM spread d̂ = the DM "
      "spread of the relative scale, `apply_to_dm` True). The call goes to an observed-composition view of the MAIN9 "
      "problem whose `nominal_theta()` returns θ_s, with every other attribute the problem's own, so the margin is "
      "centred at the observed composition.")
    allm = list(mrows) + list(mval)
    n_fb = sum(int(r["policy"]["lp_fallback_k"]) for r in allm)
    n_inf = sum(policy_extras(r)["lp_infeasible_k"] for r in allm)
    fb_note = (f"every LP fallback in the margin runs ({n_fb:,} states in total) was a proven-infeasible LP"
               if n_fb == n_inf else f"LP fallbacks {n_fb:,}, of which proven infeasible {n_inf:,}")
    a("- **Definition used (DM-only, κ = `--dm-margin`).** In every probabilistic row k and for every ingredient i: "
      "content a_ij = θ_s,ij (unchanged); DM fraction d_i = clip(d̂_i · (1 + κ · sign(s_k (c_ki(θ_s) − K_k))), 1e−6, 1) in "
      "the concentration rows (PN-T1 … PN-T5) and d_i = clip(d̂_i · (1 + κ · sign(s_k c_ki(θ_s))), 1e−6, 1) in the supply "
      "rows (PN-NEL-FIXEDDMI, PN-CP-SUP, PN-CA-ABS, PN-P-ABS); structural rows as in M0; minimise as-fed cost; "
      f"q = x / d̂. A non-optimal LP executes the M0 ration and is counted ({fb_note}).")
    a("- **Why DM-only.** Once θ_s is observed the composition is known; the only uncertainty left in the reference event "
      "is the realised DM d_s. A composition margin would guard against uncertainty that no longer exists, so the "
      "DM-only margin is the fair comparator. The all-coefficient variant (M1 exactly in its official setting, centred "
      "at θ_s; `--margin-scope all`) is implemented and was run on the 500-state validation only (table below).")
    a("- **Checks in every margin run** (first 50 states, stop on failure): the engine M1 on the view at κ = 0 "
      "reproduces the zero-margin re-solve (rows max |ΔA| ≤ 1e−12, ration max |Δq| ≤ 1e−9, same status); the engine's "
      "margin rows at κ equal the definition re-derived independently from the compiled rows (content and DM fraction "
      "of every probabilistic row, max |Δ| ≤ 1e−12); in the DM-only scope the contents equal θ_s exactly; the official "
      "M1 setting is relative with the DM margin on.")
    a("")
    a("### Validation before scaling (500 states, seed 202610031)")
    a("")
    a("| case | scope | κ | M0 fail rate | policy fail rate | LP infeasible / fallback | policy fail rate among LP-solved "
      "states (n solved) | cost ratio (mean, all) | cost ratio (mean, solved) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in list(vbase) + list(mval):
        p, mg, ex = s["policy"], margin_record(s), policy_extras(s)
        fr = "—" if ex["fail_rate_solved"] is None else f"{ex['fail_rate_solved']:.3f} ({ex['lp_solved_k']:,})"
        a(f"| {s['case_id']} | {mg['scope']} | {mg['kappa']:g} | {s['m0']['fail_rate']:.3f} | {p['fail_rate']:.3f} | "
          f"{ex['lp_infeasible_k']:,} / {p['lp_fallback_k']:,} | {fr} | {p['mean_cost_ratio_vs_m0_all_states']:.4f} | "
          f"{_r(p['mean_cost_ratio_vs_m0_solved_states'])} |")
    a("")
    full = [r for r in mrows if r["n_states"] == n_states]
    scopes = [sc for sc in MARGIN_SCOPES if any(margin_record(r)["scope"] == sc for r in full)]
    for sc in scopes:
        a(f"### Ranges over the three development streams ({n_states:,} states each; scope `{sc}`)")
        a("")
        a("| case | κ | streams | policy fail rate | policy CP upper | M0 fail rate | paired diff (policy − M0) | "
          "n10 / n01 | LP infeasible per stream | LP fallback per stream | policy fail rate among LP-solved states | "
          "cost ratio (mean, all) | cost ratio (mean, solved) |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for case in CASES:
            for kappa in (0.0,) + DM_MARGIN_GRID:
                grp = ([r for r in base if r["case_id"] == case] if kappa == 0.0 else
                       [r for r in full if r["case_id"] == case and margin_record(r)["kappa"] == kappa
                        and margin_record(r)["scope"] == sc])
                if not grp:
                    continue
                ex = [policy_extras(r) for r in grp]
                inf_s = " / ".join(f"{e['lp_infeasible_k']:,}" for e in ex)
                fb_s = " / ".join(f"{r['policy']['lp_fallback_k']:,}" for r in grp)
                a(f"| {case} ({ARM_NAME[case]}) | {kappa:g}{' (no margin)' if kappa == 0 else ''} | {len(grp)} | "
                  f"{_rng([r['policy']['fail_rate'] for r in grp])} | "
                  f"{_rng([r['policy']['cp_upper_per_claim'] for r in grp])} | "
                  f"{_rng([r['m0']['fail_rate'] for r in grp])} | "
                  f"{_rng([r['paired']['difference_policy_minus_m0'] for r in grp], '{:+.4f}')} | "
                  f"{_rng([r['paired']['n10_policy_fails_m0_passes'] for r in grp], '{:,}')} / "
                  f"{_rng([r['paired']['n01_policy_passes_m0_fails'] for r in grp], '{:,}')} | "
                  f"{inf_s} | {fb_s} | "
                  f"{_rng([e['fail_rate_solved'] for e in ex])} | "
                  f"{_rng([r['policy']['mean_cost_ratio_vs_m0_all_states'] for r in grp])} | "
                  f"{_rng([r['policy']['mean_cost_ratio_vs_m0_solved_states'] for r in grp])} |")
        a("")
        a("Ranges are min–max over the three streams (not pooled). Policy fail rate includes the fallback states "
          "(M0 executed). The fail rate among LP-solved states is conditional on the LP being feasible in that state, "
          "i.e. on a selected subset of states; it is descriptive, not a policy rate.")
        a("")
        a(f"### Per cell (scope `{sc}`)")
        a("")
        a("| case | κ | seed | policy fail k | policy rate | policy CP upper | M0 fail k | M0 rate | n10 | n01 | "
          "paired diff | LP infeasible | LP fallback | fail k in fallback states | cost ratio (mean, all) | "
          "cost ratio (mean, solved) |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in [r for r in full if margin_record(r)["scope"] == sc]:
            p, m, pr, ex = s["policy"], s["m0"], s["paired"], policy_extras(s)
            a(f"| {s['case_id']} ({s['arm']}) | {margin_record(s)['kappa']:g} | {s['seed']} | {p['fail_k']:,} | "
              f"{p['fail_rate']:.4f} | {p['cp_upper_per_claim']:.4f} | {m['fail_k']:,} | {m['fail_rate']:.4f} | "
              f"{pr['n10_policy_fails_m0_passes']:,} | {pr['n01_policy_passes_m0_fails']:,} | "
              f"{pr['difference_policy_minus_m0']:+.4f} | {ex['lp_infeasible_k']:,} | {p['lp_fallback_k']:,} | "
              f"{p['fail_k_among_fallback_states']:,} | {p['mean_cost_ratio_vs_m0_all_states']:.4f} | "
              f"{_r(p['mean_cost_ratio_vs_m0_solved_states'])} |")
        a("")
        # best kappa per case: lowest mean policy fail rate over the streams (ties -> smaller kappa)
        a(f"### Per-row violation pattern at the best κ per case (scope `{sc}`)")
        a("")
        best_rows: list[dict] = []
        for case in CASES:
            cand = []
            for kappa in DM_MARGIN_GRID:
                grp = [r for r in full if r["case_id"] == case and margin_record(r)["kappa"] == kappa
                       and margin_record(r)["scope"] == sc]
                if grp:
                    cand.append((float(np.mean([r["policy"]["fail_rate"] for r in grp])), kappa, grp))
            if cand:
                fr, kb, grp = min(cand, key=lambda x: (x[0], x[1]))
                a(f"- {case}: best κ = {kb:g} (mean policy fail rate over the {len(grp)} streams {fr:.4f}; κ = 0: "
                  f"{np.mean([r['policy']['fail_rate'] for r in base if r['case_id'] == case]):.4f})")
                best_rows += grp
        a("")
        a("Best κ = the grid value with the lowest mean policy fail rate over the three streams (ties: smaller κ). "
          "Per-row counts (policy | M0), same format as the zero-margin table:")
        a("")
        L.extend(_per_row_table(best_rows))
        a("")
        a("Linear rows by the public evaluator at the realised (θ_s, d_s); the energy row by the Chapter 3 reference "
          "chain. Fallback states carry the M0 ration and therefore the M0 verdicts.")
        a("")
    a("### Notes (DM-margin re-solves)")
    a("")
    if full:
        tot = sum(r["runtime_s"] for r in full)
        a(f"- Size: {n_states:,} states per cell, {len(full)} cells, about {tot / len(full) / 60:.1f} min per cell, "
          f"{tot / 60:.1f} min in total, one process at a time (`nice -n 10`).")
        same_draws, same_m0 = [], []
        for r in full:
            b = [x for x in base if x["case_id"] == r["case_id"] and x["seed"] == r["seed"]]
            if b:
                same_draws.append(r["draws_fingerprint"] == b[0]["draws_fingerprint"])
                same_m0.append(r["m0"]["fail_k"] == b[0]["m0"]["fail_k"]
                               and r["m0"]["per_row_violated_k"] == b[0]["m0"]["per_row_violated_k"])
        a(f"- Same states as κ = 0: draws fingerprint equal to the κ = 0 run of the same case and stream in "
          f"{sum(same_draws)}/{len(same_draws)} margin cells; M0 failure count and per-row M0 counts equal in "
          f"{sum(same_m0)}/{len(same_m0)}.")
        dm = [r for r in full if margin_record(r)["scope"] == "dm_only"]
        if dm:
            thc = sum(int(r["margin"]["n_theta_cells_clipped_total"]) for r in dm)
            a(f"- Clipping (DM-only cells): composition cells clipped in total: {thc}. The engine clips "
              "d̂_i (1 + κ) at 1 for ingredients whose d̂ is close to 1; states with at least one clipped DM cell: "
              f"{_rng([r['margin']['states_with_dm_clip'] for r in dm], '{:,}')} of {n_states:,} per cell (counts "
              "only; such a cell gets a smaller margin than κ d̂).")
        a("- What the DM-only margin does, read off its definition: in the four ≥ supply rows (energy, CP supply, Ca and "
          "P absorbed) it multiplies the DM fraction of every positively contributing ingredient by (1 − κ) (negative "
          "contributors by 1 + κ), while the planned-DM equality and the executed ration stay at d̂. In those rows the "
          "LP must meet each requirement from about (1 − κ) of the planned dry matter. Where the LP has no such "
          "solution the policy executes M0 (fallback), so where most states fall back the policy rate approaches the "
          "M0 rate by construction.")
        a("- The fallback rule (M0) is the one specified for this control. A rule that falls back to the largest "
          "feasible κ, or to the zero-margin re-solve, was not run.")
    a("")
    return L


def public_md(rows: list[dict], tag: str, n_states: int, validation: Sequence[dict] = (),
              k0_regression: Sequence[dict] = ()) -> str:
    base = [r for r in rows if margin_record(r)["kappa"] == 0.0]
    mrows = [r for r in rows if margin_record(r)["kappa"] > 0.0]
    vbase = [v for v in validation if margin_record(v)["kappa"] == 0.0]
    mval = [v for v in validation if margin_record(v)["kappa"] > 0.0]
    L: list[str] = []
    a = L.append
    a(f"# E4 observed-composition re-solve control — result ({tag[:4]}-{tag[4:6]}-{tag[6:]})")
    a("")
    a("**EXPLORATORY, UNREGISTERED, DEVELOPMENT-STREAM CONTROL.** Not part of the frozen v2 / v3 protocol, not a member "
      "of any comparable set, not merged into any official table. Development seeds only. Rates are about the declared "
      "model constraints under the declared distribution; nothing is said about animals. Only counts, rates, bounds and "
      "cost ratios appear here (no ration, no absolute cost, no restricted table value).")
    a("")
    if base:
        s0 = base[0]
        a(f"- Script: `{PUBLIC_DIR.as_posix()}/resolve_lp_control.py`; restricted outputs: "
          f"`{OUT_DEFAULT.as_posix()}/<case>_seed<seed>_n<n>[_<scope>_k<kappa>]/`")
        for lab, grp in (("zero-margin runs", base), ("DM-margin runs", mrows)):
            if not grp:
                continue
            heads = sorted({r['git_head_short'] for r in grp})
            a(f"- Git HEAD at run ({lab}): `{', '.join(heads)}`; tracked files modified under the engine scope "
              f"({', '.join(ENGINE_SCOPE)}) at run time: {sorted({r['git_engine_scope_files_modified'] for r in grp})} "
              f"(other tracked files modified, outside the engine scope: "
              f"{sorted({r['git_tracked_files_modified'] - r['git_engine_scope_files_modified'] for r in grp})})")
        if mrows:
            a("- The one engine-scope file modified during the DM-margin runs is this script (the `--dm-margin` option, "
              "uncommitted at run time); `src/`, `configs/`, `experiments/E0_verification/` and "
              "`experiments/E1_cost_reliability/` were unmodified. The script's sha256 is recorded per run (Checks).")
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
      "ration (solved once at the nominal composition) on the same states (paired). The DM-margin variant is described "
      "in its own section below.")
    a("")
    a("## Validation before scaling (500 states, seed 202610031)")
    a("")
    a("| case | M0 fail rate (500) | policy fail rate (500) | official v3 M0 rate, SD-H0, roots 0/1/2 (orientation) |")
    a("|---|---|---|---|")
    for v in vbase:
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
    for s in base:
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
    for s in base:
        p, m, pr = s["policy"], s["m0"], s["paired"]
        mp = _p(pr["mcnemar_exact_p_two_sided"]) or "—"
        a(f"| {s['case_id']} | {s['seed']} | {p['violated_k']:,} / {p['unknown_k']:,} | {m['violated_k']:,} / "
          f"{m['unknown_k']:,} | {m['cp_upper_per_claim']:.4f} | {mp} | "
          f"{p['plan_out_of_domain_k']:,} | {m['plan_domain_status']} | {p['min_cost_ratio_vs_m0']:.4f} / "
          f"{p['max_cost_ratio_vs_m0']:.4f} |")
    a("")
    a("### Per-row violation counts of the main-event members (policy | M0)")
    a("")
    if base:
        ids = list(base[0]["policy"]["per_row_violated_k"])
        a("| case | seed | " + " | ".join(ids) + " |")
        a("|---|---|" + "---|" * len(ids))
        for s in base:
            pp, mm = s["policy"]["per_row_violated_k"], s["m0"]["per_row_violated_k"]
            a(f"| {s['case_id']} | {s['seed']} | " + " | ".join(f"{pp.get(i, 0):,} \\| {mm.get(i, 0):,}" for i in ids)
              + " |")
        a("")
        a("Linear rows by the public evaluator at the realised (θ_s, d_s); the energy row by the Chapter 3 reference "
          "chain (the event's energy verdict). A state can violate several rows.")
        a("")
    a("## Notes and deviations")
    a("")
    tot = sum(r["runtime_s"] for r in base)
    a(f"- Size: {n_states:,} states per cell as specified (about {tot / max(1, len(base)) / 60:.1f} min per cell, "
      f"{tot / 60:.1f} min for the {len(base)} cells, one process at a time).")
    a("- The streams are not pooled; each row is one independent development stream.")
    if all(int(r["policy"]["lp_fallback_k"]) == 0 for r in base):
        a("- The LP fallback rule (execute M0 when the re-solve LP is not optimal) was never triggered at κ = 0: every "
          "state's LP was optimal.")
    else:
        a(f"- LP fallback at κ = 0: {[int(r['policy']['lp_fallback_k']) for r in base]} states per cell.")
    a("- Structural rows are taken at the nominal state; they are composition-free in both cases (checked), so this is "
      "identical to building them from θ_s.")
    a("- Additional descriptive quantities beyond the specification: the violated / unknown split, M0 Clopper–Pearson "
      "bound, exact McNemar p, the plan-level Table 5-1 status of the re-solved rations, per-row violation counts and "
      "the min / max cost ratio.")
    a("- The control does not isolate why the re-solve still fails: the DM fractions stay unobserved (rations are "
      "formulated at d̂ and judged at d_s) and the energy verdict is the Chapter 3 reference chain while the LP uses the "
      "linear fixed-DMI row. A d = d̂ counterfactual was not run. The DM-margin re-solves below add a margin for the "
      "DM channel; they do not observe d_s either.")
    a("")
    if mrows or mval:
        L.extend(margin_md(base, mrows, vbase, mval, n_states))
    a("## Checks (every run)")
    a("")
    cur = file_sha256(Path(__file__))
    for lab, grp in (("zero-margin runs and their validation", list(base) + list(vbase)),
                     ("DM-margin runs and their validation", list(mrows) + list(mval))):
        if not grp:
            continue
        shas = sorted({r["code_sha256"] for r in grp})
        note = ("" if shas == [cur] or not k0_regression else
                " (these runs predate the `--dm-margin` option; the zero-margin path of the current script is checked "
                "against them below)")
        a(f"- Script sha256 at run time ({lab}): `{', '.join(x[:16] for x in shas)}`; equal to the script that wrote "
          f"this file (`{cur[:16]}`): {shas == [cur]}{note}")
    for k in k0_regression:
        a(f"- Zero-margin path unchanged ({k['case_id']}): the current script (`{k['current_sha16']}`) re-run at κ = 0 on "
          f"the 500-state validation stream reproduces the original run (`{k['original_sha16']}`) byte for byte in "
          f"every per-state array (rations, costs, LP statuses, verdicts, row violations): {k['all_identical']}; draws "
          f"fingerprint equal: {k['draws_fingerprint_equal']}")
    for s in base + [r for r in mrows if r["n_states"] == n_states]:
        c = s["checks"]
        e = s["energy_relinearisation_check"]
        mg = margin_record(s)
        line = (f"- {s['case_id']} seed {s['seed']}" + ("" if mg["kappa"] == 0 else f" κ {mg['kappa']:g} ({mg['scope']})")
                + f": stream/label guard {all(c[k] for k in ('stream_is_test', 'stream_id_is_development_root'))}; "
                f"labels aligned {c['labels_match_problem'] and c['labels_match_reference']}; world C0 {c['world_correlation_C0']}; "
                f"structural rows composition-free {c['structural_rows_composition_free']}; linearisation DMI = planned DM "
                f"{c['linearisation_dmi_equals_planned_dm']}; energy column = builder linearisation at θ_s "
                f"{e['energy_column_equals_affine_linearisation']} (direct per-feed check {e['energy_column_equals_direct_feed_evaluation']}); "
                f"re-solve at nominal θ reproduces engine M0 {s['regression_resolve_at_nominal_equals_engine_m0']}; "
                f"M0 rate in the validation window {M0_EXPECTED_RANGE} {s['m0']['in_expected_range']}; ")
        if mg["kappa"] > 0:
            dc = s["margin"]["definition_check"]
            line += (f"official M1 setting relative + DM margin {c['m1_official_setting_relative_with_dm_margin']}; "
                     f"engine M1 on the view at κ = 0 = zero-margin re-solve {dc['k0_reproduces_resolve']}; margin rows = "
                     f"definition {dc['rows_follow_definition']} (content max |Δ| {dc['content_max_abs_diff']:.1e}, DM max "
                     f"|Δ| {dc['dm_max_abs_diff']:.1e}, {dc['n_states_checked']} states); ")
        line += f"problem fingerprint `{s['problem_fingerprint'][:16]}`; runtime {s['runtime_s'] / 60:.1f} min"
        a(line)
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
    ap.add_argument("--dm-margin", type=float, default=0.0, metavar="KAPPA",
                    help="M1 safety margin kappa for the re-solve at theta_s (default 0 = zero-margin re-solve; "
                         f"declared grid {list(DM_MARGIN_GRID)})")
    ap.add_argument("--margin-scope", choices=MARGIN_SCOPES, default="dm_only",
                    help="dm_only (default): M1's relative DM margin only; all: M1 in its official setting "
                         "(composition + DM), both centred at theta_s; ignored when --dm-margin is 0")
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
        k0reg = k0_regression_check(out_root)
        for p in write_public(rows, args.tag, int(args.n_states), validation, k0reg):
            print(f"wrote {os.path.relpath(p, REPO)} ({len(rows)} cells)")
        return 0
    if args.case is None:
        raise SystemExit("--case is required (or --summarise)")
    if int(args.seed) not in DEV_SEEDS:
        print(f"note: seed {args.seed} is not one of the declared development seeds {list(DEV_SEEDS)}", flush=True)
    kappa = float(args.dm_margin)
    if not (math.isfinite(kappa) and kappa >= 0.0):
        raise SystemExit("--dm-margin must be finite and >= 0")
    if kappa > 0.0 and kappa not in DM_MARGIN_GRID:
        print(f"note: kappa {kappa:g} is not in the declared grid {list(DM_MARGIN_GRID)}", flush=True)
    s = run_cell(args.case, int(args.n_states), int(args.seed), out_root, chunk=int(args.chunk), kappa=kappa,
                 scope=args.margin_scope)
    p, m, pr = s["policy"], s["m0"], s["paired"]
    print(json.dumps({"case": s["case_id"], "seed": s["seed"], "n": s["n_states"], "kappa": margin_record(s)["kappa"],
                      "margin_scope": margin_record(s)["scope"], "lp_infeasible": p["lp_infeasible_k"],
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
