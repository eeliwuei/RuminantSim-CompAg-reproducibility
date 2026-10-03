# E4 — observed-composition re-solve control ("observe the composition, then re-solve the nominal LP")

**Status: exploratory, unregistered, development-stream control.** It is not part of the frozen v2 or v3
protocol, not a member of any comparable set, and never merged into an official table. No number here replaces an
official one. "Failure" means a violation (or an unresolved verdict) of the declared model constraints under the
declared distribution; nothing is said about animals.

## Question

How reliable would a policy be that could see the realised composition of every ingredient batch before
formulating, but still formulated by the nominal least-cost LP (no safety margin, no chance constraint)? The
control separates "the nominal LP lacks information" from "the nominal LP sits on its constraint boundaries".

A reviewer called the zero-margin re-solve a straw man. The `--dm-margin KAPPA` option answers with a fair version:
the same LP re-solved at the observed composition, with a safety margin against the DM fractions, which stay hidden.
The margin is the original study's M1 coefficient-shift margin with its DM margin switched on (see "DM-margin
re-solves" below). KAPPA = 0, the default, is the zero-margin policy, unchanged.

## Cases, world, streams

- Cases: reference problem v3 arm A (`dev_case_v3a`, planned DM supply = Eq 2-1 × 1.1) and arm C
  (`dev_case_v3c`, arm A + whole cottonseed). Both are built by the official v3 path
  (`official_v3.activate` + `official_v3.build_official_context_v3`): the case problem + the premise planning row
  (`official_v2.build_v2_cfg0`), the MAIN9 objective arm (`run_endpoint_ablation.planned_arm_cfg`), the arm's
  v2-rule-set reference table and `ReferenceSpec`. Inputs are the built restricted case inputs under
  `data/restricted_local/dev_case_v3a/` and `dev_case_v3c/`, checked by `require_dev_case_inputs` and the build
  report hashes.
- World: SD-H0 (TAB; `official_v3.build_sd_point_spec_v3`), independent cells (C0), energy column from
  `EnergyColumnModel` (the case's fixed-DMI linearisation at the planned intake).
- Streams: three development roots declared for this control, `202610031`, `202610032`, `202610033`
  (`DEV_SEEDS`), test stream only (`root=<seed>/test/0`), 20,000 states each. `run_endpoint_ablation.check_seed`
  refuses a reserved formal root before any draw, and `RandomStreams` refuses one at construction. The same three roots
  are used for both cases; the states differ between cases (arm C carries the cottonseed cells).

## Policy

For every test state s = (theta_s, d_s):

1. **Observed-composition re-solve.** The MAIN9 LP of the nominal M0 problem, assembled exactly as
   `optimization/m0_nominal.solve` assembles it (`linear_rows` of the imposed rows → `assemble_x_space_lp` →
   `highs.run_linprog`, official method solver options), with theta_s in place of the nominal composition in every
   probabilistic (nutrient) row: CP supply, Ca and P absorbed, the five Table 5-1 rows and the fixed-DMI energy
   row. The energy row's per-ingredient coefficient is the state's `NEL_fixedDMI` column. That column is the builder's
   fixed-DMI linearisation (`nutrition/energy.linearise_nel_fixed_dmi`) evaluated at theta_s at the planned DM supply.
   The run checks it against the direct per-feed evaluation (`feed_energy_terms`). The bound NEL_req − C0 does not
   depend on composition. DM fractions are held at the planned d_hat (`problem.dm_estimates()`, the M0 value). Every
   structural row is taken at the nominal state, exactly as in M0. These rows are the planned-DM equality SH-DM-PLAN,
   the inclusion caps, the premise planning row SH-PLAN-T51-DGC-SHARE and the planned CP / EE limits SH-PLAN-CP-HI /
   SH-PLAN-EE-HI; they are composition-free, which the run checks. The objective is minimum as-fed cost. If the LP is
   not optimal (infeasible, failed, or missing a coefficient), the policy executes the nominal M0 ration in that state,
   and the state is counted as an LP fallback.
2. **Scoring.** `q_s` is scored with the canonical reference evaluator `evaluate_reference` on the realised state
   (theta_s, d_s) only (a one-state `DrawSet`). The main event is `main_reference_plan_domain`, and failure =
   violated OR unknown.
3. **Paired comparator.** The nominal M0 ration is solved once at the nominal composition and scored on the same
   states.

## DM-margin re-solves (`--dm-margin KAPPA`, `--margin-scope`)

- **Engine M1.** `src/ration_reliability/optimization/safety_margin.py` (`M1_safety_margin`) shifts the coefficients
  of every probabilistic row to the row-wise worst case of the box centre ± KAPPA·spread. The content moves as
  a_ij → μ_ij + KAPPA·spread_ij·sign(s_k W_kij). With `apply_to_dm`, the DM fraction moves as
  d_i → d̂_i + KAPPA·spread_d,i·sign(s_k (c_ki − K_k)) in concentration rows and sign(s_k c_ki) in supply rows, with
  s_k = −1 for ≥ rows and +1 for ≤ rows. DM is clipped to [1e−6, 1] and mass fractions to [0, 1]. Structural rows
  are unchanged, and the ration is executed as q = x / d̂. The official setting is
  `official_v2.OFFICIAL_SETTINGS["M1"]`, checked at run time: coef_directional, `margin_scale` relative
  (spread = |μ| for the composition, d̂ for DM), and `apply_to_dm` True in the main setting.
- **Centre at θ_s.** The engine function is called, not re-implemented, on `ObservedCompositionView`. This view is the
  MAIN9 problem with `nominal_theta()` returning θ_s; every other attribute is the problem's own. M1 reads the
  composition only through `nominal_theta()`.
- **`--margin-scope dm_only` (default, used for the 20,000-state grid).** M1's margin is defined on all coefficients;
  the engine has no switch that turns the composition margin off. The DM-only variant therefore uses the engine's
  explicit-spread interface: composition spread 0, DM spread d̂ (the DM spread of the relative scale), `apply_to_dm`
  True. Per probabilistic row k and ingredient i, the content stays at θ_s and
  d_i = clip(d̂_i (1 + KAPPA sign(s_k (c_ki(θ_s) − K_k))), 1e−6, 1) in concentration rows, or
  clip(d̂_i (1 + KAPPA sign(s_k c_ki(θ_s))), 1e−6, 1) in supply rows. This is the fair comparator: once θ is observed,
  the only remaining uncertainty is DM.
- **`--margin-scope all`.** M1 exactly in its official setting (composition and DM margins), centred at θ_s. It was run
  on the 500-state validation only.
- **Fallback.** If the margin LP is not optimal, the M0 ration is executed and the state counted (`lp_fallback_k`;
  `lp_infeasible_k` counts the proven-infeasible LPs).
- **Grid.** KAPPA ∈ {0.025, 0.05, 0.10, 0.15} (`DM_MARGIN_GRID`) on the same three development streams and 20,000
  states as KAPPA = 0, with the same seeds. The draws fingerprints are checked equal.

## Reported per cell (case × stream)

n; failure count and rate of the policy (with the violated / unknown split); one-sided Clopper–Pearson upper bound at
per-claim error 0.05 / 10,000 (`scipy.stats.beta.ppf(1 - 5e-6, k + 1, n - k)`); M0 failure rate on the same states;
paired counts n10 (policy fails, M0 passes) and n01 (policy passes, M0 fails); paired difference (n10 − n01) / n; exact
McNemar p (descriptive); LP fallback count; mean as-fed cost of the executed policy rations divided by the M0 cost
(a ratio only); per-row violation counts of the main-event members. For the margin runs also: KAPPA and scope, the
proven-infeasible count, the number of LP-solved states, the failure count among LP-solved states (descriptive and
conditional on feasibility), and the clipping counts of the margin.

## Checks run in every cell

Each check stops the run if it fails:

- the test stream comes from a development root (stream / label guard);
- the draw labels are aligned with the problem and the reference spec;
- the world is C0;
- the linearisation DMI equals the planned DM;
- the energy column equals the builder linearisation at theta_s, and the direct per-feed evaluation;
- the structural rows are composition-free;
- the re-solve LP at the nominal composition reproduces the engine M0 (ration and objective within 1e-9);
- the single-state evaluation path reproduces the batch M0 verdicts on the first 200 states.

The 500-state validation also checks that the M0 failure rate falls in the expected TAB range.

Margin runs (KAPPA > 0) add these checks, on the first 50 states:

- the official M1 setting is relative with the DM margin on;
- the engine M1 on the observed-composition view at k = 0 reproduces the zero-margin re-solve (rows and ration);
- the engine's margin rows at KAPPA equal the definition, re-derived independently from the compiled rows (content
  and DM fraction of every probabilistic row, max |Δ| ≤ 1e−12);
- in the DM-only scope, the contents equal θ_s exactly.

The zero-margin path of the current script was re-run on the 500-state validation stream (`_regression_k0_500/`).
`--summarise` compares it with the original validation run and reports whether every per-state array (rations,
costs, statuses, verdicts, row violations) is byte-identical.

## Files

- `resolve_lp_control.py`: the control. Arguments: `--case`, `--n-states`, `--seed`, `--out`, `--dm-margin`,
  `--margin-scope`, plus `--summarise`.
  It records the git HEAD, the engine problem fingerprint (MAIN9 and FULL11 compiled), and the world, linearisation,
  reference-spec and draw fingerprints.
- `RESULT_20261003.md`, `results_public.csv`: public results (counts, rates, bounds, cost ratios only). The CSV holds
  one row per (case, seed, KAPPA), with columns `margin_scope` and `kappa` (`none` / `0` for the zero-margin rows).
- Restricted: `data/restricted_local/exploratory_v2_1/resolve_lp_control/<case>_seed<seed>_n<n>/` (zero margin) and
  `<case>_seed<seed>_n<n>_<scope>_k<KAPPA>/` (margin runs); validation under `_validation500/`, the zero-margin
  regression under `_regression_k0_500/`, run logs `run_full_20000*.log` and `run_validation500_dm_margin.log`. Each directory
  holds `per_state.npz` (rations, absolute costs, per-state verdicts), `run_record_restricted.json` and
  `summary_public.json` (the public subset). The `--out` guard refuses any location outside
  `data/restricted_local/exploratory_v2_1/` and every official output location.

## Rerun

```
cd <repo root>
for c in dev_case_v3a dev_case_v3c; do for s in 202610031 202610032 202610033; do
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 /opt/homebrew/opt/python@3.11/bin/python3.11 \
    experiments/E4_resolve_lp_control/resolve_lp_control.py --case $c --n-states 20000 --seed $s
done; done
# DM-margin re-solves (DM-only scope), same streams
for k in 0.025 0.05 0.1 0.15; do for c in dev_case_v3a dev_case_v3c; do for s in 202610031 202610032 202610033; do
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 /opt/homebrew/opt/python@3.11/bin/python3.11 \
    experiments/E4_resolve_lp_control/resolve_lp_control.py --case $c --n-states 20000 --seed $s --dm-margin $k
done; done; done
# 500-state validation (both scopes) and the zero-margin regression run
R=data/restricted_local/exploratory_v2_1/resolve_lp_control
for c in dev_case_v3a dev_case_v3c; do
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 /opt/homebrew/opt/python@3.11/bin/python3.11 \
    experiments/E4_resolve_lp_control/resolve_lp_control.py --case $c --n-states 500 --seed 202610031 --out $R/_regression_k0_500
  for sc in dm_only all; do for k in 0.025 0.05 0.1 0.15; do
    PYTHONDONTWRITEBYTECODE=1 nice -n 10 /opt/homebrew/opt/python@3.11/bin/python3.11 \
      experiments/E4_resolve_lp_control/resolve_lp_control.py --case $c --n-states 500 --seed 202610031 \
      --dm-margin $k --margin-scope $sc --out $R/_validation500
  done; done
done
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 \
  experiments/E4_resolve_lp_control/resolve_lp_control.py --summarise --n-states 20000
```

Run one process at a time.
