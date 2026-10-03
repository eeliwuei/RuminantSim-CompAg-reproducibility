# English overview of the specification documents

The documents in `docs/`, mostly written in Chinese, specify the reference problem, uncertainty layers, decision timing, official-run plan, package scopes and engine interface of this repository. This overview translates their structure and key terms and points to the files they govern, without adding results or any withheld value.

The documents are dated 24–26 September 2026. They cite internal files that are not included here, among them `reports/`, `audit/`, `scripts/package_release.py`, `docs/INFORMATION_VALUE_DESIGN.md`, `docs/value_definition.md` and `docs/value_semantics_decision.md`.

| Document | What it specifies | Read it when |
|---|---|---|
| `reference_problem_v1.md` | The single definition that scores every ration: scope, constraint table, adjudication rules, events, energy reference, domain | You need a constraint or event id |
| `reference_problem_v2.md` | Premise planning row for all methods; ration-level domain reading | Read with v1; v2 is the later version used by the official-run plan (not frozen when written) |
| `uncertainty_data_layers.md` | Uncertainty layers, evidence tiers, permitted wording | You need to interpret an SD world |
| `decision_timing_and_units.md` | Information flow, units, q = x/d̂, no renormalisation, prices | You need to know what a rule may use when it chooses a ration |
| `official_run_v2_plan_20260926.md` | Checklist from v2 to freeze and official run | You trace the official-run design |
| `package_scopes.md` | Contents of the two release packages and their gates | You find a file or value missing |
| `ENGINE_API.md` | Engine data model, evaluator, methods, streams, run records | You use or extend `src/ration_reliability/` |

## reference_problem_v1.md

Headings: 0 Key points; 1 Scope (one cow, one inventory, one price); 2 Constraints; 3 Adjudication rules; 4 Reference events and rates; 5 Energy reference; 6 Applicability domain; 7 Data layers; 8 Stream policy; 9 Solve and evaluation flow; 10 Commands and compute; 11 Freeze fingerprint; 12 Checks; 13 Open items; 14 Non-claims.

- Every ration is scored by one function, `evaluate_reference(q, draws, spec)`, against `configs/dev_case_v1/reference_constraints.csv` (27 rows: the 22 rows of `constraints.yaml` plus 5 planning rows).
- **Nine-row main reference event**: `PN-NEL-FIXEDDMI` (energy, judged with the project's chapter-3 reference chain), `PN-CP-SUP`, `PN-CA-ABS`, `PN-P-ABS`, `PN-T1`–`PN-T5` (Table 5-1 rows). A state fails if any member is violated. `PN-CP-HI` and `PN-EE-HI` are labelled research-assumption rows: optimised and reported, but outside the main event.
- Adjudication rules A1–A7 decide a row's status from its source only. They were written after development results had been seen, as disclosed.
- Rates use the full denominator; a state with no violation but an undefined member is `unknown`, and rates are reported as the pair [violated/S, (violated + unknown)/S] with Clopper–Pearson bounds on the upper value.
- Code: `evaluation/reference.py`, `nutrition/energy_reference.py`, `nutrition/domain.py`, `run_endpoint_ablation.py` (in `experiments/E1_cost_reliability/`).

## reference_problem_v2.md (later version)

Headings: 1 What changed relative to v1; 2 Why (not for feasibility); 3 Non-claims and limits; 4 Two domain-rule implementations; 5 Code and tests; 6 Items open before freeze.

- Every arm and method, including M0, gets the structural planning row `SH-PLAN-T51-DGC-SHARE`, which requires dry-ground corn to supply at least a project-assumed share τ of planned dietary starch (computed from q, d̂ and nominal starch).
- The main event uses a **ration-level ("plan") domain reading**: whether the Table 5-1 premise holds is decided once per ration from its planned diet; inside the domain the T rows are judged in every state, outside it they are `unknown`. The ration-level event is `main_reference_plan_domain`; the per-state event `main_reference` is kept and reported alongside it.
- The constraint table is `reference_constraints_v2.csv` (28 rows, column `optimization_all_arms`); the v1 file is unchanged. Both changes followed development results (disclosed) and are justified nutritionally, not by feasibility.
- Code: `nutrition/domain.py` (`premise_planning_coefficients`, `plan_domain_status`), `experiments/E1_cost_reliability/official_v2.py`, `evaluation/reference.py`.

## uncertainty_data_layers.md

Headings: 0 Key points; 1 Terms; 2 Data-layer table and permitted wording; 3 H0; 4 Narrowed SD of S2; 5 Declared sensitivity range; 6 Random-stream policy; 7 Wording of conclusions; 8 Open items.

- **H0**: NASEM Table 19-1 means and observed total SDs treated as the true composition variation of the current batch (a strong assumption, not a real-world baseline); independent marginals, `TN_MM` with `BETA_MM` fallback.
- **S2 / SD-S2**: the table SD multiplied by within-farm ratios; only 3 of 48 stochastic cells (corn-silage DM, NDF and starch, 14-day basis) use a ratio registered for that same cell, and the other 45 borrow the 12-month corn-silage DM ratio. All 48 cells are labelled `unidentified` / `research_scenario_assumption`; SD-S2 was screened for attainability and is a method-calibration point, not a supported lower limit.
- **C1** (corn-silage correlations), **H1** (two-level farm world) and the sensitivity points `SD-SRC14`, `SD-12MO`, `SD-SRC12` are declared scenarios. The official plan scores rations in five SD worlds: SD-H0, SD-S2 and the three sensitivity points; C1 and H1 are not among them.
- Evidence tiers: `code_tested`, `holder_verified`, `independently_reproduced`, `assumption_only`, `blocked`.
- Files: `configs/uncertainty.yaml`, `configs/streams_policy.yaml`, `uncertainty/spec.py`, `uncertainty/factory.py`.

## decision_timing_and_units.md

Headings: 1 Information flow t0→t4; 2 Symbols and units; 3 Decision variable q and DM representation x; 4 Evaluation without renormalisation; 5 Two DM modes; 6 Price basis and cost units; 7 Assay fees; 8 Inventory and batches; 9 "Supply ≠ intake"; 10 Contract mapping; 11 Leakage and unit tests; 12 Open items.

- t0 inventory, prior and d̂ known; t1 assay decision (t0 information only); t2 assay result Z (with sampling and laboratory error); t3 the as-fed ration q is chosen; t4 feeding and evaluation. **Hidden at selection**: the batch composition θ, including the true DM d_i(θ), and any untested value.
- The executed decision is q in kg as-fed/head/d; a method working in DM uses q_i = x_i / d̂_i. Concentration constraints are written in D-multiplied linear form with the state's own D; scaling actual DM back to the planned total is forbidden.
- DM modes: `DM-KNOWN` (controlled sub-model) and `DM-UNCERTAIN` (used in the official plan). Cost C(q) = Σ p_i q_i does not vary with θ; supplied feed is assumed fully eaten. The break-even assay-fee item in §7 is withdrawn in the document.
- Files: `configs/constraints.yaml`, `nutrients.yaml`, `prices.yaml`, `assays.yaml`, `methods.yaml`, `normalization/units.py`.

## official_run_v2_plan_20260926.md

Headings: 0 Five problems to solve before freeze (F1–F5); 1 Official design; 2 Files changed, in batches B1–B6; 3 New tests; 4 Order; 5 Compute; 6 Items the research lead must fix before freeze; 7 Not allowed; 8 Planner's uncertainties.

An execution checklist, not a result. Design: main arm MAIN9 in five SD worlds, transparency arms FULL11 and PART6P5 in two; methods M0, M1 (with and without DM margin), M2, M3a, M3b; an N ladder; parameter selection by one-sided Clopper–Pearson upper bound ≤ α on the validation stream; membership of the comparable set by the upper bound of the main event on the test stream. §7 forbids changing α, thresholds, τ, SD points, grids, rows or tolerances for feasibility, and drawing from reserved roots before the freeze. Files: `experiments/E1_cost_reliability/official_v2.py`, `uncertainty/streams.py`, `configs/protocol_freeze.json`, `scripts/px_official_*.sh`, `scripts/verify_official_outputs.py`.

## package_scopes.md

Headings: 1 The two packages; 2 Scripts for licensed holders; 3 Receipts and seals; 4 Items for the research lead to decide (not legal conclusions); 5 Packaging program and gates; 6 Development-run tables in the internal package; 7 Later revisions (tenth gate: restricted ration-value fingerprint).

- The **external reproduction package** lets readers without restricted inputs run code, tests and the synthetic example; the **internal audit package**, meant for the research lead and for checkers the research lead names, adds process records, decision logs, blocker lists, receipts and audit reports. Neither carries restricted data.
- Gates cover forbidden paths, secrets, NASEM fingerprints, local user paths, absorption coefficients and identifiability: public q, costs, prices and the planned-DM equation must not allow ingredient DM to be solved. In the September development-run tables covered by this document, rations therefore appear only as `q_hash`. The present repository additionally releases the as-fed rations of every fixed rule (one decimal) together with an identifiability check (`data/aggregate_results/released_rations_and_requirements/`, see README *Data policy*). The full-precision vectors of the complete candidate pools remain unreleased.

## ENGINE_API.md

Headings: 0 Quick start; 1 Units and decision quantities; 2 Data structures; 3 Constraints; 4 Common evaluator; 5 SolveResult; 6 Methods; 7 Random streams; 8 UncertaintyModel; 9 YAML configuration and validation; 10 Run record; 11 Worked example; 12 Status and limits; 13 Uncertainty-model factory; 14 Reference energy check and Table 5-1 domain.

- Data model (`datamodel.py`): `Provenance` (status `sourced`, `research_scenario_assumption`, `pending_user_decision` or `synthetic_test_only`), `IngredientRecord` (d̂ as `dm_estimate`), `ConstraintSpec`, `RationProblem`, `RationDecision`, `SolveResult`, `EvaluationResult`.
- Constraint classes: `structural_hard` (q and d̂ only), `probabilistic_nutrition` (only these enter the joint event), `diagnostic_only`; kinds `concentration`, `supply`, `as_fed`.
- The common evaluator returns signed margins, `joint_violation`, `joint_unknown`, `rate_lower`, `rate_upper` and Clopper–Pearson intervals (Monte Carlo error only). Failure statuses of `SolveResult` carry no ration and no cost.
- Methods: `M0_nominal`, `M1_safety_margin`, `M2_joint_chance_saa`, `M2b_marginal_bonferroni_saa`, `M3a_box_robust`, `M3b_budget_robust`, `M3c_scenario_set_robust` (`optimization/`). Streams `opt`, `validation`, `test`, `outer` are spawned from one root (`uncertainty/streams.py`).
- A **run record** (`io/run_record.py`) stores command, times, exit status, stream ids, solver version, tolerances (`mip_rel_gap` required for pilot/official runs), config, data and output hashes and code identity (`src_tree_sha256`, git HEAD, dirty state); the hash of an input that was not supplied is recorded as `None`, not invented.

## Terms used in the article and README

The nine-row main reference event (`main_reference`, with the ration-level domain reading of v2) is the failure event of the article. The methods M0–M3 specified here are the original formulation methods; each returns one fixed as-fed ration. These documents date from before the candidate pools and the fixed, nutrient-informed and uncertainty-aware selection rules of the later protocols, which are described in the article, the Supplementary Information and `extensions/`. Where the Supplementary Information or data files say menu, recipe or legal, read candidate pool, ration or admissible.

## Glossary

| Chinese | English |
|---|---|
| 参考问题 | reference problem |
| 主参考事件 | main reference event |
| 联合违约 | joint violation |
| 违约率 / 上界率 | violation rate / upper rate |
| 余量 | margin |
| 计划行 | planning row |
| 结构约束 | structural constraint |
| 概率营养约束 | probabilistic nutrition constraint |
| 诊断行 | diagnostic row |
| 适用域 / 域外 | applicability domain / out of domain |
| 配方层域读法 / 逐状态读法 | ration-level (plan) domain reading / per-state reading |
| 裁决规则 | adjudication rule |
| 有来源 / 研究假设 | sourced / research assumption |
| 受限值 | restricted value |
| 持有方 | holder (of a licensed copy) |
| 开发冻结 / 协议冻结 | development freeze / protocol freeze |
| 随机流 / 预留根 | random stream / reserved root |
| 可比较可行集合 | comparable feasible set |
| 选参 | parameter selection |
| 原物质（鲜重）/ 干物质 | as-fed / dry matter (DM) |
| 决策时估计 d̂ | decision-time estimate d̂ |
| 参考能量链 | reference energy chain |
| 证据层级 | evidence tier |
| 回执 / 闸门 | receipt / gate |
| 内部审计包 / 外部复现包 | internal audit package / external reproduction package |

## Withheld numbers

NASEM (2021) values, journal table values with an undecided licence, reserved roots and quantities that return them in one step are not printed. In `configs/` such a field holds `null` plus a `value_ref` string beginning "withheld (public copy)" that names the source table, page or equation; the documents show ⟨withheld⟩. In the reference constraint tables the column `threshold_public` gives a `value_ref` pointer in place of a restricted bound, and drivers redact nominal planning-row margins. A holder of a licensed copy places the transcribed tables, feed-library extract and restricted value files under `data/restricted_local/` (listed in `data/README.md`), runs `scripts/preflight_dev_case.py` (lists missing inputs, prints no value), then `scripts/build_dev_case.py`, which writes build outputs, including `build_report.json`, only to `data/restricted_local/`. Public files are never filled in place.
