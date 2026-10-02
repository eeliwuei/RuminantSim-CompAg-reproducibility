# protocol.yaml 字段候选说明（不修改 protocol.yaml）

- 编写：P2b_constraints_units 子任务（项目记录），2026-09-24。
- 本文件状态（合同 §2.3）：`implemented`。只是候选说明；**没有改动** `configs/protocol.yaml`（当前 `version: "0.1-draft"`、`status: "planned"`、`freeze.is_frozen: false`）。
- 用法：由总负责人或用户逐项确认后，再写入 `configs/protocol.yaml`。标“可填”的是本阶段已有依据的候选值；标“待拍板”的列出决策编号；标“不可填”的要等运行、数据或冻结。
- 决策编号：D-xx 见 `audit/_parts/phase1_user_approvals.md` §3；PUD-P2a-xx / B-P2a-xx 见 `configs/animal_profile.yaml`、`audit/_parts/P2a_log_additions.md`；P0-x/P1-x 见数据可行性审计 §10；B-P2b-xx 见 `audit/_parts/P2b_log_additions.md`。

---

## 1 顶层

| 字段 | 当前值 | 建议 | 类别 | 依据 |
|---|---|---|---|---|
| `route` | null | `summary_based_computation` | 可填 | `audit/route_decision.md` §0（路线 B） |
| `status` | planned | 保持 `planned`，直到冻结 | 不改 | 合同 §2.3 |
| `scientific_results_exist` | false | 保持 false | 不改 | 无任何正式运行 |

## 2 `inputs`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `actual_engine_entrypoint` | null（`src/ration_reliability/` 已有代码但来源待核，BLOCKERS B-010；尚未通过数值测试） | 不可填 | 合同 §8 |
| `actual_engine_verified` | false | 不改 | — |

## 3 `data`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `source_registry_path` | `sources/source_registry.csv`（本任务期间已出现，88,899 B，17:01 由其他子任务写入；本任务只读，已在各 config 的 `registry_crosswalk` / `registry_source_id` 中引用其 source_id） | 可填（待总负责人确认该文件为定稿） | 合同 §5.3 |
| `empirical_or_summary` | `summary` | 可填 | route_decision §0；数据可行性审计 §0 Q1 |
| `legal_use_verified` | false | 不改 | NASEM All rights reserved；P0-7 未办 |
| `redistribution_verified` | false | 不改 | P0-7 |
| `independent_unit` | `lineage (descriptive only, ≤4)`；外检模块另按各自单位 | 可填（文字） | 数据可行性审计 §3.2、§9.3 |
| `paired_nutrients_available` | false | 可填 | NASEM p.361（各成分 N 不同，不能配对）；`nutrients.yaml` NR-02 |
| `observation_error_decomposition_supported` | `partial`（只有青贮 × DM/NDF/淀粉/灰分/CP 有三分量；精料只有残差区间） | 可填（文字） | 数据可行性审计 §4.1–4.6；`assays.yaml` |
| `raw_manifest_sha256` 等 | null | 不可填 | 未冻结 |

## 4 `scope`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `animal_profile_file` | `configs/animal_profile.yaml` | 可填（文件存在；内容待 PUD-P2a-01…07） | P2a |
| `nutrition_standard_and_version` | `NASEM 2021, 8th rev. ed., DOI 10.17226/25806`（B 层评价器 `nasem_dairy==1.0.2` 未安装） | 可填 | model_audit；数据可行性审计 §5.1 |
| `target_ingredient_count_range` | 保持 [15, 25]；实际核心 21 种（15 随机 + 6 确定性） | 不改 | `inventories.yaml` |
| `claim_is_complete_nutritional_adequacy` | false | 不改 | `constraints.yaml`（无能量、MP 约束） |

## 5 `units_and_decisions`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `decision_unit` | 保持 `kg_as_fed_per_head_per_day` | 不改 | 合同 §6.3 |
| `cost_unit` | `<CUR>_per_head_per_day`，货币随价格情景（USD / CNY 待 D-01） | 可填结构；货币待拍板 | `docs/decision_timing_and_units.md` §6；`prices.yaml` |
| `price_basis` | 保持 `per_kg_as_fed` | 不改 | 合同 T2.2 |
| `decision_information_timing_file` | `docs/decision_timing_and_units.md` | 可填 | 本阶段 |
| `dry_matter_mode` | 候选 `uncertain_with_available_information`（主）+ `known_for_controlled_submodel`（对照） | 待拍板 | `uncertainty.yaml` 维度 f；数据可行性审计 §5.3-2 |
| `allow_hidden_dm_recoursing` | 保持 false | 不改 | 合同 T1 |
| `offered_dm_is_observed_intake` | 保持 false | 不改 | 合同 §6.3 |
| `assumptions_about_refusal_and_sorting` | `not_modelled; offered ration assumed fully consumed, identical for all methods; mixing/delivery error not modelled` | 可填 | `docs/decision_timing_and_units.md` §9 |

## 6 `constraints`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `constraint_file` | `configs/constraints.yaml` | 可填（文件存在；多条阈值为 null） | 本阶段 |
| `source_audit_passed` | false | 不改 | CP、EE、添加量、能量未定 |
| `independent_evaluator_tested` | false | 不改 | 未实现测试 |
| `joint_nutritional_event_defined` | false | 待拍板（D-09） | `constraints.yaml` `joint_event` |
| `numerical_tolerances_file` | `configs/constraints.yaml`（`globals.numerical_tolerance_policy` 与逐条 `numerical_tolerance`；数值 1e-6 为开发起点） | 可填路径；数值待冻结 | 合同 T3 |

## 7 `uncertainty`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `model_file` | `configs/uncertainty.yaml` | 可填 | 本阶段 |
| `primary_distribution` | null（候选 TN 截断正态为开发起点） | 待拍板（D-19） | `uncertainty.yaml` 维度 b |
| `covariance_source` | null（候选：C0 独立为基线 + C1 Yoder 2014 表 6；完整矩阵 blocked P0-6） | 待拍板 | `uncertainty.yaml` 维度 c |
| `unknown_correlation_sensitivity_required` | 保持 true | 不改 | 合同 §7.4–7.5 |
| `sampling_and_lab_error_file` | `configs/assays.yaml` | 可填 | 本阶段 |
| `batch_variation_double_count_audited` | false（防护规则已写 DC-01…DC-06，未实现审计） | 不改 | `uncertainty.yaml` |

## 8 `methods`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `core` | 保持 [nominal, calibrated_safety_margin, joint_chance_saa, robust] | 不改 | `methods.yaml` |
| `optional` | [rfeu_adapter] 保留，但 `methods.yaml` 记 M4 = not_applicable（D-007） | 待总负责人确认是否从 optional 删除 | DECISIONS D-007 |
| `safety_margin_development_grid` | 保持 [0, 0.025, 0.05, 0.075, 0.10] | 不改 | 合同 T4 |
| `robust_uncertainty_set_kind` | null（候选 box_mu_k_sigma 为主、budget 为对照） | 待拍板（D-11） | `methods.yaml` M3 |
| 新增字段建议 | `safety_margin_semantics`（候选 coef_directional） | 待拍板 | 合同 T4 M1 要求先定义余量作用对象 |

## 9 `risk_and_outcomes`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `primary_target_alpha` / `secondary_target_alphas` | 保持 0.05 / [0.10, 0.01] | 不改 | 合同 §10、T3 |
| `endpoint` | 保持 `joint_model_constraint_violation`；成员另见 D-09 | 不改 | — |
| 新增字段建议 | `joint_event_members`、`attainment_rule`（测试流判定“达标”的规则，例如区间上界 ≤ α 或点估计 ≤ α） | 待拍板 | `audit/endpoint_rq_map.md` §2 |

## 10 `splits`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `split_strategy` | `simulation_streams (optimisation/validation/test/outer_parameter) + source_scenarios (P_C baseline; P_A/P_B/P_D blocked; P_CN mean-shift)`；外检模块另行登记 | 可填（文字） | 数据可行性审计 §9.2；合同 T5 |
| `test_previously_seen` | 核心层：无测试数据（模拟流）；但审计阶段已看过 AFZ、CVB 候选原料均值/SD 与 Dairy One 2024-25 汇总（D-12） | 可填（文字，须披露） | 数据可行性审计 §9.2“审计阶段已查看数据登记” |
| 其余 | 保持 | 不改 | — |

## 11 `computation`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `optimization_scenario_development_grid` | 保持 [128, 512, 1024] | 不改 | 合同 T6 |
| `final_optimization_scenario_count` | null | 不可填（开发阶段收敛检查后定） | `methods.yaml` M2 |
| `development_seeds` | 保持 [1103, 2207, 3301] | 不改 | 合同 T6 |
| `final_mc_precision_rule` | 候选：从 10,000 次起加倍，直到 Wilson 区间半宽 ≤ 0.2·max(α, r̂)，上限 200,000 | 待拍板 | `methods.yaml` draws |
| `solver_and_version` | 候选 `HiGHS via scipy.optimize (linprog/milp)`；版本在服务器上登记 | 部分可填 | `methods.yaml` solver；D-13 |
| `tolerance_and_gap_file` | `configs/methods.yaml`（`solver.dev_start_settings`） | 可填路径；数值待冻结 | 验收 C11 |

## 12 `information_value`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `information_actions_file` | `configs/assays.yaml`（面板）+ `configs/methods.yaml` `information_value`（策略） | 可填 | 本阶段 |
| `observation_likelihood_supported` | false；只有青贮部分“原料 × 成分”有三分量依据 | 不改 | `assays.yaml` `evsi_feasibility` |
| `allow_classical_evsi_claim` | 保持 false | 不改 | 合同 T7.2–T7.3 |
| `candidate_policy_class` | `finite_candidate_library_with_signal_bins`（合同 T7.3） | 可填 | `methods.yaml` |
| `panel_cost_source` | null（文献锚点 St-Pierre & Cobanov 2007a 默认值只作参照线） | 不可填 | P1-4 |
| `covered_heads` / `covered_days` | null | 待拍板 | `inventories.yaml` `batch_and_coverage` |
| `no_cost_data_report_break_even_only` | 保持 true | 不改 | 合同 §12.4 |

## 13 `statistics`

| 字段 | 建议 | 类别 | 依据 |
|---|---|---|---|
| `analysis_plan_file` | null（阶段 6 另写；`audit/endpoint_rq_map.md` §5 可作骨架） | 不可填 | 合同 §10 |
| `primary_contrast` | 候选：α = 0.05 下 M1、M2、M3 相对 M0 的成本差，仅在双方达标时解释为“同可靠性成本差” | 待拍板 | 合同 §11 E1 |
| `cluster_unit` | `lineage (descriptive)`；外检按各自真实单位 | 可填（文字） | 数据可行性审计 §9.3 |

## 14 `freeze`

全部不可填：须在开发、收敛检查与上述拍板完成后，由冻结流程生成（合同 §10）。当前 `unresolved_critical_blockers` 至少应包含：D-03、D-04、D-09、D-11、D-19、PUD-P2a-01、Q-01（能量约束）、P1-3（价格）。

## 15 `reporting`

保持不变。`journal_candidate: "Agriculture"` 与专家意见一致；SoftwareX 状态冲突留阶段 11（route_decision §6.2）。
