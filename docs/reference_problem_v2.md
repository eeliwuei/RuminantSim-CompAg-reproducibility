# 参考问题 v2（reference problem v2）：域前提计划行与配方层域读法

- 编写：总负责人，2026-09-26（KST），据 Batch 1 实现（`experiments/E1_cost_reliability/official_v2.py`、`src/ration_reliability/nutrition/domain.py`、`src/ration_reliability/evaluation/reference.py`、`configs/dev_case_v1/reference_constraints_v2.csv`）。
- 性质：**开发期研究决定**（D-533、D-535），尚未 `protocol_frozen`；本文件将进入官方运行的 `SPEC_FILES`（Batch 4），冻结后不得改动。
- 受限值规则同 v1：不写 NASEM 表值或可一步还原它们的量；受限界值只写 `value_ref`。
- **如实披露**：v2 的两项改动都是在看过开发结果（`pilot-20260925T055921Z-fb4f75af` 与第三轮消融）之后做出的；理由只依据营养学与来源（Table 5-1 的前提是所配日粮的设计属性），不依据可行性；正式结果在冻结前未看过（预留根未抽样）。

## 1 相对 v1 改了什么

| 项 | v1 | v2 |
|---|---|---|
| Table 5-1 适用前提（干粉碎玉米为主要淀粉源，τ = 0.50，项目假设） | 只在评估侧逐状态判定：域外状态 T 行记 unknown（计入上界率）；任何方法在训练时都不被要求满足前提 | **两处**：① 所有臂、所有方法（含 M0）在优化问题里加入结构计划行 `SH-PLAN-T51-DGC-SHARE`：Σ_i q_i d̂_i (1{i ∈ 干粉碎玉米} − τ)·St̂_i ≥ 0（St̂ 为名义表值淀粉；`nutrition/domain.py::premise_planning_coefficients`；τ = 0.50 时系数等于 `DIAG-T51-DGC-STARCH-SHARE` 的名义内容）；② 评估侧新增**配方层**域读法 `table51_domain = "plan"`：域状态对每份配方只按其计划日粮（q、d̂、名义成分）判定一次（`domain.py::plan_domain_status`），域内则 T 行在每个状态照常判定，域外则 T 行在每个状态记 unknown |
| 主参考事件 | `main_reference`（逐状态域条件） | `main_reference`（逐状态，保留）+ 伴随事件 `main_reference_plan_domain`（配方层）；`main_reference_plus_cp_hi` 亦有 `_plan_domain` 伴随事件；headline 同时给出配方层域状态与逐状态域外份额 |
| 主事件读法 | 逐状态 | **F1 决定（D-535）：主事件取配方层读法**，逐状态读法并列报告；冻结时写入协议，不按结果更改 |
| 约束表 | `reference_constraints.csv`（27 行） | `reference_constraints_v2.csv`（28 行；每行 `table_version = reference_problem_v2`；新增 `optimization_all_arms` 列；加载器要求 v2 表恰有一条前提计划行，v1 表不得含该行；v1 文件字节不变） |
| 编译核对 | — | `ReferenceSpec` 在 v2 表下核对已编译的前提行确实是主变体（τ、计数原料）的结构 d̂ 形式（`_premise_row_errors`） |

其余（参考牛、8 种原料、价格、SD 声明、能量参考链、裁决规则 A1–A7、九行主参考、MAIN9 臂、方法 M0–M3 与网格、比率口径、Clopper–Pearson 与 MC 标准误）与 v1 相同（`docs/reference_problem_v1.md`）。

## 2 为什么这样改（不是为了可行）

- 第三轮复核 F3 与第三轮消融显示：S2 选出的配方在主变体下 96–100% 状态域外，主事件上界接近 1，可比集合按构造为空；原因是**评估事件含一个没有任何方法被要求满足的条件**（`投稿就绪评估` §1.4 第 1 条）。
- 计划行把这一前提写进每个方法的决策问题，使训练与评估的适用域一致（B-123/B-429 列出的选项）。
- 但仅有计划行仍不够（计划 §0 F1）：求解器会把该行压到取等，逐状态随机成分下约一半状态会低于 τ，逐状态读法下主事件仍近乎全不达标。Table 5-1 的三项前提（TMR、粒度、主要淀粉源）都是**日粮设计属性**，不随单次成分抽样变化，所以主事件按配方层判域在营养学上更贴合原文；逐状态读法保留为伴随事件，供读者看到两种读法的差异。
- 不改 τ、不改任何 T 行界值、不删行、不改 α 与 SD。

## 3 不声称与限制

- 配方层域内 ≠ 该配方满足 Table 5-1 各行；域内只表示 T 行的判定有效。
- 前提中的 TMR 与粒度仍是 `assumption_only`。
- 计划行使用名义表值淀粉（受限值）：其计划余量属受限输出（`REDACT_NOMINAL_V2`），不进公开表。
- 主事件读法的选择在正式结果之前固定；若日后按新来源改 τ 或读法，登记 v3 并重跑。

## 4 B-442（两套域条件实现）

`evaluation/reference.py` 为主规则（域外 T 行不作任何判定）；`nutrition/domain.py::premise_conditioned_event` 保留域外 T 违约为违约。同一事件、同一状态与域下：两者上界率相等，后者下界率 ≥ 前者，差值恰为“域外且唯一违约成员是 T 行”的状态数（`tests/unit/test_domain_rule_cross_consistency.py`，含配方层模式）。

## 5 代码与测试（Batch 1，2026-09-26）

- `domain.py`：`premise_planning_coefficients`、`plan_domain_status`、`PREMISE_PLANNING_ROW_ID`；docstring 记 B-442 关系。
- `official_v2.py`：`PREMISE_ROW_ID`、`add_domain_premise_row`、`build_v2_cfg0`、`REDACT_NOMINAL_V2`（尚未接入驱动默认路径；Batch 3 接入）。
- `reference.py`：v2 表识别与校验；`TABLE51_DOMAIN_MODES = ("primary", "ignored", "plan")`；伴随事件 `main_reference_plan_domain`、`main_reference_plus_cp_hi_plan_domain`；headline 配方层域状态。
- 测试：`test_domain_premise_row.py`、`test_reference_constraints_v2.py`、`test_domain_rule_cross_consistency.py`（+ `test_evaluate_reference.py` 更新）；全量 1,073 passed（Mac，持有方）。
- `run_endpoint_ablation.py --dry-run`：预检 READY；冻结状态 `differs_from_frozen`（驱动、reference.py、stats.py 自 pin 46c22cb 起已改；Batch 4 重写 v2 pin）。

## 6 冻结前仍待定（见 `docs/official_run_v2_plan_20260926.md` §6）

τ 敏感性是否作为计划行格；诊断参数；成员统计量与覆盖率口径；主 N 与收敛判据；时限；M1 无 DM 余量变体角色；成本公开形式；矩阵与 worker 数；B-454 比值位置；B-127 分布族/相关/U3；偏离登记；`access_scope`。
