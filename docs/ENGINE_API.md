# 求解核心 API 说明（ENGINE_API，engine-api/0.1）

- 编写：P4a_engine_core 子任务，2026-09-24。
- 修订：W2 子任务，2026-09-24（KST）。按 M1–M3 已实现并登记、以及引擎红队修复（`audit/_parts/FIX_engine_record.md` §1、§4 第 4 条）更新了开头状态、§0、§2、§3.3、§4、§5、§6、§9、§10、§12；其余各节保持 P4a 原文。接口版本号仍为 `engine-api/0.1`（`ration_reliability.ENGINE_API_VERSION` 未改）。
- 对象：使用或扩展求解核心（方法、不确定性模型、信息价值、实验脚本）的 agent。
- 代码：`src/ration_reliability/`（不依赖、不复制旧 RuminantSim 代码；旧实现情况见 `audit/_parts/legacy_code_map_engine.md` §1）。
- 状态（合同 §2.3）：核心模块、M1、M2、M2b、M3a/b/c 和信息价值原型都是 `implemented`、`unit_passed`（2026-09-24 全量 387 passed，日志 `reports/engine_test_log_FIX_engine.md`；P4a 首版时为 118 项，见 `reports/engine_test_log_P4a.md`）。M0（`nominal_point`）与 M2 另为 `smoke_passed`（当前有效的 smoke 运行见 `reports/smoke_report.md` §9）。合成数据端到端测试属于软件自检，不是研究运行；没有 pilot，也没有 official。
- 测试通过只说明程序正确计算、正确求解了所定义的问题，不说明任何营养或动物效果（合同 §8）。

---

## 0 快速开始

项目不安装成包。把 `src` 加到 `PYTHONPATH`；测试由 `tests/conftest.py` 自动加。仓库根目录的 `pytest.ini`（`testpaths = tests`，并排除 `logs legacy results release_staging data manuscript experiments sources .git`）使根目录不带路径运行也只收集 `tests/`（W2 实测 387 项，不再收集 `logs/legacy_exec/` 下第三方包的测试）。

```bash
cd <repository root>
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest -q -p no:cacheprovider
```

```python
from ration_reliability.io import load_problem
from ration_reliability.optimization import get_method
from ration_reliability.uncertainty import RandomStreams, IndependentNormalModel
from ration_reliability.evaluation import evaluate_drawset

problem, report = load_problem("data/synthetic_test_only/engine_toy_problem_v1.yaml", mode="unit_test")
res = get_method("M0_nominal")(problem)                  # SolveResult
model = IndependentNormalModel(...)                       # 或其他 UncertaintyModel（参数必须有来源）
draws = model.draw(RandomStreams(1103), "test", 10_000)   # DrawSet，带流标签
ev = evaluate_drawset(res.decision, draws, problem.compiled, prices=problem.prices)
print(ev.summary())
```

依赖：Python 3.11、numpy 2.4、scipy 1.17（自带 HiGHS 1.12.0）、PyYAML 6.0。不需要其他包。

---

## 1 单位、基准与决策量

| 量 | 符号 | 内部标准单位 | 说明 |
|---|---|---|---|
| 执行决策 | `q_i` | kg as-fed/头/日 | 唯一交给评估器的量 |
| 计划干物质配方 | `x_i = q_i·d̂_i` | kg DM/头/日 | 方法内部可用；执行量 `q = x / d̂`（T2.1） |
| DM 含量 | `d_i` | kg DM/kg as-fed（`fraction`） | `d̂` = 决策时可用估计；`d`（抽样）= 本批真实值 |
| 成分 | `a_ij` | kg/kg DM（质量类）；Mcal/kg DM（能量类） | NaN = 缺失，绝不当 0 |
| 价格 | `p_i` | 货币/kg as-fed | 载入时一次换算，不随测试抽样变化（T2.2） |
| 供给 | `D`、`N_j` | kg/头/日，或 Mcal/头/日 | 供给不等于实际采食（C04） |

单位登记表在 `normalization/units.py`。未登记的单位一律抛 `UnitError`。价格单位写成 `<CUR>/<kg|g|t|lb>`，`CUR` 为 3 个大写字母；合成数据用 `XXX`（ISO 4217 的“无货币”代码）。不支持币种换算。

精确常数：1 lb = 0.45359237 kg，1 Mcal = 4.184 MJ（NIST SP 811 附录 B.8，https://www.nist.gov/pml/special-publication-811 ；项目核对要求见 `audit/phase1_survey_20260924/DATA_FEASIBILITY_AUDIT.md` §5.3）。

**基准（basis）与单位分开记。** `%` 在 DM 基准和鲜重基准下含义不同。基准换算必须显式给出 DM：

- 浓度：`dm_to_as_fed_concentration`、`as_fed_to_dm_concentration`；
- 质量：`as_fed_mass_to_dm`、`dm_mass_to_as_fed`；
- 价格：`price_per_tonne_to_per_kg`、`price_dm_to_as_fed`、`price_as_fed_to_dm`；
- 能量：`mcal_to_mj`、`mj_to_mcal`、`mcal_per_lb_to_mcal_per_kg`；
- 其他：`percent_to_g_per_kg` 等。

YAML 载入器只接受 DM 基准的成分。鲜重基准的数据须在上游用同一样本的 DM 换算，并留日志。

---

## 2 数据结构（`ration_reliability.datamodel`）

| 类 | 作用 | 要点 |
|---|---|---|
| `Provenance` | 来源 | `status` ∈ `sourced` / `research_scenario_assumption` / `pending_user_decision` / `synthetic_test_only`。`sourced` 要有 `source_id` 和 `locator`；假设要有 `rationale`；待定值必须是 `None` |
| `SourcedValue` | 带单位、基准、来源的标量 | `.issues()` 检查规则；`.canonical()` 换成标准单位 |
| `NutrientSpec` | 成分列 `j` | `dimension` ∈ `mass_fraction` / `energy_density` |
| `IngredientRecord` | 原料 `i` | `dm_estimate`（d̂）、`composition`（DM 基准、标准单位）、`group_weights`（如 `{"forage": 1.0}`，取值 [0,1]）、`coefficients`（具名实数系数，如吸收系数）、`is_stochastic`、`is_synthetic`、`provenance` |
| `ObservationRecord` | 来源中的单条观测或统计量 | 字段按 `evidence_contract.yaml`；可选字段不得编造 |
| `AnimalProfile` | 画像 | `attributes: {name: SourcedValue}`；`dmi_kg_per_day` 属性 |
| `ConstraintSpec` | 约束（自然单位） | 见 §3 |
| `PriceScenario` | 价格（货币/kg as-fed） | `is_scenario`、`conversion_log` |
| `RationProblem` | 完整问题 | 原料、成分、约束、价格与画像；`.compiled` 为缓存的编译结果；`.reordered()` 用于不变性测试 |
| `RationDecision` | 可执行配方 | `q_as_fed`、`d_hat`、`x_planned_dm = q·d̂`（只读） |
| `SolverOptions` | 求解设置 | `time_limit_s`、`presolve`、可行性容差（默认 1e-7，同 HiGHS 默认）、`mip_rel_gap`、`residual_check_rel_tol`（1e-6）。`mip_rel_gap` 默认 `None`（用 HiGHS 默认）；pilot/official 的运行记录必须显式记录它（§10） |
| `SolveResult` | 方法输出 | 见 §5 |
| `EvaluationResult` | 公共评估器输出 | 见 §4 |
| `AssaySpec`、`InformationPolicy` | 检测与信息政策容器 | 数据结构在此；信息价值算法在 `ration_reliability.information`（`implemented`、`unit_passed`，只用合成数据），设计见 `docs/INFORMATION_VALUE_DESIGN.md`。未知的误差或费用保持 `None` |

---

## 3 约束：表达式语法、类别与线性化

### 3.1 表达式项（`ConstraintSpec.terms`：项键 → 系数）

| 项键 | 对 `S(q,θ)` 的贡献 |
|---|---|
| `<nutrient>` | `Σ_i q_i d_i a_ij` |
| `G:<group>:<nutrient>` | `Σ_i q_i d_i w_ig a_ij`，例如 `G:forage:NDF` = 粗饲料 NDF（fNDF） |
| `C:<coef>:<nutrient>` | `Σ_i q_i d_i c_i a_ij`，`c_i` = `IngredientRecord.coefficients[coef]`，每种原料都必须有 |
| `DM`、`DM:<ingredient>`、`G:<group>:DM`、`C:<coef>:DM` | 干物质质量项 |
| `AF`、`AF:<ingredient>` | 鲜重项（只用于 `kind=as_fed`） |

编译后得到系数矩阵 `W[K,I,J]`、`w0[K,I]`、`v[K,I]`：

```
c_ki(θ) = Σ_j W[k,i,j]·a_ij + w0[k,i]
S_k(q,θ) = Σ_i q_i·d_i·c_ki(θ) + Σ_i v[k,i]·q_i
```

### 3.2 种类（`kind`）

| kind | 比较量 | 线性化（合同 T2） | 单位 |
|---|---|---|---|
| `concentration` | `E = S/D`，D 用该情景自己的 `Σ q_i d_i` | ge：`K·D − S ≤ 0`；le：`S − K·D ≤ 0` | `%`、`g/kg`、`fraction`、`Mcal/kg`、`MJ/kg`；basis 须为 `DM` |
| `supply` | `E = S`（每头每日） | ge：`K − S ≤ 0`；le：`S − K ≤ 0` | `kg/d`、`g/d`、`Mcal/d`、`MJ/d` |
| `as_fed` | `E = Σ v_i q_i` | 同上 | `kg/d`；basis 须为 `as_fed` |

系数作用于标准单位（kg/d、Mcal/d），所以 supply 表达式中 `DM` 项的系数本身要写成 kg（或 Mcal）每 kg DM。

### 3.3 类别规则（编译时强制）

- `structural_hard`：不得含成分项（W = 0）。DM 项必须 `dm_source=decision_estimate`，即用 d̂ 描述“计划配方”。允许 `eq`。评估器从 `q` 和 d̂ 确定性地检查这类约束，不看抽样，也不受风险预算放松。
- `probabilistic_nutrition`：必须 `dm_source=scenario`；不得用 `eq`；不得是 `as_fed`。只有这类约束进入联合违约事件 `I`。
- `diagnostic_only`：同上，只报告，不施加于优化，也不进入 `I`。
- `bound` 为 `None` 的约束（待定阈值）不能编译。所有错误一次列全，抛 `InvalidProblemError`。
- 每条约束的 `numerical_tolerance` 必须是有限正数（声明单位下），为 0 或负数时编译报错（红队 C10）：容差为 0 时，最优点上恰好取等的约束会因浮点舍入被误判为违约（复现：M0 在自身设计点上余量 −2.78e-15 被判违约）。判定规则不变，仍是 `−margin > numerical_tolerance`，没有另加相对 epsilon。构造函数 `concentration_constraint` 的 `tolerance` 为必填参数（无默认值）；`dm_offer_constraint`（1e-6）、`as_fed_upper_bound` 与 `inventory_constraint`（1e-9）保留正的默认值。

### 3.4 常用形式示例（阈值必须带来源，或标为假设；下面不写 NASEM 数值）

```python
{"CP": 1}                                   # CP %DM，ge 或 le
{"G:forage:NDF": 1}                         # fNDF ≥ K1
{"NDF": 1, "G:forage:NDF": 2}               # NDF + 2·fNDF ≥ K2（Table 5-1 的形式）
{"starch": 1, "G:forage:NDF": -2}           # starch ≤ 2·fNDF − K3  → sense=le, bound=−K3
{"C:AC_Ca:Ca": 1, "DM": -r1}                # 吸收 Ca ≥ R0 + r1·D（supply，r1 单位 kg/kg DM）
{"DM:limestone": 1}, dm_source=decision_estimate   # 计划 DM 添加比例上限（structural）
{"AF:corn_silage": 1}                       # H·T·q ≤ B → q ≤ B/(H·T)（inventory_constraint）
```

构造函数：`concentration_constraint`、`dm_offer_constraint`（计划 DM 供给 = DMI，structural，用 d̂）、`as_fed_upper_bound`、`inventory_constraint`。

### 3.5 线性行（给优化方法用）

`linear_rows(cc, theta, d, d_hat)` 返回 `LinearRows`：

- `A[S,K,I]`、`b[K]`、`is_eq[K]`、`missing[S,K]`；
- `g_k(q; s) = A[s,k,:] @ q − b[k]`，ge/le 行为 `≤ 0`，eq 行为 `= 0`；
- `.mean()` 给出样本期望。g 对 q 线性，所以期望是精确的，并且按联合乘积 `E[d·a]` 计算，不用 `E[d]E[a]`（T4）；
- `.residual(q)` 返回 `[S,K]` 残差。

M2 SAA 可以直接用逐情景的 `A[s]`，Big-M 用 `optimization.lp_builder.big_m_from_box(A, b, q_lb, q_ub)` 从变量界推导（T4：不许无说明地设 1e6）。

---

## 4 公共评估器（唯一，所有方法共用）

```python
from ration_reliability.evaluation import evaluate, evaluate_drawset

evaluate(q, theta_draws, d_draws, constraints, *,
         d_hat=None, prices=None, draw_stream_id=None, is_synthetic=False,
         chunk_size=20000, q_nonneg_tol=1e-9) -> EvaluationResult
evaluate_drawset(q, draws: DrawSet, constraints, **kw) -> EvaluationResult
```

- `q`：`[I]` 数组，或 `RationDecision`（此时用它记录的 d̂ 检查 structural 约束）。原料顺序必须等于 `constraints.ingredient_ids`，否则报错，不会自动重排。
- `theta_draws [S,I,J]`、`d_draws [S,I]`：本批真实状态；也可以只给单个状态 `[I,J]`、`[I]`。
- `constraints`：`problem.compiled`，包含全部类别。

评估器的计算：

1. 实际 DM：`x_real = q·d_true`。**不重新归一化，也不用隐藏真实 DM 反推 q**（T2.1，C05/C06；`tests/leakage/` 已验证）。
2. `D = Σ x_real`；`N_j = Σ x_real·a_ij`。
3. 对每条 probabilistic/diagnostic 约束，在其声明单位下计算：
   - `margin[s,k]`：带符号余量，> 0 表示满足；eq 为 `−|E−K|`；
   - `violation_amount = max(0, −margin)`；
   - `violated = (−margin > numerical_tolerance)`。
4. 使用中（`q_i ≠ 0`）的原料成分或 DM 为 NaN 时，该格 `missing_data` 且 `undefined`（D ≤ 0 的浓度约束也记 `undefined`）。这种格不算违约，只计入 `data_quality_flag`（T3）。
5. 联合事件：
   - `joint_violation[s]`：至少一条 probabilistic 约束违约；
   - `joint_unknown[s]`：没有违约，但至少一条 undefined。
6. `summary()` 给出：
   - `rate_lower = 违约数/S`；
   - `rate_upper = (违约数 + 未知数)/S`；
   - `rate_among_evaluable`；
   - Clopper–Pearson 区间，标注为 `MC_only_fixed_distribution`，只表示固定分布下的蒙特卡洛误差，不是数据不确定性区间（T8.3）。
7. 结构约束：从 `q`、d̂ 确定性计算，另加内置规则 `q ≥ 0`；结果在 `structural_ok`。
8. `cost = p @ q`（只要给了价格）。

输出数组全部只读。结果与 `chunk_size` 无关（已测试）。同一 `q` 不论来自哪个方法，输出完全相同（已测试）。

**批量结构检查（红队修复后新增，D05、C09）：**

```python
from ration_reliability.evaluation import structural_check
margin, violated, nonneg_ok = structural_check(Q, d_hat, constraints, q_nonneg_tol=1e-9)
```

- `Q`：`[n, I]`（或 `[I]`）个执行配方；`d_hat`：`[I]`（所有配方共用一个决策时估计）或 `[n, I]`（每个配方各用一个，例如信息价值中不同信号箱的决策时 DM）。结构约束含 DM 项而不给 `d_hat` 时报错。
- 返回 `margin [n, Ks]`、`violated [n, Ks]`（声明单位下，按 `numerical_tolerance` 判定）和 `nonneg_ok [n]`。
- 这就是 `evaluate` 的结构检查代码路径，所以两者的结论逐元素相同（`tests/unit/test_redteam_engine_fixes.py::test_structural_check_equals_public_evaluator`）。信息价值模块用它按决策时可得的 DM 估计确定合法行动集（`docs/INFORMATION_VALUE_DESIGN.md` §2.2、§8）。

---

## 5 SolveResult 约定

| status | 含义 | decision / objective |
|---|---|---|
| `optimal` | HiGHS 报 Optimal，且引擎自检残差 ≤ `residual_check_rel_tol` | 有 |
| `feasible_time_limit` | 到时间或迭代上限，有通过残差检查的可行解 | 有（不保证最优） |
| `proven_infeasible` | HiGHS 报 Infeasible（原问题，未放松） | **无** |
| `no_feasible_solution_found` | 到上限，没有可行解（不等于无解） | **无** |
| `numerical_error` | 最优或可行性自检失败、状态含糊（放松 presolve 重解后仍为 “unbounded or infeasible”）或其他求解器错误 | **无**（原始候选放在 `diagnostics["raw_candidate_x"]`） |
| `invalid_input` | 输入不合法：参数未知、d̂ 越界、名义系数缺失、问题无界或 ModelError | **无** |

- `SolveResult` 的构造函数强制上表：失败状态附带配方或成本会直接报错。禁止用空配方或零成本表示无解。
- 不会自动放松约束。`diagnostics["imposed_constraint_ids"]` 列出实际施加的约束。
- 最优解是全零配方时（例如漏写了计划 DM 供给规则），状态仍按数学结果记为 `optimal`，同时在 `diagnostics["warning"]` 给出提示，不改动结果。
- 各枚举（`SolveStatus`、`ConstraintClass` 等）的 `str()` 就是其取值，例如 `"optimal"`，可直接写日志和 JSON。
- 必有字段：`method_id`、`status`、`decision`、`objective`（货币/头/日，由 `p @ q` 重算）、`objective_unit`、`solver`、`solver_version`（如 `HiGHS 1.12.0 (git 4f96ee8) via scipy 1.17.1`）、`tolerances`、`wall_time_s`、`input_hash`、`mip_gap`（LP 为 `None`）、`iterations`、`n_evaluations`（精确求解为 `None`，启发式必须填）、`message`、`constraint_residuals`（按约束 id 的线性化残差）、`params`、`streams_used`、`diagnostics`、`is_synthetic`。
- HiGHS 状态映射表见 `optimization/highs.py` 模块文档。`run_milp` 另记录 `mip_gap` 和节点数。
- M2、M2b 的 `tolerances` 另记 `mip_rel_gap_source`：`explicit`（调用方显式设置）或 `solver_default_not_set_explicitly`（红队 C11）。
- M0、M1、M3a、M3c 的 `diagnostics["equivalent_methods"]` 列出其所属等价对编号（§6.3）。M2、M2b、M3b 的诊断中没有这个键。

---

## 6 方法：已登记的实现、参数与新增规则

### 6.1 已登记方法（`optimization/__init__.py` 的 `REGISTRY`）

M1、M2、M2b、M3a/b/c 都已实现并登记（M3 三行由 SMOKE 子任务加入 `REGISTRY`，见 `audit/_parts/SMOKE_log_additions.md` §2）。所有方法都按 `get_method(id)(problem, *, d_hat=None, opt_draws=None, params=None, solver_options=None) -> SolveResult` 调用。

| method id | 函数 | 求解器 | 状态（合同 §2.3） | 测试 |
|---|---|---|---|---|
| `M0_nominal` | `m0_nominal:solve` | LP | `unit_passed`；`nominal_point` 另为 `smoke_passed` | `tests/numerical/test_m0_small_lp.py` 等 |
| `M1_safety_margin` | `safety_margin:solve` | LP | `unit_passed`（含合成端到端） | `tests/unit/test_safety_margin.py`；`tests/unit/test_redteam_engine_fixes.py` |
| `M2_joint_chance_saa` | `chance_saa:solve` | MILP（共同 `z_s`） | `unit_passed`；另为 `smoke_passed`（N = 64，α_train 未校准） | `tests/numerical/test_chance_saa.py` |
| `M2b_marginal_bonferroni_saa` | `chance_saa:solve_marginal_bonferroni` | MILP（逐行 `z_{s,k}`） | `unit_passed` | 同上 |
| `M3a_box_robust` | `robust:solve_box_robust` | LP | `unit_passed` | `tests/unit/test_robust.py` |
| `M3b_budget_robust` | `robust:solve_budget_robust` | LP（Bertsimas & Sim 2004 对偶形式） | `unit_passed` | 同上 |
| `M3c_scenario_set_robust` | `robust:solve_scenario_set_robust` | LP | `unit_passed` | 同上 |

合成端到端测试 `tests/integration/test_end_to_end_synthetic.py` 让 7 个方法在同一问题、同一 `opt` 抽样上求解，并在同一 `test` 抽样上由公共评估器评分。`optimization/robust.py` 模块文档末段仍写着“本模块不登记到 REGISTRY”，这句已过时（该文件不在 W2 的修改范围内，已记入 `audit/_parts/W2_record.md`）。

### 6.2 参数要点（完整说明见各模块文档）

**M0 `M0_nominal`**

- `coefficient_mode="nominal_point"`（默认）：用名义成分和 d̂ 计算系数，即“按表值配方”的点估计；
- `coefficient_mode="draw_mean"`：用 `opt` 抽样的行均值，即联合乘积均值；
- 在零不确定性且以名义值为中心时，两者等价，只能算一种方法（等价对 `EQ-M0-MODES`，§6.3）。主结果用哪一种由 protocol 决定（待定，见 `audit/_parts/P4a_log_additions.md`）。

**M1 `M1_safety_margin`（含量系数安全余量）**

- `k`：必填，≥ 0。
- `margin_scale`：**必填，无默认值**（红队 D02）。`"sd"`：余量 = k·σ（σ 取 `opt` 抽样的样本 SD，`sd_source="opt_draws"`；或显式给出 `theta_sd [I][J]`、`d_sd [I]`，`sd_source="explicit"`）；`"relative"`：余量 = k·|μ|（DM 为 k·d̂）。缺省或取其他值时返回 `invalid_input`。
- `apply_to_dm`：**必填的布尔值，无默认值**（`configs/methods.yaml` 中 M1 的 DM 余量尚未决定）。`True` 时 DM 也按行方向取保守值。
- 余量只作用于 `probabilistic_nutrition` 行的含量系数，按上/下限方向逐项收紧；不改要求量 K，不改 SD，不动结构约束。M1 在 k 处的 LP 就是盒子 `{|a − μ| ≤ k·spread, |d − d̂| ≤ k·spread_d}` 上的逐行最坏情形 LP，所以与同一盒子的 M3a 是同一个数学问题（`EQ-M1-M3a`）。
- 选参：`select_margin_on_validation(problem, validation_draws, grid=…, target_alpha=…, screening_rule=…, params={"margin_scale": …, "apply_to_dm": …}, opt_draws=…)`，或通用的 `select_parameter_on_validation`（§6.2 末）。选 M1 的 `k` 时自动调用 `check_margin_grid(grid, margin_scale)`：尺度未声明拒绝；`sd` 尺度下网格最大值 ≤ 0.10（即合同相对网格 `{0, 0.025, 0.05, 0.075, 0.10}` 的上端，`DEVELOPMENT_GRID_RELATIVE`）拒绝，因为那只相当于 ≤ 0.1σ 的余量；网格值必须有限且 ≥ 0。合同网格应配 `margin_scale="relative"`；若选 sd 尺度，须在冻结前另定网格（`FIX_engine_record.md` §4 第 5 条）。

**M2 `M2_joint_chance_saa` 与 M2b `M2b_marginal_bonferroni_saa`**

- `alpha_train`：必填，[0, 1)。允许的训练违约数 `m = ⌊α_train·N⌋`，按十进制精确计算（`allowed_violations`）。
- `n_scenarios`：`None` = 全部 `opt` 抽样，否则取前 N 个（嵌套阶梯）。
- `big_m_mode`：`"box_quantile"`（默认）/ `"box"` / `"lp_tight"`，推导与有效性证明见 `chance_saa.py` 模块文档；`polish=True`（默认）固定 `z` 后重解 LP，消除整数容差泄漏；`mip_node_limit` 可选。
- M2b 另有 `alpha_allocation`（默认 `"equal"`，α_k = α_train / K_p）；Σα_k > α_train 返回 `invalid_input`。当 N < K_p/α_train 时每行预算 ⌊α_k·N⌋ = 0，M2b 与 M3c 完全相同（`SMOKE_log_additions.md` §6 第 3 条；集成测试已断言）。
- 选参：`select_alpha_train_on_validation(problem, validation_draws, opt_draws=…, grid=…, target_alpha=…, screening_rule=…, marginal_bonferroni=False)`；`alpha_train` 不能同时放进 `params`。
- `SolveResult.tolerances["mip_rel_gap_source"]` 记录 gap 是显式设置还是求解器默认（§5）。所有状态只针对这 N 个 `opt` 情景上的 SAA 实例，不是对真实机会约束问题的保证。

**M3a/M3b/M3c（按实际不确定集命名）**

- M3a `params`：`uncertainty_set`（`BoxUncertaintySet`，来源写入记录），或 `k`（以 `opt` 抽样均值 ± k·SD 建盒），或 `quantiles=(q_lo, q_hi)`（`opt` 抽样逐坐标分位数）。保证：盒内每个状态下全部施加的概率约束同时成立；不给任何概率陈述。
- M3b `params`：`gamma` 必填（数值 ≥ 0、`"full"`，或 `{约束 id: 值}`），盒子同 M3a；`nominal_mode="nominal_point"`（默认，Γ = 0 即 M0）或 `"box_center"`。只有逐行保证，不声称 B&S 2004 的概率界。
- M3c：`opt_draws` 就是有限情景集（只接受 `opt` 流）；只在这 N 个状态上成立，即 α_train = 0 的 M2（`EQ-M3c-M2-ALPHA0`）。

**validation 选参（所有方法通用）**

`safety_margin.select_parameter_on_validation(solve_fn, problem, validation_draws, *, param_name, grid, target_alpha, screening_rule, confidence=None, base_params=None, opt_draws=None, d_hat=None, solver_options=None, consumer=…)`：网格中每个值都求解，用公共评估器在 `validation` 抽样上评分，取满足筛选规则的最低成本值；筛选规则 `rate_upper_le_alpha` 或 `cp_upper_le_alpha`（后者须给 `confidence`），`target_alpha` 与 `screening_rule` 没有默认值；无合格值时 `status="not_met"`，不回退、不放松；只接受 `validation` 流，`test` 流抛 `LeakageError`。

### 6.3 等价方法表 `METHOD_EQUIVALENCES`（验收 D06）

`ration_reliability.optimization.METHOD_EQUIVALENCES` 登记数学上等价的方法对；`equivalence_annotations(method_id)` 返回涉及该 method id 的记录（包括 `M2_joint_chance_saa[alpha_train=0]` 这类带条件的写法）。

| pair_id | 方法 | 等价条件 | 报告规则 |
|---|---|---|---|
| `EQ-M1-M3a` | `M1_safety_margin`、`M3a_box_robust` | M1 取 `margin_scale="sd"`、`apply_to_dm=True`，M3a 用以名义成分表与 d̂ 为中心、同一 σ 的盒子 μ ± kσ（`BoxUncertaintySet.from_mean_sd`）；两者是同一盒子上的逐行最坏情形 LP。M3a 用参数 `k` 时以 `opt` 样本均值为中心，中心不同 | 只报一次，或两行都标“等价，EQ-M1-M3a”。剩下的差别只是 M1 在 validation 流上选 k |
| `EQ-M0-MODES` | `M0_nominal` 的 `draw_mean` 与 `nominal_point` | 零不确定性且以名义状态为中心（或 d_i·a_ij 的抽样均值等于名义乘积） | 算一种方法（M0） |
| `EQ-M3c-M2-ALPHA0` | `M3c_scenario_set_robust`、`M2_joint_chance_saa[alpha_train=0]` | 同一组 `opt` 情景：⌊αN⌋ = 0 迫使所有 `z_s = 0` | 作为 M2 曲线 α → 0 的端点报告，不算额外方法 |

- 任何结果表都要把等价对合并，或在两行旁标注等价，不能算两项方法贡献。
- `equivalence_annotations` 是**方法层面**的标注：它不检查各对的条件是否在某次运行中成立。例如 M2 在 α_train = 0.05 时也会得到 `EQ-M3c-M2-ALPHA0`，M1 在 `margin_scale="relative"` 时也会得到 `EQ-M1-M3a`。合并两行之前要读该对的 `condition`。
- 结果长表：`experiments/E0_verification/smoke_pipeline.long_table` 有 `equivalent_to` 列，内容是 `equivalence_annotations(method_id)` 的 `pair_id`（多个时以 `;` 连接，没有时为空单元格）；它不改变任何数值（W2 起）。

### 6.4 新增方法的规则

1. 在 `src/ration_reliability/optimization/` 新建模块，暴露：

```python
def solve(problem: RationProblem, *, d_hat=None, opt_draws: DrawSet | None = None,
          params: Mapping | None = None, solver_options: SolverOptions | None = None) -> SolveResult
```

2. 在 `optimization/__init__.py` 的 `REGISTRY` 里加**一行**，例如 `"M4_xxx": "ration_reliability.optimization.m4_xxx:solve"`。若与已有方法在某条件下数学等价，同时在 `METHOD_EQUIVALENCES` 登记一条记录，并在 `diagnostics["equivalent_methods"]` 中带上编号。不要改其他引擎文件；需要改接口时写进自己的 `_log_additions.md`，交总负责人处理。
3. 必须遵守：
   - 输入和其他方法相同：同一 `RationProblem`、同一约束和价格；
   - 只施加 `structural_hard` 和 `probabilistic_nutrition` 约束（用 `lp_builder.optimization_indices`）；
   - 在 x 空间求解时用 `assemble_x_space_lp`，并按 `q = x / d̂` 转换，不能用抽样 DM；
   - 拟合只能用 `opt` 流，调用 `require_stream(opt_draws, {"opt"}, METHOD_ID)`；`validation` 只能在显式的选择或校准函数里使用，并在结果中记录；**不许接触 `test`**；
   - 无解就返回无配方的 `SolveResult`，不修复、不放松；
   - 评分只能交给公共评估器，方法内部不另写评分；
   - 研究设定参数（余量尺度、风险水平、不确定集大小等）不给默认值，由调用方显式声明；
   - 余量或不确定集的作用对象和方向在模块文档中写明（T4），候选网格和选择准则写进结果的 `params`；
   - 联合事件用共同情景的 `z_s` 表示；换成 Bonferroni 等边际形式时要改名；
   - 鲁棒方法按实际实现的不确定集命名。
4. 测试放在 `tests/`，文件名要唯一，因为测试目录没有 `__init__.py`。至少包括：小例子的枚举或解析核对、不可行状态、与 M0 相同 `q` 时评估器输出一致。

---

## 7 随机流约定（T5/T6）

```python
from ration_reliability.uncertainty import RandomStreams
s = RandomStreams(root_seed=1103)          # 开发种子 1103 / 2207 / 3301
g = s.generator("opt")                     # SeedSequence(entropy=1103, spawn_key=(0,))
g = s.generator("outer", 17)               # 外层第 17 次参数抽样：spawn_key=(3, 17)
s.stream_id("test")                        # "root=1103/test"
```

| 名称 | spawn 索引 | 用途 |
|---|---|---|
| `opt` | 0 | 优化抽样（SAA 情景、draw_mean） |
| `validation` | 1 | 开发阶段选择与校准（如安全余量） |
| `test` | 2 | 冻结后的独立模拟流 |
| `outer` | 3 | 参数估计不确定性的外层抽样 |

其他名称（如检测信号）的索引由名称的 sha256 派生，≥ 2³²，不会挤占这 4 个。流与创建顺序无关，已测试。流 id 写入 `DrawSet.stream_id`、`SolveResult.streams_used` 和 run record。

---

## 8 UncertaintyModel 接口

```python
class UncertaintyModel(abc.ABC):
    model_id: str; ingredient_ids: tuple; nutrient_ids: tuple; is_synthetic: bool
    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]
        # 返回 theta [S,I,J]（标准单位、DM 基准，NaN=缺失）与 d [S,I]
    def params_for_fingerprint(self) -> dict
    def draw(self, streams, stream, n_draws, *sub) -> DrawSet   # 已实现
```

- 随机数只能来自传入的 `rng`，不许用全局状态。
- 已有 3 个参考机制，参数一概不提供：
  - `PointMassModel`：零不确定性；
  - `IndependentNormalModel`：独立正态，可截断。截断会移动均值；独立只是基线，不是事实（合同 7.4）；
  - `ScenarioSetModel`：有限联合状态集重采样，重复使用同一批记录不增加真实 n。
- 相关结构、批次/采样/分析误差分解等模型由后续 agent 以子类形式新增在 `uncertainty/` 下的新模块中，并在模块文档写明每个参数的来源和假设性质。
- 〔第二轮 K3，复核 R3〕研究入口（smoke/开发/正式运行、鲁棒集合、信息价值先验）一律经 §13 的不确定性模型工厂构建模型；上面三个参考机制保留给引擎单元测试作合成机制，不再在研究入口里直接实例化（`tests/unit/test_uncertainty_factory.py::test_no_new_private_naive_model_entry_points` 静态守卫）。
- `DrawSet` 只读，带 `stream`、`stream_id`、`model_fingerprint`、`is_synthetic`。`.reordered()` 用于原料、成分或情景重排。

---

## 9 YAML 问题配置与校验（T9）

Schema `ration_reliability.problem/0.1`。完整样例见 `data/synthetic_test_only/engine_toy_problem_v1.yaml`（全为合成值）。值块格式：

```yaml
{value: 16.0, unit: "%", basis: DM, status: sourced, source_id: SRC-X, locator: "Table N, p.M"}
{value: null, unit: "%", basis: DM, status: pending_user_decision, rationale: "CP 下限来源待定"}
```

`load_problem(path, mode)` / `build_problem(cfg, mode)` 在有任何错误时抛 `ConfigValidationError`，并列出全部问题；`validate_problem_config` 只返回报告。`mode` ∈ `unit_test` / `smoke` / `pilot` / `official`。

拒绝（error）以下情况：

- 结构：未知键、YAML 重复键、缺必需键；
- 单位与基准：未定义单位或基准；单位量纲与字段不符；成分不是 DM 基准；按 kg DM 计价却没有 `dm_basis_semantics`，或其值不在枚举 `DM_PRICE_SEMANTICS` 中（自由文本一律拒绝，红队 C08）。枚举含两个值：`converted_with_decision_time_dm_estimate`（按 kg DM 报价，载入时用决策时 d̂ 一次换成固定的鲜重价格，不随抽样变化）被接受；`settled_on_measured_dm`（供应商按交货批次实测 DM 结算）在默认的鲜重计价模型中报错，须另建合同情景；
- 来源：`sources` 为空；`sourced` 缺来源或定位、或指向合成来源；假设缺理由；待定值不是 `null`；
- 合成数据：`synthetic_test_only` 值出现在非合成配置里；`is_synthetic` 与 `dataset_status` 不一致；`synthetic_test_only` 目录下的文件未声明为合成；`empirical` 数据含合成源或合成值（即 smoke 输入冒充 empirical）；
- 数据使用：`run_context.fit_sets` 含 `test`；
- 按模式：pilot 和 official 模式下出现合成输入、待定值或许可不明的来源；official 缺 `protocol_sha256`（无冻结哈希）或 `primary_assumption_id`（主假设未选）；
- 约束编译失败；
- 任一约束的 `numerical_tolerance` ≤ 0（红队 C10；原来只对 probabilistic 约束给警告，现为所有约束的错误，见 §3.3）。

警告：存在 `research_scenario_assumption`。

---

## 10 Run record

```python
from ration_reliability.io import build_run_record, write_run_record, utc_now
rec = build_run_record(run_type="smoke", command=..., repo_root=..., started_at=..., completed_at=...,
                       exit_status=0, rng_streams={"test": draws.stream_id}, solver_version=...,
                       tolerances=..., config_paths=[...], data_paths=[...], protocol_path=None,
                       output_paths=[...], is_synthetic=True)
write_run_record(rec, "results/smoke/<run_id>/run_record.json")   # 已存在则拒绝覆盖
```

字段覆盖 `evidence_contract.yaml` 的全部 `run_record_required`（已测试），另有硬件、Python 和包版本、`is_synthetic`。

- 代码身份：`src_tree_sha256`（`src/ration_reliability` 下全部 `*.py`，不依赖 git，始终记录）；有 git 时另记 HEAD、dirty 标志，以及 `dirty_diff_hash`（`git diff HEAD -- src tests configs` 加这三处未跟踪文件的内容哈希）。只使用只读的 git 命令。
- 代码身份**不覆盖** `experiments/` 下的脚本（`_GIT_PATHS = ("src", "tests", "configs")`）。smoke 结果目录另写 `code_identity_addendum.json` 补记脚本 sha256 与修改时间（`reports/smoke_report.md`）；是否把 `experiments` 纳入 `_GIT_PATHS` 待总负责人决定（`audit/_parts/SMOKE_log_additions.md` §6 第 4 条）。
- **pilot/official 必须显式记录 `mip_rel_gap`**（红队 C11）：`build_run_record` 调用 `check_solver_settings_for_run_type(run_type, tolerances)`，`tolerances["mip_rel_gap"]` 须为有限数值 ≥ 0，或在没有任何 MILP 的运行中写字符串 `"not_applicable"`；缺失、`None` 或非法值直接拒绝（`ValueError`）。unit_test 与 smoke 不检查。
- 未提供的输入对应的哈希写 `None`，不补造。

---

## 11 算例（合成，`tests/numerical/test_m0_small_lp.py`）

两种原料：

| 原料 | DM | CP（%DM） | 价格（/kg as-fed） | 折合（/kg DM） |
|---|---|---|---|---|
| 粗饲料 F | 0.40 | 10 | 0.04 | 0.10 |
| 精料 C | 0.80 | 40 | 0.32 | 0.40 |

约束：计划 DM = 20 kg/d（structural），CP ≥ 16 %DM。

解析解：

- `x_C = 20·(0.16 − 0.10)/0.30 = 4`，`x_F = 16` kg DM；
- `q = (16/0.40, 4/0.80) = (40, 5)` kg as-fed；
- 成本 3.2。

M0 求得的结果与解析解一致，误差 ≤ 1e-9。

---

## 12 实现状态、未实现项与限制（W2 更新，2026-09-24）

P4a 首版此节写的“M1、M2、M3 未实现”“信息价值算法未实现”已过时，按现状改写如下。

**已实现（`implemented`、`unit_passed`；没有 pilot、official）**

- M1、M2、M2b、M3a、M3b、M3c：都已实现并登记在 `REGISTRY`（§6.1）。M2 另为 `smoke_passed`（N = 64，α_train 未校准，只说明管线跑通）；M1、M2b、M3 只在合成数据的集成测试中端到端走通。
- 信息价值原型 `ration_reliability.information`：有限候选库、信号分箱、政策 MILP、按决策时信息确定的行动集、~~扣除随机化通道的净值~~（B-435 更正注：R1 起没有“扣除随机化后的净值”，该字段是诊断对照差 `contrast_vs_matched_uninformative_bins`，可为负；FIX_A 的第四口径已由 R3B 降为启发式 `heuristic_min_of_two_policy_values`；金额须完整决策问题，见 §14.5）、理论顺序诊断、经济换算、检测策略对照、冻结政策评估（设计与红队修复后的语义见 `docs/INFORMATION_VALUE_DESIGN.md`）。
- 红队修复后的接口收紧（`audit/_parts/FIX_engine_record.md` §1）：M1 的 `margin_scale`、`apply_to_dm` 必填（§6.2）；每条约束 `numerical_tolerance` > 0（§3.3、§9）；`dm_basis_semantics` 为枚举（§9）；pilot/official 须显式 `mip_rel_gap`（§10）；批量结构检查 `evaluation.structural_check`（§4）；等价方法表 `METHOD_EQUIVALENCES`（§6.3）。

**未实现（`planned`）**

- `reporting` 子包仍为空。
- B 层 NEL 事后评价需要 `nasem_dairy==1.0.2`，本机未安装（需要用户批准）。引擎不计算 NEL、MP，也不把 CP 当作 MP。

**阻断与限制（不能靠改代码解除）**

- D02（M1 的开发期校准）、D08（M2 的 N 阶梯 128/512/1024 与多种子收敛）、F09（候选库扩大、分箱细化、小问题穷举的开发检查）仍为 `blocked`：要在服务器上按协议网格做开发期运行，且 `configs/methods.yaml` 中 M1 的 `margin_semantics.chosen`、`library_size_grid`、`signal_bins_grid` 仍未定。
- M2b 在 N < K_p/α_train 时退化为 M3c（§6.2）；协议选 N 与是否报告 M2b 时须考虑。
- `equivalence_annotations` 只做方法层面的标注，不检查等价条件（§6.3）。
- 合同 §8 第 10 项（正态单约束解析机会约束与独立抽样核对）已由 M2 的数值测试覆盖（`tests/numerical/test_chance_saa.py::test_normal_single_chance_constraint_formula_vs_mc`、`::test_analytic_chance_optimum_has_violation_alpha_by_mc`）；P4a 另做的评估器核对（独立正态、DM 固定时，蒙特卡洛违约率与解析正态概率之差在 4 倍 MC 标准误以内）仍然有效。这些只核对实现，不说明成分服从正态。
- 本机只跑单元测试和小型 smoke；正式实验按用户规定在服务器上运行。

---

## 13 不确定性模型工厂（`uncertainty/spec.py`、`uncertainty/factory.py`；第二轮复核 R3，K3 子任务，2026-09-25）

- 状态（合同 §2.3）：`implemented`、`unit_passed`（`tests/unit/test_uncertainty_factory.py`，全部为合成数据）。`experiments/E0_verification/run_smoke_nasem_dryrun.py` 已改为经工厂构建模型，K3 只执行过 `--check-only`（内存中跑通，不写文件、没有 run id），**新代码上的 smoke 运行尚未执行，不称 `smoke_passed`**；旧 smoke 运行（`IndependentNormalModel` 朴素截断）的数值对新代码失效，只作历史记录。
- 目的：一个 `UncertaintySpec` → 一个 `FactoryModel` → 一个评估世界。优化抽样（`opt`）、开发选择（`validation`）、冻结后测试（`test`）、鲁棒集合与信息价值先验都从同一个模型取；名义方法只是在优化时忽略不确定性，评估时与其他方法在同一世界。
- 审计：`scripts/audit_uncertainty_factory.py` → `reports/uncertainty_factory_audit.json`（只含元数据与合成例、计数和键名）；受限逐格审计 → `data/restricted_local/uncertainty_factory_audit_restricted.json`（不进 git）。

### 13.1 用法

```python
from ration_reliability.uncertainty import RandomStreams, UncertaintySpec, build_uncertainty_model

spec = UncertaintySpec.from_config(cfg)      # 或 UncertaintySpec.from_arrays(...)；两者走同一套校验
model = build_uncertainty_model(spec)        # FactoryModel（UncertaintyModel 子类）
w = model.draw_world(RandomStreams(1103), n_opt=512, n_validation=20000, n_test=10000)   # 同一模型的具名流
box = model.box(k)                           # BoxUncertaintySet.from_factory_model：采样边际实际矩 ± k·SD，裁到物理区间
prior = model.prior_states(RandomStreams(1103), "opt", S)   # 信息价值先验；只接受 opt/validation，test 抛 LeakageError
basis = model.prior_variance_basis()         # {ObservedComponent: variance_basis}，取自对象元数据而非调用方声明
theta0, d0 = model.nominal_state()           # 目标均值（名义方法的规划值）
model.check_problem(problem)                 # 轴、名义值与目标均值一致；被排除的原料不得出现在问题中
model.metadata.to_dict(include_values=False) # 可公开的元数据（不含目标/实际矩与参数）
```

`experiments/E0_verification/smoke_pipeline.py`：

- `build_world(spec_or_model, streams, *, n_opt, n_validation, n_test)`：只接受 `UncertaintySpec` 或 `FactoryModel`，其他模型类型抛 `TypeError`；
- `run_methods`：validation 抽样与 opt 抽样来自不同模型时抛 `WorldMismatchError`；每个 `MethodRun` 记录 `world_fingerprint`（= opt 抽样的 `model_fingerprint`）；
- `evaluate_on_test(..., world_shift_scenario=None)`：test 抽样的模型指纹与 `world_fingerprint` 不同时抛 `WorldMismatchError`；只有显式给出情景 id（如 E2 类有意换世界）才放行，并记入 `MethodRun.world_shift_scenario`。

### 13.2 配置 schema `ration_reliability.uncertainty_spec/0.1`

顶层键（未知键、缺键一律报错，全部问题一次列出，`ConfigValidationError`）：

| 键 | 取值 | 说明 |
|---|---|---|
| `spec_id` | 非空字符串 | |
| `purpose` | `main_analysis` / `sensitivity_scenario` / `smoke` / `diagnostic` / `unit_test` | 只有 `diagnostic` 可以使用朴素截断族 |
| `moment_semantics` | `target_marginal_moments` / `naive_parent_parameters` | 前者 = 给定均值/SD 就是抽样分布的矩；后者只允许 `diagnostic` |
| `is_synthetic` | bool | 非合成规格中不得出现 `synthetic_test_only` 的值或族规则 |
| `ingredient_ids`、`nutrient_ids` | 列表 | `DM` 不是成分列 |
| `family_rule` | 见 13.3 | 六个键全部必填 |
| `cell_defaults` | 元数据字段的默认值 | 只是书写便利；每个字段必须在格或此处显式给出，代码里没有隐含默认 |
| `cells` | 每个 (原料, 成分) 与 (原料, `DM`) 各一行 | 必须全部列出；`mean: null` = 缺失（抽样为 NaN，绝不当 0）；`sd: 0` = 点值；有均值无 SD 报错（点值政策待定） |
| `correlation` | 可选，见 13.5 | |
| `two_layer` | 可选，见 13.6 | 只作扩展情景 |
| `notes` | 字符串 | 进入指纹 |

每格元数据字段：`lower`、`upper`（物理区间，显式给出）、`variance_basis`（与 `information.signal.PRIOR_VARIANCE_BASES` 同一词表：`true_batch_state` / `observed_incl_sampling_and_lab` / `observed_incl_lab_only` / `unidentified`；点值与缺失格自动记 `not_applicable`）、`data_fingerprint`（来源行的哈希；合成值写 `synthetic:<label>`）、`decomposition_id`（未做方差分解时写 `none`）、`decomposition_source`、`measurement_model_id`、`provenance_status`（`sourced` / `research_scenario_assumption` / `synthetic_test_only`；待定值不能建模）、`source_id`、`locator`；可选 `family` + `family_choice_reason`（逐格改族必须写理由）。

一致性规则（与 R2 对齐）：观测口径（`observed_*`）必须写 `measurement_model_id`；非合成数据声明 `true_batch_state` 必须有方差分解（`decomposition_id` + `decomposition_source` + 被去除的 `measurement_model_id`），不能靠一句声明把观测 SD 当成真实批次 SD；`sourced` 必须有 `source_id` 与 `locator`。

### 13.3 族规则与 `infeasible_moment_match`

```yaml
family_rule:
  primary_family: TN_MM            # TN_MM / LN_MM / BETA_MM，或只在诊断规格中用 TN_NAIVE_DIAGNOSTIC
  fallback_families: [BETA_MM]     # 按顺序尝试；可为空
  on_exhausted: error              # error / exclude_cell_as_missing / exclude_ingredient
  status: research_scenario_assumption
  rationale: "为什么选这个族、这个回退顺序"
  selection_basis: declared_rule   # 或 development_data（13.8）
```

- 每格依次尝试：逐格覆盖的族（或开发数据选出的族）→ 规则主族 → 回退族。某族不能复现目标均值与 SD 时（截断正态相对下界 CV ≥ 1、Bhatia–Davis 界、数值不收敛、族不适用于该区间），这次尝试记 `fit_status = "infeasible_moment_match"`，保留原始状态与标志；成功的族与失败轨迹写进 `family_choice_reason`。
- 回退用尽时按 `on_exhausted`：`error` 抛 `FactoryBuildError` 并列出每个格；`exclude_cell_as_missing` 把该格设为缺失（NaN，评估器记 `data_quality_flag`，不当 0；DM 格不允许）；`exclude_ingredient` 把该原料移出模型轴，`check_problem` 要求问题也移出它。
- **目标矩从不修改**，也不静默裁剪；元数据保留原目标值。
- 工厂层 `fit_status`：`matched` / `point` / `missing` / `diagnostic_drift` / `infeasible_moment_match`（被排除的格）。

### 13.4 朴素截断正态只作诊断

- `TN_NAIVE_DIAGNOSTIC` = 父正态取目标均值/SD 再截断（旧 smoke 行为，均值、SD 漂移）。只允许 `purpose: diagnostic` 且 `moment_semantics: naive_parent_parameters`，且不得有回退族；在任何声明“给定均值与 SD”的规格中（主族、回退族、逐格覆盖）都会被拒绝。字面量 `TN_NAIVE` 与 `N_UNTRUNCATED` 不可请求。
- 诊断格的元数据：`fit_status = "diagnostic_drift"`、`moment_drift_flag = True`、`is_diagnostic = True`，并给出实际矩、`mean_rel_shift`、`sd_rel_error`。
- 合成验收（目标均值 0.02、SD 0.015、[0, 1]）：TN_MM 实际矩与目标一致（相对误差 < 1e-12），40 万次抽样均值在 5 个标准误内；TN_NAIVE_DIAGNOSTIC 实际均值 0.0227071、SD 0.0127879（+13.54 %、−14.75 %），与复核包 `independent_probes.json` 逐位一致，并被标记。

### 13.5 相关：潜变量相关与变换后相关分开

```yaml
correlation:
  structure_id: C1_example
  labels: [[corn_silage, NDF], [corn_silage, starch]]
  matrix: [[1, ⟨withheld⟩], [⟨withheld⟩, 1]]
  input_scale: transformed_pearson     # latent_gaussian / transformed_pearson
  handling: declared_latent_scenario   # latent_as_declared / declared_latent_scenario / calibrated_mapping
  status: research_scenario_assumption
  provenance: "..."
  source_id: null
  locator: null
```

- `latent_gaussian` 只能配 `latent_as_declared`：矩阵就是高斯 copula 的潜变量相关。
- `transformed_pearson`（例如文献的 Pearson r）不能当潜变量相关直接使用而不声明：`declared_latent_scenario` = 按声明把 r 当潜变量相关（情景），元数据报告由此得到的变换后相关（与 r 不同）；`calibrated_mapping` = 逐对求解潜变量相关，使变换后 Pearson 等于 r（二维 Gauss–Hermite 求积，96×96 节点，确定性）；r 超出这两个边际可达到的范围时报错，不强行取边界。
- 联合潜变量矩阵必须半正定，否则 `FactoryBuildError`（不做 Higham 修复）。
- 元数据 `CorrelationMetadata` 同时给出 `input_matrix`、`latent_correlation`、`transformed_correlation`、`latent_lambda_min`、`max_abs_latent_minus_transformed`、逐对校准记录与口径说明。合成例：两个右偏边际下潜变量 −0.8 对应变换后 −0.686；文献 r = −0.6 按声明使用时变换后只有 −0.524，校准映射需要潜变量 −0.693。

### 13.6 两层农场模型只作扩展情景

- `two_layer` 块生成 `ExtensionScenario`（`model.extension("two_layer_farm")`），**不改变主模型、主模型抽样或其指纹**（有无该块，主模型指纹与抽样逐位相同；该块本身进入完整规格指纹和扩展情景指纹）。`role: primary` 会被降级并记 `demoted_from_primary`。
- 真值端方差口径不另行声明，直接取被覆盖格的 `variance_basis`（对象元数据；口径混杂时 `invalid_extension_config`）。
- 识别条件：比值 `ratio_status = sourced` 且覆盖格为 `true_batch_state`。不满足时：`allow_unidentified_scenario: true` → 建成并标 `unidentified_extension_scenario`；否则 `not_built_unidentified`。当前项目数据（比值为推算、NASEM SD 为观测口径）属于后者。

### 13.7 指纹与复现

- `FactoryModel.fingerprint()`（= 所有抽样的 `DrawSet.model_fingerprint`，即世界 id）哈希：工厂版本、主规格指纹（除 `two_layer` 外的全部配置字段）、拟合后的边际与相关、完整元数据指纹。任何一个相关字段（目标矩、区间、族规则及其理由、逐格族、方差口径、数据指纹、分解、测量模型、来源、相关值/口径/状态、用途、`spec_id`、`notes`）改变，指纹都变（参数化测试逐项核对）。
- 固定规格 + 固定 `RandomStreams(root_seed)` + 同一流名 → 抽样逐位相同。
- `metadata_for_drawset(draws)` / `metadata_for_fingerprint(fp)`：本进程内按指纹查元数据（便利查找）；权威副本是 `model.metadata`，运行记录应序列化它（smoke 脚本写 `uncertainty_model` 摘要与受限全量元数据）。

### 13.8 开发数据选族

- `select_family_on_development_data(cell, observations, *, split, data_label)`：在目标矩固定的前提下，按开发观测的负对数似然在候选矩匹配族中选族；`split` 只能是 `development` / `validation` / `opt`（或传 `opt`/`validation` 流的 `DrawSet`），其他（包括 `test`）抛 `LeakageError`。
- `family_rule.selection_basis: development_data` 时，`build_uncertainty_model(spec, family_selection={...})` 必须为每个随机格提供选择结果，且其 split 与目标矩与规格一致；`declared_rule` 规格传入选择结果会报错。

### 13.9 与 R2（prior / truth / signal 防护）的接口

- 工厂元数据逐格给出 `variance_basis`、`data_fingerprint`、`decomposition_id/source`、`measurement_model_id`、`is_synthetic`；`FactoryModel.prior_variance_basis()` 可直接作为 `compute_information_value(prior_variance_basis=...)` 的来源，`metadata_for_drawset` 可让 `PriorStates.from_drawset` 绑定对象元数据。K3 没有修改 `information/` 下的文件；把元数据写进 `PriorStates` 并让防护核对 prior/truth/signal 三端，属于 R2 的任务（见 `audit/_parts/round2/K3_uncertainty_factory_log_additions.md`）。

### 13.10 实例化点与遗留

- 全仓 `IndependentNormalModel(`、`naive_truncnorm(`、`"TN_NAIVE"`、`from_independent_normal(` 的出现位置与分类见审计报告 `instantiation_inventory`：研究入口（smoke 脚本）已改走工厂；测试中的独立正态夹具是被测机制的合成样例，保留；`BoxUncertaintySet.from_independent_normal` 保留为遗留适配器，构造参数记 `parameter_basis = declared_parent_parameters_not_sampled_moments`；`scripts/build_phase3_tables.py` 的 `TN_NAIVE` 是带标签的漂移诊断。
- 仍可绕过工厂的地方（类本身未加限制，只由静态守卫测试把关）：直接构造 `IndependentNormalModel`、`GaussianCopulaModel.from_targets(..., allow_drifted=True)`、`TwoLayerFarmModel`（`family` 可取 `TN_NAIVE`）。

## 14 参考能量检查与 Table 5-1 适用域（第三轮复核 F3/F4，执行指令 D；R3D，2026-09-25）

状态：`implemented`、`unit_passed`（`tests/numerical/test_energy_reference.py`、`tests/unit/test_domain_applicability.py`，合成手算例）；证据层级 `code_tested`。阈值“主要淀粉源 ≥ 0.50”为 `assumption_only`（项目研究假设，不归于原文）。说明与开发诊断见 `reports/energy_domain_audit.md`。

### 14.1 `nutrition/energy_reference.py`：线性能量行 vs 项目非线性第 3 章参考链（逐状态）

```python
from ration_reliability.nutrition.energy_reference import (
    EnergyReferenceSpec, reference_energy_check, joint_with_reference_energy)

spec = EnergyReferenceSpec.from_constraint(lin, constraint_spec)   # PN-NEL-FIXEDDMI 行：bound、容差、id 取自约束
# 或 EnergyReferenceSpec(lin, bound_mcal_d, tolerance_mcal_d, constraint_id="PN-NEL-FIXEDDMI",
#                        fmcp_g_per_kg_dmi=16.5, column_check_atol=1e-9)
res = reference_energy_check(q, draws, spec) -> EnergyReferenceResult
res.summary() -> dict                   # 计数、假通过/假不通过、误差分布
res.class_counts(mask) -> dict          # 任意状态子集内的分类计数（例如按 Table 5-1 适用域）
jt = joint_with_reference_energy(eval_result, res, member_ids=None) -> dict
```

- 输入：`q [I]`（原物质，固定；原料顺序 = `draws.ingredient_ids` = `lin.ingredient_ids`，不自动重排；q < 0 报错）；`draws` 为 `DrawSet`，须含 `lin.nutrient_map` 的成分列；若含 `NEL_fixedDMI` 列则线性行用该列（公共评估器用的就是它），并先与 `lin.density` 对同一成分的结果比对，差 > `column_check_atol` 即判为另一世界、报错。
- 每个状态 s：实际 DM `x_s = q·d_s`，**不归一化，不用隐藏 DM 重投料**（函数没有 d̂ 或计划 DM 参数）。线性供给 `L_s = Σ x e + C0`，线性余量 `L_s − R` 与公共评估器该行的 margin 相同（测试核对到 1e-12）；参考供给 `N_s = energy.nonlinear_diet_nel`（只传入使用中的原料；DMI = 该状态供给的 DM，假定全部采食；fMCP 固定 16.5 g/kg DMI；FA、木质素固定于原料值）；`R = bound + C0`；两者都用该行的数值容差判定。
- `EnergyReferenceResult` 的逐状态数组（只读，长度 S）：`linear_supply_mcal_d`、`linear_margin_mcal_d`、`linear_defined`、`linear_violated`、`reference_supply_mcal_d`、`reference_margin_mcal_d`、`reference_defined`、`reference_violated`、`difference_mcal_d`（= L − N，> 0 表示线性行高于参考链）、`state_class`、`undefined_reason`、`analysis_anomaly`、`support_violation`、`diet_dmi_kg_d`、`diet_starch_pct`；另有 `q_hash`（与评估器同式）、`stream_id`、`model_fingerprint`、`spec_fingerprint`、`energy_column_source`、`label = "model_difference_not_animal_outcome"`。
- `state_class ∈ {agree_pass, agree_fail, false_pass, false_fail, reference_undefined, linear_undefined}`：false_pass = 线性过、参考不过；false_fail = 线性不过、参考过。`reference_undefined` 的原因：`missing_data`、`support_violation`（使用中原料木质素 > NDF 或分数越出 [0, 1]）、`no_dm_supplied`、`chain_error`。未定义状态留在分母：`summary()` 给参考违约率上下界 `[n_fail/S, (n_fail + n_undefined)/S]`。
- `analysis_anomaly`（使用中原料 CP+NDF+淀粉+EE+灰分 > 100 % DM 或 Eq 3-1 ROM < 0，类 `analysis_overlap_or_measurement_anomaly`）只作标记：参考链照算，状态不删不裁剪；`summary()` 另给这些状态内的分类计数。
- `joint_with_reference_energy`：用参考链判定替换能量行的线性判定，其余行保留评估器判定，给出 `joint_violation`、`joint_unknown`、`n_violated`、`n_unknown`、`rate_lower`、`rate_upper`（分母 = 全部状态）。先核对同一 q（`q_hash`）、同一状态数、同一流，并核对评估器能量行 margin 与参考检查的线性 margin 一致（1e-9），否则报错。`member_ids` 默认取评估器结果中全部 `probabilistic_nutrition` 行；能量行不在成员中（如 S2 的计划行）时联合事件不变。
- 供 R3C 的 `evaluate_reference(q, draws, spec)`：对每份 q 调用 `reference_energy_check` 与本节 `joint_with_reference_energy` 即可得到“能量按参考链判定”的全参考事件；线性模型仍可用于生成候选，但参考检查不得进入候选生成后的事后重投料。若把参考检查用于筛选或修复，所有方法须同规则、同计算预算，并在保留测试前固定（本轮未实现任何修复或割平面）。（B-435 更正注：`evaluate_reference` 已由 R3C / FIX3_BC 实现在 `evaluation/reference.py`，接口、事件 id 与域条件主事件见 §14.4。）

### 14.2 `nutrition/energy.py`：成分闭合分类（F4）

```python
cls = classify_composition_states(theta, nutrient_ids, ingredient_ids, *, lin=None,
                                  sum_columns=CLOSURE_SUM_COLUMNS, fraction_columns=None, tol=1e-9)
cls.analysis_anomaly     # [S, I] = sum_gt_100pct | rom_lt_0   -> ANALYSIS_ANOMALY_CLASS
cls.support_violation    # [S, I] = lignin_gt_ndf | fraction_outside_unit_interval -> SUPPORT_VIOLATION_CLASS
cls.summary()
```

- `analysis_overlap_or_measurement_anomaly`：分析定义重叠（NDF 本身含部分含氮化合物与灰分，CP、灰分又单独计一次，NASEM 2021 p.22–23；EE 不等于 FA）、采样/检测误差或推导量异常；触发检查，但不能单凭此宣布状态物理上不可能。
- `support_violation`：木质素 > NDF、质量分数越出 [0, 1]；第 3 章链在此未定义（Eq 3-3a），`nonlinear_diet_nel` 报错且错误信息带类名。
- 两类都只标记：不强制闭合、不归一化、不裁剪、不删样本。`composition_closure_report` 保留原字段并新增 `share_support_violation`、`share_lignin_gt_ndf`、`classes`。

### 14.3 `nutrition/domain.py`：Table 5-1 适用域（逐状态 / 逐配方）

```python
from ration_reliability.nutrition import domain as DOM
variants = DOM.default_table51_variants(dgc_ids, corn_silage_ingredient_ids=())   # 主 0.50 + 敏感性 1/3、2/3（+玉米籽粒读法）
r = DOM.table5_1_domain(q, draws, spec) -> Table51DomainResult          # 逐状态
r = DOM.table5_1_domain_arrays(q, theta, d, ingredient_ids, nutrient_ids, spec)   # 逐配方名义：theta=名义成分、d=d̂
r.summary(); r.counts(); r.crosstab(violated, undefined)
v, u = DOM.rows_verdict(eval_result, row_ids)                           # 例如 PN-T1..PN-T5 的“任一违约”
```

- 状态：`in_domain_conditional`（主要淀粉源判据满足；TMR 与粗饲料粒度为 `assumption_only`，所以只是“在这两项假设下可按 Table 5-1 读”）、`not_assessable`（不满足：Table 5-1 行不能据此判定或外推；留在分母；不称安全，也不称致病）、`undefined`（无淀粉供给或使用中原料缺 DM/淀粉）。
- 判据按 DIAG-T51 行的线性形式计算：`margin = Σ_counted x·St − τ·Σ_all x·St ≥ −tol`（kg/d，默认 tol 1e-6）；τ = 0.50 时与评估器对 DIAG-T51 型行（权重 +0.5/−0.5）的判定逐状态相同（测试与开发诊断均核对）。
- `Table51DomainSpec` 拒绝把阈值标为 sourced、把 TMR/粒度标为已满足，也拒绝主变体使用“含玉米青贮”的读法；变体须在看结果前固定，不按结果挑选。

### 14.4 `evaluation/reference.py`：统一参考评估与域条件主事件（R3C、FIX3_BC C-1；B-435 补写，W2_state_docs，2026-09-25）

状态：`implemented`、`unit_passed`（`tests/unit/test_evaluate_reference.py`，合成手算例），证据层级 `code_tested`；在第三轮完整消融 `pilot-20260925T055921Z-fb4f75af` 中对全部配方实际调用（开发资料，`holder_verified`，数字不得进稿件）。说明与裁决见 `docs/reference_problem_v1.md` §4、§6、§9。本节只写接口，不写受限数值。

```python
from ration_reliability.evaluation.reference import (
    load_reference_constraints, ReferenceSpec, default_events, evaluate_reference, MAIN_EVENT_T51_IGNORED)

table = load_reference_constraints("configs/dev_case_v1/reference_constraints.csv")   # 27 行；加载器校验 11 / 6 / 5 划分与角色
spec = ReferenceSpec.from_problem(problem, lin, table, dgc_ingredient_ids=[...],
                                  corn_silage_ingredient_ids=(), events=None, domain_variants=None, confidence=0.95)
ev = evaluate_reference(q, draws, spec) -> ReferenceEvaluation   # q 固定：RationDecision 或 [I] 原物质 kg/头/日
ev.summary()["headline"]; ev.summary()["events"]; ev.per_constraint_rows()
ev.event_violation[eid], ev.event_unknown[eid]                  # 逐状态只读数组 [S]
```

- 一个物理定义：`spec` 固定约束、参考能量、Table 5-1 变体与事件；所有方法、所有格的配方都用同一个 `spec` 评估（`spec.fingerprint()` 进运行身份）。`q` 不重投料、不用隐藏 DM 重配；`d_hat` 只用于结构计划行。抽样标签与参考问题不一致时报错（不自动重排）。
- 比率口径：分母恒为全部状态 S；成员未定义且无成员违约的状态记 `unknown`，比率报成对 `[n_violated/S, (n_violated + n_unknown)/S]`，筛选惯例用上值，单侧 Clopper–Pearson 上界按上值计算，另报 MC 标准误（只表示固定声明分布下的模拟误差）。
- `default_events(table)` 的事件 id（`EventDefinition(event_id, members, energy_verdict, description, table51_domain)`；`energy_verdict ∈ {reference, linear, not_member}`，`table51_domain ∈ {primary, ignored}`）：
  - `main_reference`：主参考 9 行（`role = main_reference`），能量行按参考链判定，**Table 5-1 行只在主变体域内判定**（`table51_domain="primary"`）：主变体域外（`not_assessable` / `undefined`）状态下 PN-T1–T5 记 `unknown`，该状态除非其他成员违约，否则计入比率上值。这是开发期主终点。
  - `main_reference_linear_energy`：同上，能量行用线性固定 DMI 行（只作对照）。
  - `main_reference_t51_verdicts_used_out_of_domain`（常量 `MAIN_EVENT_T51_IGNORED`）：R3C 读法，域外照用 T 行判定（`table51_domain="ignored"`）；只作对照，不是主终点。
  - `main_reference_plus_cp_hi`：主参考 + 研究假设行 PN-CP-HI（显示裁决 A2 的影响；与主事件并报，不替代）。
  - `h0_eleven`（H0 声明的 11 行，线性能量行，= FULL11 训练事件）、`h0_eleven_reference_energy`、`h0_eleven_domain_conditioned`。
  - `s2_six`（S2 声明的 6 行，= PART6P5 训练事件）、`s2_six_domain_conditioned`。
  - `excluded_five`（S2 移出的 5 行，能量按参考链）。
- `summary()["headline"]`：`primary_domain_variant`、`t51_primary_not_assessable_share`、`t51_primary_undefined_share`、`t51_primary_in_domain_share`、`main_reference_rate_lower`、`main_reference_rate_upper`、`main_reference_cp_upper`、`main_reference_unknown_only_because_out_of_domain`（因域规则才成为 unknown 的状态数）、`main_reference_t51_verdicts_used_out_of_domain_rate_upper`、`note`。每个主比率旁边都报主变体域外份额。
- `summary()["domain"][variant_id]`：每个声明变体（主变体 `T51-DGC-0.50`，τ = 0.50 为项目研究假设；敏感性 1/3、2/3 与“含玉米青贮籽粒淀粉”读法）的域状态份额、T 行与事件的域内 / 域外交叉表，以及 `main_reference_conditioned_on_this_variant`（按本变体域条件的主事件比率）。变体在看结果前固定，不按结果挑选。
- 其余输出：逐约束行（违约 / 未定义计数、自然单位的亏缺、余量分位数）、参考能量的线性 / 参考判定与假通过 / 假不通过（§14.1）、使用中原料的成分类别计数（§14.2）、结构计划行。`REFERENCE_SCHEMA = "ration_reliability.reference_evaluation/2"`。
- 已知残项：`evaluation/reference.py` 的域条件主事件与 `nutrition/domain.py::premise_conditioned_event`（FIX3_DEF）是两套实现，尚无交叉一致性测试（B-442）。

### 14.5 信息价值：`heuristic` 口径与金额接口的 `decision_problem` 参数（R3B、FIX3_BC；B-435 补写）

状态：`implemented`、`unit_passed`（`tests/unit/test_garbling_counterexample.py`、`tests/unit/test_money_needs_a_complete_decision_problem.py`、`tests/unit/test_value_callsite_scan.py`），证据层级 `code_tested`；研究决定见 `docs/value_semantics_decision.md`（VSD）。

- 口径（`ValueDefinition`）：`operational_deterministic_cost_difference`（Δ_op，确定性政策类）、`randomized_same_class_information_reference`（Δ_R，随机化同类参考，理论参考）、`contrast_vs_matched_uninformative_bins`（诊断，可为负）、`heuristic_min_of_two_policy_values` = min(Δ_op, Δ_R)（角色 `heuristic`；旧名 `executable_information_supported_value` 为弃用别名）。启发式最小值不混淆单调（garbling 反例 0 → 0.018857），不作任何默认，不作支付上限、盈亏平衡、净值或检测优先级依据；点名使用时输出带 `heuristic=True`、`excluded_from_paper_main_results=True`。诊断比例 B/Δ_op 名为 `matched_uninformative_contrast_ratio`（原 `randomization_share_of_operational`），不是随机化份额；`no_randomization_channel_measured` 只表示该匹配基准未检出。
- 两政策类分列：`InformationValueResult.policy_class_comparison()`；库内风险诊断 `min_attainable_ex_ante_joint_risk`（不换钱）。V0 与 VT 都无解时各口径为 `None` + 原因（不是 0）。
- 金额接口（`information/economics.py`）：

```python
per_head_day_value(result, value_definition, *, scenario=False,
                   decision_problem=None, diagnostic_without_decision_problem=False) -> PerHeadDayValue
batch_gross_value(dc, coverage, *, inventory_check, value_definition=None, scenario=False,
                  unbound_value_declaration=None, decision_problem=None, diagnostic_without_decision_problem=False)
break_even_max_cost_per_batch(...同上...)
net_value_per_batch(dc, coverage, cost_components, *, inventory_check, ...同上...)
strategy_decision_value(options, prior, risk, alpha, budget, *, ..., value_definition, decision_problem=None)
```

  - 既无 `decision_problem` 又未显式 `diagnostic_without_decision_problem=True` → `DecisionProblemRequiredError`（排在口径、库存、绑定、识别标签、随机化通道等原有拒绝之后）；两者同时给出也报错。
  - `decision_problem` = `CompleteDecisionProblem(problem_id, information_processing, allowed_policies, baseline, risk_timing, assay_cost, alpha, declared_in)`，五要素各为 `DecisionProblemElement(statement, status ∈ {sourced, research_scenario_assumption, synthetic_test_only}, source, choice)`。(a) `information_processing.choice ∈ INFORMATION_PROCESSING_CONVENTIONS` 决定唯一政策类与口径（确定性箱→配方映射 → operational，输出 `garbling_monotone=False`；可免费随机化 → 随机化参考，`True`）；只收该口径、α 须与结果一致、(d) 须为 `ex_ante_joint_risk`。另一政策类的量不参与放行，只保留只拒不加的 `randomization_only` 规则。
  - 显式开发诊断：FIX_A 的拒绝照旧（只拒不放），输出 `money_basis="development_diagnostic_without_decision_problem"`、`excluded_from_paper_main_results=True` 并发 `UserWarning`。
  - 所有金额输出都带 `excluded_from_paper_main_results=True` 与 `exclusion_reason`（`RQ3_MONEY_IN_PAPER_MAIN_RESULTS = False`，RQ3 为探索附录）。
  - `strategy_decision_value` 无决策问题时标 `assay_priority_basis="none_without_complete_decision_problem"`；给定时排序口径与 α 必须等于该问题的口径与 α，不用 FIX_A 最小值筛子；无定义 / 被筛除分数为 `None`，原因在 `details["ranking_status"]`。
- 现状：项目没有任何写定的完整决策问题（五要素未定，VSD §2），所以没有合法金额数字；单元测试中的决策问题全部为 `synthetic_test_only`。第三轮完整消融驱动不含信息价值步骤。
