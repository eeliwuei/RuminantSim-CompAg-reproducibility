# 冻结的参考问题与适用域说明（reference problem v1；开发冻结）

- 编写：R3C_reference_ablation（项目记录），2026-09-25。依据：第三轮复核报告 §4、§5、§9 与下一轮执行指令 C（`docs/review_20260925_round3/`）。这是自查整理，不是独立复核。
- 性质：**开发冻结**（development freeze）——在开发期把“用什么物理定义评估一切配方”固定下来，并用文件 sha256 指纹（§11）锁住；它**不是**合同意义上的正式协议冻结（`protocol_frozen`），`configs/protocol.yaml` 未改动。
- 阶段状态（合同 §2.3）：参考约束表、参考评估器、消融驱动 = `implemented`；参考评估器 `unit_passed`；驱动只做过 `--dry-run` 与 `--tiny`（smoke/debug）；完整 2×3 开发求解（FIX3_BC 前为 2×2）= `blocked`（需要用户授权算力与授权文件；Mac 只做交付端）。
- 证据层级（本轮新增）：代码行为 `code_tested`；页码与受限输入的核对 `holder_verified`（持有方环境，复核方未复算）；裁决规则与各项研究设定 `assumption_only`；本文件没有任何 `independently_reproduced` 的项。
- 受限值规则：本文不写 NASEM 表值或可一步还原它们的量；受限界值只写 `value_ref` 指针，数值在 `data/restricted_local/dev_case_v1/`。
- 不声称：本文件与相关代码不给出任何动物结局（产奶、健康、采食、疾病、利润）；“可靠性”只指声明分布下模型约束的满足频率。
- **修订（FIX3_BC，第三轮红队 C-1–C-5，2026-09-25）**：①主参考事件改为**按域条件判定**：主变体 Table 5-1 域外（`not_assessable` / `undefined`）状态下 PN-T1–T5 记 `unknown`（进入比率对的上值），R3C 的读法（域外照用 T 判定）改名 `main_reference_t51_verdicts_used_out_of_domain` 只作对照；H0 / S2 训练事件照原声明，另加域条件版本；可比较集合按域条件主事件判定成员，并排报告 R3C 读法下的成员与每个配方的域外份额（§4、§6、§9.5）。②“sourced”有两套词汇：`constraints.yaml` 的值块状态与本表的来源裁决；PART6P5 改称“S2 声明的 6 行 + 5 条计划行”，并新增第三个训练臂 **MAIN9**（9 行主参考作机会约束，CP-HI / EE-HI 作标注研究假设计划行），消融由 2×2 扩为 2×3（§3.2、§9）。③冻结指纹移入受代码清单覆盖的 pin 文件 `experiments/E1_cost_reliability/reference_problem_v1_freeze.json`，`development` 标签另要求 pin 被 git 跟踪、HEAD 恰为最后修改 pin 的提交、工作树干净、清单完整且与导入时一致（§11）。④`--dry-run` 的 0 抽样 / 0 求解 / 0 写文件改为包装入口后的实测计数；`--full` 另要求用户写的授权文件列出本机（§10）。⑤两条矿物添加上限行的界值来源状态改为 `sourced_other_quantity`；主结果表固定并报 `h0_eleven` 与“主参考 + CP-HI”（§2、§4）。修改记录：`audit/_parts/round3/FIX3_BC_record.md`。

---

## 0 要点

1. **一个参考问题，评估一切配方。** 不管配方由哪个格子、哪种方法、哪个不确定性世界训练出来，最后都由同一个 `evaluate_reference(q, draws, spec)` 按同一套物理定义（`configs/dev_case_v1/reference_constraints.csv`）评分，在两个 SD 世界的同一 test 流上各评一次。
2. **主参考事件 = 9 行**（按来源裁决，§3）：`PN-NEL-FIXEDDMI`（能量，按第 3 章参考链判定）、`PN-CP-SUP`、`PN-CA-ABS`、`PN-P-ABS`、`PN-T1`–`PN-T5`。`PN-CP-HI`（来源无数值）与 `PN-EE-HI`（来源数值针对总 FA 而非 EE）是标注的研究假设行：照常参与两种优化目标、逐行报告，但不进入主参考事件。
3. **同时报告**：H0 原 11 行事件（线性能量行，历史定义）及其参考能量版本与域条件版本、“主参考 + PN-CP-HI”（显示裁决 A2 的影响）、S2 的 6 行子集及其域条件版本、被 S2 移出的 5 行（并集与逐行）、逐约束违约与自然单位幅度、Table 5-1 适用域（`not_assessable` 单列，分母完整；主参考事件在域外不用 T 行判定，FIX3_BC）、线性 vs 参考能量判定（假通过/假不通过）、分析异常与支持集违规计数；每个比率给 Clopper–Pearson 上界与 Monte Carlo 标准误。
4. **2×3 消融**（原 2×2 + FIX3_BC 的 MAIN9 臂）：SD（H0 表 SD / 已声明的窄化点 SD-S2）×优化目标（11 行联合机会约束 / S2 声明的 6 行机会约束 + 5 条名义计划行 / 9 行主参考机会约束 + 2 条研究假设计划行）。方法只用既有 M0/M1/M2/M3，一牛、一库存、一价格，核心 α = 0.05，补充 0.10 / 0.01。
5. **裁决在看过开发结果之后做出，如实披露**（§3.3）：规则只看来源（有没有给数、给的是不是本行约束的量），不看可行与否。

---

## 1 范围：一牛、一库存、一价格

### 1.1 参考牛

| 项 | 取值 | 出处 / 状态 |
|---|---|---|
| 参考牛 | NASEM 2021 Table 21-1（p.471）经产（Mature, 700 kg）荷斯坦 DIM 200 列；`profile_id = HOL_MULTI_DIM200_NASEM21T21_v0` | `configs/animal_profile.yaml`、`configs/dev_case_v1/animal.yaml`；选择本身为研究设定（`research_scenario_assumption`） |
| 产奶、乳成分、妊娠、体重变化 | 固定情景输入（Table 21-1 / 21-3 及脚注，p.471–473） | 同上；不是预测 |
| BCS | 3.0 | 研究假设（Table 21-1 未列） |
| 计划 DMI | ⟨withheld⟩ kg/d（Eq 2-1，p.12） | 方程有来源；BCS 为假设 |
| 饲养条件 | 舍饲 TMR、热中性、非放牧 | 研究假设；TMR 是 Table 5-1 的前提之一，引擎不能观测（`assumption_only`） |
| 版本裁决 | 书与软件不一致处按 nasem_dairy commit 9b0b28e（`animal.yaml version_adjudication`） | 技术裁决，只读核对，未安装运行（B-109） |

### 1.2 库存

8 种原料（`configs/dev_case_v1/inventory.yaml`）：`corn_silage_typical`、`legume_hay_mid`、`corn_grain_dry_ground_medium`、`soybean_meal_solvent_48cp`、`canola_meal_solvent`、`soybean_hulls`（6 种随机）与 `limestone_ground`、`dicalcium_phosphate`（2 种矿物点值）。库存不受限（研究假设）。20 种剔除原料及理由见该文件；其中快速发酵淀粉源的剔除与 Table 5-1 前提有关（p.63–64）。

### 1.3 价格

`DEV-PRICE-USMW-2026W38`（USD；美国中西部 2026-09-14 至 09-23 的公开报价与推广机构参考价；不含运费、加工、盐与预混料；`configs/dev_case_v1/prices.yaml`）。成本单位 USD/头/日；按决策时 d̂ 一次换算为原物质价格。价格情景是研究设定；成本差只在这一价格情景、且只在可比较可行集合内有意义（§9.5）。

---

## 2 约束：`configs/dev_case_v1/reference_constraints.csv`

27 行，列出 `constraints.yaml` 的全部 22 行与在内存中构建的 5 条计划行（`SH-PLAN-*`）。每行给出：变量与单位、表达式、公开界值（受限值只写 `value_ref`）、界值出处（页码 / DOI 10.17226/25806）、界值来源状态（`threshold_source_status`）、模型形式状态（`model_form_status`）、适用域、裁决状态（`status`）与角色（`role`）、判定模型（`verdict_model`）、所用裁决规则、假设理由、是否属于 H0 的 11 条 / S2 的 6 条 / 排除的 5 条、在三种优化目标中的作用（FIX3_BC 新增列 `optimization_main9_arm`）、`constraints.yaml` 原状态（不改）、证据层级、裁决时是否已看过开发结果。FIX3_BC（红队 C-5 注记一）：`SH-INCL-LIMESTONE-MAX`、`SH-INCL-DCP-MAX` 的界值来源状态由 `sourced` 改为 `sourced_other_quantity`——来源数值是全日粮 Ca/P 的 MTL，换算成单一原料添加上限（忽略其他饲料带入的 Ca/P）属项目推导，与受限 yaml 的 `research_scenario_assumption` 一致；两行只是计划行，不影响主参考事件。

| 组 | 行 | status / role | 在主参考事件中 |
|---|---|---|---|
| 主参考（9） | PN-NEL-FIXEDDMI（参考链判定）、PN-CP-SUP、PN-CA-ABS、PN-P-ABS、PN-T1–T5 | sourced / main_reference | 是 |
| 研究假设（2） | PN-CP-HI（A2）、PN-EE-HI（A3） | research_assumption / optimization_only | 否（逐行报告，属“排除五条”） |
| 计划行（5 + 5） | SH-DM-PLAN、三条添加上限、SH-NONNEG；S2 的 SH-PLAN-*（五条） | planning_only / optimization_only | 否（按 q 与 d̂ 确定性检查，报告 structural_ok） |
| 诊断（6） | DIAG-DM-REAL-LO/HI、DIAG-CA-MTL、DIAG-P-MTL、DIAG-T51-DGC-STARCH-SHARE、DIAG-NEL-LINEARISATION-RESIDUAL | diagnostic / diagnostic | 否（逐行报告） |

“H0 的 11 条”= 主参考 9 条 + PN-CP-HI + PN-EE-HI；“S2 的 6 条”= PN-CP-SUP + PN-T1–T5；“排除的 5 条”= PN-NEL-FIXEDDMI、PN-CP-HI、PN-EE-HI、PN-CA-ABS、PN-P-ABS。表的加载器（`evaluation/reference.py::load_reference_constraints`）会拒绝不满足这三组划分或裁决规则的表。

`constraints.yaml` 与构建产物没有改动（它们是构建输入，改动会使预检失败、需重建）；新表是 `.csv`，不在构建输入之内。

---

## 3 裁决规则（只依据来源）与如实披露

### 3.1 规则

| 规则 | 内容 | 用于 |
|---|---|---|
| A1 | 界值由来源给出——表值、正文数值，或来源方程在声明参考牛输入下的结果——且本行约束的量与来源相同 → `sourced`，可作主参考。行层面的建模选择（供给形式、跨 DMI 搬用 g/d、维持项按实际 DM、AC 分配、书/软件版本裁决）记为假设，不改变界值的来源性质。候选只限 H0 在结果之前声明的 11 行（不新增行、不把诊断行升入事件）。 | NEL、CP-SUP、CA、P、T1–T5 |
| A2 | 来源没有给数值（或只有定性说法），数值由项目设定 → `research_assumption`，不进主参考；作为标注的研究假设行逐行报告。 | PN-CP-HI |
| A3 | 来源数值针对另一个量，本行把它施加在代理量上 → `research_assumption`（代理映射），不进主参考。 | PN-EE-HI（总 FA 上限施加在 EE 上） |
| A4 | 确定性计划行（q 与 d̂）→ `planning_only`；不是随机事件，报告 structural_ok。 | SH-*、SH-PLAN-* |
| A5 | 案例设计时（结果之前）声明为诊断的行保持诊断，不论其界值是否有来源；不升入主参考事件（不新增指标）。 | DIAG-* |
| A6 | 有来源的能量需要量，其主参考判定用项目的第 3 章参考链：NASEM p.22 要求按整个日粮计算能量，线性行把部分消化率冻结在均值、不是逐状态下界（复核 F3）；线性判定并列报告，差异记为模型差异。 | PN-NEL-FIXEDDMI |
| A7 | Table 5-1 各行在每个状态照常计算；前提的操作化判为域外的状态单列 `not_assessable`，分母完整，不删、不称安全、不称致病。FIX3_BC（红队 C-1）：主参考事件在主变体域外不用这些判定（记 unknown，进入比率对的上值），域外照用判定的读法只作对照（§4、§6）。 | PN-T1–T5 |

### 3.2 与 `constraints.yaml` 状态字段的关系

`constraints.yaml` 的 `status` 是“值块”状态，把界值来源与建模选择混在一起。例如 PN-CA-ABS、PN-P-ABS 标 `research_scenario_assumption`，理由是参考牛选择与书/软件版本裁决（`animal.yaml row_status_rationale`：“方程与输入有页码、版本经代码裁决；参考牛与 BCS 为开发案例研究设定”）；PN-NEL-FIXEDDMI 的该状态指线性化与固定 DMI。新表把两者拆成 `threshold_source_status` 与 `model_form_status` 两列，原字段原样保留在 `case_yaml_status` 列，便于对照。参考牛本身是全体行共同的情景设定，不是某一行的界值问题。

**两套“sourced”并存（FIX3_BC，红队 C-2）**：运行时 S2 / PART6P5 的 6 行由 `constraints.yaml` 的值块状态推出（其中 PN-NEL-FIXEDDMI、PN-CA-ABS、PN-P-ABS 标 `research_scenario_assumption`），而本表按 A1 把这三行裁为 `sourced` / `main_reference`。所以 PART6P5 不能称“界值 sourced 的 6 行”，只能称“S2 声明的 6 行 + 5 条计划行”——按本表裁决，它把 3 条有来源的主参考行降成了名义计划行。`constraints.yaml` 是构建输入，本轮不改；代码与文档改用上述称呼（`evaluation/reference.py` 的 `S2_SIX` 注释、消融驱动的 `objective_arms`）。FULL11 与 PART6P5 都不以 9 行主参考为训练目标（FULL11 还把研究假设行 PN-CP-HI 作概率约束训练），因此新增 MAIN9 臂（§9.1）；任何结论都与 `h0_eleven`、“主参考 + CP-HI”并报。

### 3.3 如实披露

- 本裁决写于 2026-09-25，**此时已经看过** H0 与 S2 的开发结果（`pilot-20260924T205621Z-93d8654c`；`correction_report.md` §5）与 R3D 的参考能量 / 适用域诊断（`reports/energy_domain_audit.md` §7）。表中每行的 `decided_after_seeing_dev_results` 都写明这一点。
- 规则只问两件事：来源有没有给这个数；给的是不是本行约束的那个量。规则没有看任何可行性或违约率。
- 已知后果两面都有，不是单向“变容易”：A2 把 PN-CP-HI 移出主参考事件，而开发诊断中 PN-CP-SUP 与 PN-CP-HI 是已知冲突对之一；另一方面 A1 保留了能量行与吸收 Ca、P 行，开发报告中它们在 S2 情景下的违约率约为 0.50、0.49、0.26–0.39（持有方报告值，本轮未复算），在 H0 下能量行本身就使名义配方约一半状态违约。任何后续结论都必须同时给出 H0 的 11 行事件，让读者看到这一裁决的影响。
- 若以后取得新的来源（例如泌乳牛 CP 上限的数值依据，或能按 FA 施加的随机 FA 数据），按新版本登记（`reference_problem_v2`），在新版本上重新运行；本版本与其结果保留为开发历史，不合并成同一试验。

---

## 4 参考事件与比率口径（`src/ration_reliability/evaluation/reference.py`）

| 事件 id | 成员 | 能量行判定 | Table 5-1 行（FIX3_BC） | 用途 |
|---|---|---|---|---|
| `main_reference` | 主参考 9 行 | 参考链 | 只在主变体域内判定，域外记 unknown | 主终点（开发期） |
| `main_reference_linear_energy` | 同上 | 线性行 | 同上 | 显示参考能量检查的影响（对照） |
| `main_reference_t51_verdicts_used_out_of_domain` | 同上 | 参考链 | 域外照用（R3C 读法） | 对照，不作主终点 |
| `main_reference_plus_cp_hi` | 主参考 9 行 + PN-CP-HI | 参考链 | 域内判定 | 显示裁决 A2 的影响，始终并报 |
| `h0_eleven` | H0 的 11 行 | 线性行 | 域外照用（原声明） | 历史 H0 事件（= FULL11 问题的公共评估器联合事件，测试逐状态核对） |
| `h0_eleven_reference_energy` | H0 的 11 行 | 参考链 | 域外照用 | 对照 |
| `h0_eleven_domain_conditioned` | H0 的 11 行 | 线性行 | 域内判定 | 对照 |
| `s2_six` | S2 声明的 6 行 | 不含能量 | 域外照用（原声明） | S2 事件（= PART6P5 问题的联合事件） |
| `s2_six_domain_conditioned` | 同上 | 不含能量 | 域内判定 | 对照 |
| `excluded_five` | 排除的 5 行 | 参考链 | 不含 T 行 | 被 S2 移出的部分 |
| `main9_training_event`（仅消融驱动加入） | 主参考 9 行 | 线性行 | 域外照用 | MAIN9 训练事件核对，不是终点 |

（表头原为“事件 id / 成员 / 能量行判定 / 用途”四列，FIX3_BC 加入 Table 5-1 列与新事件。）

- 分母一律是全部状态 S。某状态有成员行无定义（缺格、无 DM、参考链支持集违规）而没有成员违约时记为 `unknown`：比率给成对值 `[n_violated / S, (n_violated + n_unknown) / S]`；项目筛选惯例把 unknown 计为违约，即取上值；Clopper–Pearson 单侧/双侧界与 MC 标准误按上值计数计算。它们只量化固定声明分布下的模拟误差（合同 T8.2），不是数据不确定性区间。
- 逐约束：每个被评估行给出定义数、违约数、未定义数、比率对、CP 上界、MC 标准误，以及自然单位（该行声明单位：Mcal/d、kg/d、g/d、% DM）的违约幅度（违约时的平均与最大缺口、余量 1 %/5 %/50 % 分位）。能量行出现两次：线性行与参考链。
- 决策信息：q 固定；每个状态实际 DM `x = q·d_s`，不归一化，不用隐藏 DM 重投料（函数签名只有 `q, draws, spec`）；计划行只用决策时 d̂。

---

## 5 能量参考

- 候选生成：各方法仍用固定 DMI 线性化能量行（`NASEM2021_ch3_fixedDMI_linear_v1`）。
- 统一核查：`nutrition/energy_reference.py`（R3D）的项目第 3 章参考链 `NASEM2021_ch3_chain_fixedFMCP_suppliedDMI_v1`：DMI = 该状态实际供给的 DM（假定全部采食，采食行为未建模、未验证）；未消化微生物 CP 固定 16.5 g/kg DMI、内源粪 ROM 34.3 g/kg DMI；FA 与木质素取原料值；假定 RDP 与粗饲料 NDF 充足；无补充 NPN、脂肪补充料、莫能菌素。**不是全功能 NASEM 模型或 nasem_dairy 软件，也不是动物验证。**
- 报告：线性违约率、参考违约率（上下界）、假通过（线性过、参考不过）、假不通过（线性不过、参考过）、差值分布，并按 Table 5-1 主变体的域状态拆分。差异标注 `model_difference_not_animal_outcome`。
- 本版本**没有**把参考链接入任何方法的筛选、修复或割平面；若以后接入，所有方法须同规则、同计算预算，并在任何保留测试前固定（执行指令 D.3）。

---

## 6 适用域

- Table 5-1（p.63）的三项前提：以 TMR 饲喂、粗饲料粒度适宜、干粉碎玉米为主要淀粉源。前两项引擎不能观测，对每个状态都是 `assumption_only`；第三项逐状态按实际供给计算“干粉碎玉米淀粉占日粮淀粉”的份额。
- 原文没有给“主要”的数值。主变体 τ = 0.50（项目研究假设，`T51-DGC-0.50`），敏感性 1/3、2/3 与“含玉米青贮籽粒淀粉”读法（`nutrition/domain.py` 的 R3D 声明变体）；所有变体并列报告，不按结果挑选。
- 域外状态记 `not_assessable`：Table 5-1 行在其中照算，但不能据此判定或外推；单列计数，分母完整，不删除，不称安全，也不称致病。主参考事件、H0 事件、S2 事件都按域状态给交叉表。**FIX3_BC（红队 C-1）**：R3C 版本的主参考事件、H0、S2 事件都把域外状态下的 T 行通过 / 违约直接计入比率，域外份额只作附带列——这正是复核 F3 批评的“把前提失效当小警告”。现在主参考事件在主变体域外不用 T 行判定：T 行记 `unknown`，该状态除非其他成员违约，否则计入比率对的上值（筛选惯例把 unknown 计为违约）；`summary()["headline"]` 把域外份额与主比率并列；每个声明变体另给“按本变体域条件的主事件”比率（阈值是研究假设）；T 行逐行另给域内 / 域外违约计数。可比较集合按此判定（§9.5）。后果（开发诊断，R3D 持有方报告）：主变体下 S2 配方 96–100 % 域外，其主事件上值接近 1，不会进入可比较集合——这是域前提失效的如实结果，不是方法失败。
- 开发诊断已显示（R3D，持有方报告）：S2 配方是否“在域内”随阈值与读法翻转。本版本不因此改日粮范围或阈值；若要改（例如改原料或改用前提相容的模型），按新版本登记并重跑。
- 成分闭合（F4）：使用中原料的“CP + NDF + 淀粉 + EE + 灰分 > 100 % DM”或 Eq 3-1 ROM < 0 记为 `analysis_overlap_or_measurement_anomaly`（NDF 本身含部分含氮化合物与灰分，p.22–23），木质素 > NDF 或分数越出 [0, 1] 记为 `support_violation`；都只计数，主参考事件按这两类状态拆分报告，不闭合、不归一化、不裁剪、不删样本。

---

## 7 数据层级（详见 `docs/uncertainty_data_layers.md`）

| SD 臂 | 定义 | 方差口径与标签 | 证据层级 |
|---|---|---|---|
| `SD-H0` | NASEM Table 19-1 送检观测总 SD 当作本批波动 | `observed_incl_sampling_and_lab`（DC-01） | 数值转录 `holder_verified`；“当作本批波动”`assumption_only`（强假设） |
| `SD-S2` | 声明的窄化点：玉米青贮 DM 1/⟨withheld⟩、NDF 1/⟨withheld⟩、淀粉 1/⟨withheld⟩；其余 45 格借用 1/⟨withheld⟩ | 48 格 `unidentified` / `research_scenario_assumption` | 比值编码 `code_tested`；比值本身 `assumption_only`；原表抽查 `blocked`（B-320） |

- 驱动在任何抽样前核对两臂与 `reports/sd_scaling_sources.csv` 的 `ratio_SD_H0`、`ratio_SD_S2` 逐格一致（48/48），不一致即停止。
- `SD-S2` 是 FIX_B 按“使目标 α 可达”筛选出的窄端点，不是中性或中心估计；从 SD-H0 到 SD-S2 的变化改变的是问题的声明，不是方法改进。本版本不再往下找“更可行”的 SD；R3E 声明的其余敏感性点（SD-SRC14、SD-12MO、SD-SRC12）不在 2×3 消融之内，若要运行须先书面声明并共用同一参考评估器。
- 结论只能写成“给定模型与所列假设下的模型约束可靠性”。

---

## 8 随机流政策（`configs/streams_policy.yaml`）

- 本消融是开发运行，用开发根 `1103`（整根已归开发资料）；test 流在 H0/C1/S2 上已被查看，不是未接触的保留集。
- 三个预留的正式评估根被驱动显式拒绝（`check_seed`，SP-3 的代码守卫只限本驱动）。
- 共同随机数：两个 SD 世界用同一根下的 `opt`、`validation`、`test`（流 id 相同，组 `CRN-ablation`）；母分布不同，状态字节不同，逐世界记录模型指纹与生成方式（执行指令 C.6）。N = 128 用 512 个 opt 抽样的前 128 个。
- 新随机流不是新的真实数据；它只降低同一声明分布下的 Monte Carlo 误差。

---

## 9 求解与评估流程（`experiments/E1_cost_reliability/run_endpoint_ablation.py`）

### 9.1 六个格（原四格 + FIX3_BC 的 MAIN9 两格）

| 格 | SD 臂 | 优化目标 | 说明 |
|---|---|---|---|
| `SDH0_FULL11` | SD-H0 | 11 行联合机会约束 | = 开发运行中的 H0（同问题、同世界、同流、同方法） |
| `SDS2_FULL11` | SD-S2 | 11 行联合机会约束 | 新格 |
| `SDH0_PART6P5` | SD-H0 | 6 行机会约束 + 5 条名义计划行 | 新格 |
| `SDS2_PART6P5` | SD-S2 | 6 行机会约束 + 5 条名义计划行 | = 开发运行中的 S2 |
| `SDH0_MAIN9` | SD-H0 | 9 行主参考机会约束（能量按线性行训练）+ PN-CP-HI、PN-EE-HI 名义计划行 | 新格（FIX3_BC，红队 C-2；在任何消融运行之前声明） |
| `SDS2_MAIN9` | SD-S2 | 同上 | 新格（FIX3_BC） |

MAIN9 由驱动内 `planned_arm_cfg` 构建，规则与 `run_dev_case_v1.s2_problem_cfg` 相同（计划行 = 名义表值、决策时 d̂ 下的结构行 `SH-PLAN-*`，原行改为 diagnostic_only 情景行；不删行、不改界值与容差），计划集合由本表推出（H0 行中角色不是 main_reference 的行，运行时核对恰为 PN-CP-HI、PN-EE-HI）；`planned_arm_consistency` 核对 MAIN9 的 M0 与 FULL11 的 M0 逐位相同、计划行余量等于 FULL11 名义余量。

### 9.2 每个格

1. 预检（R3F `require_dev_case_inputs`，含本驱动所需的参考表、流政策、SD 登记表）；环境锁核对（完整运行须 `matches_lock`）；拒绝预留正式根；SD 登记表核对；开始时的代码/配置清单与规格指纹。
2. 问题：FULL11 = `dev_case_v1` 问题；PART6P5 = `run_dev_case_v1.s2_problem_cfg`（运行时核对：6 行正是 `constraints.yaml` 值块状态为 sourced 的行——注意这与本表的来源裁决不是同一词汇，§3.2；S2 的 M0 与 FULL11 的 M0 逐位相同）；MAIN9 = `planned_arm_cfg`（§9.1）。
3. 世界：`build_h0_spec` / `build_s2_spec` → 统一不确定性工厂（TN_MM，回退 BETA_MM，独立边际）→ `EnergyColumnModel`；名义状态须等于问题名义值。
4. 方法（`run_dev_case_v1.RUN_CONFIG["methods"]`，经 `run_method_block`，不新增方法或网格）：M0；M1 相对余量（含/不含 DM 余量，网格 0–0.10）；M2 联合机会 SAA（N = 128、512；α_train = α × {1, 0.75, 0.5, 0.25}）；M3a 盒式鲁棒（k ∈ {1, 1.5, 2, 2.5, 3}）；M3b 预算鲁棒（k 与 Γ ∈ {0, …, 6}）。在该格自己的 validation 流上按 `rate_upper_le_alpha`（unknown 计为违约）选参；α = 0.05 为核心，0.10、0.01 为补充。
5. 诊断（不是方法）：每个 N 的精确最小训练违约 MILP（时限 60 / 180 s；保留状态、上下界、是否证明），标注“仅训练有限情景”；前沿点 = M2 取 α_train = m*/N（最便宜的该违约数配方，反向核对少一次违约应无解）。
6. 无解处理：保留求解状态、候选逐点状态、MIP gap、对偶界、时限与用时；成本为空，绝不记 0；声明网格或候选库内无解只写“该网格/候选库内无解”，不写成全决策域无解；SAA 的无解只针对那 N 个训练情景。

### 9.3 统一评估

每个有配方的方法（含前沿点）在 **SD-H0 与 SD-S2 两个世界的 test 流**上各调用一次 `evaluate_reference`；另用公共评估器记录该格自己的训练事件在自己世界 test 流上的违约率（检查：FULL11 的训练事件 = `h0_eleven`，PART6P5 的训练事件 = `s2_six`，MAIN9 的训练事件 = `main9_training_event`，`--tiny` 测试逐配方核对相等）。评估前核对没有任何方法用过 test 流（`smoke_pipeline.evaluate_on_test` 的泄漏检查）。

### 9.4 读法

- 沿 SD 轴移动 = 改变不确定性声明（问题），不是方法改进；沿目标轴移动 = 改变训练事件。
- 六条指标的改善不冒充十一条或主参考事件的改善：每行同时给出 `main_reference`（域条件）、R3C 读法、`main_reference_plus_cp_hi`、`h0_eleven`（及域条件版）、`s2_six`（及域条件版）、`excluded_five` 与主变体域外份额。
- 最小违约只说明那 N 个训练情景、那个世界与那套模型。

### 9.5 成本只在可比较可行集合内解释

对每个评估世界与目标 α：可比较可行集合 = 该 α 下（M0 进入每个 α）各格声明方法中，有配方、structural_ok、且主参考事件（域条件，§6）`rate_upper ≤ α` 的配方；另标出 CP 上界是否 ≤ α。FIX3_BC：同时给出 R3C 读法下的成员（`members_t51_verdicts_used_out_of_domain`），每个成员附主变体域外份额、`h0_eleven` 与“主参考 + CP-HI”的比率，集合的 `headline` 给出所考虑配方的域外份额范围；两种成员资格并报，判定只用域条件主事件。覆盖率 = 成员数 / 该 α 下尝试的方法条目数（无配方的条目也在分母中）。成本差只在集合成员之间给出（相对集合内最便宜者），不称“节约”；集合为空时不比较成本。前沿点是诊断，不进入集合。

### 9.6 输出与身份

- `endpoint_ablation.csv`（每个方法 × 评估世界一行：训练目标、生成方式、求解与选参状态、gap、对偶界、时限、无解范围、成本、各事件比率与 CP 上界、能量判定、适用域、闭合计数、structural_ok、是否进入可比较集合）；`reference_residuals.csv`（统一逐约束残差，自然单位）；`solve_status.json`（逐方法逐候选的状态、最小违约与前沿、选参表、S2/FULL11 的 M0 一致性，完整运行时另与开发运行的 H0/S2 方法汇总逐方法对照状态与成本）；`comparable_sets.json`；`ablation_config.json`；受限目录 `rations_q.csv`（逐配方 q；公开包是否含 q 由总负责人按 B-147 决定）；运行记录。
- 输出路由：只有完整运行，且（FIX3_BC）代码/配置清单在开始与结束时都完整、从导入到结束不变（导入前用标准库对将被导入的源码取哈希，与开始清单比对），规格指纹等于 §11 的 pin，且 pin 已被 git 锚定（被跟踪、HEAD 恰为最后修改 pin 的提交、工作树在清单范围、规格文件与 pin 上干净），才标 `development` 写入 `results/pilot/<run_id>/`（公开）与 `data/restricted_local/pilot/<run_id>/`（受限）；否则整份输出标 `debug` 写入 `data/restricted_local/debug/`。`--tiny` 永远是 `debug`。

---

## 10 将要执行的完整命令与算力估计

完整命令（在获授权算力上、冻结代码快照后执行；本轮**没有**执行）：

```bash
cd <project root>
PY=/opt/homebrew/opt/python@3.11/bin/python3.11      # 本机路径；服务器上换成与 environment.lock.json 一致的 Python 3.11
PYTHONDONTWRITEBYTECODE=1 $PY scripts/preflight_dev_case.py --driver run_dev_case_v1
# 冻结（总负责人，合并第三轮全部改动并提交之后）：写 pin → 提交 pin → 不再提交任何东西
PYTHONDONTWRITEBYTECODE=1 $PY experiments/E1_cost_reliability/run_endpoint_ablation.py --write-freeze-pin "<冻结说明>"
git add experiments/E1_cost_reliability/reference_problem_v1_freeze.json && git commit -m "freeze reference_problem_v1"
PYTHONDONTWRITEBYTECODE=1 $PY experiments/E1_cost_reliability/run_endpoint_ablation.py --dry-run   # checks.freeze.status 须为 matches_frozen_anchored
PYTHONDONTWRITEBYTECODE=1 $PY experiments/E1_cost_reliability/run_endpoint_ablation.py --full --authorised-compute \
    --authorisation-file data/restricted_local/compute_authorisation.json
```

- `--full` 必须同时带 `--authorised-compute`，并且（FIX3_BC，红队 C-4）要有**用户本人写的**授权文件（默认 `data/restricted_local/compute_authorisation.json`，受限、不入 git），含 `authorised_hosts`（列出本机 `hostname`）、`authorised_by`、`authorised_on`、`scope`（含 `run_endpoint_ablation`）；缺文件、主机不在列表或字段缺失时在读取任何开发输入之前停止（退出码 2）。运行记录只存该文件的 sha256 与匹配的主机名。agent 不写这个文件。
- `--dry-run` 的“0 抽样、0 求解、0 写文件、0 建目录”是**实测**：干跑期间包装了求解入口（`get_method` 返回的方法、`run_method_block`、`min_violations`、`s2_consistency`、scipy `linprog`/`milp` 及引擎 HiGHS 包装里的绑定名）、抽样入口（`RandomStreams.generator`、`EnergyColumnModel.draw_world`）与写入口（写模式的 `open`、`os.mkdir`/`os.makedirs`），输出 `measured_counts`（含每个入口的调用次数）。仪表本身有测试（`test_dry_run_counters_measure_solves_draws_and_writes`）。

`--dry-run`（本轮在持有方 Mac 上执行，约 0.9 s；实测 0 求解、0 抽样、0 写文件、0 建目录）给出的计划（六格合计）：

| 项 | 数量 |
|---|---:|
| LP 求解（M0、M1、M3a、M3b） | 906 |
| MILP（M2 方法） | 144（每个时限 300 s） |
| MILP（最小违约） | 12（时限 60 / 180 s） |
| MILP（前沿点与反向核对） | ≤ 24（时限 300 s） |
| validation 候选评估 | 1,044 次 × 20,000 状态 |
| 参考评估 | ≤ 126 份配方 × 2 个世界 × 10,000 状态 = 2,520,000 状态 |

- 最坏上界（每个 MILP 都触及时限）：51,840 s ≈ 14.4 h（原四格 34,560 s；MAIN9 两格增加 50 %）。
- 参考评估：按 `--tiny` 实测约 3 × 10⁻⁵ s/状态（smoke，不是基准），约 1–2 分钟。
- 参照（持有方报告，开发运行 `pilot-20260924T205621Z-93d8654c`，本机）：H0 方法块 3.1 s；H0 最小违约与诊断 516 s；S2 块（含信息价值）307 s。据此粗估典型总耗时在 1–2 小时量级，但新格的 N = 512 最小违约与 M2 可能触及时限，**实际耗时须在授权算力上实测**；内存远低于 1 GB。
- 主机：用户授权的计算服务器（B-110、B-415、B-436）；Mac 只做交付端。

---

## 11 冻结指纹（开发冻结；FIX3_BC 起改为 git 锚定的 pin 文件）

- 范围：定义参考问题与本消融的规格文件（驱动 `SPEC_FILES`）逐个 sha256，另以哈希形式记录受限构建报告 `data/restricted_local/dev_case_v1/build_report.json`；`digest` = `stable_hash("R3C/reference_problem_v1/spec_fingerprint/v1", 排序后的文件表, 排序后的受限表)`。
- **FIX3_BC（红队 C-3）**：R3C 把冻结块写在本文件里，而本文件未被 git 跟踪、不在代码清单与 `SPEC_FILES` 内，谁都能重算后粘贴进来得到 `matches_frozen`（自我指涉）。现在：
  1. 冻结值只存在于 pin 文件 `experiments/E1_cost_reliability/reference_problem_v1_freeze.json`（在代码清单范围 `experiments/` 内；由 `--write-freeze-pin` 写出）。驱动不再读本文件的块。
  2. `freeze_status` 除内容比对外还做 git 锚定（只读 git 命令）：pin 被 git 跟踪；HEAD 恰为最后修改 pin 的提交（运行的提交**就是**冻结提交）；`git status` 在代码清单范围（src、experiments、scripts、tests、configs 与三个根文件）、全部规格文件与 pin 上为空。状态：`no_frozen_fingerprint` / `differs_from_frozen` / `matches_frozen_unanchored` / `matches_frozen_anchored`，只有最后一个允许 `development`。
  3. `development` 另要求开始与结束的代码清单都 `complete`、二者相同，且与导入前用标准库对将被导入源码（`src/ration_reliability`、`experiments/E0_verification`、`experiments/E1_cost_reliability` 的 .py）取的哈希一致（`import_time_sources`）。
- 当前状态（2026-09-25，本轮不提交）：pin 文件由 FIX3_BC 写出一次作为格式与初值，但**未被 git 跟踪**，所以 `--dry-run` 报 `matches_frozen_unanchored`（或在其他 agent 改动规格文件后报 `differs_from_frozen`）；任何完整运行都会降为 `debug`，直到总负责人合并第三轮全部改动、重写 pin、提交并在该提交上运行。重写 pin 本身是一次（开发）冻结决定，不能在看过完整运行结果后再改。
- 历史：R3C 在本文件记录的块（digest `3ded0103…59ddb`，2026-09-25T04:12Z）已撤下，不再被读取；其后 R3C / FIX3_BC 等对规格文件的改动都使它失效。

---

## 12 本轮已执行的检查（smoke / debug，不是结果）

| 检查 | 结果 | 层级 |
|---|---|---|
| `tests/unit/test_evaluate_reference.py` | 合成手算：事件 = 独立并集（域条件事件按主变体域内并集，域外 T 行为 unknown）；H0 事件 = 公共评估器联合事件；假通过/假不通过在两种能量判定间的移动；CP 与 MC 公式；自然单位幅度；`not_assessable` 单列；闭合计数；q 固定；FIX3_BC：域外只有 T 行违约的状态在主事件中为 unknown、在 R3C 读法中为违约，headline 域外份额与 unknown 计数，逐变体域条件主事件 | code_tested |
| `tests/integration/test_endpoint_ablation_dryrun.py` | 预检接线；无受限输入时 `--dry-run` 退出码 2 且无回溯；预留根拒绝；SD 登记表一致与篡改检测；指纹检测改动；FIX3_BC：pin 在临时 git 仓库中只有“被跟踪 + HEAD = pin 提交 + 干净”才锚定（未跟踪、脏树、后续提交、重贴均不锚定）；导入时源码核对；干跑计数仪表自检；授权文件；MAIN9 声明；可比较集合两种成员资格；输出路由；（持有方）`--dry-run` 实测 0 计数与六格、`--tiny` 只写 debug 并逐配方核对训练事件 = 对应参考事件（含 MAIN9）、MAIN9 与 FULL11 的 M0 一致 | code_tested；持有方部分 holder_verified |
| `--dry-run` | 预检 READY；环境锁 matches_lock；SD 登记 48/48；六格问题（MAIN9 校验通过）与世界指纹；实测计数 0；计划与算力估计（§10） | holder_verified（smoke） |
| `--tiny`（N = 16，opt 16，validation 500，test 500） | 六格全流程数秒；输出只在 `data/restricted_local/debug/`，标 debug，测试结束删除自身输出；数字不是任何比率或成本的估计 | holder_verified（smoke/debug） |

---

## 13 未决

1. 完整 2×3 开发求解：`blocked`，待用户授权算力与授权文件（Mac 不做重计算）；执行前须提交并重写 pin（§11）。
2. 裁决规则 A1–A7、主参考 9 行与 MAIN9 臂是开发期决定（裁决在看过开发结果之后，已披露），待总负责人确认后写入 `CURRENT_STATE.json`（本任务不写该文件）；正式协议冻结时再定是否沿用。
3. Table 5-1 适用域：主变体下 S2 配方大多域外、而弱读法下全部域内（R3D 诊断）；FIX3_BC 起主事件对域外状态不作 T 判定，所以这些配方的主事件上值接近 1。是否调整日粮范围或改用前提相容的模型，须在完整运行之前决定，并按新版本登记（`reference_problem_v2`），不能按结果挑阈值或读法。
4. 公开包是否包含逐配方 q（已知多步还原途径，B-147）：本驱动只把 q 写入受限目录。
5. 预留正式根只有本驱动内的守卫；`RandomStreams` 层的全局守卫与 `configs/protocol.yaml splits.test_previously_seen` 仍待总负责人（R3E 未决 1）。
6. `nasem_dairy` 未安装运行（B-109）；参考链的结构简化（固定 fMCP、DMI = 供给等）只是声明，未经外部模型或动物数据检验。
7. `constraints.yaml` 的值块状态与本表的来源裁决仍是两套词汇（构建输入不改）；若将来统一，须重建并按新版本登记。
8. `docs/ENGINE_API.md` §14 尚未写入域条件主事件、新事件 id 与 `summary()["headline"]`（不属本任务文件）。

## 14 不声称

不声称产奶、健康、采食、繁殖、疾病或利润结局；不声称真实牧场风险；不声称外部批次验证；不把持有方回执写成独立复核；新随机流不是新数据；候选网格或候选库内无解不是全决策域无解；最小违约只针对训练情景。
