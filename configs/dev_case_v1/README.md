# 开发案例 dev_case_v1（R4：一个可解释、可复算的营养开发案例）

- 编写：K4_nutrition_case（第二轮整改），2026-09-25（KST）；K4c（收尾与独立复核）同日修订，改动见 `audit/_parts/round2/K4_nutrition_case_record.md`。
- 状态（合同 §2.3）：规格与代码 `implemented`；能量与需要量调用层 `unit_passed`；**没有 smoke、pilot 或 official 运行**（运行由 K7 做）。本目录不是 `protocol_frozen`，所有文件 `frozen: false`。
- 审计报告：`reports/minimum_case_nutrition_audit.md`（逐原料表、逐约束表、价格表、能量方法、DMI/矿物裁决、自检）。
- 〔FIX_B，2026-09-25，第二轮红队 B〕改动（数值均未变）：石粉、磷酸氢钙价格状态改为 `research_scenario_assumption`（`prices.yaml`，期间与上游来源未知）；`constraints.yaml` 给 PN-CP-SUP 增加“跨 DMI 搬用 g/d”这一步的研究假设标注与 E5 敏感性登记、更正 PN-EE-HI 的 TFA/EE 表述、登记开发场景 S2 的替代联合事件；`energy.yaml` 更正能量行“保守”的适用范围（只在表均值、D = DMI、淀粉 ≤ S_ref 处成立）并登记成分闭合诊断。改动后按下文重跑构建脚本（133/133），旧构建产物保存在受限目录 `fixb_prev_build_k4c/`。运行由 K7 与 FIX_B 做（`reports/dev_case_v1_report.md`）。

## 一句话

一个参考牛（NASEM 2021 Table 21-1 经产荷斯坦的一列，列标识与输入值见 `configs/animal_profile.yaml`）、8 种原料（6 种随机 + 2 种矿物点值）、一个有日期的价格情景（美国中西部 2026 年第 38 周），11 条进入联合违约事件的营养行 + 4 条结构行 + 6 条诊断行；能量按“固定 DMI 条件下线性化的 NASEM 第 3 章 NEL”处理，所有方法用同一能量行和同一公共评估器。

## 文件

| 文件 | 内容 | 受限数值 |
|---|---|---|
| `animal.yaml` | 参考牛输入（`value_from` 引用 `configs/animal_profile.yaml`，不复制数值）；DMI/矿物版本裁决（nasem_dairy 源码只读核对，commit 9b0b28e = v1.0.2 的 src，逐函数行号）；调用层机器字段 `requirement_version`；推导期望值 | 无新增 |
| `inventory.yaml` | 8 种入选原料（理由、分类、fNDF 权重、AC 依据、矿物处理）；20 种剔除原料与理由；添加量上下限；随机成分范围 | 无（数值在受限表） |
| `prices.yaml` | 每个价格的 URL、报告名、日期、单位、基础、币种与换算；青贮定价规则；敏感性 | 无（公开价格） |
| `constraints.yaml` | 每条约束的表达式、引擎映射、单位、界值（或受限引用）、来源页码、状态 | CP 供给下限写作 `value_ref` |
| `energy.yaml` | 能量链逐步方程与页码、固定 DMI 设置、需要量分项、适用范围、诊断 | 无 |

受限文件（`data/restricted_local/dev_case_v1/`，git 忽略，NASEM 数值）：

- `nasem_feed_library_extract_9b0b28e.csv`：nasem_dairy 饲料库 12 行原文摘录（sha256 `20d0885c…b0d5`）；
- `restricted_values.yaml`：Table 21-3 的 CP 需要量与 Table 21-1 对照值；
- `build_dev_case_v1.py`：参考构建脚本（读本目录与受限表，写出下列文件；K4 原有 128 项一致性检查 + K4c 新增 5 项工厂预检，共 133 项）；
- `ingredient_parameters.csv`、`uncertainty_cells.csv`、`energy_linearisation.json`、`dev_case_v1_problem.yaml`、`build_report.json`：构建产物。

## 给运行者（K7）的用法

```bash
cd <project root>
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 data/restricted_local/dev_case_v1/build_dev_case_v1.py
```

0. 先跑构建脚本：`value_from` / `value_ref` 由 `ration_reliability.nutrition.requirements.resolve_value_refs` 解析（K4 检查点上的脚本因未解析 `value_from` 而报 KeyError，K4c 已修）。
1. 问题：`load_problem("data/restricted_local/dev_case_v1/dev_case_v1_problem.yaml", mode=...)`；构建时已在 `smoke` 与 `pilot` 两种模式下通过校验（0 个 pending、0 个合成值、32 个研究假设值）。
2. 不确定性：用 R3 统一工厂按 `uncertainty_cells.csv` 建基础模型，**成分顺序为 (CP, NDF, starch, EE, ash, Ca, P)**，再用 `ration_reliability.nutrition.energy.EnergyColumnModel(base, lin)` 追加 `NEL_fixedDMI` 列（或对已有 DrawSet 用 `append_energy_column`）。`lin` 由 `energy.linearise_nel_fixed_dmi` 按 `energy.yaml` 设置重建（构建脚本第 4 步），或从 `energy_linearisation.json` 核对。构建脚本第 8 步已用一个声明的预检规格（TN_MM，回退 BETA_MM，独立边际）实际建出工厂模型并包上能量列：48 个随机格全部矩匹配、16 个矿物格为点值，包装后模型的名义状态与问题名义值一致（差 < 1e-12）。族规则与方差口径由 K7 / D-19 / R2 定，预检规格不是选择。
3. 名义方法：问题中的 `NEL_fixedDMI` 名义值 = `lin.nominal_density`（线性，与抽样均值一致）。
4. 每个评估情景另算诊断 `DIAG-NEL-LINEARISATION-RESIDUAL`（`energy.nonlinear_diet_nel`）。
5. 价格为 USD/lb 原物质（玉米、玉米青贮为 DM 报价，载入时按 d̂ 一次换算）；成本单位 USD/头/日，不含运费、加工、盐与预混料。

构建脚本放在受限目录是因为它写出受限数值；若要纳入运行身份（R6），由总负责人把它移到 `experiments/` 并保持输出仍写入 `data/restricted_local/`。

## 不声称

不声称产奶、健康、采食量、繁殖或生产安全的改善；CP 行不等于 MP 充足；能量行是固定 DMI 下的模型 NEL 供给，不是能量平衡预测；Na/K/Cl/Mg/微量元素/维生素不在模型内。
