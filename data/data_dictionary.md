# 数据字典（v0.2，2026-09-24，P3_reconcile；FIX_data 修订；G3 补 B 层对账表）

- 状态：`implemented`（字典已起草）。原料 id、成分 id 是**建议版 v0**，未冻结（`protocol_frozen` 之前可改名，改名须同时改 `nasem_t19_1_core.csv` 的生成脚本并记入 DECISIONS）。
- v0.1 修订（FIX_data，按红队 D 的数据发现 B04/B06/B08/B10）：核心表新增第 33 列 `concept_id`；F14 的 `ingredient_id` 由 `beet_pulp_dry` 改为 `beet_pulp_dry_f14`（消除一词两义，§3.2）；F95/F93/F58 中文名不再添加 NASEM 没有的信息；`sd_definition` 加 `_secondary_unverified`；新增 §3.1 P-14–P-16、§3.2 原料 id 对照表、§4.4 枚举 `estimated_TFA_CF_x_0.5678`、§5 第 9–11 条。**没有改动任何数值字段**（mean、sd、n 等与 v0 逐格相同，见 `reports/data_audit.md` §11）。
- v0.2 修订（G3_data_assays，2026-09-24）：A 单录入的 576 个键做了独立第二读并与 A 对账（B-131），新增 §6 说明两份新文件 `nasem_t19_1_entry_A2_second_read.csv`、`nasem_t19_1_blayer_reconciled.csv` 的字段；§4.1 新增 `calculated_italic_no_sd_n`；按 W1 的配置改动同步 §1 末段与 §3.1 P-14、P-16 的处理列（B-143）；§1.1、§5 第 7 条随 B-131 更新。**核心表与它的数值没有改动**（sha256 仍为 `6c732510…2fdd`）。
- 〔FIX5_data_docs，2026-09-24〕§5 第 7 条的 B-131 状态改为“建议关闭，待总负责人确认”（红队「data_docs」发现：状态写在台账前面）。其余内容未改。
- 覆盖对象：`data/restricted_local/nasem_t19_1_core.csv`（NASEM 2021 Table 19-1 / 19-3 双录入裁决后的核心表），以及它的两个输入 `nasem_t19_1_entry_A.csv`、`nasem_t19_1_entry_B.csv`；自 v0.2 起另含 B 层对账表 `nasem_t19_1_blayer_reconciled.csv` 及其第二读输入 `nasem_t19_1_entry_A2_second_read.csv`（§6）。
- **版权**：本文件只含字段定义、单位约定、原料名称与代码、页码，**不含任何 NASEM 数值**，可以进本地 git。数值只在 `data/restricted_local/`（`.gitignore` 第 1 行，已用 `git check-ignore -v` 核实），不进 git、不进交付 zip。依据：NASEM 2021 PDF p.2 版权页写明 “All rights reserved”。
- 页码约定：`pdf_page` = 本地 PDF 物理页；`printed_page` = 书上印刷页码 = `pdf_page − 20`（A、B 两份录入均逐页核对过页眉）。本地 PDF：`data/restricted_local/nasem_dairy_2021_ncbi_bookshelf.pdf`，sha256 `e03e9415807b5e5f623cc84b54ab772b82f474d0dd874a7020a930154a95d49e`。
- 转录审计与裁决记录见 `reports/data_audit.md`。

---

## 1 核心表 `nasem_t19_1_core.csv` 的字段

一行 = 一个（原料 × 成分）单元。共 283 行、33 列（v0 为 32 列，v0.1 在末尾加 `concept_id`），UTF-8，逗号分隔。缺失值一律写字面量 `null`（pandas `read_csv` 默认会把 `null` 读成 NaN）。**`null` 永远不表示 0。**

| # | 列 | 类型 | 含义 / 取值 | 依据 |
|---|---|---|---|---|
| 1 | `core_id` | str | `<表>__<NASEM 代码>__<nutrient_id>`，如 `T19-1__NRC16F48__CP`；预混料为 `NONE__PREMIX__ALL`。唯一 | 本任务定义 |
| 2 | `ingredient_id` | str | **条目级**内部原料 id（一个 NASEM 条目一个 id，§3），不含冒号（`src/ration_reliability/datamodel.py` `IngredientRecord` 的规则）。不得与 §3.2 的概念 id 一词两义 | §3、§3.2 |
| 3 | `ingredient_name_zh` | str | 中文名（本研究用名，不是 NASEM 原文）。只写与 NASEM 条目名同义的内容；NASEM 没有的信息（如物种、加工方式）不写进名称，写进 §3.1 | 调研 §0 Q3；NASEM p.363 |
| 4 | `survey_role` | enum | `core_stochastic` / `core_stochastic_candidate`（须二选一）/ `core_deterministic` / `alternate` / `alternate_candidate` | 调研 §0 Q3；取自 A 录入 |
| 5 | `category` | enum | `forage` / `energy_concentrate` / `protein_concentrate` / `deterministic_input` / `alternate_forage` / `alternate_concentrate`（调研分类，**不是** NASEM 的饲料类型字段） | 调研 §0 Q3；取自 B 录入 |
| 6 | `table_id` | enum | `NASEM2021_Table19-1` / `NASEM2021_Table19-3` / `not_in_NASEM2021_Tables19-1_19-3` | — |
| 7 | `nasem_feed_name` | str | NASEM 印刷名原文（折行已拼接） | 表头 |
| 8 | `nasem_feed_id` | str | NASEM Feed ID Code 原文，如 `NRC16F48`；预混料为 `null` | 表头 |
| 9 | `nutrient_id` | str | 内部成分 id（§2） | §2 |
| 10 | `nasem_row_label` | str | Table 19-1：印刷行标签（去掉脚注字母）；Table 19-3：`<Primary/Secondary/Minor mineral> <元素>; <% column / Absorption Coefficient column>`，或表注 a/e 的说明，或 “not listed … footnote b” | 印刷表 |
| 11 | `unit` | enum | `%`（质量百分数）；`fraction`（吸收系数，无量纲）；`null` | 行标签；Table 19-3 表头 |
| 12 | `basis` | enum | `as_fed`（只用于 DM）；`DM`（其余成分）；`none`（吸收系数）；`null`。取值集合与 `normalization/units.py` 的 `BASES` 一致 | NASEM p.360（pdf 380）：除 DM 外全部按 DM 基础；Table 19-3 表题 “100 Percent DM Basis” |
| 13 | `mean` | str→float | 印刷均值，**按印刷字符串保留末位 0**（印刷为两位小数且末位是 0 的，不改写成一位小数） | 印刷表 |
| 14 | `sd` | str→float | 印刷 SD；口径见 `sd_definition` 与 `reports/data_audit.md` §7 | 印刷表 |
| 15 | `n` | int | 印刷 N，已去千分位逗号 | 印刷表 |
| 16 | `n_raw_printed` | str | N 的印刷原文（保留逗号，或原文就缺逗号的写法） | A 录入 |
| 17 | `value_status` | enum | 见 §4.1 | 表注 a、b；本任务 |
| 18 | `statistic_type` | enum | 见 §4.2（对应 `templates/evidence_contract.yaml` 的 `statistic_type` 必填字段） | 本任务 |
| 19 | `is_measured_or_derived` | enum | `measured` / `equation` / `unknown` / `not_applicable`，见 §4.3。前三个与 `datamodel.ObservationRecord` 的允许值一致；`not_applicable` 只用于无数值的行，加载器必须跳过这些行 | NASEM p.360–362 |
| 20 | `analytical_method` | str | 见 §4.4 | NASEM p.360–362 |
| 21 | `sd_definition` | str | 有 SD 的行 = `within_lab_SD_N_weighted_average_across_4_labs_after_outlier_screening_secondary_unverified`；其他 `null`。**后缀 `_secondary_unverified` 表示其中“实验室内 SD 按 N 线性加权”这一部分未经本项目核对原文**，取得并核对 Tran 2020 原文前不得去掉 | “after outlier screening”：NASEM p.361（pdf 381），±3.5 SD 单变量剔除、主成分得分 ±3.5 SD 剔除、聚类识别亚群；“within-lab N-weighted average”：只来自调研 S1§2.6 对 Tran et al. 2020 的转引（`sources/source_registry.csv` 中 SRC-C-TRAN2020 为 `not_downloaded`）。NASEM p.360–361 正文**没有**描述 SD 的合成方式 |
| 22 | `calculated_italic` | bool | 印刷斜体（= 计算值，Table 19-1 表注 c、i）。核心表 11 项成分中没有斜体值，全部 `false` | A 录入（`pdftohtml` 斜体标记 + 目视） |
| 23 | `pdf_page` | int | 该格所在 PDF 页（跨页原料逐行记录） | A、B 一致 |
| 24 | `printed_page` | int | 印刷页 | A、B 一致 |
| 25 | `entry_A_id` | str | A 录入的 `entry_id`；吸收系数行加后缀 `#absorption_coefficient`；A 无此行时 `null` | — |
| 26 | `entry_B_id` | str | B 录入的 `entry_id` | — |
| 27 | `entry_agreement` | enum | `both_agree`（A、B 独立录入逐字符一致）/ `resolved`（不一致或一方缺行，已看图裁决）/ `unresolved`（未裁决；本版为 0 行） | 本任务 |
| 28 | `resolution_note` | str | 一致性说明、裁决依据（含看图文件名） | 本任务 |
| 29 | `lineage_id` | str | Table 19-1 行 = `NASEM2021_T19_1`；Table 19-3 行 = `NASEM2021_T19_3`；预混料 `null` | 见 §4.5 |
| 30 | `provenance_cluster_id` | str | 见 §4.5 | NASEM p.360；调研 §3.1 |
| 31 | `copyright_status` | str | `NASEM2021_all_rights_reserved__restricted_local_only`；预混料 `not_applicable` | NASEM PDF p.2 |
| 32 | `notes` | str | 数据质量标记（SD ≥ 均值、N < 100、跨页、加工变体同值、TFA 与粗脂肪的关系、定义风险、推断说明等），多条以 `；` 分隔 | 本任务 |
| 33 | `concept_id` | str | 概念级原料 id = `configs/inventories.yaml`（v0 草案）`ingredient_catalog` 的 `ingredient_id`。多个候选条目可共用一个概念 id（如 F14、F15 → `beet_pulp_dry`），用前必须显式选定条目（§3.2、§5 第 11 条） | FIX_data；`configs/inventories.yaml` |

### 1.1 输入文件（只读，不再修改）

| 文件 | 行数 | 对齐键 | 与核心表的关系 |
|---|---|---|---|
| `nasem_t19_1_entry_A.csv` | 845 行 × 23 列；sha256 `917d2d40…9b502` | (`nasem_feed_id`, `nutrient_code`)；`TFAs`→`TFA`、`Crude fat`→`EE`；Table 19-3 的 `absorption_coefficient` 列拆成 `<元素>_absorption_coefficient` 行 | 25 个 Table 19-1 条目 × 33 行 + 5 个 19-3 条目。只有与 B 重叠的键进入核心表 |
| `nasem_t19_1_entry_B.csv` | 283 行 × 14 列；sha256 `7e67a47c…dc42` | (`nasem_feed_id`, `nutrient`) | 23 个 Table 19-1 条目 × 11 行 + 19-3 共 29 行 + 预混料 1 行；核心表的行集合与 B 相同 |

A 中只有单录入、**没有进入核心表**的单元（576 个）见 `reports/data_audit.md` §3.3。2026-09-24 G3 已对这 576 个键做独立第二读（PyMuPDF 渲染看图，读图前没有看 A 的值；Table 19-3 碳酸钙 F1003 的 4 个键除外，见 §6）并与 A 对账：576/576 个键、2,876 次字段比较 0 处不一致，结果写入 `nasem_t19_1_blayer_reconciled.csv`（§6；`reports/data_audit.md` §12）。这些键**仍不进入核心表**，核心表的行集合与数值不变。

---

## 2 成分 id（`nutrient_id`）

内部计算的规范单位是 kg/kg DM（`datamodel.NUTRIENT_DIMENSIONS["mass_fraction"] = "fraction"`）。核心表保留印刷单位 `%`；加载时 `% → /100`（`normalization/units.py` 的 `percent_to_fraction`），DM 的 `% as fed → /100` 得到 d_i（kg DM/kg 原物质）。

| `nutrient_id` | NASEM 印刷行标签 | 中文 | 单位 / 基础 | 本研究角色（调研 §0 Q3、§5.2） | 注意 |
|---|---|---|---|---|---|
| `DM` | DM, % as fed | 干物质 | % / as_fed | 主约束成分（DM 通道） | **唯一按原物质基础的行**。在 `datamodel` 中 `DM` 不是 `NutrientSpec` id，映射到 `IngredientRecord.dm_estimate`（决策时 d̂_i）或情景值 d_s |
| `ash` | Ash, % DM | 粗灰分 | % / DM | 辅助（质量守恒、ROM） | — |
| `CP` | CP, % DM | 粗蛋白 | % / DM | 主约束成分 | CP ≠ MP（合同 §2.1） |
| `ADF` | ADF, % DM | 酸性洗涤纤维 | % / DM | 辅助 | — |
| `NDF` | NDF, % DM | 中性洗涤纤维 | % / DM | 主约束成分 | NDF ≠ peNDF（合同 §2.1） |
| `lignin` | Lignin, % DM | 木质素 | % / DM | 辅助（B 层 NDF 消化率） | NASEM p.362：多数为硫酸法 ADL |
| `starch` | Starch, % DM | 淀粉 | % / DM | 主约束成分 | — |
| `TFA` | TFAs, % DM | 总脂肪酸 | % / DM | 候选（EE 上限若改用 FA 口径） | NASEM 能量计算用 TFA，不用 CF（p.362）；多个原料的 TFA 没有 N/SD（§4.3）；若干条目的 TFA 与粗脂肪关系异常（相等或不符合 p.362 的估算规则），逐格标在 `notes`，汇总见 `reports/data_audit.md` §6 |
| `EE` | Crude fat, % DM | 粗脂肪（乙醚浸出物） | % / DM | 主约束成分（EE 上限） | **NASEM 第 19 章的缩写 “CF” 指 crude fat，不是粗纤维**；粗纤维不列出（p.362） |
| `Ca` | Ca, % DM；Table 19-3 的 Ca % | 钙 | % / DM | 主约束成分 | Table 19-1 矿物只来自湿化学（p.361） |
| `P` | P, % DM；Table 19-3 的 P % | 磷 | % / DM | 主约束成分 | 同上 |
| `Mg`、`Fe`、`Cl`、`Na` | 仅 Table 19-3（矿物原料的主/次/微量元素） | 镁、铁、氯、钠 | % / DM | 不作主约束 | 只是为了保留同一矿物原料的完整行 |
| `<元素>_absorption_coefficient` | Table 19-3 “Absorption Coefficient” 列 | 吸收系数 | fraction / none | NASEM 析因法 Ca、P 需要量换算用 | **模型系数，不是成分含量** |
| `ALL` | — | — | — | 仅预混料占位行 | — |

与并行草案 `configs/nutrients.yaml`（P2b_constraints_units，未冻结）的对应：成分 id `DM`、`ash`、`CP`、`ADF`、`NDF`、`lignin`、`starch`、`TFA`、`EE`、`Ca`、`P`、`Mg`、`Na` 两边写法相同。该文件中的 `Ca_abs`、`P_abs` 是日粮层的吸收量（派生量），不是本表的列；本表的 `Ca_absorption_coefficient`、`P_absorption_coefficient` 是其中要用到的逐原料吸收系数，但只有 Table 19-3 的矿物原料有，基础饲料的吸收系数印刷表没有列出。参数来源指针（红队 D 发现 B08/B10）**已由 W1 处理（2026-09-24）**：`configs/nutrients.yaml` 的 `composition_parameter_file`、`configs/inventories.yaml` 的 `parameter_file_restricted`、`configs/uncertainty.yaml` 的 `base_distribution.parameter_file` 三处都改为指向本核心表 v2（附 sha256 `6c732510b563c5e1f6bbe9c9f82dfd678a9e02763ca6df4ea4b8ee2731bd2fdd`），并写入 `entry_agreement` 行过滤与 `concept_id` 连接规则；A 录入只留在各自的 `*_superseded` 历史字段中（`audit/_parts/W1_record.md` §1、§2.4；G3 于 2026-09-24 重新 grep 三个文件核实）。B 层对账表（§6）不是这三个字段的参数来源，配置目前也没有引用它。

**forage NDF（fNDF）不是 Table 19-1 的行**，核心表里没有它。它在日粮层面由粗饲料原料的 NDF 贡献推导：`datamodel` 用分组项 `G:forage:NDF`，要求每种原料在输入中显式给出 `group_weights["forage"]`（不按名字猜）。各原料的粗饲料标记见 §3 的 `forage_group_weight` 列，其状态见该列说明。

---

## 3 原料 id 映射（内部 id ↔ NASEM 名称/代码 ↔ 中文名）

`forage_group_weight`：用于 fNDF 的粗饲料归属。NASEM p.360（pdf 380）说明每种饲料在模型里另有“类型”字段（dry forage / wet forage / concentrate），但**印刷版 Table 19-1 没有印出该字段**。因此下表的取值都是 `research_scenario_assumption`（按原料性质判断），须在 `nasem_dairy` 饲料库获批安装后（审计 §10 P1-7）逐条对照 NASEM 的类型字段；全棉籽单列为 `pending_user_decision`（决策 D-07）。

| `ingredient_id` | 中文名 | NASEM 印刷名 | NASEM 代码 | 表 | `survey_role` | 调研等级（Q3） | `forage_group_weight` | pdf 页（印刷页） |
|---|---|---|---|---|---|---|---|---|
| `corn_silage_typical` | 玉米青贮（全株，Typical） | Corn Silage, Typical | NRC16F48 | 19-1 | core_stochastic | A | 1（assumption） | 393（373） |
| `legume_silage_mid` | 豆科青贮，中等成熟度（NASEM 把苜蓿、三叶草、百脉根合为一类） | Legume Silage, Mid-Maturity | NRC16F95 | 19-1 | core_stochastic | C/V | 1（assumption） | 400–401（380–381） |
| `legume_hay_mid` | 豆科干草，中等成熟度（NASEM 把苜蓿、三叶草、百脉根合为一类） | Legume Hay, Mid-Maturity | NRC16F93 | 19-1 | core_stochastic | B | 1（assumption） | 400（380） |
| `grass_hay_cool_season_mid` | 冷季禾本科干草，中等成熟度 | Cool-Season Grass Hay, Mid-Maturity | NRC16F34 | 19-1 | core_stochastic | C | 1（assumption） | 389（369） |
| `corn_grain_dry_ground_medium` | 干玉米粒，粉碎（中粉） | Corn Grain Dry, Medium Grind | NRC16F44 | 19-1 | core_stochastic | B | 0（assumption） | 392（372） |
| `corn_grain_steam_flaked` | 蒸汽压片玉米 | Corn Grain, Steam-Flaked | NRC16F46 | 19-1 | core_stochastic | C | 0（assumption） | 392–393（372–373） |
| `barley_grain_dry_ground` | 大麦粒（干，粉碎）；成分统计与蒸汽压片 F1074 相同，加工形态只是名义标签（P-14） | Barley Grain, Dry, Ground | NRC16F8 | 19-1 | core_stochastic | B | 0（assumption） | 384–385（364–365） |
| `beet_pulp_dry_f14`（v0 为 `beet_pulp_dry`） | 干甜菜粕（候选 F14） | Beet Pulp, Dry | NRC16F14 | 19-1 | core_stochastic_candidate | C | 0（assumption） | 386（366） |
| `beet_pulp_dry_molasses` | 干甜菜粕加糖蜜（候选 F15） | Beet Pulp, Dry, Molasses Added | NRC16F15 | 19-1 | core_stochastic_candidate | C（选 F15 可上调） | 0（assumption） | 386（366） |
| `wheat_bran` | 小麦麸皮 | Wheat Bran | NRC16F169 | 19-1 | core_stochastic | B− | 0（assumption） | 412–413（392–393） |
| `soybean_hulls` | 大豆皮 | Soybean Hulls | NRC16F144 | 19-1 | core_stochastic | B− | 0（assumption；调研 §2.2 记“非粗饲料纤维”） | 408–409（388–389） |
| `soybean_meal_solvent_48cp` | 豆粕（溶剂浸提，48% CP） | Soybean Meal, Solvent Extracted, 48% CP | NRC16F134 | 19-1 | core_stochastic | B | 0（assumption） | 408–409（388–389） |
| `canola_meal_solvent` | 菜籽粕（NASEM 条目为北美 canola） | Canola Meal, Solvent Extracted | NRC16F28 | 19-1 | core_stochastic | B（定义风险） | 0（assumption） | 388–389（368–369） |
| `cottonseed_meal` | 棉籽粕（NASEM 条目未注明溶剂浸提或压榨，P-15） | Cottonseed Meal | NRC16F58 | 19-1 | core_stochastic | C+ | 0（assumption） | 394（374） |
| `ddgs_low_fat` | DDGS 低脂（候选 F61） | Distillers Grains and Solubles, Dried, Low Fat | NRC16F61 | 19-1 | core_stochastic_candidate | B（需固定亚型） | 0（assumption） | 394–395（374–375） |
| `ddgs_high_fat` | DDGS 高脂（候选 F59） | Distillers Grains and Solubles, Dried, High Fat | NRC16F59 | 19-1 | core_stochastic_candidate | B（需固定亚型） | 0（assumption） | 394（374） |
| `cottonseed_whole_linted` | 全棉籽（带绒） | Cottonseed Whole, Linted | NRC16F56 | 19-1 | core_stochastic | B | **null（pending_user_decision，D-07）** | 394（374） |
| `fat_calcium_soaps` | 脂肪粉（钙皂） | Calcium Soaps | NRC16F25 | 19-1 | core_deterministic | D | 0（assumption） | 388（368） |
| `limestone_ground` | 石粉 | Limestone, ground | NRC16F1013 | 19-3 | core_deterministic | D | 0 | 429（409）；表注 433（413） |
| `dicalcium_phosphate` | 磷酸氢钙 | Calcium phosphate (dibasic), CaHPO4 | NRC16F1007 | 19-3 | core_deterministic | D | 0 | 428（408），在 431（411）重复列出；表注 433（413） |
| `salt_nacl` | 食盐 | Sodium chloride (salt), NaCl | NRC16F1017 | 19-3 | core_deterministic | D | 0 | 429（409），在 432（412）重复列出；表注 433（413） |
| `sodium_bicarbonate` | 小苏打 | Sodium bicarbonate, NaHCO3 | NRC16F1033 | 19-3 | core_deterministic | D | 0 | 432（412）；表注 433（413） |
| `premix_unspecified` | 预混料（未指定产品） | **NASEM 无条目** | null | — | core_deterministic | D | 0 | — |
| `wheat_straw` | 小麦秸（备选） | Wheat Straw | NRC16F176 | 19-1 | alternate | C | 1（assumption） | 414（394） |
| `corn_gluten_feed_dry` | 玉米蛋白饲料（干，备选） | Corn Gluten Feed, Dry | NRC16F40 | 19-1 | alternate | C | 0（assumption） | 390–391（370–371） |
| `corn_grain_high_moisture_coarse` | 高水分玉米，粗粉（备选候选） | Corn Grain High Moisture, Coarse Grind | NRC16F1072 | 19-1 | alternate_candidate | 待评 | 0（assumption） | 392（372） |
| `corn_grain_high_moisture_fine` | 高水分玉米，细粉（备选候选） | Corn Grain High Moisture, Fine Grind | NRC16F45 | 19-1 | alternate_candidate | 待评 | 0（assumption） | 392（372） |
| `wheat_middlings` | 小麦次粉（备选） | Wheat Middlings | NRC16F173 | 19-1 | alternate | 待评 | 0（assumption） | 413（393） |

只在 A 录入、未进核心表的条目（id 预留，暂不使用；2026-09-24 G3 起这三个条目的 A 单录入键已有第二读，进入 B 层对账表 §6，仍不在核心表）：

| 预留 `ingredient_id` | NASEM 名称 | 代码 | A 录入用途 |
|---|---|---|---|
| `corn_grain_dry_ground_fine` | Corn Grain Dry, Fine Grind | NRC16F1070 | 核对“粉碎度不影响成分统计”（variant_check） |
| `corn_grain_dry_ground_coarse` | Corn Grain Dry, Coarse Grind | NRC16F1071 | 同上 |
| `calcium_carbonate` | Calcium carbonate, CaCO3 | NRC16F1003 | “石粉”映射的候选（`core_deterministic_candidate`） |

### 3.1 定义风险与待定项（均未在本任务中决定）

| 编号 | 事项 | 现状（有来源的事实） | 待谁定 / 对应决策 |
|---|---|---|---|
| P-1 | **菜籽粕（canola）定义风险** | NASEM F28 的条目名是 Canola Meal, Solvent Extracted（北美 canola 溶剂浸提粕）；AFZ 是 00 型；中国常用菜籽粕两者都不是（调研 §0 Q3、§5.3-5）。本表只给 F28 的统计 | 用户/专家；D-15 建议写作 “canola/00 型菜籽粕”，中国菜籽粕只进情景分析 |
| P-2 | **DDGS 脂肪档** | F61（低脂）与 F59（高脂）两条都已双录入；按条目名即为不同脂肪亚群，EE 与 TFA 参数不同，选哪一条会影响 EE 上限约束是否起作用。表中另有 High Protein 条目 F60，未录入 | 用户；D-05（按目标地域常用产品） |
| P-3 | **甜菜粕 F14 / F15** | F14（Beet Pulp, Dry）各主成分 N 都 < 100（小样本）；F15 为加糖蜜产品，N 约高一个数量级；两条均已双录入 | 用户；D-08（按中国常用产品形态，须有依据） |
| P-4 | 高水分玉米 F1072 / F45 | 两条在 11 项成分上逐格相同（加工变体共用一组统计），只有 DE base 不同（DE base 不在核心表）；FIX_data 用 PDF 文字层对全部 33 个印刷行复核：只有斜体 DE base 不同 | 仅 B 层 NEL 需要选；是否纳入备选也待定 |
| P-5 | 干玉米粒粉碎度 | F44（中粉）与 F1070/F1071 成分统计相同（A 录入 33 行均相同；B 目视 11 项相同；FIX_data 文字层复核 33 个印刷行，只有斜体 DE base 不同）；NASEM 能量模型的淀粉消化率随粉碎度不同（调研 Q3）。粉碎度在成分统计上只是名义标签 | D-14，暂定中粉 |
| P-6 | 石粉映射 | 主映射 Limestone, ground（F1013，已双录入）；候选 Calcium carbonate（F1003）：A 录入的 4 个键（Ca、Ca 吸收系数、表注 DM、表注灰分）已由 G3 第二读并与 A 一致，但这 4 个键的第二读**不是盲读**（§6），不在核心表 | 用户 |
| P-7 | 预混料 | NASEM Table 19-1（184 个条目）与 19-3 都没有预混料 | 用户提供具体产品标签；D-04 |
| P-8 | 钙皂的 Ca（及 P、CP、NDF 等） | Table 19-1 中 F25 只有 DM、Ash、TFA、Crude fat 与斜体 DE base 的单一来源点值，其余格空白 = 无数据（表注 a） | 用户提供产品标签或其他有出处来源（B-P3B-1）；不得按 0 |
| P-9 | 矿物原料未列元素 | 石粉的 P，食盐、小苏打的 Ca 和 P 在 Table 19-3 未列出；表注 b 只说明 <1% 不列出 | 是否按 0 近似须作为显式 `research_scenario_assumption` 另行登记，核心表保持 `null` |
| P-10 | 全棉籽是否计入 fNDF | 影响 fNDF ≥ ⟨withheld⟩、NDF + 2·fNDF ≥ ⟨withheld⟩、淀粉 ≤ 2·fNDF − ⟨withheld⟩ 三条约束 | D-07 |
| P-11 | 物种合并 | NASEM p.363（pdf 383）：冷季禾草各物种合并为一类；苜蓿、三叶草、百脉根合并为“legume”。中国羊草、分茬苜蓿不能直接对应（调研 Q3） | 写入局限；中国情景另议（D-01） |
| P-12 | 玉米青贮条目 | 用 Typical F48；成熟度子群（F49/F50）未录入。NASEM p.363：玉米青贮成熟度按 DM 估计 | D-06（代理建议 F48） |
| P-13 | 粗饲料标记的来源 | 印刷表无饲料类型字段，§3 取值为假设 | 装 `nasem_dairy` 后核对（P1-7） |
| P-14 | **大麦加工形态只是名义标签** | NASEM F8（Barley Grain, Dry, Ground，pdf p.384–385）与 F1074（Barley Grain, Steam Rolled，pdf p.385）在全部 33 个印刷行中只有斜体 DE base 不同，11 项核心成分的 Mean/SD/N 逐格相同（B 录入说明“印刷表中的重复条目”；红队 D 看图；FIX_data 文字层全表比对 + 看图 `_page_images/fix_data_F8_p384_p385_labels_plus_col.png`、`fix_data_F1074_p385_labels_plus_col.png`）。所以“干、粉碎”这一加工形态在成分统计上无法与蒸汽压片区分，两者不是独立群体；抽样时共用一组参数，结论不得写成“干粉碎大麦特有的变异”。NASEM p.360 称加工差异“反映在名称中”，但这两条的成分统计并未区分 | 不需要选条目（成分统计相同）；B 层 NEL 若按加工形态区分，只能来自 DE base 等模型计算量。配置层**已由 W1 处理（2026-09-24）**：`configs/inventories.yaml` 大麦条目已补 `variants_same_statistics: ["NRC16F1074"]` 与 `variants_same_statistics_note`（第 140–147 行；W1 先用文字层按列复核 F8/F1074 共 33 行，只有 DE base 不同，`audit/_parts/W1_record.md` §2.1） |
| P-15 | **棉籽粕加工方式未知（定义风险）** | Table 19-1 全表 184 个条目中只有一个 Cottonseed Meal 条目（F58，pdf p.394），条目名未注明溶剂浸提或压榨（FIX_data 与红队 D 对全表条目名的抽取结果一致）。NASEM p.363（pdf 383）对豆粕说明机械压榨与溶剂浸提的粕成分不同、并分条目，棉籽粕没有分。F58 粗脂肪的 SD 相对均值很大（比值只写在受限核心表该行的 `notes` 中），与两种加工混在一起的情况相符【推断，未核实】。中国常用棉籽粕的加工方式与 CP 档也未定（`configs/prices.yaml` 已记“NASEM F58 未注明 CP 档”） | 用户/专家；作为定义风险写入局限，与 P-1（canola）同级处理；EE 上限相关结论对 F58 的依赖须在敏感性中报告 |
| P-16 | **中文名不得添加 NASEM 没有的信息** | NASEM p.363（pdf 383）：苜蓿、三叶草、百脉根合为一类，条目名为 Legume Hay / Legume Silage，没有“苜蓿为主”，也没有“半干”（haylage）的含义；本字典与核心表 v0.1 已改为“豆科青贮 / 豆科干草”。F59/F61 的条目名 Distillers Grains and Solubles, Dried, Low/High Fat 未写谷物来源。v0.1 时 `configs/inventories.yaml`（F95、F93、DDGS）与 `audit/data_coverage_table.csv`（ING02、ING03、ING14）还含 NASEM 名称里没有的“苜蓿”“半干”“玉米” | **已由 W1 处理（2026-09-24）**：`configs/inventories.yaml` F95、F93 改为“豆科青贮 / 豆科干草，中等成熟度（NASEM 把苜蓿、三叶草、百脉根合为一类）”，DDGS 概念名改为“干酒糟及其可溶物（DDGS；NASEM 条目未注明谷物来源）”，各附 `name_zh_basis`（第 65–66、79–80、229–230 行）；覆盖表 ING02、ING03、ING14 第 3 列同步（363 格；`audit/_parts/W1_record.md` §1、§2.3）。G3 于 2026-09-24 重新 grep 核实。条目级中文名仍以核心表为准 |

### 3.2 原料 id 对照表（概念 id ↔ 条目 id ↔ 调研覆盖表 id）

项目里有三套原料 id：调研覆盖表 `audit/data_coverage_table.csv`（`ING01…ING21`、`ALT01…ALT04` 与一个英文 slug）、配置文件（`configs/inventories.yaml` 及引用它的 `prices.yaml`、`assays.yaml`、`uncertainty.yaml`，这里称**概念 id**）、核心表（`ingredient_id`，**条目 id**，一个 NASEM 条目一个）。v0 的问题：`beet_pulp_dry` 在配置里指候选概念 {F14, F15}，在核心表里只指 F14，按 id 直接连接时选 F15 会拉到 F14 的参数。v0.1 把核心表的 F14 改为 `beet_pulp_dry_f14`，并在核心表末列写入 `concept_id`。

| 概念 id（`concept_id`，inventories v0） | 核心表条目 id（`ingredient_id`） | NASEM 代码 | 覆盖表 id | 说明 |
|---|---|---|---|---|
| `corn_silage` | `corn_silage_typical` | NRC16F48 | ING01 `corn_silage` | 1:1 |
| `legume_silage_mid` | `legume_silage_mid` | NRC16F95 | ING02 `legume_silage_mid` | 1:1 |
| `legume_hay_mid` | `legume_hay_mid` | NRC16F93 | ING03 `legume_hay_mid` | 1:1 |
| `cool_season_grass_hay_mid` | `grass_hay_cool_season_mid` | NRC16F34 | ING04 `grass_hay_coolseason_mid` | 1:1（三套写法各不相同） |
| `corn_grain_dry_medium` | `corn_grain_dry_ground_medium` | NRC16F44 | ING05 `corn_grain_dry_medium` | 1:1；F1070/F1071 统计相同（P-5） |
| `corn_grain_steam_flaked` | `corn_grain_steam_flaked` | NRC16F46 | ING06 `corn_steam_flaked` | 1:1 |
| `barley_grain_dry_ground` | `barley_grain_dry_ground` | NRC16F8 | ING07 `barley_dry_ground` | 1:1；F1074 统计相同（P-14） |
| `beet_pulp_dry` | `beet_pulp_dry_f14`；`beet_pulp_dry_molasses` | NRC16F14；NRC16F15 | ING08 `beet_pulp_dry` | **1:2 候选**，D-08 待定；必须显式选条目 |
| `wheat_bran` | `wheat_bran` | NRC16F169 | ING09 `wheat_bran` | 1:1 |
| `soybean_hulls` | `soybean_hulls` | NRC16F144 | ING10 `soybean_hulls` | 1:1 |
| `soybean_meal_48` | `soybean_meal_solvent_48cp` | NRC16F134 | ING11 `soybean_meal_48` | 1:1 |
| `canola_meal` | `canola_meal_solvent` | NRC16F28 | ING12 `canola_meal` | 1:1；定义风险 P-1 |
| `cottonseed_meal` | `cottonseed_meal` | NRC16F58 | ING13 `cottonseed_meal` | 1:1；定义风险 P-15 |
| `ddgs` | `ddgs_low_fat`；`ddgs_high_fat` | NRC16F61；NRC16F59 | ING14 `ddgs` | **1:2 候选**，D-05 待定；必须显式选条目 |
| `cottonseed_whole_linted` | `cottonseed_whole_linted` | NRC16F56 | ING15 `whole_cottonseed` | 1:1；fNDF 归属 D-07 |
| `calcium_soaps` | `fat_calcium_soaps` | NRC16F25 | ING16 `fat_calcium_soap` | 1:1 |
| `limestone` | `limestone_ground` | NRC16F1013 | ING17 `limestone` | 1:1（核心表内）；候选 F1003 不在核心表，其 4 个键只在 B 层对账表（非盲第二读，P-6、§6） |
| `dicalcium_phosphate` | `dicalcium_phosphate` | NRC16F1007 | ING18 `dicalcium_phosphate` | 1:1 |
| `salt` | `salt_nacl` | NRC16F1017 | ING19 `salt` | 1:1 |
| `sodium_bicarbonate` | `sodium_bicarbonate` | NRC16F1033 | ING20 `sodium_bicarbonate` | 1:1 |
| `premix` | `premix_unspecified` | null | ING21 `premix` | 1:1；无 NASEM 条目（P-7） |
| `wheat_straw` | `wheat_straw` | NRC16F176 | ALT01 `wheat_straw` | 1:1（备选） |
| `corn_gluten_feed_dry` | `corn_gluten_feed_dry` | NRC16F40 | ALT02 `corn_gluten_feed_dry` | 1:1（备选） |
| `corn_grain_high_moisture` | `corn_grain_high_moisture_coarse`；`corn_grain_high_moisture_fine` | NRC16F1072；NRC16F45 | ALT03 `corn_grain_high_moisture` | **1:2 候选**（统计相同，P-4） |
| `wheat_middlings` | `wheat_middlings` | NRC16F173 | ALT04 `wheat_middlings` | 1:1（备选） |

核对（FIX_data，2026-09-24，脚本见 `reports/data_audit.md` §11）：核心表 25 个 `concept_id` 与 `configs/inventories.yaml` 的 25 个 `ingredient_id` 一一对应，两边都没有多余 id；每个概念 id 下的 NASEM 代码集合与 inventories 的 `nasem_feed_id`/`candidates` 一致，唯一差别是石粉候选 F1003（只在 A 录入）；条目 id 与某个概念 id 同名的 13 个，都只对应同名概念本身，**不再有一词两义**。

连接规则：

1. 配置层与核心表之间只用 `concept_id` 连接，再按用户选定的 `nasem_feed_id` 取条目；覆盖表 id 只用于追溯调研记录，不作连接键。
2. 概念 id 对应多个条目（`beet_pulp_dry`、`ddgs`、`corn_grain_high_moisture`）时，加载器必须要求显式给出 `nasem_feed_id`，否则报错；不得按“条目 id = 概念 id”静默连接。
3. 任何一层改名，都要同步本表与核心表生成脚本，并记入 DECISIONS。

---

## 4 枚举定义

### 4.1 `value_status`

| 值 | 含义 | 依据 |
|---|---|---|
| `mean_sd_n` | 印刷了 Mean、SD、N | Table 19-1 |
| `mean_only_single_source` | 只有 Mean，SD 与 N 空白 = 单一来源均值 | 表注 a；NASEM p.360 |
| `calculated_italic_no_sd_n` | 只有 Mean、且印刷为斜体（= 计算值），SD 与 N 空白。只出现在 B 层对账表（§6）的 RUP 与 DE base 行；不是单一来源实测均值 | Table 19-1 表注 c、i（pdf p.414：“Italics signifies calculated value as described in text”） |
| `blank_no_data` | Mean 空白 = 无数据 | 表注 a |
| `point_value_no_sd_n` | Table 19-3 的点值（元素百分含量或吸收系数），无 SD/N | Table 19-3 |
| `table_footnote_statement` | 由 Table 19-3 表注给出的值（表注 a：DM；表注 e：灰分），不是表行值 | Table 19-3 表注（pdf p.433） |
| `not_listed_below_1pct` | Table 19-3 对该原料未列此元素；表注 b 规定 <1% 不列出 → 未列 ≠ 0，数值未知 | Table 19-3 表注 b |
| `missing_not_in_table` | 该原料不在 NASEM 表中（预混料） | 逐条目检查 |

### 4.2 `statistic_type`

| 值 | 含义 |
|---|---|
| `summary_mean_sd_n` | 汇总统计：均值 + SD + 样本数（不是单样本值，也不是批次值） |
| `summary_mean_only` | 只有均值的汇总/单一来源值，无离散度信息 |
| `no_data` | 无数值 |
| `point_value` | 参考点值（Table 19-3 元素含量） |
| `footnote_point_value` | 表注给出的点值 |
| `model_coefficient_point_value` | 模型系数点值（吸收系数） |
| `not_listed` | 表中未列出 |
| `not_in_source` | 来源中没有该原料 |

### 4.3 `is_measured_or_derived`

| 值 | 用于 | 依据 |
|---|---|---|
| `measured` | 有 Mean/SD/N 的 Table 19-1 单元（230 行）：4 家商业实验室的分析值汇总。注意其中湿化学与 NIRS 结果混合、无法区分（NIRS 本身是定标预测值） | NASEM p.360–361 |
| `equation` | 3 个只有均值的 TFA 单元（F8、F28、F61）：数值与同条目 Crude fat 按 “TFA = CF − 1” 逐位吻合，**判为方程估算值属【推断】**，表内并未逐格标注。按 p.362，CF − 1 是一般规则，粗饲料改用 TFA = CF × 0.5678；生成脚本按条目是否为粗饲料只检验适用的那一条（本版粗饲料条目的 TFA 都有 N/SD，0 行命中 0.5678 规则）。F8 的 TFA 在印刷精度下同时满足两条规则，大麦不是粗饲料，按 CF − 1 记 | NASEM p.362 |
| `unknown` | 其余只有均值的单元（其他 TFA、钙皂各值）；Table 19-3 全部数值（来源未说明） | NASEM p.360、p.362 |
| `not_applicable` | 无数值的行（空白、未列、预混料）；加载器必须跳过 | — |

### 4.4 `analytical_method`

| 值 | 用于 | 依据 |
|---|---|---|
| `commercial_lab_wet_chemistry_or_NIRS_indistinguishable` | 有 SD 的 DM、ash、CP、ADF、NDF、starch、TFA、EE | NASEM p.360–361 |
| `commercial_lab_wet_chemistry_or_NIRS_indistinguishable;mostly_sulfuric_acid_ADL` | 有 SD 的 lignin | NASEM p.362 |
| `commercial_lab_wet_chemistry` | 有 SD 的 Ca、P | NASEM p.361：矿物数据只来自湿化学 |
| `estimated_from_crude_fat_CF_minus_1` | 上述 3 个推断为方程估算的 TFA | NASEM p.362（推断） |
| `estimated_TFA_CF_x_0.5678` | 粗饲料条目中只有均值、且与 TFA = CF × 0.5678（Daley et al. 2018）逐位吻合的 TFA。**本版 0 行**（生成脚本有此分支，v0 字典漏列） | NASEM p.362（pdf 382）原文：“For forages … TFA = CF × 0.5678” |
| `unstated_single_source` | 其余只有均值的单元 | NASEM p.360 |
| `unstated` | Table 19-3 数值 | 表题、表注均未说明 |
| `null` | 无数值的行 | — |

### 4.5 `lineage_id` 与 `provenance_cluster_id`

| `provenance_cluster_id` | 用于 | 依据与说明 |
|---|---|---|
| `US_commercial_labs_2011_2015` | 有 Mean/SD/N 的 Table 19-1 单元（230 行） | NASEM p.360：数据来自 CVAS、Rock River、Dairyland、Dairy One 四家商业实验室，2015 年春索取各家 5 年数据；年份窗 2011–2015 取自 Tran et al. 2020（调研 S1§2.6 转引，本任务未复核原文）。等同调研 §3.1 的“谱系 C”：NANP 在线库、NASEM Dairy-8 软件库、`nasem_dairy` 饲料库、Dairy One 等上游实验室库都属这一簇，**不能互作独立验证** |
| `NASEM2021_single_source_unspecified` | 只有均值的 Table 19-1 单元（16 行） | NASEM p.360：无 N/SD 的均值来自其他来源（文献、NASEM 2016 肉牛表、大学未发表数据）；p.362：缺 TFA 实测时通常由 CF 估算。**不归入上面的商业实验室簇**（这是对任务给定常量的有意偏离，理由见 `audit/_parts/P3_log_additions.md` D-P3R-2） |
| `NASEM2021_T19_3_source_unstated` | Table 19-3 的数值行（24 行） | 表题与表注 a–f 都没有说明数据来源 |
| `null` | 无数值的行 | — |

`lineage_id` 是证据所在的表：`NASEM2021_T19_1` 或 `NASEM2021_T19_3`。Table 19-3 与 19-1 是不同的表、口径不同（点值、无 SD），所以不共用 `NASEM2021_T19_1`（同为 D-P3R-2）。

---

## 5 加载与使用规则（下游代码必须遵守）

1. `null` 不是 0。空白均值、未列元素、预混料都不得补 0；需要近似时，另建一条带 `research_scenario_assumption` 或 `pending_user_decision` 的记录并写理由（`datamodel.Provenance` 的规则）。
2. 只有 `statistic_type = summary_mean_sd_n` 的单元能提供离散度。`summary_mean_only` 与 `point_value` 单元只能作确定性输入或另建情景，不能给出 SD。
3. SD 是美国送检样本的群体离散度（口径见 `reports/data_audit.md` §7），不是某个牧场的批次间波动，也不等于检测误差；**不得把 SD/√n 当作新批次的成分波动**（合同 §7.4）。
4. 同一原料不同成分的 N 不同，说明它们来自不同的样本子集。表中没有相关系数；不能把不同成分的均值、SD 当作同一批样本的配对统计来构造协方差（合同 §7.2、验收 B07）。
5. 印刷字符串保留末位 0；转换成 float 时不要四舍五入。
6. `DM` 行是原物质基础，其余都是 DM 基础；`% → fraction` 除以 100。
7. 核心表与本字典之外的 NASEM 数值（IVNDFD48、RUP、DE base、微量元素等 576 个 A 单录入键）自 2026-09-24 起已有 G3 第二读与对账（§6；B-131 **建议关闭，待总负责人确认**——`BLOCKERS.md` 中仍为 open；FIX5 更正：原写“B-131 关闭”，子 agent 无权关闭 blocker），但**仍不在核心表**：进入任何计算前须 ① 只从 `nasem_t19_1_blayer_reconciled.csv` 读取、且只取 `entry_agreement ∈ {both_agree, resolved}`；② 由协议决定这些量的用途（B 层 NEL、RUP、微量元素约束目前都没有定义）；③ 斜体计算值（`calculated_italic_no_sd_n`）不能当实测值，也不能给出离散度；④ F1003 的 4 个键是非盲第二读，用于正式计算前宜再做一次盲读。
8. 版权：任何从核心表导出的派生参数文件都必须留在 `data/restricted_local/`，直到 P0-7 许可办完。
9. **印刷 N 不是独立同分布样本量。** N 是四家实验室的客户送检记录数，有同场多次送检（伪重复）、实验室聚类和 ±3.5 SD 截尾（NASEM p.360–361；`reports/data_audit.md` §7.2）。不得用 N 计算 SD/√n、σ̂²/n 或按 N 的区间来表示均值或 SD 的参数不确定性；参数不确定性只能用有依据的有效样本量（如按实验室或来源簇），或只作假设范围（合同 T8.3 的“假设范围”层）。
10. **加载器的最低校验**：只接受 `entry_agreement ∈ {both_agree, resolved}` 的行；需要离散度时只接受 `statistic_type = summary_mean_sd_n`；随机原料的某个约束成分 `sd` 为 `null`（`summary_mean_only`、`point_value` 等）时必须显式报错或走预先登记的点值策略，不得让 NaN 静默进入抽样（红队 D 对 `experiments/E0_verification/run_smoke_nasem_dryrun.py` 的发现；该脚本不在本字典负责范围）。
11. **原料连接**：按 §3.2 用 `concept_id` 连接配置层；一个概念对应多个条目时必须显式选定 `nasem_feed_id`。成分统计相同的加工变体（F8/F1074、F44/F1070/F1071、F1072/F45）是同一组参数，不能当作不同群体，也不能当作相互独立的证据。

---

## 6 B 层对账表与第二读（v0.2 新增，G3_data_assays，2026-09-24；B-131）

两份文件都只在 `data/restricted_local/`（`.gitignore` 第 1 行，`git check-ignore -v` 已核实），含 NASEM 数值，不进 git、不进交付 zip。生成与核对过程见 `reports/data_audit.md` §12 与附录 F。

| 文件 | 行数 × 列数 | sha256 | 内容 |
|---|---|---|---|
| `nasem_t19_1_entry_A2_second_read.csv` | 576 × 21 | `bec95b64394cb5b69f3881c04403d300f527da0e2eb57d65fd44e442700e0b10` | G3 的独立第二读（A2）：A 单录入的全部 576 个键，每键一行 |
| `nasem_t19_1_blayer_reconciled.csv` | 576 × 28 | `3291e8da4fbe63c01e0b50481c384c40bbf04548c52d984b92031c9e511af6e8` | A 与 A2 对账后的 B 层表；数值只在 A、A2 一致或已看图裁决时写入 |

### 6.1 `nasem_t19_1_entry_A2_second_read.csv` 的字段

| 列 | 含义 |
|---|---|
| `entry_id` | `A2_<T19-1 或 T19-3>_<NASEM 代码>_<nutrient_id>` |
| `table_id`、`nasem_feed_id`、`nasem_feed_name` | 同核心表；条目名按切图表头抄录 |
| `nutrient_code_A` | A 录入的 `nutrient_code`（对齐键；A 的 Table 19-3 吸收系数列在此拆为 `Ca_absorption_coefficient`） |
| `nutrient_id` | 内部成分 id：核心 11 项同 §2；B 层 22 项为 `A_fraction`、`B_fraction`、`C_fraction`、`Kd_B`、`RUP`、`dRUP`、`soluble_protein`、`ADIP`、`NDIP`、`IVNDFD48`、`WSC`、`DE_base`、`Mg`、`K`、`Na`、`Cl`、`S`、`Cu`、`Fe`、`Mn`、`Zn`、`Mo` |
| `nasem_row_label` | 印刷行标签（带脚注字母说明，如 “RUP, % CP (footnote c)”） |
| `unit`、`basis` | 由印刷行标签得出：`%`/`DM`（% DM）、`%`/`as_fed`（DM）、`% of CP`/`fraction_of_CP`（A、B、C 组分、RUP、可溶蛋白）、`%/h`/`not_applicable_rate`（Kd of B）、`% of RUP`/`fraction_of_RUP`（dRUP）、`% of NDF`/`fraction_of_NDF`（IVNDFD48）、`Mcal/kg`/`DM`（DE base）、`mg/kg`/`DM`（Cu、Fe、Mn、Zn、Mo）、`fraction`/`none`（吸收系数）。与 A 的写法逐键比较，0 处不同 |
| `mean`、`sd`、`n_raw_printed` | 印刷字符串（保留末位 0 与千分位逗号或原文缺逗号的写法）；空白写 `null` |
| `n` | `n_raw_printed` 去逗号后的整数；空白为 `null` |
| `calculated_italic` | 数值印为斜体（= 计算值，表注 c、i）为 `true`；空白格一律 `false` |
| `value_status` | §4.1 的枚举；本文件用到 `mean_sd_n`、`mean_only_single_source`、`calculated_italic_no_sd_n`（新增）、`blank_no_data`、`point_value_no_sd_n`、`table_footnote_statement` |
| `pdf_page`、`printed_page` | 该格所在页；`printed_page = pdf_page − 20` |
| `method` | 读数方法（PyMuPDF 渲染 300 dpi“行标签列 + 该原料三列”切图、目视读数；PyMuPDF 词坐标文字层自查） |
| `textlayer_selfcheck` | Table 19-1 行为 `match`（572 行 × 3 格与文字层一致）；Table 19-3 的 4 行为 `not_run` |
| `read_blind_to_A` | `true`；F1003 的 4 行为 `false_A_row_label_seen_before_read`（G3 查看 A 的键列时，A 的 Table 19-3 `row_label_raw` 字段含数值，读图前已看到） |
| `image_file` | 读数所用切图（`_page_images/g3_*.png`） |

### 6.2 `nasem_t19_1_blayer_reconciled.csv` 的字段

| 列 | 含义 |
|---|---|
| `recon_id` | `<T19-1 或 T19-3>__<NASEM 代码>__<nutrient_id>`，唯一 |
| `layer` | `B_layer`（550 行：23 个核心条目 + F1070、F1071 的 22 项 B 层成分）/ `variant_check_core11`（22 行：F1070、F1071 的 11 项核心成分，只用于核对粉碎度变体，P-5）/ `T19-3_candidate_entry`（4 行：F1003，P-6） |
| `table_id`、`nasem_feed_id`、`nasem_feed_name`、`nutrient_id`、`nasem_row_label`、`unit`、`basis` | 同 §6.1 |
| `ingredient_id`、`concept_id` | 23 个核心条目取核心表的值（§3、§3.2）；F1070 → `corn_grain_dry_ground_fine` / `corn_grain_dry_medium`，F1071 → `corn_grain_dry_ground_coarse` / `corn_grain_dry_medium`，F1003 → `calcium_carbonate` / `limestone`（§3 预留 id） |
| `mean`、`sd`、`n`、`n_raw_printed`、`calculated_italic`、`value_status`、`pdf_page`、`printed_page` | 对账后的值；`entry_agreement = unresolved` 时数值一律写 `null`、`value_status = unresolved`（本版 0 行） |
| `entry_A_id`、`entry_A2_id` | 两份录入的行 id（A 的吸收系数行加后缀 `#absorption_coefficient`） |
| `entry_agreement` | `both_agree`（mean、sd、N 印刷字符串、斜体标记、页码逐字符一致）/ `resolved`（不一致，已看图裁决）/ `unresolved`。本版 576 行全部 `both_agree` |
| `fields_disagreeing`、`schema_label_difference` | 不一致的字段名；unit/basis 写法是否与 A 不同。本版全部 `none` |
| `resolution_note` | 一致性说明；非盲读的键另加说明 |
| `second_read_blind` | 第二读是否对 A 盲（F1003 的 4 行为 `false`） |
| `copyright_status` | `NASEM2021_all_rights_reserved__restricted_local_only` |
| `use_status` | `not_for_formal_use_until_protocol_decision`：已双读，但用途未由协议规定（§5 第 7 条） |

### 6.3 使用限制

1. B 层表与核心表分开存放，行集合不重叠（核心表 283 个键、B 层 576 个键，合起来是 A、B 两份录入的全部 859 个键）。加载器不得把两表拼成一张后绕过 §5 第 10 条的过滤规则。
2. F1070、F1071 的 11 项核心成分与 F44 相同（P-5），F1072 与 F45 除 DE base 外相同（P-4）；B 层表只证明“印的是什么”，不改变“加工变体共用一组参数”的规定（§5 第 11 条）。
3. 斜体的 RUP、DE base 是 NASEM 按正文方法计算的值（表注 c、i），不是实测；A、B、C 组分、Kd、dRUP 等只有均值的行来自单一来源（表注 a），没有离散度。
4. Table 19-1 的 SD 口径同核心表（§1 第 21 列、`reports/data_audit.md` §7），不得用 SD/√N（§5 第 9 条）。
5. 第二读仍是 软件辅助录入 读同一个 PDF，不是人类双录入，也发现不了原书排印错误（`reports/data_audit.md` §9、§12.3）。
