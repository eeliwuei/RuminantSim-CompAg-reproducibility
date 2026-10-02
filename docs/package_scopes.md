# 交付包范围：内部审计包与外部复现包

- 编写：K5_evidence 子任务（项目记录），2026-09-25，第二轮整改 R5 第 4 条（与 R7 第 3 条衔接）。
- **FIX_C 修订（2026-09-25，第二轮红队镜头 C）**：① 本文件的范围表改由受版本控制的 `scripts/package_release.py` 执行（显式清单 + 路径闸门 + 内容闸门，见 §5），取代会话 scratchpad 中未入库的 `pack_rrs.py`；② 补上第二轮新增的全部路径（`results/pilot/`、`reports/dev_case_v1_report.md`、`reports/smoke_current_report.md`、`docs/value_definition.md`、`configs/dev_case_v1/`、`sources/visual_spotcheck_images.csv` 等，§1）；③ 新规则：**含 DM 计价原料的配方的公开 q 与总成本视同受限输出**（它们与公开价格、计划 DM 等式一起可以一步线性求解出 NASEM DM 均值，红队发现 1），`results/pilot/` 与 `reports/dev_case_v1_report.md` 在 B-K7-3 定案前两种包都不收；④ 写明外部包的代码身份比对方法（§2.2）、干净解压的实测结果（§2.1）与封印的真实含义（§3）。修改明细见 `audit/_parts/round2/FIX_C_record.md`。
- **W3 修订（2026-09-25，第三轮 W3_package_scope）**：内部审计包（只限内部包）按显式清单收入开发运行 `pilot-20260925T055921Z-fb4f75af`（px 授权算力、冻结提交 46c22cb、`output_label = development`）的 7 个公开表与 15 个逐个判定过的主机日志，以及 W1 由该运行生成的第三轮报告；受限目录、其他 `results/pilot` 旧运行照旧两包都不收；外部包维持 BLOCKED。可识别性闸门新增三项检查（任意版式的 CSV q 表、q 不公开核对、余量一步还原受限界值），见 §6。明细见 `audit/_parts/round3/W3_package_scope_record.md`。
- **FIX_PKG 修订（2026-09-26，第三轮红队 D-1 / D-4 / D-7 / D-9）**：① 可识别性闸门的 q 表解析扩到 `.tsv` / `.psv` 文件与嵌在 `.md` / `.txt` / `.log` 中的表格文本、列名无 `q_` 前缀的宽表、带任一数值列的长表、首列为原料的行标签表、JSON 中以原料 id 为键的数值映射（W3 所说“任意版式”实际只覆盖 `q_` 前缀宽表与 `ingredient_id` + `q_as_fed…` 长表，红队实测 4 种版式漏过）；② 按项目既有规则（驱动 `public_redaction.restricted_only`、USER_ACTIONS c13、D-452），内部包中该运行 `reference_residuals.csv` 的浓度类余量在打包副本中扣留（`FIELD_REDACTIONS` 规则 `FIXPKG-RESID-CONC-MARGINS`，364 行），并新增检查“界值可读时余量能否一步得出受限浓度”；③ 新增第 9 道闸门“运行输出哈希”：内部包中的运行公开表逐个与 `run_record.json` 的 `output_hashes` 比对（运行记录本身与登记的 sha256 比对）；④ 打包描述更正（`ablation_config.json` 为 2 × 3 声明配置；`solve_status.json` 的对偶界字段本次全为 null；`dryrun.json` 的原因降为推断），并按逐文件判定收入总负责人补写的 `mac_bytes_replay.md`。见 §7 与 `audit/_parts/round3/FIX_PKG_record.md`。
- **Round3-fix 修订（2026-09-26，FIX_DOCS，第三轮红队 D-10、D-5）**：§6.1 “px 自建产物因 OpenBLAS 末位浮点差被冻结检查判为不同”改为推测（未逐字段核验），写明冻结检查只覆盖 `build_report.json`，引用总负责人补写的 `logs/px_runs/pilot-20260925T055921Z-fb4f75af/mac_bytes_replay.md`（B-445）；§5 中两处 “`CURRENT_STATE.json` `packaging.last_check`” 改指实际位置（第二轮状态页的只读副本 `audit/_history/CURRENT_STATE_round2.json`；第三轮的最近一次检查登记在 `CURRENT_STATE.json` `packaging`）。
- **FIX2 修订（2026-09-26，FIX_PKG 核验的 D-1 残留与 D-10）**：① 可识别性闸门的版式解析补三个漏报（核验在文档声称的覆盖范围内找到）：配方标识列不再只认固定键名（任何非数值、非原料 id 的首列或唯一文本列都可作配方键，如 `recipe`、`name`、`配方`）、解析前跳过 `#` 注释行、整份分隔表的分隔符用 `csv.Sniffer` 并逐一尝试 `,` `;` 制表符 `|`（非逗号分隔时另按小数逗号读）；② 新增第 10 道闸门“受限配方数值指纹”（`restricted_q_fingerprint`），与版式无关，只对“同一配方 ≥ 3 个投料量同窗口共现”生效；③ §7.1 写明三层防线的实际强弱（路径白名单是第一道防线，版式解析是启发式，数值指纹覆盖任意版式但单个数值泄漏不在其覆盖范围）。见 §7.1、§7.6 与 `audit/_parts/round3/FIX2_record.md`。
- **FINAL 修订（2026-09-26，总负责人收尾）**：① 第 10 道闸门修两处残留（第三轮红队 G10）：全精度打印按二进制精确值比较；非 UTF-8 文本（GBK、CP949、UTF-16/32、Latin-1）先解码再扫描（§7.6）；② B-450 处理：审计报告的名义预检 q 列移入受限目录（§7.6 末）；③ 主机日志新增 `authorisation_confirmation.md`（§6.2）。
- 依据：复核报告 `docs/review_20260924_round2/RuminantSim_Second_Review/01_复核报告.md` §6 末段、§7.2、§7.3；整改指令 R5、R7；`交付说明_20260924.md` ⑥（现行打包规则）；`sources/access_and_license_log.md` §2、§7；`BLOCKERS.md` B-105、B-139、B-147、B-150、B-310、B-318、B-320；K7 的 B-K7-3、D-K7-10。
- **本文件不是法律结论。** 哪些资料可以再发布、在什么许可下发布，由用户或其许可负责人决定。这里只按项目既有规则和来源登记的许可状态划定两种包的内容范围，并说明理由。遇到未决事项时一律按“不收”处理，并列入 §4 交用户决定。
- 两种包都**不再发布任何受限数据**：NASEM 2021 的数值、页面图与转录表，CVB、AFZ、Dairy One 数值，会话缓存及其持久副本中的期刊全文，以及能一步反推这些数值的派生输出。受限数据留在合法持有者本机的 `data/restricted_local/`；需要核对时，由持有者运行 §2 的脚本，交出只含哈希、计数与判定的回执。

---

## 1 两种包的用途与内容

- **外部复现包**：给期刊补充材料、公开仓或任何用户以外的读者。目标是让没有受限资料的人也能运行代码、测试和合成示例，并知道受限参数从哪里、按什么步骤合法取得和核对。
- **内部审计包**：给用户本人及用户指定的复核人。在外部复现包之上，增加过程记录、决策日志、阻断清单、回执和审计报告，便于复核“做了什么、证据在哪里”。受限数据仍不随包；复核人需要核对受限数据时，在持有者的环境运行脚本，或核对持有者交出的回执。
- 另有的“CYRP 继承总档（内部）”包含服务地址、服务器路径等运维信息，继续单独隔离，不属于这两种包，也不作论文补充材料（复核报告 §7.3；B-140）。

下表是规则；**可执行版本在 `scripts/package_release.py` 的 `SCOPE`、`EXCLUDE`、`FORBIDDEN`、`REDACTIONS`、`EXTERNAL_PENDING` 中**，两者不一致时以脚本为准并应同时修正本表。

| 内容 | 内部审计包 | 外部复现包 | 理由 |
|---|---|---|---|
| `src/`、`tests/`、`experiments/`、`scripts/`（不含 `__pycache__`、`*.pyc`） | 收 | 收 | 项目自写代码；代码许可另定（§4）。外部包不改写代码文件（本机路径须在源文件里改掉，§5 闸门 5） |
| `pytest.ini`、`requirements-lock.txt`、`environment.lock.json` | 收 | 收 | 复现环境（R6）；PyMuPDF 是锁文件中的可选组 `holder_verification`，只供持有者核对脚本 |
| `data/synthetic_test_only/` | 收 | 收 | 合成示例，显式标为合成 |
| `data/data_dictionary.md`、`data/processed/phase3_manifest.json` | 收 | 收 | 只有字段说明、文件名、哈希、行数、列名与计数（⑥ 已确认） |
| `sources/source_registry.csv`、`sources/access_and_license_log.md`、`sources/error_source_locators.csv`、`sources/visual_spotcheck_images.csv` | 收 | 收（本机用户路径在打包时改写为 `~/`、`-Users-<user>-`） | 只有定位、许可状态与哈希；定位表不含参数值，会话路径已改为 `<local_session>/…`（FIX_C） |
| `reports/restricted_inputs_receipt.json`、`reports/evidence_closure.md`、`reports/data_audit.md` | 收 | 收 | 回执与核验表不含数值；外部读者可用 `--check-seal` 复算封印（只能发现意外损坏，§3） |
| `reports/` 其他文件（阶段报告、引擎测试日志、`minimum_case_nutrition_audit.md`、`smoke_current_report.md`、`uncertainty_factory_audit.json`、`value_definition_counterexample_results.json` 等） | 收（须通过 §5 内容闸门） | 不收（未列入外部范围即不收） | 过程记录；外部只需复现所需内容 |
| `docs/ENGINE_API.md`、`docs/INFORMATION_VALUE_DESIGN.md`、`docs/decision_timing_and_units.md`、`docs/value_definition.md`、本文件 | 收 | 收 | 接口与设计说明对复现有用 |
| `docs/EXPERT_DIRECTION_20260924.md` | 收 | 不收 | 内部方向说明 |
| `configs/*.yaml` | 收；按 ⑥ 删去两个原料级吸收系数（`inventories.yaml`） | **暂缓（BLOCKED）**：`assays.yaml` 的期刊表格数值、`uncertainty.yaml` 的 Yoder 2014 相关系数（B-310），`constraints.yaml`、`animal_profile.yaml`、`inventories.yaml` 中的 NASEM 表值与表注陈述（B-318）须先定处理方式（null + 定位，或取得许可） | 再发布许可未定 |
| `configs/dev_case_v1/`（第二轮新增） | 收 | **暂缓（BLOCKED）**：含 NASEM 派生陈述（如 `prices.yaml` 注记中矿物原料的决策时 DM 估计）与约束界值，按 B-318 / B-K7-3 处理 | 同上 |
| `results/pilot/`（第二轮新增） | **不收**（W3 明列的一个开发运行的公开表除外，见下一行） | 不收 | **FIX_C 发现 1**：公开 q、公开总成本、公开价格与计划 DM 等式联立，一步线性求解即得 NASEM Table 19-1/19-3 的原料 DM 均值（本机只读复算：5 条配方、秩 7、5 种原料 DM 精确解出，相对误差 ≤ 1e-11）。B-K7-3 定案（把成本与 q 表移入受限目录，或只给不可反解的汇总）之前两包都不收；第二轮及更早的运行（`pilot-20260924T…`）公开目录仍带逐配方 q，照旧不收 |
| `results/pilot/pilot-20260925T055921Z-fb4f75af/` 的 `NOT_FOR_MANUSCRIPT.md`、`ablation_config.json`、`comparable_sets.json`、`endpoint_ablation.csv`、`reference_residuals.csv`、`run_record.json`、`solve_status.json`（W3） | **收**（逐文件显式清单 `INTERNAL_RUN_PUBLIC_FILES`；目录里以后多出的文件不收，清单报告为“未判定”） | 不收 | 该运行的驱动只把 q 写入受限目录，公开表里配方只以 `q_hash`（sha256）出现；成本在 `endpoint_ablation.csv`。可识别性闸门核对：q 不在包内任何文件、`q_hash` 是摘要、包内 JSON 无 q/d̂/占比键值；持有方对照：把受限 `rations_q.csv` 当作包内文件读入，经包内公开成本与公开计划 DMI 必须解出全部 8 种原料 DM（证明闸门看得见这条途径，若 q 被公开会判 FAIL）。`reference_residuals.csv` 的余量见 §6.3；**FIX_PKG**：打包副本扣留 PN-CP-SUP、PN-T1–T5、DIAG-T51-DGC-STARCH-SHARE 的余量与亏缺字段（§7.2，运行产物不改）；7 个文件的源字节须与 `run_record.json` 的 `output_hashes` 一致（§7.3）。开发材料，不得进入正式结果或稿件 |
| `logs/px_runs/pilot-20260925T055921Z-fb4f75af/`（W3） | **逐文件判定后收**（`INTERNAL_RUN_LOG_FILES`，15 个，§6.2） | 不收 | 运行前检查、构建摘要、测试、干跑与完整运行日志；无受限值、无可一步还原者。含主机目录路径（`/home/<用户>/rrs_runs`），无地址、无凭据 |
| `reports/round3_ablation_results.md`、`reports/round3_ablation_summary.csv`、`reports/*_pilot-20260925T055921Z-fb4f75af.csv`（W1 生成，W3 规则） | 收（随 `reports/` 收入，须通过全部闸门；文件不存在时清单记“缺”） | 不收 | 第三轮开发结果的汇总；是否可入稿由总负责人定，本包只作复核 |
| `data/restricted_local/pilot/pilot-20260925T055921Z-fb4f75af/`（`rations_q.csv`、`reference_residuals_full.csv`） | 不收 | 不收 | 逐配方 q（加公开 DMI 或成本即解出原料 DM）与 PN-CP-HI 余量（与 PN-CP-SUP 余量配对即还原 Table 21-3 的 CP 下限） |
| `reports/dev_case_v1_report.md`（第二轮新增） | **不收** | 不收 | 同上：§5 的 q 表（3 位小数）与成本行（4 位小数）加公开价格即可把玉米与玉米青贮的 DM 解到 NASEM 印刷精度（1 位小数）。报告改写（删成本行或只给不可反解的汇总）之前不收；改写后须重新通过 §5 的可识别性闸门 |
| `reports/model_audit.md` | **暂不收** | 不收 | 按类别列出 NASEM 吸收系数、完整 Table 5-1（B-318）。⑥ 规定决定前不进 zip；上一版研究 ZIP 实际收入了它（复核报告 §7.3），是 scratchpad 打包脚本未执行排除规则所致（FIX_C 发现 2），现由 `package_release.py` 的路径闸门强制 |
| `reports/smoke_report.md`、`results/smoke/` | 不收 | 不收 | 配方占比与营养浓度可反推 NASEM 数值（B-147）；`results/smoke/smoke-20260924T085518Z-185279b9/`、`…085519Z-06864c0b/` 的 `rations.csv` 仍带计划 DM 占比且已被 git 跟踪（一步反推，见 FIX_C 记录），须按 D-226 移入受限目录，git 历史中的副本在任何推送前清理（B-150 同类） |
| `reports/packaging_record_*.md` | 不收 | 不收 | 外层 ZIP 回执放在 ZIP 外（复核报告 §7.2） |
| 共享日志与交接（`AGENT_HANDOFF.md`、`AGENT_PROGRESS.md`、`DECISIONS.md`、`BLOCKERS.md`、`CHANGELOG.md`、`USER_ACTIONS_*.md`、交付说明、`CURRENT_STATE.json`、`correction_report.md`、`changed_files_manifest.json`）、`audit/`（不含 `audit/phase1_survey_20260924/`） | 收；按 ⑥ 的同一规则删去三份审计记录中同样的两个原料级吸收系数（FIX_C） | 不收（`CURRENT_STATE.json` 与 `README.md` 除外，作为状态页） | 过程记录供复核；对外只需可复现所需内容 |
| `docs/contract/`、`docs/review_20260924_round2/` | 收 | 不收（合同原文与复核记录是否外发由用户定，默认不收） | 外部包的测试不再依赖合同文件（§2.1） |
| `data/restricted_local/`（PDF、转录表、核心表、B 层表、AC 表、页面图、阶段 3 产物、smoke/pilot 受限输出、dev_case_v1 构建产物、调研原文、盲读读数、期刊页面持久副本 `_journal_cache/`、打包禁止清单 `_package_denylist.json`） | 不收 | 不收 | NASEM All rights reserved；CVB 条款禁止未经许可转成机器可读；Dairy One 与期刊全文再发布权未核 |
| `audit/phase1_survey_20260924`（指向受限目录的符号链接） | 不收，打包不跟随链接 | 不收 | 同上（B-150） |
| 本地会话缓存（期刊全文页面、WebFetch 缓存的 PDF） | 不收 | 不收 | 第三方受版权文字，只作本地核验证据（`access_and_license_log.md` §7） |
| `logs/legacy_exec/` | 只收 ⑥ 白名单（各子目录顶层 `RUN_LOG.md` 与文件名含 sha256 的 `.txt`） | 不收 | 未公开源码、生产代码副本与反汇编（B-007a） |
| `legacy/`、`.git/`、`docs/inheritance_20260924/` | 不收 | 不收 | 用户已持有；git 历史含受限笔记（B-150）；运维信息（B-140） |

## 2 给合法持有者的核对与提取脚本

持有者在项目根、`data/restricted_local/` 就位时运行。所有脚本只读受限输入，受限输出只写回 `data/restricted_local/`。

| 脚本 | 作用 | 受限输入 | 输出 |
|---|---|---|---|
| `scripts/verify_restricted_inputs.py` | A 阶段 3 清单全部输入与 17 个输出的哈希与行数，并按完整代码清单判断产物是否由当前代码生成（`code_currency`）；B 受限源文件哈希；C 配置指针与页码定位；D 转录一致；E PDF 文字层全覆盖定位（含输入端负对照）；F 看图盲读比对（图像哈希对登记值、内存重渲染逐字节复现）；G 误差参数定位表与期刊页面持久副本；I dev_case_v1 构建输入与每条运行记录的受限输入（完整性）及其代码/配置是否仍当前（时效）；`--rebuild-check` 在受限临时目录重建阶段 3 与 dev_case_v1 产物并比对 | PDF、录入 A/B/A2、核心表、B 层表、AC 表、阶段 3 产物、dev_case_v1 构建输入与产物、期刊页面持久副本；可选会话缓存 | `reports/restricted_inputs_receipt.json`（无数值，带封印）；`--render-visual` 时切图与只有键的模板写入受限目录；`--persist-session-caches` 写 `_journal_cache/` |
| `scripts/package_release.py` | 按本文件生成两种包的显式清单并执行全部打包闸门；闸门全部 PASS 时 `--build` 才压缩并写 ZIP 外回执 | 核心表、B 层表、AC 表、dev_case_v1 参数表、受限问题 YAML（只作阳性对照与比对，不输出数值）、哈希化禁止清单 | 输出目录（必须在仓库外）中的 `package_manifest_<包>.json`、ZIP 与回执 |
| `scripts/check_environment.py` | 核对解释器与锁文件（`--require-group holder_verification` 另核 PyMuPDF）；`--compare-run-record` 逐文件比较当前代码与运行记录 | 无 | 终端输出 / `--json-out` |
| `scripts/build_phase3_tables.py` | 由核心表重建阶段 3 清洗表与诊断；自 FIX_C 起清单记录完整代码清单与构建脚本 sha256 | 核心表 | 受限目录的 `processed/`；被跟踪的 `data/processed/phase3_manifest.json` |
| `data/restricted_local/dev_case_v1/build_dev_case_v1.py`（受限目录内，不在代码清单中，B-K4-3 / B-K7-5） | 由配置与受限表构建 dev_case_v1 问题 | 核心表、B 层表、饲料库摘录、受限值 | 受限目录 |
| `reports/data_audit.md` 附录 B–G | 由两份录入重建核心表、看图裁决、B 层第二读对账、误差参数登记的逐格核对 | 录入 A/B/A2、PDF、期刊页面副本 | 核心表、B 层表（受限） |
| `experiments/E0_verification/run_smoke_current.py`、`run_dev_case_v1.py` | 当前代码 smoke 与 dev_case_v1 开发运行 | 核心表、dev_case_v1 构建产物 | `results/<类型>/<run_id>/`（运行记录与脱敏输出）与受限目录中的逐配方输出 |
| `python -m pytest`（`tests/`） | 工程自检 | 无（需要受限资料的少数测试缺资料时跳过并说明原因） | 测试日志 |

### 2.1 外部包干净解压的实测（FIX_C，2026-09-24T20:59Z，同一台机器的隔离目录，不是第三方复现）

- 按 `package_release.py` 的外部清单（136 个文件，已做路径改写，未做尚待决定的配置脱敏）解压到会话 scratchpad 的空目录，无 `.git`、无 `data/restricted_local/`、无 `docs/contract/`：`pytest -q` → **756 passed，0 failed，3 skipped**。3 个跳过均写明原因：两个需要受限表的真实数据测试；`test_run_record_and_hashing.py` 中“冻结副本与合同原文一致”的核对（合同不在包内；字段检查本身改用冻结副本 `RUN_RECORD_REQUIRED_FROZEN` 照常执行，合同在场时两者必须逐项相等）。红队此前在同样条件下得到 1 failed，原因即该测试直接读合同文件。
- `experiments/E0_verification/run_dev_case_v1.py --dry-run` 在缺受限输入时仍直接抛出 `FileNotFoundError`（退出码 1），**不是**约定的 BLOCKED 信息。该驱动不在 FIX_C 的修改范围，已提交修改请求（`audit/_parts/round2/FIX_C_log_additions.md`）；在修好之前，外部读者应把退出码 1 + `data/restricted_local/...` 缺失的报错理解为访问范围所致。
- 需要真实参数的核对（`verify_restricted_inputs.py`）在缺受限资料时各节报 `BLOCKED`，不是 FAIL。

### 2.2 外部包的代码身份怎样与内部运行记录比对

外部包按规定要改动 `configs/`（期刊值与 NASEM 值置空、原料级吸收系数删去），而 `configs/` 属于代码清单（R6），所以**外部包的 `code_manifest_sha256` 永远不会等于任何内部运行记录的值**，这不是篡改。比对时只比较代码子集：

```bash
python scripts/check_environment.py --compare-run-record <内部运行目录>/run_record.json
```

输出按区域（src、experiments、scripts、tests、configs、根文件）列出差异，并给出双方 `src experiments scripts tests` 子集的摘要 `code_subset_sha256`（`ration_reliability.io.run_record.code_manifest_subset_sha256`）。代码子集相同、只有 configs 不同，说明外部包的代码与内部树一致。2026-09-24T20:59Z 实测：解压后的外部包与当前工作树的代码子集摘要相同（`e934e235…68e9`）；两者与 dev_case_v1 开发运行 `pilot-20260924T192615Z-1fc52728` 记录的子集（`b645043e…0388`）不同，因为该运行之后第二轮整改又改了代码——那条运行的证据时效为 stale（回执 `evidence_currency`）。

## 3 回执的交换方式与封印的含义

1. 持有者运行 `verify_restricted_inputs.py --holder "<姓名或角色>"`（可加 `--hmac-key-file` 用自己保管的密钥签 HMAC），得到回执。
2. 回执可放入两种包。`--check-seal` 复算的是**不带密钥的 sha256**：它只能发现回执在传递中意外损坏；任何人改完内容都能重新计算它，所以**不能证明回执没有被篡改，也不能证明是谁运行的**（FIX_C 更正：K5 原文“外部读者可以用 `--check-seal` 复算封印”容易被读成防篡改）。需要防篡改时，只能依靠持有者的 HMAC（只有掌握密钥的一方能验证）或外部签名；复核人在持有者处用同一密钥复核 HMAC。
3. 回执中的哈希把判定绑定到具体文件：复核人若另行合法取得同一 PDF 与转录表，可重跑脚本并比较哈希与判定。
4. 回执只证明所列文件、转录与定位的一致性，不证明原书无误，也不构成外部验证（`reports/evidence_closure.md` §4–§5）。回执中的 `evidence_currency` 另报时效：阶段 3 产物与各运行是否仍由当前代码与配置生成，“stale / not_established”表示证据描述的是更早的代码或配置，不表示文件被改动。

## 4 交用户决定（不是法律结论）

| 事项 | 影响的包内容 | 登记 |
|---|---|---|
| dev_case_v1 的公开 q 与总成本如何处理（移入受限目录，或只给不可反解的汇总） | `results/pilot/`、`reports/dev_case_v1_report.md`（决定前两包都不收） | B-K7-3、FIX_C 发现 1 |
| NASEM 摘录与派生参数能否作带页码引用进入补充材料 | `reports/model_audit.md`、配置与测试中的少量 NASEM 表值、`configs/dev_case_v1/` | B-318（USER_ACTIONS W5-1）、B-105（P0-7） |
| 期刊表格数值的再发布 | `configs/assays.yaml`、`configs/uncertainty.yaml` 对外版本 | B-310 |
| git 历史中的受限笔记与已跟踪的旧 smoke `rations.csv` | 任何推送或公开仓之前 | B-150（C9）、B-147 |
| 会话缓存与其受限持久副本的保存与删除；是否调整本地会话保留设置（用户配置，本项目不改） | 240 条误差参数的复核依据 | B-139、B-320 |
| 项目代码的许可证 | 外部复现包 | 未登记，发布前须定 |
| 合同原文与复核记录是否外发 | `docs/contract/`、`docs/review_20260924_round2/` | 默认不外发 |
| 开发运行 `pilot-20260925T055921Z-fb4f75af` 的公开表与主机日志进入内部审计包（W3 按本表规则执行，§6） | 内部包 22 个文件；外部包不收 | W3 记录；总负责人确认后才打包外发给复核方 |
| ~~内部包内 Table 5-1 界值（`configs/constraints.yaml`，B-318）与该运行 PN-T1–T5 的公开余量同处一包：读者可由余量 + 界值得到配方层面的 NDF / fNDF / 淀粉浓度分位数。按 D-K7-10 / D-452 的理由（“余量 + 浓度一步即得 Table 5-1/21-3 值”），这些浓度要保护的正是界值，而界值已在内部包内，故闸门不判为新增暴露；若总负责人认为 q 不公开时配方浓度本身也须受限，可在 `FIELD_REDACTIONS` 加一条打包期脱敏规则（运行产物不改）~~ **FIX_PKG 更正（红队 D-4）**：项目既有规则（驱动 `public_redaction.restricted_only` 含 CP / NDF / 淀粉 / 粗饲料 NDF 浓度；USER_ACTIONS c13；D-452）本身就把浓度列为受限，不需要另行决定；已在打包副本中扣留（§7.2），闸门对遗留的浓度类余量判 FAIL。本行不再待决 | 内部包 `reference_residuals.csv` | W3 记录 §4；FIX_PKG 记录 |
| W1 报告正文写明 M0 名义预检中 PN-T4、PN-T5、PN-CP-SUP 等行“名义余量为 0”（引用受限构建报告 `nominal_margins`）：按上行同一理由不还原新的受限值（CP 下限仍需 CP 浓度，后者在包内不可得），但属于引用受限产物的文字，是否保留由总负责人定 | `reports/round3_ablation_results.md` | W3 记录 §4 |

## 5 打包程序与闸门（R7 执行）

```bash
PY=/opt/homebrew/opt/python@3.11/bin/python3.11
# 只检查（两种包的显式清单 + 全部闸门），输出目录必须在仓库外
PYTHONDONTWRITEBYTECODE=1 $PY scripts/package_release.py --out-dir <仓库外目录>
# 某个包的闸门全部 PASS 后才打包：ZIP 内只放成员清单，整个 ZIP 的 sha256 写在 ZIP 外的 .receipt.json
PYTHONDONTWRITEBYTECODE=1 $PY scripts/package_release.py --out-dir <仓库外目录> --package internal --build
```

退出码：0 全部 PASS；1 有闸门 FAIL；2 BLOCKED（待决事项，或缺受限参照数据以致无法扫描——扫描不了不算通过）；3 参数错误。闸门（每个内容闸门先跑阳性对照，对照不触发则该闸门判 FAIL）：

1. **路径**：在最终清单上独立检查禁止路径（受限数据、`results/smoke`、`results/pilot`、`model_audit.md`、`dev_case_v1_report.md`、`smoke_report.md`、`phase1_survey`、`inheritance`、`legacy/`、`.git/`、缓存；外部包另禁 `docs/contract`、复核记录、`audit/`、共享日志）；清单中不得有符号链接；`logs/legacy_exec/` 只许白名单。**W3**：只在内部包、只对 `INTERNAL_RUN_PUBLIC_FILES` 中的精确路径豁免 `results/pilot/*` 这一条规则（豁免逐条记入 `waived_internal_run_files`）；`logs/` 下另许 `INTERNAL_RUN_LOG_FILES` 中的精确路径；这些文件必须是内容闸门读得到的文本（`unscannable_allowed_run_files` 非空即 FAIL；`.log`、`.sh` 自 W3 起计为文本，此前这两类文件不经任何内容闸门）。
2. **⑥ 改写**：每条改写（两个原料级吸收系数在 `inventories.yaml` 与三份审计记录中的全部出现、两份报告中三处原料级矿物值）必须恰好命中预期次数，否则判 FAIL（规则过时）。改写用上下文正则，脚本本身不含这些数值。
3. **密钥与服务器地址**：通用模式 + 受限目录中的哈希化禁止清单（`data/restricted_local/_package_denylist.json`，只存 sha256，项目中不存任何明文凭据）；清单缺失判 BLOCKED。
4. **NASEM 指纹**：核心表与 B 层表的 mean/SD/N 同行三元组；核心表、B 层、dev_case_v1 参数表与不确定性格的 mean+SD 同行（≥ 3 位有效数字，含 ÷100）。
5. **本机用户路径**（外部包）：非代码文本中的 `/Users/<名>/`、`-Users-<名>-` 改写后不得残留；代码文件不改写，出现即 FAIL，须在源文件里改。
6. **吸收系数**：受限 AC 字符串出现在谈吸收系数的行上即 FAIL；内部包只接受 ⑥ 已放行、并逐文件限定行数的三处（B-318 类的文字级/类别级值）。
7. **可识别性**（FIX_C 发现 1）：对包内每个配方表（`rations_q.csv` 目录、带计划 DM 占比的 CSV、Markdown 中以原料为行的 q 表），把公开 q、DM 计价原料的总成本（用运行所用价格）、计划 DM 等式或计划 DM 占比列成线性方程组求原料 DM；任何受限 DM 值被解到印刷精度（±0.05 个百分点）即 FAIL；找不到对应受限参照时，只要有可识别的 DM 坐标也判 FAIL——**CLOSE2 起，这一无真值分支用读者手里有的方程：公开 DMI / 计划 DM 占比，以及 `configs/` 下每个公开价格情景（现为 `configs/dev_case_v1/prices.yaml`）写出的成本方程**（此前该分支不用任何价格，成本行被整体丢掉，“q + 成本”途径对它不可见）；有成本行而没有任何公开价格情景覆盖其原料时判 `cost_route_unverifiable`（FAIL，不算通过）。阳性对照：由受限问题合成的 5 条配方必须解出；**CLOSE1（B-433）起，真实文件对照改为冻结副本**：`data/restricted_local/package_gate_controls/`（受限、git 忽略、从不打包）中每条解析路径各一个对照——开发运行公开目录（`rations_q.csv` + `method_summary.csv` + `nutrient_profile_public.csv`，`pilot-20260924T205621Z-93d8654c`）、带计划 DM 占比的 smoke `rations.csv` + `nutrient_profile.csv`（`smoke-20260924T190623Z-ece10eb7`）、按报告印刷精度（q 3 位、成本 4 位小数）生成的 Markdown q 表；真值是各运行受限问题 YAML 的缩减副本（只有 DM 估计与价格）；`controls.json` 登记每个文件的 sha256 与冻结时解出的原料数。对照文件缺失、被改动、少于冻结时的解出数或不再触发，都判**仪器失效**（`instrument_status = instrument_failure`，闸门 FAIL，`files_failing` 不受影响，不归咎于包内文件）；缺 `controls.json` 判 BLOCKED（`instrument_unavailable`，扫描不了不算通过）；每条解析路径都必须有对照。冻结：`scripts/package_release.py --freeze-controls --control-run-dir … --control-share-csv … --control-share-profile …`（全部对照触发才写清单；已有清单须 `--force`）。回归测试：`tests/unit/test_package_release.py` 第 5 节。**CLOSE2（收尾红队）补充**：CLOSE1 的三个冻结对照都通过注入真值（与 Markdown 的上下文目录）运行，绕过了真实包内文件要走的两步查找（按 run id 找受限真值、Markdown 按 `results/*/<run id>` 找公开 DMI），也没有覆盖无真值分支——所以当时“仪器 ok”只证明解析器与求解器正常，**不证明真实文件的判定链路正常**（B-433 的 CLOSE1 关闭证据据此补正）。现在每个对照另做默认路径子检查（与包内文件同一调用，不注入任何参数）：`default_lookup`（三个对照：由 run id 查到受限真值并解出不少于冻结时的原料数）；`no_run_id`（Markdown 对照：删去含 run id 的行、放在不含 run id 的包路径形态下，必须经无真值分支判 FAIL，且理由与可识别坐标数不低于冻结时）；冻结时把子检查结果记入 `default_path_at_freeze`。对照清单 `controls.json` 的 sha256 与逐对照阈值登记在被跟踪的 `docs/package_gate_controls.pin.json`（`--freeze-controls` 自动写入；内部包收入），闸门比对不一致或缺登记文件都判仪器失效；合法的重新冻结（`--force`）会改写这个被跟踪文件，改动在 git 中可见。回归测试：同文件第 6 节（9 项）。子检查仍依赖 `results/pilot/pilot-20260924T205621Z-93d8654c/run_record.json`、受限问题 YAML 与公开价格文件：它们被移走或改动时闸门报仪器失效（响亮失败），须重新冻结。CSV 两类解析路径的“无 run id”形态只由第 6 节合成测试覆盖，没有真实冻结对照。FIX_C 原先直接复算的三处仓库文件（`results/pilot/…1fc52728/rations_q.csv`、`reports/dev_case_v1_report.md`、旧 smoke 的 `rations.csv`）不再作对照：其中报告在 FIX_B 改写后不再触发，缺失的文件会被静默跳过。范围：本闸门覆盖 DM 途径；浓度行等其他途径靠路径规则关闭，不能证明不存在。**W3 补充**（§6.3）：任意版式的 CSV q 表（宽表 `q_<原料>` 列或长表 `ingredient_id` + `q_as_fed…`，任意文件名；此前只读长表 `rations_q.csv`，宽表会被当作“无配方表”直接放过）；成本可来自包内另一文件（按 `q_hash` / 运行 id + 标签连接）；计划 DMI 用运行配置中公开的 `SH-DM-PLAN` 值；q 不公开核对与持有方真实对照；余量一步还原受限界值的线性核对（合成对照 + 持有方真实对照）。这些新对照缺失、不触发时同样判仪器失效 / 仪器不可用。
8. **待决事项**（外部包）：`EXTERNAL_PENDING` 所列文件在用户决定之前使外部包为 BLOCKED。

2026-09-24T20:5xZ 的检查结果（输出在会话 scratchpad，不入库）：内部包 280 个文件，全部闸门 PASS（可识别性阳性对照：合成 5 条配方解出 2 种原料；三处真实公开输出全部判出——开发运行 5 种原料精确解出、报告表 2 种原料解到印刷精度、旧 smoke 8 种原料一步解出）；外部包 136 个文件，**BLOCKED**（B-310、B-318 待决）且本机用户路径闸门 **FAIL**（`src/ration_reliability/README.md` 第 29、38 行与 `experiments/E0_verification/run_dev_case_v1.py` 第 70 行写有 `cd /Users/<用户>/…`，不在 FIX_C 修改范围，已请求改为 `cd <project root>`）。

- ZIP 总哈希写在 ZIP 外的回执中，ZIP 内只放成员清单（复核报告 §7.2）。
- 外部复现包在干净目录解压后运行测试与合成示例，记录结果（§2.1）；这是同一机器的隔离复现，不是第三方复现。
- 此前的打包脚本是会话 scratchpad 中未入库的 `pack_rrs.py`（`reports/packaging_record_20260924.md` 记载；FIX_C 读到的是 2026-09-25 00:28 KST 修改后的版本）：它的排除表没有 `reports/model_audit.md`，目录表包含 `results/pilot`、`docs/contract` 与复核记录，没有删去 ⑥ 要求的吸收系数，只打一个包，扫描只查核心表三元组，且把明文凭据写在脚本里作扫描模式。它不再使用；打包一律用 `scripts/package_release.py`。

- **CLOSE1 检查（2026-09-24T22:18:35Z，会话 scratchpad 输出，未打包；最终复查见 ~~`CURRENT_STATE.json` `packaging.last_check`~~ 第二轮状态页只读副本 `audit/_history/CURRENT_STATE_round2.json` `packaging.last_check`〔Round3-fix：`CURRENT_STATE.json` 自 schema /2 起不再有该字段〕）**：内部包 293 个文件，8 个闸门全部 PASS（可识别性：合成对照解出 2 种原料，三个冻结对照分别解出 5、7、2 种，`instrument_status = ok`，包内 1 个含原料表的文件未判出）；外部包 139 个文件，路径、改写、本机用户路径（B-431 已在源文件修复）、密钥、指纹、可识别性 PASS，吸收系数与待决事项 BLOCKED（B-310、B-318 待用户），总状态 BLOCKED。FIX_C 与 K8 记录的外部包本机路径 FAIL、两包可识别性 FAIL 已消除。

- **CLOSE2 检查（见 ~~`CURRENT_STATE.json` `packaging.last_check`~~ `audit/_history/CURRENT_STATE_round2.json` `packaging.last_check`〔Round3-fix 改指〕，会话 scratchpad 输出，未打包）**：可识别性闸门改为上文 CLOSE2 规则后，对照清单重新冻结（2026-09-24T22:56:01Z，`--force`；8 个对照文件与 CLOSE1 冻结时逐字节相同，三个对照仍解出 5、7、2 种原料；Markdown 对照删去 run id 后经公开价格判出 2 个可识别坐标）；清单 sha256 `fbd3db85881ea7520bff4e02724711b40ede8f2a82bf5d02e2646814553982f6`（登记于 `docs/package_gate_controls.pin.json`）；CLOSE1 清单 `fb94efad…aab5` 已备份到受限目录 `superseded/`。内部包全部闸门 PASS（包内唯一含原料表的 `audit/data_feasibility_report.md` 走无真值分支：无成本行、无可识别坐标，PASS）；外部包 BLOCKED（吸收系数与待决事项，B-310、B-318）。

## 6 第三轮开发运行的公开表进入内部审计包（W3）

### 6.1 范围与判定原则

- 对象：开发运行 `pilot-20260925T055921Z-fb4f75af`（px 授权算力，冻结提交 46c22cb，`output_label = development`，`code_and_spec_unchanged_during_run = true`；运行所用受限产物是 Mac 确定性构建的字节，px 自建产物~~因 OpenBLAS 末位浮点差~~被冻结检查判为不同而未采用〔**Round3-fix 更正（第三轮红队 D-10）**：冻结检查的规格指纹只覆盖 `build_report.json` 这一个受限构建文件（`dryrun.json` 的 `differs` 只列它），其他构建产物是否不同没有被冻结检查比较；px 自建产物也改变了参考规格指纹（`11d5aaec…`，换回后 `e7f91072…`）。差异原因“平台数值末位差（OpenBLAS / Accelerate）”是**推测（未逐字段核验）**，B-445；Mac 字节回放与 dryrun2 的命令只有总负责人事后补写的记录 `logs/px_runs/pilot-20260925T055921Z-fb4f75af/mac_bytes_replay.md`，`px_run.sh` 到第一次 dry-run 为止〕）。这是开发材料：本包只供复核方审阅，任何数字都不是正式结果或稿件结论；本次可比较可行集合为空，成本不作跨方法比较。
- 只进内部包；外部包的 `SCOPE` 不含 `results/`、`logs/`，其禁止规则照旧，外部包维持 BLOCKED（B-310、B-318）。
- 显式清单，不按目录收：`INTERNAL_RUN_PUBLIC_FILES`（7 个公开表）与 `INTERNAL_RUN_LOG_FILES`（15 个日志）。目录中以后出现的其他文件不收，清单的 `run_files.not_collected_no_decision` 会列出它们，须先逐个判定。受限目录 `data/restricted_local/pilot/<run id>/` 与其他 `results/pilot` 运行照旧两包都不收。
- 第三轮报告（W1 生成）：`reports/round3_ablation_results.md`、`reports/round3_ablation_summary.csv`、`reports/*_pilot-20260925T055921Z-fb4f75af.csv` 随 `reports/` 收入内部包，与其他报告一样须通过全部闸门；清单的 `round3_reports` 记录每条规则当前命中的文件（不存在即为空）。
- 判定标准（与 D-226、D-K4-9、D-K7-10 / D-452 一致）：不含 NASEM 等受限表值，也不含与包内其他内容联立可一步还原受限值的量（原料 DM、计划 DM 占比、d̂、受限界值）。

### 6.2 主机日志逐个判定（`logs/px_runs/pilot-20260925T055921Z-fb4f75af/`）

| 文件 | 判定 | 内容与理由 |
|---|---|---|
| `FULL_STARTED`、`FULL_ENDED`、`FULL_EXIT` | 收 | 开始、结束 UTC 时间与退出码 0 |
| `px_setup.sh`、`px_run.sh`、`px_full.sh` | 收 | 实际执行的命令（检出 4cd0f51 / 46c22cb、按文件名复制受限输入、venv 与锁安装、构建、预检、测试、环境核对、干跑、完整运行）；含主机目录路径，无地址、无凭据、无数值 |
| `append_env.log`、`envcheck.json` | 收 | 环境锁追加与核对（`matches_lock`，指纹摘要） |
| `build.log` | 收 | `scripts/build_dev_case.py` 打印的摘要：检查数、验证器计数、预检状态、M0 目标值（即公开表中的 M0 成本）与 NEL 余量上限百分比（一个 LP 最优值相对公开需要量的比例）；不含表值、q、占比或 d̂ |
| `preflight.log` | 收 | 预检 READY 一行（文件数） |
| `pytest_px.log`、`pytest_px2.log` | 收 | 测试进度、警告（含主机路径）、跳过原因与计数 |
| `dryrun.json` | 收 | 用 px 自建受限产物的干跑：冻结检查 `differs_from_frozen`（这正是改用 Mac 构建字节的依据）；只有 id、计数与摘要 |
| `dryrun2.json` | 收 | 用 Mac 构建字节的干跑：`matches_frozen_anchored`；只有 id、计数与摘要 |
| `full_run.log` | 收 | 驱动摘要（行数、耗时、冻结状态）与 HiGHS 进度行 |
| `mac_bytes_replay.md`（FIX_PKG 加入） | 收 | 总负责人事后补写的 Mac 构建字节回放与 dryrun2 命令记录（红队 D-10）：命令、主机别名与目录路径、构建文件 sha256 的前 16 位（完整 sha256 已在 `run_record.json` 中）；无地址、无凭据、无数值；原因写明为推测（B-445） |
| `authorisation_confirmation.md`（FINAL 加入，2026-09-26） | 收 | 总负责人登记的用户对话确认（「确认授权，继续」），关闭 B-449；原授权文件未改，只引其 sha256（已在 `run_record.json`）；无地址、无凭据、无数值 |
| （排除） | 无 | 17 个文件均判为不含受限值；`INTERNAL_RUN_LOG_EXCLUDED` 为空 |

8 道闸门（FIX_PKG 起为 9 道，新增运行输出哈希闸门只针对 7 个公开表；FIX2 起为 10 道，第 10 道数值指纹闸门扫描包内全部文本文件）对这些文件全部适用（`.log`、`.sh` 自 W3 起计为文本；此前内部包中的 3 个复核 `.log` 从未被内容闸门读过，现在也一并扫描）。

### 6.3 可识别性：q 不公开、余量与公开价格 / 成本

1. **DM 途径**。公开表中配方只以 `q_hash`（对全精度 q 的 sha256）出现，成本在 `endpoint_ablation.csv`；公开计划 DMI（`SH-DM-PLAN` = ⟨withheld⟩，`configs/dev_case_v1/reference_constraints.csv`）与公开价格（`configs/dev_case_v1/prices.yaml`）都在包内。每条配方公开的只有“DMI 等式 + 成本”两个方程，而未知的 q 有 8 维，原料 DM 在线性系统中没有已知系数的方程——只要 q 不公开，这条线性途径是闭合的。闸门的做法：
   - 任意版式的 CSV q 表都被解析（宽表 `q_<原料>`、长表 `ingredient_id` + `q_as_fed…`、任意文件名），缺成本时按 `q_hash` / 运行 id + 标签从包内其他文件连接成本，计划 DMI 取该运行配置中的公开值（另试“DMI 未知但各配方相同”）；有受限真值时按印刷精度判还原，无真值时任何可识别坐标即 FAIL。〔**FIX_PKG 更正（红队 D-1）**：“任意版式”说过了头——W3 实际只解析 `q_` 前缀宽表与 `ingredient_id` + `q_as_fed…` / `q` 长表，且只看 `.csv` / `.md`；红队用真实受限 q 实测，列名不带 `q_` 前缀、长表数量列另名、`.tsv`、JSON 四种版式全部漏过。现在的实际覆盖见 §7.1。〕
   - q 不公开核对：带 `q_hash` 的表中每个 `q_hash` 必须是 64 位十六进制摘要；包内 JSON（含 JSON 格式的 `.log`）不得带 `q_as_fed…`、`d_hat`、`planned_dm_share…`、`x_box_*`、`raw_candidate_x`、`dm_estimate(s)` 等键的数值（字符串 `restricted…` 除外）。
   - 阳性对照：合成宽表与长表（成本放在另一个文件、按 `q_hash` / 标签连接）必须解出受限 DM；**持有方真实对照**：把该运行的受限 `rations_q.csv` 当作包内文件在内存中读入（虚拟路径，不写盘），经包内公开成本与公开 DMI 必须解出受限 DM——本次 26 行、8 种原料全部解出，说明一旦 q 被公开闸门必判 FAIL。受限 q 表缺失时判仪器不可用（BLOCKED），不算通过。
2. **余量**（`reference_residuals.csv`）。每行余量 = 配方量 − 界值（或反之），逐状态分位数。D-K7-10 / D-452 要保护的是“余量 + 浓度一步即得 Table 5-1/21-3 值”：两行若共享同一配方量（如 PN-CP-HI 的 CP 浓度与 PN-CP-SUP 的 CP 供给 = 浓度 × 计划 DM），一行界值公开时，另一行的受限界值就被还原。闸门把每个余量统计量写成线性方程组（未知数 = 每条配方、每个状态的基础量 + 读者不知道的界值；读者知道的界值 = 参考表中的公开数值，或受限值已在包内印出者），任一受限界值可识别即 FAIL；只报告约束 id，不报告数值。结果：
   - PN-CP-SUP 的界值（Table 21-3）受限且包内没有印出；它的配对行 PN-CP-HI（19 % 公开）的余量已由运行驱动写为 `restricted`（`REDACT_TEST_MARGINS`）——公开表不还原任何受限界值；持有方真实对照（受限 `reference_residuals_full.csv`，含 PN-CP-HI 余量）必须还原 PN-CP-SUP 的界值，本次触发。
   - PN-T1–T5 的界值（Table 5-1）本身已随 `configs/constraints.yaml` 进入内部包（B-318 内部放行），读者由余量可得到配方层面的 NDF / fNDF / 淀粉浓度分位数，但不会因此得到包内尚无的受限值；q 不公开时这些浓度也不构成线性 DM 途径。闸门不判为新增暴露，列入 §4 由总负责人确认。〔**FIX_PKG 更正（红队 D-4）**：这一判断与项目既有规则不符（浓度本身列为受限，见 §4 该行更正）；打包副本现扣留这些余量，闸门 (d) 检查见 §7.2。〕
   - 运行驱动自己声明要隐去的行（`REDACT_NOMINAL`、`REDACT_TEST_MARGINS`，由源码 AST 读取）在包内任何残差表中必须隐去。
   - 阳性对照：合成 PN-CP-HI + PN-CP-SUP 配对必须还原 CP 下限，隐去 PN-CP-HI 后必须不再还原。
3. **未覆盖 / 残余风险**（不宣称已证明不存在）：
   - 多行余量在 q 未知时的双线性（矩阵分解）重建：没有已知系数的线性途径，结构上每条配方有 8 维未知的计划 DM，本闸门只作论证，不作证明（`assumption_only`）。
   - Markdown 表的列名只按完整标签或 `q_hash` 与其他文件的成本连接，缩写标签（如只写 “M0”）不连接；Markdown 中说明“某行名义余量为 0”的文字不被扫描（§4 已列 W1 报告中的一处）。
   - `run_record.json` 的 `output_hashes` 含受限 `rations_q.csv` 与 `reference_residuals_full.csv` 的 sha256，`q_hash` 是全精度 q 的 sha256：候选空间是全精度浮点，不构成可行的穷举校验器（与 FIX3_DEF 所指“被删短数字 + 集合摘要”不同）。
   - 日志与运行记录含授权主机的用户目录路径（内部包保留本机路径的既有规则）；外部包不收这些文件。

### 6.4 检查结果

见 `audit/_parts/round3/W3_package_scope_record.md`（命令、时间、8 道闸门状态、文件数、对照计数）。输出目录在仓库外，检查完已删除，未打包。〔Round3-fix：W3 的结果已被 FIX_PKG（§7，9 道闸门）取代；最近一次检查登记在 `CURRENT_STATE.json` `packaging`。〕

### 6.5 证据层级

- 范围规则、路径闸门、CSV q 解析、q 不公开核对、余量线性核对与各阳性对照：`code_tested`（`tests/unit/test_package_release.py` 第 7 节，全合成）。
- 对真实运行文件的检查结果与两个持有方真实对照：`holder_verified`（持有方本机，复核方未复算）。
- 日志逐个判定：人工逐文件阅读 + 闸门扫描，`holder_verified`。
- 双线性途径不存在：`assumption_only`。无 `independently_reproduced` 项。

## 7 FIX_PKG 修订（第三轮红队 D-1 / D-4 / D-7 / D-9，2026-09-26）

本节只描述 `scripts/package_release.py` 现在实际做的事，不扩大说法。详细记录与检查结果见 `audit/_parts/round3/FIX_PKG_record.md`。

### 7.1 q 表解析的实际覆盖（红队 D-1）

**FIX2：三层防线各自管什么（先读这一段）**

1. **路径白名单是第一道防线**：`SCOPE`、`EXCLUDE`、`FORBIDDEN` 与运行文件的逐文件显式清单决定受限表根本不进包（`data/restricted_local/` 两包都不收，路径闸门在最终清单上独立复查）。后两层只是在“某个文件里仍写着受限数值”时的补救。
2. **版式解析闸门（第 7 道，可识别性）是启发式**：它只读下文列出的版式，凡是没有列出的写法（YAML、HTML、按量逐条的 JSON 记录、列名带单位、正文句子等）它看不见，看不见不等于没有泄漏。第三轮核验就在它声称的覆盖范围内找到 3 个漏报（见下文 FIX2 条）。
3. **数值指纹闸门（第 10 道，§7.6）覆盖任意版式**，但**只对“同一配方的 ≥ 3 个投料量出现在同一窗口（同一行或相邻 20 行）”生效**：单个投料量（或两个）的泄漏不在其覆盖范围；换算成其他单位（g、lb、DM 基础）、占比、合计、有效数字不足 3 位的数，以及非文本文件（如 ZIP）也不在其覆盖范围。

三层都不能证明包内没有受限内容；闸门全 PASS 只表示这三种检查没有发现。

- 读哪些文件：整份分隔表 `.csv` / `.tsv` / `.psv`；嵌在 `.md` / `.txt` / `.log` 文本中的表格（连续 ≥ 2 行、用同一分隔符切出相同列数：`,`、制表符、`;`、`|`（Markdown 管道表，分隔行跳过）、连续空白（空白读法跳过含 `|` 的行；表头比数据行少一列时视为 pandas 式索引列））；`.json` 与 JSON 格式的 `.log`。
- 每张表的三种读法：① 宽表：列名去掉反引号 / 星号、至多一个前缀（`q_as_fed…_`、`q_`、`kg_as_fed_`、`as_fed_`、`kg_`）与一个后缀（`_kg_as_fed…`、`_kg_per_head_d`、`_as_fed`、`_kg`、`_q` 等）后等于已知原料 id 或中文原料名，即为 q 列（`q_` 前缀可有可无）；② 长表：原料列（`ingredient_id`，否则取值全为原料 id 的列）加任一数值列，逐个数值列（非键、非成本）当作 q 试算，取最坏结果；③ 行标签表：首列为原料、每列一条配方、可有成本行（原 Markdown 规则，现也用于分隔表）。JSON：同一映射中 ≥ 2 个原料 id 键带正数值即为一条配方，配方键与成本取自映射本身或其所在记录，缺成本时按 `q_hash` / 标签从包内成本索引连接。
- 新读法中一行（或一列）至少有 2 种正数原料才算配方（同 Markdown 规则）：单一原料的“配方”会让任何 DM 都“可识别”，初版在 `reports/data_audit.md` 上因此误报，已排除；W3 原有的 `q_` 前缀宽 CSV 与 `ingredient_id` + `q_as_fed…` 长 CSV 维持 ≥ 1。
- `q` 不公开核对另加一条：包内 JSON（含 JSON 格式 `.log`）中凡是以 ≥ 2 个原料 id 为键的数值映射都判为 q（FAIL），无论键名叫什么。
- 阳性对照：闸门内置合成对照新增 9 条读法（无前缀宽表 + 成本、任意数值列长表、`.tsv`、`.psv`、`.txt` 中的 CSV 块、`.log` 中的空白表、`.md` 中按配方成行的管道表、行标签分隔表、JSON 映射），每条都必须经 `identify_file` 解出受限 DM，否则判仪器失效；回归测试见 `tests/unit/test_package_release.py` 第 8 节。持有方在内存中用真实受限 q（26 条配方、8 种原料）按红队 V1、V3–V7 与行标签等 10 种版式构造虚拟包内文件（不写盘），逐个经 `identify_file` 全部判 FAIL（路径带 run id 时 8 种原料 DM 全部还原，数值未打印）；其中 5 种另注入整包经可识别性闸门判 FAIL。注意：`RUN_ID_RE` 以 `\b` 定界，run id 紧跟在下划线后（如 `x_pilot-…`）时不被识别，文件走无真值分支（仍判 FAIL，但不做还原核对）。
- 不覆盖：YAML（配置中有公开的逐原料价格）、代码、HTML / TeX、二进制文件（如 `docs/` 下两个 ZIP）、表格字段前带日志前缀的行、标签含空格的空白表（只读到切分整齐的行）、被正文打断的表、正文陈述（如“名义余量为 0”）。
- **FIX2 更正（FIX_PKG 核验 D-1 残留）**：核验用真实受限 q 在上面声称覆盖的范围内找到 3 个漏报，整包 9 道闸门全 PASS——(M1) 长表配方列名为 `recipe` / `name` / `配方` 时，`_row_key` 只认固定键名，26 条配方被并成 1 条、可识别坐标 0；(M2) 表前加一行 `#` 注释；(M3) 分号分隔的 `.csv`（`.csv` 只按逗号整表解析）。此前的“不覆盖”清单没有列出这三项，属于说过了头。现已修正：
  - 配方键：已知键名（`label`、`method_label`、`method_id`、`ration_id`、`ration`）都不在时，任何非数值、非原料 id 的文本列可作配方键——优先首列，否则取唯一文本列；长表中某候选若会把同一原料放进同一配方两次（如取值恒定的 `unit` 列），不作配方键（`_text_key_column`）。
  - 注释行：整份分隔表先去掉以 `#` 开头的行再解析（同时保留原样读法，以防表头本身以 `#` 开头）；嵌在文本中的表遇到 `#` 行跳过，不打断表。
  - 分隔符：整份分隔表（`.csv` / `.tsv` / `.psv`）先用 `csv.Sniffer` 判定，再逐一尝试文中出现的 `,` `;` 制表符 `|`，取最坏结果；分隔符不是逗号时，形如 `29,35` 的单元格另按小数逗号读一遍。
  - 各有合成阳性对照（闸门内 `q_columns_synthetic_control` 的 `fix2_paths` 6 条：`recipe` 键长表、首列 `配方` 键长表、`#` 注释、分号、制表符、分号 + 小数逗号）与回归测试（`tests/unit/test_package_release.py` 第 9 节）；用修正前的脚本跑同一批测试，M1–M3 相关测试全部失败，说明测试能区分新旧代码。持有方用真实受限 q 在内存中复测 M1–M3（`audit/_parts/round3/FIX2_holder_checks_inmemory.py`），结果见 FIX2 记录。
  - 仍不解析：列名带单位（`id (kg/d)`、`玉米青贮(kg)`）、长格式 JSON 记录（`[{ingredient, kg}]`）、YAML、HTML、正文陈述等；这些由第 10 道闸门在“≥ 3 个投料量同窗口”的条件下兜底（§7.6），单个数值仍不覆盖。`RUN_ID_RE` 的下划线定界问题仍未修（已登记）。

### 7.2 浓度类余量（红队 D-4）

- 依据：驱动 `run_dev_case_v1.py` 的 `public_redaction.restricted_only` 含“CP/NDF/starch/forage-NDF concentrations”；USER_ACTIONS c13；D-452。余量 = 浓度 − 界值（或反之），界值可读时一步得出浓度的逐状态分位数。
- 界定：`ROW_EXPRESSIONS` 中基础量属于受限浓度类（`RESTRICTED_CONCENTRATION_BASES`：CP、NDF、粗饲料 NDF、淀粉、干粉碎玉米淀粉份额）的行 = PN-CP-HI、SH-PLAN-CP-HI、PN-CP-SUP、PN-T1–T5、DIAG-T51-DGC-STARCH-SHARE。PN-CP-HI 与 SH-PLAN-CP-HI 已由驱动隐去；其余 7 行由打包规则 `FIXPKG-RESID-CONC-MARGINS` 扣留 `mean_deficit_given_violation`、`max_deficit`、`margin_q01`、`margin_q05`、`margin_q50`（52 行 × 7 = 364 行，精确计数，不符即改写闸门 FAIL）。只改打包副本；运行产物 `results/pilot/…/reference_residuals.csv` 不改，其源字节仍由 `run_record.json` 绑定。计数与违约率列保留。
- 不扣留：EE、Ca、P 浓度行（驱动的公开剖面量）、供给行（NEL、吸收 Ca / P、DM）。
- 闸门检查 (d)“界值可读时余量能否一步得出受限浓度”：浓度类行带数值余量 / 亏缺，且界值可读——公开数值、受限值已在包内印出，或受限的来源表值（`value_ref`，保守地视为持有原书的读者可读）——即判 FAIL；界值不可读（无参考行、公式文字）的行只列出不判。阳性对照：合成表必须恰好标出两行浓度类（一行包内可读、一行来源表可读）且在扣留后不再标出；持有方对照：被改写的残差表的源字节必须被标出，否则判仪器失效。本次：源文件中 7 个约束共 364 行被标出，打包副本 0 行。

### 7.3 运行输出哈希闸门（红队 D-7）

- 第 9 道闸门 `run_output_hashes`（只对内部包）：`INTERNAL_RUN_PUBLIC_FILES` 中的 6 个表逐个用源字节的 sha256 与 `run_record.json` 的 `output_hashes` 比对（打包副本经脱敏的，比对源文件）；`run_record.json` 本身不在自己的 `output_hashes` 中，与脚本登记的 `W3_RUN_RECORD_SHA256`（f716062b…63b4）比对，因此“改了表、又把记录里的哈希改成一致”也会被判出；运行 id 不符、缺条目、缺运行记录都判 FAIL。阳性对照：内存中改一个字节的公开表与运行记录都必须被判出。
- 主机日志（`INTERNAL_RUN_LOG_FILES`）不在 `output_hashes` 中，本闸门不绑定它们（残余风险）。

### 7.4 描述更正（红队 D-9、D-3、D-10 的打包描述部分）

- `ablation_config.json`：“declared 2 x 3 configuration”（运行产物自身文字中的 “2 x 2” 与 “not executed in round 3” 不改，见 B-444）。
- `solve_status.json`：注明其对偶界字段本次全为 null（驱动键名不一致，D-3；驱动已修正，只影响以后重新冻结的运行）。
- `logs/…/dryrun.json`：原因写为推断（冻结检查只覆盖 `build_report.json`；平台数值差异是总负责人的推断，未逐字段核验，B-445）。§6.1 中“因 OpenBLAS 末位浮点差”的说法属 D-10，本修订未改。〔Round3-fix：§6.1 已改为推测并注明冻结检查的覆盖范围。〕

### 7.5 证据层级

- 新解析路径、浓度类余量检查、字段扣留、运行输出哈希闸门与各阳性对照：`code_tested`（第 8 节，全合成；另一项读取公开运行表的测试在该目录存在时运行）。
- 对真实运行文件的闸门结果、真实受限 q 的 10 种版式注入、源残差表被标出 364 行：`holder_verified`（持有方本机，复核方未复算）。
- “以上版式之外不存在其他可识别写法”：不宣称；未覆盖项见 §7.1 末条。无 `independently_reproduced` 项。

### 7.6 第 10 道闸门：受限配方数值指纹（FIX2）

本节只描述 `scripts/package_release.py` 的 `gate_restricted_q_fingerprint` 实际做的事。

- **数值来源**（`RESTRICTED_Q_SOURCES`，持有方本机的受限目录）：`data/restricted_local/pilot/*/rations_q.csv`（第三轮宽表）、`data/restricted_local/pilot/*/*rations_full.csv`（早先开发运行的长表）、`data/restricted_local/smoke/*/rations.csv`、各 dev_case 构建报告（`dev_case_v1/build_report.json`、`dev_case_v1/*/build_report.json`、`_r3f_build_check_NOT_A_RUN/*/build_report.json`）中的名义预检配方 `q_as_fed`。每份配方取非零投料量（> 1e-6 kg；更小的视为求解噪声）。只在内存中读，不写出、不打印。
- **打印形式**（`q_value_forms`）：每个投料量（FINAL 起同时取最短 repr 与二进制精确值两种十进制展开）按 3、4、5、6 位有效数字与 1–6 位固定小数，各做四舍五入（half-up 与 half-even）和截断，保留与去掉末尾 0 两种写法，小数点与小数逗号两种分隔；只保留有效数字 ≥ 3 位的形式。文本中有效数字多于 6 位的数（全精度输出）按它自己的打印精度做数值比较（四舍五入或截断相等即算）；FINAL 起比较同时用最短 repr 与二进制精确值（此前只用 repr，`%.20f`、`%.18e` 这类印出精确二进制展开的写法会漏掉）。科学计数法只认完整的词（十六进制摘要里的 `…56e1234…` 不算）。
- **判定**：扫描包内每个文本文件（不看后缀）。FINAL 起非 UTF-8 文本先解码再扫描（`decode_text_like`：字节序标记 UTF-8/16/32；UTF-8；ASCII 与 NUL 交错的 UTF-16 LE/BE；前 4 KiB 无 NUL 的文件依次试 GB18030、CP949，最后 Latin-1）；此前只扫 UTF-8，GBK 或 UTF-16 的配方表会被跳过。含 NUL 且不是上述编码的文件（ZIP、图片等）判为二进制，只计数、不扫描。同一文件的一个窗口——同一行，或相邻 20 行（超过 2,000 字符的行按 2,000 字符分段计行）——内出现同一配方 ≥ 3 种原料的投料量，且每个量对应文本中不同的数，即判 FAIL。结果只报受限来源路径、配方个数、窗口内命中个数与行号，不报数值。
- **适用性**：受限目录中找不到任何配方表时判 **BLOCKED**（`applicability = not_applicable_without_restricted_q_tables`），明确标注，不静默 PASS；外部读者的环境正是这种情况。
- **阳性对照**：① 合成对照（不用受限数值）：两份合成配方以 8 种版式（YAML、HTML 表、JSON 记录、列名带单位 + 小数逗号、截断的正文句子、宽 CSV、长 CSV、单行 JSON）写出必须全部判出，另有 3 个阴性版式必须不判出（只有 2 个量；3 个量彼此相隔超过窗口；只印 2 位有效数字）；② 持有方内存对照：每个受限来源中 ≥ 3 个投料量的全部配方，以 5 种解析器不读的非常规版式（YAML、HTML 表、JSON 记录、带单位列名 + 小数逗号、正文句子）在内存中写出，必须全部判出。任一对照不触发即仪器失效（FAIL，不归咎于包内文件）。
- **覆盖与不覆盖**：覆盖任意版式，但只对“≥ 3 个投料量同窗口共现”生效。**单个投料量（或同一配方的两个）的泄漏不在覆盖范围**；换算成其他单位（g、lb、DM 基础）、计划 DM 占比、合计、有效数字不足 3 位的数（如 1–10 kg 之间印 1 位小数的量）、跨越 20 行以上的写法、非文本文件都不覆盖。数值巧合可能造成误报：判 FAIL 后须人工看命中行再定性。编码识别是启发式：非常短的 UTF-16 文件、混合编码文件、压缩或加密内容不保证读到。
- **首次运行的发现（FIX2，真阳性）**：内部包的 `reports/minimum_case_nutrition_audit.md` §4 逐原料表“名义预检 q”列（第 103–110 行）印出了名义预检 M0 配方的投料量（3 位小数，其中 6 个量有效数字 ≥ 3 位），与受限 `dev_case_v1/build_report.json` 的名义预检配方、本运行受限 `rations_q.csv` 中 6 个 M0 配方及早先运行的 M0 配方一致；下一行还印有名义预检总成本。版式解析闸门没有看见这张表（首列标签带 NASEM 条目号，不在 `ZH_LABELS` 中）。按 FIX_C 发现 1 的规则（含 DM 计价原料的配方的公开 q 与总成本视同受限输出；`reports/dev_case_v1_report.md` 正是因同类 q 表被排除），这一列属受限类。本次只报告、未改包范围或打包规则：处理方式（打包副本中扣留该列，或内部包不收该文件）以及此前已发出的内部包是否含此文件，由总负责人决定（BLOCKERS B-450）。**〔FINAL 处理，2026-09-26〕** 总负责人在源文件中把该列 8 个数值改为“受限”，原值移到 `data/restricted_local/dev_case_v1/nominal_precheck_q_from_audit_report.md`（同一组数值本来就在受限构建报告）；总成本单独保留（没有 q 时只是一个数）。复查：用本闸门扫描此前发出的 3 个 ZIP，第二轮内部包 `ration_reliability_study_internal_20260924T231311Z.zip` 只有这一个文件命中（共 298 个文件，296 个文本文件被扫描，其余 295 个 0 命中），阶段 0–4 包与继承总档包 0 命中；本地 git 历史（提交 e1ed163、6cad2a1）仍含该列，推送前须清史（与 0b161a9 同）。
- **证据层级**：规则与合成对照 `code_tested`（第 9 节）；真实包检查与持有方内存对照 `holder_verified`；“除上述条件外不存在其他泄漏写法”不宣称。
