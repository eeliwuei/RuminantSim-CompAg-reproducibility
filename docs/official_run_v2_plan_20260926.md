# 正式运行计划（reference problem v2 → protocol freeze → official run）

- 日期：2026-09-26。依据：用户决定 D-532（故事线与贡献排序）、D-533（采纳《投稿就绪评估》§3 决定包）、D-534（GitHub）；本文由规划 agent 的实施计划整理而成，是执行清单，不是结果。
- 不写任何受限值、比值或预留根。

## 0 冻结前必须解决的五个问题

| # | 问题 | 处理 |
|---|---|---|
| F1（关键，研究负责人定） | 只加“计划层域前提行”（t0 名义成分下干粉碎玉米淀粉占比 ≥ τ）不足以让可比集合非空：LP 会把该行压到恰好取等，而终点评估仍按**逐状态**判定域内外，约一半状态会落在 τ 以下 → T 行 unknown → 上界率仍接近 1。须在冻结前按营养学理由（不按可行性）预先选定主事件的域读法：(a) 保持逐状态判定并如实报告；**(b) v2 主事件按“配方层”（所配日粮在 t0 名义成分下是否满足前提）判定域，逐状态读法并列报告；** (c) 把逐状态前提行放进联合机会事件（超出“计划行”，需新决定）。 | 代码同时实现 (a)(b)，冻结时选定；建议 (b)：Table 5-1 的前提“干粉碎玉米为主要淀粉源”是所配日粮的设计属性，不是某次成分抽样的属性 |
| F2 | 选参筛的是各臂训练事件（线性能量行、T 行全域判定），成员判定用的是主参考事件（参考能量链、域条件）——两者不同 | 保留并披露；Methods 明写 |
| F3 | 预留根一旦抽样即“用掉”（SP-4）：任何门槛检查都必须在创建 `RandomStreams(预留根)` 之前完成；只有运行结束的“清单未变”检查可在抽样后失败，产出 `official_invalidated` 并披露、同根重跑 | 官方路径按此顺序实现 |
| F4（B-445） | 规格指纹含受限 `build_report.json`，px 自建字节与 Mac 不同 | px 上用 `scripts/replay_build_outputs.py apply` 换回 Mac 字节并留日志，再跑 pytest |
| F5 | `io/config.py` 官方模式要求 `run_context.protocol_sha256` 与 `primary_assumption_id` | 官方驱动在内存中为每臂配置注入 |

## 1 正式设计（冻结内容）

- **Reference problem v2** = dev_case_v1 FULL11 问题 + 结构计划行 `SH-PLAN-T51-DGC-SHARE`（系数 (1{i∈干粉碎玉米} − τ)·名义淀粉_i，作用于 q·d̂，≥ 0，τ = 0.50），写入 cfg0，所有臂、所有方法（含 M0）继承；四个 τ 读法变体仍在评估中并列报告。
- **格（每根）**：主臂 MAIN9 × {SD-H0, SD-S2, SD-SRC14, SD-12MO, SD-SRC12}；透明度臂 FULL11、PART6P5 × {SD-H0, SD-S2}；五个 SD 点共用 root/opt、root/validation、root/test（共同随机数）。
- **评估**：每份配方在五个世界的 test 流上评分；SD-H0 为主；对角线（自身世界）+ SD-H0 列构成假设范围，不称 CI。
- **方法与网格**：与开发消融相同（M0；M1 含/不含 DM 余量；M2；M3a；M3b）；N 阶梯 {128, 512, 1024}，opt = 1024 抽样、小 N 取前缀；validation 20,000、test 10,000；CP 精确区间；固定规模（不采用加倍规则）。
- **选参**：`cp_upper_le_alpha`（单侧 CP 上界 ≤ α，c = 0.95），validation 流。
- **成员**：主参考事件在评估世界 test 流上的单侧 CP 上界 p̄ ≤ α；覆盖率按条目与按不同 q_hash 并报。
- **诊断**（标 `diagnostic`，永不进集合）：DIAG-T45（只放宽 T4/T5 的名义 LP 余量试验：A0 全行；A1 去 T4/T5；A2 求使名义余量 ≥ h 的最小放宽 δ）；DIAG-E1（能量行单行 SAA）。
- **DM 模式**：DM-uncertain（现管线即此）；范围：8 种原料、一库存、一价格。

## 2 改动文件（分批）

- **B1**：`nutrition/domain.py` 计划行系数函数与配方层域状态；新 `experiments/E1_cost_reliability/official_v2.py`（premise row 构建、`REDACT_NOMINAL_V2`）；新 `configs/dev_case_v1/reference_constraints_v2.csv`（v1 不动）；`evaluation/reference.py` 接受/要求 v2 行、加 `table51_domain="plan"` 读法与伴随事件；`docs/reference_problem_v2.md`；B-442 交叉一致性测试。
- **B2**：`run_dev_case_v1.run_method_block` 加 `screening_rule`/`confidence` 参数；消融驱动 `_membership`/`comparable_sets` 加 `membership_stat`（官方 = cp_upper）与 `coverage_distinct_rations`；新增 `per_method_report`（每方法所选配方的率对、p̄、达标、成本〔标“非可比成本”〕、声明网格内最低主事件 r̂⁺ 及其成本、前沿诊断）、`candidate_rations`（去重 q_hash 后全部评估，纯描述）、`matched_cost_curve`（专家的“同成本预算下约束满足”描述曲线）；`configs/methods.yaml` 与代码对齐（B-453 六项 + N_final、M1 语义/尺度/DM、M3 集合类型）；`configs/protocol.yaml` 除 freeze.* 外全部填写（DM 模式、主对比、空集合报告、test_previously_seen=true、删 no_cost_data_report_break_even_only、范围偏离、U3 不可识别、T2.3/调整幅度/B-428 偏离登记）。
- **B3**：SD 点数组与登记核对（5 列 × 48 格；H0 全 1；S2 与既有一致；各点在登记区间内）；`OFFICIAL_CONFIG`；两项诊断实现；收敛报告；`build_context`/`plan()` 泛化到 5 世界 3 N；N=1024 最小违约时限。
- **B4**：`uncertainty/streams.py` 预留根守卫（只有 `official_v2.require_protocol_frozen()` 签发的令牌才能创建预留根的流；关闭 SB-05）；`require_protocol_frozen`（protocol_freeze.json 存在且 git 锚定、is_frozen、test_previously_seen、规格指纹 = v2 pin、代码清单子集摘要、数据哈希、access_scope、SD 点/格/根齐全）；`write_protocol_freeze`；官方输出路由 `results/official/<run_id>/` + 受限镜像；作业编排 `run_jobs(workers)`（作业 = (根 k, 格) 或 (根 k, 诊断)，各自重建流，写 `jobs/<job_id>/` + DONE + sha 清单；`--official-merge` 拒绝缺失/重复；表按根分列，无汇总 n）；驱动新模式 `--official`（须 `--authorised-compute` + 授权范围含 `run_official_v2` + 只允许预留根 + `--root-k` + `--workers`）、`--official --dry-run`（实测 0 抽样）、`--official-rehearsal`（开发根、tiny、debug 标签）、`--official-merge`；`check_seed` 官方模式反向；`FREEZE_PIN` → `reference_problem_v2_freeze.json`；`SPEC_FILES` 增 official_v2.py、reference_constraints_v2.csv、protocol.yaml、reference_problem_v2.md、stats.py、safety_margin.py、streams.py；运行记录扩展；`scripts/px_official_prepare.sh`、`scripts/px_official_run.sh`、`scripts/verify_official_outputs.py`。
- **B5 冻结**（负责人 + agent）：Methods/协议定稿；BLOCKERS 更新 B-126/127/129/442/447/452/453；**一次冻结提交**含 protocol.yaml（is_frozen: true、时间戳）、configs/protocol_freeze.json、v2 pin；git bundle 自该提交生成；用户写授权文件。门：Mac `--official --dry-run` = matches_frozen_anchored、协议已冻结、0 抽样。
- **B6 运行**：px prepare → run → merge → 取回 → verify → 数值审计 → 更新 CURRENT_STATE。

## 3 测试（新增）

计划行系数与结构判定；v2 表加载；B-442 交叉一致；methods.yaml = 执行配置；SD 点登记（按模式断言，不印数值）；预留根守卫；官方门控（临时 git 仓：未冻结/篡改/未锚定/test_previously_seen 非 true/pin 不符/授权范围错/开发根 → 退出 2 且 0 抽样）；选参 CP 筛查（spy）；成员 p̄ 与按配方覆盖率；诊断（δ*=0 当不绑定；单调；E1 臂只含能量行；诊断不进集合）；收敛判据；配对成本曲线单调；作业合并；官方彩排（debug）。

## 4 顺序

B1 → B2 → B3 → B4（每批一个 agent，顺序执行；全量测试绿才进下一批）→ B5 冻结（需用户：F1 与 §6 决定、授权文件）→ B6 运行。

## 5 算力（compute host）

- 开发运行实测：6 格（N∈{128,512}）3,180 s；方法块约 10 s/格（PART6P5 下 M2 可行时 491 s）；其余是打到时限的诊断；参考评估 5.6e-5 s/状态。
- 估计每格每根：N{128,512} 约 600 s；加 N=1024：最小违约 360 s + 前沿 ≤ 600 s + 12 个 M2 MILP（不可行约 60 s，可行时可达 3,600 s）；五世界候选评分 ≤ 600 s。典型 0.5–0.8 h，重格 1.5–2 h，最坏 3.7 h。
- 矩阵：根 0：9 格；根 1、2：MAIN9 × 5 点各 5 格；三根各一诊断作业 → 19 格作业 + 3 诊断作业；串行期望 15–17 h，最坏 72 h。
- 并行：每作业一进程、BLAS 单线程；8 worker 期望 2.5–3 h、最坏 9–10 h；4 worker 期望 5 h（最坏超预算 → 冻结时改为根 1、2 只跑 MAIN9 × {SD-H0, SD-S2}，绝不中途裁剪）。冻结前在 px 查 `nproc`、`free -g`；workers = min(8, nproc − 2)。
- 调用：Mac 打 bundle → px `git fetch` 并 checkout 冻结提交 → `bash scripts/px_official_prepare.sh`（回放 Mac 构建字节、pytest、预检、环境检查、`--official --dry-run`）→ `screen -dmS rrs_official bash scripts/px_official_run.sh --workers N`（export OPENBLAS/OMP/MKL_NUM_THREADS=1；写 OFFICIAL_STARTED/EXIT/ENDED）→ Mac `rsync --checksum` 取回公开与受限输出 → `scripts/verify_official_outputs.py`。

## 6 冻结前须写定（研究负责人）

1. F1 域读法；τ = 1/3、2/3 是否作为计划行敏感性格（+2 格）或只作评估变体。
2. 诊断参数：T4/T5 的 δ 单位与尺度、h 目标网格、DIAG-E1 的 N/α 网格、跑哪些根。
3. 成员统计量（test 上 p̄）与覆盖率两种口径。
4. 主 N = 1024 预先固定；收敛判据（每根、每 SD 点、每 α：512→1024 M2 选参状态相同、|Δ validation r̂⁺| ≤ 2 MC SE、成本相对变化 ≤ 1%；否则报“未证实”，不延长）。
5. 求解时限：N=1024 最小违约 360 s；M2 300 s。
6. M1 无 DM 余量变体：完整条目还是敏感性线（P45）。
7. 候选与成本：喂给“最低可达风险”图与配对成本曲线的候选集、成本网格、成本的公开形式（B-420：相对 M0 或只留受限）。
8. 矩阵与 worker 数（查 nproc 后）；validation/test 规模。
9. B-454：比值存放位置（改变规格文件）。
10. B-127：分布族 TN_MM + BETA_MM 回退；C0 为主；C1 是否保留在外；U3 决定。
11. 登记偏离：T2.3、调整幅度、B-428、RQ3 不跑、题目更改。
12. `access_scope`：全部开发资料、已看运行、看过结果后的决定清单。
13. 用户动作：授权文件（主机、范围 `run_official_v2`）、批准把受限输入放到 px、SSH/rsync 授权。

## 7 不允许

改 α/阈值/τ/SD 点/网格/行/容差以求可行；任何求解后增删 SD 点；冻结前从预留根抽样（含测试、smoke、干跑、彩排、“看一眼”）；看结果后换根、加抽样或加 N（因 bug 重跑须同根，旧输出保留并披露）；冻结提交与 protocol_freeze.json 存在之前看官方 test 输出；冻结提交到运行结束之间改任何文件；把根合并成一个 n；把假设范围叫 CI；把诊断或前沿当方法或集合成员；集合外比成本或称“节约”；公开文件印比值、根或受限值。

## 8 规划者不确定的点

决定包第 6 条“validation 上筛查与成员”——成员按 test 上 p̄ 理解；“放宽使之可行”按 δ 余量 LP 定义；px 核数与 scipy/HiGHS 是否多线程；`replay_build_outputs.py` 接口是否与 prepare 脚本假设一致；`protocol.yaml freeze.code_commit_or_hash` 如何避免自引用；是否要 τ 计划行敏感性格。
