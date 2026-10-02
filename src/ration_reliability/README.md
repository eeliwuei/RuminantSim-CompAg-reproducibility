# 求解与共同评价核心

本目录包括单位与数据结构、约束编译、M0/M1/M2/M3 求解、不确定性模型工厂、共同评价与参考能量判定、信息价值探索模块。它不依赖旧投稿代码的保存分数。

当前入口为项目根目录 `README.md` 与 `CURRENT_STATE.json`。旧的“尚未实现/没有 pilot/387 项测试”说明已移入 `review_20260926/history/`，不能继续用作当前状态。

## 接口

- `io.load_problem`：加载并验证有来源/状态标识的案例。
- `optimization.get_method`：获取命名求解方法。
- `uncertainty`：命名随机流、模型工厂、分布与先验信息。
- `evaluation.evaluate_drawset`：相同原物质投料决策的共同基础判定。
- `evaluation.reference.evaluate_reference`：领域条件、参考能量及多个明确命名的风险事件。
- `evaluation.stats`：有限值、整数计数与概率验证后的 Monte Carlo 区间。
- `information`：政策类别与价值定义必须显式选择；启发式差额不作默认检测支付上限。

代码使用例可直接运行 `python scripts/run_synthetic_demo.py`。该示例全为合成测试，不是日粮建议。

2026-09-26 修复了统计边界和回放输入检查：非法概率、非整数计数、无穷值和重复保存键不再被静默当作有效结果。合法输入的原统计定义、营养阈值、风险目标与来源选择未改变。详见本轮报告与测试日志。
