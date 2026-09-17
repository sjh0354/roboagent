# Experiment report

状态：正式矩阵尚未开始。所有 phase/timeout 版 pilot 已作废；只有聚合概率协议的新 pilot 通过后才会启动正式矩阵。

全部结果均为 **SIMULATED PHYSICAL EXECUTION；NOT REAL ROBOT PERFORMANCE**。模型固定为 `kimi-k3`。底层单次物理成功率假设为 0.50；有 monitor 的方法把最多两次恢复和耗时直接折算为 0.85，无 monitor 方法按 0.60。每个规范化具身目标只用不可设 seed 的系统随机数判定一次，不模拟 phase、timeout 或显式物理重试。`0,1,2` 仅是三次重复编号。

## 执行器校准（非 agent 成绩）

所有旧校准均仅作历史审计。聚合协议各执行 2000 次有效调用：无 monitor 成功 1208 次、失败 792 次，观测率 0.604；有 monitor 成功 1720 次、失败 280 次，观测率 0.860。二者均处于各自三倍标准差容差内，显式物理重试均为 0。

## 协议 pilot

`formal_pilot_v7_p60_r2_s3` 与 `formal_pilot_v8_p60_r2_s3_independent_timeout` 均失效，不报告分数。

新 pilot 目录为 `formal_pilot_v9_p60_p85_atomic_random`：2 个任务 × 4 个核心方法 × 3 次重复，共 24 episode。通过条件包括零非 completed、零重复键、每个有效物理调用恰好一个原子判定、显式物理重试为 0、无 monitor 行使用 0.60、有 monitor 行使用 0.85、repeat id 不控制 RNG。它只是协议验证，不能当作 40 任务结论。

当前进度为 6/24、零协议违规：`iterative_posthoc` 在一个任务的 3 次重复中成功 1 次，`iterative_monitor` 成功 2 次。该 3-repeat 小样本只验证概率分配和数据链路，不解释为稳定方法差异。

## 旧工程 smoke（不与正式表合并）

- 下列 smoke 均早于本次终止协议修正，只证明各 runtime/插件曾真实连通，不提供可复用成绩。
- OpenClaw 原生 seed 17：任务成功；4 个有效物理调用中 2 个语义失败，1 次重试；24 次 Kimi 调用，388,773 native-reported aggregate tokens。
- Hermes+A-MEM 跨任务：第二任务加载 1 条 note，检索得分 0.3366，并在新增 note 时触发关联/演化调用。
- Hermes+adapted CaM seed 17：监控在异常中间态触发 3 次中断，但 Hermes 未恢复，任务失败。这是方法负面结果。
- Hermes+A-MEM+adapted CaM seed 0：任务成功；A-MEM 1 次写入模型调用；CaM 生成 2 次调用，运行期 4 次评估、0 次触发。

## 正式矩阵

新 pilot 通过后，正式核心矩阵目标为 40 任务 × 4 方法 × 3 次重复 = 480 episode，目录为 `core_ablation_p60_p85_atomic_random_v1`。不使用任何旧 pilot 数字补齐。

正式系统主表另含 Hermes、OpenClaw、Hermes+A-MEM、Hermes+adapted CaM、Hermes+A-MEM+adapted CaM，共 40 × 5 × 3 = 600 episode；其中 adapted CaM 两行只列诊断结果，并明确“不等价于完整 CaM”。

机制矩阵使用冻结的 8 任务子集，12 方法 × 8 任务 × 3 次重复 = 288 episode：每个领域含 1 个低复杂度、1 个中复杂度和 2 个高复杂度任务。比较 timing、单帧/时序帧、Raw/Sliding、同模型摘要、TF-IDF RAG、本文 memory、−episodic、−reflective 和 A-MEM。

## 费用口径

只报告真实 API 尝试数、输入/输出/cache/reasoning/total token 与墙钟时间。provider 未提供可验证费率或账单，因此金额统一为 unknown，不把 runtime 的 `estimated_cost_usd=0` 当作零成本。
