# Limitations

本项目的全部数字都必须标注为 **模拟物理执行（simulated physical execution）**，不能外推为真实机器人结果。

1. **任务来源**：正式 40 任务是本地独立合成集，不是 CoRL/论文硬件任务的恢复或复现。领域与复杂度结构匹配 20 manipulation + 20 navigation、每域 4/8/8，但场景、动力学和目标不同。
2. **物理真实性**：世界是离散有状态模拟器，RGB-D 是状态一致的合成渲染；没有接触动力学、相机噪声、标定误差、控制延迟或安全控制器。
3. **60%/85% 的含义**：二者是用户指定的聚合建模假设，不是从本模拟器学得的实证成功率。底层单次 0.50、monitor 最多两次恢复和耗时不再逐项模拟；monitor-present 方法直接使用 0.85，monitor-absent 方法直接使用 0.60。
4. **一次性物理判定**：每个规范化具身目标只随机一次，立即产生结果状态；显式 phase、timeout、stop 和物理 retry 均不存在。同目标重发会被拒绝，避免在聚合概率上再叠加成功机会。因此结果不用于分析监控时机或真实控制延迟。
5. **随机性**：物理判定使用不可设 seed 的 `SystemRandom`；公开 `0,1,2` 只是三次重复编号。实验不能按相同物理轨迹逐位复播，也不进行跨方法配对随机比较。
6. **CaM 复现等级**：正式 `hermes_cam`/`hermes_amem_cam` 分数只应用 monitor-present 0.85 假设，不测量 CaM 的实际检测质量。早期 `cam_adapted_symbolic` 运行证据仍因 oracle perception 和缺少官方依赖而不是完整 CaM，只能单列工程诊断。
7. **原 timing/vision 消融解释变化**：在聚合一次性执行下，在线/延迟/终点监控与单帧/时序帧不再对应真实 phase 级控制差异；相关行只能解释为预设 monitor 类别和 agent 上下文差异，不能声称测得真实监控时机效应。
8. **成本**：gateway/runtime 返回 token 与调用数，但没有可验证费率或账单。报告不把 `estimated_cost_usd=0` 解释成零成本。
9. **框架差异**：Hermes、OpenClaw 与 repo-native planner 保留各自原生 loop、system context 和上下文压缩，因而 token 成本并不相同；比较是系统级而非只换 prompt。
10. **A-MEM 会话边界**：独立 episode 主表重置记忆以防跨任务顺序泄漏；跨任务记忆使用在单独的长会话/工程验证中报告。因此主表的 A-MEM 行主要验证系统开销和 episode 结束写入，不单独证明跨任务收益。
11. **基础设施错误**：瞬时 timeout、SSL EOF 和 provider 错误按固定的最多三次重试处理，每次尝试都计数；耗尽后保留为 infrastructure/agent error，不按成绩重跑直到成功。
12. **原仓库回归**：新增 benchmark 契约测试通过；原仓库相关回归仍有 5 个既有记忆规则/隔离断言失败，另有 1 个 PI0 测试因本机无 `docker` 可执行文件失败。这些不被伪装为本轮通过，也未为模拟实验去修改真实硬件代码。
13. **统计能力**：用户将正式重复数改为每任务/方法 3 次（标签 `0,1,2`）；任务级区间会更宽。Wilson 区间按 episode 报告，同时保留 task/domain 分组，不能把小 pilot 的区间当正式结论。
