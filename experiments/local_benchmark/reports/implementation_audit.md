# Implementation audit

更新日期：2026-09-17。本文只描述本地代码与模拟评测工程，不代表真实机器人性能。

## 基线与保护

- 仓库：`/Users/michaelshi/Localwork/roboagent`
- 远端：`https://github.com/sjh0354/roboagent.git`
- 分支：`demo/reading-setup-base-20260602`
- 原始 SHA：`4ee8399ded2c92ca51b2aac8eac7e436f212b062`
- 开始工作时工作树干净；新增与修改均留在本地，未 reset、覆盖、提交或推送用户工作。
- 没有启动 UR5e、G1、Keenon、PI0 Docker 或真实 IoT/外部账户动作。

## 原仓库已有并已核对

- `planner/base_vlm_planner.py` 是共享逐步规划循环；`none` 关闭层次记忆与主动压缩，但保留普通 conversation/working history。
- 原实现遇到未知 `OMNICLAW_MEMORY_MODE` 会回退 `full`，不适合消融的 fail-closed 要求。
- PI0 的真实监管链位于 `executor/arm_executor_vision.py`：`_execute_pi0_script` → `_wait_for_pi0_action_completion` → `_assess_pi0_action_progress` → `_stop_pi0_session`；顶层空 `_check_interrupt()` 不等于没有监管。
- `POST_ACTION_VLM_COMPARE` 位于 `planner/arm_planner_vlm.py`，与 `PI0_ACTION_VLM_MONITOR` 是不同控制点。
- 成功过程帧受 `TRANSIENT_MEMORY_ON_SUCCESS`、动作时长和帧数条件限制。
- VLA 不可用时原执行器可回退 recorded replay；因此本实验完全绕开真实执行入口并锁定新的模拟 backend。
- 原仓库的 `openclaw_lark` 是消息传输兼容层，不是 OpenClaw agent baseline。
- G1 导航入口是 Unitree/G1 执行代码，不能当作论文 Keenon 的完整实现。

## 本轮修正

- 未知记忆模式现在明确抛错，不再静默回退。
- memory candidate inbox 支持 `AGENT_MEMORY_INBOX_ROOT`，每个 episode 隔离。
- 层次记忆上下文支持独立关闭 episodic 或 reflective 层，用于 `−episodic`、`−reflective`。
- 新增公共、有状态、部分可观测的合成 RGB-D 世界；私有随机结果和评分不进入 agent 工具。
- 当前正式协议不再模拟 VLA phase、超时或逐次重试。底层单次成功率假设为 0.50；monitor 最多两次恢复及耗时被直接折算为聚合 `p=0.85`，无 monitor 的模型自检/打断折算为聚合 `p=0.60`。
- 每个满足前置条件的规范化具身目标只用 `SystemRandom` 纯随机判定一次并立即写入可观察状态；公开重复编号 `0,1,2` 不进入 RNG，也不存在可控/可复播的物理 seed。
- 同一目标再次调用会返回 `aggregate_probability_already_consumed_for_goal`，不再次掷骰子，避免把已聚合的 60%/85% 重复累计。非法动作同样不掷骰子。
- 失败仍可表现为无效果、部分完成或可恢复停滞，但它们是一次性结果状态，不再展开过程帧控制。
- 新增 MCP 与 loopback HTTP 两种薄传输；二者使用同一个 `StatefulWorld` 与同一评分器。
- 新增 repo-native `BaseVLMPlanner` 适配、官方 Hermes AIAgent+MCP、官方 OpenClaw embedded agent+native plugin。
- 新增普通 TF-IDF chunk RAG、同模型摘要、Raw/Sliding、时机、单帧/时序帧和层次记忆消融。
- Agent 与记忆相关模型请求均锁定 `kimi-k3`，没有 fallback model；正式聚合协议不再额外调用在线视觉 monitor。瞬时网络错误最多重试三次，每次尝试都写原始审计日志。

## 新增而非原仓库既有

- `experiments/local_benchmark/` 下的模拟器、40 任务 manifest、框架适配器、评估器、矩阵启动器与报告均为本轮新增。
- 40 个正式任务是独立本地合成任务集，不是恢复出的 CoRL/论文硬件 40 任务。
- OpenClaw baseline 是本轮新增的官方 runtime 插件适配，不是原 `openclaw_lark`。
- Hermes+A-MEM 使用官方 A-MEM 源码；Hermes 原生行未注入 OmniAct 计划、摘要或监管结论。

## 仍缺外部条件

- Code-as-Monitor 官方实现和其完整视觉/跟踪依赖未取得。本轮 `cam_adapted_symbolic` 使用公共状态 oracle perception、受限表达式与 Kimi 生成代码，只能作为诊断变体，不能替代完整 CaM。
- 网关没有返回可核对的账单价格，成本字段只能报告 tokens/调用次数和 `cost_status=unknown`，不能把 runtime 返回的零美元当真实费用。
- 真实机器人、安全系统、摄像机标定误差、动力学与真实网络/IoT 均未评测。
