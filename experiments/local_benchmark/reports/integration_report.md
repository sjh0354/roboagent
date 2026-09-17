# Integration report

所有结果均为 **SIMULATED PHYSICAL EXECUTION；不是实际机器人性能**。

## 固定版本

| 组件 | 固定版本/提交 | 实际接入 |
|---|---|---|
| RoboAgent | `4ee8399ded2c92ca51b2aac8eac7e436f212b062` + 本地未提交改动 | repo-native `BaseVLMPlanner` |
| 模型 | `kimi-k3` | `/Users/michaelshi/Localwork/test.sh` 中同一 OpenAI-compatible endpoint；脚本只解析、不执行 |
| Python | 3.11.15 | 项目 `.venv` |
| Hermes Agent | v0.19.0；checkout `7651764ce63f44f4e02b1595798e73a67f678ebc`；CLI 内嵌 upstream `1be70d63` | 官方 `AIAgent` 原生循环、stdio MCP，隔离 `HERMES_HOME` |
| OpenClaw | 2026.7.1 (`2d2ddc4`) | 官方 embedded agent、`api.registerTool` native plugin，隔离 state/config |
| Node | 24.16.0 | 项目 `.runtime/node24` |
| A-MEM | `ceffb860f0712bbae97b184d440df62bc910ca8d` | 官方 `AgenticMemorySystem`：note、检索、关联/演化；Kimi transport |
| CaM | `cam_adapted_symbolic` | 非官方/非完整 CaM；公共状态 oracle perception、Kimi 生成受限表达式 |

Python 环境的完整快照见 `requirements-local-benchmark.lock.txt`。

## 当前协议工程证据

- 原 0.70、稳定 SHA 配对和 phase/timeout 执行器均已被聚合协议取代，仅保留作历史审计。当前无 monitor `p=0.60` 的 2000 次纯随机校准成功 1208 次（0.604）；有 monitor `p=0.85` 成功 1720 次（0.860），均在三倍标准差容差内。
- 当前契约测试验证：每个规范化具身目标只做一次原子判定；显式物理重试为 0；重复目标不会再次随机；Hermes/OpenClaw 只需调用一次 start_skill 并检查返回图像/状态。
- 下列原生 runtime smoke 早于这次随机性/终止协议修正，只是连通性证据，成绩不进入新表：OpenClaw `corrected_native_smoke_v1`、Hermes `hermes_smoke_v7`、A-MEM/CaM 组合 smoke。
- Hermes 原生：`hermes_smoke_v7`，官方 AIAgent 发现 MCP 工具并完成多步任务。
- Hermes+A-MEM：`hermes_amem_smoke_v3` 有真实写入；`hermes_amem_cross_task_v1` 从前一任务加载 1 条 note、检索并触发新 note 的关联/演化。
- Hermes+adapted CaM：`corrected_cam_smoke_v3` 中生成代码通过正常轨迹探针并执行；seed 17 检出异常中间态但恢复失败，任务成绩为 0，原始负面结果保留。
- Hermes+A-MEM+adapted CaM：`hermes_amem_cam_smoke_v1` 成功；A-MEM 检索/写入真实发生，CaM 初稿因正常轨迹错误被拒绝，第二稿执行 4 次且无误停。
- OmniAct：planner 审计日志含实际 PNG SHA-256 与 API 返回的 multimodal image token；Hermes/OpenClaw 的原生 session/tool result 也保存图像内容而非路径字符串。

## 隔离边界

- 私有评分、`planned_success` 和 `failure_mode` 只存在于世界私有对象/trace；MCP/HTTP 工具不暴露。
- 不生成或保存可控物理 RNG seed；公开 repeat id 不进入 RNG。私有 trace 只保存实际随机 draw 和结果，供审计而非复播。
- 正式执行不再含运行中 VLA、cancel 或共同超时；start_skill 原子解析后立即返回结果图像/公共状态。
- 每个主表 episode 有独立世界、输出目录、Hermes home/OpenClaw state 和 OmniAct memory root。
- A-MEM 主表 episode 默认重置；跨任务 smoke 明确使用共享 store，单独报告，不混入独立 episode 主表。
- 数字工具只改变本地模拟状态；没有真实消息、账户或设备副作用。
