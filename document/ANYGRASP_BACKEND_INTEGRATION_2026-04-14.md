# AnyGrasp 后端集成说明（2026-04-14）

本文档整理本次将 UR5e `pick_and_place` 执行后端从纯 PI0/VLA 路径扩展为“可切换 AnyGrasp 传统抓取路径”的改动。

## 1. 目标

- 保留当前可用的 `pi0` 执行链路，避免破坏现有流程。
- 新增 `anygrasp` 执行链路，使 `pick_and_place` 可走传统抓取方案。
- 把 AnyGrasp 与具体机械臂执行解耦：
  - AnyGrasp 负责抓取位姿检测。
  - UR 执行器脚本负责将抓取位姿转换为真实机械臂动作。

## 2. 改动概览

### 2.1 执行器新增后端切换

文件：
- `executor/arm_executor_vision.py`

新增环境变量：
- `ARM_ACT_BACKEND=pi0|anygrasp`（默认 `pi0`）

行为：
- 当 `simulation_mode=False` 且 `action_type=act` 时：
  - `ARM_ACT_BACKEND=pi0`：沿用原 `_execute_pi0_script(...)`。
  - `ARM_ACT_BACKEND=anygrasp`：走 `_execute_anygrasp_action(...)`。

### 2.2 AnyGrasp runner 调用协议

文件：
- `executor/arm_executor_vision.py`

调用方式：
- 执行器调用：
  - `$ANYGRASP_RUNNER_CMD --request-json '<JSON_PAYLOAD>'`
- 若未配置 `ANYGRASP_RUNNER_CMD`，会默认尝试：
  - `python utils/anygrasp_runner.py`

请求 JSON（关键字段）：
- `action`
- `item_name`
- `source`
- `target`
- `instruction`
- `observation_image`（执行前快照路径，若可获取）
- `timestamp`

runner 输出要求：
- 最后一行 stdout 必须是 JSON 对象：
  - `success: bool`
  - `feedback: str`
  - `error: str|null`
  - `data: dict`

### 2.3 新增 AnyGrasp 可执行 runner

文件：
- `utils/anygrasp_runner.py`

功能：
- 解析 `--request-json`
- 加载 AnyGrasp SDK
- 读取 RGB 与 depth（depth 支持 `.png`/`.npy`）
- 依据相机内参构建点云
- 调用 AnyGrasp 生成抓取候选并选最优抓取
- （可选）调用 UR 执行器命令完成真实抓取放置
- 返回统一 JSON 结果给 arm executor

### 2.4 新增 UR 执行器模板

文件：
- `utils/ur_grasp_executor_template.py`

用途：
- 接收 AnyGrasp 输出的 best grasp。
- 由你填充真实控制逻辑（RTDE/MoveIt/自研控制器）。

### 2.5 保留模板 runner

文件：
- `utils/anygrasp_runner_template.py`

用途：
- 作为更简化的协议示例。

### 2.6 README 更新

文件：
- `README.md`

新增内容：
- 后端切换说明（`pi0`/`anygrasp`）。
- AnyGrasp runner 调用协议。
- AnyGrasp 推荐环境变量配置。
- UR 执行器命令配置示例。

## 3. 新增/更新环境变量

### 3.1 执行后端选择

- `ARM_ACT_BACKEND`
  - `pi0`（默认）
  - `anygrasp`

### 3.2 AnyGrasp runner 层

- `ANYGRASP_RUNNER_CMD`
- `ANYGRASP_RUNNER_TIMEOUT`
- `ANYGRASP_SDK_ROOT`
- `ANYGRASP_CHECKPOINT_PATH`（必需）
- `ANYGRASP_COLOR_PATH`（可选，request 无 RGB 时兜底）
- `ANYGRASP_DEPTH_PATH`（通常必需）
- `ANYGRASP_INTRINSICS_JSON`
- `ANYGRASP_LIMS_JSON`
- `ANYGRASP_COLLISION_DETECTION`
- `ANYGRASP_DENSE_GRASP`
- `ANYGRASP_APPLY_OBJECT_MASK`
- `ANYGRASP_TOP_K`

### 3.3 UR 执行层

- `UR_GRASP_EXECUTOR_CMD`
- `UR_GRASP_EXECUTOR_TIMEOUT`
- `ANYGRASP_REQUIRE_UR_EXECUTION`

## 4. 典型运行方式

```bash
export ARM_ACT_BACKEND=anygrasp
export ANYGRASP_RUNNER_CMD='python /media/user/B29202FA9202C2B91/RAS_interactive_planner/utils/anygrasp_runner.py'
export ANYGRASP_RUNNER_TIMEOUT=240

export ANYGRASP_SDK_ROOT='/abs/path/to/anygrasp_sdk'
export ANYGRASP_CHECKPOINT_PATH='/abs/path/to/checkpoint.tar'
export ANYGRASP_DEPTH_PATH='/abs/path/to/depth.png'   # 或 depth.npy
export ANYGRASP_INTRINSICS_JSON='{"fx":927.17,"fy":927.37,"cx":651.32,"cy":349.62,"scale":1000.0}'
export ANYGRASP_LIMS_JSON='[-0.19,0.12,0.02,0.15,0.0,1.0]'

export UR_GRASP_EXECUTOR_CMD='python /media/user/B29202FA9202C2B91/RAS_interactive_planner/utils/ur_grasp_executor_template.py'
export ANYGRASP_REQUIRE_UR_EXECUTION=1

python planner/arm_planner_vlm.py --log
```

## 5. 调用链路（真实模式）

1. `planner/arm_planner_vlm.py` 生成 `next_step=pick_and_place`
2. `executor/arm_executor_vision.py` 根据 `ARM_ACT_BACKEND` 分流
3. `anygrasp` 分支组装 request payload 并调用 runner
4. `utils/anygrasp_runner.py` 输出统一 JSON
5. executor 将结果封装为 `ExecutionResult` 回传 planner
6. vision mixin 继续执行动作后观测采集并推进下一轮规划

## 6. 当前已知限制

- 本仓库当前未直接提供 AnyGrasp 依赖与模型权重，需外部准备：
  - `open3d`
  - AnyGrasp SDK
  - checkpoint
- 目前 planner 侧默认只提供彩色图路径；AnyGrasp 实际通常需要 depth 输入，因此需要通过：
  - request 中提供 `depth_path`，或
  - `ANYGRASP_DEPTH_PATH` 环境变量兜底
- `ur_grasp_executor_template.py` 仍需按你的真实机器人控制栈实现动作执行。

## 7. 回滚方式

若 AnyGrasp 路径不稳定，可立即切回原路径：

```bash
export ARM_ACT_BACKEND=pi0
```

无需改代码。

## 8. 本次涉及文件清单

修改：
- `executor/arm_executor_vision.py`
- `README.md`

新增：
- `utils/anygrasp_runner.py`
- `utils/anygrasp_runner_template.py`
- `utils/ur_grasp_executor_template.py`
- `document/ANYGRASP_BACKEND_INTEGRATION_2026-04-14.md`
