# Robot Agent System

This repository contains a modular embodied-agent system for two hardware profiles:

- `Unitree-G1` humanoid planner
- `UR5e` robotic arm planner

The current codebase has been refactored from large in-code prompt templates into
a modular runtime built around:

- `agent/` for identity, soul, skills, and long-term memory
- shared planner runtime
- shared vision executor runtime
- action schema and dynamic skill resolution

## Requirements

- Python 3.8+
- `google-genai` access configured through environment variables
- camera dependencies only if you want real hardware mode
  - `pyrealsense2` for UR5e RealSense flow
  - OpenCV / camera dependencies for humanoid camera flow

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key:

```bash
export GENAI_API_KEY='your-api-key'
```

If your environment uses a compatible custom endpoint, you can also set:

```bash
export GENAI_BASE_URL='your-compatible-endpoint'
```

## Current Launch Modes

Both planners now default to `real` mode.

- no extra flag: real hardware mode
- `--simulation`: simulation mode using local images
- `--log`: detailed logs
- `--transport`: choose `voice`, `lark`, `openclaw_lark`, or `none`

### Humanoid Planner

Real mode:

```bash
python planner/humanoid_planner_vlm.py
```

Simulation mode:

```bash
python planner/humanoid_planner_vlm.py --simulation
```

Simulation mode with full logs:

```bash
python planner/humanoid_planner_vlm.py --simulation --log
```

Simulation mode using the native Lark gateway:

```bash
python server/lark_gateway.py

python planner/humanoid_planner_vlm.py \
  --simulation \
  --transport lark \
  --lark-target 'group:oc_xxx'
```

### Arm Planner

Real mode:

```bash
python planner/arm_planner_vlm.py
```

Simulation mode:

```bash
python planner/arm_planner_vlm.py --simulation
```

Simulation mode with full logs:

```bash
python planner/arm_planner_vlm.py --simulation --log
```

Simulation mode using the native Lark gateway:

```bash
python server/lark_gateway.py

python planner/arm_planner_vlm.py \
  --simulation \
  --transport lark \
  --lark-target 'group:oc_xxx'
```

## Interaction Transports

The planner input/output path is now transport-based instead of being tied only
to human speech.

Supported modes:

- `voice`: compatibility mode using `FunASRManager`
- `lark`: read and send messages through the project-native Lark gateway server
- `openclaw_lark`: legacy compatibility mode using the local `openclaw` CLI
- `none`: keyboard-only interaction

Relevant environment variables:

```bash
bash scripts/init_env.sh

source env/common.env.sh
source env/planner.g1.env.sh
```

The new remote inter-agent action is `send_agent_message`. Local human-facing
speech remains `speak`.

When using native `lark` transport:

- incoming human messages are normalized to `human: ...`
- known robot senders can be mapped with `LARK_AGENT_SENDER_MAP`
- local `speak` responses are mirrored back into the current group chat
- group-chat messages are only consumed when the bot is `@` mentioned
- `send_agent_message(recipient=...)` sends a structured message like `@ur5e [agent:g1] [to:ur5e] ...`

Environment layout:

- `env/*.example.sh`: checked-in templates
- `env/common.env.sh`: shared model keys and common services
- `env/planner.g1.env.sh`: G1 planner-side interaction config
- `env/planner.ur5e.env.sh`: UR5e planner-side interaction config
- `env/planner.env.sh`: backward-compatible default that currently points to G1
- `env/lark.env.sh`: native Lark gateway and bridge config such as `LARK_ACCOUNT_IDS`, bot `APP_ID`, `APP_SECRET`
- `apikey.sh`: compatibility entry that sources all three files above

## Native Lark Gateway

Run the project-native gateway with:

```bash
cd server
pip install -r requirements.txt

source ../env/common.env.sh
source ../env/lark.env.sh
python lark_gateway.py
```

For WebSocket-based event subscription, start the bridge in another terminal:

```bash
source env/common.env.sh
source env/lark.env.sh
node server/lark_ws_bridge.js
```

For a single detached background session, use:

```bash
bash scripts/start_lark_tmux.sh
```

Stop it with:

```bash
bash scripts/stop_lark_tmux.sh
```

For a one-command G1 demo session that starts the Lark stack and the humanoid
planner together:

```bash
bash scripts/start_g1_demo_tmux.sh
```

This defaults to:

```bash
python planner/humanoid_planner_vlm.py --simulation --log
```

You can pass custom planner args after the session name:

```bash
bash scripts/start_g1_demo_tmux.sh ras-g1-real --log
```

Stop it with:

```bash
bash scripts/stop_g1_demo_tmux.sh
```

For the UR5e side, use:

```bash
bash scripts/start_ur5e_demo_tmux.sh
```

Stop it with:

```bash
bash scripts/stop_ur5e_demo_tmux.sh
```

Multi-bot configuration is environment-based.

Single bot:

```bash
export LARK_APP_ID='cli_xxx'
export LARK_APP_SECRET='xxx'
export LARK_VERIFICATION_TOKEN='xxx'
export LARK_DEFAULT_ACCOUNT='default'
```

Multiple bots:

```bash
export LARK_ACCOUNT_IDS='g1,ur5e'

export LARK_G1_APP_ID='cli_xxx'
export LARK_G1_APP_SECRET='xxx'
export LARK_G1_VERIFICATION_TOKEN='xxx'

export LARK_UR5E_APP_ID='cli_yyy'
export LARK_UR5E_APP_SECRET='yyy'
export LARK_UR5E_VERIFICATION_TOKEN='yyy'
```

Preferred mode for this project:

- Feishu/Lark persistent WebSocket connection via `server/lark_ws_bridge.js`

Optional fallback mode:

- webhook callback to `http://<host>:18889/webhook/g1`
- webhook callback to `http://<host>:18889/webhook/ur5e`

Current limitations:

- encrypted event payloads are not implemented yet; keep Feishu event encryption disabled for now
- the WebSocket bridge currently depends on `@larksuiteoapi/node-sdk`; by default it reuses the copy bundled inside your OpenClaw installation

## What Simulation Mode Does

Simulation mode uses files under `simulation_images/` instead of real camera input.

This is the safest way to test:

- prompt assembly
- planner JSON generation
- end-to-end task flow
- memory candidate writing
- CLI interaction behavior

## Current Architecture

### Agent Layer

`agent/` contains modular agent definitions:

- `agent/bootstrap.md`
- `agent/soul.md`
- `agent/profiles/...`
- `agent/skills/...`
- `agent/memory/...`

### Prompt Runtime

Key files:

- `template/modular_prompt_loader.py`
- `utils/action_registry.py`
- `utils/skill_resolver.py`

Behavior:

- runtime system prompts are assembled from modular markdown files
- relevant skills are loaded dynamically
- action validation uses a shared schema

### Planner Runtime

Key files:

- `planner/base_vlm_planner.py`
- `planner/humanoid_planner_vlm.py`
- `planner/arm_planner_vlm.py`

Behavior:

- shared half-open-loop execution flow
- profile-specific observation and execution behavior
- memory candidate writing at interaction boundaries

### Executor Runtime

Key files:

- `executor/vision_enabled_mixin.py`
- `executor/humanoid_executor_vision.py`
- `executor/arm_executor_vision.py`

Behavior:

- shared visual observation enhancement
- image-based scene description
- simulation image management
- hardware-specific camera and actuation handling

## Memory

The system now distinguishes three memory layers:

- session memory: conversation context only
- hardware memory: embodiment-specific stable lessons
- global memory: long-lived cross-agent lessons

At runtime, new memory candidates are written into:

- `agent/memory/inbox/hardware/`
- `agent/memory/inbox/global/`

These inbox files are intentionally separate from the stable memory files.

## Useful Commands

Show CLI help:

```bash
python planner/humanoid_planner_vlm.py --help
python planner/arm_planner_vlm.py --help
```

Syntax-check key files:

```bash
python -m py_compile \
  planner/base_vlm_planner.py \
  planner/humanoid_planner_vlm.py \
  planner/arm_planner_vlm.py \
  executor/vision_enabled_mixin.py \
  executor/humanoid_executor_vision.py \
  executor/arm_executor_vision.py
```

## Notes

- Real online image analysis and planner generation have been validated in a network-enabled environment.
- In restricted sandbox environments, Gemini calls may still fail due to DNS/network policy.
- If you want the most recent refactor summary and testing status, see:

`document/REFACTOR_HANDOFF_2026-03-10.md`
