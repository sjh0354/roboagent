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
