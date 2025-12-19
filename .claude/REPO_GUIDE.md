# Interactive Planner - Repository Guide

**Quick Context**: This is a robot planning system with TWO modes:
1. **Interactive Mode**: Human-in-the-loop planning with text-based feedback
2. **VLM Autonomous Mode**: Vision-based autonomous planning with direct image observations (NEW!)

## What This Project Does

An **Interactive Robot Planning Framework** that operates differently from traditional batch planners:
- **Plans ONE step at a time** (not entire sequences upfront)
- **Waits for human feedback** after each step execution
- **Adapts dynamically** based on real-world results
- **Maintains conversation history** for context-aware planning

Think: "Plan → Execute → Get Feedback → Plan Next" in a loop until task completion.

## Planning Modes

### VLM Autonomous Mode (Vision-Based) ⭐ NEW
- **Visual observations** - VLM sees images directly
- **Fully autonomous** execution until completion
- **Vision-Language Model** (qwen-vl-plus/max)
- **Human interaction** only for: task requests, clarifications, completions
- Use for: routine tasks, maximum automation

## Key Files & Their Roles

### Core Planning (Main Classes)

**VLM Autonomous Mode (Vision-Based):** ⭐ NEW

*Humanoid Robot:*
```
humanoid_planner_vlm.py (576 lines)
├─ AutonomousVLMPlanner - VLM-based autonomous planner
├─ start_new_task(request, run_autonomously=True) - Start autonomous execution
├─ provide_human_response(response) - Answer clarification questions
└─ _run_autonomous_loop() - Fully autonomous execution until complete

humanoid_executor_vision.py (200+ lines)
├─ VisionEnabledExecutor - Executor with visual observation
├─ execute_action() - Execute with automatic image capture
└─ get_current_observation() - Get current observation image + state
```

*Robotic Arm:* ⭐ NEW
```
arm_planner_vlm.py (570+ lines)
├─ AutonomousArmVLMPlanner - VLM-based autonomous arm planner
├─ start_new_task(request, run_autonomously=True) - Start autonomous execution
├─ provide_humanoid_response(response) - Answer humanoid clarifications
└─ _run_autonomous_loop() - Fully autonomous execution until complete
```

*Shared Vision Components:*
```
qwen_vlm_client.py (350+ lines)
├─ QwenVLMClient - Qwen VLM API client
├─ analyze_image() - Analyze single image with query
├─ compare_images() - Compare before/after for verification
└─ get_observation_description() - Get detailed scene description

simulation_image_manager.py (300+ lines)
├─ SimulationImageManager - Manages observation images for simulation
├─ update_state() - Update robot state
├─ get_current_observation_image() - Get image for current state
└─ register_custom_image() - Add custom observation images
```

### LLM Prompt Templates (System Instructions)

**Interactive Mode:**
```
humanoid_prompt_template_interactive.py (488 lines)
└─ get_system_prompt() - Returns LLM system prompt for humanoid planning
   ├─ Scene descriptions (Room 01=Home, Room 02=Store)
   ├─ Available actions (talk/tool/act/sense)
   ├─ Output format (JSON schema)
   └─ Planning guidelines with examples

arm_prompt_template_interactive.py (675 lines)
└─ get_system_prompt() - LLM system prompt for robotic arm planning
```

**VLM Autonomous Mode:** ⭐ NEW
```
humanoid_prompt_template_vlm.py (600+ lines)
└─ get_humanoid_vlm_system_prompt() - VLM system prompt for humanoid
   ├─ Vision-based planning instructions
   ├─ Visual verification guidelines
   ├─ Autonomous execution protocol
   └─ Examples with visual observations

arm_prompt_template_vlm.py (500+ lines)
└─ get_arm_vlm_system_prompt() - VLM system prompt for arm
   ├─ Vision-based planning for robotic arm
   ├─ Workspace observation guidelines
   ├─ Manipulation action specifications
   └─ Examples with store/warehouse scenarios
```

### Testing & Examples


**VLM Autonomous Mode:** ⭐ NEW
```
test/test_vlm_planner.py - Humanoid VLM planner test suite
test/test_arm_vlm_planner.py - Arm VLM planner test suite
```

### Documentation

**Core Documentation:**
```
document/README.md - Complete system documentation
document/IMPLEMENTATION_SUMMARY.md - Architecture overview
document/HARDWARE_INTEGRATION_GUIDE.md - Real hardware integration
document/README_EXECUTOR.md - Executor-specific docs
```

**VLM-Specific Documentation:** ⭐ NEW
```
VLM_PLANNER_README.md - Complete VLM planner guide
VISION_BASED_ARCHITECTURE_DESIGN.md - VLM architecture design document
```

## Quick Start

### Interactive Mode (Text-Based)

**Test Without API:**
```bash
python test/test_interactive_mock.py
```

**Run With Real API:**
```bash
export DASHSCOPE_API_KEY='your-key-here'
python humanoid_planner_interactive.py
```

### VLM Autonomous Mode (Vision-Based) ⭐ NEW

**Test System:**
```bash
python test/test_vlm_planner.py
```

**Run VLM Planner:**
```bash
export DASHSCOPE_API_KEY='your-key-here'
python humanoid_planner_vlm.py
```

**Programmatic Usage:**
```python
from humanoid_planner_vlm import AutonomousVLMPlanner

planner = AutonomousVLMPlanner(simulation_mode=True)
result = planner.start_new_task("I'm feeling cold")
# Runs autonomously until complete!
```

**Provide Custom Images:**
```bash
# Place images in simulation_images/ directory:
simulation_images/home/ac_on_22.jpg    # For humanoid
simulation_images/home/light_on.jpg    # For humanoid
simulation_images/store/default.jpg    # For arm
simulation_images/store/picked_water.jpg  # For arm
# Or register programmatically:
planner.register_observation_image("home_ac_on_22", "/path/to/image.jpg")
```

### Robotic Arm VLM Mode ⭐ NEW

**Test System:**
```bash
python test/test_arm_vlm_planner.py
```

**Run Arm VLM Planner:**
```bash
export DASHSCOPE_API_KEY='your-key'
python arm_planner_vlm.py
```

**Programmatic Usage:**
```python
from arm_planner_vlm import AutonomousArmVLMPlanner

planner = AutonomousArmVLMPlanner(simulation_mode=True)
result = planner.start_new_task("Get water for humanoid")
# Runs autonomously until complete!
```

## Architecture Overview

### Closed-Loop Flow
```
Human Request
    ↓
┌───────────────────────────────┐
│ 1. Plan Single Step (LLM)    │ ← Context: Request + History
│ 2. Execute Step (Executor)   │
│ 3. Wait for Feedback (Human) │
│ 4. Update History             │
└───────────────────────────────┘
    ↓ (repeat until task_complete)
Task Complete
```

### Context Memory (Cumulative)
Each planning call receives:
1. **Original Request** - The initial task goal
2. **Execution History** - All completed steps with timestamps
3. **Conversation History** - Full LLM message chain

### LLM Response Format
```json
{
  "current_step_analysis": {
    "task_progress": "What's been done",
    "current_situation": "Current state",
    "next_action_reasoning": "Why this action"
  },
  "next_step": {
    "step_number": 2,
    "agent": "Unitree-G1 humanoid_robot",
    "action": "control_air_conditioner",
    "action_type": "tool",
    "parameters": {"mode": "cooling", "temperature": 22},
    "expected_outcome": "Temperature will decrease",
    "verification_method": "Check if user feels cooler"
  },
  "feedback_request": "Has the temperature improved?",
  "contingency": {
    "if_step_fails": "Try adjusting temperature lower",
    "alternative_approach": "Open windows for ventilation"
  }
}
```

When task is complete, `next_step` becomes `null`.

## Action Types

### Humanoid Robot (Unitree-G1)
- **Talk**: `talk_with_human()`, `request_item_from_store()`
- **Tool**: `control_air_conditioner()`, `control_light()`, `web_search()`
- **Act**: `navigate_to_store()`, `return_home_with_item()`, `wait_for_item()`
- **Sense**: `get_observation()` (camera + VLM analysis)

### Robotic Arm (UR5e)
- **Talk**: `communicate_with_humanoid()`
- **Tool**: `verify_item_availability()`, `update_inventory()`
- **Act**: `pick_from_shelf()`, `place_on_counter()`, `organize_shelf()`
- **Sense**: `get_observation()` (workspace camera analysis)

## Key Configuration

### LLM Backend
- **API**: Alibaba DashScope (OpenAI-compatible)
- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Models**: `qwen-turbo` (default), `qwen-plus`, `qwen-max`, `qwen2.5-72b-instruct`
- **Temperature**: 0.7
- **Max Tokens**: 2000

### Environment Variables
```bash
DASHSCOPE_API_KEY=your-key  # Required for LLM calls
```

### Execution Modes
- **Simulation Mode** (default): Returns mock feedback, no hardware needed
- **Hardware Mode**: Integrates with real robot APIs (requires implementation)

## Common Patterns

### Starting a New Task
```python
planner = InteractiveHumanoidPlanner(simulation_mode=True)
step_plan = planner.start_new_task("Help me get groceries")
```

### Providing Feedback & Planning Next Step
```python
next_step = planner.provide_feedback("I successfully arrived at the store")
```

### Checking Task Status
```python
if planner.is_task_complete:
    summary = planner.get_task_summary()
```

### Switching LLM Models
```python
planner.switch_model("qwen-max")  # Use more powerful model
```

### Resetting for New Task
```python
planner.reset_conversation()  # Clear all history
```

## Implementation Status

✅ **Complete**
- Interactive humanoid planner (closed-loop)
- Interactive arm planner (closed-loop)
- Mock testing suite (no API required)
- Context memory & conversation history
- Single-step planning framework
- Structured action executor
- Complete documentation

⚠️ **Planned** (skeleton code provided)
- Real Unitree-G1 API integration
- Vision/VLM system integration
- Smart home controller APIs
- Real UR5e manipulation APIs

## Architecture Comparison

| Feature | Open-Loop (Traditional) | Closed-Loop (This System) |
|---------|------------------------|---------------------------|
| Planning | All steps upfront | One step at a time |
| Execution | Batch sequence | Step → Pause → Feedback |
| Adaptation | Pre-planned only | Dynamic, real-time |
| Human Control | Once at start | After every step |
| API Calls | 1 (entire plan) | N (one per step) |
| Safety | Lower (no intervention) | Higher (constant checkpoints) |

## Repository Statistics

- **Total Lines**: ~3,619 lines of Python
- **Core Planning**: 3 files (1,414 lines)
- **Prompt Templates**: 2 files (1,163 lines)
- **Test Suite**: 4 files (842 lines)
- **Documentation**: 4 comprehensive markdown files

## Branch & Git Status

- **Current Branch**: `interactive_planner_1125`
- **Recent Changes**: Added interactive planner files and updated humanoid planner
- **Repository**: Clean working directory

## Tips for Understanding the Code

1. **Start with tests**: `test/test_interactive_mock.py` shows complete workflow
2. **Read prompts**: Prompt templates define all available actions and output format
3. **Follow the loop**: `start_new_task()` → `plan_next_step()` → `provide_feedback()` cycle
4. **Check executor**: `humanoid_executor.py` shows how actions map to robot APIs
5. **Simulation first**: Run with `simulation_mode=True` to understand flow without APIs

## Common Modifications

### Adding New Actions
1. Define action in prompt template (`humanoid_prompt_template_interactive.py`)
2. Implement executor method in `humanoid_executor.py`
3. Add examples to prompt

### Supporting New Robot Types
1. Create new planner (e.g., `quadruped_planner_interactive.py`)
2. Create prompt template with available actions
3. Create executor for robot-specific APIs

### Changing LLM Provider
1. Update `openai.OpenAI()` initialization
2. Change `base_url` to new API endpoint
3. Update environment variable for API key

## Debugging Tips

- Set `simulation_mode=True` to test logic without hardware
- Print `planner.execution_history` to see all steps
- Check `planner.conversation_history` for full LLM context
- Use mock tests to verify changes without API calls

---

**Last Updated**: 2025-11-25
**Primary Contact**: Check repository documentation for maintainer info
