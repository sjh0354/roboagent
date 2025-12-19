# Interactive Planner - Closed-Loop Robot Control System

## Overview

This module implements a **closed-loop, interactive planning system** for robotic agents. Unlike traditional open-loop systems that generate complete multi-step plans upfront, this system operates in a **turn-based manner**:

1. **Plan ONE step** at a time
2. **Execute** that single step
3. **Wait** for human feedback/observation
4. **Update** context with feedback
5. **Repeat** until task complete

## Architecture Comparison

### Open-Loop (Original System)
```
Human Request → Generate Full Plan → Execute All Steps → Done
                 (5-10 steps)         (no interruption)
```

### Closed-Loop (Interactive System)
```
Human Request → Plan Step 1 → Execute → Wait for Feedback →
                                              ↓
                Plan Step 2 → Execute → Wait for Feedback →
                                              ↓
                Plan Step 3 → Execute → ... → Task Complete
```

## Key Features

- **Single-Step Planning**: LLM plans only the immediate next action
- **Context Accumulation**: Each step's feedback is concatenated into the conversation history
- **Adaptive Planning**: Next step adapts based on previous outcomes
- **Error Recovery**: Failed steps trigger recovery planning
- **Human-in-the-Loop**: System pauses after each step for human input

## File Structure

```
interactive_planner/
├── humanoid_prompt_template_interactive.py  # Interactive prompt for single-step planning
├── humanoid_planner_interactive.py          # Main interactive planner class
├── interactive_planner_usage.py             # Usage examples
├── test_interactive_mock.py                 # Mock tests (no API needed)
├── README.md                                # This file
│
├── humanoid_prompt_template.py              # Original open-loop prompt
├── humanoid_planner.py                      # Original open-loop planner
├── arm_prompt_template.py                   # Robotic arm prompt
└── arm_planner.py                           # Robotic arm planner
```

## Quick Start

### 1. Test Without API (Mock Mode)

```bash
cd /media/user/B29202FA9202C2B91/robot_agent_system/interactive_planner
python test_interactive_mock.py
```

This demonstrates the closed-loop workflow without requiring API keys.

### 2. Run with Real LLM API

```bash
# Set your API key
export DASHSCOPE_API_KEY='your-api-key-here'

# Run interactive mode
python humanoid_planner_interactive.py
```

### 3. Run Usage Examples

```bash
# All examples
python interactive_planner_usage.py

# Specific example
python interactive_planner_usage.py simple
python interactive_planner_usage.py retrieval
python interactive_planner_usage.py error
python interactive_planner_usage.py programmatic
```

## Usage Example

```python
from humanoid_planner_interactive import InteractiveHumanoidPlanner

# Initialize planner
planner = InteractiveHumanoidPlanner()

# Start a task
task_request = "I'm feeling cold"
current_step = planner.start_new_task(task_request)

# Interactive loop
while not planner.is_task_complete:
    # Execute the planned step
    feedback_request = planner.execute_step_and_wait(current_step)

    # Get feedback from human/sensors
    print(f"System needs: {feedback_request}")
    feedback = input("Your feedback: ")

    # Plan next step based on feedback
    current_step = planner.provide_feedback(feedback)

# Task complete
print("Task finished!")
print(planner.get_task_summary())
```

## How Context Memory Works

The system maintains conversation history across planning calls:

### Step 1: Initial Planning
```
Input to LLM:
  [ORIGINAL REQUEST]: "I'm feeling cold"
  [STEPS EXECUTED SO FAR]: None (first step)
  [CURRENT STATUS]: Planning step #1

Output from LLM:
  next_step: get_observation()
```

### Step 2: Planning with Feedback
```
Input to LLM:
  [ORIGINAL REQUEST]: "I'm feeling cold"
  [STEPS EXECUTED SO FAR]: 1
    - Step 1: get_observation with {}
  [FEEDBACK/OBSERVATION]: "Room temperature is cool, AC is off"
  [CURRENT STATUS]: Planning step #2

Output from LLM:
  next_step: control_air_conditioner(turn_on, 24)
```

### Step 3: Continued Planning
```
Input to LLM:
  [ORIGINAL REQUEST]: "I'm feeling cold"
  [STEPS EXECUTED SO FAR]: 2
    - Step 1: get_observation
    - Step 2: control_air_conditioner
  [FEEDBACK]: "AC turned on at 24°C"
  [CURRENT STATUS]: Planning step #3

Output from LLM:
  next_step: talk_with_human("AC is on, room warming up")
```

## API Response Format

Each planning call returns a JSON structure:

```json
{
  "current_step_analysis": {
    "task_progress": "What has been done so far",
    "current_situation": "Current state from feedback",
    "next_action_reasoning": "Why this action is chosen"
  },
  "next_step": {
    "step_number": 2,
    "action": "control_air_conditioner",
    "action_type": "tool",
    "parameters": {"action": "turn_on", "temperature": 24},
    "expected_outcome": "AC turns on",
    "verification_method": "Check AC status"
  },
  "task_status": {
    "is_complete": false,
    "completion_percentage": 60,
    "remaining_actions": "Verify and confirm with human"
  },
  "feedback_request": "Did the AC turn on successfully?",
  "contingency": {
    "if_step_fails": "Retry or check for errors",
    "alternative_approach": "Manual control suggestion"
  }
}
```

## Key Methods

### `InteractiveHumanoidPlanner`

#### `start_new_task(human_request)`
Start a new task and plan the first step.

**Args:**
- `human_request` (str): Natural language task request

**Returns:**
- dict: First step plan

#### `plan_next_step()`
Plan the next single step based on current context and history.

**Returns:**
- dict: Next step plan

#### `provide_feedback(feedback)`
Provide feedback about the executed step and plan the next one.

**Args:**
- `feedback` (str): Observation or result from the executed step

**Returns:**
- dict: Next step plan, or None if task complete

#### `execute_step_and_wait(step_plan)`
Execute a step (simulation) and return feedback request.

**Args:**
- `step_plan` (dict): Step plan from `plan_next_step()`

**Returns:**
- str: What feedback is needed from human

#### `get_task_summary()`
Get summary of the current task.

**Returns:**
- dict: Task summary with execution history

#### `reset_conversation()`
Reset conversation state for a new task.

## Conversation History Management

The planner maintains three types of memory:

1. **Original Request**: The initial task goal (never changes)
2. **Execution History**: List of all executed steps with timestamps
3. **Conversation History**: Full LLM conversation (prompt + response pairs)

All three are concatenated into each planning request to provide full context.

## Error Handling

The system handles errors gracefully:

```python
# Example: Navigation failure
step_plan = planner.plan_next_step()
feedback = "Navigation FAILED - obstacle detected"

# Next planning call will adapt:
recovery_step = planner.provide_feedback(feedback)
# Returns: "Inform human about obstacle" or "Try alternative route"
```

## Configuration

### Model Selection

```python
# Use different Qwen models
planner = InteractiveHumanoidPlanner(model_name="qwen-max")

# Available models:
# - qwen-turbo (fast)
# - qwen-plus (balanced, default)
# - qwen-max (best reasoning)
# - qwen2.5-72b-instruct (open source)
```

### Switch Models During Runtime

```python
planner.switch_model("qwen-max")
```

## Comparison with Original System

| Feature | Open-Loop (Original) | Closed-Loop (Interactive) |
|---------|---------------------|---------------------------|
| Planning | Full multi-step plan | Single step at a time |
| Execution | All steps in sequence | One step, then pause |
| Feedback | After all steps | After each step |
| Adaptation | No adaptation | Adapts based on feedback |
| Error Recovery | Pre-planned contingencies | Dynamic recovery |
| Human Control | Approve plan upfront | Approve/guide each step |
| Context Memory | Single planning call | Cumulative across steps |

## Advanced Usage

### Programmatic Integration

```python
# Integrate with real robot control
planner = InteractiveHumanoidPlanner()
current_step = planner.start_new_task("Turn on AC")

while not planner.is_task_complete:
    next_action = current_step.get("next_step", {})

    # Call actual robot control functions
    if next_action["action"] == "control_air_conditioner":
        result = robot.control_ac(**next_action["parameters"])
        feedback = f"AC control result: {result}"

    elif next_action["action"] == "get_observation":
        observation = robot.camera.get_scene_description()
        feedback = f"Observation: {observation}"

    # Continue loop
    current_step = planner.provide_feedback(feedback)
```

### Custom Feedback Processing

```python
# Use sensor data as feedback
sensor_data = robot.sensors.read_all()
feedback = f"Temperature: {sensor_data['temp']}°C, Light: {sensor_data['light']}"
planner.provide_feedback(feedback)
```

## Testing

### Run Mock Tests
```bash
python test_interactive_mock.py
```

### Run All Examples
```bash
python interactive_planner_usage.py
```

### Interactive CLI
```bash
python humanoid_planner_interactive.py
```

## Troubleshooting

### API Key Not Set
```
Error: API key not found
Solution: export DASHSCOPE_API_KEY='your-key'
```

### Task Not Completing
```
Issue: Planner keeps generating steps
Solution: Ensure feedback indicates task completion
```

### Invalid JSON Response
```
Issue: LLM returns malformed JSON
Solution: Check prompt template or try different model
```

## Future Enhancements

- [ ] Add robotic arm interactive planner
- [ ] Implement multi-agent coordination
- [ ] Add visual feedback support (image input)
- [ ] Create web interface for human feedback
- [ ] Add voice input/output
- [ ] Implement plan rollback/undo
- [ ] Add parallel action support

## License

Same as parent project.

## Contact

For questions or issues, please refer to the main project documentation.
