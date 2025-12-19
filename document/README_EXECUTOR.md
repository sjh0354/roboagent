# Interactive Humanoid Planner with Executor

## What Was Built

A complete **closed-loop planning and execution system** for your Unitree-G1 humanoid robot:

1. **HumanoidExecutor** (`humanoid_executor.py`): Executes real-world actions
2. **InteractiveHumanoidPlanner** (`humanoid_planner_interactive.py`): Plans and orchestrates tasks
3. **Integration**: Automatic feedback loop between planner and executor

## Architecture

```
User Request
    ↓
[Planner] → Plans next step using LLM
    ↓
[Executor] → Executes action on robot/environment
    ↓
[Executor] → Returns feedback (success/failure + observations)
    ↓
[Planner] → Plans next step based on feedback
    ↓
Loop until task complete
```

## Quick Start

### 1. Run Interactive CLI (Simulation Mode)

```bash
export DASHSCOPE_API_KEY="your_api_key"
python humanoid_planner_interactive.py
```

**Commands:**
- Just type your request: `"I'm feeling cold"`
- `mode sim` / `mode real`: Switch execution modes
- `stats`: Show execution statistics
- `status`: Show task progress
- `reset`: Start new task
- `quit`: Exit

### 2. Use Programmatically

```python
from humanoid_planner_interactive import InteractiveHumanoidPlanner

# Initialize
planner = InteractiveHumanoidPlanner(
    simulation_mode=True,  # False for real hardware
    verbose=True
)

# Execute a task (fully automated)
current_plan = planner.start_new_task("I need some water")

while not planner.is_task_complete:
    feedback = planner.execute_step_and_wait(current_plan)
    current_plan = planner.provide_feedback(feedback)

# Get summary
summary = planner.get_task_summary()
print(f"Completed in {summary['steps_executed']} steps")
```

## Files Overview

| File | Purpose |
|------|---------|
| `humanoid_executor.py` | Executes actions on robot/environment |
| `humanoid_planner_interactive.py` | LLM-based planning + executor integration |
| `humanoid_prompt_template_interactive.py` | System prompt for LLM planner |
| `humanoid_usage_example.py` | Example usage patterns |
| `HARDWARE_INTEGRATION_GUIDE.md` | How to integrate real Unitree-G1 APIs |

## Execution Flow Example

**Task**: "I'm feeling cold"

```
Step 1: [Planner] → get_observation()
        [Executor] → Captures scene, returns "Human on couch, AC is off"
        [Feedback] → Observation result

Step 2: [Planner] → control_air_conditioner(turn_on, 24)
        [Executor] → Sends command to AC, returns "AC turned on at 24°C"
        [Feedback] → Success

Step 3: [Planner] → talk_with_human("I've turned on the AC to 24°C")
        [Executor] → Speaks message, returns "Message delivered"
        [Feedback] → Success

Task Complete! ✅
```

## Current Status: Simulation Mode

**What works now:**
- ✅ Complete planning and execution loop
- ✅ All action types (talk, tool, act, sense)
- ✅ Automatic feedback generation
- ✅ Error handling and recovery
- ✅ Execution statistics

**What needs real hardware:**
- 🔧 Unitree-G1 navigation APIs
- 🔧 Camera + Vision-Language Model
- 🔧 Smart home device APIs
- 🔧 Text-to-speech system
- 🔧 Inter-robot communication

## Next Steps for Real Hardware

### 1. Get Unitree-G1 SDK
Contact Unitree and get:
- Official Python SDK
- Network connection details (IP, ports)
- API documentation

### 2. Implement TODO Sections
In `humanoid_executor.py`, replace all `raise NotImplementedError` with real API calls:

```python
# Current (simulation):
def _navigate_to_store(self):
    if self.simulation_mode:
        # Simulation code
        ...
    else:
        raise NotImplementedError("Real navigation not yet implemented")

# Your task:
def _navigate_to_store(self):
    if self.simulation_mode:
        # Keep simulation code
        ...
    else:
        # Add real implementation
        result = self.robot_controller.navigate_to(store_position)
        return ExecutionResult(...)
```

### 3. Test Incrementally

```bash
# Stage 1: Test in simulation
python humanoid_executor.py  # Unit test
python humanoid_usage_example.py  # Integration test

# Stage 2: Test with mock hardware
# (Create mock classes that simulate timing/responses)

# Stage 3: Test with real hardware (supervised)
planner = InteractiveHumanoidPlanner(simulation_mode=False)

# Stage 4: Full deployment
```

## Action Categories

### TALK (Communication)
- `talk_with_human(message)`: Speak to user
- `request_item_from_store(item)`: Inter-robot communication

**APIs needed**: TTS library, network communication

### TOOL (Device Control)
- `control_air_conditioner(action, temperature)`: HVAC control
- `control_light(action)`: Lighting control
- `web_search(URL, query)`: Web search

**APIs needed**: Smart home APIs (HomeAssistant, MQTT, etc.)

### ACT (Physical Actions)
- `navigate_to_store()`: Move to store location
- `return_home_with_item(item)`: Navigate while carrying object
- `wait_for_item(time)`: Wait/idle

**APIs needed**: Unitree navigation SDK, SLAM, path planning

### SENSE (Perception)
- `get_observation()`: Capture and analyze scene

**APIs needed**: Camera, VLM (GPT-4V, Qwen-VL, or local model)

## Safety Features to Add

Before deploying on real hardware:

1. **Emergency Stop**: Physical button + software flag
2. **Collision Avoidance**: Distance sensors, obstacle detection
3. **Human Detection**: Don't move if human too close
4. **Battery Monitoring**: Return to charge station when low
5. **Error Logging**: Track failures for debugging
6. **Timeout Protection**: Max execution time per action

## Example Use Cases

1. **Home Assistant**
   - "Turn on the lights"
   - "Set AC to 22 degrees"
   - "What's the weather today?"

2. **Item Retrieval**
   - "Get me a bottle of water"
   - "Bring me the book from the shelf"

3. **Information Query**
   - "What do you see in the kitchen?"
   - "Is the package at the door?"

4. **Multi-Step Tasks**
   - "Make sure the living room is comfortable" → observe → adjust AC/lights → confirm

## Troubleshooting

**Issue**: "API key not found"
- Set environment variable: `export DASHSCOPE_API_KEY="sk-..."`

**Issue**: "Hardware not implemented"
- You're in real hardware mode but APIs aren't implemented
- Switch to simulation: `planner.switch_execution_mode(simulation_mode=True)`

**Issue**: Planning errors
- Check LLM model availability
- Verify API key has credits
- Check prompt template format

**Issue**: Execution failures
- Check executor verbose logs
- Use `stats` command to see success rate
- Review execution history in task summary

## Performance Tips

1. **Model Selection**: Use `qwen-plus` for balance of speed/quality
2. **Timeout Settings**: Adjust based on action complexity
3. **Feedback Quality**: Better observations → better next steps
4. **Error Handling**: Implement retries for transient failures

## Contributing

When you implement real hardware APIs:

1. Test thoroughly in isolation
2. Add error handling
3. Document API requirements
4. Share working examples
5. Update this README with lessons learned

## Resources

- **Unitree-G1 Docs**: [unitree.com](https://www.unitree.com)
- **VLM Options**: OpenAI GPT-4V, Qwen-VL, LLaVA
- **Smart Home**: Home Assistant, MQTT
- **Integration Guide**: See `HARDWARE_INTEGRATION_GUIDE.md`

---

**Status**: ✅ Simulation ready | 🔧 Hardware integration pending

Good luck with your Unitree-G1 robot! 🤖
