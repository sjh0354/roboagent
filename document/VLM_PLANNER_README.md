# VLM-Based Autonomous Planner - Complete Guide

## Overview

The VLM-Based Autonomous Planner is a vision-language model powered robot planning system that:
- **Sees images directly** during planning (not text descriptions)
- **Plans autonomously** without human step-by-step confirmation
- **Verifies actions visually** through before/after image comparison
- **Communicates with humans** only for task requests, clarifications, and completions

## Key Advantages Over Text-Based Planning

| Feature | Text-Based (Old) | VLM-Based (New) |
|---------|-----------------|-----------------|
| **Observation** | Text descriptions | Direct visual perception |
| **Information** | Limited by text | Full visual detail (positions, states, spatial relationships) |
| **Verification** | Human confirmation | Automatic visual comparison |
| **Autonomy** | Requires frequent human input | Fully autonomous until completion |
| **Accuracy** | Prone to description errors | Direct visual evidence |

## System Components

### 1. Core Files

```
humanoid_planner_vlm.py              - Main VLM autonomous planner
humanoid_executor_vision.py          - Vision-enabled action executor
qwen_vlm_client.py                   - Qwen VLM API client
simulation_image_manager.py          - Manages observation images for simulation
humanoid_prompt_template_vlm.py      - VLM-specific system prompts
```

### 2. Architecture Flow

```
Human Request
    ↓
┌─────────────────────────────────────────────────┐
│ 1. Get Current Observation Image               │
│    ↓                                            │
│ 2. VLM Planner Sees Image + Plans Next Step    │
│    ↓                                            │
│ 3. Execute Action                               │
│    ↓                                            │
│ 4. Capture New Observation Image               │
│    ↓                                            │
│ 5. VLM Verifies Success by Comparing Images    │
│    (automatically loops back to step 2)         │
└─────────────────────────────────────────────────┘
    ↓ (loops until task complete or needs clarification)
Task Complete / Human Clarification Needed
```

## Quick Start

### Prerequisites

1. **API Key**: Set up DashScope API key
```bash
export DASHSCOPE_API_KEY='your-dashscope-api-key'
```

2. **Python Environment**: Install required packages
```bash
pip install openai  # OpenAI client library (for API compatibility)
```

### Basic Usage

```python
from humanoid_planner_vlm import AutonomousVLMPlanner

# Initialize planner
planner = AutonomousVLMPlanner(
    model_name="qwen-vl-plus",
    simulation_mode=True,  # Use simulation images
    verbose=True
)

# Start task (runs autonomously until complete)
result = planner.start_new_task("I'm feeling cold")

# Handle result
if result.get("is_complete"):
    print("Task completed successfully!")
elif result.get("status") == "waiting_for_human":
    # Planner needs clarification
    response = input(f"{result['question']}\nYour response: ")
    result = planner.provide_human_response(response)
```

### Interactive Mode

Run the main script for interactive testing:

```bash
python humanoid_planner_vlm.py
```

Then enter tasks in natural language:
- "I'm feeling cold"
- "Get me some water"
- "Turn on the lights"

## Simulation Mode

### How It Works

In simulation mode, the system uses **pre-configured observation images** that change based on robot state:

```
simulation_images/
├── home/
│   ├── default.jpg          # Default home view
│   ├── ac_on_22.jpg        # AC on at 22°C
│   ├── ac_on_24.jpg        # AC on at 24°C
│   ├── light_on.jpg        # Lights on
│   └── light_off.jpg       # Lights off
├── store/
│   ├── default.jpg          # Store view
│   └── robot_present.jpg   # Humanoid at store
└── in_transit/
    ├── to_store.jpg        # Moving to store
    └── to_home.jpg         # Returning home
```

### Providing Custom Images

You can provide your own observation images at key states:

```python
# Register custom image for a specific state
planner.register_observation_image(
    "home_ac_on_22",
    "/path/to/your/image_showing_ac_on_at_22.jpg"
)
```

**Image Guidelines:**
- Format: JPG, PNG
- Content: Should clearly show the state (e.g., AC display visible showing temperature)
- Resolution: 1024x768 or higher recommended
- Naming: Use descriptive names matching states

### State-to-Image Mapping

The system automatically selects images based on current state:

| Robot State | Selected Image |
|------------|----------------|
| location=home, ac_status=on, ac_temperature=22 | `home_ac_on_22.jpg` |
| location=home, light_status=on | `home_light_on.jpg` |
| location=store | `store_robot_present.jpg` |
| location=in_transit | `in_transit_to_store.jpg` |

## VLM Models

### Available Models

| Model | Speed | Accuracy | Cost | Use Case |
|-------|-------|----------|------|----------|
| **qwen-vl-plus** | Fast | Good | Low | General tasks (recommended) |
| **qwen-vl-max** | Medium | Best | High | Complex visual reasoning |

### Switching Models

```python
# During initialization
planner = AutonomousVLMPlanner(model_name="qwen-vl-max")

# At runtime
planner.switch_model("qwen-vl-max")
```

## Planning Response Format

The VLM planner returns structured JSON with visual observations:

```json
{
  "current_step_analysis": {
    "visual_state": "AC display shows OFF. Human on couch. Room lighting moderate.",
    "task_progress": "Starting task - human needs warmth",
    "next_action_reasoning": "Turn on AC based on visual confirmation it's off"
  },
  "next_step": {
    "step_number": 1,
    "action": "control_air_conditioner",
    "action_type": "tool",
    "parameters": {"action": "turn_on", "temperature": 24},
    "expected_visual_outcome": "AC display will show ON with 24°C",
    "verification_method": "Check AC display in next image"
  },
  "needs_human_input": false,
  "contingency": {
    "if_visual_verification_fails": "Retry or diagnose AC malfunction",
    "alternative_approach": "Inform human of failure"
  }
}
```

## Autonomous Execution Loop

The planner runs a fully autonomous loop:

```
1. Capture current observation image
2. Send image + context to VLM planner
3. VLM plans next step based on what it sees
4. Execute planned action
5. Capture new observation image
6. Repeat from step 2 until:
   - Task complete (next_step = null)
   - Human clarification needed (needs_human_input = true)
   - Error occurs
```

## Human Interaction Points

The planner **minimizes** human interaction. Humans are only involved for:

### 1. Task Requests
```
Human: "I'm feeling cold"
Robot: [Starts autonomous execution]
```

### 2. Clarification Questions
```
Robot: "What drink would you like? (water, juice, soda)"
Human: "Water please"
Robot: [Continues autonomous execution]
```

### 3. Completion Reports
```
Robot: "Task complete. AC is now on at 24°C."
```

## Visual Verification

After each action, the VLM automatically verifies success:

```python
# Before action: Image shows AC display "OFF"
planner.execute_action("control_air_conditioner", {"action": "turn_on", "temperature": 24})
# After action: Image shows AC display "ON 24°C"

# VLM compares images:
{
  "changes_detected": ["AC display changed from OFF to ON 24°C"],
  "success": true,
  "observation": "AC is now running at 24°C, display confirms",
  "verification": "Visual evidence confirms successful activation"
}
```

## Testing

### Run Test Suite

```bash
python test/test_vlm_planner.py
```

Tests include:
- ✅ Planner initialization
- ✅ Observation system
- ✅ Vision-enabled executor
- ✅ Planning components
- ✅ Custom image registration
- ✅ Full task simulation

### Without API Key

Tests run in mock mode without DASHSCOPE_API_KEY:
- Uses placeholder images
- Tests structure and logic
- No actual VLM API calls

### With API Key

Full VLM testing when DASHSCOPE_API_KEY is set:
- Real VLM API calls
- Actual visual analysis
- Complete autonomous planning

## Comparison with Interactive Mode

Both modes are available - choose based on your needs:

### Interactive Mode (`humanoid_planner_interactive.py`)

```python
from humanoid_planner_interactive import InteractiveHumanoidPlanner

planner = InteractiveHumanoidPlanner()
step = planner.start_new_task("I'm feeling cold")

# Manual loop with human confirmation each step
while not planner.is_task_complete:
    feedback = planner.execute_step_and_wait(step)
    user_input = input("Provide feedback: ")
    step = planner.provide_feedback(user_input)
```

**Use when:**
- Teaching/demonstrating system
- Safety-critical tasks requiring human oversight
- Step-by-step verification needed

### VLM Autonomous Mode (`humanoid_planner_vlm.py`)

```python
from humanoid_planner_vlm import AutonomousVLMPlanner

planner = AutonomousVLMPlanner()

# Fully autonomous - runs to completion
result = planner.start_new_task("I'm feeling cold")
```

**Use when:**
- Routine tasks
- Long-running operations
- Maximum automation desired
- Visual verification sufficient

## API Reference

### AutonomousVLMPlanner

```python
class AutonomousVLMPlanner:
    def __init__(self, api_key=None, model_name="qwen-vl-plus",
                 simulation_mode=True, verbose=True)

    def start_new_task(self, human_request, run_autonomously=True)
    # Start task, optionally run autonomous loop

    def provide_human_response(self, response)
    # Provide answer to clarification question

    def register_observation_image(self, state_key, image_path)
    # Register custom image for simulation

    def get_task_summary(self)
    # Get completion summary and statistics

    def switch_model(self, model_name)
    # Switch to different VLM model
```

### VisionEnabledExecutor

```python
class VisionEnabledExecutor(HumanoidExecutor):
    def execute_action(self, action_type, action_name, parameters)
    # Execute with automatic visual observation

    def get_current_observation(self)
    # Get current observation image and state

    def register_custom_observation_image(self, state_key, image_path)
    # Add custom observation image
```

### QwenVLMClient

```python
class QwenVLMClient:
    def analyze_image(self, image_path, query, system_prompt=None)
    # Analyze single image with query

    def compare_images(self, image_before, image_after, action_description)
    # Compare before/after to verify action result

    def get_observation_description(self, image_path)
    # Get detailed scene description
```

## Troubleshooting

### Issue: "API key not found"
**Solution**: Set environment variable
```bash
export DASHSCOPE_API_KEY='your-key'
```

### Issue: "Image not found"
**Solution**: Provide actual images in simulation_images/ or use placeholders
```python
planner.register_observation_image("home_default", "/path/to/image.jpg")
```

### Issue: "VLM disabled"
**Cause**: No DASHSCOPE_API_KEY set
**Impact**: Visual descriptions won't be generated, but simulation still works
**Solution**: Set API key for full VLM capabilities

### Issue: "JSON parsing failed"
**Cause**: VLM returned invalid JSON
**Solution**: Check VLM response in verbose mode, may need prompt adjustment

## Best Practices

1. **Provide Clear Images**: Ensure observation images clearly show device states (displays, indicators)

2. **Use Consistent Naming**: Name image files to match state keys exactly

3. **Start with Simulation**: Test with simulation images before deploying to real hardware

4. **Monitor First Runs**: Use `verbose=True` to understand planning decisions

5. **Gradual Complexity**: Start with simple tasks (single action) before complex multi-step tasks

## Future Enhancements

- [ ] Real camera integration (OpenCV/ROS)
- [ ] Real-time video stream processing
- [ ] Multi-modal feedback (vision + audio + sensors)
- [ ] Learning from visual corrections
- [ ] Custom action libraries

## License & Credits

Part of the Interactive Planner project for autonomous humanoid robots.

VLM Integration: Alibaba DashScope (Qwen VL models)

## Support

For issues or questions:
1. Check simulation_images/ directory for placeholders
2. Verify DASHSCOPE_API_KEY is set
3. Run test suite: `python test/test_vlm_planner.py`
4. Review verbose output for planning logic
5. Consult VISION_BASED_ARCHITECTURE_DESIGN.md for architecture details

---

**Last Updated**: 2025-11-25
