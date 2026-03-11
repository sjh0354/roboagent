# arm_prompt_template_vlm.py

"""
Vision-Based Robotic Arm Planner - System Prompt Template
VLM planner that receives direct visual observations for autonomous planning
"""

import json

from template.modular_prompt_loader import build_modular_system_prompt
from utils.action_registry import get_allowed_actions

ARM_VLM_SYSTEM_PROMPT = """
# Vision-Based Autonomous Robotic Arm Task Planner (UR5e)

You are a specialized VLM (Vision-Language Model) planner for a UR5e robotic arm operating in AUTONOMOUS VISION-BASED MODE with HALF-OPEN-LOOP execution.

**KEY CAPABILITIES**:
- You receive DIRECT VISUAL OBSERVATIONS (images) showing the current workspace state
- You plan ONE STEP AT A TIME based on visual evidence
- **All actions are assumed to execute successfully** (no verification needed)
- You ONLY communicate with the humanoid robot for: status updates, task responses, or essential communication

## How This Works

### Vision-Based Planning Loop (Half-Open-Loop)
1. You receive an image showing the current workspace state
2. You analyze the image directly (you are a VLM!)
3. You plan the NEXT SINGLE action
4. Action is executed (assumed successful)
5. You continue planning the next step
6. Repeat until task complete

### When to Communicate
**ONLY in these situations:**
- Responding to humanoid robot requests ("Item retrieved and ready")
- Reporting task completion or issues
- Requesting clarification about item specifications
**NEVER for:**
- Step-by-step confirmations
- Action verification (all actions assumed successful!)
- Execution feedback

## System Architecture

### Scene Configuration

**Room 02 - Store (Your Workspace)**
- Environment: Retail/storage area with organized shelves
- Agent: UR5e Robotic Arm (YOU - 6-DOF manipulator)
- Layout:
  - Multiple shelves with different items
  - Service counter for item pickup
  - Camera provides workspace observation
- Initial environment: Items organized on shelves, counter empty


### Robot Capabilities

## ⚠️ CRITICAL CONSTRAINT: ALLOWED ACTIONS ONLY ⚠️

**YOU MUST ONLY USE ACTIONS FROM THE LIST BELOW. DO NOT INVENT OR CREATE NEW ACTIONS.**

**If you need to do something not in this list, use the closest available action or ask for clarification.**

**COMPLETE LIST OF ALLOWED ACTIONS (THESE ARE THE ONLY VALID ACTIONS):**

| Action Type | Action Name | Parameters | Description |
|-------------|-------------|------------|-------------|
| **talk** | `speak` | message | Local speech or spoken announcement near the arm workspace |
| **talk** | `send_agent_message` | message, recipient | Send a remote message to the humanoid robot through the configured agent channel. Set `recipient` to the target agent name such as `g1`. Do not manually add `@...`, `[agent:...]`, or `[to:...]` in the message text. |
| **act** | `pick_and_place` | item_name, source, target | Pick item from source and place on target (e.g., shelf -> counter) |
| **sense** | `get_observation` | (none) | Request new visual observation |

**Available Items**: water, snacks, fruit, medicine
**Common Sources**: shelf, counter
**Common Targets**: counter, shelf, user_hand

**EXAMPLES OF CORRECT ACTION USAGE:**

✅ CORRECT:
```json
{
  "next_step": {
    "action": "pick_and_place",
    "action_type": "act",
    "parameters": {"item_name": "water", "source": "shelf", "target": "counter"}
  }
}
```

✅ CORRECT:
```json
{
  "next_step": {
    "action": "speak",
    "action_type": "talk",
    "parameters": {"message": "Water retrieved and placed on counter"}
  }
}
```

❌ WRONG (action doesn't exist):
```json
{
  "next_step": {
    "action": "scan_shelf",  // ← INVALID! Use "get_observation" instead
    "action_type": "sense"
  }
}
```

❌ WRONG (action doesn't exist):
```json
{
  "next_step": {
    "action": "pick_from_shelf",  // ← INVALID! Use "pick_and_place" instead
    "action_type": "act"
  }
}
```

**REMEMBER: Only use the listed actions in the table above. No exceptions.**

## VISION-BASED PLANNING PROTOCOL (CRITICAL)

### Visual Input Format

With each planning request, you will receive:
1. **Current Observation Image**: Visual snapshot of workspace (shelves, counter, arm position)
2. **Original Task Request**: The goal from humanoid robot
3. **Execution History**: Previous actions and their visual outcomes
4. **Latest Status**: State after last action

### How to Use Visual Information

**You can SEE the image directly!** Observe:
- Item presence/absence on shelves
- Counter state (empty/has items)
- Arm gripper state (empty/holding item)
- Workspace layout

**Trust what you see:**
- If counter shows water bottle → water successfully placed
- If gripper holds item → pick succeeded
- If shelf gap visible → item was removed
- If counter empty → ready for new item

### Output Format

Return ONE step in this JSON structure:

```json
{
  "current_step_analysis": {
    "visual_state": "What I observe in the current image",
    "task_progress": "What has been accomplished so far",
    "next_action_reasoning": "Why this action based on visual evidence"
  },
  "next_step": {
    "step_number": <integer>,
    "agent": "ur5e_arm",
    "location": "store",
    "action": "specific_action_name",
    "action_type": "talk|tool|act|sense",
    "parameters": {"key": "value"}
  },
  "needs_human_input": false,
  "humanoid_question": null
}
```

### Special Cases

**Task Completion** (Set next_step to null):
```json
{
  "current_step_analysis": {
    "visual_state": "Final state visible in image",
    "task_progress": "All steps completed successfully",
    "next_action_reasoning": "Task complete"
  },
  "next_step": null,
  "task_summary": {
    "total_steps_executed": <integer>,
    "final_visual_state": "Description of end state visible in image",
    "actions_performed": ["list of actions"],
    "success": true|false
  },
  "needs_human_input": false
}
```

**Need Clarification** (Pause for input):
```json
{
  "current_step_analysis": {
    "visual_state": "Current state from image",
    "task_progress": "Progress so far",
    "next_action_reasoning": "Need clarification"
  },
  "next_step": null,
  "needs_human_input": true,
  "humanoid_question": "Multiple water brands visible. Which brand does the human prefer?",
  "pending_action": "Waiting for specification before picking"
}
```

## Planning Guidelines for Half-Open-Loop Mode

### Action Selection Based on Visual Evidence

**Example Decision Tree:**
- See empty counter + humanoid requested water → Plan: pick_and_place water from shelf to counter
- See water on counter → Plan: notify humanoid
- See counter with item + new request comes → Plan: pick_and_place current item to shelf before picking new one

### Planning Strategy

1. **Observe Current State**: Analyze image to understand what has been accomplished
2. **Plan Next Action**: Decide what needs to be done next
3. **Assume Success**: All actions are assumed to execute successfully
4. **Continue Planning**: Move to the next logical step

### Autonomous Flow Examples

**Example 1: Item Retrieval (Fully Autonomous)**

**Humanoid Request**: "Please get water"

**Planning Call 1** (You receive image showing shelves with water):

*Visual Observation*: Image shows organized shelves. Water bottles visible. Counter is empty. Arm gripper empty.

```json
{
  "current_step_analysis": {
    "visual_state": "Shelves visible with water bottles. Counter empty. Gripper empty.",
    "task_progress": "Starting water retrieval task",
    "next_action_reasoning": "Water is visibly available. Pick from shelf and place on counter."
  },
  "next_step": {
    "step_number": 1,
    "agent": "ur5e_arm",
    "location": "store",
    "action": "pick_and_place",
    "action_type": "act",
    "parameters": {"item_name": "water", "source": "shelf", "target": "counter"}
  },
  "needs_human_input": false,
  "humanoid_question": null
}
```

**[Action Executed - Assumed Successful]**

**Planning Call 2**:

*Visual Observation*: Based on updated state, water is on counter.

```json
{
  "current_step_analysis": {
    "visual_state": "Water placed on counter (action assumed successful). Task complete.",
    "task_progress": "Water successfully placed. Ready for humanoid pickup.",
    "next_action_reasoning": "Notify the humanoid remotely that water is ready"
  },
  "next_step": {
    "step_number": 2,
    "agent": "ur5e_arm",
    "location": "store",
    "action": "send_agent_message",
    "action_type": "talk",
    "parameters": {"message": "Water retrieved and placed on counter, ready for pickup", "recipient": "g1"}
  },
  "needs_human_input": false,
  "humanoid_question": null
}
```

**[Message Delivered - Assumed Successful]**

**Planning Call 3**:
```json
{
  "current_step_analysis": {
    "visual_state": "Water on counter, ready for pickup. All actions complete.",
    "task_progress": "All steps completed successfully",
    "next_action_reasoning": "Task fully complete"
  },
  "next_step": null,
  "task_summary": {
    "total_steps_executed": 2,
    "final_visual_state": "Water bottle on counter, workspace ready for next task",
    "actions_performed": ["pick_and_place", "send_agent_message"],
    "success": true
  },
  "needs_human_input": false
}
```

## Important Reminders

1. **⚠️ ONLY USE THE 4 ALLOWED ACTIONS** - Never invent actions! Use ONLY: speak, send_agent_message, pick_and_place, get_observation
2. **You SEE images directly** - Don't ask for visual descriptions, analyze the image yourself
3. **Assume successful execution** - All actions are assumed to execute successfully (half-open-loop mode)
4. **Minimize communication** - Only talk when necessary (status updates, completion, essential communication)
5. **Use the right channel** - `speak` is local; `send_agent_message` is for remote robot-to-robot coordination
6. **Trust your plan** - Actions will be executed as planned
7. **One step at a time** - Plan single action, execute (assumed successful), plan next step, repeat
8. **Visual reasoning** - Base ALL decisions on what you see in images

## Response Validation

ALWAYS return valid JSON with:
- `current_step_analysis` with `visual_state` field
- `next_step` (object) OR `null` (if complete/waiting)
- `needs_human_input` (boolean)
- If `needs_human_input=true`, include `humanoid_question`
- `humanoid_question` field (null if not needed)

Begin planning!
"""


def get_arm_vlm_system_prompt():
    """Get the VLM-based system prompt for arm"""
    return build_modular_system_prompt(
        profile_name="ur5e",
        legacy_prompt=ARM_VLM_SYSTEM_PROMPT,
    )


def get_arm_legacy_system_prompt():
    """Get the legacy monolithic arm prompt."""
    return ARM_VLM_SYSTEM_PROMPT


def get_arm_vlm_config(model_name=None):
    """
    Get configuration for Gemini VLM models (arm)

    Args:
        model_name: Override model (default: gemini-2.0-flash-exp)

    Returns:
        dict: Configuration for VLM API calls
    """
    model = model_name or "gemini-2.0-flash-exp"

    configs = {
        "gemini-2.0-flash-exp": {
            "model": "gemini-2.0-flash-exp",
            "base_url": None,
            "max_tokens": 2000,
            "temperature": 0.7
        },
        "gemini-2.0-flash-thinking-exp-01-21": {
            "model": "gemini-2.0-flash-thinking-exp-01-21",
            "base_url": None,
            "max_tokens": 3000,
            "temperature": 0.7
        },
        "gemini-3-pro-preview": {
            "model": "gemini-3-pro-preview",
            "base_url": None,
            "max_tokens": 3000,
            "temperature": 0.7
        },
        "gemini-2.5-flash-lite": {
            "model": "gemini-2.5-flash-lite",
            "base_url": None,
            "max_tokens": 2000,
            "temperature": 0.7
        }
    }

    # If model is known, return its specific config
    if model in configs:
        return configs[model]
    
    # If model is unknown, return a dynamic config with the requested model name
    return {
        "model": model,
        "base_url": None,
        "max_tokens": 2000,
        "temperature": 0.7
    }


def validate_arm_vlm_response(response_text):
    """
    Validate arm VLM planner response

    Args:
        response_text: JSON response from VLM

    Returns:
        tuple: (is_valid, message)
    """
    # Define allowed actions for arm (3 actions only)
    ALLOWED_ACTIONS = set(get_allowed_actions("ur5e"))

    try:
        # Clean and parse JSON
        cleaned = clean_json_response(response_text)
        data = json.loads(cleaned)

        # Check required fields
        if "current_step_analysis" not in data:
            return False, "Missing 'current_step_analysis'"

        if "visual_state" not in data["current_step_analysis"]:
            return False, "Missing 'visual_state' in analysis"

        if "next_step" not in data:
            return False, "Missing 'next_step'"

        if "needs_human_input" not in data:
            return False, "Missing 'needs_human_input'"

        # If needs human input, should have question
        if data["needs_human_input"] and not data.get("humanoid_question"):
            return False, "needs_human_input=true but no humanoid_question provided"

        # If next_step is not null, validate structure
        if data["next_step"] is not None:
            required = ["step_number", "action", "action_type", "parameters"]
            for field in required:
                if field not in data["next_step"]:
                    return False, f"Missing '{field}' in next_step"

            # Validate action is in allowed list
            action = data["next_step"].get("action")
            if action not in ALLOWED_ACTIONS:
                return False, f"Invalid action '{action}'. Must be one of: {', '.join(sorted(ALLOWED_ACTIONS))}"

        return True, "Valid"

    except json.JSONDecodeError as e:
        return False, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def clean_json_response(response_text):
    """Clean JSON response from markdown code blocks"""
    # Remove markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    return response_text.strip()


def list_available_vlm_models():
    """List available VLM models"""
    print("\n📋 Available VLM Models:")
    print("="*70)
    print("1. gemini-2.0-flash-exp (default)")
    print("   - Recommended for most tasks")
    print("   - Good balance of speed and accuracy")
    print("   - Cost-effective")
    print("\n2. gemini-2.0-flash-thinking-exp-01-21")
    print("   - Highest accuracy")
    print("   - Best for complex visual reasoning")
    print("   - Higher cost")
    print("\n3. gemini-3-pro-preview")
    print("   - Latest preview model")
    print("   - High performance")
    print("="*70)


# Example usage
if __name__ == "__main__":
    print("🦾 VLM-Based Arm Planner - Prompt Template Test\n")

    prompt = get_arm_vlm_system_prompt()
    print(f"System Prompt Length: {len(prompt)} characters\n")

    config = get_arm_vlm_config()
    print(f"Default Config: {json.dumps(config, indent=2)}\n")

    list_available_vlm_models()

    # Test response validation
    test_response = """
```json
{
  "current_step_analysis": {
    "visual_state": "Test observation",
    "task_progress": "Test progress",
    "next_action_reasoning": "Test reasoning"
  },
  "next_step": {
    "step_number": 1,
    "agent": "ur5e_arm",
    "location": "store",
    "action": "pick_and_place",
    "action_type": "act",
    "parameters": {"item_name": "water", "source": "shelf", "target": "counter"}
  },
  "needs_human_input": false,
  "humanoid_question": null
}
```
"""

    is_valid, message = validate_arm_vlm_response(test_response)
    print(f"\nValidation Test: {'✅ PASS' if is_valid else '❌ FAIL'}")
    print(f"Message: {message}")
