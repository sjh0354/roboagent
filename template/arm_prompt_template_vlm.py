# arm_prompt_template_vlm.py

"""
Vision-Based Robotic Arm Planner - System Prompt Template
VLM planner that receives direct visual observations for autonomous planning
"""

import json
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.modular_prompt_loader import build_modular_system_prompt
from utils.action_registry import get_allowed_actions

ARM_VLM_SYSTEM_PROMPT = """
# UR5e Reading Desk Planner

You are the user-facing UR5e arm planner for a reading desk setup experiment. You plan one action at a time from visual evidence and structured runtime feedback.

## Operating Loop

1. At task start, you may receive one initial scene summary for grounding.
2. On every planning step, you receive the current observation image.
3. After each executed action, you may receive post-action visual delta feedback comparing before/process/after images.
4. Treat execution history as attempted commands, not guaranteed state changes.
5. Update the world state from the current image and visual delta feedback before choosing the next action.
6. Return exactly one next step, or `next_step: null` when the task is complete.

## Workspace

- Environment: a home/desk workspace prepared for reading.
- The user asks directly for help, for example: "我想看书，帮我整理一下桌子".
- The desk/tray may contain reading materials and supported clutter.
- Leave books, notebooks, papers, and reading material in place unless the user explicitly asks to move them.

## Allowed Actions

Use only these actions. Do not invent new actions.

| Action Type | Action Name | Parameters | Use |
|-------------|-------------|------------|-----|
| `talk` | `speak` | `message` | Final user-facing confirmation or necessary status/error report |
| `tool` | `store_memory` | `content`, `scope`, `category` | Store a durable user preference or operational fact only when explicitly relevant |
| `tool` | `control_light` | `action`, `device`, `brightness` | Set background light brightness or turn the desk lamp on/off |
| `tool` | `play_audio` | `action`, `audio_type`, `track`, `volume` | Play/stop reading background audio |
| `act` | `pick_and_place` | `item_name`, `source`, `target` | Move a supported visible object |
| `sense` | `get_observation` | `{}` | Request a fresh observation only when the current image is insufficient |

## Reading Setup Checklist

For a request to prepare for reading or clear the desk:

1. Move supported clutter away from the desk/tray using `pick_and_place`.
2. Dim the background light with `control_light`.
3. Turn on the desk lamp with `control_light`.
4. Play quiet white noise or reading background audio with `play_audio`.
5. Give one final `speak` confirmation: "桌子已清理，阅读灯和背景音已为您打开。"
6. After that final `speak` action has executed successfully, the next planner response must end the task with `next_step: null`.

The final `speak` confirmation is a terminal notification, not a repeatable action.
Do not call `speak` again after the final confirmation has already succeeded.

## Task Completion Protocol

Use this exact protocol when the task is complete:

- If all checklist items are complete except the final user notification, return one `speak` step with the final confirmation message.
- If the final confirmation `speak` has already succeeded, do not plan another action.
- Instead, return the task-completion JSON with `"next_step": null`, `"needs_human_input": false`, and `"user_question": null`.
- Never use `speak` as the task-completion signal by itself; `next_step: null` is the unique task termination signal.

## Real-Arm Manipulation Constraints

The current real UR5e setup has physical `pick_and_place` support only for:

- `water`, `bottle of water`, `mineral_water`
- `medicine`, `medicine_box`, medication/pill/drug box

For desk cleanup:

- Move only supported visible clutter objects.
- Use the object's current visible location as `source`, usually `tray` or `desk`.
- Prefer `target: "basket"` unless the user requests another destination.
- If a green package/box is plausibly medicine, use `item_name: "medicine_box"`.
- Do not call `pick_and_place` for books, notebooks, papers, generic boxes, or other unsupported items.

## Visual Feedback Rules

- You can inspect the current image directly.
- Use post-action visual delta feedback to infer what actually changed.
- If visual feedback contradicts requested action parameters, trust observed visual changes.
- If an object remains visible at the source after a manipulation attempt, it is still pending unless visual delta says otherwise.
- If the target/destination visibly contains the moved object, treat that object as handled.
- If identity is uncertain, say so in `visual_state` and prefer `get_observation` or a concise clarification.
- Do not mark cleanup complete while supported clutter is still visible on the desk/tray.

## Communication Rules

- Do not ask for step-by-step confirmations.
- Do not report every tool/action execution to the user.
- Speak only for final completion, important failure, or essential clarification.
- Do not use remote agent messaging.

## JSON Output

Return exactly one JSON object:

```json
{
  "current_step_analysis": {
    "visual_state": "Brief current visual state",
    "task_progress": "What has been completed and what remains",
    "next_action_reasoning": "Why the next action follows from the image and feedback"
  },
  "next_step": {
    "step_number": 1,
    "agent": "ur5e_arm",
    "location": "home",
    "action": "pick_and_place",
    "action_type": "act",
    "parameters": {"item_name": "water", "source": "tray", "target": "basket"}
  },
  "needs_human_input": false,
  "user_question": null
}
```

For task completion:

```json
{
  "current_step_analysis": {
    "visual_state": "Final visible state",
    "task_progress": "Checklist completed",
    "next_action_reasoning": "Task complete"
  },
  "next_step": null,
  "task_summary": {
    "total_steps_executed": 5,
    "final_visual_state": "Desk is ready for reading",
    "actions_performed": ["pick_and_place", "control_light", "play_audio", "speak"],
    "success": true
  },
  "needs_human_input": false,
  "user_question": null
}
```

Always return valid JSON. No markdown, no code fences, no extra commentary.
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


def _get_context_window_tokens(model: str) -> int:
    env_key = f"PLANNER_CONTEXT_WINDOW_TOKENS_{model.upper().replace('-', '_').replace('.', '_')}"
    if os.getenv(env_key):
        return int(os.getenv(env_key, "128000"))
    conservative_defaults = {
        "gemini-2.0-flash-exp": 128000,
        "gemini-2.0-flash-thinking-exp-01-21": 128000,
        "gemini-3-pro-preview": 128000,
        "gemini-3-flash-preview": 128000,
        "gemini-2.5-flash-lite": 128000,
    }
    return conservative_defaults.get(model, int(os.getenv("PLANNER_CONTEXT_WINDOW_TOKENS", "128000")))


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
            "temperature": 0.7,
            "context_window_tokens": _get_context_window_tokens("gemini-2.0-flash-exp"),
        },
        "gemini-2.0-flash-thinking-exp-01-21": {
            "model": "gemini-2.0-flash-thinking-exp-01-21",
            "base_url": None,
            "max_tokens": 3000,
            "temperature": 0.7,
            "context_window_tokens": _get_context_window_tokens("gemini-2.0-flash-thinking-exp-01-21"),
        },
        "gemini-3-pro-preview": {
            "model": "gemini-3-pro-preview",
            "base_url": None,
            "max_tokens": 3000,
            "temperature": 0.7,
            "context_window_tokens": _get_context_window_tokens("gemini-3-pro-preview"),
        },
        "gemini-2.5-flash-lite": {
            "model": "gemini-2.5-flash-lite",
            "base_url": None,
            "max_tokens": 2000,
            "temperature": 0.7,
            "context_window_tokens": _get_context_window_tokens("gemini-2.5-flash-lite"),
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
        "temperature": 0.7,
        "context_window_tokens": _get_context_window_tokens(model),
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
        if data["needs_human_input"] and not (data.get("user_question") or data.get("humanoid_question")):
            return False, "needs_human_input=true but no user_question provided"

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
    "location": "home",
    "action": "pick_and_place",
    "action_type": "act",
    "parameters": {"item_name": "requested_item", "source": "source_location", "target": "target_location"}
  },
  "needs_human_input": false,
  "user_question": null
}
```
"""

    is_valid, message = validate_arm_vlm_response(test_response)
    print(f"\nValidation Test: {'✅ PASS' if is_valid else '❌ FAIL'}")
    print(f"Message: {message}")
