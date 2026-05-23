import json

from template.arm_prompt_template_vlm import validate_arm_vlm_response


def _response_for(action, action_type, parameters=None):
    return json.dumps(
        {
            "current_step_analysis": {
                "visual_state": "workspace visible",
                "task_progress": "reading setup in progress",
                "next_action_reasoning": "next required checklist action",
            },
            "next_step": {
                "step_number": 1,
                "agent": "ur5e_arm",
                "location": "store",
                "action": action,
                "action_type": action_type,
                "parameters": parameters or {},
            },
            "needs_human_input": False,
            "humanoid_question": None,
        }
    )


def test_arm_validator_accepts_single_device_iot_and_audio_actions():
    light_ok, light_msg = validate_arm_vlm_response(
        _response_for(
            "control_light",
            "tool",
            {"action": "set_brightness", "device": "background_light", "brightness": 30},
        )
    )
    audio_ok, audio_msg = validate_arm_vlm_response(
        _response_for(
            "play_audio",
            "tool",
            {"action": "play", "audio_type": "white_noise", "volume": 35},
        )
    )

    assert light_ok, light_msg
    assert audio_ok, audio_msg


def test_arm_validator_rejects_robot_to_robot_message_action():
    is_valid, message = validate_arm_vlm_response(
        _response_for("send_agent_message", "talk", {"message": "please clean desk", "recipient": "g1"})
    )

    assert is_valid is False
    assert "Invalid action 'send_agent_message'" in message
