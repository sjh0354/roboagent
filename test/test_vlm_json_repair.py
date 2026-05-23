import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.base_vlm_planner import BaseVLMPlanner


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeVLMClient:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _Response("")
        return _Response(
            '{"current_step_analysis":{"visual_state":"clear","task_progress":"done",'
            '"next_action_reasoning":"finish"},"next_step":null,"needs_human_input":false}'
        )


class _RepairPlanner(BaseVLMPlanner):
    def __init__(self):
        super().__init__(
            profile_name="ur5e",
            api_key="test_api_key",
            model_name="gemini-2.5-flash-lite",
            verbose=False,
            config_getter=lambda _model: {
                "model": "gemini-2.5-flash-lite",
                "max_tokens": 512,
                "temperature": 0.2,
                "context_window_tokens": 32000,
            },
            system_prompt_getter=lambda: "",
            legacy_prompt_getter=lambda: "",
        )
        self.vlm_client = _FakeVLMClient()
        self.is_task_complete = False
        self.original_request = "reading setup"
        self.conversation_history = []
        self.execution_history = []

    def _reset_runtime_state(self):
        return None

    def _print_task_started(self, request: str):
        return None

    def _prepare_current_observation(self):
        return "unused.jpg"

    def _get_current_image_for_planning(self) -> str:
        return "unused.jpg"

    def _execute_step(self, step_plan):
        return None

    def _build_context_message(self) -> str:
        return "context"

    def _parse_step_plan(self, response_text: str):
        if not response_text:
            return {"error": "JSON parsing failed: empty response", "raw_response": ""}
        return {
            "current_step_analysis": {},
            "next_step": None,
            "needs_human_input": False,
        }

    def _display_step_plan(self, step_plan):
        return None

    def _is_waiting_for_input(self) -> bool:
        return False

    def _set_waiting_state(self, question):
        return None

    def _clear_waiting_state(self):
        return None

    def _get_pending_question(self):
        return None

    def _get_waiting_status(self) -> str:
        return "returned"

    def _get_question_field(self) -> str:
        return "question"

    def _get_input_role_name(self) -> str:
        return "human"

    def _get_response_prefix(self) -> str:
        return "human"

    def _build_observation_user_content(self, observation_packet):
        return []


def test_plan_retries_once_when_vlm_response_is_not_json():
    planner = _RepairPlanner()

    result = planner.plan_next_step_with_image("unused.jpg")

    assert result["next_step"] is None
    assert planner.is_task_complete is True
    assert len(planner.vlm_client.calls) == 2
    assert planner.vlm_client.calls[0]["response_mime_type"] == "application/json"
    assert planner.vlm_client.calls[1]["temperature"] == 0
    assert "Return ONLY one valid JSON object" in planner.vlm_client.calls[1]["messages"][-1]["content"]
