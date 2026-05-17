import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.base_vlm_planner import BaseVLMPlanner


class _MemoryModePlanner(BaseVLMPlanner):
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

    def _reset_runtime_state(self):
        return None

    def _print_task_started(self, request: str):
        return None

    def _prepare_current_observation(self):
        return None

    def _get_current_image_for_planning(self) -> str:
        return ""

    def _execute_step(self, step_plan):
        return None

    def _build_context_message(self) -> str:
        return ""

    def _parse_step_plan(self, response_text: str):
        return {}

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


def test_memory_mode_none_disables_runtime_memory_context(monkeypatch):
    monkeypatch.setenv("OMNICLAW_MEMORY_MODE", "none")
    planner = _MemoryModePlanner()
    planner.reset_conversation()

    assert planner.memory_mode == "none"
    assert planner._build_hierarchical_memory_context() is None


def test_memory_mode_fixed_records_step_summaries(monkeypatch):
    monkeypatch.setenv("OMNICLAW_MEMORY_MODE", "fixed")
    monkeypatch.setenv("OMNICLAW_FIXED_MEMORY_STEP_INTERVAL", "2")
    planner = _MemoryModePlanner()
    planner.reset_conversation()
    planner.original_request = "reading setup"
    planner.step_count = 2
    planner.execution_history = [
        {"step_number": 1, "action": "control_light", "parameters": {"device": "background_light"}, "assumed_successful": True},
        {"step_number": 2, "action": "play_audio", "parameters": {"audio_type": "white_noise"}, "assumed_successful": True},
    ]

    planner._maybe_record_fixed_frequency_memory()
    context = planner._build_hierarchical_memory_context()

    assert planner.memory_mode == "fixed"
    assert context is not None
    assert "[NAIVE FIXED-FREQUENCY MEMORY]" in context
    assert "play_audio" in context
