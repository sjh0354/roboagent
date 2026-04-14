import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.base_vlm_planner import BaseVLMPlanner


class _TestPlanner(BaseVLMPlanner):
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

    def _did_just_speak(self, step_plan) -> bool:
        return False

    def _normalize_observation_packet(self, observation_input):
        return observation_input

    def _build_observation_user_content(self, observation_packet):
        return []

    def _build_transient_memory_context(self):
        return None

    def _prepare_transient_frames(self, frames):
        return []


def test_long_term_rule_summary_keeps_stable_preferences_in_runtime_summary_only(tmp_path, monkeypatch):
    planner = _TestPlanner()
    source_text = "\n".join(
        [
            "Task request: 帮我拿一瓶喝的。",
            "USER: 我平时严格控糖，以后默认按无糖偏好处理，不要拿含糖饮料。",
            "USER: 今晚很困，需要提神。",
        ]
    )

    payload = planner._generate_hierarchical_memory_summary(
        level="long_term",
        source_text=source_text,
        metadata={"fallback_topic": "drink_choice"},
    )

    assert payload["topic"] == "sugar_free_preference"
    assert "Stable user preferences" in payload["summary"]
    assert "Decision rules" in payload["summary"]
    assert "Default to sugar-free drink options for this user" in payload["summary"]
    preferences_path = tmp_path / "memory" / "local" / "user_preferences.md"
    assert not preferences_path.exists()


def test_short_term_rule_summary_keeps_temporary_need_bounded_by_durable_constraints():
    planner = _TestPlanner()
    source_text = "\n".join(
        [
            "USER: 我平时严格控糖，以后默认按无糖偏好处理，不要拿含糖饮料。",
            "USER: 今晚很困，需要提神。",
            "Task: choose one drink for the user.",
        ]
    )

    payload = planner._generate_hierarchical_memory_summary(
        level="short_term",
        source_text=source_text,
        metadata={"fallback_topic": "drink_choice"},
    )

    assert payload["topic"] == "sugar_free_preference"
    assert "Current needs" in payload["summary"]
    assert "The current task favors a drink that helps the user stay awake." in payload["summary"]
    assert "Apply temporary needs only after preserving durable user preferences and hard constraints." in payload["summary"]
    assert "Treat stimulation as a temporary need only within the boundary of durable user preferences and hard constraints." in payload["summary"]
    assert "Never override durable user preferences, dietary restrictions, or safety constraints with a short-term convenience need." in payload["summary"]
    assert "After satisfying hard constraints, prefer an allowed drink that better matches the current need for stimulation." not in payload["summary"]


def test_short_term_merge_sanitizes_overly_absolute_stimulation_rules():
    planner = _TestPlanner()

    heuristic_payload = planner._build_rule_memory_payload(
        level="short_term",
        source_text="\n".join(
            [
                "USER: 我平时严格控糖，以后默认按无糖偏好处理，不要拿含糖饮料。",
                "USER: 今晚很困，需要提神。",
            ]
        ),
        metadata={"fallback_topic": "drink_choice"},
        fallback_topic="sugar_free_preference",
    )
    parsed_payload = {
        "topic": "coordination_completion",
        "hard_preferences": [],
        "soft_preferences": [
            "prioritize items that provide sustained energy",
            "prioritize items that are mentally stimulating",
        ],
        "current_needs": ["user is feeling sleepy"],
        "forbidden_items": [],
        "decision_rules": [
            "when selecting items, prioritize those that offer sustained energy and mental stimulation, while avoiding overly sweet options."
        ],
        "stable_user_preferences": [],
    }

    payload = planner._merge_rule_memory_payload(
        level="short_term",
        heuristic_payload=heuristic_payload,
        parsed_payload=parsed_payload,
        fallback_topic="sugar_free_preference",
    )

    assert "prioritize items that provide sustained energy" not in payload["summary"]
    assert "prioritize items that are mentally stimulating" not in payload["summary"]
    assert "prioritize those that offer sustained energy and mental stimulation" not in payload["summary"]
    assert "Apply temporary needs only after preserving durable user preferences and hard constraints." in payload["summary"]
    assert "Never override durable user preferences, dietary restrictions, or safety constraints with a short-term convenience need." in payload["summary"]


def test_caffeine_avoidance_is_not_treated_as_stimulation_need():
    planner = _TestPlanner()
    source_text = "\n".join(
        [
            "USER: 我晚上不想摄入咖啡因。",
            "Task: choose one drink for the user.",
        ]
    )

    payload = planner._generate_hierarchical_memory_summary(
        level="long_term",
        source_text=source_text,
        metadata={"fallback_topic": "drink_choice"},
    )

    assert "Avoid caffeinated drinks when choosing beverages for this user." in payload["summary"]
    assert "Treat caffeine avoidance as a durable drink-selection constraint." in payload["summary"]
    assert "The current task favors a drink that helps the user stay awake." not in payload["summary"]


def test_normalize_legacy_action_payload_maps_to_next_step():
    planner = _TestPlanner()
    payload = planner._normalize_step_plan_payload(
        {
            "action": {
                "skill": "store_memory",
                "parameters": {
                    "content": "strictly sugar-free; do not offer sugary drinks",
                    "scope": "global",
                    "category": "preference",
                },
            },
            "state": {"step": 1, "task": "remember user preference"},
        }
    )

    assert payload["next_step"]["action"] == "store_memory"
    assert payload["next_step"]["action_type"] == "tool"
    assert payload["next_step"]["parameters"]["category"] == "preference"
    assert payload["needs_human_input"] is False


def test_normalize_next_step_fills_parameters_from_action_parameters_alias():
    planner = _TestPlanner()
    payload = planner._normalize_step_plan_payload(
        {
            "action": "store_memory",
            "action_parameters": {
                "content": "default beverage is plain water; do not offer flavored drinks unless explicitly requested",
                "scope": "global",
                "category": "preference",
            },
            "next_step": {
                "step_number": 1,
                "action": "store_memory",
                "action_type": "tool",
                "parameters": {},
            },
        }
    )

    assert payload["next_step"]["action"] == "store_memory"
    assert payload["next_step"]["parameters"]["content"].startswith("default beverage is plain water")
    assert payload["next_step"]["parameters"]["category"] == "preference"


def test_normalize_next_step_fills_parameters_from_action_config_alias():
    planner = _TestPlanner()
    payload = planner._normalize_step_plan_payload(
        {
            "action": "store_memory",
            "action_config": {
                "content": "default snack preference is original flavor, light taste, no sweet snacks unless explicitly requested",
                "scope": "global",
                "category": "preference",
            },
            "next_step": {
                "step_number": 1,
                "action": "store_memory",
                "action_type": "tool",
                "parameters": {},
            },
        }
    )

    assert payload["next_step"]["action"] == "store_memory"
    assert payload["next_step"]["parameters"]["content"].startswith("default snack preference")
    assert payload["next_step"]["parameters"]["scope"] == "global"


def test_normalize_legacy_action_object_with_action_name_maps_to_next_step():
    planner = _TestPlanner()
    payload = planner._normalize_step_plan_payload(
        {
            "action": {
                "action_name": "store_memory",
                "action_type": "tool",
                "parameters": {
                    "content": "default snack preference: plain, mild flavors; avoid sweet snacks unless specified",
                    "scope": "global",
                    "category": "preference",
                },
            }
        }
    )

    assert payload["next_step"]["action"] == "store_memory"
    assert payload["next_step"]["action_type"] == "tool"
    assert payload["next_step"]["parameters"]["category"] == "preference"


def test_normalize_tool_code_print_wrapper_maps_to_next_step():
    planner = _TestPlanner()
    payload = planner._normalize_step_plan_payload(
        {
            "tool_code": "print(store_memory.store_memory(content='default preference: original flavor, light snacks, no sweet snacks', scope='global', category='preference'))"
        }
    )

    assert payload["next_step"]["action"] == "store_memory"
    assert payload["next_step"]["action_type"] == "tool"
    assert payload["next_step"]["parameters"]["scope"] == "global"
    assert payload["next_step"]["parameters"]["category"] == "preference"
