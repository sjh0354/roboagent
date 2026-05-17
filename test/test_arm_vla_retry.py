import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.arm_executor import ExecutionResult
from planner.arm_planner_vlm import AutonomousArmVLMPlanner


class _RetryExecutor:
    def __init__(self):
        self.calls = 0
        self.act_backend = "vla"

    def execute_action(self, action_type, action_name, parameters):
        self.calls += 1
        if self.calls == 1:
            return ExecutionResult(
                success=False,
                feedback="PI0 launched but made no progress",
                data={
                    "backend": "vla",
                    "retryable": True,
                    "failure_kind": "pi0_no_progress",
                    "pi0_steps": 0,
                    "observation_image": "simulation_images/store/default.jpg",
                },
                error="Detected 0 PI0 steps",
            )
        return ExecutionResult(
            success=True,
            feedback="PI0 retry succeeded",
            data={
                "backend": "vla",
                "pi0_steps": 12,
                "observation_image": "simulation_images/store/item_ready.jpg",
            },
        )

    def set_message_transport(self, transport):
        return None

    def get_current_observation(self):
        return {"image_path": "simulation_images/store/default.jpg"}

    def get_buffered_observations_between(self, start_ts, end_ts, max_frames=6):
        return []


class _RetryPlanner:
    _canonical_action_name = AutonomousArmVLMPlanner._canonical_action_name
    _infer_action_type = AutonomousArmVLMPlanner._infer_action_type
    _normalize_step_action = AutonomousArmVLMPlanner._normalize_step_action
    _normalize_step_plan_payload = AutonomousArmVLMPlanner._normalize_step_plan_payload
    _extract_aliased_parameters = AutonomousArmVLMPlanner._extract_aliased_parameters
    _max_vla_retries = AutonomousArmVLMPlanner._max_vla_retries
    _should_retry_failed_vla_action = AutonomousArmVLMPlanner._should_retry_failed_vla_action
    _build_retry_context = AutonomousArmVLMPlanner._build_retry_context
    _execute_step_real = AutonomousArmVLMPlanner._execute_step_real
    _build_context_message = AutonomousArmVLMPlanner._build_context_message
    _completed_recorded_cleanup_items = lambda self: set()
    _uses_recorded_trajectory_backend = lambda self: False
    _is_reading_desk_cleanup_request = lambda self: False
    _build_hierarchical_memory_context = lambda self: None
    _build_transient_memory_context = AutonomousArmVLMPlanner._build_transient_memory_context

    def __init__(self):
        self.executor = _RetryExecutor()
        self.profile_name = "ur5e"
        self.simulation_mode = False
        self.verbose = False
        self.execution_history = []
        self.step_count = 0
        self.is_task_complete = False
        self.task_failed = False
        self.task_error = None
        self.pending_retry_context = None
        self.transient_memory_packet = None
        self.vla_retry_count = 0
        self.original_request = "clear the reading desk"
        self.current_observation_image = "simulation_images/store/default.jpg"

    def _apply_real_trajectory_policy(self, next_step, parameters):
        return None

    def _build_transient_memory_packet(self, action_name, start_ts, end_ts, primary_image):
        return {
            "primary_image": primary_image,
            "frames": [
                {"image_path": "simulation_images/store/default.jpg", "label": "Frame 1 - before"},
                {"image_path": primary_image, "label": "Frame 2 - after"},
            ],
            "is_sequence": True,
            "summary_text": f"Long action '{action_name}' showed limited progress",
        }


def _step():
    return {
        "next_step": {
            "step_number": 1,
            "action_type": "act",
            "action": "pick_and_place",
            "parameters": {"item_name": "water", "source": "desk", "target": "basket"},
        }
    }


def test_retryable_vla_failure_keeps_arm_planner_alive(monkeypatch):
    monkeypatch.setenv("ARM_VLA_MAX_RETRIES", "1")
    planner = _RetryPlanner()

    planner._execute_step_real(_step())

    assert planner.is_task_complete is False
    assert planner.pending_retry_context is not None
    assert planner.pending_retry_context["failure_kind"] == "pi0_no_progress"
    assert planner.transient_memory_packet is not None
    assert "RETRYABLE PI0/VLA FAILURE" in planner._build_context_message()

    planner._execute_step_real(_step())

    assert planner.executor.calls == 2
    assert planner.pending_retry_context is None
    assert planner.vla_retry_count == 0


def test_observe_workspace_is_normalized_to_get_observation():
    planner = _RetryPlanner()

    payload = planner._normalize_step_plan_payload(
        {
            "current_step_analysis": {
                "visual_state": "workspace visible",
                "task_progress": "starting",
                "next_action_reasoning": "inspect before moving",
            },
            "next_step": {
                "step_number": 1,
                "action": "observe_workspace",
                "action_type": "sense",
                "parameters": {},
            },
            "needs_human_input": False,
        }
    )

    assert payload["next_step"]["action"] == "get_observation"
    assert payload["next_step"]["action_type"] == "sense"


def test_non_retryable_real_failure_marks_task_failed():
    planner = _RetryPlanner()
    planner.executor = type(
        "FailingExecutor",
        (),
        {
            "act_backend": "vla",
            "execute_action": lambda self, action_type, action_name, parameters: ExecutionResult(
                success=False,
                feedback="Unknown sense action",
                error="Action 'observe_workspace' not implemented",
                data={},
            ),
        },
    )()

    planner._execute_step_real(
        {
            "next_step": {
                "step_number": 1,
                "action_type": "sense",
                "action": "observe_workspace",
                "parameters": {},
            }
        }
    )

    assert planner.is_task_complete is True
    assert planner.task_failed is True
    assert planner.task_error == "Action 'observe_workspace' not implemented"
