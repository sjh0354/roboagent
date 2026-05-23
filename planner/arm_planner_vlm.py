# arm_planner_vlm.py

"""
Vision-Based Autonomous Robotic Arm Planner
Uses VLM (Vision-Language Model) with direct visual observations for planning
"""

import os
import sys
# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from typing import Dict, Optional

from executor.arm_executor_vision import VisionEnabledArmExecutor
# from utils.realsense_manager import RealSenseCameraManager # Removed: Handled by Executor
from template.arm_prompt_template_vlm import (
    get_arm_vlm_system_prompt,
    get_arm_legacy_system_prompt,
    get_arm_vlm_config,
    validate_arm_vlm_response,
    clean_json_response,
    list_available_vlm_models
)
from planner.base_vlm_planner import BaseVLMPlanner
from utils.message_transport import MessageTransport, create_message_transport


class AutonomousArmVLMPlanner(BaseVLMPlanner):
    """
    Autonomous Robotic Arm Planner with Vision-Language Model (Half-Open-Loop Mode)

    Key features:
    - VLM planner receives images directly (not text descriptions)
    - Autonomous execution loop without interactive feedback waiting (half-open-loop)
    - Executor failures stop the loop instead of being treated as completed actions
    - Direct user interaction for task responses, clarifications, and status updates
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
                 simulation_mode: bool = False,
                 verbose: bool = True,
                 volume: float = 1.0,
                 voice: str = "male",
                 transport: Optional[MessageTransport] = None,
                 control_method: Optional[str] = None):
        """
        Initialize VLM-based autonomous arm planner

        Args:
            api_key: Google GenAI API key
            model_name: VLM model name (default: from env DEFAULT_VLM_MODEL or gemini-2.0-flash-exp)
            simulation_mode: Use simulation images (True) or real camera (False)
            verbose: Print detailed logs
            volume: TTS playback volume (0.0 to 1.0)
            voice: Voice tone (default: 'male')
        """
        self.verbose = verbose
        self.simulation_mode = simulation_mode
        self.transport = transport
        self.task_failed = False
        self.task_error = None
        self.previous_observation_image = None
        self.initial_scene_summary = None
        super().__init__(
            profile_name="ur5e",
            api_key=api_key,
            model_name=model_name,
            verbose=verbose,
            config_getter=get_arm_vlm_config,
            system_prompt_getter=get_arm_vlm_system_prompt,
            legacy_prompt_getter=get_arm_legacy_system_prompt,
        )
        
        # Initialize Executor
        # Note: Executor handles Camera initialization internally. 
        # We must not initialize RealSenseCameraManager here to avoid "Device busy" errors.
        self.executor = VisionEnabledArmExecutor(
            simulation_mode=self.simulation_mode,
            verbose=verbose,
            enable_vision=True,
            vlm_model=model_name,
            volume=volume,
            voice=voice,
            act_backend=control_method,
        )
        self.executor.set_message_transport(self.transport)
        self.reset_conversation()

        if verbose:
            print(f"✅ AutonomousArmVLMPlanner initialized with model: {self.config['model']}")
            print(f"   Simulation mode: {simulation_mode}")
            print(f"   Volume: {int(volume * 100)}%")

    def reset_conversation(self):
        super().reset_conversation()
        self.vla_retry_count = 0

    def _reset_runtime_state(self):
        self.waiting_for_humanoid = False
        self.humanoid_question = None
        self.waiting_for_user = False
        self.user_question = None
        self.pending_retry_context = None
        self.task_failed = False
        self.task_error = None
        self.latest_visual_outcome = None
        self.initial_scene_summary = None

    def start_new_task(self, request: str, run_autonomously: bool = True,
                      observation_image: Optional[str] = None) -> Dict:
        if observation_image:
            self.current_observation_image = observation_image
            self.initial_scene_summary = None
        else:
            initial_obs = self._get_default_observation(include_vlm_description=True)
            self.current_observation_image = initial_obs.get("image_path") or "simulation_images/store/default.jpg"
            self.initial_scene_summary = initial_obs.get("vlm_description")
        return super().start_new_task(request, run_autonomously)

    def _print_task_started(self, request: str):
        if self.verbose:
            print("\n" + "="*70)
            print("🦾 NEW ARM TASK STARTED (VLM Half-Open-Loop Mode)")
            print("="*70)
            print(f"📝 Request: {request}")
            print(f"🕐 Started at: {self.task_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⚙️  Mode: Half-open-loop; executor failures stop the task")
            print("="*70 + "\n")

    def _get_default_observation_image(self) -> str:
        """Get default observation image for store workspace"""
        obs = self._get_default_observation(include_vlm_description=False)
        if obs.get("image_path"):
            return obs["image_path"]
        return "simulation_images/store/default.jpg"

    def _get_default_observation(self, include_vlm_description: bool = False) -> Dict:
        """Get initial/current workspace observation from the executor."""
        # Delegate to executor
        if hasattr(self, 'executor'):
             try:
                 obs = self.executor.get_current_observation(
                     include_vlm_description=include_vlm_description
                 )
             except TypeError:
                 obs = self.executor.get_current_observation()
             if obs.get('image_path'):
                 return obs

        # Fallback for simulation or failure
        return {"image_path": "simulation_images/store/default.jpg", "vlm_description": None}

    def provide_humanoid_response(self, response: str) -> Dict:
        return self.provide_input_response(response)

    def provide_user_response(self, response: str) -> Dict:
        return self.provide_input_response(response)

    def set_transport(self, transport: Optional[MessageTransport]):
        self.transport = transport
        self.executor.set_message_transport(transport)

    def plan_next_step_with_image(self, image_path: str) -> Dict:
        return super().plan_next_step_with_image(image_path)

    def plan_next_step(self) -> Dict:
        return super().plan_next_step()

    def _prepare_current_observation(self):
        if self.verbose:
            print(f"\n📸 Current observation: {self.current_observation_image}")
        if self.transient_memory_packet:
            packet = dict(self.transient_memory_packet)
            packet["primary_image"] = packet.get("primary_image") or self.current_observation_image
            return packet
        return self.current_observation_image

    def _get_current_image_for_planning(self) -> str:
        return self.current_observation_image

    def _handle_pause_prompt(self, just_spoke: bool, question: Optional[str]):
        if self.verbose:
            print(f"\n⏸️  PAUSED: Waiting for user input")
            print(f"❓ Question: {question}")

    def _execute_step(self, step_plan: Dict):
        """
        Execute step, dispatching based on simulation mode.

        Args:
            step_plan: Step plan from VLM planner
        """
        if self.simulation_mode:
            self._execute_step_simulation(step_plan)
        else:
            self._execute_step_real(step_plan)

    def _execute_step_real(self, step_plan: Dict):
        """Execute step using Executor on real hardware"""
        if "error" in step_plan or step_plan.get("next_step") is None:
            return

        next_step = dict(step_plan["next_step"])
        parameters = dict(next_step.get("parameters") or {})
        policy_adjustment = self._apply_real_trajectory_policy(next_step, parameters)
        if policy_adjustment == "complete":
            self.is_task_complete = True
            if self.verbose:
                print("\n🧭 Real trajectory policy: supported cleanup trajectories are already complete.")
            return

        # Record in execution history
        history_entry = {
            "step_number": next_step.get("step_number"),
            "action": next_step.get("action"),
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }
        if policy_adjustment:
            history_entry["policy_adjustment"] = policy_adjustment
        self.execution_history.append(history_entry)

        self.step_count += 1

        if self.verbose:
            print("\n" + "="*70)
            print(f"⚡ EXECUTING STEP {next_step.get('step_number')} (Real Hardware)")
            print("="*70)
            print(f"🎯 Action: {next_step.get('action')}")
            print(f"📦 Parameters: {json.dumps(parameters, indent=2)}")
            if policy_adjustment:
                print(f"🧭 Policy adjustment: {policy_adjustment}")
            print("="*70 + "\n")

        step_started_at = time.time()
        before_observation_image = self.current_observation_image
        # Execute via Executor
        result = self.executor.execute_action(
            next_step.get("action_type"),
            next_step.get("action"),
            parameters
        )
        step_finished_at = time.time()
        
        # Update observation image based on result
        if result.data.get("observation_image"):
             self.current_observation_image = result.data.get("observation_image")

        self.execution_history[-1]["assumed_successful"] = bool(result.success)
        self.execution_history[-1]["execution_result"] = result.to_dict()
        self.execution_history[-1]["duration_seconds"] = round(step_finished_at - step_started_at, 2)
        self.latest_visual_outcome = self._infer_post_action_visual_outcome(
            next_step=next_step,
            result=result,
            before_image=before_observation_image,
            after_image=self.current_observation_image,
            start_ts=step_started_at,
            end_ts=step_finished_at,
        )
        if self.latest_visual_outcome:
            self.execution_history[-1]["visual_outcome"] = self.latest_visual_outcome
        self.transient_memory_packet = None
        keep_success_frames = os.getenv("TRANSIENT_MEMORY_ON_SUCCESS", "0") == "1"
        if (not result.success) or keep_success_frames:
            self.transient_memory_packet = self._build_transient_memory_packet(
                action_name=next_step.get("action", "unknown"),
                start_ts=step_started_at,
                end_ts=step_finished_at,
                primary_image=result.data.get("observation_image") or self.current_observation_image,
            )

        if not result.success and self._should_retry_failed_vla_action(next_step, result):
            self.vla_retry_count += 1
            self.pending_retry_context = self._build_retry_context(next_step, result)
            if self.verbose:
                print(f"\n{'='*70}")
                print("📊 EXECUTION STATUS:")
                print(f"{'='*70}")
                print("✗ Action failed, but retry is allowed.")
                print(f"Retry budget: {self.vla_retry_count}/{self._max_vla_retries()}")
                print(f"Error: {result.error}")
                print("→ Replanning with transient visual memory...")
                print(f"{'='*70}\n")
            return

        if not result.success:
            self.is_task_complete = True
            self.task_failed = True
            self.task_error = result.error or result.feedback or "Arm action failed"
            if self.verbose:
                print(f"\n{'='*70}")
                print("📊 EXECUTION STATUS:")
                print(f"{'='*70}")
                print("✗ Action failed on hardware; stopping autonomous loop.")
                print(f"Error: {result.error}")
                print(f"{'='*70}\n")
            return

        self.pending_retry_context = None
        if (result.data or {}).get("backend") in {"vla", "pi0"}:
            self.vla_retry_count = 0

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📊 EXECUTION STATUS:")
            print(f"{'='*70}")
            print(f"✓ Action executed on hardware")
            print(f"→ Continuing to next step...")
            print(f"{'='*70}\n")

    def _max_vla_retries(self) -> int:
        return max(0, int(os.getenv("ARM_VLA_MAX_RETRIES", "1")))

    def _should_retry_failed_vla_action(self, next_step: Dict, result) -> bool:
        if self.simulation_mode:
            return False
        if next_step.get("action_type") != "act":
            return False
        data = result.data or {}
        backend = str(data.get("backend") or "").lower()
        if backend not in {"vla", "pi0"}:
            return False
        if not data.get("retryable"):
            return False
        return self.vla_retry_count < self._max_vla_retries()

    def _build_retry_context(self, next_step: Dict, result) -> Dict:
        data = result.data or {}
        return {
            "failed_action": next_step.get("action"),
            "failed_parameters": next_step.get("parameters") or {},
            "error": result.error,
            "feedback": result.feedback,
            "failure_kind": data.get("failure_kind"),
            "pi0_steps": data.get("pi0_steps"),
            "retry_count": self.vla_retry_count,
            "max_retries": self._max_vla_retries(),
        }

    def _infer_post_action_visual_outcome(
        self,
        *,
        next_step: Dict,
        result,
        before_image: Optional[str],
        after_image: Optional[str],
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> Optional[Dict]:
        """Build action-agnostic visual feedback for the next planning step."""
        if not after_image:
            return None

        parameters = next_step.get("parameters") or {}
        outcome = {
            "requested_action": next_step.get("action"),
            "requested_action_type": next_step.get("action_type"),
            "requested_parameters": parameters,
            "executor_success": bool(result.success),
            "after_image": after_image,
            "visual_delta_mode": "action_blind",
            "instruction": (
                "Use requested_action/requested_parameters as attempted-command metadata only. "
                "The visual_delta is generated without task intent by default; infer the next world state "
                "from observed visual changes, not from what was requested."
            ),
        }

        data = result.data or {}
        if data.get("vlm_observation"):
            outcome["post_action_observation"] = data.get("vlm_observation")

        process_images: list[str] = []
        if (
            start_ts is not None
            and end_ts is not None
            and hasattr(self, "_get_transient_memory_frames")
        ):
            max_process_frames = max(0, int(os.getenv("POST_ACTION_DELTA_MAX_PROCESS_FRAMES", "2")))
            if max_process_frames:
                frames = self._get_transient_memory_frames(start_ts, end_ts)
                prepared = self._prepare_transient_frames(frames)
                for frame in prepared[:max_process_frames]:
                    image_path = frame.get("image_path")
                    if image_path and image_path not in {before_image, after_image}:
                        process_images.append(image_path)

        should_compare = os.getenv("POST_ACTION_VLM_COMPARE", "1") == "1"
        can_compare = (
            should_compare
            and before_image
            and after_image
            and before_image != after_image
            and hasattr(self.vlm_client, "analyze_action_visual_delta")
        )
        if can_compare:
            try:
                include_action_context = os.getenv("POST_ACTION_DELTA_INCLUDE_ACTION_CONTEXT", "0") == "1"
                if include_action_context:
                    outcome["visual_delta_mode"] = "action_context"
                visual_delta = self.vlm_client.analyze_action_visual_delta(
                    before_image=before_image,
                    after_image=after_image,
                    process_images=process_images,
                    action_name=str(next_step.get("action") or "unknown"),
                    action_parameters=parameters,
                    include_action_context=include_action_context,
                )
                outcome["visual_delta"] = visual_delta
            except Exception as error:
                if self.verbose:
                    print(f"⚠️  Post-action visual delta failed: {str(error)}")
                outcome["visual_delta_error"] = str(error)

        return outcome

    def _apply_real_trajectory_policy(self, next_step: Dict, parameters: Dict) -> Optional[str]:
        """Constrain desk-cleaning real execution to the recorded replay trajectories."""
        if not self._uses_recorded_trajectory_backend():
            return None
        if next_step.get("action_type") != "act" or next_step.get("action") != "pick_and_place":
            return None
        if not self._is_reading_desk_cleanup_request():
            return None

        planned_text = " ".join(
            str(parameters.get(key, ""))
            for key in ("item_name", "instruction", "task_description", "description")
        )
        planned_item = self._trajectory_item_from_text(planned_text)
        completed_items = self._completed_recorded_cleanup_items()
        remaining_items = [item for item in ("water", "medicine") if item not in completed_items]

        if planned_item in {"water", "medicine"}:
            before = dict(parameters)
            parameters["item_name"] = planned_item
            parameters["source"] = parameters.get("source") or "tray"
            parameters["target"] = "basket"
            if parameters != before:
                return f"normalized supported cleanup item to {planned_item} -> basket"
            return None

        if remaining_items:
            replacement_item = remaining_items[0]
            original_item = parameters.get("item_name", "unknown")
            parameters.clear()
            parameters.update({
                "item_name": replacement_item,
                "source": "tray",
                "target": "basket",
            })
            next_step["action_type"] = "act"
            next_step["action"] = "pick_and_place"
            return (
                f"replaced unsupported cleanup item '{original_item}' "
                f"with recorded trajectory {replacement_item} -> basket"
            )

        return "complete"

    def _uses_recorded_trajectory_backend(self) -> bool:
        executor = getattr(self, "executor", None)
        if executor and hasattr(executor, "_select_backend_for_action"):
            backend = executor._select_backend_for_action(quiet=True)
        else:
            backend = getattr(executor, "act_backend", "")
        return str(backend).strip().lower() in {"trajectory_replay", "trajectory", "replay"}

    def _is_reading_desk_cleanup_request(self) -> bool:
        request = (self.original_request or "").lower()
        cleanup_terms = ("clean", "clear", "tidy", "organize", "整理", "清理", "收拾")
        reading_terms = ("read", "book", "desk", "table", "看书", "读书", "桌")
        return any(term in request for term in cleanup_terms) and any(term in request for term in reading_terms)

    def _trajectory_item_from_text(self, text: str) -> Optional[str]:
        lowered = text.lower().replace("_", " ")
        if "water" in lowered or "mineral" in lowered:
            return "water"
        medicine_terms = ("medicine", "medication", "pill", "drug", "green box")
        if any(term in lowered for term in medicine_terms):
            return "medicine"
        return None

    def _completed_recorded_cleanup_items(self) -> set:
        completed = set()
        for exec_step in self.execution_history:
            if not exec_step.get("assumed_successful"):
                continue
            params = exec_step.get("parameters") or {}
            item = self._trajectory_item_from_text(
                " ".join(str(params.get(key, "")) for key in ("item_name", "instruction", "task_description"))
            )
            if item:
                completed.add(item)
        return completed

    def _execute_step_simulation(self, step_plan: Dict):
        """
        Simulate step execution (Internal Planner Simulation Logic)

        Args:
            step_plan: Step plan from VLM planner
        """
        if "error" in step_plan or step_plan.get("next_step") is None:
            return

        next_step = step_plan["next_step"]
        action_type = next_step.get("action_type")
        action_name = next_step.get("action")
        parameters = next_step.get("parameters") or {}

        # Record in execution history
        self.execution_history.append({
            "step_number": next_step.get("step_number"),
            "action": action_name,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        })

        self.step_count += 1

        # Display execution header
        if self.verbose:
            print("\n" + "="*70)
            print(f"⚡ EXECUTING STEP {next_step.get('step_number')} (Simulation Mode)")
            print("="*70)
            print(f"🎯 Action: {next_step.get('action')}")
            print(f"📦 Parameters: {json.dumps(next_step.get('parameters', {}), indent=2)}")
            print(f"✓ Assumed: SUCCESS (no verification)")
            print("="*70 + "\n")

        step_started_at = time.time()
        execution_result = None
        if action_type in {"tool", "talk", "sense"}:
            result = self.executor.execute_action(action_type, action_name, parameters)
            execution_result = result.to_dict()
            if result.data.get("observation_image"):
                self.current_observation_image = result.data.get("observation_image")
        else:
            # Update observation image based on action (simulation)
            self._update_observation_after_action(next_step)
        step_finished_at = time.time()

        # Mark as assumed successful
        self.execution_history[-1]["assumed_successful"] = True
        if execution_result:
            self.execution_history[-1]["execution_result"] = execution_result
        self.execution_history[-1]["duration_seconds"] = round(step_finished_at - step_started_at, 2)
        self.transient_memory_packet = self._build_transient_memory_packet(
            action_name=next_step.get("action", "unknown"),
            start_ts=step_started_at,
            end_ts=step_finished_at,
            primary_image=self.current_observation_image,
        )

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📊 EXECUTION STATUS:")
            print(f"{'='*70}")
            print(f"✓ Action assumed successful (half-open-loop mode)")
            print(f"→ Continuing to next step...")
            print(f"{'='*70}\n")

    def _update_observation_after_action(self, next_step: Dict):
        """Update observation image based on executed action (simulation)"""
        action = next_step.get("action")
        params = next_step.get("parameters", {})

        # Map actions to observation images (simulation)
        if action == "pick_and_place":
            item = params.get("item_name", "item")
            target = params.get("target", "counter")
            if target == "counter":
                self.current_observation_image = f"simulation_images/store/counter_{item}.jpg"
            else:
                self.current_observation_image = f"simulation_images/store/{target}_{item}.jpg"
        elif action == "speak":
            # No visual change for communication
            pass
        elif action == "get_observation":
            # No visual change for observation request
            pass
        else:
            # Default: keep current observation
            pass

    def _build_context_message(self) -> str:
        """Build context message for VLM planner"""
        context_parts = []

        # Original request
        context_parts.append(f"[ORIGINAL USER REQUEST]: {self.original_request}")

        initial_scene_summary = getattr(self, "initial_scene_summary", None)
        if initial_scene_summary:
            context_parts.append(
                "\n[INITIAL SCENE SUMMARY]: Captured once at task start for grounding. "
                "Do not treat it as current after actions; use the current image and "
                "post-action visual delta for state updates.\n"
                f"{initial_scene_summary}"
            )

        # Execution history
        if self.execution_history:
            context_parts.append(f"\n[STEPS EXECUTED SO FAR]: {len(self.execution_history)}")
            for exec_step in self.execution_history[-3:]:
                result = exec_step.get("execution_result") or {}
                status = result.get("success")
                if status is None:
                    status = exec_step.get("assumed_successful")
                status_text = "executor_success" if status else "executor_failed"
                context_parts.append(
                    f"  - Step {exec_step['step_number']}: {exec_step['action']} "
                    f"with {exec_step['parameters']} -> {status_text}"
                )
        else:
            context_parts.append("\n[STEPS EXECUTED SO FAR]: None (this is the first step)")

        if self._is_reading_desk_cleanup_request():
            context_parts.append(
                "\n[READING DESK CLEANUP COMPLETION RULE]: Treat execution history as attempted commands, "
                "not guaranteed state changes. Use the current image and post-action visual outcome feedback "
                "to decide what actually changed before choosing the next step."
            )
            if self._final_reading_confirmation_spoken():
                context_parts.append(
                    "\n[TERMINATION REQUIRED]: The final reading setup confirmation has already been spoken "
                    "successfully. Do not call `speak` again. Return the task-completion JSON now with "
                    "`next_step: null`, `needs_human_input: false`, and `user_question: null`."
                )

        if self.latest_visual_outcome:
            context_parts.append(
                "\n[POST-ACTION VISUAL OUTCOME FEEDBACK]: This feedback compares the image before the "
                "last action with the image after it. By default the visual_delta is generated action-blind, "
                "without seeing the requested action parameters. Trust observed visual changes over the "
                "requested action parameters when updating the world state.\n"
                f"{json.dumps(self.latest_visual_outcome, ensure_ascii=False, indent=2)}"
            )

        # Current status
        context_parts.append(f"\n[CURRENT STATUS]: Planning step #{self.step_count + 1}")

        hierarchical_memory_context = self._build_hierarchical_memory_context()
        if hierarchical_memory_context:
            context_parts.append(f"\n{hierarchical_memory_context}")

        transient_memory_context = self._build_transient_memory_context()
        if transient_memory_context:
            context_parts.append(f"\n{transient_memory_context}")

        if self.pending_retry_context:
            retry_context = json.dumps(
                self.pending_retry_context,
                ensure_ascii=False,
                indent=2,
            )
            context_parts.append(
                "\n[RETRYABLE PI0/VLA FAILURE]: The previous physical action did not "
                "complete successfully. Use the ordered transient visual memory frames "
                "above to decide whether the arm is stuck, made no progress, or needs "
                "the same skill called again with clearer parameters. Retry only if the "
                "object and destination are still valid.\n"
                f"{retry_context}"
            )

        if self._uses_recorded_trajectory_backend() and self._is_reading_desk_cleanup_request():
            completed_items = self._completed_recorded_cleanup_items()
            remaining_items = [item for item in ("water", "medicine") if item not in completed_items]
            context_parts.append(
                "\n[REAL TRAJECTORY POLICY]: This real robot can only replay recorded cleanup "
                "trajectories for water and medicine. For this reading/desk-cleaning task, "
                "move remaining supported clutter to basket in this order: "
                f"{remaining_items or 'none'}. Never plan tray, book, notebook, paper, or counter moves."
            )

        # Instruction
        context_parts.append(
            "\n[INSTRUCTION]: Based on the visual observation (image(s) provided above) and execution history, "
            "plan the NEXT SINGLE STEP. Use the JSON format specified in the system prompt."
        )

        return "\n".join(context_parts)

    def _final_reading_confirmation_spoken(self) -> bool:
        """Return True once the one-time reading setup final confirmation succeeded."""
        final_message = "桌子已清理，阅读灯和背景音已为您打开。"
        for exec_step in reversed(self.execution_history):
            if exec_step.get("action") != "speak":
                continue
            params = exec_step.get("parameters") or {}
            message = str(params.get("message") or "")
            if final_message not in message:
                continue
            result = exec_step.get("execution_result") or {}
            status = result.get("success")
            if status is None:
                status = exec_step.get("assumed_successful")
            return bool(status)
        return False

    def _parse_step_plan(self, response_text: str) -> Dict:
        """Parse VLM response into step plan"""
        response_text = response_text or ""
        cleaned_text = clean_json_response(response_text)

        is_valid, message = validate_arm_vlm_response(response_text)
        if not is_valid:
            if self.verbose:
                print(f"⚠️  Validation warning: {message}")

        try:
            payload = json.loads(cleaned_text)
            if isinstance(payload, dict):
                normalized = self._normalize_step_plan_payload(payload)
                if normalized.get("user_question") is None and normalized.get("humanoid_question"):
                    normalized["user_question"] = normalized["humanoid_question"]
                return normalized
            return payload
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ JSON parsing failed: {str(e)}")
                print(f"Raw VLM response preview: {repr(response_text[:500])}")
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response_text[:500]
            }

    def _display_step_plan(self, step_plan: Dict):
        """Display step plan in readable format"""
        if not self.verbose:
            return

        print("\n" + "="*70)
        print("📋 NEXT STEP PLAN (Half-Open-Loop)")
        print("="*70)

        # Analysis
        analysis = step_plan.get("current_step_analysis", {})
        print(f"\n🔍 ANALYSIS:")
        print(f"  Visual State: {analysis.get('visual_state', 'N/A')}")
        print(f"  Progress: {analysis.get('task_progress', 'N/A')}")
        print(f"  Reasoning: {analysis.get('next_action_reasoning', 'N/A')}")

        # Next step
        next_step = step_plan.get("next_step")
        if next_step:
            print(f"\n⚡ NEXT STEP #{next_step.get('step_number', '?')}:")
            print(f"  Action: {next_step.get('action')}")
            print(f"  Type: {next_step.get('action_type')}")
            print(f"  Parameters: {json.dumps(next_step.get('parameters', {}), indent=4)}")

        print("="*70 + "\n")

    def _is_waiting_for_input(self) -> bool:
        return self.waiting_for_user

    def _set_waiting_state(self, question: Optional[str]):
        self.waiting_for_user = True
        self.user_question = question
        self.waiting_for_humanoid = True
        self.humanoid_question = question

    def _clear_waiting_state(self):
        self.waiting_for_user = False
        self.user_question = None
        self.waiting_for_humanoid = False
        self.humanoid_question = None

    def _get_pending_question(self) -> Optional[str]:
        return self.user_question

    def _get_waiting_status(self) -> str:
        return "waiting_for_user"

    def _get_question_field(self) -> str:
        return "user_question"

    def _get_input_role_name(self) -> str:
        return "user"

    def _get_response_prefix(self) -> str:
        return "USER RESPONSE"

    def _get_summary_extras(self) -> Dict:
        return {
            "success": not self.task_failed,
            "error": self.task_error,
        }

    def _get_transient_memory_frames(self, start_ts: float, end_ts: float):
        if hasattr(self.executor, "get_buffered_observations_between"):
            return self.executor.get_buffered_observations_between(start_ts, end_ts, max_frames=6)
        return []

    def switch_model(self, model_name: str):
        """Switch to different VLM model"""
        old_model = self.config['model']
        self.config = get_arm_vlm_config(model_name)
        self.vlm_client.switch_model(model_name)
        if self.verbose:
            print(f"🔄 Switched VLM model: {old_model} → {model_name}")

    def close(self):
        """Release executor-owned resources such as camera pipelines."""
        executor = getattr(self, "executor", None)
        if executor and hasattr(executor, "close"):
            executor.close()


def main():
    """Main function for testing"""
    import argparse
    import select

    parser = argparse.ArgumentParser(description="Autonomous arm VLM planner")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode using local simulation images")
    parser.add_argument("--log", action="store_true", help="Show detailed planner/executor logs")
    parser.add_argument(
        "--model",
        default=os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
        help="VLM model name. Default: DEFAULT_VLM_MODEL or gemini-2.0-flash-exp",
    )
    parser.add_argument(
        "--transport",
        default=os.getenv("INTERACTION_TRANSPORT", "voice"),
        help="Interaction transport: voice | lark | openclaw_lark | none",
    )
    parser.add_argument("--lark-target", default=os.getenv("LARK_TARGET") or os.getenv("OPENCLAW_LARK_TARGET"), help="Lark/Feishu target chat id")
    parser.add_argument("--lark-account", default=os.getenv("LARK_ACCOUNT_ID") or os.getenv("OPENCLAW_LARK_ACCOUNT"), help="Lark bot account id")
    parser.add_argument(
        "--control-method",
        choices=["vla", "pi0", "replay", "trajectory_replay", "anygrasp"],
        default=os.getenv("ARM_CONTROL_METHOD") or os.getenv("ARM_ACT_BACKEND") or "vla",
        help="Real arm control backend. Default: vla/pi0; if its policy port is unavailable, fallback to replay.",
    )
    args = parser.parse_args()
    verbose = args.log
    simulation_mode = args.simulation

    print("🦾 Autonomous VLM-Based Robotic Arm Planner")
    print("="*70)

    # Check API key
    if not os.getenv("GENAI_API_KEY"):
        print("⚠️  GENAI_API_KEY environment variable not set")
        print("Set it using: export GENAI_API_KEY='your-key'")
        return

    transport = None
    planner = None
    try:
        # Initialize planner
        transport = create_message_transport(
            args.transport,
            verbose=verbose,
            target=args.lark_target,
            account_id=args.lark_account,
        )
        if transport:
            transport.start()

        planner = AutonomousArmVLMPlanner(
            model_name=args.model,
            simulation_mode=simulation_mode,
            verbose=verbose,
            transport=transport,
            control_method=args.control_method,
        )

        print("\n" + "="*70)
        print("🎮 AUTONOMOUS ARM VLM PLANNER - Ready")
        print("="*70)
        print("📋 Commands:")
        print(f"  - Mode: {'simulation' if simulation_mode else 'real'}")
        print(f"  - Control method: {planner.executor.act_backend}")
        print(f"  - Transport: {args.transport}")
        print("  - Speak wake word (e.g. '你好机器人') followed by your request when using voice transport")
        print("  - Type a user request directly")
        print("  - 'models' or 'm': List available VLM models")
        print("  - 'switch <model>': Switch VLM model")
        print("  - 'status' or 's': Show task status")
        print("  - 'reset' or 'r': Reset for new task")
        print("  - 'quit' or 'q': Exit")
        print("="*70)

        while True:
            user_input = None
            
            # 1. Check for voice command (non-blocking)
            voice_cmd = transport.get_command() if transport else None
            if voice_cmd:
                print(f"\n🎙️  Transport command detected: {voice_cmd}")
                user_input = voice_cmd
            
            # 2. Check for keyboard input (non-blocking)
            if not user_input:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    user_input = sys.stdin.readline().strip()
            
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'q']:
                print("👋 Goodbye!")
                break

            elif user_input.lower() in ['models', 'm']:
                list_available_vlm_models()
                continue

            elif user_input.lower().startswith('switch '):
                model_name = user_input[7:].strip()
                planner.switch_model(model_name)
                continue

            elif user_input.lower() in ['status', 's']:
                summary = planner.get_task_summary()
                print("\n📊 TASK STATUS:")
                if verbose:
                    print(json.dumps(summary, indent=2))
                else:
                    print(f"  request: {summary.get('original_request')}")
                    print(f"  steps: {summary.get('steps_executed')}")
                    print(f"  complete: {summary.get('is_complete')}")
                continue

            elif user_input.lower() in ['reset', 'r']:
                planner.reset_conversation()
                print("🔄 Reset complete. Ready for new task.")
                continue

            # Start new task
            result = planner.start_new_task(user_input, run_autonomously=True)

            # Handle result (Clarification)
            if result.get("status") in {"waiting_for_user", "waiting_for_humanoid"}:
                print(f"\n❓ {result['question']}")
                print("Your response (speak or type) > ", end="", flush=True)
                
                response = None
                while not response:
                    # Check voice
                    v_res = transport.get_response() if transport else None
                    if v_res:
                        print(f"{v_res} (transport)")
                        response = v_res
                    
                    # Check keyboard
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        response = sys.stdin.readline().strip()
                
                result = planner.provide_user_response(response)

            # Show final result
            if result.get("error") or result.get("success") is False:
                print(f"\n❌ Task failed: {result.get('error') or 'Unknown error'}")
            elif result.get("is_complete"):
                print("\n✅ Task completed successfully!")
                if not verbose:
                    print(f"   steps: {result.get('steps_executed')}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if planner:
            planner.close()
        if transport:
            transport.stop()


if __name__ == "__main__":
    main()
