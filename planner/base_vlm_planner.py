"""
Shared runtime for modular VLM planners.
"""

import os
from datetime import datetime
from typing import Dict, Optional

from template.modular_prompt_loader import build_runtime_system_prompt
from utils.gemini_vlm_client import GeminiVLMClient
from utils.memory_manager import MemoryManager


class BaseVLMPlanner:
    """Shared planner runtime for embodied VLM agents."""

    def __init__(
        self,
        profile_name: str,
        api_key: Optional[str],
        model_name: str,
        verbose: bool,
        config_getter,
        system_prompt_getter,
        legacy_prompt_getter,
    ):
        self.api_key = api_key or os.getenv("GENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Set GENAI_API_KEY environment variable or provide api_key parameter."
            )

        self.profile_name = profile_name
        self.config = config_getter(model_name)
        self.verbose = verbose
        self.system_prompt = system_prompt_getter()
        self.legacy_system_prompt = legacy_prompt_getter()
        self.vlm_client = GeminiVLMClient(
            api_key=self.api_key,
            model_name=model_name,
            verbose=verbose,
        )
        self.memory_manager = MemoryManager(profile_name=self.profile_name, verbose=verbose)

    def reset_conversation(self):
        """Reset planner state for a new task."""
        self.original_request = None
        self.conversation_history = []
        self.execution_history = []
        self.step_count = 0
        self.is_task_complete = False
        self.task_start_time = None
        self.memory_candidates = {}
        self.memory_recorded = False
        self._reset_runtime_state()

    def start_new_task(self, request: str, run_autonomously: bool = True) -> Dict:
        """Start a new task using the shared autonomous loop."""
        self.reset_conversation()
        self.original_request = request
        self.task_start_time = datetime.now()
        self._print_task_started(request)

        if run_autonomously:
            return self._run_autonomous_loop()
        return self.plan_next_step()

    def _run_autonomous_loop(self) -> Dict:
        """Shared half-open-loop execution."""
        while not self.is_task_complete and not self._is_waiting_for_input():
            interrupt_result = self._check_interrupt()
            if interrupt_result:
                return self._finalize_interaction(interrupt_result)

            current_image = self._prepare_current_observation()
            step_plan = self.plan_next_step_with_image(current_image)

            if step_plan.get("error"):
                if self.verbose:
                    print(f"❌ Planning error: {step_plan['error']}")
                return self._finalize_interaction(step_plan)

            if step_plan.get("needs_human_input"):
                just_spoke = False
                if step_plan.get("next_step"):
                    self._execute_step(step_plan)
                    just_spoke = self._did_just_speak(step_plan)

                question = step_plan.get(self._get_question_field())
                self._set_waiting_state(question)
                self._handle_pause_prompt(just_spoke=just_spoke, question=question)
                return self._finalize_interaction(step_plan)

            if step_plan.get("next_step") is None:
                self.is_task_complete = True
                if self.verbose:
                    print("\n✅ TASK COMPLETE (Half-Open-Loop Mode)")
                return self._finalize_interaction(step_plan)

            self._execute_step(step_plan)

        if self.is_task_complete:
            return self._finalize_interaction(self.get_task_summary())
        if self._is_waiting_for_input():
            return self._finalize_interaction(
                {
                    "status": self._get_waiting_status(),
                    "question": self._get_pending_question(),
                    "step_count": self.step_count,
                }
            )
        return self._finalize_interaction({"status": "returned"})

    def provide_input_response(self, response: str) -> Dict:
        """Resume the shared autonomous loop after user/agent input."""
        if not self._is_waiting_for_input():
            return {"error": f"Not waiting for {self._get_input_role_name()} input"}

        if self.verbose:
            print(f"\n💬 {self._get_input_role_name().capitalize()} response received: {response}")

        self.conversation_history.append(
            {
                "role": "user",
                "content": f"[{self._get_response_prefix()}]: {response}",
            }
        )

        self._clear_waiting_state()
        self.memory_recorded = False
        return self._run_autonomous_loop()

    def plan_next_step_with_image(self, image_path: str) -> Dict:
        """Shared VLM planning call."""
        if self.is_task_complete:
            if self.verbose:
                print("⚠️  Task already complete")
            return {"error": "Task already complete"}

        context_text = self._build_context_message()
        system_prompt = self._build_runtime_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)

        user_content = []
        if os.path.exists(image_path):
            user_content.append(self.vlm_client.create_image_message(image_path))
        else:
            user_content.append({"type": "text", "text": f"[Image not available: {image_path}]"})
        user_content.append({"type": "text", "text": context_text})
        messages.append({"role": "user", "content": user_content})

        try:
            if self.verbose:
                print(f"\n🧠 Planning next step with VLM ({self.config['model']})...")

            response = self.vlm_client.create_chat_completion(
                model=self.config["model"],
                messages=messages,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
            )
            response_text = response.choices[0].message.content
            step_plan = self._parse_step_plan(response_text)
            self.conversation_history.append({"role": "assistant", "content": response_text})

            if step_plan.get("next_step"):
                self._display_step_plan(step_plan)
            elif step_plan.get("next_step") is None and not step_plan.get("needs_human_input"):
                self.is_task_complete = True

            return step_plan
        except Exception as error:
            error_msg = f"Planning failed: {str(error)}"
            if self.verbose:
                print(f"❌ Error: {error_msg}")
            return {"error": error_msg}

    def plan_next_step(self) -> Dict:
        """Plan the next step using the current observation source."""
        return self.plan_next_step_with_image(self._get_current_image_for_planning())

    def get_task_summary(self) -> Dict:
        """Return the shared task summary payload."""
        duration = None
        if self.task_start_time:
            duration = (datetime.now() - self.task_start_time).total_seconds()

        summary = {
            "profile_name": self.profile_name,
            "original_request": self.original_request,
            "steps_executed": self.step_count,
            "is_complete": self.is_task_complete,
            "duration_seconds": duration,
            "execution_history": self.execution_history,
        }
        summary.update(self._get_summary_extras())
        return summary

    def _build_runtime_system_prompt(self) -> str:
        return build_runtime_system_prompt(
            profile_name=self.profile_name,
            original_request=self.original_request or "",
            execution_history=self.execution_history,
            current_location=self._get_current_location_for_prompt(),
            legacy_prompt=self.legacy_system_prompt,
        )

    def _finalize_interaction(self, result: Dict) -> Dict:
        if self.memory_recorded:
            return result

        status = result.get("status")
        if not status:
            if result.get("is_complete"):
                status = "completed"
            elif result.get("needs_human_input"):
                status = self._get_waiting_status()
            elif result.get("error"):
                status = "error"
            else:
                status = "returned"

        self.memory_candidates = self.memory_manager.record_interaction(
            status=status,
            original_request=self.original_request,
            execution_history=self.execution_history,
            result=result,
        )
        self.memory_recorded = True
        result["memory_candidates"] = self.memory_candidates
        return result

    def _did_just_speak(self, step_plan: Dict) -> bool:
        action = step_plan.get("next_step", {}).get("action")
        return action in {"speak", "talk_with_human"}

    def _check_interrupt(self) -> Optional[Dict]:
        return None

    def _handle_pause_prompt(self, just_spoke: bool, question: Optional[str]):
        return None

    def _get_current_location_for_prompt(self) -> Optional[str]:
        return None

    def _get_summary_extras(self) -> Dict:
        return {}

    def _reset_runtime_state(self):
        raise NotImplementedError

    def _print_task_started(self, request: str):
        raise NotImplementedError

    def _prepare_current_observation(self) -> str:
        raise NotImplementedError

    def _get_current_image_for_planning(self) -> str:
        raise NotImplementedError

    def _execute_step(self, step_plan: Dict):
        raise NotImplementedError

    def _build_context_message(self) -> str:
        raise NotImplementedError

    def _parse_step_plan(self, response_text: str) -> Dict:
        raise NotImplementedError

    def _display_step_plan(self, step_plan: Dict):
        raise NotImplementedError

    def _is_waiting_for_input(self) -> bool:
        raise NotImplementedError

    def _set_waiting_state(self, question: Optional[str]):
        raise NotImplementedError

    def _clear_waiting_state(self):
        raise NotImplementedError

    def _get_pending_question(self) -> Optional[str]:
        raise NotImplementedError

    def _get_waiting_status(self) -> str:
        raise NotImplementedError

    def _get_question_field(self) -> str:
        raise NotImplementedError

    def _get_input_role_name(self) -> str:
        raise NotImplementedError

    def _get_response_prefix(self) -> str:
        raise NotImplementedError
