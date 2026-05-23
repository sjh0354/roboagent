"""
Shared runtime for modular VLM planners.
"""

import ast
import json
import os
import re
from typing import Any, List
from datetime import datetime
from typing import Dict, Optional

from template.modular_prompt_loader import build_runtime_system_prompt
from utils.gemini_vlm_client import GeminiVLMClient
from utils.hierarchical_memory_manager import HierarchicalMemoryManager
from utils.memory_manager import MemoryManager
from utils.action_registry import get_action_schema, get_action_to_skill


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
        self.api_key = (
            api_key
            or os.getenv("GENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("PLANNER_VLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SHARESAI_API_KEY")
            or os.getenv("PI0_ACTION_MONITOR_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "API key not found. Set GENAI_API_KEY, PLANNER_VLM_API_KEY, OPENAI_API_KEY, "
                "SHARESAI_API_KEY, or provide api_key parameter."
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
        self.hierarchical_memory_manager = HierarchicalMemoryManager(
            profile_name=self.profile_name,
            verbose=verbose,
        )
        self.active_context_trigger_ratio = float(
            os.getenv("ACTIVE_CONTEXT_TRIGGER_RATIO", os.getenv("MEMORY_TRIGGER_RATIO", "0.25"))
        )
        self.active_context_keep_recent_messages = int(
            os.getenv("ACTIVE_CONTEXT_KEEP_RECENT_MESSAGES", "4")
        )
        self.context_window_tokens = int(
            self.config.get("context_window_tokens")
            or os.getenv("PLANNER_CONTEXT_WINDOW_TOKENS", "32000")
        )
        self.hierarchical_memory_manager.configure_runtime(
            context_window_tokens=self.context_window_tokens,
        )
        self.memory_mode = self._normalize_memory_mode(os.getenv("OMNICLAW_MEMORY_MODE", "full"))
        self.fixed_memory_step_interval = int(os.getenv("OMNICLAW_FIXED_MEMORY_STEP_INTERVAL", "2"))
        self.fixed_memory_max_summaries = int(os.getenv("OMNICLAW_FIXED_MEMORY_MAX_SUMMARIES", "4"))

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
        self.transient_memory_packet = None
        self.runtime_memory_session_id = None
        self.runtime_memory_cursor = {"conversation_history": 0, "execution_history": 0}
        self.active_context_last_compaction_at = None
        self.fixed_memory_summaries = []
        self.fixed_memory_last_step = 0
        self._reset_runtime_state()

    def start_new_task(self, request: str, run_autonomously: bool = True) -> Dict:
        """Start a new task using the shared autonomous loop."""
        self.reset_conversation()
        self.original_request = request
        self.task_start_time = datetime.now()
        self.runtime_memory_session_id = self.hierarchical_memory_manager.start_session(
            started_at=self.task_start_time,
        )
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

            observation_input = self._prepare_current_observation()
            step_plan = self.plan_next_step_with_image(observation_input)

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
            self._maybe_record_fixed_frequency_memory()

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

    def plan_next_step_with_image(self, observation_input: Any) -> Dict:
        """Shared VLM planning call."""
        if self.is_task_complete:
            if self.verbose:
                print("⚠️  Task already complete")
            return {"error": "Task already complete"}

        self._maybe_compact_active_context()
        observation_packet = self._normalize_observation_packet(observation_input)
        context_text = self._build_context_message()
        system_prompt = self._build_runtime_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)

        user_content = self._build_observation_user_content(observation_packet)
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
                response_mime_type="application/json",
            )
            response_text = response.choices[0].message.content
            step_plan = self._parse_step_plan(response_text)
            if step_plan.get("error"):
                repair_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON for the planner. "
                            "Return ONLY one valid JSON object matching the required schema. "
                            "Do not include markdown, comments, explanations, or code fences."
                        ),
                    }
                ]
                if self.verbose:
                    print("🔁 Retrying VLM planning with strict JSON-only instruction...")
                response = self.vlm_client.create_chat_completion(
                    model=self.config["model"],
                    messages=repair_messages,
                    max_tokens=self.config["max_tokens"],
                    temperature=0,
                    response_mime_type="application/json",
                )
                response_text = response.choices[0].message.content
                step_plan = self._parse_step_plan(response_text)
            self.conversation_history.append({"role": "assistant", "content": response_text})

            if step_plan.get("next_step"):
                self._display_step_plan(step_plan)
            elif step_plan.get("next_step") is None and not step_plan.get("needs_human_input"):
                self.is_task_complete = True

            self.transient_memory_packet = None
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
            include_memory=self.memory_mode != "none",
        )

    def _finalize_interaction(self, result: Dict) -> Dict:
        if self.memory_recorded:
            return result

        status = result.get("status")
        if not status:
            if result.get("error") or result.get("success") is False:
                status = "error"
            elif result.get("is_complete"):
                status = "completed"
            elif result.get("needs_human_input"):
                status = self._get_waiting_status()
            else:
                status = "returned"

        if self.memory_mode == "none":
            self.memory_recorded = True
            return result

        self.memory_candidates = self.memory_manager.record_interaction(
            status=status,
            original_request=self.original_request,
            execution_history=self.execution_history,
            result=result,
        )
        self.memory_recorded = True
        result["memory_candidates"] = self.memory_candidates
        hierarchical_memory = self._record_hierarchical_memory(status=status, result=result)
        if hierarchical_memory:
            result["hierarchical_memory"] = hierarchical_memory
        return result

    def _did_just_speak(self, step_plan: Dict) -> bool:
        action = step_plan.get("next_step", {}).get("action")
        return action in {"speak", "talk_with_human"}

    def _normalize_observation_packet(self, observation_input: Any) -> Dict[str, Any]:
        if isinstance(observation_input, dict):
            packet = dict(observation_input)
            frames = packet.get("frames") or []
            packet["frames"] = frames
            packet["primary_image"] = packet.get("primary_image") or (
                frames[-1].get("image_path") if frames else None
            )
            packet["is_sequence"] = len(frames) > 1
            return packet

        image_path = str(observation_input)
        return {
            "primary_image": image_path,
            "frames": [{"image_path": image_path, "label": "Current frame"}],
            "is_sequence": False,
        }

    def _build_observation_user_content(self, observation_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = []
        frames = observation_packet.get("frames") or []

        if frames:
            for index, frame in enumerate(frames, start=1):
                image_path = frame.get("image_path")
                label = frame.get("label") or f"Frame {index}"
                user_content.append({"type": "text", "text": f"[{label}]"})
                if image_path and os.path.exists(image_path):
                    user_content.append(self.vlm_client.create_image_message(image_path))
                else:
                    user_content.append({"type": "text", "text": f"Image not available: {image_path}"})
        else:
            image_path = observation_packet.get("primary_image")
            if image_path and os.path.exists(image_path):
                user_content.append(self.vlm_client.create_image_message(image_path))
            else:
                user_content.append({"type": "text", "text": f"[Image not available: {image_path}]"})

        summary_text = observation_packet.get("summary_text")
        if summary_text:
            user_content.append({"type": "text", "text": summary_text})
        return user_content

    def _build_transient_memory_context(self) -> Optional[str]:
        packet = self.transient_memory_packet
        if not packet or not packet.get("frames"):
            return None

        lines = [
            "[TRANSIENT VISUAL MEMORY]",
            packet.get("summary_text", "").strip(),
        ]
        for frame in packet.get("frames", []):
            label = frame.get("label")
            if label:
                lines.append(f"  - {label}")
        return "\n".join(line for line in lines if line)

    def _build_transient_memory_packet(
        self,
        action_name: str,
        start_ts: float,
        end_ts: float,
        primary_image: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        duration = end_ts - start_ts
        threshold = float(os.getenv("TRANSIENT_MEMORY_THRESHOLD_SECONDS", "5.0"))
        if duration < threshold:
            return None

        frames = self._get_transient_memory_frames(start_ts, end_ts)
        prepared_frames = self._prepare_transient_frames(frames)
        if primary_image and (
            not prepared_frames or prepared_frames[-1].get("image_path") != primary_image
        ):
            prepared_frames.append(
                {
                    "image_path": primary_image,
                    "label": f"Frame {len(prepared_frames) + 1} - t=+{duration:.0f}s",
                }
            )

        if len(prepared_frames) < 2:
            return None

        return {
            "primary_image": primary_image or prepared_frames[-1].get("image_path"),
            "frames": prepared_frames,
            "is_sequence": True,
            "summary_text": f"Long action '{action_name}' took {duration:.1f}s",
        }

    def _get_transient_memory_frames(self, start_ts: float, end_ts: float) -> List[Any]:
        return []

    def _prepare_transient_frames(self, frames: List[Any]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        if not frames:
            return prepared

        base_ts = frames[0].timestamp
        for index, frame in enumerate(frames, start=1):
            image_path = getattr(frame, "image_path", None)
            if not image_path:
                continue
            delta = max(0.0, getattr(frame, "timestamp", base_ts) - base_ts)
            prepared.append(
                {
                    "image_path": image_path,
                    "label": f"Frame {index} - t=+{delta:.0f}s",
                }
            )
        return prepared

    def _check_interrupt(self) -> Optional[Dict]:
        return None

    def _handle_pause_prompt(self, just_spoke: bool, question: Optional[str]):
        return None

    def _get_current_location_for_prompt(self) -> Optional[str]:
        return None

    def _get_summary_extras(self) -> Dict:
        return {}

    def _build_hierarchical_memory_context(self) -> Optional[str]:
        if self.memory_mode == "none":
            return None
        if self.memory_mode == "fixed":
            return self._build_fixed_frequency_memory_context()
        return self.hierarchical_memory_manager.build_context_block(
            session_id=self.runtime_memory_session_id,
        )

    def _maybe_compact_active_context(self) -> None:
        if self.memory_mode != "full":
            return
        keep_recent = max(0, self.active_context_keep_recent_messages)
        if len(self.conversation_history) <= keep_recent:
            return

        older_messages = self.conversation_history[:-keep_recent]
        source_text = self._render_conversation_messages(older_messages)
        if not source_text:
            return

        threshold_tokens = max(1, int(self.context_window_tokens * self.active_context_trigger_ratio))
        should_compact = self._estimate_tokens(source_text) >= threshold_tokens
        if not should_compact and self.active_context_last_compaction_at:
            elapsed = datetime.now() - self.active_context_last_compaction_at
            should_compact = elapsed.total_seconds() >= int(
                os.getenv("MEMORY_FORCE_SUMMARY_SECONDS", "7200")
            )
        if not should_compact:
            return

        if not self.runtime_memory_session_id:
            self.runtime_memory_session_id = self.hierarchical_memory_manager.start_session()

        summary_source = "\n".join(
            [
                f"Task request: {self.original_request or 'N/A'}",
                "",
                "[OLDER CONVERSATION TO COMPRESS]",
                source_text,
            ]
        ).strip()
        entry = self.hierarchical_memory_manager.store_active_context_summary(
            session_id=self.runtime_memory_session_id,
            source_text=summary_source,
            summary_callback=self._generate_hierarchical_memory_summary,
            trigger_reason="active_context_threshold",
            metadata_extra={
                "fallback_topic": self._fallback_topic_from_request(),
            },
        )
        if not entry:
            return

        self.conversation_history = self.conversation_history[-keep_recent:]
        self.runtime_memory_cursor["conversation_history"] = min(
            self.runtime_memory_cursor["conversation_history"],
            len(self.conversation_history),
        )
        self.active_context_last_compaction_at = datetime.now()
        if self.verbose:
            print(
                f"🧠 Active context compacted into short-term memory: {entry['id']}"
            )

    def _record_hierarchical_memory(self, status: str, result: Dict) -> Optional[Dict[str, Any]]:
        if self.memory_mode != "full":
            return None
        if not self.runtime_memory_session_id:
            self.runtime_memory_session_id = self.hierarchical_memory_manager.start_session()

        new_messages = self.conversation_history[self.runtime_memory_cursor["conversation_history"]:]
        new_steps = self.execution_history[self.runtime_memory_cursor["execution_history"]:]
        self.runtime_memory_cursor["conversation_history"] = len(self.conversation_history)
        self.runtime_memory_cursor["execution_history"] = len(self.execution_history)

        event_text = self._render_runtime_event_text(
            new_messages=new_messages,
            new_steps=new_steps,
            status=status,
            result=result,
        )
        event = self.hierarchical_memory_manager.append_raw_event(
            session_id=self.runtime_memory_session_id,
            text=event_text,
            status=status,
            metadata={
                "step_count": self.step_count,
                "is_complete": bool(result.get("is_complete")),
            },
        )
        force_rollup = bool(
            result.get("is_complete")
            or status.startswith("waiting_")
            or status in {"error", "interrupted"}
        )
        rollup = self.hierarchical_memory_manager.maybe_rollup_raw(
            session_id=self.runtime_memory_session_id,
            summary_callback=self._generate_hierarchical_memory_summary,
            force=force_rollup,
            trigger_reason="interaction_finalize",
            metadata_extra={
                "status": status,
                "is_complete": bool(result.get("is_complete")),
                "latest_action": new_steps[-1].get("action") if new_steps else None,
                "fallback_topic": self._fallback_topic_from_request(),
            },
        )
        if not event and not rollup:
            return None
        return {
            **({"event_id": event["id"]} if event else {}),
            **({"rollup": rollup} if rollup else {}),
        }

    def _normalize_memory_mode(self, mode: str) -> str:
        normalized = str(mode or "full").strip().lower()
        aliases = {
            "off": "none",
            "without": "none",
            "without_memory": "none",
            "w/o": "none",
            "w/o_memory": "none",
            "naive": "fixed",
            "fixed_frequency": "fixed",
            "fixed-frequency": "fixed",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"full", "none", "fixed"}:
            if self.verbose:
                print(f"⚠️  Unknown OMNICLAW_MEMORY_MODE={mode!r}; falling back to full")
            return "full"
        return normalized

    def _maybe_record_fixed_frequency_memory(self) -> None:
        if self.memory_mode != "fixed":
            return
        interval = max(1, self.fixed_memory_step_interval)
        if len(self.execution_history) - self.fixed_memory_last_step < interval:
            return

        new_steps = self.execution_history[self.fixed_memory_last_step:]
        if not new_steps:
            return
        summary = self._render_fixed_frequency_memory_summary(new_steps)
        self.fixed_memory_summaries.append(summary)
        max_summaries = max(1, self.fixed_memory_max_summaries)
        self.fixed_memory_summaries = self.fixed_memory_summaries[-max_summaries:]
        self.fixed_memory_last_step = len(self.execution_history)

    def _render_fixed_frequency_memory_summary(self, steps: List[Dict[str, Any]]) -> str:
        lines = [
            f"Naive fixed-frequency summary after step {self.step_count}.",
            f"Task request: {self.original_request or 'N/A'}",
        ]
        for step in steps:
            result = step.get("execution_result") or {}
            success = result.get("success")
            if success is None:
                success = step.get("assumed_successful")
            lines.append(
                "Step {step_number}: {action} {parameters} -> {status}".format(
                    step_number=step.get("step_number", "?"),
                    action=step.get("action", "unknown"),
                    parameters=step.get("parameters") or {},
                    status="success" if success else "failed",
                )
            )
        return "\n".join(lines)

    def _build_fixed_frequency_memory_context(self) -> Optional[str]:
        if not self.fixed_memory_summaries:
            return None
        lines = ["[NAIVE FIXED-FREQUENCY MEMORY]"]
        for index, summary in enumerate(self.fixed_memory_summaries[-self.fixed_memory_max_summaries:], start=1):
            lines.append(f"Summary {index}:")
            lines.append(summary)
        return "\n".join(lines)

    def _render_runtime_event_text(
        self,
        *,
        new_messages: List[Dict[str, Any]],
        new_steps: List[Dict[str, Any]],
        status: str,
        result: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        if self.original_request:
            lines.append(f"Task request: {self.original_request}")
        lines.append(f"Interaction status: {status}")

        rendered_messages = self._render_conversation_messages(new_messages)
        if rendered_messages:
            lines.extend(["", "[NEW CONVERSATION]", rendered_messages])

        if new_steps:
            lines.extend(["", "[NEW EXECUTION STEPS]"])
            for step in new_steps:
                result_dict = step.get("execution_result") or {}
                lines.append(
                    f"- Step {step.get('step_number')}: {step.get('action')} "
                    f"with {step.get('parameters', {})} "
                    f"=> {'SUCCESS' if result_dict.get('success', True) else 'FAILED'}"
                )
                feedback = result_dict.get("feedback")
                if feedback:
                    lines.append(f"  feedback: {feedback}")

        if result.get("question"):
            lines.append(f"\nPending question: {result['question']}")
        if result.get("human_question"):
            lines.append(f"\nPending question: {result['human_question']}")
        if result.get("error"):
            lines.append(f"\nError: {result['error']}")
        if result.get("success") is False:
            lines.append("\nResult: Task failed.")
        elif result.get("is_complete"):
            lines.append("\nResult: Task marked complete.")
        return "\n".join(line for line in lines if line).strip()

    def _render_conversation_messages(self, messages: List[Dict[str, Any]]) -> str:
        rendered: List[str] = []
        for message in messages:
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content_text = "\n".join(part for part in text_parts if part)
            else:
                content_text = str(content)
            content_text = content_text.strip()
            if not content_text:
                continue
            rendered.append(f"{role}: {content_text}")
        return "\n".join(rendered).strip()

    def _estimate_tokens(self, text: str) -> int:
        cleaned = (text or "").strip()
        if not cleaned:
            return 0
        return max(1, len(cleaned) // 4)

    def _normalize_step_plan_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        next_step_number = getattr(self, "step_count", 0) + 1
        if isinstance(payload.get("next_step"), dict):
            next_step = payload["next_step"]
            next_step.setdefault("step_number", next_step_number)
            self._normalize_step_action(next_step)
            if not next_step.get("action_type"):
                next_step["action_type"] = self._infer_action_type(next_step.get("action"))
            aliased_parameters = self._extract_aliased_parameters(payload)
            current_parameters = next_step.get("parameters")
            if isinstance(current_parameters, dict) and current_parameters:
                next_step["parameters"] = current_parameters
            elif isinstance(aliased_parameters, dict) and aliased_parameters:
                next_step["parameters"] = aliased_parameters
            else:
                next_step.setdefault("parameters", {})
            payload.setdefault("needs_human_input", False)
            return payload

        legacy_action = payload.get("action")
        if isinstance(legacy_action, dict):
            skill = (
                legacy_action.get("skill")
                or legacy_action.get("action")
                or legacy_action.get("action_name")
            )
            parameters = (
                legacy_action.get("parameters")
                or legacy_action.get("args")
                or legacy_action.get("action_arguments")
                or {}
            )
            if isinstance(skill, str):
                payload["next_step"] = {
                    "step_number": next_step_number,
                    "action": skill,
                    "action_type": legacy_action.get("action_type") or self._infer_action_type(skill),
                    "parameters": parameters if isinstance(parameters, dict) else {},
                }
                self._normalize_step_action(payload["next_step"])
                payload.setdefault(
                    "current_step_analysis",
                    {
                        "visual_state": "",
                        "task_progress": "",
                        "next_action_reasoning": "",
                    },
                )
                payload.setdefault("needs_human_input", False)
            return payload

        if isinstance(payload.get("action"), str):
            action_name = payload.get("action")
            parameters = self._extract_aliased_parameters(payload) or {}
            payload["next_step"] = {
                "step_number": next_step_number,
                "action": action_name,
                "action_type": self._infer_action_type(action_name),
                "parameters": parameters if isinstance(parameters, dict) else {},
            }
            self._normalize_step_action(payload["next_step"])
            payload.setdefault(
                "current_step_analysis",
                {
                    "visual_state": "",
                    "task_progress": "",
                    "next_action_reasoning": "",
                },
            )
            payload.setdefault("needs_human_input", False)
            return payload

        tool_code_step = self._extract_tool_code_next_step(payload.get("tool_code"), next_step_number)
        if tool_code_step:
            payload["next_step"] = tool_code_step
            self._normalize_step_action(payload["next_step"])
            payload.setdefault(
                "current_step_analysis",
                {
                    "visual_state": "",
                    "task_progress": "",
                    "next_action_reasoning": "",
                },
            )
            payload.setdefault("needs_human_input", False)
            return payload

        if isinstance(payload.get("module"), str):
            module_name = str(payload.get("module", "")).strip()
            action_name = module_name.split("/")[-1] if module_name else ""
            parameters = self._extract_aliased_parameters(payload) or {}
            if action_name:
                payload["next_step"] = {
                    "step_number": next_step_number,
                    "action": action_name,
                    "action_type": self._infer_action_type(action_name),
                    "parameters": parameters if isinstance(parameters, dict) else {},
                }
                self._normalize_step_action(payload["next_step"])
                payload.setdefault(
                    "current_step_analysis",
                    {
                        "visual_state": "",
                        "task_progress": "",
                        "next_action_reasoning": "",
                    },
                )
                payload.setdefault("needs_human_input", False)
        return payload

    def _normalize_step_action(self, next_step: Dict[str, Any]) -> None:
        action_name = next_step.get("action")
        canonical_action = self._canonical_action_name(action_name)
        if canonical_action != action_name:
            next_step["action"] = canonical_action
        inferred_type = self._infer_action_type(canonical_action)
        if inferred_type:
            next_step["action_type"] = inferred_type

    def _canonical_action_name(self, action_name: Any) -> Any:
        if not isinstance(action_name, str) or not action_name.strip():
            return action_name

        normalized = action_name.strip()
        if get_action_schema(self.profile_name, normalized):
            return normalized

        for action, skill in get_action_to_skill(self.profile_name).items():
            if normalized == skill:
                return action

        return normalized

    def _extract_aliased_parameters(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for key in (
            "parameters",
            "args",
            "action_args",
            "action_arguments",
            "action_parameters",
            "action_config",
        ):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        return {}

    def _extract_tool_code_next_step(
        self,
        tool_code: Any,
        step_number: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(tool_code, str) or not tool_code.strip():
            return None
        call_node = self._parse_tool_code_call(tool_code)
        if call_node is None:
            return None
        action_name = self._extract_call_name(call_node.func)
        if not action_name:
            return None
        parameters: Dict[str, Any] = {}
        for keyword in call_node.keywords:
            if not keyword.arg:
                continue
            try:
                parameters[keyword.arg] = ast.literal_eval(keyword.value)
            except Exception:
                continue
        return {
            "step_number": step_number,
            "action": action_name,
            "action_type": self._infer_action_type(action_name),
            "parameters": parameters,
        }

    def _parse_tool_code_call(self, tool_code: str) -> Optional[ast.Call]:
        try:
            parsed = ast.parse(tool_code.strip(), mode="exec")
        except SyntaxError:
            return None
        if not parsed.body:
            return None
        first_stmt = parsed.body[0]
        if not isinstance(first_stmt, ast.Expr) or not isinstance(first_stmt.value, ast.Call):
            return None
        outer_call = first_stmt.value
        if self._extract_call_name(outer_call.func) == "print" and outer_call.args:
            first_arg = outer_call.args[0]
            if isinstance(first_arg, ast.Call):
                return first_arg
        return outer_call

    def _extract_call_name(self, func_node: ast.AST) -> Optional[str]:
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None

    def _infer_action_type(self, action_name: Optional[str]) -> str:
        if not action_name:
            return "act"
        schema = get_action_schema(self.profile_name, action_name)
        action_type = schema.get("action_type") if isinstance(schema, dict) else None
        if isinstance(action_type, str) and action_type:
            return action_type
        return "act"

    def _generate_hierarchical_memory_summary(
        self,
        level: str,
        source_text: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, str]:
        fallback_topic = metadata.get("fallback_topic") or f"{level}_{self.profile_name}"
        rule_topic = self._detect_topic_from_rules(level=level, source_text=source_text, metadata=metadata)
        heuristic_payload = self._build_rule_memory_payload(
            level=level,
            source_text=source_text,
            metadata=metadata,
            fallback_topic=rule_topic or fallback_topic,
        )
        if self._should_skip_summary_model():
            return {
                "topic": heuristic_payload["topic"],
                "summary": heuristic_payload["summary"],
            }

        system_prompt = (
            "You are a memory compaction assistant for an embodied planner. "
            "Do not summarize chronology. Extract the durable decision rules future planning must obey. "
            "Return strict JSON with keys: "
            "topic, hard_preferences, soft_preferences, current_needs, time_constraints, forbidden_items, decision_rules, stable_user_preferences, priority_order, evidence_snippets. "
            "The topic must be a short lowercase slug with underscores. "
            "Each list item must be concise, factual, and action-guiding. "
            "Preserve hard constraints and decision priority even if they seem repetitive. "
            "Never collapse a hard constraint into a soft preference."
        )
        level_specific_guidance = [
            "Extract only the information that should affect future decisions.",
            "Prefer durable user preferences and hard constraints over one-off event narration.",
            "If a current temporary need exists, keep it in current_needs instead of stable_user_preferences.",
        ]
        if level == "short_term":
            level_specific_guidance.extend(
                [
                    "For short_term summaries, capture only the current-session objective, active deadline, and a fixed low-priority notice.",
                    "Treat the temporary need as conditional guidance inside existing durable constraints.",
                    "Never let current_needs or decision_rules override durable user preferences, dietary rules, or safety constraints.",
                    "If there is tension, encode the rule as: satisfy the temporary need only within the boundary of durable constraints.",
                    "Use time_constraints only for the single most important active deadline or immediate timing pressure in this task.",
                    "current_needs should contain at most one concise item.",
                    "time_constraints should contain at most one concise item.",
                    "soft_preferences should usually be empty for short_term summaries.",
                    "stable_user_preferences should usually be empty for short_term summaries unless the source is explicitly a durable preference statement that should later be promoted.",
                    "Do not create a new durable preference from a one-off request.",
                    "priority_order should usually be a single fixed notice that this short-term objective is lower priority than long-term preferences, safety rules, and dietary constraints.",
                ]
            )
        elif level == "long_term_overview":
            level_specific_guidance.extend(
                [
                    "For long_term_overview, retain only the most decision-critical stable rules that still matter across cases.",
                    "Keep hard constraints, stable tradeoff rules, and recurring time-sensitive patterns if they influence future choices.",
                ]
            )
        user_prompt = "\n".join(
            [
                f"Level: {level}",
                f"Device: {self.profile_name}",
                f"Trigger: {metadata.get('trigger_reason', 'unknown')}",
                f"Suggested topic: {rule_topic or fallback_topic}",
                "",
                *level_specific_guidance,
                "",
                "Source text:",
                source_text,
            ]
        )
        try:
            response = self.vlm_client.create_chat_completion(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=min(800, self.config.get("max_tokens", 800)),
                temperature=0.2,
            )
            parsed = self._parse_summary_json(response.choices[0].message.content)
            merged = self._merge_rule_memory_payload(
                level=level,
                heuristic_payload=heuristic_payload,
                parsed_payload=parsed,
                fallback_topic=rule_topic or fallback_topic,
            )
            return {"topic": merged["topic"], "summary": merged["summary"]}
        except Exception:
            return {
                "topic": heuristic_payload["topic"],
                "summary": heuristic_payload["summary"],
            }

    def _parse_summary_json(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        if not cleaned:
            return {}
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    return {}
        if not isinstance(data, dict):
            return {}

        return {
            "topic": str(data.get("topic", "")).strip(),
            "hard_preferences": self._coerce_list_field(data.get("hard_preferences")),
            "soft_preferences": self._coerce_list_field(data.get("soft_preferences")),
            "current_needs": self._coerce_list_field(data.get("current_needs")),
            "time_constraints": self._coerce_list_field(data.get("time_constraints")),
            "forbidden_items": self._coerce_list_field(data.get("forbidden_items")),
            "decision_rules": self._coerce_list_field(data.get("decision_rules")),
            "stable_user_preferences": self._coerce_list_field(data.get("stable_user_preferences")),
            "priority_order": self._coerce_list_field(data.get("priority_order")),
            "evidence_snippets": self._coerce_list_field(data.get("evidence_snippets")),
        }

    def _coerce_list_field(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _merge_rule_memory_payload(
        self,
        *,
        level: str,
        heuristic_payload: Dict[str, Any],
        parsed_payload: Dict[str, Any],
        fallback_topic: str,
    ) -> Dict[str, Any]:
        merged = {
            "topic": parsed_payload.get("topic") or heuristic_payload["topic"] or fallback_topic,
            "hard_preferences": parsed_payload.get("hard_preferences") or heuristic_payload["hard_preferences"],
            "soft_preferences": parsed_payload.get("soft_preferences") or heuristic_payload["soft_preferences"],
            "current_needs": parsed_payload.get("current_needs") or heuristic_payload["current_needs"],
            "time_constraints": parsed_payload.get("time_constraints") or heuristic_payload["time_constraints"],
            "forbidden_items": parsed_payload.get("forbidden_items") or heuristic_payload["forbidden_items"],
            "decision_rules": parsed_payload.get("decision_rules") or heuristic_payload["decision_rules"],
            "stable_user_preferences": parsed_payload.get("stable_user_preferences") or heuristic_payload["stable_user_preferences"],
            "priority_order": parsed_payload.get("priority_order") or heuristic_payload["priority_order"],
            "evidence_snippets": parsed_payload.get("evidence_snippets") or heuristic_payload["evidence_snippets"],
        }
        if level == "short_term":
            merged = self._sanitize_short_term_rule_payload(
                merged_payload=merged,
                heuristic_payload=heuristic_payload,
            )
        merged["summary"] = self._render_rule_memory_summary(merged)
        return merged

    def _sanitize_short_term_rule_payload(
        self,
        *,
        merged_payload: Dict[str, Any],
        heuristic_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        def _keep_short_term_soft_preference(rule: str) -> bool:
            lowered = str(rule or "").lower()
            if not lowered:
                return False
            blocked_markers = [
                "prioritize items that provide sustained energy",
                "prioritize items that are mentally stimulating",
                "mental stimulation",
                "sustained energy",
                "provide energy",
                "offer sustained energy",
            ]
            if any(marker in lowered for marker in blocked_markers):
                return False
            allowed_markers = [
                "allowed",
                "within the boundary",
                "within durable constraints",
                "safe fallback",
                "water remains a safe fallback",
            ]
            return any(marker in lowered for marker in allowed_markers)

        def _keep_short_term_decision_rule(rule: str) -> bool:
            lowered = str(rule or "").lower()
            if not lowered:
                return False
            required_markers = [
                "durable user preferences",
                "hard constraints",
                "dietary restrictions",
                "satisfy the temporary need only within",
                "temporary needs only after preserving",
                "never override durable user preferences",
                "allowed drink",
                "allowed option",
            ]
            if any(marker in lowered for marker in required_markers):
                return True
            blocked_markers = [
                "prioritize those that offer sustained energy",
                "mental stimulation",
                "provide sustained energy",
                "provide energy",
            ]
            return not any(marker in lowered for marker in blocked_markers)

        sanitized = dict(merged_payload)
        parsed_soft_preferences = merged_payload.get("soft_preferences") or []
        parsed_decision_rules = merged_payload.get("decision_rules") or []

        kept_soft_preferences = [
            rule for rule in parsed_soft_preferences if _keep_short_term_soft_preference(rule)
        ]
        kept_decision_rules = [
            rule for rule in parsed_decision_rules if _keep_short_term_decision_rule(rule)
        ]

        heuristic_soft_preferences = heuristic_payload.get("soft_preferences") or []
        heuristic_decision_rules = heuristic_payload.get("decision_rules") or []

        if heuristic_soft_preferences:
            kept_soft_preferences.extend(heuristic_soft_preferences)
        if heuristic_decision_rules:
            kept_decision_rules = heuristic_decision_rules + kept_decision_rules

        fixed_low_priority_notice = (
            "Lower priority than long-term preferences, safety rules, and dietary constraints. "
            "Use only to choose among options that already satisfy those constraints."
        )
        current_needs = self._dedupe_rules(merged_payload.get("current_needs") or [])
        time_constraints = self._dedupe_rules(merged_payload.get("time_constraints") or [])
        task_like_short_term = bool(current_needs or time_constraints)

        if task_like_short_term:
            sanitized["current_needs"] = current_needs[:1]
            sanitized["time_constraints"] = time_constraints[:1]
            sanitized["priority_order"] = [fixed_low_priority_notice]
            sanitized["soft_preferences"] = []
            sanitized["stable_user_preferences"] = []
            sanitized["forbidden_items"] = []
            sanitized["decision_rules"] = []
            sanitized["evidence_snippets"] = []
            return sanitized

        sanitized["soft_preferences"] = self._dedupe_rules(kept_soft_preferences)
        sanitized["decision_rules"] = self._dedupe_rules(kept_decision_rules)
        return sanitized

    def _build_rule_memory_payload(
        self,
        *,
        level: str,
        source_text: str,
        metadata: Dict[str, Any],
        fallback_topic: str,
    ) -> Dict[str, Any]:
        lowered = source_text.lower()
        hard_preferences: List[str] = []
        soft_preferences: List[str] = []
        current_needs: List[str] = []
        time_constraints: List[str] = []
        forbidden_items: List[str] = []
        decision_rules: List[str] = []
        stable_user_preferences: List[str] = []
        priority_order: List[str] = []
        evidence_snippets: List[str] = []

        def add_evidence_from_markers(markers: List[str], limit: int = 2) -> None:
            extracted = self._extract_relevant_source_lines(
                source_text,
                markers=markers,
                limit=limit,
            )
            for line in extracted:
                if line not in evidence_snippets:
                    evidence_snippets.append(line)

        if any(token in lowered for token in ["strictly sugar-free", "sugar-free", "sugar free", "no sugar", "无糖", "控糖"]):
            hard_preferences.append("Default to sugar-free drink options for this user.")
            forbidden_items.append("Do not offer sugary drinks.")
            stable_user_preferences.append("Default to sugar-free drink options for this user.")
            decision_rules.append("Enforce the sugar-free constraint before optimizing for stimulation, convenience, or taste.")
            priority_order.append("Priority: sugar-free constraint before temporary drink-selection needs.")
            add_evidence_from_markers(["strictly sugar-free", "sugar-free", "sugar free", "no sugar", "无糖", "控糖"])

        if any(token in lowered for token in ["allergic", "allergy", "过敏"]):
            extracted = self._extract_relevant_source_lines(
                source_text,
                markers=["allergic", "allergy", "过敏", "anaphyl", "坚果"],
                limit=2,
            )
            for line in extracted:
                normalized = self._normalize_rule_sentence(f"Honor the allergy or safety restriction: {line}")
                hard_preferences.append(normalized)
                stable_user_preferences.append(normalized)
            decision_rules.append("Treat allergy and safety restrictions as hard constraints.")
            priority_order.append("Priority: allergy and safety restrictions before all convenience preferences.")
            add_evidence_from_markers(["allergic", "allergy", "过敏", "anaphyl", "坚果"])

        if any(token in lowered for token in ["vegan", "vegetarian", "纯素", "素食"]):
            hard_preferences.append("Respect the user's dietary restriction when choosing food or drinks.")
            stable_user_preferences.append("Respect the user's dietary restriction when choosing food or drinks.")
            priority_order.append("Priority: dietary restriction before temporary food preference.")
            add_evidence_from_markers(["vegan", "vegetarian", "纯素", "素食"])

        if any(token in lowered for token in ["avoid caffeine", "no caffeine", "不想摄入咖啡因", "避免咖啡因", "不要咖啡因"]):
            hard_preferences.append("Avoid caffeinated drinks when choosing beverages for this user.")
            stable_user_preferences.append("Avoid caffeinated drinks when choosing beverages for this user.")
            decision_rules.append("Treat caffeine avoidance as a durable drink-selection constraint.")
            priority_order.append("Priority: caffeine avoidance before stimulation needs.")
            add_evidence_from_markers(["avoid caffeine", "no caffeine", "不想摄入咖啡因", "避免咖啡因", "不要咖啡因"])

        if any(token in lowered for token in ["afraid of heights", "恐高", "high-floor", "高层", "stairs", "楼梯", "noise", "吵", "loud", "噪音", "安全优先", "safety first"]):
            extracted = self._extract_relevant_source_lines(
                source_text,
                markers=["afraid of heights", "恐高", "high-floor", "高层", "stairs", "楼梯", "noise", "吵", "loud", "噪音", "安全优先", "safety first"],
                limit=3,
            )
            for line in extracted:
                normalized = self._normalize_rule_sentence(line)
                if normalized and normalized not in hard_preferences and normalized not in soft_preferences:
                    hard_preferences.append(normalized)
                if normalized and normalized not in stable_user_preferences and self._looks_like_stable_preference(line):
                    stable_user_preferences.append(normalized)
            if extracted:
                priority_order.append("Priority: durable safety and avoidance constraints before convenience.")
                evidence_snippets.extend([line for line in extracted if line not in evidence_snippets])

        if any(token in lowered for token in ["不要", "不喜欢", "prefer", "preference", "偏好", "default"]):
            extracted = self._extract_relevant_source_lines(
                source_text,
                markers=["不要", "不喜欢", "prefer", "preference", "偏好", "default", "只考虑", "优先"],
                limit=3,
            )
            for line in extracted:
                normalized = self._normalize_rule_sentence(line)
                if self._looks_like_stable_preference(line) and normalized not in stable_user_preferences:
                    stable_user_preferences.append(normalized)
                if normalized not in soft_preferences and normalized not in hard_preferences:
                    soft_preferences.append(normalized)
            evidence_snippets.extend([line for line in extracted if line not in evidence_snippets])

        if any(token in lowered for token in ["sleepy", "stay awake", "tired", "熬夜", "很困", "提神", "困"]):
            current_needs.append("The current task favors a drink that helps the user stay awake.")
            if level == "short_term":
                decision_rules.append(
                    "Treat stimulation as a temporary need only within the boundary of durable user preferences and hard constraints."
                )
                decision_rules.append(
                    "Never override durable user preferences, dietary restrictions, or safety constraints with a short-term convenience need."
                )
            else:
                decision_rules.append(
                    "After satisfying hard constraints, prefer an allowed drink that better matches the current need for stimulation."
                )
            add_evidence_from_markers(["sleepy", "stay awake", "tired", "熬夜", "很困", "提神", "困"])

        if any(token in lowered for token in ["今晚", "下午", "明天", "before", "by ", "九点前", "早上", "今晚前", "right now", "马上", "尽快", "等会", "稍后"]):
            extracted = self._extract_relevant_source_lines(
                source_text,
                markers=["今晚", "下午", "明天", "before", "by ", "九点前", "早上", "今晚前", "right now", "马上", "尽快", "等会", "稍后"],
                limit=3,
            )
            for line in extracted:
                normalized = self._normalize_rule_sentence(line)
                if normalized and normalized not in time_constraints:
                    time_constraints.append(normalized)
                if normalized and normalized not in evidence_snippets:
                    evidence_snippets.append(normalized)

        if any(token in lowered for token in ["water", "白水", "矿泉水"]):
            soft_preferences.append("If no stimulating drink is allowed, water remains a safe fallback option.")

        hard_preferences = self._dedupe_rules(hard_preferences)
        soft_preferences = self._dedupe_rules(soft_preferences)
        current_needs = self._dedupe_rules(current_needs)
        time_constraints = self._dedupe_rules(time_constraints)
        forbidden_items = self._dedupe_rules(forbidden_items)
        stable_user_preferences = self._dedupe_rules(stable_user_preferences)
        priority_order = self._dedupe_rules(priority_order)
        evidence_snippets = self._dedupe_rules(evidence_snippets)

        if level == "short_term":
            decision_rules.insert(
                0,
                "Apply temporary needs only after preserving durable user preferences and hard constraints."
            )
            if not priority_order:
                priority_order.append("Priority: hard constraints > current need > soft preferences.")
        elif not priority_order:
            priority_order.append("Priority: stable hard constraints before temporary convenience or soft preference.")

        if not decision_rules:
            decision_rules.append("Preserve durable user preferences and hard constraints ahead of transient convenience.")
        decision_rules = self._dedupe_rules(decision_rules)

        topic = fallback_topic
        if stable_user_preferences:
            topic = self._topic_from_preference(stable_user_preferences[0], fallback_topic)
        elif current_needs:
            topic = self._topic_from_preference(current_needs[0], fallback_topic)

        payload = {
            "topic": topic,
            "hard_preferences": hard_preferences,
            "soft_preferences": soft_preferences,
            "current_needs": current_needs,
            "time_constraints": time_constraints,
            "forbidden_items": forbidden_items,
            "decision_rules": decision_rules,
            "stable_user_preferences": stable_user_preferences,
            "priority_order": priority_order,
            "evidence_snippets": evidence_snippets,
        }
        payload["summary"] = self._render_rule_memory_summary(payload)
        return payload

    def _render_rule_memory_summary(self, payload: Dict[str, Any]) -> str:
        sections: List[str] = []
        section_map = [
            ("Stable user preferences", payload.get("stable_user_preferences", [])),
            ("Hard constraints", payload.get("hard_preferences", [])),
            ("Time constraints", payload.get("time_constraints", [])),
            ("Current needs", payload.get("current_needs", [])),
            ("Forbidden items", payload.get("forbidden_items", [])),
            ("Priority order", payload.get("priority_order", [])),
            ("Decision rules", payload.get("decision_rules", [])),
            ("Soft preferences", payload.get("soft_preferences", [])),
            ("Evidence", payload.get("evidence_snippets", [])),
        ]
        for title, items in section_map:
            clean_items = [str(item).strip() for item in items if str(item).strip()]
            if not clean_items:
                continue
            section_lines = [f"{title}:"]
            section_lines.extend([f"- {item}" for item in clean_items])
            sections.append("\n".join(section_lines))
        if not sections:
            lines = [line.strip() for line in (payload.get("source_text") or "").splitlines() if line.strip()]
            if not lines:
                return "No durable decision rules extracted."
            return "\n".join(lines[:8])[:1200]
        return "\n\n".join(sections)

    def _extract_relevant_source_lines(
        self,
        source_text: str,
        *,
        markers: List[str],
        limit: int,
    ) -> List[str]:
        results: List[str] = []
        for line in source_text.splitlines():
            stripped = line.strip(" -")
            lowered = stripped.lower()
            if not stripped:
                continue
            if not any(marker.lower() in lowered for marker in markers):
                continue
            if stripped.startswith(("Task request:", "Interaction status:", "Metadata:", "Summary:")):
                continue
            normalized = self._normalize_rule_sentence(stripped)
            if normalized and normalized not in results:
                results.append(normalized)
            if len(results) >= limit:
                break
        return results

    def _normalize_rule_sentence(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = cleaned.strip("- ")
        cleaned = re.sub(r"^(USER|ASSISTANT|SYSTEM):\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.rstrip("。.!！?？")
        return cleaned[:220]

    def _looks_like_stable_preference(self, text: str) -> bool:
        lowered = str(text or "").lower()
        stable_markers = [
            "以后",
            "默认",
            "always",
            "default",
            "平时",
            "strictly",
            "长期",
            "只考虑",
            "不要拿",
            "do not offer",
        ]
        return any(marker in lowered for marker in stable_markers)

    def _dedupe_rules(self, items: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for item in items:
            cleaned = self._normalize_rule_sentence(item)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _topic_from_preference(self, text: str, fallback_topic: str) -> str:
        lowered = text.lower()
        if "sugar-free" in lowered or "无糖" in text or "控糖" in text:
            return "sugar_free_preference"
        if "allergy" in lowered or "过敏" in text:
            return "allergy_constraint"
        if "dietary restriction" in lowered or "vegan" in lowered or "素食" in text:
            return "dietary_preference"
        if "stay awake" in lowered or "提神" in text:
            return "stimulation_need"
        return fallback_topic

    def _should_skip_summary_model(self) -> bool:
        return not self.api_key or "test_api_key" in self.api_key

    def _detect_topic_from_rules(
        self,
        *,
        level: str,
        source_text: str,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        if level == "long_term_overview":
            return f"{self.profile_name}_overview"

        lowered = source_text.lower()
        if any(token in lowered for token in ["strictly sugar-free", "sugar-free", "sugar free", "no sugar", "无糖", "控糖"]):
            return "sugar_free_preference"
        if any(token in lowered for token in ["allergic", "allergy", "过敏"]):
            return "allergy_constraint"

        status = str(metadata.get("status") or "")
        if status.startswith("waiting_") or "Pending question:" in source_text:
            return "clarification"
        if metadata.get("is_complete") or "Result: Task marked complete." in source_text:
            latest_action = metadata.get("latest_action") or self._extract_latest_action_from_source(source_text)
            category = self._classify_action_topic(latest_action)
            if category:
                return f"{category}_completion"
            return "task_completion"

        latest_action = metadata.get("latest_action") or self._extract_latest_action_from_source(source_text)
        category = self._classify_action_topic(latest_action)
        if category:
            return category

        fallback = metadata.get("fallback_topic") or self._fallback_topic_from_request()
        return fallback

    def _extract_latest_action_from_source(self, source_text: str) -> Optional[str]:
        matches = re.findall(r"Step\s+\d+:\s+([a-zA-Z0-9_]+)", source_text)
        if not matches:
            return None
        return matches[-1]

    def _classify_action_topic(self, action_name: Optional[str]) -> Optional[str]:
        if not action_name:
            return None
        action = action_name.strip().lower()
        action_groups = {
            "navigation": {"navigate_to"},
            "manipulation": {"pick", "place", "pick_from_shelf", "place_on_counter", "pick_and_place"},
            "home_devices": {"control_air_conditioner", "control_light", "set_home_mode"},
            "weather": {"query_weather"},
            "web_research": {"web_search", "choose_web_option"},
            "coordination": {"speak", "send_agent_message"},
            "memory_update": {"store_memory"},
            "observation": {"get_observation", "observe_scene", "observe_workspace"},
        }
        for topic, actions in action_groups.items():
            if action in actions:
                return topic
        return None

    def _fallback_topic_from_request(self) -> str:
        request = (self.original_request or "").strip().lower()
        if not request:
            return f"{self.profile_name}_task"
        compact = re.sub(r"[^a-z0-9]+", "_", request)
        compact = re.sub(r"_+", "_", compact).strip("_")
        return compact[:48] or f"{self.profile_name}_task"

    def _reset_runtime_state(self):
        raise NotImplementedError

    def _print_task_started(self, request: str):
        raise NotImplementedError

    def _prepare_current_observation(self) -> Any:
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
