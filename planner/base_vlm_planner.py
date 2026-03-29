"""
Shared runtime for modular VLM planners.
"""

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
from utils.action_registry import get_action_schema


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
        return self.hierarchical_memory_manager.build_context_block(
            session_id=self.runtime_memory_session_id,
        )

    def _maybe_compact_active_context(self) -> None:
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
        if result.get("is_complete"):
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
            if not next_step.get("action_type"):
                next_step["action_type"] = self._infer_action_type(next_step.get("action"))
            next_step.setdefault("parameters", {})
            payload.setdefault("needs_human_input", False)
            return payload

        legacy_action = payload.get("action")
        if isinstance(legacy_action, dict):
            skill = legacy_action.get("skill") or legacy_action.get("action")
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
                    "action_type": self._infer_action_type(skill),
                    "parameters": parameters if isinstance(parameters, dict) else {},
                }
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
            parameters = (
                payload.get("parameters")
                or payload.get("args")
                or payload.get("action_args")
                or payload.get("action_arguments")
                or {}
            )
            payload["next_step"] = {
                "step_number": next_step_number,
                "action": action_name,
                "action_type": self._infer_action_type(action_name),
                "parameters": parameters if isinstance(parameters, dict) else {},
            }
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
            parameters = payload.get("parameters") or {}
            if action_name:
                payload["next_step"] = {
                    "step_number": next_step_number,
                    "action": action_name,
                    "action_type": self._infer_action_type(action_name),
                    "parameters": parameters if isinstance(parameters, dict) else {},
                }
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
            "topic, hard_preferences, soft_preferences, current_needs, forbidden_items, decision_rules, stable_user_preferences. "
            "The topic must be a short lowercase slug with underscores. "
            "Each list item must be concise, factual, and action-guiding."
        )
        user_prompt = "\n".join(
            [
                f"Level: {level}",
                f"Device: {self.profile_name}",
                f"Trigger: {metadata.get('trigger_reason', 'unknown')}",
                f"Suggested topic: {rule_topic or fallback_topic}",
                "",
                "Extract only the information that should affect future decisions.",
                "Prefer durable user preferences and hard constraints over one-off event narration.",
                "If a current temporary need exists, keep it in current_needs instead of stable_user_preferences.",
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
            "forbidden_items": self._coerce_list_field(data.get("forbidden_items")),
            "decision_rules": self._coerce_list_field(data.get("decision_rules")),
            "stable_user_preferences": self._coerce_list_field(data.get("stable_user_preferences")),
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
        heuristic_payload: Dict[str, Any],
        parsed_payload: Dict[str, Any],
        fallback_topic: str,
    ) -> Dict[str, Any]:
        merged = {
            "topic": parsed_payload.get("topic") or heuristic_payload["topic"] or fallback_topic,
            "hard_preferences": parsed_payload.get("hard_preferences") or heuristic_payload["hard_preferences"],
            "soft_preferences": parsed_payload.get("soft_preferences") or heuristic_payload["soft_preferences"],
            "current_needs": parsed_payload.get("current_needs") or heuristic_payload["current_needs"],
            "forbidden_items": parsed_payload.get("forbidden_items") or heuristic_payload["forbidden_items"],
            "decision_rules": parsed_payload.get("decision_rules") or heuristic_payload["decision_rules"],
            "stable_user_preferences": parsed_payload.get("stable_user_preferences") or heuristic_payload["stable_user_preferences"],
        }
        merged["summary"] = self._render_rule_memory_summary(merged)
        return merged

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
        forbidden_items: List[str] = []
        decision_rules: List[str] = []
        stable_user_preferences: List[str] = []

        if any(token in lowered for token in ["strictly sugar-free", "sugar-free", "sugar free", "no sugar", "无糖", "控糖"]):
            hard_preferences.append("Default to sugar-free drink options for this user.")
            forbidden_items.append("Do not offer sugary drinks.")
            stable_user_preferences.append("Default to sugar-free drink options for this user.")
            decision_rules.append("Enforce the sugar-free constraint before optimizing for stimulation, convenience, or taste.")

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

        if any(token in lowered for token in ["vegan", "vegetarian", "纯素", "素食"]):
            hard_preferences.append("Respect the user's dietary restriction when choosing food or drinks.")
            stable_user_preferences.append("Respect the user's dietary restriction when choosing food or drinks.")

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

        if any(token in lowered for token in ["sleepy", "stay awake", "tired", "caffeine", "熬夜", "很困", "提神", "困"]):
            current_needs.append("The current task favors a drink that helps the user stay awake.")
            decision_rules.append("After satisfying hard constraints, prefer an allowed drink that better matches the current need for stimulation.")

        if any(token in lowered for token in ["water", "白水", "矿泉水"]):
            soft_preferences.append("If no stimulating drink is allowed, water remains a safe fallback option.")

        hard_preferences = self._dedupe_rules(hard_preferences)
        soft_preferences = self._dedupe_rules(soft_preferences)
        current_needs = self._dedupe_rules(current_needs)
        forbidden_items = self._dedupe_rules(forbidden_items)
        stable_user_preferences = self._dedupe_rules(stable_user_preferences)

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
            "forbidden_items": forbidden_items,
            "decision_rules": decision_rules,
            "stable_user_preferences": stable_user_preferences,
        }
        payload["summary"] = self._render_rule_memory_summary(payload)
        return payload

    def _render_rule_memory_summary(self, payload: Dict[str, Any]) -> str:
        sections: List[str] = []
        section_map = [
            ("Stable user preferences", payload.get("stable_user_preferences", [])),
            ("Hard constraints", payload.get("hard_preferences", [])),
            ("Current needs", payload.get("current_needs", [])),
            ("Forbidden items", payload.get("forbidden_items", [])),
            ("Decision rules", payload.get("decision_rules", [])),
            ("Soft preferences", payload.get("soft_preferences", [])),
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
