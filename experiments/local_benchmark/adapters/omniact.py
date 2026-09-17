"""OmniAct adapter using the repository's native BaseVLMPlanner loop."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from planner.base_vlm_planner import BaseVLMPlanner

from ..model import MODEL_NAME, ModelGateway, aggregate_jsonl_usage, configure_roboagent_model_environment
from ..memory.rag import TurnChunkRAG
from ..simulator.world import PHYSICAL_SKILLS, StatefulWorld


METHODS = {
    "iterative_posthoc": {"memory": "none", "monitor": "none", "strategy": "raw"},
    "iterative_memory": {"memory": "full", "monitor": "none", "strategy": "hierarchical"},
    "iterative_monitor": {"memory": "none", "monitor": "online_stop", "strategy": "raw"},
    "omniact": {"memory": "full", "monitor": "online_stop", "strategy": "hierarchical"},
    "memory_raw_sliding": {"memory": "none", "monitor": "online_stop", "strategy": "raw"},
    "memory_same_model_summary": {"memory": "none", "monitor": "online_stop", "strategy": "summary"},
    "memory_rag": {"memory": "none", "monitor": "online_stop", "strategy": "rag"},
    "omniact_no_episodic": {"memory": "full", "monitor": "online_stop", "strategy": "hierarchical", "episodic": False, "reflective": True},
    "omniact_no_reflective": {"memory": "full", "monitor": "online_stop", "strategy": "hierarchical", "episodic": True, "reflective": False},
    "timing_posthoc": {"memory": "full", "monitor": "posthoc", "strategy": "hierarchical"},
    "timing_online_stop": {"memory": "full", "monitor": "online_stop", "strategy": "hierarchical"},
    "timing_online_delayed": {"memory": "full", "monitor": "online_delayed", "strategy": "hierarchical"},
    "vision_endpoint": {"memory": "full", "monitor": "online_stop", "strategy": "hierarchical", "temporal": False},
    "vision_temporal": {"memory": "full", "monitor": "online_stop", "strategy": "hierarchical", "temporal": True},
}


class OmniActLocalPlanner(BaseVLMPlanner):
    """Thin environment binding around the repository-native planner."""

    def __init__(
        self,
        *,
        world: StatefulWorld,
        method_id: str,
        run_dir: Path,
        temporal_frames: bool = True,
        max_steps: int = 12,
        verbose: bool = False,
    ):
        if method_id not in METHODS:
            raise ValueError(f"unsupported OmniAct method: {method_id}")
        configure_roboagent_model_environment()
        self.world = world
        self.method_id = method_id
        self.method = METHODS[method_id]
        self.run_dir = Path(run_dir)
        self.temporal_frames = bool(self.method.get("temporal", temporal_frames))
        self.max_steps = int(max_steps)
        self.gateway = ModelGateway(self.run_dir / "model_calls.jsonl")
        self.rag = TurnChunkRAG(self.run_dir / "rag_events.jsonl", token_budget=600)
        self.summary_memories: List[str] = []
        self._recent_frames: List[Dict[str, Any]] = []
        self.pending_question = None
        self.call_audit_path = self.run_dir / "planner_calls.jsonl"
        os.environ["OMNICLAW_MEMORY_MODE"] = self.method["memory"]
        memory_root = self.run_dir / "memory"
        os.environ["AGENT_LOCAL_MEMORY_ROOT"] = str(memory_root)
        os.environ["AGENT_MEMORY_INBOX_ROOT"] = str(memory_root / "inbox")
        os.environ["MEMORY_CONTEXT_BUDGET_TOKENS"] = "600"
        os.environ["MEMORY_RAW_TAIL_MAX_TOKENS"] = "300"
        super().__init__(
            profile_name="ur5e",
            api_key=os.environ["PLANNER_VLM_API_KEY"],
            model_name=MODEL_NAME,
            verbose=verbose,
            config_getter=lambda model: {"model": model, "max_tokens": 1800, "temperature": 0.0, "context_window_tokens": 131072},
            system_prompt_getter=lambda: "",
            legacy_prompt_getter=lambda: "",
        )
        self._instrument_native_client()

    def _instrument_native_client(self) -> None:
        original = self.vlm_client.create_chat_completion

        def audited(**kwargs):
            response = None
            started = time.monotonic()
            for attempt in range(1, 4):
                call_started = time.monotonic()
                try:
                    response = original(**kwargs)
                    break
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    retryable = any(marker in message.lower() for marker in (
                        "timed out", "timeout", "connection error", "unexpected_eof", "temporarily unavailable",
                        "rate limit", "429", "502", "503", "504",
                    ))
                    with self.call_audit_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "role": "planner", "model": MODEL_NAME, "attempt": attempt,
                            "api_call_attempt": True, "error": message, "retryable": retryable,
                            "wall_seconds": time.monotonic() - call_started,
                        }, ensure_ascii=False) + "\n")
                    if not retryable or attempt == 3:
                        raise
                    time.sleep(float(5 * attempt))
            assert response is not None
            usage = getattr(response, "usage", None)
            usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
            message = response.choices[0].message.content
            image_records = []
            for msg in kwargs.get("messages", []):
                for item in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                    path = item.get("_internal_path") if isinstance(item, dict) else None
                    if path and Path(path).exists():
                        image_records.append({"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
            with self.call_audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "role": "planner", "model": MODEL_NAME, "images": image_records, "response": message,
                    "usage": usage_dict, "api_call_attempt": True, "wall_seconds": time.monotonic() - started,
                }, ensure_ascii=False) + "\n")
            return response

        self.vlm_client.create_chat_completion = audited

    def _reset_runtime_state(self):
        self.pending_question = None

    def _print_task_started(self, request: str):
        return None

    def _build_runtime_system_prompt(self) -> str:
        memory = "Hierarchical event memory is enabled." if self.method["memory"] == "full" else "No hierarchical memory is available; ordinary working history remains."
        monitor = (
            "This arm is assigned the monitor-present aggregate physical policy (85%)."
            if self.world.monitor_present
            else "This arm is assigned the no-monitor aggregate physical policy (60%)."
        )
        return (
            "You are the OmniAct embodied planner running in a synthetic, stateful simulated physical environment. "
            "Plan exactly one tool-like step at a time and recover from observable failures. Never claim success without observing it. "
            "Each normalized physical goal is resolved atomically by one aggregate random draw. Monitoring/recovery effects "
            "are already included in that probability, so do not retry the same physical goal. Inspect the returned image/state. "
            f"{memory} {monitor} "
            "Return only JSON: {current_step_analysis:{visual_state:string,task_progress:string,next_action_reasoning:string}, "
            "next_step:null OR {action:string,parameters:object}, needs_human_input:false}. "
            "Keep each analysis string under 80 words and do not emit private chain-of-thought or prose outside the JSON object. "
            "Allowed actions: observe; execute_skill with exact schemas: pick arguments {item}; place {item,target}; "
            "navigate {destination}; pick_and_place {item,source,target}. Prefer pick_and_place for a direct move when allowed. "
            "set_device with {device,state}; remember with {text}; speak with {message}. "
            "Set next_step to null only after the requested state is visibly satisfied."
        )

    def _prepare_current_observation(self) -> Dict[str, Any]:
        obs = self.world.observe()
        self._recent_frames.append(obs)
        self._recent_frames = self._recent_frames[-4:]
        selected = self._recent_frames if self.temporal_frames else self._recent_frames[-1:]
        return {
            "primary_image": obs["rgb_paths"][0],
            "frames": [
                {"image_path": frame["rgb_paths"][0], "label": f"chronological frame {index + 1}, t={frame['public_state']['sim_time']}"}
                for index, frame in enumerate(selected)
            ],
            "summary_text": "Public state: " + json.dumps(obs["public_state"], ensure_ascii=False, sort_keys=True),
        }

    def _get_current_image_for_planning(self) -> str:
        return self._prepare_current_observation()["primary_image"]

    def _build_context_message(self) -> str:
        memory_context = self._build_hierarchical_memory_context()
        lines = [
            f"Original request: {self.original_request}",
            f"Step {self.step_count + 1} of {self.max_steps}.",
            "Current public state: " + json.dumps(self.world.public_snapshot(), ensure_ascii=False, sort_keys=True),
        ]
        if self.execution_history:
            lines.append("Recent action results: " + json.dumps(self.execution_history[-4:], ensure_ascii=False, default=str))
        if memory_context:
            lines.append(memory_context)
        if self.method["strategy"] == "rag":
            recalled = self.rag.search(self.original_request or "")
            if recalled:
                lines.append(recalled)
        elif self.method["strategy"] == "summary" and self.summary_memories:
            lines.append("[SAME-MODEL STEP SUMMARIES]\n" + "\n".join(self.summary_memories[-4:]))
        return "\n".join(lines)

    def _trim_controlled_memory_history(self) -> None:
        """Apply the frozen raw-tail budget only to memory-organization arms."""
        if self.method_id == "memory_raw_sliding":
            self.conversation_history = self.conversation_history[-8:]
        elif self.method["strategy"] in {"summary", "rag"}:
            self.conversation_history = self.conversation_history[-4:]

    def _build_hierarchical_memory_context(self) -> Optional[str]:
        if self.memory_mode != "full":
            return super()._build_hierarchical_memory_context()
        return self.hierarchical_memory_manager.build_context_block(
            session_id=self.runtime_memory_session_id,
            include_episodic=self.method.get("episodic", True),
            include_reflective=self.method.get("reflective", True),
        )

    def _parse_step_plan(self, response_text: str) -> Dict[str, Any]:
        text = (response_text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            return {"error": f"invalid planner JSON: {error}"}
        payload.setdefault("current_step_analysis", {})
        payload.setdefault("needs_human_input", False)
        step = payload.get("next_step")
        if step is not None:
            if not isinstance(step, dict):
                return {"error": "planner selected an unavailable action"}
            if step.get("action") in PHYSICAL_SKILLS:
                direct_skill = step["action"]
                direct_parameters = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
                step = {
                    "action": "execute_skill",
                    "parameters": {"skill": direct_skill, "arguments": direct_parameters},
                }
                payload["next_step"] = step
            if step.get("action") not in {"observe", "execute_skill", "set_device", "remember", "speak"}:
                return {"error": "planner selected an unavailable action"}
            step.setdefault("parameters", {})
            step.setdefault("step_number", self.step_count + 1)
            step.setdefault("action_type", "act" if step["action"] == "execute_skill" else "tool")
        return payload

    def _execute_step(self, step_plan: Dict[str, Any]):
        step = step_plan["next_step"]
        action = step["action"]
        params = step.get("parameters") or {}
        started = time.monotonic()
        frames: List[str] = []
        monitor_decisions: List[Dict[str, Any]] = []
        if action == "observe":
            observation = self.world.observe()
            frames.extend(observation["rgb_paths"])
            result = {"accepted": True, "diagnostic": "observation_captured", "observation": observation}
        elif action == "set_device":
            result = self.world.set_device(str(params.get("device") or ""), params.get("state"))
        elif action == "remember":
            result = self.world.store_preference(str(params.get("text") or ""))
        elif action == "speak":
            result = {"accepted": True, "diagnostic": "simulated_local_report", "message": str(params.get("message") or "")}
        else:
            skill = str(params.get("skill") or "")
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            if skill not in PHYSICAL_SKILLS:
                result = {"accepted": False, "diagnostic": "unknown_physical_skill"}
            else:
                result = self.world.start_skill(skill, arguments)
                observation = result.get("observation") or {}
                if observation.get("rgb_paths"):
                    frames.append(observation["rgb_paths"][0])
                    self._recent_frames.append(observation)
        self.step_count += 1
        record = {
            "step_number": self.step_count,
            "action": action,
            "parameters": params,
            "execution_result": result,
            "wall_seconds": time.monotonic() - started,
            "sim_time": self.world.sim_time,
            "monitor_decisions": monitor_decisions,
            "execution_control_policy": "direct_aggregate_bernoulli",
            "monitor_effect_integrated": self.world.monitor_present,
            "effective_physical_success_probability": self.world.physical_success_prob,
            "explicit_physical_retries_simulated": False,
        }
        self.execution_history.append(record)
        serialized_record = json.dumps(record, ensure_ascii=False, default=str)
        if self.method["strategy"] == "rag":
            self.rag.add(serialized_record)
        elif self.method["strategy"] == "summary":
            summary = self.gateway.complete_json(
                role="memory_summary",
                system="Summarize one embodied-agent step as compact factual JSON. Never infer hidden success.",
                prompt=(
                    "Return JSON with summary (max 80 words), observed_outcome, and next_recovery_hint. "
                    "Use only this public step record:\n" + serialized_record
                ),
                max_tokens=500,
            )
            self.summary_memories.append(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        self.conversation_history.append({"role": "user", "content": "[EXECUTION RESULT]\n" + json.dumps(result, ensure_ascii=False, default=str)})
        self._trim_controlled_memory_history()
        if frames and self.temporal_frames:
            self.transient_memory_packet = {
                "primary_image": frames[-1],
                "frames": [{"image_path": path, "label": f"execution frame {i + 1}"} for i, path in enumerate(frames[-4:])],
                "summary_text": f"Ordered execution frames for {action}",
            }
        if self.step_count >= self.max_steps:
            self.is_task_complete = True

    def _display_step_plan(self, step_plan: Dict[str, Any]):
        return None

    def _is_waiting_for_input(self) -> bool:
        return False

    def _set_waiting_state(self, question: Optional[str]):
        self.pending_question = question

    def _clear_waiting_state(self):
        self.pending_question = None

    def _get_pending_question(self) -> Optional[str]:
        return self.pending_question

    def _get_waiting_status(self) -> str:
        return "waiting_human"

    def _get_question_field(self) -> str:
        return "question"

    def _get_input_role_name(self) -> str:
        return "human"

    def _get_response_prefix(self) -> str:
        return "human"


def run_omniact_episode(world: StatefulWorld, method_id: str, run_dir: Path, *, temporal_frames: bool = True, max_steps: int = 12, verbose: bool = False) -> Dict[str, Any]:
    planner = OmniActLocalPlanner(world=world, method_id=method_id, run_dir=run_dir, temporal_frames=temporal_frames, max_steps=max_steps, verbose=verbose)
    started = time.monotonic()
    result = planner.start_new_task(world.task["request"], run_autonomously=True)
    score = world.score()
    planner_usage = aggregate_jsonl_usage(planner.call_audit_path)
    auxiliary_usage = aggregate_jsonl_usage(planner.gateway.log_path)
    combined_usage = {key: planner_usage[key] + auxiliary_usage[key] for key in planner_usage}
    return {
        "method_id": method_id,
        "framework": "RoboAgent BaseVLMPlanner",
        "model": MODEL_NAME,
        "simulated_physical_execution": True,
        "task_id": world.task["task_id"],
        "seed": world.seed,
        "agent_result": result,
        "agent_error": result.get("error") if isinstance(result, dict) else None,
        "score": score,
        "planner_steps": planner.step_count,
        "model_calls": sum(1 for _ in planner.call_audit_path.open(encoding="utf-8")) if planner.call_audit_path.exists() else 0,
        "monitor_calls": planner.gateway.role_counts.get("online_monitor", 0),
        "memory_model_calls": planner.gateway.role_counts.get("memory_summary", 0),
        "planner_usage": planner_usage,
        "auxiliary_usage": auxiliary_usage,
        "usage": combined_usage,
        "monitor_timing_policy": planner.method["monitor"],
        "execution_control_policy": "direct_aggregate_bernoulli",
        "monitor_present": world.monitor_present,
        "monitor_effect_integrated": world.monitor_present,
        "effective_physical_success_probability": world.physical_success_prob,
        "explicit_physical_retries_simulated": False,
        "seed_controls_physics": False,
        "temporal_frames": planner.temporal_frames,
        "memory_strategy": planner.method["strategy"],
        "memory_context_budget_tokens": 600,
        "wall_seconds": time.monotonic() - started,
    }
