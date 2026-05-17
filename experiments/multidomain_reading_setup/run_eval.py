"""
Run the multidomain reading setup benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experiments.multidomain_reading_setup.score import score_trace
from experiments.preference_write_then_recall_ablation import initialize_clean_local_memory, local_memory_root
from executor.arm_executor import ExecutionResult
from executor.humanoid_executor import HumanoidExecutor
from planner.base_vlm_planner import BaseVLMPlanner
from utils.memory_manager import MemoryManager


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases.jsonl"
DEFAULT_OUTPUT = HERE / "outputs"
DEFAULT_GOLDEN = HERE / "golden_traces"
MEMORY_MODES = ("full", "none", "fixed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multidomain reading setup evaluation")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to cases.jsonl")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for results")
    parser.add_argument(
        "--memory-modes",
        default="full,none,fixed",
        help="Comma-separated modes: full,none,fixed",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["mock", "hardware"],
        default="mock",
        help="mock is deterministic; hardware is reserved for real demo integration",
    )
    parser.add_argument(
        "--trace-mode",
        choices=["scripted", "planner", "replay"],
        default="scripted",
        help="scripted emits fixed traces; planner runs BaseVLMPlanner loops; replay loads golden traces",
    )
    parser.add_argument(
        "--golden-dir",
        default=str(DEFAULT_GOLDEN),
        help="Directory for recording or replaying golden traces",
    )
    parser.add_argument(
        "--record-golden",
        action="store_true",
        help="Write generated traces to --golden-dir for later --trace-mode replay",
    )
    parser.add_argument(
        "--failure-suite",
        action="store_true",
        help="Also run synthetic failure variants to validate scorer discrimination",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    modes = parse_memory_modes(args.memory_modes)
    output_dir = Path(args.output_dir)
    golden_dir = Path(args.golden_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.record_golden:
        golden_dir.mkdir(parents=True, exist_ok=True)

    if args.execution_mode == "hardware":
        raise SystemExit(
            "hardware mode requires live planner/transport wiring. Use --execution-mode mock for reproducible scoring."
        )

    records = []
    for mode in modes:
        with memory_mode(mode):
            for case in cases:
                seed_case_memory(case)
                if args.trace_mode == "replay":
                    trace = load_golden_trace(golden_dir, case, mode)
                elif args.trace_mode == "planner":
                    trace = run_mock_planner_case(case, mode)
                else:
                    trace = run_mock_case(case, mode)
                if args.record_golden:
                    save_golden_trace(golden_dir, trace)
                records.append(build_record(case, trace))
                if args.failure_suite:
                    for failure_name in FAILURE_VARIANTS:
                        failed_trace = apply_failure_variant(trace, failure_name)
                        records.append(build_record(case, failed_trace))

    write_results(output_dir, records)
    print_summary(records)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                cases.append(json.loads(stripped))
    return cases


def parse_memory_modes(raw: str) -> List[str]:
    modes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in MEMORY_MODES]
    if invalid:
        raise ValueError(f"Invalid memory modes: {invalid}")
    return modes


@contextmanager
def memory_mode(mode: str):
    previous_mode = os.getenv("OMNICLAW_MEMORY_MODE")
    previous_local_root = os.getenv("AGENT_LOCAL_MEMORY_ROOT")
    os.environ["OMNICLAW_MEMORY_MODE"] = mode
    os.environ["AGENT_LOCAL_MEMORY_ROOT"] = str(DEFAULT_OUTPUT / "local_memory" / mode)
    initialize_clean_local_memory()
    try:
        yield
    finally:
        if previous_mode is None:
            os.environ.pop("OMNICLAW_MEMORY_MODE", None)
        else:
            os.environ["OMNICLAW_MEMORY_MODE"] = previous_mode
        if previous_local_root is None:
            os.environ.pop("AGENT_LOCAL_MEMORY_ROOT", None)
        else:
            os.environ["AGENT_LOCAL_MEMORY_ROOT"] = previous_local_root


def seed_case_memory(case: Dict[str, Any]) -> None:
    manager = MemoryManager(profile_name="humanoid_g1", verbose=False)
    for entry in case.get("seed_memory") or []:
        manager.store_memory(content=entry, scope="global", category="preference")


def run_mock_case(case: Dict[str, Any], memory_mode_name: str) -> Dict[str, Any]:
    """Create a deterministic trace for scorer and paper-table validation."""
    expected = case.get("expected", {})
    clutter = expected.get("move_to_basket") or []
    utterance = case.get("utterance", "")

    g1_steps = [
        step(1, "send_agent_message", "talk", {
            "recipient": "ur5e",
            "message": f"请清理阅读桌面，只移动不相关杂物到 basket，保留书和笔记本。用户请求：{utterance}",
        }),
        step(2, "control_light", "tool", {
            "action": "set_brightness",
            "device": "background_light",
            "brightness": min(30, int(expected.get("background_brightness_max", 35))),
        }),
        step(3, "control_light", "tool", {
            "action": "turn_on",
            "device": "desk_lamp",
            "brightness": 80,
        }),
        step(4, "play_audio", "tool", {
            "action": "play",
            "audio_type": expected.get("audio_type", "white_noise"),
            "track": "reading_white_noise",
            "volume": 35,
        }),
        step(5, "speak", "talk", {
            "message": "桌子已清理，阅读灯和背景音已为您打开。",
        }),
    ]
    ur5e_steps = [
        step(index + 1, "pick_and_place", "act", {
            "item_name": item,
            "source": "desk",
            "target": "basket",
        })
        for index, item in enumerate(clutter)
    ]

    return {
        "case_id": case.get("case_id"),
        "memory_mode": memory_mode_name,
        "trace_mode": "scripted",
        "transcript": utterance,
        "local_memory_root": str(local_memory_root()),
        "g1_execution_history": g1_steps,
        "ur5e_execution_history": ur5e_steps,
    }


def run_mock_planner_case(case: Dict[str, Any], memory_mode_name: str) -> Dict[str, Any]:
    """Run a deterministic planner-in-loop trace without external model calls."""
    utterance = case.get("utterance", "")
    expected = case.get("expected", {})
    clutter = expected.get("move_to_basket") or []

    ur5e_planner = ScriptedPlanner(
        profile_name="ur5e",
        executor=MockArmCleanupExecutor(),
        scripted_steps=[
            step(index + 1, "pick_and_place", "act", {
                "item_name": item,
                "source": "desk",
                "target": "basket",
            })
            for index, item in enumerate(clutter)
        ],
    )
    ur5e_planner.start_new_task(
        "请清理阅读桌面，只移动不相关杂物到 basket，保留书和笔记本。",
        run_autonomously=True,
    )

    g1_planner = ScriptedPlanner(
        profile_name="humanoid_g1",
        executor=HumanoidExecutor(simulation_mode=True, verbose=False),
        scripted_steps=[
            step(1, "send_agent_message", "talk", {
                "recipient": "ur5e",
                "message": f"请清理阅读桌面，只移动不相关杂物到 basket，保留书和笔记本。用户请求：{utterance}",
            }),
            step(2, "control_light", "tool", {
                "action": "set_brightness",
                "device": "background_light",
                "brightness": min(30, int(expected.get("background_brightness_max", 35))),
            }),
            step(3, "control_light", "tool", {
                "action": "turn_on",
                "device": "desk_lamp",
                "brightness": 80,
            }),
            step(4, "play_audio", "tool", {
                "action": "play",
                "audio_type": expected.get("audio_type", "white_noise"),
                "track": "reading_white_noise",
                "volume": 35,
            }),
            step(5, "speak", "talk", {
                "message": "桌子已清理，阅读灯和背景音已为您打开。",
            }),
        ],
    )
    g1_planner.start_new_task(utterance, run_autonomously=True)

    return {
        "case_id": case.get("case_id"),
        "memory_mode": memory_mode_name,
        "trace_mode": "planner",
        "transcript": utterance,
        "local_memory_root": str(local_memory_root()),
        "g1_execution_history": g1_planner.execution_history,
        "ur5e_execution_history": ur5e_planner.execution_history,
    }


def save_golden_trace(golden_dir: Path, trace: Dict[str, Any]) -> Path:
    path = golden_trace_path(golden_dir, trace.get("case_id"), trace.get("memory_mode"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_golden_trace(golden_dir: Path, case: Dict[str, Any], memory_mode_name: str) -> Dict[str, Any]:
    path = golden_trace_path(golden_dir, case.get("case_id"), memory_mode_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing golden trace for case={case.get('case_id')} memory_mode={memory_mode_name}: {path}"
        )
    trace = json.loads(path.read_text(encoding="utf-8"))
    trace["trace_mode"] = "replay"
    trace["golden_source"] = str(path)
    return trace


def golden_trace_path(golden_dir: Path, case_id: Any, memory_mode_name: Any) -> Path:
    return golden_dir / str(memory_mode_name or "unknown") / f"{case_id or 'unknown_case'}.json"


FAILURE_VARIANTS = (
    "missing_audio",
    "no_desk_lamp",
    "wrong_route_type",
    "moves_notebook",
    "no_tts",
)


def build_record(case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    score = score_trace(case, trace)
    return {
        **score,
        "trace": trace,
        "failure_variant": trace.get("failure_variant", "none"),
    }


def apply_failure_variant(trace: Dict[str, Any], failure_name: str) -> Dict[str, Any]:
    mutated = clone_jsonable(trace)
    mutated["failure_variant"] = failure_name
    if failure_name == "missing_audio":
        mutated["g1_execution_history"] = [
            step for step in mutated.get("g1_execution_history", [])
            if step.get("action") != "play_audio"
        ]
    elif failure_name == "no_desk_lamp":
        mutated["g1_execution_history"] = [
            step for step in mutated.get("g1_execution_history", [])
            if not (
                step.get("action") == "control_light"
                and (step.get("parameters") or {}).get("device") == "desk_lamp"
            )
        ]
    elif failure_name == "wrong_route_type":
        for step_item in mutated.get("g1_execution_history", []):
            if step_item.get("action") == "play_audio":
                step_item["action_type"] = "talk"
                break
    elif failure_name == "moves_notebook":
        ur5e_steps = mutated.setdefault("ur5e_execution_history", [])
        for protected_item in ("book", "notebook"):
            ur5e_steps.append(
                step(len(ur5e_steps) + 1, "pick_and_place", "act", {
                    "item_name": protected_item,
                    "source": "desk",
                    "target": "basket",
                })
            )
    elif failure_name == "no_tts":
        mutated["g1_execution_history"] = [
            step for step in mutated.get("g1_execution_history", [])
            if step.get("action") != "speak"
        ]
    else:
        raise ValueError(f"Unknown failure variant: {failure_name}")
    return mutated


def clone_jsonable(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def step(step_number: int, action: str, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step_number": step_number,
        "action": action,
        "action_type": action_type,
        "parameters": parameters,
        "assumed_successful": True,
        "execution_result": {
            "success": True,
            "feedback": "mock success",
            "data": {},
            "error": None,
        },
    }


class ScriptedPlanner(BaseVLMPlanner):
    """Small deterministic planner that exercises BaseVLMPlanner and executors."""

    def __init__(self, profile_name: str, executor: Any, scripted_steps: List[Dict[str, Any]]):
        self.executor = executor
        self.scripted_steps = scripted_steps
        super().__init__(
            profile_name=profile_name,
            api_key="test_api_key",
            model_name="scripted-vlm",
            verbose=False,
            config_getter=lambda _model: {
                "model": "scripted-vlm",
                "max_tokens": 512,
                "temperature": 0.0,
                "context_window_tokens": 32000,
            },
            system_prompt_getter=lambda: "",
            legacy_prompt_getter=lambda: "",
        )

    def _reset_runtime_state(self):
        self.current_location = "Room 01" if self.profile_name == "humanoid_g1" else "Room 02"

    def _print_task_started(self, request: str):
        return None

    def _prepare_current_observation(self):
        return "simulation_images/home/default.jpg"

    def _get_current_image_for_planning(self) -> str:
        return "simulation_images/home/default.jpg"

    def _execute_step(self, step_plan: Dict[str, Any]):
        next_step = step_plan.get("next_step")
        if not next_step:
            return
        result = self.executor.execute_action(
            next_step.get("action_type"),
            next_step.get("action"),
            next_step.get("parameters") or {},
        )
        self.execution_history.append({
            "step_number": next_step.get("step_number"),
            "action": next_step.get("action"),
            "action_type": next_step.get("action_type"),
            "parameters": next_step.get("parameters") or {},
            "assumed_successful": bool(result.success),
            "execution_result": result.to_dict(),
        })
        self.step_count += 1

    def _build_context_message(self) -> str:
        return f"Scripted context for: {self.original_request}"

    def _parse_step_plan(self, response_text: str) -> Dict[str, Any]:
        return {}

    def _display_step_plan(self, step_plan):
        return None

    def plan_next_step_with_image(self, observation_input: Any) -> Dict[str, Any]:
        if self.step_count >= len(self.scripted_steps):
            self.is_task_complete = True
            return {
                "current_step_analysis": {
                    "visual_state": "scripted final state",
                    "task_progress": "all scripted steps complete",
                    "next_action_reasoning": "task complete",
                },
                "next_step": None,
                "needs_human_input": False,
            }
        next_step = dict(self.scripted_steps[self.step_count])
        return {
            "current_step_analysis": {
                "visual_state": "scripted observation",
                "task_progress": f"{self.step_count} steps complete",
                "next_action_reasoning": "scripted benchmark action",
            },
            "next_step": next_step,
            "needs_human_input": False,
            self._get_question_field(): None,
        }

    def _is_waiting_for_input(self) -> bool:
        return False

    def _set_waiting_state(self, question: Optional[str]):
        return None

    def _clear_waiting_state(self):
        return None

    def _get_pending_question(self):
        return None

    def _get_waiting_status(self) -> str:
        return "waiting_for_human"

    def _get_question_field(self) -> str:
        return "human_question"

    def _get_input_role_name(self) -> str:
        return "human"

    def _get_response_prefix(self) -> str:
        return "HUMAN RESPONSE"


class MockArmCleanupExecutor:
    """Minimal UR5e cleanup executor for planner-in-loop benchmark mode."""

    def execute_action(self, action_type: str, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        if action_type == "act" and action_name == "pick_and_place":
            return ExecutionResult(
                success=True,
                feedback="mock pick_and_place completed",
                data={"observation_image": None},
                error=None,
            )
        return ExecutionResult(
            success=False,
            feedback="unsupported mock arm action",
            error=f"{action_type}.{action_name} is not supported",
        )


def write_results(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "memory_mode",
                "failure_variant",
                "cases",
                "task_success_rate",
                "voice",
                "embodied",
                "iot",
                "cyber_api",
                "tts",
                "route_correctness",
            ],
        )
        writer.writeheader()
        for row in summarize(records):
            writer.writerow(row)


def summarize(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_mode: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (record["memory_mode"], record.get("failure_variant", "none"))
        by_mode.setdefault(key, []).append(record)

    rows = []
    for (mode, failure_variant), mode_records in sorted(by_mode.items()):
        count = len(mode_records)
        submetric_names = ["voice", "embodied", "iot", "cyber_api", "tts", "route_correctness"]
        row = {
            "memory_mode": mode,
            "failure_variant": failure_variant,
            "cases": count,
            "task_success_rate": _ratio(record["success"] for record in mode_records),
        }
        for name in submetric_names:
            row[name] = _ratio(record["submetrics"][name] for record in mode_records)
        rows.append(row)
    return rows


def _ratio(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(1 for item in items if item) / len(items), 4)


def print_summary(records: List[Dict[str, Any]]) -> None:
    for row in summarize(records):
        print(
            "{memory_mode}/{failure_variant}: TSR={task_success_rate:.4f} cases={cases}".format(**row)
        )


if __name__ == "__main__":
    main()
