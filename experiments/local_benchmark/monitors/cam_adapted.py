"""Kimi-generated restricted symbolic monitor, explicitly not full CaM."""

from __future__ import annotations

import ast
import json
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..model import ModelGateway
from ..simulator.render import render_observation


REPRODUCTION_LEVEL = "cam_adapted_symbolic (public-state oracle perception; not official/full Code-as-Monitor)"

ALLOWED_NODES = {
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load,
    ast.Constant, ast.Subscript, ast.List, ast.Tuple, ast.Dict, ast.Slice,
    ast.And, ast.Or, ast.Not, ast.Add, ast.Eq, ast.NotEq, ast.In, ast.NotIn,
    ast.Is, ast.IsNot, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
}
ALLOWED_NAMES = {"current", "previous", "action", "False", "True", "None"}


def _compile(expression: str):
    if len(expression) > 2000:
        raise ValueError("monitor expression is too long")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise ValueError(f"disallowed monitor syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise ValueError(f"disallowed monitor name: {node.id}")
    return compile(tree, "<cam_adapted>", "eval")


def _normal_progress_probes(task: Dict[str, Any], initial: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Build public-only states that a conservative monitor must not stop.

    These probes are protocol states, not evaluator answers: an object moving to
    ``near_*`` and a robot moving through ``corridor_to_*`` are documented normal
    phases of every method's shared executor.
    """
    probes: list[Dict[str, Any]] = []
    base = copy.deepcopy(initial)
    probes.append({"name": "initial_idle", "previous": base, "current": base, "action": {}})
    success = task.get("success", {})
    for item, origin in initial.get("objects", {}).items():
        near = copy.deepcopy(base)
        near["objects"][item] = f"near_{origin}"
        near["active_action"] = {"skill": "pick", "phase": 1, "timeout_phase": 3, "status": "vla_running"}
        probes.append({
            "name": f"normal_pick_approach:{item}", "previous": base, "current": near,
            "action": {"skill": "pick", "arguments": {"item": item}},
        })
        held = copy.deepcopy(base)
        held["objects"][item] = "held"
        held["held_object"] = item
        probes.append({
            "name": f"normal_pick_held:{item}", "previous": near, "current": held,
            "action": {"skill": "pick", "arguments": {"item": item}},
        })
        target = success.get("objects", {}).get(item)
        if target is not None:
            placed = copy.deepcopy(base)
            placed["objects"][item] = target
            probes.append({
                "name": f"normal_object_endpoint:{item}", "previous": held, "current": placed,
                "action": {"skill": "place", "arguments": {"item": item, "target": target}},
            })
    destination = success.get("robot_location")
    if destination is not None:
        corridor = copy.deepcopy(base)
        corridor["robot_location"] = f"corridor_to_{destination}"
        corridor["active_action"] = {"skill": "navigate", "phase": 2, "timeout_phase": 4, "status": "vla_running"}
        probes.append({
            "name": "normal_navigation_corridor", "previous": base, "current": corridor,
            "action": {"skill": "navigate", "arguments": {"destination": destination}},
        })
        arrived = copy.deepcopy(base)
        arrived["robot_location"] = destination
        probes.append({
            "name": "normal_navigation_endpoint", "previous": corridor, "current": arrived,
            "action": {"skill": "navigate", "arguments": {"destination": destination}},
        })
    return probes


def _validate_normal_progress(compiled: Any, probes: list[Dict[str, Any]]) -> list[str]:
    checked: list[str] = []
    for probe in probes:
        try:
            triggered = bool(eval(  # noqa: S307 - AST-whitelisted expression
                compiled,
                {"__builtins__": {}},
                {"current": probe["current"], "previous": probe["previous"], "action": probe["action"]},
            ))
        except Exception as exc:
            raise ValueError(f"monitor failed normal-progress probe {probe['name']}: {exc}") from exc
        if triggered:
            raise ValueError(f"monitor stops documented normal-progress probe {probe['name']}")
        checked.append(probe["name"])
    return checked


def prepare_cam_monitor(task: Dict[str, Any], episode_dir: Path) -> Path:
    """Ask the real model for one task-level monitor and validate it before use."""
    output = Path(episode_dir) / "cam_adapted"
    output.mkdir(parents=True, exist_ok=True)
    initial = task["initial_state"]
    public = {
        "sim_time": 0.0,
        "robot_location": initial.get("robot_location", "desk"),
        "held_object": initial.get("held_object"),
        "objects": dict(initial.get("objects", {})),
        "devices": dict(initial.get("devices", {})),
        "preferences": list(initial.get("preferences", [])),
        "active_action": None,
    }
    rendered = render_observation(public, output / "initial_frames", 1)
    gateway = ModelGateway(output / "generation_model_call.jsonl")
    probes = _normal_progress_probes(task, public)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "stop_expression": {"type": "string"},
            "declared_constraints": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["stop_expression", "declared_constraints"],
    }
    normal_examples = []
    for probe in probes:
        normal_examples.append({
            "name": probe["name"],
            "robot_location": probe["current"].get("robot_location"),
            "held_object": probe["current"].get("held_object"),
            "objects": probe["current"].get("objects"),
            "active_action": probe["current"].get("active_action"),
        })
    base_prompt = (
        "Task request: " + task["request"] + "\nInitial public state: " + json.dumps(public, ensure_ascii=False)
        + "\nDocumented normal current-state examples (the expression MUST be false for all): "
        + json.dumps(normal_examples, ensure_ascii=False)
        + "\nJSON schema: {\"stop_expression\":\"Python boolean expression\",\"declared_constraints\":[\"text\"]}. "
        "Expression variables are only current, previous, action dictionaries. Allowed: subscripts, literals, "
        "comparisons, and/or/not, membership, string/list +. Forbidden: calls, attributes, loops, comprehensions, "
        "imports, assignment. Stop only for observable unsafe divergence or a clear pre-endpoint stall. "
        "Use the literal False if no conservative rule can distinguish a failure from the normal examples."
    )
    rejected: list[Dict[str, str]] = []
    response: Dict[str, Any] = {}
    expression = ""
    checked: list[str] = []
    for attempt in range(1, 4):
        repair = ""
        if rejected:
            repair = (
                "\nThe previous expression was rejected by the mandatory public-protocol validator: "
                + rejected[-1]["error"]
                + ". Return a corrected expression; False is valid and preferable to a false positive."
            )
        response = gateway.complete_tool_json(
            role=f"cam_adapted_code_generation_attempt_{attempt}",
            system=(
                "Output one JSON object immediately, with no reasoning or markdown. Generate a conservative runtime "
                "safety/stall expression for a synthetic embodied task. Never use hidden success or evaluator state."
            ),
            prompt=base_prompt + repair,
            tool_name="submit_monitor",
            schema=schema,
            image_paths=rendered["rgb_paths"],
            max_tokens=3000,
        )
        expression = str(response.get("stop_expression", ""))
        try:
            compiled = _compile(expression)
            checked = _validate_normal_progress(compiled, probes)
            break
        except ValueError as exc:
            rejected.append({"expression": expression, "error": str(exc)})
    else:
        (output / "rejected_monitors.json").write_text(
            json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise ValueError("Kimi-generated CaM-adapted monitor failed mandatory normal-progress validation")
    spec = {
        "reproduction_level": REPRODUCTION_LEVEL,
        "model": "kimi-k3",
        "stop_expression": expression,
        "declared_constraints": response.get("declared_constraints", []),
        "normal_progress_validation": {"passed": True, "probes": checked, "rejected_attempts": rejected},
        "input_contract": ["previous public state", "current public state", "current action metadata"],
        "image_generation_input": rendered["rgb_paths"],
    }
    path = output / "monitor_spec.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class CamAdaptedRuntime:
    def __init__(self, spec_path: Path, log_path: Path):
        self.spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        self.compiled = _compile(self.spec["stop_expression"])
        self.log_path = Path(log_path)
        self.previous: Optional[Dict[str, Any]] = None

    def evaluate(self, current: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        try:
            stop = bool(eval(self.compiled, {"__builtins__": {}}, {  # noqa: S307 - AST-whitelisted expression
                "current": current, "previous": self.previous or current, "action": action,
            }))
            decision = "stop" if stop else "continue"
            error = None
        except Exception as exc:
            # Fail open so monitor bugs cannot manufacture favorable stops.
            stop, decision, error = False, "continue", f"{type(exc).__name__}: {exc}"
        result = {
            "decision": decision,
            "recommended_stop": stop,
            "error": error,
            "reproduction_level": REPRODUCTION_LEVEL,
            "expression": self.spec["stop_expression"],
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"current": current, "previous": self.previous, "action": action, "result": result}, ensure_ascii=False) + "\n")
        self.previous = current
        return result


def from_environment(episode_dir: Path) -> Optional[CamAdaptedRuntime]:
    if os.environ.get("LOCAL_BENCHMARK_CAM_MODE") != "cam_adapted_symbolic":
        return None
    return CamAdaptedRuntime(Path(os.environ["LOCAL_BENCHMARK_CAM_SPEC"]), Path(episode_dir) / "cam_adapted/runtime_events.jsonl")
