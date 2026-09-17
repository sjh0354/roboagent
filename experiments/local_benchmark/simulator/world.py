"""Stateful synthetic world with direct aggregate physical resolution."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .render import render_observation


PHYSICAL_SKILLS = {"pick", "place", "navigate", "pick_and_place"}
FAILURE_MODES = ("no_effect", "partial", "recoverable_stall")


class StatefulWorld:
    """One episode of the simplified simulated-physical benchmark.

    Every valid normalized embodied goal is resolved exactly once. The supplied
    probability is already an aggregate policy assumption (0.85 with a monitor,
    0.60 without one), so phases and explicit physical retries are deliberately
    not simulated. SystemRandom is used directly; the public repetition label
    never controls physical outcomes.
    """

    def __init__(
        self,
        task: Dict[str, Any],
        seed: int,
        output_dir: Path,
        physical_success_prob: float = 0.60,
        *,
        monitor_present: bool = False,
        base_single_attempt_prob: float = 0.50,
        integrated_monitor_retry_limit: int = 2,
        render_enabled: bool = True,
    ):
        if not 0.0 <= physical_success_prob <= 1.0:
            raise ValueError("physical_success_prob must be in [0, 1]")
        if not 0.0 <= base_single_attempt_prob <= 1.0:
            raise ValueError("base_single_attempt_prob must be in [0, 1]")
        if int(integrated_monitor_retry_limit) < 0:
            raise ValueError("integrated_monitor_retry_limit must be non-negative")
        self.task = task
        # Compatibility field only: this repeat id never enters an RNG.
        self.seed = int(seed)
        self.physical_success_prob = float(physical_success_prob)
        self.monitor_present = bool(monitor_present)
        self.base_single_attempt_prob = float(base_single_attempt_prob)
        self.integrated_monitor_retry_limit = int(integrated_monitor_retry_limit)
        self._physical_rng = secrets.SystemRandom()
        self.output_dir = Path(output_dir).resolve()
        self.render_enabled = bool(render_enabled)
        self.frame_dir = self.output_dir / "frames"
        self.private_trace_path = self.output_dir / "private_trace.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        initial = task["initial_state"]
        self.objects = dict(initial.get("objects", {}))
        self.robot_location = initial.get("robot_location", "desk")
        self.held_object: Optional[str] = initial.get("held_object")
        self.devices = dict(initial.get("devices", {}))
        self.preferences = list(initial.get("preferences", []))
        self.sim_time = 0.0
        self._resolved_goals: set[str] = set()
        self._frame_index = 0
        self._last_frame: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()
        self.counts = {
            "all_physical_attempts": 0,
            "valid_physical_calls": 0,
            "illegal_calls": 0,
            "atomic_physical_resolutions": 0,
            "semantic_successes": 0,
            "semantic_failures": 0,
            "duplicate_goal_rejections": 0,
            "explicit_physical_retries": 0,
            "digital_calls": 0,
        }
        self._write_private("reset", {
            "task_id": task["task_id"],
            "repeat_id": self.seed,
            "seed_controls_physics": False,
            "physical_rng_mode": "system_random_unseeded",
            "effective_success_probability": self.physical_success_prob,
            "monitor_present": self.monitor_present,
            "base_single_attempt_probability": self.base_single_attempt_prob,
            "integrated_monitor_retry_limit": self.integrated_monitor_retry_limit if self.monitor_present else 0,
            "explicit_physical_retries_simulated": False,
        })
        self.observe()

    def _write_private(self, event: str, payload: Dict[str, Any]) -> None:
        record = {"event": event, "sim_time": round(self.sim_time, 3), **payload}
        with self.private_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _action_key(skill: str, arguments: Dict[str, Any]) -> str:
        item = str(arguments.get("item") or arguments.get("item_name") or "-")
        target = str(arguments.get("target") or arguments.get("destination") or "-")
        goal_kind = "move_item" if item != "-" and target != "-" else skill
        return f"{goal_kind}|{item}|{target}"

    def _check_preconditions(self, skill: str, args: Dict[str, Any]) -> Optional[str]:
        item = args.get("item") or args.get("item_name")
        if skill == "pick":
            if not item or item not in self.objects:
                return "unknown_item"
            if self.held_object is not None:
                return "gripper_not_empty"
            if self.objects[item] != self.robot_location:
                return "item_not_at_robot_location"
        elif skill == "place":
            if not item or self.held_object != item:
                return "item_not_held"
            if not (args.get("target") or args.get("destination")):
                return "missing_target"
        elif skill == "navigate":
            destination = args.get("destination") or args.get("target")
            if not destination or destination == self.robot_location:
                return "invalid_destination"
        elif skill == "pick_and_place":
            source = args.get("source")
            target = args.get("target") or args.get("destination")
            if not item or item not in self.objects:
                return "unknown_item"
            if self.held_object is not None:
                return "gripper_not_empty_use_place"
            if source and self.objects[item] != source:
                return "item_not_at_source"
            if self.objects[item] != self.robot_location:
                return "item_not_at_robot_location"
            if not target:
                return "missing_target"
        return None

    def _action_budget_exhausted(self) -> bool:
        limit = self.task.get("max_actions")
        if limit is None:
            return False
        used = self.counts["all_physical_attempts"] + self.counts["digital_calls"]
        return used >= int(limit)

    def _apply_success(self, skill: str, arguments: Dict[str, Any]) -> None:
        item = arguments.get("item") or arguments.get("item_name")
        target = arguments.get("target") or arguments.get("destination")
        if skill == "navigate":
            self.robot_location = str(target)
        elif skill == "pick" and item:
            self.objects[item] = "held"
            self.held_object = str(item)
        elif skill == "place" and item:
            self.objects[item] = str(target)
            self.held_object = None
        elif skill == "pick_and_place" and item:
            self.objects[item] = str(target)

    def _apply_failure(self, skill: str, arguments: Dict[str, Any], mode: str) -> None:
        item = arguments.get("item") or arguments.get("item_name")
        target = arguments.get("target") or arguments.get("destination")
        source = arguments.get("source")
        if skill == "navigate":
            if mode != "no_effect":
                self.robot_location = f"corridor_to_{target}"
            return
        if not item:
            return
        original = str(source or self.task["initial_state"].get("objects", {}).get(item, "desk"))
        if mode == "no_effect":
            self.objects[item] = original
        elif mode == "partial":
            self.objects[item] = f"floor_near_{target or original}"
            if self.held_object == item:
                self.held_object = None
        else:
            self.objects[item] = f"near_{original}"

    def start_skill(self, skill: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resolve one embodied operation atomically and return observable state."""
        arguments = dict(arguments or {})
        with self._lock:
            if skill not in PHYSICAL_SKILLS:
                return {"accepted": False, "diagnostic": "unknown_physical_skill"}
            if self._action_budget_exhausted():
                self.counts["all_physical_attempts"] += 1
                self.counts["illegal_calls"] += 1
                return {"accepted": False, "diagnostic": "action_budget_exhausted"}
            self.counts["all_physical_attempts"] += 1
            goal_key = self._action_key(skill, arguments)
            if goal_key in self._resolved_goals:
                self.counts["illegal_calls"] += 1
                self.counts["duplicate_goal_rejections"] += 1
                self._write_private("duplicate_aggregate_goal_rejected", {"skill": skill, "arguments": arguments})
                return {
                    "accepted": False,
                    "diagnostic": "aggregate_probability_already_consumed_for_goal",
                    "explicit_retry_available": False,
                }
            invalid = self._check_preconditions(skill, arguments)
            if invalid:
                self.counts["illegal_calls"] += 1
                self._write_private("illegal_skill", {"skill": skill, "arguments": arguments, "reason": invalid})
                return {"accepted": False, "diagnostic": invalid}
            self._resolved_goals.add(goal_key)
            self.counts["valid_physical_calls"] += 1
            self.counts["atomic_physical_resolutions"] += 1
            random_draw = self._physical_rng.random()
            success = random_draw < self.physical_success_prob
            failure_mode = self._physical_rng.choice(FAILURE_MODES)
            if success:
                self._apply_success(skill, arguments)
                self.counts["semantic_successes"] += 1
            else:
                self._apply_failure(skill, arguments, failure_mode)
                self.counts["semantic_failures"] += 1
            self.sim_time += 1.0
            observation = self.observe()
            self._write_private("physical_operation_resolved", {
                "skill": skill,
                "arguments": arguments,
                "goal_key": goal_key,
                "random_draw": random_draw,
                "effective_success_probability": self.physical_success_prob,
                "success": success,
                "failure_mode": None if success else failure_mode,
                "frame_index": observation["frame_index"],
            })
            return {
                "accepted": True,
                "execution_resolved": True,
                "diagnostic": "aggregate_physical_operation_resolved_inspect_observation",
                "effective_success_probability": self.physical_success_prob,
                "explicit_retry_available": False,
                "observation": observation,
            }

    def set_device(self, device: str, state: Any) -> Dict[str, Any]:
        with self._lock:
            if self._action_budget_exhausted():
                return {"accepted": False, "diagnostic": "action_budget_exhausted"}
            if not device:
                return {"accepted": False, "diagnostic": "missing_device"}
            self.devices[str(device)] = state
            self.sim_time += 0.2
            self.counts["digital_calls"] += 1
            frame = self.observe()
            self._write_private("device_set", {"device": device, "state": state, "frame_index": frame["frame_index"]})
            return {"accepted": True, "diagnostic": "device_state_updated", "observation": frame}

    def store_preference(self, text: str) -> Dict[str, Any]:
        with self._lock:
            if self._action_budget_exhausted():
                return {"accepted": False, "diagnostic": "action_budget_exhausted"}
            cleaned = str(text or "").strip()
            if not cleaned:
                return {"accepted": False, "diagnostic": "empty_preference"}
            self.preferences.append(cleaned)
            self.counts["digital_calls"] += 1
            self._write_private("preference_stored", {"text": cleaned})
            return {"accepted": True, "diagnostic": "preference_recorded"}

    def public_snapshot(self) -> Dict[str, Any]:
        return {
            "sim_time": round(self.sim_time, 3),
            "robot_location": self.robot_location,
            "held_object": self.held_object,
            "objects": dict(self.objects),
            "devices": dict(self.devices),
            "preferences": list(self.preferences),
            "active_action": None,
        }

    def observe(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = self.public_snapshot()
            self._frame_index += 1
            if self.render_enabled:
                rendered = render_observation(snapshot, self.frame_dir, self._frame_index)
            else:
                rendered = {
                    "frame_index": self._frame_index,
                    "rgb_paths": [],
                    "depth_npy": None,
                    "depth_png": None,
                    "intrinsics": {},
                    "extrinsics": {},
                    "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
                    "metadata_path": None,
                }
            public = {**rendered, "public_state": snapshot}
            self._last_frame = public
            return public

    def score(self) -> Dict[str, Any]:
        """Private evaluator; never exposed through agent tools."""
        checks = []
        predicate = self.task.get("success", {})
        for item, expected in predicate.get("objects", {}).items():
            actual = self.objects.get(item)
            checks.append({"kind": "object", "name": item, "expected": expected, "actual": actual, "passed": actual == expected})
        for device, expected in predicate.get("devices", {}).items():
            actual = self.devices.get(device)
            checks.append({"kind": "device", "name": device, "expected": expected, "actual": actual, "passed": actual == expected})
        if "robot_location" in predicate:
            expected = predicate["robot_location"]
            checks.append({"kind": "robot_location", "expected": expected, "actual": self.robot_location, "passed": self.robot_location == expected})
        required_preference = predicate.get("preference_contains")
        if required_preference:
            passed = any(required_preference.lower() in p.lower() for p in self.preferences)
            checks.append({"kind": "preference_contains", "expected": required_preference, "passed": passed})
        return {
            "task_success": bool(checks) and all(item["passed"] for item in checks),
            "checks": checks,
            "counts": dict(self.counts),
            "sim_time": self.sim_time,
            "repeat_id": self.seed,
            "seed_controls_physics": False,
            "physical_rng_mode": "system_random_unseeded",
            "execution_model": "direct_aggregate_bernoulli_per_normalized_embodied_goal",
            "base_single_attempt_probability": self.base_single_attempt_prob,
            "effective_physical_success_probability": self.physical_success_prob,
            "monitor_present": self.monitor_present,
            "integrated_monitor_retry_limit": self.integrated_monitor_retry_limit if self.monitor_present else 0,
            "explicit_physical_retries_simulated": False,
        }
