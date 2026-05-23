"""Scoring utilities for the long-horizon orchestration benchmark."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional


CYBER_ACTION_TYPES = {"tool"}
PHYSICAL_ACTION_TYPES = {"act"}
VISION_ACTION_TYPES = {"sense"}


def score_trace(case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    expected_steps = case.get("expected_trace") or []
    actual_steps = extract_steps(trace)
    matched = [_find_matching_step(expected, actual_steps) for expected in expected_steps]

    action_correctness = all(item is not None for item in matched)
    cyber_iot_success = _all_expected_type_matched(expected_steps, matched, CYBER_ACTION_TYPES)
    physical_command_success = _all_expected_type_matched(expected_steps, matched, PHYSICAL_ACTION_TYPES)
    vision_success = _all_expected_type_matched(expected_steps, matched, VISION_ACTION_TYPES)
    final_report_success = _score_final_report(expected_steps, matched)
    physical_execution_success = _score_physical_execution(expected_steps, matched)

    submetrics = {
        "action_correctness": action_correctness,
        "cyber_iot_success": cyber_iot_success,
        "physical_command_success": physical_command_success,
        "physical_execution_success": physical_execution_success,
        "vision_success": vision_success,
        "final_report_success": final_report_success,
    }
    overall = all(submetrics.values())
    return {
        "case_id": case.get("case_id"),
        "method": trace.get("method"),
        "success": overall,
        "submetrics": submetrics,
        "physical_action_count": len(_expected_by_type(expected_steps, PHYSICAL_ACTION_TYPES)),
        "physical_action_success_count": _count_physical_execution_success(expected_steps, matched),
        "failure_reasons": [name for name, passed in submetrics.items() if not passed],
    }


def apply_physical_execution(
    trace: Dict[str, Any],
    success_rate: float = 0.7,
    seed: int = 7,
) -> Dict[str, Any]:
    """Attach deterministic sampled execution results to physical action steps."""
    mutated = json.loads(json.dumps(trace, ensure_ascii=False))
    case_id = str(mutated.get("case_id") or "unknown_case")
    method = str(mutated.get("method") or "unknown_method")
    for index, step in enumerate(extract_steps(mutated)):
        if step.get("action_type") != "act":
            continue
        sampled_success = _deterministic_success(
            seed=seed,
            success_rate=success_rate,
            method=method,
            case_id=case_id,
            index=index,
            action=str(step.get("action") or ""),
            parameters=step.get("parameters") or {},
        )
        step["execution_result"] = {
            "success": sampled_success,
            "feedback": "mock physical execution succeeded" if sampled_success else "mock physical execution failed",
            "error": None if sampled_success else "sampled_physical_failure",
            "data": {"physical_success_rate": success_rate, "seed": seed},
        }
        step["assumed_successful"] = sampled_success
    mutated["physical_success_rate"] = success_rate
    mutated["physical_seed"] = seed
    return mutated


def extract_steps(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(trace.get("execution_history"), list):
        return trace["execution_history"]
    if isinstance(trace.get("steps"), list):
        return trace["steps"]

    steps: List[Dict[str, Any]] = []
    for key in (
        "g1_execution_history",
        "humanoid_g1_execution_history",
        "ur5e_execution_history",
        "arm_execution_history",
    ):
        value = trace.get(key)
        if isinstance(value, list):
            steps.extend(value)
    return steps


def _find_matching_step(expected: Dict[str, Any], actual_steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for actual in actual_steps:
        if actual.get("action") != expected.get("action"):
            continue
        if actual.get("action_type") != expected.get("action_type"):
            continue
        if _parameters_match(expected.get("parameters") or {}, actual.get("parameters") or {}):
            return actual
    return None


def _parameters_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    expected_contains = expected.get("message_contains")
    if expected_contains is not None:
        actual_message = str(actual.get("message") or "")
        if not actual_message and isinstance(actual.get("message_contains"), list):
            actual_message = " ".join(str(item) for item in actual.get("message_contains") or [])
        return all(str(marker) in actual_message for marker in expected_contains)

    for key, expected_value in expected.items():
        if key not in actual:
            return False
        if actual[key] != expected_value:
            return False
    return True


def _all_expected_type_matched(
    expected_steps: List[Dict[str, Any]],
    matched: List[Optional[Dict[str, Any]]],
    action_types: set[str],
) -> bool:
    required = _expected_by_type(expected_steps, action_types)
    if not required:
        return True
    for expected, actual in zip(expected_steps, matched):
        if expected.get("action_type") in action_types and actual is None:
            return False
    return True


def _expected_by_type(expected_steps: List[Dict[str, Any]], action_types: set[str]) -> List[Dict[str, Any]]:
    return [step for step in expected_steps if step.get("action_type") in action_types]


def _score_final_report(
    expected_steps: List[Dict[str, Any]],
    matched: List[Optional[Dict[str, Any]]],
) -> bool:
    talk_steps = [
        actual
        for expected, actual in zip(expected_steps, matched)
        if expected.get("action_type") == "talk" and expected.get("action") == "speak" and actual is not None
    ]
    expected_talk = [
        expected
        for expected in expected_steps
        if expected.get("action_type") == "talk" and expected.get("action") == "speak"
    ]
    return not expected_talk or bool(talk_steps)


def _score_physical_execution(
    expected_steps: List[Dict[str, Any]],
    matched: List[Optional[Dict[str, Any]]],
) -> bool:
    expected_physical = _expected_by_type(expected_steps, PHYSICAL_ACTION_TYPES)
    if not expected_physical:
        return True
    for expected, actual in zip(expected_steps, matched):
        if expected.get("action_type") != "act":
            continue
        if actual is None:
            return False
        result = actual.get("execution_result") or {}
        if result.get("success") is not True:
            return False
    return True


def _count_physical_execution_success(
    expected_steps: List[Dict[str, Any]],
    matched: List[Optional[Dict[str, Any]]],
) -> int:
    count = 0
    for expected, actual in zip(expected_steps, matched):
        if expected.get("action_type") == "act" and actual is not None:
            if (actual.get("execution_result") or {}).get("success") is True:
                count += 1
    return count


def _deterministic_success(
    seed: int,
    success_rate: float,
    method: str,
    case_id: str,
    index: int,
    action: str,
    parameters: Dict[str, Any],
) -> bool:
    bounded_rate = min(1.0, max(0.0, float(success_rate)))
    payload = json.dumps(
        {
            "seed": seed,
            "method": method,
            "case_id": case_id,
            "index": index,
            "action": action,
            "parameters": parameters,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return value < bounded_rate


def ratio(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(1 for item in items if item) / len(items), 4)
