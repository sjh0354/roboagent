"""
Scoring utilities for the multidomain reading setup benchmark.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def score_trace(case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    expected = case.get("expected", {})
    g1_steps = trace.get("g1_execution_history", [])
    ur5e_steps = trace.get("ur5e_execution_history", [])
    all_steps = [*g1_steps, *ur5e_steps]

    voice = bool(trace.get("transcript"))
    embodied = _score_embodied(expected, ur5e_steps)
    iot = _score_iot(expected, g1_steps)
    cyber_api = _score_audio(expected, g1_steps)
    tts = _score_tts(g1_steps)
    route = _score_route_correctness(all_steps)

    submetrics = {
        "voice": voice,
        "embodied": embodied,
        "iot": iot,
        "cyber_api": cyber_api,
        "tts": tts,
        "route_correctness": route,
    }
    success = all(submetrics.values())
    return {
        "case_id": case.get("case_id"),
        "memory_mode": trace.get("memory_mode"),
        "success": success,
        "submetrics": submetrics,
        "failure_reasons": [
            name for name, passed in submetrics.items() if not passed
        ],
    }


def _score_embodied(expected: Dict[str, Any], steps: List[Dict[str, Any]]) -> bool:
    moved = set()
    wrongly_moved = set()
    protected = set(expected.get("do_not_move") or [])
    for step in steps:
        if step.get("action") != "pick_and_place":
            continue
        params = step.get("parameters") or {}
        item = _normalize_item(params.get("item_name"))
        target = str(params.get("target") or "").lower()
        if target == "basket":
            moved.add(item)
        if item in protected:
            wrongly_moved.add(item)
    required = {_normalize_item(item) for item in expected.get("move_to_basket") or []}
    return required.issubset(moved) and not wrongly_moved


def _score_iot(expected: Dict[str, Any], steps: List[Dict[str, Any]]) -> bool:
    background_dimmed = False
    desk_lamp_on = not expected.get("desk_lamp_on", False)
    max_brightness = int(expected.get("background_brightness_max", 35))
    for step in steps:
        if step.get("action") != "control_light":
            continue
        params = step.get("parameters") or {}
        device = str(params.get("device") or "room_light").lower()
        action = str(params.get("action") or "").lower()
        brightness = params.get("brightness")
        if device == "background_light" and action == "set_brightness":
            background_dimmed = isinstance(brightness, int) and brightness <= max_brightness
        if device == "desk_lamp" and action == "turn_on":
            desk_lamp_on = True
    return background_dimmed and desk_lamp_on


def _score_audio(expected: Dict[str, Any], steps: List[Dict[str, Any]]) -> bool:
    expected_audio = expected.get("audio_type", "white_noise")
    for step in steps:
        if step.get("action") != "play_audio":
            continue
        params = step.get("parameters") or {}
        if params.get("audio_type") == expected_audio and params.get("action", "play") == "play":
            return True
    return False


def _score_tts(steps: List[Dict[str, Any]]) -> bool:
    required_markers = ["桌", "清", "阅读", "灯", "背景"]
    for step in steps:
        if step.get("action") != "speak":
            continue
        message = str((step.get("parameters") or {}).get("message") or "")
        if all(marker in message for marker in required_markers):
            return True
    return False


def _score_route_correctness(steps: Iterable[Dict[str, Any]]) -> bool:
    expected_types = {
        "send_agent_message": "talk",
        "control_light": "tool",
        "play_audio": "tool",
        "speak": "talk",
        "pick_and_place": "act",
    }
    for step in steps:
        action = step.get("action")
        if action in expected_types and step.get("action_type") != expected_types[action]:
            return False
    return True


def _normalize_item(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    if text in {"medicine", "green_box", "medication_box"}:
        return "medicine_box"
    if text in {"bottle", "bottle_of_water", "mineral_water"}:
        return "water"
    return text
