"""Generate the frozen 40-task local synthetic formal manifest.

This is an independently designed local suite, not the paper's CoRL task set.
The 4/8/8 complexity split is applied separately to manipulation/navigation.
"""

from __future__ import annotations

import json
from pathlib import Path


OBJECTS = [
    "water_bottle", "medicine_box", "red_block", "blue_cup", "notebook",
    "snack_box", "remote_control", "tissue_pack", "green_block", "mug",
    "glasses_case", "tea_tin", "charger", "pencil_case", "small_towel",
    "fruit_bowl", "soap_box", "plant_pot", "mail_bundle", "tool_pouch",
]
SOURCES = ["desk", "counter", "shelf", "side_table"]
TARGETS = ["basket", "storage_bin", "tray", "upper_shelf"]
LOCATIONS = ["kitchen", "reading_corner", "entryway", "workbench", "living_room", "bedroom", "pantry", "hallway"]


def complexity(index: int) -> int:
    return 1 if index < 4 else 2 if index < 12 else 3


def manipulation(index: int) -> dict:
    level = complexity(index)
    item = OBJECTS[index]
    source = SOURCES[index % len(SOURCES)]
    target = TARGETS[index % len(TARGETS)]
    objects = {item: source, f"distractor_{index:02d}": source}
    success = {"objects": {item: target, f"distractor_{index:02d}": source}}
    request = f"Move the {item} from the {source} to the {target}, leave the distractor untouched, and verify the result."
    allowed = ["observe", "pick", "place", "pick_and_place"]
    devices = {}
    if level >= 2:
        device = f"work_light_{index:02d}"
        devices[device] = "off"
        success["devices"] = {device: "on"}
        request += f" Then turn {device} on."
        allowed.append("set_device")
    if level == 3:
        second = OBJECTS[(index + 5) % len(OBJECTS)] + "_secondary"
        second_target = TARGETS[(index + 1) % len(TARGETS)]
        objects[second] = source
        success["objects"][second] = second_target
        request += f" Also move the {second} to the {second_target}."
    return {
        "task_id": f"formal_manipulation_{index + 1:02d}", "split": "formal", "domain": "manipulation",
        "complexity": level, "request": request,
        "initial_state": {"robot_location": source, "held_object": None, "objects": objects, "devices": devices, "preferences": []},
        "success": success, "allowed_tools": allowed, "max_actions": 6 + level * 4,
        "interaction_events": [{"at_sim_time": 1.0, "kind": "observable_progress"}] if level == 3 else [],
        "failure_modes": ["no_effect", "partial", "recoverable_stall"],
        "provenance": "independent local synthetic formal suite; not a CoRL hardware task",
    }


def navigation(index: int) -> dict:
    level = complexity(index)
    start = LOCATIONS[(index + 4) % len(LOCATIONS)]
    destination = LOCATIONS[index % len(LOCATIONS)]
    if start == destination:
        start = "charging_station"
    request = f"Navigate from the {start} to the {destination} and verify arrival."
    devices = {}
    success = {"robot_location": destination}
    allowed = ["observe", "navigate"]
    if level >= 2:
        device = f"zone_light_{index:02d}"
        devices[device] = "off"
        success["devices"] = {device: "on"}
        request += f" Turn {device} on after arrival."
        allowed.append("set_device")
    if level == 3:
        preference = f"quiet route {index:02d}"
        success["preference_contains"] = preference
        request = f"Remember that the user prefers {preference}. " + request
        allowed.append("remember")
    return {
        "task_id": f"formal_navigation_{index + 1:02d}", "split": "formal", "domain": "navigation",
        "complexity": level, "request": request,
        "initial_state": {"robot_location": start, "held_object": None, "objects": {}, "devices": devices, "preferences": []},
        "success": success, "allowed_tools": allowed, "max_actions": 5 + level * 3,
        "interaction_events": [{"at_sim_time": 2.0, "kind": "observable_progress"}] if level == 3 else [],
        "failure_modes": ["no_effect", "partial", "recoverable_stall"],
        "provenance": "independent local synthetic formal suite; not a CoRL hardware task",
    }


def main() -> None:
    rows = [manipulation(i) for i in range(20)] + [navigation(i) for i in range(20)]
    target = Path(__file__).with_name("formal_40.jsonl")
    target.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(target, len(rows))


if __name__ == "__main__":
    main()
