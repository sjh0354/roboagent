import pytest

from experiments.local_benchmark.monitors.cam_adapted import (
    _compile,
    _normal_progress_probes,
    _validate_normal_progress,
)


def _task():
    return {
        "task_id": "cam-contract",
        "request": "move bottle to basket",
        "initial_state": {
            "robot_location": "desk",
            "held_object": None,
            "objects": {"bottle": "desk"},
            "devices": {},
            "preferences": [],
        },
        "success": {"objects": {"bottle": "basket"}, "robot_location": "shelf"},
    }


def test_generated_monitor_cannot_stop_documented_normal_progress():
    task = _task()
    initial = {**task["initial_state"], "sim_time": 0.0, "active_action": None}
    probes = _normal_progress_probes(task, initial)
    bad = _compile('(current["held_object"] != "bottle") and (current["objects"]["bottle"] not in ["desk", "basket"])')
    with pytest.raises(ValueError, match="normal_pick_approach"):
        _validate_normal_progress(bad, probes)


def test_false_monitor_passes_normal_progress_contract():
    task = _task()
    initial = {**task["initial_state"], "sim_time": 0.0, "active_action": None}
    probes = _normal_progress_probes(task, initial)
    checked = _validate_normal_progress(_compile("False"), probes)
    assert "normal_navigation_corridor" in checked
