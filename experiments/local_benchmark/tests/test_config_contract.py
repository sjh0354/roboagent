import json
from pathlib import Path

import pytest

from experiments.local_benchmark.io import REPO_ROOT, load_json_config, load_manifest
from experiments.local_benchmark.__main__ import _method_has_monitor

from test.test_memory_modes import _MemoryModePlanner


def test_unknown_memory_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("OMNICLAW_MEMORY_MODE", "typo_mode")
    with pytest.raises(ValueError, match="full/none/fixed"):
        _MemoryModePlanner()


def test_frozen_formal_protocol_uses_three_repeat_ids_and_40_tasks():
    config = load_json_config(REPO_ROOT / "experiments/local_benchmark/configs/formal_simulation.yaml")
    assert config["episode_seeds"] == [0, 1, 2]
    assert config["base_single_attempt_success_prob"] == 0.50
    assert config["effective_physical_success_prob"] == {"monitor_present": 0.85, "monitor_absent": 0.60}
    assert config["integrated_monitor_retry_limit"] == 2
    assert config["explicit_physical_retries_simulated"] is False
    assert config["physical_rng_mode"] == "system_random_unseeded"
    assert config["seed_controls_physics"] is False
    assert config["inter_episode_cooldown_seconds"] == 5
    tasks = load_manifest(config["task_manifest"])
    assert len(tasks) == 40
    assert sum(task["domain"] == "manipulation" for task in tasks) == 20
    assert sum(task["domain"] == "navigation" for task in tasks) == 20


def test_matrix_plan_is_frozen_to_same_three_seeds():
    plan_path = Path(REPO_ROOT) / "experiments/local_benchmark/configs/matrix_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["episode_seeds"] == [0, 1, 2]
    assert plan["physical_execution_model"]["monitor_present_effective_prob"] == 0.85
    assert plan["physical_execution_model"]["monitor_absent_effective_prob"] == 0.60
    assert plan["physical_execution_model"]["explicit_retries_simulated"] is False
    assert plan["formal_pilot"]["tasks"] == ["formal_manipulation_01", "formal_navigation_01"]


def test_monitor_presence_maps_to_85_percent_policy_arms():
    assert _method_has_monitor("iterative_monitor") is True
    assert _method_has_monitor("omniact") is True
    assert _method_has_monitor("hermes_cam") is True
    assert _method_has_monitor("hermes_amem_cam") is True
    assert _method_has_monitor("iterative_posthoc") is False
    assert _method_has_monitor("iterative_memory") is False
    assert _method_has_monitor("hermes_native") is False
    assert _method_has_monitor("openclaw_native") is False
