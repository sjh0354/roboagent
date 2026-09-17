from experiments.local_benchmark.simulator.render import LOCATIONS
from experiments.local_benchmark.simulator.world import StatefulWorld


def task():
    return {
        "task_id": "contract",
        "request": "move block",
        "initial_state": {"robot_location": "desk", "objects": {"block": "desk"}, "devices": {}, "preferences": []},
        "success": {"objects": {"block": "basket"}},
    }


def task_with_budget(limit):
    value = task()
    value["max_actions"] = limit
    return value


def _sample_rate(tmp_path, probability, monitor_present, trials=2000):
    world = StatefulWorld(
        task(), 77, tmp_path, physical_success_prob=probability,
        monitor_present=monitor_present, render_enabled=False,
    )
    successes = 0
    for index in range(trials):
        destination = f"target_{index}"
        world.robot_location = "desk"
        world.start_skill("navigate", {"destination": destination})
        successes += world.robot_location == destination
    return world, successes / trials


def test_repeat_id_never_controls_physical_rng(tmp_path):
    world = StatefulWorld(task(), 12, tmp_path, render_enabled=False)
    score = world.score()
    assert score["repeat_id"] == 12
    assert score["seed_controls_physics"] is False
    assert score["physical_rng_mode"] == "system_random_unseeded"
    assert not hasattr(world, "physical_rng_seed")


def test_no_monitor_aggregate_probability_is_point_six(tmp_path):
    world, rate = _sample_rate(tmp_path, 0.60, False)
    assert abs(rate - 0.60) < 0.04
    assert world.counts["valid_physical_calls"] == 2000
    assert world.counts["atomic_physical_resolutions"] == 2000


def test_monitor_aggregate_probability_is_point_eight_five(tmp_path):
    world, rate = _sample_rate(tmp_path, 0.85, True)
    assert abs(rate - 0.85) < 0.04
    score = world.score()
    assert score["monitor_present"] is True
    assert score["integrated_monitor_retry_limit"] == 2
    assert score["explicit_physical_retries_simulated"] is False


def test_physical_operation_resolves_atomically(tmp_path):
    world = StatefulWorld(task(), 0, tmp_path, physical_success_prob=1.0, render_enabled=False)
    result = world.start_skill("pick_and_place", {"item": "block", "source": "desk", "target": "basket"})
    assert result["accepted"] is True
    assert result["execution_resolved"] is True
    assert result["explicit_retry_available"] is False
    assert "action_id" not in result
    assert world.objects["block"] == "basket"
    assert world.public_snapshot()["active_action"] is None


def test_same_aggregate_goal_cannot_be_sampled_twice(tmp_path):
    world = StatefulWorld(task(), 0, tmp_path, physical_success_prob=0.0, render_enabled=False)
    arguments = {"item": "block", "source": "desk", "target": "basket"}
    first = world.start_skill("pick_and_place", arguments)
    second = world.start_skill("pick_and_place", arguments)
    assert first["accepted"] is True
    assert second["diagnostic"] == "aggregate_probability_already_consumed_for_goal"
    assert second["explicit_retry_available"] is False
    assert world.counts["valid_physical_calls"] == 1
    assert world.counts["duplicate_goal_rejections"] == 1
    assert world.counts["explicit_physical_retries"] == 0


def test_duplicate_budget_cannot_be_evaded_by_switching_skill_name(tmp_path):
    world = StatefulWorld(task(), 0, tmp_path, physical_success_prob=0.0, render_enabled=False)
    world.start_skill("pick_and_place", {"item": "block", "source": "desk", "target": "basket"})
    world.objects["block"] = "held"
    world.held_object = "block"
    rejected = world.start_skill("place", {"item": "block", "target": "basket"})
    assert rejected["diagnostic"] == "aggregate_probability_already_consumed_for_goal"


def test_illegal_call_never_rolls_for_success(tmp_path):
    world = StatefulWorld(task(), 1, tmp_path, physical_success_prob=1.0, render_enabled=False)
    result = world.start_skill("pick", {"item": "missing"})
    assert result == {"accepted": False, "diagnostic": "unknown_item"}
    assert world.counts["valid_physical_calls"] == 0
    assert world.counts["illegal_calls"] == 1


def test_pick_then_place_resolve_as_two_distinct_goals(tmp_path):
    world = StatefulWorld(task(), 5, tmp_path, physical_success_prob=1.0, render_enabled=False)
    picked = world.start_skill("pick", {"item": "block"})
    assert picked["execution_resolved"] is True
    rejected = world.start_skill("pick_and_place", {"item": "block", "target": "basket"})
    assert rejected == {"accepted": False, "diagnostic": "gripper_not_empty_use_place"}
    placed = world.start_skill("place", {"item": "block", "target": "basket"})
    assert placed["execution_resolved"] is True
    assert world.objects["block"] == "basket"
    assert world.held_object is None


def test_pick_and_place_requires_robot_and_object_colocation(tmp_path):
    world = StatefulWorld(task(), 5, tmp_path, physical_success_prob=1.0, render_enabled=False)
    world.robot_location = "shelf"
    rejected = world.start_skill("pick_and_place", {"item": "block", "source": "desk", "target": "basket"})
    assert rejected == {"accepted": False, "diagnostic": "item_not_at_robot_location"}
    assert world.counts["valid_physical_calls"] == 0


def test_public_interface_does_not_expose_hidden_draw_or_failure_mode(tmp_path):
    world = StatefulWorld(task(), 3, tmp_path, render_enabled=True)
    result = world.start_skill("pick_and_place", {"item": "block", "source": "desk", "target": "basket"})
    public_text = repr(world.public_snapshot()) + repr(result) + repr(world.observe())
    assert "random_draw" not in public_text
    assert "failure_mode" not in public_text
    assert "success':" not in public_text
    assert list((tmp_path / "frames").glob("*.png"))


def test_task_action_budget_blocks_extra_state_changes(tmp_path):
    world = StatefulWorld(task_with_budget(1), 9, tmp_path, physical_success_prob=1.0, render_enabled=False)
    world.start_skill("pick_and_place", {"item": "block", "source": "desk", "target": "basket"})
    assert world.objects["block"] == "basket"
    assert world.set_device("lamp", "on") == {"accepted": False, "diagnostic": "action_budget_exhausted"}
    assert "lamp" not in world.devices


def test_formal_renderer_locations_are_distinct():
    assert len(LOCATIONS) == 16
    assert len(set(LOCATIONS.values())) == len(LOCATIONS)
