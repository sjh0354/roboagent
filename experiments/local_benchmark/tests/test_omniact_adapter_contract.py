from experiments.local_benchmark.adapters.omniact import OmniActLocalPlanner
from experiments.local_benchmark.simulator.world import StatefulWorld


class _NoMonitorGateway:
    def monitor(self, **_kwargs):
        raise AssertionError("monitor must not be called when the policy is none")


def test_monitor_off_resolves_physical_skill_once_without_monitor_call(tmp_path):
    task = {
        "task_id": "monitor-off-contract",
        "request": "move block",
        "initial_state": {"robot_location": "desk", "held_object": None, "objects": {"block": "desk"}, "devices": {}, "preferences": []},
        "success": {"objects": {"block": "basket"}},
        "max_actions": 4,
    }
    world = StatefulWorld(task, 0, tmp_path, physical_success_prob=1.0, render_enabled=True)
    planner = OmniActLocalPlanner.__new__(OmniActLocalPlanner)
    planner.world = world
    planner.method_id = "iterative_posthoc"
    planner.method = {"memory": "none", "monitor": "none", "strategy": "raw"}
    planner.temporal_frames = True
    planner.gateway = _NoMonitorGateway()
    planner.rag = None
    planner.summary_memories = []
    planner._recent_frames = []
    planner.step_count = 0
    planner.max_steps = 4
    planner.is_task_complete = False
    planner.execution_history = []
    planner.conversation_history = []
    planner.transient_memory_packet = None
    planner._execute_step({
        "next_step": {
            "action": "execute_skill",
            "parameters": {"skill": "pick_and_place", "arguments": {"item": "block", "source": "desk", "target": "basket"}},
        }
    })
    assert world.objects["block"] == "basket"
    assert world.counts["atomic_physical_resolutions"] == 1
    assert world.counts["explicit_physical_retries"] == 0
    record = planner.execution_history[-1]
    assert record["execution_control_policy"] == "direct_aggregate_bernoulli"
    assert record["monitor_effect_integrated"] is False
    assert record["effective_physical_success_probability"] == 1.0
    assert record["explicit_physical_retries_simulated"] is False
    assert record["execution_result"]["execution_resolved"] is True
