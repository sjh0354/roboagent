import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.humanoid_executor import HumanoidExecutor
from experiments.multidomain_reading_setup.run_eval import (
    FAILURE_VARIANTS,
    apply_failure_variant,
    load_golden_trace,
    run_mock_case,
    run_mock_planner_case,
    save_golden_trace,
)
from experiments.multidomain_reading_setup.score import score_trace
from utils.mock_music_api import MockMusicAPI


def _case():
    return {
        "case_id": "demo",
        "utterance": "我准备看书了，帮我收拾一下桌子。",
        "expected": {
            "move_to_basket": ["water", "medicine_box"],
            "do_not_move": ["book", "notebook"],
            "background_brightness_max": 35,
            "desk_lamp_on": True,
            "audio_type": "white_noise",
        },
    }


def test_multidomain_mock_trace_scores_success():
    case = _case()
    trace = run_mock_case(case, "full")

    score = score_trace(case, trace)

    assert score["success"] is True
    assert all(score["submetrics"].values())


def test_multidomain_planner_trace_scores_success(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ROOT", str(tmp_path / "local_memory"))
    monkeypatch.setenv("OMNICLAW_MEMORY_MODE", "fixed")
    case = _case()
    trace = run_mock_planner_case(case, "fixed")

    score = score_trace(case, trace)

    assert trace["trace_mode"] == "planner"
    assert score["success"] is True
    assert len(trace["g1_execution_history"]) == 5
    assert len(trace["ur5e_execution_history"]) == 2


def test_golden_trace_record_and_replay(tmp_path):
    case = _case()
    trace = run_mock_case(case, "full")

    path = save_golden_trace(tmp_path, trace)
    replayed = load_golden_trace(tmp_path, case, "full")
    score = score_trace(case, replayed)

    assert path.exists()
    assert replayed["trace_mode"] == "replay"
    assert replayed["golden_source"] == str(path)
    assert score["success"] is True


def test_multidomain_score_catches_missing_audio():
    case = _case()
    trace = run_mock_case(case, "full")
    trace["g1_execution_history"] = [
        step for step in trace["g1_execution_history"]
        if step["action"] != "play_audio"
    ]

    score = score_trace(case, trace)

    assert score["success"] is False
    assert score["submetrics"]["cyber_api"] is False


def test_failure_variants_hit_expected_submetrics():
    case = _case()
    trace = run_mock_case(case, "full")
    expected_failed_metric = {
        "missing_audio": "cyber_api",
        "no_desk_lamp": "iot",
        "wrong_route_type": "route_correctness",
        "moves_notebook": "embodied",
        "no_tts": "tts",
    }

    assert set(expected_failed_metric) == set(FAILURE_VARIANTS)
    for failure_name, metric in expected_failed_metric.items():
        score = score_trace(case, apply_failure_variant(trace, failure_name))
        assert score["success"] is False
        assert score["submetrics"][metric] is False


def test_mock_music_api_validates_white_noise_request():
    api = MockMusicAPI()

    result = api.play({"action": "play", "audio_type": "white_noise", "volume": 35})

    assert result.success is True
    assert result.data["audio_type"] == "white_noise"


def test_humanoid_executor_light_supports_brightness_and_device():
    executor = HumanoidExecutor(simulation_mode=True, verbose=False)

    result = executor.execute_action(
        "tool",
        "control_light",
        {"action": "set_brightness", "device": "background_light", "brightness": 30},
    )

    assert result.success is True
    assert result.data["device"] == "background_light"
    assert result.data["brightness"] == 30


def test_humanoid_executor_play_audio_tool():
    executor = HumanoidExecutor(simulation_mode=True, verbose=False)

    result = executor.execute_action(
        "tool",
        "play_audio",
        {"action": "play", "audio_type": "white_noise", "volume": 35},
    )

    assert result.success is True
    assert result.data["audio"]["audio_type"] == "white_noise"
