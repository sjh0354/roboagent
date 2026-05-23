import os

from executor.arm_executor_vision import VisionEnabledArmExecutor


def test_pi0_waits_until_action_log_is_idle(monkeypatch, tmp_path):
    executor = object.__new__(VisionEnabledArmExecutor)
    executor.verbose = False

    log_path = tmp_path / "pi0_actions_test.log"
    log_path.write_text("# header\nstep1\n", encoding="utf-8")

    timeline = {"now": 1000.0, "polls": 0}

    def fake_time():
        return timeline["now"]

    def fake_sleep(seconds):
        timeline["now"] += seconds
        timeline["polls"] += 1
        if timeline["polls"] == 2:
            with open(log_path, "a", encoding="utf-8") as file:
                file.write("step2\n")
            os.utime(log_path, (timeline["now"], timeline["now"]))

    monkeypatch.setattr("executor.arm_executor_vision.time.time", fake_time)
    monkeypatch.setattr("executor.arm_executor_vision.time.sleep", fake_sleep)
    monkeypatch.setenv("PI0_ACTION_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("PI0_ACTION_MIN_WAIT_SECONDS", "4")
    monkeypatch.setenv("PI0_ACTION_IDLE_SECONDS", "3")
    monkeypatch.setenv("PI0_ACTION_POLL_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR", "0")
    monkeypatch.setenv("PI0_ACTION_ALLOW_LOG_IDLE_STOP", "1")

    latest_log, steps, wait_info = executor._wait_for_pi0_action_completion(
        str(tmp_path),
        started_at=1000.0,
    )

    assert latest_log == str(log_path)
    assert steps == 2
    assert wait_info["reason"] == "log_idle"
    assert wait_info["elapsed_seconds"] >= 5


def test_pi0_vlm_monitor_stops_completed_action(monkeypatch, tmp_path):
    executor = object.__new__(VisionEnabledArmExecutor)
    executor.verbose = False
    executor.camera_manager = object()
    executor.observation_buffer = None
    executor.vlm_client = type(
        "MonitorClient",
        (),
        {
            "assess_long_action_progress": lambda self, **kwargs: {
                "decision": "complete",
                "task_complete": True,
                "stuck_or_ood": False,
                "confidence": "high",
                "visual_evidence": ["object is in basket"],
                "recommended_stop": True,
                "reason": "goal is complete",
            }
        },
    )()

    log_path = tmp_path / "pi0_actions_test.log"
    log_path.write_text("# header\nstep1\n", encoding="utf-8")

    timeline = {"now": 2000.0, "stopped": False}

    monkeypatch.setattr("executor.arm_executor_vision.time.time", lambda: timeline["now"])

    def fake_sleep(seconds):
        timeline["now"] += seconds

    monkeypatch.setattr("executor.arm_executor_vision.time.sleep", fake_sleep)
    monkeypatch.setenv("PI0_ACTION_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("PI0_ACTION_MIN_WAIT_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_MIN_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_INTERVAL_SECONDS", "1")

    executor._capture_observation_snapshot = lambda **kwargs: {"image_path": "current.jpg"}

    def fake_stop(container_name):
        timeline["stopped"] = True

    executor._stop_pi0_session = fake_stop

    latest_log, steps, wait_info = executor._wait_for_pi0_action_completion(
        str(tmp_path),
        started_at=2000.0,
        instruction="pick up water and place into basket",
        start_image="start.jpg",
    )

    assert latest_log == str(log_path)
    assert steps == 1
    assert wait_info["reason"] == "vlm_complete"
    assert wait_info["monitor"]["decision"] == "complete"
    assert timeline["stopped"] is True


def test_pi0_stops_at_max_useful_time_when_monitor_keeps_continuing(monkeypatch, tmp_path):
    executor = object.__new__(VisionEnabledArmExecutor)
    executor.verbose = False
    executor.camera_manager = object()
    executor.observation_buffer = None
    executor.vlm_client = type(
        "MonitorClient",
        (),
        {
            "assess_long_action_progress": lambda self, **kwargs: {
                "decision": "continue",
                "task_complete": False,
                "stuck_or_ood": False,
                "confidence": "medium",
                "visual_evidence": ["arm is still moving"],
                "recommended_stop": False,
                "reason": "still moving",
            }
        },
    )()

    log_path = tmp_path / "pi0_actions_test.log"
    log_path.write_text("# header\nstep1\n", encoding="utf-8")

    timeline = {"now": 3000.0, "stopped": False}
    monkeypatch.setattr("executor.arm_executor_vision.time.time", lambda: timeline["now"])

    def fake_sleep(seconds):
        timeline["now"] += seconds

    monkeypatch.setattr("executor.arm_executor_vision.time.sleep", fake_sleep)
    monkeypatch.setenv("PI0_ACTION_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("PI0_ACTION_MAX_USEFUL_SECONDS", "5")
    monkeypatch.setenv("PI0_ACTION_MIN_WAIT_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_MIN_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_POLL_SECONDS", "1")

    executor._capture_observation_snapshot = lambda **kwargs: {"image_path": "current.jpg"}

    def fake_stop(container_name):
        timeline["stopped"] = True

    executor._stop_pi0_session = fake_stop

    _, steps, wait_info = executor._wait_for_pi0_action_completion(
        str(tmp_path),
        started_at=3000.0,
        instruction="pick up medicine and place into basket",
        start_image="start.jpg",
    )

    assert steps == 1
    assert wait_info["reason"] == "max_useful_time"
    assert timeline["stopped"] is True


def test_pi0_ignores_early_stuck_decision(monkeypatch, tmp_path):
    executor = object.__new__(VisionEnabledArmExecutor)
    executor.verbose = False
    executor.camera_manager = object()
    executor.observation_buffer = None
    executor.vlm_client = type(
        "MonitorClient",
        (),
        {
            "assess_long_action_progress": lambda self, **kwargs: {
                "decision": "stuck",
                "task_complete": False,
                "stuck_or_ood": True,
                "confidence": "medium",
                "visual_evidence": ["target remains at source"],
                "recommended_stop": True,
                "reason": "too early to trust",
            }
        },
    )()

    log_path = tmp_path / "pi0_actions_test.log"
    log_path.write_text("# header\nstep1\n", encoding="utf-8")

    timeline = {"now": 4000.0, "stopped": False}
    monkeypatch.setattr("executor.arm_executor_vision.time.time", lambda: timeline["now"])
    monkeypatch.setattr("executor.arm_executor_vision.time.sleep", lambda seconds: timeline.update(now=timeline["now"] + seconds))
    monkeypatch.setenv("PI0_ACTION_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("PI0_ACTION_MAX_USEFUL_SECONDS", "8")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_MIN_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("PI0_ACTION_STUCK_MIN_SECONDS", "10")
    monkeypatch.setenv("PI0_ACTION_POLL_SECONDS", "1")

    executor._capture_observation_snapshot = lambda **kwargs: {"image_path": "current.jpg"}
    executor._stop_pi0_session = lambda container_name: timeline.update(stopped=True)

    _, _, wait_info = executor._wait_for_pi0_action_completion(
        str(tmp_path),
        started_at=4000.0,
        instruction="pick up water and place into basket",
        start_image="start.jpg",
    )

    assert wait_info["reason"] == "max_useful_time"
    assert timeline["stopped"] is True


def test_pi0_monitor_receives_dense_process_sequence(monkeypatch):
    executor = object.__new__(VisionEnabledArmExecutor)
    executor.camera_manager = object()
    calls = {"count": 0}

    def fake_snapshot(**kwargs):
        calls["count"] += 1
        return {"image_path": f"frame_{calls['count']}.jpg"}

    monkeypatch.setattr("executor.arm_executor_vision.time.sleep", lambda seconds: None)
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_SEQUENCE_FRAMES", "3")
    monkeypatch.setenv("PI0_ACTION_VLM_MONITOR_SEQUENCE_INTERVAL_SECONDS", "0.1")
    executor._capture_observation_snapshot = fake_snapshot

    frames = executor._capture_monitor_process_sequence(max_frames=4, exclude_paths=set())

    assert [frame["image_path"] for frame in frames] == ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]
    assert [frame["order"] for frame in frames] == [1, 2, 3]
    assert frames[-1]["label"] == "t"
