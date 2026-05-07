#!/usr/bin/env python3
"""
UR policy bridge client (v2 safety gate).

Goals:
- Provide a stable CLI contract for AnyGrasp -> UR policy bridge.
- Default to safe `dry_run` mode.
- v2 only allows restricted `motion_only` when explicit safety gates pass.
- `full` mode defaults fail-close; can be opened for one controlled run with
  UR_ENABLE_FULL_ONCE=1.
- Return one structured JSON object on the last stdout line.
"""

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _response(success: bool, feedback: str, error: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "success": success,
        "feedback": feedback,
        "error": error,
        "data": data or {},
    }


def _looks_like_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _validate_grasp_json(grasp: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    # Minimal finite-number checks to avoid unsafe propagation.
    score = grasp.get("score")
    if score is not None and not _looks_like_number(score):
        return False, "grasp.score is not finite"

    translation = grasp.get("translation")
    if translation is not None:
        if not isinstance(translation, list) or len(translation) != 3:
            return False, "grasp.translation must be a 3-element list"
        for idx, value in enumerate(translation):
            if not _looks_like_number(value):
                return False, f"grasp.translation[{idx}] is not finite"

    rotation = grasp.get("rotation_matrix")
    if rotation is not None:
        if not isinstance(rotation, list) or len(rotation) != 3:
            return False, "grasp.rotation_matrix must be 3x3 list"
        for r, row in enumerate(rotation):
            if not isinstance(row, list) or len(row) != 3:
                return False, f"grasp.rotation_matrix row {r} is invalid"
            for c, value in enumerate(row):
                if not _looks_like_number(value):
                    return False, f"grasp.rotation_matrix[{r}][{c}] is not finite"

    return True, None


def _read_joint_limits() -> List[List[float]]:
    raw = os.getenv("UR_JOINT_LIMITS_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                limits = []
                for item in parsed:
                    if (
                        isinstance(item, list)
                        and len(item) == 2
                        and _looks_like_number(item[0])
                        and _looks_like_number(item[1])
                    ):
                        lo = float(item[0])
                        hi = float(item[1])
                        if lo <= hi:
                            limits.append([lo, hi])
                if limits:
                    return limits
        except Exception:
            pass
    # Conservative default for UR joints when explicit limits are not provided.
    return [[-3.14, 3.14] for _ in range(6)]


def _extract_target_joint_info(
    request: Dict[str, Any],
    grasp: Dict[str, Any],
    fallback_joint_state: List[Any],
) -> Tuple[List[float], str, Any]:
    source = "none"
    candidate: Any = None

    # Priority required by workflow: policy actions -> request.target_joint.
    policy_action = request.get("policy_action")
    if isinstance(policy_action, dict):
        if policy_action.get("target_joint") is not None:
            candidate = policy_action.get("target_joint")
            source = "policy.target_joint"
        elif isinstance(policy_action.get("action"), list) and len(policy_action.get("action")) >= 6:
            candidate = policy_action.get("action")
            source = "policy.action[:6]"
    if candidate is None and isinstance(request.get("policy_actions"), list):
        policy_actions = request.get("policy_actions")
        if policy_actions and isinstance(policy_actions[0], dict):
            pa0 = policy_actions[0]
            if pa0.get("target_joint") is not None:
                candidate = pa0.get("target_joint")
                source = "policy_actions[0].target_joint"
            elif isinstance(pa0.get("action"), list) and len(pa0.get("action")) >= 6:
                candidate = pa0.get("action")
                source = "policy_actions[0].action[:6]"

    if candidate is None and request.get("target_joint") is not None:
        candidate = request.get("target_joint")
        source = "request"
    if candidate is None and grasp.get("target_joint") is not None:
        candidate = grasp.get("target_joint")
        source = "grasp"
    if candidate is None and isinstance(fallback_joint_state, list) and fallback_joint_state:
        candidate = fallback_joint_state
        source = "current_joint_fallback"

    raw_candidate = candidate
    if not isinstance(candidate, list) or not candidate:
        return [], source, raw_candidate
    result: List[float] = []
    for value in candidate:
        if not _looks_like_number(value):
            return [], source, raw_candidate
        result.append(float(value))
    return result, source, raw_candidate


def _recover_current_joint_from_state_bridge(request: Dict[str, Any]) -> Tuple[List[float], Optional[str]]:
    robot_ip = str(os.getenv("UR_ROBOT_IP", "192.168.24.245")).strip() or "192.168.24.245"
    arm_name = str(os.getenv("UR_ARM_NAME", "arm_1")).strip() or "arm_1"
    timeout_sec = float(os.getenv("UR_STATE_BRIDGE_TIMEOUT", "15"))
    python_bin = shutil.which("python3") or shutil.which("python")
    if not python_bin:
        return [], "state_bridge_python_not_found"
    local_bridge_script = "/home/ef/projects/RAS_interactivate_planner/utils/ur_state_bridge_client.py"
    local_bridge_cmd = (
        f"{shlex.quote(python_bin)} {shlex.quote(local_bridge_script)} "
        f"--robot-ip {shlex.quote(robot_ip)} --arm-name {shlex.quote(arm_name)} --timeout-sec 3"
    )
    bridge_cmd = os.getenv(
        "UR_STATE_BRIDGE_CMD",
        local_bridge_cmd,
    ).strip()
    if not bridge_cmd:
        return [], "state_bridge_cmd_empty"

    def _run_cmd(cmd: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

    try:
        completed = _run_cmd(bridge_cmd)
    except Exception as error:
        return [], f"state_bridge_run_exception: {error}"
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        # If bridge is executed inside a container, docker-wrapper based state
        # command fails by design. Fall back to local RTDE state bridge client.
        docker_missing = "docker: command not found" in stderr.lower()
        wrapper_selected = "ur_state_bridge_container_wrapper.sh" in bridge_cmd or bridge_cmd.strip().startswith("docker ")
        if docker_missing and wrapper_selected:
            try:
                completed = _run_cmd(local_bridge_cmd)
            except Exception as error:
                return [], f"state_bridge_local_fallback_exception: {error}"
            if completed.returncode != 0:
                fallback_stderr = (completed.stderr or "").strip()
                return [], f"state_bridge_local_fallback_nonzero_exit: {fallback_stderr or completed.returncode}"
        else:
            return [], f"state_bridge_nonzero_exit: {stderr or completed.returncode}"

    parsed = None
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if lines:
        try:
            parsed = json.loads(lines[-1])
        except Exception:
            parsed = None
    if not isinstance(parsed, dict) or not parsed.get("success"):
        return [], "state_bridge_unsuccessful"
    joint_state = (((parsed.get("data") or {}).get("observation") or {}).get("joint_state"))
    if not isinstance(joint_state, list) or len(joint_state) < 6:
        return [], "state_bridge_joint_state_invalid"
    result: List[float] = []
    for value in joint_state[:6]:
        if not _looks_like_number(value):
            return [], "state_bridge_joint_state_non_numeric"
        result.append(float(value))
    request["joint_state"] = result
    return result, None


def _target_joint_complete(target_joint: List[float], expected_dim: int = 6) -> bool:
    if not isinstance(target_joint, list) or len(target_joint) != expected_dim:
        return False
    for value in target_joint:
        if not _looks_like_number(value):
            return False
    return True


def _joint_limit_ok(target_joint: List[float], limits: List[List[float]]) -> bool:
    if not target_joint:
        return False
    for index, value in enumerate(target_joint):
        if index >= len(limits):
            return False
        lo, hi = limits[index]
        if value < lo or value > hi:
            return False
    return True


def _is_all_zero_target(target_joint: List[float], eps: float = 1e-6) -> bool:
    if not target_joint:
        return False
    return all(abs(value) <= eps for value in target_joint)


def _max_abs_joint_delta(target_joint: List[float], current_joint: List[float]) -> Optional[float]:
    if not target_joint or not current_joint:
        return None
    if len(target_joint) > len(current_joint):
        return None
    delta = 0.0
    for idx, value in enumerate(target_joint):
        delta = max(delta, abs(value - current_joint[idx]))
    return delta


def _extract_observation(request: Dict[str, Any]) -> Dict[str, Any]:
    observation = request.get("observation")
    if isinstance(observation, dict):
        obs = dict(observation)
    else:
        obs = {}

    # Backward-compatible aliases.
    if "rgb_path" not in obs:
        obs["rgb_path"] = request.get("observation_image") or request.get("rgb_path")
    if "depth_path" not in obs:
        obs["depth_path"] = request.get("depth_path")
    if "intrinsics" not in obs:
        obs["intrinsics"] = request.get("intrinsics")
    if "joint_state" not in obs:
        obs["joint_state"] = request.get("joint_state")
    if "ee_pose" not in obs:
        obs["ee_pose"] = request.get("ee_pose")
    if "timestamp" not in obs:
        obs["timestamp"] = request.get("timestamp")
    return obs


def _joint_names() -> List[str]:
    raw = os.getenv("UR_JOINT_NAMES", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                names = [str(x) for x in parsed]
                if len(names) == 6:
                    return names
        except Exception:
            pass
    return [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]


def _publish_motion_only_trajectory(target_joint: List[float], command_topic: str, control_rate: float) -> Dict[str, Any]:
    started_at = time.time()
    ros2_bin = shutil.which("ros2")
    env_timeout_raw = os.getenv("UR_ROS2_PUB_TIMEOUT_SEC", os.getenv("UR_MOTION_PUBLISH_TIMEOUT_SEC", "3.0"))
    try:
        timeout_sec = float(env_timeout_raw)
    except (TypeError, ValueError):
        timeout_sec = 3.0
    if timeout_sec <= 0:
        timeout_sec = 3.0

    if not ros2_bin:
        ended_at = time.time()
        return {
            "command": ["ros2", "topic", "pub", "--once", command_topic, "trajectory_msgs/msg/JointTrajectory", "<payload>"],
            "returncode": 127,
            "stdout": "",
            "stderr": "ros2_not_found: ros2 CLI is not available in current execution environment",
            "duration_sec": 0.0,
            "timeout_sec_used": timeout_sec,
            "env_timeout_raw": env_timeout_raw,
            "started_at": started_at,
            "ended_at": ended_at,
        }

    duration_sec = max(0.5, min(5.0, 1.0 / max(0.1, control_rate)))
    payload = {
        "joint_names": _joint_names(),
        "points": [
            {
                "positions": target_joint,
                "time_from_start": {"sec": int(duration_sec), "nanosec": int((duration_sec % 1.0) * 1e9)},
            }
        ],
    }
    helper_script = os.getenv(
        "UR_ROS2_PUBLISH_HELPER",
        "/home/ef/projects/RAS_interactivate_planner/utils/ros2_publish_once.py",
    )
    python_bin = shutil.which("python3") or shutil.which("python")
    if not python_bin:
        ended_at = time.time()
        return {
            "command": ["python3", helper_script],
            "returncode": 127,
            "stdout": "",
            "stderr": "python_not_found: python3 is not available in current execution environment",
            "duration_sec": duration_sec,
            "timeout_sec_used": timeout_sec,
            "env_timeout_raw": env_timeout_raw,
            "started_at": started_at,
            "ended_at": ended_at,
        }
    cmd = [
        python_bin,
        helper_script,
        "--topic",
        command_topic,
        "--payload-json",
        json.dumps(payload, ensure_ascii=False),
        "--timeout-sec",
        str(timeout_sec),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)
    except FileNotFoundError as error:
        ended_at = time.time()
        return {
            "command": cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": str(error),
            "duration_sec": duration_sec,
            "timeout_sec_used": timeout_sec,
            "env_timeout_raw": env_timeout_raw,
            "started_at": started_at,
            "ended_at": ended_at,
        }
    except subprocess.TimeoutExpired as error:
        ended_at = time.time()
        return {
            "command": cmd,
            "returncode": 124,
            "stdout": (error.stdout or "").strip() if isinstance(error.stdout, str) else "",
            "stderr": (error.stderr or "").strip() if isinstance(error.stderr, str) else "ros2 topic pub timeout",
            "duration_sec": duration_sec,
            "timeout_sec_used": timeout_sec,
            "env_timeout_raw": env_timeout_raw,
            "started_at": started_at,
            "ended_at": ended_at,
        }
    ended_at = time.time()
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
        "duration_sec": duration_sec,
        "timeout_sec_used": timeout_sec,
        "env_timeout_raw": env_timeout_raw,
        "started_at": started_at,
        "ended_at": ended_at,
    }


def _publish_gripper_close(gripper_topic: str) -> Dict[str, Any]:
    started_at = time.time()
    env_timeout_raw = os.getenv("UR_ROS2_PUB_TIMEOUT_SEC", os.getenv("UR_MOTION_PUBLISH_TIMEOUT_SEC", "3.0"))
    try:
        timeout_sec = float(env_timeout_raw)
    except (TypeError, ValueError):
        timeout_sec = 3.0
    if timeout_sec <= 0:
        timeout_sec = 3.0

    helper_script = os.getenv(
        "UR_ROS2_GRIPPER_PUBLISH_HELPER",
        "/home/ef/projects/RAS_interactivate_planner/utils/ros2_publish_gripper_once.py",
    )
    python_bin = shutil.which("python3") or shutil.which("python")
    if not python_bin:
        ended_at = time.time()
        return {
            "command": ["python3", helper_script],
            "returncode": 127,
            "stdout": "",
            "stderr": "python_not_found: python3 is not available in current execution environment",
            "duration_sec": 0.0,
            "timeout_sec_used": timeout_sec,
            "env_timeout_raw": env_timeout_raw,
            "started_at": started_at,
            "ended_at": ended_at,
        }

    close_payload = {"data": [255, 200, 85]}
    cmd = [
        python_bin,
        helper_script,
        "--topic",
        gripper_topic,
        "--payload-json",
        json.dumps(close_payload, ensure_ascii=False),
        "--timeout-sec",
        str(timeout_sec),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)
    except FileNotFoundError as error:
        ended_at = time.time()
        return {
            "command": cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": str(error),
            "duration_sec": 0.0,
            "timeout_sec_used": timeout_sec,
            "env_timeout_raw": env_timeout_raw,
            "started_at": started_at,
            "ended_at": ended_at,
        }
    except subprocess.TimeoutExpired as error:
        ended_at = time.time()
        return {
            "command": cmd,
            "returncode": 124,
            "stdout": (error.stdout or "").strip() if isinstance(error.stdout, str) else "",
            "stderr": (error.stderr or "").strip() if isinstance(error.stderr, str) else "ros2 gripper pub timeout",
            "duration_sec": 0.0,
            "timeout_sec_used": timeout_sec,
            "env_timeout_raw": env_timeout_raw,
            "started_at": started_at,
            "ended_at": ended_at,
        }

    ended_at = time.time()
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
        "duration_sec": 0.0,
        "timeout_sec_used": timeout_sec,
        "env_timeout_raw": env_timeout_raw,
        "started_at": started_at,
        "ended_at": ended_at,
    }


def run_bridge(
    request: Dict[str, Any],
    grasp: Dict[str, Any],
    mode: str,
    policy_host: str,
    policy_port: int,
    action_chunk_size: int,
    control_rate: float,
    max_joint_step: float,
) -> Dict[str, Any]:
    is_valid, validation_error = _validate_grasp_json(grasp)
    if not is_valid:
        return _response(
            success=False,
            feedback="Bridge input validation failed.",
            error=validation_error,
            data={"mode": mode},
        )

    observation = _extract_observation(request)
    final_joint_state = observation.get("joint_state")
    if not isinstance(final_joint_state, list):
        final_joint_state = []
    if len(final_joint_state) < 6:
        recovered_joint, recover_error = _recover_current_joint_from_state_bridge(request)
        if recovered_joint:
            observation = _extract_observation(request)
            final_joint_state = observation.get("joint_state") or recovered_joint
        else:
            result = _response(
                success=False,
                feedback="Fail-close: current_joint unavailable and state bridge recovery failed.",
                error="current_joint_unavailable",
                data={
                    "mode": mode,
                    "current_joint": final_joint_state,
                    "state_bridge_error": recover_error,
                },
            )
            print(json.dumps(result, ensure_ascii=False))
            return result

    used_topics = {
        "joint_state_topic": os.getenv("UR_JOINT_STATE_TOPIC", "/arm_1/arm_joint_states"),
        "command_topic": os.getenv("UR_COMMAND_TOPIC", "/arm_1/command/joint_trajectory"),
        "gripper_command_topic": os.getenv("UR_GRIPPER_COMMAND_TOPIC", "/arm_1/command/gripper_command"),
        "base_camera_topic": os.getenv("UR_BASE_CAMERA_TOPIC", "/front_camera/front_camera/color/image_raw"),
        "wrist_camera_topic": os.getenv("UR_WRIST_CAMERA_TOPIC", "/left_camera/left_camera/image_raw"),
    }

    joint_limits = _read_joint_limits()
    target_joint, target_joint_source, raw_target_joint = _extract_target_joint_info(request, grasp, final_joint_state)
    target_joint_complete = _target_joint_complete(target_joint, expected_dim=6)

    ur_enable_motion = str(os.getenv("UR_ENABLE_MOTION", "0")).strip().lower() in {"1", "true", "yes", "on"}
    allow_full_once = str(os.getenv("UR_ENABLE_FULL_ONCE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    step_ok = max_joint_step <= 0.02
    rate_ok = control_rate <= 10.0
    joint_limit_ok = _joint_limit_ok(target_joint, joint_limits)
    zero_target_ok = not _is_all_zero_target(target_joint)
    delta_max = _max_abs_joint_delta(target_joint, final_joint_state)
    delta_ok = delta_max is not None and delta_max <= 0.03
    safety_gate = {
        "ur_enable_motion": ur_enable_motion,
        "joint_limit_ok": joint_limit_ok,
        "step_ok": step_ok,
        "rate_ok": rate_ok,
        "zero_target_ok": zero_target_ok,
        "delta_ok": delta_ok,
        "delta_max": delta_max,
        "gripper_blocked": mode != "full",
    }

    data = {
        "mode": mode,
        "policy_steps": action_chunk_size,
        "final_joint_state": final_joint_state,
        "target_joint": target_joint,
        "raw_target_joint": raw_target_joint,
        "target_joint_source": target_joint_source,
        "current_joint": final_joint_state,
        "target_joint_complete": target_joint_complete,
        "joint_limits": joint_limits,
        "used_topics": used_topics,
        "used_policy_host_port": {"host": policy_host, "port": policy_port},
        "bridge_params": {
            "control_rate": control_rate,
            "max_joint_step": max_joint_step,
        },
        "observation_paths": {
            "rgb_path": observation.get("rgb_path"),
            "depth_path": observation.get("depth_path"),
        },
        "grasp_summary": {
            "score": grasp.get("score"),
            "translation": grasp.get("translation"),
        },
        "safety_gate": safety_gate,
    }

    if mode == "full":
        if not allow_full_once:
            print("SAFETY_GATE: full mode blocked (fail-close).")
            return _response(
                success=False,
                feedback="Fail-close: full mode is disabled in bridge v2 unless UR_ENABLE_FULL_ONCE=1.",
                error="full_mode_blocked",
                data=data,
            )
        if not ur_enable_motion:
            print("SAFETY_GATE: full blocked because UR_ENABLE_MOTION != 1.")
            return _response(
                success=False,
                feedback="Fail-close: full requires UR_ENABLE_MOTION=1.",
                error="motion_disabled",
                data=data,
            )
        if not step_ok:
            print("SAFETY_GATE: full blocked by max_joint_step > 0.02.")
            return _response(
                success=False,
                feedback="Fail-close: max_joint_step exceeds v2 safety threshold.",
                error="max_joint_step_exceeded",
                data=data,
            )
        if not rate_ok:
            print("SAFETY_GATE: full blocked by control_rate > 10.")
            return _response(
                success=False,
                feedback="Fail-close: control_rate exceeds v2 safety threshold.",
                error="control_rate_exceeded",
                data=data,
            )
        if not target_joint_complete:
            print("SAFETY_GATE: full blocked by target_joint completeness check.")
            return _response(
                success=False,
                feedback="Fail-close: full requires target_joint as a 6D numeric vector.",
                error="target_joint_missing_or_invalid",
                data=data,
            )
        if not joint_limit_ok:
            print("SAFETY_GATE: full blocked by joint limit check.")
            return _response(
                success=False,
                feedback="Fail-close: target_joint missing or out of conservative joint limits.",
                error="joint_limit_violation",
                data=data,
            )
        if not zero_target_ok:
            print("SAFETY_GATE: full blocked by all-zero target_joint.")
            return _response(
                success=False,
                feedback="Fail-close: all-zero target_joint is blocked by v2 safety policy.",
                error="zero_target_joint_blocked",
                data=data,
            )
        if not delta_ok:
            print("SAFETY_GATE: full blocked by excessive target-current joint delta.")
            return _response(
                success=False,
                feedback="Fail-close: max(abs(target-current)) exceeds 0.03 rad.",
                error="joint_delta_exceeded",
                data=data,
            )

        print("SAFETY_GATE: full gate passed (single-run unlock).")
        publish_result = _publish_motion_only_trajectory(
            target_joint=target_joint,
            command_topic=used_topics["command_topic"],
            control_rate=control_rate,
        )
        if publish_result["returncode"] != 0:
            print("SAFETY_GATE: full publish failed; fail-close.")
            data["motion_plan"] = {
                "profile": "full_single_pick_trial",
                "gripper_command": "close",
                "execution_stub": False,
                "publish_result": publish_result,
            }
            return _response(
                success=False,
                feedback="full gate passed but trajectory publish failed (fail-close).",
                error="motion_publish_failed",
                data=data,
            )
        gripper_result = _publish_gripper_close(used_topics["gripper_command_topic"])
        if gripper_result["returncode"] != 0:
            print("SAFETY_GATE: full gripper publish failed; fail-close.")
            data["motion_plan"] = {
                "profile": "full_single_pick_trial",
                "gripper_command": "close",
                "execution_stub": False,
                "publish_result": publish_result,
                "gripper_publish_result": gripper_result,
            }
            return _response(
                success=False,
                feedback="full gate passed but gripper close publish failed (fail-close).",
                error="gripper_publish_failed",
                data=data,
            )
        data["motion_plan"] = {
            "profile": "full_single_pick_trial",
            "gripper_command": "close",
            "execution_stub": False,
            "publish_result": publish_result,
            "gripper_publish_result": gripper_result,
        }
        return _response(
            success=True,
            feedback="full single-run gate passed: trajectory and gripper close commands published.",
            error=None,
            data=data,
        )

    if mode == "motion_only":
        if not target_joint_complete:
            print("SAFETY_GATE: motion_only blocked by target_joint completeness check.")
            return _response(
                success=False,
                feedback="Fail-close: motion_only requires target_joint as a 6D numeric vector.",
                error="target_joint_missing_or_invalid",
                data=data,
            )
        if not ur_enable_motion:
            print("SAFETY_GATE: motion_only blocked because UR_ENABLE_MOTION != 1.")
            return _response(
                success=False,
                feedback="Fail-close: motion_only requires UR_ENABLE_MOTION=1.",
                error="motion_disabled",
                data=data,
            )
        if not step_ok:
            print("SAFETY_GATE: motion_only blocked by max_joint_step > 0.02.")
            return _response(
                success=False,
                feedback="Fail-close: max_joint_step exceeds v2 safety threshold.",
                error="max_joint_step_exceeded",
                data=data,
            )
        if not rate_ok:
            print("SAFETY_GATE: motion_only blocked by control_rate > 10.")
            return _response(
                success=False,
                feedback="Fail-close: control_rate exceeds v2 safety threshold.",
                error="control_rate_exceeded",
                data=data,
            )
        if not joint_limit_ok:
            print("SAFETY_GATE: motion_only blocked by joint limit check.")
            return _response(
                success=False,
                feedback="Fail-close: target_joint missing or out of conservative joint limits.",
                error="joint_limit_violation",
                data=data,
            )
        if not zero_target_ok:
            print("SAFETY_GATE: motion_only blocked by all-zero target_joint.")
            return _response(
                success=False,
                feedback="Fail-close: all-zero target_joint is blocked by v2 safety policy.",
                error="zero_target_joint_blocked",
                data=data,
            )
        if not delta_ok:
            print("SAFETY_GATE: motion_only blocked by excessive target-current joint delta.")
            return _response(
                success=False,
                feedback="Fail-close: max(abs(target-current)) exceeds 0.03 rad.",
                error="joint_delta_exceeded",
                data=data,
            )

        print("SAFETY_GATE: motion_only gate passed.")
        print("SAFETY_GATE: pre-grasp high-point trajectory only; gripper close blocked.")
        publish_result = _publish_motion_only_trajectory(
            target_joint=target_joint,
            command_topic=used_topics["command_topic"],
            control_rate=control_rate,
        )
        if publish_result["returncode"] != 0:
            print("SAFETY_GATE: motion_only publish failed; fail-close.")
            data["motion_plan"] = {
                "profile": "pre_grasp_high_point_only",
                "gripper_command": "blocked",
                "execution_stub": False,
                "publish_result": publish_result,
            }
            return _response(
                success=False,
                feedback="motion_only gate passed but trajectory publish failed (fail-close).",
                error="motion_publish_failed",
                data=data,
            )
        data["motion_plan"] = {
            "profile": "pre_grasp_high_point_only",
            "gripper_command": "blocked",
            "execution_stub": False,
            "publish_result": publish_result,
        }
        return _response(
            success=True,
            feedback="motion_only gate passed: restricted pre-grasp-high-point trajectory published; gripper close blocked.",
            error=None,
            data=data,
        )

    return _response(
        success=True,
        feedback="dry_run succeeded: bridge input/output contract validated, no robot motion executed.",
        error=None,
        data=data,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="UR policy bridge client (v2 safety gate)")
    parser.add_argument("--request-json", required=True, help="Task context JSON")
    parser.add_argument("--grasp-json", required=True, help="AnyGrasp best_grasp JSON")
    parser.add_argument("--mode", choices=["dry_run", "motion_only", "full"], default="dry_run")
    parser.add_argument("--policy-host", default=os.getenv("POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--policy-port", type=int, default=int(os.getenv("POLICY_PORT", "8001")))
    parser.add_argument("--action-chunk-size", type=int, default=int(os.getenv("ACTION_CHUNK_SIZE", "5")))
    parser.add_argument("--control-rate", type=float, default=float(os.getenv("CONTROL_RATE", "20.0")))
    parser.add_argument("--max-joint-step", type=float, default=float(os.getenv("MAX_JOINT_STEP", "0.03")))
    args = parser.parse_args()

    try:
        request = json.loads(args.request_json)
        grasp = json.loads(args.grasp_json)
    except json.JSONDecodeError as error:
        result = _response(False, "Invalid JSON input.", error=str(error), data={"mode": args.mode})
        print(json.dumps(result, ensure_ascii=False))
        return 2

    result = run_bridge(
        request=request,
        grasp=grasp,
        mode=args.mode,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        action_chunk_size=args.action_chunk_size,
        control_rate=args.control_rate,
        max_joint_step=args.max_joint_step,
    )

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
