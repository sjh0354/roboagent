#!/usr/bin/env python3
"""
Direct RTDE executor adapter for RAS side.

Purpose:
- Consume request/grasp JSON from AnyGrasp runner.
- Keep existing safety gates strict.
- Execute one servoJ command directly (bypass ROS topic publish timing issues).

Notes:
- motion_only only (no gripper command in this adapter).
- Returns one JSON object as the last stdout line.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _resp(success: bool, feedback: str, error: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "success": success,
        "feedback": feedback,
        "error": error,
        "data": data or {},
    }


def _is_num(x: Any) -> bool:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(y)


def _load_joint_limits() -> List[List[float]]:
    raw = os.getenv("UR_JOINT_LIMITS_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                out = []
                for item in parsed:
                    if isinstance(item, list) and len(item) == 2 and _is_num(item[0]) and _is_num(item[1]):
                        lo = float(item[0])
                        hi = float(item[1])
                        if lo <= hi:
                            out.append([lo, hi])
                if out:
                    return out
        except Exception:
            pass
    return [[-3.14, 3.14] for _ in range(6)]


def _extract_target_joint(request: Dict[str, Any], grasp: Dict[str, Any], fallback: List[float]) -> List[float]:
    candidate = request.get("target_joint")
    if candidate is None:
        policy_action = request.get("policy_action")
        if isinstance(policy_action, dict):
            candidate = policy_action.get("target_joint")
    if candidate is None:
        candidate = grasp.get("target_joint")
    if candidate is None:
        candidate = fallback
    if not isinstance(candidate, list) or len(candidate) < 6:
        return []
    out: List[float] = []
    for idx, v in enumerate(candidate[:6]):
        if not _is_num(v):
            return []
        out.append(float(v))
    return out


def _max_abs_delta(target: List[float], current: List[float]) -> Optional[float]:
    if len(target) < 6 or len(current) < 6:
        return None
    d = 0.0
    for i in range(6):
        d = max(d, abs(target[i] - current[i]))
    return d


def _joint_limit_ok(target: List[float], limits: List[List[float]]) -> bool:
    if len(target) < 6 or len(limits) < 6:
        return False
    for i in range(6):
        lo, hi = limits[i]
        if target[i] < lo or target[i] > hi:
            return False
    return True


def _all_zero(target: List[float], eps: float = 1e-6) -> bool:
    return len(target) >= 6 and all(abs(v) <= eps for v in target[:6])


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct RTDE servoJ executor (RAS adapter)")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--grasp-json", required=True)
    args = parser.parse_args()

    try:
        request = json.loads(args.request_json)
        grasp = json.loads(args.grasp_json)
    except json.JSONDecodeError as e:
        print(json.dumps(_resp(False, "Invalid JSON input.", error=str(e), data={}), ensure_ascii=False))
        return 2

    robot_ip = os.getenv("ROBOT_IP", "192.168.24.245")
    control_rate = float(os.getenv("CONTROL_RATE", "3"))
    max_joint_step = float(os.getenv("MAX_JOINT_STEP", "0.005"))
    ur_enable_motion = str(os.getenv("UR_ENABLE_MOTION", "0")).strip().lower() in {"1", "true", "yes", "on"}

    rtde_control = None
    rtde_receive = None
    try:
        import rtde_control  # type: ignore
        import rtde_receive  # type: ignore
    except Exception as e:
        print(json.dumps(_resp(False, "RTDE python modules unavailable.", error=str(e), data={}), ensure_ascii=False))
        return 1

    # Prefer live robot state as current joint for delta gate.
    current_joint: List[float] = []
    try:
        r_recv = rtde_receive.RTDEReceiveInterface(robot_ip)
        cj = r_recv.getActualQ()
        if isinstance(cj, list) and len(cj) >= 6:
            current_joint = [float(x) for x in cj[:6]]
    except Exception:
        pass
    if not current_joint:
        req_joint = request.get("joint_state")
        if isinstance(req_joint, list) and len(req_joint) >= 6 and all(_is_num(v) for v in req_joint[:6]):
            current_joint = [float(v) for v in req_joint[:6]]

    target_joint = _extract_target_joint(request, grasp, current_joint)
    limits = _load_joint_limits()
    delta_max = _max_abs_delta(target_joint, current_joint)

    safety_gate = {
        "ur_enable_motion": ur_enable_motion,
        "joint_limit_ok": _joint_limit_ok(target_joint, limits),
        "step_ok": max_joint_step <= 0.02,
        "rate_ok": control_rate <= 10.0,
        "zero_target_ok": not _all_zero(target_joint),
        "delta_ok": (delta_max is not None and delta_max <= 0.03),
        "delta_max": delta_max,
        "gripper_blocked": True,
    }

    data = {
        "mode": "motion_only",
        "executor": "ur_rtde_executor",
        "robot_ip": robot_ip,
        "current_joint": current_joint,
        "target_joint": target_joint,
        "safety_gate": safety_gate,
        "motion_plan": {
            "profile": "direct_servoj_single_step",
            "gripper_command": "blocked",
            "execution_stub": False,
            "servoj_params": {"speed": 0.0, "acceleration": 0.0, "time": 0.05, "lookahead": 0.1, "gain": 300},
        },
    }

    if not safety_gate["ur_enable_motion"]:
        print(json.dumps(_resp(False, "Fail-close: UR_ENABLE_MOTION=1 required.", error="motion_disabled", data=data), ensure_ascii=False))
        return 1
    if not safety_gate["step_ok"]:
        print(json.dumps(_resp(False, "Fail-close: max_joint_step exceeds threshold.", error="max_joint_step_exceeded", data=data), ensure_ascii=False))
        return 1
    if not safety_gate["rate_ok"]:
        print(json.dumps(_resp(False, "Fail-close: control_rate exceeds threshold.", error="control_rate_exceeded", data=data), ensure_ascii=False))
        return 1
    if not safety_gate["joint_limit_ok"]:
        print(json.dumps(_resp(False, "Fail-close: target joint out of limits.", error="joint_limit_violation", data=data), ensure_ascii=False))
        return 1
    if not safety_gate["zero_target_ok"]:
        print(json.dumps(_resp(False, "Fail-close: all-zero target blocked.", error="zero_target_joint_blocked", data=data), ensure_ascii=False))
        return 1
    if not safety_gate["delta_ok"]:
        print(json.dumps(_resp(False, "Fail-close: target-current delta exceeds 0.03 rad.", error="joint_delta_exceeded", data=data), ensure_ascii=False))
        return 1

    try:
        # Minimal robustness: ensure controller script channel is ready before servoJ.
        # Retry a few times to absorb transient startup races after owner switch.
        r_ctrl = None
        last_err = None
        for _ in range(3):
            try:
                r_ctrl = rtde_control.RTDEControlInterface(robot_ip)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0)
        if r_ctrl is None:
            raise RuntimeError(f"rtde_control_connect_failed_after_retries: {last_err}")
        print("SERVOJ_EXEC: sending target via RTDE servoJ")
        r_ctrl.servoJ(target_joint, 0.0, 0.0, 0.05, 0.1, 300)
        time.sleep(max(0.2, 1.0 / max(0.1, control_rate)))
    except Exception as e:
        print(json.dumps(_resp(False, "servoJ execution failed.", error=str(e), data=data), ensure_ascii=False))
        return 1

    print(json.dumps(_resp(True, "direct_servo motion_only executed.", error=None, data=data), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
