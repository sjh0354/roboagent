#!/usr/bin/env python3
"""
UR state bridge client (RAS-side adapter).

Design goal:
- Do not modify upstream ur5e-arm-teleoperation repository.
- Reuse the same arm topic conventions in returned payload.
- Read state directly from RTDE receive interface (no motion publish).
"""

import argparse
import json
import math
import sys
import time
from typing import Any, Dict, Optional


def _response(success: bool, feedback: str, error: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "success": success,
        "feedback": feedback,
        "error": error,
        "data": data or {},
    }


def _to_float_list(values: Any) -> list[float]:
    result: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite value")
        result.append(number)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Read UR state via RTDE receive and emit one-line JSON.")
    parser.add_argument("--robot-ip", default="192.168.24.245")
    parser.add_argument("--arm-name", default="arm_1")
    parser.add_argument("--joint-state-topic", default=None, help="Compatibility arg; topic convention is returned in JSON.")
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    args = parser.parse_args()

    started_at = time.time()

    try:
        import rtde_receive  # type: ignore
    except Exception as error:
        print(json.dumps(_response(False, "rtde_receive unavailable", f"rtde_import_error: {error}"), ensure_ascii=False))
        return 127

    try:
        receiver = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    except Exception as error:
        print(
            json.dumps(
                _response(
                    False,
                    "rtde receive connect failed",
                    f"rtde_connect_error: {error}",
                    data={"robot_ip": args.robot_ip},
                ),
                ensure_ascii=False,
            )
        )
        return 1

    deadline = time.time() + max(0.5, args.timeout_sec)
    last_error: Optional[str] = None
    joint_state: Optional[list[float]] = None
    joint_velocity: Optional[list[float]] = None
    joint_effort: Optional[list[float]] = None
    tcp_pose: Optional[list[float]] = None
    while time.time() < deadline:
        try:
            joint_state = _to_float_list(receiver.getActualQ())
            joint_velocity = _to_float_list(receiver.getActualQd())
            joint_effort = _to_float_list(receiver.getActualCurrent())
            tcp_pose = _to_float_list(receiver.getActualTCPPose())
            if len(joint_state) >= 6:
                break
        except Exception as error:
            last_error = str(error)
            time.sleep(0.1)

    ended_at = time.time()
    if not joint_state or len(joint_state) < 6:
        print(
            json.dumps(
                _response(
                    False,
                    "rtde state read timeout",
                    f"rtde_state_timeout: {last_error or 'no samples'}",
                    data={"robot_ip": args.robot_ip, "duration_sec": round(ended_at - started_at, 3)},
                ),
                ensure_ascii=False,
            )
        )
        return 1

    result = _response(
        True,
        "state read ok",
        data={
            "robot_ip": args.robot_ip,
            "observation": {
                "joint_state": joint_state[:6],
                "joint_velocity": (joint_velocity or [])[:6],
                "joint_effort": (joint_effort or [])[:6],
                "ee_pose": tcp_pose or [],
                "timestamp": time.time(),
            },
            "topics_convention": {
                "joint_state_topic": f"/{args.arm_name}/arm_joint_states",
                "ee_pose_topic": f"/{args.arm_name}/ee_pose",
                "command_topic": f"/{args.arm_name}/command/joint_trajectory",
                "gripper_topic": f"/{args.arm_name}/command/gripper_command",
            },
            "meta": {
                "source": "rtde_receive",
                "duration_sec": round(ended_at - started_at, 3),
            },
        },
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
