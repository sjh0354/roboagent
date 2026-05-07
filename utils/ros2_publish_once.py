#!/usr/bin/env python3
"""
Publish one JointTrajectory message with rclpy and exit.
"""

import argparse
import json
import sys
import time


def _build_msg(payload):
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    msg = JointTrajectory()
    msg.joint_names = [str(x) for x in payload.get("joint_names", [])]
    points = payload.get("points", [])
    for item in points:
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in item.get("positions", [])]
        pt.velocities = [float(x) for x in item.get("velocities", [])]
        pt.accelerations = [float(x) for x in item.get("accelerations", [])]
        pt.effort = [float(x) for x in item.get("effort", [])]
        tfs = item.get("time_from_start", {}) or {}
        sec = int(tfs.get("sec", 0))
        nanosec = int(tfs.get("nanosec", 0))
        pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points.append(pt)
    return msg


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS2 one-shot publisher")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--payload-json", required=True)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    args = parser.parse_args()

    try:
        payload = json.loads(args.payload_json)
    except json.JSONDecodeError as error:
        print(f"invalid_payload_json: {error}", file=sys.stderr)
        return 2

    try:
        import rclpy
    except Exception as error:
        print(f"rclpy_import_failed: {error}", file=sys.stderr)
        return 3

    timeout_sec = max(0.1, float(args.timeout_sec))
    started = time.time()
    rclpy.init(args=None)
    node = rclpy.create_node("slock_publish_once")
    try:
        publisher = node.create_publisher(
            __import__("trajectory_msgs.msg", fromlist=["JointTrajectory"]).JointTrajectory,
            args.topic,
            10,
        )
        msg = _build_msg(payload)

        # Wait briefly for at least one subscriber.
        deadline = started + timeout_sec
        while time.time() < deadline and publisher.get_subscription_count() < 1:
            rclpy.spin_once(node, timeout_sec=0.05)

        if publisher.get_subscription_count() < 1:
            print("no_subscriber_for_topic", file=sys.stderr)
            return 124

        # Publish a few times over a short window to avoid one-shot drop.
        for _ in range(3):
            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
        print("published_once")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())

