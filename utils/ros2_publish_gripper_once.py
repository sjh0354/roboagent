#!/usr/bin/env python3
"""
Publish one Int32MultiArray gripper command with rclpy and exit.
Expected payload: {"data":[position,speed,force]}.
"""

import argparse
import json
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS2 one-shot gripper publisher")
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
        data = payload.get("data", [])
        if not isinstance(data, list) or len(data) < 3:
            print("invalid_payload: expected data=[position,speed,force]", file=sys.stderr)
            return 2
        values = [int(data[0]), int(data[1]), int(data[2])]
    except Exception as error:
        print(f"invalid_payload_values: {error}", file=sys.stderr)
        return 2

    try:
        import rclpy
        from std_msgs.msg import Int32MultiArray
    except Exception as error:
        print(f"rclpy_import_failed: {error}", file=sys.stderr)
        return 3

    timeout_sec = max(0.1, float(args.timeout_sec))
    started = time.time()
    rclpy.init(args=None)
    node = rclpy.create_node("slock_gripper_publish_once")
    try:
        publisher = node.create_publisher(Int32MultiArray, args.topic, 10)
        msg = Int32MultiArray()
        msg.data = values

        deadline = started + timeout_sec
        while time.time() < deadline and publisher.get_subscription_count() < 1:
            rclpy.spin_once(node, timeout_sec=0.05)

        if publisher.get_subscription_count() < 1:
            print("no_subscriber_for_topic", file=sys.stderr)
            return 124

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
