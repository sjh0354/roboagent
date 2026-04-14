#!/usr/bin/env python3
"""
Template UR grasp executor invoked by anygrasp_runner.py.

Input:
  --request-json '<json>'
  --grasp-json '<json>'

Output:
  Print ONE JSON object on the last stdout line.
"""

import argparse
import json
import sys
from typing import Any, Dict


def _response(success: bool, feedback: str, error: str = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "success": success,
        "feedback": feedback,
        "error": error,
        "data": data or {},
    }


def execute_pick_and_place(request: Dict[str, Any], grasp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace this with your real UR execution logic:
      1) move to pre-grasp pose
      2) descend and close gripper
      3) lift and move to target
      4) open gripper and retreat
    """
    return _response(
        success=False,
        feedback="Template executor called. Implement execute_pick_and_place().",
        error="template_not_implemented",
        data={"request": request, "grasp": grasp},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="UR grasp executor template")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--grasp-json", required=True)
    args = parser.parse_args()

    try:
        request = json.loads(args.request_json)
        grasp = json.loads(args.grasp_json)
    except json.JSONDecodeError as error:
        print(json.dumps(_response(False, "Invalid JSON input.", error=str(error))))
        return 2

    result = execute_pick_and_place(request, grasp)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
