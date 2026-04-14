#!/usr/bin/env python3
"""
Template AnyGrasp runner for ARM_ACT_BACKEND=anygrasp.

Contract:
  - Input:  --request-json '<json>'
  - Output: print ONE JSON object on the LAST stdout line
            {"success": bool, "feedback": str, "error": str|None, "data": dict}
"""

import argparse
import json
import os
import sys
from typing import Any, Dict


def _build_response(success: bool, feedback: str, error: str = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "success": success,
        "feedback": feedback,
        "error": error,
        "data": data or {},
    }


def run_anygrasp_pick_and_place(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace this function with your real AnyGrasp + robot execution pipeline.

    Expected request keys:
      - action, item_name, source, target, instruction, observation_image, timestamp
    """
    item_name = request.get("item_name", "item")
    source = request.get("source", "shelf")
    target = request.get("target", "counter")
    observation_image = request.get("observation_image", "")

    # Example: verify your runtime path before running the true pipeline.
    anygrasp_root = os.getenv("ANYGRASP_SDK_ROOT", "")
    if anygrasp_root and not os.path.exists(anygrasp_root):
        return _build_response(
            success=False,
            feedback="AnyGrasp root path not found.",
            error=f"invalid_ANYGRASP_SDK_ROOT: {anygrasp_root}",
            data={"request": request},
        )

    # TODO:
    # 1) Load RGB-D/intrinsics for `observation_image`
    # 2) Run AnyGrasp inference to get best grasp pose
    # 3) Convert grasp pose to robot base frame
    # 4) Execute pick-and-place with your UR controller / MoveIt / RTDE
    # 5) Return real execution result + key telemetry
    return _build_response(
        success=False,
        feedback="Template runner called. Implement run_anygrasp_pick_and_place() for real execution.",
        error="template_not_implemented",
        data={
            "item_name": item_name,
            "source": source,
            "target": target,
            "observation_image": observation_image,
            "request": request,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AnyGrasp runner template")
    parser.add_argument("--request-json", required=True, help="Request payload from arm executor")
    args = parser.parse_args()

    try:
        request = json.loads(args.request_json)
    except json.JSONDecodeError as error:
        print(json.dumps(_build_response(False, "Invalid request JSON.", error=str(error))))
        return 2

    response = run_anygrasp_pick_and_place(request)
    print(json.dumps(response, ensure_ascii=False))
    return 0 if response.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
