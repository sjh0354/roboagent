#!/usr/bin/env python3
"""
AnyGrasp backend runner for ARM_ACT_BACKEND=anygrasp.

Input:
  --request-json '<json>'

Output:
  Print ONE JSON object on the last stdout line:
  {
    "success": bool,
    "feedback": str,
    "error": str | null,
    "data": {...}
  }

Environment:
  ANYGRASP_SDK_ROOT: path to anygrasp_sdk (for imports and optional default assets)
  ANYGRASP_CHECKPOINT_PATH: AnyGrasp checkpoint path (required)
  ANYGRASP_COLOR_PATH / ANYGRASP_DEPTH_PATH: fallback image paths when request lacks them
  ANYGRASP_INTRINSICS_JSON: e.g. {"fx":927.17,"fy":927.37,"cx":651.32,"cy":349.62,"scale":1000.0}
  ANYGRASP_LIMS_JSON: optional workspace limits [xmin,xmax,ymin,ymax,zmin,zmax]
  ANYGRASP_COLLISION_DETECTION: 1/0, default 1
  ANYGRASP_DENSE_GRASP: 1/0, default 0
  ANYGRASP_APPLY_OBJECT_MASK: 1/0, default 1
  ANYGRASP_TOP_K: number of top grasps returned, default 20
  UR_GRASP_EXECUTOR_CMD: optional command for robot execution
  ANYGRASP_REQUIRE_UR_EXECUTION: 1/0, default 1
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _json_env(key: str, default: Any) -> Any:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _response(success: bool, feedback: str, error: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
    return {
        "success": success,
        "feedback": feedback,
        "error": error,
        "data": data or {},
    }


def _load_depth(depth_path: str) -> np.ndarray:
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path)
    else:
        depth = np.array(Image.open(depth_path))
    if depth.ndim != 2:
        raise ValueError(f"Depth must be single-channel, got shape={depth.shape}")
    return depth


def _load_rgb(rgb_path: str) -> np.ndarray:
    color = np.array(Image.open(rgb_path), dtype=np.float32)
    if color.ndim == 2:
        color = np.stack([color, color, color], axis=-1)
    if color.shape[-1] > 3:
        color = color[..., :3]
    return color / 255.0


def _build_points_and_colors(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    scale = float(intrinsics.get("scale", 1000.0))
    z_min = float(intrinsics.get("z_min", 0.0))
    z_max = float(intrinsics.get("z_max", 1.5))

    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB and depth resolution mismatch: rgb={rgb.shape[:2]}, depth={depth.shape[:2]}")

    xmap, ymap = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    points_z = depth.astype(np.float32) / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > z_min) & (points_z < z_max)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    colors = rgb[mask].astype(np.float32)
    if points.size == 0:
        raise ValueError("No valid points after depth filtering.")
    return points, colors


def _to_serializable(value: Any):
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return str(value)


def _grasp_to_dict(grasp: Any) -> Dict[str, Any]:
    fields = {
        "score": getattr(grasp, "score", None),
        "width": getattr(grasp, "width", None),
        "height": getattr(grasp, "height", None),
        "depth": getattr(grasp, "depth", None),
        "translation": getattr(grasp, "translation", None),
        "rotation_matrix": getattr(grasp, "rotation_matrix", None),
    }
    return {k: _to_serializable(v) for k, v in fields.items()}


def _import_anygrasp_modules(sdk_root: str):
    if sdk_root and sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)
        detection_dir = os.path.join(sdk_root, "grasp_detection")
        if os.path.isdir(detection_dir):
            sys.path.insert(0, detection_dir)

    from gsnet import AnyGrasp  # type: ignore

    return AnyGrasp


def _pick_best_grasp(
    request: Dict[str, Any],
    intrinsics: Dict[str, float],
    lims: List[float],
) -> Dict[str, Any]:
    sdk_root = os.getenv("ANYGRASP_SDK_ROOT", "").strip()
    AnyGrasp = _import_anygrasp_modules(sdk_root)

    checkpoint_path = os.getenv("ANYGRASP_CHECKPOINT_PATH", "").strip()
    if not checkpoint_path:
        raise RuntimeError("ANYGRASP_CHECKPOINT_PATH is required.")

    rgb_path = request.get("rgb_path") or request.get("observation_image") or os.getenv("ANYGRASP_COLOR_PATH", "")
    depth_path = request.get("depth_path") or os.getenv("ANYGRASP_DEPTH_PATH", "")
    if not rgb_path:
        raise RuntimeError("No RGB path provided. Need request.rgb_path/observation_image or ANYGRASP_COLOR_PATH.")
    if not depth_path:
        raise RuntimeError("No depth path provided. Need request.depth_path or ANYGRASP_DEPTH_PATH.")

    rgb = _load_rgb(rgb_path)
    depth = _load_depth(depth_path)
    points, colors = _build_points_and_colors(rgb, depth, intrinsics)

    cfgs = SimpleNamespace(
        checkpoint_path=checkpoint_path,
        max_gripper_width=float(os.getenv("ANYGRASP_MAX_GRIPPER_WIDTH", "0.1")),
        gripper_height=float(os.getenv("ANYGRASP_GRIPPER_HEIGHT", "0.03")),
        top_down_grasp=_bool_env("ANYGRASP_TOP_DOWN_GRASP", False),
        debug=_bool_env("ANYGRASP_DEBUG", False),
    )

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()
    gg, _ = anygrasp.get_grasp(
        points,
        colors,
        lims=lims,
        apply_object_mask=_bool_env("ANYGRASP_APPLY_OBJECT_MASK", True),
        dense_grasp=_bool_env("ANYGRASP_DENSE_GRASP", False),
        collision_detection=_bool_env("ANYGRASP_COLLISION_DETECTION", True),
    )
    if len(gg) == 0:
        raise RuntimeError("AnyGrasp returned no grasps.")

    gg = gg.nms().sort_by_score()
    top_k = max(1, int(os.getenv("ANYGRASP_TOP_K", "20")))
    best = gg[0]
    top_list = [_grasp_to_dict(gg[i]) for i in range(min(len(gg), top_k))]
    return {
        "best_grasp": _grasp_to_dict(best),
        "top_grasps": top_list,
        "point_count": int(points.shape[0]),
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "intrinsics": intrinsics,
        "lims": lims,
    }


def _run_ur_executor(request: Dict[str, Any], grasp_result: Dict[str, Any]) -> Dict[str, Any]:
    cmd_text = os.getenv("UR_GRASP_EXECUTOR_CMD", "").strip()
    if not cmd_text:
        if _bool_env("ANYGRASP_REQUIRE_UR_EXECUTION", True):
            raise RuntimeError("UR_GRASP_EXECUTOR_CMD is required for real pick-and-place execution.")
        return {"executed": False, "note": "No UR executor configured; detection-only mode."}

    command = shlex.split(cmd_text) + [
        "--request-json",
        json.dumps(request, ensure_ascii=False),
        "--grasp-json",
        json.dumps(grasp_result.get("best_grasp", {}), ensure_ascii=False),
    ]
    timeout = int(os.getenv("UR_GRASP_EXECUTOR_TIMEOUT", "120"))
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    parsed = None
    if stdout:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if lines:
            try:
                parsed = json.loads(lines[-1])
            except Exception:
                parsed = None

    if completed.returncode != 0:
        raise RuntimeError(stderr or f"UR executor failed with exit code {completed.returncode}")

    return {
        "executed": True,
        "executor_returncode": completed.returncode,
        "executor_stdout": stdout,
        "executor_stderr": stderr,
        "executor_result": parsed if isinstance(parsed, dict) else None,
    }


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    intrinsics = _json_env(
        "ANYGRASP_INTRINSICS_JSON",
        {"fx": 927.17, "fy": 927.37, "cx": 651.32, "cy": 349.62, "scale": 1000.0, "z_min": 0.0, "z_max": 1.5},
    )
    lims = _json_env("ANYGRASP_LIMS_JSON", [-0.19, 0.12, 0.02, 0.15, 0.0, 1.0])
    if not isinstance(lims, list) or len(lims) != 6:
        raise RuntimeError("ANYGRASP_LIMS_JSON must be a 6-element list.")

    grasp_result = _pick_best_grasp(request, intrinsics=intrinsics, lims=lims)
    execution_result = _run_ur_executor(request, grasp_result)

    data = {
        "backend": "anygrasp",
        "request": request,
        **grasp_result,
        "execution": execution_result,
    }
    return _response(
        success=True,
        feedback=(
            f"AnyGrasp detected grasp for item='{request.get('item_name', 'item')}' and "
            f"{'executed UR action' if execution_result.get('executed') else 'skipped UR execution'}."
        ),
        data=data,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AnyGrasp runner")
    parser.add_argument("--request-json", required=True, help="request payload")
    args = parser.parse_args()

    try:
        request = json.loads(args.request_json)
    except json.JSONDecodeError as error:
        print(json.dumps(_response(False, "Invalid request JSON.", error=str(error))))
        return 2

    try:
        result = run(request)
    except Exception as error:
        result = _response(
            success=False,
            feedback="AnyGrasp execution failed.",
            error=str(error),
            data={"backend": "anygrasp", "request": request},
        )

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
