#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="${UR_ROS2_CONTAINER_NAME:-exp_ef_ur5e-arm-teleopration}"
MODE="${UR_EXECUTION_MODE:-motion_only}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8001}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-5}"
CONTROL_RATE="${CONTROL_RATE:-10}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.02}"
ROS2_PUB_TIMEOUT_SEC="${UR_ROS2_PUB_TIMEOUT_SEC:-${UR_MOTION_PUBLISH_TIMEOUT_SEC:-3.0}}"

docker exec -i \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-131}" \
  -e UR_ENABLE_MOTION="${UR_ENABLE_MOTION:-0}" \
  -e UR_ENABLE_FULL_ONCE="${UR_ENABLE_FULL_ONCE:-0}" \
  -e UR_JOINT_LIMITS_JSON="${UR_JOINT_LIMITS_JSON:-}" \
  -e UR_EXEC_MODE_INNER="${MODE}" \
  -e UR_POLICY_HOST_INNER="${POLICY_HOST}" \
  -e UR_POLICY_PORT_INNER="${POLICY_PORT}" \
  -e UR_ACTION_CHUNK_SIZE_INNER="${ACTION_CHUNK_SIZE}" \
  -e UR_CONTROL_RATE_INNER="${CONTROL_RATE}" \
  -e UR_MAX_JOINT_STEP_INNER="${MAX_JOINT_STEP}" \
  -e UR_ROS2_PUB_TIMEOUT_SEC="${ROS2_PUB_TIMEOUT_SEC}" \
  -e UR_MOTION_PUBLISH_TIMEOUT_SEC="${ROS2_PUB_TIMEOUT_SEC}" \
  "${CONTAINER_NAME}" bash -lc '
[ -n "${LD_LIBRARY_PATH:-}" ] && unset LD_LIBRARY_PATH
[ -n "${PYTHONPATH:-}" ] && unset PYTHONPATH
[ -n "${CONDA_PREFIX:-}" ] && unset CONDA_PREFIX
[ -n "${CONDA_DEFAULT_ENV:-}" ] && unset CONDA_DEFAULT_ENV
[ -n "${VIRTUAL_ENV:-}" ] && unset VIRTUAL_ENV
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f /home/ef/projects/ur5e-arm-teleoperation/install/setup.bash ] && source /home/ef/projects/ur5e-arm-teleoperation/install/setup.bash
python /home/ef/projects/RAS_interactivate_planner/utils/ur_policy_bridge_client.py \
  --mode "${UR_EXEC_MODE_INNER}" \
  --policy-host "${UR_POLICY_HOST_INNER}" \
  --policy-port "${UR_POLICY_PORT_INNER}" \
  --action-chunk-size "${UR_ACTION_CHUNK_SIZE_INNER}" \
  --control-rate "${UR_CONTROL_RATE_INNER}" \
  --max-joint-step "${UR_MAX_JOINT_STEP_INNER}" \
  "$@"
' -- "$@"
