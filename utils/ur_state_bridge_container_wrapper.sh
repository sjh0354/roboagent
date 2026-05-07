#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${UR_ROS2_CONTAINER_NAME:-exp_ef_ur5e-arm-teleopration}"

docker exec -i \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-131}" \
  "${CONTAINER_NAME}" bash -lc '
[ -n "${LD_LIBRARY_PATH:-}" ] && unset LD_LIBRARY_PATH
[ -n "${PYTHONPATH:-}" ] && unset PYTHONPATH
[ -n "${CONDA_PREFIX:-}" ] && unset CONDA_PREFIX
[ -n "${CONDA_DEFAULT_ENV:-}" ] && unset CONDA_DEFAULT_ENV
[ -n "${VIRTUAL_ENV:-}" ] && unset VIRTUAL_ENV
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f /home/ef/projects/ur5e-arm-teleoperation/install/setup.bash ] && source /home/ef/projects/ur5e-arm-teleoperation/install/setup.bash
python3 /home/ef/projects/RAS_interactivate_planner/utils/ur_state_bridge_client.py "$@"
' -- "$@"
