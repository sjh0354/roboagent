#!/usr/bin/env bash
set -euo pipefail

# UR RTDE preflight for RAS workstation runs.
# Purpose:
# 1) clear residual UR control processes (host + container)
# 2) verify RTDE/Dashboard ports are reachable
# 3) verify ROS2 chain visibility gate before motion test

ROBOT_IP="${ROBOT_IP:-192.168.24.245}"
CONTAINER_NAME="${CONTAINER_NAME:-exp_ef_ur5e-arm-teleopration}"
ROS_DOMAIN_ID_VAL="${ROS_DOMAIN_ID:-131}"
UR_WS_PATH="${UR_WS_PATH:-/home/ef/projects/ur5e-arm-teleoperation}"

RUN_CLEANUP=1
RUN_GATE=1
GATE_LEVEL="${GATE_LEVEL:-motion}"
STATE_SOURCE="${STATE_SOURCE:-ros_topic}"

usage() {
  cat <<'EOF'
Usage:
  scripts/ur_rtde_preflight.sh [--check-only] [--cleanup-only] [--state-gate|--motion-gate] [--state-source ros_topic|rtde]

Env vars (optional):
  ROBOT_IP            default: 192.168.24.245
  CONTAINER_NAME      default: exp_ef_ur5e-arm-teleopration
  ROS_DOMAIN_ID       default: 131
  UR_WS_PATH          default: /home/ef/projects/ur5e-arm-teleoperation
  GATE_LEVEL          default: motion (state|motion)
  STATE_SOURCE        default: ros_topic (ros_topic|rtde)

Exit codes:
  0  pass
  1  preflight failed (see logs)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      RUN_CLEANUP=0
      RUN_GATE=1
      shift
      ;;
    --cleanup-only)
      RUN_CLEANUP=1
      RUN_GATE=0
      shift
      ;;
    --state-gate)
      GATE_LEVEL="state"
      shift
      ;;
    --motion-gate)
      GATE_LEVEL="motion"
      shift
      ;;
    --state-source)
      [[ $# -ge 2 ]] || fail "--state-source requires a value"
      STATE_SOURCE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

log() { printf '[preflight] %s\n' "$*"; }

fail() {
  printf '[preflight] FAIL: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

require_cmd docker
require_cmd python

safe_kill_by_pattern() {
  # Kill by pattern but never kill the current shell/process tree.
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if [[ "$pid" == "$$" || "$pid" == "$PPID" ]]; then
      continue
    fi
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
}

if [[ "$RUN_CLEANUP" -eq 1 ]]; then
  log "cleanup host residual processes"
  safe_kill_by_pattern 'ur5e_single_arm.launch.py'
  safe_kill_by_pattern '(^|/)ur5e_robot($| )'
  safe_kill_by_pattern '(^|/)ur_rtde($| )'

  log "cleanup container residual processes: ${CONTAINER_NAME}"
  docker exec "${CONTAINER_NAME}" bash -lc \
    '
for patt in "ur5e_single_arm.launch.py" "(^|/)ur5e_robot($| )" "(^|/)ur_rtde($| )"; do
  pids=$(pgrep -f "$patt" || true)
  [ -z "$pids" ] && continue
  while read -r pid; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
done
' \
    || fail "container cleanup failed (${CONTAINER_NAME})"
fi

log "socket check ${ROBOT_IP}:29999/30004"
SOCK_RESULT="$(python - <<PY
import socket
ip="${ROBOT_IP}"
results=[]
for p in (29999,30004):
    s=socket.socket(); s.settimeout(2)
    try:
        s.connect((ip,p)); results.append((p,"open",""))
    except Exception as e:
        results.append((p,"fail",str(e)))
    finally:
        s.close()
for p,st,msg in results:
    print(f"{p} {st} {msg}".rstrip())
PY
)"
printf '%s\n' "$SOCK_RESULT"

echo "$SOCK_RESULT" | grep -q '^29999 open' || fail "29999 not open"
echo "$SOCK_RESULT" | grep -q '^30004 open' || fail "30004 not open"

if [[ "$RUN_GATE" -eq 1 ]]; then
  if [[ "$GATE_LEVEL" == "state" ]]; then
    if [[ "$STATE_SOURCE" == "ros_topic" ]]; then
      log "state gate via ROS topic (single RTDE owner: ur5e_robot)"
      GATE_LOG="$(docker exec "${CONTAINER_NAME}" bash -lc "
set -e
source /ros_entrypoint.sh
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f ${UR_WS_PATH}/install/setup.bash ] && source ${UR_WS_PATH}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_VAL}
echo NODES:
ros2 node list || true
echo TOPICS:
ros2 topic list | grep -E '/arm_1/arm_joint_states' || true
echo JOINT_ONCE:
timeout 6s ros2 topic echo /arm_1/arm_joint_states --once --field position || true
" )" || fail "state gate (ros_topic) failed"
      printf '%s\n' "$GATE_LOG"
      echo "$GATE_LOG" | grep -q '/arm_1/arm_joint_states' || fail "topic /arm_1/arm_joint_states not visible"
      echo "$GATE_LOG" | grep -q '\[' || fail "joint_state one-shot empty"
    elif [[ "$STATE_SOURCE" == "rtde" ]]; then
      log "state gate via RAS RTDE bridge (diagnostic fallback)"
      STATE_OUT="$(/home/ef/projects/RAS_interactivate_planner/utils/ur_state_bridge_container_wrapper.sh \
        --robot-ip "${ROBOT_IP}" \
        --arm-name "arm_1" \
        --timeout-sec "3.0" 2>&1)" || fail "state bridge command failed: ${STATE_OUT}"
      printf '%s\n' "$STATE_OUT"
      echo "$STATE_OUT" | tail -n 1 | grep -q '"success"[[:space:]]*:[[:space:]]*true' || fail "state gate failed"
    else
      fail "invalid --state-source: ${STATE_SOURCE} (expected ros_topic|rtde)"
    fi
  else
    log "ROS2 visibility gate (${GATE_LEVEL}) in container: ${CONTAINER_NAME}"
    run_motion_gate_once() {
      docker exec "${CONTAINER_NAME}" bash -lc "
set -e
source /ros_entrypoint.sh
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f ${UR_WS_PATH}/install/setup.bash ] && source ${UR_WS_PATH}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_VAL}
echo NODES:
ros2 node list || true
echo TOPICS:
ros2 topic list | grep -E '/arm_1/(arm_joint_states|command/joint_trajectory|command/gripper_command)' || true
echo JOINT_ONCE:
timeout 6s ros2 topic echo /arm_1/arm_joint_states --once --field position || true
" 
    }

    GATE_LOG="$(run_motion_gate_once)" || fail "ros2 gate check failed"
    printf '%s\n' "$GATE_LOG"

    # Motion gate uses topic visibility as source of truth.
    # Retry once with startup wait to absorb node naming/startup timing jitter.
    if ! echo "$GATE_LOG" | grep -q '/arm_1/command/joint_trajectory' || \
       ! echo "$GATE_LOG" | grep -q '/arm_1/arm_joint_states'; then
      log "motion gate topics not fully visible, wait 10s and retry once"
      sleep 10
      GATE_LOG_RETRY="$(run_motion_gate_once)" || fail "ros2 gate retry failed"
      printf '%s\n' "$GATE_LOG_RETRY"
      echo "$GATE_LOG_RETRY" | grep -q '/arm_1/command/joint_trajectory' || fail "topic /arm_1/command/joint_trajectory not visible"
      echo "$GATE_LOG_RETRY" | grep -q '/arm_1/arm_joint_states' || fail "topic /arm_1/arm_joint_states not visible"
    fi
  fi
fi

log "PASS"
