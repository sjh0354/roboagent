#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ras_cpy}"
CONDA_SH_PATH="${CONDA_SH_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
POLICY_HOST="${POLICY_HOST:-10.11.18.197}"
POLICY_PORT="${POLICY_PORT:-8000}"
TASK_PROMPT="${TASK_PROMPT:-我想看书，帮我整理一下桌子}"
ARM_READING_VLM_MODEL="${ARM_READING_VLM_MODEL:-gpt-5.5}"
PLANNER_VLM_PROVIDER="${PLANNER_VLM_PROVIDER:-openai_compatible}"
PLANNER_VLM_BASE_URL="${PLANNER_VLM_BASE_URL:-https://api.sharesai.xyz/v1}"
PLANNER_VLM_FALLBACK_MODEL="${PLANNER_VLM_FALLBACK_MODEL:-gemini-2.5-flash-lite}"
PI0_ACTION_MONITOR_PROVIDER="${PI0_ACTION_MONITOR_PROVIDER:-openai_compatible}"
PI0_ACTION_MONITOR_BASE_URL="${PI0_ACTION_MONITOR_BASE_URL:-https://api.sharesai.xyz/v1}"
PI0_ACTION_MONITOR_MODEL="${PI0_ACTION_MONITOR_MODEL:-gpt-5.5}"
ARM_CONTROL_METHOD="${ARM_CONTROL_METHOD:-pi0}"
ARM_VLA_FALLBACK_TO_REPLAY="${ARM_VLA_FALLBACK_TO_REPLAY:-0}"
POST_ACTION_VLM_COMPARE="${POST_ACTION_VLM_COMPARE:-1}"
POST_ACTION_DELTA_MAX_PROCESS_FRAMES="${POST_ACTION_DELTA_MAX_PROCESS_FRAMES:-2}"
PI0_ACTION_VLM_MONITOR="${PI0_ACTION_VLM_MONITOR:-1}"
PI0_ACTION_VLM_MONITOR_MIN_SECONDS="${PI0_ACTION_VLM_MONITOR_MIN_SECONDS:-40}"
PI0_ACTION_VLM_MONITOR_INTERVAL_SECONDS="${PI0_ACTION_VLM_MONITOR_INTERVAL_SECONDS:-10}"
PI0_ACTION_VLM_MONITOR_MAX_PROCESS_FRAMES="${PI0_ACTION_VLM_MONITOR_MAX_PROCESS_FRAMES:-4}"
PI0_ACTION_VLM_MONITOR_SEQUENCE_FRAMES="${PI0_ACTION_VLM_MONITOR_SEQUENCE_FRAMES:-4}"
PI0_ACTION_VLM_MONITOR_SEQUENCE_INTERVAL_SECONDS="${PI0_ACTION_VLM_MONITOR_SEQUENCE_INTERVAL_SECONDS:-2.5}"
PI0_ACTION_STUCK_MIN_SECONDS="${PI0_ACTION_STUCK_MIN_SECONDS:-55}"
PI0_ACTION_MAX_USEFUL_SECONDS="${PI0_ACTION_MAX_USEFUL_SECONDS:-90}"
PI0_ACTION_TIMEOUT_SECONDS="${PI0_ACTION_TIMEOUT_SECONDS:-120}"
TRANSPORT_MODE="${TRANSPORT_MODE:-none}"
EXTRA_PLANNER_ARGS=()
ASSUME_SAFE="${ARM_TEST_ASSUME_SAFE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y)
            ASSUME_SAFE="1"
            shift
            ;;
        --simulation)
            EXTRA_PLANNER_ARGS+=("--simulation")
            shift
            ;;
        --prompt)
            TASK_PROMPT="$2"
            shift 2
            ;;
        --model)
            ARM_READING_VLM_MODEL="$2"
            shift 2
            ;;
        --no-visual-delta)
            POST_ACTION_VLM_COMPARE="0"
            shift
            ;;
        --visual-delta-frames)
            POST_ACTION_DELTA_MAX_PROCESS_FRAMES="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_PLANNER_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_PLANNER_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$ASSUME_SAFE" != "1" ]]; then
    cat <<'EOF'
Safety reminder: this script sends a task to the real arm planner and the arm may move.
Confirm before running:
- Human operator is watching the robot.
- E-stop is reachable and tested.
- Workspace is clear of hands and unexpected objects.
EOF
fi

if [[ ! -f "$CONDA_SH_PATH" ]]; then
    echo "conda init script not found: $CONDA_SH_PATH" >&2
    exit 1
fi

set +u
source "$CONDA_SH_PATH"
conda activate "$CONDA_ENV_NAME"
set -u

cd "$REPO_ROOT"

echo "Arm reading setup:"
echo "  prompt: $TASK_PROMPT"
echo "  vlm model: $ARM_READING_VLM_MODEL"
echo "  planner provider: $PLANNER_VLM_PROVIDER"
echo "  planner base url: $PLANNER_VLM_BASE_URL"
echo "  planner fallback model: $PLANNER_VLM_FALLBACK_MODEL"
echo "  pi0 monitor model: $PI0_ACTION_MONITOR_MODEL"
echo "  pi0 monitor provider: $PI0_ACTION_MONITOR_PROVIDER"
echo "  pi0 monitor base url: $PI0_ACTION_MONITOR_BASE_URL"
echo "  control method: $ARM_CONTROL_METHOD"
echo "  post-action visual delta: $POST_ACTION_VLM_COMPARE"
echo "  visual delta process frames: $POST_ACTION_DELTA_MAX_PROCESS_FRAMES"
echo "  pi0 vlm monitor: $PI0_ACTION_VLM_MONITOR"
echo "  pi0 monitor min seconds: $PI0_ACTION_VLM_MONITOR_MIN_SECONDS"
echo "  pi0 monitor interval seconds: $PI0_ACTION_VLM_MONITOR_INTERVAL_SECONDS"
echo "  pi0 monitor process frames: $PI0_ACTION_VLM_MONITOR_MAX_PROCESS_FRAMES"
echo "  pi0 monitor sequence frames: $PI0_ACTION_VLM_MONITOR_SEQUENCE_FRAMES"
echo "  pi0 monitor sequence interval seconds: $PI0_ACTION_VLM_MONITOR_SEQUENCE_INTERVAL_SECONDS"
echo "  pi0 stuck min seconds: $PI0_ACTION_STUCK_MIN_SECONDS"
echo "  pi0 max useful seconds: $PI0_ACTION_MAX_USEFUL_SECONDS"
echo "  pi0 action timeout seconds: $PI0_ACTION_TIMEOUT_SECONDS"

export POLICY_HOST
export POLICY_PORT
export ARM_CONTROL_METHOD
export ARM_VLA_FALLBACK_TO_REPLAY
export POST_ACTION_VLM_COMPARE
export POST_ACTION_DELTA_MAX_PROCESS_FRAMES
export PI0_ACTION_VLM_MONITOR
export PI0_ACTION_VLM_MONITOR_MIN_SECONDS
export PI0_ACTION_VLM_MONITOR_INTERVAL_SECONDS
export PI0_ACTION_VLM_MONITOR_MAX_PROCESS_FRAMES
export PI0_ACTION_VLM_MONITOR_SEQUENCE_FRAMES
export PI0_ACTION_VLM_MONITOR_SEQUENCE_INTERVAL_SECONDS
export PI0_ACTION_STUCK_MIN_SECONDS
export PI0_ACTION_MAX_USEFUL_SECONDS
export PI0_ACTION_TIMEOUT_SECONDS
export DEFAULT_VLM_MODEL="$ARM_READING_VLM_MODEL"
export PLANNER_VLM_PROVIDER
export PLANNER_VLM_BASE_URL
export PLANNER_VLM_FALLBACK_MODEL
export PI0_ACTION_MONITOR_PROVIDER
export PI0_ACTION_MONITOR_BASE_URL
export PI0_ACTION_MONITOR_MODEL

printf '%s\nq\n' "$TASK_PROMPT" | python -u planner/arm_planner_vlm.py \
    --log \
    --model "$ARM_READING_VLM_MODEL" \
    --transport "$TRANSPORT_MODE" \
    --control-method "$ARM_CONTROL_METHOD" \
    "${EXTRA_PLANNER_ARGS[@]}"
