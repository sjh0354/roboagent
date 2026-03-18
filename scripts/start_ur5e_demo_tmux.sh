#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="${1:-ras-ur5e-demo}"
shift || true
# PLANNER_ARGS="${*:---simulation --log}"
PLANNER_ARGS="${*:---log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV_NAME="${TMUX_CONDA_ENV:-ras}"
CONDA_SH_PATH="${CONDA_SH_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed"
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists"
    echo "attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

COMMON_ENV_CMD="source \"$REPO_ROOT/env/common.env.sh\""
LARK_ENV_CMD="source \"$REPO_ROOT/env/lark.env.sh\""
PLANNER_ENV_CMD="source \"$REPO_ROOT/env/planner.ur5e.env.sh\""

CONDA_ACTIVATE_CMD="true"
if [[ -f "$CONDA_SH_PATH" ]]; then
    CONDA_ACTIVATE_CMD="source \"$CONDA_SH_PATH\" && conda activate \"$CONDA_ENV_NAME\""
else
    echo "warning: conda init script not found at $CONDA_SH_PATH; continuing without explicit conda activation"
fi

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
tmux rename-window -t "$SESSION_NAME:0" "ur5e-demo"

tmux send-keys -t "$SESSION_NAME:0.0" "bash -lc '$CONDA_ACTIVATE_CMD && $COMMON_ENV_CMD && $LARK_ENV_CMD && python \"$REPO_ROOT/server/lark_gateway.py\"'" C-m

tmux split-window -h -t "$SESSION_NAME:0" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:0.1" "bash -lc '$CONDA_ACTIVATE_CMD && $COMMON_ENV_CMD && $LARK_ENV_CMD && node \"$REPO_ROOT/server/lark_ws_bridge.js\"'" C-m

tmux split-window -v -t "$SESSION_NAME:0.1" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:0.2" "bash -lc '$CONDA_ACTIVATE_CMD && $COMMON_ENV_CMD && $PLANNER_ENV_CMD && python \"$REPO_ROOT/planner/arm_planner_vlm.py\" $PLANNER_ARGS'" C-m

tmux select-layout -t "$SESSION_NAME:0" tiled >/dev/null 2>&1 || true

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "conda env: $CONDA_ENV_NAME"
echo "pane 0: lark_gateway.py"
echo "pane 1: lark_ws_bridge.js"
echo "pane 2: arm_planner_vlm.py $PLANNER_ARGS"
