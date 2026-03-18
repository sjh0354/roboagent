#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="${1:-ras-lark}"
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

CONDA_ACTIVATE_CMD="true"
if [[ -f "$CONDA_SH_PATH" ]]; then
    CONDA_ACTIVATE_CMD="source \"$CONDA_SH_PATH\" && conda activate \"$CONDA_ENV_NAME\""
else
    echo "warning: conda init script not found at $CONDA_SH_PATH; continuing without explicit conda activation"
fi

ENV_CMD="source \"$REPO_ROOT/env/common.env.sh\" && source \"$REPO_ROOT/env/lark.env.sh\""

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
tmux rename-window -t "$SESSION_NAME:0" "lark"
tmux send-keys -t "$SESSION_NAME:0.0" "bash -lc '$CONDA_ACTIVATE_CMD && $ENV_CMD && python \"$REPO_ROOT/server/lark_gateway.py\"'" C-m

tmux split-window -h -t "$SESSION_NAME:0" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:0.1" "bash -lc '$CONDA_ACTIVATE_CMD && $ENV_CMD && node \"$REPO_ROOT/server/lark_ws_bridge.js\"'" C-m

tmux select-layout -t "$SESSION_NAME:0" even-horizontal >/dev/null 2>&1 || true

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "conda env: $CONDA_ENV_NAME"
echo "left pane: lark_gateway.py"
echo "right pane: lark_ws_bridge.js"
