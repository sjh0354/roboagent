#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="${1:-ras-g1-demo}"
shift || true
PLANNER_ARGS="${*:---simulation --log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
PLANNER_ENV_CMD="source \"$REPO_ROOT/env/planner.g1.env.sh\""

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
tmux rename-window -t "$SESSION_NAME:0" "g1-demo"

tmux send-keys -t "$SESSION_NAME:0.0" "bash -lc '$COMMON_ENV_CMD && $LARK_ENV_CMD && python \"$REPO_ROOT/server/lark_gateway.py\"'" C-m

tmux split-window -h -t "$SESSION_NAME:0" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:0.1" "bash -lc '$COMMON_ENV_CMD && $LARK_ENV_CMD && node \"$REPO_ROOT/server/lark_ws_bridge.js\"'" C-m

tmux split-window -v -t "$SESSION_NAME:0.1" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME:0.2" "bash -lc '$COMMON_ENV_CMD && $PLANNER_ENV_CMD && python \"$REPO_ROOT/planner/humanoid_planner_vlm.py\" $PLANNER_ARGS'" C-m

tmux select-layout -t "$SESSION_NAME:0" tiled >/dev/null 2>&1 || true

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "pane 0: lark_gateway.py"
echo "pane 1: lark_ws_bridge.js"
echo "pane 2: humanoid_planner_vlm.py $PLANNER_ARGS"
