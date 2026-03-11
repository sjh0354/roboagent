#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="${1:-ras-ur5e-demo}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed"
    exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' does not exist"
    exit 0
fi

tmux kill-session -t "$SESSION_NAME"
echo "stopped tmux session: $SESSION_NAME"
