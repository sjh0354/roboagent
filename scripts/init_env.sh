#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$REPO_ROOT/env"

copy_if_missing() {
    local src="$1"
    local dst="$2"

    if [ -f "$dst" ]; then
        echo "skip existing: ${dst#$REPO_ROOT/}"
        return
    fi

    cp "$src" "$dst"
    echo "created: ${dst#$REPO_ROOT/}"
}

copy_if_missing "$ENV_DIR/common.env.example.sh" "$ENV_DIR/common.env.sh"
copy_if_missing "$ENV_DIR/lark.env.example.sh" "$ENV_DIR/lark.env.sh"
copy_if_missing "$ENV_DIR/planner.g1.env.example.sh" "$ENV_DIR/planner.g1.env.sh"
copy_if_missing "$ENV_DIR/planner.ur5e.env.example.sh" "$ENV_DIR/planner.ur5e.env.sh"

echo
echo "local env files are ready"
echo "next step: edit env/*.env.sh with your real keys and chat ids"
