#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Executing Big Memory Vanish..."

LOCAL_MEMORY_DIR="$REPO_ROOT/agent/memory/local"
if [ -d "$LOCAL_MEMORY_DIR" ]; then
    rm -rf "$LOCAL_MEMORY_DIR"
    echo "Removed personalized memory overlay: $LOCAL_MEMORY_DIR"
else
    echo "No personalized memory overlay found."
fi

if command -v git >/dev/null 2>&1 && [ -d "$REPO_ROOT/.git" ]; then
    git -C "$REPO_ROOT" restore --worktree \
        agent/skills \
        agent/profiles/humanoid_g1/skills_index.md \
        agent/profiles/ur5e/skills_index.md
    echo "Restored skills and skill indexes to repository baseline."
else
    echo "Skipped skill restore because this directory is not a git repository."
fi

echo "Big Memory Vanish complete."
