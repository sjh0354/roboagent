#!/usr/bin/env bash

# Backward-compatible default planner config.
# Source the dedicated planner env directly when you want an explicit robot role:
#   source env/planner.g1.env.sh
#   source env/planner.ur5e.env.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/planner.g1.env.sh"
