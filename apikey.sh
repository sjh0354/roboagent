#!/usr/bin/env bash

_ras_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env"

for _ras_env_file in \
    "$_ras_env_dir/common.env.sh" \
    "$_ras_env_dir/planner.env.sh" \
    "$_ras_env_dir/lark.env.sh"
do
    if [ -f "$_ras_env_file" ]; then
        # shellcheck disable=SC1090
        source "$_ras_env_file"
    fi
done

unset _ras_env_dir
unset _ras_env_file
