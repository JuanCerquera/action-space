#!/usr/bin/env bash
# Upload and validate the scene on an existing Isaac Lab node, then stop billing.
set -euo pipefail

node_name="${1:-action-space}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
    local result=$?
    trap - EXIT
    if ! autor node stop "$node_name"; then
        echo "Failed to stop $node_name; run: autor node stop $node_name" >&2
        if (( result == 0 )); then result=1; fi
    fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

autor cp "$script_dir/01_scene.py" "$node_name:~/action-space/scripts/01_scene.py"
autor run "$node_name" --timeout 300 -- timeout 280 python -u \
    /home/isaaclab/action-space/scripts/01_scene.py \
    --device cuda:0 --viz none --steps 2100 --check-hover
