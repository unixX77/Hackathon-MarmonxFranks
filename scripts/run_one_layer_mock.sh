#!/usr/bin/env bash
# Run the main coordinator with simulated manipulation/perception only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src/lego_sorter${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/runtime"
task_runtime="$(mktemp -d "$ROOT/runtime/one-layer-mock-XXXXXXXX")"
echo 'SIMULATION ONLY: no robot or camera connections. Default: 30 bricks; use --bricks N and --step.'
echo "Results: $task_runtime"
exec /usr/bin/python3 -s -m lego_sorter.cli mock \
  --config "$ROOT/src/lego_sorter/config/loose_one_layer_mock.yaml" \
  --runtime "$task_runtime" "$@"
