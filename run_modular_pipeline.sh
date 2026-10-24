#!/usr/bin/env bash
# Separate pipeline: D435 -> top-plane boxes -> quality classifier -> camera poses.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/sam3/bin/python}
exec "$PY" run_modular_pipeline.py "$@"
