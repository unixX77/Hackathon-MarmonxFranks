#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/sam3/bin/python}
exec "$PY" classifier/colour_model.py --colour-space cielab --out colour_setups/cielab/model.json "$@"
