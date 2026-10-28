#!/usr/bin/env bash
# Normal pipeline geometry + modular standalone classifier only.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/sam3/bin/python}
exec "$PY" run_pipeline_classifier.py "$@"
