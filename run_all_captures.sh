#!/usr/bin/env bash
# Run the classifier pipeline on every saved D435 capture; see run_all_captures.py --help.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/sam3/bin/python}
exec "$PY" run_all_captures.py "$@"
