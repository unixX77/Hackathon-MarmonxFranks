#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash run_pipeline_classifier.sh --colour-model colour_setups/cielab/model.json --out data/pipeline_cielab_results "$@"
