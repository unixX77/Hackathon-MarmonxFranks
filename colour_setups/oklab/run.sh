#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash run_pipeline_classifier.sh --colour-model colour_setups/oklab/model.json --out data/pipeline_oklab_results "$@"
