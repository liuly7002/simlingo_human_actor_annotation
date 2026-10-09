#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"
ANNOTATOR_ID="${2:-}"

python prepare_samples.py --config "$CONFIG"
python annotation_app.py --config "$CONFIG" --annotator-id "$ANNOTATOR_ID"
