#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "[3/3] Predicting and evaluating ..."
python sft/predict.py --config sft/conf/sft.conf "$@"
echo "Done."
