#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "[2/3] Training SFT model ..."
python sft/train.py --config sft/conf/sft.conf "$@"
echo "Done."
