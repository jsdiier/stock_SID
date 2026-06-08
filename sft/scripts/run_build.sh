#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "[1/3] Building SFT datasets ..."
python sft/build_dataset.py --config sft/conf/sft.conf "$@"
echo "Done."
