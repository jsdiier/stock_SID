#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "[rqvae] Training RQ-VAE model ..."
python rqvae/train.py --config rqvae/conf/common.conf "$@"
echo "Done."
