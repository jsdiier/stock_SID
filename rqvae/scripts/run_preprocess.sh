#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "[rqvae] Preprocessing stock data ..."
python rqvae/preprocess.py --config rqvae/conf/common.conf "$@"
echo "Done."
