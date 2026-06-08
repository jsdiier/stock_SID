#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "[rqvae] Exporting SID cache ..."
# --rebuild-cache encodes all embeddings with the trained model and saves sid_cache.npy
# --sid is required by the script but only used for the query step after cache is built
python rqvae/inspect_sid.py --config rqvae/conf/common.conf --rebuild-cache --sid "<a_0><b_0><c_0>"
echo "Done."
