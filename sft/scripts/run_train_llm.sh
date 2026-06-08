#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "[2/3] Training LLM (Qwen3-0.6B + LoRA) ..."
python sft/train_llm.py --config sft/conf/sft.conf "$@"
echo "Done."
