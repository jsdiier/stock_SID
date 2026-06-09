#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# --- 环境变量 ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=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

echo ">>> 正在初始化环境..."
# --- 动态识别根目录 ---
if [ -d "/nfs/dataset-ofs-rank-ssl" ]; then
    DATA_PREFIX="/nfs/dataset-ofs-rank-ssl"
    echo ">>> 检测到训练平台环境 (NFS)，使用路径: ${DATA_PREFIX}"
else
    DATA_PREFIX="/home/luban/rank-ssl"
    echo ">>> 检测到本地 SSH 环境，使用路径: ${DATA_PREFIX}"
fi

# --- 激活环境：直接把 env bin 加到 PATH 最前面，绕过 conda shebang ---
VAE_ENV_PATH="${DATA_PREFIX}/chenpinyuan/miniconda_base/envs/Stock_SID"
export PATH="${VAE_ENV_PATH}/bin:$PATH"
echo ">>> 当前 Python 路径: $(which python)"
python --version

# --- 自动创建模型软链接 ---
MODEL_SRC="${DATA_PREFIX}/chenpinyuan/MODEL/Qwen3-0.6B"
MODEL_DST="${ROOT}/sft/models/qwen3-0.6b"
echo ">>> 创建模型软链接: ${MODEL_DST} -> ${MODEL_SRC}"
mkdir -p "${ROOT}/sft/models"
ln -sfn "$MODEL_SRC" "$MODEL_DST"

echo "[2/3] Training LLM (Qwen3-0.6B + LoRA) ..."
python sft/train_llm.py --config sft/conf/sft.conf "$@"
echo "Done."
