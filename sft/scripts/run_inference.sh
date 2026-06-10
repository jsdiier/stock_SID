#!/bin/bash
# 行情数据增量更新（幂等） + 周度 SID 推理，一键执行。
#
# Usage:
#   bash sft/scripts/run_inference.sh                  # 以今天为基准
#   bash sft/scripts/run_inference.sh --date 20260610  # 额外参数透传给 inference.py
#   bash sft/scripts/run_inference.sh --max-stocks 50  # 冒烟测试
#
# 幂等性保证（三层）:
#   1. 周级跳过 : data/raw/.last_complete_week 记录已同步到的完整周，
#                同一周内重复执行直接跳过抓取（stamp 仅在成功后写入，set -e 保证）
#   2. 数据去重 : preprocess --incremental 按 week label 去重，重复抓取不产生重复行
#   3. 缓存自愈 : inference.py 检测 train.npy 与 sid_cache.npy 行数/mtime 不一致时
#                自动用 RQ-VAE 重编码，保证收益表对齐
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# --- 环境变量 ---
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
if [ -d "$VAE_ENV_PATH" ]; then
    export PATH="${VAE_ENV_PATH}/bin:$PATH"
fi
echo ">>> 当前 Python 路径: $(which python)"

# --- 自动创建模型软链接（幂等）---
MODEL_SRC="${DATA_PREFIX}/chenpinyuan/MODEL/Qwen3-0.6B"
MODEL_DST="${ROOT}/sft/models/qwen3-0.6b"
if [ -d "$MODEL_SRC" ]; then
    mkdir -p "${ROOT}/sft/models"
    ln -sfn "$MODEL_SRC" "$MODEL_DST"
fi

# ============================================================
# [1/2] 行情数据增量更新（幂等）
# ============================================================
RAW_DIR="data/raw"
STAMP="${RAW_DIR}/.last_complete_week"

# 上一个完整 ISO 周（python 计算，跨平台）
EXPECTED_WEEK=$(python -c "
import datetime as dt
d = dt.date.today() - dt.timedelta(days=7)
y, w, _ = d.isocalendar()
print(f'{y}-W{w:02d}')")

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$EXPECTED_WEEK" ]; then
    echo "[1/2] 行情数据已同步到 ${EXPECTED_WEEK}，跳过抓取。"
else
    # 只回看 70 天：覆盖最近 ~10 周，--incremental 按 week label 去重
    FETCH_START=$(python -c "
import datetime as dt
print((dt.date.today() - dt.timedelta(days=70)).strftime('%Y%m%d'))")
    echo "[1/2] 增量更新行情数据  (start=${FETCH_START}, 目标完整周=${EXPECTED_WEEK}) ..."
    python rqvae/preprocess.py \
        --config rqvae/conf/common.conf \
        --incremental \
        --start-date "$FETCH_START"
    echo "$EXPECTED_WEEK" > "$STAMP"
    echo "[1/2] 数据更新完成，stamp → ${EXPECTED_WEEK}"
fi

# ============================================================
# [2/2] 推理
# ============================================================
# --- 推理任务概览 tag ---
python -c "
import datetime as dt, glob, os, configparser
today = dt.date.today()
y, w, _ = today.isocalendar()
mon = today - dt.timedelta(days=today.weekday())
fri = mon + dt.timedelta(days=4)
cfg = configparser.ConfigParser(); cfg.read('sft/conf/sft.conf', encoding='utf-8')
excl  = tuple(p.strip() for p in cfg.get('inference','exclude_prefixes',fallback='').split(',') if p.strip())
k     = cfg.get('inference','beam_k',fallback='5')
files = glob.glob('data/raw/*.npz')
n_all  = len(files)
n_keep = len([f for f in files if not os.path.basename(f).startswith(excl)])
print(f'>>> ───────────── 推理任务概览 ─────────────')
print(f'>>> 预测目标周   : {y}-W{w:02d}  ({mon} ~ {fri})')
print(f'>>> 输入数据截至 : 上一完整周 (目标周之前最近52周)')
print(f'>>> 股票池       : {n_all} 支, 排除前缀 {list(excl)} 后待推理 {n_keep} 支')
print(f'>>> beam_k       : {k} (每支股票输出{k}个候选SID)')
print(f'>>> ──────────────────────────────────────')
"
echo "[2/2] Running inference ..."
python sft/inference.py --config sft/conf/sft.conf "$@"
echo "Done."
