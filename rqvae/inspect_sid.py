"""
查询某个 SID token 对应的所有周内样本，并可视化收盘价走势。

Usage:
  python inspect_sid.py --sid "<a_114><b_11><c_55>"
  python inspect_sid.py --sid "<a_114><b_11><c_55>" --max-samples 20
  python inspect_sid.py --sid "<a_114><b_11><c_55>" --rebuild-cache
"""
import argparse
import configparser
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model.rqvae import RQVAE

LEVEL_NAMES   = list('abcdefghijklmnopqrstuvwxyz')
CLOSE_INDICES = [3, 10, 17, 24, 31]   # close 在每天 7 指标中是第 4 个（index 3）
DAY_LABELS    = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5']


def _load_model(cfg, device):
    hidden_dims = [int(x) for x in cfg.get('rqvae', 'hidden_dims').split(',')]
    model = RQVAE(
        input_dim         = cfg.getint('data',   'input_dim'),
        hidden_dims       = hidden_dims,
        latent_dim        = cfg.getint('rqvae',  'latent_dim'),
        num_codebooks     = cfg.getint('rqvae',  'num_codebooks'),
        codebook_size     = cfg.getint('rqvae',  'codebook_size'),
        ema_decay         = cfg.getfloat('rqvae', 'ema_decay'),
        commitment_weight = cfg.getfloat('rqvae', 'commitment_weight'),
    ).to(device)

    ckpt_dir = cfg.get('paths', 'checkpoint_dir')
    ckpts    = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_epoch_*.pt')))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model loaded: {os.path.basename(ckpts[-1])}")
    return model


def _build_cache(model, train_npy, cache_path, index_path, device, batch_size=4096):
    """编码全量 embedding，保存 sid_cache.npy 和 sid_index.json。"""
    data      = np.load(train_npy)
    all_codes = []
    total     = len(data)

    for i in range(0, total, batch_size):
        batch  = torch.from_numpy(data[i:i + batch_size]).float().to(device)
        codes  = model.encode_to_sid(batch).cpu().numpy()
        all_codes.append(codes)
        print(f"\r  encoding {min(i + batch_size, total):,}/{total:,}", end='', flush=True)
    print()

    all_codes = np.concatenate(all_codes, axis=0)
    np.save(cache_path, all_codes)
    print(f"SID cache saved: {cache_path}")

    # 构建倒排索引
    from collections import defaultdict
    token_index = defaultdict(list)
    for idx, codes in enumerate(all_codes):
        token = ''.join(f'<{LEVEL_NAMES[i]}_{v}>' for i, v in enumerate(codes))
        token_index[token].append(idx)

    with open(index_path, 'w') as f:
        json.dump(token_index, f)
    print(f"SID index saved: {index_path}  ({len(token_index):,} unique SIDs)")
    return token_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid',           required=True,
                        help='SID token，如 "<a_114><b_11><c_55>"')
    parser.add_argument('--config',        default='conf/common.conf')
    parser.add_argument('--max-samples',   type=int, default=50)
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='强制重新编码（模型更新后使用）')
    args = parser.parse_args()

    query_token = args.sid.strip()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dir = cfg.get('paths', 'embedding_dir')
    train_npy     = os.path.join(embedding_dir, 'train.npy')
    cache_path    = os.path.join(embedding_dir, 'sid_cache.npy')
    index_path    = os.path.join(embedding_dir, 'sid_index.json')

    # ---------- 构建或加载缓存 ----------
    need_build = args.rebuild_cache or not os.path.exists(index_path)
    if need_build:
        model       = _load_model(cfg, device)
        token_index = _build_cache(model, train_npy, cache_path, index_path, device)
    else:
        with open(index_path, 'r') as f:
            token_index = json.load(f)
        print(f"SID index loaded: {len(token_index):,} unique SIDs")

    # ---------- 查找匹配样本 ----------
    match_indices = token_index.get(query_token, [])
    print(f"\n{query_token} → {len(match_indices):,} 个匹配样本")

    if not match_indices:
        print("未找到匹配样本，请检查 SID 格式或换一个 token")
        return

    match_indices = np.array(match_indices)

    # ---------- 加载数据 ----------
    all_vecs = np.load(train_npy)
    meta_df  = pd.read_csv(os.path.join(embedding_dir, 'train_meta.csv'))

    # 随机采样（样本过多时）
    plot_indices = match_indices
    if len(match_indices) > args.max_samples:
        rng          = np.random.default_rng(42)
        plot_indices = rng.choice(match_indices, args.max_samples, replace=False)
        print(f"样本过多，随机展示 {args.max_samples} 条")

    # ---------- 提取收盘价 ----------
    close_matrix = all_vecs[plot_indices][:, CLOSE_INDICES]             # (M, 5)
    mean_close   = all_vecs[match_indices][:, CLOSE_INDICES].mean(axis=0)  # (5,)

    # ---------- 画图 ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = cm.tab20(np.linspace(0, 1, len(plot_indices)))

    for i, (idx, color) in enumerate(zip(plot_indices, colors)):
        ts_code   = meta_df.iloc[idx]['ts_code']
        year_week = meta_df.iloc[idx]['year_week']
        ax.plot(range(1, 6), close_matrix[i],
                color=color, alpha=0.35, linewidth=1,
                label=f'{ts_code} {year_week}')

    ax.plot(range(1, 6), mean_close,
            color='red', linewidth=2.5, zorder=5,
            label=f'Mean (n={len(match_indices):,})')

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(DAY_LABELS)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Normalized Close  (relative to Day1 close)')
    ax.set_title(f'{query_token}\n{len(match_indices):,} samples')
    ax.grid(True, alpha=0.25)

    if len(plot_indices) <= 10:
        ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    safe_name = query_token.replace('<', '').replace('>', '').replace('_', '-')
    out_name  = f'sid_{safe_name}.png'
    plt.savefig(out_name, dpi=150)
    plt.show()
    print(f"图片已保存: {out_name}")


if __name__ == '__main__':
    main()
