"""
把所有出现过的 SID 组合导出到 txt，同时构建 token→样本索引 映射并保存到本地。

Usage:
  python export_sids.py
"""
import configparser
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

LEVEL_NAMES = list('abcdefghijklmnopqrstuvwxyz')

def codes_to_token(codes) -> str:
    return ''.join(f'<{LEVEL_NAMES[i]}_{v}>' for i, v in enumerate(codes))


cfg = configparser.ConfigParser()
cfg.read('conf/common.conf', encoding='utf-8')

embedding_dir = cfg.get('paths', 'embedding_dir')
cache_path    = os.path.join(embedding_dir, 'sid_cache.npy')

if not os.path.exists(cache_path):
    print("sid_cache.npy 不存在，请先运行 inspect_sid.py --rebuild-cache")
    sys.exit(1)

all_codes = np.load(cache_path)   # (N, K)

# 构建 token → 样本索引列表 的倒排索引
print("构建 token 索引...")
token_index = defaultdict(list)
for idx, codes in enumerate(all_codes):
    token = codes_to_token(codes)
    token_index[token].append(idx)

# 保存倒排索引
index_path = os.path.join(embedding_dir, 'sid_index.json')
with open(index_path, 'w') as f:
    json.dump(token_index, f)
print(f"索引已保存: {index_path}  ({len(token_index):,} 个唯一 SID)")

# 保存 all_sids.txt
unique_tokens = sorted(token_index.keys())
sids_path = os.path.join(embedding_dir, 'all_sids.txt')
with open(sids_path, 'w') as f:
    for token in unique_tokens:
        f.write(token + '\n')

print(f"总样本数:    {len(all_codes):,}")
print(f"唯一 SID 数: {len(token_index):,}")
print(f"SID 列表:    {sids_path}")
