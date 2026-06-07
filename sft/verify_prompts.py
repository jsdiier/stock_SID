"""
Quick sanity check: load tokenizer, extend with SID tokens, print 3 prompts.
Run this BEFORE training to verify the format looks right.

Usage:
  python verify_prompts.py
  python verify_prompts.py --config conf/sft.conf
"""
import argparse, configparser, os, sys
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline.llm_dataset import (
    LLMSIDDataset, all_sid_tokens, sid_tokens_str
)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='conf/sft.conf')
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.config, encoding='utf-8')

K   = cfg.getint('rqvae', 'num_codebooks')
CS  = cfg.getint('rqvae', 'codebook_size')
model_dir = cfg.get('llm', 'model_dir')
max_len   = cfg.getint('llm', 'max_seq_len')

print(f"Loading tokenizer from: {model_dir}")
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Extend tokenizer with SID tokens
sid_tokens = all_sid_tokens(K, CS)
n_before = len(tok)
tok.add_tokens(sid_tokens, special_tokens=True)
n_added  = len(tok) - n_before
print(f"Tokenizer vocab: {n_before:,} → {len(tok):,}  (+{n_added} SID tokens)")

# Verify a few SID tokens are atomic
tests = ['<a_0>', '<a_127>', '<b_64>', '<c_1>']
print("\nSID token atomicity check:")
for t in tests:
    ids     = tok.encode(t, add_special_tokens=False)
    decoded = tok.decode(ids)
    atomic  = '✓' if len(ids) == 1 else '✗ NOT atomic!'
    print(f"  {t!r:15s} → ids={ids}  decoded={decoded!r}  {atomic}")

# Load data and print 3 prompts
data_path = os.path.join(cfg.get('paths', 'data_dir'), 'train.npz')
d = np.load(data_path)
dataset = LLMSIDDataset(d['contexts'], d['targets'], tok, K, max_len)

sep = '=' * 72
print(f"\n{sep}")
print(f"  3 SAMPLE PROMPTS  (dataset has {len(dataset):,} samples)")
print(sep)

idxs = [0, len(dataset)//2, len(dataset)-1]
for rank, idx in enumerate(idxs):
    prompt_txt, next_str = dataset.sample_prompt_text(idx)

    p_ids = tok.encode(prompt_txt, add_special_tokens=False)
    n_ids = tok.encode(next_str,   add_special_tokens=False)
    decoded_comp = [tok.decode([t]) for t in n_ids]

    print(f"\n  ── Prompt {rank+1}  (index {idx}) ──")
    print(f"  Last 300 chars of prompt:")
    for line in prompt_txt.splitlines():
        print(f"    {line}")
    print(f"  Completion : {next_str}")
    print(f"  Comp tokens: {decoded_comp}")
    print(f"  Tokens     : prompt={len(p_ids)}  completion={len(n_ids)}  total={len(p_ids)+len(n_ids)}")

print(f"\n{sep}")
print("If each completion token decodes to exactly one SID (e.g. '<a_19>'), format is correct.")
