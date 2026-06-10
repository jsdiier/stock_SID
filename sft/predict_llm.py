"""
Inference: use the fine-tuned LoRA model to predict next-week SIDs.

Autoregressive mode (default): predicted SID fed back as context.
Teacher-forced mode (--tf):    actual SID fed back (oracle upper-bound).

Usage
-----
  python predict_llm.py
  python predict_llm.py --tf
  python predict_llm.py --config conf/sft.conf
"""
import argparse
import configparser
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline.llm_dataset import LLMSIDDataset, all_sid_tokens, sid_tokens_str

LEVEL_NAMES = 'abcdefghijklmnopqrstuvwxyz'


def load_model_and_tokenizer(cfg, device):
    model_dir   = cfg.get('llm', 'model_dir')
    adapter_dir = cfg.get('llm', 'adapter_dir')

    # Find latest saved adapter.
    # New layout: adapter_dir/run_<timestamp>/epoch_XX  (one folder per training
    # run; timestamp sorts chronologically, so the last path = newest run's
    # newest epoch). Legacy layout adapter_dir/epoch_XX is the fallback.
    adapters = (sorted(glob.glob(os.path.join(adapter_dir, 'run_*', 'epoch_*')))
                or sorted(glob.glob(os.path.join(adapter_dir, 'epoch_*'))))
    if not adapters:
        raise FileNotFoundError(f"No adapters in {adapter_dir} — run train_llm.py first")
    latest = adapters[-1]

    log = logging.getLogger(__name__)
    log.info(f"Loading base model from {model_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    log.info(f"Loading tokenizer + LoRA adapter from {latest}")
    tokenizer = AutoTokenizer.from_pretrained(latest, trust_remote_code=True)

    # Resize base model embeddings to match extended tokenizer
    base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base, latest)
    model = model.to(device).eval()
    log.info(f"Model loaded  vocab={len(tokenizer):,}")
    return model, tokenizer


@torch.no_grad()
def predict_next_sid(model, tokenizer, context_str: str, K: int, device) -> list:
    """
    Given a history string (concatenated SID tokens), generate the next SID.
    Returns list of K integer codes.
    """
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': context_str}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
    ).to(device)

    # We want exactly K new tokens (one per codebook level)
    out = model.generate(
        prompt_ids,
        max_new_tokens = K,
        do_sample      = False,    # greedy
        pad_token_id   = tokenizer.pad_token_id,
    )

    new_token_ids = out[0, prompt_ids.shape[1]:].tolist()
    # Decode each new token and parse the code index
    codes = []
    for tid in new_token_ids[:K]:
        tok_str = tokenizer.convert_ids_to_tokens(tid)   # e.g. '<a_19>'
        try:
            code = int(tok_str.split('_')[-1].rstrip('>'))
        except Exception:
            code = 0
        codes.append(code)

    # Pad if fewer than K tokens were generated
    while len(codes) < K:
        codes.append(0)
    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conf/sft.conf')
    parser.add_argument('--tf', action='store_true',
                        help='Teacher-forced mode (use ground-truth SID as next context)')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    result_dir = cfg.get('paths', 'result_dir')
    log_dir    = cfg.get('paths', 'log_dir')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)

    mode_str = 'teacher-forced' if args.tf else 'autoregressive'
    suffix   = 'tf'             if args.tf else 'ar'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, f'predict_llm_{suffix}.log'), mode='w'),
        ],
    )
    log = logging.getLogger(__name__)

    K  = cfg.getint('rqvae', 'num_codebooks')
    CS = cfg.getint('rqvae', 'codebook_size')
    H  = cfg.getint('sequence', 'history_weeks')

    use_cuda = torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')
    log.info(f"Device: {device}  |  Mode: {mode_str}")

    model, tokenizer = load_model_and_tokenizer(cfg, device)

    # ── test data ─────────────────────────────────────────────
    test_path = os.path.join(cfg.get('paths', 'data_dir'), 'test.npz')
    test_data = np.load(test_path, allow_pickle=True)
    ts_codes  = test_data['ts_codes']
    log.info(f"Test stocks: {len(ts_codes):,}")

    rows       = []
    exact_hits = 0
    level_hits = np.zeros(K, dtype=int)
    total      = 0

    for ts_code in ts_codes:
        safe   = ts_code.replace('.', '_')
        ctx    = test_data[f"{safe}_ctx"]       # (H, K)
        labels = test_data[f"{safe}_labels"]    # (T, K)
        weeks  = test_data[f"{safe}_weeks"]     # (T,)

        # Build initial context string (strip PAD rows)
        ctx_list = [
            tuple(ctx[i].tolist())
            for i in range(len(ctx))
            if not (ctx[i] >= CS).all()
        ]

        for t, (week, gt) in enumerate(zip(weeks, labels)):
            # Build context string from sliding window
            context_str = '\n'.join(
                sid_tokens_str(np.array(c), K) for c in ctx_list
            )

            pred_codes = predict_next_sid(model, tokenizer, context_str, K, device)
            pred = np.array(pred_codes, dtype=np.int64)

            exact = int((pred == gt).all())
            exact_hits += exact
            total      += 1
            for k in range(K):
                level_hits[k] += int(pred[k] == gt[k])

            row = {'ts_code': ts_code, 'week': week, 'exact_match': exact}
            for k in range(K):
                lname = LEVEL_NAMES[k]
                row[f'pred_{lname}'] = int(pred[k])
                row[f'gt_{lname}']   = int(gt[k])
            rows.append(row)

            # Update sliding context
            next_codes = tuple(gt.tolist()) if args.tf else tuple(pred_codes)
            ctx_list.append(next_codes)
            if len(ctx_list) > H:
                ctx_list = ctx_list[-H:]

    # ── metrics ───────────────────────────────────────────────
    lines = [
        f"Mode         : {mode_str}",
        f"Total samples: {total:,}",
        f"Exact match  : {exact_hits / max(total, 1):.4f}  ({exact_hits}/{total})",
    ]
    for k in range(K):
        lname = LEVEL_NAMES[k]
        lines.append(
            f"  Level-{lname} acc : {level_hits[k] / max(total, 1):.4f}"
            f"  ({level_hits[k]}/{total})"
        )

    sep = '=' * 52
    log.info('\n' + sep + '\n' + '\n'.join(lines) + '\n' + sep)

    mpath = os.path.join(result_dir, f'llm_metrics_{suffix}.txt')
    with open(mpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    cpath = os.path.join(result_dir, f'llm_predictions_{suffix}.csv')
    pd.DataFrame(rows).to_csv(cpath, index=False)
    log.info(f"Predictions → {cpath}")
    log.info(f"Metrics     → {mpath}")


if __name__ == '__main__':
    main()
