"""
End-to-end weekly inference: raw weekly embeddings → RQ-VAE SIDs → LLM beam
search → top-k next-week SIDs + historical avg returns → global stock ranking.

Time semantics
--------------
Run date (e.g. Wed 2026-06-10, ISO week 2026-W24):
  * target week  = the ISO week containing the run date (Mon 06-08 … Fri 06-12)
  * model input  = the latest 52 complete weeks STRICTLY BEFORE the target week
The script never looks at any data from the target week.

Pipeline
--------
1. Read per-stock weekly embeddings from rqvae raw dir (data/raw/*.npz)
2. Encode the context weeks with the RQ-VAE encoder (fresh, no sid_cache)
3. Build the same chat prompt as training; beam-search K=3 SID tokens with
   constrained decoding (level a → level b → level c)
4. For each candidate SID, look up its historical average intra-week return
   (mean of day5 normalized close over all historical samples with that SID)
5. Write per-stock JSON + global ranking CSV sorted by Σ prob_i × avg_return_i

Usage
-----
  python sft/inference.py --config sft/conf/sft.conf
  python sft/inference.py --date 20260610            # explicit as-of date
  python sft/inference.py --max-stocks 50            # smoke test
"""
import argparse
import configparser
import datetime as dt
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_SFT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SFT_DIR)
sys.path.insert(0, _SFT_DIR)   # sft/data_pipeline takes priority
sys.path.insert(1, _ROOT)      # rqvae.* resolved as namespace package

from data_pipeline.llm_dataset import LLMSIDDataset, sid_tokens_str, PAD_CODE
from predict_llm import load_model_and_tokenizer
from rqvae.model.rqvae import RQVAE

LEVEL_NAMES = 'abcdefghijklmnopqrstuvwxyz'
RET_COL     = 31   # day-5 normalized close = close5/close1 - 1 (intra-week return)


# ── week helpers ──────────────────────────────────────────────────────────────

def iso_week_label(d: dt.date) -> str:
    """date → '2026-W24'  (same format as rqvae fetcher)."""
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


# ── RQ-VAE ────────────────────────────────────────────────────────────────────

def load_rqvae(rq_cfg_path: str, device) -> RQVAE:
    rq = configparser.ConfigParser()
    rq.read(rq_cfg_path, encoding='utf-8')

    hidden_dims = [int(x) for x in rq.get('rqvae', 'hidden_dims').split(',')]
    model = RQVAE(
        input_dim         = rq.getint('data',   'input_dim'),
        hidden_dims       = hidden_dims,
        latent_dim        = rq.getint('rqvae',  'latent_dim'),
        num_codebooks     = rq.getint('rqvae',  'num_codebooks'),
        codebook_size     = rq.getint('rqvae',  'codebook_size'),
        ema_decay         = rq.getfloat('rqvae', 'ema_decay'),
        commitment_weight = rq.getfloat('rqvae', 'commitment_weight'),
    ).to(device)

    ckpt_dir = rq.get('paths', 'checkpoint_dir')
    ckpts    = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_epoch_*.pt')))
    if not ckpts:
        raise FileNotFoundError(f"No RQ-VAE checkpoint in {ckpt_dir}")
    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logging.getLogger(__name__).info(f"RQ-VAE loaded: {os.path.basename(ckpts[-1])}")
    return model, rq


# ── historical SID → avg return table ─────────────────────────────────────────

def ensure_sid_cache(rqvae: RQVAE, embedding_dir: str, device) -> None:
    """
    sid_cache.npy must stay row-aligned with train.npy. After an incremental
    preprocess run, merge_to_dataset rebuilds train.npy (rows shift), so the
    cache is re-encoded here whenever it is missing, older, or size-mismatched.
    Idempotent: a fresh cache is left untouched.
    """
    log        = logging.getLogger(__name__)
    train_path = os.path.join(embedding_dir, 'train.npy')
    cache_path = os.path.join(embedding_dir, 'sid_cache.npy')

    stale = (not os.path.exists(cache_path)
             or os.path.getmtime(train_path) > os.path.getmtime(cache_path)
             or len(np.load(cache_path, mmap_mode='r'))
                != len(np.load(train_path, mmap_mode='r')))
    if not stale:
        return

    log.info("sid_cache.npy is stale/missing — re-encoding train.npy with RQ-VAE …")
    data  = np.load(train_path, mmap_mode='r')
    codes = []
    for i in tqdm(range(0, len(data), 4096), desc='Rebuild sid_cache', unit='batch'):
        x = torch.from_numpy(np.array(data[i:i + 4096])).float().to(device)
        codes.append(rqvae.encode_to_sid(x).cpu().numpy())
    codes = np.concatenate(codes, axis=0)
    np.save(cache_path, codes)
    log.info(f"  sid_cache rebuilt: {len(codes):,} rows → {cache_path}")


def build_return_table(embedding_dir: str, CS: int) -> dict:
    """
    {sid_key: (avg_return, n_samples)} where sid_key = a*CS² + b*CS + c.
    avg_return = mean over all historical stock-weeks with that SID of the
    intra-week close return (normalized vector column RET_COL).
    """
    codes = np.load(os.path.join(embedding_dir, 'sid_cache.npy'))            # (N, K)
    rets  = np.load(os.path.join(embedding_dir, 'train.npy'),
                    mmap_mode='r')[:, RET_COL]                               # (N,)

    key = (codes[:, 0].astype(np.int64) * CS + codes[:, 1]) * CS + codes[:, 2]
    g   = pd.DataFrame({'key': key, 'ret': np.asarray(rets)}) \
            .groupby('key')['ret'].agg(['mean', 'count'])
    return {int(k): (float(m), int(c)) for k, m, c in
            zip(g.index, g['mean'], g['count'])}


# ── prompt builder (reuses LLMSIDDataset's fast integer tables) ───────────────

class PromptBuilder:
    def __init__(self, tokenizer, K: int, max_len: int):
        dummy_ctx = np.full((1, 1, K), PAD_CODE, dtype=np.int64)
        dummy_tgt = np.zeros((1, K), dtype=np.int64)
        self._ds  = LLMSIDDataset(dummy_ctx, dummy_tgt, tokenizer, K, max_len)
        self.max_len = max_len

    def build(self, ctx: np.ndarray) -> list:
        """(H, K) codes (PAD rows allowed) → prompt token-id list (left-truncated)."""
        ids = self._ds.prefix_ids + self._ds._content_ids(ctx) + self._ds.suffix_ids
        if len(ids) > self.max_len:
            ids = ids[len(ids) - self.max_len:]
        return ids


# ── constrained beam search ───────────────────────────────────────────────────

@torch.no_grad()
def beam_predict(model, prompts: list, level_ids: list, pad_id: int,
                 K: int, beam_k: int, device):
    """
    prompts   : list of token-id lists (one per stock)
    level_ids : [list_of_a_ids, list_of_b_ids, list_of_c_ids]
    Returns   : (codes, probs)
                codes (B, beam_k, K) int  |  probs (B, beam_k) float
    """
    B    = len(prompts)
    maxL = max(len(p) for p in prompts)
    input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxL), dtype=torch.long)
    for i, p in enumerate(prompts):                       # left padding
        input_ids[i, maxL - len(p):] = torch.tensor(p, dtype=torch.long)
        attn_mask[i, maxL - len(p):] = 1
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    allowed = [torch.tensor(ids) for ids in level_ids]

    def _prefix_fn(batch_id, seq):
        g = seq.shape[-1] - maxL                          # tokens generated so far
        return allowed[g].tolist() if g < K else allowed[-1].tolist()

    out = model.generate(
        input_ids                  = input_ids,
        attention_mask             = attn_mask,
        max_new_tokens             = K,
        num_beams                  = beam_k,
        num_return_sequences       = beam_k,
        do_sample                  = False,
        length_penalty             = 0.0,                 # raw sum of log-probs
        early_stopping             = False,
        pad_token_id               = pad_id,
        prefix_allowed_tokens_fn   = _prefix_fn,
        renormalize_logits         = True,   # prob over the 128 valid SID tokens
        return_dict_in_generate    = True,
        output_scores              = True,
    )

    new_tok = out.sequences[:, maxL:].reshape(B, beam_k, -1)[:, :, :K]   # (B, k, K)
    probs   = torch.exp(out.sequences_scores).reshape(B, beam_k)         # joint prob
    return new_tok.cpu().numpy(), probs.cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Weekly SID inference + ranking')
    parser.add_argument('--config',     default='sft/conf/sft.conf')
    parser.add_argument('--date',       default=None,
                        help='As-of date YYYYMMDD (default: today)')
    parser.add_argument('--max-stocks', type=int, default=0,
                        help='Limit number of stocks (0 = all, for smoke tests)')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    log_dir    = cfg.get('paths', 'log_dir')
    result_dir = cfg.get('paths', 'result_dir')
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(log_dir, 'inference.log'), mode='a')],
    )
    log = logging.getLogger(__name__)

    K        = cfg.getint('rqvae', 'num_codebooks')
    CS       = cfg.getint('rqvae', 'codebook_size')
    H        = cfg.getint('sequence', 'history_weeks')
    min_ctx  = cfg.getint('sequence', 'min_context_weeks')
    max_len  = cfg.getint('llm', 'max_seq_len')
    beam_k   = cfg.getint('inference', 'beam_k',        fallback=5)
    inf_bs   = cfg.getint('inference', 'batch_size',    fallback=32)
    rq_conf  = cfg.get('inference',    'rqvae_config',  fallback='rqvae/conf/common.conf')
    excl     = tuple(p.strip() for p in
                     cfg.get('inference', 'exclude_prefixes', fallback='').split(',')
                     if p.strip())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device} | beam_k={beam_k} | batch_size={inf_bs}")

    # ── time window ───────────────────────────────────────────
    asof = (dt.datetime.strptime(args.date, '%Y%m%d').date()
            if args.date else dt.date.today())
    target_week    = iso_week_label(asof)                     # prediction window
    last_week_done = iso_week_label(asof - dt.timedelta(days=7))
    log.info(f"As-of {asof}  →  target week {target_week}  "
             f"(input = last {H} weeks < {target_week})")

    # ── RQ-VAE + raw weekly embeddings ────────────────────────
    rqvae, rq_cfg = load_rqvae(rq_conf, device)
    raw_dir       = rq_cfg.get('paths', 'raw_data_dir')
    embedding_dir = rq_cfg.get('paths', 'embedding_dir')

    npz_files = sorted(glob.glob(os.path.join(raw_dir, '*.npz')))
    if excl:
        n_before  = len(npz_files)
        npz_files = [p for p in npz_files
                     if not os.path.basename(p).startswith(excl)]
        log.info(f"Excluded {n_before - len(npz_files):,} stocks "
                 f"with prefixes {list(excl)}")
    if args.max_stocks:
        npz_files = npz_files[:args.max_stocks]
    if not npz_files:
        log.error(f"No raw stock files in {raw_dir} — run rqvae/preprocess.py first")
        sys.exit(1)
    log.info(f"Raw stock files: {len(npz_files):,}")

    # Collect per-stock context embeddings (weeks strictly before target week)
    stocks, all_vecs, spans = [], [], []      # spans = (start, end) into all_vecs
    latest_week_seen = ''
    for path in npz_files:
        d       = np.load(path, allow_pickle=True)
        labels  = d['labels'].tolist()
        if labels:
            latest_week_seen = max(latest_week_seen, max(labels))
        keep = [i for i, w in enumerate(labels) if w < target_week]
        if len(keep) < min_ctx:
            continue
        keep = keep[-H:]                                       # latest H weeks
        s    = len(all_vecs)
        all_vecs.extend(d['vectors'][keep])
        spans.append((s, len(all_vecs)))
        stocks.append(str(d['ts_code']))

    log.info(f"Stocks with ≥{min_ctx} context weeks: {len(stocks):,}")
    if latest_week_seen < last_week_done:
        log.warning(f"Raw data is STALE: latest week on disk = {latest_week_seen}, "
                    f"expected ≥ {last_week_done}. "
                    f"Run `python rqvae/preprocess.py --incremental` first!")

    # ── encode embeddings → SIDs (fresh) ──────────────────────
    all_vecs = np.asarray(all_vecs, dtype=np.float32)
    sids = []
    for i in range(0, len(all_vecs), 4096):
        x = torch.from_numpy(all_vecs[i:i + 4096]).to(device)
        sids.append(rqvae.encode_to_sid(x).cpu().numpy())
    sids = np.concatenate(sids, axis=0)                        # (M, K)
    log.info(f"Encoded {len(sids):,} stock-weeks → SIDs")

    # ── LLM + prompt builder ──────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, device)
    builder = PromptBuilder(tokenizer, K, max_len)
    pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id

    level_ids = [
        [tokenizer.convert_tokens_to_ids(f'<{LEVEL_NAMES[k]}_{i}>') for i in range(CS)]
        for k in range(K)
    ]
    id2code = {tid: (k, i) for k in range(K)
               for i, tid in enumerate(level_ids[k])}

    # ── return table ──────────────────────────────────────────
    ensure_sid_cache(rqvae, embedding_dir, device)
    ret_table = build_return_table(embedding_dir, CS)
    log.info(f"Return table: {len(ret_table):,} unique historical SIDs")

    # ── batched beam search ───────────────────────────────────
    results = []
    for b0 in tqdm(range(0, len(stocks), inf_bs),
                   desc='Beam search', unit='batch',
                   total=(len(stocks) + inf_bs - 1) // inf_bs):
        batch_idx = range(b0, min(b0 + inf_bs, len(stocks)))
        prompts   = []
        for i in batch_idx:
            s, e = spans[i]
            ctx  = np.full((e - s, K), PAD_CODE, dtype=np.int64)
            ctx[:] = sids[s:e]
            prompts.append(builder.build(ctx))

        codes, probs = beam_predict(model, prompts, level_ids, pad_id,
                                    K, beam_k, device)

        for j, i in enumerate(batch_idx):
            preds, score = [], 0.0
            for r in range(beam_k):
                # token ids → (level, code); fall back to 0 on anomaly
                trip = [id2code.get(int(t), (k, 0))[1]
                        for k, t in enumerate(codes[j, r])]
                key  = (trip[0] * CS + trip[1]) * CS + trip[2]
                prob = float(probs[j, r])
                ret, n_hist = ret_table.get(key, (None, 0))
                if ret is not None:
                    score += prob * ret
                preds.append({
                    'sid':            sid_tokens_str(np.array(trip), K),
                    'beamsearch_prob': float(f'{prob:.4g}'),   # keep tiny probs visible
                    'avg_return':      None if ret is None else round(ret, 6),
                    'n_hist':          n_hist,
                })
            results.append({
                'ts_code':     stocks[i],
                'target_week': target_week,
                'predictions': preds,
                'score':       float(f'{score:.4g}'),
            })

    # ── outputs ───────────────────────────────────────────────
    results.sort(key=lambda r: r['score'], reverse=True)

    json_path = os.path.join(result_dir, f'inference_{target_week}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    rank_rows = [{
        'rank':        n + 1,
        'ts_code':     r['ts_code'],
        'score':       r['score'],
        'sid_top1':    r['predictions'][0]['sid'],
        'prob_top1':   r['predictions'][0]['beamsearch_prob'],
        'return_top1': r['predictions'][0]['avg_return'],
    } for n, r in enumerate(results)]
    csv_path = os.path.join(result_dir, f'inference_rank_{target_week}.csv')
    pd.DataFrame(rank_rows).to_csv(csv_path, index=False)

    log.info(f"Top 10 by prob-weighted return:")
    for r in rank_rows[:10]:
        log.info(f"  #{r['rank']:<3} {r['ts_code']}  score={r['score']:+.4f}  "
                 f"top1={r['sid_top1']} (p={r['prob_top1']:.3f}, "
                 f"ret={r['return_top1']})")
    log.info(f"Detail JSON → {json_path}")
    log.info(f"Ranking CSV → {csv_path}")


if __name__ == '__main__':
    main()
