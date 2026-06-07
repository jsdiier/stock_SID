"""
Build SFT train and test datasets from the RQ-VAE SID cache.

Reads
-----
  sid_cache.npy   + train_meta.csv   (produced by the RQ-VAE pipeline)

Writes
------
  <data_dir>/train.npz
      contexts    : (N, H, K) int64  — left-padded context windows
      targets     : (N, K)   int64  — next-week ground-truth codes
      ts_codes    : (N,)     str
      week_labels : (N,)     str    — the target week for each sample

  <data_dir>/test.npz
      ts_codes            : (M,)    str  — stocks that have test-range data
      {safe_code}_ctx     : (H, K)  int64  — context before test_start
      {safe_code}_labels  : (T, K)  int64  — actual codes for test weeks
      {safe_code}_weeks   : (T,)    str    — test week strings

  where safe_code = ts_code with '.' replaced by '_'.

Usage
-----
  python build_dataset.py
  python build_dataset.py --config conf/sft.conf
"""
import argparse
import configparser
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline.date_utils import weeks_in_range
from data_pipeline.sid_loader import load_stock_sids, filter_universe


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_key(ts_code: str) -> str:
    """'000001.SZ' → '000001_SZ'  (safe as npz array key)."""
    return ts_code.replace('.', '_')


def _pad_context(hist: list, history_weeks: int, K: int, pad_code: int) -> np.ndarray:
    """Left-pad a list of code arrays to shape (history_weeks, K)."""
    if len(hist) >= history_weeks:
        hist = hist[-history_weeks:]
    else:
        n_pad = history_weeks - len(hist)
        pad   = [np.full(K, pad_code, dtype=np.int64)] * n_pad
        hist  = pad + hist
    return np.array(hist, dtype=np.int64)   # (H, K)


# ── dataset builders ──────────────────────────────────────────────────────────

def build_train(stock_sids: dict, train_weeks_set: set,
                history_weeks: int, min_context: int,
                pad_code: int, K: int):
    """
    Sliding-window next-step prediction samples for all target weeks
    that fall inside train_weeks_set.

    Returns (contexts, targets, ts_codes, week_labels) as numpy arrays.
    """
    contexts, targets, ts_codes, week_labels = [], [], [], []

    for ts_code, seq in stock_sids.items():
        for i, (week, codes) in enumerate(seq):
            if week not in train_weeks_set:
                continue

            hist = [c for _, c in seq[:i]]          # strictly before target week
            if len(hist) < min_context:
                continue

            ctx = _pad_context(hist, history_weeks, K, pad_code)
            contexts.append(ctx)
            targets.append(codes.astype(np.int64))
            ts_codes.append(ts_code)
            week_labels.append(week)

    return (
        np.array(contexts,    dtype=np.int64),
        np.array(targets,     dtype=np.int64),
        np.array(ts_codes),
        np.array(week_labels),
    )


def build_test(stock_sids: dict, test_weeks_list: list,
               history_weeks: int, pad_code: int, K: int) -> dict:
    """
    Per-stock test data: context before the test window + ground-truth labels.

    Returns a dict suitable for np.savez_compressed.
    Each stock contributes three arrays keyed by _safe_key(ts_code):
      {safe}_ctx    : (H, K)   context before test_start
      {safe}_labels : (T, K)   actual codes for test weeks the stock has
      {safe}_weeks  : (T,)     test week strings
    Plus:
      ts_codes : (M,) array of original ts_codes included in the test set
    """
    test_weeks_set  = set(test_weeks_list)
    first_test_week = test_weeks_list[0]   # sorted, so [0] is earliest

    out      = {}
    included = []

    for ts_code, seq in stock_sids.items():
        test_entries = [(w, c) for w, c in seq if w in test_weeks_set]
        if not test_entries:
            continue

        hist = [c for w, c in seq if w < first_test_week]
        ctx  = _pad_context(hist, history_weeks, K, pad_code)

        labels  = np.array([c for _, c in test_entries], dtype=np.int64)   # (T, K)
        t_weeks = np.array([w for w, _ in test_entries])                    # (T,) str

        safe = _safe_key(ts_code)
        out[f"{safe}_ctx"]    = ctx
        out[f"{safe}_labels"] = labels
        out[f"{safe}_weeks"]  = t_weeks
        included.append(ts_code)

    out['ts_codes'] = np.array(included)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build SFT train/test datasets')
    parser.add_argument('--config', default='conf/sft.conf')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    log = logging.getLogger(__name__)

    # ── config ────────────────────────────────────────────────
    train_start = cfg.get('dates', 'train_start')
    train_end   = cfg.get('dates', 'train_end')
    test_start  = cfg.get('dates', 'test_start')
    test_end    = cfg.get('dates', 'test_end')
    universe    = cfg.get('stocks', 'universe')
    max_stocks  = cfg.getint('stocks', 'max_stocks')
    H           = cfg.getint('sequence', 'history_weeks')
    min_ctx     = cfg.getint('sequence', 'min_context_weeks')
    K           = cfg.getint('rqvae',   'num_codebooks')
    PAD         = cfg.getint('rqvae',   'codebook_size')   # 128

    sid_cache   = cfg.get('paths', 'sid_cache')
    meta_path   = cfg.get('paths', 'train_meta')
    data_dir    = cfg.get('paths', 'data_dir')
    os.makedirs(data_dir, exist_ok=True)

    # ── load & filter ─────────────────────────────────────────
    log.info("Loading SID sequences …")
    stock_sids = load_stock_sids(sid_cache, meta_path)
    log.info(f"  Total stocks in cache: {len(stock_sids):,}")
    stock_sids = filter_universe(stock_sids, universe, max_stocks)
    log.info(f"  After universe filter: {len(stock_sids):,} stocks")

    # ── train dataset ─────────────────────────────────────────
    train_weeks = weeks_in_range(train_start, train_end)
    train_set   = set(train_weeks)
    log.info(f"Building train dataset  [{train_start} → {train_end}]  ({len(train_weeks)} weeks) …")

    ctx, tgt, tsc, wlb = build_train(stock_sids, train_set, H, min_ctx, PAD, K)

    if len(ctx) == 0:
        log.warning("No training samples generated — check date range and stock universe.")
    else:
        train_path = os.path.join(data_dir, 'train.npz')
        np.savez_compressed(train_path,
                            contexts=ctx, targets=tgt,
                            ts_codes=tsc, week_labels=wlb)
        log.info(f"  → {len(ctx):,} samples  saved: {train_path}")
        log.info(f"  context shape: {ctx.shape}  |  target shape: {tgt.shape}")

    # ── test dataset ──────────────────────────────────────────
    test_weeks = weeks_in_range(test_start, test_end)
    log.info(f"Building test dataset   [{test_start} → {test_end}]  ({len(test_weeks)} weeks) …")

    test_data = build_test(stock_sids, test_weeks, H, PAD, K)

    if len(test_data['ts_codes']) == 0:
        log.warning("No test stocks found — check test date range.")
    else:
        test_path = os.path.join(data_dir, 'test.npz')
        np.savez_compressed(test_path, **test_data)
        log.info(f"  → {len(test_data['ts_codes']):,} stocks  saved: {test_path}")

    log.info("Done.")


if __name__ == '__main__':
    main()
