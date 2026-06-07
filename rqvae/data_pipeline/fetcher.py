"""
Data fetching and weekly embedding construction for all A-share stocks.

Flow:
  1. get_stock_list()         → list of all listed ts_codes
  2. process_all_stocks()     → per-stock .npz in raw_data_dir  (resumable)
  3. merge_to_dataset()       → train.npy / val.npy + meta CSVs in embedding_dir
"""
import os
import time
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import chinadata.ca_data as ts

logger = logging.getLogger(__name__)

OHLC_COLS  = ['open', 'high', 'low', 'close']
VOL_COLS   = ['vol', 'amount']
KEEP_COLS  = ['pct_chg']
INDICATORS = OHLC_COLS + VOL_COLS + KEEP_COLS   # 7 features, order fixed
_COL_IDX   = {c: i for i, c in enumerate(INDICATORS)}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def init_api(token: str):
    ts.set_token(token)
    return ts.pro_api(token)


def get_stock_list(pro) -> List[str]:
    """Return ts_codes for all currently listed A-share stocks."""
    df = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code')
    return df['ts_code'].tolist()


def _fetch_daily(pro, ts_code: str, start_date: str, end_date: str,
                 sleep: float) -> Optional[pd.DataFrame]:
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        time.sleep(sleep)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        return df
    except Exception as e:
        logger.warning(f"{ts_code} fetch failed: {e}")
        time.sleep(sleep * 3)
        return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_week(raw: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize a single week's (5, 7) raw price matrix in-place.
      OHLC      → (price / close_day1) - 1
      vol/amount → log1p, then intra-week z-score
      pct_chg   → keep as-is (already a daily return %)
    Returns None if close_day1 <= 0 (bad data).
    """
    g = raw.astype(np.float64).copy()
    close0 = g[0, _COL_IDX['close']]
    if close0 <= 0:
        return None

    for c in OHLC_COLS:
        g[:, _COL_IDX[c]] = g[:, _COL_IDX[c]] / close0 - 1.0

    for c in VOL_COLS:
        vals = np.log1p(np.maximum(g[:, _COL_IDX[c]], 0.0))
        mu, sigma = vals.mean(), vals.std()
        g[:, _COL_IDX[c]] = (vals - mu) / (sigma + 1e-8)

    return g.astype(np.float32)


# ---------------------------------------------------------------------------
# Weekly vector builder
# ---------------------------------------------------------------------------

def build_weekly_vectors(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Split daily data into ISO-week groups (only full 5-day weeks),
    normalize each week, flatten to (N, 35).
    Returns (vectors, week_labels).
    """
    df = df.copy()
    iso = df['trade_date'].dt.isocalendar()
    df['year_week'] = (
        iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)
    )

    week_sizes  = df.groupby('year_week').size()
    valid_weeks = week_sizes[week_sizes == 5].index

    samples, labels = [], []
    for week in valid_weeks:
        grp = df[df['year_week'] == week].sort_values('trade_date')
        raw    = grp[INDICATORS].values          # (5, 7)
        normed = _normalize_week(raw)
        if normed is None:
            continue
        samples.append(normed.flatten())          # (35,)
        labels.append(week)

    if not samples:
        return np.empty((0, 35), dtype=np.float32), []

    return np.array(samples, dtype=np.float32), labels


# ---------------------------------------------------------------------------
# Per-stock processing (resumable)
# ---------------------------------------------------------------------------

def process_all_stocks(
    pro,
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    raw_data_dir: str,
    sleep: float = 0.3,
    incremental: bool = False,
) -> None:
    """
    Fetch and preprocess every stock; save one .npz per stock to raw_data_dir.
    Already-processed files are skipped (safe to resume after interruption).

    Args:
        incremental: if True, extend existing per-stock data instead of skipping.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    total = len(stock_codes)

    for i, code in enumerate(stock_codes, 1):
        out_path = _stock_path(raw_data_dir, code)

        if os.path.exists(out_path) and not incremental:
            logger.debug(f"[{i}/{total}] {code} already processed, skip")
            continue

        logger.info(f"[{i}/{total}] {code}")
        df = _fetch_daily(pro, code, start_date, end_date, sleep=sleep)
        if df is None or len(df) < 5:
            continue

        vectors, labels = build_weekly_vectors(df)
        if len(vectors) == 0:
            continue

        if incremental and os.path.exists(out_path):
            existing = np.load(out_path, allow_pickle=True)
            existing_labels = set(existing['labels'].tolist())
            mask = [lbl not in existing_labels for lbl in labels]
            new_vecs   = vectors[[j for j, m in enumerate(mask) if m]]
            new_labels = [lbl for lbl, m in zip(labels, mask) if m]
            if len(new_vecs) == 0:
                continue
            vectors = np.concatenate([existing['vectors'], new_vecs], axis=0)
            labels  = existing['labels'].tolist() + new_labels

        np.savez_compressed(
            out_path,
            vectors=vectors,
            labels=np.array(labels),
            ts_code=np.array(code),
        )
        logger.debug(f"  saved {len(vectors)} weeks")


# ---------------------------------------------------------------------------
# Merge to train / val dataset
# ---------------------------------------------------------------------------

def merge_to_dataset(
    raw_data_dir: str,
    embedding_dir: str,
) -> None:
    """
    Merge all per-stock .npz files into a single training set:
      embedding_dir/train.npy       (N, 35)
      embedding_dir/train_meta.csv  ts_code + year_week per sample
    """
    os.makedirs(embedding_dir, exist_ok=True)

    all_vecs, all_meta = [], []

    npz_files = sorted(f for f in os.listdir(raw_data_dir) if f.endswith('.npz'))
    logger.info(f"Merging {len(npz_files)} stock files...")

    for fname in npz_files:
        data    = np.load(os.path.join(raw_data_dir, fname), allow_pickle=True)
        vecs    = data['vectors']           # (N, 35)
        labels  = data['labels'].tolist()
        ts_code = str(data['ts_code'])

        for vec, label in zip(vecs, labels):
            all_vecs.append(vec)
            all_meta.append({'ts_code': ts_code, 'year_week': label})

    all_vecs = np.array(all_vecs, dtype=np.float32)
    meta_df  = pd.DataFrame(all_meta)

    np.save(os.path.join(embedding_dir, 'train.npy'), all_vecs)
    meta_df.to_csv(os.path.join(embedding_dir, 'train_meta.csv'), index=False)

    logger.info(f"Dataset ready: {len(all_vecs):,} samples  →  {embedding_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stock_path(raw_data_dir: str, ts_code: str) -> str:
    safe = ts_code.replace('.', '_')
    return os.path.join(raw_data_dir, f"{safe}.npz")
