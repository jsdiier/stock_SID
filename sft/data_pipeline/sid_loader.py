"""
Load per-stock, time-ordered SID sequences from the RQ-VAE cache.
"""
import numpy as np
import pandas as pd


def load_stock_sids(sid_cache_path: str, train_meta_path: str) -> dict:
    """
    Load all SID sequences and group them by stock, sorted chronologically.

    Parameters
    ----------
    sid_cache_path  : path to sid_cache.npy  — shape (N, K) int64
    train_meta_path : path to train_meta.csv — columns: ts_code, year_week

    Returns
    -------
    dict  {ts_code: [(year_week, codes_array), ...]}
          where codes_array is shape (K,) int64, and entries are sorted by year_week.
    """
    codes = np.load(sid_cache_path)            # (N, K) int64
    meta  = pd.read_csv(train_meta_path)       # ts_code, year_week

    if len(codes) != len(meta):
        raise ValueError(
            f"Row count mismatch: sid_cache has {len(codes)}, train_meta has {len(meta)}"
        )

    meta = meta.copy()
    meta['_row'] = np.arange(len(meta))

    stock_dict: dict = {}
    for ts_code, grp in meta.groupby('ts_code', sort=False):
        grp  = grp.sort_values('year_week').reset_index(drop=True)
        idxs = grp['_row'].values
        weeks = grp['year_week'].tolist()
        stock_codes = codes[idxs]              # (T, K)
        stock_dict[ts_code] = list(zip(weeks, stock_codes))

    return stock_dict


def filter_universe(stock_sids: dict, universe: str, max_stocks: int) -> dict:
    """
    Apply universe filter and optional stock count cap.

    Parameters
    ----------
    universe   : 'all' or comma-separated ts_codes
    max_stocks : if > 0, cap the result to the first max_stocks stocks
    """
    if universe.strip().lower() != 'all':
        keep = {s.strip() for s in universe.split(',')}
        stock_sids = {k: v for k, v in stock_sids.items() if k in keep}

    if max_stocks > 0:
        stock_sids = dict(list(stock_sids.items())[:max_stocks])

    return stock_sids
