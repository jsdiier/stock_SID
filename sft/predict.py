"""
Predict next-week SIDs for each stock in the test set and evaluate accuracy.

Two prediction modes
--------------------
  autoregressive  (default)
      Each predicted SID is appended to the context for the next step.
      This simulates real-world usage where ground truth is unknown.

  teacher-forced  (--teacher-force)
      The actual ground-truth SID is appended at each step.
      Upper-bound for the autoregressive mode; useful for model diagnostics.

Outputs
-------
  <result_dir>/predictions_ar.csv   or   predictions_tf.csv
      ts_code, week, pred_a..pred_c, gt_a..gt_c, exact_match

  <result_dir>/metrics.txt
      Summary: exact-match rate and per-level accuracy

Usage
-----
  python predict.py
  python predict.py --teacher-force
  python predict.py --config conf/sft.conf
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

sys.path.insert(0, os.path.dirname(__file__))
from model.sft_model import build_model


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(cfg, device):
    model = build_model(cfg).to(device)

    ckpt_dir = cfg.get('paths', 'checkpoint_dir')
    ckpts    = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_epoch_*.pt')))
    if not ckpts:
        raise FileNotFoundError(
            f"No SFT checkpoints in {ckpt_dir}\n"
            "Run  python train.py  first."
        )

    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()

    log = logging.getLogger(__name__)
    log.info(f"Model loaded: {os.path.basename(ckpts[-1])}  "
             f"(trained epoch {ckpt.get('epoch','?')})")
    return model


def _predict_stock(model, ctx_np, labels_np, teacher_force, device):
    """
    Autoregressively predict T steps given an initial context.

    Parameters
    ----------
    ctx_np        : (H, K) int64  — initial context window
    labels_np     : (T, K) int64  — ground-truth test codes
    teacher_force : bool

    Returns
    -------
    preds : (T, K) int64
    """
    context = torch.from_numpy(ctx_np).long().to(device)   # (H, K)
    preds   = []

    for t in range(len(labels_np)):
        pred = model.predict_next(context)                  # (K,) on device
        preds.append(pred.cpu().numpy())

        next_code = (
            torch.from_numpy(labels_np[t]).long().to(device)
            if teacher_force
            else pred
        )
        # Slide the context window forward by one step
        context = torch.cat([context[1:], next_code.unsqueeze(0)], dim=0)

    return np.array(preds, dtype=np.int64)                  # (T, K)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Predict and evaluate SFT model')
    parser.add_argument('--config',        default='conf/sft.conf')
    parser.add_argument('--teacher-force', action='store_true',
                        help='Feed ground-truth SID at each step (oracle / upper-bound mode)')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    result_dir = cfg.get('paths', 'result_dir')
    log_dir    = cfg.get('paths', 'log_dir')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)

    mode_str = 'teacher-forced' if args.teacher_force else 'autoregressive'
    suffix   = 'tf'             if args.teacher_force else 'ar'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, f'predict_{suffix}.log'), mode='w'),
        ],
    )
    log = logging.getLogger(__name__)

    use_cuda = cfg.get('training', 'device') == 'cuda' and torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')
    log.info(f"Device: {device}  |  Mode: {mode_str}")

    # ── model ─────────────────────────────────────────────────
    model = _load_model(cfg, device)
    K     = model.K

    # ── test data ─────────────────────────────────────────────
    test_path = os.path.join(cfg.get('paths', 'data_dir'), 'test.npz')
    if not os.path.exists(test_path):
        log.error(f"Test data not found: {test_path}")
        log.error("Run  python build_dataset.py  first.")
        sys.exit(1)

    test_data = np.load(test_path, allow_pickle=True)
    ts_codes  = test_data['ts_codes']
    log.info(f"Loaded test data: {len(ts_codes):,} stocks")

    # ── predict ───────────────────────────────────────────────
    rows       = []
    exact_hits = 0
    level_hits = np.zeros(K, dtype=int)
    total      = 0

    for ts_code in ts_codes:
        safe   = ts_code.replace('.', '_')
        ctx    = test_data[f"{safe}_ctx"]        # (H, K)
        labels = test_data[f"{safe}_labels"]     # (T, K)
        weeks  = test_data[f"{safe}_weeks"]      # (T,)

        preds = _predict_stock(model, ctx, labels, args.teacher_force, device)

        for t, (week, pred, gt) in enumerate(zip(weeks, preds, labels)):
            exact = int((pred == gt).all())
            exact_hits += exact
            total      += 1
            for k in range(K):
                level_hits[k] += int(pred[k] == gt[k])

            row = {'ts_code': ts_code, 'week': week, 'exact_match': exact}
            for k in range(K):
                lname = chr(ord('a') + k)
                row[f'pred_{lname}'] = int(pred[k])
                row[f'gt_{lname}']   = int(gt[k])
            rows.append(row)

    # ── metrics ───────────────────────────────────────────────
    lines = [
        f"Mode         : {mode_str}",
        f"Total samples: {total:,}",
        f"Exact match  : {exact_hits / max(total,1):.4f}  ({exact_hits}/{total})",
    ]
    for k in range(K):
        lname = chr(ord('a') + k)
        lines.append(f"  Level-{lname} acc: {level_hits[k] / max(total,1):.4f}"
                     f"  ({level_hits[k]}/{total})")

    sep = '=' * 52
    log.info('\n' + sep + '\n' + '\n'.join(lines) + '\n' + sep)

    metrics_path = os.path.join(result_dir, f'metrics_{suffix}.txt')
    with open(metrics_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log.info(f"Metrics saved: {metrics_path}")

    # ── detailed results ──────────────────────────────────────
    out_csv = os.path.join(result_dir, f'predictions_{suffix}.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info(f"Predictions saved: {out_csv}")


if __name__ == '__main__':
    main()
