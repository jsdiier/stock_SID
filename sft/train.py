"""
Entry point: train the SFT SID-predictor model.

The model learns to predict the next week's SID (3 discrete codes) from a
fixed-length context window of past weekly SIDs.

Usage
-----
  # Fresh training
  python train.py

  # Resume from the latest checkpoint (model weights only; schedule restarts)
  python train.py --resume

  # Use a different config
  python train.py --config conf/sft.conf
"""
import argparse
import configparser
import glob
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline.dataset import SIDDataset
from model.sft_model import build_model
from trainer.sft_trainer import SFTTrainer


def _setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='a'),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description='Train SFT SID predictor')
    parser.add_argument('--config', default='conf/sft.conf')
    parser.add_argument('--resume', action='store_true',
                        help='Load weights from the latest checkpoint before training')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    _setup_logging(cfg.get('paths', 'log_dir'))
    log = logging.getLogger(__name__)

    # ── device ────────────────────────────────────────────────
    use_cuda = cfg.get('training', 'device') == 'cuda' and torch.cuda.is_available()
    device   = torch.device('cuda' if use_cuda else 'cpu')
    log.info(f"Device: {device}")

    # ── dataset ───────────────────────────────────────────────
    data_path = os.path.join(cfg.get('paths', 'data_dir'), 'train.npz')
    if not os.path.exists(data_path):
        log.error(f"Training data not found: {data_path}")
        log.error("Run  python build_dataset.py  first.")
        sys.exit(1)

    log.info(f"Loading training data: {data_path}")
    d       = np.load(data_path)
    dataset = SIDDataset(d['contexts'], d['targets'])
    loader  = DataLoader(
        dataset,
        batch_size = cfg.getint('training', 'batch_size'),
        shuffle    = True,
        num_workers= 0,
        pin_memory = (device.type == 'cuda'),
        drop_last  = True,
    )
    log.info(f"  {len(dataset):,} samples  |  {len(loader)} batches/epoch")

    # ── model ─────────────────────────────────────────────────
    model       = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"SIDPredictor  params: {total_params:,}")

    start_epoch = 0
    if args.resume:
        ckpt_dir = cfg.get('paths', 'checkpoint_dir')
        ckpts    = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_epoch_*.pt')))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model'])
            start_epoch = ckpt.get('epoch', 0)
            log.info(f"Resumed from {os.path.basename(ckpts[-1])} (epoch {start_epoch})")
        else:
            log.warning(f"No checkpoint found in {ckpt_dir}, starting fresh.")

    # ── train ─────────────────────────────────────────────────
    trainer = SFTTrainer(model, loader, cfg, device, start_epoch=start_epoch)
    trainer.train()
    log.info("Training complete.")


if __name__ == '__main__':
    main()
