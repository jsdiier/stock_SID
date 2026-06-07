import configparser
import glob
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.rqvae import RQVAE
from data_pipeline.dataset import WeeklyEmbeddingDataset

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, cfg: configparser.ConfigParser):
        self.cfg = cfg

        # device
        requested = cfg.get('training', 'device', fallback='cuda')
        self.device = torch.device(
            'cuda' if requested == 'cuda' and torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")

        # paths
        self.embedding_dir  = cfg.get('paths', 'embedding_dir')
        self.checkpoint_dir = cfg.get('paths', 'checkpoint_dir')
        self.log_dir        = cfg.get('paths', 'log_dir')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir,        exist_ok=True)

        # model
        hidden_dims = [int(x) for x in cfg.get('rqvae', 'hidden_dims').split(',')]
        self.model = RQVAE(
            input_dim         = cfg.getint('data',   'input_dim'),
            hidden_dims       = hidden_dims,
            latent_dim        = cfg.getint('rqvae',  'latent_dim'),
            num_codebooks     = cfg.getint('rqvae',  'num_codebooks'),
            codebook_size     = cfg.getint('rqvae',  'codebook_size'),
            ema_decay         = cfg.getfloat('rqvae', 'ema_decay'),
            commitment_weight = cfg.getfloat('rqvae', 'commitment_weight'),
        ).to(self.device)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = cfg.getfloat('training', 'learning_rate'),
            weight_decay = cfg.getfloat('training', 'weight_decay'),
        )

        self.epochs        = cfg.getint('training', 'epochs')
        self.grad_clip     = cfg.getfloat('training', 'grad_clip', fallback=1.0)
        self.save_every    = cfg.getint('training', 'save_every_n_epochs')
        self.keep_last     = cfg.getint('training', 'keep_last_n_checkpoints')
        self.warmup_epochs = cfg.getint('training', 'warmup_epochs', fallback=5)
        self.batch_size    = cfg.getint('training', 'batch_size')

        # dataloader（全量训练，无 val split）
        train_path = os.path.join(self.embedding_dir, 'train.npy')
        self.train_loader = DataLoader(
            WeeklyEmbeddingDataset(train_path),
            batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == 'cuda'),
        )

        # lr scheduler (cosine with linear warmup)
        total_steps  = self.epochs * len(self.train_loader)
        warmup_steps = self.warmup_epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_with_warmup(step, warmup_steps, total_steps),
        )

        # log file
        self.log_path = os.path.join(self.log_dir, 'train.log')

        # wandb
        self._wandb = None
        if cfg.getboolean('wandb', 'enabled', fallback=False):
            self._wandb = _init_wandb(cfg)

    # ------------------------------------------------------------------

    def train(self, resume: bool = False) -> None:
        start_epoch = 0
        if resume:
            ckpt = self._latest_checkpoint()
            if ckpt:
                start_epoch = self._load_checkpoint(ckpt) + 1
                logger.info(f"Resumed from {ckpt}, starting epoch {start_epoch}")
        elif self.cfg.get('training', 'mode') == 'full':
            for f in glob.glob(os.path.join(self.checkpoint_dir, 'ckpt_epoch_*.pt')):
                os.remove(f)

        try:
            for epoch in range(start_epoch, self.epochs):
                metrics = self._run_epoch(epoch)
                self._log(epoch, metrics)

                if (epoch + 1) % self.save_every == 0 or epoch == self.epochs - 1:
                    self._save_checkpoint(epoch)
                    self._prune_checkpoints()
        finally:
            if self._wandb:
                self._wandb.finish()

    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = recon_loss_sum = vq_loss_sum = 0.0
        util_sum   = [0.0] * self.model.quantizer.num_levels
        n_batches  = 0

        for x in self.train_loader:
            x = x.to(self.device)
            loss, recon, vq, codes = self.model.compute_loss(x)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss     += loss.item()
            recon_loss_sum += recon.item()
            vq_loss_sum    += vq.item()
            util = self.model.quantizer.codebook_utilisation(codes)
            for lvl, u in enumerate(util):
                util_sum[lvl] += u
            n_batches += 1

        n = max(n_batches, 1)
        return {
            'total': total_loss     / n,
            'recon': recon_loss_sum / n,
            'vq':    vq_loss_sum    / n,
            'util':  [u / n for u in util_sum],
            'lr':    self.scheduler.get_last_lr()[0],
        }

    def _log(self, epoch: int, m: dict) -> None:
        util_str = ' '.join(f'{u:.2f}' for u in m['util'])
        msg = (
            f"epoch {epoch:04d} | "
            f"total={m['total']:.5f} recon={m['recon']:.5f} vq={m['vq']:.5f} | "
            f"cb_util=[{util_str}] | lr={m['lr']:.2e}"
        )
        logger.info(msg)
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')

        if self._wandb:
            log_dict = {
                'train/loss_total': m['total'],
                'train/loss_recon': m['recon'],
                'train/loss_vq':    m['vq'],
                'train/lr':         m['lr'],
            }
            for lvl, u in enumerate(m['util']):
                log_dict[f'codebook/util_level{lvl}'] = u
            self._wandb.log(log_dict, step=epoch)

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.checkpoint_dir, f'ckpt_epoch_{epoch:04d}.pt')
        torch.save({
            'epoch':     epoch,
            'model':     self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        return ckpt['epoch']

    def _latest_checkpoint(self) -> Optional[str]:
        files = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ckpt_epoch_*.pt')))
        return files[-1] if files else None

    def _prune_checkpoints(self) -> None:
        files = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ckpt_epoch_*.pt')))
        for old in files[:-self.keep_last]:
            os.remove(old)


# ---------------------------------------------------------------------------

def _init_wandb(cfg: configparser.ConfigParser):
    import wandb
    api_key  = cfg.get('wandb', 'api_key',  fallback=None)
    project  = cfg.get('wandb', 'project',  fallback='stock-rqvae')
    run_name = cfg.get('wandb', 'run_name', fallback=None)

    if api_key:
        os.environ['WANDB_API_KEY'] = api_key

    # 上传 hyperparameters
    config = {
        'input_dim':         cfg.getint('data',     'input_dim'),
        'hidden_dims':       cfg.get('rqvae',        'hidden_dims'),
        'latent_dim':        cfg.getint('rqvae',     'latent_dim'),
        'num_codebooks':     cfg.getint('rqvae',     'num_codebooks'),
        'codebook_size':     cfg.getint('rqvae',     'codebook_size'),
        'ema_decay':         cfg.getfloat('rqvae',   'ema_decay'),
        'commitment_weight': cfg.getfloat('rqvae',   'commitment_weight'),
        'batch_size':        cfg.getint('training',  'batch_size'),
        'epochs':            cfg.getint('training',  'epochs'),
        'learning_rate':     cfg.getfloat('training','learning_rate'),
        'warmup_epochs':     cfg.getint('training',  'warmup_epochs', fallback=5),
    }

    run = wandb.init(project=project, name=run_name, config=config)
    logger.info(f"wandb run: {run.url}")
    return run


def _cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
