"""
Training loop for the SFT SID-predictor.
"""
import glob
import logging
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    Trains SIDPredictor on next-step SID prediction.

    Loss: mean cross-entropy across all K codebook levels at the last
    sequence position (next-step prediction, teacher-forced during training).
    Schedule: linear warmup → cosine decay.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        cfg,
        device: torch.device,
        start_epoch: int = 0,
    ):
        self.model  = model.to(device)
        self.loader = train_loader
        self.device = device
        self.K      = model.K

        self.epochs      = cfg.getint('training', 'epochs')
        self.save_every  = cfg.getint('training', 'save_every')
        self.grad_clip   = cfg.getfloat('training', 'grad_clip')
        self.ckpt_dir    = cfg.get('paths', 'checkpoint_dir')
        self.start_epoch = start_epoch

        self.criterion = nn.CrossEntropyLoss()

        lr = cfg.getfloat('training', 'learning_rate')
        wd = cfg.getfloat('training', 'weight_decay')
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        total_steps  = self.epochs * len(train_loader)
        warmup_steps = cfg.getint('training', 'warmup_epochs') * len(train_loader)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

    def _loss(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        context : (B, H, K)
        target  : (B, K)
        """
        logits = self.model(context)                       # K × (B, H, vocab)
        loss = sum(
            self.criterion(logits[k][:, -1, :], target[:, k])
            for k in range(self.K)
        ) / self.K
        return loss

    def _save(self, epoch: int, loss: float):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt")
        torch.save({
            'epoch':     epoch,
            'loss':      loss,
            'model':     self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _prune(self, keep: int = 3):
        ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, 'ckpt_epoch_*.pt')))
        for old in ckpts[:-keep]:
            os.remove(old)
            logger.debug(f"Removed old checkpoint: {old}")

    def train(self):
        end_epoch = self.start_epoch + self.epochs
        for epoch in range(self.start_epoch, end_epoch):
            self.model.train()
            total_loss = 0.0

            for context, target in self.loader:
                context = context.to(self.device)
                target  = target.to(self.device)

                self.optimizer.zero_grad()
                loss = self._loss(context, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

            avg  = total_loss / len(self.loader)
            lr_v = self.scheduler.get_last_lr()[0]
            logger.info(f"epoch {epoch + 1:04d}/{end_epoch} | loss={avg:.5f} | lr={lr_v:.3e}")

            if (epoch + 1) % self.save_every == 0 or epoch + 1 == end_epoch:
                self._save(epoch + 1, avg)
                self._prune(keep=3)
