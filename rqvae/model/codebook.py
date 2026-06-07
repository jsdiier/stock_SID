import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCodebook(nn.Module):
    """
    Single-level VQ codebook with Exponential Moving Average (EMA) updates.
    EMA keeps codebook updates off the gradient graph → more stable than
    straight gradient descent on the codebook.
    """

    def __init__(self, codebook_size: int, embed_dim: int,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(codebook_size, embed_dim)
        nn.init.uniform_(embed, -1 / codebook_size, 1 / codebook_size)

        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.ones(codebook_size))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D) encoder output (or residual from previous level)
        Returns:
            quantized_st: (B, D)  straight-through quantized
            indices:       (B,)   nearest codebook index per sample
            commit_loss:   scalar  ||z - sg(e)||^2
        """
        # L2 distance to all codebook entries
        dist = (
            z.pow(2).sum(dim=1, keepdim=True)          # (B, 1)
            - 2.0 * (z @ self.embed.T)                 # (B, K)
            + self.embed.pow(2).sum(dim=1)             # (K,)
        )  # (B, K)

        indices = dist.argmin(dim=1)        # (B,)
        quantized = self.embed[indices]     # (B, D)

        if self.training:
            self._ema_update(z, indices)

        # straight-through estimator: gradient passes through z unchanged
        quantized_st = z + (quantized - z).detach()

        commit_loss = F.mse_loss(z, quantized.detach())

        return quantized_st, indices, commit_loss

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.codebook_size).float()   # (B, K)

        # cluster size EMA
        self.cluster_size.mul_(self.decay).add_(
            one_hot.sum(0), alpha=1.0 - self.decay
        )

        # embedding average EMA
        self.embed_avg.mul_(self.decay).add_(
            one_hot.T @ z, alpha=1.0 - self.decay
        )

        # Laplace smoothing to avoid dead codes
        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.eps)
            / (n + self.codebook_size * self.eps)
            * n
        )
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

    def active_codes(self, indices: torch.Tensor) -> float:
        """Fraction of codebook entries used in this batch (codebook utilisation)."""
        return indices.unique().numel() / self.codebook_size
