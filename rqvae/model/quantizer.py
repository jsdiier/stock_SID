from typing import Tuple
import torch
import torch.nn as nn

from .codebook import VQCodebook


class ResidualQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RQ).

    Applies K VQ codebooks sequentially, each quantizing the residual
    left by the previous level:
        residual_0 = z
        q_1, code_1 = VQ_1(residual_0);  residual_1 = residual_0 - q_1
        q_2, code_2 = VQ_2(residual_1);  residual_2 = residual_1 - q_2
        ...
        z_q = q_1 + q_2 + ... + q_K

    The SID for a sample is the tuple (code_1, code_2, ..., code_K).
    """

    def __init__(self, num_levels: int, codebook_size: int, embed_dim: int,
                 ema_decay: float = 0.99):
        super().__init__()
        self.num_levels = num_levels
        self.codebooks = nn.ModuleList([
            VQCodebook(codebook_size, embed_dim, decay=ema_decay)
            for _ in range(num_levels)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D) latent from encoder
        Returns:
            z_q:        (B, D)            sum of all level embeddings
            codes:      (B, num_levels)   integer SID codes
            total_loss: scalar            sum of per-level commitment losses
        """
        residual = z
        z_q = torch.zeros_like(z)
        all_codes = []
        total_loss = torch.tensor(0.0, device=z.device)

        for cb in self.codebooks:
            q, idx, loss = cb(residual)
            residual = residual - q.detach()
            z_q = z_q + q
            all_codes.append(idx)
            total_loss = total_loss + loss

        codes = torch.stack(all_codes, dim=1)   # (B, num_levels)
        return z_q, codes, total_loss

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode to SID codes only (no gradients, no EMA update).
        Returns codes: (B, num_levels)
        """
        residual = z
        all_codes = []
        for cb in self.codebooks:
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * (residual @ cb.embed.T)
                + cb.embed.pow(2).sum(dim=1)
            )
            idx = dist.argmin(dim=1)
            q = cb.embed[idx]
            residual = residual - q
            all_codes.append(idx)
        return torch.stack(all_codes, dim=1)

    def codebook_utilisation(self, codes: torch.Tensor) -> list:
        """Per-level fraction of active codes in a batch."""
        return [
            codes[:, level].unique().numel() / self.codebooks[level].codebook_size
            for level in range(self.num_levels)
        ]
