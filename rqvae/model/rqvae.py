from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import ResidualQuantizer


class RQVAE(nn.Module):
    """
    Residual-Quantization VAE.

    Forward pass:
        x (B, input_dim)
        → Encoder → z (B, latent_dim)
        → ResidualQuantizer → z_q (B, latent_dim), codes (B, K)
        → Decoder → x_recon (B, input_dim)

    SID for sample i = codes[i]  shape: (K,)  each value ∈ [0, codebook_size)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.commitment_weight = commitment_weight

        self.encoder    = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder    = Decoder(latent_dim, hidden_dims, input_dim)
        self.quantizer  = ResidualQuantizer(num_codebooks, codebook_size,
                                            latent_dim, ema_decay)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        z_q, codes, vq_loss = self.quantizer(z)
        x_rec = self.decoder(z_q)
        return x_rec, codes, vq_loss

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, recon_loss, vq_loss, codes
        """
        x_rec, codes, vq_loss = self.forward(x)
        recon_loss = F.mse_loss(x_rec, x)
        total_loss = recon_loss + self.commitment_weight * vq_loss
        return total_loss, recon_loss, vq_loss, codes

    @torch.no_grad()
    def encode_to_sid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input embeddings to discrete SID codes.
        Returns: (B, num_codebooks) integer tensor
        """
        z = self.encoder(x)
        return self.quantizer.encode(z)

    @torch.no_grad()
    def decode_from_sid(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct embedding from SID codes (lookup + decode).
        codes: (B, num_codebooks)
        Returns: (B, input_dim)
        """
        z_q = torch.zeros(
            codes.shape[0], self.quantizer.codebooks[0].embed_dim,
            device=codes.device
        )
        for level, cb in enumerate(self.quantizer.codebooks):
            z_q = z_q + cb.embed[codes[:, level]]
        return self.decoder(z_q)
