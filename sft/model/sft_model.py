"""
Decoder-only (GPT-style) Transformer that predicts the next week's SID
given a sequence of past weekly SID codes.

Architecture
------------
- Each week is K integer codes (one per RQ-VAE codebook level).
- The K embeddings are summed → one d_model-dimensional vector per week.
- Learnable positional encoding is added.
- Causal (auto-regressive) Transformer encoder layers.
- K independent output heads, each producing logits over [0, codebook_size-1].

PAD_CODE = codebook_size (e.g. 128) is used for left-padding shorter sequences.
"""
import torch
import torch.nn as nn


class SIDPredictor(nn.Module):

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.K             = num_codebooks
        self.codebook_size = codebook_size
        self.PAD_CODE      = codebook_size      # index used for left-padding

        # One embedding table per codebook level (valid codes 0..codebook_size-1 + PAD)
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model)
            for _ in range(num_codebooks)
        ])

        # Learnable positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,    # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # K output classification heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size)
            for _ in range(num_codebooks)
        ])

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.trunc_normal_(emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        for head in self.heads:
            nn.init.zeros_(head.bias)

    def forward(self, src: torch.Tensor) -> list:
        """
        Parameters
        ----------
        src : (B, T, K) long  — input sequence of weekly SID codes

        Returns
        -------
        list of K tensors, each (B, T, codebook_size) — per-level logits
        The prediction for position t is the next-step (t+1) logit at position t.
        """
        B, T, K = src.shape

        # Sum embeddings across codebook levels → (B, T, d_model)
        x = sum(self.embeddings[k](src[:, :, k]) for k in range(K))

        # Add positional embedding
        pos = torch.arange(T, device=src.device).unsqueeze(0)  # (1, T)
        x   = x + self.pos_embed(pos)
        x   = self.input_norm(x)

        # Causal mask: position t may only attend to positions 0..t
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=src.device
        )
        x = self.transformer(x, mask=causal_mask)              # (B, T, d_model)

        return [head(x) for head in self.heads]                # K × (B, T, codebook_size)

    @torch.no_grad()
    def predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict codes for the single NEXT time step.

        Parameters
        ----------
        context : (T, K) or (B, T, K) long — past weeks

        Returns
        -------
        (K,) or (B, K) long — predicted codes for the next week
        """
        squeeze = context.dim() == 2
        if squeeze:
            context = context.unsqueeze(0)                     # (1, T, K)

        logits = self.forward(context)                         # K × (1, T, vocab)
        preds  = torch.stack(
            [logits[k][:, -1, :].argmax(dim=-1) for k in range(self.K)],
            dim=-1,
        )                                                      # (B, K)

        return preds.squeeze(0) if squeeze else preds


def build_model(cfg) -> 'SIDPredictor':
    """Construct SIDPredictor from a ConfigParser object."""
    H = cfg.getint('sequence', 'history_weeks')
    return SIDPredictor(
        num_codebooks   = cfg.getint('rqvae',  'num_codebooks'),
        codebook_size   = cfg.getint('rqvae',  'codebook_size'),
        d_model         = cfg.getint('model',  'd_model'),
        nhead           = cfg.getint('model',  'nhead'),
        num_layers      = cfg.getint('model',  'num_layers'),
        dim_feedforward = cfg.getint('model',  'dim_feedforward'),
        dropout         = cfg.getfloat('model','dropout'),
        max_seq_len     = H + 64,
    )
