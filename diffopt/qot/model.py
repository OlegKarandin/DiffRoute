"""SpanAttentionQoT: Transformer-based QoT estimator for optical segments."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpanAttentionQoT(nn.Module):
    """Predict GSNR (dB) of a transparent segment from per-span features.

    Architecture:
        Input: (batch, max_spans, feature_dim=5)
        -> Linear projection: 5 -> model_dim
        -> Learned positional encoding: (max_spans, model_dim)
        -> TransformerEncoder: num_layers layers, num_heads heads
        -> Mean pool over non-padded spans
        -> MLP: model_dim -> ff_dim//2 -> 1

    Args:
        feature_dim: Number of features per span (default 5).
        model_dim: Transformer embedding dimension (default 64).
        num_heads: Number of attention heads (default 4).
        num_layers: Number of transformer encoder layers (default 2).
        max_spans: Maximum number of spans (default 60).
        ff_dim: Feed-forward dimension inside transformer (default 128).
    """

    def __init__(
        self,
        feature_dim: int = 5,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_spans: int = 60,
        ff_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.max_spans = max_spans

        # Input projection
        self.input_proj = nn.Linear(feature_dim, model_dim)

        # Learned positional encoding
        self.pos_encoding = nn.Embedding(max_spans, model_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, ff_dim // 2),
            nn.ReLU(),
            nn.Linear(ff_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_encoding.weight, std=0.02)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, span_features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            span_features: (batch, max_spans, feature_dim)
            padding_mask: (batch, max_spans) BoolTensor, True for real spans

        Returns:
            gsnr_db: (batch,) predicted GSNR in dB
        """
        batch_size, seq_len, _ = span_features.shape

        # Project features
        x = self.input_proj(span_features)  # (B, S, D)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_encoding(positions).unsqueeze(0)  # (B, S, D)

        # TransformerEncoder expects src_key_padding_mask where True means IGNORE
        # Our padding_mask: True = real span -> invert for transformer
        src_key_padding_mask = ~padding_mask  # True = padding (ignored)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, S, D)

        # Mean pool over real spans only
        mask_float = padding_mask.float().unsqueeze(-1)  # (B, S, 1)
        n_real = mask_float.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = (x * mask_float).sum(dim=1) / n_real  # (B, D)

        # MLP head
        out = self.mlp(pooled).squeeze(-1)  # (B,)
        return out
