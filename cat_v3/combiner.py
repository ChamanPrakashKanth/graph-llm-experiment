"""Tiny Combiner module for CAT V3."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCombiner(nn.Module):
    """A lightweight transformer module that organizes fused concepts into semantic chunks."""

    def __init__(self, concept_dim: int, nhead: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=concept_dim,
            nhead=nhead,
            dim_feedforward=concept_dim * 2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
        """Organizes concepts based on self-attention.
        
        Args:
            fused_embeddings: [batch_size, top_m, concept_dim]
            
        Returns:
            [batch_size, top_m, concept_dim] organized concept representations
        """
        # Self-attention allows concepts to establish relational context
        return self.transformer(fused_embeddings)
