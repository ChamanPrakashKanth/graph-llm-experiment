"""Tiny Encoder module for CAT V3."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyEncoder(nn.Module):
    """A self-contained lightweight transformer encoder for processing text queries."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode token sequence to a dense vector.
        
        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs
            attention_mask: [batch_size, seq_len] mask tensor (1 for active, 0 for pad)
            
        Returns:
            [batch_size, hidden_size] dense sentence representation
        """
        seq_len = input_ids.size(1)
        # Add word embeddings and positional encodings
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len]
        
        # PyTorch src_key_padding_mask requires True on padding positions
        padding_mask = (attention_mask == 0)
        
        # Run Transformer
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Perform mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(out * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        return pooled
