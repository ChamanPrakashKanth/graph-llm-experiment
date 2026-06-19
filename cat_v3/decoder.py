"""Tiny Decoder module for CAT V3."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class TinyDecoder(nn.Module):
    """A lightweight causal transformer decoder for converting concept embeddings to text."""

    def __init__(
        self,
        vocab_size: int,
        concept_dim: int,
        hidden_size: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.concept_proj = nn.Linear(concept_dim, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Construct standard upper-triangular causal mask for transformer self-attention."""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        organized_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive training forward pass (teacher forcing).
        
        Args:
            organized_embeddings: [batch_size, top_m, concept_dim]
            target_ids: [batch_size, seq_len]
            target_mask: [batch_size, seq_len] (1 for active, 0 for pad)
            
        Returns:
            [batch_size, seq_len, vocab_size] token logits
        """
        batch_size, seq_len = target_ids.size()
        device = target_ids.device

        # Map concept states to decoder hidden size (cross-attention memory)
        memory = self.concept_proj(organized_embeddings)

        # Target embeddings + positional encodings
        x = self.embedding(target_ids) + self.pos_embedding[:, :seq_len]

        # Masks
        tgt_mask = self._generate_causal_mask(seq_len, device)
        tgt_key_padding_mask = (target_mask == 0) if target_mask is not None else None

        # Transformer decoding
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.output_head(out)

    @torch.no_grad()
    def generate(
        self,
        organized_embeddings: torch.Tensor,
        max_length: int = 32,
        start_id: int = 0,
        eos_id: int = 1,
    ) -> torch.Tensor:
        """Autoregressively generate English tokens from concept embeddings.
        
        Args:
            organized_embeddings: [batch_size, top_m, concept_dim]
            max_length: Maximum sequence length to generate
            start_id: ID of the PAD/start token
            eos_id: ID of the EOS token
            
        Returns:
            [batch_size, gen_seq_len] generated token IDs
        """
        batch_size = organized_embeddings.size(0)
        device = organized_embeddings.device

        # Initialize with start token
        generated = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
        memory = self.concept_proj(organized_embeddings)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            seq_len = generated.size(1)
            x = self.embedding(generated) + self.pos_embedding[:, :seq_len]
            tgt_mask = self._generate_causal_mask(seq_len, device)

            out = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
            logits = self.output_head(out[:, -1, :])  # Take logits of last step
            next_tokens = logits.argmax(dim=-1)       # [batch_size]

            # Enforce EOS if finished
            next_tokens = torch.where(finished, torch.tensor(eos_id, device=device), next_tokens)
            
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

        return generated
