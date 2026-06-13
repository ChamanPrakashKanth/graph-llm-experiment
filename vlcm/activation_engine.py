"""Concept Activation Engine for VLCM.

Uses a Tiny Transformer Encoder and ConceptActivator to turn text queries
into initial activations, then propagates them over the graph memory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel

from reasoning_dataset import ConceptVocabulary


class TinyTransformerEncoder(nn.Module):
    """Transformer encoder for encoding question texts into dense representations."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        pretrained_encoder_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        if pretrained_encoder_name:
            self.encoder = AutoModel.from_pretrained(pretrained_encoder_name)
            self.hidden_size = int(self.encoder.config.hidden_size)
        else:
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=512,
                pad_token_id=0,
            )
            self.encoder = BertModel(config)
            self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Return CLS embedding
        return outputs.last_hidden_state[:, 0]


class ConceptActivator(nn.Module):
    """Maps the dense question embedding to initial concept activation probabilities."""

    def __init__(self, encoder_dim: int, num_concepts: int, special_ids: List[int]) -> None:
        super().__init__()
        self.activation_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, num_concepts),
        )
        mask = torch.ones(num_concepts, dtype=torch.bool)
        for concept_id in special_ids:
            mask[concept_id] = False
        self.register_buffer("real_concept_mask", mask, persistent=False)

    def forward(self, question_embedding: torch.Tensor, top_k: int = 5) -> Dict[str, torch.Tensor]:
        logits = self.activation_head(question_embedding)
        logits = logits.masked_fill(~self.real_concept_mask.unsqueeze(0), -1.0e4)
        probs = torch.sigmoid(logits)
        
        k = min(top_k, int(self.real_concept_mask.sum().item()))
        scores, concept_ids = torch.topk(probs, k=max(k, 1), dim=-1)
        return {
            "activation_logits": logits,
            "activation_probs": probs,
            "concept_ids": concept_ids,
            "concept_scores": scores,
        }


class ConceptActivationEngine:
    """Helper class to query activation models and inspect concepts."""

    def __init__(
        self,
        encoder: TinyTransformerEncoder,
        activator: ConceptActivator,
        vocab: ConceptVocabulary,
    ) -> None:
        self.encoder = encoder
        self.activator = activator
        self.vocab = vocab

    def activate_query(
        self,
        question: str,
        tokenizer,
        device: torch.device,
        top_k: int = 5,
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """Encodes question text and returns active concepts and their activation values."""
        self.encoder.eval()
        self.activator.eval()

        encoded = tokenizer.encode(question, max_length=64)
        input_ids = encoded["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            emb = self.encoder(input_ids, attention_mask)
            act_outputs = self.activator(emb, top_k=top_k)
            probs = act_outputs["activation_probs"].squeeze(0).cpu().tolist()

        concept_scores = {}
        for concept_name in self.vocab.concept_names():
            c_id = self.vocab.concept_to_id[concept_name]
            concept_scores[concept_name] = probs[c_id]

        sorted_concepts = sorted(concept_scores.items(), key=lambda x: -x[1])
        return concept_scores, sorted_concepts
