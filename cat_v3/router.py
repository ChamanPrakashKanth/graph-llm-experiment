"""Semantic Router module for CAT V3."""

from __future__ import annotations

from typing import Dict, List
import torch
import torch.nn as nn


class SemanticRouter(nn.Module):
    """A lightweight MLP routing model that selects active experts for a given query."""

    def __init__(self, encoder_dim: int, num_experts: int = 6) -> None:
        super().__init__()
        self.router_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, num_experts)
        )
        self.num_experts = num_experts

    def forward(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 2,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Routes query embedding to active experts.
        
        Args:
            query_embedding: [batch_size, encoder_dim]
            top_k: Minimum number of experts to activate
            threshold: Probability threshold above which an expert is activated
            
        Returns:
            Dict containing:
                logits: [batch_size, num_experts]
                probs: [batch_size, num_experts]
                activation_mask: [batch_size, num_experts] (bool tensor indicating active experts)
        """
        logits = self.router_head(query_embedding)
        probs = torch.sigmoid(logits)
        
        batch_size = query_embedding.size(0)
        
        # Get top-k indices to guarantee minimum activation
        k = min(top_k, self.num_experts)
        _, top_k_indices = torch.topk(probs, k=k, dim=-1)
        
        # Build activation mask
        activation_mask = torch.zeros_like(probs, dtype=torch.bool)
        activation_mask.scatter_(1, top_k_indices, True)
        
        # Add any expert that exceeds the absolute probability threshold
        activation_mask = activation_mask | (probs >= threshold)
        
        return {
            "logits": logits,
            "probs": probs,
            "activation_mask": activation_mask
        }
