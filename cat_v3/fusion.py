"""Concept Fusion Layer module for CAT V3."""

from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn


class ConceptFusionLayer(nn.Module):
    """Fuses path predictions and embeddings from multiple active GAT experts."""

    def __init__(self, num_concepts: int, pad_id: int, eos_id: int, top_m: int = 8) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.top_m = top_m

    def forward(
        self,
        expert_reports: Dict[str, Dict[str, torch.Tensor]],
        router_probs: torch.Tensor,
        router_mask: torch.Tensor,
        global_embeddings: torch.Tensor,
        domain_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Fuses expert reports into a unified tensor representation.
        
        Args:
            expert_reports: Dict mapping domain -> GATExpert output dict
            router_probs: [batch_size, num_experts] tensor of router probabilities
            router_mask: [batch_size, num_experts] boolean mask of active experts
            global_embeddings: [num_concepts, concept_dim] concept embedding table
            domain_names: List of domain names mapping to indices in router outputs
            
        Returns:
            Dict containing:
                fused_concept_ids: [batch_size, top_m]
                fused_embeddings: [batch_size, top_m, concept_dim]
                fused_scores: [batch_size, num_concepts]
                top_scores: [batch_size, top_m]
        """
        batch_size = router_probs.size(0)
        device = router_probs.device
        path_len = next(iter(expert_reports.values()))["predicted_path"].size(1)

        # Accumulator for scores of each concept: [batch_size, num_concepts]
        fused_scores = torch.zeros(batch_size, self.num_concepts, device=device)

        for idx, domain in enumerate(domain_names):
            report = expert_reports[domain]
            pred_path = report["predicted_path"]  # [batch_size, path_length]
            path_scores = report["path_scores"]   # [batch_size, path_length] (log probabilities)
            path_probs = torch.exp(path_scores)   # [batch_size, path_length] (probabilities)

            # Mask and weight based on router probability for this domain
            # active_weight: [batch_size]
            active_weight = router_probs[:, idx] * router_mask[:, idx].float()

            # Scatter probabilities into the concept accumulator for each step
            for step in range(path_len):
                step_concepts = pred_path[:, step]  # [batch_size]
                step_probs = path_probs[:, step]    # [batch_size]

                # Weight step probability by expert activation weight
                weighted_probs = step_probs * active_weight  # [batch_size]
                fused_scores.scatter_add_(1, step_concepts.unsqueeze(1), weighted_probs.unsqueeze(1))

        # Clear scores for special pad and eos tokens to avoid selecting them
        fused_scores[:, self.pad_id] = 0.0
        fused_scores[:, self.eos_id] = 0.0

        # Retrieve the top-M most activated concepts
        scores, fused_concept_ids = torch.topk(fused_scores, k=self.top_m, dim=-1)

        # Retrieve unified concept embeddings: [batch_size, top_m, concept_dim]
        fused_embeddings = global_embeddings[fused_concept_ids]

        return {
            "fused_concept_ids": fused_concept_ids,
            "fused_embeddings": fused_embeddings,
            "fused_scores": fused_scores,
            "top_scores": scores
        }

    def get_symbolic_report(
        self,
        vocab: Any,
        expert_reports: Dict[str, Dict[str, torch.Tensor]],
        router_mask: torch.Tensor,
        domain_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Converts raw tensor results into human-readable concept graphs and paths.
        
        Args:
            vocab: ConceptVocabulary instance
            expert_reports: Dict of expert outputs
            router_mask: [batch_size, num_experts]
            domain_names: List of domains
            
        Returns:
            List of dicts, one per batch item:
                {
                    "concepts": List[str],
                    "reasoning_paths": List[List[str]],
                    "confidence": List[float]
                }
        """
        batch_size = router_mask.size(0)
        symbolic_reports = []

        for b in range(batch_size):
            concepts_set = set()
            reasoning_paths = []
            confidences = []

            for idx, domain in enumerate(domain_names):
                if router_mask[b, idx].item():
                    report = expert_reports[domain]
                    path_ids = report["predicted_path"][b].tolist()
                    path = vocab.decode_path(path_ids)
                    if path:
                        reasoning_paths.append(path)
                        for c in path:
                            concepts_set.add(c)
                        
                        # Mean path confidence calculation
                        path_scores = report["path_scores"][b].tolist()
                        path_probs = [torch.exp(torch.tensor(s)).item() for s in path_scores[:len(path)]]
                        mean_prob = sum(path_probs) / max(len(path_probs), 1)
                        confidences.append(mean_prob)

            symbolic_reports.append({
                "concepts": sorted(list(concepts_set)),
                "reasoning_paths": reasoning_paths,
                "confidence": confidences
            })

        return symbolic_reports
