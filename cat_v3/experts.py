"""GAT Experts module for CAT V3 using PyTorch Geometric."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATExpert(nn.Module):
    """A domain expert that reasons over a specific concept graph using GAT layers."""

    def __init__(
        self,
        domain_name: str,
        num_concepts: int,
        concept_dim: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        pad_id: int,
        eos_id: int,
        path_length: int = 8,
    ) -> None:
        super().__init__()
        self.domain_name = domain_name
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.path_length = path_length

        # Graph buffers
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # GAT message passing layers
        self.gat1 = GATConv(concept_dim, concept_dim, heads=2, concat=False)
        self.gat2 = GATConv(concept_dim, concept_dim, heads=1, concat=False)

        # Path reasoning decoder
        self.start_embedding = nn.Parameter(torch.zeros(concept_dim))
        self.question_proj = nn.Linear(concept_dim, concept_dim)
        self.gru = nn.GRUCell(concept_dim, concept_dim)
        self.output_head = nn.Linear(concept_dim, num_concepts)

        # Construct topological transition mask
        # transition[u, v] is True if there is a valid edge u -> v
        transition = torch.zeros(num_concepts, num_concepts, dtype=torch.bool)
        src = edge_index[0]
        dst = edge_index[1]
        transition[src, dst] = True

        # Allow transitioning to EOS from any concept except pad
        for c_id in range(num_concepts):
            if c_id != pad_id:
                transition[c_id, eos_id] = True
        transition[eos_id, eos_id] = True
        transition[pad_id, eos_id] = True
        self.register_buffer("transition_mask", transition)

        # First-step mask: starting nodes must have outgoing edges in this domain
        first_step = torch.zeros(num_concepts, dtype=torch.bool)
        first_step[src] = True
        first_step[eos_id] = False
        first_step[pad_id] = False
        self.register_buffer("first_step_mask", first_step)

    def forward(
        self,
        global_embeddings: torch.Tensor,
        query_context: torch.Tensor,
        target_paths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform GAT message passing and generate a reasoning path.
        
        Args:
            global_embeddings: [num_concepts, concept_dim]
            query_context: [batch_size, concept_dim]
            target_paths: [batch_size, path_length] (optional for teacher forcing)
            
        Returns:
            Dict containing expert reports: predicted path, logits, scores, and updated node states.
        """
        batch_size = query_context.size(0)
        device = query_context.device

        # 1. GAT message passing over expert's graph
        # edge_weight acts as edge importance weights
        h = F.elu(self.gat1(global_embeddings, self.edge_index))
        node_states = self.gat2(h, self.edge_index) # [num_concepts, concept_dim]

        # 2. Path generation loop
        hidden = torch.tanh(self.question_proj(query_context))
        prev_embedding = self.start_embedding.unsqueeze(0).expand(batch_size, -1)
        prev_ids = torch.full((batch_size,), self.eos_id, dtype=torch.long, device=device)

        logits_steps = []
        prediction_steps = []
        score_steps = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(self.path_length):
            # Input is the state of previously predicted concept node
            if step > 0:
                prev_embedding = node_states[prev_ids]

            hidden = self.gru(prev_embedding, hidden)
            logits = self.output_head(hidden)

            # Apply domain transition constraint mask
            if step == 0:
                allowed = self.first_step_mask.unsqueeze(0).expand(batch_size, -1)
            else:
                allowed = self.transition_mask[prev_ids]
            
            logits = logits.masked_fill(~allowed, -1e4)
            predicted = logits.argmax(dim=-1)

            # Handle finished batches (early stop mask)
            if target_paths is None:
                predicted = torch.where(
                    finished,
                    torch.tensor(self.eos_id, device=device),
                    predicted
                )

            log_probs = F.log_softmax(logits, dim=-1)
            scores = log_probs.gather(1, predicted.unsqueeze(1)).squeeze(1)

            logits_steps.append(logits)
            prediction_steps.append(predicted)
            score_steps.append(scores)

            if target_paths is not None:
                next_ids = target_paths[:, step].clone()
                # Map pad to eos during training evaluation
                next_ids = torch.where(
                    next_ids == self.pad_id,
                    torch.full_like(next_ids, self.eos_id),
                    next_ids
                )
            else:
                next_ids = predicted

            if target_paths is None:
                finished = finished | (predicted == self.eos_id) | (predicted == self.pad_id)
                if finished.all():
                    # Pad remaining steps
                    remaining = self.path_length - len(prediction_steps)
                    for _ in range(remaining):
                        pad_logits = torch.full_like(logits, -1e4)
                        pad_logits[:, self.eos_id] = 0.0
                        logits_steps.append(pad_logits)
                        prediction_steps.append(torch.full((batch_size,), self.eos_id, dtype=torch.long, device=device))
                        score_steps.append(torch.zeros((batch_size,), device=device))
                    break

            prev_ids = next_ids.clamp(min=0, max=self.num_concepts - 1)

        return {
            "predicted_path": torch.stack(prediction_steps, dim=1),
            "path_logits": torch.stack(logits_steps, dim=1),
            "path_scores": torch.stack(score_steps, dim=1),
            "node_states": node_states
        }
