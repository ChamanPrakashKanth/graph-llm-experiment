"""CAT V2 neural model: concept activation, graph propagation, path prediction."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertConfig, BertModel

from reasoning_dataset import ConceptVocabulary
from reasoning_graph import ReasoningGraph


def graph_to_tensors(
    graph: Optional[ReasoningGraph],
    vocab: ConceptVocabulary,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build propagation, transition, and first-step masks from an explicit graph."""

    num_concepts = vocab.size()
    propagation = torch.eye(num_concepts, dtype=torch.float)
    transition = torch.zeros(num_concepts, num_concepts, dtype=torch.bool)
    first_step = torch.ones(num_concepts, dtype=torch.bool)
    first_step[vocab.pad_id] = False
    first_step[vocab.eos_id] = False

    if graph is not None:
        for source, targets in graph.edges.items():
            if source not in vocab.concept_to_id:
                continue
            source_id = vocab.concept_to_id[source]
            for edge in targets.values():
                if edge.target not in vocab.concept_to_id:
                    continue
                target_id = vocab.concept_to_id[edge.target]
                transition[source_id, target_id] = True
                propagation[target_id, source_id] += float(edge.weight)

    for concept_id in range(num_concepts):
        if concept_id != vocab.pad_id:
            transition[concept_id, vocab.eos_id] = True
    transition[vocab.eos_id, vocab.eos_id] = True
    transition[vocab.pad_id, vocab.eos_id] = True

    row_sum = propagation.sum(dim=1, keepdim=True).clamp_min(1e-6)
    propagation = propagation / row_sum

    if device is not None:
        propagation = propagation.to(device)
        transition = transition.to(device)
        first_step = first_step.to(device)

    return propagation, transition, first_step


class TinyTransformerEncoder(nn.Module):
    """Locally initialized transformer encoder using Hugging Face modules."""

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
        return outputs.last_hidden_state[:, 0]


class ConceptMemory(nn.Module):
    def __init__(self, num_concepts: int, concept_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_concepts, concept_dim)

    def forward(self, concept_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(concept_ids)

    def all_embeddings(self) -> torch.Tensor:
        ids = torch.arange(self.embeddings.num_embeddings, device=self.embeddings.weight.device)
        return self.embeddings(ids)


class ConceptActivator(nn.Module):
    """Map the question representation to a supervised concept activation field."""

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


class GraphMessagePassing(nn.Module):
    """Pure PyTorch message passing over the explicit concept graph."""

    def __init__(self, concept_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(concept_dim, concept_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(concept_dim, concept_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(concept_dim) for _ in range(num_layers)])

    def forward(self, state: torch.Tensor, propagation_matrix: torch.Tensor) -> torch.Tensor:
        hidden = state
        for layer, norm in zip(self.layers, self.norms):
            messages = torch.einsum("ts,bsd->btd", propagation_matrix, hidden)
            hidden = norm(hidden + layer(messages))
        return hidden


class PathGenerator(nn.Module):
    """Stateful graph-constrained path decoder."""

    def __init__(
        self,
        concept_dim: int,
        num_concepts: int,
        path_length: int,
        pad_id: int,
        eos_id: int,
    ) -> None:
        super().__init__()
        self.path_length = path_length
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.start_embedding = nn.Parameter(torch.zeros(concept_dim))
        self.question_init = nn.Linear(concept_dim, concept_dim)
        self.gru = nn.GRUCell(concept_dim * 2, concept_dim)
        self.output_head = nn.Linear(concept_dim, num_concepts)
        self.context_projection = nn.Linear(concept_dim, concept_dim)

    def forward(
        self,
        memory: ConceptMemory,
        input_norm: nn.LayerNorm,
        graph_reasoner: GraphMessagePassing,
        propagation_matrix: torch.Tensor,
        initial_activation_probs: torch.Tensor,
        question_context: torch.Tensor,
        activation_logits: torch.Tensor,
        transition_mask: torch.Tensor,
        first_step_mask: torch.Tensor,
        target_paths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = question_context.size(0)
        num_concepts = initial_activation_probs.size(1)
        hidden = torch.tanh(self.question_init(question_context))
        context = torch.tanh(self.context_projection(question_context))
        prev_embedding = self.start_embedding.unsqueeze(0).expand(batch_size, -1)
        prev_ids = torch.full(
            (batch_size,),
            self.eos_id,
            dtype=torch.long,
            device=question_context.device,
        )

        logits_steps: List[torch.Tensor] = []
        prediction_steps: List[torch.Tensor] = []
        score_steps: List[torch.Tensor] = []

        # Track GNN activations and states iteratively
        activation_probs = initial_activation_probs.clone()
        final_graph_state = None

        for step in range(self.path_length):
            # Dynamic GNN memory propagation at each iteration
            concept_memory = memory.all_embeddings().unsqueeze(0).expand(batch_size, -1, -1)
            current_state = input_norm(
                concept_memory
                + question_context.unsqueeze(1) * activation_probs.unsqueeze(-1)
            )
            graph_state = graph_reasoner(current_state, propagation_matrix)
            final_graph_state = graph_state

            if step > 0:
                prev_embedding = graph_state[
                    torch.arange(batch_size, device=graph_state.device),
                    prev_ids,
                ]

            decoder_input = torch.cat([prev_embedding, context], dim=-1)
            hidden = self.gru(decoder_input, hidden)

            logits = self.output_head(hidden) + 0.25 * activation_logits
            if step == 0:
                allowed = first_step_mask.unsqueeze(0).expand(batch_size, -1)
            else:
                allowed = transition_mask[prev_ids]
            logits = logits.masked_fill(~allowed, -1.0e4)

            predicted = logits.argmax(dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            scores = log_probs.gather(1, predicted.unsqueeze(1)).squeeze(1)

            logits_steps.append(logits)
            prediction_steps.append(predicted)
            score_steps.append(scores)

            if target_paths is not None:
                next_ids = target_paths[:, step].clone()
                next_ids = torch.where(
                    next_ids == self.pad_id,
                    torch.full_like(next_ids, self.eos_id),
                    next_ids,
                )
            else:
                next_ids = predicted

            prev_ids = next_ids.clamp(min=0, max=num_concepts - 1)
            
            # Recursive feedback loop: Inject predicted/target concept back into activations
            activation_probs = activation_probs.clone()
            activation_probs.scatter_(1, prev_ids.unsqueeze(1), 1.0)

        return {
            "path_logits": torch.stack(logits_steps, dim=1),
            "predicted_path": torch.stack(prediction_steps, dim=1),
            "path_scores": torch.stack(score_steps, dim=1),
            "final_graph_state": final_graph_state,
        }


class CATReasoningModel(nn.Module):
    """Concept Attention Transformer V2."""

    def __init__(
        self,
        num_concepts: int,
        tokenizer_vocab_size: int,
        graph: Optional[ReasoningGraph],
        vocab: ConceptVocabulary,
        concept_dim: int = 128,
        path_length: int = 8,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        graph_layers: int = 2,
        top_k: int = 5,
        dropout: float = 0.1,
        pretrained_encoder_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = {
            "num_concepts": num_concepts,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "concept_dim": concept_dim,
            "path_length": path_length,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "graph_layers": graph_layers,
            "top_k": top_k,
            "dropout": dropout,
            "pretrained_encoder_name": pretrained_encoder_name,
        }
        self.vocab_pad_id = vocab.pad_id
        self.vocab_eos_id = vocab.eos_id
        self.top_k = top_k

        self.encoder = TinyTransformerEncoder(
            vocab_size=tokenizer_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            pretrained_encoder_name=pretrained_encoder_name,
        )
        encoder_dim = self.encoder.hidden_size
        self.question_projection = nn.Linear(encoder_dim, concept_dim)
        self.memory = ConceptMemory(num_concepts=num_concepts, concept_dim=concept_dim)
        self.activator = ConceptActivator(
            encoder_dim=encoder_dim,
            num_concepts=num_concepts,
            special_ids=[vocab.pad_id, vocab.eos_id],
        )
        self.input_norm = nn.LayerNorm(concept_dim)
        self.graph_reasoner = GraphMessagePassing(
            concept_dim=concept_dim,
            num_layers=graph_layers,
            dropout=dropout,
        )
        self.path_generator = PathGenerator(
            concept_dim=concept_dim,
            num_concepts=num_concepts,
            path_length=path_length,
            pad_id=vocab.pad_id,
            eos_id=vocab.eos_id,
        )

        propagation, transition, first_step = graph_to_tensors(graph, vocab)
        self.register_buffer("propagation_matrix", propagation)
        self.register_buffer("transition_mask", transition)
        self.register_buffer("first_step_mask", first_step)

    def encode_question(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    def set_graph(self, graph: ReasoningGraph, vocab: ConceptVocabulary) -> None:
        propagation, transition, first_step = graph_to_tensors(
            graph,
            vocab,
            device=self.propagation_matrix.device,
        )
        self.propagation_matrix = propagation
        self.transition_mask = transition
        self.first_step_mask = first_step

    def _build_traversal_trace(
        self,
        predicted_path: torch.Tensor,
        path_scores: torch.Tensor,
    ) -> List[List[Dict[str, float]]]:
        traces: List[List[Dict[str, float]]] = []
        for row, scores in zip(predicted_path.detach().cpu(), path_scores.detach().cpu()):
            traces.append(
                [
                    {
                        "step": float(step),
                        "concept_id": float(int(concept_id)),
                        "log_probability": float(score),
                    }
                    for step, (concept_id, score) in enumerate(zip(row.tolist(), scores.tolist()))
                ]
            )
        return traces

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_paths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        question_embedding = self.encode_question(input_ids, attention_mask)
        projected_question = self.question_projection(question_embedding)

        activated = self.activator(question_embedding, top_k=self.top_k)
        activation_probs = activated["activation_probs"]

        path_output = self.path_generator(
            memory=self.memory,
            input_norm=self.input_norm,
            graph_reasoner=self.graph_reasoner,
            propagation_matrix=self.propagation_matrix,
            initial_activation_probs=activation_probs,
            question_context=projected_question,
            activation_logits=activated["activation_logits"],
            transition_mask=self.transition_mask,
            first_step_mask=self.first_step_mask,
            target_paths=target_paths,
        )

        return {
            "question_embedding": question_embedding,
            "activation_logits": activated["activation_logits"],
            "activated_concepts": activated["concept_ids"],
            "activation_scores": activated["concept_scores"],
            "graph_state": path_output["final_graph_state"],
            "path_logits": path_output["path_logits"],
            "predicted_path": path_output["predicted_path"],
            "path_scores": path_output["path_scores"],
            "traversal_trace": self._build_traversal_trace(
                path_output["predicted_path"],
                path_output["path_scores"],
            ),
        }


def build_model_from_dataset(
    dataset,
    concept_dim: int = 128,
    hidden_size: int = 128,
    path_length: Optional[int] = None,
    **kwargs,
) -> CATReasoningModel:
    return CATReasoningModel(
        num_concepts=dataset.vocab.size(),
        tokenizer_vocab_size=dataset.tokenizer.vocab_size,
        graph=dataset.graph,
        vocab=dataset.vocab,
        concept_dim=concept_dim,
        hidden_size=hidden_size,
        path_length=path_length or dataset.max_path_length,
        **kwargs,
    )


if __name__ == "__main__":
    from reasoning_dataset import ReasoningCollator, ReasoningDataset, write_example_dataset

    path = write_example_dataset()
    dataset = ReasoningDataset(path, max_path_length=8)
    batch = ReasoningCollator()([dataset[0], dataset[1]])
    model = build_model_from_dataset(dataset, concept_dim=64, hidden_size=64, num_hidden_layers=1)
    out = model(batch["input_ids"], batch["attention_mask"], target_paths=batch["path_ids"])
    print(out["predicted_path"].shape)

