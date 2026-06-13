"""Graph Neural Memory and Reasoning substrate for VLCM.

Contains ReasoningGraph (logical structure, beam search, visualizer)
and GraphNeuralMemory (trainable neural propagation layer).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning_dataset import ConceptVocabulary


@dataclass
class ConceptNode:
    """A named concept in the reasoning graph."""
    name: str
    activation: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ConceptEdge:
    """A directed, weighted relationship between two concepts."""
    source: str
    target: str
    weight: float = 1.0
    relation: str = "related"
    evidence_count: int = 1
    metadata: Dict[str, object] = field(default_factory=dict)


class ReasoningGraph:
    """Directed concept graph used by VLCM as the reasoning substrate."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[str, Dict[str, ConceptEdge]] = defaultdict(dict)

    def add_node(
        self,
        concept: str,
        activation: float = 0.0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ConceptNode:
        concept = concept.strip()
        if not concept:
            raise ValueError("concept names must be non-empty")

        if concept not in self.nodes:
            self.nodes[concept] = ConceptNode(
                name=concept,
                activation=float(activation),
                metadata=dict(metadata or {}),
            )
        else:
            node = self.nodes[concept]
            node.activation = max(float(activation), node.activation)
            node.metadata.update(metadata or {})
        return self.nodes[concept]

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        relation: str = "related",
        metadata: Optional[Dict[str, object]] = None,
    ) -> ConceptEdge:
        if weight <= 0:
            raise ValueError("edge weights must be positive")

        self.add_node(source)
        self.add_node(target)

        existing = self.edges[source].get(target)
        if existing is not None:
            total = existing.evidence_count + 1
            existing.weight = (
                existing.weight * existing.evidence_count + float(weight)
            ) / total
            existing.evidence_count = total
            existing.relation = relation or existing.relation
            existing.metadata.update(metadata or {})
            return existing

        edge = ConceptEdge(
            source=source,
            target=target,
            weight=float(weight),
            relation=relation,
            metadata=dict(metadata or {}),
        )
        self.edges[source][target] = edge
        return edge

    def neighbors(self, concept: str) -> List[ConceptEdge]:
        return sorted(
            self.edges.get(concept, {}).values(),
            key=lambda edge: (-edge.weight, edge.target),
        )

    def has_edge(self, source: str, target: str) -> bool:
        return target in self.edges.get(source, {})

    def edge_weight(self, source: str, target: str, default: float = 0.0) -> float:
        edge = self.edges.get(source, {}).get(target)
        return edge.weight if edge is not None else default

    def set_activation(self, concept: str, value: float) -> None:
        if concept in self.nodes:
            self.nodes[concept].activation = float(value)

    def reset_activations(self) -> None:
        for node in self.nodes.values():
            node.activation = 0.0

    def grow_from_paths(
        self,
        paths: Iterable[Sequence[str]],
        relation: str = "reasoning_step",
        weight: float = 1.0,
    ) -> None:
        for path in paths:
            cleaned = [concept.strip() for concept in path if concept and concept.strip()]
            for concept in cleaned:
                self.add_node(concept)
            for source, target in zip(cleaned, cleaned[1:]):
                self.add_edge(source, target, weight=weight, relation=relation)

    def grow_from_documents(
        self,
        documents: Iterable[str],
        concepts: Sequence[str],
        window: int = 2,
        relation: str = "document_cooccurrence",
        weight: float = 0.35,
    ) -> None:
        concept_patterns = [
            (concept, re.compile(r"\b" + re.escape(concept.lower()) + r"\b"))
            for concept in concepts
            if concept and not concept.startswith("<")
        ]

        for concept, _ in concept_patterns:
            self.add_node(concept)

        for document in documents:
            sentences = re.split(r"(?<=[.!?])\s+", document)
            for sentence in sentences:
                lowered = sentence.lower()
                mentions: List[Tuple[int, str]] = []
                for concept, pattern in concept_patterns:
                    match = pattern.search(lowered)
                    if match:
                        mentions.append((match.start(), concept))

                mentions = sorted(set(mentions))
                ordered = [concept for _, concept in mentions]
                for i, source in enumerate(ordered):
                    for target in ordered[i + 1 : i + 1 + window]:
                        if source != target:
                            self.add_edge(
                                source,
                                target,
                                weight=weight,
                                relation=relation,
                                metadata={"source": "document"},
                            )

    def path_score(
        self,
        path: Sequence[str],
        activation_scores: Optional[Dict[str, float]] = None,
        transition_weight: float = 1.0,
        activation_weight: float = 0.25,
        length_penalty: float = 0.01,
    ) -> float:
        if not path:
            return -math.inf

        score = 0.0
        activations = activation_scores or {
            name: node.activation for name, node in self.nodes.items()
        }

        for concept in path:
            score += activation_weight * float(activations.get(concept, 0.0))

        for source, target in zip(path, path[1:]):
            if not self.has_edge(source, target):
                return -math.inf
            score += transition_weight * self.edge_weight(source, target)

        return score - length_penalty * max(len(path) - 1, 0)

    def beam_search(
        self,
        starts: Sequence[str],
        max_depth: int = 4,
        beam_width: int = 5,
        activation_scores: Optional[Dict[str, float]] = None,
        end_concepts: Optional[Sequence[str]] = None,
    ) -> List[Tuple[float, List[str]]]:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        end_set = set(end_concepts or [])
        beam: List[Tuple[float, List[str]]] = []
        for start in starts:
            if start in self.nodes:
                path = [start]
                beam.append((self.path_score(path, activation_scores), path))

        completed: List[Tuple[float, List[str]]] = list(beam)

        for _ in range(max_depth - 1):
            candidates: List[Tuple[float, List[str]]] = []
            for _, path in beam:
                current = path[-1]
                if current in end_set:
                    candidates.append((self.path_score(path, activation_scores), path))
                    continue

                for edge in self.neighbors(current):
                    if edge.target in path:
                        continue
                    next_path = path + [edge.target]
                    candidates.append(
                        (self.path_score(next_path, activation_scores), next_path)
                    )

            if not candidates:
                break

            candidates.sort(key=lambda item: (-item[0], item[1]))
            beam = candidates[:beam_width]
            completed.extend(beam)

        completed.sort(key=lambda item: (-item[0], item[1]))
        return completed[:beam_width]

    def best_path(
        self,
        starts: Sequence[str],
        max_depth: int = 4,
        beam_width: int = 5,
        activation_scores: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        paths = self.beam_search(
            starts=starts,
            max_depth=max_depth,
            beam_width=beam_width,
            activation_scores=activation_scores,
        )
        return paths[0][1] if paths else []

    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": [asdict(node) for node in sorted(self.nodes.values(), key=lambda n: n.name)],
            "edges": [
                asdict(edge)
                for source in sorted(self.edges)
                for edge in sorted(self.edges[source].values(), key=lambda e: e.target)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ReasoningGraph":
        graph = cls()
        for node_payload in payload.get("nodes", []):
            node_data = dict(node_payload)
            graph.add_node(
                str(node_data["name"]),
                activation=float(node_data.get("activation", 0.0)),
                metadata=dict(node_data.get("metadata", {})),
            )
        for edge_payload in payload.get("edges", []):
            edge_data = dict(edge_payload)
            graph.add_edge(
                str(edge_data["source"]),
                str(edge_data["target"]),
                weight=float(edge_data.get("weight", 1.0)),
                relation=str(edge_data.get("relation", "related")),
                metadata=dict(edge_data.get("metadata", {})),
            )
            edge = graph.edges[str(edge_data["source"])][str(edge_data["target"])]
            edge.evidence_count = int(edge_data.get("evidence_count", 1))
        return graph

    @classmethod
    def from_paths(cls, paths: Iterable[Sequence[str]]) -> "ReasoningGraph":
        graph = cls()
        graph.grow_from_paths(paths)
        return graph

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ReasoningGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def visualize(
        self,
        output_path: str | Path,
        highlight_path: Optional[Sequence[str]] = None,
        title: str = "VLCM Concept Reasoning Graph",
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        graph = nx.DiGraph()
        for node in self.nodes:
            graph.add_node(node)
        for source, targets in self.edges.items():
            for edge in targets.values():
                graph.add_edge(
                    edge.source,
                    edge.target,
                    weight=edge.weight,
                    relation=edge.relation,
                )

        highlight_edges = set()
        highlight_nodes = set(highlight_path or [])
        if highlight_path:
            highlight_edges = set(zip(highlight_path, highlight_path[1:]))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, seed=7, k=0.7)
        node_colors = [
            "#f2a65a" if node in highlight_nodes else "#7eb6ff"
            for node in graph.nodes
        ]
        edge_colors = [
            "#d64242" if edge in highlight_edges else "#667085"
            for edge in graph.edges
        ]
        widths = [
            2.8 if edge in highlight_edges else 1.0 + graph.edges[edge].get("weight", 1.0)
            for edge in graph.edges
        ]

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1500, alpha=0.95)
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=edge_colors,
            width=widths,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.08",
        )
        nx.draw_networkx_labels(graph, pos, font_size=8)
        edge_labels = {
            (source, target): data.get("relation", "related")
            for source, target, data in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output, dpi=180)
        plt.close()
        return output


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


class GraphNeuralMemory(nn.Module):
    """GNN memory module over concepts with learnable edge weights."""

    def __init__(
        self,
        num_concepts: int,
        concept_dim: int,
        graph: Optional[ReasoningGraph],
        vocab: ConceptVocabulary,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim

        # Initial graph matrix construction
        initial_prop, transition_mask, first_step_mask = graph_to_tensors(graph, vocab)
        
        # Binary mask of valid transitions including self-loops (diagonal)
        self.register_buffer(
            "edge_mask",
            (initial_prop > 0.0) | torch.eye(num_concepts, dtype=torch.bool),
        )

        # Trainable edge weights initialized with default graph weights
        self.edge_importance = nn.Parameter(initial_prop.clone())

        # GNN message passing layers
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

    def get_propagation_matrix(self) -> torch.Tensor:
        """Filter trainable weights by edge mask and normalize rows."""
        # Restrict backprop only to valid edges and ensure weights are non-negative
        A = F.relu(self.edge_importance) * self.edge_mask
        row_sum = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return A / row_sum

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Propagate concept state representation matrix through GNN layers."""
        prop_matrix = self.get_propagation_matrix()
        hidden = state
        for layer, norm in zip(self.layers, self.norms):
            # Batch message passing: B x T x D
            messages = torch.einsum("ts,bsd->btd", prop_matrix, hidden)
            hidden = norm(hidden + layer(messages))
        return hidden


def build_engineering_graph() -> ReasoningGraph:
    graph = ReasoningGraph()
    graph.grow_from_paths(
        [
            ["Pressure", "Friction", "Energy Loss", "Pressure Drop"],
            ["Pressure", "Velocity", "Boundary Layer", "Turbulence"],
            ["Load", "Stress", "Strain", "Failure"],
            ["Residual", "Mesh Quality", "Numerical Instability", "Convergence Failure"],
            ["Temperature Gradient", "Heat Flux", "Thermal Stress", "Cracking"],
        ]
    )
    return graph

