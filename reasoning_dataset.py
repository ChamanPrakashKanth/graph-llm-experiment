"""Dataset, tokenizer, and concept vocabulary for CAT V2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from reasoning_graph import ReasoningGraph


PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"


class SimpleTokenizer:
    """A tiny deterministic tokenizer for offline research runs.

    The model still uses a transformer encoder; this tokenizer simply avoids
    requiring a network download for a pretrained tokenizer.
    """

    def __init__(self, token_to_id: Optional[Dict[str, int]] = None) -> None:
        if token_to_id is None:
            token_to_id = {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                SEP_TOKEN: 3,
            }
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)?", text.lower())

    def build(self, texts: Iterable[str]) -> None:
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def encode(self, text: str, max_length: int) -> Dict[str, torch.Tensor]:
        tokens = [CLS_TOKEN] + self.tokenize(text)[: max_length - 2] + [SEP_TOKEN]
        ids = [self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]) for token in tokens]
        attention = [1] * len(ids)

        pad = max_length - len(ids)
        if pad > 0:
            ids.extend([self.token_to_id[PAD_TOKEN]] * pad)
            attention.extend([0] * pad)

        return {
            "input_ids": torch.tensor(ids[:max_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention[:max_length], dtype=torch.long),
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> Dict[str, object]:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SimpleTokenizer":
        return cls(token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()})


class ConceptVocabulary:
    """Stable concept vocabulary with path-specific special tokens."""

    def __init__(self, concepts: Optional[Iterable[str]] = None) -> None:
        self.concept_to_id: Dict[str, int] = {PAD_TOKEN: 0, EOS_TOKEN: 1}
        self.id_to_concept: Dict[int, str] = {0: PAD_TOKEN, 1: EOS_TOKEN}
        if concepts is not None:
            self.build(concepts)

    @property
    def pad_id(self) -> int:
        return self.concept_to_id[PAD_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.concept_to_id[EOS_TOKEN]

    def add(self, concept: str) -> int:
        concept = concept.strip()
        if not concept:
            raise ValueError("concept names must be non-empty")
        if concept not in self.concept_to_id:
            idx = len(self.concept_to_id)
            self.concept_to_id[concept] = idx
            self.id_to_concept[idx] = concept
        return self.concept_to_id[concept]

    def build(self, concepts: Iterable[str]) -> None:
        for concept in sorted({c.strip() for c in concepts if c and c.strip()}):
            if concept not in (PAD_TOKEN, EOS_TOKEN):
                self.add(concept)

    def build_from_paths(self, paths: Iterable[Sequence[str]]) -> None:
        self.build(concept for path in paths for concept in path)

    def encode_path(
        self,
        concepts: Sequence[str],
        max_path_length: int,
        add_eos: bool = True,
    ) -> torch.Tensor:
        ids = [self.concept_to_id[concept] for concept in concepts]
        if add_eos:
            ids.append(self.eos_id)
        ids = ids[:max_path_length]
        if len(ids) < max_path_length:
            ids.extend([self.pad_id] * (max_path_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def decode_path(
        self,
        ids: Sequence[int],
        stop_at_eos: bool = True,
        skip_special: bool = True,
    ) -> List[str]:
        concepts: List[str] = []
        for raw_id in ids:
            concept_id = int(raw_id)
            concept = self.id_to_concept.get(concept_id)
            if concept is None:
                continue
            if concept == EOS_TOKEN and stop_at_eos:
                break
            if skip_special and concept in {PAD_TOKEN, EOS_TOKEN}:
                continue
            concepts.append(concept)
        return concepts

    def activation_target(self, path_ids: torch.Tensor) -> torch.Tensor:
        target = torch.zeros(len(self.concept_to_id), dtype=torch.float)
        for raw_id in path_ids.tolist():
            concept_id = int(raw_id)
            if concept_id not in {self.pad_id, self.eos_id}:
                target[concept_id] = 1.0
        return target

    def concept_names(self) -> List[str]:
        return [
            self.id_to_concept[idx]
            for idx in range(len(self.id_to_concept))
            if idx not in {self.pad_id, self.eos_id}
        ]

    def size(self) -> int:
        return len(self.concept_to_id)

    def to_dict(self) -> Dict[str, object]:
        return {"concept_to_id": self.concept_to_id}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ConceptVocabulary":
        vocab = cls()
        vocab.concept_to_id = {str(k): int(v) for k, v in payload["concept_to_id"].items()}
        vocab.id_to_concept = {idx: concept for concept, idx in vocab.concept_to_id.items()}
        return vocab


@dataclass
class ReasoningSample:
    question: str
    reasoning_path: List[str]
    answer: str


def load_reasoning_samples(path: str | Path) -> List[ReasoningSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: List[ReasoningSample] = []
    for row in payload:
        if not row.get("question") or not row.get("reasoning_path"):
            raise ValueError("each sample needs question and reasoning_path")
        samples.append(
            ReasoningSample(
                question=str(row["question"]),
                reasoning_path=[str(concept) for concept in row["reasoning_path"]],
                answer=str(row.get("answer", "")),
            )
        )
    return samples


class ReasoningDataset(Dataset):
    """Supervised question to reasoning-path dataset."""

    def __init__(
        self,
        dataset_file: str | Path,
        max_length: int = 64,
        max_path_length: int = 8,
        tokenizer: Optional[SimpleTokenizer] = None,
        vocab: Optional[ConceptVocabulary] = None,
    ) -> None:
        self.dataset_file = Path(dataset_file)
        self.max_length = max_length
        self.max_path_length = max_path_length
        self.samples = load_reasoning_samples(self.dataset_file)

        self.vocab = vocab or ConceptVocabulary()
        if vocab is None:
            self.vocab.build_from_paths(sample.reasoning_path for sample in self.samples)

        self.tokenizer = tokenizer or SimpleTokenizer()
        if tokenizer is None:
            self.tokenizer.build(sample.question for sample in self.samples)

        self.graph = ReasoningGraph.from_paths(
            sample.reasoning_path for sample in self.samples
        )
        for concept in self.vocab.concept_names():
            self.graph.add_node(concept)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        encoded = self.tokenizer.encode(sample.question, max_length=self.max_length)
        path_ids = self.vocab.encode_path(
            sample.reasoning_path,
            max_path_length=self.max_path_length,
            add_eos=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "path_ids": path_ids,
            "activation_targets": self.vocab.activation_target(path_ids),
            "question": sample.question,
            "reasoning_path": sample.reasoning_path,
            "answer": sample.answer,
        }


class ReasoningCollator:
    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "path_ids": torch.stack([item["path_ids"] for item in batch]),
            "activation_targets": torch.stack(
                [item["activation_targets"] for item in batch]
            ),
            "questions": [str(item["question"]) for item in batch],
            "reasoning_paths": [list(item["reasoning_path"]) for item in batch],
            "answers": [str(item["answer"]) for item in batch],
        }


EXAMPLE_DATA = [
    {
        "question": "Why does pressure drop in a pipe?",
        "reasoning_path": ["Pressure", "Friction", "Energy Loss", "Pressure Drop"],
        "answer": "Pressure drops because wall friction converts mechanical energy into losses.",
    },
    {
        "question": "Why does turbulence increase at high speed?",
        "reasoning_path": ["Velocity", "Instability", "Vortex", "Turbulence"],
        "answer": "Higher velocity amplifies instabilities, which form vortices and increase turbulence.",
    },
    {
        "question": "Why can poor mesh quality cause convergence failure?",
        "reasoning_path": [
            "Mesh Quality",
            "Numerical Instability",
            "Residual",
            "Convergence Failure",
        ],
        "answer": "Poor mesh quality introduces numerical instability, causing residuals to stagnate or diverge.",
    },
    {
        "question": "How does load cause structural failure?",
        "reasoning_path": ["Load", "Stress", "Strain", "Failure"],
        "answer": "External load produces stress, stress produces strain, and excessive strain leads to failure.",
    },
    {
        "question": "Why does a boundary layer separate?",
        "reasoning_path": [
            "Adverse Pressure Gradient",
            "Boundary Layer",
            "Flow Separation",
            "Drag Increase",
        ],
        "answer": "An adverse pressure gradient slows the boundary layer until flow separates and drag rises.",
    },
    {
        "question": "Why does thermal stress create cracking?",
        "reasoning_path": [
            "Temperature Gradient",
            "Thermal Expansion",
            "Thermal Stress",
            "Cracking",
        ],
        "answer": "Uneven temperature creates differential expansion, thermal stress, and eventually cracking.",
    },
    {
        "question": "Why does cavitation damage pumps?",
        "reasoning_path": ["Low Pressure", "Vapor Bubble", "Collapse", "Surface Erosion"],
        "answer": "Low pressure forms vapor bubbles whose collapse produces local impacts and erosion.",
    },
    {
        "question": "Why does a column buckle under compression?",
        "reasoning_path": ["Compression", "Slenderness", "Lateral Deflection", "Buckling"],
        "answer": "Compression in a slender member amplifies lateral deflection until buckling occurs.",
    },
    {
        "question": "Why does cyclic loading cause fatigue?",
        "reasoning_path": ["Cyclic Load", "Stress Concentration", "Crack Initiation", "Fatigue Failure"],
        "answer": "Repeated loads concentrate stress, initiate cracks, and accumulate fatigue damage.",
    },
    {
        "question": "Why does heat flux increase with temperature gradient?",
        "reasoning_path": ["Temperature Gradient", "Heat Flux", "Heat Transfer", "Cooling Rate"],
        "answer": "A larger temperature gradient increases heat flux and raises the cooling rate.",
    },
    {
        "question": "Why do residuals oscillate during a CFD solve?",
        "reasoning_path": ["Residual", "Under Relaxation", "Numerical Stability", "Convergence"],
        "answer": "Residual oscillation often indicates insufficient damping or unstable numerical updates.",
    },
    {
        "question": "Why can combustion become unstable?",
        "reasoning_path": ["Heat Release", "Pressure Wave", "Acoustic Feedback", "Combustion Instability"],
        "answer": "Heat release can couple with pressure waves, creating acoustic feedback and instability.",
    },
]


def write_example_dataset(path: str | Path = "data/reasoning_dataset.json") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(EXAMPLE_DATA, indent=2), encoding="utf-8")
    return output


def build_graph_from_dataset(dataset_file: str | Path) -> ReasoningGraph:
    samples = load_reasoning_samples(dataset_file)
    return ReasoningGraph.from_paths(sample.reasoning_path for sample in samples)


if __name__ == "__main__":
    target = write_example_dataset()
    dataset = ReasoningDataset(target)
    print(f"wrote {target}")
    print(f"samples={len(dataset)} concepts={dataset.vocab.size()}")

