"""Supervised losses and path metrics for VLCM (Very Large Concepts Model)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathPredictionLoss(nn.Module):
    """Cross-entropy loss over predicted concept sequence."""

    def __init__(self, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, path_logits: torch.Tensor, target_paths: torch.Tensor) -> torch.Tensor:
        batch, length, vocab_size = path_logits.shape
        return self.loss_fn(
            path_logits.reshape(batch * length, vocab_size),
            target_paths.reshape(batch * length),
        )


class ConceptActivationLoss(nn.Module):
    """Multi-label BCE loss for query concept activations."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        activation_logits: torch.Tensor,
        activation_targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(activation_logits, activation_targets)


class TransitionPriorLoss(nn.Module):
    """Penalize model for violating graph transition rules."""

    def __init__(self, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id

    def forward(
        self,
        path_logits: torch.Tensor,
        target_paths: torch.Tensor,
        transition_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if transition_mask is None or path_logits.size(1) < 2:
            return path_logits.new_tensor(0.0)

        penalties = []
        probabilities = F.softmax(path_logits[:, 1:], dim=-1)
        previous_ids = target_paths[:, :-1]
        valid_positions = previous_ids != self.pad_id

        for step in range(probabilities.size(1)):
            valid = valid_positions[:, step]
            if not torch.any(valid):
                continue
            prev = previous_ids[valid, step]
            allowed = transition_mask[prev]
            invalid_mass = probabilities[valid, step].masked_fill(allowed, 0.0).sum(dim=-1)
            penalties.append(invalid_mass.mean())

        if not penalties:
            return path_logits.new_tensor(0.0)
        return torch.stack(penalties).mean()


class VLCMReasoningLoss(nn.Module):
    """Combined VLCM objective function."""

    def __init__(
        self,
        pad_id: int = 0,
        path_weight: float = 1.0,
        activation_weight: float = 0.2,
        transition_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.path_weight = path_weight
        self.activation_weight = activation_weight
        self.transition_weight = transition_weight
        self.path_loss = PathPredictionLoss(pad_id=pad_id)
        self.activation_loss = ConceptActivationLoss()
        self.transition_loss = TransitionPriorLoss(pad_id=pad_id)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_paths: torch.Tensor,
        activation_targets: torch.Tensor,
        transition_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        path = self.path_loss(outputs["path_logits"], target_paths)
        activation = self.activation_loss(
            outputs["activation_logits"],
            activation_targets,
        )
        transition = self.transition_loss(
            outputs["path_logits"],
            target_paths,
            transition_mask,
        )

        total = (
            self.path_weight * path
            + self.activation_weight * activation
            + self.transition_weight * transition
        )

        return total, {
            "path_loss": float(path.detach().cpu()),
            "activation_loss": float(activation.detach().cpu()),
            "transition_loss": float(transition.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }


def strip_path(path: torch.Tensor, pad_id: int = 0, eos_id: int = 1) -> list[int]:
    stripped: list[int] = []
    for raw_id in path.detach().cpu().tolist():
        concept_id = int(raw_id)
        if concept_id == pad_id:
            continue
        if concept_id == eos_id:
            break
        stripped.append(concept_id)
    return stripped


def compute_path_metrics(
    predicted_paths: torch.Tensor,
    target_paths: torch.Tensor,
    pad_id: int = 0,
    eos_id: int = 1,
) -> Dict[str, float]:
    exact = 0
    token_correct = 0
    token_total = 0
    transition_correct = 0
    transition_total = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    batch_size = predicted_paths.size(0)

    for predicted, target in zip(predicted_paths, target_paths):
        pred = strip_path(predicted, pad_id=pad_id, eos_id=eos_id)
        gold = strip_path(target, pad_id=pad_id, eos_id=eos_id)

        exact += int(pred == gold)

        max_len = min(len(pred), len(gold))
        for idx in range(max_len):
            token_correct += int(pred[idx] == gold[idx])
        token_total += max(len(gold), 1)

        pred_transitions = set(zip(pred, pred[1:]))
        gold_transitions = set(zip(gold, gold[1:]))
        transition_correct += len(pred_transitions & gold_transitions)
        transition_total += max(len(gold_transitions), 1)

        pred_set = set(pred)
        gold_set = set(gold)
        overlap = len(pred_set & gold_set)
        precision = overlap / max(len(pred_set), 1)
        recall = overlap / max(len(gold_set), 1)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    return {
        "exact_match": exact / max(batch_size, 1),
        "token_accuracy": token_correct / max(token_total, 1),
        "transition_accuracy": transition_correct / max(transition_total, 1),
        "concept_precision": precision_sum / max(batch_size, 1),
        "concept_recall": recall_sum / max(batch_size, 1),
        "concept_f1": f1_sum / max(batch_size, 1),
    }
