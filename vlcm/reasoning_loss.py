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


class SecondOrderDifferentialLoss(nn.Module):
    """Penalize abrupt changes (acceleration) in the path logit trajectory.

    Computes the discrete second derivative of path logits across time steps:
        d2[t] = logits[t+1] - 2*logits[t] + logits[t-1]
    and returns the mean squared magnitude of these acceleration vectors.

    This encourages smooth transitions through concept space, helping the
    model avoid attractor traps and erratic path jumps at scale.
    """

    def __init__(self, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id

    def forward(
        self,
        path_logits: torch.Tensor,
        target_paths: torch.Tensor,
    ) -> torch.Tensor:
        # path_logits shape: (B, T, V)
        # Need at least 3 time steps to compute second-order differences
        if path_logits.size(1) < 3:
            return path_logits.new_tensor(0.0)

        # Use softmax probabilities for smoother gradients
        probs = F.softmax(path_logits, dim=-1)  # B x T x V

        # Compute first-order differences (velocity): delta[t] = probs[t+1] - probs[t]
        first_diff = probs[:, 1:, :] - probs[:, :-1, :]  # B x (T-1) x V

        # Compute second-order differences (acceleration): d2[t] = delta[t+1] - delta[t]
        second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]  # B x (T-2) x V

        # Build validity mask: only penalize steps where all three involved
        # path positions are non-padding
        valid = (
            (target_paths[:, :-2] != self.pad_id)
            & (target_paths[:, 1:-1] != self.pad_id)
            & (target_paths[:, 2:] != self.pad_id)
        )  # B x (T-2)

        if not torch.any(valid):
            return path_logits.new_tensor(0.0)

        # Mean squared acceleration over valid positions
        sq_accel = (second_diff ** 2).sum(dim=-1)  # B x (T-2)
        masked_accel = sq_accel * valid.float()
        return masked_accel.sum() / valid.float().sum().clamp_min(1.0)


class VLCMReasoningLoss(nn.Module):
    """Combined VLCM objective function."""

    def __init__(
        self,
        pad_id: int = 0,
        path_weight: float = 1.0,
        activation_weight: float = 0.2,
        transition_weight: float = 0.05,
        second_order_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.path_weight = path_weight
        self.activation_weight = activation_weight
        self.transition_weight = transition_weight
        self.second_order_weight = second_order_weight
        self.path_loss = PathPredictionLoss(pad_id=pad_id)
        self.activation_loss = ConceptActivationLoss()
        self.transition_loss = TransitionPriorLoss(pad_id=pad_id)
        self.second_order_loss = SecondOrderDifferentialLoss(pad_id=pad_id)

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
        second_order = self.second_order_loss(
            outputs["path_logits"],
            target_paths,
        )

        total = (
            self.path_weight * path
            + self.activation_weight * activation
            + self.transition_weight * transition
            + self.second_order_weight * second_order
        )

        return total, {
            "path_loss": float(path.detach().cpu()),
            "activation_loss": float(activation.detach().cpu()),
            "transition_loss": float(transition.detach().cpu()),
            "second_order_loss": float(second_order.detach().cpu()),
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
