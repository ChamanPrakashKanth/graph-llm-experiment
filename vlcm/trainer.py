"""Training and checkpoint loading loops for VLCM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from reasoning_dataset import ConceptVocabulary, SimpleTokenizer
from vlcm.graph_reasoner import ReasoningGraph
from vlcm.path_generator import VLCMModel
from vlcm.reasoning_loss import VLCMReasoningLoss, compute_path_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VLCMTrainer:
    """Supervised trainer for Very Large Concepts Model (VLCM)."""

    def __init__(
        self,
        model: VLCMModel,
        train_loader: DataLoader,
        vocab: ConceptVocabulary,
        graph: ReasoningGraph,
        tokenizer: SimpleTokenizer,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        checkpoint_dir: str | Path = "checkpoints/vlcm",
        device: Optional[str] = None,
        second_order_weight: float = 0.0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.graph = graph
        self.tokenizer = tokenizer
        self.gradient_clip = gradient_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = VLCMReasoningLoss(
            pad_id=vocab.pad_id,
            second_order_weight=second_order_weight,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[Dict[str, float]] = []

    def _move_batch(self, batch: Dict[str, object]) -> Dict[str, object]:
        moved = dict(batch)
        for key in ("input_ids", "attention_mask", "path_ids", "activation_targets"):
            moved[key] = batch[key].to(self.device)
        return moved

    def train_step(self, batch: Dict[str, object]) -> Dict[str, float]:
        self.model.train()
        batch = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            target_paths=batch["path_ids"],
        )
        loss, loss_metrics = self.loss_fn(
            outputs,
            batch["path_ids"],
            batch["activation_targets"],
            transition_mask=self.model.transition_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()

        path_metrics = compute_path_metrics(
            outputs["predicted_path"].detach(),
            batch["path_ids"].detach(),
            pad_id=self.vocab.pad_id,
            eos_id=self.vocab.eos_id,
        )
        return {**loss_metrics, **path_metrics}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        progress = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            metrics = self.train_step(batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            progress.set_postfix(loss=f"{metrics['total_loss']:.4f}")

        return {
            key: value / max(len(self.train_loader), 1)
            for key, value in totals.items()
        }

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        eval_loader = loader or self.val_loader or self.train_loader
        self.model.eval()
        totals: Dict[str, float] = {}

        for batch in eval_loader:
            batch = self._move_batch(batch)
            teacher_outputs = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                target_paths=batch["path_ids"],
            )
            loss, loss_metrics = self.loss_fn(
                teacher_outputs,
                batch["path_ids"],
                batch["activation_targets"],
                transition_mask=self.model.transition_mask,
            )
            autoregressive_outputs = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                target_paths=None,
            )
            path_metrics = compute_path_metrics(
                autoregressive_outputs["predicted_path"],
                batch["path_ids"],
                pad_id=self.vocab.pad_id,
                eos_id=self.vocab.eos_id,
            )
            metrics = {**loss_metrics, **path_metrics, "eval_loss": float(loss.cpu())}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)

        return {
            key: value / max(len(eval_loader), 1)
            for key, value in totals.items()
        }

    def fit(self, epochs: int = 5, save_every: int = 1) -> Dict[str, float]:
        last_metrics: Dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            eval_metrics = self.evaluate()
            last_metrics = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"eval_{key}": value for key, value in eval_metrics.items()},
            }
            self.history.append(last_metrics)
            if save_every and epoch % save_every == 0:
                self.save_checkpoint(epoch, last_metrics)
            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics['total_loss']:.4f} "
                f"eval_f1={eval_metrics['concept_f1']:.4f}"
            )
        return last_metrics

    def checkpoint_payload(
        self,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.model.config,
            "vocab": self.vocab.to_dict(),
            "graph": self.graph.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "metrics": metrics or {},
            "history": self.history,
        }

    def save_checkpoint(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> Path:
        path = self.checkpoint_dir / f"vlcm_epoch_{epoch}.pt"
        torch.save(self.checkpoint_payload(epoch, metrics), path)
        return path


def load_vlcm_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> Dict[str, object]:
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    vocab = ConceptVocabulary.from_dict(payload["vocab"])
    graph = ReasoningGraph.from_dict(payload["graph"])
    tokenizer = SimpleTokenizer.from_dict(payload["tokenizer"])
    config = dict(payload["config"])

    model = VLCMModel(
        num_concepts=vocab.size(),
        tokenizer_vocab_size=tokenizer.vocab_size,
        graph=graph,
        vocab=vocab,
        concept_dim=int(config.get("concept_dim", 128)),
        path_length=int(config.get("path_length", 8)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_hidden_layers=int(config.get("num_hidden_layers", 2)),
        num_attention_heads=int(config.get("num_attention_heads", 4)),
        intermediate_size=int(config.get("intermediate_size", 256)),
        graph_layers=int(config.get("graph_layers", 2)),
        top_k=int(config.get("top_k", 5)),
        dropout=float(config.get("dropout", 0.1)),
        pretrained_encoder_name=config.get("pretrained_encoder_name", None),
    )
    model.load_state_dict(payload["model_state"])
    model.to(target_device)
    model.eval()
    return {
        "model": model,
        "vocab": vocab,
        "graph": graph,
        "tokenizer": tokenizer,
        "payload": payload,
        "device": target_device,
    }


def latest_vlcm_checkpoint(checkpoint_dir: str | Path = "checkpoints/vlcm") -> Optional[Path]:
    checkpoints = list(Path(checkpoint_dir).glob("vlcm_epoch_*.pt"))
    if not checkpoints:
        return None
    def get_epoch(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0
    checkpoints.sort(key=get_epoch)
    return checkpoints[-1]
