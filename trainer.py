# =============================================================================
# trainer.py
# CAT Research Training Framework
# =============================================================================

import os
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from graph_builder import DynamicGraphBuilder
from losses import CATLoss
from losses import AdaptiveWeightScheduler
from losses import SNRMetric


# =============================================================================
# CAT Trainer
# =============================================================================

class CATTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=2e-5,
        weight_decay=0.01,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        device=None
    ):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model.to(
            self.device
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.loss_fn = CATLoss()

        self.scaler = GradScaler()

        self.graph_builder = (
            DynamicGraphBuilder()
        )

        self.weight_scheduler = (
            AdaptiveWeightScheduler()
        )

        self.loss_history = []

        self.global_step = 0

        self.checkpoint_dir = (
            Path(checkpoint_dir)
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.writer = SummaryWriter(
            log_dir=log_dir
        )

    # =========================================================================
    # Training Epoch
    # =========================================================================

    def train_epoch(
        self,
        epoch
    ):

        self.model.train()

        running_loss = 0.0

        previous_state = None

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}"
        )

        for batch in progress:

            input_ids = (
                batch["input_ids"]
                .to(self.device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(self.device)
            )

            batch_size = (
                input_ids.size(0)
            )

            edge_index = self._build_batch_graph(
                batch_size
            )

            targets = torch.randint(
                0,
                50000,
                (batch_size,),
                device=self.device
            )

            self.optimizer.zero_grad()

            with autocast():

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    edge_index=edge_index
                )

                weights = (
                    self.weight_scheduler
                    .update(
                        self.current_snr()
                    )
                )

                loss, metrics = (
                    self.loss_fn(
                        outputs,
                        targets,
                        previous_state,
                        weights
                    )
                )

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            previous_state = (
                outputs[
                    "graph_output"
                ]
                .mean(dim=1)
                .detach()
            )

            running_loss += (
                loss.item()
            )

            self.loss_history.append(
                loss.item()
            )

            self.log_metrics(
                metrics
            )

            progress.set_postfix({

                "loss":
                    f"{loss.item():.4f}",

                "snr":
                    f"{self.current_snr():.2f}"
            })

            self.global_step += 1

        epoch_loss = (
            running_loss /
            len(self.train_loader)
        )

        return epoch_loss

    # =========================================================================
    # Validation
    # =========================================================================

    @torch.no_grad()
    def validate(self):

        if self.val_loader is None:
            return None

        self.model.eval()

        total_loss = 0

        for batch in self.val_loader:

            input_ids = (
                batch["input_ids"]
                .to(self.device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(self.device)
            )

            batch_size = (
                input_ids.size(0)
            )

            edge_index = self._build_batch_graph(
                batch_size
            )

            targets = torch.randint(
                0,
                50000,
                (batch_size,),
                device=self.device
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                edge_index=edge_index
            )

            loss, _ = (
                self.loss_fn(
                    outputs,
                    targets
                )
            )

            total_loss += (
                loss.item()
            )

        return (
            total_loss /
            len(self.val_loader)
        )

    # =========================================================================
    # Graph Creation
    # =========================================================================

    def _build_batch_graph(
        self,
        batch_size
    ):

        concepts = torch.randn(
            batch_size,
            self.model.num_concepts,
            self.model.concept_dim,
            device=self.device
        )

        edge_index = (
            self.graph_builder
            .build_batch_graph(
                concepts
            )
        )

        return edge_index.to(
            self.device
        )

    # =========================================================================
    # Logging
    # =========================================================================

    def log_metrics(
        self,
        metrics
    ):

        for key, value in metrics.items():

            self.writer.add_scalar(
                key,
                value,
                self.global_step
            )

        self.writer.add_scalar(
            "snr",
            self.current_snr(),
            self.global_step
        )

    # =========================================================================
    # Checkpoint
    # =========================================================================

    def save_checkpoint(
        self,
        epoch,
        val_loss=None
    ):

        path = (
            self.checkpoint_dir /
            f"cat_epoch_{epoch}.pt"
        )

        torch.save({

            "epoch":
                epoch,

            "model":
                self.model.state_dict(),

            "optimizer":
                self.optimizer.state_dict(),

            "loss_history":
                self.loss_history,

            "val_loss":
                val_loss

        }, path)

        print(
            f"Checkpoint saved: {path}"
        )

    # =========================================================================
    # SNR
    # =========================================================================

    def current_snr(
        self
    ):

        return (
            SNRMetric.compute_db(
                self.loss_history
            )
        )

    # =========================================================================
    # Train Loop
    # =========================================================================

    def fit(
        self,
        epochs
    ):

        best_val = float("inf")

        for epoch in range(
            1,
            epochs + 1
        ):

            train_loss = (
                self.train_epoch(
                    epoch
                )
            )

            val_loss = (
                self.validate()
            )

            print(
                f"\nEpoch {epoch}"
            )

            print(
                f"Train Loss: {train_loss:.4f}"
            )

            if val_loss is not None:

                print(
                    f"Val Loss: {val_loss:.4f}"
                )

                if val_loss < best_val:

                    best_val = val_loss

                    self.save_checkpoint(
                        epoch,
                        val_loss
                    )

            else:

                self.save_checkpoint(
                    epoch
                )

        self.writer.close()


# =============================================================================
# Utility
# =============================================================================

def count_parameters(
    model
):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":

    print(
        "trainer.py loaded successfully"
    )