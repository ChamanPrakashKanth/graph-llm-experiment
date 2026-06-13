# =============================================================================
# losses.py
# CAT Research Loss Framework
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Prediction Loss
# =============================================================================

class PredictionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        logits,
        targets
    ):

        return self.loss_fn(
            logits,
            targets
        )


# =============================================================================
# Contrastive Concept Loss
# =============================================================================

class ContrastiveConceptLoss(nn.Module):
    """
    InfoNCE-style loss

    Pulls related concepts together
    Pushes unrelated concepts apart
    """

    def __init__(
        self,
        temperature=0.07
    ):
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        concept_vectors
    ):
        """
        concept_vectors:
            [batch, num_concepts, dim]
        """

        batch_size = concept_vectors.size(0)

        pooled = concept_vectors.mean(dim=1)

        pooled = F.normalize(
            pooled,
            dim=-1
        )

        similarity = torch.matmul(
            pooled,
            pooled.T
        )

        similarity = (
            similarity /
            self.temperature
        )

        labels = torch.arange(
            batch_size,
            device=concept_vectors.device
        )

        loss = F.cross_entropy(
            similarity,
            labels
        )

        return loss


# =============================================================================
# Directional Consistency Loss
# =============================================================================

class DirectionalConsistencyLoss(nn.Module):
    """
    Encourages smooth gradient evolution
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_features,
        previous_features
    ):
        """
        current_features:
            [batch, hidden]

        previous_features:
            [batch, hidden]
        """

        current_features = F.normalize(
            current_features,
            dim=-1
        )

        previous_features = F.normalize(
            previous_features,
            dim=-1
        )

        cosine = (
            current_features *
            previous_features
        ).sum(dim=-1)

        loss = (
            1.0 - cosine
        ).mean()

        return loss


# =============================================================================
# Curvature Approximation Loss
# =============================================================================

class CurvatureLoss(nn.Module):
    """
    Cheap curvature approximation

    Instead of:

        d²L/dθ²

    Uses:

        ||g_t - g_t-1||
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_hidden,
        previous_hidden
    ):
        """
        current_hidden:
            [batch, hidden]

        previous_hidden:
            [batch, hidden]
        """

        difference = (
            current_hidden -
            previous_hidden
        )

        curvature = torch.norm(
            difference,
            p=2,
            dim=-1
        )

        return curvature.mean()


# =============================================================================
# Attention Entropy Loss
# =============================================================================

class AttentionEntropyLoss(nn.Module):
    """
    Prevents attention collapse
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        attention_weights
    ):
        """
        attention:
            [batch, heads, q, k]
            OR
            [batch, q, k]
        """

        eps = 1e-8

        entropy = -(
            attention_weights *
            torch.log(
                attention_weights + eps
            )
        )

        entropy = entropy.sum(dim=-1)

        entropy = entropy.mean()

        return -entropy


# =============================================================================
# Graph Sparsity Loss
# =============================================================================

class GraphSparsityLoss(nn.Module):
    """
    Prevent graph explosion
    """

    def __init__(
        self,
        target_density=0.10
    ):
        super().__init__()

        self.target_density = target_density

    def forward(
        self,
        edge_index,
        num_nodes
    ):

        edges = edge_index.size(1)

        density = (
            edges /
            (num_nodes * num_nodes)
        )

        loss = (
            density -
            self.target_density
        ) ** 2

        return loss


# =============================================================================
# SNR Metric
# =============================================================================

class SNRMetric:

    @staticmethod
    def compute(loss_history):

        if len(loss_history) < 5:
            return 0.0

        losses = torch.tensor(
            loss_history,
            dtype=torch.float
        )

        signal = losses.mean()

        noise = (
            losses.std() +
            1e-8
        )

        snr = signal / noise

        return snr.item()

    @staticmethod
    def compute_db(loss_history):

        snr = (
            SNRMetric.compute(
                loss_history
            )
        )

        return (
            10 *
            torch.log10(
                torch.tensor(
                    snr + 1e-8
                )
            )
        ).item()


# =============================================================================
# Adaptive Weight Scheduler
# =============================================================================

class AdaptiveWeightScheduler:

    def __init__(self):

        self.pred_weight = 1.0
        self.contrast_weight = 0.10
        self.direction_weight = 0.05
        self.curvature_weight = 0.05
        self.attention_weight = 0.01

    def update(
        self,
        snr_db
    ):

        if snr_db > 15:

            self.direction_weight *= 0.95
            self.curvature_weight *= 0.95

        elif snr_db < 5:

            self.direction_weight *= 1.05
            self.curvature_weight *= 1.05

        return {
            "pred": self.pred_weight,
            "contrast": self.contrast_weight,
            "direction": self.direction_weight,
            "curvature": self.curvature_weight,
            "attention": self.attention_weight
        }


# =============================================================================
# Combined CAT Loss
# =============================================================================

class CATLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.prediction_loss = PredictionLoss()

        self.contrastive_loss = (
            ContrastiveConceptLoss()
        )

        self.direction_loss = (
            DirectionalConsistencyLoss()
        )

        self.curvature_loss = (
            CurvatureLoss()
        )

        self.attention_loss = (
            AttentionEntropyLoss()
        )

    def forward(
        self,
        outputs,
        targets,
        previous_state=None,
        weights=None
    ):

        if weights is None:

            weights = {
                "pred": 1.0,
                "contrast": 0.1,
                "direction": 0.05,
                "curvature": 0.05,
                "attention": 0.01
            }

        logits = outputs["logits"]

        concept_vectors = (
            outputs["concept_vectors"]
        )

        attention_weights = (
            outputs["attention_weights"]
        )

        pred_loss = (
            self.prediction_loss(
                logits,
                targets
            )
        )

        contrast_loss = (
            self.contrastive_loss(
                concept_vectors
            )
        )

        total_loss = (
            weights["pred"] *
            pred_loss
        )

        total_loss += (
            weights["contrast"] *
            contrast_loss
        )

        direction_loss = torch.tensor(
            0.0,
            device=logits.device
        )

        curvature_loss = torch.tensor(
            0.0,
            device=logits.device
        )

        if previous_state is not None:

            current_hidden = (
                outputs["graph_output"]
                .mean(dim=1)
            )

            direction_loss = (
                self.direction_loss(
                    current_hidden,
                    previous_state
                )
            )

            curvature_loss = (
                self.curvature_loss(
                    current_hidden,
                    previous_state
                )
            )

            total_loss += (
                weights["direction"] *
                direction_loss
            )

            total_loss += (
                weights["curvature"] *
                curvature_loss
            )

        entropy_loss = (
            self.attention_loss(
                attention_weights
            )
        )

        total_loss += (
            weights["attention"] *
            entropy_loss
        )

        metrics = {
            "prediction_loss":
                pred_loss.item(),

            "contrastive_loss":
                contrast_loss.item(),

            "direction_loss":
                direction_loss.item(),

            "curvature_loss":
                curvature_loss.item(),

            "attention_entropy":
                entropy_loss.item(),

            "total_loss":
                total_loss.item()
        }

        return (
            total_loss,
            metrics
        )