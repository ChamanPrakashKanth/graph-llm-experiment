# =============================================================================
# config.py
# CAT Research Configuration
# =============================================================================

from dataclasses import dataclass


# =============================================================================
# Model Config
# =============================================================================

@dataclass
class ModelConfig:

    encoder_name: str = "bert-base-uncased"

    hidden_dim: int = 768

    concept_dim: int = 512

    num_concepts: int = 64

    num_transformer_layers: int = 4

    num_attention_heads: int = 8

    num_classes: int = 50000

    dropout: float = 0.1


# =============================================================================
# Graph Config
# =============================================================================

@dataclass
class GraphConfig:

    top_k: int = 8

    similarity_threshold: float = 0.20

    self_loops: bool = False


# =============================================================================
# Loss Config
# =============================================================================

@dataclass
class LossConfig:

    prediction_weight: float = 1.0

    contrastive_weight: float = 0.10

    directional_weight: float = 0.05

    curvature_weight: float = 0.05

    attention_weight: float = 0.01


# =============================================================================
# Training Config
# =============================================================================

@dataclass
class TrainingConfig:

    epochs: int = 20

    batch_size: int = 8

    learning_rate: float = 2e-5

    weight_decay: float = 0.01

    gradient_clip: float = 1.0

    mixed_precision: bool = True

    save_every: int = 1

    log_every: int = 10


# =============================================================================
# Dataset Config
# =============================================================================

@dataclass
class DatasetConfig:

    tokenizer_name: str = "bert-base-uncased"

    max_length: int = 256

    chunk_size: int = 8

    overlap: int = 2


# =============================================================================
# Experiment Config
# =============================================================================

@dataclass
class ExperimentConfig:

    experiment_name: str = "CAT_CFD_Research"

    dataset_path: str = "data"

    checkpoint_dir: str = "checkpoints"

    log_dir: str = "logs"

    report_dir: str = "reports"


# =============================================================================
# Master Config
# =============================================================================

@dataclass
class CATConfig:

    model = ModelConfig()

    graph = GraphConfig()

    loss = LossConfig()

    training = TrainingConfig()

    dataset = DatasetConfig()

    experiment = ExperimentConfig()


CONFIG = CATConfig()