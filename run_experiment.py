# =============================================================================
# run_experiment.py
# CAT Research Pipeline
# =============================================================================

import torch

from pathlib import Path
from torch.utils.data import DataLoader

from config import CONFIG

from dataset import (
    CATDataset,
    CATCollator,
    CorpusLoader
)

from trainer import (
    CATTrainer,
    count_parameters
)

from cat_model import (
    ConceptAttentionTransformer
)


# =============================================================================
# Device
# =============================================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# Load Corpus
# =============================================================================

def load_corpus():

    dataset_path = (
        CONFIG.experiment.dataset_path
    )

    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"{dataset_path} not found"
        )

    documents = (
        CorpusLoader
        .load_txt_folder(
            dataset_path
        )
    )

    print(
        f"Loaded {len(documents)} documents"
    )

    return documents


# =============================================================================
# Dataset
# =============================================================================

def build_dataset(
    documents
):

    dataset = CATDataset(
        documents=documents,
        tokenizer_name=(
            CONFIG.dataset
            .tokenizer_name
        ),
        max_length=(
            CONFIG.dataset
            .max_length
        )
    )

    return dataset


# =============================================================================
# DataLoader
# =============================================================================

def build_dataloader(
    dataset
):

    loader = DataLoader(

        dataset,

        batch_size=(
            CONFIG.training
            .batch_size
        ),

        shuffle=True,

        collate_fn=(
            CATCollator()
        ),

        num_workers=0
    )

    return loader


# =============================================================================
# Model
# =============================================================================

def build_model():

    model = (
        ConceptAttentionTransformer(

            encoder_name=
            CONFIG.model.encoder_name,

            concept_dim=
            CONFIG.model.concept_dim,

            hidden_dim=
            CONFIG.model.hidden_dim,

            num_concepts=
            CONFIG.model.num_concepts,

            num_transformer_layers=
            CONFIG.model
            .num_transformer_layers,

            num_classes=
            CONFIG.model.num_classes,

            dropout=
            CONFIG.model.dropout
        )
    )

    return model


# =============================================================================
# Trainer
# =============================================================================

def build_trainer(
    model,
    train_loader
):

    trainer = CATTrainer(

        model=model,

        train_loader=train_loader,

        val_loader=None,

        lr=(
            CONFIG.training
            .learning_rate
        ),

        weight_decay=(
            CONFIG.training
            .weight_decay
        ),

        checkpoint_dir=(
            CONFIG.experiment
            .checkpoint_dir
        ),

        log_dir=(
            CONFIG.experiment
            .log_dir
        ),

        device=DEVICE
    )

    return trainer


# =============================================================================
# Report
# =============================================================================

def save_report(
    trainer,
    model
):

    report_dir = Path(
        CONFIG.experiment
        .report_dir
    )

    report_dir.mkdir(
        exist_ok=True
    )

    report_file = (
        report_dir /
        "final_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=================================\n"
        )

        f.write(
            "CAT EXPERIMENT REPORT\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Parameters: "
            f"{count_parameters(model)}\n"
        )

        f.write(
            f"Final SNR: "
            f"{trainer.current_snr():.4f}\n"
        )

        f.write(
            f"Training Steps: "
            f"{trainer.global_step}\n"
        )

        if len(
            trainer.loss_history
        ) > 0:

            f.write(
                f"Final Loss: "
                f"{trainer.loss_history[-1]:.6f}\n"
            )

    print(
        f"Report saved to "
        f"{report_file}"
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "CONCEPT ATTENTION TRANSFORMER"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {DEVICE}"
    )

    print()

    # ---------------------------------------------------------
    # Corpus
    # ---------------------------------------------------------

    documents = (
        load_corpus()
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = (
        build_dataset(
            documents
        )
    )

    print(
        f"Dataset Size: "
        f"{len(dataset)}"
    )

    # ---------------------------------------------------------
    # Loader
    # ---------------------------------------------------------

    loader = (
        build_dataloader(
            dataset
        )
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = (
        build_model()
    )

    print()

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    print()

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = (
        build_trainer(
            model,
            loader
        )
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    trainer.fit(

        epochs=(
            CONFIG.training
            .epochs
        )
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    save_report(
        trainer,
        model
    )

    print()

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


# =============================================================================
# Entry
# =============================================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            42
        )

    main()