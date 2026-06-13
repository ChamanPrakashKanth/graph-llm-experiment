"""Canonical CAT V2 command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

from answer_decoder import CATReasoningSystem, OptionalLLMAnswerDecoder, TemplateAnswerDecoder
from cat_reasoning_model import build_model_from_dataset
from reasoning_dataset import (
    ReasoningCollator,
    ReasoningDataset,
    SimpleTokenizer,
    build_graph_from_dataset,
    write_example_dataset,
)
from reasoning_graph import ReasoningGraph
from reasoning_loss import compute_path_metrics
from reasoning_trainer import (
    CATReasoningTrainer,
    latest_checkpoint,
    load_checkpoint,
    set_seed,
)


def ensure_dataset(path: str | Path) -> Path:
    dataset_path = Path(path)
    if not dataset_path.exists():
        write_example_dataset(dataset_path)
    return dataset_path


def build_dataset_and_loader(args, shuffle: bool = False):
    dataset_path = ensure_dataset(args.dataset)
    dataset = ReasoningDataset(
        dataset_path,
        max_length=args.max_length,
        max_path_length=args.path_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=ReasoningCollator(),
        num_workers=0,
    )
    return dataset, loader


def resolve_checkpoint(args) -> Optional[Path]:
    if getattr(args, "checkpoint", None):
        return Path(args.checkpoint)
    return latest_checkpoint(getattr(args, "checkpoint_dir", "checkpoints/cat_v2"))


def train_command(args) -> None:
    set_seed(args.seed)
    dataset, loader = build_dataset_and_loader(args, shuffle=True)
    model = build_model_from_dataset(
        dataset,
        concept_dim=args.concept_dim,
        hidden_size=args.hidden_size,
        path_length=args.path_length,
        num_hidden_layers=args.encoder_layers,
        num_attention_heads=args.attention_heads,
        intermediate_size=args.intermediate_size,
        graph_layers=args.graph_layers,
        top_k=args.top_k,
        dropout=args.dropout,
        pretrained_encoder_name=args.pretrained_encoder,
    )
    trainer = CATReasoningTrainer(
        model=model,
        train_loader=loader,
        vocab=dataset.vocab,
        graph=dataset.graph,
        tokenizer=dataset.tokenizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    print(f"samples={len(dataset)} concepts={dataset.vocab.size()} parameters={sum(p.numel() for p in model.parameters()):,}")
    metrics = trainer.fit(epochs=args.epochs, save_every=args.save_every)
    final_checkpoint = latest_checkpoint(args.checkpoint_dir)
    print(json.dumps(metrics, indent=2))
    if final_checkpoint:
        print(f"checkpoint={final_checkpoint}")


def load_or_build_system(args) -> CATReasoningSystem:
    checkpoint = resolve_checkpoint(args)
    if checkpoint and checkpoint.exists():
        loaded = load_checkpoint(checkpoint, device=args.device)
        decoder = (
            OptionalLLMAnswerDecoder(args.decoder_model, device=loaded["device"])
            if getattr(args, "decoder_model", None)
            else TemplateAnswerDecoder()
        )
        return CATReasoningSystem(
            loaded["model"],
            loaded["vocab"],
            loaded["tokenizer"],
            decoder=decoder,
            device=loaded["device"],
        )

    dataset, _ = build_dataset_and_loader(args, shuffle=False)
    model = build_model_from_dataset(
        dataset,
        concept_dim=args.concept_dim,
        hidden_size=args.hidden_size,
        path_length=args.path_length,
        num_hidden_layers=args.encoder_layers,
        num_attention_heads=args.attention_heads,
        intermediate_size=args.intermediate_size,
        graph_layers=args.graph_layers,
        top_k=args.top_k,
        dropout=args.dropout,
    )
    return CATReasoningSystem(
        model,
        dataset.vocab,
        dataset.tokenizer,
        decoder=TemplateAnswerDecoder(),
        device=args.device,
    )


def infer_command(args) -> None:
    system = load_or_build_system(args)
    result = system.answer(args.question, max_length=args.max_length)
    print("Question:")
    print(result["question"])
    print("\nActivated Concepts:")
    print(" -> ".join(result["activated_concepts"]))
    print("\nReasoning Path:")
    print(result["reasoning_path_text"])
    print("\nAnswer:")
    print(result["answer"])


@torch.no_grad()
def evaluate_command(args) -> None:
    checkpoint = resolve_checkpoint(args)
    if not checkpoint or not checkpoint.exists():
        raise FileNotFoundError("evaluation requires a checkpoint; run train first or pass --checkpoint")

    loaded = load_checkpoint(checkpoint, device=args.device)
    dataset = ReasoningDataset(
        ensure_dataset(args.dataset),
        max_length=args.max_length,
        max_path_length=int(loaded["model"].config["path_length"]),
        tokenizer=loaded["tokenizer"],
        vocab=loaded["vocab"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ReasoningCollator(),
        num_workers=0,
    )
    totals = {}
    model = loaded["model"]
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(loaded["device"])
        attention_mask = batch["attention_mask"].to(loaded["device"])
        target = batch["path_ids"].to(loaded["device"])
        outputs = model(input_ids, attention_mask)
        metrics = compute_path_metrics(
            outputs["predicted_path"],
            target,
            pad_id=loaded["vocab"].pad_id,
            eos_id=loaded["vocab"].eos_id,
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)

    averaged = {key: value / max(len(loader), 1) for key, value in totals.items()}
    print(json.dumps(averaged, indent=2))


def visualize_command(args) -> None:
    checkpoint = resolve_checkpoint(args)
    highlight_path = None
    if checkpoint and checkpoint.exists():
        loaded = load_checkpoint(checkpoint, device=args.device)
        graph = loaded["graph"]
        if args.question:
            system = CATReasoningSystem(
                loaded["model"],
                loaded["vocab"],
                loaded["tokenizer"],
                decoder=TemplateAnswerDecoder(),
                device=loaded["device"],
            )
            highlight_path = system.answer(args.question, max_length=args.max_length)["reasoning_path"]
    else:
        dataset = ReasoningDataset(
            ensure_dataset(args.dataset),
            max_length=args.max_length,
            max_path_length=args.path_length,
        )
        graph = dataset.graph
    output = graph.visualize(args.output, highlight_path=highlight_path)
    print(f"graph_png={output}")


def read_documents(paths: Iterable[str | Path]) -> list[str]:
    documents: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for file in sorted(path.glob("*.txt")):
                documents.append(file.read_text(encoding="utf-8"))
        elif path.exists():
            documents.append(path.read_text(encoding="utf-8"))
    return documents


def grow_graph_command(args) -> None:
    dataset = ReasoningDataset(
        ensure_dataset(args.dataset),
        max_length=args.max_length,
        max_path_length=args.path_length,
    )
    graph = ReasoningGraph.from_dict(dataset.graph.to_dict())
    documents = read_documents(args.documents)
    if not documents:
        raise FileNotFoundError("no readable document files were found")

    graph.grow_from_documents(
        documents,
        concepts=dataset.vocab.concept_names(),
        window=args.window,
    )
    graph.save_json(args.output)
    if args.png:
        graph.visualize(args.png)
    print(f"graph_json={args.output}")
    if args.png:
        print(f"graph_png={args.png}")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="data/reasoning_dataset.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/cat_v2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--path-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concept-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAT V2 reasoning path prototype")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train")
    add_shared_arguments(train)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--save-every", type=int, default=1)
    train.add_argument("--pretrained-encoder", default=None)
    train.set_defaults(func=train_command)

    infer = subcommands.add_parser("infer")
    add_shared_arguments(infer)
    infer.add_argument("--question", required=True)
    infer.add_argument("--decoder-model", default=None)
    infer.set_defaults(func=infer_command)

    evaluate = subcommands.add_parser("evaluate")
    add_shared_arguments(evaluate)
    evaluate.set_defaults(func=evaluate_command)

    visualize = subcommands.add_parser("visualize")
    add_shared_arguments(visualize)
    visualize.add_argument("--output", default="reports/cat_v2_graph.png")
    visualize.add_argument("--question", default=None)
    visualize.set_defaults(func=visualize_command)

    grow = subcommands.add_parser("grow-graph")
    add_shared_arguments(grow)
    grow.add_argument("--documents", nargs="+", default=["data"])
    grow.add_argument("--output", default="reports/cat_v2_graph.json")
    grow.add_argument("--png", default="reports/cat_v2_grown_graph.png")
    grow.add_argument("--window", type=int, default=2)
    grow.set_defaults(func=grow_graph_command)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

