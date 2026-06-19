import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader

from cat_v3.train import train_cat_v3
from cat_v3.eval import evaluate_model, run_single_inference
from cat_v3.benchmark import run_all_benchmarks
from cat_v3.dataset import CATV3Dataset, grow_dataset, build_expert_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="CAT V3 CLI Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the CAT V3 model")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/cat_v3", help="Directory to save model checkpoints")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model")
    eval_parser.add_argument("--model-path", type=str, default="checkpoints/cat_v3/cat_v3_model.pt", help="Path to saved model checkpoint")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run single question inference")
    infer_parser.add_argument("--question", type=str, required=True, help="Question to analyze")
    infer_parser.add_argument("--model-path", type=str, default="checkpoints/cat_v3/cat_v3_model.pt", help="Path to saved model checkpoint")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run the profiling and scalability benchmarks")

    args = parser.parse_args()

    # Add workspace root to python path to resolve absolute imports correctly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    if args.command == "train":
        print(f"Starting training for {args.epochs} epochs...")
        train_cat_v3(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir
        )

    elif args.command == "evaluate":
        print(f"Loading checkpoint from {args.model_path} for evaluation...")
        if not os.path.exists(args.model_path):
            print(f"Error: checkpoint {args.model_path} does not exist.")
            sys.exit(1)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        vocab = checkpoint["vocab"]
        tokenizer = checkpoint["tokenizer"]
        expert_graphs = checkpoint["expert_graphs"]
        
        from cat_v3.model import CATV3Model
        model = CATV3Model(
            num_concepts=vocab.size(),
            tokenizer_vocab_size=tokenizer.vocab_size(),
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
            expert_graphs=expert_graphs,
            concept_dim=128,
            hidden_size=128,
            path_length=8,
            top_m=8,
            decoder_vocab_size=tokenizer.vocab_size()
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load validation data
        raw_data = grow_dataset()
        dataset = CATV3Dataset(raw_data, concept_vocab=vocab, token_tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        metrics = evaluate_model(model, dataloader, vocab, tokenizer)
        print("\n" + "="*40)
        print("  CAT V3 EVALUATION METRICS")
        print("="*40)
        for k, v in metrics.items():
            if "latency" in k:
                print(f"  {k:30s}: {v:8.2f} ms")
            else:
                print(f"  {k:30s}: {v*100:8.2f}%")
        print("="*40)

    elif args.command == "infer":
        if not os.path.exists(args.model_path):
            print(f"Error: checkpoint {args.model_path} does not exist.")
            sys.exit(1)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        vocab = checkpoint["vocab"]
        tokenizer = checkpoint["tokenizer"]
        expert_graphs = checkpoint["expert_graphs"]
        
        from cat_v3.model import CATV3Model
        model = CATV3Model(
            num_concepts=vocab.size(),
            tokenizer_vocab_size=tokenizer.vocab_size(),
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
            expert_graphs=expert_graphs,
            concept_dim=128,
            hidden_size=128,
            path_length=8,
            top_m=8,
            decoder_vocab_size=tokenizer.vocab_size()
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        res = run_single_inference(model, args.question, vocab, tokenizer)
        print("\n" + "="*50)
        print(f"QUERY: \"{res['question']}\"")
        print("="*50)
        print(f"ACTIVATED EXPERTS: {res['activated_domains']}")
        print(f"DOMAIN PROBABILITIES:")
        for dom, prob in res['domain_probabilities'].items():
            print(f"  - {dom:12s}: {prob:.4f}")
            
        print("\nEXPERT REASONING PATHS:")
        for dom, path in res['expert_paths'].items():
            print(f"  - {dom:12s}: {' -> '.join(path)}")
            
        print("\nFUSED CONCEPT GRAPH:")
        print(f"  - Concepts: {res['fusion_report']['concepts']}")
        print(f"  - Paths: {res['fusion_report']['reasoning_paths']}")
        print(f"  - Confidences: {res['fusion_report']['confidence']}")
        
        print(f"\nDECODER RESPONSE:\n\"{res['answer']}\"")
        print("="*50)

    elif args.command == "benchmark":
        run_all_benchmarks()


if __name__ == "__main__":
    main()
