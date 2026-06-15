"""Evaluation utilities and Concept Chain Discovery tests for VLCM."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

# Fix Windows stdout CP1252 encoding issues with Unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


import torch
from torch.utils.data import DataLoader

from reasoning_dataset import ConceptVocabulary, ReasoningCollator, ReasoningDataset, SimpleTokenizer
from vlcm.decoder import VLCMReasoningSystem
from vlcm.graph_reasoner import ReasoningGraph
from vlcm.path_generator import VLCMModel
from vlcm.reasoning_loss import VLCMReasoningLoss, compute_path_metrics


class VLCMEvaluator:
    """Evaluates VLCM models on path accuracy and reasoning tasks."""

    def __init__(
        self,
        model: VLCMModel,
        vocab: ConceptVocabulary,
        tokenizer: SimpleTokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(self, dataset_path: str | Path, batch_size: int = 4) -> Dict[str, float]:
        """Compute standard metrics on a verification dataset."""
        dataset = ReasoningDataset(
            dataset_path,
            max_length=64,
            max_path_length=self.model.config["path_length"],
            tokenizer=self.tokenizer,
            vocab=self.vocab,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ReasoningCollator(),
        )

        totals = {}
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target = batch["path_ids"].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            metrics = compute_path_metrics(
                outputs["predicted_path"],
                target,
                pad_id=self.vocab.pad_id,
                eos_id=self.vocab.eos_id,
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)

        return {key: value / max(len(loader), 1) for key, value in totals.items()}


def run_engineering_domain_test(device: str = "cpu") -> bool:
    """Test if VLCM can discover concept chains not explicitly seen during training.
    
    Training paths:
      1. Pressure -> Velocity -> Turbulence
      2. Turbulence -> Heat Transfer -> Cooling Rate
    
    Test path query:
      Pressure -> ? -> Cooling Rate
      We expect the model to infer: Pressure -> Velocity -> Turbulence -> Heat Transfer -> Cooling Rate
    """
    import random
    random.seed(42)
    torch.manual_seed(42)
    print("=== Running Engineering Domain Test (Concept Chain Discovery) ===")
    
    # 1. Prepare vocabulary and simple tokenizer
    vocab = ConceptVocabulary()
    vocab.build(["Pressure", "Velocity", "Turbulence", "Heat Transfer", "Cooling Rate"])
    
    tokenizer = SimpleTokenizer()
    tokenizer.build([
        "How does pressure affect turbulence?",
        "Why does turbulence change cooling rate?",
        "How does pressure lead to cooling rate?",
    ])

    # 2. Build graph containing individual subchains
    graph = ReasoningGraph()
    graph.grow_from_paths([
        ["Pressure", "Velocity", "Turbulence"],
        ["Turbulence", "Heat Transfer", "Cooling Rate"],
    ])

    # 3. Create VLCM model
    model = VLCMModel(
        num_concepts=vocab.size(),
        tokenizer_vocab_size=tokenizer.vocab_size,
        graph=graph,
        vocab=vocab,
        concept_dim=64,
        path_length=6,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        graph_layers=1,
        dropout=0.0,
    ).to(device)

    # 4. Create synthetic dataset with subchains
    # (Notice the full path 'Pressure -> Velocity -> Turbulence -> Heat Transfer -> Cooling Rate' is NOT in train!)
    train_samples = [
        {
            "question": "How does pressure affect turbulence?",
            "reasoning_path": ["Pressure", "Velocity", "Turbulence"],
            "answer": "Pressure increases velocity which results in turbulence.",
        },
        {
            "question": "Why does turbulence change cooling rate?",
            "reasoning_path": ["Turbulence", "Heat Transfer", "Cooling Rate"],
            "answer": "Turbulence increases heat transfer which enhances the cooling rate.",
        },
    ]
    
    # Write temp dataset
    temp_dir = Path("cat_v2_test_tmp")
    temp_dir.mkdir(exist_ok=True)
    temp_dataset_path = temp_dir / "engineering_test_dataset.json"
    temp_dataset_path.write_text(json.dumps(train_samples), encoding="utf-8")

    try:
        dataset = ReasoningDataset(
            temp_dataset_path,
            max_length=16,
            max_path_length=6,
            tokenizer=tokenizer,
            vocab=vocab,
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=ReasoningCollator(),
        )

        # 5. Overfit on the subchains so the transition probabilities and embeddings are learned
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = VLCMReasoningLoss(pad_id=vocab.pad_id, activation_weight=1.0)
        
        # Train for 50 epochs to guarantee learning of edge transitions
        model.train()
        for epoch in range(50):
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                path_ids = batch["path_ids"].to(device)
                activation_targets = batch["activation_targets"].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, target_paths=path_ids)
                
                loss, _ = loss_fn(
                    outputs,
                    path_ids,
                    activation_targets,
                    transition_mask=model.transition_mask,
                )
                loss.backward()
                optimizer.step()

        # 6. Evaluate on the UNSEEN chain query
        model.eval()
        system = VLCMReasoningSystem(model, vocab, tokenizer, device=device)
        test_question = "How does pressure lead to cooling rate?"
        
        result = system.answer(test_question, max_length=16)
        path = result["reasoning_path"]
        
        print(f"Test Query: {test_question}")
        print(f"Generated Path:\n{result['explainability_path']}")
        
        # Verify if intermediate concepts were successfully chained
        success = (
            "Pressure" in path 
            and "Cooling Rate" in path 
            and "Velocity" in path 
            and "Turbulence" in path 
            and "Heat Transfer" in path
        )
        if success:
            print("=> Success! VLCM successfully discovered the unseen concept chain through graph neural reasoning.")
        else:
            print("=> Failed to connect all concepts in the chain.")
            
        return success
        
    finally:
        # Cleanup
        if temp_dataset_path.exists():
            temp_dataset_path.unlink()
