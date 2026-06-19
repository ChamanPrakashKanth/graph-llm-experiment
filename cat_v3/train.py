"""Training pipeline for CAT V3."""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cat_v3.dataset import (
    DOMAINS,
    ConceptVocabulary,
    SimpleCharTokenizer,
    CATV3Dataset,
    grow_dataset,
    build_expert_graphs
)
from cat_v3.model import CATV3Model


def cat_v3_loss(
    outputs: Dict[str, torch.Tensor],
    router_target: torch.Tensor,
    path_ids: torch.Tensor,
    response_targets: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes joint multi-task loss: Router BCE + GAT Experts CE + Causal Decoder CE."""
    batch_size = router_target.size(0)
    num_concepts = outputs["fused_scores"].size(1)
    
    # 1. Router Loss (BCE)
    router_logits = outputs["router_logits"]
    loss_router = F.binary_cross_entropy_with_logits(router_logits, router_target)
    
    # 2. Experts Loss (Cross-Entropy on paths for active experts)
    loss_experts = torch.tensor(0.0, device=router_target.device)
    active_expert_count = 0
    
    for idx, domain in enumerate(DOMAINS):
        expert_out = outputs["expert_reports"][domain]
        expert_logits = expert_out["path_logits"]  # [B, path_length, num_concepts]
        path_len = expert_logits.size(1)
        
        # Calculate CE step-by-step
        ce = F.cross_entropy(
            expert_logits.view(-1, num_concepts),
            path_ids[:, :path_len].reshape(-1),
            reduction="none"
        ).view(batch_size, path_len).mean(dim=1)  # [B]
        
        # Mask out samples where this expert is NOT active in ground truth
        active_weight = router_target[:, idx]  # [B]
        if active_weight.sum() > 0:
            loss_experts = loss_experts + (ce * active_weight).sum() / active_weight.sum()
            active_expert_count += 1
            
    if active_expert_count > 0:
        loss_experts = loss_experts / active_expert_count
        
    # 3. Decoder Loss (Cross-Entropy on target words)
    decoder_logits = outputs["decoder_logits"]  # [B, seq_len-1, vocab_size]
    loss_decoder = F.cross_entropy(
        decoder_logits.reshape(-1, decoder_logits.size(-1)),
        response_targets.reshape(-1),
        ignore_index=pad_id
    )
    
    # Total loss
    total_loss = 0.5 * loss_router + 1.0 * loss_experts + 1.0 * loss_decoder
    
    return total_loss, {
        "loss_total": total_loss.item(),
        "loss_router": loss_router.item(),
        "loss_experts": loss_experts.item(),
        "loss_decoder": loss_decoder.item()
    }


def train_cat_v3(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 5e-4,
    checkpoint_dir: str = "checkpoints/cat_v3",
) -> CATV3Model:
    """Trains the full CAT V3 prototype model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Prepare data
    raw_data = grow_dataset()
    
    # Get all texts to fit the decoder tokenizer vocab
    all_texts = [d["question"] for d in raw_data] + [d["response"] for d in raw_data]
    tokenizer = SimpleCharTokenizer(all_texts)
    
    vocab = ConceptVocabulary({d: list(set(grow_dataset()[0]["concept_paths"][0])) for d in DOMAINS})
    # Re-extract all concepts from dataset to build clean vocabulary
    concept_lists = {}
    for d in DOMAINS:
        concept_lists[d] = []
    
    for item in raw_data:
        # Sort concepts by domain
        for domain in item["active_experts"]:
            for path in item["concept_paths"]:
                for concept in path:
                    concept_lists[domain].append(concept)
                    
    # Ensure every domain has some concepts
    for d in DOMAINS:
        concept_lists[d] = list(set(concept_lists[d]))
        if not concept_lists[d]:
            # fallback
            concept_lists[d] = ["entropy", "load"]
            
    vocab = ConceptVocabulary(concept_lists)
    expert_graphs = build_expert_graphs(vocab)
    
    dataset = CATV3Dataset(raw_data, concept_vocab=vocab, token_tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CAT V3 on device: {device}")
    
    # 2. Initialize Model
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 3. Training loop
    model.train()
    for epoch in range(epochs):
        epoch_losses = {"loss_total": 0.0, "loss_router": 0.0, "loss_experts": 0.0, "loss_decoder": 0.0}
        steps = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            router_target = batch["router_target"].to(device)
            path_ids = batch["path_ids"].to(device)
            response_ids = batch["response_ids"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Shift targets for causal language modeling
            decoder_inputs = response_ids[:, :-1]
            decoder_targets = response_ids[:, 1:]
            decoder_attention_mask = response_mask[:, :-1]
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_paths=path_ids,
                target_responses=decoder_inputs,
                target_responses_mask=decoder_attention_mask,
            )
            
            loss, metrics = cat_v3_loss(
                outputs=outputs,
                router_target=router_target,
                path_ids=path_ids,
                response_targets=decoder_targets,
                pad_id=tokenizer.pad_id
            )
            
            loss.backward()
            optimizer.step()
            
            for k, v in metrics.items():
                epoch_losses[k] += v
            steps += 1
            
        # Log epoch summary
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = {k: v / steps for k, v in epoch_losses.items()}
            print(
                f"Epoch {epoch+1:02d}/{epochs:02d} - "
                f"Total: {avg_loss['loss_total']:.4f} | "
                f"Router: {avg_loss['loss_router']:.4f} | "
                f"Experts: {avg_loss['loss_experts']:.4f} | "
                f"Decoder: {avg_loss['loss_decoder']:.4f}"
            )
            
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "cat_v3_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "tokenizer": tokenizer,
        "expert_graphs": expert_graphs,
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    return model
