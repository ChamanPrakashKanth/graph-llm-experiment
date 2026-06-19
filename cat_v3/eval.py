"""Evaluation and metrics pipeline for CAT V3."""

from __future__ import annotations

import time
import torch
from typing import Dict, List, Any
import numpy as np

from cat_v3.dataset import DOMAINS, ConceptVocabulary, SimpleCharTokenizer


def compute_routing_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Computes F1-score, Precision, and Recall for multi-label expert classification."""
    preds_bool = preds.bool().cpu().numpy()
    targets_bool = targets.bool().cpu().numpy()
    
    tp = np.logical_and(preds_bool, targets_bool).sum()
    fp = np.logical_and(preds_bool, np.logical_not(targets_bool)).sum()
    fn = np.logical_and(np.logical_not(preds_bool), targets_bool).sum()
    
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)
    exact_match = np.all(preds_bool == targets_bool, axis=-1).mean()
    
    return {
        "router_precision": float(precision),
        "router_recall": float(recall),
        "router_f1": float(f1),
        "router_exact_match": float(exact_match)
    }


def compute_path_metrics(pred_paths: List[List[str]], target_paths: List[List[str]]) -> Dict[str, float]:
    """Computes F1, precision, recall, and exact sequence match for reasoning paths."""
    tp_c, fp_c, fn_c = 0, 0, 0
    exact_matches = 0
    
    for pred, target in zip(pred_paths, target_paths):
        p_set = set(pred)
        t_set = set(target)
        
        tp = len(p_set.intersection(t_set))
        fp = len(p_set - t_set)
        fn = len(t_set - p_set)
        
        tp_c += tp
        fp_c += fp
        fn_c += fn
        
        if pred == target:
            exact_matches += 1
            
    precision = tp_c / max(tp_c + fp_c, 1e-9)
    recall = tp_c / max(tp_c + fn_c, 1e-9)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)
    exact_match = exact_matches / max(len(target_paths), 1)
    
    return {
        "path_precision": float(precision),
        "path_recall": float(recall),
        "path_f1": float(f1),
        "path_exact_match": float(exact_match)
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    vocab: ConceptVocabulary,
    tokenizer: SimpleCharTokenizer,
) -> Dict[str, float]:
    """Evaluates router routing, reasoning path, and decoder text generation metrics."""
    model.eval()
    device = next(model.parameters()).device
    
    all_router_preds = []
    all_router_targets = []
    
    all_pred_paths = []
    all_target_paths = []
    
    exact_response_matches = 0
    total_samples = 0
    
    latencies = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            router_target = batch["router_target"].to(device)
            path_ids = batch["path_ids"].to(device)
            response_ids = batch["response_ids"].to(device)
            
            # Measure latency
            start_time = time.perf_counter()
            outputs = model.generate_response(
                input_ids=input_ids,
                attention_mask=attention_mask,
                router_top_k=2,
                router_threshold=0.5
            )
            latency = (time.perf_counter() - start_time) * 1000.0
            latencies.append(latency / input_ids.size(0))
            
            # Router evaluations
            all_router_preds.append(outputs["router_mask"])
            all_router_targets.append(router_target)
            
            # Decode paths & responses
            for b in range(input_ids.size(0)):
                total_samples += 1
                
                # Active experts target
                active_domains = [DOMAINS[idx] for idx, val in enumerate(router_target[b].tolist()) if val > 0]
                
                # Decoded target path
                t_ids = path_ids[b].cpu().tolist()
                t_path = vocab.decode_path(t_ids)
                all_target_paths.append(t_path)
                
                # Decode predicted path from active experts or fusion layer
                fused_ids = outputs["fused_concept_ids"][b].cpu().tolist()
                f_path = vocab.decode_path(fused_ids)
                all_pred_paths.append(f_path)
                
                # Evaluate decoder exact response match
                target_response = tokenizer.decode(response_ids[b].cpu().tolist())
                generated_response = tokenizer.decode(outputs["generated_tokens"][b].cpu().tolist())
                
                if target_response.strip().lower() == generated_response.strip().lower():
                    exact_response_matches += 1
                    
        router_preds_tensor = torch.cat(all_router_preds, dim=0)
        router_targets_tensor = torch.cat(all_router_targets, dim=0)
        
        router_metrics = compute_routing_metrics(router_preds_tensor, router_targets_tensor)
        path_metrics = compute_path_metrics(all_pred_paths, all_target_paths)
        
        eval_metrics = {
            **router_metrics,
            **path_metrics,
            "decoder_exact_match": exact_response_matches / max(total_samples, 1),
            "inference_latency_ms": float(np.mean(latencies)),
        }
        
    return eval_metrics


def run_single_inference(
    model: torch.nn.Module,
    question: str,
    vocab: ConceptVocabulary,
    tokenizer: SimpleCharTokenizer,
) -> Dict[str, Any]:
    """Runs a single inference query and returns the detailed multi-expert reasoning report."""
    model.eval()
    device = next(model.parameters()).device
    
    input_ids, attention_mask = tokenizer.encode(question, max_length=32)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.generate_response(
            input_ids=input_ids,
            attention_mask=attention_mask,
            router_top_k=2,
            router_threshold=0.5
        )
        
        # 1. Router probabilities
        router_probs = outputs["router_probs"][0].cpu().tolist()
        active_mask = outputs["router_mask"][0].cpu().tolist()
        
        activated_domains = [DOMAINS[idx] for idx, val in enumerate(active_mask) if val > 0]
        domain_probs = {DOMAINS[idx]: router_probs[idx] for idx in range(len(DOMAINS))}
        
        # 2. Expert local paths
        expert_paths = {}
        for idx, domain in enumerate(DOMAINS):
            if active_mask[idx]:
                path_ids = outputs["expert_reports"][domain]["predicted_path"][0].cpu().tolist()
                path = vocab.decode_path(path_ids)
                expert_paths[domain] = path
                
        # 3. Concept fusion report
        fusion_report = model.fusion.get_symbolic_report(
            vocab=vocab,
            expert_reports=outputs["expert_reports"],
            router_mask=outputs["router_mask"],
            domain_names=DOMAINS
        )[0]
        
        # 4. Final Text Generation
        gen_tokens = outputs["generated_tokens"][0].cpu().tolist()
        response = tokenizer.decode(gen_tokens)
        
    return {
        "question": question,
        "activated_domains": activated_domains,
        "domain_probabilities": domain_probs,
        "expert_paths": expert_paths,
        "fusion_report": fusion_report,
        "answer": response
    }
