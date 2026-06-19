"""Benchmarking and scalability profiling suite for CAT V3."""

from __future__ import annotations

import json
import time
import os
import psutil
import torch
import numpy as np
from typing import Dict, List, Any

from cat_v3.dataset import (
    DOMAINS,
    ConceptVocabulary,
    SimpleCharTokenizer,
    CATV3Dataset,
    grow_dataset,
    build_expert_graphs,
    generate_synthetic_scale_graph
)
from cat_v3.model import CATV3Model
from cat_v3.eval import evaluate_model, run_single_inference


def measure_memory_usage() -> float:
    """Measures current system memory usage in Megabytes (MB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_scalability_tests() -> List[Dict[str, Any]]:
    """Runs stress tests at 100, 1,000, and 10,000 concepts to measure latency and memory footprint."""
    scales = [100, 1000, 10000]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    print("\n" + "="*60)
    print("  CAT V3 SCALABILITY BENCHMARKING (100 -> 10,000 CONCEPTS)")
    print("="*60)
    
    for scale in scales:
        print(f"\n>>> Running Scalability Test with {scale} concepts...")
        
        # 1. Generate synthetic graph of target scale
        vocab, expert_graphs = generate_synthetic_scale_graph(scale, density=0.01)
        
        # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        mem_before = measure_memory_usage()
        vram_before = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0.0
        
        # 2. Instantiate Model
        model = CATV3Model(
            num_concepts=vocab.size(),
            tokenizer_vocab_size=100,
            pad_id=vocab.pad_id,
            eos_id=vocab.eos_id,
            expert_graphs=expert_graphs,
            concept_dim=64, # small dim to run on local machines comfortably
            hidden_size=64,
            path_length=8,
            top_m=8,
            decoder_vocab_size=100
        ).to(device)
        
        mem_after = measure_memory_usage()
        vram_after = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0.0
        
        # Weight sizes
        model_memory = mem_after - mem_before
        vram_usage = (vram_after - vram_before) / (1024 * 1024)
        
        # 3. Simulate forward pass
        batch_size = 4
        dummy_input = torch.randint(0, 99, (batch_size, 32), device=device)
        dummy_mask = torch.ones((batch_size, 32), dtype=torch.long, device=device)
        
        # Warmup
        for _ in range(3):
            _ = model(dummy_input, dummy_mask)
            
        latencies = []
        activation_counts = []
        
        # Run profiling iterations
        for _ in range(20):
            start = time.perf_counter()
            outputs = model(dummy_input, dummy_mask)
            latencies.append((time.perf_counter() - start) * 1000.0)
            
            # Count activated experts
            mask = outputs["router_mask"] # [B, num_experts]
            activation_counts.append(mask.float().sum(dim=1).mean().item())
            
        avg_latency = float(np.mean(latencies))
        avg_activations = float(np.mean(activation_counts))
        
        print(f"  Result ({scale} concepts):")
        print(f"    Inference Latency: {avg_latency:.2f} ms")
        print(f"    RAM Footprint increase: {model_memory:.2f} MB")
        if torch.cuda.is_available():
            print(f"    VRAM Usage: {vram_usage:.2f} MB")
        print(f"    Avg Expert Activations: {avg_activations:.1f} experts")
        
        results.append({
            "concept_count": scale,
            "latency_ms": avg_latency,
            "ram_footprint_mb": model_memory,
            "vram_footprint_mb": vram_usage,
            "avg_activations": avg_activations,
            "multi_hop_depth": 8
        })
        
        # Clean up model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    return results


def run_all_benchmarks(trained_model_path: str = "checkpoints/cat_v3/cat_v3_model.pt") -> Dict[str, Any]:
    """Runs all 5 benchmarks: Single Expert, Multi-Expert, Fusion Quality, Multi-Hop, and Scalability."""
    results = {}
    
    # 1. Load trained model
    if not os.path.exists(trained_model_path):
        print(f"No trained model found at {trained_model_path}. Please train a model first using 'run_cat_v3.py train'.")
        # Run scalability tests anyway
        scalability_results = run_scalability_tests()
        return {"scalability": scalability_results}
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(trained_model_path, map_location=device)
    
    vocab = checkpoint["vocab"]
    tokenizer = checkpoint["tokenizer"]
    expert_graphs = checkpoint["expert_graphs"]
    
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
    
    # Benchmark 1: Single Expert Reasoning
    print("\n>>> Running Benchmark 1: Single Expert Reasoning...")
    single_q = "How does syntax affect semantic interpretation of a sentence?"
    single_res = run_single_inference(model, single_q, vocab, tokenizer)
    print(f"  Query: \"{single_q}\"")
    print(f"  Activated Experts: {single_res['activated_domains']}")
    print(f"  Expert Reasoning Path: {single_res['expert_paths']}")
    print(f"  Generated Answer: \"{single_res['answer']}\"")
    results["single_expert"] = single_res
    
    # Benchmark 2: Multi-Expert Reasoning
    print("\n>>> Running Benchmark 2: Multi-Expert Reasoning...")
    multi_q = "Why does compressor pressure ratio affect turbine efficiency?"
    multi_res = run_single_inference(model, multi_q, vocab, tokenizer)
    print(f"  Query: \"{multi_q}\"")
    print(f"  Activated Experts: {multi_res['activated_domains']}")
    print(f"  Expert Reasoning Paths: {list(multi_res['expert_paths'].values())}")
    print(f"  Generated Answer: \"{multi_res['answer']}\"")
    results["multi_expert"] = multi_res
    
    # Benchmark 3: Concept Fusion Quality
    print("\n>>> Running Benchmark 3: Concept Fusion Quality...")
    print(f"  Fused Concepts: {multi_res['fusion_report']['concepts']}")
    print(f"  Fused Paths: {multi_res['fusion_report']['reasoning_paths']}")
    print(f"  Path Confidences: {multi_res['fusion_report']['confidence']}")
    results["fusion_quality"] = multi_res["fusion_report"]
    
    # Benchmark 4: Multi-Hop Concept Prediction
    print("\n>>> Running Benchmark 4: Multi-Hop Concept Prediction...")
    # Check max depth reachable
    total_steps = 8
    overlap_q = "How does a foundation load affect beam buckling?"
    overlap_res = run_single_inference(model, overlap_q, vocab, tokenizer)
    print(f"  Query: \"{overlap_q}\"")
    print(f"  Reasoning Path Depth: {len(overlap_res['fusion_report']['reasoning_paths'][0])} hops")
    print(f"  Reasoning Path Trace: {overlap_res['fusion_report']['reasoning_paths']}")
    results["multi_hop"] = {
        "query": overlap_q,
        "depth": len(overlap_res['fusion_report']['reasoning_paths'][0]),
        "path": overlap_res['fusion_report']['reasoning_paths']
    }
    
    # Benchmark 5: Scalability stress testing
    scalability_results = run_scalability_tests()
    results["scalability"] = scalability_results
    
    # Save benchmark results to reports
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    benchmark_file = os.path.join(reports_dir, "scalability_metrics.json")
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results saved to {benchmark_file}")
    
    return results
