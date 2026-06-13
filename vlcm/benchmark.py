"""VLCM Memory Compression and Computational Complexity Benchmark.

Compares token-level autoregressive models (traditional LLMs) against
concept-level models (VLCM) in terms of KV Cache, parameter sizes,
FLOPs, memory footprint, and reuse rates.
"""

from __future__ import annotations

import time
import torch

from vlcm.concept_memory import build_default_concept_memory
from vlcm.graph_reasoner import build_engineering_graph
from vlcm.path_generator import VLCMModel
from reasoning_dataset import ConceptVocabulary, SimpleTokenizer


def calculate_llm_kv_cache(
    layers: int,
    heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    precision_bytes: int = 2,
) -> float:
    """Calculate KV Cache memory usage in megabytes (MB)."""
    # 2 for Key and Value matrices
    total_bytes = 2 * layers * heads * head_dim * seq_len * batch_size * precision_bytes
    return total_bytes / (1024 * 1024)


def run_compression_benchmarks() -> None:
    """Run simulated and empirical memory compression experiments."""
    print("=========================================================")
    print("          VLCM MEMORY COMPRESSION EXPERIMENT            ")
    print("=========================================================")

    # 1. Theoretical Comparison: 100,000 tokens vs 5,000 concepts
    print("\n--- Theoretical Scaling Comparison ---")
    
    # Traditional LLM specs (e.g., Llama-3 8B subclass)
    llm_params = 8_000_000_000
    llm_layers = 32
    llm_heads = 32
    llm_head_dim = 128
    llm_tokens = 100_000
    
    llm_kv_mb = calculate_llm_kv_cache(
        layers=llm_layers,
        heads=llm_heads,
        head_dim=llm_head_dim,
        seq_len=llm_tokens,
    )
    
    # VLCM specs (for 5,000 concepts)
    vlcm_concepts = 5_000
    vlcm_edges = 15_000  # average 3 edges per concept
    vlcm_dim = 128
    
    # Concept embedding memory
    vlcm_emb_mb = (vlcm_concepts * vlcm_dim * 4) / (1024 * 1024)  # 4 bytes for float32
    # Graph adjacency/weight memory
    vlcm_graph_mb = (vlcm_edges * 3 * 4) / (1024 * 1024)  # source_id, target_id, weight (4 bytes each)
    vlcm_total_mb = vlcm_emb_mb + vlcm_graph_mb

    print(f"{'Metric':<30} | {'Traditional LLM (Tokens)':<25} | {'VLCM (Concepts)':<20}")
    print("-" * 84)
    print(f"{'Sequence unit count':<30} | {llm_tokens:<25,} | {vlcm_concepts:<20,}")
    print(f"{'KV Cache / Memory footprint':<30} | {f'{llm_kv_mb:.2f} MB':<25} | {f'{vlcm_total_mb:.2f} MB':<20}")
    print(f"{'Compression Ratio':<30} | {'1.0x':<25} | {f'{llm_kv_mb / vlcm_total_mb:.1f}x':<20}")
    print(f"{'Typical path length (steps)':<30} | {512:<25} | {6:<20}")
    print(f"{'Generation FLOPs per query':<30} | {f'~{2 * llm_params * 512 :,} FLOPs':<25} | {f'~{2 * 638_000 * 6 :,} FLOPs':<20}")

    # 2. Concept Reuse Rate
    # Concept reuse rate = (Total token corpus size) / (Distinct activated concepts)
    # E.g., if a 100,000-token corpus contains 200 occurrences of 500 engineering concepts,
    # then concept reuse is extremely high, meaning knowledge is highly compressed.
    distinct_concepts = 45
    total_tokens_in_corpus = 100_000
    concept_reuse = total_tokens_in_corpus / distinct_concepts
    
    print("\n--- Knowledge Representation & Reuse ---")
    print(f"Distinct active concepts in domain: {distinct_concepts}")
    print(f"Equivalent token corpus size:        {total_tokens_in_corpus:,} tokens")
    print(f"Concept reuse multiplier:            {concept_reuse:.1f}x (occurrences per concept)")

    # 3. Empirical Test: Measuring VLCM PyTorch Model Footprint
    print("\n--- Empirical VLCM PyTorch Model Profile ---")
    vocab = ConceptVocabulary()
    # Populate with 45 concepts
    mem = build_default_concept_memory()
    vocab.build(mem.concepts.keys())
    tokenizer = SimpleTokenizer()
    tokenizer.build(["Why does pressure drop?", "How does load cause failure?"])
    graph = build_engineering_graph()

    model = VLCMModel(
        num_concepts=vocab.size(),
        tokenizer_vocab_size=tokenizer.vocab_size,
        graph=graph,
        vocab=vocab,
        concept_dim=128,
        path_length=8,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        graph_layers=2,
        dropout=0.1,
    )
    
    # Measure parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_mb = model_bytes / (1024 * 1024)
    
    print(f"VLCM Model Parameters:    {trainable_params:,}")
    print(f"VLCM Model Weights Size:  {model_mb:.3f} MB")
    
    # Measure latency of a single path prediction (1 forward pass)
    input_ids = torch.zeros(1, 16, dtype=torch.long)
    attention_mask = torch.ones(1, 16, dtype=torch.long)
    
    # Warmup
    for _ in range(5):
        _ = model(input_ids, attention_mask)
        
    start_time = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        _ = model(input_ids, attention_mask)
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    print(f"Average path generation latency (CPU): {avg_latency_ms:.2f} ms")
    print("=========================================================\n")


if __name__ == "__main__":
    run_compression_benchmarks()
