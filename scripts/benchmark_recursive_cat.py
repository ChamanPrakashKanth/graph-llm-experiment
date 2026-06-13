import time
import torch
import sys
from pathlib import Path

# Ensure root workspace folder is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cat_reasoning_model import CATReasoningModel
from reasoning_dataset import ConceptVocabulary, SimpleTokenizer
from reasoning_graph import ReasoningGraph

def run_benchmarks():
    print("=========================================================")
    print("      RECURSIVE CAT V2 EMPIRICAL CPU BENCHMARK          ")
    print("=========================================================")
    
    # 1. Setup vocabulary and graph
    vocab = ConceptVocabulary()
    # Add dummy concepts representing Python coding domain
    concepts = [
        "File Input", "Line Iteration", "Substring Search", "Match Extraction",
        "List Input", "Modulo Condition", "List Comprehension", "Filtered Output",
        "HTTP Request", "API Call", "JSON Parsing", "Dictionary Access"
    ]
    vocab.build(concepts)
    
    graph = ReasoningGraph.from_paths([
        ["File Input", "Line Iteration", "Substring Search", "Match Extraction"],
        ["List Input", "Modulo Condition", "List Comprehension", "Filtered Output"],
        ["HTTP Request", "API Call", "JSON Parsing", "Dictionary Access"]
    ])
    
    tokenizer = SimpleTokenizer()
    tokenizer.build([
        "How to read a file line by line and find a word?",
        "How to filter a list of numbers to find even numbers?",
        "How to fetch a URL and parse JSON in Python?"
    ])
    
    # 2. Build recursive CAT V2 model
    model = CATReasoningModel(
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
    
    # 3. Profile parameters and weights size
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_mb = model_bytes / (1024 * 1024)
    
    # 4. Measure CPU Latency
    input_ids = torch.zeros(1, 16, dtype=torch.long)
    attention_mask = torch.ones(1, 16, dtype=torch.long)
    
    # Warmup
    for _ in range(10):
        _ = model(input_ids, attention_mask)
        
    start = time.perf_counter()
    iters = 100
    for _ in range(iters):
        _ = model(input_ids, attention_mask)
    end = time.perf_counter()
    
    avg_latency = ((end - start) / iters) * 1000
    
    print(f"Vocabulary Size (Concepts): {vocab.size()}")
    print(f"Model Parameters:           {trainable_params:,}")
    print(f"Model Weights Size:         {model_mb:.3f} MB")
    print(f"Average CPU Latency (100 runs): {avg_latency:.2f} ms")
    print("=========================================================")

if __name__ == "__main__":
    run_benchmarks()
