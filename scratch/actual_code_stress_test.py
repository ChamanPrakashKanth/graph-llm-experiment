# scratch/actual_code_stress_test.py
import os
import sys
import time
import json
import psutil
import torch

# Ensure workspace root is in sys.path
sys.path.append(os.getcwd())

from reasoning_graph import ReasoningGraph
from agent_executor import OllamaClient, SandboxExecutor, AutonomousCodingAgent

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def run_actual_code_stress_test():
    print("=" * 70)
    print("STARTING 100K CONCEPTS STRESS TEST ON ACTUAL CODING TASKS")
    print("=" * 70)

    # 1. Profile Loading
    mem_before = get_ram_usage()
    t_start = time.perf_counter()
    
    print(f"Loading 100,000 Concepts Coding Graph...")
    graph_path = "data/coding_concepts_graph.json"
    if not os.path.exists(graph_path):
        print(f"Error: {graph_path} not found. Run scale_coding_kb.py first.")
        return
        
    graph = ReasoningGraph.load_json(graph_path)
    t_load = time.perf_counter() - t_start
    mem_after = get_ram_usage()
    mem_increase = mem_after - mem_before
    
    print(f"Graph Loaded Successfully in {t_load:.4f} seconds!")
    print(f"RAM Memory Footprint Increase: {mem_increase:.2f} MB")
    print(f"Graph Nodes Count: {len(graph.nodes):,}")
    print(f"Graph Edges Count: {sum(len(edges) for edges in graph.edges.values()):,}")

    # Pre-calculate activation scores dictionary to bypass dictionary comprehension overhead in beam search
    print("Pre-calculating graph activation dictionary...")
    acts = {name: float(node.activation) for name, node in graph.nodes.items()}
    
    # 2. Profile Traversal (Beam Search)
    print("\nProfiling Beam Search Traversal (50 iterations)...")
    all_nodes = list(graph.nodes.keys())
    test_starts = [n for n in all_nodes if "in python" in n.lower() or "in javascript" in n.lower()][:5]
    if not test_starts:
        test_starts = all_nodes[:5]
        
    # Warmup
    for _ in range(5):
        _ = graph.beam_search(starts=test_starts, max_depth=5, beam_width=5, activation_scores=acts)
        
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = graph.beam_search(starts=test_starts, max_depth=5, beam_width=5, activation_scores=acts)
        latencies.append((time.perf_counter() - t0) * 1000.0) # ms
        
    avg_search_lat = sum(latencies) / len(latencies)
    print(f"Average 5-hop Beam Search Latency: {avg_search_lat:.2f} ms")

    # 3. Setup Ollama Client & Sandbox
    print("\nSetting up Ollama client and Sandbox executor...")
    ollama = OllamaClient(model="qwen2.5-coder:3b")
    conn_ok, conn_msg = ollama.check_connection()
    print(f"Ollama Connection Status: {conn_ok} ({conn_msg})")
    if not conn_ok:
        print("Error: Ollama connection failed. Cannot proceed with code execution benchmarks.")
        return

    sandbox = SandboxExecutor()
    agent = AutonomousCodingAgent(ollama, sandbox)

    # 4. Define Actual Coding Tasks
    tasks = [
        {
            "id": "task_1_fibonacci",
            "name": "Fibonacci Sequence (Python)",
            "language": "python",
            "prompt": "Write a Python function fibonacci(n) that returns the first n Fibonacci numbers. In the main block, call this function with n=10, print the result, and do not use any interactive input() calls.",
            "concept_path": ["Array Allocation in Python", "Array Execution in Python", "Array Optimization in Python"]
        },
        {
            "id": "task_2_gcd",
            "name": "Greatest Common Divisor (Python)",
            "language": "python",
            "prompt": "Write a Python function calculate_gcd(a, b) that computes the greatest common divisor using the Euclidean algorithm. Test it with inputs a=48 and b=18, print the result, and do not use interactive input() calls.",
            "concept_path": ["Variable Normalization in Python", "Variable Execution in Python", "Variable Optimization in Python"]
        },
        {
            "id": "task_3_matrix_transpose",
            "name": "Matrix Transpose (Python)",
            "language": "python",
            "prompt": "Write a Python function transpose(matrix) that transposes a 2D list (matrix) of size 3x3. Test it with matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], print the transposed matrix, and do not use interactive input() calls.",
            "concept_path": ["Matrix Normalization in Python", "Matrix Execution in Python", "Matrix Optimization in Python"]
        },
        {
            "id": "task_4_rle",
            "name": "Run-length Encoding (JavaScript)",
            "language": "javascript",
            "prompt": "Write a JavaScript function rle(input) that performs run-length encoding on a string. For example, 'AABBBCCCC' should return 'A2B3C4'. Call it with 'AABBBCCCC' in the main block and print the result using console.log. Do not use interactive prompt() calls.",
            "concept_path": ["String Parsing in JavaScript", "String Execution in JavaScript", "String Validation in JavaScript"]
        },
        {
            "id": "task_5_bubble_sort",
            "name": "Bubble Sort (Python)",
            "language": "python",
            "prompt": "Write a Python function bubble_sort(arr) that sorts an array of integers in ascending order. Test it with inputs arr=[64, 34, 25, 12, 22, 11, 90], print the sorted array, and do not use interactive input() calls.",
            "concept_path": ["Array Optimization in Python", "Array Parsing in Python", "Array Execution in Python"]
        }
    ]

    results = []
    print("\nRunning coding tasks through agent loop...")
    for t_idx, t in enumerate(tasks):
        print("-" * 50)
        print(f"Task {t_idx+1}: {t['name']}")
        print(f"Concept Path: {' -> '.join(t['concept_path'])}")
        
        t0 = time.perf_counter()
        agent_res = agent.run_agent_loop(
            task=t["prompt"],
            language=t["language"],
            concept_path=t["concept_path"],
            max_iterations=4,
            callback=None # Run silently but capture full outcome
        )
        total_time = time.perf_counter() - t0
        
        print(f"Outcome: {'SUCCESS' if agent_res['success'] else 'FAILED'}")
        print(f"Iterations needed: {agent_res['iterations']}")
        print(f"Total time taken: {total_time:.2f} seconds")
        if agent_res['success']:
            print(f"Output: {agent_res['stdout'].strip()}")
        else:
            print(f"Error: {agent_res['stderr'].strip()}")

        results.append({
            "task_id": t["id"],
            "name": t["name"],
            "language": t["language"],
            "prompt": t["prompt"],
            "concept_path": t["concept_path"],
            "success": agent_res["success"],
            "iterations": agent_res["iterations"],
            "total_time_seconds": total_time,
            "exit_code": agent_res["exit_code"],
            "code": agent_res["code"],
            "stdout": agent_res["stdout"],
            "stderr": agent_res["stderr"],
            "is_simulated": agent_res["is_simulated"]
        })

    # Save results
    summary_results = {
        "graph_load_time_seconds": t_load,
        "graph_ram_footprint_mb": mem_increase,
        "beam_search_latency_ms": avg_search_lat,
        "total_tasks_run": len(tasks),
        "success_rate": sum(1 for r in results if r["success"]) / len(tasks),
        "avg_task_latency_seconds": sum(r["total_time_seconds"] for r in results) / len(tasks),
        "results": results
    }

    with open("scratch/actual_code_stress_test_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)
        
    print("=" * 70)
    print("STRESS TEST COMPLETED AND SAVED TO scratch/actual_code_stress_test_results.json")
    print("=" * 70)

if __name__ == "__main__":
    run_actual_code_stress_test()
