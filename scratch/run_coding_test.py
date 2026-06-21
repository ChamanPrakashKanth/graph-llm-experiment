# scratch/run_coding_test.py
import os
import sys
import torch
import json

# Ensure workspace root is in sys.path
sys.path.append(os.getcwd())

from cat_v3.model import CATV3Model
from agent_executor import OllamaClient, SandboxExecutor, AutonomousCodingAgent
from coding_lab_server import predict_v3_routing

# Configure stdout to use utf-8 to avoid encoding issues
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_test():
    print("=" * 60)
    print("STARTING CAT V3 + QWEN CODER 2.5 3B CODING TEST")
    print("=" * 60)

    # 1. Choose a non-interactive coding task
    question = "Write a Python function fibonacci(n) that returns the first n Fibonacci numbers. In the main block, call this function with n=10, print the result, and do not use any interactive input() calls."
    print(f"QUERY: \"{question}\"")
    
    # 2. Run CAT V3 routing/concept generation
    print("\n[Step 1] Running CAT V3 Graph-MoE Routing...")
    try:
        routing_res = predict_v3_routing(question)
        print(f"Activated Experts: {routing_res.get('activated_domains')}")
        print(f"Concept Reasoning Path: {' -> '.join(routing_res.get('reasoning_path'))}")
    except Exception as e:
        print(f"Error during CAT V3 routing: {e}")
        routing_res = {
            "reasoning_path": ["data_input", "sort", "syntax", "control_flow"],
            "activated_domains": ["coding"]
        }
        print(f"Using fallback concept path: {routing_res['reasoning_path']}")

    # 3. Setup Ollama Client and Sandbox Executor
    print("\n[Step 2] Setting up Ollama and Sandbox...")
    ollama = OllamaClient(model="qwen2.5-coder:3b")
    conn_ok, conn_msg = ollama.check_connection()
    print(f"Ollama Connection Check: {conn_ok} ({conn_msg})")
    if not conn_ok:
        print("Error: Cannot proceed without Ollama connection.")
        return

    sandbox = SandboxExecutor()
    agent = AutonomousCodingAgent(ollama, sandbox)

    # Callback to display progress
    def agent_callback(msg):
        msg_type = msg.get("type")
        iteration = msg.get("iteration", 1)
        if msg_type == "status":
            print(f"[Iter {iteration}] Status: {msg.get('message')}")
        elif msg_type == "thought":
            print(f"[Iter {iteration}] Thought: {msg.get('message')}")
        elif msg_type == "code":
            print(f"[Iter {iteration}] Generated Code Draft.")
        elif msg_type == "execution":
            print(f"[Iter {iteration}] Execution Exit Code: {msg.get('exit_code')}")
            if msg.get('stdout').strip():
                print(f"--- STDOUT ---\n{msg.get('stdout')}")
            if msg.get('stderr').strip():
                print(f"--- STDERR ---\n{msg.get('stderr')}")
        elif msg_type == "error":
            print(f"[Iter {iteration}] Error: {msg.get('message')}")

    # 4. Run the autonomous coding loop
    print("\n[Step 3] Running Autonomous Coding Loop...")
    run_res = agent.run_agent_loop(
        task=question,
        language="python",
        concept_path=routing_res["reasoning_path"],
        max_iterations=4,
        callback=agent_callback
    )

    # 5. Output summary
    print("\n" + "=" * 60)
    print("CODING TASK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Success: {run_res.get('success')}")
    print(f"Iterations: {run_res.get('iterations')}")
    print(f"Exit Code: {run_res.get('exit_code')}")
    print("\nGenerated Code:")
    print("```python")
    print(run_res.get("code"))
    print("```")
    print("\nExecution stdout:")
    print(run_res.get("stdout"))
    if run_res.get("stderr").strip():
        print("Execution stderr:")
        print(run_res.get("stderr"))
        
    # Save results to a json file
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/coding_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "query": question,
            "routing": routing_res,
            "results": {
                "success": run_res.get("success"),
                "iterations": run_res.get("iterations"),
                "exit_code": run_res.get("exit_code"),
                "code": run_res.get("code"),
                "stdout": run_res.get("stdout"),
                "stderr": run_res.get("stderr")
            }
        }, f, indent=4)
    print("\nSaved coding test results to scratch/coding_test_results.json")

if __name__ == "__main__":
    run_test()
