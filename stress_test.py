# stress_test.py
import sys
import time
import json
import torch
import psutil
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add workspace to path
sys.path.append(str(Path(__file__).parent))

from chat_server import get_system, predict_reasoning
import gate_router

def run_stress_test():
    print("="*60)
    print("      GATE Mechanical Engineering Model Stress Test")
    print("="*60)
    
    # 1. Load system
    start_load = time.time()
    loaded = get_system("mechanical_engineering")
    load_time = time.time() - start_load
    if not loaded:
        print("Error: Could not load mechanical_engineering system.")
        return
    print(f"Model and tokenizers loaded successfully in {load_time:.2f}s.")
    
    # 2. Gather questions
    questions = []
    
    # Load mechanical gate questions
    gate_path = Path("data/mechanical_gate_questions.json")
    if gate_path.exists():
        with open(gate_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                questions.append({
                    "id": item.get("id", "GATE-ME-NAT"),
                    "type": "NAT",
                    "question": item["question"],
                    "expected": item["correct_answer"],
                })
                
    # Load mechanical numerical bank
    num_path = Path("data/mechanical_numerical_bank.json")
    if num_path.exists():
        with open(num_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                questions.append({
                    "id": "NUMERICAL",
                    "type": "NAT",
                    "question": item["question"],
                    "expected": item["answer"],
                })
                
    # Load mechanical MCQ bank
    mcq_path = Path("data/mechanical_mcq_bank.json")
    if mcq_path.exists():
        with open(mcq_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                questions.append({
                    "id": "MCQ",
                    "type": "MCQ",
                    "question": item["question"],
                    "expected": item["answer"],
                })
                
    print(f"Total test questions gathered: {len(questions)}")
    print("-"*60)
    
    process = psutil.Process(os.getpid()) if hasattr(os, "getpid") else None
    
    results = []
    correct_count = 0
    total_latency = 0.0
    
    for idx, item in enumerate(questions, 1):
        q_text = item["question"]
        q_type = item["type"]
        expected = item["expected"]
        
        # Measure latency
        start_time = time.time()
        
        # Run model path prediction
        pred = predict_reasoning(loaded, q_text, beam_width=3)
        
        # Run gate orchestration
        gate_res = gate_router.process_gate_query(
            q_text,
            model_path=pred.get("reasoning_path"),
            top_concepts=pred.get("top_concepts"),
        )
        
        latency = (time.time() - start_time) * 1000  # in ms
        total_latency += latency
        
        # Validate output
        is_correct = False
        computed_str = "N/A"
        
        if q_type == "NAT":
            solved = gate_res.get("solved")
            if solved:
                computed_str = str(solved.get("solved_value", ""))
                try:
                    got_val = float(computed_str.replace(",", ""))
                    exp_val = float(str(expected).replace(",", ""))
                    # Allow 2% tolerance
                    rel_err = abs(got_val - exp_val) / max(abs(exp_val), 1e-9)
                    if rel_err <= 0.02:
                        is_correct = True
                except ValueError:
                    if computed_str == str(expected):
                        is_correct = True
            else:
                # Fallback to dataset match
                match = gate_res.get("gate_match")
                if match:
                    computed_str = str(match.get("correct_answer", match.get("answer", "")))
                    if computed_str == str(expected):
                        is_correct = True
        else:
            # MCQ validation
            match = gate_res.get("gate_match")
            if match:
                computed_str = str(match.get("answer", ""))
                if computed_str == str(expected):
                    is_correct = True
                    
        if is_correct:
            correct_count += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        mem_use = process.memory_info().rss / (1024 * 1024) if process else 0.0
        
        print(f"[{idx}] Type: {q_type:<4} | Latency: {latency:6.1f}ms | Status: {status:<4} | Expected: {expected:<10} | Computed: {computed_str:<10}")
        
        results.append({
            "index": idx,
            "id": item["id"],
            "type": q_type,
            "question": q_text,
            "expected": expected,
            "computed": computed_str,
            "latency_ms": latency,
            "status": status,
            "memory_mb": mem_use,
        })
        
    avg_latency = total_latency / len(questions)
    accuracy = (correct_count / len(questions)) * 100
    peak_mem = max(r["memory_mb"] for r in results) if results else 0.0
    
    print("="*60)
    print("                 BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Questions : {len(questions)}")
    print(f"Correct Answers : {correct_count} / {len(questions)}")
    print(f"Accuracy        : {accuracy:.2f}%")
    print(f"Average Latency : {avg_latency:.2f} ms")
    print(f"Peak Memory RSS : {peak_mem:.2f} MB")
    print("="*60)
    
    # Save report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    with open(report_dir / "stress_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "peak_memory_mb": peak_mem,
            "total_questions": len(questions),
            "correct_questions": correct_count,
            "details": results
        }, f, indent=2)
        
    # Generate markdown table for README
    md_table = []
    md_table.append("| Question Index | Question Type | Expected Output | Model/Solver Output | Latency (ms) | Status |")
    md_table.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        q_short = r["question"] if len(r["question"]) <= 60 else r["question"][:57] + "..."
        md_table.append(f"| {r['index']} | {r['type']} | {r['expected']} | {r['computed']} | {r['latency_ms']:.1f} | {r['status']} |")
        
    return {
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "peak_memory": peak_mem,
        "md_table": "\n".join(md_table),
    }

if __name__ == "__main__":
    import os
    run_stress_test()
