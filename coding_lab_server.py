# coding_lab_server.py
# Antigravity Autonomous Coding Lab Server
# Powered by CAT V3 (Graph-MoE) & Ollama qwen2.5-coder:3b

import os
import sys
import json
import math
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

# Ensure workspace root is in python path
sys.path.append(str(Path(__file__).parent))

try:
    import torch
    from cat_v3.model import CATV3Model
    from agent_executor import OllamaClient, SandboxExecutor, AutonomousCodingAgent
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

_LOADED_SYSTEM = None

def load_system():
    global _LOADED_SYSTEM
    if _LOADED_SYSTEM is not None:
        return _LOADED_SYSTEM

    if not BACKEND_AVAILABLE:
        print("Backend dependencies not available.")
        return None

    checkpoint_path = "checkpoints/cat_v3/cat_v3_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Run training first.")
        return None

    try:
        device = "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
        model.eval()

        _LOADED_SYSTEM = {
            "model": model,
            "vocab": vocab,
            "tokenizer": tokenizer,
            "expert_graphs": expert_graphs,
            "device": device
        }
        print("CAT V3 model loaded successfully for Coding Lab.")
        return _LOADED_SYSTEM
    except Exception as e:
        print(f"Error loading CAT V3 model: {e}")
        return None

def predict_v3_routing(question):
    loaded = load_system()
    if not loaded:
        # Fallback simulated routing if no checkpoint
        return {
            "reasoning_path": ["data_input", "sort", "syntax", "control_flow"],
            "activated_domains": ["coding", "mathematics"],
            "domain_probabilities": {
                "mechanical": 0.05, "civil": 0.05, "electrical": 0.02,
                "physics": 0.08, "mathematics": 0.72, "english": 0.12, "coding": 0.95
            },
            "nodes": [{"name": "syntax", "activation": 0.8}, {"name": "sort", "activation": 0.9}],
            "edges": []
        }

    model = loaded["model"]
    vocab = loaded["vocab"]
    tokenizer = loaded["tokenizer"]
    device = loaded["device"]

    DOMAINS_LIST = ["mechanical", "civil", "electrical", "physics", "mathematics", "english", "coding"]

    with torch.no_grad():
        input_ids, attention_mask = tokenizer.encode(question, max_length=32)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

        outputs = model.generate_response(
            input_ids=input_ids,
            attention_mask=attention_mask,
            router_top_k=2,
            router_threshold=0.4
        )

        router_probs = outputs["router_probs"][0].cpu().tolist()
        active_mask = outputs["router_mask"][0].cpu().tolist()

        activated_domains = [DOMAINS_LIST[i] for i, mask in enumerate(active_mask) if mask]
        domain_probs = {DOMAINS_LIST[i]: prob for i, prob in enumerate(router_probs)}

        fusion_report = model.fusion.get_symbolic_report(
            vocab=vocab,
            expert_reports=outputs["expert_reports"],
            router_mask=outputs["router_mask"],
            domain_names=DOMAINS_LIST
        )[0]

        reasoning_path = []
        if fusion_report["reasoning_paths"]:
            reasoning_path = fusion_report["reasoning_paths"][0]
        else:
            # Fallback path if empty
            reasoning_path = ["data_input", "syntax", "control_flow"]

        nodes_list = []
        for name in vocab.concepts:
            if name not in ("<PAD>", "<EOS>"):
                act = 0.95 if name in fusion_report["concepts"] else 0.05
                nodes_list.append({
                    "name": name,
                    "activation": act
                })

        edges_list = []
        for d_idx, d_name in enumerate(DOMAINS_LIST):
            if active_mask[d_idx]:
                expert = model.experts[d_name]
                edge_index = expert.edge_index.cpu()
                for i in range(edge_index.size(1)):
                    u = vocab.id_to_concept[edge_index[0, i].item()]
                    v = vocab.id_to_concept[edge_index[1, i].item()]
                    if u not in ("<PAD>", "<EOS>") and v not in ("<PAD>", "<EOS>"):
                        edges_list.append({
                            "from": u,
                            "to": v,
                            "weight": 1.0
                        })

        return {
            "reasoning_path": reasoning_path,
            "activated_domains": activated_domains,
            "domain_probabilities": domain_probs,
            "nodes": nodes_list,
            "edges": edges_list
        }

class CodingLabRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(LAB_HTML.encode("utf-8"))
        elif parsed.path == "/api/status":
            ollama = OllamaClient()
            conn_ok, conn_msg = ollama.check_connection()
            runtimes = SandboxExecutor.get_supported_runtimes()
            
            status_data = {
                "ollama_connected": conn_ok,
                "ollama_message": conn_msg,
                "runtimes": runtimes,
                "cat_v3_loaded": load_system() is not None
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        elif parsed.path == "/api/agent/run":
            query_params = urllib.parse.parse_qs(parsed.query)
            question = query_params.get("question", [""])[0].strip()
            language = query_params.get("language", ["python"])[0].strip().lower()
            max_iterations = int(query_params.get("max_iterations", ["3"])[0])
            temperature = float(query_params.get("temperature", ["0.2"])[0])

            try:
                if not question:
                    raise ValueError("Question cannot be empty.")

                # Set headers for Server-Sent Events (SSE)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def send_event(event_type, payload):
                    event_data = {"event": event_type, "data": payload}
                    try:
                        self.wfile.write(f"data: {json.dumps(event_data)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        pass # Connection closed

                # Step 1: Run CAT V3 Routing
                send_event("status", "Running query through CAT V3 Graph-MoE router...")
                routing_res = predict_v3_routing(question)
                
                send_event("routing", {
                    "reasoning_path": routing_res["reasoning_path"],
                    "activated_domains": routing_res["activated_domains"],
                    "domain_probabilities": routing_res["domain_probabilities"],
                    "nodes": routing_res["nodes"],
                    "edges": routing_res["edges"]
                })

                # Step 2: Initialize Agent
                ollama = OllamaClient(model="qwen2.5-coder:3b")
                conn_ok, conn_msg = ollama.check_connection()
                if not conn_ok:
                    send_event("error", f"Ollama Error: {conn_msg}")
                    return

                sandbox = SandboxExecutor()
                agent = AutonomousCodingAgent(ollama, sandbox)

                def agent_callback(msg):
                    # Direct forwarding of callback logs to SSE client
                    send_event("agent_log", msg)

                # Step 3: Run the loop
                run_res = agent.run_agent_loop(
                    task=question,
                    language=language,
                    concept_path=routing_res["reasoning_path"],
                    max_iterations=max_iterations,
                    callback=agent_callback
                )

                # Step 4: Finished
                send_event("done", run_res)

            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                except Exception:
                    pass
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/agent/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode("utf-8"))
                question = data.get("question", "").strip()
                language = data.get("language", "python").strip().lower()
                max_iterations = int(data.get("max_iterations", 3))
                temperature = float(data.get("temperature", 0.2))

                if not question:
                    raise ValueError("Question cannot be empty.")

                # Set headers for Server-Sent Events (SSE)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def send_event(event_type, payload):
                    event_data = {"event": event_type, "data": payload}
                    try:
                        self.wfile.write(f"data: {json.dumps(event_data)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        pass # Connection closed

                # Step 1: Run CAT V3 Routing
                send_event("status", "Running query through CAT V3 Graph-MoE router...")
                routing_res = predict_v3_routing(question)
                
                send_event("routing", {
                    "reasoning_path": routing_res["reasoning_path"],
                    "activated_domains": routing_res["activated_domains"],
                    "domain_probabilities": routing_res["domain_probabilities"],
                    "nodes": routing_res["nodes"],
                    "edges": routing_res["edges"]
                })

                # Step 2: Initialize Agent
                ollama = OllamaClient(model="qwen2.5-coder:3b")
                conn_ok, conn_msg = ollama.check_connection()
                if not conn_ok:
                    send_event("error", f"Ollama Error: {conn_msg}")
                    return

                sandbox = SandboxExecutor()
                agent = AutonomousCodingAgent(ollama, sandbox)

                def agent_callback(msg):
                    # Direct forwarding of callback logs to SSE client
                    send_event("agent_log", msg)

                # Step 3: Run the loop
                run_res = agent.run_agent_loop(
                    task=question,
                    language=language,
                    concept_path=routing_res["reasoning_path"],
                    max_iterations=max_iterations,
                    callback=agent_callback
                )

                # Step 4: Finished
                send_event("done", run_res)

            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                except Exception:
                    pass
        else:
            self.send_response(404)
            self.end_headers()

# ─── FRONTEND HTML / CSS / JS ───────────────────────────────────────────────

LAB_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Antigravity Autonomous Coding Lab</title>
    <meta name="description" content="State of the art multi-language autonomous coding agent visual interface.">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        :root {
            --bg-deep: #05070a;
            --bg-base: #0a0d14;
            --bg-surface: #101422;
            --bg-elevated: #161c30;
            --bg-glass: rgba(10, 13, 20, 0.75);

            --border-glow: rgba(99, 102, 241, 0.25);
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-medium: rgba(255, 255, 255, 0.15);

            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #475569;

            --accent-primary: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.15);
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --accent-cyan: #06b6d4;

            --font-main: 'Outfit', sans-serif;
            --font-code: 'Fira Code', monospace;
            --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.5);
            --radius-md: 12px;
            --radius-lg: 16px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-main);
            background: var(--bg-deep);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 20%;
            width: 60%;
            height: 40%;
            background: radial-gradient(ellipse at center, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        header {
            height: 70px;
            background: var(--bg-base);
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: justify;
            padding: 0 2rem;
            position: relative;
            z-index: 10;
            backdrop-filter: blur(10px);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: 800;
            color: #fff;
            box-shadow: 0 0 15px rgba(99,102,241,0.4);
        }

        .title-group h1 {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.3px;
        }

        .title-group p {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .header-status {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.8rem;
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--accent-green);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        #main-container {
            flex: 1;
            display: grid;
            grid-template-columns: 360px 1fr 1fr;
            overflow: hidden;
            position: relative;
            z-index: 1;
        }

        /* Panels layout */
        .panel {
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-subtle);
            background: var(--bg-base);
            overflow: hidden;
        }

        .panel:last-child {
            border-right: none;
        }

        .panel-header {
            padding: 1.2rem;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .panel-header h2 {
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: var(--text-secondary);
        }

        .panel-body {
            flex: 1;
            padding: 1.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        /* Controls Form */
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        textarea {
            width: 100%;
            height: 120px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.8rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
            resize: none;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px rgba(99,102,241,0.1);
        }

        select {
            width: 100%;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.8rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
        }

        .slider-group {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.85rem;
        }

        .slider-group input {
            flex: 1;
            margin-left: 1rem;
            accent-color: var(--accent-primary);
        }

        .btn-run {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            border: none;
            border-radius: var(--radius-md);
            color: #fff;
            font-family: var(--font-main);
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(99,102,241,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-run:hover {
            box-shadow: 0 6px 20px rgba(99,102,241,0.45);
        }

        .btn-run:active {
            transform: scale(0.98);
        }

        .btn-run:disabled {
            background: var(--bg-elevated);
            color: var(--text-secondary);
            cursor: not-allowed;
            box-shadow: none;
        }

        /* Compiler check table */
        .system-check-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1rem;
        }

        .system-check-card h3 {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 0.8rem;
        }

        .compiler-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.78rem;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }

        .compiler-row:last-child {
            border-bottom: none;
        }

        .badge-ok {
            color: var(--accent-green);
            font-weight: 600;
        }

        .badge-warn {
            color: var(--accent-amber);
            font-weight: 600;
        }

        /* Visualizer Panel (Middle) */
        #network-canvas {
            flex: 1;
            background: #07090f;
            position: relative;
        }

        .router-bars {
            height: 150px;
            border-top: 1px solid var(--border-subtle);
            background: var(--bg-surface);
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            overflow-y: auto;
        }

        .router-bars h3 {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
        }

        .bar-row {
            display: flex;
            align-items: center;
            font-size: 0.7rem;
            gap: 0.5rem;
        }

        .bar-label {
            width: 80px;
            text-align: right;
            font-weight: 600;
        }

        .bar-container {
            flex: 1;
            height: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            background: var(--accent-primary);
            width: 0%;
            transition: width 0.4s ease-out;
            border-radius: 4px;
        }

        .bar-val {
            width: 35px;
            text-align: left;
            font-weight: 500;
        }

        /* Code/Logs Panel (Right) */
        .tabs-header {
            display: flex;
            border-bottom: 1px solid var(--border-subtle);
            background: var(--bg-surface);
        }

        .tab-btn {
            flex: 1;
            padding: 1rem 0;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            font-family: var(--font-main);
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: rgba(255,255,255,0.02);
        }

        .tab-btn.active {
            color: var(--accent-primary);
            border-bottom-color: var(--accent-primary);
            background: rgba(99,102,241,0.03);
        }

        .tab-content {
            flex: 1;
            display: none;
            padding: 1.2rem;
            background: #080b12;
            overflow: auto;
            position: relative;
        }

        .tab-content.active {
            display: block;
        }

        /* Logging Terminal */
        .terminal-log {
            font-family: var(--font-code);
            font-size: 0.78rem;
            line-height: 1.5;
            color: #d1d5db;
        }

        .log-entry {
            margin-bottom: 0.8rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid rgba(255,255,255,0.02);
        }

        .log-time {
            color: var(--text-muted);
            margin-right: 0.5rem;
        }

        .log-tag {
            font-weight: 700;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.68rem;
            margin-right: 0.5rem;
        }

        .tag-status { background: rgba(99,102,241,0.15); color: var(--accent-primary); }
        .tag-thought { background: rgba(6,182,212,0.15); color: var(--accent-cyan); }
        .tag-code { background: rgba(245,158,11,0.15); color: var(--accent-amber); }
        .tag-execution { background: rgba(16,185,129,0.15); color: var(--accent-green); }
        .tag-error { background: rgba(244,63,94,0.15); color: var(--accent-rose); }

        pre {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.8rem;
            margin-top: 0.4rem;
            overflow-x: auto;
            font-family: var(--font-code);
            color: #f1f5f9;
        }

        .code-container {
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .code-pre {
            flex: 1;
            background: #020408;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 1.2rem;
            font-family: var(--font-code);
            font-size: 0.85rem;
            color: #e2e8f0;
            white-space: pre-wrap;
            overflow-y: auto;
            position: relative;
        }

        .copy-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.4rem 0.8rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            color: var(--text-primary);
            font-family: var(--font-main);
            font-size: 0.72rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .copy-btn:hover {
            background: var(--accent-primary);
            border-color: var(--accent-primary);
        }

        /* Spinner */
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <header>
        <div class="header-left">
            <div class="logo">Ag</div>
            <div class="title-group">
                <h1>Antigravity Coding Lab</h1>
                <p>CAT V3 (Graph-MoE) + Ollama qwen2.5-coder:3b Autonomous Agent Loop</p>
            </div>
        </div>
        <div class="header-status">
            <div class="status-pill" id="ollama-status">
                <div class="status-dot"></div>
                Ollama: Checking...
            </div>
            <div class="status-pill" id="cat-status" style="background: rgba(99,102,241,0.08); border-color: rgba(99,102,241,0.2); color: var(--accent-primary);">
                <div class="status-dot" style="background: var(--accent-primary);"></div>
                Model: Checking...
            </div>
        </div>
    </header>

    <div id="main-container">
        
        <!-- Controls Panel -->
        <div class="panel">
            <div class="panel-header">
                <h2>Configuration</h2>
            </div>
            <div class="panel-body">
                <div class="form-group">
                    <label for="prompt-input">Coding Task</label>
                    <textarea id="prompt-input" placeholder="e.g. Write a script to calculate fibonacci values recursively."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="lang-select">Programming Language</label>
                    <select id="lang-select">
                        <option value="python">Python</option>
                        <option value="javascript">JavaScript (Node.js)</option>
                        <option value="cpp">C++</option>
                        <option value="go">Go</option>
                        <option value="sql">SQL (SQLite3)</option>
                        <option value="html">HTML/CSS (Validator)</option>
                        <option value="java">Java</option>
                        <option value="rust">Rust</option>
                    </select>
                </div>

                <div class="form-group">
                    <div class="slider-group">
                        <label for="temp-slider">Temperature</label>
                        <span id="temp-val">0.2</span>
                    </div>
                    <input type="range" id="temp-slider" min="0.1" max="1.0" step="0.05" value="0.2" oninput="document.getElementById('temp-val').innerText = this.value">
                </div>

                <div class="form-group">
                    <div class="slider-group">
                        <label for="iter-slider">Max Debug Retries</label>
                        <span id="iter-val">3</span>
                    </div>
                    <input type="range" id="iter-slider" min="1" max="5" step="1" value="3" oninput="document.getElementById('iter-val').innerText = this.value">
                </div>

                <button class="btn-run" id="btn-execute">
                    <span>Run Coding Agent</span>
                </button>

                <div class="system-check-card">
                    <h3>Runtime Diagnostics</h3>
                    <div id="diagnostics-list">
                        <!-- Filled by JS -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Visualizer Panel -->
        <div class="panel">
            <div class="panel-header">
                <h2>Concept Planning Graph (CAT V3)</h2>
            </div>
            <div id="network-canvas"></div>
            
            <div class="router-bars">
                <h3>MoE Routing Probabilities</h3>
                <div id="routing-bars-list">
                    <!-- Bars filled dynamically -->
                </div>
            </div>
        </div>

        <!-- Outputs Panel -->
        <div class="panel">
            <div class="tabs-header">
                <button class="tab-btn active" onclick="switchTab('trace')">Execution Trace</button>
                <button class="tab-btn" onclick="switchTab('code')">Generated Code</button>
                <button class="tab-btn" onclick="switchTab('stdout')">Stdout Output</button>
            </div>
            
            <!-- Trace content -->
            <div class="tab-content active" id="tab-trace">
                <div class="terminal-log" id="log-container">
                    <div class="log-entry"><span class="log-time">00:00:00</span><span class="log-tag tag-status">System</span>Agent ready. Enter task details and click run.</div>
                </div>
            </div>

            <!-- Code Content -->
            <div class="tab-content" id="tab-code">
                <div class="code-container">
                    <div class="code-pre" id="code-output">
                        <button class="copy-btn" onclick="copyGeneratedCode()">Copy</button>
                        <code id="code-content">// Code will be shown here...</code>
                    </div>
                </div>
            </div>

            <!-- Output Content -->
            <div class="tab-content" id="tab-stdout">
                <pre id="stdout-output">Execution stdout/stderr output will be displayed here...</pre>
            </div>
        </div>

    </div>

    <script>
        let network = null;
        let activeEventSource = null;

        // Initialize Diagnostics and Status
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update Ollama status badge
                const ollamaPill = document.getElementById('ollama-status');
                if (data.ollama_connected) {
                    ollamaPill.innerHTML = '<div class="status-dot"></div>Ollama: Online';
                    ollamaPill.style.background = 'rgba(16, 185, 129, 0.08)';
                    ollamaPill.style.color = 'var(--accent-green)';
                } else {
                    ollamaPill.innerHTML = '<div class="status-dot" style="background: var(--accent-rose);"></div>Ollama: Offline';
                    ollamaPill.style.background = 'rgba(244, 63, 94, 0.08)';
                    ollamaPill.style.color = 'var(--accent-rose)';
                }

                // Update CAT V3 badge
                const catPill = document.getElementById('cat-status');
                if (data.cat_v3_loaded) {
                    catPill.innerHTML = '<div class="status-dot"></div>CAT V3: Ready';
                    catPill.style.background = 'rgba(99, 102, 241, 0.08)';
                    catPill.style.color = 'var(--accent-primary)';
                } else {
                    catPill.innerHTML = '<div class="status-dot" style="background: var(--accent-amber);"></div>CAT V3: Simulated';
                    catPill.style.background = 'rgba(245, 158, 11, 0.08)';
                    catPill.style.color = 'var(--accent-amber)';
                }

                // Fill compiler rows
                const diagList = document.getElementById('diagnostics-list');
                diagList.innerHTML = '';
                for (const [lang, ok] of Object.entries(data.runtimes)) {
                    const row = document.createElement('div');
                    row.className = 'compiler-row';
                    row.innerHTML = `
                        <span>${lang.toUpperCase()}</span>
                        <span class="${ok ? 'badge-ok' : 'badge-warn'}">${ok ? 'Executable' : 'Simulated Check'}</span>
                    `;
                    diagList.appendChild(row);
                }

            } catch (e) {
                console.error("Failed to load server status: ", e);
            }
        }

        // Tab Switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            const idx = ['trace', 'code', 'stdout'].indexOf(tabId);
            document.querySelectorAll('.tab-btn')[idx].classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }

        // Log entry helper
        function appendLog(tag, msg, detail = null) {
            const container = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            const timeStr = new Date().toTimeString().split(' ')[0];
            
            let tagClass = 'tag-status';
            if (tag === 'Thought') tagClass = 'tag-thought';
            if (tag === 'Code') tagClass = 'tag-code';
            if (tag === 'Execution') tagClass = 'tag-execution';
            if (tag === 'Error') tagClass = 'tag-error';

            entry.innerHTML = `
                <span class="log-time">${timeStr}</span>
                <span class="log-tag ${tagClass}">${tag}</span>
                <span>${msg}</span>
            `;

            if (detail) {
                const pre = document.createElement('pre');
                pre.innerText = detail;
                entry.appendChild(pre);
            }

            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }

        function copyGeneratedCode() {
            const codeText = document.getElementById('code-content').innerText;
            navigator.clipboard.writeText(codeText);
            const btn = document.querySelector('.copy-btn');
            btn.innerText = 'Copied!';
            setTimeout(() => btn.innerText = 'Copy', 2000);
        }

        // Draw Concept Network Graph
        function drawGraph(nodesData, edgesData, highlightPath = []) {
            const container = document.getElementById('network-canvas');
            
            // Build Network Nodes
            const nodes = new vis.DataSet(
                nodesData.map(n => {
                    const isHighlighted = highlightPath.includes(n.name);
                    return {
                        id: n.name,
                        label: n.name.replace('_', ' '),
                        color: {
                            background: isHighlighted ? '#6366f1' : '#161c30',
                            border: isHighlighted ? '#a5b4fc' : '#334155',
                            highlight: { background: '#6366f1', border: '#a5b4fc' }
                        },
                        font: { color: '#f8fafc', size: 12, face: 'Outfit' },
                        shape: 'box',
                        borderWidth: isHighlighted ? 2.5 : 1,
                        margin: 10
                    };
                })
            );

            // Deduplicate Edges
            const edgeMap = {};
            edgesData.forEach(e => {
                const key = `${e.from}->${e.to}`;
                edgeMap[key] = e;
            });

            // Build Network Edges
            const edges = new vis.DataSet(
                Object.values(edgeMap).map(e => {
                    let isHighlighted = false;
                    for (let i = 0; i < highlightPath.length - 1; i++) {
                        if (highlightPath[i] === e.from && highlightPath[i+1] === e.to) {
                            isHighlighted = true;
                            break;
                        }
                    }

                    return {
                        from: e.from,
                        to: e.to,
                        arrows: 'to',
                        color: {
                            color: isHighlighted ? '#10b981' : '#334155',
                            highlight: isHighlighted ? '#10b981' : '#475569'
                        },
                        width: isHighlighted ? 3 : 1,
                        smooth: { type: 'curvedCW', roundness: 0.15 }
                    };
                })
            );

            const data = { nodes, edges };
            const options = {
                physics: {
                    stabilization: true,
                    barnesHut: { gravitationalConstant: -2000, springLength: 150 }
                }
            };

            network = new vis.Network(container, data, options);
        }

        // Run Routing Bars animation
        function updateRoutingBars(probs) {
            const list = document.getElementById('routing-bars-list');
            list.innerHTML = '';
            
            const sorted = Object.entries(probs).sort((a,b) => b[1] - a[1]);
            for (const [dom, p] of sorted) {
                const pct = Math.round(p * 100);
                const row = document.createElement('div');
                row.className = 'bar-row';
                
                // Highlight active experts
                const isActive = p >= 0.4;
                const fillStyle = isActive ? 'background: linear-gradient(90deg, #6366f1, #8b5cf6)' : 'background: #475569';
                
                row.innerHTML = `
                    <div class="bar-label" style="color: ${isActive ? 'var(--text-primary)' : 'var(--text-secondary)'}">${dom}</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${pct}%; ${fillStyle}"></div>
                    </div>
                    <div class="bar-val" style="color: ${isActive ? 'var(--accent-indigo-light)' : 'var(--text-secondary)'}">${pct}%</div>
                `;
                list.appendChild(row);
            }
        }

        // Execute Agent
        document.getElementById('btn-execute').addEventListener('click', async () => {
            const question = document.getElementById('prompt-input').value.trim();
            const language = document.getElementById('lang-select').value;
            const maxIterations = document.getElementById('iter-slider').value;
            const temperature = document.getElementById('temp-slider').value;

            if (!question) {
                alert("Please enter a coding task first!");
                return;
            }

            const btn = document.getElementById('btn-execute');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> <span>Running Agent...</span>';

            // Clear visual logs
            document.getElementById('log-container').innerHTML = '';
            document.getElementById('code-content').innerText = '// Initializing generation...';
            document.getElementById('stdout-output').innerText = 'Awaiting run output...';
            switchTab('trace');

            appendLog('System', `Initiating autonomous execution loop for language: ${language.toUpperCase()}`);

            if (activeEventSource) {
                activeEventSource.close();
            }

            // Start SSE request
            const eventUrl = `/api/agent/run?` + new URLSearchParams({
                question: question,
                language: language,
                max_iterations: maxIterations,
                temperature: temperature
            }).toString();

            activeEventSource = new EventSource(eventUrl);

            activeEventSource.onmessage = (event) => {
                const res = JSON.parse(event.data);
                const type = res.event;
                const payload = res.data;

                if (type === 'status') {
                    appendLog('System', payload);
                } else if (type === 'thought') {
                    appendLog('Thought', payload);
                } else if (type === 'routing') {
                    // Update VIS Network and MoE Routing Bars
                    updateRoutingBars(payload.domain_probabilities);
                    drawGraph(payload.nodes, payload.edges, payload.reasoning_path);
                    appendLog('System', `CAT V3 Router selected experts: [${payload.activated_domains.join(', ')}]. Path predicted: ${payload.reasoning_path.join(' -> ')}`);
                } else if (type === 'agent_log') {
                    const logType = payload.type;
                    if (logType === 'status') {
                        appendLog('System', `[Iter ${payload.iteration}] ${payload.message}`);
                    } else if (logType === 'thought') {
                        appendLog('Thought', `[Iter ${payload.iteration}] ${payload.message}`);
                    } else if (logType === 'code') {
                        appendLog('Code', `[Iter ${payload.iteration}] Generated code solution.`, payload.explanation);
                        document.getElementById('code-content').innerText = payload.code;
                    } else if (logType === 'execution') {
                        let execDetail = `Exit Code: ${payload.exit_code}\n`;
                        if (payload.stdout.trim()) execDetail += `\nSTDOUT:\n${payload.stdout}\n`;
                        if (payload.stderr.trim()) execDetail += `\nSTDERR:\n${payload.stderr}\n`;
                        
                        appendLog('Execution', `[Iter ${payload.iteration}] Executed sandbox run (Simulated: ${payload.is_simulated})`, execDetail);
                    } else if (logType === 'error') {
                        appendLog('Error', `[Iter ${payload.iteration}] ${payload.message}`);
                    }
                } else if (type === 'done') {
                    appendLog('System', `Autonomous loop finished. Success: ${payload.success} in ${payload.iterations} iterations.`);
                    
                    // Display final output in Stdout tab
                    let stdoutText = `Status: ${payload.success ? 'Success' : 'Failure'}\n`;
                    stdoutText += `Iterations run: ${payload.iterations}\n`;
                    stdoutText += `Exit Code: ${payload.exit_code}\n\n`;
                    stdoutText += `=== STDOUT ===\n${payload.stdout}\n\n`;
                    stdoutText += `=== STDERR ===\n${payload.stderr}\n`;
                    
                    document.getElementById('stdout-output').innerText = stdoutText;
                    document.getElementById('code-content').innerText = payload.code;
                    
                    // Switch tab to code if success, otherwise stdout
                    if (payload.success) {
                        switchTab('code');
                    } else {
                        switchTab('stdout');
                    }

                    btn.disabled = false;
                    btn.innerHTML = '<span>Run Coding Agent</span>';
                    activeEventSource.close();
                } else if (type === 'error') {
                    appendLog('Error', payload);
                    btn.disabled = false;
                    btn.innerHTML = '<span>Run Coding Agent</span>';
                    activeEventSource.close();
                }
            };

            activeEventSource.onerror = (e) => {
                appendLog('Error', "Connection to server EventStream closed unexpectedly.");
                btn.disabled = false;
                btn.innerHTML = '<span>Run Coding Agent</span>';
                if (activeEventSource) activeEventSource.close();
            };
        });

        // Load baseline status on page show
        fetchStatus();
        
        // Poll status every 10 seconds
        setInterval(fetchStatus, 10000);
    </script>
</body>
</html>
"""

def run(port=8002):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, CodingLabRequestHandler)
    print(f"Starting Antigravity Autonomous Coding Lab server on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8002
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port)
