# gui_server.py
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import urllib.parse

import torch
import torch.nn.functional as F

# Ensure root folder is in python path
sys.path.append(str(Path(__file__).parent))

from reasoning_trainer import load_checkpoint, latest_checkpoint

# Load checkpoints on demand to save memory
_LOADED_CHECKPOINTS = {}

def get_checkpoint_system(name):
    if name == "structural":
        path = latest_checkpoint("checkpoints/cat_v2_structural")
    elif name == "python_coding":
        path = latest_checkpoint("checkpoints/cat_v2_python_coding")
    else:
        path = latest_checkpoint("checkpoints/cat_v2")
        
    if not path or not path.exists():
        raise FileNotFoundError(f"Checkpoint for '{name}' not found. Run training first.")
        
    path_str = str(path)
    if path_str not in _LOADED_CHECKPOINTS:
        print(f"Loading checkpoint: {path_str}")
        loaded = load_checkpoint(path, device="cpu")
        _LOADED_CHECKPOINTS[path_str] = loaded
    return _LOADED_CHECKPOINTS[path_str]

@torch.no_grad()
def predict_next_concepts(loaded, question, partial_path):
    model = loaded["model"]
    vocab = loaded["vocab"]
    tokenizer = loaded["tokenizer"]
    device = loaded["device"]
    
    model.eval()
    
    # Tokenize question
    encoded = tokenizer.encode(question, max_length=64)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
    
    # Run question encoder
    question_embedding = model.encode_question(input_ids, attention_mask)
    projected_question = model.question_projection(question_embedding)
    
    # Get activations
    activated = model.activator(question_embedding, top_k=model.top_k)
    activation_logits = activated["activation_logits"]
    activation_probs = activated["activation_probs"].squeeze(0).cpu().tolist()
    
    # Iterative GNN and path step updates
    hidden = torch.tanh(model.path_generator.question_init(projected_question))
    context = torch.tanh(model.path_generator.context_projection(projected_question))
    prev_embedding = model.path_generator.start_embedding.unsqueeze(0)
    
    num_concepts = vocab.size()
    prev_ids = torch.full((1,), model.vocab_eos_id, dtype=torch.long, device=device)
    
    # Track dynamic GNN activations
    gui_activation_probs = activated["activation_probs"].clone()
    
    # 1. Step through partial path recursively
    step = 0
    for concept_name in partial_path:
        concept_memory = model.memory.all_embeddings().unsqueeze(0)
        current_state = model.input_norm(
            concept_memory + projected_question.unsqueeze(1) * gui_activation_probs.unsqueeze(-1)
        )
        graph_state = model.graph_reasoner(current_state, model.propagation_matrix)
        
        if step > 0:
            prev_embedding = graph_state[0, prev_ids]
            
        decoder_input = torch.cat([prev_embedding, context], dim=-1)
        hidden = model.path_generator.gru(decoder_input, hidden)
        
        concept_id = vocab.concept_to_id.get(concept_name)
        if concept_id is None:
            break
        prev_ids = torch.tensor([concept_id], dtype=torch.long, device=device)
        
        # Feedback update
        gui_activation_probs = gui_activation_probs.clone()
        gui_activation_probs[0, prev_ids] = 1.0
        
        step += 1

    # 2. Get predictions for the next concept
    concept_memory = model.memory.all_embeddings().unsqueeze(0)
    current_state = model.input_norm(
        concept_memory + projected_question.unsqueeze(1) * gui_activation_probs.unsqueeze(-1)
    )
    graph_state = model.graph_reasoner(current_state, model.propagation_matrix)
    
    if step > 0:
        prev_embedding = graph_state[0, prev_ids]

    decoder_input = torch.cat([prev_embedding, context], dim=-1)
    hidden = model.path_generator.gru(decoder_input, hidden)
    
    if step == 0:
        allowed = model.first_step_mask.unsqueeze(0)
    else:
        allowed = model.transition_mask[prev_ids]
        
    logits = model.path_generator.output_head(hidden) + 0.25 * activation_logits
    logits = logits.masked_fill(~allowed, -1.0e4)
    
    probs = F.softmax(logits, dim=-1).squeeze(0)
    
    # Fetch all allowed transitions sorted by probability
    allowed_indices = torch.where(allowed[0])[0].tolist()
    predictions = []
    for idx in allowed_indices:
        concept_name = vocab.id_to_concept[idx]
        prob = float(probs[idx].item())
        predictions.append({
            "concept": concept_name,
            "probability": prob,
            "id": idx
        })
    predictions.sort(key=lambda x: -x["probability"])
    
    # 3. Auto-complete remainder of path from here recursively
    autocomplete_ids = []
    curr_prev_ids = prev_ids.clone()
    curr_prev_embedding = prev_embedding.clone()
    curr_hidden = hidden.clone()
    curr_activation_probs = gui_activation_probs.clone()
    
    for auto_step in range(step, model.path_generator.path_length):
        # Dynamic GNN inside autocomplete step
        concept_memory = model.memory.all_embeddings().unsqueeze(0)
        current_state = model.input_norm(
            concept_memory + projected_question.unsqueeze(1) * curr_activation_probs.unsqueeze(-1)
        )
        auto_graph_state = model.graph_reasoner(current_state, model.propagation_matrix)
        
        if auto_step > 0:
            curr_prev_embedding = auto_graph_state[0, curr_prev_ids]
            
        auto_decoder_input = torch.cat([curr_prev_embedding, context], dim=-1)
        curr_hidden = model.path_generator.gru(auto_decoder_input, curr_hidden)
        
        auto_allowed = model.transition_mask[curr_prev_ids]
        auto_logits = model.path_generator.output_head(curr_hidden) + 0.25 * activation_logits
        auto_logits = auto_logits.masked_fill(~auto_allowed, -1.0e4)
        
        predicted = auto_logits.argmax(dim=-1)
        pred_id = int(predicted.item())
        autocomplete_ids.append(pred_id)
        if pred_id == model.vocab_eos_id or pred_id == model.vocab_pad_id:
            break
        curr_prev_ids = predicted
        
        # Feedback update in autocomplete loop
        curr_activation_probs = curr_activation_probs.clone()
        curr_activation_probs[0, curr_prev_ids] = 1.0
        
    autocomplete_path = vocab.decode_path(autocomplete_ids)
    
    # Collect graph info
    nodes_list = []
    for concept_name in vocab.concept_names():
        c_id = vocab.concept_to_id[concept_name]
        nodes_list.append({
            "name": concept_name,
            "activation": float(activation_probs[c_id])
        })
        
    edges_list = []
    for src, targets in loaded["graph"].edges.items():
        for edge in targets.values():
            edges_list.append({
                "from": edge.source,
                "to": edge.target,
                "weight": float(edge.weight)
            })
            
    return {
        "predictions": predictions,
        "autocomplete_path": autocomplete_path,
        "nodes": nodes_list,
        "edges": edges_list
    }

class CATGUIRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging to keep terminal output clean
        pass
        
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/api/checkpoints":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            checkpoints = []
            if latest_checkpoint("checkpoints/cat_v2"):
                checkpoints.append({"id": "cfd", "name": "CFD / Fluid Dynamics (CAT V2)"})
            if latest_checkpoint("checkpoints/cat_v2_structural"):
                checkpoints.append({"id": "structural", "name": "Structural Engineering (CAT V2)"})
            if latest_checkpoint("checkpoints/cat_v2_python_coding"):
                checkpoints.append({"id": "python_coding", "name": "Python Coding AI (CAT V2)"})
            self.wfile.write(json.dumps({"checkpoints": checkpoints}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                checkpoint_name = data.get("checkpoint", "structural")
                question = data.get("question", "")
                partial_path = data.get("partial_path", [])
                
                loaded = get_checkpoint_system(checkpoint_name)
                result = predict_next_concepts(loaded, question, partial_path)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAT V2 Interactive Concept Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: rgba(22, 27, 34, 0.7);
            --border-color: rgba(48, 54, 61, 0.6);
            --primary: #58a6ff;
            --primary-hover: #79c0ff;
            --success: #3fb950;
            --accent-orange: #f2a65a;
            --text-color: #c9d1d9;
            --text-muted: #8b949e;
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            padding: 1.5rem 2rem;
            background: linear-gradient(180deg, rgba(22,27,34,0.8) 0%, rgba(13,17,23,0) 100%);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-badge {
            background: linear-gradient(135deg, var(--primary) 0%, #1f6feb 100%);
            color: #fff;
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.9rem;
            box-shadow: 0 4px 12px rgba(88,166,255,0.3);
        }

        h1 {
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.5px;
        }

        main {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 1.5rem;
            padding: 1.5rem 2rem;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        .control-pane {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--glass-shadow);
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        select, input[type="text"] {
            width: 100%;
            background-color: rgba(22, 27, 34, 0.9);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem;
            color: #fff;
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        select:focus, input[type="text"]:focus {
            border-color: var(--primary);
        }

        .quick-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .btn-quick {
            background-color: rgba(48, 54, 61, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            color: var(--text-color);
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-quick:hover {
            background-color: rgba(88, 166, 255, 0.15);
            border-color: var(--primary);
            color: #fff;
        }

        /* Path Builder UI */
        .path-container {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            background-color: rgba(13, 17, 23, 0.5);
            border: 1px dashed var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            min-height: 4.5rem;
        }

        .path-node {
            background: linear-gradient(135deg, rgba(88,166,255,0.2) 0%, rgba(31,111,235,0.2) 100%);
            border: 1px solid rgba(88, 166, 255, 0.5);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            animation: popIn 0.3s ease-out;
        }

        .path-node.start {
            background: rgba(48, 54, 61, 0.4);
            border-color: var(--border-color);
            color: var(--text-muted);
        }

        .path-arrow {
            color: var(--text-muted);
            font-weight: bold;
        }

        .btn-delete-node {
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .btn-delete-node:hover {
            color: #f85149;
        }

        /* Predictions UI */
        .predictions-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .prediction-item {
            background-color: rgba(22, 27, 34, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
            overflow: hidden;
        }

        .prediction-item:hover {
            border-color: var(--primary);
            transform: translateX(4px);
        }

        .prediction-progress {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            background-color: var(--primary);
            opacity: 0.6;
            transition: width 0.3s ease;
        }

        .prediction-label {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            z-index: 1;
        }

        .prediction-probability {
            font-weight: 600;
            color: var(--primary);
            z-index: 1;
        }

        .badge-eos {
            background-color: rgba(248, 81, 73, 0.15);
            color: #f85149;
            border: 1px solid rgba(248, 81, 73, 0.4);
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-weight: bold;
        }

        .button-bar {
            display: flex;
            gap: 0.75rem;
        }

        .btn {
            flex: 1;
            padding: 0.8rem 1.2rem;
            border-radius: 8px;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }

        .btn-primary {
            background-color: #1f6feb;
            border: 1px solid #388bfd;
            color: #fff;
        }

        .btn-primary:hover {
            background-color: #388bfd;
        }

        .btn-secondary {
            background-color: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-color);
        }

        .btn-secondary:hover {
            background-color: rgba(48, 54, 61, 0.4);
            border-color: var(--text-color);
        }

        /* Visualization Pane */
        .visualization-pane {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 100px);
            min-height: 600px;
        }

        #network-container {
            flex: 1;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            background-color: rgba(13, 17, 23, 0.8);
            position: relative;
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
            text-align: center;
            padding: 2rem;
            gap: 1rem;
        }

        @keyframes popIn {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .alert-error {
            background-color: rgba(248, 81, 73, 0.1);
            border: 1px solid rgba(248, 81, 73, 0.4);
            color: #ff7b72;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.9rem;
            display: none;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <span class="logo-badge">CAT V2</span>
            <h1>Concept Attention Transformer Reasoning Lab</h1>
        </div>
    </header>

    <main>
        <div class="control-pane">
            <div class="card">
                <div class="card-title">Setup Environment</div>
                <div class="alert-error" id="error-alert"></div>
                
                <div class="input-group">
                    <label for="checkpoint-select">Select Model Checkpoint</label>
                    <select id="checkpoint-select">
                        <option value="structural">Structural Engineering (Latest 30 epochs)</option>
                        <option value="cfd">CFD / Fluid Dynamics (Latest 3 epochs)</option>
                        <option value="python_coding">Python Coding AI (CAT V2)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="question-input">Enter Reasoning Question</label>
                    <input type="text" id="question-input" placeholder="Type an engineering question here...">
                </div>

                <div class="input-group">
                    <label>Suggested Questions</label>
                    <div class="quick-questions" id="suggestions-box">
                        <!-- Loaded dynamically -->
                    </div>
                </div>
            </div>

            <div class="card" id="path-builder-card" style="opacity: 0.6; pointer-events: none;">
                <div class="card-title">
                    Step-by-Step Path Builder
                    <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Predict next concept</span>
                </div>
                
                <div class="input-group">
                    <label>Current Reasoning Path</label>
                    <div class="path-container" id="path-box">
                        <span class="path-node start">&lt;START&gt;</span>
                    </div>
                </div>

                <div class="input-group">
                    <label id="predictions-title">Next Concept Probabilities</label>
                    <div class="predictions-list" id="predictions-box">
                        <div style="color: var(--text-muted); font-size: 0.9rem; text-align: center; padding: 1rem;">
                            Set a question above to activate predictions.
                        </div>
                    </div>
                </div>

                <div class="button-bar">
                    <button class="btn btn-secondary" id="btn-reset">Reset Path</button>
                    <button class="btn btn-primary" id="btn-autocomplete">Auto-Complete Path</button>
                </div>
            </div>
        </div>

        <div class="visualization-pane">
            <div class="card" style="height: 100%;">
                <div class="card-title">Interactive Concept Graph Visualization</div>
                <div id="network-container">
                    <div class="empty-state" id="network-empty-state">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <circle cx="12" cy="12" r="9"/>
                            <line x1="12" y1="8" x2="12" y2="16"/>
                            <line x1="8" y1="12" x2="16" y2="12"/>
                        </svg>
                        <p>Graph visualization will render here once a question is loaded.</p>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        const checkpointSelect = document.getElementById("checkpoint-select");
        const questionInput = document.getElementById("question-input");
        const suggestionsBox = document.getElementById("suggestions-box");
        const pathBox = document.getElementById("path-box");
        const predictionsBox = document.getElementById("predictions-box");
        const pathBuilderCard = document.getElementById("path-builder-card");
        const errorAlert = document.getElementById("error-alert");
        
        const btnReset = document.getElementById("btn-reset");
        const btnAutocomplete = document.getElementById("btn-autocomplete");
        
        let currentPath = [];
        let network = null;
        let graphData = { nodes: [], edges: [] };

        const SUGGESTED_QUESTIONS = {
            structural: [
                "How does load cause structural failure?",
                "Why does a column buckle under compression?",
                "Why does cyclic loading cause fatigue?",
                "Why does thermal stress create cracking?"
            ],
            cfd: [
                "Why does pressure drop in a pipe?",
                "Why does turbulence increase at high speed?",
                "Why can poor mesh quality cause convergence failure?",
                "Why does cavitation damage pumps?"
            ],
            python_coding: [
                "How to read a file line by line and find a word?",
                "How to filter a list of numbers to find even numbers?",
                "How to fetch a URL and parse JSON in Python?",
                "How to sort a list of dictionaries by a key?"
            ]
        };

        // Initialize suggestions based on default select value
        updateSuggestions();
        
        checkpointSelect.addEventListener("change", () => {
            updateSuggestions();
            resetPath();
            updatePredictions();
        });

        questionInput.addEventListener("input", () => {
            if (questionInput.value.trim().length > 3) {
                pathBuilderCard.style.opacity = "1";
                pathBuilderCard.style.pointerEvents = "auto";
                updatePredictions();
            } else {
                pathBuilderCard.style.opacity = "0.6";
                pathBuilderCard.style.pointerEvents = "none";
            }
        });

        btnReset.addEventListener("click", resetPath);
        btnAutocomplete.addEventListener("click", autocompletePath);

        function updateSuggestions() {
            suggestionsBox.innerHTML = "";
            const key = checkpointSelect.value;
            SUGGESTED_QUESTIONS[key].forEach(q => {
                const btn = document.createElement("button");
                btn.className = "btn-quick";
                btn.innerText = q;
                btn.onclick = () => {
                    questionInput.value = q;
                    pathBuilderCard.style.opacity = "1";
                    pathBuilderCard.style.pointerEvents = "auto";
                    resetPath();
                };
                suggestionsBox.appendChild(btn);
            });
        }

        function resetPath() {
            currentPath = [];
            renderPath();
            updatePredictions();
        }

        function renderPath() {
            pathBox.innerHTML = '<span class="path-node start">&lt;START&gt;</span>';
            currentPath.forEach((concept, idx) => {
                const arrow = document.createElement("span");
                arrow.className = "path-arrow";
                arrow.innerText = "➔";
                pathBox.appendChild(arrow);

                const node = document.createElement("span");
                node.className = "path-node";
                node.innerHTML = `<span>${concept}</span>`;
                
                // Allow removing last element
                if (idx === currentPath.length - 1) {
                    const delBtn = document.createElement("button");
                    delBtn.className = "btn-delete-node";
                    delBtn.innerHTML = "×";
                    delBtn.onclick = (e) => {
                        e.stopPropagation();
                        currentPath.pop();
                        renderPath();
                        updatePredictions();
                    };
                    node.appendChild(delBtn);
                }
                pathBox.appendChild(node);
            });
        }

        async function updatePredictions() {
            const question = questionInput.value.trim();
            if (!question) {
                predictionsBox.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem; text-align: center; padding: 1rem;">Set a question above to activate predictions.</div>';
                return;
            }

            try {
                errorAlert.style.display = "none";
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        checkpoint: checkpointSelect.value,
                        question: question,
                        partial_path: currentPath
                    })
                });

                if (!response.ok) throw new Error("Backend prediction error");

                const data = await response.json();
                renderPredictions(data.predictions);
                renderGraph(data.nodes, data.edges);
            } catch (err) {
                errorAlert.innerText = err.message;
                errorAlert.style.display = "block";
            }
        }

        function renderPredictions(predictions) {
            predictionsBox.innerHTML = "";
            if (predictions.length === 0) {
                predictionsBox.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem; text-align: center; padding: 1rem;">No valid next concept connections allowed by graph transition mask.</div>';
                return;
            }

            predictions.forEach(pred => {
                const item = document.createElement("div");
                item.className = "prediction-item";
                
                const isEos = pred.concept === "<EOS>";
                
                item.onclick = () => {
                    if (isEos) {
                        alert("Path generation finished via End-Of-Sequence <EOS>.");
                        return;
                    }
                    currentPath.push(pred.concept);
                    renderPath();
                    updatePredictions();
                };

                const progress = document.createElement("div");
                progress.className = "prediction-progress";
                progress.style.width = `${pred.probability * 100}%`;
                item.appendChild(progress);

                const label = document.createElement("div");
                label.className = "prediction-label";
                if (isEos) {
                    label.innerHTML = `<span>End Sequence</span> <span class="badge-eos">&lt;EOS&gt;</span>`;
                } else {
                    label.innerText = pred.concept;
                }
                item.appendChild(label);

                const prob = document.createElement("div");
                prob.className = "prediction-probability";
                prob.innerText = `${(pred.probability * 100).toFixed(1)}%`;
                item.appendChild(prob);

                predictionsBox.appendChild(item);
            });
        }

        async function autocompletePath() {
            const question = questionInput.value.trim();
            if (!question) return;

            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        checkpoint: checkpointSelect.value,
                        question: question,
                        partial_path: currentPath
                    })
                });

                if (!response.ok) throw new Error("Auto-complete error");

                const data = await response.json();
                if (data.autocomplete_path && data.autocomplete_path.length > 0) {
                    currentPath = currentPath.concat(data.autocomplete_path);
                    renderPath();
                    updatePredictions();
                }
            } catch (err) {
                errorAlert.innerText = err.message;
                errorAlert.style.display = "block";
            }
        }

        function renderGraph(nodes, edges) {
            const container = document.getElementById("network-container");
            const emptyState = document.getElementById("network-empty-state");
            if (emptyState) emptyState.remove();

            // Check if vis.js is loaded
            if (typeof vis === "undefined") {
                container.innerHTML = '<div class="empty-state">Unable to load vis.js library. Connect to internet or verify CDN.</div>';
                return;
            }

            const visNodes = [];
            const visEdges = [];

            // Determine if a node is in the current path
            const pathSet = new Set(currentPath);

            nodes.forEach(node => {
                const isPath = pathSet.has(node.name);
                const isStart = currentPath[0] === node.name;
                
                let color = {
                    background: "#1f2937",
                    border: "#4b5563",
                    highlight: { background: "#1f2937", border: "#4b5563" }
                };

                if (isPath) {
                    color = {
                        background: "#1f6feb",
                        border: "#58a6ff",
                        highlight: { background: "#388bfd", border: "#58a6ff" }
                    };
                } else if (node.activation > 0.1) {
                    color = {
                        background: "rgba(242, 166, 90, 0.25)",
                        border: "var(--accent-orange)",
                        highlight: { background: "rgba(242, 166, 90, 0.35)", border: "var(--accent-orange)" }
                    };
                }

                visNodes.push({
                    id: node.name,
                    label: node.name,
                    color: color,
                    font: { color: "#c9d1d9", face: "Outfit" },
                    shape: "box",
                    margin: 10,
                    borderWidth: isPath ? 2 : 1,
                    shadow: true
                });
            });

            // Keep track of highlighted edges in the path
            const pathEdges = new Set();
            for (let i = 0; i < currentPath.length - 1; i++) {
                pathEdges.add(`${currentPath[i]}->${currentPath[i+1]}`);
            }

            edges.forEach(edge => {
                const inPath = pathEdges.has(`${edge.from}->${edge.to}`);
                visEdges.push({
                    from: edge.from,
                    to: edge.to,
                    width: inPath ? 3 : 1,
                    color: {
                        color: inPath ? "#58a6ff" : "rgba(139, 148, 158, 0.3)",
                        highlight: inPath ? "#58a6ff" : "rgba(139, 148, 158, 0.6)",
                        hover: "rgba(139, 148, 158, 0.5)"
                    },
                    arrows: { to: { enabled: true, scaleFactor: 0.8 } },
                    smooth: { type: "curvedCW", roundness: 0.15 }
                });
            });

            const data = {
                nodes: new vis.DataSet(visNodes),
                edges: new vis.DataSet(visEdges)
            };

            const options = {
                physics: {
                    solver: "forceAtlas2Based",
                    forceAtlas2Based: {
                        gravitationalConstant: -26,
                        centralGravity: 0.005,
                        springLength: 100,
                        springConstant: 0.18
                    }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 200
                }
            };

            if (network === null) {
                network = new vis.Network(container, data, options);
            } else {
                network.setData(data);
            }
        }
    </script>
</body>
</html>
"""

def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, CATGUIRequestHandler)
    print(f"CAT V2 Concept Prediction GUI running at http://localhost:{port}/")
    print(f"To test: open browser at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server(8080)
