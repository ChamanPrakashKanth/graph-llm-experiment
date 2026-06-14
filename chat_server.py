# chat_server.py
# MIT Engineering Mathematics Chat UI — powered by CAT V2
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import urllib.parse

# Ensure root folder is in python path
sys.path.append(str(Path(__file__).parent))

# Try to import the model backend (graceful fallback for demo mode)
try:
    import torch
    import torch.nn.functional as F
    from reasoning_trainer import load_checkpoint, latest_checkpoint
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

_LOADED_CHECKPOINTS = {}

def get_mit_math_system():
    """Load the MIT Math checkpoint."""
    if not BACKEND_AVAILABLE:
        return None
    path = latest_checkpoint("checkpoints/cat_v2_mit_math")
    if not path or not path.exists():
        return None
    path_str = str(path)
    if path_str not in _LOADED_CHECKPOINTS:
        print(f"Loading MIT Math checkpoint: {path_str}")
        loaded = load_checkpoint(path, device="cpu")
        _LOADED_CHECKPOINTS[path_str] = loaded
    return _LOADED_CHECKPOINTS[path_str]

@torch.no_grad()
def predict_reasoning(loaded, question):
    """Run full auto-complete prediction for a question."""
    model = loaded["model"]
    vocab = loaded["vocab"]
    tokenizer = loaded["tokenizer"]
    device = loaded["device"]

    model.eval()

    encoded = tokenizer.encode(question, max_length=64)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)

    question_embedding = model.encode_question(input_ids, attention_mask)
    projected_question = model.question_projection(question_embedding)

    activated = model.activator(question_embedding, top_k=model.top_k)
    activation_logits = activated["activation_logits"]
    activation_probs = activated["activation_probs"].squeeze(0).cpu().tolist()

    hidden = torch.tanh(model.path_generator.question_init(projected_question))
    context = torch.tanh(model.path_generator.context_projection(projected_question))
    prev_embedding = model.path_generator.start_embedding.unsqueeze(0)

    prev_ids = torch.full((1,), model.vocab_eos_id, dtype=torch.long, device=device)
    gui_activation_probs = activated["activation_probs"].clone()

    path_ids = []
    path_probs = []

    for step in range(model.path_generator.path_length):
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
        predicted = logits.argmax(dim=-1)
        pred_id = int(predicted.item())
        pred_prob = float(probs[pred_id].item())

        path_ids.append(pred_id)
        path_probs.append(pred_prob)

        if pred_id == model.vocab_eos_id or pred_id == model.vocab_pad_id:
            break

        prev_ids = predicted
        gui_activation_probs = gui_activation_probs.clone()
        gui_activation_probs[0, prev_ids] = 1.0

    reasoning_path = vocab.decode_path(path_ids)

    # Collect top activated concepts
    top_concepts = []
    for idx, prob in enumerate(activation_probs):
        if prob > 0.05:
            top_concepts.append({
                "name": vocab.id_to_concept[idx],
                "activation": prob
            })
    top_concepts.sort(key=lambda x: -x["activation"])

    # Collect graph nodes/edges
    nodes_list = []
    for name in vocab.concept_names():
        c_id = vocab.concept_to_id[name]
        nodes_list.append({
            "name": name,
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
        "reasoning_path": reasoning_path,
        "path_probabilities": path_probs,
        "top_concepts": top_concepts[:12],
        "nodes": nodes_list,
        "edges": edges_list
    }


# ─── Dataset-based knowledge for demo / fallback ─────────────────────────────

def load_mit_math_dataset():
    """Load the MIT math Q&A dataset for fallback answers."""
    ds_path = Path(__file__).parent / "data" / "mit_math_dataset.json"
    if ds_path.exists():
        with open(ds_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

MIT_MATH_QA = load_mit_math_dataset()


def find_best_match(question):
    """Simple keyword matching to find the closest dataset entry."""
    q_lower = question.lower()
    best_score = 0
    best_entry = None
    for entry in MIT_MATH_QA:
        words = set(entry["question"].lower().split())
        q_words = set(q_lower.split())
        overlap = len(words & q_words)
        if overlap > best_score:
            best_score = overlap
            best_entry = entry
    return best_entry


class ChatRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(CHAT_HTML.encode("utf-8"))
        elif parsed.path == "/api/status":
            loaded = get_mit_math_system() if BACKEND_AVAILABLE else None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "model_loaded": loaded is not None,
                "dataset_size": len(MIT_MATH_QA),
                "backend": "cat_v2_mit_math" if loaded else "dataset_fallback"
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                question = data.get("question", "").strip()
                if not question:
                    raise ValueError("Empty question")

                response_data = {}

                # Try model inference first
                loaded = get_mit_math_system() if BACKEND_AVAILABLE else None
                if loaded:
                    result = predict_reasoning(loaded, question)
                    response_data["reasoning_path"] = result["reasoning_path"]
                    response_data["path_probabilities"] = result["path_probabilities"]
                    response_data["top_concepts"] = result["top_concepts"]
                    response_data["nodes"] = result["nodes"]
                    response_data["edges"] = result["edges"]
                    response_data["source"] = "model"

                # Always add dataset match for the answer text
                match = find_best_match(question)
                if match:
                    response_data["answer"] = match["answer"]
                    if "reasoning_path" not in response_data:
                        response_data["reasoning_path"] = match["reasoning_path"]
                        response_data["source"] = "dataset"
                    response_data["matched_question"] = match["question"]
                else:
                    if "reasoning_path" not in response_data:
                        response_data["reasoning_path"] = []
                        response_data["source"] = "none"
                    response_data["answer"] = "I don't have a specific answer for this question in my knowledge base. Try rephrasing or selecting a suggested question."

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

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


# ─── HTML / CSS / JS ─────────────────────────────────────────────────────────

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIT Engineering Mathematics — CAT V2 Chat</title>
    <meta name="description" content="Interactive chat interface for MIT OCW Engineering Mathematics powered by CAT V2 Concept Attention Transformer reasoning.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        /* ═══════════════════════════════════════════════════════════════
           Design System Tokens
           ═══════════════════════════════════════════════════════════════ */
        :root {
            --bg-deep: #07090e;
            --bg-base: #0c1018;
            --bg-surface: #111827;
            --bg-elevated: #1a2236;
            --bg-hover: #1e293b;
            --bg-glass: rgba(17, 24, 39, 0.72);

            --border-subtle: rgba(55, 65, 81, 0.45);
            --border-medium: rgba(75, 85, 99, 0.5);
            --border-accent: rgba(99, 102, 241, 0.35);

            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --text-dim: #475569;

            --accent-indigo: #818cf8;
            --accent-indigo-light: #a5b4fc;
            --accent-indigo-dim: rgba(99, 102, 241, 0.15);
            --accent-blue: #60a5fa;
            --accent-cyan: #22d3ee;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --accent-rose: #fb7185;
            --accent-violet: #c084fc;

            --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
            --gradient-surface: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.04) 100%);
            --gradient-glow: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.12) 0%, transparent 70%);

            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.35);
            --shadow-lg: 0 8px 32px rgba(0,0,0,0.45);
            --shadow-glow: 0 0 40px rgba(99,102,241,0.15);

            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --radius-full: 9999px;

            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 400ms cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ═══════════════════════════════════════════════════════════════
           Reset & Base
           ═══════════════════════════════════════════════════════════════ */
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html { height: 100%; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* Ambient glow behind header */
        body::before {
            content: '';
            position: fixed;
            top: -120px;
            left: 50%;
            transform: translateX(-50%);
            width: 800px;
            height: 400px;
            background: radial-gradient(ellipse, rgba(99,102,241,0.10) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        /* ═══════════════════════════════════════════════════════════════
           Header
           ═══════════════════════════════════════════════════════════════ */
        #app-header {
            position: relative;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.5rem;
            height: 64px;
            background: rgba(12, 16, 24, 0.85);
            backdrop-filter: blur(20px) saturate(1.5);
            border-bottom: 1px solid var(--border-subtle);
            flex-shrink: 0;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 0.875rem;
        }

        .logo-icon {
            width: 38px;
            height: 38px;
            background: var(--gradient-primary);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: 800;
            color: #fff;
            box-shadow: 0 2px 12px rgba(99,102,241,0.35);
            letter-spacing: -0.5px;
        }

        .header-title {
            display: flex;
            flex-direction: column;
        }

        .header-title h1 {
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.3px;
            line-height: 1.2;
            color: var(--text-primary);
        }

        .header-title span {
            font-size: 0.72rem;
            color: var(--text-muted);
            font-weight: 500;
            letter-spacing: 0.3px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.7rem;
            background: rgba(52, 211, 153, 0.1);
            border: 1px solid rgba(52, 211, 153, 0.25);
            border-radius: var(--radius-full);
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--accent-emerald);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            background: var(--accent-emerald);
            border-radius: 50%;
            animation: pulse-dot 2s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* ═══════════════════════════════════════════════════════════════
           Main Layout
           ═══════════════════════════════════════════════════════════════ */
        #app-main {
            flex: 1;
            display: flex;
            overflow: hidden;
            position: relative;
            z-index: 1;
        }

        /* ─── Sidebar ───────────────────────────────────────────────── */
        #sidebar {
            width: 300px;
            min-width: 300px;
            background: var(--bg-base);
            border-right: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .sidebar-header {
            padding: 1.25rem 1.25rem 0.75rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .sidebar-header h2 {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }

        .course-tabs {
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
        }

        .course-tab {
            padding: 0.35rem 0.65rem;
            font-size: 0.7rem;
            font-weight: 600;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-fast);
            font-family: inherit;
        }

        .course-tab:hover {
            background: var(--accent-indigo-dim);
            border-color: var(--border-accent);
            color: var(--accent-indigo-light);
        }

        .course-tab.active {
            background: var(--accent-indigo-dim);
            border-color: var(--accent-indigo);
            color: var(--accent-indigo-light);
            box-shadow: 0 0 12px rgba(99,102,241,0.15);
        }

        .sidebar-questions {
            flex: 1;
            overflow-y: auto;
            padding: 0.75rem;
        }

        .sidebar-questions::-webkit-scrollbar { width: 4px; }
        .sidebar-questions::-webkit-scrollbar-track { background: transparent; }
        .sidebar-questions::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 4px; }

        .question-card {
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.4rem;
            border: 1px solid transparent;
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            line-height: 1.45;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-fast);
            position: relative;
        }

        .question-card::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 3px;
            height: 0;
            background: var(--gradient-primary);
            border-radius: 2px;
            transition: height var(--transition-base);
        }

        .question-card:hover {
            background: var(--bg-elevated);
            border-color: var(--border-subtle);
            color: var(--text-primary);
            padding-left: 1.1rem;
        }

        .question-card:hover::before {
            height: 60%;
        }

        .question-tag {
            display: inline-block;
            font-size: 0.6rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            margin-bottom: 0.3rem;
        }

        .tag-calculus { background: rgba(96, 165, 250, 0.12); color: var(--accent-blue); }
        .tag-multivariable { background: rgba(52, 211, 153, 0.12); color: var(--accent-emerald); }
        .tag-odes { background: rgba(251, 191, 36, 0.12); color: var(--accent-amber); }
        .tag-linalg { background: rgba(192, 132, 252, 0.12); color: var(--accent-violet); }
        .tag-cross { background: rgba(251, 113, 133, 0.12); color: var(--accent-rose); }

        /* ─── Chat Area ─────────────────────────────────────────────── */
        #chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
            background: var(--bg-deep);
            position: relative;
        }

        /* Chat background pattern */
        #chat-area::before {
            content: '';
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 25% 25%, rgba(99,102,241,0.03) 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, rgba(139,92,246,0.02) 0%, transparent 50%);
            pointer-events: none;
        }

        #messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            position: relative;
            z-index: 1;
        }

        #messages-container::-webkit-scrollbar { width: 6px; }
        #messages-container::-webkit-scrollbar-track { background: transparent; }
        #messages-container::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 4px; }
        #messages-container::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.35); }

        /* ─── Message Bubbles ───────────────────────────────────────── */
        .message {
            display: flex;
            gap: 0.75rem;
            max-width: 85%;
            animation: msg-slide-up 0.35s ease-out;
        }

        @keyframes msg-slide-up {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.assistant { align-self: flex-start; }

        .message-avatar {
            width: 34px;
            height: 34px;
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 700;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #fff;
        }

        .message.assistant .message-avatar {
            background: var(--gradient-primary);
            color: #fff;
        }

        .message-body {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .message-bubble {
            padding: 0.85rem 1.1rem;
            border-radius: var(--radius-md);
            font-size: 0.88rem;
            line-height: 1.6;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.2));
            border: 1px solid rgba(99,102,241,0.3);
            color: var(--text-primary);
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-bubble {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            border-bottom-left-radius: 4px;
        }

        /* ─── Reasoning Path Visualization ──────────────────────────── */
        .reasoning-path-container {
            background: rgba(17, 24, 39, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1rem;
            margin-top: 0.25rem;
        }

        .reasoning-path-label {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .reasoning-path-label svg {
            width: 14px;
            height: 14px;
            opacity: 0.7;
        }

        .reasoning-path-nodes {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.35rem;
        }

        .path-concept {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.35rem 0.7rem;
            border-radius: var(--radius-full);
            font-size: 0.75rem;
            font-weight: 600;
            animation: concept-pop 0.3s ease-out backwards;
            white-space: nowrap;
        }

        @keyframes concept-pop {
            from { opacity: 0; transform: scale(0.85); }
            to { opacity: 1; transform: scale(1); }
        }

        .path-concept:nth-child(1) { animation-delay: 0.1s; }
        .path-concept:nth-child(3) { animation-delay: 0.2s; }
        .path-concept:nth-child(5) { animation-delay: 0.3s; }
        .path-concept:nth-child(7) { animation-delay: 0.4s; }
        .path-concept:nth-child(9) { animation-delay: 0.5s; }
        .path-concept:nth-child(11) { animation-delay: 0.6s; }
        .path-concept:nth-child(13) { animation-delay: 0.7s; }
        .path-concept:nth-child(15) { animation-delay: 0.8s; }

        .path-concept.step-0 { background: rgba(96,165,250,0.15); color: var(--accent-blue); border: 1px solid rgba(96,165,250,0.3); }
        .path-concept.step-1 { background: rgba(99,102,241,0.15); color: var(--accent-indigo-light); border: 1px solid rgba(99,102,241,0.3); }
        .path-concept.step-2 { background: rgba(139,92,246,0.15); color: var(--accent-violet); border: 1px solid rgba(139,92,246,0.3); }
        .path-concept.step-3 { background: rgba(52,211,153,0.15); color: var(--accent-emerald); border: 1px solid rgba(52,211,153,0.3); }
        .path-concept.step-4 { background: rgba(251,191,36,0.15); color: var(--accent-amber); border: 1px solid rgba(251,191,36,0.3); }
        .path-concept.step-5 { background: rgba(251,113,133,0.15); color: var(--accent-rose); border: 1px solid rgba(251,113,133,0.3); }
        .path-concept.step-6 { background: rgba(34,211,238,0.15); color: var(--accent-cyan); border: 1px solid rgba(34,211,238,0.3); }

        .path-arrow {
            color: var(--text-dim);
            font-size: 0.8rem;
            flex-shrink: 0;
        }

        /* ─── Concept Activations ───────────────────────────────────── */
        .concepts-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            margin-top: 0.25rem;
        }

        .concept-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.55rem;
            border-radius: var(--radius-full);
            font-size: 0.68rem;
            font-weight: 500;
            background: rgba(99,102,241,0.08);
            border: 1px solid rgba(99,102,241,0.15);
            color: var(--text-secondary);
        }

        .concept-chip .chip-bar {
            width: 28px;
            height: 3px;
            background: rgba(99,102,241,0.2);
            border-radius: 2px;
            overflow: hidden;
        }

        .concept-chip .chip-bar-fill {
            height: 100%;
            background: var(--accent-indigo);
            border-radius: 2px;
            transition: width var(--transition-base);
        }

        /* ─── Welcome Screen ────────────────────────────────────────── */
        #welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            text-align: center;
            position: relative;
            z-index: 1;
        }

        .welcome-icon {
            width: 80px;
            height: 80px;
            background: var(--gradient-primary);
            border-radius: var(--radius-lg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-glow);
            animation: float 4s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }

        .welcome-title {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-indigo-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .welcome-sub {
            font-size: 0.9rem;
            color: var(--text-secondary);
            max-width: 500px;
            line-height: 1.6;
            margin-bottom: 2rem;
        }

        .welcome-courses {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            max-width: 520px;
            width: 100%;
        }

        .welcome-course-card {
            padding: 1rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            text-align: left;
            cursor: pointer;
            transition: all var(--transition-base);
        }

        .welcome-course-card:hover {
            border-color: var(--border-accent);
            background: var(--bg-elevated);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .welcome-course-card .course-number {
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 0.3rem;
        }

        .welcome-course-card .course-name {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-primary);
            line-height: 1.3;
        }

        /* ─── Input Area ────────────────────────────────────────────── */
        #input-area {
            position: relative;
            z-index: 2;
            padding: 1rem 2rem 1.25rem;
            background: linear-gradient(180deg, transparent 0%, rgba(12,16,24,0.95) 20%);
        }

        .input-wrapper {
            display: flex;
            align-items: flex-end;
            gap: 0.6rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-medium);
            border-radius: var(--radius-lg);
            padding: 0.5rem 0.6rem 0.5rem 1rem;
            transition: all var(--transition-base);
            box-shadow: var(--shadow-md);
        }

        .input-wrapper:focus-within {
            border-color: var(--accent-indigo);
            box-shadow: var(--shadow-md), 0 0 0 3px rgba(99,102,241,0.1);
        }

        #chat-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.92rem;
            line-height: 1.5;
            resize: none;
            max-height: 120px;
            min-height: 24px;
            padding: 0.35rem 0;
        }

        #chat-input::placeholder { color: var(--text-dim); }

        #send-btn {
            width: 38px;
            height: 38px;
            border: none;
            border-radius: var(--radius-sm);
            background: var(--gradient-primary);
            color: #fff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all var(--transition-fast);
            flex-shrink: 0;
        }

        #send-btn:hover { transform: scale(1.05); box-shadow: 0 2px 12px rgba(99,102,241,0.4); }
        #send-btn:active { transform: scale(0.95); }
        #send-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

        #send-btn svg {
            width: 18px;
            height: 18px;
        }

        .input-hint {
            text-align: center;
            font-size: 0.67rem;
            color: var(--text-dim);
            margin-top: 0.5rem;
        }

        /* ─── Typing Indicator ──────────────────────────────────────── */
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 0.85rem 1.1rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            border-bottom-left-radius: 4px;
            width: fit-content;
        }

        .typing-dot {
            width: 7px;
            height: 7px;
            background: var(--accent-indigo);
            border-radius: 50%;
            animation: typing-bounce 1.4s ease-in-out infinite;
        }

        .typing-dot:nth-child(2) { animation-delay: 0.15s; }
        .typing-dot:nth-child(3) { animation-delay: 0.3s; }

        @keyframes typing-bounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-6px); opacity: 1; }
        }

        /* ─── Source badge ──────────────────────────────────────────── */
        .source-badge {
            font-size: 0.6rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            width: fit-content;
        }

        .source-model {
            background: rgba(52,211,153,0.1);
            color: var(--accent-emerald);
            border: 1px solid rgba(52,211,153,0.2);
        }

        .source-dataset {
            background: rgba(96,165,250,0.1);
            color: var(--accent-blue);
            border: 1px solid rgba(96,165,250,0.2);
        }

        /* ─── Right Panel ───────────────────────────────────────────── */
        #right-panel {
            width: 320px;
            min-width: 320px;
            background: var(--bg-base);
            border-left: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .panel-section {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-section-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .panel-section-title svg {
            width: 14px;
            height: 14px;
            opacity: 0.6;
        }

        /* Path timeline */
        .path-timeline {
            display: flex;
            flex-direction: column;
            gap: 0;
        }

        .timeline-step {
            display: flex;
            align-items: flex-start;
            gap: 0.7rem;
            position: relative;
            padding-bottom: 0.6rem;
        }

        .timeline-marker {
            width: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            flex-shrink: 0;
        }

        .timeline-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            border: 2px solid var(--accent-indigo);
            background: var(--bg-deep);
            z-index: 1;
        }

        .timeline-line {
            width: 2px;
            flex: 1;
            min-height: 14px;
            background: rgba(99,102,241,0.2);
        }

        .timeline-step:last-child .timeline-line { display: none; }

        .timeline-content {
            flex: 1;
            padding-bottom: 0.2rem;
        }

        .timeline-concept {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .timeline-prob {
            font-size: 0.68rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Activation bars */
        .activation-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .activation-name {
            font-size: 0.72rem;
            color: var(--text-secondary);
            min-width: 110px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .activation-bar {
            flex: 1;
            height: 6px;
            background: rgba(99,102,241,0.1);
            border-radius: 3px;
            overflow: hidden;
        }

        .activation-bar-fill {
            height: 100%;
            border-radius: 3px;
            background: var(--gradient-primary);
            transition: width var(--transition-slow);
        }

        .activation-value {
            font-size: 0.65rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            min-width: 36px;
            text-align: right;
        }

        .panel-scrollable {
            flex: 1;
            overflow-y: auto;
            padding: 0 1.25rem 1rem;
        }

        .panel-scrollable::-webkit-scrollbar { width: 4px; }
        .panel-scrollable::-webkit-scrollbar-track { background: transparent; }
        .panel-scrollable::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 4px; }

        .panel-empty {
            text-align: center;
            padding: 2rem 1rem;
            color: var(--text-dim);
            font-size: 0.8rem;
            line-height: 1.6;
        }

        .panel-empty svg {
            width: 40px;
            height: 40px;
            margin-bottom: 0.75rem;
            opacity: 0.3;
        }

        /* ─── Responsive ────────────────────────────────────────────── */
        @media (max-width: 1100px) {
            #right-panel { display: none; }
        }
        @media (max-width: 800px) {
            #sidebar { display: none; }
            .message { max-width: 95%; }
        }
    </style>
</head>
<body>

<!-- ═══ HEADER ═══ -->
<header id="app-header">
    <div class="header-left">
        <div class="logo-icon">∫</div>
        <div class="header-title">
            <h1>MIT Engineering Mathematics</h1>
            <span>CAT V2 Concept Reasoning · 18.01 · 18.02 · 18.03 · 18.06</span>
        </div>
    </div>
    <div class="header-right">
        <div class="status-badge" id="status-badge">
            <span class="status-dot"></span>
            <span id="status-text">Initializing...</span>
        </div>
    </div>
</header>

<!-- ═══ MAIN ═══ -->
<main id="app-main">

    <!-- ─── Sidebar ─── -->
    <aside id="sidebar">
        <div class="sidebar-header">
            <h2>Explore Topics</h2>
            <div class="course-tabs">
                <button class="course-tab active" data-course="all" id="tab-all">All</button>
                <button class="course-tab" data-course="18.01" id="tab-1801">18.01</button>
                <button class="course-tab" data-course="18.02" id="tab-1802">18.02</button>
                <button class="course-tab" data-course="18.03" id="tab-1803">18.03</button>
                <button class="course-tab" data-course="18.06" id="tab-1806">18.06</button>
                <button class="course-tab" data-course="cross" id="tab-cross">Cross</button>
            </div>
        </div>
        <div class="sidebar-questions" id="questions-list">
            <!-- Populated by JS -->
        </div>
    </aside>

    <!-- ─── Chat ─── -->
    <section id="chat-area">
        <div id="welcome-screen">
            <div class="welcome-icon">∇</div>
            <div class="welcome-title">Ask anything about engineering mathematics</div>
            <div class="welcome-sub">
                This AI uses a Concept Attention Transformer to reason through mathematical concepts
                step by step — producing verifiable reasoning paths with 0% logic hallucinations.
            </div>
            <div class="welcome-courses">
                <div class="welcome-course-card" onclick="filterCourse('18.01')">
                    <div class="course-number" style="color: var(--accent-blue);">18.01</div>
                    <div class="course-name">Single Variable Calculus</div>
                </div>
                <div class="welcome-course-card" onclick="filterCourse('18.02')">
                    <div class="course-number" style="color: var(--accent-emerald);">18.02</div>
                    <div class="course-name">Multivariable Calculus</div>
                </div>
                <div class="welcome-course-card" onclick="filterCourse('18.03')">
                    <div class="course-number" style="color: var(--accent-amber);">18.03</div>
                    <div class="course-name">Differential Equations</div>
                </div>
                <div class="welcome-course-card" onclick="filterCourse('18.06')">
                    <div class="course-number" style="color: var(--accent-violet);">18.06</div>
                    <div class="course-name">Linear Algebra</div>
                </div>
            </div>
        </div>

        <div id="messages-container" style="display: none;"></div>

        <div id="input-area">
            <div class="input-wrapper">
                <textarea id="chat-input" rows="1" placeholder="Ask about limits, eigenvalues, Laplace transforms, or any MIT math topic..." maxlength="500"></textarea>
                <button id="send-btn" aria-label="Send message">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
            <div class="input-hint">Press Enter to send · Shift+Enter for new line</div>
        </div>
    </section>

    <!-- ─── Right Panel ─── -->
    <aside id="right-panel">
        <div class="panel-section">
            <div class="panel-section-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                Reasoning Path
            </div>
            <div id="panel-path">
                <div class="panel-empty">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                    <div>Ask a question to see the<br>concept reasoning path</div>
                </div>
            </div>
        </div>
        <div class="panel-section" style="border-bottom: none;">
            <div class="panel-section-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4m0 14v4m-9.5-9.5h4m14 0h4m-3.5-7l-2.83 2.83M6.33 17.67l-2.83 2.83m14-14l2.83-2.83M6.33 6.33L3.5 3.5"/></svg>
                Top Concept Activations
            </div>
            <div id="panel-activations">
                <div class="panel-empty">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                    <div>Concept activations will<br>appear here</div>
                </div>
            </div>
        </div>
    </aside>
</main>

<script>
// ═══════════════════════════════════════════════════════════════
//  Data: MIT Math Suggested Questions
// ═══════════════════════════════════════════════════════════════
const QUESTIONS = [
    // ── 18.01 Single Variable Calculus ──
    { course: "18.01", q: "Why does a limit describe the behavior of a function near a point?", tag: "Limits" },
    { course: "18.01", q: "Why does a derivative measure instantaneous rate of change?", tag: "Derivatives" },
    { course: "18.01", q: "How does the power rule simplify differentiation?", tag: "Derivatives" },
    { course: "18.01", q: "Why does the chain rule decompose composite function derivatives?", tag: "Chain Rule" },
    { course: "18.01", q: "How does the Fundamental Theorem of Calculus connect derivatives and integrals?", tag: "FTC" },
    { course: "18.01", q: "How does a Taylor series approximate a function near a point?", tag: "Series" },
    { course: "18.01", q: "How does L'Hopital's rule resolve indeterminate forms?", tag: "Limits" },
    { course: "18.01", q: "How does Newton's method find roots iteratively?", tag: "Roots" },
    { course: "18.01", q: "Why does the definite integral compute area under a curve?", tag: "Integration" },
    { course: "18.01", q: "How does integration by parts handle products of functions?", tag: "Integration" },

    // ── 18.02 Multivariable Calculus ──
    { course: "18.02", q: "Why does the gradient point in the direction of steepest ascent?", tag: "Gradient" },
    { course: "18.02", q: "How does a double integral compute volume under a surface?", tag: "Integration" },
    { course: "18.02", q: "How does the divergence theorem relate volume and surface integrals?", tag: "Divergence" },
    { course: "18.02", q: "How does Stokes' theorem connect line and surface integrals?", tag: "Stokes" },
    { course: "18.02", q: "How does Lagrange multiplier optimization handle constraints?", tag: "Optimization" },
    { course: "18.02", q: "Why does curl measure local rotation of a vector field?", tag: "Curl" },
    { course: "18.02", q: "Why does a saddle point fail both the max and min tests?", tag: "Critical Pts" },
    { course: "18.02", q: "How does the directional derivative measure slope in any direction?", tag: "Gradient" },

    // ── 18.03 Differential Equations ──
    { course: "18.03", q: "Why does separation of variables solve certain first-order ODEs?", tag: "1st Order" },
    { course: "18.03", q: "How does an integrating factor solve first-order linear ODEs?", tag: "1st Order" },
    { course: "18.03", q: "Why does the Laplace transform convert ODEs to algebraic equations?", tag: "Laplace" },
    { course: "18.03", q: "How do eigenvalues determine stability of a linear system?", tag: "Stability" },
    { course: "18.03", q: "Why does resonance cause unbounded growth in forced oscillators?", tag: "Resonance" },
    { course: "18.03", q: "How does a Fourier series decompose periodic functions?", tag: "Fourier" },
    { course: "18.03", q: "How does the matrix exponential solve linear systems of ODEs?", tag: "Systems" },
    { course: "18.03", q: "Why does a phase portrait reveal qualitative ODE behavior?", tag: "Phase Space" },

    // ── 18.06 Linear Algebra ──
    { course: "18.06", q: "How does Gaussian elimination solve systems of linear equations?", tag: "Systems" },
    { course: "18.06", q: "How does an eigenvalue decomposition diagonalize a matrix?", tag: "Eigenvalues" },
    { course: "18.06", q: "Why does the singular value decomposition reveal the rank?", tag: "SVD" },
    { course: "18.06", q: "How does orthogonal projection minimize distance to a subspace?", tag: "Projection" },
    { course: "18.06", q: "How does least squares solve overdetermined systems?", tag: "Least Squares" },
    { course: "18.06", q: "Why does the determinant equal zero for singular matrices?", tag: "Determinants" },
    { course: "18.06", q: "Why does a positive definite matrix have all positive eigenvalues?", tag: "Definiteness" },
    { course: "18.06", q: "How does diagonalization simplify computing matrix powers?", tag: "Eigenvalues" },

    // ── Cross-domain ──
    { course: "cross", q: "How does the matrix exponential from linear algebra solve systems of differential equations?", tag: "LA + ODE" },
    { course: "cross", q: "Why does the gradient from multivariable calculus connect to eigenvalue analysis in optimization?", tag: "MVC + LA" },
    { course: "cross", q: "How does positive definiteness connect quadratic forms in linear algebra to stability in ODEs?", tag: "LA + ODE" },
    { course: "cross", q: "Why does the Fourier series from ODEs connect to orthogonality in linear algebra?", tag: "ODE + LA" },
];

const COURSE_COLORS = {
    "18.01": { tag: "tag-calculus", name: "Calculus" },
    "18.02": { tag: "tag-multivariable", name: "Multivariable" },
    "18.03": { tag: "tag-odes", name: "ODEs" },
    "18.06": { tag: "tag-linalg", name: "Linear Algebra" },
    "cross": { tag: "tag-cross", name: "Cross-Domain" },
};

// ═══════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════
let currentCourse = "all";
let isProcessing = false;
let lastResult = null;

// ═══════════════════════════════════════════════════════════════
//  DOM Refs
// ═══════════════════════════════════════════════════════════════
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const messagesContainer = document.getElementById("messages-container");
const welcomeScreen = document.getElementById("welcome-screen");
const questionsList = document.getElementById("questions-list");
const panelPath = document.getElementById("panel-path");
const panelActivations = document.getElementById("panel-activations");

// ═══════════════════════════════════════════════════════════════
//  Initialize
// ═══════════════════════════════════════════════════════════════
renderSidebarQuestions();
checkStatus();

// ─── Event listeners ────
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});

sendBtn.addEventListener("click", sendMessage);

document.querySelectorAll(".course-tab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".course-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        currentCourse = tab.dataset.course;
        renderSidebarQuestions();
    });
});

// ═══════════════════════════════════════════════════════════════
//  Functions
// ═══════════════════════════════════════════════════════════════

async function checkStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        const statusText = document.getElementById("status-text");
        if (data.model_loaded) {
            statusText.textContent = "Model Online";
        } else {
            statusText.textContent = `Dataset (${data.dataset_size} Q&A)`;
        }
    } catch (e) {
        document.getElementById("status-text").textContent = "Offline";
    }
}

function filterCourse(course) {
    currentCourse = course;
    document.querySelectorAll(".course-tab").forEach(t => {
        t.classList.toggle("active", t.dataset.course === course);
    });
    renderSidebarQuestions();
}

function renderSidebarQuestions() {
    questionsList.innerHTML = "";
    const filtered = currentCourse === "all" ? QUESTIONS : QUESTIONS.filter(q => q.course === currentCourse);

    filtered.forEach(item => {
        const card = document.createElement("div");
        card.className = "question-card";

        const tagClass = COURSE_COLORS[item.course]?.tag || "tag-calculus";
        card.innerHTML = `
            <span class="question-tag ${tagClass}">${item.course} · ${item.tag}</span>
            <div>${item.q}</div>
        `;

        card.addEventListener("click", () => {
            chatInput.value = item.q;
            chatInput.style.height = "auto";
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
            sendMessage();
        });

        questionsList.appendChild(card);
    });
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isProcessing) return;

    // Switch from welcome to chat
    welcomeScreen.style.display = "none";
    messagesContainer.style.display = "flex";

    // Add user message
    addMessage("user", text);
    chatInput.value = "";
    chatInput.style.height = "auto";

    isProcessing = true;
    sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = addTypingIndicator();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: text })
        });

        if (!res.ok) throw new Error("Server error");

        const data = await res.json();
        lastResult = data;

        // Remove typing
        typingEl.remove();

        // Add assistant response
        addAssistantMessage(data);

        // Update right panel
        updateRightPanel(data);

    } catch (err) {
        typingEl.remove();
        addMessage("assistant", "Sorry, an error occurred while processing your question. Please try again.");
    }

    isProcessing = false;
    sendBtn.disabled = false;
    scrollToBottom();
}

function addMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "U" : "∫";

    const body = document.createElement("div");
    body.className = "message-body";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;

    body.appendChild(bubble);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function addAssistantMessage(data) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "∫";

    const body = document.createElement("div");
    body.className = "message-body";

    // Answer bubble
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = data.answer || "Path generated — see reasoning below.";
    body.appendChild(bubble);

    // Reasoning path
    if (data.reasoning_path && data.reasoning_path.length > 0) {
        const pathContainer = document.createElement("div");
        pathContainer.className = "reasoning-path-container";

        const pathLabel = document.createElement("div");
        pathLabel.className = "reasoning-path-label";
        pathLabel.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            Concept Reasoning Path
        `;
        pathContainer.appendChild(pathLabel);

        const nodesDiv = document.createElement("div");
        nodesDiv.className = "reasoning-path-nodes";

        data.reasoning_path.forEach((concept, idx) => {
            if (idx > 0) {
                const arrow = document.createElement("span");
                arrow.className = "path-arrow";
                arrow.textContent = "→";
                nodesDiv.appendChild(arrow);
            }
            const node = document.createElement("span");
            node.className = `path-concept step-${idx % 7}`;
            node.textContent = concept;
            nodesDiv.appendChild(node);
        });

        pathContainer.appendChild(nodesDiv);
        body.appendChild(pathContainer);
    }

    // Source badge
    if (data.source) {
        const badge = document.createElement("div");
        badge.className = `source-badge ${data.source === "model" ? "source-model" : "source-dataset"}`;
        badge.textContent = data.source === "model" ? "⚡ CAT V2 Model" : "📚 Knowledge Base";
        body.appendChild(badge);
    }

    // Top concepts (inline chips)
    if (data.top_concepts && data.top_concepts.length > 0) {
        const grid = document.createElement("div");
        grid.className = "concepts-grid";
        data.top_concepts.slice(0, 8).forEach(c => {
            const chip = document.createElement("div");
            chip.className = "concept-chip";
            chip.innerHTML = `
                <span>${c.name}</span>
                <div class="chip-bar"><div class="chip-bar-fill" style="width: ${(c.activation * 100).toFixed(0)}%"></div></div>
            `;
            grid.appendChild(chip);
        });
        body.appendChild(grid);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

function addTypingIndicator() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    msgDiv.id = "typing-msg";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "∫";

    const body = document.createElement("div");
    body.className = "message-body";

    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    typing.innerHTML = `<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>`;

    body.appendChild(typing);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function updateRightPanel(data) {
    // Path timeline
    if (data.reasoning_path && data.reasoning_path.length > 0) {
        let html = '<div class="path-timeline">';
        data.reasoning_path.forEach((concept, idx) => {
            const prob = data.path_probabilities && data.path_probabilities[idx]
                ? (data.path_probabilities[idx] * 100).toFixed(1) + '%'
                : '';
            html += `
                <div class="timeline-step">
                    <div class="timeline-marker">
                        <div class="timeline-dot"></div>
                        <div class="timeline-line"></div>
                    </div>
                    <div class="timeline-content">
                        <div class="timeline-concept">${concept}</div>
                        ${prob ? `<div class="timeline-prob">confidence: ${prob}</div>` : ''}
                    </div>
                </div>
            `;
        });
        html += '</div>';
        panelPath.innerHTML = html;
    }

    // Activations
    if (data.top_concepts && data.top_concepts.length > 0) {
        let html = '';
        data.top_concepts.slice(0, 10).forEach(c => {
            const pct = (c.activation * 100).toFixed(1);
            html += `
                <div class="activation-item">
                    <div class="activation-name" title="${c.name}">${c.name}</div>
                    <div class="activation-bar">
                        <div class="activation-bar-fill" style="width: ${pct}%"></div>
                    </div>
                    <div class="activation-value">${pct}%</div>
                </div>
            `;
        });
        panelActivations.innerHTML = html;
    }
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}
</script>
</body>
</html>
"""

def run_chat_server(port=8090):
    server_address = ("", port)
    httpd = HTTPServer(server_address, ChatRequestHandler)
    print(f"\n{'='*60}")
    print(f"  MIT Engineering Mathematics Chat UI")
    print(f"  Powered by CAT V2 Concept Attention Transformer")
    print(f"{'='*60}")
    print(f"  -> Open: http://localhost:{port}/")
    print(f"  -> Dataset: {len(MIT_MATH_QA)} Q&A pairs loaded")
    if BACKEND_AVAILABLE:
        loaded = get_mit_math_system()
        if loaded:
            print(f"  -> Model: CAT V2 MIT Math checkpoint ONLINE")
        else:
            print(f"  -> Model: No checkpoint found (using dataset fallback)")
    else:
        print(f"  -> Model: PyTorch not available (using dataset fallback)")
    print(f"{'='*60}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down chat server.")
        httpd.server_close()

if __name__ == "__main__":
    run_chat_server(8090)
