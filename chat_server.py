# chat_server.py
# Multi-Domain Engineering Chat UI — powered by CAT V2 & VLCM
import json
import os
import sys
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import urllib.parse

# Ensure root folder is in python path
sys.path.append(str(Path(__file__).parent))

# Try to import the model backend (graceful fallback for demo mode)
try:
    import torch
    import torch.nn.functional as F
    from reasoning_trainer import load_checkpoint as load_cat_checkpoint, latest_checkpoint as latest_cat_checkpoint
    from vlcm.trainer import load_vlcm_checkpoint, latest_vlcm_checkpoint
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

_LOADED_SYSTEMS = {}
_LOADED_DATASETS = {}

def load_cat_v3_checkpoint(checkpoint_path, device="cpu"):
    import torch
    from cat_v3.model import CATV3Model
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
    
    return {
        "model": model,
        "vocab": vocab,
        "tokenizer": tokenizer,
        "expert_graphs": expert_graphs,
        "device": device
    }

DOMAINS_V3_LIST = ["mechanical", "civil", "electrical", "physics", "mathematics", "english"]

def predict_reasoning_cat_v3(loaded, question):
    import torch
    import math
    model = loaded["model"]
    vocab = loaded["vocab"]
    tokenizer = loaded["tokenizer"]
    device = loaded["device"]
    
    model.eval()
    
    with torch.no_grad():
        input_ids, attention_mask = tokenizer.encode(question, max_length=32)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        
        outputs = model.generate_response(
            input_ids=input_ids,
            attention_mask=attention_mask,
            router_top_k=2,
            router_threshold=0.5
        )
        
        router_probs = outputs["router_probs"][0].cpu().tolist()
        active_mask = outputs["router_mask"][0].cpu().tolist()
        
        gen_tokens = outputs["generated_tokens"][0].cpu().tolist()
        answer = tokenizer.decode(gen_tokens)
        
        fusion_report = model.fusion.get_symbolic_report(
            vocab=vocab,
            expert_reports=outputs["expert_reports"],
            router_mask=outputs["router_mask"],
            domain_names=DOMAINS_V3_LIST
        )[0]
        
        reasoning_path = []
        if fusion_report["reasoning_paths"]:
            reasoning_path = fusion_report["reasoning_paths"][0]
            
        fused_scores = outputs["fused_concept_ids"][0].cpu().tolist()
        top_concepts = []
        for idx in fused_scores:
            if idx in vocab.id_to_concept:
                concept_name = vocab.id_to_concept[idx]
                if concept_name not in ("<PAD>", "<EOS>"):
                    top_concepts.append({
                        "name": concept_name,
                        "activation": 0.8
                    })
                    
        nodes_list = []
        for name in vocab.concepts:
            if name not in ("<PAD>", "<EOS>"):
                act = 0.9 if name in fusion_report["concepts"] else 0.05
                nodes_list.append({
                    "name": name,
                    "activation": act
                })
                
        edges_list = []
        for domain_idx, domain_name in enumerate(DOMAINS_V3_LIST):
            if active_mask[domain_idx]:
                expert = model.experts[domain_name]
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
                        
        active_experts_str = ", ".join([d.capitalize() for idx, d in enumerate(DOMAINS_V3_LIST) if active_mask[idx]])
        answer = f"**[GAT Router activated experts: {active_experts_str}]**\n\n{answer}"
        
        return {
            "reasoning_path": reasoning_path,
            "path_probabilities": [0.9] * len(reasoning_path),
            "top_concepts": top_concepts,
            "nodes": nodes_list,
            "edges": edges_list,
            "answer": answer
        }


DOMAINS = {
    "cat_v3_moe": {
        "name": "CAT V3 / Graph-MoE (Prototype)",
        "checkpoint_dir": "checkpoints/cat_v3",
        "dataset_path": "data/reasoning_dataset.json",
        "is_cat_v3": True,
        "symbol": "🔮",
        "desc": "Multi-expert routing with GAT graph reasoning & fusion across 6 engineering domains",
    },
    "mechanical_engineering": {
        "name": "Mechanical Engineering (VLCM)",
        "checkpoint_dir": "checkpoints/vlcm_mech_2nd_order",
        "dataset_path": "data/mechanical_engineering_dataset.json",
        "is_vlcm": True,
        "symbol": "⚙️",
        "desc": "Scaled 10k-concept dataset (140k edges, 58k causal) with Second-Order Loss",
    },
    "mit_stanford_mech": {
        "name": "MIT & Stanford ME Curriculum (VLCM)",
        "checkpoint_dir": "checkpoints/vlcm_mit_stanford_curriculum",
        "dataset_path": "data/mechanical_reasoning_paths.json",
        "is_vlcm": True,
        "symbol": "🎓",
        "desc": "Complete syllabus (10k+ concepts, 50k+ reasoning paths)",
    },
    "mit_math": {
        "name": "MIT OCW Mathematics (CAT V2)",
        "checkpoint_dir": "checkpoints/cat_v2_mit_math",
        "dataset_path": "data/mit_math_dataset.json",
        "is_vlcm": False,
        "symbol": "∫",
        "desc": "Calculus, Linear Algebra, ODEs, and Multivariable Calculus",
    },
    "structural": {
        "name": "Structural Engineering (CAT V2)",
        "checkpoint_dir": "checkpoints/cat_v2_structural",
        "dataset_path": "data/structural_reasoning_dataset.json",
        "is_vlcm": False,
        "symbol": "🏗️",
        "desc": "Beams, stress-strain, columns buckling, fatigue, and materials science",
    },
    "cfd": {
        "name": "CFD & Fluid Dynamics (CAT V2)",
        "checkpoint_dir": "checkpoints/cat_v2",
        "dataset_path": "data/reasoning_dataset.json",
        "is_vlcm": False,
        "symbol": "🌪️",
        "desc": "Pipe flow, pressure drop, turbulence, boundary layers, and mesh quality",
    },
    "python_coding": {
        "name": "Python Coding AI (CAT V2)",
        "checkpoint_dir": "checkpoints/cat_v2_python_coding",
        "dataset_path": "data/python_coding_dataset.json",
        "is_vlcm": False,
        "symbol": "🐍",
        "desc": "Autoregressive code generation, lists, sorting, file I/O, and API requests",
    },
}

def get_system(domain):
    if not BACKEND_AVAILABLE:
        return None
    if domain not in DOMAINS:
        return None

    info = DOMAINS[domain]
    checkpoint_dir = info["checkpoint_dir"]
    is_vlcm = info.get("is_vlcm", False)
    is_cat_v3 = info.get("is_cat_v3", False)

    if is_cat_v3:
        path = Path(checkpoint_dir) / "cat_v3_model.pt"
        if not path.exists():
            return None
        path_str = str(path)
        if path_str not in _LOADED_SYSTEMS:
            print(f"Loading checkpoint for {domain}: {path_str}")
            try:
                workspace_root = str(Path(__file__).parent)
                if workspace_root not in sys.path:
                    sys.path.insert(0, workspace_root)
                loaded = load_cat_v3_checkpoint(path_str, device="cpu")
                _LOADED_SYSTEMS[path_str] = loaded
            except Exception as e:
                print(f"Failed to load checkpoint {path_str}: {e}")
                return None
        return _LOADED_SYSTEMS[path_str]

    if is_vlcm:
        path = latest_vlcm_checkpoint(checkpoint_dir)
        loader = load_vlcm_checkpoint
    else:
        path = latest_cat_checkpoint(checkpoint_dir)
        loader = load_cat_checkpoint

    if not path or not path.exists():
        return None

    path_str = str(path)
    if path_str not in _LOADED_SYSTEMS:
        print(f"Loading checkpoint for {domain}: {path_str}")
        try:
            loaded = loader(path, device="cpu")
            import os
            from answer_decoder import T5AnswerDecoder, TemplateAnswerDecoder
            t5_path = "checkpoints/answer_decoder_t5"
            if os.path.exists(t5_path) and any(os.listdir(t5_path)):
                try:
                    loaded["decoder"] = T5AnswerDecoder(model_dir=t5_path, device="cpu")
                    print(f"Loaded and cached T5AnswerDecoder in chat_server for domain: {domain}")
                except Exception as ex:
                    print(f"Failed to load T5 decoder: {ex}. Falling back to TemplateAnswerDecoder.")
                    loaded["decoder"] = TemplateAnswerDecoder()
            else:
                loaded["decoder"] = TemplateAnswerDecoder()
            _LOADED_SYSTEMS[path_str] = loaded
        except Exception as e:
            print(f"Failed to load checkpoint {path_str}: {e}")
            return None
    return _LOADED_SYSTEMS[path_str]

def get_dataset(domain):
    if domain not in DOMAINS:
        return []
    if domain not in _LOADED_DATASETS:
        info = DOMAINS[domain]
        path = Path(info["dataset_path"])
        if path.exists():
            try:
                print(f"Loading dataset {domain} from {path}...")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Precompute token sets for fast search matching
                for entry in data:
                    entry["_q_words"] = set(entry["question"].lower().split())
                _LOADED_DATASETS[domain] = data
                print(f"Dataset {domain} loaded successfully. Size: {len(data)}.")
            except Exception as e:
                print(f"Failed to load dataset {path}: {e}")
                _LOADED_DATASETS[domain] = []
        else:
            _LOADED_DATASETS[domain] = []
    return _LOADED_DATASETS[domain]

def find_best_match(question, dataset):
    if not dataset:
        return None
    q_words = set(question.lower().split())
    if not q_words:
        return None
    best_score = 0
    best_entry = None
    for entry in dataset:
        words = entry.get("_q_words")
        if words is None:
            words = set(entry["question"].lower().split())
            entry["_q_words"] = words
        overlap = len(words & q_words)
        if overlap > best_score:
            best_score = overlap
            best_entry = entry
    return best_entry

def predict_reasoning(loaded, question, beam_width=1):
    import grammar_parser
    normalized_question = grammar_parser.normalize_query(question)

    model = loaded["model"]
    vocab = loaded["vocab"]
    tokenizer = loaded["tokenizer"]
    device = loaded["device"]

    model.eval()

    with torch.no_grad():
        encoded = tokenizer.encode(normalized_question, max_length=64)
        input_ids = encoded["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)

        outputs = model(input_ids, attention_mask, beam_width=beam_width)

        path_ids = outputs["predicted_path"][0].detach().cpu().tolist()
        path_scores = outputs["path_scores"][0].detach().cpu().tolist()

        path_probs = []
        for s in path_scores:
            try:
                p = math.exp(s)
            except OverflowError:
                p = 0.0
            path_probs.append(min(max(p, 0.0), 1.0))

        reasoning_path = vocab.decode_path(path_ids)

        # Generate answer using cached decoder
        decoder = loaded.get("decoder")
        if decoder is not None:
            try:
                answer = decoder.generate_answer(normalized_question, reasoning_path)
            except Exception as ex:
                print(f"Error generating answer in predict_reasoning: {ex}")
                answer = ""
        else:
            answer = ""

        # Sigmoid activations
        if "activation_logits" in outputs:
            activation_logits = outputs["activation_logits"]
            activation_probs = torch.sigmoid(activation_logits).squeeze(0).cpu().tolist()
        else:
            activation_probs = [0.0] * vocab.size()

    # Collect top activated concepts
    top_concepts = []
    for idx, prob in enumerate(activation_probs):
        if idx in vocab.id_to_concept:
            concept_name = vocab.id_to_concept[idx]
            if concept_name not in ("<PAD>", "<EOS>") and prob > 0.05:
                top_concepts.append({
                    "name": concept_name,
                    "activation": prob
                })
    top_concepts.sort(key=lambda x: -x["activation"])

    # Collect graph nodes/edges
    nodes_list = []
    for name in vocab.concept_names():
        c_id = vocab.concept_to_id[name]
        nodes_list.append({
            "name": name,
            "activation": float(activation_probs[c_id]) if c_id < len(activation_probs) else 0.0
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
        "edges": edges_list,
        "answer": answer
    }

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
            status_data = {}
            for domain, info in DOMAINS.items():
                loaded = get_system(domain)
                ds = get_dataset(domain)
                status_data[domain] = {
                    "model_loaded": loaded is not None,
                    "dataset_size": len(ds),
                    "checkpoint_dir": info["checkpoint_dir"]
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        elif parsed.path == "/api/gate/suggestions":
            try:
                import gate_router
                suggestions = gate_router.get_gate_suggested_questions()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(suggestions).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
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
                domain = data.get("domain", "mechanical_engineering").strip()
                beam_width = int(data.get("beam_width", 1))
                beam_width = max(1, min(beam_width, 5))

                if not question:
                    raise ValueError("Empty question")
                if domain not in DOMAINS:
                    domain = "mechanical_engineering"

                response_data = {}

                # Try model inference first
                loaded = get_system(domain)
                dataset = get_dataset(domain)

                if loaded:
                    info = DOMAINS[domain]
                    if info.get("is_cat_v3"):
                        result = predict_reasoning_cat_v3(loaded, question)
                    else:
                        result = predict_reasoning(loaded, question, beam_width=beam_width)
                    response_data["reasoning_path"] = result["reasoning_path"]
                    response_data["path_probabilities"] = result["path_probabilities"]
                    response_data["top_concepts"] = result["top_concepts"]
                    response_data["nodes"] = result["nodes"]
                    response_data["edges"] = result["edges"]
                    response_data["answer"] = result["answer"]
                    response_data["source"] = "model"

                # Dataset match for the answer text (only if model didn't load or didn't generate an answer)
                if not response_data.get("answer"):
                    match = find_best_match(question, dataset)
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

                # Integrate GATE orchestration: multi-concept routing + symbolic solver
                try:
                    import gate_router
                    gate_result = gate_router.process_gate_query(
                        question,
                        model_path=response_data.get("reasoning_path"),
                        top_concepts=response_data.get("top_concepts"),
                    )
                    if gate_result.get("concept_path"):
                        response_data["reasoning_path"] = gate_result["concept_path"]
                        response_data["gate_routing"] = gate_result["routing"]
                    if gate_result.get("solved"):
                        response_data["solved_equation"] = gate_result["solved"]
                    if gate_result.get("gate_match"):
                        response_data["gate_match"] = gate_result["gate_match"]
                    if gate_result.get("concept_notes"):
                        response_data["concept_notes"] = gate_result["concept_notes"]
                    response_data["relevant_equations"] = gate_result.get("relevant_equations", [])
                    gate_text = gate_result.get("composed_answer", "")
                    if gate_text:
                        if gate_result.get("solved") or gate_result.get("gate_match"):
                            response_data["answer"] = gate_text
                            response_data["source"] = "gate_solver"
                        else:
                            response_data["answer"] += f"\n\n{gate_text}"
                except Exception as ex:
                    print(f"Error in GATE orchestration: {ex}")

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
    <title>CAT V2 & VLCM — Engineering Reasoning Lab Chat</title>
    <meta name="description" content="Interactive multi-domain chat interface powered by VLCM and Concept Attention Transformer reasoning.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
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

        #app-main {
            flex: 1;
            display: flex;
            overflow: hidden;
            position: relative;
            z-index: 1;
        }

        #sidebar {
            width: 320px;
            min-width: 320px;
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

        .domain-tabs {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .domain-tab {
            padding: 0.6rem 0.85rem;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            background: rgba(17, 24, 39, 0.3);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-base);
            font-family: inherit;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            text-align: left;
        }

        .domain-tab:hover {
            background: var(--accent-indigo-dim);
            border-color: var(--border-accent);
            color: var(--text-primary);
        }

        .domain-tab.active {
            background: var(--accent-indigo-dim);
            border-color: var(--accent-indigo);
            color: var(--accent-indigo-light);
            box-shadow: 0 0 12px rgba(99,102,241,0.12);
        }

        .domain-tab .domain-symbol {
            font-size: 1rem;
        }

        .sidebar-questions-title {
            padding: 1rem 1.25rem 0.25rem;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
        }

        .sidebar-questions {
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem 0.75rem;
        }

        .sidebar-questions::-webkit-scrollbar { width: 4px; }
        .sidebar-questions::-webkit-scrollbar-track { background: transparent; }
        .sidebar-questions::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: 4px; }

        .question-card {
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.4rem;
            border: 1px solid transparent;
            border-radius: var(--radius-sm);
            font-size: 0.78rem;
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
            font-size: 0.58rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            margin-bottom: 0.3rem;
        }

        .tag-featured { background: rgba(251, 113, 133, 0.12); color: var(--accent-rose); border: 1px solid rgba(251, 113, 133, 0.2); }
        .tag-general { background: rgba(99, 102, 241, 0.12); color: var(--accent-indigo-light); border: 1px solid rgba(99, 102, 241, 0.2); }

        #chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
            background: var(--bg-deep);
            position: relative;
        }

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
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
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

        .welcome-details {
            padding: 1.25rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-accent);
            border-radius: var(--radius-lg);
            max-width: 520px;
            width: 100%;
            text-align: left;
            box-shadow: var(--shadow-md);
            animation: popIn 0.3s ease-out;
        }

        .welcome-details-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--accent-indigo-light);
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .welcome-details-desc {
            font-size: 0.82rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

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

        .equation-card-right {
            background: rgba(99, 102, 241, 0.05);
            border: 1px solid var(--border-accent);
            border-radius: var(--radius-md);
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            animation: popIn 0.3s ease-out;
        }
        .equation-title-right {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--accent-indigo-light);
            margin-bottom: 0.3rem;
        }
        .equation-formula-right {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-primary);
            background: rgba(17, 24, 39, 0.4);
            padding: 0.35rem 0.5rem;
            border-radius: 4px;
            margin-bottom: 0.4rem;
            text-align: center;
        }
        .equation-vars-right {
            font-size: 0.65rem;
            color: var(--text-muted);
            line-height: 1.4;
        }

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

        @keyframes popIn {
            0% { transform: scale(0.95); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

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
        <div class="logo-icon" id="header-logo-icon">⚙️</div>
        <div class="header-title">
            <h1 id="header-domain-title">Mechanical Engineering Reasoning</h1>
            <span>VLCM + CAT V2 Concept Reasoning System</span>
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
            <h2>Select Domain</h2>
            <div class="domain-tabs">
                <button class="domain-tab active" data-domain="mechanical_engineering" id="tab-mech">
                    <span class="domain-symbol">⚙️</span>
                    <span>Mechanical Engineering</span>
                </button>
                <button class="domain-tab" data-domain="cat_v3_moe" id="tab-catv3">
                    <span class="domain-symbol">🔮</span>
                    <span>CAT V3 / Graph-MoE</span>
                </button>
                <button class="domain-tab" data-domain="mit_stanford_mech" id="tab-curriculum">
                    <span class="domain-symbol">🎓</span>
                    <span>MIT & Stanford Curriculum</span>
                </button>
                <button class="domain-tab" data-domain="mit_math" id="tab-math">
                    <span class="domain-symbol">∫</span>
                    <span>MIT OCW Math</span>
                </button>
                <button class="domain-tab" data-domain="structural" id="tab-struct">
                    <span class="domain-symbol">🏗️</span>
                    <span>Structural Eng.</span>
                </button>
                <button class="domain-tab" data-domain="cfd" id="tab-cfd">
                    <span class="domain-symbol">🌪️</span>
                    <span>CFD / Fluids</span>
                </button>
                <button class="domain-tab" data-domain="python_coding" id="tab-python">
                    <span class="domain-symbol">🐍</span>
                    <span>Python Coding AI</span>
                </button>
            </div>
        </div>
        <div class="sidebar-header" style="border-top: 1px solid var(--border-subtle); padding-top: 1rem;">
            <h2>Decoding Settings</h2>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-secondary);">
                <span>Beam Width:</span>
                <select id="beam-width-select" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); color: var(--text-primary); border-radius: var(--radius-sm); padding: 0.25rem 0.5rem; outline: none; font-size: 0.8rem; cursor: pointer; font-family: inherit; font-weight: 600;">
                    <option value="1" selected>1 (Greedy)</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="5">5</option>
                </select>
            </div>
        </div>
        <div class="sidebar-questions-title">Recommended Questions</div>
        <div class="sidebar-questions" id="questions-list">
            <!-- Populated by JS -->
        </div>
    </aside>

    <!-- ─── Chat Area ─── -->
    <section id="chat-area">
        <div id="welcome-screen">
            <div class="welcome-icon" id="welcome-logo-icon">⚙️</div>
            <div class="welcome-title" id="welcome-domain-title">Mechanical Engineering Reasoning</div>
            <div class="welcome-sub" id="welcome-domain-desc">
                1000-chunk dataset trained with Second-Order Differential Loss
            </div>
            
            <div class="welcome-details">
                <div class="welcome-details-title">🧠 Explainable Concept-Level Reasoning</div>
                <div class="welcome-details-desc">
                    Unlike standard LLMs which generate tokens directly, this system searches for a logical sequence of engineering concepts first. The reasoning path is displayed dynamically alongside the answer, ensuring complete transparency.
                </div>
            </div>
        </div>

        <div id="messages-container" style="display: none;"></div>

        <div id="input-area">
            <div class="input-wrapper">
                <textarea id="chat-input" rows="1" placeholder="Ask an engineering question..." maxlength="500"></textarea>
                <button id="send-btn" aria-label="Send message" disabled>
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
        <div class="panel-section">
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
        <div class="panel-section" style="border-bottom: none;">
            <div class="panel-section-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                Relevant Equations & Solvers
            </div>
            <div id="panel-equations">
                <div class="panel-empty">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                    <div>Formulas, equations and solvers will<br>appear here</div>
                </div>
            </div>
        </div>
    </aside>
</main>

<script>
// ═══════════════════════════════════════════════════════════════
//  Data: Multi-Domain Suggested Questions
// ═══════════════════════════════════════════════════════════════
const DOMAIN_QUESTIONS = {
    "cat_v3_moe": [
        { q: "Why does compressor pressure ratio affect turbine efficiency?", tag: "Mechanical/Physics" },
        { q: "How does a foundation load affect beam buckling?", tag: "Civil/Mechanical" },
        { q: "Why does capacitor impedance shift current phase angle?", tag: "Electrical/Physics" },
        { q: "How do eigenvalues explain decay differential equations?", tag: "Mathematics/Physics" },
        { q: "How does syntax affect semantic interpretation of a sentence?", tag: "English/Language" },
        { q: "How does thermal stress lead to cracking in turbine blades?", tag: "Mechanical/Physics" }
    ],
    "mechanical_engineering": [
        { q: "Calculate stress if load equals ten kN and area measures five m^2", tag: "GATE NAT" },
        { q: "A steel column of length 2.0 m has pinned ends. If E = 200e9 Pa and I = 1.0e-5 m^4, what is the critical buckling load in kN?", tag: "GATE NAT" },
        { q: "For a column of length L, if one end is fixed and the other is free, what is the effective length?", tag: "GATE MCQ" },
        { q: "Calculate the Reynolds number for water (density = 1000 kg/m^3, viscosity = 0.001 Pa-s) flowing at 2.0 m/s in a 0.05 m diameter pipe.", tag: "GATE Numerical" },
        { q: "Why do slender columns fail by buckling rather than crushing?", tag: "Columns / Buckling" },
        { q: "Why does turbulence increase the heat transfer coefficient in pipe flow?", tag: "Fluids / Turbulence" },
        { q: "Calculate the engineering strain if a 2.0 m rod stretches by 0.004 m under tensile load.", tag: "Materials / Strain" },
        { q: "Calculate the critical damping coefficient for a system with mass 5.0 kg and stiffness 500 N/m.", tag: "Vibrations / Damping" },
        { q: "Given a matrix A with trace of 5.0 and determinant is 6.0, what is the eigenvalue?", tag: "Math / Cayley-Hamilton" },
        { q: "If output load is 100 N and output displacement is 0.05 m and input displacement is 0.2 m, find input force", tag: "Mechanics / Virtual Work" },
        { q: "If gear 1 speed is 100 rad/s, pitch radius of gear 1 is 0.05 m, and pitch radius of gear 2 is 0.1 m, find the gear 2 speed", tag: "Gears / Law of Gearing" },
        { q: "Determine the logarithmic decrement if damping ratio is 0.1", tag: "Vibrations / Log Dec" },
        { q: "Find the factor of safety if mean stress is 100 MPa, yield strength is 300 MPa, stress amplitude is 50 MPa, and endurance limit is 150 MPa", tag: "Design / Soderberg" },
        { q: "Calculate the tool life in min if cutting speed is 120 m/min, exponent n is 0.25, and constant C is 240", tag: "Mfg / Taylor's Tool Life" },
        { q: "Compute the economic order quantity if annual demand is 10000 units, ordering cost is 50, and holding cost is 4", tag: "Industrial / EOQ" }
    ],
    "mit_math": [
        { q: "Why does the gradient point in the direction of steepest ascent?", tag: "18.02 Calculus" },
        { q: "How do eigenvalues determine stability of a linear system?", tag: "18.03 ODEs" },
        { q: "Why does resonance cause unbounded growth in forced oscillators?", tag: "18.03 ODEs" },
        { q: "Why does the Laplace transform convert ODEs to algebraic equations?", tag: "18.03 ODEs" },
        { q: "How does an eigenvalue decomposition diagonalize a matrix?", tag: "18.06 Linalg" },
        { q: "How does Gaussian elimination solve systems of linear equations?", tag: "18.06 Linalg" }
    ],
    "structural": [
        { q: "How does load cause structural failure?", tag: "Load & Failure" },
        { q: "Why does a column buckle under compression?", tag: "Compression" },
        { q: "Why does cyclic loading cause fatigue?", tag: "Fatigue Loading" },
        { q: "Why does thermal stress create cracking?", tag: "Thermal Stress" }
    ],
    "cfd": [
        { q: "Why does pressure drop in a pipe?", tag: "Pressure Drop" },
        { q: "Why does turbulence increase at high speed?", tag: "Turbulence" },
        { q: "Why can poor mesh quality cause convergence failure?", tag: "Mesh Quality" },
        { q: "Why does cavitation damage pumps?", tag: "Cavitation" }
    ],
    "python_coding": [
        { q: "How to read a file line by line and find a word?", tag: "File I/O" },
        { q: "How to filter a list of numbers to find even numbers?", tag: "Lists" },
        { q: "How to fetch a URL and parse JSON in Python?", tag: "Network/APIs" },
        { q: "How to sort a list of dictionaries by a key?", tag: "Sorting" }
    ],
    "mit_stanford_mech": [
        { q: "Calculate stress if load equals ten kN and area measures five m^2", tag: "GATE NAT" },
        { q: "A cantilever column of length 3.0 m has E = 210e9 Pa and I = 2.0e-6 m^4. Find the critical buckling load in kN.", tag: "GATE NAT" },
        { q: "For a column with both ends pinned, the effective length factor K equals:", tag: "GATE MCQ" },
        { q: "Find Carnot efficiency if hot reservoir temperature is 800 K and cold reservoir temperature is 300 K.", tag: "GATE NAT" },
        { q: "Why do slender columns fail by buckling rather than crushing?", tag: "Columns / Buckling" },
        { q: "Why does turbulence increase the heat transfer coefficient in pipe flow?", tag: "Fluids / Turbulence" },
        { q: "Calculate the engineering strain if a 2.0 m rod stretches by 0.004 m under tensile load.", tag: "Materials / Strain" },
        { q: "Calculate the critical damping coefficient for a system with mass 5.0 kg and stiffness 500 N/m.", tag: "Vibrations / Damping" },
        { q: "Given a matrix A with trace of 5.0 and determinant is 6.0, what is the eigenvalue?", tag: "Math / Cayley-Hamilton" },
        { q: "If output load is 100 N and output displacement is 0.05 m and input displacement is 0.2 m, find input force", tag: "Mechanics / Virtual Work" },
        { q: "If gear 1 speed is 100 rad/s, pitch radius of gear 1 is 0.05 m, and pitch radius of gear 2 is 0.1 m, find the gear 2 speed", tag: "Gears / Law of Gearing" },
        { q: "Determine the logarithmic decrement if damping ratio is 0.1", tag: "Vibrations / Log Dec" },
        { q: "Find the factor of safety if mean stress is 100 MPa, yield strength is 300 MPa, stress amplitude is 50 MPa, and endurance limit is 150 MPa", tag: "Design / Soderberg" },
        { q: "Calculate the tool life in min if cutting speed is 120 m/min, exponent n is 0.25, and constant C is 240", tag: "Mfg / Taylor's Tool Life" },
        { q: "Compute the economic order quantity if annual demand is 10000 units, ordering cost is 50, and holding cost is 4", tag: "Industrial / EOQ" }
    ]
};

const DOMAIN_DETAILS = {
    "cat_v3_moe": { title: "CAT V3 / Graph-MoE (Prototype)", icon: "🔮", desc: "Multi-expert routing with GAT graph reasoning & fusion across 6 engineering domains", placeholder: "Ask multi-domain queries (e.g. pressure ratios, foundation loads, eigenvalues)..." },
    "mechanical_engineering": { title: "Mechanical Engineering (VLCM)", icon: "⚙️", desc: "Scaled 10k-concept dataset (140k edges, 58k causal) with Second-Order Loss", placeholder: "Ask about buckling, heat exchangers, fluid dynamics, stress tensors..." },
    "mit_stanford_mech": { title: "MIT & Stanford ME Curriculum (VLCM)", icon: "🎓", desc: "Complete syllabus (10k+ concepts, 50k+ reasoning paths)", placeholder: "Ask about buckling, heat exchangers, control systems, Navier-Stokes..." },
    "mit_math": { title: "MIT OCW Mathematics (CAT V2)", icon: "∫", desc: "Calculus, Linear Algebra, ODEs, and Multivariable Calculus", placeholder: "Ask about eigenvalues, gradients, Laplace transforms, Fourier series..." },
    "structural": { title: "Structural Engineering (CAT V2)", icon: "🏗️", desc: "Beams, stress-strain, columns buckling, fatigue, and materials science", placeholder: "Ask about load distributions, Euler buckling, S-N curves, strain..." },
    "cfd": { title: "CFD & Fluid Dynamics (CAT V2)", icon: "🌪️", desc: "Pipe flow, pressure drop, turbulence, boundary layers, and mesh quality", placeholder: "Ask about boundary layers, adverse gradients, Navier-Stokes residuals..." },
    "python_coding": { title: "Python Coding AI (CAT V2)", icon: "🐍", desc: "Autoregressive code generation, lists, sorting, file I/O, and API requests", placeholder: "Ask about file line parsing, list comprehensions, sorting dicts, JSON..." }
};

// ═══════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════
let currentDomain = "mechanical_engineering";
let isProcessing = false;
let statusInfo = null;

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
const panelEquations = document.getElementById("panel-equations");

const headerLogoIcon = document.getElementById("header-logo-icon");
const headerDomainTitle = document.getElementById("header-domain-title");
const welcomeLogoIcon = document.getElementById("welcome-logo-icon");
const welcomeDomainTitle = document.getElementById("welcome-domain-title");
const welcomeDomainDesc = document.getElementById("welcome-domain-desc");

// ═══════════════════════════════════════════════════════════════
//  Initialize
// ═══════════════════════════════════════════════════════════════
renderSidebarQuestions();
checkStatus();

// Event listeners
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
    sendBtn.disabled = chatInput.value.trim().length === 0;
});

sendBtn.addEventListener("click", sendMessage);

document.querySelectorAll(".domain-tab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".domain-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        currentDomain = tab.dataset.domain;
        
        // Update Domain details across UI
        const details = DOMAIN_DETAILS[currentDomain];
        headerLogoIcon.textContent = details.icon;
        headerDomainTitle.textContent = details.title;
        welcomeLogoIcon.textContent = details.icon;
        welcomeDomainTitle.textContent = details.title;
        welcomeDomainDesc.textContent = details.desc;
        chatInput.placeholder = details.placeholder;
        
        renderSidebarQuestions();
        updateStatusBadge();
        
        // Hide messages, show welcome screen when switching domains
        messagesContainer.style.display = "none";
        messagesContainer.innerHTML = "";
        welcomeScreen.style.display = "flex";
        
        // Clear right panels
        panelPath.innerHTML = `
            <div class="panel-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <div>Ask a question to see the<br>concept reasoning path</div>
            </div>
        `;
        panelActivations.innerHTML = `
            <div class="panel-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                <div>Concept activations will<br>appear here</div>
            </div>
        `;
        panelEquations.innerHTML = `
            <div class="panel-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                <div>Formulas, equations and solvers will<br>appear here</div>
            </div>
        `;
    });
});

// ═══════════════════════════════════════════════════════════════
//  Functions
// ═══════════════════════════════════════════════════════════════

async function checkStatus() {
    try {
        const res = await fetch("/api/status");
        statusInfo = await res.json();
        updateStatusBadge();
    } catch (e) {
        document.getElementById("status-text").textContent = "Offline";
    }
}

function updateStatusBadge() {
    const statusText = document.getElementById("status-text");
    const statusDot = document.querySelector(".status-dot");
    if (!statusInfo || !statusInfo[currentDomain]) {
        statusText.textContent = "Offline";
        statusDot.style.background = "#ef4444";
        return;
    }
    const info = statusInfo[currentDomain];
    if (info.model_loaded) {
        statusText.textContent = "Model Online";
        statusDot.style.background = "var(--accent-emerald)";
    } else {
        statusText.textContent = `Dataset (${info.dataset_size} samples)`;
        statusDot.style.background = "var(--accent-amber)";
    }
}

function renderSidebarQuestions() {
    questionsList.innerHTML = "";
    const list = DOMAIN_QUESTIONS[currentDomain] || [];

    list.forEach((item, idx) => {
        const card = document.createElement("div");
        card.className = "question-card";

        const tagClass = idx === 0 ? "tag-featured" : "tag-general";
        const tagText = idx === 0 ? "🔥 Featured" : item.tag;
        card.innerHTML = `
            <span class="question-tag ${tagClass}">${tagText}</span>
            <div>${item.q}</div>
        `;

        card.addEventListener("click", () => {
            chatInput.value = item.q;
            chatInput.style.height = "auto";
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
            sendBtn.disabled = false;
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
    sendBtn.disabled = true;

    isProcessing = true;

    // Show typing indicator
    const typingEl = addTypingIndicator();

    try {
        const beamWidth = parseInt(document.getElementById("beam-width-select")?.value || "1", 10);
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: text, domain: currentDomain, beam_width: beamWidth })
        });

        if (!res.ok) throw new Error("Server error");

        const data = await res.json();

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
    scrollToBottom();
}

function formatMarkdown(text) {
    if (!text) return "";
    // Replace double dollars $$formula$$ or $formula$ with styled mono boxes
    let html = text.replace(/\$\$(.*?)\$\$/g, '<div class="equation-formula-right">$1</div>');
    html = html.replace(/\$(.*?)\$/g, '<code style="background: rgba(17, 24, 39, 0.4); padding: 0.1rem 0.3rem; border-radius: 4px; font-family: monospace;">$1</code>');
    // Replace double newlines with <br><br>
    html = html.replace(/\n\n/g, "<br><br>");
    // Replace single newlines with <br>
    html = html.replace(/\n/g, "<br>");
    // Replace **text** with <strong>text</strong>
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    return html;
}

function addMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "U" : DOMAIN_DETAILS[currentDomain].icon;

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
    avatar.textContent = DOMAIN_DETAILS[currentDomain].icon;

    const body = document.createElement("div");
    body.className = "message-body";

    // Answer bubble
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = formatMarkdown(data.answer || "Path generated — see reasoning below.");
    body.appendChild(bubble);

    // Reasoning path inline
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
        badge.textContent = data.source === "model" ? "⚡ VLCM/CAT Model" : "📚 Knowledge Base";
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
    avatar.textContent = DOMAIN_DETAILS[currentDomain].icon;

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

    // Equations & Math Solvers
    let eqHtml = '';
    
    // 1. Show solved equation if present
    if (data.solved_equation) {
        const solved = data.solved_equation;
        const inputsStr = Object.entries(solved.inputs)
            .map(([k, v]) => `${k} = ${v}`)
            .join(', ');
        
        eqHtml += `
            <div class="equation-card-right" style="background: rgba(52, 211, 153, 0.08); border-color: rgba(52, 211, 153, 0.35);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div class="equation-title-right" style="color: var(--accent-emerald); font-weight: 800;">⚡ Solver: ${solved.equation}</div>
                    <span class="source-badge" style="background: rgba(52, 211, 153, 0.15); color: var(--accent-emerald); border-color: rgba(52, 211, 153, 0.3); font-size: 0.58rem;">SOLVED</span>
                </div>
                <div class="equation-formula-right" style="border-left: 2px solid var(--accent-emerald); font-size: 0.9rem;">${solved.formula}</div>
                <div class="equation-vars-right" style="color: var(--text-secondary); margin-bottom: 0.4rem;">
                    <strong>Inputs:</strong> ${inputsStr}
                </div>
                <div style="background: rgba(52, 211, 153, 0.12); padding: 0.5rem; border-radius: var(--radius-sm); font-size: 0.75rem; border: 1px dashed rgba(52, 211, 153, 0.25);">
                    <span style="color: var(--text-muted);">Result:</span>
                    <strong style="color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">${solved.solved_variable} = ${solved.solved_value}</strong>
                </div>
            </div>
        `;
    }

    // 2. Show relevant equations
    if (data.relevant_equations && data.relevant_equations.length > 0) {
        data.relevant_equations.forEach(eq => {
            // Avoid repeating the solved one if it's already shown
            if (data.solved_equation && data.solved_equation.equation === eq.name) {
                return;
            }
            
            let varsList = '';
            if (eq.variables) {
                varsList = Object.entries(eq.variables)
                    .map(([k, v]) => `<div><strong>${k}</strong>: ${v}</div>`)
                    .join('');
            }
            
            eqHtml += `
                <div class="equation-card-right">
                    <div class="equation-title-right">⚙️ ${eq.name}</div>
                    <div class="equation-formula-right">${eq.formula}</div>
                    <div class="equation-vars-right">
                        ${varsList}
                    </div>
                </div>
            `;
        });
    }

    // If nothing matched, show empty message
    if (!eqHtml) {
        eqHtml = `
            <div class="panel-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                <div>Formulas, equations and solvers will<br>appear here</div>
            </div>
        `;
    }
    
    panelEquations.innerHTML = eqHtml;
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
    print(f"  Multi-Domain Engineering Chat Server")
    print(f"  Powered by CAT V2 & VLCM Models")
    print(f"{'='*60}")
    print(f"  -> Open: http://localhost:{port}/")
    for domain, info in DOMAINS.items():
        ds = get_dataset(domain)
        loaded = get_system(domain)
        status = "ONLINE" if loaded else "FALLBACK (dataset)"
        print(f"  -> {info['name']}: {len(ds)} QA, Model: {status}")
    print(f"{'='*60}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down chat server.")
        httpd.server_close()

if __name__ == "__main__":
    run_chat_server(8090)
