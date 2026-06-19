"""Dataset and Vocabulary definitions for CAT V3."""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple, Optional
import torch
from torch.utils.data import Dataset

# Domain lists
DOMAINS = [
    "mechanical",
    "civil",
    "electrical",
    "physics",
    "mathematics",
    "english"
]

# Domain-specific concept vocabularies
DOMAIN_CONCEPTS = {
    "mechanical": [
        "compressor", "pressure_ratio", "turbine", "efficiency", "entropy",
        "temperature_rise", "fluid_flow", "thermal_stress", "cracking", "buckling"
    ],
    "civil": [
        "foundation", "concrete", "rebar", "soil_bearing", "beam",
        "load", "bending_moment", "shear_force", "buckling"
    ],
    "electrical": [
        "voltage", "current", "impedance", "resistor", "inductor",
        "capacitor", "frequency", "power_factor", "phase_angle"
    ],
    "physics": [
        "entropy", "thermodynamics", "energy", "force", "acceleration",
        "velocity", "heat", "temperature_rise", "pressure", "gravity"
    ],
    "mathematics": [
        "eigenvalue", "matrix", "characteristic_equation", "derivative", "integral",
        "differential_equation", "decay", "logarithm", "acceleration"
    ],
    "english": [
        "syntax", "grammar", "semantics", "vocabulary", "metaphor",
        "sentence", "noun", "verb", "adjective"
    ]
}

# Simple Character / Word level tokenizer for English Decoder
class SimpleCharTokenizer:
    """A simple vocabulary and tokenizer for query encoding and response decoding."""

    def __init__(self, texts: List[str]) -> None:
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        # Build token list (split by words/spaces and lowercase)
        tokens = set()
        for text in texts:
            for word in self._tokenize_text(text):
                tokens.add(word)
                
        self.vocab = [self.pad_token, self.eos_token, self.unk_token] + sorted(list(tokens))
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}
        
        self.pad_id = self.token_to_id[self.pad_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.unk_id = self.token_to_id[self.unk_token]

    def _tokenize_text(self, text: str) -> List[str]:
        # Simple punctuation splitter
        cleaned = text.lower().replace(".", " . ").replace(",", " , ").replace("?", " ? ").replace("!", " ! ")
        return cleaned.split()

    def encode(self, text: str, max_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        words = self._tokenize_text(text)
        ids = [self.token_to_id.get(w, self.unk_id) for w in words]
        ids = ids[:max_length - 1] + [self.eos_id]
        
        # Pad
        attention_mask = [1] * len(ids)
        if len(ids) < max_length:
            padding = [self.pad_id] * (max_length - len(ids))
            ids = ids + padding
            attention_mask = attention_mask + [0] * len(padding)
            
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.unk_id):
                continue
            words.append(self.id_to_token.get(idx, ""))
        # Re-assemble text
        text = " ".join(words)
        # Fix basic punctuation spaces
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        return text

    def vocab_size(self) -> int:
        return len(self.vocab)


class ConceptVocabulary:
    """Vocabulary mapping domain concepts to integer IDs."""

    def __init__(self, domains_dict: Dict[str, List[str]]) -> None:
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        
        # Build global list of unique concepts
        all_concepts = set()
        for concepts in domains_dict.values():
            for concept in concepts:
                all_concepts.add(concept.strip().lower())
                
        self.concepts = [self.pad_token, self.eos_token] + sorted(list(all_concepts))
        self.concept_to_id = {c: idx for idx, c in enumerate(self.concepts)}
        self.id_to_concept = {idx: c for idx, c in enumerate(self.concepts)}
        
        self.pad_id = self.concept_to_id[self.pad_token]
        self.eos_id = self.concept_to_id[self.eos_token]

    def size(self) -> int:
        return len(self.concepts)

    def encode_path(self, path: List[str], max_length: int = 8) -> torch.Tensor:
        ids = [self.concept_to_id[c.strip().lower()] for c in path if c.strip().lower() in self.concept_to_id]
        ids = ids[:max_length - 1] + [self.eos_id]
        if len(ids) < max_length:
            ids = ids + [self.pad_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode_path(self, ids: List[int]) -> List[str]:
        path = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx == self.pad_id:
                continue
            path.append(self.id_to_concept.get(idx, str(idx)))
        return path


# Sample queries and ground truth paths/responses
RAW_DATASET = [
    # Mechanical + Physics + Maths
    {
        "question": "Why does compressor pressure ratio affect turbine efficiency?",
        "active_experts": ["mechanical", "physics", "mathematics"],
        "concept_paths": [
            ["compressor", "pressure_ratio", "temperature_rise", "entropy", "efficiency"]
        ],
        "response": "Increasing compressor pressure ratio raises outlet temperature. This increases entropy generation and can reduce overall efficiency."
    },
    {
        "question": "How does thermal stress lead to cracking in turbine blades?",
        "active_experts": ["mechanical", "physics"],
        "concept_paths": [
            ["temperature_rise", "thermal_stress", "cracking"]
        ],
        "response": "High temperature rise induces severe thermal stress, which eventually leads to cracking."
    },
    # Civil + Mechanical + Physics
    {
        "question": "How does a foundation load affect beam buckling?",
        "active_experts": ["civil", "mechanical", "physics"],
        "concept_paths": [
            ["foundation", "load", "bending_moment", "beam", "buckling"]
        ],
        "response": "The foundation transfers the load, causing a high bending moment in the beam which results in buckling."
    },
    {
        "question": "What is the relation between shear force and bending moment in concrete foundations?",
        "active_experts": ["civil"],
        "concept_paths": [
            ["concrete", "foundation", "load", "shear_force", "bending_moment"]
        ],
        "response": "Applying load on concrete foundation structures produces shear force that integrates into bending moment distributions."
    },
    # Electrical + Physics + Maths
    {
        "question": "Why does capacitor impedance shift current phase angle?",
        "active_experts": ["electrical", "physics", "mathematics"],
        "concept_paths": [
            ["capacitor", "impedance", "frequency", "phase_angle", "current"]
        ],
        "response": "A capacitor impedance varies with frequency, causing current phase angle to lead the voltage."
    },
    {
        "question": "How do resistor and inductor values determine power factor?",
        "active_experts": ["electrical", "physics"],
        "concept_paths": [
            ["resistor", "inductor", "impedance", "phase_angle", "power_factor"]
        ],
        "response": "Combining resistor and inductor impedances increases the phase angle, which lowers the power factor."
    },
    # Mathematics + Physics
    {
        "question": "How do eigenvalues explain decay differential equations?",
        "active_experts": ["mathematics", "physics"],
        "concept_paths": [
            ["matrix", "eigenvalue", "differential_equation", "decay", "logarithm"]
        ],
        "response": "Using matrix eigenvalues helps solve the differential equation, showing exponential decay governed by logarithms."
    },
    # English
    {
        "question": "How does syntax affect semantic interpretation of a sentence?",
        "active_experts": ["english"],
        "concept_paths": [
            ["syntax", "grammar", "sentence", "semantics"]
        ],
        "response": "Proper syntax and grammar organize words in a sentence to deliver clear semantics."
    },
    {
        "question": "How are nouns and verbs combined to form a metaphor?",
        "active_experts": ["english"],
        "concept_paths": [
            ["vocabulary", "noun", "verb", "sentence", "metaphor"]
        ],
        "response": "Selecting noun and verb combinations from vocabulary builds a sentence representing a metaphor."
    }
]

# Grow additional mock samples for better training size
def grow_dataset() -> List[Dict[str, object]]:
    data = list(RAW_DATASET)
    # Generate variations
    for _ in range(30):
        # Pick a base
        base = random.choice(RAW_DATASET)
        q = base["question"]
        # Add slight noise to question
        prefix = random.choice(["Could you explain ", "Explain ", "Analyze ", "Question: ", ""])
        suffix = random.choice(["", " please.", " in detail.", " brief."])
        new_q = f"{prefix}{q}{suffix}".strip()
        data.append({
            "question": new_q,
            "active_experts": base["active_experts"],
            "concept_paths": base["concept_paths"],
            "response": base["response"]
        })
    return data


class CATV3Dataset(Dataset):
    """PyTorch Dataset for CAT V3."""

    def __init__(self, raw_data: List[Dict[str, object]], concept_vocab: ConceptVocabulary, token_tokenizer: SimpleCharTokenizer, max_len_q: int = 32, max_len_p: int = 8, max_len_r: int = 32) -> None:
        self.data = raw_data
        self.concept_vocab = concept_vocab
        self.tokenizer = token_tokenizer
        self.max_len_q = max_len_q
        self.max_len_p = max_len_p
        self.max_len_r = max_len_r

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize query
        input_ids, attention_mask = self.tokenizer.encode(item["question"], max_length=self.max_len_q)
        
        # Router targets (multilabel binary vector of size 6)
        router_target = torch.zeros(len(DOMAINS), dtype=torch.float)
        for domain in item["active_experts"]:
            if domain in DOMAINS:
                router_target[DOMAINS.index(domain)] = 1.0
                
        # Primary concept path
        path_concepts = item["concept_paths"][0]
        path_ids = self.concept_vocab.encode_path(path_concepts, max_length=self.max_len_p)
        
        # Response tokens
        response_ids, response_mask = self.tokenizer.encode(item["response"], max_length=self.max_len_r)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "router_target": router_target,
            "path_ids": path_ids,
            "response_ids": response_ids,
            "response_mask": response_mask
        }


# Expert-specific GAT adjacency construction helpers
def build_expert_graphs(vocab: ConceptVocabulary) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Returns edge_index and edge_weight for each GAT expert, ensuring dataset paths are valid edges."""
    expert_graphs = {}
    
    for domain in DOMAINS:
        concepts = DOMAIN_CONCEPTS[domain]
        concept_ids = [vocab.concept_to_id[c] for c in concepts if c in vocab.concept_to_id]
        
        sources = []
        targets = []
        weights = []
        
        # 1. Connect sequential concepts to form default traversal edges
        for i in range(len(concept_ids) - 1):
            sources.append(concept_ids[i])
            targets.append(concept_ids[i+1])
            weights.append(1.0)
            
            # Bidirectional/backwards with lower weight
            sources.append(concept_ids[i+1])
            targets.append(concept_ids[i])
            weights.append(0.5)
            
        # 2. Add self-loops
        for c_id in concept_ids:
            sources.append(c_id)
            targets.append(c_id)
            weights.append(1.0)
            
        # 3. Add transitions from RAW_DATASET for this domain to ensure paths are topologically valid
        for item in RAW_DATASET:
            if domain in item["active_experts"]:
                for path in item["concept_paths"]:
                    for i in range(len(path) - 1):
                        u = path[i].strip().lower()
                        v = path[i+1].strip().lower()
                        if u in vocab.concept_to_id and v in vocab.concept_to_id:
                            u_id = vocab.concept_to_id[u]
                            v_id = vocab.concept_to_id[v]
                            # Check if edge already exists
                            idx_match = -1
                            for k in range(len(sources)):
                                if sources[k] == u_id and targets[k] == v_id:
                                    idx_match = k
                                    break
                            if idx_match == -1:
                                sources.append(u_id)
                                targets.append(v_id)
                                weights.append(1.5) # Dataset paths get a weight boost
                            else:
                                weights[idx_match] = 1.5 # Boost weight to 1.5 if it exists
                                
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        expert_graphs[domain] = (edge_index, edge_weight)
        
    return expert_graphs


# Scaling testing graph generator
def generate_synthetic_scale_graph(num_concepts: int, density: float = 0.05) -> Tuple[ConceptVocabulary, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Generates synthetic scale graphs for stress testing."""
    # Create fake domains and concepts
    syn_domains = {d: [f"c_{d}_{i}" for i in range(num_concepts // len(DOMAINS))] for d in DOMAINS}
    vocab = ConceptVocabulary(syn_domains)
    
    expert_graphs = {}
    for d in DOMAINS:
        concepts = syn_domains[d]
        concept_ids = [vocab.concept_to_id[c] for c in concepts if c in vocab.concept_to_id]
        
        # Form scale-free or random edges
        sources = []
        targets = []
        weights = []
        
        # Ensure it is at least sequentially connected
        for i in range(len(concept_ids) - 1):
            sources.append(concept_ids[i])
            targets.append(concept_ids[i+1])
            weights.append(1.0)
            
        # Add some random connections to match density
        num_random_edges = int(len(concept_ids) * len(concept_ids) * density)
        num_random_edges = min(num_random_edges, 100000) # cap to prevent memory explosion
        
        for _ in range(num_random_edges):
            u = random.choice(concept_ids)
            v = random.choice(concept_ids)
            if u != v:
                sources.append(u)
                targets.append(v)
                weights.append(random.uniform(0.1, 1.0))
                
        # Self-loops
        for c_id in concept_ids:
            sources.append(c_id)
            targets.append(c_id)
            weights.append(1.0)
            
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        expert_graphs[d] = (edge_index, edge_weight)
        
    return vocab, expert_graphs
