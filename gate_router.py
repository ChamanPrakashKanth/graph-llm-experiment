# gate_router.py
# GATE Mechanical Engineering orchestration: multi-concept routing, prerequisite
# hopping, and LLM-style step-by-step answers — without changing CAT/VLCM architecture.
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from grammar_parser import normalize_query

DATA_DIR = Path(__file__).parent / "data"

# Keyword seeds map natural language to domain concepts for routing.
CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "Euler Buckling": [
        "buckling", "euler", "critical load", "p_cr", "p cr", "slenderness",
        "column", "compressive", "lateral deflection",
    ],
    "Critical Load": ["critical load", "buckling load", "p_cr", "p cr"],
    "Slenderness": ["slenderness", "slender", "long column"],
    "Compressive Load": ["compressive", "compression", "axial load", "axial force"],
    "Boundary Condition": [
        "pinned", "fixed", "free end", "cantilever", "end condition",
        "boundary condition", "effective length",
    ],
    "Effective Length": ["effective length", "l_e", "le ", " k factor", "k-factor"],
    "Stress": ["stress", "sigma", "normal stress", "axial stress"],
    "Load": ["load", "force", "axial load", "applied force"],
    "Strain": ["strain", "epsilon", "deformation"],
    "Yielding": ["yield", "yielding", "yield stress"],
    "Reynolds Number": [
        "reynolds", "re ", "laminar", "turbulent", "flow regime", "pipe flow",
    ],
    "Turbulence": ["turbulent", "turbulence", "eddies", "chaotic flow"],
    "Flow Speed": ["flow speed", "velocity", "flow rate", "speed of flow"],
    "Viscosity": ["viscosity", "dynamic viscosity", "mu", "viscous"],
    "Bending Moment": ["bending moment", "moment", "flexure"],
    "Bending Stress": ["bending stress", "flexural stress"],
    "Torque": ["torque", "torsion", "twisting moment"],
    "Angular Acceleration": ["angular acceleration", "alpha", "rotational accel"],
    "Fourier Law": ["fourier", "conduction", "heat flux", "thermal conductivity"],
    "Temperature Gradient": ["temperature gradient", "gradient", "dT/dx"],
    "Carnot cycle": ["carnot", "efficiency", "heat engine", "reversible cycle"],
    "Convection": ["convection", "nusselt", "heat transfer coefficient"],
    "Newton Second Law": ["newton", "f=ma", "acceleration", "mass"],
    "Euler-Bernoulli Beam": ["beam", "deflection", "neutral axis"],
    "Cayley-Hamilton Theorem": ["cayley", "hamilton", "characteristic equation", "eigenvalue", "matrix power", "trace", "determinant"],
    "Virtual Work Principle": ["virtual work", "displacement", "equilibrium", "active force", "constraint"],
    "Law of Gearing": ["law of gearing", "gear", "angular velocity", "pitch radius", "velocity ratio"],
    "Logarithmic Decrement": ["logarithmic decrement", "damping ratio", "amplitude decay", "decay rate", "damped vibration"],
    "Soderberg Line": ["soderberg", "fatigue", "endurance limit", "yield strength", "mean stress", "stress amplitude", "factor of safety", "fos"],
    "Taylor's Tool Life Equation": ["taylor", "tool life", "cutting speed", "tool wear", "exponent n", "constant c"],
    "Economic Order Quantity (EOQ)": ["economic order quantity", "eoq", "inventory cost", "holding cost", "ordering cost", "annual demand"]
}

# Curated GATE exam multi-hop chains (prerequisite routing templates).
GATE_REASONING_CHAINS: Dict[str, List[str]] = {
    "Euler Buckling": [
        "Compressive Load", "Boundary Condition", "Effective Length",
        "Euler Buckling", "Critical Load",
    ],
    "Reynolds Number": [
        "Flow Speed", "Viscosity", "Reynolds Number", "Turbulence",
    ],
    "Stress": ["Load", "Cross-sectional Area", "Stress"],
    "Hooke's Law": ["Stress", "Strain", "Elastic Deformation"],
    "Bending Stress": [
        "Bending Moment", "Euler-Bernoulli Beam", "Bending Stress",
    ],
    "Torque": ["Torque", "Angular Acceleration", "Moment of Inertia"],
    "Fourier's Law": ["Temperature Gradient", "Fourier Law", "Conduction"],
    "Carnot Efficiency": ["Carnot cycle", "Efficiency"],
    "Newton's Second Law": ["Force", "Acceleration", "Newton Second Law"],
    "Cayley-Hamilton Theorem": ["Characteristic Polynomial", "Matrix Power", "Cayley-Hamilton Theorem"],
    "Virtual Work Principle": ["Static Equilibrium", "Virtual Displacement", "Virtual Work Principle"],
    "Law of Gearing": ["Conjugate Teeth Profile", "Angular Velocity", "Law of Gearing"],
    "Logarithmic Decrement": ["Damping Ratio", "Amplitude Decay", "Logarithmic Decrement"],
    "Soderberg Line": ["Yield Strength", "Endurance Limit", "Mean Stress", "Stress Amplitude", "Soderberg Line"],
    "Taylor's Tool Life Equation": ["Cutting Speed", "Tool Wear", "Taylor's Tool Life Equation"],
    "Economic Order Quantity (EOQ)": ["Holding Cost", "Ordering Cost", "Inventory Cost", "Economic Order Quantity (EOQ)"]
}

# Maps routed concepts to symbolic solver equations (priority order).
CONCEPT_EQUATION_MAP: Dict[str, str] = {
    "Euler Buckling": "Euler Buckling",
    "Critical Load": "Euler Buckling",
    "Slenderness": "Euler Buckling",
    "Buckling": "Euler Buckling",
    "Stress": "Stress",
    "Load": "Stress",
    "Reynolds Number": "Reynolds Number",
    "Turbulence": "Reynolds Number",
    "Flow Speed": "Reynolds Number",
    "Bending Stress": "Bending Stress",
    "Bending Moment": "Bending Stress",
    "Torque": "Torque",
    "Fourier Law": "Fourier's Law",
    "Temperature Gradient": "Fourier's Law",
    "Carnot cycle": "Carnot Efficiency",
    "Efficiency": "Carnot Efficiency",
    "Convection": "Nusselt Number",
    "Newton Second Law": "Newton's Second Law",
    "Elastic Deformation": "Hooke's Law",
    "Strain": "Hooke's Law",
    "Cayley-Hamilton Theorem": "Cayley-Hamilton Theorem",
    "Eigenvalue": "Cayley-Hamilton Theorem",
    "Matrix": "Cayley-Hamilton Theorem",
    "Characteristic Polynomial": "Cayley-Hamilton Theorem",
    "Virtual Work Principle": "Virtual Work Principle",
    "Static Equilibrium": "Virtual Work Principle",
    "Virtual Displacement": "Virtual Work Principle",
    "Law of Gearing": "Law of Gearing",
    "Gear": "Law of Gearing",
    "Velocity Ratio": "Law of Gearing",
    "Logarithmic Decrement": "Logarithmic Decrement",
    "Damping Ratio": "Logarithmic Decrement",
    "Damped Vibration": "Logarithmic Decrement",
    "Soderberg Line": "Soderberg Line",
    "Fatigue Design": "Soderberg Line",
    "Yield Strength": "Soderberg Line",
    "Endurance Limit": "Soderberg Line",
    "Taylor's Tool Life Equation": "Taylor's Tool Life Equation",
    "Tool Wear": "Taylor's Tool Life Equation",
    "Cutting Speed": "Taylor's Tool Life Equation",
    "Economic Order Quantity (EOQ)": "Economic Order Quantity (EOQ)",
    "Inventory Cost": "Economic Order Quantity (EOQ)",
    "Holding Cost": "Economic Order Quantity (EOQ)",
    "Ordering Cost": "Economic Order Quantity (EOQ)"
}


class GateKnowledgeBase:
    """Lazy-loaded GATE mechanical engineering knowledge base."""

    _cache: Optional["GateKnowledgeBase"] = None

    def __init__(self) -> None:
        self.concepts: List[Dict[str, Any]] = self._load("mechanical_concepts.json")
        self.gate_questions: List[Dict[str, Any]] = self._load("mechanical_gate_questions.json")
        self.mcq_bank: List[Dict[str, Any]] = self._load("mechanical_mcq_bank.json")
        self.numerical_bank: List[Dict[str, Any]] = self._load("mechanical_numerical_bank.json")
        self.formula_sheet: List[Dict[str, Any]] = self._load("mechanical_formula_sheet.json")
        self.hierarchy: Dict[str, Any] = self._load("mechanical_hierarchy.json")
        self._concept_index: Dict[str, Dict[str, Any]] = {
            c["concept"]: c for c in self.concepts
        }
        self._prereq_graph: Dict[str, Set[str]] = self._build_prereq_graph()

    @staticmethod
    def _load(filename: str) -> Any:
        path = DATA_DIR / filename
        if not path.exists():
            return [] if filename.endswith(".json") and filename != "mechanical_hierarchy.json" else {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _build_prereq_graph(self) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = {}
        for concept in self.concepts:
            name = concept["concept"]
            neighbors: Set[str] = set()
            for prereq in concept.get("prerequisites", []):
                neighbors.add(prereq)
            for related in concept.get("related_concepts", []):
                neighbors.add(related)
            graph[name] = neighbors
        for chain in GATE_REASONING_CHAINS.values():
            for i in range(len(chain) - 1):
                graph.setdefault(chain[i], set()).add(chain[i + 1])
                graph.setdefault(chain[i + 1], set()).add(chain[i])
        return graph

    @classmethod
    def get(cls) -> "GateKnowledgeBase":
        if cls._cache is None:
            cls._cache = cls()
        return cls._cache


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap_score(a: str, b: str) -> float:
    wa, wb = _tokenize(a), _tokenize(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def seed_concepts_from_text(question: str) -> List[Tuple[str, float]]:
    """Score concepts by keyword overlap with the question."""
    q = question.lower()
    scored: List[Tuple[str, float]] = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if hits:
            scored.append((concept, hits / len(keywords)))
    scored.sort(key=lambda x: -x[1])
    return scored


def hop_concepts(
    seeds: List[str],
    max_hops: int = 5,
) -> List[str]:
    """
    Multi-hop concept routing: traverse prerequisite and related-concept edges,
    then stitch curated GATE reasoning chains.
    """
    kb = GateKnowledgeBase.get()
    visited: List[str] = []
    seen: Set[str] = set()
    frontier = list(seeds)

    for _ in range(max_hops):
        if not frontier:
            break
        next_frontier: List[str] = []
        for concept in frontier:
            if concept in seen:
                continue
            seen.add(concept)
            visited.append(concept)
            for neighbor in kb._prereq_graph.get(concept, set()):
                if neighbor not in seen:
                    next_frontier.append(neighbor)
        frontier = next_frontier

    # Append curated chain tail for the strongest seed match.
    for seed in seeds:
        if seed in GATE_REASONING_CHAINS:
            for step in GATE_REASONING_CHAINS[seed]:
                if step not in seen:
                    visited.append(step)
                    seen.add(step)
            break

    return visited


def route_concepts(
    question: str,
    model_path: Optional[List[str]] = None,
    top_concepts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Fuse neural model path, activation seeds, and rule-based keyword routing
    into a single multi-hop concept path for GATE-style reasoning.
    """
    seeds: List[str] = []
    seed_scores: Dict[str, float] = {}

    for concept, score in seed_concepts_from_text(question):
        seeds.append(concept)
        seed_scores[concept] = score

    if model_path:
        for c in model_path:
            if c not in ("<PAD>", "<EOS>") and c not in seeds:
                seeds.insert(0, c)
                seed_scores[c] = seed_scores.get(c, 0.0) + 0.5

    if top_concepts:
        for entry in top_concepts[:5]:
            name = entry.get("name", "")
            if name and name not in seeds:
                seeds.append(name)
                seed_scores[name] = seed_scores.get(name, 0.0) + entry.get("activation", 0.1)

    concept_path = hop_concepts(seeds)

    # Deduplicate while preserving order; model path prefix takes priority.
    merged: List[str] = []
    seen: Set[str] = set()
    if model_path:
        for c in model_path:
            if c not in ("<PAD>", "<EOS>") and c not in seen:
                merged.append(c)
                seen.add(c)
    for c in concept_path:
        if c not in seen:
            merged.append(c)
            seen.add(c)

    ranked_equations: List[str] = []
    for c in merged:
        eq = CONCEPT_EQUATION_MAP.get(c)
        if eq and eq not in ranked_equations:
            ranked_equations.append(eq)

    return {
        "seed_concepts": seeds[:8],
        "concept_path": merged,
        "ranked_equations": ranked_equations,
        "seed_scores": seed_scores,
    }


def match_gate_bank(question: str) -> Optional[Dict[str, Any]]:
    """Match question against GATE NAT, MCQ, and numerical banks."""
    kb = GateKnowledgeBase.get()
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for entry in kb.gate_questions:
        score = _overlap_score(question, entry["question"])
        if score > best_score:
            best_score = score
            best = {"source": "gate_nat", "match_score": score, **entry}

    for entry in kb.mcq_bank:
        score = _overlap_score(question, entry["question"])
        if score > best_score:
            best_score = score
            best = {"source": "gate_mcq", "match_score": score, **entry}

    for entry in kb.numerical_bank:
        score = _overlap_score(question, entry["question"])
        if score > best_score:
            best_score = score
            best = {"source": "gate_numerical", "match_score": score, **entry}

    if best and best_score >= 0.25:
        return best
    return None


def get_concept_notes(concept_path: List[str]) -> List[Dict[str, str]]:
    """Fetch exam tips and typical mistakes for concepts on the routed path."""
    kb = GateKnowledgeBase.get()
    notes: List[Dict[str, str]] = []
    for name in concept_path:
        entry = kb._concept_index.get(name)
        if entry:
            notes.append({
                "concept": name,
                "exam_summary": entry.get("exam_summary", ""),
                "typical_mistakes": entry.get("typical_mistakes", ""),
                "memory_aid": entry.get("memory_aid", ""),
            })
    return notes


def _validate_nat_answer(solved_value: str, expected: str, tolerance: float = 0.02) -> Dict[str, Any]:
    try:
        got = float(solved_value.replace(",", ""))
        exp = float(str(expected).replace(",", ""))
        rel_err = abs(got - exp) / max(abs(exp), 1e-9)
        return {
            "expected": expected,
            "computed": solved_value,
            "within_tolerance": rel_err <= tolerance,
            "relative_error": round(rel_err, 4),
        }
    except ValueError:
        return {"expected": expected, "computed": solved_value, "within_tolerance": False}


def compose_gate_answer(
    question: str,
    routing: Dict[str, Any],
    solved: Optional[Dict[str, Any]],
    gate_match: Optional[Dict[str, Any]],
    concept_notes: List[Dict[str, str]],
) -> str:
    """Build an LLM-style step-by-step GATE exam answer from routed concepts."""
    parts: List[str] = []

    path = routing.get("concept_path", [])
    if path:
        chain = " → ".join(path[:8])
        parts.append(f"**Concept Routing Path:** {chain}")

    if gate_match:
        if gate_match["source"] == "gate_mcq":
            parts.append(
                f"\n**GATE MCQ Match** (confidence {gate_match['match_score']:.0%})\n"
                f"**Answer: {gate_match['answer']}**\n"
                f"{gate_match.get('explanation', '')}"
            )
        elif gate_match["source"] == "gate_nat":
            steps = gate_match.get("calculation_steps", [])
            if steps:
                parts.append("\n**Reference Solution Steps:**")
                parts.extend(f"  {s}" for s in steps)
            interp = gate_match.get("physical_interpretation", "")
            if interp:
                parts.append(f"\n**Physical Interpretation:** {interp}")
        elif gate_match["source"] == "gate_numerical":
            steps = gate_match.get("steps", [])
            if steps:
                parts.append("\n**Reference Numerical Steps:**")
                parts.extend(f"  {s}" for s in steps)

    if solved:
        calc_steps = solved.get("calculation_steps", [])
        if calc_steps:
            parts.append("\n**Symbolic Solver Steps:**")
            parts.extend(f"  {s}" for s in calc_steps)
        parts.append(
            f"\n**Result:** {solved['solved_variable']} = **{solved['solved_value']}** "
            f"(using {solved['equation']}: $${solved['formula']}$$)"
        )
        validation = solved.get("nat_validation")
        if validation:
            status = "✓ Within GATE tolerance" if validation["within_tolerance"] else "✗ Outside tolerance"
            parts.append(
                f"\n**GATE NAT Check:** {status} "
                f"(expected {validation['expected']}, computed {validation['computed']})"
            )

    exam_tips = [n for n in concept_notes if n.get("exam_summary")]
    if exam_tips:
        parts.append("\n**GATE Exam Tips:**")
        for tip in exam_tips[:3]:
            parts.append(f"  • *{tip['concept']}*: {tip['exam_summary']}")

    mistakes = [n for n in concept_notes if n.get("typical_mistakes")]
    if mistakes:
        parts.append("\n**Common Mistakes to Avoid:**")
        for m in mistakes[:2]:
            parts.append(f"  • *{m['concept']}*: {m['typical_mistakes']}")

    if not parts:
        parts.append(
            "I routed your question through the mechanical engineering concept graph. "
            "Try providing numeric values (e.g. E = 200e9, I = 1e-5, L = 2.0) for a full solution."
        )

    return "\n".join(parts)


def process_gate_query(
    question: str,
    model_path: Optional[List[str]] = None,
    top_concepts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main orchestration entry point for chat_server.
    Routes concepts, solves symbolically, matches GATE banks, composes answer.
    """
    import equations_database

    routing = route_concepts(question, model_path, top_concepts)
    gate_match = match_gate_bank(question)
    solved = equations_database.check_and_solve_chain(
        question,
        concept_path=routing.get("concept_path"),
        ranked_equations=routing.get("ranked_equations"),
    )

    if solved and gate_match and gate_match.get("correct_answer"):
        solved["nat_validation"] = _validate_nat_answer(
            solved["solved_value"], gate_match["correct_answer"]
        )
    elif solved and gate_match and gate_match.get("answer") and gate_match["source"] != "gate_mcq":
        solved["nat_validation"] = _validate_nat_answer(
            solved["solved_value"], gate_match["answer"]
        )

    concept_notes = get_concept_notes(routing.get("concept_path", []))
    composed = compose_gate_answer(question, routing, solved, gate_match, concept_notes)

    relevant = equations_database.lookup_equations_by_concepts(routing.get("concept_path", []))

    return {
        "routing": routing,
        "gate_match": gate_match,
        "solved": solved,
        "concept_notes": concept_notes,
        "composed_answer": composed,
        "relevant_equations": relevant,
        "concept_path": routing.get("concept_path", []),
    }


def get_gate_suggested_questions() -> List[Dict[str, str]]:
    """Return GATE bank questions for chat UI suggestions."""
    kb = GateKnowledgeBase.get()
    suggestions: List[Dict[str, str]] = []
    for entry in kb.gate_questions:
        suggestions.append({
            "q": entry["question"],
            "tag": f"GATE {entry.get('type', 'NAT')}",
        })
    for entry in kb.mcq_bank:
        suggestions.append({
            "q": entry["question"],
            "tag": "GATE MCQ",
        })
    for entry in kb.numerical_bank:
        suggestions.append({
            "q": entry["question"],
            "tag": "GATE Numerical",
        })
    return suggestions
