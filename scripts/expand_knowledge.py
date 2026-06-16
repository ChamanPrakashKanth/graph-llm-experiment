# scripts/expand_knowledge.py
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure we can import from workspace root
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from reasoning_graph import ReasoningGraph, ConceptNode, ConceptEdge

# List of allowed relationship types
ALLOWED_RELATIONS = {
    "depends_on", "requires", "causes", "affects", "increases", "decreases",
    "governs", "controls", "determines", "derived_from", "part_of", "used_in",
    "measured_by", "calculated_by", "prerequisite_for", "related_to",
    "constrained_by", "optimized_by"
}

# Source references
MIT_OCW_SOURCE = "MIT OpenCourseWare (Course 2: Mechanical Engineering)"
TEXTBOOK_SOURCE = "Standard Engineering Textbooks (Shigley, Cengel, Incropera)"

# Contradictions database
CONTRADICTIONS = [
    {
        "concept": "Chvorinov's Rule",
        "parameter": "exponent n",
        "claims": [
            {
                "claim": "The exponent n equals exactly 2.0 based on 1D heat conduction through a mold.",
                "source": "Analytical Heat Transfer Physics (MIT OCW Course 2)",
                "confidence": 0.95
            },
            {
                "claim": "The exponent n ranges from 1.5 to 2.0 empirically due to 3D corner effects, finite mold size, and varying shape factors.",
                "source": "Metal Casting Foundry Practice (Textbooks & Empirical Studies)",
                "confidence": 0.90
            }
        ],
        "status": "marked_for_review"
    }
]

# New concepts data
NEW_CONCEPTS = [
    {
        "concept": "Euler Turbomachinery Equation",
        "definition": "A fundamental equation expressing the relationship between torque/power and the change in angular momentum of fluid in a rotating turbomachine.",
        "prerequisites": ["Angular Momentum Conservation", "Velocity Triangle", "Torque"],
        "difficulty": "Hard",
        "revision_1_line": "Euler's equation relates fluid angular momentum change to rotor work.",
        "revision_5_lines": "Euler turbomachinery equation applies to pumps and turbines.\nIt is based on conservation of angular momentum.\nPower is P = mass_flow * omega * (r2*Vu2 - r1*Vu1).\nVelocity triangles resolve fluid velocities.\nAssumes steady, one-dimensional, and adiabatic flow.",
        "explanation": "The Euler turbomachinery equation determines the energy transfer between a fluid and a rotor. The torque exerted on the rotor equals the rate of change of angular momentum of the fluid. Combining this with rotor angular velocity yields power.",
        "exam_summary": "Highly tested in turbomachinery questions. Be careful with absolute velocity tangential components (Vu1, Vu2) from velocity triangles.",
        "typical_mistakes": "Using relative velocity instead of absolute velocity tangential component, or confusing inlet and outlet radius.",
        "memory_aid": "Euler's turbomachine equation relates torque to delta angular momentum.",
        "related_concepts": ["Velocity Triangle", "Rotor Speed", "Mass Flow Rate"],
        "aliases": ["Euler's pump equation", "Euler turbine equation", "Euler's turbomachinery equation"],
        "synonyms": ["Euler equation for turbomachines"]
    },
    {
        "concept": "Velocity Triangle",
        "definition": "A vector diagram representing the relationship between absolute velocity, relative velocity, and blade velocity of a fluid in turbomachinery.",
        "prerequisites": ["Absolute Velocity", "Relative Velocity", "Blade Speed"],
        "difficulty": "Medium",
        "revision_1_line": "Vector sum V = U + W relates absolute, blade, and relative velocities.",
        "revision_5_lines": "Velocity triangles are drawn at rotor inlet and outlet.\nAbsolute velocity V is vector sum of blade speed U and relative velocity W.\nTangential component Vu drives torque.\nAxial component Vf determines mass flow rate.\nAngles alpha and beta define flow directions.",
        "explanation": "Velocity triangles are crucial for analyzing turbomachinery. They graphically represent fluid velocities in stationary and rotating frames of reference at the inlet and exit of a blade row.",
        "exam_summary": "Drawing velocity triangles correctly is key to solving any turbomachinery problem. Remember Vf = V * sin(alpha) and Vu = V * cos(alpha).",
        "typical_mistakes": "Confusing flow angles (absolute vs relative) or drawing the vector sum in the wrong direction.",
        "memory_aid": "Absolute = Blade + Relative (V = U + W).",
        "related_concepts": ["Absolute Velocity", "Relative Velocity", "Blade Speed", "Euler Turbomachinery Equation"],
        "aliases": ["velocity vector diagram"],
        "synonyms": ["fluid velocity triangle"]
    },
    {
        "concept": "Open Feedwater Heater",
        "definition": "A direct-contact heat exchanger in a regenerative Rankine cycle where extracted steam mixes directly with feedwater.",
        "prerequisites": ["Regenerative Rankine Cycle", "Steam Extraction"],
        "difficulty": "Medium",
        "revision_1_line": "Direct-contact mixing chamber preheats feedwater with turbine bleed steam.",
        "revision_5_lines": "Preheats feedwater to increase boiler inlet temperature.\nDirect contact mixing occurs at constant pressure.\nExit state is assumed to be saturated liquid.\nExtraction fraction y is determined by energy balance.\nReduces required boiler heat input, increasing efficiency.",
        "explanation": "An open feedwater heater (OFWH) mixes steam bled from the turbine with low-temperature condensate. Preheating the feedwater before it enters the boiler reduces fuel consumption and increases overall cycle thermal efficiency.",
        "exam_summary": "Frequently tested in Rankine cycle questions. Apply energy balance: y*h_extracted + (1-y)*h_in = h_out (where h_out is saturated liquid enthalpy).",
        "typical_mistakes": "Forgetting that the exit state is saturated liquid, or using incorrect pump work terms.",
        "memory_aid": "Open heater = direct mixing = saturated liquid exit.",
        "related_concepts": ["Regenerative Rankine Cycle", "Extraction Fraction", "Energy Balance"],
        "aliases": ["OFWH", "direct-contact feedwater heater"],
        "synonyms": ["open feed heater"]
    },
    {
        "concept": "Chvorinov's Rule",
        "definition": "An empirical relationship relating the solidification time of a casting to its volume and surface area.",
        "prerequisites": ["Casting", "Solidification Time", "Casting Modulus"],
        "difficulty": "Easy",
        "revision_1_line": "t = B * (V/A)^n relates casting shape to solidification time.",
        "revision_5_lines": "Total solidification time depends on casting modulus V/A.\nChvorinov's formula is t = B * (V/A)^n.\nExponent n is theoretically 2.0 but empirically 1.5 to 2.0.\nMold constant B depends on metal and mold properties.\nUsed to design risers that solidify after the casting.",
        "explanation": "Chvorinov's Rule predicts solidification time, which is crucial to prevent shrinkage cavities. By ensuring the riser has a higher volume-to-area ratio (modulus) than the casting, the riser solidifies last, feeding molten metal to the casting.",
        "exam_summary": "A favorite topic in manufacturing engineering. Remember to calculate V and A for different shapes (sphere, cylinder, slab).",
        "typical_mistakes": "Using total surface area for a slab instead of heat-dissipating surface area, or forgetting to square the modulus.",
        "memory_aid": "Thicker shapes solidify slower; t is proportional to modulus squared.",
        "related_concepts": ["Solidification Time", "Mold Constant", "Casting Modulus", "Riser Design"],
        "aliases": ["Chvorinov solidification rule"],
        "synonyms": ["Chvorinov's formula"]
    },
    {
        "concept": "Bode Plot",
        "definition": "A frequency response plot representing open-loop magnitude and phase of a system over a logarithmic frequency scale.",
        "prerequisites": ["Frequency Response", "Transfer Function"],
        "difficulty": "Medium",
        "revision_1_line": "Logarithmic plot of system magnitude and phase versus frequency.",
        "revision_5_lines": "Magnitude is plotted in decibels (dB = 20 * log10(gain)).\nPhase is plotted in degrees.\nDetermines gain margin (GM) and phase margin (PM).\nUsed to evaluate closed-loop feedback stability.\nCrossover frequencies are key transition points.",
        "explanation": "Bode plots display system behavior in the frequency domain. They enable engineers to determine stability margins and frequency-domain performance without computing closed-loop poles directly.",
        "exam_summary": "Look for slope changes at corner frequencies: -20 dB/dec per pole, +20 dB/dec per zero.",
        "typical_mistakes": "Confusing gain crossover and phase crossover frequencies when calculating margins.",
        "memory_aid": "Bode plot = stability health check in frequency domain.",
        "related_concepts": ["Gain Margin", "Phase Margin", "Gain Crossover Frequency", "Phase Crossover Frequency", "Closed Loop Stability"],
        "aliases": ["Bode diagram"],
        "synonyms": ["logarithmic frequency response plot"]
    },
    {
        "concept": "Gain Margin",
        "definition": "The amount of open-loop gain increase required at the phase crossover frequency to make the closed-loop system unstable.",
        "prerequisites": ["Bode Plot", "Phase Crossover Frequency"],
        "difficulty": "Medium",
        "revision_1_line": "Reciprocal of open-loop gain at phase crossover frequency.",
        "revision_5_lines": "Measured at phase crossover frequency omega_pc.\nGM_dB = 0 - |G(j*omega_pc)|_dB.\nPositive GM in dB indicates stable system gain headroom.\nNegative GM in dB indicates instability.\nGATE tests both formula and graphical reading.",
        "explanation": "Gain Margin is the safety margin for system gain. It indicates how much the loop gain can be multiplied before the system oscillates.",
        "exam_summary": "Find omega where phase = -180, read gain, GM is negative of that gain in dB.",
        "typical_mistakes": "Using gain crossover frequency instead of phase crossover frequency.",
        "memory_aid": "Gain Margin protects against gain drift.",
        "related_concepts": ["Bode Plot", "Phase Crossover Frequency", "Closed Loop Stability"],
        "aliases": ["gain stability margin"],
        "synonyms": ["GM"]
    },
    {
        "concept": "Phase Margin",
        "definition": "The amount of additional phase lag required at the gain crossover frequency to make the closed-loop system unstable.",
        "prerequisites": ["Bode Plot", "Gain Crossover Frequency"],
        "difficulty": "Medium",
        "revision_1_line": "PM = 180 + phase_angle at gain crossover frequency.",
        "revision_5_lines": "Measured at gain crossover frequency omega_gc.\nPM = 180 + angle(G(j*omega_gc)).\nPositive PM indicates stability.\nNegative PM indicates instability.\nDetermines system damping ratio and overshoot.",
        "explanation": "Phase Margin indicates how much delay or phase lag can be added to the system before it becomes unstable.",
        "exam_summary": "Find omega where gain = 0 dB, read phase angle, add 180.",
        "typical_mistakes": "Using phase angle in radians instead of degrees, or using phase crossover frequency.",
        "memory_aid": "Phase Margin protects against system delay.",
        "related_concepts": ["Bode Plot", "Gain Crossover Frequency", "Closed Loop Stability"],
        "aliases": ["phase stability margin"],
        "synonyms": ["PM"]
    }
]

# New relationships
NEW_EDGES = [
    # Euler Turbomachinery
    ("Velocity Triangle", "Absolute Velocity", "depends_on"),
    ("Velocity Triangle", "Relative Velocity", "depends_on"),
    ("Velocity Triangle", "Blade Speed", "depends_on"),
    ("Euler Turbomachinery Equation", "Angular Momentum Conservation", "derived_from"),
    ("Euler Turbomachinery Equation", "Turbomachinery Power", "governs"),
    ("Turbomachinery Power", "Torque", "measured_by"),
    ("Euler Turbomachinery Equation", "Velocity Triangle", "requires"),
    
    # OFWH
    ("Regenerative Rankine Cycle", "Open Feedwater Heater", "requires"),
    ("Open Feedwater Heater", "Feedwater Preheating", "causes"),
    ("Feedwater Preheating", "Thermal Efficiency", "increases"),
    ("Open Feedwater Heater", "Extraction Fraction", "determines"),
    ("Extraction Fraction", "Energy Balance", "calculated_by"),
    ("Open Feedwater Heater", "Saturated Liquid", "constrained_by"),

    # Chvorinov's Rule
    ("Chvorinov's Rule", "Solidification Time", "governs"),
    ("Solidification Time", "Casting Modulus", "determines"),
    ("Solidification Time", "Mold Constant", "determines"),
    ("Chvorinov's Rule", "Riser Design", "used_in"),
    ("Riser Design", "Shrinkage Cavity", "decreases"),

    # Bode Plot & margins
    ("Bode Plot", "Gain Crossover Frequency", "determines"),
    ("Bode Plot", "Phase Crossover Frequency", "determines"),
    ("Gain Crossover Frequency", "Phase Margin", "determines"),
    ("Phase Crossover Frequency", "Gain Margin", "determines"),
    ("Phase Margin", "Closed Loop Stability", "governs"),
    ("Gain Margin", "Closed Loop Stability", "governs"),
    ("Closed Loop Stability", "Feedback Loop", "optimized_by")
]

# New multi-hop reasoning paths
REASONING_PATHS = [
    {
        "question": "How does rotor speed affect power in a turbomachine?",
        "reasoning_path": ["Rotor Speed", "Blade Speed", "Velocity Triangle", "Euler Turbomachinery Equation", "Turbomachinery Power"],
        "answer": "Rotor speed determines blade speed, which changes the velocity triangle components. This alters the fluid angular momentum change governed by the Euler turbomachinery equation, directly affecting power."
    },
    {
        "question": "Why does preheating feedwater improve Rankine cycle efficiency?",
        "reasoning_path": ["Regenerative Rankine Cycle", "Open Feedwater Heater", "Feedwater Preheating", "Thermal Efficiency"],
        "answer": "A regenerative Rankine cycle uses an open feedwater heater to perform feedwater preheating with bleed steam. This reduces the heat input required by the boiler, which increases thermal efficiency."
    },
    {
        "question": "How does riser design prevent shrinkage defects in casting?",
        "reasoning_path": ["Chvorinov's Rule", "Casting Modulus", "Solidification Time", "Riser Design", "Shrinkage Cavity"],
        "answer": "Chvorinov's rule determines solidification time from the casting modulus. Designing the riser with a higher modulus ensures it has a longer solidification time, feeding molten metal to prevent shrinkage cavities."
    },
    {
        "question": "How does phase margin affect closed loop stability in feedback systems?",
        "reasoning_path": ["Bode Plot", "Gain Crossover Frequency", "Phase Margin", "Closed Loop Stability"],
        "answer": "From the Bode plot of open loop response, the gain crossover frequency is determined. This dictates the phase margin, which directly governs closed loop stability."
    }
]

# Formulas list
FORMULAS = [
    {
        "name": "Euler Turbomachinery Equation",
        "formula": "P = \\dot{m} \\omega (r_2 V_{u2} - r_1 V_{u1})",
        "variables": ["P: Power (W)", "m_dot: Mass flow rate (kg/s)", "omega: Angular velocity (rad/s)", "r: Radius (m)", "V_u: Tangential absolute velocity (m/s)"]
    },
    {
        "name": "Open Feedwater Heater Energy Balance",
        "formula": "y \\cdot h_{extracted} + (1 - y) \\cdot h_{feedwater,in} = h_{out}",
        "variables": ["y: Extraction fraction (dimensionless)", "h_extracted: Steam extraction enthalpy (J/kg)", "h_feedwater,in: Inlet feedwater enthalpy (J/kg)", "h_out: Exit saturated liquid enthalpy (J/kg)"]
    },
    {
        "name": "Chvorinov's Rule",
        "formula": "t = B \\left( \\frac{V}{A} \\right)^n",
        "variables": ["t: Solidification time (s)", "B: Mold constant (s/m^2)", "V: Casting volume (m^3)", "A: Surface area (m^2)", "n: Exponent (typically 2.0)"]
    },
    {
        "name": "Gain Margin",
        "formula": "GM_{dB} = 0 - |G(j\\omega_{pc})|_{dB}",
        "variables": ["GM_dB: Gain margin (dB)", "omega_pc: Phase crossover frequency (rad/s)", "G(j*omega): Open-loop transfer function"]
    },
    {
        "name": "Phase Margin",
        "formula": "PM = 180^\\circ + \\angle G(j\\omega_{gc})",
        "variables": ["PM: Phase margin (degrees)", "omega_gc: Gain crossover frequency (rad/s)", "G(j*omega): Open-loop transfer function"]
    }
]

def search_existing_nodes(graph: ReasoningGraph, term: str) -> Optional[str]:
    """Search graph nodes, aliases, and synonyms for matches to prevent duplicates."""
    term_normalized = term.strip().lower()
    for node_name, node in graph.nodes.items():
        if node_name.lower() == term_normalized:
            return node_name
        aliases = node.metadata.get("aliases", [])
        synonyms = node.metadata.get("synonyms", [])
        for a in aliases:
            if str(a).strip().lower() == term_normalized:
                return node_name
        for s in synonyms:
            if str(s).strip().lower() == term_normalized:
                return node_name
    return None

def main():
    print("=== MECHANICAL ENGINEERING CONCEPT GRAPH EXPANSION ===")
    
    # 1. Load Existing Graph
    graph_path = Path("data/mechanical_engineering_graph.json")
    if not graph_path.exists():
        print(f"Error: {graph_path} not found.")
        return
        
    print(f"Loading existing graph from {graph_path}...")
    graph = ReasoningGraph.load_json(graph_path)
    print(f"Loaded {len(graph.nodes)} nodes and {sum(len(edges) for edges in graph.edges.values())} edges.")

    # 2. Load Existing Concept DB
    concept_db_path = Path("data/mechanical_concepts.json")
    concepts_db = []
    if concept_db_path.exists():
        print(f"Loading existing concept database from {concept_db_path}...")
        with open(concept_db_path, "r", encoding="utf-8") as f:
            concepts_db = json.load(f)
    existing_concept_names = {c["concept"].strip().lower() for c in concepts_db}

    # Tracking lists for output report
    new_nodes_report = []
    updated_nodes_report = []
    new_edges_report = []
    updated_edges_report = []
    evidence_report = []

    # 3. Integrate New Concepts (Nodes)
    print("\nIntegrating concept nodes...")
    for item in NEW_CONCEPTS:
        name = item["concept"]
        # Duplicate check
        matched_name = search_existing_nodes(graph, name)
        
        metadata = {
            "description": item["definition"],
            "difficulty": item["difficulty"],
            "aliases": item.get("aliases", []),
            "synonyms": item.get("synonyms", []),
            "sources": [MIT_OCW_SOURCE, TEXTBOOK_SOURCE]
        }
        
        if matched_name:
            print(f" - [UPDATE] Concept '{name}' already exists in graph as '{matched_name}'. Updating metadata and strengthening confidence.")
            node = graph.nodes[matched_name]
            node.activation = 1.0 # Strengthen confidence
            node.metadata.update(metadata)
            
            updated_nodes_report.append({
                "name": matched_name,
                "previous_activation": node.activation,
                "new_activation": 1.0,
                "metadata_keys_updated": list(metadata.keys())
            })
            
            evidence_report.append({
                "target_type": "node",
                "target_name": matched_name,
                "source": MIT_OCW_SOURCE,
                "action": "strengthened_confidence"
            })
        else:
            print(f" - [CREATE] Concept '{name}' does not exist. Creating new node.")
            graph.add_node(name, activation=1.0, metadata=metadata)
            new_nodes_report.append({
                "name": name,
                "activation": 1.0,
                "metadata": metadata
            })
            evidence_report.append({
                "target_type": "node",
                "target_name": name,
                "source": TEXTBOOK_SOURCE,
                "action": "created_node"
            })
            
        # Update concept DB list
        if name.strip().lower() not in existing_concept_names:
            concepts_db.append({
                "concept": name,
                "definition": item["definition"],
                "prerequisites": item["prerequisites"],
                "difficulty": item["difficulty"],
                "revision_1_line": item["revision_1_line"],
                "revision_5_lines": item["revision_5_lines"],
                "explanation": item["explanation"],
                "exam_summary": item["exam_summary"],
                "typical_mistakes": item["typical_mistakes"],
                "memory_aid": item["memory_aid"],
                "related_concepts": item["related_concepts"]
            })
            existing_concept_names.add(name.strip().lower())

    # 4. Integrate New Relationships (Edges)
    print("\nIntegrating relationships (edges)...")
    for source, target, relation in NEW_EDGES:
        if relation not in ALLOWED_RELATIONS:
            print(f"Warning: Relation '{relation}' not in allowed relations list. Skipping.")
            continue
            
        # Resolve source and target names in case they mapped to existing nodes (or create them if missing)
        src_resolved = search_existing_nodes(graph, source) or source
        tgt_resolved = search_existing_nodes(graph, target) or target
        
        # Ensure they are added as nodes in case they are completely new helper nodes
        if src_resolved not in graph.nodes:
            graph.add_node(src_resolved, activation=0.5, metadata={"sources": [TEXTBOOK_SOURCE]})
            new_nodes_report.append({"name": src_resolved, "activation": 0.5, "metadata": {"sources": [TEXTBOOK_SOURCE]}})
        if tgt_resolved not in graph.nodes:
            graph.add_node(tgt_resolved, activation=0.5, metadata={"sources": [TEXTBOOK_SOURCE]})
            new_nodes_report.append({"name": tgt_resolved, "activation": 0.5, "metadata": {"sources": [TEXTBOOK_SOURCE]}})
            
        # Add or update edge
        has_edge_before = graph.has_edge(src_resolved, tgt_resolved)
        edge = graph.add_edge(src_resolved, tgt_resolved, weight=1.0, relation=relation, metadata={"source": TEXTBOOK_SOURCE})
        
        if has_edge_before:
            print(f" - [UPDATE] Edge '{src_resolved}' -> '{tgt_resolved}' already exists. Incrementing evidence count (count={edge.evidence_count}).")
            updated_edges_report.append({
                "source": src_resolved,
                "target": tgt_resolved,
                "relation": relation,
                "new_evidence_count": edge.evidence_count
            })
            evidence_report.append({
                "target_type": "edge",
                "source_node": src_resolved,
                "target_node": tgt_resolved,
                "source": MIT_OCW_SOURCE,
                "action": "incremented_evidence"
            })
        else:
            print(f" - [CREATE] Edge '{src_resolved}' -> ({relation}) -> '{tgt_resolved}'.")
            new_edges_report.append({
                "source": src_resolved,
                "target": tgt_resolved,
                "relation": relation,
                "weight": 1.0
            })
            evidence_report.append({
                "target_type": "edge",
                "source_node": src_resolved,
                "target_node": tgt_resolved,
                "source": TEXTBOOK_SOURCE,
                "action": "created_edge"
            })

    # 5. Sanity Checks & Quality Rules
    print("\nRunning quality verification checks...")
    isolated_nodes = []
    for node_name in graph.nodes:
        # Check degree (inward and outward)
        has_in = any(node_name in targets for targets in graph.edges.values())
        has_out = len(graph.edges.get(node_name, {})) > 0
        if not has_in and not has_out:
            isolated_nodes.append(node_name)
            
    if isolated_nodes:
        print(f"Warning: Found {len(isolated_nodes)} isolated nodes in graph: {isolated_nodes}")
        # Connect isolated nodes to a general category node to maintain connectivity
        for node in isolated_nodes:
            graph.add_edge("Mechanical Engineering", node, relation="related_to")
            print(f" - Fixed: Connected isolated node '{node}' to 'Mechanical Engineering'")
    else:
        print("Success: No isolated nodes found in the expanded graph.")

    # Validate relation types
    invalid_edges = []
    for src, targets in graph.edges.items():
        for tgt, edge in targets.items():
            if edge.relation not in ALLOWED_RELATIONS and edge.relation not in ("document_cooccurrence", "reasoning_step", "related", "applies_to", "prerequisite_of", "causes"):
                invalid_edges.append((src, tgt, edge.relation))
    if invalid_edges:
        print(f"Warning: Found {len(invalid_edges)} edges with non-standard relation types. (Legacy types preserved: document_cooccurrence, reasoning_step).")

    # 6. Save Updated Files
    print(f"\nSaving expanded graph to {graph_path}...")
    graph.save_json(graph_path)
    
    print(f"Saving updated concept database to {concept_db_path}...")
    with open(concept_db_path, "w", encoding="utf-8") as f:
        json.dump(concepts_db, f, indent=2)

    # 7. Write Results JSON
    report_data = {
        "new_nodes": new_nodes_report,
        "updated_nodes": updated_nodes_report,
        "new_edges": new_edges_report,
        "updated_edges": updated_edges_report,
        "reasoning_paths": REASONING_PATHS,
        "formulas": FORMULAS,
        "contradictions": CONTRADICTIONS,
        "evidence": evidence_report
    }
    
    report_path = Path("reports/expansion_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Expansion report saved successfully to {report_path}!")
    
    # 8. Print Output format for Agent verification
    print("\n=== EXPANSION OUTPUT REPORT (JSON) ===")
    print(json.dumps(report_data, indent=2))

if __name__ == "__main__":
    main()
