# scripts/expand_curriculum.py
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure we can import from workspace root
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from reasoning_graph import ReasoningGraph, ConceptNode, ConceptEdge

# List of allowed relationship types for this phase
ALLOWED_RELATIONS = {
    "causes", "affects", "depends_on", "requires", "increases", "decreases",
    "governs", "controls", "leads_to", "fails_due_to", "derived_from",
    "part_of", "used_in", "prerequisite_for"
}

SOURCE_NAME = "GATE Mechanical Engineering Syllabus Curriculum"

# 13 Syllabus Area Concepts
NEW_CONCEPTS = [
    {
        "concept": "Cayley-Hamilton Theorem",
        "definition": "A mathematical theorem stating that every square matrix satisfies its own characteristic equation.",
        "prerequisites": ["Matrix", "Characteristic Polynomial", "Eigenvalue"],
        "difficulty": "Medium",
        "revision_1_line": "Every square matrix satisfies its own characteristic equation: p(A) = 0.",
        "revision_5_lines": "Cayley-Hamilton theorem applies to square matrices.\nIt states that p(A) = 0 where p(lambda) is the characteristic polynomial.\nUsed to find inverse matrices efficiently.\nUsed to calculate high powers of matrices.\nFundamental in state-space control engineering.",
        "explanation": "The Cayley-Hamilton theorem provides a powerful algebraic method to calculate matrix powers and matrix inverses. By replacing the scalar eigenvalue lambda with the matrix A itself in the characteristic equation, it creates a relation that equates A^n to a linear combination of lower powers.",
        "exam_summary": "Highly tested in GATE Mathematics. Use it to solve for A^-1 or A^9 by reducing powers of A using the characteristic equation.",
        "typical_mistakes": "Forgetting to multiply the constant term by the identity matrix I when substituting A.",
        "memory_aid": "A satisfies its own characteristic equation.",
        "related_concepts": ["Eigenvalue", "Characteristic Equation", "Matrix Power"],
        "aliases": ["Cayley Hamilton theorem"],
        "synonyms": ["Cayley-Hamilton formula"]
    },
    {
        "concept": "Virtual Work Principle",
        "definition": "A principle stating that for a system in static equilibrium under constraints, the virtual work done by external forces for any virtual displacement is zero.",
        "prerequisites": ["Static Equilibrium", "Constraint", "Virtual Displacement"],
        "difficulty": "Hard",
        "revision_1_line": "delta W = sum of F_i * delta r_i = 0 for equilibrium.",
        "revision_5_lines": "Applies to systems in static equilibrium.\nVirtual displacement must be compatible with constraints.\nVirtual work done by active forces is zero.\nEliminates constraint forces from equilibrium equations.\nForms the basis of Lagrangian mechanics.",
        "explanation": "The principle of virtual work simplifies equilibrium analysis of complex constrained systems by focusing only on work-producing active forces, effectively ignoring passive constraint forces like normal reactions at smooth joints.",
        "exam_summary": "Excellent for multi-link mechanism equilibrium questions in GATE. Define virtual coordinate delta theta and equate delta W to zero.",
        "typical_mistakes": "Including constraint forces that do no work, or choosing virtual displacements that violate constraints.",
        "memory_aid": "Virtual work of active forces is zero: delta W = 0.",
        "related_concepts": ["Virtual Displacement", "Generalized Coordinates", "Lagrangian Mechanics"]
    },
    {
        "concept": "Mohr's Circle",
        "definition": "A graphical representation of stress transformation equations, displaying normal and shear stress on any plane.",
        "prerequisites": ["Stress", "Stress Transformation", "Principal Stress"],
        "difficulty": "Medium",
        "revision_1_line": "Graphical representation of plane stress transformation.",
        "revision_5_lines": "Plots normal stress (sigma) on x-axis and shear stress (tau) on y-axis.\nCenter of circle is at (sigma_avg, 0).\nRadius of circle equals maximum shear stress.\nPrincipal stresses are the x-axis intercepts.\nAngle of rotation in Mohr's circle is double the physical plane angle.",
        "explanation": "Mohr's Circle provides a visual method for transforming stress components from one coordinate system to another. It enables quick determination of principal stresses, maximum shear stress, and the orientation of the principal planes.",
        "exam_summary": "One of the most highly tested topics in GATE Strength of Materials. Practice finding the radius R and maximum shear stress tau_max = R.",
        "typical_mistakes": "Forgetting that a plane rotation of theta corresponds to a 2*theta rotation on Mohr's Circle.",
        "memory_aid": "Mohr's circle center = average stress, radius = max shear stress.",
        "related_concepts": ["Principal Stress", "Maximum Shear Stress", "Yield Criteria"]
    },
    {
        "concept": "Law of Gearing",
        "definition": "A condition for gears to transmit a constant angular velocity ratio, stating that the common normal at the point of contact must pass through the pitch point.",
        "prerequisites": ["Gear", "Conjugate Action", "Angular Velocity"],
        "difficulty": "Medium",
        "revision_1_line": "Common normal at contact point must pass through the pitch point.",
        "revision_5_lines": "Ensures constant angular velocity ratio.\nPitch point lies on the line of centers.\nInvolute profiles naturally satisfy the law of gearing.\nPrevents velocity fluctuations and wear.\nEnables smooth torque transmission.",
        "explanation": "The Law of Gearing governs tooth profiles. If the velocity ratio varies, it causes acceleration and deceleration of the gears, leading to vibrations, noise, and tooth failure under load.",
        "exam_summary": "Tested in Theory of Machines. Understand the involute profile geometry and path of contact.",
        "typical_mistakes": "Confusing pitch point with pressure angle or path of contact.",
        "memory_aid": "Normal at contact point always intersects the pitch point.",
        "related_concepts": ["Involute Profile", "Pitch Point", "Velocity Ratio"]
    },
    {
        "concept": "Logarithmic Decrement",
        "definition": "The natural logarithm of the ratio of any two successive amplitudes in a damped free vibration.",
        "prerequisites": ["Damped Vibration", "Damping Ratio", "Amplitude Decay"],
        "difficulty": "Medium",
        "revision_1_line": "delta = ln(x_n / x_{n+1}) = 2 * pi * zeta / sqrt(1 - zeta^2).",
        "revision_5_lines": "Measures the rate of decay of vibration amplitude.\nDirectly related to the damping ratio (zeta).\nFor small damping: delta approx 2 * pi * zeta.\nCalculated from successive peak values.\nUsed to experimentally determine damping in systems.",
        "explanation": "Logarithmic decrement quantifies how rapidly a system dissipates energy during oscillations. It allows engineers to estimate the damping ratio of a system by observing its decay profile over time.",
        "exam_summary": "GATE frequently asks to find the damping ratio zeta from the ratio of amplitudes after n cycles: delta = (1/n) * ln(x_0 / x_n).",
        "typical_mistakes": "Forgetting to divide by the number of cycles n when using non-successive peaks.",
        "memory_aid": "Log decrement = rate of decay of peaks.",
        "related_concepts": ["Damping Ratio", "Damped Natural Frequency", "Vibration Isolation"]
    },
    {
        "concept": "Soderberg Line",
        "definition": "A design limit line on a fatigue diagram connecting the endurance limit on the stress amplitude axis to the yield strength on the mean stress axis.",
        "prerequisites": ["Mean Stress", "Stress Amplitude", "Endurance Limit", "Yield Strength"],
        "difficulty": "Medium",
        "revision_1_line": "Fatigue design line: sigma_a / S_e + sigma_m / S_y = 1/FOS.",
        "revision_5_lines": "Used for design under fluctuating loads.\nMean stress is plotted on x-axis; stress amplitude on y-axis.\nUses yield strength S_y, making it more conservative than Goodman.\nEndurance limit S_e represents fatigue limit.\nProtects against both fatigue failure and static yielding.",
        "explanation": "The Soderberg Line is a conservative fatigue design criterion. Because it connects to yield strength rather than ultimate tensile strength, it ensures the component does not yield even under peak stresses.",
        "exam_summary": "Common in GATE Machine Design. Be prepared to calculate mean stress sigma_m and amplitude stress sigma_a from maximum and minimum stresses.",
        "typical_mistakes": "Using ultimate strength S_ut instead of yield strength S_y in the denominator of the mean stress term.",
        "memory_aid": "Soderberg uses Yield strength S_y (Goodman uses Ultimate S_ut).",
        "related_concepts": ["Goodman Relation", "Mean Stress", "Stress Amplitude", "Fatigue Limit"]
    },
    {
        "concept": "Navier-Stokes Equations",
        "definition": "Partial differential equations expressing conservation of momentum for viscous, incompressible fluid flow.",
        "prerequisites": ["Conservation of Momentum", "Viscous Flow", "Incompressible Flow"],
        "difficulty": "Hard",
        "revision_1_line": "Momentum equations for viscous Newtonian fluid flow.",
        "revision_5_lines": "Expresses Newton's second law for fluid particles.\nIncludes pressure forces, viscous forces, and body forces.\nNonlinear convective term makes analytical solutions difficult.\nForms the mathematical foundation of CFD.\nEuler equations are the inviscid limit.",
        "explanation": "The Navier-Stokes equations model fluid motion. They describe how velocity, pressure, temperature, and density of a moving fluid are related, serving as the governing physics for aerodynamics and fluid engineering.",
        "exam_summary": "GATE Fluid Mechanics often tests simplified 1D forms (e.g. Couette flow or Poiseuille flow) where convective acceleration terms vanish.",
        "typical_mistakes": "Forgetting that pressure gradient is a force term, or using incorrect boundary conditions (like no-slip) for inviscid flow cases.",
        "memory_aid": "Newton's F=ma applied to viscous fluids.",
        "related_concepts": ["Viscous Flow", "Boundary Layer", "Euler Equations"]
    },
    {
        "concept": "Stefan-Boltzmann Law",
        "definition": "A law stating that the total energy radiated per unit surface area of a blackbody per unit time is directly proportional to the fourth power of its absolute temperature.",
        "prerequisites": ["Thermal Radiation", "Blackbody", "Absolute Temperature"],
        "difficulty": "Easy",
        "revision_1_line": "E = sigma * A * T^4 for blackbody radiation.",
        "revision_5_lines": "Governs radiation heat transfer.\nRadiated energy is proportional to T^4.\nStefan-Boltzmann constant sigma = 5.67e-8 W/m^2-K^4.\nReal surfaces include emissivity (epsilon < 1).\nIndependent of any conducting medium.",
        "explanation": "The Stefan-Boltzmann law quantifies heat transfer by electromagnetic waves. Because it depends on the fourth power of temperature, radiation dominates all other heat transfer modes at high temperatures.",
        "exam_summary": "Frequently tested in GATE Heat Transfer. Pay close attention to absolute temperature conversions to Kelvin.",
        "typical_mistakes": "Using Celsius instead of Kelvin, which leads to massive calculation errors due to the T^4 term.",
        "memory_aid": "Radiation scales with absolute temperature to the fourth power.",
        "related_concepts": ["Emissivity", "Radiation Shield", "Cooling Rate"]
    },
    {
        "concept": "Clausius Inequality",
        "definition": "A thermodynamic theorem stating that the cyclic integral of heat transfer divided by absolute temperature is less than or equal to zero for any thermodynamic cycle.",
        "prerequisites": ["Thermodynamic Cycle", "Heat Transfer", "Entropy"],
        "difficulty": "Medium",
        "revision_1_line": "oint(dQ / T) <= 0 for any thermodynamic cycle.",
        "revision_5_lines": "Derived from the Second Law of Thermodynamics.\nCyclic integral equals 0 for reversible cycles.\nCyclic integral is less than 0 for irreversible cycles.\nCyclic integral greater than 0 is thermodynamically impossible.\nUsed to define the property of entropy.",
        "explanation": "The Clausius Inequality provides a mathematical criterion for evaluating thermodynamic cycle feasibility. It shows that irreversibilities (like friction and unrestrained expansion) always generate entropy, leading to negative cyclic integrals.",
        "exam_summary": "A common conceptual and numerical topic in GATE Thermodynamics. Use it to check if a proposed engine cycle is reversible, irreversible, or impossible.",
        "typical_mistakes": "Using heat transfers without correct sign conventions (heat in is positive, heat out is negative) in the summation.",
        "memory_aid": "dQ/T cycle integral is 0 for reversible, < 0 for real/irreversible.",
        "related_concepts": ["Entropy Generation", "Second Law", "Carnot Cycle"]
    },
    {
        "concept": "Taylor's Tool Life Equation",
        "definition": "An empirical relationship relating cutting tool life to cutting speed.",
        "prerequisites": ["Cutting Speed", "Tool Wear", "Machining"],
        "difficulty": "Easy",
        "revision_1_line": "V * T^n = C relates cutting speed V and tool life T.",
        "revision_5_lines": "Exponent n depends on tool material (e.g. HSS: 0.1, Carbide: 0.25).\nConstant C represents cutting speed for a tool life of 1 minute.\nCutting speed is the most critical factor affecting tool life.\nUsed to determine optimum cutting speed for minimum cost.\nHigh speed reduces tool life exponentially.",
        "explanation": "Taylor's Tool Life Equation enables engineers to optimize machining parameters. It quantifies how increasing cutting speed accelerates tool wear and reduces tool life, allowing balancing of machining speed and tool replacement costs.",
        "exam_summary": "Highly tested in GATE Manufacturing. Be prepared to solve for exponent n or constant C given two operating conditions.",
        "typical_mistakes": "Confusing cutting speed units (m/min) with tool life units (minutes), or making algebra errors with exponents.",
        "memory_aid": "V * T^n = C.",
        "related_concepts": ["Tool Wear", "Machining Cost", "Cutting Speed"]
    },
    {
        "concept": "Economic Order Quantity (EOQ)",
        "definition": "The ideal order quantity a company should purchase to minimize total inventory costs, balancing holding and ordering costs.",
        "prerequisites": ["Inventory Demand", "Ordering Cost", "Holding Cost"],
        "difficulty": "Easy",
        "revision_1_line": "Q* = sqrt(2 * D * S / H) minimizes total inventory cost.",
        "revision_5_lines": "Balances ordering costs and inventory holding costs.\nAssumes constant demand and lead time.\nTotal cost curve is flat near the minimum point.\nQ* represents the minimum point of total cost.\nFundamental concept in operations research.",
        "explanation": "The Economic Order Quantity (EOQ) model determines the optimal order size. Ordering larger quantities reduces the frequency (and thus cost) of orders but increases the average inventory held (and thus holding cost). EOQ resolves this tradeoff.",
        "exam_summary": "A standard topic in GATE Industrial Engineering. Practice calculating total cost TC = D*C + (D/Q)*S + (Q/2)*H.",
        "typical_mistakes": "Using monthly demand instead of annual demand D, or mixing holding cost units.",
        "memory_aid": "EOQ = sqrt(2 * Demand * Ordering_Cost / Holding_Cost).",
        "related_concepts": ["Inventory Cost", "Ordering Cost", "Holding Cost"]
    },
    {
        "concept": "Iron-Carbon Phase Diagram",
        "definition": "A graphical representation of the phases present in steel and cast iron at various temperatures and carbon compositions under equilibrium.",
        "prerequisites": ["Carbon Composition", "Phase Transformation", "Steel"],
        "difficulty": "Medium",
        "revision_1_line": "Maps temperature and carbon content to equilibrium steel phases.",
        "revision_5_lines": "Shows phase fields: ferrite, austenite, cementite, pearlite.\nEutectoid point occurs at 727 deg C and 0.76% carbon.\nEutectic point occurs at 1147 deg C and 4.3% carbon.\nDetermines phase transformations during slow cooling.\nForms basis of steel heat treatment design.",
        "explanation": "The Iron-Carbon Phase Diagram is the map for steel metallurgy. It shows which phases are stable at different temperatures and carbon concentrations, enabling design of heat treatments like annealing and normalizing.",
        "exam_summary": "GATE Materials Science frequently tests invariant reactions (Eutectoid, Eutectic, Peritectic) and phase calculations using the lever rule.",
        "typical_mistakes": "Applying the lever rule in the wrong direction or using incorrect temperature limits for phases.",
        "memory_aid": " austenite transforms to pearlite at eutectoid point.",
        "related_concepts": ["Austenite", "Ferrite", "Phase Transformation", "Heat Treatment"]
    },
    {
        "concept": "Routh-Hurwitz Criterion",
        "definition": "A mathematical test that determines whether all poles of a system's transfer function lie in the open left-half of the complex s-plane, ensuring absolute stability.",
        "prerequisites": ["Characteristic Equation", "Transfer Function", "Poles"],
        "difficulty": "Medium",
        "revision_1_line": "Checks closed-loop stability using characteristic equation coefficients.",
        "revision_5_lines": "Requires constructing a Routh array.\nNo sign changes in the first column indicates stability.\nNumber of sign changes equals the number of right-half plane poles.\nUsed to find range of control gains for stability.\nBypasses the need to solve for roots explicitly.",
        "explanation": "The Routh-Hurwitz Criterion allows stability analysis of high-order systems without factoring polynomials. It evaluates the coefficients of the characteristic equation to guarantee all closed-loop poles are in the stable left-half plane.",
        "exam_summary": "Extremely common in GATE Control Systems. Solve for parameter ranges (e.g. K) that keep the first column of the Routh array strictly positive.",
        "typical_mistakes": "Sign errors during array row calculations, or missing terms in the characteristic equation.",
        "memory_aid": "No sign changes in Routh first column = Stable system.",
        "related_concepts": ["Closed Loop Stability", "Transfer Function", "Characteristic Equation"]
    }
]

# New relationships for GATE curriculum (strictly using allowed relations)
NEW_EDGES = [
    # Cayley-Hamilton
    ("Cayley-Hamilton Theorem", "Characteristic Polynomial", "depends_on"),
    ("Cayley-Hamilton Theorem", "Matrix Power", "governs"),
    ("Matrix Power", "Systems of ODEs", "used_in"),

    # Virtual Work
    ("Virtual Work Principle", "Lagrangian Mechanics", "prerequisite_for"),
    ("Virtual Work Principle", "Static Equilibrium", "governs"),
    ("Virtual Work Principle", "Structural Analysis", "used_in"),

    # Mohr's Circle
    ("Mohr's Circle", "Stress Transformation", "governs"),
    ("Stress Transformation", "Principal Stress", "leads_to"),
    ("Principal Stress", "Yield Criteria", "prerequisite_for"),

    # Law of Gearing
    ("Law of Gearing", "Angular Velocity", "controls"),
    ("Angular Velocity", "Torque Transfer", "affects"),
    ("Law of Gearing", "Conjugate Teeth Profile", "requires"),

    # Logarithmic Decrement
    ("Logarithmic Decrement", "Damping Ratio", "depends_on"),
    ("Damping Ratio", "Amplitude Decay", "controls"),
    ("Amplitude Decay", "Vibration Mitigation", "leads_to"),

    # Soderberg Line
    ("Soderberg Line", "Fatigue Design", "governs"),
    ("Fatigue Design", "Cyclic Loading Failure", "controls"),
    ("Soderberg Line", "Yield Strength", "depends_on"),

    # Navier-Stokes
    ("Navier-Stokes Equations", "Viscous Flow", "governs"),
    ("Viscous Flow", "Boundary Layer Development", "leads_to"),
    ("Boundary Layer Development", "Skin Friction Drag", "causes"),

    # Stefan-Boltzmann
    ("Stefan-Boltzmann Law", "Thermal Radiation", "governs"),
    ("Thermal Radiation", "Heat Flux", "increases"),
    ("Heat Flux", "Cooling Rate", "leads_to"),

    # Clausius Inequality
    ("Clausius Inequality", "Cycle Reversibility", "governs"),
    ("Cycle Reversibility", "Thermal Efficiency", "affects"),
    ("Clausius Inequality", "Entropy Generation", "leads_to"),

    # Taylor's Tool Life
    ("Taylor's Tool Life Equation", "Tool Wear", "governs"),
    ("Tool Wear", "Tool Failure", "leads_to"),
    ("Cutting Speed", "Tool Wear", "increases"),

    # EOQ
    ("Economic Order Quantity (EOQ)", "Inventory Cost", "controls"),
    ("Inventory Cost", "Holding Cost", "depends_on"),
    ("Inventory Cost", "Ordering Cost", "depends_on"),

    # Iron-Carbon Phase
    ("Iron-Carbon Phase Diagram", "Phase Transformation", "governs"),
    ("Phase Transformation", "Microstructure", "affects"),
    ("Microstructure", "Mechanical Properties", "controls"),

    # Routh-Hurwitz
    ("Routh-Hurwitz Criterion", "Closed Loop Stability", "governs"),
    ("Closed Loop Stability", "Left Half Plane Poles", "requires")
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
    print("=== GATE MECHANICAL ENGINEERING SYLLABUS EXPANSION ===")
    
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
        matched_name = search_existing_nodes(graph, name)
        
        metadata = {
            "description": item["definition"],
            "difficulty": item["difficulty"],
            "aliases": item.get("aliases", []),
            "synonyms": item.get("synonyms", []),
            "sources": [SOURCE_NAME]
        }
        
        if matched_name:
            print(f" - [UPDATE] Concept '{name}' already exists in graph as '{matched_name}'. Updating metadata.")
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
                "source": SOURCE_NAME,
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
                "source": SOURCE_NAME,
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
            
        src_resolved = search_existing_nodes(graph, source) or source
        tgt_resolved = search_existing_nodes(graph, target) or target
        
        # Ensure they are added as nodes if missing
        if src_resolved not in graph.nodes:
            graph.add_node(src_resolved, activation=0.5, metadata={"sources": [SOURCE_NAME]})
            new_nodes_report.append({"name": src_resolved, "activation": 0.5, "metadata": {"sources": [SOURCE_NAME]}})
        if tgt_resolved not in graph.nodes:
            graph.add_node(tgt_resolved, activation=0.5, metadata={"sources": [SOURCE_NAME]})
            new_nodes_report.append({"name": tgt_resolved, "activation": 0.5, "metadata": {"sources": [SOURCE_NAME]}})
            
        # Add or update edge
        has_edge_before = graph.has_edge(src_resolved, tgt_resolved)
        edge = graph.add_edge(src_resolved, tgt_resolved, weight=1.0, relation=relation, metadata={"source": SOURCE_NAME})
        
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
                "source": SOURCE_NAME,
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
                "source": SOURCE_NAME,
                "action": "created_edge"
            })

    # 5. Sanity Checks & Quality Rules
    print("\nRunning quality verification checks...")
    isolated_nodes = []
    for node_name in graph.nodes:
        has_in = any(node_name in targets for targets in graph.edges.values())
        has_out = len(graph.edges.get(node_name, {})) > 0
        if not has_in and not has_out:
            isolated_nodes.append(node_name)
            
    if isolated_nodes:
        print(f"Warning: Found {len(isolated_nodes)} isolated nodes in graph: {isolated_nodes}")
        for node in isolated_nodes:
            graph.add_edge("Mechanical Engineering", node, relation="part_of")
            print(f" - Fixed: Connected isolated node '{node}' to 'Mechanical Engineering'")
    else:
        print("Success: No isolated nodes found in the expanded graph.")

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
        "evidence": evidence_report
    }
    
    report_path = Path("reports/curriculum_expansion_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Expansion report saved successfully to {report_path}!")

if __name__ == "__main__":
    main()
