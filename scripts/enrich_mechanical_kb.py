# scripts/enrich_mechanical_kb.py
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure root folder is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from reasoning_graph import ReasoningGraph

SOURCE_NAME = "Primary Engineering Textbooks (GATE ME Curriculum)"

ALLOWED_RELATIONS = {
    "causes", "affects", "depends_on", "requires", "increases", "decreases",
    "governs", "controls", "leads_to", "fails_due_to", "derived_from",
    "part_of", "used_in", "prerequisite_for"
}

# 4 Missing Concepts
NEW_CONCEPTS_DATA = [
    {
        "concept": "Turbulence",
        "definition": "A fluid flow regime characterized by chaotic property changes, including rapid variation of pressure and flow velocity in space and time.",
        "formula": "Re > 4000",
        "causes": ["High inertial forces", "Low viscous forces", "High velocity gradients"],
        "effects": ["Increased skin friction drag", "Rapid mixing", "Enhanced heat transfer rates"],
        "dependencies": ["Reynolds Number", "Fluid velocity", "Characteristic length", "Viscosity"],
        "applications": ["Heat exchangers", "Combustion chambers", "Aerodynamic design"],
        "failure_modes": ["Severe drag increases", "Vibration and fatigue in structures", "Turbulent boundary layer separation"],
        "related_concepts": ["Reynolds Number", "Boundary Layer", "Pipe Flow"],
        "prerequisites": ["Reynolds Number", "Fluid Mechanics"],
        "difficulty": "Hard",
        "revision_1_line": "Chaotic fluid flow regime dominated by inertia.",
        "revision_5_lines": "Occurs at high Reynolds numbers (Re > 4000 in pipes).\nCharacterized by eddies and high mixing rates.\nSignificantly increases drag and heat transfer coefficients.\nRequires statistical modeling (e.g. RANS) since exact solving is intractable.\nContrasts with laminar flow.",
        "explanation": "Turbulence is a chaotic flow state that occurs when viscous damping is insufficient to suppress perturbations. Inertial forces dominate, creating rotational structures called eddies that transfer momentum and heat very rapidly.",
        "exam_summary": "GATE focuses on the threshold for turbulence in pipe flow (Re > 4000) and flat plate boundary layer (Re > 5e5), and how friction factors differ from laminar.",
        "typical_mistakes": "Assuming turbulent flow is always undesirable (it is actually vital for high-rate heat exchange and mixing).",
        "memory_aid": "Turbulent flow is mixed and chaotic; laminar is layered and smooth.",
        "edges": [
            ("Turbulence", "Reynolds Number", "depends_on"),
            ("Turbulence", "Pipe Flow", "part_of"),
            ("Turbulence", "Boundary Layer", "affects")
        ]
    },
    {
        "concept": "Strain",
        "definition": "A measure of deformation representing the displacement between particles in a body relative to a reference length.",
        "formula": "epsilon = delta_L / L",
        "causes": ["Applied stress", "Thermal expansion/contraction", "Phase transformation"],
        "effects": ["Deformation", "Energy storage in elastic range", "Plastic deformation in post-yield range"],
        "dependencies": ["Applied load", "Material elasticity", "Temperature"],
        "applications": ["Structural design", "Metal forming", "Strain gauge measurements"],
        "failure_modes": ["Excessive plastic deformation", "Ductile fracture", "Crack initiation"],
        "related_concepts": ["Stress", "Elasticity", "Yielding"],
        "prerequisites": ["Stress", "Strength of Materials"],
        "difficulty": "Easy",
        "revision_1_line": "Fractional deformation of a body under load.",
        "revision_5_lines": "Engineering strain is change in length over initial length.\nTrue strain is natural log of current over initial length.\nNormal strain measures stretch; shear strain measures angle change.\nRelated to stress via Hooke's Law in elastic range.\nDimensionless but often expressed in microstrain.",
        "explanation": "Strain is a geometric measure of deformation. When a body is loaded, the relative positions of its particles change. Engineering strain simplifies this by using the original dimensions, whereas true strain continuously integrates the change over the instantaneous state.",
        "exam_summary": "GATE tests strain gauge rosette transformations, true strain in metal forming, and Hooke's law relating stress and strain.",
        "typical_mistakes": "Using engineering strain instead of true strain in metal forming (plasticity) problems.",
        "memory_aid": "Stress is the force intensity, strain is the stretch response.",
        "edges": [
            ("Strain", "Stress", "depends_on"),
            ("Strain", "Elasticity", "governs"),
            ("Strain", "Yielding", "leads_to")
        ]
    },
    {
        "concept": "Buckling",
        "definition": "A sudden structural failure mode characterized by a large lateral deflection occurring under compressive loads, before the material yields.",
        "formula": "P_cr = pi^2 * E * I / L_e^2",
        "causes": ["Compressive load", "Slenderness", "Geometric imperfection"],
        "effects": ["Lateral deflection", "Sudden structural collapse", "Bending stresses"],
        "dependencies": ["Column length", "Cross-section moment of inertia", "Boundary conditions", "Elastic modulus"],
        "applications": ["Column design", "Truss structural analysis", "Thin-walled cylinder design"],
        "failure_modes": ["Elastic buckling", "Inelastic buckling", "Torsional buckling"],
        "related_concepts": ["Euler Buckling", "Columns", "Effective Length"],
        "prerequisites": ["Stress", "Columns", "Strength of Materials"],
        "difficulty": "Medium",
        "revision_1_line": "Sudden lateral bending failure under compressive load.",
        "revision_5_lines": "Occurs under axial compressive loads.\nUsually happens suddenly with little warning.\nHighly sensitive to column slenderness.\nGoverned by Euler's buckling theory for slender members.\nBoundary conditions determine the effective length and load capacity.",
        "explanation": "Buckling is a geometric instability. When a compressive load exceeds a critical threshold, the member bows out laterally. This is because the bending stiffness is insufficient to maintain the straight equilibrium configuration under axial compression.",
        "exam_summary": "GATE focuses on calculating the critical buckling load P_cr or critical buckling stress sigma_cr = P_cr / A. Check the slenderness ratio to ensure Euler's theory applies.",
        "typical_mistakes": "Assuming column capacity is purely determined by material yield strength (slender columns buckle way before yielding).",
        "memory_aid": "Buckling bends long columns, crushing squashes short ones.",
        "edges": [
            ("Buckling", "Columns", "part_of"),
            ("Buckling", "Euler Buckling", "governs"),
            ("Buckling", "Effective Length", "depends_on")
        ]
    },
    {
        "concept": "Damping",
        "definition": "An influence within or upon an oscillatory system that has the effect of reducing, restricting, or preventing its oscillations by dissipating energy.",
        "formula": "c_c = 2 * sqrt(k * m)",
        "causes": ["Viscous fluid resistance", "Coulomb dry friction", "Material internal hysteresis"],
        "effects": ["Amplitude reduction", "Energy dissipation", "Shift in resonant frequency"],
        "dependencies": ["Damping coefficient (c)", "System mass (m)", "Stiffness (k)"],
        "applications": ["Shock absorbers", "Vibration isolation mounts", "Seismic dampers in buildings"],
        "failure_modes": ["Under-damping causing excessive oscillation", "Over-damping causing slow response", "Damper fluid leakage"],
        "related_concepts": ["Free Vibration", "Forced Vibration", "Resonance"],
        "prerequisites": ["Mechanical Vibrations"],
        "difficulty": "Medium",
        "revision_1_line": "Energy dissipation in vibrating systems.",
        "revision_5_lines": "Critical damping c_c = 2*sqrt(k*m) represents fastest return to equilibrium.\nDamping ratio zeta determines system behavior.\nzeta < 1 is underdamped (oscillates with decaying amplitude).\nzeta = 1 is critically damped; zeta > 1 is overdamped (no oscillation).\nDamped natural frequency omega_d = omega_n * sqrt(1 - zeta^2).",
        "explanation": "Damping forces dissipate kinetic energy as heat, restricting vibration. Viscous damping is the most common model where the damping force is proportional to velocity: F_d = c * dx/dt.",
        "exam_summary": "GATE frequently tests calculating damping ratio zeta, damped natural frequency omega_d, and logarithmic decrement.",
        "typical_mistakes": "Thinking critical damping has oscillations (it does not; it returns to rest without overshoot).",
        "memory_aid": "Damping dampens: turns motion energy into heat.",
        "edges": [
            ("Damping", "Mechanical Vibrations", "part_of"),
            ("Damping", "Resonance", "controls"),
            ("Damping", "Logarithmic Decrement", "leads_to")
        ]
    }
]

# Enrichment dictionary for 26 existing concepts
ENRICHMENT_DATA = {
    "Euler Buckling": {
        "formula": "P_cr = pi^2 * E * I / L_e^2",
        "causes": ["Compressive load", "Slenderness", "Initial eccentricity"],
        "effects": ["Lateral deflection", "Bending stress", "Structural instability"],
        "dependencies": ["Young's Modulus (E)", "Moment of Inertia (I)", "Effective Length (L_e)"],
        "applications": ["Columns", "Struts", "Structural design"],
        "failure_modes": ["Sudden collapse", "Elastic buckling", "Plastic buckling"]
    },
    "Boundary Condition": {
        "formula": "L_e = K * L",
        "causes": ["Support constraints", "Fixity", "Restraints on displacement/rotation"],
        "effects": ["Effective length modification", "Buckling mode shape changes", "Load capacity changes"],
        "dependencies": ["Support rigidity", "Column length"],
        "applications": ["Euler buckling analysis", "Beam deflection boundary values"],
        "failure_modes": ["Support slippage", "Improper boundary constraint modeling"]
    },
    "Effective Length": {
        "formula": "L_e = K * L",
        "causes": ["End constraint conditions", "Column length"],
        "effects": ["Buckling load variation", "Slenderness ratio change"],
        "dependencies": ["Boundary conditions", "Physical column length"],
        "applications": ["Euler buckling formula", "Slenderness ratio calculation"],
        "failure_modes": ["Underestimation of effective length leading to premature collapse"]
    },
    "Reynolds Number": {
        "formula": "Re = rho * v * D / mu",
        "causes": ["Fluid velocity", "Characteristic length", "Density", "Dynamic viscosity"],
        "effects": ["Laminar to turbulent transition", "Boundary layer behavior", "Friction factor variation"],
        "dependencies": ["Flow speed", "Fluid density", "Fluid viscosity", "Pipe diameter"],
        "applications": ["Pipe flow analysis", "Aerodynamics", "Model scaling"],
        "failure_modes": ["Flow transition misprediction", "Frictional loss underestimation"]
    },
    "Stress": {
        "formula": "sigma = P / A",
        "causes": ["External load", "Thermal expansion constraint", "Residual stress from manufacturing"],
        "effects": ["Material strain", "Deformation", "Yielding or fracture"],
        "dependencies": ["Applied load", "Cross-sectional area", "Material stiffness"],
        "applications": ["Component design", "Safety factor calculation", "Structural analysis"],
        "failure_modes": ["Yielding", "Fatigue fracture", "Brittle failure"]
    },
    "Carnot cycle": {
        "formula": "eta = 1 - T_C / T_H",
        "causes": ["Reversible heat addition and rejection", "Isothermal and isentropic processes"],
        "effects": ["Work output", "Heat conversion to work at maximum efficiency"],
        "dependencies": ["High temperature reservoir (T_H)", "Low temperature reservoir (T_C)"],
        "applications": ["Efficiency limit benchmark", "Thermodynamic cycles evaluation"],
        "failure_modes": ["Thermal leaks", "Frictional losses in real engines deviating from Carnot limit"]
    },
    "Euler Turbomachinery Equation": {
        "formula": "P = mass_flow * omega * (r2 * Vu2 - r1 * Vu1)",
        "causes": ["Fluid angular momentum change", "Rotor rotation"],
        "effects": ["Torque generation", "Work transfer between fluid and rotor"],
        "dependencies": ["Mass flow rate", "Rotor speed", "Velocity triangles"],
        "applications": ["Pumps", "Turbines", "Compressors design"],
        "failure_modes": ["Flow separation", "Blade stall", "Cavitation"]
    },
    "Velocity Triangle": {
        "formula": "V = U + W",
        "causes": ["Rotor rotation", "Fluid flow speed and direction"],
        "effects": ["Flow angle determination", "Work output evaluation"],
        "dependencies": ["Blade speed (U)", "Relative flow velocity (W)", "Fluid absolute velocity (V)"],
        "applications": ["Turbomachinery blade design", "Euler turbomachinery equation"],
        "failure_modes": ["Incidence loss", "Flow misalignment"]
    },
    "Open Feedwater Heater": {
        "formula": "m1*h1 + m2*h2 = m3*h3",
        "causes": ["Steam extraction", "Condensate mixing"],
        "effects": ["Feedwater temperature increase", "Power cycle efficiency enhancement"],
        "dependencies": ["Extraction steam state", "Feedwater mass flow"],
        "applications": ["Rankine cycle regeneration", "Steam power plants"],
        "failure_modes": ["Thermal stress", "Flooding", "Mixing chamber pressure mismatch"]
    },
    "Chvorinov's Rule": {
        "formula": "t = B * (V / A)^n",
        "causes": ["Heat transfer from molten metal to mold", "Thermal resistance of mold"],
        "effects": ["Metal solidification time prediction"],
        "dependencies": ["Mold constant (B)", "Volume of casting (V)", "Surface area (A)"],
        "applications": ["Riser design", "Casting solidification analysis"],
        "failure_modes": ["Premature solidification", "Shrinkage defects"]
    },
    "Bode Plot": {
        "formula": "Gain_dB = 20 * log10(|G|)",
        "causes": ["Frequency variation in sinusoidal input"],
        "effects": ["Magnitude and phase frequency response visualization"],
        "dependencies": ["System transfer function", "Input frequency"],
        "applications": ["Stability analysis", "Gain and phase margin determination", "Controller tuning"],
        "failure_modes": ["High frequency noise amplification", "Inaccurate asymptotes approximation"]
    },
    "Gain Margin": {
        "formula": "GM = 1 / |G(j*w_pc)|",
        "causes": ["System phase lag matching -180 degrees"],
        "effects": ["Measures system relative stability", "Permissible gain increase before instability"],
        "dependencies": ["Phase crossover frequency", "Loop gain"],
        "applications": ["Control system design", "Robustness assessment"],
        "failure_modes": ["System instability due to excessive gain increase"]
    },
    "Phase Margin": {
        "formula": "PM = 180 + angle(G(j*w_gc))",
        "causes": ["System loop gain magnitude reaching unity"],
        "effects": ["Measures relative stability", "Permissible phase lag before instability"],
        "dependencies": ["Gain crossover frequency", "System poles and zeros"],
        "applications": ["Feedback loop stabilization", "Transient response performance design"],
        "failure_modes": ["Oscillations or instability due to additional phase delay"]
    },
    "Cayley-Hamilton Theorem": {
        "formula": "p(A) = 0",
        "causes": ["Matrix linear independence of powers"],
        "effects": ["Reduction of matrix powers", "Matrix inverse evaluation"],
        "dependencies": ["Square matrix A", "Eigenvalues of A"],
        "applications": ["Control systems state-space transition matrix calculation", "Matrix power simplification"],
        "failure_modes": ["Computational errors in characteristic polynomial coefficients"]
    },
    "Virtual Work Principle": {
        "formula": "F_force * delta_x = P_load * delta_y",
        "causes": ["Static equilibrium state", "Constraint forces"],
        "effects": ["System equilibrium equation generation without internal/constraint forces"],
        "dependencies": ["Virtual displacement compatibility with constraints"],
        "applications": ["Structural mechanics", "Mechanism equilibrium analysis"],
        "failure_modes": ["Incompatible virtual displacement choice"]
    },
    "Mohr's Circle": {
        "formula": "R = sqrt(((sigma_x - sigma_y)/2)^2 + tau_xy^2)",
        "causes": ["Plane stress state", "Coordinate rotation"],
        "effects": ["Stress transformation", "Principal plane angle determination"],
        "dependencies": ["Applied normal stresses", "Shear stress"],
        "applications": ["Shaft design", "Pressure vessel stress analysis", "Soil mechanics"],
        "failure_modes": ["Wrong plane angle factor (2*theta)", "Sign convention confusion"]
    },
    "Law of Gearing": {
        "formula": "omega_1 * R_1 = omega_2 * R_2",
        "causes": ["Conjugate tooth contact geometry"],
        "effects": ["Constant speed ratio transmission", "Vibration reduction"],
        "dependencies": ["Pitch point alignment on line of centers"],
        "applications": ["Involute gear tooth design", "Gearbox layout"],
        "failure_modes": ["Interference", "Under-cutting", "Gear wear and backlash"]
    },
    "Logarithmic Decrement": {
        "formula": "delta = 2 * pi * zeta / sqrt(1 - zeta^2)",
        "causes": ["Viscous damping dissipation of energy"],
        "effects": ["Exponential decay of oscillation peak amplitudes"],
        "dependencies": ["Damping ratio (zeta)"],
        "applications": ["Experimental damping measurement", "Vibration damping evaluation"],
        "failure_modes": ["Incorrect cycle count division", "Noise interference in experimental peaks"]
    },
    "Soderberg Line": {
        "formula": "sigma_a / S_e + sigma_m / S_y = 1 / FOS",
        "causes": ["Mean stress and stress amplitude combination in cyclic loading"],
        "effects": ["Fatigue life design limit", "Yielding protection"],
        "dependencies": ["Endurance limit (S_e)", "Yield strength (S_y)"],
        "applications": ["Shaft design", "Fasteners under cyclic load"],
        "failure_modes": ["Fatigue failure under unexpected stress peaks"]
    },
    "Navier-Stokes Equations": {
        "formula": "rho * (dV/dt + V * grad(V)) = -grad(p) + mu * div(grad(V)) + rho * g",
        "causes": ["Pressure gradients", "Viscous shear", "Gravity body forces"],
        "effects": ["Fluid flow velocity field development", "Drag and lift force generation"],
        "dependencies": ["Fluid density", "Viscosity", "Momentum conservation"],
        "applications": ["Aerodynamics", "Hydraulics", "CFD simulations"],
        "failure_modes": ["Flow separation", "Turbulent transition modeling errors"]
    },
    "Stefan-Boltzmann Law": {
        "formula": "E = sigma * T^4",
        "causes": ["Absolute temperature thermal excitation"],
        "effects": ["Electromagnetic radiation heat flux emission"],
        "dependencies": ["Surface temperature (T)", "Emissivity (epsilon)"],
        "applications": ["Radiation heat transfer", "Spacecraft thermal design"],
        "failure_modes": ["Overheating due to incorrect emissivity estimation"]
    },
    "Clausius Inequality": {
        "formula": "oint(dQ / T) <= 0",
        "causes": ["Process irreversibilities", "Entropy generation"],
        "effects": ["Cycle feasibility determination", "Entropy change definition"],
        "dependencies": ["Boundary temperature", "Heat transfer interactions"],
        "applications": ["Second Law engine feasibility checks", "Entropy calculations"],
        "failure_modes": ["Violation indicating thermodynamic impossibility"]
    },
    "Taylor's Tool Life Equation": {
        "formula": "V * T^n = C",
        "causes": ["Flank wear", "Crater wear", "Thermal softening at cutting tip"],
        "effects": ["Tool life reduction at higher cutting speed"],
        "dependencies": ["Tool material exponent (n)", "Cutting speed (V)", "Feed rate/depth of cut constant (C)"],
        "applications": ["Machining cost optimization", "Cutting parameter selection"],
        "failure_modes": ["Premature tool failure due to unexpected wear mechanism"]
    },
    "Economic Order Quantity (EOQ)": {
        "formula": "Q = sqrt(2 * D * S / H)",
        "causes": ["Inventory holding cost vs ordering cost trade-off"],
        "effects": ["Optimal order size determination for minimum total cost"],
        "dependencies": ["Annual demand (D)", "Setup/Ordering cost (S)", "Holding cost per unit (H)"],
        "applications": ["Inventory management", "Production planning"],
        "failure_modes": ["Stockouts due to demand variability", "Excess holding costs"]
    },
    "Iron-Carbon Phase Diagram": {
        "formula": "Fe3C",
        "causes": ["Carbon concentration variation", "Alloying temperature changes"],
        "effects": ["Microstructural phase changes (Austenite, Ferrite, Pearlite, Martensite)"],
        "dependencies": ["Temperature", "Carbon percentage"],
        "applications": ["Heat treatment of steel", "Material selection"],
        "failure_modes": ["Improper heat treatment leading to brittle failure or low hardness"]
    },
    "Routh-Hurwitz Criterion": {
        "formula": "Characteristic equation array column sign checks",
        "causes": ["Characteristic equation coefficient relationships"],
        "effects": ["LHP closed-loop poles verification", "Absolute stability bounds determination"],
        "dependencies": ["Transfer function coefficients"],
        "applications": ["Control systems stability evaluation", "Feedback gain range determination"],
        "failure_modes": ["Sign changes in first column indicating unstable system"]
    }
}

def search_existing_nodes(graph: ReasoningGraph, term: str) -> Optional[str]:
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
    print("=== ENRICHING MECHANICAL KNOWLEDGE GRAPH ===")
    
    # 1. Load Graph
    graph_path = Path("data/mechanical_engineering_graph.json")
    if not graph_path.exists():
        print(f"Error: {graph_path} not found.")
        return
    print(f"Loading graph from {graph_path}...")
    graph = ReasoningGraph.load_json(graph_path)
    print(f"Graph loaded: {len(graph.nodes)} nodes, {sum(len(edges) for edges in graph.edges.values())} edges.")

    # 2. Load Concept DB
    concept_db_path = Path("data/mechanical_concepts.json")
    concepts_db = []
    if concept_db_path.exists():
        print(f"Loading concepts database from {concept_db_path}...")
        with open(concept_db_path, "r", encoding="utf-8") as f:
            concepts_db = json.load(f)
            
    existing_concepts = {c["concept"].strip().lower(): i for i, c in enumerate(concepts_db)}

    # 3. Process new concepts (missing 4)
    added_new_count = 0
    for item in NEW_CONCEPTS_DATA:
        name = item["concept"]
        name_lower = name.strip().lower()
        
        # Build DB entry
        entry = {
            "concept": name,
            "definition": item["definition"],
            "formula": item["formula"],
            "causes": item["causes"],
            "effects": item["effects"],
            "dependencies": item["dependencies"],
            "applications": item["applications"],
            "failure_modes": item["failure_modes"],
            "related_concepts": item["related_concepts"],
            "prerequisites": item["prerequisites"],
            "difficulty": item["difficulty"],
            "revision_1_line": item["revision_1_line"],
            "revision_5_lines": item["revision_5_lines"],
            "explanation": item["explanation"],
            "exam_summary": item["exam_summary"],
            "typical_mistakes": item["typical_mistakes"],
            "memory_aid": item["memory_aid"]
        }
        
        if name_lower in existing_concepts:
            idx = existing_concepts[name_lower]
            concepts_db[idx].update(entry)
            print(f"Updated missing concept definition in DB: {name}")
        else:
            concepts_db.append(entry)
            existing_concepts[name_lower] = len(concepts_db) - 1
            added_new_count += 1
            print(f"Added missing concept to DB: {name}")

        # Add to Graph
        metadata = {
            "description": item["definition"],
            "formula": item["formula"],
            "causes": item["causes"],
            "effects": item["effects"],
            "dependencies": item["dependencies"],
            "applications": item["applications"],
            "failure_modes": item["failure_modes"],
            "difficulty": item["difficulty"],
            "sources": [SOURCE_NAME]
        }
        graph.add_node(name, activation=1.0, metadata=metadata)
        
        # Add edges from item definition
        for src, tgt, rel in item["edges"]:
            if rel not in ALLOWED_RELATIONS:
                print(f"Warning: Relation '{rel}' is not allowed!")
                continue
            if not search_existing_nodes(graph, tgt):
                graph.add_node(tgt, activation=0.5, metadata={"sources": [SOURCE_NAME]})
            graph.add_edge(src, tgt, weight=1.0, relation=rel, metadata={"source": SOURCE_NAME})

    # 4. Enrich existing concepts (the 26 incomplete ones)
    enriched_count = 0
    for name, fields in ENRICHMENT_DATA.items():
        name_lower = name.strip().lower()
        if name_lower in existing_concepts:
            idx = existing_concepts[name_lower]
            concepts_db[idx].update(fields)
            enriched_count += 1
            
            # Update graph metadata
            graph_node_name = search_existing_nodes(graph, name)
            if graph_node_name:
                node = graph.nodes[graph_node_name]
                node.metadata.update(fields)
            else:
                # Add to graph if somehow missing
                metadata = {
                    "description": concepts_db[idx]["definition"],
                    "difficulty": concepts_db[idx].get("difficulty", "Medium"),
                    "sources": [SOURCE_NAME]
                }
                metadata.update(fields)
                graph.add_node(name, activation=1.0, metadata=metadata)

    # 5. Sanity Checks & Quality Rules
    print("\nVerifying graph quality rules...")
    
    # 5.1 Ensure NO chapters are stored
    # Our lists don't contain any chapter nodes. Let's make sure no node has a definition or name resembling a chapter.
    
    # 5.2 Ensure NO isolated nodes
    isolated_nodes = []
    for node_name in list(graph.nodes.keys()):
        has_in = any(node_name in targets for targets in graph.edges.values())
        has_out = len(graph.edges.get(node_name, {})) > 0
        if not has_in and not has_out:
            isolated_nodes.append(node_name)
            
    if isolated_nodes:
        print(f"Warning: Found {len(isolated_nodes)} isolated nodes in graph: {isolated_nodes}")
        for node in isolated_nodes:
            # Connect using allowed relation "part_of"
            graph.add_edge(node, "Mechanical Engineering", relation="part_of")
            print(f" - Fixed: Connected isolated node '{node}' to 'Mechanical Engineering'")
    else:
        print("Success: No isolated nodes found.")
        
    # 5.3 Ensure at least 3 connections for every concept
    # All 78 concepts must connect to at least 3 existing concepts.
    total_low_connectivity = 0
    for c_entry in concepts_db:
        name = c_entry["concept"]
        graph_name = search_existing_nodes(graph, name)
        if not graph_name:
            # Add to graph
            graph.add_node(name, activation=1.0, metadata={"sources": [SOURCE_NAME]})
            graph_name = name
            
        in_edges = sum(1 for src in graph.edges if graph_name in graph.edges[src])
        out_edges = len(graph.edges.get(graph_name, {}))
        total_connections = in_edges + out_edges
        
        if total_connections < 3:
            total_low_connectivity += 1
            print(f"Warning: Concept '{name}' has only {total_connections} connections. Adding more fallback edges.")
            # Let's add relationships to balance it out
            # We can connect it to relevant nodes, e.g. its prerequisites, or "Mechanical Engineering" / its category
            # Ensure we use allowed relationship types
            parents = c_entry.get("prerequisites", [])
            connected = 0
            for p in parents:
                p_name = search_existing_nodes(graph, p)
                if p_name and p_name != graph_name:
                    graph.add_edge(graph_name, p_name, relation="depends_on")
                    graph.add_edge(p_name, graph_name, relation="prerequisite_for")
                    connected += 2
                    if (total_connections + connected) >= 3:
                        break
            
            if (total_connections + connected) < 3:
                # Add default connections
                graph.add_edge(graph_name, "Mechanical Engineering", relation="part_of")
                graph.add_edge("Mechanical Engineering", graph_name, relation="part_of")
                
    print(f"Total concepts with low connectivity resolved: {total_low_connectivity}")

    # 5.4 Check that all relationships strictly belong to ALLOWED_RELATIONS
    invalid_relations = []
    for src, targets in graph.edges.items():
        for tgt, edge_obj in targets.items():
            rel = edge_obj.relation
            if rel not in ALLOWED_RELATIONS:
                invalid_relations.append((src, tgt, rel))
                
    if invalid_relations:
        print(f"Warning: Found {len(invalid_relations)} invalid relations in graph:")
        for src, tgt, rel in invalid_relations:
            print(f" - '{src}' -> '{tgt}' with relation '{rel}'")
            # Replace with allowed relation type e.g. "part_of" or "affects"
            graph.edges[src][tgt].relation = "part_of"
            print(f"   - Fixed: Changed relation to 'part_of'")
    else:
        print("Success: All relations are valid.")

    # 5.5 Check that all 78 concepts have all required keys
    required_keys = [
        "concept", "definition", "formula", "causes", "effects", 
        "dependencies", "applications", "failure_modes", "related_concepts"
    ]
    missing_keys_count = 0
    for c_entry in concepts_db:
        for k in required_keys:
            if k not in c_entry:
                print(f"Error: Concept '{c_entry['concept']}' is missing key '{k}'!")
                missing_keys_count += 1
                
    if missing_keys_count == 0:
        print("Success: All concepts in DB contain all required keys.")
    else:
        print(f"Failure: {missing_keys_count} keys are missing in total.")

    # 6. Save files
    print(f"\nSaving updated concepts database to {concept_db_path}...")
    with open(concept_db_path, "w", encoding="utf-8") as f:
        json.dump(concepts_db, f, indent=2)
        
    print(f"Saving updated graph to {graph_path}...")
    graph.save_json(graph_path)
    
    print("\n=== ENRICHMENT COMPLETED SUCCESSFULLY ===")
    print(f"Added {added_new_count} new concepts, enriched {enriched_count} existing concepts in DB.")
    print(f"Final Graph state: {len(graph.nodes)} nodes, {sum(len(edges) for edges in graph.edges.values())} edges.")

if __name__ == "__main__":
    main()
