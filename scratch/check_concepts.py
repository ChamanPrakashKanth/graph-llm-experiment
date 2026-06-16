import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Define lists of concepts expected from the user's prompt
EXPECTED_CONCEPTS = [
    # Thermodynamics
    "First Law", "Second Law", "Entropy", "Availability", "Power Cycles", "Refrigeration", "Gas Turbines",
    # Fluid Mechanics
    "Reynolds Number", "Bernoulli Equation", "Boundary Layer", "Turbulence", "Pipe Flow", "Dimensional Analysis",
    # Heat Transfer
    "Conduction", "Convection", "Radiation", "Thermal Resistance", "Heat Exchangers", "Boiling", "Condensation",
    # Strength of Materials
    "Stress", "Strain", "Elasticity", "Bending", "Torsion", "Columns", "Buckling",
    # Machine Design
    "Fatigue", "Failure Theories", "Shafts", "Bearings", "Gears", "Springs", "Bolted Joints",
    # Theory of Machines
    "Mechanisms", "Velocity Analysis", "Governors", "Balancing", "Gyroscopes",
    # Manufacturing
    "Casting", "Welding", "Machining", "Metal Forming", "CNC",
    # Vibrations
    "Free Vibration", "Forced Vibration", "Resonance", "Damping", "Vibration Isolation",
    # Control Systems
    "Transfer Functions", "Stability", "Root Locus", "Frequency Response",
    # Engineering Mathematics
    "Differential Equations", "Linear Algebra", "Laplace Transform", "Numerical Methods", "Probability"
]

def main():
    print("=== Concept Verification ===")
    concept_db_path = Path("data/mechanical_concepts.json")
    if not concept_db_path.exists():
        print("Error: mechanical_concepts.json not found")
        return

    with open(concept_db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    db_concepts = {x["concept"].strip().lower(): x for x in db}

    missing = []
    incomplete_fields = []
    required_keys = [
        "concept", "definition", "formula", "causes", "effects", 
        "dependencies", "applications", "failure_modes", "related_concepts"
    ]

    for c in EXPECTED_CONCEPTS:
        c_lower = c.strip().lower()
        if c_lower not in db_concepts:
            missing.append(c)
        else:
            entry = db_concepts[c_lower]
            missing_keys = [k for k in required_keys if k not in entry]
            if missing_keys:
                incomplete_fields.append((c, missing_keys))

    print(f"Total expected: {len(EXPECTED_CONCEPTS)}")
    print(f"Total missing in database: {len(missing)}")
    if missing:
        print(f"Missing list: {missing}")

    print(f"Total incomplete field sets: {len(incomplete_fields)}")
    for c, keys in incomplete_fields:
        print(f" - {c} is missing keys: {keys}")

    # Let's also check graph connectivity
    graph_path = Path("data/mechanical_engineering_graph.json")
    if graph_path.exists():
        from reasoning_graph import ReasoningGraph
        graph = ReasoningGraph.load_json(graph_path)
        print(f"\nLoaded graph: {len(graph.nodes)} nodes")
        
        low_connectivity = []
        for c in EXPECTED_CONCEPTS:
            c_name = None
            # Find matching node name in graph
            for node_name in graph.nodes:
                if node_name.lower() == c.lower():
                    c_name = node_name
                    break
            
            if not c_name:
                print(f" - Concept '{c}' NOT found in graph!")
                continue

            # Calculate degree
            in_edges = sum(1 for src in graph.edges if c_name in graph.edges[src])
            out_edges = len(graph.edges.get(c_name, {}))
            total_edges = in_edges + out_edges
            if total_edges < 3:
                low_connectivity.append((c_name, total_edges))
                
        print(f"Total concepts with < 3 connections in graph: {len(low_connectivity)}")
        if low_connectivity:
            for name, deg in low_connectivity:
                print(f" - {name} has only {deg} connections")
    else:
        print("Graph file not found")

if __name__ == "__main__":
    main()
