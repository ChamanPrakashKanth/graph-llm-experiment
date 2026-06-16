# scratch/inspect_graph.py
import json
from pathlib import Path

def main():
    graph_path = Path("data/mechanical_engineering_graph.json")
    if not graph_path.exists():
        print(f"Error: {graph_path} not found")
        return
        
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
        
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    
    print("\nFirst 10 nodes:")
    for n in nodes[:10]:
        print(f" - {n.get('name')}: {n.get('metadata')}")
        
    print("\nFirst 10 edges:")
    for e in edges[:10]:
        print(f" - {e.get('source')} -> {e.get('target')} ({e.get('relation')}) [weight={e.get('weight')}, evidence={e.get('evidence_count')}]")

    # Get relation types
    relations = {}
    for e in edges:
        rel = e.get("relation", "unknown")
        relations[rel] = relations.get(rel, 0) + 1
    print("\nRelation type distribution:")
    for rel, count in relations.items():
        print(f" - {rel}: {count}")

if __name__ == "__main__":
    main()
