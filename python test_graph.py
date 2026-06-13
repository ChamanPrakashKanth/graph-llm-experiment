# test_graph.py

from reasoning_graph import build_cfd_graph
from reasoning_graph import ActivationEngine
from reasoning_graph import GraphReasoner

graph = build_cfd_graph()

engine = ActivationEngine(graph)

engine.activate(["Pressure"])

reasoner = GraphReasoner(graph)

path = reasoner.best_path(["Pressure"])

print(path)