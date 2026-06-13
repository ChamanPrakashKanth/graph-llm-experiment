"""Concept Memory Layer for VLCM.

Represents concepts, hierarchies (with multiple parents), and weighted activations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set


class ConceptNode:
    """A single node in the concept memory representing a concept."""

    def __init__(
        self,
        name: str,
        parents: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.parents = parents or []
        self.children: List[str] = []
        self.description = description
        self.activation = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parents": self.parents,
            "children": self.children,
            "description": self.description,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConceptNode:
        node = cls(
            name=data["name"],
            parents=data.get("parents", []),
            description=data.get("description", ""),
        )
        node.children = data.get("children", [])
        node.activation = data.get("activation", 0.0)
        return node


class ConceptMemory:
    """Manages persistent hierarchical concepts with multi-parent support and activations."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}

    def add_concept(
        self,
        name: str,
        parents: Optional[List[str]] = None,
        description: str = "",
    ) -> ConceptNode:
        name = name.strip()
        if not name:
            raise ValueError("Concept name cannot be empty.")

        if name not in self.concepts:
            self.concepts[name] = ConceptNode(name, parents, description)
        else:
            if parents:
                for parent in parents:
                    if parent not in self.concepts[name].parents:
                        self.concepts[name].parents.append(parent)
            if description:
                self.concepts[name].description = description

        # Register children on parents
        if parents:
            for parent in parents:
                self.add_concept(parent)  # Ensure parent exists
                if name not in self.concepts[parent].children:
                    self.concepts[parent].children.append(name)

        return self.concepts[name]

    def get_concept(self, name: str) -> Optional[ConceptNode]:
        return self.concepts.get(name)

    def set_activation(self, name: str, value: float) -> None:
        if name in self.concepts:
            self.concepts[name].activation = float(value)

    def get_activation(self, name: str) -> float:
        node = self.concepts.get(name)
        return node.activation if node is not None else 0.0

    def reset_activations(self) -> None:
        for node in self.concepts.values():
            node.activation = 0.0

    def propagate_activations(
        self,
        upward_decay: float = 0.3,
        downward_decay: float = 0.5,
        iterations: int = 2,
    ) -> None:
        """Propagate activations through the hierarchy.
        
        When a concept is active, it spreads activation to its parents (upward)
        and children (downward) with decay.
        """
        for _ in range(iterations):
            updates: Dict[str, float] = {}
            for name, node in self.concepts.items():
                if node.activation > 0:
                    # Propagate to parents (multiple parents supported)
                    for parent in node.parents:
                        updates[parent] = max(
                            updates.get(parent, 0.0),
                            node.activation * upward_decay,
                        )
                    # Propagate to children
                    for child in node.children:
                        updates[child] = max(
                            updates.get(child, 0.0),
                            node.activation * downward_decay,
                        )
            for name, val in updates.items():
                if name in self.concepts:
                    self.concepts[name].activation = max(self.concepts[name].activation, val)

    def to_dict(self) -> dict:
        return {name: node.to_dict() for name, node in self.concepts.items()}

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> ConceptMemory:
        memory = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, concept_data in data.items():
            memory.concepts[name] = ConceptNode.from_dict(concept_data)
        return memory


def build_default_concept_memory() -> ConceptMemory:
    """Build the default engineering concept hierarchy."""
    mem = ConceptMemory()
    # 1. Pressure hierarchy
    mem.add_concept("Pressure", description="Force per unit area.")
    mem.add_concept("Hydrostatic Pressure", parents=["Pressure"], description="Pressure exerted by a fluid at equilibrium.")
    mem.add_concept("Dynamic Pressure", parents=["Pressure"], description="Kinetic energy per unit volume of fluid.")
    mem.add_concept("Gauge Pressure", parents=["Pressure"], description="Pressure relative to atmospheric pressure.")
    mem.add_concept("Absolute Pressure", parents=["Pressure"], description="Pressure relative to absolute zero pressure.")
    mem.add_concept("Stagnation Pressure", parents=["Pressure", "Dynamic Pressure"], description="Pressure fluid obtains when forced to cease flowing.")
    mem.add_concept("Pressure Drop", parents=["Pressure"], description="Difference in pressure between two points.")
    mem.add_concept("Low Pressure", parents=["Pressure"])

    # 2. Velocity and Boundary Layer
    mem.add_concept("Velocity")
    mem.add_concept("Boundary Layer", parents=["Velocity"])
    mem.add_concept("Turbulence", parents=["Velocity"])
    mem.add_concept("Vortex", parents=["Turbulence"])
    mem.add_concept("Flow Separation", parents=["Boundary Layer"])
    mem.add_concept("Drag Increase", parents=["Flow Separation"])
    mem.add_concept("Friction", parents=["Velocity"])

    # 3. Load & Stress
    mem.add_concept("Load")
    mem.add_concept("Stress", parents=["Load"])
    mem.add_concept("Strain", parents=["Load"])
    mem.add_concept("Failure", parents=["Stress", "Strain"])
    mem.add_concept("Slenderness")
    mem.add_concept("Compression", parents=["Load"])
    mem.add_concept("Lateral Deflection", parents=["Compression", "Slenderness"])
    mem.add_concept("Buckling", parents=["Lateral Deflection", "Failure"])
    mem.add_concept("Cyclic Load", parents=["Load"])
    mem.add_concept("Stress Concentration", parents=["Stress"])
    mem.add_concept("Crack Initiation", parents=["Stress Concentration"])
    mem.add_concept("Fatigue Failure", parents=["Crack Initiation", "Failure"])

    # 4. Thermal
    mem.add_concept("Temperature Gradient")
    mem.add_concept("Thermal Expansion", parents=["Temperature Gradient"])
    mem.add_concept("Thermal Stress", parents=["Stress", "Thermal Expansion"])
    mem.add_concept("Cracking", parents=["Thermal Stress", "Failure"])
    mem.add_concept("Heat Flux", parents=["Temperature Gradient"])
    mem.add_concept("Heat Transfer", parents=["Heat Flux", "Turbulence"])
    mem.add_concept("Cooling Rate", parents=["Heat Transfer"])

    # 5. CFD Convergence & Solves
    mem.add_concept("Mesh Quality")
    mem.add_concept("Numerical Instability", parents=["Mesh Quality"])
    mem.add_concept("Residual")
    mem.add_concept("Convergence Failure", parents=["Numerical Instability", "Residual"])
    mem.add_concept("Under Relaxation")
    mem.add_concept("Numerical Stability", parents=["Under Relaxation"])
    mem.add_concept("Convergence", parents=["Numerical Stability"])

    # 6. Combustion
    mem.add_concept("Heat Release")
    mem.add_concept("Pressure Wave", parents=["Pressure"])
    mem.add_concept("Acoustic Feedback", parents=["Pressure Wave"])
    mem.add_concept("Combustion Instability", parents=["Heat Release", "Acoustic Feedback"])

    return mem
