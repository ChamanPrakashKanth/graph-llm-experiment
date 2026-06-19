# cat_v3/generate_large_moe_dataset.py
import json
import random
import os

# Define the DOMAINS and DOMAIN_CONCEPTS to be exactly consistent with dataset.py
DOMAINS = ["mechanical", "civil", "electrical", "physics", "mathematics", "english"]

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

base_templates = [
    # 1. Mechanical: compressor, pressure_ratio, temperature_rise, entropy, efficiency
    {
        "active_experts": ["mechanical", "physics", "mathematics"],
        "concept_path": ["compressor", "pressure_ratio", "temperature_rise", "entropy", "efficiency"],
        "questions": [
            "Why does {c0} {c1} affect turbine {c4}?",
            "How does the {c1} of a {c0} influence turbine {c4}?",
            "Explain the relationship between {c0} {c1} and overall turbine {c4}.",
            "In a turbine system, why does changing the {c0} {c1} lead to a change in {c4}?",
            "What is the thermodynamic link between {c0} {c1}, {c2}, {c3}, and {c4}?",
            "How do {c0} {c1} and subsequent {c2} dictate {c4}?",
            "Analyze the impact of {c0} {c1} on {c3} generation and turbine {c4}.",
            "Can you trace how {c0} {c1} affects {c2}, resulting in {c3} and lower {c4}?"
        ],
        "responses": [
            "Increasing {c0} {c1} raises the outlet {c2}. This increases {c3} generation and can reduce overall turbine {c4}.",
            "The {c0} {c1} dictates the {c2}. High {c2} leads to higher {c3}, lowering the turbine's overall {c4}.",
            "An elevated {c0} {c1} leads to a higher {c2}, which causes {c3} to rise and ultimately degrades {c4}.",
            "Thermodynamically, {c0} {c1} drives {c2}, which produces {c3} and thus limits turbine {c4}."
        ]
    },
    # 2. Mechanical: temperature_rise, thermal_stress, cracking
    {
        "active_experts": ["mechanical", "physics"],
        "concept_path": ["temperature_rise", "thermal_stress", "cracking"],
        "questions": [
            "How does {c0} lead to {c2} in turbine blades?",
            "Why does high {c0} cause {c2} via {c1}?",
            "Analyze the cause of {c2} due to {c1} and {c0}.",
            "What leads to blade {c2} when {c1} increases?",
            "Explain how {c0} induces {c1} and results in material {c2}.",
            "What is the connection between {c0}, localized {c1}, and mechanical {c2}?"
        ],
        "responses": [
            "High {c0} induces severe {c1}, which eventually leads to {c2}.",
            "An elevated {c0} creates localized {c1}, causing {c2} in structural materials.",
            "Material {c2} is driven by the severe {c1} that arises from a rapid {c0}.",
            "The cycle of {c0} produces intense internal {c1}, which initiates structural {c2}."
        ]
    },
    # 3. Civil: foundation, load, bending_moment, beam, buckling
    {
        "active_experts": ["civil", "mechanical", "physics"],
        "concept_path": ["foundation", "load", "bending_moment", "beam", "buckling"],
        "questions": [
            "How does a {c0} {c1} affect {c3} {c4}?",
            "Explain the path from {c0} {c1} to structural {c3} {c4}.",
            "Why does transferring {c0} {c1} lead to {c4} in support {c3}s?",
            "How does {c1} on the {c0} cause {c3} {c4}?",
            "What role does {c2} play when {c0} {c1} causes {c3} {c4}?",
            "Detail how {c0} {c1} generates a {c2} that causes {c3} {c4}."
        ],
        "responses": [
            "The {c0} transfers the {c1}, causing a high {c2} in the {c3} which results in {c4}.",
            "Loading the {c0} produces a high {c2} along the {c3}, leading to structural {c4}.",
            "The {c3} undergoes {c4} because the {c0} {c1} induces an excessive internal {c2}.",
            "Applying a structural {c1} to the {c0} increases the {c2} on the {c3}, triggering lateral {c4}."
        ]
    },
    # 4. Civil: concrete, foundation, load, shear_force, bending_moment
    {
        "active_experts": ["civil"],
        "concept_path": ["concrete", "foundation", "load", "shear_force", "bending_moment"],
        "questions": [
            "What is the relation between {c3} and {c4} in {c0} {c1}s?",
            "How does a {c2} generate {c3} and {c4} in {c0} {c1} structures?",
            "Analyze {c4} and {c3} distributions in a loaded {c0} {c1}.",
            "Why does {c2} on a {c0} {c1} produce both {c3} and {c4}?",
            "Explain the conversion of {c0} {c1} {c2} into {c3} and {c4}."
        ],
        "responses": [
            "Applying a {c2} on {c0} {c1} structures produces {c3} that integrates into {c4} distributions.",
            "The {c2} on a {c0} {c1} induces {c3}, which integrates directly into the {c4}.",
            "In {c0} {c1}s, {c2} gives rise to internal {c3}, which changes along the span to define {c4}."
        ]
    },
    # 5. Electrical: capacitor, impedance, frequency, phase_angle, current
    {
        "active_experts": ["electrical", "physics", "mathematics"],
        "concept_path": ["capacitor", "impedance", "frequency", "phase_angle", "current"],
        "questions": [
            "Why does {c0} {c1} shift {c4} {c3}?",
            "How do {c0} {c1} and {c2} affect the {c4} {c3}?",
            "Explain how {c2} changes the {c0} {c1} and shifts {c4} {c3}.",
            "Analyze the relationship between {c0} {c1}, {c2}, and the resulting {c4} {c3}.",
            "In what way does the AC {c2} determine {c0} {c1} and {c4} {c3}?"
        ],
        "responses": [
            "A {c0} {c1} varies with {c2}, causing the {c4} {c3} to lead the voltage.",
            "The {c0} {c1} is inversely proportional to {c2}, shifting the {c4} {c3} relative to voltage.",
            "As AC {c2} changes, the {c0} {c1} shifts, which alters the {c4} {c3}."
        ]
    },
    # 6. Electrical: resistor, inductor, impedance, phase_angle, power_factor
    {
        "active_experts": ["electrical", "physics"],
        "concept_path": ["resistor", "inductor", "impedance", "phase_angle", "power_factor"],
        "questions": [
            "How do {c0} and {c1} values determine the circuit {c4}?",
            "Why do {c0} and {c1} {c2} values alter the {c4}?",
            "What is the role of {c0} and {c1} {c2} in defining the {c3} and {c4}?",
            "Explain how the {c4} is calculated from {c0} and {c1} electrical {c2}."
        ],
        "responses": [
            "Combining {c0} and {c1} {c2} increases the {c3}, which lowers the {c4}.",
            "The total circuit {c2} from the {c0} and {c1} alters the {c3}, modifying the {c4}.",
            "The reactive {c2} of the {c1} combined with the {c0} shifts the {c3}, directly lowering the {c4}."
        ]
    },
    # 7. Mathematics: matrix, eigenvalue, differential_equation, decay, logarithm
    {
        "active_experts": ["mathematics", "physics"],
        "concept_path": ["matrix", "eigenvalue", "differential_equation", "decay", "logarithm"],
        "questions": [
            "How do {c1}s explain {c3} {c2}s?",
            "Why do {c0} {c1}s solve {c2}s describing exponential {c3}?",
            "Explain the use of {c0} {c1}s and {c4}s in solving a {c3} {c2}.",
            "Analyze the connection between a system {c0}, its {c1}s, and a {c3} {c2}."
        ],
        "responses": [
            "Using {c0} {c1}s helps solve the {c2}, showing exponential {c3} governed by {c4}s.",
            "The {c1}s of the system {c0} solve the {c2}, yielding a {c3} rate that is analyzed with {c4}s.",
            "Solving the {c2} using {c0} {c1}s indicates a {c3} process characterized by a natural {c4}."
        ]
    },
    # 8. English: syntax, grammar, sentence, semantics
    {
        "active_experts": ["english"],
        "concept_path": ["syntax", "grammar", "sentence", "semantics"],
        "questions": [
            "How does {c0} affect {c3} interpretation of a {c2}?",
            "Why do {c0} and {c1} govern the {c3} of a {c2}?",
            "Explain how {c0}, {c1}, and {c2} structure determine {c3}.",
            "Analyze the relationship between {c0}, {c1}, and {c2} {c3}."
        ],
        "responses": [
            "Proper {c0} and {c1} organize words in a {c2} to deliver clear {c3}.",
            "A {c2} relies on correct {c0} and {c1} to convey its intended {c3}.",
            "Without sound {c0} and {c1}, a {c2} cannot resolve to coherent {c3}."
        ]
    },
    # 9. English: vocabulary, noun, verb, sentence, metaphor
    {
        "active_experts": ["english"],
        "concept_path": ["vocabulary", "noun", "verb", "sentence", "metaphor"],
        "questions": [
            "How are {c1}s and {c2}s combined to form a {c4}?",
            "Why do {c1}s and {c2}s in a {c3} create a figurative {c4}?",
            "Explain how combining {c0} {c1}s and {c2}s yields a {c4} in a {c3}.",
            "Analyze {c1} and {c2} combinations from the {c0} to form a {c4}."
        ],
        "responses": [
            "Selecting {c1} and {c2} combinations from {c0} builds a {c3} representing a {c4}.",
            "A {c4} is formed in a {c3} by combining non-literal {c1}s and {c2}s from our {c0}.",
            "We construct a {c4} inside a written {c3} by selecting descriptive {c1}s and {c2}s from the {c0}."
        ]
    },
    # 10. Civil/Mechanical: beam, load, force, bending_moment, buckling
    {
        "active_experts": ["civil", "mechanical", "physics"],
        "concept_path": ["beam", "load", "force", "bending_moment", "buckling"],
        "questions": [
            "How does applying a {c3} {c2} cause {c4} in a structural {c0}?",
            "Explain {c0} {c4} under the action of a {c3} {c2}.",
            "Why does a {c0} under a critical {c1} or {c2} experience {c4} due to {c3}?",
            "Analyze the structural risk of {c4} when a {c0} is subjected to {c3}."
        ],
        "responses": [
            "Applying {c2} to a {c0} generates a {c3} that exceeds critical thresholds, leading to {c4}.",
            "A structural {c0} under compressive {c2} develops a {c3}, resulting in lateral {c4}.",
            "The {c0} fails through {c4} when the applied {c1} creates an excessive internal {c3}."
        ]
    },
    # 11. Mathematics/Physics: gravity, force, acceleration, differential_equation
    {
        "active_experts": ["mathematics", "physics"],
        "concept_path": ["gravity", "force", "acceleration", "differential_equation"],
        "questions": [
            "How is {c0} modeled as a {c3} for {c2}?",
            "Why does gravitational {c1} lead to a {c3} describing {c2}?",
            "Explain how the {c1} of {c0} is expressed as an {c2} {c3}.",
            "Analyze how Newton's laws model {c0} {c1} and {c2} via a {c3}."
        ],
        "responses": [
            "Gravitational {c1} produces an {c2} that is written as a second-order {c3}.",
            "Modeling {c0} as a {c1} yields a {c3} representing {c2} and velocity over time.",
            "The physical {c1} of {c0} accelerates the mass, which is mathematically solved as a {c3}."
        ]
    },
    # 12. Mechanical: fluid_flow, compressor, pressure_ratio
    {
        "active_experts": ["mechanical"],
        "concept_path": ["fluid_flow", "compressor", "pressure_ratio"],
        "questions": [
            "How does {c0} influence the {c1} {c2}?",
            "In what way does the {c0} determine the {c1} {c2}?",
            "Explain the impact of {c0} velocity on the {c1} {c2}.",
            "Does changing {c0} alter the operating {c1} {c2}?"
        ],
        "responses": [
            "The rate of {c0} governs the {c1}'s performance, directly altering the {c2}.",
            "Velocity of {c0} affects the pressure rise in the {c1}, modifying the operating {c2}.",
            "Steady {c0} maintains a stable {c1} {c2} across the blades."
        ]
    },
    # 13. Mechanical: fluid_flow, turbine, efficiency
    {
        "active_experts": ["mechanical"],
        "concept_path": ["fluid_flow", "turbine", "efficiency"],
        "questions": [
            "How does {c0} affect turbine {c2}?",
            "Why does the {c0} pattern determine the {c2} of the turbine?",
            "Analyze how turbine {c2} depends on the characteristics of {c0}.",
            "Explain why unstable {c0} reduces turbine {c2}."
        ],
        "responses": [
            "Optimal {c0} minimizes boundary layer separation, resulting in higher turbine {c2}.",
            "The behavior of the {c0} inside the blades directly determines the aerodynamic turbine {c2}.",
            "Turbine {c2} is maximized when {c0} matches the blade design angles."
        ]
    },
    # 14. Civil: foundation, concrete, rebar, soil_bearing
    {
        "active_experts": ["civil"],
        "concept_path": ["foundation", "concrete", "rebar", "soil_bearing"],
        "questions": [
            "How do {c1} and {c2} reinforce a {c0} for {c3}?",
            "Why are {c1} and {c2} essential for foundation {c3} capacity?",
            "Analyze structural {c1} and {c2} requirements to support foundation {c3}."
        ],
        "responses": [
            "Reinforcing the {c0} with {c1} and {c2} distributes the load to match the {c3} capacity.",
            "The structural integrity of a {c0} is achieved using {c1} and {c2} to prevent exceeding the {c3} limit."
        ]
    },
    # 15. Civil: rebar, concrete, beam, shear_force
    {
        "active_experts": ["civil"],
        "concept_path": ["rebar", "concrete", "beam", "shear_force"],
        "questions": [
            "How do {c0} and {c1} reinforce a {c2} against {c3}?",
            "Why is {c0} placement necessary to resist {c3} in a {c1} {c2}?",
            "Analyze the role of structural {c0} and {c1} in preventing {c3} failure in a {c2}."
        ],
        "responses": [
            "Embedding {c0} inside the {c1} {c2} provides tensile reinforcement to resist high {c3}.",
            "The {c1} handles compression while {c0} prevents diagonal tension cracks caused by {c3} in the {c2}."
        ]
    },
    # 16. Electrical: voltage, current, impedance, frequency
    {
        "active_experts": ["electrical", "physics"],
        "concept_path": ["voltage", "current", "impedance", "frequency"],
        "questions": [
            "How do {c0} and {c1} relate to {c2} and AC {c3}?",
            "Why does the ratio of {c0} to {c1} depend on the {c3} through {c2}?",
            "Explain the effect of AC {c3} on circuit {c2}, {c0}, and {c1}.",
            "Analyze how changing {c3} alters the relation between {c0} and {c1} via {c2}."
        ],
        "responses": [
            "AC {c2} varies with {c3}, changing the resulting {c1} for a constant applied {c0}.",
            "As the AC {c3} shifts, the circuit {c2} changes, which alters the {c1} response under the applied {c0}."
        ]
    },
    # 17. Electrical: capacitor, frequency, impedance, power_factor
    {
        "active_experts": ["electrical"],
        "concept_path": ["capacitor", "frequency", "impedance", "power_factor"],
        "questions": [
            "How does a {c0} react to {c1} to alter {c2} and improve {c3}?",
            "Why does {c0} {c2} at a specific {c1} optimize the system {c3}?",
            "Analyze the impact of {c0} charging and AC {c1} on overall circuit {c3} through {c2}."
        ],
        "responses": [
            "The {c0} {c2} decreases at high {c1}, allowing reactive power correction which improves the overall {c3}.",
            "By adjusting AC {c1}, the {c0} {c2} is tuned to balance the inductive load and optimize {c3}."
        ]
    },
    # 18. Physics: thermodynamics, entropy, energy, heat
    {
        "active_experts": ["physics", "mechanical"],
        "concept_path": ["thermodynamics", "entropy", "energy", "heat"],
        "questions": [
            "Explain the connection between {c0}, {c1}, and {c3} {c2}.",
            "Why does the study of {c0} show that {c3} transfer increases {c1}?",
            "Detail how transfer of {c3} {c2} changes a system's {c1} according to {c0}."
        ],
        "responses": [
            "According to {c0}, transferring {c3} {c2} between systems always increases total net {c1}.",
            "In {c0}, {c3} is a form of {c2} in transit, and its flow inevitably generates system {c1}."
        ]
    },
    # 19. Physics: force, acceleration, velocity, gravity
    {
        "active_experts": ["physics"],
        "concept_path": ["force", "acceleration", "velocity", "gravity"],
        "questions": [
            "How does gravitational {c0} affect a body's {c1} and {c2} under {c3}?",
            "Explain how the {c0} of {c3} determines free-fall {c1} and {c2}.",
            "Analyze the kinematic relationship between {c3} {c0}, acceleration, and velocity."
        ],
        "responses": [
            "The downward {c0} of {c3} induces a constant {c1}, causing a linear increase in {c2}.",
            "Under the influence of {c3}, the gravitational {c0} drives a constant {c1} that increases the object's terminal {c2}."
        ]
    },
    # 20. Physics: energy, force, acceleration, velocity
    {
        "active_experts": ["physics", "mathematics"],
        "concept_path": ["energy", "force", "acceleration", "velocity"],
        "questions": [
            "How does work done by a {c1} alter kinetic {c0}, {c2}, and {c3}?",
            "Why does applying a physical {c1} cause {c2}, leading to higher {c3} and {c0}?",
            "Analyze how {c1} creates {c2} to change a particle's {c3} and mechanical {c0}."
        ],
        "responses": [
            "A net {c1} causes {c2}, which increases the particle's {c3} and therefore raises its kinetic {c0}.",
            "The mechanical {c0} changes as the applied {c1} drives {c2}, resulting in an increased {c3}."
        ]
    },
    # 21. Mathematics: characteristic_equation, eigenvalue, matrix, derivative
    {
        "active_experts": ["mathematics"],
        "concept_path": ["characteristic_equation", "eigenvalue", "matrix", "derivative"],
        "questions": [
            "How do we use the {c0} to find {c1}s of a {c2} of {c3}s?",
            "Why does solving the {c0} of a {c2} relate to {c1}s in {c3} systems?",
            "Explain the derivation of the {c0} for a system {c2} of linear {c3}s."
        ],
        "responses": [
            "For a system of {c3}s, we represent the coefficients in a {c2} and solve the {c0} to find the {c1}s.",
            "The {c1}s of a linear {c3} operator are the roots of the {c0} derived from its coefficient {c2}."
        ]
    },
    # 22. Mathematics: derivative, integral, differential_equation
    {
        "active_experts": ["mathematics"],
        "concept_path": ["derivative", "integral", "differential_equation"],
        "questions": [
            "What is the role of the {c0} and the {c1} in solving a {c2}?",
            "How do we combine {c0}s and {c1}s to solve a multi-variable {c2}?",
            "Explain how the fundamental theorem connecting {c0} and {c1} solves a first-order {c2}."
        ],
        "responses": [
            "A {c2} describes a relation using {c0}s, which is solved by applying the {c1} operator.",
            "We integrate the terms of the {c2} containing {c0}s to find the general solution."
        ]
    },
    # 23. Mathematics: characteristic_equation, derivative, differential_equation
    {
        "active_experts": ["mathematics"],
        "concept_path": ["characteristic_equation", "derivative", "differential_equation"],
        "questions": [
            "How is the {c0} formulated for a high-order {c2} containing {c1}s?",
            "Why does substituting the {c1} with algebraic variables yield the {c0} of the {c2}?",
            "Analyze the connection between linear {c2}s, their {c1}s, and the {c0}."
        ],
        "responses": [
            "Substituting the {c1} operator with a scalar variable transforms the {c2} into its algebraic {c0}.",
            "We solve the algebraic {c0} to find the exponential roots of the homogeneous {c2}."
        ]
    },
    # 24. English: semantics, noun, verb, adjective
    {
        "active_experts": ["english"],
        "concept_path": ["semantics", "noun", "verb", "adjective"],
        "questions": [
            "How do {c1}s, {c2}s, and {c3}s shape lexical {c0}?",
            "Why is the {c0} of a statement determined by its {c1}, {c2}, and modifying {c3}?",
            "Explain the grammatical connection between a {c1}, {c2}, {c3}, and sentence {c0}."
        ],
        "responses": [
            "Lexical {c0} is built from structural relationships between the {c1}, the action {c2}, and modifying {c3}s.",
            "The {c1} and {c2} establish the core state, while the {c3} refines the sentence {c0}."
        ]
    },
    # 25. English: grammar, sentence, adjective, syntax
    {
        "active_experts": ["english"],
        "concept_path": ["grammar", "sentence", "adjective", "syntax"],
        "questions": [
            "How does English {c0} govern the placement of an {c2} in {c3} of a {c1}?",
            "Why does the {c3} of a {c1} depend on the {c0} rules for {c2}s?",
            "Analyze adjective modifiers in {c1} structure using {c0} and {c3}."
        ],
        "responses": [
            "Standard {c0} rules dictate that an {c2} precedes its noun in the {c3} of a normal {c1}.",
            "The {c3} of a written {c1} is determined by the universal {c0} that governs {c2} modifiers."
        ]
    },
    # 26. Cross-Domain: heat, temperature_rise, pressure, entropy
    {
        "active_experts": ["physics", "mechanical"],
        "concept_path": ["heat", "temperature_rise", "pressure", "entropy"],
        "questions": [
            "How do {c0} and subsequent {c1} increase gas {c2} and system {c3}?",
            "Why does adding thermal {c0} produce a {c1}, leading to higher {c2} and {c3}?",
            "Analyze the thermodynamic path where {c0} causes {c1}, altering {c2} and generating {c3}."
        ],
        "responses": [
            "Adding thermal {c0} drives a {c1}, which increases the gas {c2} and generates internal {c3}.",
            "The input of {c0} yields a rapid {c1}, boosting the pressure against restrictions and increasing total {c3}."
        ]
    },
    # 27. Cross-Domain: voltage, current, energy, power_factor
    {
        "active_experts": ["electrical", "physics"],
        "concept_path": ["voltage", "current", "energy", "power_factor"],
        "questions": [
            "How do AC {c0} and {c1} affect the efficiency of {c2} transmission and {c3}?",
            "Why does the phase difference between {c0} and {c1} decrease delivered {c2} and lower {c3}?",
            "Explain how the {c3} of electrical {c2} delivery depends on {c0} and {c1} phase."
        ],
        "responses": [
            "The phase angle between {c0} and {c1} dictates how much electrical {c2} is usable, defining the {c3}.",
            "A shift in {c0} relative to {c1} lowers the {c3}, which reduces the rate of useful {c2} delivery."
        ]
    },
    # 28. Cross-Domain: differential_equation, acceleration, force, gravity
    {
        "active_experts": ["mathematics", "physics"],
        "concept_path": ["differential_equation", "acceleration", "force", "gravity"],
        "questions": [
            "How is the motion under gravitational {c2} written as a {c0} for {c1}?",
            "Why does the physical {c2} of {c3} yield a solvable {c0} representing particle {c1}?",
            "Explain the derivation of the {c0} of motion starting from {c3} {c2} and {c1}."
        ],
        "responses": [
            "Starting with the gravitational {c2} of {c3}, Newton's second law yields a second-order {c0} for {c1}.",
            "We model the {c2} of {c3} acting on a mass to formulate a {c0} describing {c1} and velocity."
        ]
    },
    # 29. Cross-Domain: syntax, grammar, semantics, metaphor
    {
        "active_experts": ["english"],
        "concept_path": ["syntax", "grammar", "semantics", "metaphor"],
        "questions": [
            "How do {c0} and {c1} enable the {c2} of a complex {c3}?",
            "Why does the interpretation of a literary {c3} rely on sentence {c0}, {c1}, and {c2}?",
            "Analyze structural {c0} and {c1} as requirements to resolve the {c2} of a {c3}."
        ],
        "responses": [
            "Coherent sentence {c0} and {c1} structure the language, allowing the reader to decode the {c2} of a {c3}.",
            "A literary {c3} depends on standard {c0} and {c1} to successfully convey its deeper {c2}."
        ]
    },
    # 30. Cross-Domain: thermal_stress, cracking, concrete, beam
    {
        "active_experts": ["mechanical", "civil", "physics"],
        "concept_path": ["thermal_stress", "cracking", "concrete", "beam"],
        "questions": [
            "How does localized {c0} cause structural {c1} in a reinforced {c2} {c3}?",
            "Why do temperature variations induce {c0} that leads to {c1} in {c2} {c3} elements?",
            "Analyze structural failure via {c1} when a structural {c2} {c3} undergoes high {c0}."
        ],
        "responses": [
            "High internal {c0} within the reinforced {c2} {c3} exceeds tensile strength, producing structural {c1}.",
            "Severe {c0} in the structural {c2} {c3} initiates micro-fractures, which propagate into visible {c1}."
        ]
    }
]

prefixes = [
    "",
    "explain the following: ",
    "analyze this: ",
    "question: ",
    "regarding this topic, ",
    "can you explain: ",
    "please address: ",
    "in engineering systems, "
]

suffixes = [
    "",
    " (in detail)",
    " (briefly)",
    " for industrial systems",
    " using mathematical formulations",
    " please"
]

response_prefixes = [
    "",
    "in general, ",
    "based on engineering analysis, ",
    "typically, ",
    "under standard conditions, ",
    "fundamentally, "
]

def format_sentence(s, concepts):
    # Map placeholders like {c0} -> concept name (with underscores replaced by spaces)
    fmt_dict = {}
    for i, c in enumerate(concepts):
        fmt_dict[f"c{i}"] = c.replace("_", " ")
    return s.format(**fmt_dict)

def capitalize_first(s):
    if not s:
        return s
    return s[0].upper() + s[1:]

def generate_dataset():
    dataset = []
    
    for idx, template in enumerate(base_templates):
        path = template["concept_path"]
        experts = template["active_experts"]
        
        formatted_questions = [format_sentence(q, path) for q in template["questions"]]
        formatted_responses = [format_sentence(r, path) for r in template["responses"]]
        
        # Generate combinatorial variations
        # We want to make sure we get enough samples per template.
        # With 30 templates, if we generate ~50 samples per template, we get 1,500 samples.
        count = 0
        local_seen = set()
        
        # Shuffle prefixes/suffixes to get a nice distribution
        random.seed(idx)
        shuffled_prefixes = list(prefixes)
        shuffled_suffixes = list(suffixes)
        shuffled_resp_prefixes = list(response_prefixes)
        
        # Loop over questions and responses, applying prefixes/suffixes
        for q_base in formatted_questions:
            for r_base in formatted_responses:
                for pref in shuffled_prefixes:
                    for suff in shuffled_suffixes:
                        for resp_pref in shuffled_resp_prefixes:
                            
                            # Construct question
                            q_str = f"{pref}{q_base}{suff}".strip()
                            q_str = capitalize_first(q_str)
                            # Ensure proper ending punctuation
                            if not q_str.endswith("?") and not q_str.endswith(".") and not q_str.endswith(")"):
                                q_str += "?"
                                
                            # Construct response
                            r_base_lowered = r_base[0].lower() + r_base[1:] if r_base else r_base
                            r_str = f"{resp_pref}{r_base_lowered}".strip()
                            r_str = capitalize_first(r_str)
                            if not r_str.endswith("."):
                                r_str += "."
                                
                            key = (q_str.lower(), r_str.lower())
                            if key not in local_seen:
                                local_seen.add(key)
                                dataset.append({
                                    "question": q_str,
                                    "active_experts": experts,
                                    "concept_paths": [path],
                                    "response": r_str
                                })
                                count += 1
                                # Limit to 50 variations per base template to avoid giant redundant files
                                if count >= 55:
                                    break
                        if count >= 55:
                            break
                    if count >= 55:
                        break
                if count >= 55:
                    break
                    
    print(f"Generated {len(dataset)} unique samples.")
    return dataset

def main():
    dataset = generate_dataset()
    
    os.makedirs("data", exist_ok=True)
    out_path = "data/cat_v3_reasoning_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset successfully written to {out_path}.")

if __name__ == "__main__":
    main()
