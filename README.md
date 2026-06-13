# CAT V2 & VLCM: Concept-Based Neural-Symbolic Reasoning

CAT V2 (Concept Attention Transformer) and VLCM (Very Large Concepts Model) are advanced research prototypes for **explicit neural-symbolic concept reasoning**. Unlike traditional language models that generate free-form text token-by-token (frequently leading to logical drift and hallucinations), these architectures activate concept nodes on a structured domain graph and perform neural message passing to generate valid, verifiable planning paths.

```text
Traditional LLM:  Question ➔ Tokens ➔ Attention ➔ Tokens ➔ Answer (High Hallucinations)
CAT V2 / VLCM:    Question ➔ Concept Activations ➔ GNN Message Passing ➔ Causal Path Decoding ➔ Exact Answer (0% Logic Hallucinations)
```

---

## 🏛️ Architectures

### 1. CAT V2 Flow
CAT V2 maps user queries to concept activations, performs message passing over a concept graph, and decodes the reasoning path step-by-step using a Gated Recurrent Unit (GRU) path generator constrained by transition masks.

```mermaid
graph TD
    Q[Question Text] -->|Simple Tokenizer| TE[Tiny Transformer Encoder]
    TE -->|CLS Embedding| CA[Concept Activator]
    TE -->|CLS Embedding| QP[Question Projection]
    CA -->|Multi-Label BCE Probabilities| CM[Concept Memory Embeddings]
    QP -->|Linear Projection| CM
    CM -->|Initial Graph State| GMP[Graph Message Passing Layer]
    GMP -->|Propagation Matrix / Graph State| PG[GRU Path Generator]
    QP -->|Context Vector| PG
    PG -->|Transition Mask Constraints| PL[Next Concept Logits]
    PL -->|Softmax Probability| NC[Step-by-Step Path Output]
    NC -->|Template Decoder| A[Final Answer]
```

### 2. VLCM (Very Large Concepts Model) Flow
VLCM extends this paradigm with a multi-parent DAG concept memory layer, learnable edge importances, and a causal Transformer Decoder for reasoning path generation.

```mermaid
graph TD
    Q[User Question] -->|TinyTransformerEncoder| QE[Dense Query Embedding]
    QE -->|ConceptActivator Head| ICA[Initial Concept Activations]
    
    subgraph Concept Memory Layer
        VCM[Very Large Concept Memory]
        H[Multi-parent DAG Hierarchies]
    end
    
    ICA -->|Weighted activations| VCM
    
    subgraph Graph Neural Memory
        GMM[GraphNeuralMemory]
        LEW[Learnable Edge Importance]
    end
    
    VCM -->|Propagation matrix| GMM
    QE -->|Context projection| GMM
    GMM -->|Soft propagation| LEW
    LEW -->|GNN Message Passing| GS[Propagated Graph States]
    
    subgraph Concept Reasoning Transformer
        CRT[ConceptReasoningTransformer Decoder]
        TM[Transition Mask Constraints]
    end
    
    GS -->|Concept Embeddings| CRT
    QE -->|Cross-Attention Memory| CRT
    CRT -->|Causal Self-Attention| TM
    TM -->|Next Concept Logits| RP[Explicit Reasoning Path]
    
    RP -->|show_reasoning_path| EXP[Explainable Output]
    RP -->|Language Decoder| ANS[Final Technical Answer]
```

---

## 🚀 Key Features

* **Strict Graph Constraint**: Uses a topological transition mask to restrict path generation at each step. Logic-leap and path hallucination rate is **0%**.
* **Domain Checkpoints**: Includes pre-trained checkpoints for:
  * 🧮 **Computational Fluid Dynamics (CFD)**
  * 🏗️ **Structural Engineering**
  * 🐍 **Python Coding AI** (Planning and mapping programming tasks to code concepts)
* **Interactive Lab GUI**: A built-in web interface featuring vis.js network graphs to inspect node activations and next-concept probability bars in real time.
* **Extreme Memory Compression**: Eliminates standard token KV Cache scaling, enabling deployment of high-order reasoning graphs on edge CPUs.

---

## 🐍 Real-World Case Study: Python Coding AI

The Python Coding AI domain demonstrates how CAT V2 can be used as a **logical planning engine** for software development tasks. Instead of jumping straight to code generation, the model maps a natural language programming query to a sequence of execution concepts, which can then be compiled into clean, robust Python snippets.

### GNN Semantics & Graph Growth
The model represents programming semantics by growing a GNN graph directly from general Python documentation (`data/python_docs.txt`) via sentence co-occurrence scanning. For example:
* **The Input Task**: `"How to filter a list of numbers to find even numbers?"`
* **Semantic Capture**: Embedded into a dense vector by the encoder, mapping query intent to initial GNN states.
* **Next-Concept Prediction**: The GNN maps this onto the vocabulary, generating the path:
  $$\text{List Input} \rightarrow \text{Modulo Condition} \rightarrow \text{List Comprehension} \rightarrow \text{Filtered Output}$$

---

## 📊 Comparison & Benchmarks

Here is an empirical and theoretical comparison of the CAT V2 Concept SLM/VLCM against traditional token-level autoregressive models:

| Dimension / Metric | Traditional Token LLM (e.g., Llama-3 8B) | CAT V2 / VLCM (Concept SLM) |
| :--- | :--- | :--- |
| **Fundamental Sequence Unit** | Token (Characters/Words) | Concept (Nodes/Edges) |
| **Model Size (Parameters)** | 8,000,000,000 | **~638K - 868K** (Ultra-lightweight) |
| **Reasoning Constraint** | Soft (Token Probability-based) | **Strict 100%** (Transition Mask) |
| **Path Hallucinations** | High (frequently skips logical steps) | **0%** (Topologically constrained) |
| **Memory Footprint (KV Cache / Graph)** | **50,000.00 MB** (at 100k context) | **2.61 MB** (~19,200x compression) |
| **Generation Compute Cost** | ~8.2 Trillion FLOPs | **~7.6 Million FLOPs** (~1,000,000x saving) |
| **Inference Hardware** | Multi-GPU Cloud Clusters / High-end RAM | CPU (Runs on microcontrollers & edge devices) |
| **Average Latency (CPU)** | Seconds to Minutes | **~27.06 ms** |
| **GNN Graph Size (Python)** | N/A | **45 concepts**, **132 directed edges** |

---

## 💥 Where it Breaks: Structural Failure Modes

While your architecture provides **0% path hallucinations** inside the graph boundaries, it has severe structural breaking points when evaluated against open-ended programming tasks:

### 1. The Syntax Generation Void (The "Formatting" Gap)
* **The Break**: The model cannot output code syntax (e.g. `def check_prime(n):`). It only outputs concept plans (`["List Input", "Modulo Condition", "Filtered Output"]`).
* **Consequence**: It requires a secondary translation layer (System 1) like a template compiler or a token-level LLM to generate the final source code. If that translation layer fails, the code is broken despite the correct plan.

### 2. Semantic Noise from Co-Occurrence Graph Growth
* **The Break**: Growing the GNN graph from text documentation via window co-occurrence is noisy. If a sentence mentions both "list comprehensions" and "regular expressions" in the same context, the GNN adds an edge between them.
* **Consequence**: The GNN may create invalid shortcuts or transition rules (e.g., transitioning directly from `List Comprehension` to `Compile Pattern`), allowing the path generator to construct nonsensical plans that are topographically valid on the noisy graph.

### 3. Nested Control Flows and Tree/DAG Architectures
* **The Break**: Your model generates flat, sequential paths of concepts ($C_1 \rightarrow C_2 \rightarrow C_3$). 
* **Consequence**: Real programs are structured with complex nested loops, conditional branches (`if-else`), scope limits, and recursion. Representing a program with multiple nested branches as a single flat path of concepts is mathematically impossible. The architecture breaks down on any task requiring non-linear program graphs.

### 4. Error Propagation & No Self-Correction
* **The Break**: If the initial Concept Activator misinterprets the query and fails to activate the correct entry point node, the GNN will propagate activations to the wrong graph neighborhood.
* **Consequence**: Because path generation is strictly constrained by the transition mask, the model will be forced to generate a path within the incorrect neighborhood. There is no mid-path self-correction mechanism to jump to a disjoint subgraph.

### 5. Infinite Search Space & Out-of-Vocabulary Code
* **The Break**: Coding is open-ended. New libraries, custom functions, and specific variables are created constantly.
* **Consequence**: Since the vocabulary is closed (defined by the Concept Vocabulary), the model cannot handle queries involving libraries or concepts outside its trained vocabulary. It cannot "invent" a concept at test time.

---

## 🧠 Recursive Semantic Reasoning Loop Evaluation (CAT V2 Deep Dive)

The CAT V2 architecture performs iterative semantic reasoning loops:
$$\text{Token} \longrightarrow \text{Concept} \longrightarrow \text{Graph Update} \longrightarrow \text{Next Concept Prediction} \longrightarrow \text{Graph Update} \longrightarrow \dots$$

Here is an in-depth critique of this recursive formulation:

### 1. Reasoning Depth: Decoupling Layers from Logic
* **Dynamic Planning Horizon**: The reasoning depth is determined by the number of autoregressive loop iterations $T$, rather than the model's layer depth. The effective depth is $D = T \times G$ (where $G$ is GNN propagation layers).
* **Multi-Hop Traversal**: Even with a 1-layer GNN ($G=1$), the model can reason across an arbitrary $N$-hop concept chain by running the loop $N$ times. This permits complex long-range reasoning chains without scaling the parameter count or layer depth of the neural network.

### 2. GNN State Updates as a Structured Working Memory
* **Explicit State Space**: The memory state is represented as a distribution of activations over a discrete vocabulary of concepts: $\mathbf{h}_t \in \mathbb{R}^{|V| \times d}$. Every element in this memory is directly projectable back to a human-understandable concept node.
* **Topological Conservation**: During GNN message passing, activations propagate only along the pre-defined graph edges. The memory state cannot drift into arbitrary, uninterpretable vector states; its trajectory is strictly bounded by the rules of the GNN.
* **Contextual Persistence**: By injecting the newly predicted concept back into the graph, the GNN acts as an attractor network. The current planning focus is continually modulated by the query context projection, keeping the active reasoning state grounded in the original prompt.

### 3. Failure Modes: Semantic Graph Drift & Attractor Traps
* **Attractor Traps**: If the model predicts an incorrect intermediate concept $C_{\text{err}}$, this concept is injected back into the graph. GNN message passing will immediately propagate this erroneous activation to its neighbors. The "working memory" shifts its center of gravity to an incorrect region of the graph.
* **Irreversible Topography**: Because path generation is strictly bounded by the topological transition mask, once the model enters an incorrect sub-graph, it is trapped. It cannot "teleport" back to the correct path if no edge exists between the current erroneous state and the correct target state.
* **Gradient Decay in the Loop**: Backpropagating through the feedback loop (BPTT over GNN updates) makes the model highly susceptible to vanishing gradients, making it difficult for the network to learn long-term conceptual dependencies during training.

### 4. Scalability of the Iterative Loop
* **Memory Footprint ($O(1)$ Scaling)**: Unlike Transformers, where the KV Cache memory footprint grows linearly $O(T)$ with the sequence length (leading to massive memory requirements at long contexts), CAT V2's working memory size is static: $\text{Memory} = O(|V| \cdot d)$ which remains constant regardless of the planning path length. This represents an enormous scalability advantage for running long-chain reasoning on edge hardware.
* **Computation ($O(E)$ Sparsity)**: Each iteration requires a GNN forward pass. For a sparse graph where the number of edges $E \ll |V|^2$, GNN propagation scales as $O(E \cdot d)$. This is highly compute-efficient compared to the $O(T^2)$ self-attention cost in transformers.
* **Training Bottleneck**: While inference scales exceptionally well, training is the scalability bottleneck. BPTT through recurrent GNN states prevents parallelization across time steps, unlike the parallel training capability of transformers.

### 5. Representation Limits: Branching, Recursion, and Hierarchies
* **Branching (Limitation)**: A single autoregressive loop produces a linear path. If the code logic requires branching (e.g., `if-else` execution paths), a single-path CAT decoder fails. Representing branches requires tracking multiple active concept paths in memory simultaneously (e.g., maintaining a multi-modal activation distribution or using Beam Search decoding).
* **Recursion (Limitation)**: While the concept graph can represent cycles topologically, the path decoder does not have a stack. Without a symbolic stack memory (like a Pushdown Automaton), recursive CAT V2 cannot perform nested recursion (e.g., tracking variable scope or nested function calls).
* **Hierarchical Reasoning (VLCM Extension)**: By utilizing a directed acyclic graph (DAG) structure (as implemented in VLCM), GNN message passing can propagate activations vertically (general-to-specific). The model can successfully perform hierarchical planning (e.g., planning the abstract concept `Data Search` and refining it in subsequent loops to `Regex Match`).

### 6. Emergent Reasoning via Repeated Semantic Refinement
* **Noise Filtering**: When initialized with noisy activation inputs (System 1 errors), GNN message-passing loops act as a cleanup memory. Through repeated propagation iterations, the GNN weights act as a mathematical attractor, pulling noisy activations toward stable "concept basins" (correct nodes) before path generation begins.
* **Iterative Planning Refinement**: By feeding the predicted concept back, the model can perform semantic revision. The context projection vector continuously guides the graph update, allowing the model to refine its plan dynamically as it gathers more "steps" in its working memory.

---

## 🛠️ CLI Quick Start

### 1. Run Verification & Tests
Ensure the environment and concept propagation logic are healthy:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

### 2. Grow Graph from Documentation
Compile GNN transition edges based on documentation sentence co-occurrences:
```powershell
.\.venv\Scripts\python.exe run_reasoning.py grow-graph --dataset data/python_coding_dataset.json --documents data/python_docs.txt --output data/python_coding_graph.json --png reports/python_coding_graph.png
```

### 3. Train the Domain Models
Train models on structural, CFD, or python coding reasoning datasets:
```powershell
# Train Python Coding AI on Grown Graph (30 Epochs)
.\.venv\Scripts\python.exe run_reasoning.py train --dataset data/python_coding_dataset.json --checkpoint-dir checkpoints/cat_v2_python_coding --graph-file data/python_coding_graph.json --epochs 30

# Train Structural Engineering (30 Epochs)
.\.venv\Scripts\python.exe run_reasoning.py train --dataset data/structural_reasoning_dataset.json --checkpoint-dir checkpoints/cat_v2_structural --epochs 30

# Train CFD Reasoning (3 Epochs)
.\.venv\Scripts\python.exe run_reasoning.py train --dataset data/reasoning_dataset.json --checkpoint-dir checkpoints/cat_v2 --epochs 3
```

### 4. Path Inference
Query a trained checkpoint to generate a planning path and solution:
```powershell
.\.venv\Scripts\python.exe run_reasoning.py infer --dataset data/python_coding_dataset.json --checkpoint-dir checkpoints/cat_v2_python_coding --graph-file data/python_coding_graph.json --question "How to filter a list of numbers to find even numbers?"
```

### 5. Run the Benchmarks
Execute the theoretical and empirical profiling benchmarks:
```powershell
.\.venv\Scripts\python.exe vlcm/run_vlcm.py benchmark
```

---

## 🖥️ Interactive Lab GUI

The project features a responsive dark-mode GUI to inspect the step-by-step next-concept predictions, graph nodes, active activations, and auto-complete outputs.

### Starting the GUI Server:
```powershell
.\.venv\Scripts\python.exe gui_server.py
```
Open your browser and navigate to:
👉 **[http://localhost:8080/](http://localhost:8080/)**

### GUI Capabilities:
* **Domain Checkpoint Selector**: Switch between `Python Coding AI`, `Structural Engineering`, and `CFD` on the fly.
* **Probabilistic Path Builder**: Shows all mathematically valid next concepts along with active probability bars.
* **Interactive Node Network**: Real-time layout highlighting the active concept nodes and current planning paths.
