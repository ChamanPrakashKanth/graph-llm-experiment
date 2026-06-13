# CAT V2: Concept Attention Transformer

CAT V2 is a research prototype for explicit concept reasoning. It predicts a
reasoning path through a concept graph, then optionally decodes that path into
language.

Traditional LLM:

```text
Question -> Tokens -> Attention -> Tokens -> Answer
```

CAT V2:

```text
Question -> Concept Activation -> Concept Graph -> Reasoning Path -> Answer
```

The reasoning path is the primary output.

## Architecture

```text
Question
  -> Simple offline tokenizer
  -> Tiny transformers encoder
  -> Concept activator
  -> Concept memory
  -> Explicit concept graph
  -> Pure PyTorch graph message passing
  -> Graph-constrained path generator
  -> Reasoning path
  -> Template or optional LLM answer decoder
```

Core files:

- `cat_reasoning_model.py`: transformer encoder, concept activation, graph propagation, path decoder.
- `reasoning_graph.py`: explicit nodes, edges, graph growth, scoring, beam traversal, visualization.
- `reasoning_dataset.py`: dataset schema, tokenizer, concept vocabulary, activation targets.
- `reasoning_loss.py`: supervised path CE, concept activation BCE, transition-prior loss, metrics.
- `reasoning_trainer.py`: training, evaluation, checkpoint saving/loading.
- `answer_decoder.py`: path-first answer generation.
- `run_reasoning.py`: canonical CLI.

## Dataset Schema

```json
{
  "question": "Why does pressure drop in a pipe?",
  "reasoning_path": ["Pressure", "Friction", "Energy Loss", "Pressure Drop"],
  "answer": "Pressure drops because wall friction converts mechanical energy into losses."
}
```

The model trains against `reasoning_path`. It does not create random labels or
random target classes.

## Quick Start

Train a small offline model:

```powershell
.\.venv\Scripts\python.exe run_reasoning.py train --epochs 3
```

Infer a reasoning path:

```powershell
.\.venv\Scripts\python.exe run_reasoning.py infer --question "Why does pressure drop?"
```

Evaluate the latest checkpoint:

```powershell
.\.venv\Scripts\python.exe run_reasoning.py evaluate
```

Visualize the graph:

```powershell
.\.venv\Scripts\python.exe run_reasoning.py visualize --output reports/cat_v2_graph.png
```

Grow the graph from local text documents:

```powershell
.\.venv\Scripts\python.exe run_reasoning.py grow-graph --documents data --output reports/cat_v2_graph.json
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Checkpoint Contract

CAT V2 checkpoints contain:

- `model_state`
- `optimizer_state`
- `config`
- `vocab`
- `graph`
- `tokenizer`
- `epoch`
- `metrics`
- `history`

By default, checkpoints are written to `checkpoints/cat_v2/` and are ignored by
git.

## Model Output Contract

The model returns:

- `question_embedding`
- `activation_logits`
- `activated_concepts`
- `graph_state`
- `path_logits`
- `predicted_path`
- `path_scores`
- `traversal_trace`

## Design Corrections From V1

- No random labels: every training target is a concept path from the dataset.
- No random graphs: the graph is built from supervised path edges and optional document growth.
- No ID-distance losses: path supervision uses cross-entropy; activation uses multi-label BCE.
- No repeated pooled logits: the path decoder is recurrent and graph-constrained.
- Language decoding is downstream only; the graph/path engine is the reasoner.

## Architectural Weaknesses And Future Improvements

- The included dataset is intentionally small, so learned generalization is limited.
- Document graph growth is heuristic exact phrase matching, not robust relation extraction.
- Edges encode supervised or co-occurrence evidence, not causal proof.
- The concept vocabulary is fixed at training time.
- Beam traversal and neural decoding are local search procedures.
- The default answer decoder is template-based; optional LLM quality depends on the supplied model.
- There is no large external benchmark yet for path-level scientific reasoning.

Useful next steps:

- Add a larger curated engineering path corpus.
- Replace heuristic document growth with relation extraction and evidence spans.
- Add uncertainty calibration for concept activations and path scores.
- Support dynamic vocabulary expansion with embedding initialization.
- Evaluate against held-out path and graph-retrieval benchmarks.

