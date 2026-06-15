import json
import shutil
import uuid
from contextlib import contextmanager
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from answer_decoder import CATReasoningSystem
from cat_reasoning_model import build_model_from_dataset
from reasoning_dataset import ReasoningCollator, ReasoningDataset
from reasoning_graph import ReasoningGraph
from reasoning_loss import CATReasoningLoss, compute_path_metrics
from reasoning_trainer import CATReasoningTrainer, load_checkpoint, set_seed


SAMPLES = [
    {
        "question": "Why does pressure drop?",
        "reasoning_path": ["Pressure", "Friction", "Energy Loss", "Pressure Drop"],
        "answer": "Pressure drops because friction causes energy loss.",
    },
    {
        "question": "How does load cause failure?",
        "reasoning_path": ["Load", "Stress", "Strain", "Failure"],
        "answer": "Load creates stress, strain, and failure.",
    },
    {
        "question": "Why does mesh quality affect convergence?",
        "reasoning_path": [
            "Mesh Quality",
            "Numerical Instability",
            "Residual",
            "Convergence Failure",
        ],
        "answer": "Poor mesh quality destabilizes residual convergence.",
    },
]


TEST_TMP_ROOT = Path.cwd() / "cat_v2_test_tmp"


@contextmanager
def tempdir():
    TEST_TMP_ROOT.mkdir(exist_ok=True)
    path = TEST_TMP_ROOT / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_dataset(folder: Path) -> Path:
    path = folder / "reasoning_dataset.json"
    path.write_text(json.dumps(SAMPLES), encoding="utf-8")
    return path


def tiny_model(dataset: ReasoningDataset):
    return build_model_from_dataset(
        dataset,
        concept_dim=32,
        hidden_size=32,
        path_length=dataset.max_path_length,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        graph_layers=1,
        top_k=3,
        dropout=0.0,
    )


class GraphTests(unittest.TestCase):
    def test_graph_growth_traversal_serialization_and_visualization(self):
        with tempdir() as tmp:
            graph = ReasoningGraph.from_paths(sample["reasoning_path"] for sample in SAMPLES)
            self.assertTrue(graph.has_edge("Pressure", "Friction"))
            graph.grow_from_documents(
                ["Pressure and friction cause energy loss. Load creates stress."],
                concepts=["Pressure", "Friction", "Energy Loss", "Load", "Stress"],
            )
            paths = graph.beam_search(["Pressure"], max_depth=4, beam_width=3)
            self.assertGreaterEqual(len(paths), 1)
            self.assertEqual(paths[0][1][0], "Pressure")
            self.assertGreater(graph.path_score(["Pressure", "Friction"]), 0)

            restored = ReasoningGraph.from_dict(graph.to_dict())
            self.assertTrue(restored.has_edge("Pressure", "Friction"))
            output = restored.visualize(Path(tmp) / "graph.png", highlight_path=paths[0][1])
            self.assertTrue(output.exists())


class DatasetTests(unittest.TestCase):
    def test_dataset_vocab_padding_and_collator(self):
        with tempdir() as tmp:
            dataset = ReasoningDataset(write_dataset(Path(tmp)), max_length=16, max_path_length=6)
            item = dataset[0]
            self.assertEqual(item["path_ids"][4].item(), dataset.vocab.eos_id)
            self.assertEqual(item["path_ids"][5].item(), dataset.vocab.pad_id)
            self.assertEqual(dataset.vocab.decode_path(item["path_ids"]), SAMPLES[0]["reasoning_path"])

            batch = ReasoningCollator()([dataset[0], dataset[1]])
            self.assertEqual(batch["input_ids"].shape, (2, 16))
            self.assertEqual(batch["path_ids"].shape, (2, 6))
            self.assertEqual(batch["activation_targets"].shape[1], dataset.vocab.size())


class ModelLossTests(unittest.TestCase):
    def test_forward_loss_and_backward_are_supervised(self):
        with tempdir() as tmp:
            set_seed(7)
            dataset = ReasoningDataset(write_dataset(Path(tmp)), max_length=16, max_path_length=6)
            batch = ReasoningCollator()([dataset[0], dataset[1]])
            model = tiny_model(dataset)
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                target_paths=batch["path_ids"],
            )
            self.assertEqual(outputs["path_logits"].shape, (2, 6, dataset.vocab.size()))
            self.assertEqual(outputs["predicted_path"].shape, (2, 6))
            self.assertIn("traversal_trace", outputs)

            loss_fn = CATReasoningLoss(pad_id=dataset.vocab.pad_id)
            loss, metrics = loss_fn(
                outputs,
                batch["path_ids"],
                batch["activation_targets"],
                transition_mask=model.transition_mask,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(metrics["path_loss"], 0)
            loss.backward()
            grad_norm = model.path_generator.output_head.weight.grad.norm().item()
            self.assertGreater(grad_norm, 0)


class TrainerIntegrationTests(unittest.TestCase):
    def test_training_checkpoint_inference_and_metrics(self):
        with tempdir() as tmp:
            set_seed(13)
            root = Path(tmp)
            dataset = ReasoningDataset(write_dataset(root), max_length=16, max_path_length=6)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=ReasoningCollator(),
            )
            model = tiny_model(dataset)
            trainer = CATReasoningTrainer(
                model=model,
                train_loader=loader,
                vocab=dataset.vocab,
                graph=dataset.graph,
                tokenizer=dataset.tokenizer,
                lr=1e-3,
                checkpoint_dir=root / "checkpoints",
                device="cpu",
            )
            metrics = trainer.fit(epochs=1)
            self.assertIn("eval_concept_f1", metrics)
            checkpoint = root / "checkpoints" / "cat_v2_epoch_1.pt"
            self.assertTrue(checkpoint.exists())

            loaded = load_checkpoint(checkpoint, device="cpu")
            system = CATReasoningSystem(
                loaded["model"],
                loaded["vocab"],
                loaded["tokenizer"],
                device="cpu",
            )
            result = system.answer("Why does pressure drop?", max_length=16)
            self.assertGreaterEqual(len(result["reasoning_path"]), 1)
            self.assertIn("answer", result)

            batch = next(iter(loader))
            with torch.no_grad():
                outputs = loaded["model"](batch["input_ids"], batch["attention_mask"])
            path_metrics = compute_path_metrics(
                outputs["predicted_path"],
                batch["path_ids"],
                pad_id=loaded["vocab"].pad_id,
                eos_id=loaded["vocab"].eos_id,
            )
            self.assertIn("concept_f1", path_metrics)


class CATBeamSearchAndEarlyStoppingTests(unittest.TestCase):
    def test_beam_search_and_early_stopping(self):
        with tempdir() as tmp:
            root = Path(tmp)
            dataset = ReasoningDataset(write_dataset(root), max_length=16, max_path_length=6)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=ReasoningCollator(),
            )
            model = tiny_model(dataset)
            model.eval()

            for batch in loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                bsz = input_ids.size(0)
                
                with torch.no_grad():
                    out_greedy = model(input_ids, attention_mask, beam_width=1)
                
                self.assertIn("predicted_path", out_greedy)
                self.assertEqual(out_greedy["predicted_path"].shape, (bsz, 6))
                
                with torch.no_grad():
                    out_beam = model(input_ids, attention_mask, beam_width=3)
                
                self.assertIn("predicted_path", out_beam)
                self.assertEqual(out_beam["predicted_path"].shape, (bsz, 6))
                
                for path in out_beam["predicted_path"]:
                    for cid in path.tolist():
                        self.assertTrue(0 <= cid < dataset.vocab.size())


if __name__ == "__main__":
    unittest.main()
