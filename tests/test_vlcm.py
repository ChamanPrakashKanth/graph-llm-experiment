import json
import shutil
import sys
import uuid
import unittest
from contextlib import contextmanager
from pathlib import Path

# Fix Windows stdout CP1252 encoding issues with Unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader

from reasoning_dataset import ReasoningCollator, ReasoningDataset
from vlcm.concept_memory import build_default_concept_memory
from vlcm.graph_reasoner import ReasoningGraph, GraphNeuralMemory
from vlcm.path_generator import ConceptReasoningTransformer, VLCMModel, build_vlcm_model_from_dataset
from vlcm.reasoning_loss import VLCMReasoningLoss, compute_path_metrics
from vlcm.trainer import VLCMTrainer, load_vlcm_checkpoint, set_seed
from vlcm.decoder import VLCMReasoningSystem, show_reasoning_path
from vlcm.evaluator import run_engineering_domain_test


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


def tiny_vlcm_model(dataset: ReasoningDataset):
    return build_vlcm_model_from_dataset(
        dataset,
        concept_dim=32,
        hidden_size=32,
        path_length=dataset.max_path_length,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        graph_layers=1,
        top_k=3,
        dropout=0.0,
    )


class VLCMConceptMemoryTests(unittest.TestCase):
    def test_hierarchy_and_multi_parent_activation_propagation(self):
        mem = build_default_concept_memory()
        
        # Test existence and hierarchy
        node = mem.get_concept("Thermal Stress")
        self.assertIsNotNone(node)
        self.assertIn("Stress", node.parents)
        self.assertIn("Thermal Expansion", node.parents)
        
        # Test children mapping
        stress_node = mem.get_concept("Stress")
        self.assertIn("Thermal Stress", stress_node.children)
        
        # Test activation and propagation
        mem.reset_activations()
        mem.set_activation("Thermal Expansion", 1.0)
        self.assertEqual(mem.get_activation("Thermal Stress"), 0.0)
        
        # Propagate: child should receive activation from parent with downward_decay (0.5)
        mem.propagate_activations(iterations=1)
        self.assertGreater(mem.get_activation("Thermal Stress"), 0.0)
        self.assertEqual(mem.get_activation("Thermal Stress"), 0.5)


class VLCMGraphNeuralMemoryTests(unittest.TestCase):
    def test_learnable_edge_weights_message_passing(self):
        with tempdir() as tmp:
            dataset = ReasoningDataset(write_dataset(Path(tmp)), max_length=16, max_path_length=6)
            
            # Create a learnable graph neural memory layer
            g_mem = GraphNeuralMemory(
                num_concepts=dataset.vocab.size(),
                concept_dim=32,
                graph=dataset.graph,
                vocab=dataset.vocab,
                num_layers=1,
            )
            
            # Confirm edge mask allows propagation
            self.assertTrue(g_mem.edge_mask[dataset.vocab.concept_to_id["Friction"], dataset.vocab.concept_to_id["Pressure"]])
            self.assertFalse(g_mem.edge_mask[dataset.vocab.concept_to_id["Stress"], dataset.vocab.concept_to_id["Pressure"]])
            
            # Forward pass over states
            x = torch.randn(1, dataset.vocab.size(), 32)
            out = g_mem(x)
            self.assertEqual(out.shape, (1, dataset.vocab.size(), 32))


class VLCMModelTests(unittest.TestCase):
    def test_vlcm_transformer_decoder_and_loss(self):
        with tempdir() as tmp:
            set_seed(42)
            dataset = ReasoningDataset(write_dataset(Path(tmp)), max_length=16, max_path_length=6)
            batch = ReasoningCollator()([dataset[0], dataset[1]])
            
            model = tiny_vlcm_model(dataset)
            
            # Forward with teacher forcing
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                target_paths=batch["path_ids"],
            )
            self.assertEqual(outputs["path_logits"].shape, (2, 6, dataset.vocab.size()))
            self.assertEqual(outputs["predicted_path"].shape, (2, 6))
            
            # Check loss
            loss_fn = VLCMReasoningLoss(pad_id=dataset.vocab.pad_id)
            loss, metrics = loss_fn(
                outputs,
                batch["path_ids"],
                batch["activation_targets"],
                transition_mask=model.transition_mask,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(metrics["path_loss"], 0.0)
            
            # Backward pass on parameters, including learnable edges
            loss.backward()
            grad = model.graph_reasoner.edge_importance.grad
            self.assertIsNotNone(grad)


class VLCMTrainerIntegrationTests(unittest.TestCase):
    def test_vlcm_trainer_pipeline(self):
        with tempdir() as tmp:
            set_seed(123)
            root = Path(tmp)
            dataset = ReasoningDataset(write_dataset(root), max_length=16, max_path_length=6)
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=ReasoningCollator(),
            )
            model = tiny_vlcm_model(dataset)
            trainer = VLCMTrainer(
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
            
            checkpoint = root / "checkpoints" / "vlcm_epoch_1.pt"
            self.assertTrue(checkpoint.exists())
            
            loaded = load_vlcm_checkpoint(checkpoint, device="cpu")
            system = VLCMReasoningSystem(
                loaded["model"],
                loaded["vocab"],
                loaded["tokenizer"],
                device="cpu",
            )
            result = system.answer("Why does pressure drop?", max_length=16)
            self.assertGreaterEqual(len(result["reasoning_path"]), 1)
            self.assertIn("answer", result)
            
            # Test direct formatter call
            self.assertEqual(show_reasoning_path(["Pressure", "Friction"]), "Pressure\n↓\nFriction")


class VLCMBeamSearchAndEarlyStoppingTests(unittest.TestCase):
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
            model = tiny_vlcm_model(dataset)
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


class VCLMEngineeringDomainTest(unittest.TestCase):
    def test_concept_chain_discovery(self):
        # Verify the Concept Chain Discovery evaluation test runs successfully
        success = run_engineering_domain_test(device="cpu")
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
