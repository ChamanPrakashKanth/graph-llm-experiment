"""Unit tests for the CAT V3 model architecture."""

from __future__ import annotations

import unittest
import torch

from cat_v3.encoder import TinyEncoder
from cat_v3.router import SemanticRouter
from cat_v3.experts import GATExpert
from cat_v3.fusion import ConceptFusionLayer
from cat_v3.combiner import TinyCombiner
from cat_v3.decoder import TinyDecoder
from cat_v3.model import CATV3Model, ConceptMemory


class TestCATV3Architecture(unittest.TestCase):
    """Verifies internal shapes, masks, and forward passes of CAT V3 sub-modules."""

    def setUp(self) -> None:
        self.num_concepts = 20
        self.concept_dim = 64
        self.hidden_size = 64
        self.pad_id = 0
        self.eos_id = 1
        
        # Build mock expert graph: sequential edges
        sources = list(range(self.num_concepts - 1))
        targets = list(range(1, self.num_concepts))
        self.edge_index = torch.tensor([sources, targets], dtype=torch.long)
        self.edge_weight = torch.ones(len(sources), dtype=torch.float)

    def test_encoder(self) -> None:
        encoder = TinyEncoder(vocab_size=50, hidden_size=self.hidden_size)
        input_ids = torch.randint(0, 49, (2, 10))
        mask = torch.ones((2, 10), dtype=torch.long)
        
        out = encoder(input_ids, mask)
        self.assertEqual(out.shape, (2, self.hidden_size))

    def test_router(self) -> None:
        router = SemanticRouter(encoder_dim=self.hidden_size, num_experts=6)
        query = torch.randn(2, self.hidden_size)
        
        out = router(query, top_k=2, threshold=0.5)
        self.assertEqual(out["logits"].shape, (2, 6))
        self.assertEqual(out["probs"].shape, (2, 6))
        self.assertEqual(out["activation_mask"].shape, (2, 6))
        self.assertEqual(out["activation_mask"].dtype, torch.bool)

    def test_gat_expert(self) -> None:
        expert = GATExpert(
            domain_name="test_domain",
            num_concepts=self.num_concepts,
            concept_dim=self.concept_dim,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            path_length=5
        )
        global_embs = torch.randn(self.num_concepts, self.concept_dim)
        query_context = torch.randn(2, self.concept_dim)
        
        out = expert(global_embs, query_context)
        self.assertEqual(out["predicted_path"].shape, (2, 5))
        self.assertEqual(out["path_logits"].shape, (2, 5, self.num_concepts))
        self.assertEqual(out["path_scores"].shape, (2, 5))
        self.assertEqual(out["node_states"].shape, (self.num_concepts, self.concept_dim))

    def test_concept_fusion(self) -> None:
        fusion = ConceptFusionLayer(
            num_concepts=self.num_concepts,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            top_m=4
        )
        
        # Mock reports
        expert_reports = {
            "mechanical": {
                "predicted_path": torch.tensor([[2, 3, 4, 1, 1], [5, 6, 7, 1, 1]]),
                "path_scores": torch.zeros(2, 5), # logprob 0 -> prob 1
                "node_states": torch.randn(self.num_concepts, self.concept_dim)
            },
            "physics": {
                "predicted_path": torch.tensor([[3, 4, 8, 1, 1], [6, 7, 9, 1, 1]]),
                "path_scores": torch.zeros(2, 5),
                "node_states": torch.randn(self.num_concepts, self.concept_dim)
            }
        }
        
        router_probs = torch.tensor([[0.9, 0.8], [0.1, 0.95]])
        router_mask = torch.tensor([[True, True], [False, True]])
        global_embs = torch.randn(self.num_concepts, self.concept_dim)
        
        out = fusion(
            expert_reports=expert_reports,
            router_probs=router_probs,
            router_mask=router_mask,
            global_embeddings=global_embs,
            domain_names=["mechanical", "physics"]
        )
        
        self.assertEqual(out["fused_concept_ids"].shape, (2, 4))
        self.assertEqual(out["fused_embeddings"].shape, (2, 4, self.concept_dim))
        self.assertEqual(out["fused_scores"].shape, (2, self.num_concepts))
        
        # Verify pad/eos scores are 0
        self.assertEqual(out["fused_scores"][0, self.pad_id].item(), 0.0)
        self.assertEqual(out["fused_scores"][0, self.eos_id].item(), 0.0)

    def test_combiner(self) -> None:
        combiner = TinyCombiner(concept_dim=self.concept_dim)
        fused = torch.randn(2, 5, self.concept_dim)
        
        out = combiner(fused)
        self.assertEqual(out.shape, (2, 5, self.concept_dim))

    def test_decoder(self) -> None:
        decoder = TinyDecoder(vocab_size=30, concept_dim=self.concept_dim, hidden_size=self.hidden_size)
        organized = torch.randn(2, 4, self.concept_dim)
        target_ids = torch.randint(0, 29, (2, 8))
        
        logits = decoder(organized, target_ids)
        self.assertEqual(logits.shape, (2, 8, 30))
        
        gen = decoder.generate(organized, max_length=10, start_id=self.pad_id, eos_id=self.eos_id)
        self.assertEqual(gen.size(0), 2)
        self.assertLessEqual(gen.size(1), 10)

    def test_end_to_end_model(self) -> None:
        from cat_v3.dataset import DOMAINS
        expert_graphs = {
            dom: (self.edge_index, self.edge_weight) for dom in DOMAINS
        }
        
        model = CATV3Model(
            num_concepts=self.num_concepts,
            tokenizer_vocab_size=40,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            expert_graphs=expert_graphs,
            concept_dim=self.concept_dim,
            hidden_size=self.hidden_size,
            path_length=6,
            top_m=5,
            decoder_vocab_size=30
        )
        
        input_ids = torch.randint(0, 39, (2, 10))
        mask = torch.ones((2, 10), dtype=torch.long)
        target_paths = torch.randint(0, self.num_concepts - 1, (2, 6))
        target_responses = torch.randint(0, 29, (2, 12))
        
        # Test training forward pass
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            target_paths=target_paths,
            target_responses=target_responses
        )
        self.assertEqual(out["decoder_logits"].shape, (2, 12, 30))
        self.assertEqual(out["router_logits"].shape, (2, len(DOMAINS)))
        
        # Test generate pass
        gen_out = model.generate_response(input_ids, mask, max_length=15)
        self.assertEqual(gen_out["generated_tokens"].size(0), 2)
        self.assertLessEqual(gen_out["generated_tokens"].size(1), 15)


if __name__ == "__main__":
    unittest.main()
