"""Integrated CAT V3 model architecture."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from cat_v3.encoder import TinyEncoder
from cat_v3.router import SemanticRouter
from cat_v3.experts import GATExpert
from cat_v3.fusion import ConceptFusionLayer
from cat_v3.combiner import TinyCombiner
from cat_v3.decoder import TinyDecoder
from cat_v3.dataset import DOMAINS


class ConceptMemory(nn.Module):
    """Shared concept memory embeddings table."""

    def __init__(self, num_concepts: int, concept_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_concepts, concept_dim)

    def forward(self) -> torch.Tensor:
        """Returns all concept embeddings."""
        ids = torch.arange(self.embeddings.num_embeddings, device=self.embeddings.weight.device)
        return self.embeddings(ids)


class CATV3Model(nn.Module):
    """Concept Attention Transformer V3 (CAT V3) integrated model."""

    def __init__(
        self,
        num_concepts: int,
        tokenizer_vocab_size: int,
        pad_id: int,
        eos_id: int,
        expert_graphs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        concept_dim: int = 128,
        hidden_size: int = 128,
        path_length: int = 8,
        top_m: int = 8,
        decoder_vocab_size: int = 500,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.path_length = path_length
        self.top_m = top_m

        # 1. Tiny Encoder
        self.encoder = TinyEncoder(vocab_size=tokenizer_vocab_size, hidden_size=hidden_size)

        # 2. Semantic Router
        self.router = SemanticRouter(encoder_dim=hidden_size, num_experts=len(DOMAINS))

        # 3. Concept Memory
        self.memory = ConceptMemory(num_concepts=num_concepts, concept_dim=concept_dim)
        
        # Project encoder representation into concept space if dimensions differ
        self.query_proj = nn.Linear(hidden_size, concept_dim)

        # 4. GAT Experts registered as a ModuleDict
        self.experts = nn.ModuleDict()
        for domain in DOMAINS:
            edge_index, edge_weight = expert_graphs[domain]
            self.experts[domain] = GATExpert(
                domain_name=domain,
                num_concepts=num_concepts,
                concept_dim=concept_dim,
                edge_index=edge_index,
                edge_weight=edge_weight,
                pad_id=pad_id,
                eos_id=eos_id,
                path_length=path_length,
            )

        # 5. Concept Fusion Layer
        self.fusion = ConceptFusionLayer(num_concepts=num_concepts, pad_id=pad_id, eos_id=eos_id, top_m=top_m)

        # 6. Tiny Combiner
        self.combiner = TinyCombiner(concept_dim=concept_dim)

        # 7. Tiny Decoder
        self.decoder = TinyDecoder(
            vocab_size=decoder_vocab_size,
            concept_dim=concept_dim,
            hidden_size=hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_paths: Optional[torch.Tensor] = None,
        target_responses: Optional[torch.Tensor] = None,
        target_responses_mask: Optional[torch.Tensor] = None,
        router_top_k: int = 2,
        router_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Runs the complete forward pass of CAT V3.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            target_paths: [batch_size, path_length] (optional)
            target_responses: [batch_size, resp_len] (optional)
            target_responses_mask: [batch_size, resp_len] (optional)
            router_top_k: Minimum number of experts to activate
            router_threshold: Activation threshold for experts
            
        Returns:
            Dict containing outputs of all sub-components.
        """
        # Step 1: Tiny Encoder
        query_emb = self.encoder(input_ids, attention_mask)
        
        # Step 2: Semantic Router
        router_out = self.router(query_emb, top_k=router_top_k, threshold=router_threshold)
        router_logits = router_out["logits"]
        router_probs = router_out["probs"]
        router_mask = router_out["activation_mask"]

        # Step 3: Global Concept Embeddings
        global_embs = self.memory()
        query_context = self.query_proj(query_emb)

        # Step 4: Graph Mixture of Experts (Graph-MoE)
        expert_reports = {}
        for domain in DOMAINS:
            expert = self.experts[domain]
            # Pass through expert
            expert_reports[domain] = expert(
                global_embeddings=global_embs,
                query_context=query_context,
                target_paths=target_paths,
            )

        # Step 5: Concept Fusion Layer
        fusion_out = self.fusion(
            expert_reports=expert_reports,
            router_probs=router_probs,
            router_mask=router_mask,
            global_embeddings=global_embs,
            domain_names=DOMAINS,
        )
        fused_embs = fusion_out["fused_embeddings"]

        # Step 6: Tiny Combiner
        organized_embs = self.combiner(fused_embs)

        # Step 7: Tiny Decoder
        decoder_logits = None
        if target_responses is not None:
            # We predict next tokens, so shift targets during training
            decoder_logits = self.decoder(
                organized_embeddings=organized_embs,
                target_ids=target_responses,
                target_mask=target_responses_mask,
            )

        return {
            "query_embedding": query_emb,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "router_mask": router_mask,
            "expert_reports": expert_reports,
            "fused_concept_ids": fusion_out["fused_concept_ids"],
            "fused_scores": fusion_out["fused_scores"],
            "fused_embeddings": fused_embs,
            "organized_embeddings": organized_embs,
            "decoder_logits": decoder_logits,
        }

    @torch.no_grad()
    def generate_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 32,
        router_top_k: int = 2,
        router_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Runs inference to generate a reasoning graph and natural language answer."""
        # 1. Forward pass to get organized embeddings
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            router_top_k=router_top_k,
            router_threshold=router_threshold,
        )
        
        organized_embs = outputs["organized_embeddings"]
        router_mask = outputs["router_mask"]
        expert_reports = outputs["expert_reports"]

        # 2. Autoregressive token generation
        gen_tokens = self.decoder.generate(
            organized_embeddings=organized_embs,
            max_length=max_length,
            start_id=self.pad_id,
            eos_id=self.eos_id,
        )

        return {
            "router_probs": outputs["router_probs"],
            "router_mask": router_mask,
            "expert_reports": expert_reports,
            "fused_concept_ids": outputs["fused_concept_ids"],
            "generated_tokens": gen_tokens,
        }
