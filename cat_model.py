# =============================================================================
# cat_model.py
# Concept Attention Transformer (CAT)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from torch_geometric.nn import GATv2Conv


# =============================================================================
# Graph Encoder
# =============================================================================

class ConceptGraphEncoder(nn.Module):
    """
    Graph Attention Network for concept relationships
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gat1 = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout
        )

        self.gat2 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=1,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.gelu(x)

        return x


# =============================================================================
# Concept Attention Layer
# =============================================================================

class ConceptAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attn_out, attn_weights = self.attn(
            x,
            x,
            x
        )

        x = x + self.dropout(attn_out)
        x = self.norm(x)

        return x, attn_weights


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)

        x = self.norm1(
            x + self.dropout(attn_out)
        )

        ff_out = self.ff(x)

        x = self.norm2(
            x + self.dropout(ff_out)
        )

        return x


# =============================================================================
# Main CAT Model
# =============================================================================

class ConceptAttentionTransformer(nn.Module):

    def __init__(
        self,
        encoder_name="bert-base-uncased",
        concept_dim=512,
        hidden_dim=768,
        num_concepts=64,
        num_transformer_layers=4,
        num_classes=50000,
        dropout=0.1
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Sentence Encoder
        # ---------------------------------------------------------

        self.encoder = AutoModel.from_pretrained(
            encoder_name
        )

        encoder_dim = self.encoder.config.hidden_size

        # ---------------------------------------------------------
        # Concept Projection
        # ---------------------------------------------------------

        self.concept_projection = nn.Linear(
            encoder_dim,
            concept_dim
        )

        self.num_concepts = num_concepts
        self.concept_dim = concept_dim

        # learnable concept memory

        self.concept_memory = nn.Parameter(
            torch.randn(
                num_concepts,
                concept_dim
            )
        )

        # ---------------------------------------------------------
        # Graph Encoder
        # ---------------------------------------------------------

        self.graph_encoder = ConceptGraphEncoder(
            input_dim=concept_dim,
            hidden_dim=hidden_dim
        )

        # ---------------------------------------------------------
        # Concept Attention
        # ---------------------------------------------------------

        self.concept_attention = ConceptAttentionLayer(
            hidden_dim=hidden_dim
        )

        # ---------------------------------------------------------
        # Transformer Stack
        # ---------------------------------------------------------

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                ff_dim=hidden_dim * 4,
                num_heads=8
            )
            for _ in range(num_transformer_layers)
        ])

        # ---------------------------------------------------------
        # Prediction Head
        # ---------------------------------------------------------

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    # =====================================================================
    # Sentence Encoding
    # =====================================================================

    def encode_sentences(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0]

        return cls_embedding

    # =====================================================================
    # Concept Construction
    # =====================================================================

    def build_concepts(
        self,
        sentence_embeddings
    ):
        """
        sentence_embeddings:
        [batch, hidden]
        """

        projected = self.concept_projection(
            sentence_embeddings
        )

        projected = projected.unsqueeze(1)

        concept_memory = self.concept_memory.unsqueeze(0)

        concept_vectors = (
            projected * concept_memory
        )

        return concept_vectors

    # =====================================================================
    # Forward
    # =====================================================================

    def forward(
        self,
        input_ids,
        attention_mask,
        edge_index
    ):

        batch_size = input_ids.size(0)

        # ---------------------------------------------------------
        # Sentence Encoder
        # ---------------------------------------------------------

        sentence_embeddings = self.encode_sentences(
            input_ids,
            attention_mask
        )

        # ---------------------------------------------------------
        # Concept Construction
        # ---------------------------------------------------------

        concept_vectors = self.build_concepts(
            sentence_embeddings
        )

        # ---------------------------------------------------------
        # Flatten for GNN
        # ---------------------------------------------------------

        x = concept_vectors.reshape(
            batch_size * self.num_concepts,
            self.concept_dim
        )

        # ---------------------------------------------------------
        # Graph Encoder
        # ---------------------------------------------------------

        graph_out = self.graph_encoder(
            x,
            edge_index
        )

        graph_out = graph_out.reshape(
            batch_size,
            self.num_concepts,
            -1
        )

        # ---------------------------------------------------------
        # Concept Attention
        # ---------------------------------------------------------

        graph_out, attention_weights = (
            self.concept_attention(graph_out)
        )

        # ---------------------------------------------------------
        # Transformer Layers
        # ---------------------------------------------------------

        hidden = graph_out

        for layer in self.transformer_layers:
            hidden = layer(hidden)

        # ---------------------------------------------------------
        # Pool Concepts
        # ---------------------------------------------------------

        pooled = hidden.mean(dim=1)

        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------

        logits = self.prediction_head(
            pooled
        )

        return {
            "logits": logits,
            "sentence_embeddings": sentence_embeddings,
            "concept_vectors": concept_vectors,
            "graph_output": graph_out,
            "attention_weights": attention_weights
        }