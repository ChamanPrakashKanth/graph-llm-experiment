# =============================================================================
# graph_builder.py
# Dynamic Concept Graph Builder
# =============================================================================

import torch
import torch.nn.functional as F


class DynamicGraphBuilder:
    """
    Build graph using cosine similarity between concepts.

    Produces:
        edge_index [2, num_edges]
    """

    def __init__(
        self,
        top_k=8,
        similarity_threshold=0.25,
        self_loops=False
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.self_loops = self_loops

    # =========================================================================
    # Cosine Similarity
    # =========================================================================

    def compute_similarity_matrix(
        self,
        concept_vectors
    ):
        """
        concept_vectors:
            [num_concepts, dim]
        """

        concept_vectors = F.normalize(
            concept_vectors,
            p=2,
            dim=-1
        )

        similarity = torch.matmul(
            concept_vectors,
            concept_vectors.T
        )

        return similarity

    # =========================================================================
    # Single Graph
    # =========================================================================

    def build_graph(
        self,
        concept_vectors
    ):
        """
        concept_vectors:
            [num_concepts, dim]

        returns:
            edge_index
        """

        similarity = self.compute_similarity_matrix(
            concept_vectors
        )

        num_concepts = concept_vectors.size(0)

        edge_list = []

        for source in range(num_concepts):

            sim_scores = similarity[source].clone()

            if not self.self_loops:
                sim_scores[source] = -1.0

            top_values, top_indices = torch.topk(
                sim_scores,
                k=min(
                    self.top_k,
                    num_concepts
                )
            )

            for target, score in zip(
                top_indices,
                top_values
            ):

                if score < self.similarity_threshold:
                    continue

                edge_list.append(
                    [source, target.item()]
                )

        if len(edge_list) == 0:

            edge_index = torch.tensor(
                [[0], [0]],
                dtype=torch.long
            )

        else:

            edge_index = torch.tensor(
                edge_list,
                dtype=torch.long
            ).T.contiguous()

        return edge_index

    # =========================================================================
    # Batched Graph
    # =========================================================================

    def build_batch_graph(
        self,
        concept_batch
    ):
        """
        concept_batch:
            [batch, num_concepts, dim]

        returns:
            edge_index
        """

        batch_size = concept_batch.size(0)
        num_concepts = concept_batch.size(1)

        all_edges = []

        offset = 0

        for batch_id in range(batch_size):

            graph = self.build_graph(
                concept_batch[batch_id]
            )

            graph = graph + offset

            all_edges.append(graph)

            offset += num_concepts

        edge_index = torch.cat(
            all_edges,
            dim=1
        )

        return edge_index


# =============================================================================
# Advanced Graph Builder
# =============================================================================

class MultiRelationGraphBuilder:
    """
    Future-ready graph builder supporting:

    similarity edges
    temporal edges
    co-occurrence edges
    dependency edges
    """

    def __init__(
        self,
        top_k=8,
        threshold=0.25
    ):
        self.top_k = top_k
        self.threshold = threshold

    # =========================================================================

    def build_similarity_edges(
        self,
        concept_vectors
    ):

        builder = DynamicGraphBuilder(
            top_k=self.top_k,
            similarity_threshold=self.threshold
        )

        return builder.build_graph(
            concept_vectors
        )

    # =========================================================================

    def build_temporal_edges(
        self,
        num_concepts
    ):

        edges = []

        for i in range(num_concepts - 1):

            edges.append([i, i + 1])
            edges.append([i + 1, i])

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).T

        return edge_index

    # =========================================================================

    def merge_edges(
        self,
        *edge_sets
    ):

        merged = torch.cat(
            edge_sets,
            dim=1
        )

        merged = torch.unique(
            merged,
            dim=1
        )

        return merged

    # =========================================================================

    def build_graph(
        self,
        concept_vectors
    ):

        similarity_edges = (
            self.build_similarity_edges(
                concept_vectors
            )
        )

        temporal_edges = (
            self.build_temporal_edges(
                concept_vectors.size(0)
            )
        )

        edge_index = self.merge_edges(
            similarity_edges,
            temporal_edges
        )

        return edge_index


# =============================================================================
# Utility Functions
# =============================================================================

def graph_density(
    edge_index,
    num_nodes
):
    """
    Graph density metric
    """

    num_edges = edge_index.size(1)

    max_edges = num_nodes * (num_nodes - 1)

    return num_edges / max_edges


def graph_statistics(
    edge_index,
    num_nodes
):

    density = graph_density(
        edge_index,
        num_nodes
    )

    degrees = torch.bincount(
        edge_index[0],
        minlength=num_nodes
    )

    return {
        "num_nodes": num_nodes,
        "num_edges": edge_index.size(1),
        "density": density,
        "avg_degree": degrees.float().mean().item(),
        "max_degree": degrees.max().item()
    }


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":

    batch_size = 4
    num_concepts = 64
    dim = 512

    concepts = torch.randn(
        batch_size,
        num_concepts,
        dim
    )

    builder = DynamicGraphBuilder(
        top_k=8,
        similarity_threshold=0.20
    )

    edge_index = builder.build_batch_graph(
        concepts
    )

    print(edge_index.shape)