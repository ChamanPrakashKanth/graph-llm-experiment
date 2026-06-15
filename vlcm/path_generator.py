"""Concept Reasoning Transformer and VLCMModel architecture for VLCM.

Implements the causal Transformer Decoder and the full end-to-end model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning_dataset import ConceptVocabulary
from vlcm.activation_engine import ConceptActivator, TinyTransformerEncoder
from vlcm.graph_reasoner import GraphNeuralMemory, ReasoningGraph, graph_to_tensors


class ConceptReasoningTransformer(nn.Module):
    """Transformer decoder that predicts paths step-by-step over concept spaces."""

    def __init__(
        self,
        concept_dim: int,
        num_concepts: int,
        path_length: int,
        pad_id: int,
        eos_id: int,
        num_layers: int = 2,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.path_length = path_length
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.concept_dim = concept_dim

        # Positional embeddings for path steps
        self.pos_encoder = nn.Embedding(path_length + 1, concept_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=concept_dim,
            nhead=nhead,
            dim_feedforward=concept_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Linear(concept_dim, num_concepts)
        self.start_embedding = nn.Parameter(torch.zeros(concept_dim))

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Create upper triangular causal mask to prevent attending to future steps."""
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        graph_state: torch.Tensor,
        question_context: torch.Tensor,
        activation_logits: torch.Tensor,
        transition_mask: torch.Tensor,
        first_step_mask: torch.Tensor,
        target_paths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_concepts, concept_dim = graph_state.shape
        device = graph_state.device

        # Reshape question context for cross-attention key/values
        # Shape: B x 1 x D
        memory = question_context.unsqueeze(1)

        if target_paths is not None:
            # Training Mode: Teacher Forcing
            inputs_list = []
            
            # Step 0 input is always start_embedding
            prev_emb = self.start_embedding.unsqueeze(0).expand(batch_size, -1)
            inputs_list.append(prev_emb)

            # Accumulate target embeddings for remaining steps
            for step in range(self.path_length - 1):
                gold_ids = target_paths[:, step].clone()
                gold_ids = torch.where(
                    gold_ids == self.pad_id,
                    torch.full_like(gold_ids, self.eos_id),
                    gold_ids,
                )
                gold_ids = gold_ids.clamp(0, num_concepts - 1)
                
                prev_emb = graph_state[
                    torch.arange(batch_size, device=device),
                    gold_ids,
                ]
                inputs_list.append(prev_emb)

            decoder_input = torch.stack(inputs_list, dim=1)  # B x L x D

            # Add positional embeddings
            pos_ids = torch.arange(self.path_length, device=device).unsqueeze(0).expand(batch_size, -1)
            decoder_input = decoder_input + self.pos_encoder(pos_ids)

            # Run Transformer Decoder with causal mask
            tgt_mask = self.generate_causal_mask(self.path_length, device)
            decoded = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)

            # Predict concept logits
            logits = self.output_head(decoded) + 0.25 * activation_logits.unsqueeze(1)

            # Apply strict transition masks step-by-step
            masked_logits_list = []
            for step in range(self.path_length):
                step_logits = logits[:, step]
                if step == 0:
                    allowed = first_step_mask.unsqueeze(0).expand(batch_size, -1)
                else:
                    prev_ids = target_paths[:, step - 1].clone()
                    prev_ids = torch.where(
                        prev_ids == self.pad_id,
                        torch.full_like(prev_ids, self.eos_id),
                        prev_ids,
                    )
                    prev_ids = prev_ids.clamp(0, num_concepts - 1)
                    allowed = transition_mask[prev_ids]
                
                step_logits = step_logits.masked_fill(~allowed, -1.0e4)
                masked_logits_list.append(step_logits)

            path_logits = torch.stack(masked_logits_list, dim=1)
            predicted_path = path_logits.argmax(dim=-1)

            log_probs = F.log_softmax(path_logits, dim=-1)
            path_scores = log_probs.gather(2, predicted_path.unsqueeze(2)).squeeze(2)

        else:
            # Inference Mode: Autoregressive decoding
            prediction_steps = []
            score_steps = []
            logits_steps = []

            prev_ids = torch.full(
                (batch_size,),
                self.eos_id,
                dtype=torch.long,
                device=device,
            )
            inputs_list = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(self.path_length):
                if step == 0:
                    prev_emb = self.start_embedding.unsqueeze(0).expand(batch_size, -1)
                else:
                    prev_emb = graph_state[
                        torch.arange(batch_size, device=device),
                        prev_ids,
                    ]
                
                inputs_list.append(prev_emb)

                # Causal input so far
                decoder_input = torch.stack(inputs_list, dim=1)

                # Add positional encoding
                pos_ids = torch.arange(step + 1, device=device).unsqueeze(0).expand(batch_size, -1)
                decoder_input = decoder_input + self.pos_encoder(pos_ids)

                # Run Transformer Decoder
                tgt_mask = self.generate_causal_mask(step + 1, device)
                decoded = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)

                # Take the last prediction
                last_decoded = decoded[:, -1]

                step_logits = self.output_head(last_decoded) + 0.25 * activation_logits
                if step == 0:
                    allowed = first_step_mask.unsqueeze(0).expand(batch_size, -1)
                else:
                    allowed = transition_mask[prev_ids]

                step_logits = step_logits.masked_fill(~allowed, -1.0e4)
                predicted = step_logits.argmax(dim=-1)

                # Force completed batch items to keep predicting eos_id
                predicted = torch.where(finished, torch.tensor(self.eos_id, device=device), predicted)

                log_probs = F.log_softmax(step_logits, dim=-1)
                scores = log_probs.gather(1, predicted.unsqueeze(1)).squeeze(1)

                finished = finished | (predicted == self.eos_id) | (predicted == self.pad_id)

                logits_steps.append(step_logits)
                prediction_steps.append(predicted)
                score_steps.append(scores)

                if finished.all():
                    # Pad the remaining steps to maintain consistent sequence length shape
                    remaining = self.path_length - len(prediction_steps)
                    for _ in range(remaining):
                        pad_logits = torch.full_like(step_logits, -1.0e4)
                        pad_logits[:, self.eos_id] = 0.0
                        logits_steps.append(pad_logits)
                        prediction_steps.append(torch.full((batch_size,), self.eos_id, dtype=torch.long, device=device))
                        score_steps.append(torch.zeros((batch_size,), device=device))
                    break

                prev_ids = predicted.clamp(0, num_concepts - 1)

            path_logits = torch.stack(logits_steps, dim=1)
            predicted_path = torch.stack(prediction_steps, dim=1)
            path_scores = torch.stack(score_steps, dim=1)

        return {
            "path_logits": path_logits,
            "predicted_path": predicted_path,
            "path_scores": path_scores,
        }


class VLCMModel(nn.Module):
    """Very Large Concepts Model (VLCM) core neural network."""

    def __init__(
        self,
        num_concepts: int,
        tokenizer_vocab_size: int,
        graph: Optional[ReasoningGraph],
        vocab: ConceptVocabulary,
        concept_dim: int = 128,
        path_length: int = 8,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        graph_layers: int = 2,
        top_k: int = 5,
        dropout: float = 0.1,
        pretrained_encoder_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = {
            "num_concepts": num_concepts,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "concept_dim": concept_dim,
            "path_length": path_length,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "graph_layers": graph_layers,
            "top_k": top_k,
            "dropout": dropout,
            "pretrained_encoder_name": pretrained_encoder_name,
        }
        self.vocab_pad_id = vocab.pad_id
        self.vocab_eos_id = vocab.eos_id
        self.top_k = top_k
        self.vocab = vocab

        # Question Encoder
        self.encoder = TinyTransformerEncoder(
            vocab_size=tokenizer_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            pretrained_encoder_name=pretrained_encoder_name,
        )
        if pretrained_encoder_name:
            for p in self.encoder.parameters():
                p.requires_grad = False
        encoder_dim = self.encoder.hidden_size
        self.question_projection = nn.Linear(encoder_dim, concept_dim)

        # Concept Memory Embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, concept_dim)

        # Concept Activator
        self.activator = ConceptActivator(
            encoder_dim=encoder_dim,
            num_concepts=num_concepts,
            special_ids=[vocab.pad_id, vocab.eos_id],
        )

        self.input_norm = nn.LayerNorm(concept_dim)

        # Graph Neural Memory (Learnable Message Passing)
        self.graph_reasoner = GraphNeuralMemory(
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            graph=graph,
            vocab=vocab,
            num_layers=graph_layers,
            dropout=dropout,
        )

        # Concept Reasoning Transformer Decoder
        self.path_generator = ConceptReasoningTransformer(
            concept_dim=concept_dim,
            num_concepts=num_concepts,
            path_length=path_length,
            pad_id=vocab.pad_id,
            eos_id=vocab.eos_id,
            num_layers=2,
            nhead=4,
        )

        # Build buffers for transition mask constraints
        _, transition, first_step = graph_to_tensors(graph, vocab)
        self.register_buffer("transition_mask", transition)
        self.register_buffer("first_step_mask", first_step)

    def set_graph(self, graph: ReasoningGraph, vocab: ConceptVocabulary) -> None:
        """Dynamically update graph transition and propagation matrices."""
        propagation, transition, first_step = graph_to_tensors(
            graph,
            vocab,
            device=self.transition_mask.device,
        )
        self.transition_mask = transition
        self.first_step_mask = first_step
        self.graph_reasoner.edge_mask = (propagation > 0.0) | torch.eye(
            self.config["num_concepts"],
            dtype=torch.bool,
            device=self.transition_mask.device,
        )
        self.graph_reasoner.edge_importance.data.copy_(propagation)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_paths: Optional[torch.Tensor] = None,
        beam_width: int = 1,
    ) -> Dict[str, torch.Tensor]:
        question_embedding = self.encoder(input_ids, attention_mask)
        projected_question = self.question_projection(question_embedding)

        # Get question-based concept activations
        activated = self.activator(question_embedding, top_k=self.top_k)
        activation_probs = activated["activation_probs"]

        # Retrieve concept memory embeddings
        batch_size = input_ids.size(0)
        all_ids = torch.arange(
            self.concept_embeddings.num_embeddings,
            device=input_ids.device,
        )
        concept_memory = self.concept_embeddings(all_ids).unsqueeze(0).expand(batch_size, -1, -1)

        # Initialize GNN states with concept memory + active question projection
        initial_state = self.input_norm(
            concept_memory
            + projected_question.unsqueeze(1) * activation_probs.unsqueeze(-1)
        )

        # Neural graph propagation
        graph_state = self.graph_reasoner(initial_state)

        # Invoke beam search if requested during inference
        if target_paths is None and beam_width > 1:
            return self.beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                question_embedding=question_embedding,
                projected_question=projected_question,
                activated=activated,
                graph_state=graph_state,
                beam_width=beam_width,
            )

        # Path decoding using ConceptReasoningTransformer
        path_output = self.path_generator(
            graph_state=graph_state,
            question_context=projected_question,
            activation_logits=activated["activation_logits"],
            transition_mask=self.transition_mask,
            first_step_mask=self.first_step_mask,
            target_paths=target_paths,
        )

        # Trace predicted probabilities for GUI and visualization
        path_logits = path_output["path_logits"]
        predicted_path = path_output["predicted_path"]
        path_scores = path_output["path_scores"]

        traces = []
        for row, scores in zip(predicted_path.detach().cpu(), path_scores.detach().cpu()):
            traces.append(
                [
                    {
                        "step": float(step),
                        "concept_id": float(int(concept_id)),
                        "log_probability": float(score),
                    }
                    for step, (concept_id, score) in enumerate(zip(row.tolist(), scores.tolist()))
                ]
            )

        return {
            "question_embedding": question_embedding,
            "activation_logits": activated["activation_logits"],
            "activated_concepts": activated["concept_ids"],
            "activation_scores": activated["concept_scores"],
            "graph_state": graph_state,
            "path_logits": path_logits,
            "predicted_path": predicted_path,
            "path_scores": path_scores,
            "traversal_trace": traces,
        }

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_embedding: torch.Tensor,
        projected_question: torch.Tensor,
        activated: Dict[str, torch.Tensor],
        graph_state: torch.Tensor,
        beam_width: int = 3,
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.size(0)
        device = input_ids.device
        num_concepts = self.config["num_concepts"]
        path_length = self.config["path_length"]

        final_prediction_steps = []
        final_score_steps = []
        final_logits_steps = []

        for b in range(batch_size):
            sample_graph_state = graph_state[b]
            sample_question_context = projected_question[b].unsqueeze(0)
            sample_activation_logits = activated["activation_logits"][b]

            # Step 0: decoding from start embedding
            prev_emb = self.path_generator.start_embedding.unsqueeze(0)
            pos_id = torch.tensor([[0]], device=device)
            decoder_input = prev_emb + self.path_generator.pos_encoder(pos_id)

            memory = sample_question_context.unsqueeze(1)
            tgt_mask = self.path_generator.generate_causal_mask(1, device)
            decoded = self.path_generator.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)
            last_decoded = decoded[:, -1]

            step_logits = self.path_generator.output_head(last_decoded) + 0.25 * sample_activation_logits.unsqueeze(0)
            allowed = self.first_step_mask.unsqueeze(0)
            step_logits = step_logits.masked_fill(~allowed, -1.0e4)

            log_probs = F.log_softmax(step_logits, dim=-1).squeeze(0)
            top_scores, top_ids = torch.topk(log_probs, k=min(beam_width, num_concepts))

            active_beams = []
            for score, cid in zip(top_scores.tolist(), top_ids.tolist()):
                if score > -1.0e3:
                    active_beams.append({
                        "path": [int(cid)],
                        "score": float(score),
                        "finished": (int(cid) == self.vocab_eos_id) or (int(cid) == self.vocab_pad_id)
                    })

            if not active_beams:
                active_beams.append({
                    "path": [int(top_ids[0].item())],
                    "score": float(top_scores[0].item()),
                    "finished": False
                })

            for step in range(1, path_length):
                if all(beam["finished"] for beam in active_beams):
                    break

                candidates = []
                beams_to_run = [beam for beam in active_beams if not beam["finished"]]
                beams_finished = [beam for beam in active_beams if beam["finished"]]

                if beams_to_run:
                    num_run = len(beams_to_run)
                    decoder_inputs_list = []
                    for beam in beams_to_run:
                        embs = [self.path_generator.start_embedding]
                        for cid in beam["path"][:-1]:
                            embs.append(sample_graph_state[cid])
                        decoder_inputs_list.append(torch.stack(embs))
                    
                    decoder_inputs = torch.stack(decoder_inputs_list, dim=0)
                    pos_ids = torch.arange(step, device=device).unsqueeze(0).expand(num_run, -1)
                    decoder_inputs = decoder_inputs + self.path_generator.pos_encoder(pos_ids)

                    run_memory = sample_question_context.unsqueeze(1).expand(num_run, 1, -1)
                    tgt_mask = self.path_generator.generate_causal_mask(step, device)
                    decoded = self.path_generator.transformer_decoder(decoder_inputs, run_memory, tgt_mask=tgt_mask)
                    last_decoded = decoded[:, -1]

                    logits = self.path_generator.output_head(last_decoded) + 0.25 * sample_activation_logits.unsqueeze(0)

                    for idx, beam in enumerate(beams_to_run):
                        prev_id = beam["path"][-1]
                        allowed = self.transition_mask[prev_id]
                        beam_logits = logits[idx].masked_fill(~allowed, -1.0e4)
                        beam_log_probs = F.log_softmax(beam_logits, dim=-1)

                        top_k_scores, top_k_ids = torch.topk(beam_log_probs, k=min(beam_width, num_concepts))

                        for next_score, next_cid in zip(top_k_scores.tolist(), top_k_ids.tolist()):
                            if next_score > -1.0e3:
                                next_cid_int = int(next_cid)
                                is_eos = (next_cid_int == self.vocab_eos_id) or (next_cid_int == self.vocab_pad_id)
                                candidates.append({
                                    "path": beam["path"] + [next_cid_int],
                                    "score": beam["score"] + float(next_score),
                                    "finished": is_eos
                                })

                for beam in beams_finished:
                    candidates.append({
                        "path": beam["path"] + [self.vocab_eos_id],
                        "score": beam["score"],
                        "finished": True
                    })

                candidates.sort(key=lambda x: x["score"], reverse=True)
                active_beams = candidates[:beam_width]

            active_beams.sort(key=lambda x: x["score"], reverse=True)
            best_beam = active_beams[0]
            best_path = best_beam["path"]

            if len(best_path) < path_length:
                best_path.extend([self.vocab_eos_id] * (path_length - len(best_path)))
            best_path = best_path[:path_length]

            final_pred_tensor = torch.tensor(best_path, device=device)
            final_prediction_steps.append(final_pred_tensor)

            # Re-run logits mapping for scores and logits shape
            sample_logits_list = []
            sample_scores_list = []

            for step_idx in range(path_length):
                embs = [self.path_generator.start_embedding]
                for cid in best_path[:step_idx]:
                    embs.append(sample_graph_state[cid])
                decoder_input = torch.stack(embs).unsqueeze(0)
                
                pos_ids = torch.arange(step_idx + 1, device=device).unsqueeze(0)
                decoder_input = decoder_input + self.path_generator.pos_encoder(pos_ids)
                
                tgt_mask = self.path_generator.generate_causal_mask(step_idx + 1, device)
                decoded = self.path_generator.transformer_decoder(decoder_input, sample_question_context.unsqueeze(1), tgt_mask=tgt_mask)
                last_decoded = decoded[:, -1]
                
                step_logits = self.path_generator.output_head(last_decoded) + 0.25 * sample_activation_logits.unsqueeze(0)
                if step_idx == 0:
                    allowed = self.first_step_mask.unsqueeze(0)
                else:
                    allowed = self.transition_mask[best_path[step_idx - 1]].unsqueeze(0)
                
                step_logits = step_logits.masked_fill(~allowed, -1.0e4)
                log_probs = F.log_softmax(step_logits, dim=-1)
                
                score = log_probs[0, best_path[step_idx]].item()
                
                sample_logits_list.append(step_logits.squeeze(0))
                sample_scores_list.append(torch.tensor(score, device=device))

            final_logits_steps.append(torch.stack(sample_logits_list, dim=0))
            final_score_steps.append(torch.stack(sample_scores_list, dim=0))

        path_logits = torch.stack(final_logits_steps, dim=0)
        predicted_path = torch.stack(final_prediction_steps, dim=0)
        path_scores = torch.stack(final_score_steps, dim=0)

        traces = []
        for row, scores in zip(predicted_path.detach().cpu(), path_scores.detach().cpu()):
            traces.append(
                [
                    {
                        "step": float(step),
                        "concept_id": float(int(concept_id)),
                        "log_probability": float(score),
                    }
                    for step, (concept_id, score) in enumerate(zip(row.tolist(), scores.tolist()))
                ]
            )

        return {
            "question_embedding": question_embedding,
            "activation_logits": activated["activation_logits"],
            "activated_concepts": activated["concept_ids"],
            "activation_scores": activated["concept_scores"],
            "graph_state": graph_state,
            "path_logits": path_logits,
            "predicted_path": predicted_path,
            "path_scores": path_scores,
            "traversal_trace": traces,
        }


def build_vlcm_model_from_dataset(
    dataset,
    concept_dim: int = 128,
    hidden_size: int = 128,
    path_length: Optional[int] = None,
    **kwargs,
) -> VLCMModel:
    """Helper function to build a VLCMModel instance directly from a dataset."""
    return VLCMModel(
        num_concepts=dataset.vocab.size(),
        tokenizer_vocab_size=dataset.tokenizer.vocab_size,
        graph=dataset.graph,
        vocab=dataset.vocab,
        concept_dim=concept_dim,
        hidden_size=hidden_size,
        path_length=path_length or dataset.max_path_length,
        **kwargs,
    )
