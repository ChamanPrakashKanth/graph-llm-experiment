"""Language decoding and explainability for VLCM.

Converts explicit concept reasoning paths into formatted text outputs
and coherent natural language answers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from reasoning_dataset import ConceptVocabulary, SimpleTokenizer
from answer_decoder import T5AnswerDecoder


def show_reasoning_path(concepts: Sequence[str]) -> str:
    """Format and display the reasoning path with down arrows.
    
    Output format:
    Concept1
    ↓
    Concept2
    ↓
    Concept3
    """
    return "\n↓\n".join(concepts)


class PathFormatter:
    """Utility to format concept paths with right arrows."""

    @staticmethod
    def format_path(concepts: Sequence[str]) -> str:
        return " -> ".join(concepts)


class TemplateAnswerDecoder:
    """Generates natural language technical answers from a concept path."""

    def generate_answer(self, question: str, reasoning_path: Sequence[str]) -> str:
        if not reasoning_path:
            return "No stable reasoning path was found for the question."

        if len(reasoning_path) == 1:
            return f"The reasoning engine activated {reasoning_path[0]} as the central concept."

        start = reasoning_path[0]
        end = reasoning_path[-1]
        chain = PathFormatter.format_path(reasoning_path)
        
        if question.lower().startswith("why"):
            return f"{end} follows because the path {chain} links the initiating concept to the outcome."
        return f"The predicted reasoning path is {chain}, connecting {start} to {end}."


class VLCMReasoningSystem:
    """End-to-end inference wrapper for VLCM."""

    def __init__(
        self,
        model,
        vocab: ConceptVocabulary,
        tokenizer: SimpleTokenizer,
        decoder: Optional[object] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if decoder is None:
            import os
            t5_path = "checkpoints/answer_decoder_t5"
            if os.path.exists(t5_path) and any(os.listdir(t5_path)):
                try:
                    decoder = T5AnswerDecoder(model_dir=t5_path, device=self.device)
                    print(f"Loaded local T5 answer decoder from {t5_path}")
                except Exception as e:
                    print(f"Could not load T5 answer decoder from {t5_path}: {e}. Falling back to TemplateAnswerDecoder.")
                    decoder = TemplateAnswerDecoder()
            else:
                decoder = TemplateAnswerDecoder()
        self.decoder = decoder
        
        self.model.to(self.device)
        self.model.eval()

    def predict_path_ids(self, question: str, max_length: int = 64, beam_width: int = 1) -> Dict[str, object]:
        import grammar_parser
        normalized = grammar_parser.normalize_query(question)
        encoded = self.tokenizer.encode(normalized, max_length=max_length)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, beam_width=beam_width)

        path_ids = outputs["predicted_path"][0].detach().cpu().tolist()
        return {
            "path_ids": path_ids,
            "outputs": outputs,
        }

    def answer(self, question: str, max_length: int = 64, beam_width: int = 1) -> Dict[str, object]:
        prediction = self.predict_path_ids(question, max_length=max_length, beam_width=beam_width)
        reasoning_path = self.vocab.decode_path(prediction["path_ids"])
        answer = self.decoder.generate_answer(question, reasoning_path)
        outputs = prediction["outputs"]
        
        activated = [
            self.vocab.id_to_concept.get(int(concept_id), str(int(concept_id)))
            for concept_id in outputs["activated_concepts"][0].detach().cpu().tolist()
        ]
        
        return {
            "question": question,
            "reasoning_path": reasoning_path,
            "reasoning_path_text": PathFormatter.format_path(reasoning_path),
            "explainability_path": show_reasoning_path(reasoning_path),
            "activated_concepts": activated,
            "answer": answer,
        }
