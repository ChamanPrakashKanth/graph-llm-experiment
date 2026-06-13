"""Language decoding for CAT V2.

The reasoning path is the primary output. This module converts a predicted
path into readable prose after the graph reasoning engine has already run.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from reasoning_dataset import ConceptVocabulary, SimpleTokenizer


class PathFormatter:
    @staticmethod
    def format_path(concepts: Sequence[str]) -> str:
        return " -> ".join(concepts)


class TemplateAnswerDecoder:
    """Offline answer decoder that uses only the predicted path."""

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


class OptionalLLMAnswerDecoder:
    """Optional transformers decoder. It is never used unless a model is supplied."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate_answer(
        self,
        question: str,
        reasoning_path: Sequence[str],
        max_new_tokens: int = 128,
    ) -> str:
        path_text = PathFormatter.format_path(reasoning_path)
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Reasoning Path:\n"
            f"{path_text}\n\n"
            "Using only the reasoning path above, generate a concise technical answer."
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() or text.strip()


class CATReasoningSystem:
    """End-to-end inference wrapper for path-first CAT reasoning."""

    def __init__(
        self,
        reasoning_model,
        vocab: ConceptVocabulary,
        tokenizer: SimpleTokenizer,
        decoder: Optional[object] = None,
        device: Optional[str] = None,
    ) -> None:
        self.reasoning_model = reasoning_model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.decoder = decoder or TemplateAnswerDecoder()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reasoning_model.to(self.device)
        self.reasoning_model.eval()

    def predict_path_ids(self, question: str, max_length: int = 64) -> Dict[str, object]:
        encoded = self.tokenizer.encode(question, max_length=max_length)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.reasoning_model(input_ids, attention_mask)

        path_ids = outputs["predicted_path"][0].detach().cpu().tolist()
        return {
            "path_ids": path_ids,
            "outputs": outputs,
        }

    def answer(self, question: str, max_length: int = 64) -> Dict[str, object]:
        prediction = self.predict_path_ids(question, max_length=max_length)
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
            "activated_concepts": activated,
            "answer": answer,
            "traversal_trace": outputs["traversal_trace"][0],
        }


if __name__ == "__main__":
    decoder = TemplateAnswerDecoder()
    path = ["Pressure", "Friction", "Energy Loss", "Pressure Drop"]
    print(decoder.generate_answer("Why does pressure drop?", path))

