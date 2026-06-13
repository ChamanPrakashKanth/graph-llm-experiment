# =============================================================================
# dataset.py
# CAT Dataset Pipeline
# =============================================================================

import re
import json
import torch

from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# =============================================================================
# Text Cleaning
# =============================================================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =============================================================================
# Sentence Splitter
# =============================================================================

def split_sentences(text):

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip()) > 5
    ]

    return sentences


# =============================================================================
# Chunk Builder
# =============================================================================

class TextChunker:

    def __init__(
        self,
        chunk_size=8,
        overlap=2
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_sentences(
        self,
        sentences
    ):

        chunks = []

        step = (
            self.chunk_size -
            self.overlap
        )

        for i in range(
            0,
            len(sentences),
            step
        ):

            chunk = sentences[
                i:i+self.chunk_size
            ]

            if len(chunk) < 2:
                continue

            chunks.append(chunk)

        return chunks


# =============================================================================
# Concept Extractor
# =============================================================================

class ConceptExtractor:

    def __init__(self):

        self.patterns = [

            r"pressure",

            r"velocity",

            r"density",

            r"temperature",

            r"mesh",

            r"cell",

            r"boundary",

            r"turbulence",

            r"residual",

            r"solver",

            r"continuity",

            r"momentum",

            r"energy",

            r"heat",

            r"flow",

            r"navier[- ]stokes",

            r"finite volume",

            r"finite element",

            r"wall function",

            r"k[- ]epsilon",

            r"k[- ]omega",

            r"rans",

            r"les",

            r"cfd"
        ]

    def extract(
        self,
        text
    ):

        text = text.lower()

        concepts = []

        for pattern in self.patterns:

            if re.search(
                pattern,
                text
            ):
                concepts.append(
                    pattern
                )

        return concepts


# =============================================================================
# Graph Concept Vocabulary
# =============================================================================

class ConceptVocabulary:

    def __init__(self):

        self.concept_to_id = {}
        self.id_to_concept = {}

    def build(
        self,
        concept_lists
    ):

        unique = set()

        for concepts in concept_lists:

            unique.update(
                concepts
            )

        unique = sorted(
            list(unique)
        )

        self.concept_to_id = {
            concept: idx
            for idx, concept
            in enumerate(unique)
        }

        self.id_to_concept = {
            idx: concept
            for concept, idx
            in self.concept_to_id.items()
        }

    def encode(
        self,
        concepts
    ):

        return [
            self.concept_to_id[c]
            for c in concepts
            if c in self.concept_to_id
        ]

    def size(self):

        return len(
            self.concept_to_id
        )


# =============================================================================
# CAT Dataset
# =============================================================================

class CATDataset(Dataset):

    def __init__(
        self,
        documents,
        tokenizer_name="bert-base-uncased",
        max_length=256
    ):

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                tokenizer_name
            )
        )

        self.max_length = max_length

        self.samples = []

        chunker = TextChunker()

        extractor = ConceptExtractor()

        all_concepts = []

        for document in documents:

            sentences = split_sentences(
                document
            )

            chunks = (
                chunker
                .chunk_sentences(
                    sentences
                )
            )

            for chunk in chunks:

                text = " ".join(
                    chunk
                )

                concepts = (
                    extractor
                    .extract(text)
                )

                all_concepts.append(
                    concepts
                )

                self.samples.append({

                    "text":
                        text,

                    "concepts":
                        concepts
                })

        self.vocab = (
            ConceptVocabulary()
        )

        self.vocab.build(
            all_concepts
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = (
            self.samples[idx]
        )

        encoding = (
            self.tokenizer(
                sample["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        )

        concept_ids = (
            self.vocab.encode(
                sample["concepts"]
            )
        )

        if len(concept_ids) == 0:

            concept_ids = [0]

        return {

            "input_ids":
                encoding[
                    "input_ids"
                ].squeeze(0),

            "attention_mask":
                encoding[
                    "attention_mask"
                ].squeeze(0),

            "concept_ids":
                torch.tensor(
                    concept_ids,
                    dtype=torch.long
                ),

            "text":
                sample["text"]
        }


# =============================================================================
# File Loader
# =============================================================================

class CorpusLoader:

    @staticmethod
    def load_txt_folder(
        folder
    ):

        folder = Path(folder)

        documents = []

        for file in folder.glob(
            "*.txt"
        ):

            try:

                with open(
                    file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    documents.append(
                        f.read()
                    )

            except:

                pass

        return documents

    @staticmethod
    def load_jsonl(
        path
    ):

        docs = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                row = json.loads(
                    line
                )

                docs.append(
                    row["text"]
                )

        return docs


# =============================================================================
# Batch Collator
# =============================================================================

class CATCollator:

    def __call__(
        self,
        batch
    ):

        input_ids = torch.stack([
            x["input_ids"]
            for x in batch
        ])

        attention_mask = torch.stack([
            x["attention_mask"]
            for x in batch
        ])

        return {

            "input_ids":
                input_ids,

            "attention_mask":
                attention_mask,

            "texts":
                [
                    x["text"]
                    for x in batch
                ]
        }


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":

    docs = [

        """
        Pressure and velocity are
        fundamental CFD variables.

        Turbulence models such as
        k-epsilon are widely used.
        """,

        """
        Boundary layers influence
        heat transfer significantly.
        """
    ]

    dataset = CATDataset(
        docs
    )

    print(
        len(dataset)
    )

    print(
        dataset[0]
    )