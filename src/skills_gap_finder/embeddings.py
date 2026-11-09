"""Turn text into embedding vectors.

Models run locally on CPU through ONNX (via fastembed), so no API key or GPU is needed.
Vectors are L2-normalised, so a dot product is cosine similarity.

Some models (e.g. BGE) are trained to embed a search query slightly differently from the
documents being searched, so an embedder has separate `documents` and `queries` methods.
"""

from typing import Protocol

import numpy as np
import pandas as pd

# Best of four local models on the evaluation set (see README); about 440 MB, downloaded on first use
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
COMPARED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "snowflake/snowflake-arctic-embed-s",
]


class Embedder(Protocol):
    name: str

    def documents(self, texts: list[str]) -> np.ndarray: ...

    def queries(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model)

    def documents(self, texts: list[str]) -> np.ndarray:
        return normalise(np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32))

    def queries(self, texts: list[str]) -> np.ndarray:
        return normalise(np.asarray(list(self._model.query_embed(texts)), dtype=np.float32))


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def skill_document(skill: pd.Series) -> str:
    """The text that represents a skill in the index: its name, description and synonyms."""
    text = f"{skill['label']}. {skill['description']}"
    if skill["alt_labels"]:
        text += f" Also known as: {skill['alt_labels']}."
    return text
