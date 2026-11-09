"""Step 3: embed every skill and store it in a persistent Chroma vector database, then search it."""

import chromadb
import numpy as np
import pandas as pd

from .config import CHROMA_DIR, COLLECTION
from .embeddings import Embedder, skill_document


def get_collection(path=CHROMA_DIR) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build(skills: pd.DataFrame, embed: Embedder, path=CHROMA_DIR, batch_size: int = 64) -> chromadb.Collection:
    """(Re)build the index from scratch so it always matches skills.csv."""
    client = chromadb.PersistentClient(path=str(path))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    documents = skills.apply(skill_document, axis=1).tolist()
    for start in range(0, len(skills), batch_size):
        batch = skills.iloc[start : start + batch_size]
        docs = documents[start : start + batch_size]
        collection.add(
            ids=batch["skill_uri"].tolist(),
            documents=docs,
            embeddings=embed.documents(docs).tolist(),
            metadatas=batch[["label", "skill_type"]].to_dict("records"),
        )
    return collection


def search(collection: chromadb.Collection, embed: Embedder, query: str, k: int = 5) -> list[dict]:
    """Return the k skills closest in meaning to the query, most similar first."""
    result = collection.query(query_embeddings=embed.queries([query]).tolist(), n_results=k)
    return [
        {
            "skill_uri": uri,
            "label": meta["label"],
            "skill_type": meta["skill_type"],
            "similarity": round(1 - distance, 3),  # Chroma returns cosine distance
        }
        for uri, meta, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0])
    ]


def get_vectors(collection: chromadb.Collection, skill_uris: list[str]) -> np.ndarray:
    """Fetch stored skill vectors in the same order as skill_uris."""
    stored = collection.get(ids=skill_uris, include=["embeddings"])
    by_id = dict(zip(stored["ids"], stored["embeddings"]))
    return np.asarray([by_id[uri] for uri in skill_uris], dtype=np.float32)
