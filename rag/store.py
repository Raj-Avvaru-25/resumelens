"""Stage 3 of RAG: STORE + Stage 4: RETRIEVE.

A "vector store" is just a place that holds chunk vectors and can answer the
question: "which stored vectors are most similar to this query vector?"

Production systems use FAISS / pgvector / Pinecone. For a single resume that's
massive overkill — a NumPy matrix and one dot product is the entire engine, and
it makes the math visible instead of hiding it behind a library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk


@dataclass
class Retrieved:
    """A chunk plus how similar it was to the query (1.0 = identical meaning)."""

    chunk: Chunk
    score: float


class VectorStore:
    """In-memory cosine-similarity store over the resume chunks."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        assert len(chunks) == len(vectors), "one vector per chunk"
        self.chunks = chunks
        self.vectors = vectors  # (N, D), already L2-normalized

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Retrieved]:
        """Return the top_k most similar chunks, highest score first.

        Because both the stored vectors and the query are unit-length, the dot
        product equals the cosine similarity — a number in [-1, 1] where higher
        means "closer in meaning".
        """
        scores = self.vectors @ query_vector  # (N,) similarity for every chunk
        order = np.argsort(scores)[::-1][:top_k]
        return [Retrieved(chunk=self.chunks[i], score=float(scores[i])) for i in order]

    def all_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Similarity of the query against *every* chunk (for visualization)."""
        return self.vectors @ query_vector
