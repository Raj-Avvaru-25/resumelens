"""Stage 2 of RAG: EMBED (dense / semantic signal).

An embedding turns a piece of text into a list of numbers (a vector) such that
texts with similar *meaning* land close together in that number-space. This is
the semantic half of our hybrid retrieval: "led a team" and "managed engineers"
have no words in common but end up as nearby vectors.

We use a local sentence-transformers model so the learner can literally watch
vectors get computed on their own machine — no API, no key.

Note the asymmetry: modern retrievers like bge want a short *instruction*
prepended to the QUERY but not to the documents. `embed_documents` and
`embed_query` handle that for you.
"""

from __future__ import annotations

import functools

import numpy as np

from . import config


@functools.lru_cache(maxsize=1)
def _get_model():
    """Load the embedding model once and reuse it (it's expensive to load)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def _encode(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # unit length => cosine similarity == dot product
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed chunk texts (no query instruction) into an (N, D) matrix."""
    return _encode(texts)


# Backwards-compatible alias used by the pipeline's index builder.
embed_texts = embed_documents


def embed_query(query: str) -> np.ndarray:
    """Embed a single query, prepending the model's retrieval instruction."""
    return _encode([config.EMBEDDING_QUERY_PROMPT + query])[0]


def embedding_dim() -> int:
    """Dimensionality of the vectors this model produces (e.g. 384)."""
    return int(_get_model().get_sentence_embedding_dimension())
