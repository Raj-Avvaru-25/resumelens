"""Second-stage reranking with a cross-encoder.

First-stage retrieval (dense + BM25) scores the query and each chunk
*independently* and hopes their vectors line up. A cross-encoder instead feeds the
query and a chunk into the model TOGETHER and outputs a single relevance score —
it can actually "read" whether the chunk answers the query. It's slower, so we
only run it on the handful of first-stage candidates, not the whole corpus.

This is the highest-ROI accuracy upgrade in the whole pipeline.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder

    from . import config

    return CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, candidates: list[tuple[int, str]]) -> dict[int, float]:
    """Score (query, chunk) pairs. Returns {chunk_id: relevance_score}.

    Degrades gracefully: if the model can't load (e.g. offline), every candidate
    gets a neutral 0.0 and the caller falls back to the first-stage order.
    """
    if not candidates:
        return {}
    try:
        model = _get_model()
        pairs = [(query, text) for _, text in candidates]
        scores = model.predict(pairs)
        return {cid: float(s) for (cid, _), s in zip(candidates, scores)}
    except Exception:
        return {cid: 0.0 for cid, _ in candidates}
