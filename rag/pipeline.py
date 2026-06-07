"""Orchestrates the hierarchical (small-to-big) RAG pipeline + captures artifacts.

    LOAD -> CHUNK (parents + children) -> EMBED children -> STORE (+ BM25)   once/resume

    QUERY
      ├─ dense + BM25 over CHILDREN (bullets)   ← sharp, precise matching
      ├─ RRF fuse -> candidate children
      ├─ cross-encoder rerank of candidate children
      └─ ROLL UP children -> PARENTS (whole roles)   ← returned for full context
                                                      -> AUGMENT -> GENERATE
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import config, embedder, reranker
from .chunker import Chunk, Parent, chunk_resume
from .lexical import BM25
from .store import VectorStore


# --- Index (built once per resume) -------------------------------------------

@dataclass
class ResumeIndex:
    resume_text: str
    parents: list[Parent]
    chunks: list[Chunk]              # the CHILDREN (searchable units)
    vectors: np.ndarray             # child embeddings
    store: VectorStore = field(init=False)
    bm25: BM25 = field(init=False)
    parent_by_id: dict[int, Parent] = field(init=False)

    def __post_init__(self):
        self.store = VectorStore(self.chunks, self.vectors)
        self.bm25 = BM25([c.embed_text for c in self.chunks])
        self.parent_by_id = {p.id: p for p in self.parents}


def build_index(resume_text: str) -> ResumeIndex:
    parents, children = chunk_resume(
        resume_text, max_chars=config.MAX_CHUNK_CHARS, overlap=config.CHUNK_OVERLAP_CHARS
    )
    if not children:
        parents = [Parent(id=0, title="(empty)", text=resume_text or "(empty)",
                          section="HEADER", child_ids=[0])]
        children = [Chunk(id=0, text=resume_text or "(empty)", section="HEADER",
                          parent_id=0, embed_text=resume_text or "(empty)")]
    vectors = embedder.embed_documents([c.embed_text for c in children])
    return ResumeIndex(resume_text=resume_text, parents=parents, chunks=children, vectors=vectors)


# --- Per-query results -------------------------------------------------------

@dataclass
class Scored:
    """A CHILD with the score it earned at every retrieval stage."""

    chunk: Chunk
    dense_score: float = 0.0
    dense_rank: int = 0
    bm25_score: float = 0.0
    bm25_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


@dataclass
class ParentHit:
    """A whole role/entry returned as context, with the bullets that matched."""

    parent: Parent
    score: float                 # best child score that rolled up to this parent
    matched_child_ids: list[int]


@dataclass
class RetrievalResult:
    query: str
    query_vector: np.ndarray
    dense_scores: np.ndarray            # per CHILD
    bm25_scores: np.ndarray             # per CHILD
    scored_all: list[Scored]            # diagnostics for every CHILD
    retrieved_parents: list[ParentHit]  # the roles returned to the model
    augmented_context: str

    @property
    def all_scores(self) -> np.ndarray:
        return self.dense_scores

    @property
    def returned_parent_ids(self) -> set[int]:
        return {h.parent.id for h in self.retrieved_parents}


# --- Ranking primitives ------------------------------------------------------

def _ranks_from_scores(scores: np.ndarray) -> dict[int, int]:
    order = np.argsort(scores)[::-1]
    return {int(cid): pos + 1 for pos, cid in enumerate(order)}


def rrf_fuse(rankings: list[dict[int, int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for cid, rank in ranking.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


# child-level orderings (used by the eval harness, then rolled up to parents)
def order_dense(index: ResumeIndex, query: str) -> list[int]:
    return [int(i) for i in np.argsort(index.store.all_scores(embedder.embed_query(query)))[::-1]]


def order_bm25(index: ResumeIndex, query: str) -> list[int]:
    return [int(i) for i in np.argsort(index.bm25.scores(query))[::-1]]


def order_hybrid(index: ResumeIndex, query: str) -> list[int]:
    dense_ranks = _ranks_from_scores(index.store.all_scores(embedder.embed_query(query)))
    bm25_ranks = _ranks_from_scores(index.bm25.scores(query))
    fused = rrf_fuse([dense_ranks, bm25_ranks], config.RRF_K)
    return sorted(fused, key=lambda c: fused[c], reverse=True)


def order_hybrid_rerank(index: ResumeIndex, query: str, candidate_k: int | None = None) -> list[int]:
    candidate_k = candidate_k or config.CANDIDATE_K
    hybrid = order_hybrid(index, query)
    candidates = hybrid[:candidate_k]
    scores = reranker.rerank(query, [(cid, index.chunks[cid].text) for cid in candidates])
    reranked = sorted(candidates, key=lambda c: scores.get(c, 0.0), reverse=True)
    return reranked + hybrid[candidate_k:]


def to_parent_order(index: ResumeIndex, child_ids: list[int]) -> list[int]:
    """Collapse a child ranking into a parent ranking (first occurrence wins)."""
    seen: set[int] = set()
    out: list[int] = []
    for cid in child_ids:
        pid = index.chunks[cid].parent_id
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


# --- The full two-stage, small-to-big retrieval ------------------------------

def retrieve(
    index: ResumeIndex,
    query: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    enable_rerank: bool | None = None,
    extra_queries: list[str] | None = None,
    dense_query_vector: np.ndarray | None = None,
) -> RetrievalResult:
    """Two-stage small-to-big retrieval.

    Optional query transforms (no effect unless passed):
      * extra_queries       — multi-query: each extra query contributes its own
                              dense+BM25 ranking, all fused together via RRF.
      * dense_query_vector  — HyDE: use this precomputed vector for the main dense
                              search instead of embedding the raw query.
    """
    top_k = top_k or config.TOP_K
    candidate_k = candidate_k or config.CANDIDATE_K
    enable_rerank = config.ENABLE_RERANK if enable_rerank is None else enable_rerank

    # Build the fusion inputs across the main query + any expansion queries.
    queries = [query] + list(extra_queries or [])
    rankings: list[dict[int, int]] = []
    main_qv = None
    dense_scores = None
    for i, q in enumerate(queries):
        qv_i = dense_query_vector if (i == 0 and dense_query_vector is not None) else embedder.embed_query(q)
        dscores_i = index.store.all_scores(qv_i)
        rankings.append(_ranks_from_scores(dscores_i))
        rankings.append(_ranks_from_scores(index.bm25.scores(q)))
        if i == 0:
            main_qv, dense_scores = qv_i, dscores_i

    qv = main_qv
    bm25_scores = index.bm25.scores(query)  # main query's lexical scores (for display)
    dense_ranks = _ranks_from_scores(dense_scores)
    bm25_ranks = _ranks_from_scores(bm25_scores)
    rrf = rrf_fuse(rankings, config.RRF_K)

    candidate_ids = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:candidate_k]
    rerank_scores: dict[int, float] = {}
    if enable_rerank:
        rerank_scores = reranker.rerank(query, [(cid, index.chunks[cid].text) for cid in candidate_ids])

    scored_all = [
        Scored(
            chunk=c,
            dense_score=float(dense_scores[c.id]),
            dense_rank=dense_ranks[c.id],
            bm25_score=float(bm25_scores[c.id]),
            bm25_rank=bm25_ranks[c.id],
            rrf_score=float(rrf.get(c.id, 0.0)),
            rerank_score=rerank_scores.get(c.id) if enable_rerank else None,
        )
        for c in index.chunks
    ]

    def final_key(s: Scored):
        return (s.rerank_score if s.rerank_score is not None else -1e9, s.rrf_score)

    ranked = sorted(scored_all, key=final_key, reverse=True)

    # Roll children up to parents by FIRST-APPEARANCE in the final child ranking.
    # We order by rank position, NOT raw .score — rerank logits (~ -6) and RRF
    # scores (~0.03) live on different scales and must not be compared directly.
    # `final_key` already tiers reranked children above the rest, so first
    # appearance is the correct parent order.
    parent_order: list[int] = []
    parent_score: dict[int, float] = {}
    seen: set[int] = set()
    for s in ranked:
        pid = s.chunk.parent_id
        if pid not in seen:
            seen.add(pid)
            parent_order.append(pid)
            parent_score[pid] = s.score  # best child's score (display only)

    # Which bullets actually matched (the reranked candidates), grouped by parent.
    matched: dict[int, list[int]] = defaultdict(list)
    for s in scored_all:
        if s.rerank_score is not None:
            matched[s.chunk.parent_id].append(s.chunk.id)

    hits = [
        ParentHit(index.parent_by_id[pid], parent_score[pid], sorted(matched.get(pid, [])))
        for pid in parent_order[:top_k]
    ]

    return RetrievalResult(
        query=query,
        query_vector=qv,
        dense_scores=dense_scores,
        bm25_scores=bm25_scores,
        scored_all=scored_all,
        retrieved_parents=hits,
        augmented_context=format_context(hits),
    )


def format_context(hits: list[ParentHit]) -> str:
    """Render returned PARENTS (whole roles) as the spotlight given to the model."""
    out = []
    for h in hits:
        mb = ", ".join(f"#{c}" for c in h.matched_child_ids) or "—"
        out.append(
            f"[Role P{h.parent.id} | section: {h.parent.section} | score: {h.score:.3f} "
            f"| matched bullets: {mb}]\n{h.parent.text}"
        )
    return "\n\n".join(out)
