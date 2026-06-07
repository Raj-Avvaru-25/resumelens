"""Evaluation harness — the difference between vibes-driven and eval-driven RAG.

You cannot improve retrieval you don't measure. This module defines a small
labeled set of (question -> the chunk that truly answers it) and computes standard
IR metrics, so we can prove that hybrid beats dense-only and reranking beats
hybrid — with numbers, not opinions.

Gold relevance is defined by *anchor phrases* (a substring that uniquely lives in
the correct chunk) rather than hard-coded chunk IDs, so the labels stay valid even
if the chunker changes.

Metrics:
  * Recall@k — did at least one truly-relevant chunk make the top k?
  * MRR@k    — Mean Reciprocal Rank: 1/(rank of first relevant chunk), else 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import config, pipeline
from .pipeline import ResumeIndex

# (question, anchor phrases that identify the truly relevant chunk(s))
# Tuned for the bundled sample_resume.txt. A relevant PARENT is any whose text
# contains an anchor. Multi-anchor questions are synthesis questions that should
# pull MORE THAN ONE role.
GOLD_SET: list[tuple[str, list[str]]] = [
    # direct
    ("What consensus algorithm did they use?", ["Raft"]),
    ("Tell me about their real-time payments work.", ["payments ledger"]),
    ("Did they improve query latency?", ["p95 query latency"]),
    ("Streaming data pipeline experience?", ["streaming model on Kafka"]),
    ("Any machine learning projects?", ["learned re-ranker"]),
    ("What databases do they know?", ["PostgreSQL"]),
    ("Have they mentored or led people?", ["Mentored"]),
    ("Where did they study?", ["B.Tech"]),
    # harder / trickier phrasing (no keyword overlap with the source)
    ("How do they keep systems reliable when nodes die?", ["fault-tolerant", "leader failover"]),
    ("Have they shipped anything used by the public?", ["GitHub stars", "open-source"]),
    ("Can they handle large transaction volume?", ["4M transactions"]),
    ("How experienced are they overall?", ["6 years"]),
    ("Did they fix a production incident?", ["after a major outage"]),
    # synthesis — should retrieve MULTIPLE roles
    ("Compare their backend depth with their ML work.", ["payments ledger", "learned re-ranker"]),
    ("Where have they used Kafka?", ["Kafka"]),
    ("What measurable impact have they had?", ["18 hours to under", "reduced p95", "near zero"]),
]


def gold_ids(index: ResumeIndex, anchors: list[str]) -> set[int]:
    """PARENT IDs whose text contains any anchor phrase (case-insensitive).

    We measure at the parent level because parents (whole roles) are what the
    small-to-big pipeline actually returns.
    """
    ids = set()
    for p in index.parents:
        low = p.text.lower()
        if any(a.lower() in low for a in anchors):
            ids.add(p.id)
    return ids


@dataclass
class Metrics:
    recall_at_k: float
    mrr_at_k: float
    per_query: list[dict]  # {"query", "first_relevant_rank" or None}


def evaluate(
    index: ResumeIndex,
    ranker: Callable[[ResumeIndex, str], list[int]],
    k: int,
    gold: list[tuple[str, list[str]]] | None = None,
) -> Metrics:
    """Run a retrieval function over the gold set and compute Recall@k + MRR@k."""
    gold = gold or GOLD_SET
    recalls, rrs, per_query = [], [], []
    for query, anchors in gold:
        relevant = gold_ids(index, anchors)
        order = ranker(index, query)
        topk = order[:k]
        hit = bool(relevant & set(topk))
        recalls.append(1.0 if hit else 0.0)

        first_rank = None
        for pos, cid in enumerate(order, start=1):
            if cid in relevant:
                first_rank = pos
                break
        rrs.append(1.0 / first_rank if (first_rank and first_rank <= k) else 0.0)
        per_query.append(
            {"query": query, "first_relevant_rank": first_rank if first_rank else None}
        )

    n = max(1, len(gold))
    return Metrics(sum(recalls) / n, sum(rrs) / n, per_query)


# The three pipeline variants we compare. Each child-level ordering is rolled up
# to a PARENT ordering (what small-to-big actually returns) before scoring.
def _parents(order_fn):
    return lambda index, query: pipeline.to_parent_order(index, order_fn(index, query))


VARIANTS: dict[str, Callable[[ResumeIndex, str], list[int]]] = {
    "Dense only": _parents(pipeline.order_dense),
    "Hybrid (dense+BM25, RRF)": _parents(pipeline.order_hybrid),
    "Hybrid + rerank": _parents(pipeline.order_hybrid_rerank),
}


def compare_variants(index: ResumeIndex, k: int | None = None) -> dict[str, Metrics]:
    k = k or config.TOP_K
    return {name: evaluate(index, fn, k) for name, fn in VARIANTS.items()}
