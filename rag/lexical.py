"""The lexical half of hybrid retrieval: BM25.

Dense embeddings are great at *meaning* but can miss exact tokens — a specific
library, an error code, an acronym, "p95". BM25 is the classic keyword-relevance
score (a smarter TF-IDF) that nails exact matches. Real systems fuse both, because
each covers the other's blind spot.

This is a compact, dependency-free Okapi BM25 over the resume chunks, written out
in full so the math is visible rather than hidden in a library.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens (keeps things like 'p95')."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 ranking over a fixed set of documents (the resume chunks)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # term-frequency saturation
        self.b = b    # length-normalization strength
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.N = len(docs)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        # Inverse document frequency: rare terms are more informative.
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """Return a BM25 score for every document, aligned to chunk order."""
        out = np.zeros(self.N, dtype=np.float32)
        if not self.N:
            return out
        q_terms = tokenize(query)
        for i in range(self.N):
            dl, tf = self.doc_len[i], self.tf[i]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            out[i] = score
        return out
