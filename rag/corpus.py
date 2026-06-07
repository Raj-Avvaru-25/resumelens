"""Multi-résumé corpus — where RAG stops being optional.

On one résumé you can stuff the whole thing in context. Across a *pool* you can't —
you must retrieve. This module turns a set of résumés into searchable candidates and
supports the two corpus operations recruiters actually want:

  * FILTER on structured fields (seniority, years, skills) — a query over the
    extracted profiles (feature #3), not over vectors.
  * RANK candidates for a free-text need ("who shipped distributed systems?") by
    running each résumé through the same hybrid + rerank + small-to-big pipeline and
    comparing their best-matching evidence.

For a handful of résumés we search each candidate's own index and compare the top
match. At large scale you'd instead keep one combined ANN index with per-résumé
metadata and filter before retrieval — same idea, different plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import extraction
from .chunker import Parent
from .extraction import Profile
from .pipeline import ResumeIndex, build_index, retrieve


@dataclass
class Candidate:
    name: str
    text: str
    index: ResumeIndex
    profile: Profile | None = None


@dataclass
class CandidateHit:
    candidate: Candidate
    score: float                 # best reranked-child score (cross-candidate relevance)
    best_parent: Parent          # the role that matched the query
    matched_child_ids: list[int]


def build_corpus(docs: dict[str, str]) -> list[Candidate]:
    """Build a per-résumé index for each document (chunk → embed → store)."""
    return [Candidate(name=name, text=text, index=build_index(text)) for name, text in docs.items()]


def extract_profiles(client, candidates: list[Candidate]) -> None:
    """Fill in structured profiles (enables filtering). Needs an API key."""
    for c in candidates:
        if c.profile is None:
            c.profile = extraction.extract_profile(client, c.text)


def passes_filter(
    profile: Profile | None,
    min_years: float,
    seniorities: list[str],
    required_skills: list[str],
) -> bool:
    """Apply metadata filters. With no profile we can't filter, so we keep it."""
    if profile is None:
        return True
    if min_years and profile.years_experience < min_years:
        return False
    if seniorities and profile.seniority not in seniorities:
        return False
    if required_skills:
        haystack = " ".join(profile.skills).lower()
        if not all(s.lower().strip() in haystack for s in required_skills if s.strip()):
            return False
    return True


def search(
    candidates: list[Candidate],
    query: str,
    extra_queries: list[str] | None = None,
    dense_query_vector: np.ndarray | None = None,
) -> list[CandidateHit]:
    """Rank candidates by their single best matching role for the query."""
    hits: list[CandidateHit] = []
    for c in candidates:
        r = retrieve(c.index, query, extra_queries=extra_queries, dense_query_vector=dense_query_vector)
        if not r.retrieved_parents:
            continue
        top = r.retrieved_parents[0]
        hits.append(CandidateHit(c, top.score, top.parent, top.matched_child_ids))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits
