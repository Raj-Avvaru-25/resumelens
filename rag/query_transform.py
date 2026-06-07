"""Query transformation: multi-query expansion + HyDE.

Raw user questions are often short, vague, or compound — poor embedding queries.
Two cheap, well-established fixes, both using Claude:

* Multi-query: rewrite the question into several focused sub-queries and retrieve
  for all of them, fusing the results. Lifts recall on compound/vague asks
  ("compare their backend vs ML depth" → one query per facet).
* HyDE (Hypothetical Document Embeddings): have the model draft a hypothetical
  answer, then embed THAT. A hypothetical résumé line is lexically/semantically
  closer to the real passages than a question is.

Both require a Claude call, so they only run when the caller has an API key — base
retrieval stays key-free.
"""

from __future__ import annotations

import json

import numpy as np

from . import config, embedder

_MULTI_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


def multi_query(client, query: str, n: int = 3) -> list[str]:
    """Return up to n alternative sub-queries (excludes the original)."""
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=400,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _MULTI_SCHEMA}},
        system=(
            "You rewrite a user's question about a résumé into focused search "
            "queries for a retrieval system. Cover distinct facets; keep each short "
            "and keyword-rich. Do not answer the question."
        ),
        messages=[{"role": "user", "content": f"Question: {query}\nReturn {n} alternative search queries."}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        queries = json.loads(text).get("queries", [])
    except json.JSONDecodeError:
        queries = []
    # de-dupe and drop anything identical to the original
    out, seen = [], {query.strip().lower()}
    for q in queries:
        q = (q or "").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:n]


def hyde_text(client, query: str) -> str:
    """Draft a short hypothetical résumé excerpt that would answer the query."""
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=300,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a short, plausible résumé excerpt (2–3 sentences) that "
                    f"would directly answer this question: '{query}'. Output only the "
                    "excerpt, no preamble."
                ),
            }
        ],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def hyde_vector(client, query: str) -> np.ndarray:
    """Embed the hypothetical document for use as the dense query vector."""
    text = hyde_text(client, query) or query
    return embedder.embed_documents([text])[0]


def apply(client, query: str, mode: str):
    """Return (extra_queries, dense_query_vector, note) for the chosen transform.

    `note` is a short human-readable description for the UI ("" if none).
    """
    if mode == "multi-query":
        subs = multi_query(client, query)
        note = "multi-query → " + " · ".join(subs) if subs else "multi-query (no expansions)"
        return subs, None, note
    if mode == "HyDE":
        vec = hyde_vector(client, query)
        return None, vec, "HyDE → embedded a hypothetical answer for dense search"
    return None, None, ""
