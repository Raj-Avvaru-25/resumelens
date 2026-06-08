# Architecture

A contributor-level reference: the layering, the exact object shapes, how data
flows through a build and a query, and the seams where you'd extend or swap pieces.
For the conceptual "why", read the **Understanding the system** section of the
[README](../README.md) first.

---

## 1. Layering

```
┌─ ui/ ───────────────────────────────────────────────────────────────┐
│ app.py (sidebar, routing, per-résumé index cache)                    │
│ walkthrough · evaluation · profile · understanding · recruiter ·     │
│ talent_pool · components                                             │
└───────────────┬─────────────────────────────────────────────────────┘
                │ calls (plain Python, no Streamlit imported below this line)
┌─ rag/ ────────▼─────────────────────────────────────────────────────┐
│ pipeline  ── orchestration: build_index, retrieve, RetrievalResult   │
│ chunker · embedder · lexical(BM25) · store · reranker                 │
│ query_transform · generator · extraction · faithfulness · evaluation │
│ corpus  ── multi-résumé                                               │
│ config  ── all tunables                                              │
└───────────────┬─────────────────────────────────────────────────────┘
                │
        local models (sentence-transformers)   +   Claude (anthropic SDK)
```

**Rule:** everything under `rag/` is pure Python and Streamlit-free — it can be
imported by tests, scripts, or another frontend. Streamlit lives only in `ui/` and
`app.py`. The Anthropic SDK is touched only in `generator.py`, `extraction.py`,
`faithfulness.py`, and `query_transform.py` (the key-requiring paths).

---

## 2. Data model

All defined in `rag/chunker.py` and `rag/pipeline.py`.

```python
# chunker.py — the two chunk levels
@dataclass
class Chunk:            # CHILD: the searchable unit
    id: int            # global, sequential (== position in ResumeIndex.chunks)
    text: str          # the bullet text (shown to users, cited)
    section: str       # e.g. "EXPERIENCE"
    parent_id: int     # which Parent it belongs to
    embed_text: str    # what we embed/index: "<role title> — <bullet>"

@dataclass
class Parent:          # the CONTEXT unit that gets returned
    id: int
    title: str         # first line of the entry (role/company/dates)
    text: str          # full entry (title + all bullets)
    section: str
    child_ids: list[int]
```

```python
# pipeline.py — the index and per-query results
@dataclass
class ResumeIndex:
    resume_text: str
    parents: list[Parent]
    chunks: list[Chunk]          # CHILDREN (id == list position)
    vectors: np.ndarray          # (N_children, D), L2-normalized
    store: VectorStore           # cosine engine over children
    bm25: BM25                   # lexical engine over child.embed_text
    parent_by_id: dict[int, Parent]

@dataclass
class Scored:                    # one child, scored at every stage
    chunk: Chunk
    dense_score: float; dense_rank: int
    bm25_score: float;  bm25_rank: int
    rrf_score: float
    rerank_score: float | None   # set only for reranked candidates
    # .score -> rerank_score if present else rrf_score

@dataclass
class ParentHit:                 # a returned role
    parent: Parent
    score: float                 # best child's .score (display only — see §4)
    matched_child_ids: list[int] # the reranked bullets under this role

@dataclass
class RetrievalResult:
    query: str
    query_vector: np.ndarray
    dense_scores: np.ndarray     # per child
    bm25_scores: np.ndarray      # per child
    scored_all: list[Scored]     # diagnostics for every child (drives the UI table)
    retrieved_parents: list[ParentHit]   # the roles handed to the model
    augmented_context: str       # format_context(retrieved_parents)
    # .all_scores -> dense_scores ;  .returned_parent_ids -> {pid,...}
```

Invariant worth knowing: **`Chunk.id == its index in `ResumeIndex.chunks`** and
**`Parent.id == its index in `ResumeIndex.parents`**. Citations and rollups rely on
this (a citation's `start_block_index` *is* the child id).

---

## 3. Index build — `build_index(resume_text)`

```
resume_text
  └─ chunker.chunk_resume()            → (parents, children)
        parents: whole role/section entries (never char-split)
        children: bullets; embed_text = "<title> — <bullet>"
  └─ embedder.embed_documents([c.embed_text])  → vectors (N, D)
  └─ ResumeIndex(...)  builds:
        VectorStore(children, vectors)   (store.py)
        BM25([c.embed_text])             (lexical.py)
        parent_by_id
```

Cost is CPU-only and quick. Cached per-résumé in `st.session_state` keyed by
`INDEX_VERSION:sha256(resume_text)` (see `app.py::_get_index`); bump
`INDEX_VERSION` whenever the index structure changes so stale cached objects don't
break.

---

## 4. Query — `pipeline.retrieve(...)`

```python
retrieve(index, query,
         top_k=TOP_K, candidate_k=CANDIDATE_K, enable_rerank=ENABLE_RERANK,
         extra_queries=None,            # multi-query: extra rewrites to fuse
         dense_query_vector=None)       # HyDE: precomputed dense query vector
  -> RetrievalResult
```

Flow:

1. For the main query (and each `extra_query`): dense scores (`store.all_scores`)
   + BM25 scores → two rank dicts each. HyDE replaces the *main* dense vector with
   `dense_query_vector`.
2. **RRF fuse** all rank dicts → `rrf` score per child (`rrf_fuse`, k=`RRF_K`).
3. First stage: top `candidate_k` children by `rrf`.
4. Second stage: `reranker.rerank(query, [(id, child.text)…])` → `rerank_score` on
   those candidates.
5. Final child order = sort by `(rerank_score or -inf, rrf_score)`.
6. **Roll up to parents by first-appearance** in that order (NOT by raw `.score` —
   rerank logits ≈ −6 and RRF ≈ 0.03 are different scales and must not be compared
   numerically; rank position already tiers rerank above RRF). Take `top_k` parents.
7. `augmented_context = format_context(retrieved_parents)` (whole role texts).

**Composable orderings** (used by the eval harness, return child-id lists):
`order_dense`, `order_bm25`, `order_hybrid`, `order_hybrid_rerank`, and
`to_parent_order(index, child_ids)` which collapses a child ranking to a parent
ranking by first appearance.

---

## 5. Generation — `rag/generator.py`

Single-doc modes (understanding, recruiter, walkthrough):

```python
context_text, cacheable = select_context(resume_text, augmented_context, mode)
#   mode="full"   -> (resume_text, True)    whole résumé, cached once per résumé
#   mode="chunks" -> (augmented_context, False)  retrieved roles only (true RAG)

stream_reply(client, context_text, persona, messages,
             cacheable=True, effort=None, usage_out=None)  # yields text deltas
```

- System prompt = `[persona, context_block]`; `cache_control {ttl:"1h"}` on the
  context block when `cacheable` (the prefix is stable per résumé → cache hit on
  later turns, once it clears the model's ~4096-token minimum).
- `usage_out` (a dict) is filled with `usage` after the stream drains, so the UI can
  show token counts.

Citations path (`stream_cited`): sends **all children as a custom-content citation
document** (one text block per child) with `citations:{enabled:true}` and
`cache_control`. The API returns `content_block_location` citations whose
`start_block_index` **is the child id**. The generator parses the stream into
`segments = [{text, citations:[{cited_text, start_block, end_block}]}]` and fills
`result_out["segments"|"usage"]`. UI turns segments into `[n]` tags + a Sources list.

Persona strings: `UNDERSTANDING_PERSONA` (precise analyst) and `RECRUITER_PERSONA`
(skeptical interviewer).

---

## 6. Multi-résumé — `rag/corpus.py`

```python
Candidate(name, text, index: ResumeIndex, profile: Profile | None)
CandidateHit(candidate, score, best_parent: Parent, matched_child_ids)

build_corpus(docs: dict[name,text]) -> list[Candidate]      # one ResumeIndex each
extract_profiles(client, candidates)                        # fills .profile (needs key)
passes_filter(profile, min_years, seniorities, required_skills) -> bool
search(candidates, query, extra_queries=None, dense_query_vector=None)
    -> list[CandidateHit]   # each candidate's top role, sorted by score desc
```

`search` runs the **same `retrieve` per candidate** and ranks candidates by their
single best reranked child score — a fair cross-document signal because cross-encoder
logits are comparable across résumés. Filtering happens on `Profile` fields *before*
ranking. (Scale note in §9.)

---

## 7. Evaluation & faithfulness

- `rag/evaluation.py`: `GOLD_SET = [(question, [anchor phrases])]`. `gold_ids`
  returns **parent** ids whose text contains an anchor (we measure at the unit we
  *return*). `evaluate(index, ranker, k)` → `Metrics(recall_at_k, mrr_at_k,
  per_query)`. `VARIANTS` wraps the child orderings with `to_parent_order`.
  **Always evaluated against `data/sample_resume.txt`** (a gold set only applies to
  its own document — see `ui/evaluation.py::_sample_index`).
- `rag/faithfulness.py`: `citation_coverage(segments, chunks)` (deterministic:
  % of substantive sentences cited, % of citations whose quote is verbatim in the
  cited child) and `judge_groundedness(client, q, answer, context)` (LLM-judge,
  structured output `{grounded_score, verdict, unsupported_claims}`).

---

## 8. Query transforms — `rag/query_transform.py`

`apply(client, query, mode) -> (extra_queries, dense_query_vector, note)`:
- `"multi-query"` → `multi_query()` (structured-output rewrites) → `extra_queries`.
- `"HyDE"` → `hyde_vector()` (embed a hypothetical answer) → `dense_query_vector`.
- `"none"` → `(None, None, "")`.

The caller passes the result straight into `retrieve(...)`. Transforms need a key;
base retrieval does not.

---

## 9. Extension points

| You want to… | Do this |
|--------------|---------|
| **Swap the embedding model** | set `EMBEDDING_MODEL` (+ `EMBEDDING_QUERY_PROMPT` if it wants a query instruction) in `config.py` or env. Dim is auto-detected; no other change. |
| **Swap / disable the reranker** | set `RERANK_MODEL` (any `CrossEncoder`-compatible model), or `ENABLE_RERANK=False` (or pass `enable_rerank=False` per call). |
| **Tune retrieval** | `TOP_K`, `CANDIDATE_K`, `RRF_K`, `MAX_CHUNK_CHARS` in `config.py`. |
| **Add a real vector DB** | reimplement `store.VectorStore` keeping `all_scores(qv) -> np.ndarray`, *or* change `pipeline.retrieve` to call an ANN `top_k`. For a corpus, build **one** combined index over all children carrying `{resume_id, parent_id}` metadata and filter before ANN (replaces the per-candidate loop in `corpus.search`). |
| **Filter at scale** | extend `Profile` (in `extraction.py`), push the predicate into the vector-store query instead of `corpus.passes_filter`. |
| **Use a different LLM/provider** | reimplement `generator.get_client/stream_reply/stream_cited`. Note: native **citations are Anthropic-specific**; a different provider needs a different grounding approach. |
| **Grow the eval** | edit `evaluation.GOLD_SET` (it scores `data/sample_resume.txt`); add hard negatives / synthesis questions. |
| **Add résumés to the pool** | drop `.txt` files in `data/samples/` (auto-loaded) or upload at runtime in the Talent pool. |
| **Add a new UI mode** | add `ui/<mode>.py` with `render(index, api_key, …)`, then register it in `app.py`'s mode radio + dispatch. |

---

## 10. Performance & caching

- **Model loads** are memoized with `functools.lru_cache` (`embedder._get_model`,
  `reranker._get_model`) — one load per process.
- **Indexes** cached in `st.session_state` (résumé index by content hash; corpus by
  signature of the doc set).
- **Search** is brute-force `O(N·D)` cosine + `O(N·|q|)` BM25 — microseconds at this
  scale, intentionally not an ANN index.
- **Prompt caching** trims input cost on multi-turn single-résumé chats (full-résumé
  mode); the reranker/eval calls are local and free.
- **Token cost** lives almost entirely in generation *output* (thinking) — controlled
  by the sidebar **effort** selector; measure with the token readout / effort A/B.
