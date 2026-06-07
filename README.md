# Resume RAG 📄

A **glass-box** Retrieval-Augmented Generation app for understanding resumes — not
by matching keywords, but by reasoning over complete context.

It does two jobs and teaches one concept:

1. **Teaches RAG** — an illustrated, step-by-step tour that shows every internal
   stage (load → chunk → embed → store → retrieve → augment → generate) running on
   *your actual resume*, with the real intermediate data at each step (chunks,
   vectors, similarity scores, a 2D map of the embedding space, and the exact
   prompt sent to Claude).
2. **Deep understanding** — a grounded analyst you can interrogate. It reads the
   whole resume in context and grounds every claim in the text.
3. **Cynical recruiter** — a skeptical interviewer that grills the candidate,
   demands specifics, and pressure-tests every claim.

## How it works

```
LOAD ─► CHUNK ─► EMBED ─► STORE (+ BM25 index)          built once per resume

QUESTION
   ├─ dense (semantic) ranking          ┐
   ├─ BM25 (lexical) ranking            ├─► RRF fusion ─► cross-encoder rerank ─┐
   └─ (each covers the other's blind spot)                                       │
                                          AUGMENT (whole resume + spotlight) ─► GENERATE (Claude)
```

**Chunking is hierarchical (small-to-big / parent-document):** each **role/position**
is a *parent* (the unit returned to the model, for full context), and each **bullet**
is a *child* (the unit embedded and searched, for precision). Retrieval matches the
sharp child, then hands back its whole parent role — *index small, return big.*

This is **two-stage hybrid retrieval**, the same shape used in production systems:

- **Dense + BM25, fused with Reciprocal Rank Fusion** — semantic similarity *and*
  exact-keyword relevance, combined by rank position (no score calibration).
- **Cross-encoder reranking** — re-reads each candidate *together with* the query
  for a precise second-stage score (the biggest accuracy win).
- **Evaluation harness** — a labeled gold set with **Recall@k / MRR@k** that proves
  each upgrade helps (dense → hybrid → hybrid+rerank), with numbers, not vibes.
- **Verifiable citations** (Deep-understanding mode) — the résumé is sent as a
  custom-content citation document (one block per chunk), so the API returns
  `content_block_location` citations that map each claim back to the exact chunk —
  grounding the model can't fake, not "please cite" prompting.
- Everything except generation **runs locally** (`sentence-transformers`), no key.
- **Generation uses Claude** (`claude-opus-4-8`) with adaptive thinking. The whole
  resume is sent (cached) for complete context, plus the retrieved spotlight.

> No external vector DB on purpose: for one resume, brute-force NumPy search is
> correct and instant. The 7/10-tier value here is *retrieval engineering*
> (hybrid + rerank + eval), not infra you don't need at this scale.

> For a single resume, full-context prompting is already strong — so why RAG?
> Here, retrieval is the **teaching mechanism** and the **focusing lens**: it shows
> *which evidence* supports each answer, and it's exactly the technique you'd scale
> to thousands of documents.

## Setup

```bash
cd /Users/rajavvaru/RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # then paste your Anthropic API key into .env
# (or enter the key in the sidebar at runtime)

streamlit run app.py
```

The first run downloads the small embedding model (~90 MB) once.

## Project layout

| Path | What it is |
|------|------------|
| `app.py` | Streamlit entry point + sidebar + navigation |
| `rag/loader.py` | LOAD — PDF/txt/paste → clean text |
| `rag/chunker.py` | CHUNK — hierarchical: parents (roles) + children (bullets) |
| `rag/embedder.py` | EMBED — local dense vectors (bge, query-instruction aware) |
| `rag/lexical.py` | BM25 — dependency-free keyword relevance |
| `rag/reranker.py` | Cross-encoder second-stage reranking |
| `rag/store.py` | NumPy cosine-similarity engine |
| `rag/pipeline.py` | Hybrid retrieval (dense+BM25→RRF→rerank) + artifacts |
| `rag/evaluation.py` | Gold set + Recall@k / MRR@k harness |
| `rag/query_transform.py` | Multi-query + HyDE query transformation |
| `rag/faithfulness.py` | Answer-groundedness: citation coverage + LLM-judge |
| `rag/extraction.py` | Structured résumé → typed `Profile` (Structured Outputs) |
| `rag/corpus.py` | Multi-résumé pool: filter + cross-candidate ranking |
| `rag/generator.py` | GENERATE — Claude calls (analyst + recruiter personas) |
| `ui/walkthrough.py` | The illustrated self-guide (shows every stage) |
| `ui/evaluation.py` | Retrieval metrics + generation-faithfulness dashboard |
| `ui/profile.py` | Structured-profile view (strengths + red flags) |
| `ui/talent_pool.py` | Multi-résumé search / filter / rank |
| `ui/understanding.py` | Deep-understanding chat |
| `ui/recruiter.py` | Cynical-recruiter chat |
| `data/samples/` | Extra sample résumés for the talent pool |
| `data/sample_resume.txt` | A realistic sample to try instantly |

## Notes

- An Anthropic API key is needed only for the two Claude-powered modes. The full
  RAG walkthrough (load → retrieve, including the embedding-space plot) works
  without one.
- Token counts in the UI are rough estimates (~4 chars/token) for illustration.
