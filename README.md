# Resume RAG 📄 — a glass-box RAG you can actually learn from

A Retrieval-Augmented Generation app that **understands résumés in context** — and
**shows you exactly how it does it**. It's three things at once:

1. **A teacher** — an illustrated, step-by-step tour of every internal RAG stage,
   running on real data (no hand-waving).
2. **A serious retrieval engine** — hierarchical small-to-big chunking, hybrid
   dense+BM25 retrieval with reranking, query transforms, verifiable citations,
   and a real evaluation harness.
3. **A recruiter's tool** — deep résumé understanding, a cynical-recruiter grill
   mode, structured profile extraction, and **multi-résumé candidate ranking**.

![How RAG works — guided tour](docs/screenshots/01-hero.png)

---

## ✨ Why this is different from a tutorial RAG

Most "RAG demos" are one dense lookup over fixed-size chunks. This one is built
the way production systems actually are — and every claim is measured, not asserted.

| | Typical tutorial RAG | **This project** |
|---|---|---|
| **Chunking** | fixed N-char splits (shred sentences & projects) | **hierarchical small-to-big** — role *parents* + bullet *children* |
| **Retrieval** | single dense cosine | **dense + BM25 → RRF fusion → cross-encoder rerank** |
| **Query** | raw question only | **+ multi-query expansion & HyDE** |
| **Grounding** | "trust the model" | **verifiable API citations** (each claim → exact bullet) |
| **Quality** | vibes | **measured**: Recall@k / MRR@k **+** answer-faithfulness |
| **Understanding** | keyword echo | **structured extraction** into a typed profile |
| **Scale** | one document | **multi-résumé pool**: filter + cross-candidate ranking |
| **Transparency** | black box | **glass-box** walkthrough of every stage |

It doesn't *say* the upgrades help — it shows the numbers (dense → hybrid →
hybrid+rerank), with headroom to spot regressions:

![Retrieval evaluation](docs/screenshots/05-eval.png)

---

## 🚀 Quickstart (fork & run)

**Prerequisites:** Python 3.9+ and ~1 GB free disk (PyTorch + first-run model
downloads). No GPU needed — everything runs on CPU.

```bash
# 1. Fork this repo on GitHub, then clone YOUR fork:
git clone https://github.com/<your-username>/resume-rag.git
cd resume-rag

# 2. One command does the rest (venv + install + launch):
./run.sh
```

Then open **http://localhost:8501**. The first launch downloads ~200 MB of local
models (embedder + reranker) once; after that the retrieval features work offline.

<details>
<summary>Prefer manual setup?</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
</details>

### Optional: enable the Claude-powered modes

Retrieval, the walkthrough, the evaluation metrics, and **the whole talent-pool
ranking** work with **no API key**. To unlock generation (deep-understanding chat,
cynical recruiter, structured-profile extraction, faithfulness judging, query
transforms), add an [Anthropic API key](https://console.claude.com/):

```bash
cp .env.example .env       # then paste your key into .env
# or just type it into the sidebar at runtime
```

---

## 🧭 A tour of the six modes

### 🔍 How RAG works — the self-guide
Walks every stage on your actual résumé. The **chunking** stage shows the
hierarchy plainly: each role is a *parent* (returned for full context); each
bullet is a *child* (embedded & searched for precision):

![Hierarchical chunking](docs/screenshots/02-chunking.png)

The **retrieve** stage is the glass box — every bullet's dense / BM25 / RRF /
rerank score, a 2D map of the embedding space, and the whole matched role that
gets handed to the model:

![Two-stage hybrid retrieval](docs/screenshots/03-retrieve.png)

### 📊 Retrieval evaluation
Recall@k / MRR@k over a labeled gold set across pipeline variants (and optional
multi-query / HyDE variants), **plus** generation-faithfulness scoring (citation
coverage, verify-rate, and an LLM-judge). *Measure, don't guess.*

### 🗂️ Structured profile *(needs key)*
Parses the résumé into a typed schema — roles, dates, seniority, skills,
quantified impact — and a recruiter-style **✅ strengths / 🚩 red flags** read.

### 🧠 Deep understanding *(needs key)*
Grounded Q&A over the résumé. Toggle **📎 citations** to tag every claim `[n]` and
link it to the exact bullet — *verified by the API, not asserted by the model.*

### 😠 Cynical recruiter *(needs key)*
A skeptical interviewer that grills the candidate, demands specifics, and
pressure-tests every claim — same retrieval stack underneath.

### 🏢 Talent pool — where RAG becomes necessary *(ranking is key-free)*
Search a **pool** of résumés: filter on the structured fields, then rank
candidates for a free-text need, each shown with the role that matched and why.
Here's "who has shipped fault-tolerant distributed systems?" — the two
distributed-systems engineers rank top, frontend/ML at the bottom:

![Multi-résumé candidate ranking](docs/screenshots/04-talent-pool.png)

---

## 🛠️ How it works

```
LOAD ─► CHUNK (role parents + bullet children) ─► EMBED children ─► STORE (+ BM25)   built once

QUESTION
   ├─ (optional) query transform: multi-query / HyDE
   ├─ dense (semantic) ranking  ┐
   ├─ BM25 (lexical) ranking    ├─► RRF fusion ─► cross-encoder rerank ─► roll up to ROLES
   └─ (each covers the other's blind spot)                                        │
                                       AUGMENT (resume or roles) ─► GENERATE (Claude) ─► [cited]
```

- **Small-to-big retrieval:** match the sharp child (bullet), return the whole
  parent (role) — *index small, return big.*
- **Hybrid + rerank:** semantic recall (dense) + exact-term recall (BM25), fused
  with Reciprocal Rank Fusion, then a cross-encoder re-reads query+chunk together.
- **Local + Claude split:** embeddings (`bge-small-en-v1.5`) and reranking
  (`ms-marco-MiniLM`) run locally; generation uses **Claude `claude-opus-4-8`**
  with adaptive thinking and prompt caching.
- **No external vector DB on purpose:** for this scale a NumPy brute-force search
  is correct and instant. The value here is *retrieval engineering*, not infra.

---

## 📁 Project layout

| Path | What it is |
|------|------------|
| `app.py` | Streamlit entry point + sidebar + navigation |
| `rag/loader.py` | LOAD — PDF/txt/paste → clean text (with soft-wrap reflow) |
| `rag/chunker.py` | CHUNK — hierarchical: parents (roles) + children (bullets) |
| `rag/embedder.py` | EMBED — local dense vectors (query-instruction aware) |
| `rag/lexical.py` | BM25 — dependency-free keyword relevance |
| `rag/reranker.py` | Cross-encoder second-stage reranking |
| `rag/store.py` | NumPy cosine-similarity engine |
| `rag/pipeline.py` | Hybrid small-to-big retrieval + per-stage artifacts |
| `rag/query_transform.py` | Multi-query + HyDE |
| `rag/faithfulness.py` | Answer grounding: citation coverage + LLM-judge |
| `rag/evaluation.py` | Gold set + Recall@k / MRR@k harness |
| `rag/extraction.py` | Résumé → typed `Profile` (Structured Outputs) |
| `rag/corpus.py` | Multi-résumé pool: filter + cross-candidate ranking |
| `rag/generator.py` | GENERATE — Claude calls (analyst + recruiter personas) |
| `ui/*.py` | One module per mode |
| `data/sample_resume.txt`, `data/samples/` | Sample résumés to try instantly |
| `scripts/capture_screenshots.py` | Regenerates the docs screenshots |

---

## 📸 Regenerating the screenshots

The images in `docs/screenshots/` are captured from the live app:

```bash
pip install playwright && python -m playwright install chromium
streamlit run app.py            # in one terminal
python scripts/capture_screenshots.py   # in another
```

---

## 🔭 Honest limitations & roadmap

This is a strong RAG **for its scale**; the remaining gaps only matter in
production:

- **Persistence / real vector DB** — the in-memory index rebuilds per session;
  at thousands of résumés you'd swap to pgvector/Qdrant with ANN + metadata
  filtering before retrieval.
- **Observability** — no tracing/cost dashboards yet.
- **CI-gated eval** — the gold set is small and tuned to the samples; a real
  deployment would track metrics in CI and gate changes on them.

---

## 🔒 Notes

- The two heaviest modes spend Claude tokens; use the sidebar **effort** selector
  and **token readout** (and the **effort A/B**) to control cost.
- `.env` is gitignored — no secrets are committed.
