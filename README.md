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

## 🧠 Understanding the system

### The problem RAG solves
An LLM doesn't know your private documents, and you can't paste unlimited text into
a prompt. **RAG (Retrieval-Augmented Generation)** fixes both: *retrieve* the most
relevant slices of **your** data, then *augment* the prompt with them so the model
answers from facts, not memory. For a single résumé you could paste the whole
thing — but the moment you have *many*, you **must** retrieve. That's why the
**Talent pool** is where RAG stops being optional.

### The central trade-off (and how this app resolves it)
Chunk size is a tug-of-war:

- **Big chunks** (a whole role) → rich **context**, but **blurry retrieval** (one
  vector has to average everything in the role, so it matches nothing sharply).
- **Small chunks** (a single bullet) → **sharp retrieval**, but **no context**.

You can't win with one size — so we keep **both**: embed/search the small **bullets**
(precision) and return the whole **role** (context). That's **small-to-big**, and
it's the spine of the whole system.

### Concepts in 30 seconds
| Term | Plain meaning |
|------|---------------|
| **Embedding** | text → a vector of numbers; similar *meaning* lands at nearby vectors |
| **Dense retrieval** | match by meaning (embeddings + cosine similarity) |
| **BM25 (lexical)** | match by exact keywords (a smart TF-IDF); catches terms embeddings gloss over |
| **RRF** | Reciprocal Rank Fusion — merge two ranked lists by *position*, no score calibration needed |
| **Cross-encoder rerank** | re-reads `(query, chunk)` *together* for a precise relevance score; slow, so run only on a few candidates |
| **Small-to-big** | search the children (bullets), return the parents (roles) |
| **HyDE** | embed a *hypothetical answer* instead of the raw question |
| **Multi-query** | search several rewrites of the question and fuse the results |
| **Citations** | model output linked to exact source spans, *verified by the API* |
| **Recall@k / MRR@k** | did the right item make the top-k / how high did it rank |
| **Faithfulness** | is the generated answer actually supported by the source |

### The data model (the nouns in the code)
| Object | Is | Role in retrieval | Defined in |
|--------|----|-------------------|-----------|
| **Chunk** (child) | one bullet / line | the **searchable** unit (embedded + BM25) | `rag/chunker.py` |
| **Parent** | one role / section entry | the **returned** unit (full context) | `rag/chunker.py` |
| **Profile** | typed extraction (roles, skills, seniority…) | the **filterable** facts | `rag/extraction.py` |
| **ResumeIndex** | one résumé's children + vectors + BM25 + parents | a single searchable résumé | `rag/pipeline.py` |
| **Corpus** | many `ResumeIndex` + `Profile`s | the **Talent pool** | `rag/corpus.py` |

### Follow one query end-to-end
Trace *"What consensus algorithm did they use?"* on the sample résumé:

1. **Embed the query** (with the bge query instruction). → `embedder.embed_query`
2. **Score every bullet** two ways: dense (meaning) + BM25 (keywords). The bullet
   *"…fault-tolerant scheduler … Raft-backed metadata store…"* scores high.
3. **Fuse** the two rankings with RRF → a shortlist of candidate bullets.
4. **Rerank** the shortlist with the cross-encoder; the Raft bullet wins on a
   true `(query, bullet)` read. → `reranker.rerank`
5. **Roll up to the parent** → return the **whole Chronos project role**, not just
   the bullet. → `pipeline.retrieve`
6. **Augment**: hand Claude the returned role(s) — or the full résumé — plus the
   question. → `generator`
7. **Generate** the answer, optionally with a citation back to the exact bullet.

Every number along that path is visible live in **🔍 How RAG works → ④ Retrieve**.

### Why these design choices
| Decision | Why |
|----------|-----|
| Structure-aware chunks, not fixed-size | a résumé is *sections → roles → bullets*; cutting every N chars shreds meaning mid-thought |
| Hybrid, not dense-only | embeddings miss exact tokens (acronyms, `p95`, library names); BM25 covers that blind spot |
| A reranker | the biggest precision win — first-stage scores are noisy; the cross-encoder actually *reads* relevance |
| Full-résumé context in single-doc modes | it fits the window, so retrieval is the *focusing lens* + teaching device, not a crutch |
| Local embeddings + Claude only for generation | watch vectors build locally for free; spend tokens only where they add value |
| Eval on a labeled sample | a gold set only applies to *its own* document — see the Evaluation page's banner |

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
