"""Central configuration for the Resume RAG app.

Keeping every tunable knob in one place makes the rest of the codebase easier to
read: when you wonder "what model is this using?" or "how big is a chunk?", the
answer is always here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
load_dotenv()

# --- Claude (generation) -----------------------------------------------------
# claude-opus-4-8 is Anthropic's most capable model. We use it for the two
# "thinking" features: contextual understanding and the cynical-recruiter grilling.
CLAUDE_MODEL = "claude-opus-4-8"

# Adaptive thinking lets Claude decide how much to reason per request.
# effort trades thoroughness against cost/latency. "high" is a good default.
# Effort mostly affects OUTPUT (hidden thinking) tokens, not input.
CLAUDE_EFFORT = "high"
EFFORT_OPTIONS = ["low", "medium", "high", "xhigh", "max"]

# Approximate Opus 4.8 pricing, $ per 1M tokens (for the in-app cost estimate).
# cache_write assumes the 1-hour TTL we use (~2x base input).
PRICE_PER_1M = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 10.0}

# Upper bound on generated tokens. We stream, so this can be generous.
CLAUDE_MAX_TOKENS = 8000


# --- Embeddings (dense retrieval) --------------------------------------------
# bge-small-en-v1.5 is a strong, compact modern retriever (384-d). It runs
# locally on CPU, no API key. It expects a short *instruction* prepended to the
# QUERY (not the documents) for best short-query→passage retrieval.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_QUERY_PROMPT = os.environ.get(
    "EMBEDDING_QUERY_PROMPT",
    "Represent this sentence for searching relevant passages: ",
)


# --- Reranking (second-stage precision) --------------------------------------
# A cross-encoder reads (query, chunk) TOGETHER and scores true relevance —
# far more accurate than comparing two independent embeddings. We use it to
# re-order the first-stage candidates. Also local, no API key.
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_RERANK = True


# --- Chunking ----------------------------------------------------------------
# Resumes are short and structured, so we keep chunks coherent (one experience
# entry / project / section per chunk) rather than blindly cutting every N chars.
# MAX_CHUNK_CHARS only kicks in to split an unusually long block.
MAX_CHUNK_CHARS = 700
CHUNK_OVERLAP_CHARS = 120


# --- Retrieval ---------------------------------------------------------------
# Two-stage retrieval:
#   1. Hybrid first stage (dense + BM25, fused with RRF) -> CANDIDATE_K candidates
#   2. Cross-encoder rerank -> TOP_K final chunks handed to the model
CANDIDATE_K = 8   # how many candidates to rerank (small corpus: rerank all)
TOP_K = 4         # final chunks used as the spotlight
RRF_K = 60        # Reciprocal Rank Fusion constant (standard default)


def get_api_key() -> str | None:
    """Return the Anthropic API key from the environment, or None if unset."""
    return os.environ.get("ANTHROPIC_API_KEY")
