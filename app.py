"""Resume RAG — a glass-box RAG app for contextual resume understanding.

Run with:  streamlit run app.py

Three modes:
  • How RAG works  — an illustrated, beginner-friendly tour of the pipeline.
  • Deep understanding — a grounded analyst you can interrogate.
  • Cynical recruiter — a skeptical interviewer that grills the candidate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from rag import config
from rag.loader import load_from_pdf, load_from_text
from rag.pipeline import build_index
from ui import evaluation, profile, recruiter, talent_pool, understanding, walkthrough

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_resume.txt"

# Bump this whenever the index structure changes (e.g. new retrieval stages) so
# any stale index cached in an open session is rebuilt instead of crashing.
INDEX_VERSION = "2-hybrid-rerank"


def _resume_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_index(resume_text: str):
    """Build (or reuse) the resume index. Rebuilds when the text or version changes."""
    h = f"{INDEX_VERSION}:{_resume_hash(resume_text)}"
    if st.session_state.get("index_hash") != h:
        with st.spinner("Building index: chunk → embed → store (first run downloads the embedding model)…"):
            st.session_state["index"] = build_index(resume_text)
            st.session_state["index_hash"] = h
            # New resume invalidates old conversations.
            for k in ("understanding_history", "recruiter_history", "walk_last_result", "profile"):
                st.session_state.pop(k, None)
    return st.session_state["index"]


def _sidebar() -> tuple[str | None, str | None]:
    """Render the sidebar; return (resume_text, api_key)."""
    st.sidebar.title("📄 Resume RAG")

    # --- API key ---
    env_key = config.get_api_key()
    if env_key:
        st.sidebar.success("Anthropic API key loaded from environment.")
        api_key = env_key
    else:
        api_key = st.sidebar.text_input(
            "Anthropic API key", type="password",
            help="Needed only for the two Claude-powered modes. Retrieval works without it.",
        ) or None
        if not api_key:
            st.sidebar.info("No key yet — the RAG walkthrough still works fully.")

    st.sidebar.divider()

    # --- Resume source ---
    st.sidebar.subheader("Resume")
    source = st.sidebar.radio(
        "Source", ["Sample resume", "Upload (.pdf / .txt)", "Paste text"],
        label_visibility="collapsed",
    )

    resume_text: str | None = None
    if source == "Sample resume":
        resume_text = load_from_text(SAMPLE_PATH.read_text(encoding="utf-8"))
    elif source == "Upload (.pdf / .txt)":
        up = st.sidebar.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed")
        if up is not None:
            if up.name.lower().endswith(".pdf"):
                resume_text = load_from_pdf(up.getvalue())
            else:
                resume_text = load_from_text(up.getvalue().decode("utf-8", errors="ignore"))
    else:
        pasted = st.sidebar.text_area("Paste the resume text", height=240, label_visibility="collapsed")
        if pasted.strip():
            resume_text = load_from_text(pasted)

    st.sidebar.divider()
    st.sidebar.caption(
        f"LLM: `{config.CLAUDE_MODEL}`  \n"
        f"Embed: `{config.EMBEDDING_MODEL.split('/')[-1]}` (local)  \n"
        f"Rerank: `{config.RERANK_MODEL.split('/')[-1]}` (local)  \n"
        f"Retrieval: dense + BM25 → RRF → cross-encoder"
    )
    return resume_text, api_key


def main():
    resume_text, api_key = _sidebar()

    mode = st.sidebar.radio(
        "Mode",
        [
            "🔍 How RAG works",
            "📊 Retrieval evaluation",
            "🗂️ Structured profile",
            "🧠 Deep understanding",
            "😠 Cynical recruiter",
            "🏢 Talent pool",
        ],
    )

    # Quality/cost lever: what context Claude actually receives.
    st.sidebar.divider()
    ctx_label = st.sidebar.radio(
        "Context sent to Claude",
        ["Full resume (best answers)", "Retrieved chunks only (cheapest)"],
        help=(
            "Full resume = whole resume, cached once per resume — best answers on a "
            "small doc. Chunks only = true RAG, far fewer tokens, but can miss "
            "context the retriever didn't surface."
        ),
    )
    context_mode = "chunks" if "chunks" in ctx_label else "full"
    st.session_state["context_mode"] = context_mode  # walkthrough reads this

    effort = st.sidebar.selectbox(
        "Reasoning effort",
        config.EFFORT_OPTIONS,
        index=config.EFFORT_OPTIONS.index(config.CLAUDE_EFFORT),
        help=(
            "Controls how much Claude reasons before answering. Mostly affects "
            "OUTPUT (hidden thinking) tokens, not input. For resume Q&A, 'medium' "
            "is usually plenty; lower it to cut tokens, raise it for tough probing."
        ),
    )
    st.session_state["effort"] = effort  # walkthrough reads this

    cite = st.sidebar.checkbox(
        "📎 Cite sources (Deep understanding)",
        value=False,
        help=(
            "In Deep-understanding mode, return verifiable API citations: every "
            "claim is tagged [n] and linked to the exact résumé chunk it came from. "
            "Uses the whole résumé as a citable document (overrides the context "
            "choice above for that mode)."
        ),
    )

    transform = st.sidebar.selectbox(
        "Query transform",
        ["none", "multi-query", "HyDE"],
        help=(
            "Improve retrieval on vague/compound questions (needs API key). "
            "multi-query = search several rewrites and fuse; HyDE = embed a "
            "hypothetical answer instead of the raw question. Compare them in the "
            "Retrieval evaluation page."
        ),
    )
    st.session_state["transform"] = transform

    # Talent pool manages its own multi-résumé corpus — it doesn't need the
    # single résumé selected in the sidebar.
    if mode == "🏢 Talent pool":
        talent_pool.render(api_key)
        return

    if not resume_text:
        st.title("Resume RAG")
        st.info("Pick or upload a resume in the sidebar to begin.")
        return

    index = _get_index(resume_text)

    if mode == "🔍 How RAG works":
        walkthrough.render(index, api_key)
    elif mode == "📊 Retrieval evaluation":
        evaluation.render(index, api_key)
    elif mode == "🗂️ Structured profile":
        profile.render(index, api_key)
    elif mode == "🧠 Deep understanding":
        understanding.render(index, api_key, context_mode, effort, cite, transform)
    else:
        recruiter.render(index, api_key, context_mode, effort, transform)


if __name__ == "__main__":
    main()
