"""ResumeLens — a glass-box RAG app for contextual résumé understanding.

Run with:  streamlit run app.py

The UI is organized into three jobs, grouped in the sidebar:
  • Learn   — "How RAG works": an illustrated tour of the pipeline.
  • Ask     — Deep understanding · Cynical recruiter: interrogate one résumé.
  • Analyze — Evaluation · Structured profile · Talent pool: measure & scale.

Global setup (API key + résumé source) lives in the sidebar; per-page tuning knobs
live in an 'Advanced settings' expander on the pages that actually use them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from rag import config
from rag.loader import load_from_pdf, load_from_text
from rag.pipeline import build_index
from ui import (
    controls,
    evaluation,
    profile,
    recruiter,
    talent_pool,
    understanding,
    walkthrough,
)

st.set_page_config(page_title="ResumeLens", page_icon="🔍", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_resume.txt"

# Bump this whenever the index structure changes so a stale cached index rebuilds.
INDEX_VERSION = "2-hybrid-rerank"

# Résumé-source pill labels (with icons) for the sidebar picker.
_SRC_SAMPLE = "📄 Sample"
_SRC_UPLOAD = "⬆️ Upload"
_SRC_PASTE = "📋 Paste"


# --------------------------------------------------------------------------- #
# Shared state helpers
# --------------------------------------------------------------------------- #

def _resume_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_index(resume_text: str):
    """Build (or reuse) the résumé index. Rebuilds when text or version changes."""
    h = f"{INDEX_VERSION}:{_resume_hash(resume_text)}"
    if st.session_state.get("index_hash") != h:
        with st.spinner("Building index: chunk → embed → store (first run downloads the embedding model)…"):
            st.session_state["index"] = build_index(resume_text)
            st.session_state["index_hash"] = h
            for k in ("understanding_history", "recruiter_history", "walk_last_result", "profile"):
                st.session_state.pop(k, None)
    return st.session_state["index"]


def _api_key() -> str | None:
    return st.session_state.get("api_key")


def _resume_name() -> str:
    text = st.session_state.get("resume_text") or ""
    first = text.strip().split("\n", 1)[0].strip()
    return first[:60] or "Untitled résumé"


def _require_index():
    """Return the active index, or halt the page with guidance if no résumé yet."""
    text = st.session_state.get("resume_text")
    if not text:
        st.info("⬅️ Pick or upload a résumé in the sidebar (**Setup**) to use this page.")
        st.stop()
    st.caption(f"📄 Active résumé: **{_resume_name()}**  ·  change it in the sidebar")
    return _get_index(text)


# --------------------------------------------------------------------------- #
# Sidebar (global setup)
# --------------------------------------------------------------------------- #

def _render_setup_sidebar() -> None:
    """API key + résumé source + model info. Stores results in session_state."""
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Setup")

        # --- Résumé (step 1) — prominent pill picker --------------------------
        st.markdown("**📄 Résumé** — pick a source")
        sources = [_SRC_SAMPLE, _SRC_UPLOAD, _SRC_PASTE]
        resume_text: str | None = None
        if config.DEMO_MODE:
            # Public demo: show all options but disabled, locked to the sample.
            st.segmented_control(
                "Résumé source", sources, default=_SRC_SAMPLE,
                disabled=True, label_visibility="collapsed",
                help="Upload & paste are available when you run ResumeLens yourself.",
            )
            resume_text = load_from_text(SAMPLE_PATH.read_text(encoding="utf-8"))
            st.caption(
                "🎬 **Demo mode** — locked to the bundled sample. **Upload** & "
                "**Paste** are enabled in the full version (clone the repo)."
            )
        else:
            choice = st.segmented_control(
                "Résumé source", sources, default=_SRC_SAMPLE,
                label_visibility="collapsed",
            ) or _SRC_SAMPLE
            if choice == _SRC_SAMPLE:
                resume_text = load_from_text(SAMPLE_PATH.read_text(encoding="utf-8"))
            elif choice == _SRC_UPLOAD:
                up = st.file_uploader("Upload a .pdf or .txt résumé", type=["pdf", "txt"])
                if up is not None:
                    resume_text = (
                        load_from_pdf(up.getvalue()) if up.name.lower().endswith(".pdf")
                        else load_from_text(up.getvalue().decode("utf-8", errors="ignore"))
                    )
            else:
                pasted = st.text_area("Paste the résumé text", height=200)
                if pasted.strip():
                    resume_text = load_from_text(pasted)
        st.session_state["resume_text"] = resume_text

        # --- Claude key (step 2, optional) -----------------------------------
        st.markdown("**🔑 Claude API key** · *optional*")
        env_key = config.get_api_key()
        if env_key:
            st.success("API key loaded from environment.")
            st.session_state["api_key"] = env_key
        else:
            key = st.text_input(
                "Your Anthropic API key", type="password", placeholder="sk-ant-...",
                label_visibility="collapsed",
                help="Held only in memory for your session and used solely to call "
                     "Anthropic — never stored, logged, or shared.",
            ) or None
            st.session_state["api_key"] = key
            if key:
                st.success("Key set — Claude pages unlocked.")
            else:
                st.caption(
                    "Unlocks the Claude pages (Deep understanding · Cynical recruiter "
                    "· Structured profile). The RAG tour, eval & talent pool work with "
                    "no key.  \n[Get a key →](https://console.claude.com/) · lives only "
                    "in this session, sent only to Anthropic.  \n💡 Tip: use a "
                    "low-limit key you can revoke afterwards."
                )

        with st.expander("🔧 Models & retrieval"):
            st.caption(
                f"LLM: `{config.CLAUDE_MODEL}`  \n"
                f"Embed: `{config.EMBEDDING_MODEL.split('/')[-1]}` (local)  \n"
                f"Rerank: `{config.RERANK_MODEL.split('/')[-1]}` (local)  \n"
                f"Retrieval: dense + BM25 → RRF → cross-encoder"
            )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def page_home() -> None:
    st.title("🔍 ResumeLens")
    st.markdown(
        "#### A glass-box RAG you can actually learn from.\n"
        "It retrieves the most relevant pieces of a résumé and grounds every answer "
        "in the text — and **shows you exactly how it works at every step.**"
    )
    st.divider()
    st.markdown("**Three things at once — pick a section from the sidebar, or jump in:**")

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("#### 🔍 Learn")
        st.caption(
            "A guided, illustrated tour of every RAG stage — chunking, embedding, "
            "hybrid retrieval, reranking, generation — running on real data."
        )
        if st.button("How RAG works →", use_container_width=True):
            st.switch_page(_PAGES["walkthrough"])
    with c2, st.container(border=True):
        st.markdown("#### 🧠 Ask")
        st.caption(
            "Interrogate a single résumé. A grounded analyst that cites its sources — "
            "or a cynical recruiter that grills the candidate's boldest claims."
        )
        if st.button("Deep understanding →", use_container_width=True):
            st.switch_page(_PAGES["understanding"])
        if st.button("Cynical recruiter →", use_container_width=True):
            st.switch_page(_PAGES["recruiter"])
    with c3, st.container(border=True):
        st.markdown("#### 📊 Analyze")
        st.caption(
            "Measure quality (Recall@k / MRR@k + faithfulness), extract a structured "
            "profile, or rank a whole pool of candidates for a free-text need."
        )
        if st.button("Evaluation →", use_container_width=True):
            st.switch_page(_PAGES["evaluation"])
        if st.button("Talent pool →", use_container_width=True):
            st.switch_page(_PAGES["talent"])

    if not _api_key():
        st.info(
            "💡 No API key needed to explore the **RAG tour**, **retrieval eval**, or "
            "the **talent pool** ranking. Add a key in the sidebar to unlock the "
            "Claude-powered Q&A, the recruiter, and profile extraction."
        )


def page_walkthrough() -> None:
    index = _require_index()
    controls.advanced_controls(context=True, effort=True)
    walkthrough.render(index, _api_key())


def page_understanding() -> None:
    index = _require_index()
    v = controls.advanced_controls(context=True, effort=True, transform=True, cite=True)
    understanding.render(
        index, _api_key(), v["context_mode"], v["effort"], v["cite"], v["transform"]
    )


def page_recruiter() -> None:
    index = _require_index()
    v = controls.advanced_controls(context=True, effort=True, transform=True)
    recruiter.render(index, _api_key(), v["context_mode"], v["effort"], v["transform"])


def page_evaluation() -> None:
    # Eval runs on its own labeled sample index — no sidebar résumé required.
    evaluation.render(None, _api_key())


def page_profile() -> None:
    index = _require_index()
    profile.render(index, _api_key())


def page_talent() -> None:
    controls.advanced_controls(transform=True)
    talent_pool.render(_api_key())


# Created in main(); referenced by page_home() for st.switch_page.
_PAGES: dict = {}


def main() -> None:
    global _PAGES
    home = st.Page(page_home, title="Home", icon="🏠", default=True)
    walk = st.Page(page_walkthrough, title="How RAG works", icon="🔍")
    understand = st.Page(page_understanding, title="Deep understanding", icon="🧠")
    recruit = st.Page(page_recruiter, title="Cynical recruiter", icon="😠")
    evaluate = st.Page(page_evaluation, title="Evaluation", icon="📊")
    prof = st.Page(page_profile, title="Structured profile", icon="🗂️")
    talent = st.Page(page_talent, title="Talent pool", icon="🏢")

    _PAGES = {
        "walkthrough": walk, "understanding": understand, "recruiter": recruit,
        "evaluation": evaluate, "profile": prof, "talent": talent,
    }

    nav = st.navigation(
        {
            "": [home],
            "Learn": [walk],
            "Ask": [understand, recruit],
            "Analyze": [evaluate, prof, talent],
        }
    )
    _render_setup_sidebar()
    nav.run()


if __name__ == "__main__":
    main()
