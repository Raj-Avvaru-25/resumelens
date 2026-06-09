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
    components,
    controls,
    evaluation,
    profile,
    recruiter,
    talent_pool,
    understanding,
    walkthrough,
)

st.set_page_config(
    page_title="ResumeLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
components.inject_global_styles()

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
    """The active Anthropic key: env, else the sidebar field, else the inline
    field a page rendered at the point of need. One source of truth for both."""
    env = config.get_api_key()
    if env:
        return env
    return (
        st.session_state.get("api_key_sidebar")
        or st.session_state.get("api_key_home")
        or st.session_state.get("api_key_inline")
        or None
    )


def _resume_name() -> str:
    return st.session_state.get("resume_name") or "Bundled sample"


def _load_resume_file(up) -> str:
    if up.name.lower().endswith(".pdf"):
        return load_from_pdf(up.getvalue())
    return load_from_text(up.getvalue().decode("utf-8", errors="ignore"))


def _set_sample_resume() -> None:
    st.session_state["resume_text"] = load_from_text(SAMPLE_PATH.read_text(encoding="utf-8"))
    st.session_state["resume_name"] = "Bundled sample"


def _ensure_resume() -> None:
    """Seed the bundled sample once, so every page has a résumé to work with."""
    if not st.session_state.get("resume_text"):
        _set_sample_resume()


# --- résumé picker callbacks (update state only on real interaction) --------
def _cb_source() -> None:
    if st.session_state.get("resume_source") == _SRC_SAMPLE:
        _set_sample_resume()


def _cb_upload(widget_key: str) -> None:
    up = st.session_state.get(widget_key)
    if up is not None:
        st.session_state["resume_text"] = _load_resume_file(up)
        st.session_state["resume_name"] = up.name
        st.session_state["resume_source"] = _SRC_UPLOAD  # reflect in the sidebar picker


def _cb_paste() -> None:
    pasted = (st.session_state.get("resume_paste") or "").strip()
    if pasted:
        st.session_state["resume_text"] = load_from_text(pasted)
        st.session_state["resume_name"] = "Pasted résumé"


def _require_index():
    """Return the active index, or halt the page with guidance if no résumé yet."""
    text = st.session_state.get("resume_text")
    if not text:
        st.info("⬅️ Choose or upload a résumé from the sidebar to use this page.")
        st.stop()
    st.caption(f"📄 Active résumé: **{_resume_name()}**")
    return _get_index(text)


# --------------------------------------------------------------------------- #
# Sidebar (global setup)
# --------------------------------------------------------------------------- #

def _render_setup_sidebar() -> None:
    """API key + résumé source + model info. Stores results in session_state."""
    with st.sidebar:
        active_name = _resume_name() if st.session_state.get("resume_text") else "Bundled sample"
        st.markdown(
            f"""
            <div class="rl-sidebar-resume">
              <small>Active résumé</small>
              <strong>{active_name}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sources = [_SRC_SAMPLE, _SRC_UPLOAD, _SRC_PASTE]
        st.markdown("**Change résumé**")
        if config.DEMO_MODE:
            st.selectbox(
                "Résumé source", sources, index=0, disabled=True,
                label_visibility="collapsed",
                help="Upload & paste are available when you run ResumeLens yourself.",
            )
            _set_sample_resume()
            st.caption("Demo mode uses the bundled sample.")
        else:
            choice = st.selectbox(
                "Résumé source", sources, key="resume_source",
                on_change=_cb_source, label_visibility="collapsed",
            )
            if choice == _SRC_UPLOAD:
                st.caption("Upload a PDF or text résumé.")
                st.file_uploader(
                    "Choose résumé file", type=["pdf", "txt"],
                    key="resume_file_sidebar", on_change=_cb_upload,
                    args=("resume_file_sidebar",), label_visibility="collapsed",
                )
            elif choice == _SRC_PASTE:
                st.caption("Paste the full résumé below.")
                st.text_area(
                    "Paste résumé", height=160, key="resume_paste",
                    on_change=_cb_paste, placeholder="Paste résumé text…",
                    label_visibility="collapsed",
                )
            else:
                st.caption("Using the bundled sample résumé.")

        with st.expander("Claude API key · optional", expanded=False):
            if config.get_api_key():
                st.success("Loaded from environment.")
            else:
                st.text_input(
                    "Your Anthropic API key", type="password", placeholder="sk-ant-...",
                    label_visibility="collapsed", key="api_key_sidebar",
                    help="Held only in memory for your session and sent only to Anthropic.",
                )
                if _api_key():
                    st.success("Claude features unlocked.")
                else:
                    st.caption(
                        "[Get a key](https://console.claude.com/) to unlock generation "
                        "and profile extraction. Retrieval works without one."
                    )

        with st.expander("Models"):
            st.caption(
                f"LLM: `{config.CLAUDE_MODEL}`  \n"
                f"Embed: `{config.EMBEDDING_MODEL.split('/')[-1]}`  \n"
                f"Rerank: `{config.RERANK_MODEL.split('/')[-1]}`"
            )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def page_home() -> None:
    # Full-screen landing: sidebar hidden, full-bleed, with a page-in motion.
    components.enter_page("home", sidebar_collapsed=True)
    components.inject_home_layout()
    components.render_home_hero(_resume_name())

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True, key="rlcard_learn"):
        st.markdown('<span class="rl-chip">LEARN</span>', unsafe_allow_html=True)
        st.markdown("#### See how RAG works")
        st.caption("Chunking → retrieval → reranking → grounded generation, visualized.")
        if st.button("Open walkthrough", width="stretch"):
            st.switch_page(_PAGES["walkthrough"])
    with c2, st.container(border=True, key="rlcard_ask"):
        st.markdown('<span class="rl-chip">ASK</span>', unsafe_allow_html=True)
        st.markdown("#### Question a résumé")
        st.caption("Grounded answers, or pressure-test claims with the recruiter.")
        if st.button("Deep understanding", width="stretch"):
            st.switch_page(_PAGES["understanding"])
        if st.button("Cynical recruiter", width="stretch"):
            st.switch_page(_PAGES["recruiter"])
    with c3, st.container(border=True, key="rlcard_analyze"):
        st.markdown('<span class="rl-chip">ANALYZE</span>', unsafe_allow_html=True)
        st.markdown("#### Evaluate & rank")
        st.caption("Measure retrieval, check faithfulness, rank a candidate pool.")
        if st.button("Evaluation lab", width="stretch"):
            st.switch_page(_PAGES["evaluation"])
        if st.button("Talent pool", width="stretch"):
            st.switch_page(_PAGES["talent"])

    # Bring-your-own résumé, right on the landing (locked in the public demo).
    with st.container(border=True, key="rlcard_upload"):
        connected = bool(_api_key())
        if connected:
            res_col, key_col = st.container(), None
        else:
            res_col, key_col = st.columns(2, vertical_alignment="top")

        # Option — a single CTA: upload your résumé for personalized results
        with res_col:
            if config.DEMO_MODE:
                # Locked in the public demo; the CSS adds the 🔒 label + hover tooltip.
                st.file_uploader(
                    "Upload your résumé", type=["pdf", "txt"], disabled=True,
                    label_visibility="collapsed",
                )
                st.caption("Demo · exploring a sample résumé")
            else:
                st.file_uploader(
                    "Upload your résumé", type=["pdf", "txt"], key="resume_file_home",
                    on_change=_cb_upload, args=("resume_file_home",),
                    label_visibility="collapsed",
                )
                st.caption(f"Currently exploring · {_resume_name()}")

        # Option — Claude API key (only until connected; "?" hover explains how to get one)
        if key_col is not None:
            with key_col:
                st.text_input(
                    "🔑 Claude API key — optional", type="password",
                    placeholder="sk-ant-…", key="api_key_home",
                    help="How to get one: sign in at console.anthropic.com → API Keys → "
                         "Create Key, then paste it here.\n\nOptional · held only in this "
                         "session · sent only to Anthropic · never stored or logged.",
                )
                st.caption(
                    "Unlocks the Claude-powered pages.  "
                    "[Get a key →](https://console.claude.com/)"
                )


def page_walkthrough() -> None:
    components.enter_page("walkthrough", sidebar_collapsed=False)
    index = _require_index()
    controls.advanced_controls(context=True, effort=True)
    walkthrough.render(index, _api_key())


def page_understanding() -> None:
    components.enter_page("understanding", sidebar_collapsed=False)
    index = _require_index()
    v = controls.advanced_controls(context=True, effort=True, transform=True, cite=True)
    understanding.render(
        index, _api_key(), v["context_mode"], v["effort"], v["cite"], v["transform"]
    )


def page_recruiter() -> None:
    components.enter_page("recruiter", sidebar_collapsed=False)
    index = _require_index()
    v = controls.advanced_controls(context=True, effort=True, transform=True)
    recruiter.render(index, _api_key(), v["context_mode"], v["effort"], v["transform"])


def page_evaluation() -> None:
    components.enter_page("evaluation", sidebar_collapsed=False)
    # Eval runs on its own labeled sample index — no sidebar résumé required.
    evaluation.render(None, _api_key())


def page_profile() -> None:
    components.enter_page("profile", sidebar_collapsed=False)
    index = _require_index()
    profile.render(index, _api_key())


def page_talent() -> None:
    components.enter_page("talent", sidebar_collapsed=False)
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
    _ensure_resume()
    _render_setup_sidebar()
    nav.run()


if __name__ == "__main__":
    main()
