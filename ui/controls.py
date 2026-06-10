"""Contextual per-page controls.

The tuning knobs (context mode, reasoning effort, query transform, citations) used
to live in the global sidebar and showed even on pages that ignore them. Here they
render inside a collapsible **Advanced** expander on *only* the pages that use them,
and they sync to ``st.session_state`` so older code that reads those keys (e.g. the
walkthrough) keeps working.
"""

from __future__ import annotations

import re

import streamlit as st

from rag import config

_CTX_FULL = "Full résumé (best answers)"
_CTX_CHUNKS = "Retrieved chunks only (cheapest)"


def advanced_controls(
    *, context: bool = False, effort: bool = False,
    transform: bool = False, cite: bool = False,
) -> dict:
    """Render an 'Advanced settings' expander with only the requested controls.

    Returns a dict with keys ``context_mode``, ``effort``, ``transform``, ``cite``
    (always present, defaulted from session_state) and writes each back to
    session_state under the same name so other modules can read them.
    """
    vals = {
        "context_mode": st.session_state.get("context_mode", "full"),
        "effort": st.session_state.get("effort", config.CLAUDE_EFFORT),
        "transform": st.session_state.get("transform", "none"),
        "cite": False,
    }
    if not any((context, effort, transform, cite)):
        return vals

    with st.sidebar.expander("Page settings", expanded=False):
        if context:
            choice = st.radio(
                "Context sent to Claude",
                [_CTX_FULL, _CTX_CHUNKS],
                key="ctrl_context",
                help=(
                    "Full résumé = whole document, cached once — best answers on a "
                    "small doc. Chunks only = true RAG, far fewer tokens, but may "
                    "miss context the retriever didn't surface."
                ),
            )
            vals["context_mode"] = "chunks" if "chunks" in choice else "full"
            st.session_state["context_mode"] = vals["context_mode"]

        if effort:
            vals["effort"] = st.select_slider(
                "Reasoning effort",
                options=config.EFFORT_OPTIONS,
                value=st.session_state.get("effort", config.CLAUDE_EFFORT),
                key="ctrl_effort",
                help=(
                    "How much Claude reasons before answering. Mostly affects hidden "
                    "thinking (output) tokens. 'medium' is usually plenty."
                ),
            )
            st.session_state["effort"] = vals["effort"]

        if transform:
            vals["transform"] = st.selectbox(
                "Query transform",
                ["none", "multi-query", "HyDE"],
                key="ctrl_transform",
                help=(
                    "Improve retrieval on vague/compound questions (needs API key). "
                    "multi-query = search several rewrites and fuse; HyDE = embed a "
                    "hypothetical answer instead of the raw question."
                ),
            )
            st.session_state["transform"] = vals["transform"]

        if cite:
            vals["cite"] = st.checkbox(
                "📎 Cite sources",
                value=False,
                key="ctrl_cite",
                help=(
                    "Tag every claim [n] and link it to the exact résumé chunk it "
                    "came from — verified by the API, not asserted by the model. "
                    "Uses the whole résumé as the citable document."
                ),
            )

    return vals


def page_header(icon: str, title: str, subtitle: str) -> None:
    """Consistent page title + one-line subtitle."""
    st.markdown(
        f"""
        <header class="rl-page-head">
          <div class="rl-eyebrow">{icon} ResumeLens</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def api_key_prompt(context: str = "this feature") -> None:
    """Inline 'enter your Anthropic key' box, shown at the exact point of need.

    Instead of telling the user to hunt for the key field in the sidebar, render
    the input right where they hit the wall. The widget key is derived from
    ``context`` so a page can show more than one prompt (e.g. the walkthrough's
    GENERATE and Effort A/B steps) without colliding; _api_key() reads any
    ``api_key_inline*`` value.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", context.lower()).strip("_") or "x"
    with st.container(border=True):
        st.markdown(f"#### 🔑 Add your Anthropic API key to use {context}")
        st.caption(
            "This is the one Claude-powered step. The RAG tour, evaluation, and "
            "talent-pool ranking all work without a key."
        )
        st.text_input(
            "Anthropic API key", type="password", placeholder="sk-ant-...",
            label_visibility="collapsed", key=f"api_key_inline_{slug}",
        )
        st.caption(
            "[Get a key →](https://console.claude.com/) · held only in this session, "
            "sent only to Anthropic, never stored or logged. "
            "Tip: use a low-limit key you can revoke."
        )
