"""Contextual per-page controls.

The tuning knobs (context mode, reasoning effort, query transform, citations) used
to live in the global sidebar and showed even on pages that ignore them. Here they
render inside a collapsible **Advanced** expander on *only* the pages that use them,
and they sync to ``st.session_state`` so older code that reads those keys (e.g. the
walkthrough) keeps working.
"""

from __future__ import annotations

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

    with st.expander("⚙️ Advanced settings", expanded=False):
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
    st.title(f"{icon} {title}")
    st.caption(subtitle)
