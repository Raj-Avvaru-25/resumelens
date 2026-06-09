"""Deep-understanding mode: a grounded analyst you can interrogate about the resume.

Every question retrieves the most relevant chunks (shown for transparency) and
feeds them, plus the full resume, to Claude — so answers reason over complete
context instead of matching keywords.
"""

from __future__ import annotations

import streamlit as st

from rag import generator, query_transform
from rag.pipeline import ResumeIndex, retrieve
from ui import components, controls

_HISTORY_KEY = "understanding_history"


def render(index: ResumeIndex, api_key: str | None, context_mode: str = "full",
           effort: str = "high", cite: bool = False, transform: str = "none"):
    st.header("🧠 Deep understanding")
    st.markdown(
        "Ask anything about the resume. Claude reads it in full context and grounds "
        "every claim in the text. Each answer shows which chunks were retrieved."
    )
    if cite:
        st.info(
            "📎 **Citations on.** Each claim is tagged `[n]` and linked below to the "
            "exact résumé chunk it came from — verified by the API, not asserted by "
            "the model."
        )

    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []

    st.caption("Try these:")
    cols = st.columns(3)
    suggestions = [
        "Summarize this person's real strengths.",
        "How senior is this person, honestly?",
        "What's the most technically impressive thing here?",
    ]
    pending = None
    for col, s in zip(cols, suggestions):
        if col.button(s, key=f"sugg_{s}"):
            pending = s

    for turn in st.session_state[_HISTORY_KEY]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    typed = st.chat_input("Ask about the resume…")
    deferred = st.session_state.pop("understanding_pending_q", None)
    question = pending or typed or deferred
    if not question:
        return

    if not api_key:
        # Bring key entry to the point of need; resume the question automatically after.
        st.session_state["understanding_pending_q"] = question
        controls.api_key_prompt("Deep Understanding")
        return

    _answer(index, api_key, question, context_mode, effort, cite, transform)


def _answer(index: ResumeIndex, api_key: str, question: str, context_mode: str,
            effort: str, cite: bool, transform: str):
    st.session_state[_HISTORY_KEY].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    client = generator.get_client(api_key)
    extra, override, note = (None, None, "")
    if transform != "none":
        with st.spinner(f"Transforming query ({transform})…"):
            extra, override, note = query_transform.apply(client, question, transform)
    result = retrieve(index, question, extra_queries=extra, dense_query_vector=override)
    with st.chat_message("assistant"):
        if note:
            st.caption(f"🔁 {note}")
        if cite:
            label = "retrieval preview (Claude cites from the full résumé below)"
        elif context_mode == "chunks":
            label = "exactly what Claude sees (whole roles)"
        else:
            label = "evidence (Claude also sees the full resume)"
        with st.expander(f"🔎 Retrieved {len(result.retrieved_parents)} roles — {label}"):
            for h in result.retrieved_parents:
                mb = ", ".join(f"#{c}" for c in h.matched_child_ids) or "—"
                st.markdown(f"**{h.parent.label()}** · matched bullets: {mb}")
                st.code(h.parent.text, language=None)

        # Send full chat history so follow-ups have memory.
        messages = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state[_HISTORY_KEY]
        ]

        if cite:
            answer = _stream_cited(index, client, messages, effort)
        else:
            context_text, cacheable = generator.select_context(
                index.resume_text, result.augmented_context, context_mode
            )
            usage_out: dict = {}
            answer = st.write_stream(
                generator.stream_reply(
                    client, context_text, generator.UNDERSTANDING_PERSONA,
                    messages, cacheable=cacheable, effort=effort, usage_out=usage_out,
                )
            )
            components.render_usage(usage_out.get("usage"))

    st.session_state[_HISTORY_KEY].append({"role": "assistant", "content": answer})


def _stream_cited(index: ResumeIndex, client, messages: list[dict], effort: str) -> str:
    """Stream a cited answer, then replace it with [n]-tagged text + a Sources list.

    Returns the plain answer text (no markers) to store in chat history.
    """
    placeholder = st.empty()
    buf: list[str] = []
    out: dict = {}
    for delta in generator.stream_cited(
        client, index.chunks, generator.UNDERSTANDING_PERSONA, messages,
        effort=effort, result_out=out,
    ):
        buf.append(delta)
        placeholder.markdown("".join(buf))

    segments = out.get("segments", [])
    cited_md, sources = _format_cited(segments)
    placeholder.markdown(cited_md)  # replace raw text with [n]-tagged version
    _render_sources(sources, index)
    components.render_usage(out.get("usage"))
    return "".join(seg["text"] for seg in segments)


def _format_cited(segments: list[dict]) -> tuple[str, list[tuple[int, dict]]]:
    parts: list[str] = []
    sources: list[tuple[int, dict]] = []
    n = 0
    for seg in segments:
        parts.append(seg["text"])
        for cit in seg.get("citations", []):
            n += 1
            parts.append(f" `[{n}]`")
            sources.append((n, cit))
    return "".join(parts), sources


def _render_sources(sources: list[tuple[int, dict]], index) -> None:
    if not sources:
        st.caption("⚠️ The model produced no verifiable citations for this answer.")
        return
    chunks = index.chunks  # children = the citable document blocks
    st.markdown("**📎 Sources** — each claim linked to the exact bullet (and its role):")
    for n, cit in sources:
        sb, eb = cit.get("start_block"), cit.get("end_block")
        if sb is None or not (0 <= sb < len(chunks)):
            st.markdown(f"**[{n}]** (unresolved) — “{cit.get('cited_text', '').strip()}”")
            continue
        ids = [i for i in range(sb, (eb if eb is not None else sb + 1)) if 0 <= i < len(chunks)]
        bullets = ", ".join(f"#{i}" for i in ids) if ids else f"#{sb}"
        role = index.parent_by_id[chunks[sb].parent_id].title
        quote = cit.get("cited_text", "").strip()
        st.markdown(f"**[{n}]** bullet {bullets} · *{role}* — “{quote}”")
