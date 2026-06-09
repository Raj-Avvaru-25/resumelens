"""Cynical-recruiter mode: a skeptical interviewer that grills the candidate.

Same RAG plumbing as the understanding mode, but the persona is a hard-nosed
recruiter who pressure-tests every claim. Two ways in:
  * "Grill me" — Claude opens with its toughest questions about the resume.
  * Free chat — you answer, it pushes back like a real interview.
"""

from __future__ import annotations

import streamlit as st

from rag import generator, query_transform
from rag.pipeline import ResumeIndex, retrieve
from ui import components, controls

_HISTORY_KEY = "recruiter_history"

_OPENING_PROMPT = (
    "Open the interview. Pick the 3 most impressive or suspicious claims in this "
    "resume and grill me on them with your sharpest, most specific questions. "
    "Number them."
)


def render(index: ResumeIndex, api_key: str | None, context_mode: str = "full",
           effort: str = "high", transform: str = "none"):
    controls.page_header(
        "⚠",
        "The cynical recruiter",
        "Pressure-test bold claims, vague ownership, and technical depth with a skeptical interviewer.",
    )
    components.render_editorial_image(
        "recruiter-room.png",
        "Interview mode",
        "Make every claim defensible.",
        "A skeptical recruiter uses the retrieved résumé evidence to ask the follow-up questions that matter.",
    )

    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []

    c1, c2, _ = st.columns([1.1, 1.4, 4.5])
    start = c1.button("🔥 Grill me", type="primary")
    if c2.button("Reset interview"):
        st.session_state[_HISTORY_KEY] = []
        st.rerun()

    for turn in st.session_state[_HISTORY_KEY]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    typed = st.chat_input("Answer the recruiter…")

    # Resolve the intended action: the button, a typed answer, or one we deferred
    # while waiting for the user to add an API key.
    deferred = st.session_state.pop("recruiter_pending", None)
    if start:
        question, retrieval = _OPENING_PROMPT, "key claims projects impact ownership"
    elif typed:
        question, retrieval = typed, typed
    elif deferred:
        question, retrieval = deferred
    else:
        question = retrieval = None

    if question is None:
        return

    if not api_key:
        # Bring key entry to the point of need; resume the action automatically after.
        st.session_state["recruiter_pending"] = (question, retrieval)
        controls.api_key_prompt("the Cynical Recruiter")
        return

    _turn(index, api_key, question, retrieval, context_mode, effort, transform)


def _turn(index: ResumeIndex, api_key: str, user_text: str, retrieval_query: str,
          context_mode: str, effort: str, transform: str):
    st.session_state[_HISTORY_KEY].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    client = generator.get_client(api_key)
    extra, override, note = (None, None, "")
    if transform != "none":
        with st.spinner(f"Transforming query ({transform})…"):
            extra, override, note = query_transform.apply(client, retrieval_query, transform)
    result = retrieve(index, retrieval_query, extra_queries=extra, dense_query_vector=override)
    with st.chat_message("assistant"):
        if note:
            st.caption(f"🔁 {note}")
        with st.expander("🔎 Resume evidence the recruiter is leaning on (whole roles)"):
            for h in result.retrieved_parents:
                mb = ", ".join(f"#{c}" for c in h.matched_child_ids) or "—"
                st.markdown(f"**{h.parent.label()}** · matched bullets: {mb}")
                st.code(h.parent.text, language=None)

        context_text, cacheable = generator.select_context(
            index.resume_text, result.augmented_context, context_mode
        )
        messages = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state[_HISTORY_KEY]
        ]
        usage_out: dict = {}
        reply = st.write_stream(
            generator.stream_reply(
                client,
                context_text,
                generator.RECRUITER_PERSONA,
                messages,
                cacheable=cacheable,
                effort=effort,
                usage_out=usage_out,
            )
        )
        components.render_usage(usage_out.get("usage"))
    st.session_state[_HISTORY_KEY].append({"role": "assistant", "content": reply})
