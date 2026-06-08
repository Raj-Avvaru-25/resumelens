"""The self-guide: an illustrated, step-by-step tour of how RAG works internally.

Each stage is shown with the ACTUAL data flowing through it, so a beginner can
see — not just read about — what retrieval-augmented generation does.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from rag import config, generator
from rag.pipeline import ResumeIndex, retrieve
from ui import components


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for illustration."""
    return max(1, round(len(text) / 4))


def _context_mode() -> str:
    """Read the Full-resume vs chunks-only choice set in the sidebar."""
    return st.session_state.get("context_mode", "full")


def _effort() -> str:
    """Read the reasoning-effort choice set in the sidebar."""
    return st.session_state.get("effort", config.CLAUDE_EFFORT)


def render(index: ResumeIndex, api_key: str | None):
    st.header("🔍 How RAG works — a guided tour")
    st.markdown(
        "RAG = **Retrieval-Augmented Generation**. Instead of hoping the model "
        "memorized your resume, we *retrieve* the most relevant pieces and *augment* "
        "the prompt with them. Below, each stage runs on your actual resume."
    )

    _stage_load(index)
    _stage_chunk(index)
    _stage_embed(index)
    _stage_retrieve(index)
    _stage_generate(index, api_key)
    _stage_effort_ab(index, api_key)


def _stage_load(index: ResumeIndex):
    with st.expander("① LOAD — get clean plain text", expanded=True):
        st.markdown(
            "Whatever the input (PDF, paste, .txt), we first flatten it to plain "
            "text. Models and math operate on text, not file formats."
        )
        c1, c2 = st.columns(2)
        c1.metric("Characters", f"{len(index.resume_text):,}")
        c2.metric("Est. tokens", f"{_est_tokens(index.resume_text):,}")
        st.text_area("The normalized resume text", index.resume_text, height=200, disabled=True)


def _stage_chunk(index: ResumeIndex):
    with st.expander(
        f"② CHUNK — hierarchical: {len(index.parents)} roles (parents) → "
        f"{len(index.chunks)} bullets (children)", expanded=True
    ):
        st.markdown(
            "We keep **two levels** to win on both context *and* precision:\n\n"
            "* **Parent** = a whole role / project / section — the unit we **return** "
            "to the model, so it sees the full arc of a job, not fragments.\n"
            "* **Child** = a single bullet — the unit we **embed and search**, so "
            "matching stays sharp.\n\n"
            "Retrieval matches a child, then hands back its parent. This is "
            "*small-to-big*: **index small, return big.**"
        )
        for p in index.parents:
            st.markdown(f"**{p.label()}** · ~{_est_tokens(p.text)} tokens · {len(p.child_ids)} children")
            for cid in p.child_ids:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ `#{cid}` {index.chunks[cid].text}", unsafe_allow_html=True)


def _stage_embed(index: ResumeIndex):
    with st.expander(f"③ EMBED — turn each child (bullet) into a {index.vectors.shape[1]}-D vector", expanded=True):
        st.markdown(
            "We embed the **children** (bullets), not the big parents — small units "
            "make sharper vectors. An **embedding** maps text to numbers so similar "
            "*meaning* lands nearby, even with zero shared words. Computed locally by "
            f"`{config.EMBEDDING_MODEL.split('/')[-1]}`. Each bullet is embedded **with "
            "its role title prepended**, so the vector knows which job it belongs to. "
            "First 8 dims of each child vector:"
        )
        preview = {
            index.chunks[i].label(): np.round(index.vectors[i][:8], 3).tolist()
            for i in range(len(index.chunks))
        }
        st.dataframe(preview, use_container_width=True)
        st.caption(
            f"Each vector actually has {index.vectors.shape[1]} numbers; we show 8. "
            "Vectors are length-normalized, so similarity is just a dot product."
        )


def _stage_retrieve(index: ResumeIndex):
    with st.expander("④ RETRIEVE — small-to-big, two-stage hybrid", expanded=True):
        st.markdown(
            "We match on the sharp **children** (bullets), then return their "
            "**parents** (whole roles). The matching itself is a real pipeline — "
            "watch each stage vote:\n\n"
            "1. **Dense** (semantic) — cosine similarity of meaning, per bullet.\n"
            "2. **BM25** (lexical) — exact-keyword relevance per bullet.\n"
            "3. **RRF fusion** — combine the two rankings by position.\n"
            f"4. **Cross-encoder rerank** — re-read each candidate bullet *with* the "
            f"query. Top bullets roll up to the top {config.TOP_K} **roles** returned."
        )
        query = st.text_input(
            "Ask something about the resume",
            value="What consensus algorithm did they use?",
            key="walk_query",
        )
        if not query.strip():
            return

        result = retrieve(index, query)
        returned_pids = result.returned_parent_ids
        chosen_child_ids = {c.id for c in index.chunks if c.parent_id in returned_pids}

        st.markdown("**Every stage's score for every child (bullet)**, sorted by final rank:")
        rows = sorted(
            result.scored_all,
            key=lambda s: (s.rerank_score if s.rerank_score is not None else -1e9, s.rrf_score),
            reverse=True,
        )
        st.dataframe(
            {
                "child": [s.chunk.label() for s in rows],
                "role (parent)": [index.parent_by_id[s.chunk.parent_id].title[:24] for s in rows],
                "dense": [round(s.dense_score, 3) for s in rows],
                "BM25": [round(s.bm25_score, 2) for s in rows],
                "RRF": [round(s.rrf_score, 4) for s in rows],
                "rerank": [
                    round(s.rerank_score, 3) if s.rerank_score is not None else None
                    for s in rows
                ],
                "role returned?": ["✅" if s.chunk.parent_id in returned_pids else "" for s in rows],
            },
            use_container_width=True,
        )
        st.caption(
            "A bullet can rank low on dense but high on BM25 (or vice versa); the "
            "rerank column re-orders the fused candidates; and the matched bullet's "
            "**whole role** gets returned (see the ✅ column — entire roles light up)."
        )

        _plot_embedding_space(index, result.query_vector, chosen_child_ids)

        st.markdown("**The returned roles (parents) handed to Claude:**")
        for h in result.retrieved_parents:
            mb = ", ".join(f"#{c}" for c in h.matched_child_ids) or "—"
            st.markdown(f"**{h.parent.label()}** · matched bullets: {mb}")
            st.code(h.parent.text, language=None)

        st.session_state["walk_last_result"] = result


def _plot_embedding_space(index, query_vector, chosen_ids):
    """Project the high-D vectors down to 2D so we can SEE retrieval happen."""
    if len(index.chunks) < 2:
        return
    all_vecs = np.vstack([index.vectors, query_vector[None, :]])
    coords = PCA(n_components=2).fit_transform(all_vecs)
    chunk_xy, query_xy = coords[:-1], coords[-1]

    fig = go.Figure()
    # non-retrieved chunks
    fig.add_trace(go.Scatter(
        x=[chunk_xy[i, 0] for i in range(len(index.chunks)) if index.chunks[i].id not in chosen_ids],
        y=[chunk_xy[i, 1] for i in range(len(index.chunks)) if index.chunks[i].id not in chosen_ids],
        mode="markers+text",
        text=[index.chunks[i].label() for i in range(len(index.chunks)) if index.chunks[i].id not in chosen_ids],
        textposition="top center", marker=dict(size=10, color="lightgray"), name="chunk",
    ))
    # retrieved chunks
    fig.add_trace(go.Scatter(
        x=[chunk_xy[i, 0] for i in range(len(index.chunks)) if index.chunks[i].id in chosen_ids],
        y=[chunk_xy[i, 1] for i in range(len(index.chunks)) if index.chunks[i].id in chosen_ids],
        mode="markers+text",
        text=[index.chunks[i].label() for i in range(len(index.chunks)) if index.chunks[i].id in chosen_ids],
        textposition="top center", marker=dict(size=14, color="#2E86DE"), name="retrieved",
    ))
    # the query
    fig.add_trace(go.Scatter(
        x=[query_xy[0]], y=[query_xy[1]], mode="markers+text", text=["YOUR QUESTION"],
        textposition="bottom center", marker=dict(size=18, color="#E74C3C", symbol="star"),
        name="query",
    ))
    fig.update_layout(
        title="Embedding space (PCA to 2D) — retrieval picks the chunks nearest your question",
        height=460, showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This is a 2D shadow of the full high-dimensional space, so distances are "
        "approximate — but the blue (retrieved) points really are the closest to the "
        "red star in the true space."
    )


def _stage_generate(index: ResumeIndex, api_key: str | None):
    with st.expander("⑤ GENERATE — Claude answers using the context", expanded=True):
        st.markdown(
            "Finally we send Claude the context chosen in **⚙️ Advanced settings** — "
            "the **whole resume** (cached) for complete understanding, *or* just the "
            "**retrieved chunks** for cheapest true-RAG — and ask your question. That's the "
            "'augmented generation' in RAG. The token readout appears under the answer."
        )
        result = st.session_state.get("walk_last_result")
        if result is None:
            st.info("Ask a question in step ④ first.")
            return
        if not api_key:
            st.warning("Add your Anthropic API key in the sidebar to run generation.")
            return
        if st.button("Answer with Claude", key="walk_generate"):
            client = generator.get_client(api_key)
            context_text, cacheable = generator.select_context(
                index.resume_text, result.augmented_context, _context_mode()
            )
            messages = [{"role": "user", "content": result.query}]
            usage_out: dict = {}
            st.write_stream(
                generator.stream_reply(
                    client,
                    context_text,
                    generator.UNDERSTANDING_PERSONA,
                    messages,
                    cacheable=cacheable,
                    effort=_effort(),
                    usage_out=usage_out,
                )
            )
            components.render_usage(usage_out.get("usage"))


def _stage_effort_ab(index: ResumeIndex, api_key: str | None):
    with st.expander("⚡ Effort A/B — measure what reasoning effort costs", expanded=False):
        st.markdown(
            "Run the **same question** at several effort levels back-to-back and "
            "compare token usage side by side. Input is (near) identical across "
            "levels — the difference lands almost entirely in **output** (the hidden "
            "thinking tokens). This answers *'how much does medium actually save?'* "
            "with your numbers, not an estimate."
        )
        if not api_key:
            st.warning("Add your Anthropic API key in the sidebar to run the A/B.")
            return

        question = st.text_input(
            "Question to test",
            value="How senior is this person, and defend your answer with specifics.",
            key="ab_query",
        )
        levels = st.multiselect(
            "Effort levels to compare",
            config.EFFORT_OPTIONS,
            default=["medium", "high"],
            key="ab_levels",
        )
        if not st.button("Run A/B", key="ab_run"):
            return
        if not question.strip() or not levels:
            st.info("Enter a question and pick at least one effort level.")
            return

        result = retrieve(index, question)
        context_text, cacheable = generator.select_context(
            index.resume_text, result.augmented_context, _context_mode()
        )
        client = generator.get_client(api_key)

        ordered = [e for e in config.EFFORT_OPTIONS if e in levels]  # low → max
        rows, answers = [], {}
        for eff in ordered:
            with st.spinner(f"Running at effort='{eff}'… (high/max think longer)"):
                messages = [{"role": "user", "content": question}]
                usage_out: dict = {}
                text = "".join(
                    generator.stream_reply(
                        client, context_text, generator.UNDERSTANDING_PERSONA,
                        messages, cacheable=cacheable, effort=eff, usage_out=usage_out,
                    )
                )
                rows.append((eff, usage_out.get("usage")))
                answers[eff] = text

        _render_ab(rows, answers)


def _render_ab(rows, answers):
    rows = [(e, u) for e, u in rows if u is not None]
    if not rows:
        st.error("No usage returned.")
        return

    p = config.PRICE_PER_1M

    def cost(u):
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        return (u.input_tokens * p["input"] + u.output_tokens * p["output"]
                + cr * p["cache_read"] + cw * p["cache_write"]) / 1_000_000

    effs = [e for e, _ in rows]
    outs = [u.output_tokens for _, u in rows]
    st.dataframe(
        {
            "effort": effs,
            "input": [u.input_tokens for _, u in rows],
            "output": outs,
            "cache read": [getattr(u, "cache_read_input_tokens", 0) or 0 for _, u in rows],
            "≈ $": [round(cost(u), 4) for _, u in rows],
        },
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_bar(x=effs, y=outs, marker_color="#2E86DE")
    fig.update_layout(
        height=360, title="Output tokens by effort (this is the part effort changes)",
        yaxis_title="output tokens",
    )
    st.plotly_chart(fig, use_container_width=True)

    base = max(outs)
    if base:
        lines = [f"`{e}` → {o:,} output tokens ({o / base * 100:.0f}% of the largest)"
                 for e, o in zip(effs, outs)]
        st.success("**Output-token comparison:**  \n" + "  \n".join(lines))
    st.caption(
        "Note: with full-resume + caching, the *second* run can show 'cache read' on "
        "input (the first run wrote it) — so judge the effort effect from the "
        "**output** column, which caching doesn't touch."
    )

    for eff in effs:
        with st.expander(f"Answer at effort='{eff}'  (eyeball the quality difference)"):
            st.markdown(answers[eff])
