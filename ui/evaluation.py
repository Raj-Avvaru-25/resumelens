"""Evaluation page — prove the upgrades actually help, with numbers.

Two halves:
  1. RETRIEVAL quality (no API key) — Recall@k / MRR@k across pipeline variants,
     optionally including query-transform variants (those need a key).
  2. GENERATION faithfulness (needs key) — does the ANSWER stay grounded in the
     résumé? Measured by citation coverage + an LLM-judge groundedness score.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from rag import config, evaluation, faithfulness, generator, query_transform
from rag.loader import load_from_text
from ui import controls
from rag.pipeline import ResumeIndex, build_index, retrieve

_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_resume.txt"


def _sample_index() -> ResumeIndex:
    """Evaluation always runs on the bundled sample — the document the gold set is
    hand-labeled for. A gold set is meaningless against any other résumé."""
    if "eval_index" not in st.session_state:
        st.session_state["eval_index"] = build_index(
            load_from_text(_SAMPLE.read_text(encoding="utf-8"))
        )
    return st.session_state["eval_index"]


def render(index: ResumeIndex, api_key: str | None):
    st.header("📊 Evaluation")
    st.markdown(
        "The honest part of any RAG system: **measuring** quality instead of "
        "trusting it. Retrieval metrics below; generation faithfulness underneath."
    )
    st.info(
        "These metrics run on the **bundled sample résumé** — the document the gold "
        "set is hand-labeled for. Whatever résumé you pick in the sidebar won't "
        "change them: a gold set only applies to its own document. To evaluate a "
        "different résumé, write its labels in `rag/evaluation.py`."
    )

    sample = _sample_index()  # NOT the sidebar résumé — eval needs its labeled doc
    _retrieval_section(sample, api_key)
    st.divider()
    _faithfulness_section(sample, api_key)


# --- 1. Retrieval quality ----------------------------------------------------

def _retrieval_section(index: ResumeIndex, api_key: str | None):
    st.subheader("Retrieval quality")
    n_parents = len(index.parents)
    k = st.slider("k (top-k roles)", min_value=1, max_value=max(2, n_parents),
                  value=min(config.TOP_K, n_parents))
    st.caption(
        "**Recall@k** = fraction of questions whose correct role made the top k. "
        "**MRR@k** = average of 1/(rank of the first correct role). Measured at the "
        "**role (parent)** level, since that's what small-to-big returns."
    )
    include_transforms = st.checkbox(
        "Include query-transform variants (multi-query, HyDE) — uses API, slower",
        value=False, disabled=not api_key,
        help="Adds transformed-query pipelines so you can prove transforms help.",
    )

    if not st.button("Run retrieval eval", type="primary"):
        st.info(f"{len(evaluation.GOLD_SET)} labeled questions ready.")
        return

    with st.spinner("Scoring dense → hybrid → hybrid+rerank…"):
        results = evaluation.compare_variants(index, k)

    if include_transforms and api_key:
        client = generator.get_client(api_key)
        with st.spinner("Scoring transform variants (calling Claude per query)…"):
            results["Hybrid+rerank +multi-query"] = evaluation.evaluate(
                index, _transform_ranker(client, "multi-query"), k)
            results["Hybrid+rerank +HyDE"] = evaluation.evaluate(
                index, _transform_ranker(client, "HyDE"), k)

    names = list(results.keys())
    recalls = [results[n].recall_at_k for n in names]
    mrrs = [results[n].mrr_at_k for n in names]

    st.dataframe(
        {"pipeline": names,
         f"Recall@{k}": [round(r, 3) for r in recalls],
         f"MRR@{k}": [round(m, 3) for m in mrrs]},
        use_container_width=True,
    )
    fig = go.Figure()
    fig.add_bar(name=f"Recall@{k}", x=names, y=recalls)
    fig.add_bar(name=f"MRR@{k}", x=names, y=mrrs)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]), height=420,
                      title="Retrieval quality by pipeline variant")
    st.plotly_chart(fig, use_container_width=True)

    best = max(names, key=lambda n: (results[n].recall_at_k, results[n].mrr_at_k))
    st.success(f"Best on this set: **{best}**.")

    with st.expander("Per-question breakdown (rank of the first correct role)"):
        table = {"question": [q for q, _ in evaluation.GOLD_SET]}
        for n in names:
            table[n] = [pq["first_relevant_rank"] or "—" for pq in results[n].per_query]
        st.dataframe(table, use_container_width=True)
        st.caption("Lower rank = correct role found sooner. '—' = not retrieved.")


def _transform_ranker(client, mode: str):
    """A ranker (index, query) -> full parent order, applying a query transform."""
    def ranker(index: ResumeIndex, query: str) -> list[int]:
        extra, override, _ = query_transform.apply(client, query, mode)
        result = retrieve(index, query, top_k=len(index.parents),
                          extra_queries=extra, dense_query_vector=override)
        return [h.parent.id for h in result.retrieved_parents]
    return ranker


# --- 2. Generation faithfulness ----------------------------------------------

def _faithfulness_section(index: ResumeIndex, api_key: str | None):
    st.subheader("Generation faithfulness")
    st.markdown(
        "Retrieval metrics say nothing about whether the **answer** is grounded. "
        "Here we generate cited answers for a few questions and score:\n"
        "* **Citation coverage** — fraction of answer sentences that carry a citation.\n"
        "* **Verify rate** — fraction of citations whose quote is verbatim in the source.\n"
        "* **Groundedness (LLM-judge)** — a strict separate model call rating support."
    )
    if not api_key:
        controls.api_key_prompt("faithfulness scoring")
        return

    n = st.slider("How many questions to score", 1, min(6, len(evaluation.GOLD_SET)), 3)
    use_judge = st.checkbox("Also run the LLM-judge (extra call per question)", value=True)

    if not st.button("Run faithfulness eval"):
        st.info("Generates a cited answer per question, then scores grounding.")
        return

    client = generator.get_client(api_key)
    rows = []
    questions = [q for q, _ in evaluation.GOLD_SET][:n]
    for q in questions:
        with st.spinner(f"Answering + grading: {q[:48]}…"):
            out: dict = {}
            list(generator.stream_cited(
                client, index.chunks, generator.UNDERSTANDING_PERSONA,
                [{"role": "user", "content": q}], effort="low", result_out=out,
            ))
            segments = out.get("segments", [])
            cov = faithfulness.citation_coverage(segments, index.chunks)
            answer = "".join(s["text"] for s in segments)
            grounded = None
            if use_judge:
                grounded = faithfulness.judge_groundedness(
                    client, q, answer, index.resume_text).get("grounded_score")
            rows.append({"question": q, **cov, "grounded": grounded})

    st.dataframe(
        {
            "question": [r["question"] for r in rows],
            "coverage": [round(r["coverage"], 2) for r in rows],
            "verify rate": [round(r["verify_rate"], 2) for r in rows],
            "groundedness": [round(r["grounded"], 2) if r["grounded"] is not None else "—" for r in rows],
        },
        use_container_width=True,
    )
    avg_cov = sum(r["coverage"] for r in rows) / len(rows)
    avg_ver = sum(r["verify_rate"] for r in rows) / len(rows)
    msg = f"Avg citation coverage **{avg_cov:.0%}**, verify rate **{avg_ver:.0%}**"
    if use_judge:
        gs = [r["grounded"] for r in rows if r["grounded"] is not None]
        if gs:
            msg += f", judge groundedness **{sum(gs) / len(gs):.0%}**"
    st.success(msg + ".")
    st.caption(
        "High coverage + verify rate = the model is sticking to the résumé. The "
        "judge is a second, independent opinion. Low scores would flag hallucination."
    )
