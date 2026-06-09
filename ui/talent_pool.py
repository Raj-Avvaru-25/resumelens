"""🏢 Talent pool — multi-résumé search, filtering, and candidate ranking.

This is the use case where RAG is actually necessary: you can't fit a pool of
résumés in context, so you retrieve. Load several résumés, optionally extract
structured profiles to filter on (seniority / years / skills), then rank candidates
for a free-text need — each shown with the role that matched and why.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from rag import config, corpus, generator, query_transform
from rag.loader import load_from_pdf, load_from_text
from ui import controls

_ROOT = Path(__file__).resolve().parent.parent
_MAIN_SAMPLE = _ROOT / "data" / "sample_resume.txt"
_SAMPLE_DIR = _ROOT / "data" / "samples"

_SENIORITIES = ["Junior", "Mid", "Senior", "Staff", "Principal"]


def _sample_docs() -> dict[str, str]:
    docs: dict[str, str] = {}
    files = ([_MAIN_SAMPLE] if _MAIN_SAMPLE.exists() else []) + sorted(_SAMPLE_DIR.glob("*.txt"))
    for f in files:
        text = load_from_text(f.read_text(encoding="utf-8"))
        name = text.split("\n")[0].strip() or f.stem
        docs[name] = text
    return docs


def _signature(docs: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(docs):
        h.update(name.encode())
        h.update(docs[name].encode())
    return h.hexdigest()


def render(api_key: str | None):
    controls.page_header(
        "⌁",
        "Talent pool",
        "Filter a candidate corpus, rank it against a free-text need, and inspect the evidence behind every match.",
    )

    # --- assemble the document set -------------------------------------------
    docs: dict[str, str] = {}
    if config.DEMO_MODE:
        # Public demo: show the controls but disabled, locked to the sample pool.
        docs.update(_sample_docs())
        st.checkbox("Use bundled sample pool", value=True, disabled=True)
        st.file_uploader(
            "Add résumés (.pdf / .txt)", type=["pdf", "txt"],
            accept_multiple_files=True, disabled=True,
        )
        st.caption(
            "🎬 **Demo mode** — ranking the bundled sample pool. Adding your own "
            "résumés is enabled in the full version (clone the repo)."
        )
        uploads = None
    else:
        if st.checkbox("Use bundled sample pool", value=True):
            docs.update(_sample_docs())
        uploads = st.file_uploader(
            "Add résumés (.pdf / .txt)", type=["pdf", "txt"], accept_multiple_files=True
        )
    for up in uploads or []:
        text = (load_from_pdf(up.getvalue()) if up.name.lower().endswith(".pdf")
                else load_from_text(up.getvalue().decode("utf-8", errors="ignore")))
        name = text.split("\n")[0].strip() or up.name
        docs[name] = text

    if not docs:
        st.info("Enable the sample pool or upload résumés to begin.")
        return

    # --- build (and cache) the corpus ----------------------------------------
    sig = _signature(docs)
    if st.session_state.get("corpus_sig") != sig:
        with st.spinner(f"Building index for {len(docs)} résumés (chunk → embed → store)…"):
            st.session_state["corpus"] = corpus.build_corpus(docs)
            st.session_state["corpus_sig"] = sig
    candidates = st.session_state["corpus"]
    st.success(f"Pool ready: **{len(candidates)} candidates**.")

    # --- structured profiles (enable filtering) ------------------------------
    have_profiles = all(c.profile is not None for c in candidates)
    with st.expander("🗂️ Structured profiles (needed for filtering)", expanded=not have_profiles):
        st.caption("Extract a typed profile per résumé so you can filter on seniority / years / skills.")
        if not api_key:
            controls.api_key_prompt("profile extraction")
        elif st.button("Extract profiles for the pool", type="primary"):
            with st.spinner("Extracting profiles with Claude…"):
                corpus.extract_profiles(generator.get_client(api_key), candidates)
            have_profiles = True
        if have_profiles:
            st.dataframe(
                {
                    "candidate": [c.name for c in candidates],
                    "seniority": [c.profile.seniority for c in candidates],
                    "years": [c.profile.years_experience for c in candidates],
                    "top skills": [", ".join(c.profile.skills[:6]) for c in candidates],
                },
                use_container_width=True,
            )

    # --- filters --------------------------------------------------------------
    st.subheader("Filter")
    if not have_profiles:
        st.caption("Filters are disabled until profiles are extracted.")
    fc1, fc2, fc3 = st.columns(3)
    min_years = fc1.slider("Min years", 0, 15, 0, disabled=not have_profiles)
    seniorities = fc2.multiselect("Seniority", _SENIORITIES, disabled=not have_profiles)
    skills_raw = fc3.text_input("Required skills (comma-sep)", disabled=not have_profiles)
    required_skills = [s for s in skills_raw.split(",") if s.strip()]

    # --- search ---------------------------------------------------------------
    st.subheader("Rank candidates")
    query = st.text_input(
        "What are you looking for?",
        value="Who has actually shipped fault-tolerant distributed systems?",
    )
    if not st.button("Search the pool", type="primary"):
        return
    if not query.strip():
        st.info("Enter what you're looking for.")
        return

    # apply metadata filters first
    pool = [
        c for c in candidates
        if corpus.passes_filter(c.profile, min_years, seniorities, required_skills)
    ]
    if not pool:
        st.warning("No candidates pass the filters.")
        return

    # optional query transform (reuses the sidebar setting)
    transform = st.session_state.get("transform", "none")
    extra, override = None, None
    if transform != "none" and api_key:
        with st.spinner(f"Transforming query ({transform})…"):
            extra, override, note = query_transform.apply(generator.get_client(api_key), query, transform)
            if note:
                st.caption(f"🔁 {note}")

    with st.spinner(f"Ranking {len(pool)} candidates…"):
        hits = corpus.search(pool, query, extra_queries=extra, dense_query_vector=override)

    st.markdown(f"**{len(hits)} candidates** ranked (filtered from {len(candidates)}):")
    for rank, h in enumerate(hits, start=1):
        with st.container(border=True):
            head = f"**{rank}. {h.candidate.name}**  ·  match score `{h.score:.3f}`"
            if h.candidate.profile:
                p = h.candidate.profile
                head += f"  ·  {p.seniority} · {p.years_experience:g} yrs"
            st.markdown(head)
            mb = ", ".join(f"#{c}" for c in h.matched_child_ids) or "—"
            st.markdown(f"Best-matching role — **{h.best_parent.title}** (bullets {mb}):")
            st.code(h.best_parent.text, language=None)
    st.caption(
        "Score = the cross-encoder relevance of each candidate's single best-matching "
        "bullet to your query — a fair cross-candidate signal. Filtering happens on the "
        "structured profiles *before* ranking."
    )
