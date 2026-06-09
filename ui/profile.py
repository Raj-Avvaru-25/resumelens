"""Structured profile page — the résumé parsed into a typed schema.

Shows the extracted Profile (roles, skills, seniority, quantified impact) plus a
recruiter-style read (strengths + red flags), and the raw JSON. Cached per résumé.
"""

from __future__ import annotations

import streamlit as st

from rag import extraction, generator
from rag.pipeline import ResumeIndex
from ui import controls

_KEY = "profile"


def render(index: ResumeIndex, api_key: str | None):
    controls.page_header(
        "◇",
        "Structured profile",
        "Transform unstructured résumé text into a typed, comparable candidate intelligence layer.",
    )
    if not api_key:
        controls.api_key_prompt("Structured Profile")
        return

    if st.button("Extract structured profile", type="primary"):
        with st.spinner("Extracting with Claude (structured outputs)…"):
            st.session_state[_KEY] = extraction.extract_profile(
                generator.get_client(api_key), index.resume_text
            )

    profile = st.session_state.get(_KEY)
    if profile is None:
        st.info("Click **Extract structured profile** to parse the résumé.")
        return

    _render_profile(profile)


def _render_profile(p) -> None:
    st.subheader(p.name or "—")
    st.caption(p.headline)
    c1, c2, c3 = st.columns(3)
    c1.metric("Experience", f"{p.years_experience:g} yrs")
    c2.metric("Seniority", p.seniority or "—")
    c3.metric("Roles", str(len(p.roles)))

    if p.skills:
        st.markdown("**Skills**")
        st.markdown(" ".join(f"`{s}`" for s in p.skills))

    st.markdown("### Roles")
    for r in p.roles:
        with st.container(border=True):
            st.markdown(f"**{r.title}** · {r.company} · {r.start}–{r.end}")
            if r.summary:
                st.write(r.summary)
            if r.technologies:
                st.caption("Tech: " + ", ".join(r.technologies))
            for q in r.quantified_impact:
                st.markdown(f"- 📈 {q}")

    col_s, col_r = st.columns(2)
    with col_s:
        st.markdown("### ✅ Strengths")
        for s in p.strengths:
            st.markdown(f"- {s}")
    with col_r:
        st.markdown("### 🚩 Red flags")
        if p.red_flags:
            for r in p.red_flags:
                st.markdown(f"- {r}")
        else:
            st.caption("None surfaced.")

    with st.expander("Raw extracted JSON"):
        st.json(p.model_dump())
