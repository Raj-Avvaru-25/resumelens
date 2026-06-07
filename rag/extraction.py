"""Structured extraction — turn a free-text résumé into a typed schema.

Resume *understanding* isn't only a retrieval problem; a lot of it is extraction.
Once, at ingest, we ask Claude (via Structured Outputs) to parse the résumé into a
strict schema: roles with dates and tech, an experience estimate, skills, and a
recruiter-style read (genuine strengths + red flags).

Why it matters:
  * Standalone understanding: a clean, comparable profile instead of prose.
  * It's the data layer for filtering/ranking across MANY résumés (feature #4):
    "≥5 yrs, Senior, knows Kafka" is a query over these fields, not over vectors.

We use the SDK's `messages.parse()` with a Pydantic model, which validates the
response against the schema for us.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import config

_SYSTEM = (
    "You extract a precise, structured profile from a résumé. Be strictly faithful "
    "to the text — never invent companies, dates, or achievements. For "
    "quantified_impact, include ONLY achievements that carry a concrete number or "
    "measurable outcome. For red_flags, surface what a skeptical recruiter would "
    "probe: vague or unverifiable claims, unexplained gaps, inflated scope, and "
    "ownership ambiguity ('we built' vs 'I built')."
)


class Role(BaseModel):
    title: str = Field(description="Job title held")
    company: str = Field(description="Employer name")
    start: str = Field(description="Start year, e.g. '2020' (best effort)")
    end: str = Field(description="End year or 'Present'")
    summary: str = Field(description="One-sentence summary of what they did in this role")
    technologies: list[str] = Field(description="Concrete tools/languages used in this role")
    quantified_impact: list[str] = Field(
        description="Achievements with concrete numbers/metrics only; empty if none"
    )


class Profile(BaseModel):
    name: str
    headline: str = Field(description="One-line professional headline")
    years_experience: float = Field(description="Best estimate of total years of professional experience")
    seniority: str = Field(description="Best judgment: one of Junior, Mid, Senior, Staff, Principal")
    skills: list[str] = Field(description="De-duplicated skills/technologies across the whole résumé")
    roles: list[Role]
    education: list[str] = Field(description="Degrees / institutions")
    strengths: list[str] = Field(description="Genuine, evidence-backed strengths")
    red_flags: list[str] = Field(
        description="Vague/unquantified claims, gaps, inflated scope, or ownership ambiguity"
    )


def extract_profile(client, resume_text: str) -> Profile | None:
    """Parse a résumé into a validated Profile. Returns None on refusal/parse failure."""
    resp = client.messages.parse(
        model=config.CLAUDE_MODEL,
        max_tokens=4000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": resume_text}],
        output_format=Profile,
    )
    return resp.parsed_output
