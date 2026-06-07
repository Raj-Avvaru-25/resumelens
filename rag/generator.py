"""Stage 5 of RAG: GENERATE (with Claude).

This is where retrieval pays off. We hand Claude:

  1. The WHOLE resume (cached, so it's cheap to reuse across turns) — this gives
     it complete, contextual understanding rather than keyword matching.
  2. The RETRIEVED spotlight chunks — the evidence most relevant to the current
     question, so the model focuses on what matters.

Two personas live here:
  * understanding  — a precise analyst that explains the resume in full context.
  * recruiter      — a cynical, skeptical recruiter that grills the candidate.
"""

from __future__ import annotations

from collections.abc import Iterator

import anthropic

from . import config

# --- System prompts ----------------------------------------------------------

UNDERSTANDING_PERSONA = """You are a meticulous resume analyst. Your job is to \
understand a candidate's resume in COMPLETE CONTEXT — not by matching keywords, \
but by reasoning about what each line, role, and project actually means.

When you answer:
- Read claims in the context of the whole resume, not in isolation.
- Explain the *meaning and implications* of experience, not just restate it.
- For projects, infer scope, the candidate's specific role, the tech trade-offs, \
and the likely impact.
- Ground every claim in the resume. If something is not supported, say so plainly.
- Be concrete and specific. No flattery, no filler."""

RECRUITER_PERSONA = """You are a cynical, highly experienced technical recruiter \
who has seen thousands of inflated resumes. You are interviewing this candidate \
and you do NOT take claims at face value.

Your style:
- Probe relentlessly. For every impressive-sounding claim, ask "prove it": what \
exactly did YOU do, what was the hard part, what would have broken if you'd done \
it wrong?
- Hunt for vagueness, buzzwords, unquantified impact, and ownership ambiguity \
("we built" vs "I built").
- Pressure-test depth: ask the follow-up a real expert would ask about the tech \
or the project.
- Be sharp and skeptical, but professional — you are testing, not insulting.
- Always ground your challenges in specific lines/projects from the resume."""


# The two ways to supply context to Claude — a quality/cost trade.
CONTEXT_FULL = "full"      # whole resume, cached once per resume — best answers
CONTEXT_CHUNKS = "chunks"  # retrieved chunks only — cheapest, true RAG


def select_context(resume_text: str, augmented_context: str, mode: str) -> tuple[str, bool]:
    """Return (context_text, cacheable) for the chosen context mode.

    * Full resume  -> the whole resume; STABLE per resume, so we cache it once and
                      read it cheaply on every later question.
    * Chunks only  -> just the retrieved spotlight; true RAG, far fewer tokens, but
                      it CHANGES every question so caching it would never hit —
                      hence cacheable=False.
    """
    if mode == CONTEXT_CHUNKS:
        return augmented_context, False
    return resume_text, True


def _system_blocks(context_text: str, persona: str, cacheable: bool) -> list[dict]:
    """Build the system prompt: persona + the chosen context block.

    We send ONE context block, never the resume *and* the chunks (the chunks are
    substrings of the resume — that's double-paying). When the block is stable
    across questions (full-resume mode) we put a cache_control breakpoint on it so
    the persona+context prefix is written once and READ (~0.1× cost) thereafter.
    The volatile parts (question + chat history) live in `messages`, after this
    prefix, so they don't break the cache.

    Caveat: Anthropic only caches a prefix above a model minimum (~4096 tokens for
    Opus 4.8). A short resume may fall under that and silently won't cache —
    harmless, just no savings until the context is larger.
    """
    label = "FULL RESUME" if cacheable else "RETRIEVED CONTEXT"
    block: dict = {
        "type": "text",
        "text": f"{label} (authoritative source — ground everything here):\n\n"
        + context_text,
    }
    if cacheable:
        # 1-hour TTL so the cache survives a normal working session, not just 5 min.
        block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    return [{"type": "text", "text": persona}, block]


def get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def stream_reply(
    client: anthropic.Anthropic,
    context_text: str,
    persona: str,
    messages: list[dict],
    cacheable: bool = True,
    effort: str | None = None,
    usage_out: dict | None = None,
) -> Iterator[str]:
    """Stream Claude's reply token-by-token (yields text deltas).

    `context_text` is whatever context you chose to send (full resume or chunks);
    `cacheable` marks it stable enough to cache. `effort` overrides the default
    reasoning depth. `messages` is the running chat history.

    If `usage_out` (a dict) is passed, the final token-usage object is stored under
    `usage_out["usage"]` once the stream is fully consumed — so the caller can show
    input/output/cache token counts after `st.write_stream` finishes.
    """
    effort = effort or config.CLAUDE_EFFORT
    with client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        system=_system_blocks(context_text, persona, cacheable),
        messages=messages,
    ) as stream:
        yield from stream.text_stream
        if usage_out is not None:
            usage_out["usage"] = stream.get_final_message().usage


def stream_cited(
    client: anthropic.Anthropic,
    chunks,
    persona: str,
    messages: list[dict],
    effort: str | None = None,
    result_out: dict | None = None,
) -> Iterator[str]:
    """Stream a reply with VERIFIABLE, API-computed citations.

    We hand Claude the résumé as a custom-content citation document — one text
    block per chunk — with citations enabled. The API then returns
    `content_block_location` citations whose `start_block_index` is exactly the
    chunk id, so every cited claim is provably tied to a specific chunk (the model
    can't invent a citation that doesn't point at real source text).

    Yields text deltas for live display. After the stream completes, if
    `result_out` is passed, it is filled with:
      result_out["segments"] -> [{"text": str, "citations": [{cited_text, start_block, end_block}]}]
      result_out["usage"]    -> token usage object
    """
    effort = effort or config.CLAUDE_EFFORT

    document = {
        "type": "document",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": c.text} for c in chunks],
        },
        "title": "Candidate resume (chunked)",
        "citations": {"enabled": True},
        # Stable per resume -> cache it (only bites once it clears the model minimum).
        "cache_control": {"type": "ephemeral"},
    }
    full_messages = [
        {
            "role": "user",
            "content": [
                document,
                {
                    "type": "text",
                    "text": "Above is the candidate's resume as cited source "
                    "material. Answer my questions using it and cite the specific "
                    "lines that support each claim.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": "Understood — I'll ground every claim in the resume and cite "
            "the exact lines.",
        },
        *messages,
    ]

    segments: list[dict] = []
    cur: dict | None = None
    with client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        system=[{"type": "text", "text": persona}],
        messages=full_messages,
    ) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "content_block_start":
                block = getattr(event, "content_block", None)
                if getattr(block, "type", None) == "text":
                    cur = {"text": "", "citations": []}
            elif etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    if cur is None:
                        cur = {"text": "", "citations": []}
                    cur["text"] += delta.text
                    yield delta.text
                elif dtype == "citations_delta":
                    cit = getattr(delta, "citation", None)
                    if cit is not None:
                        if cur is None:
                            cur = {"text": "", "citations": []}
                        cur["citations"].append(
                            {
                                "cited_text": getattr(cit, "cited_text", ""),
                                "start_block": getattr(cit, "start_block_index", None),
                                "end_block": getattr(cit, "end_block_index", None),
                            }
                        )
            elif etype == "content_block_stop":
                if cur is not None:
                    segments.append(cur)
                    cur = None
        if cur is not None:
            segments.append(cur)
        final = stream.get_final_message()

    if result_out is not None:
        result_out["segments"] = segments
        result_out["usage"] = final.usage
