"""Faithfulness / groundedness evaluation for GENERATED answers.

Retrieval metrics tell you whether the right context was fetched. They say nothing
about whether the ANSWER is actually grounded in it — i.e. whether the model
hallucinated. Two complementary signals:

* Citation coverage (cheap, deterministic): with native citations enabled, what
  fraction of substantive answer sentences carry a citation, and does each
  `cited_text` truly appear in the source chunk? High coverage + high verify-rate
  means the model is sticking to the document.
* LLM-judge groundedness (optional, costs a call): a separate, strict model call
  rates how well the answer is supported by the context and lists unsupported
  claims.
"""

from __future__ import annotations

import json

from . import config


def citation_coverage(segments: list[dict], chunks) -> dict:
    """Measure citation coverage over a cited answer's segments.

    `segments` come from generator.stream_cited's result_out; `chunks` are the
    children (the citable document blocks).
    """
    sentences = cited = verified = total_cit = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if len(text) < 15:  # skip connective fragments like " and "
            continue
        sentences += 1
        cits = seg.get("citations", [])
        if cits:
            cited += 1
        for c in cits:
            total_cit += 1
            sb = c.get("start_block")
            quote = (c.get("cited_text") or "").strip().lower()
            if sb is not None and 0 <= sb < len(chunks) and quote and quote in chunks[sb].text.lower():
                verified += 1
    return {
        "sentences": sentences,
        "cited": cited,
        "coverage": (cited / sentences) if sentences else 0.0,
        "citations": total_cit,
        "verified": verified,
        "verify_rate": (verified / total_cit) if total_cit else 0.0,
    }


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded_score": {"type": "number"},
        "verdict": {"type": "string"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded_score", "verdict", "unsupported_claims"],
    "additionalProperties": False,
}


def judge_groundedness(client, question: str, answer: str, context: str) -> dict:
    """LLM-as-judge: is every claim in `answer` supported by `context`?

    Returns {grounded_score: 0..1, verdict: str, unsupported_claims: [str]}.
    """
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=900,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        system=(
            "You are a strict grounding grader. Given CONTEXT (a résumé) and an "
            "ANSWER, decide whether every factual claim in the ANSWER is supported "
            "by the CONTEXT. grounded_score is the fraction of claims supported "
            "(1.0 = fully grounded, 0.0 = unsupported). List any unsupported or "
            "exaggerated claims."
        ),
        messages=[
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}",
            }
        ],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"grounded_score": 0.0, "verdict": "parse error", "unsupported_claims": []}
