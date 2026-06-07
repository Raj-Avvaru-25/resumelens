"""Small shared UI helpers."""

from __future__ import annotations

import streamlit as st

from rag import config


def render_usage(usage) -> None:
    """Show the token usage + a rough cost estimate for one Claude response.

    `usage` is the SDK usage object. Key fields:
      * input_tokens               — uncached input (full price)
      * output_tokens              — generated tokens, INCLUDING hidden thinking
                                     (this is the part `effort` inflates/shrinks)
      * cache_read_input_tokens    — input served from cache (~0.1× price)
      * cache_creation_input_tokens — input written to cache this turn
    """
    if usage is None:
        return
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cread = getattr(usage, "cache_read_input_tokens", 0) or 0
    cwrite = getattr(usage, "cache_creation_input_tokens", 0) or 0

    p = config.PRICE_PER_1M
    cost = (
        inp * p["input"]
        + out * p["output"]
        + cread * p["cache_read"]
        + cwrite * p["cache_write"]
    ) / 1_000_000

    st.caption(
        f"🧾 **tokens** — input `{inp:,}` · output `{out:,}` "
        f"· cache read `{cread:,}` · cache write `{cwrite:,}`  ·  ≈ **${cost:.4f}**  \n"
        f"output is where reasoning effort shows up; cache read = the resume served "
        f"cheaply (only when the cached prefix exceeds the model's ~4k-token minimum)."
    )
