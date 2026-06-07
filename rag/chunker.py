"""Stage 1 of RAG: CHUNK — hierarchical, small-to-big.

The hard tension in RAG is chunk size:
  * BIG chunks (a whole role) give the model great CONTEXT but BLURRY retrieval —
    one vector has to average everything in the role, so it matches nothing sharply.
  * SMALL chunks (one bullet) give SHARP retrieval but fragmented context.

You can't win with a single size, so we keep BOTH levels:

  Parent  = one role / position / section entry   (the unit we RETURN — full context)
    └─ Child = one bullet / line / sentence        (the unit we EMBED + SEARCH — precision)

Retrieval matches on the sharp child, then returns its parent (the whole role).
That is "small-to-big" / parent-document retrieval: index small, return big.

Each child is embedded WITH its role title prepended ("Senior SWE, Meta — <bullet>")
so a bullet's vector still knows which role it belongs to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A CHILD: the small, searchable unit (a bullet / line / sentence)."""

    id: int
    text: str          # the bullet text itself (shown to users, cited)
    section: str        # resume section, e.g. "EXPERIENCE"
    parent_id: int      # which role/entry this bullet belongs to
    embed_text: str     # what we actually embed/index (role title + bullet)

    def label(self) -> str:
        return f"#{self.id} · {self.section}"


@dataclass
class Parent:
    """A role / position / section entry: the CONTEXT unit returned to the model."""

    id: int
    title: str          # first line of the entry (e.g. "Senior SWE, Meta (2020-2023)")
    text: str           # the full entry (title + all bullets)
    section: str
    child_ids: list[int] = field(default_factory=list)

    def label(self) -> str:
        t = self.title if len(self.title) <= 38 else self.title[:37] + "…"
        return f"P{self.id} · {self.section} · {t}"


# --- header detection (same heuristic as before) -----------------------------

_KNOWN_HEADINGS = {
    "summary", "objective", "experience", "work experience",
    "professional experience", "employment", "projects", "education",
    "skills", "technical skills", "certifications", "achievements",
    "awards", "publications", "interests", "contact", "profile",
}


def _looks_like_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return False
    low = stripped.lower().rstrip(":")
    if low in _KNOWN_HEADINGS:
        return True
    if stripped.endswith((".", ",", ";")) and not stripped.endswith(":"):
        return False
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    return bool(letters) and stripped.upper() == stripped and len(letters) >= 3


def _parse_blocks(text: str) -> list[tuple[str, list[str]]]:
    """Split text into (section, lines-of-one-entry) blocks, like before."""
    current_section = "HEADER"
    blocks: list[tuple[str, list[str]]] = []
    buffer: list[str] = []

    def flush():
        if buffer:
            blocks.append((current_section, buffer.copy()))
            buffer.clear()

    for line in text.split("\n"):
        if _looks_like_header(line):
            flush()
            current_section = line.strip().rstrip(":").upper()
            continue
        if line.strip() == "":
            flush()
            continue
        buffer.append(line)
    flush()
    return blocks


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_long_line(text: str, max_chars: int) -> list[str]:
    """Only an unusually long single bullet gets split — on sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    pieces, cur = [], ""
    for sent in _SENTENCE.split(text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            pieces.append(cur.strip())
            cur = sent
        else:
            cur = f"{cur} {sent}".strip() if cur else sent
    if cur.strip():
        pieces.append(cur.strip())
    return pieces or [text]


def chunk_resume(text: str, max_chars: int = 700, overlap: int = 120) -> tuple[list[Parent], list[Chunk]]:
    """Build the parent/child hierarchy.

    Returns (parents, children). Parents are whole role/section entries (never
    char-split). Children are the bullets within them — the searchable units.
    `overlap` is unused here (kept for signature stability); parents aren't split.
    """
    parents: list[Parent] = []
    children: list[Chunk] = []
    cid = 0

    for section, lines in _parse_blocks(text):
        nonempty = [ln.strip() for ln in lines if ln.strip()]
        if not nonempty:
            continue
        pid = len(parents)
        title = nonempty[0]
        block_text = "\n".join(nonempty).strip()
        child_ids: list[int] = []

        for line in nonempty:
            for piece in _split_long_line(line, max_chars):
                is_title = (piece == title) and not child_ids  # first line = title
                embed = piece if is_title else f"{title} — {piece}"
                children.append(
                    Chunk(id=cid, text=piece, section=section, parent_id=pid, embed_text=embed)
                )
                child_ids.append(cid)
                cid += 1

        parents.append(Parent(id=pid, title=title, text=block_text, section=section, child_ids=child_ids))

    return parents, children
