"""Stage 0 of RAG: LOAD.

Turn whatever the user gives us (a pasted string, a .txt, or a .pdf) into one
clean block of plain text. Everything downstream operates on plain text.
"""

from __future__ import annotations

import io
import re


def load_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF's raw bytes.

    PDFs store text per-page; we concatenate the pages with blank lines between
    them so the chunker can still see paragraph boundaries.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return normalize_text("\n\n".join(pages))


def load_from_text(raw: str) -> str:
    """Clean up a pasted or uploaded plain-text resume."""
    return normalize_text(raw)


def normalize_text(text: str) -> str:
    """Tidy whitespace without destroying the line/paragraph structure.

    We deliberately KEEP single newlines (bullets) and blank lines (paragraph
    breaks) because the chunker uses them as signals. We only collapse runs of
    spaces and excessive blank lines.
    """
    # Normalize Windows/Mac line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines down to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (but not newlines) into one space.
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Re-join sentences that a PDF / text editor hard-wrapped mid-thought.
    text = unwrap_soft_breaks(text)
    return text.strip()


# Terminal punctuation that legitimately ends a line (not a soft wrap).
_LINE_ENDERS = ".!?:;)"
# Characters that signal the *next* line starts a fresh bullet/entry.
_BULLET_STARTS = "-•*–—▪◦●"


def unwrap_soft_breaks(text: str) -> str:
    """Merge lines that were wrapped mid-sentence back into one logical line.

    Resumes (especially extracted from PDFs) often break a single bullet across
    two physical lines:

        "Redesigned the posting service, cutting duplicate
         postings to near zero after a major outage."

    We join such a continuation onto the previous line when ALL of these hold:
      * the previous line is long (looks like it hit a wrap margin),
      * it does NOT end in sentence-ending punctuation,
      * the next line starts with a lowercase letter and is not a new bullet.

    Real bullets and entries (which start capitalized or with a bullet marker)
    are left untouched, so we don't accidentally glue separate points together.
    """
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if (
            out
            and out[-1].strip()
            and stripped
            and len(out[-1]) >= 45
            and out[-1][-1] not in _LINE_ENDERS
            and stripped[0] not in _BULLET_STARTS
            and stripped[0].islower()
        ):
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(line)
    return "\n".join(out)
