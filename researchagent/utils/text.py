"""Text normalisation shared across subsystems.

PDF extraction introduces differences that are not differences in words: ligatures, soft
hyphens, line-break hyphenation, collapsed whitespace, curly quotes. Every comparison in
the system — grounding a quote, matching an entity name, scoring a query — has to look
through that noise without becoming loose enough to match invented text.

Lives in ``utils`` because four subsystems depend on it and none of them should have to
import another to get it.
"""

from __future__ import annotations

import re
import unicodedata

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}  # noqa: RUF001 - the characters normalised
_SOFT_HYPHEN = re.compile(r"[­‐‑]")  # noqa: RUF001 - ditto
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s+(\w)")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Collapse extraction artefacts, keep the words.

    Applied to both sides of every comparison, so matching survives real-world text
    without admitting paraphrase as a quote.
    """
    folded = unicodedata.normalize("NFKC", text)
    for source, target in {**_LIGATURES, **_QUOTES}.items():
        folded = folded.replace(source, target)
    folded = _SOFT_HYPHEN.sub("-", folded)
    # "distri- buted" across a line break is one word.
    folded = _HYPHEN_LINEBREAK.sub(r"\1\2", folded)
    return _WHITESPACE.sub(" ", folded).strip().lower()


_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
        "from", "how", "in", "into", "is", "it", "its", "of", "on", "or", "our", "that",
        "the", "their", "there", "these", "this", "those", "to", "via", "we", "what",
        "which", "why", "with",
    }
)  # fmt: skip


def tokenise(text: str) -> set[str]:
    return {
        token for token in normalise(text).split() if len(token) > 2 and token not in _STOPWORDS
    }


def overlap(candidate: set[str], reference: set[str]) -> float:
    """Fraction of the query vocabulary present in the candidate.

    Asymmetric on purpose: a long description should not be penalised for containing
    words the query never mentioned.
    """
    if not candidate or not reference:
        return 0.0
    return len(candidate & reference) / len(reference)
