"""Evidence grounding.

A language model asked for a verbatim quote will sometimes produce a *plausible* one
instead. Grounding is the check that catches it: every quote an extractor returns is
located in the actual document before it becomes evidence, and a quote that cannot be
located produces nothing.

This is the cheapest high-value guarantee in the system. It costs a string search and it
converts "the model said so" into "page 4, section Results, paragraph 2 says so".

Matching tolerates the noise real PDFs introduce — ligatures, soft hyphens, column
wrapping, collapsed whitespace — but not invention. The similarity floor is deliberately
high; below it the safe answer is "not found".
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.logging import get_logger
from researchagent.models.document import PaperDocument, Paragraph, Section

logger = get_logger(__name__)

# PDF text extraction routinely leaves these behind; none of them change the words.
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_SOFT_HYPHEN = re.compile(r"[­‐‑]")  # noqa: RUF001 - the characters being normalised
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s+(\w)")
_WHITESPACE = re.compile(r"\s+")
_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}  # noqa: RUF001 - ditto

MIN_QUOTE_CHARS = 20


class GroundedQuote(BaseModel):
    """A quote proven to exist in the document, with where it was found."""

    model_config = {"frozen": True}

    quote: str = Field(min_length=1, description="Verbatim text as printed in the document")
    location: SourceLocation
    similarity: float = Field(ge=0.0, le=1.0)
    exact: bool


class EvidenceGrounder:
    """Locates extractor-supplied quotes in the source document."""

    name = "evidence_grounder"

    def __init__(self, document: PaperDocument, *, similarity_threshold: float = 0.85) -> None:
        self._document = document
        self._threshold = similarity_threshold
        # Normalised paragraph text is computed once per document, not per quote: an
        # extraction pass asks about dozens of quotes against the same paragraphs.
        self._index: list[tuple[Section, Paragraph, str]] = [
            (section, paragraph, normalise(paragraph.text))
            for section in document.sections
            for paragraph in section.paragraphs
        ]
        # PDF extraction splits a printed sentence across several blocks, and those
        # blocks become separate paragraphs. A genuine quote can therefore span two
        # paragraphs and match neither. Section-level text is the fallback: less precise
        # (no paragraph index) but still a real, checkable location.
        self._sections: list[tuple[Section, str]] = [
            (section, normalise(section.text)) for section in document.sections
        ]

    def ground(self, quote: str) -> GroundedQuote | None:
        """Find ``quote`` in the document, or return None.

        None is a real answer, not a failure to try: it means the extractor produced text
        that is not in the paper, and the caller must discard the extraction.
        """
        candidate = normalise(quote)
        if len(candidate) < MIN_QUOTE_CHARS:
            # Fragments match by accident. "accuracy" appears in every paper and grounds
            # nothing, so a too-short quote is treated as no quote at all.
            return None

        for locate in (self._find_exact, self._find_similar, self._find_in_section):
            found = locate(candidate)
            if found is not None:
                return found
        return None

    def evidence_for(self, claim: str, quote: str, *, produced_by: str) -> Evidence | None:
        """Ground a quote and wrap it as :class:`Evidence`, or return None."""
        grounded = self.ground(quote)
        if grounded is None:
            logger.debug(
                "quote_not_grounded",
                document_id=self._document.paper_id,
                produced_by=produced_by,
                quote=quote[:120],
            )
            return None

        return Evidence.from_text(
            claim=claim,
            # The document's own wording, not the model's paraphrase of it.
            quote=grounded.quote,
            location=grounded.location,
            produced_by=produced_by,
        )

    def _find_exact(self, candidate: str) -> GroundedQuote | None:
        for section, paragraph, normalised in self._index:
            position = normalised.find(candidate)
            if position >= 0:
                return GroundedQuote(
                    quote=paragraph.text,
                    location=self._locate(section, paragraph, position, len(candidate)),
                    similarity=1.0,
                    exact=True,
                )
        return None

    def _find_similar(self, candidate: str) -> GroundedQuote | None:
        """Best fuzzy match above the threshold.

        Scored against same-length windows of each paragraph rather than the paragraph as
        a whole, so a genuine quote inside a long paragraph is not penalised for the text
        surrounding it — which is exactly the case that matters, since paragraphs are
        usually far longer than the sentence an extractor cites.
        """
        best: GroundedQuote | None = None
        for section, paragraph, normalised in self._index:
            if not normalised:
                continue

            ratio, offset = _best_window_ratio(candidate, normalised)
            if ratio < self._threshold or (best is not None and ratio <= best.similarity):
                continue

            best = GroundedQuote(
                quote=paragraph.text,
                location=self._locate(section, paragraph, offset, len(candidate)),
                similarity=round(ratio, 4),
                exact=False,
            )
        return best

    def _find_in_section(self, candidate: str) -> GroundedQuote | None:
        """Locate a quote that crosses paragraph boundaries, at section precision."""
        for section, text in self._sections:
            if not text:
                continue

            position = text.find(candidate)
            if position >= 0:
                return self._section_hit(section, position, len(candidate), 1.0, exact=True)

            ratio, offset = _best_window_ratio(candidate, text)
            if ratio >= self._threshold:
                return self._section_hit(
                    section, offset, len(candidate), round(ratio, 4), exact=False
                )
        return None

    def _section_hit(
        self, section: Section, start: int, length: int, similarity: float, *, exact: bool
    ) -> GroundedQuote:
        excerpt = section.text[start : start + length] or section.text[:length]
        return GroundedQuote(
            quote=excerpt,
            location=SourceLocation(
                document_id=self._document.paper_id,
                page=section.page_start,
                section_id=section.id,
                section_title=section.title,
                char_start=start,
                char_end=start + length,
            ),
            similarity=similarity,
            exact=exact,
        )

    def _locate(
        self, section: Section, paragraph: Paragraph, start: int, length: int
    ) -> SourceLocation:
        return SourceLocation(
            document_id=self._document.paper_id,
            page=paragraph.page,
            section_id=section.id,
            section_title=section.title,
            paragraph_index=paragraph.index,
            bounding_box=paragraph.bounding_box,
            char_start=start,
            char_end=start + length,
        )


def normalise(text: str) -> str:
    """Collapse the differences PDF extraction introduces, keep the words.

    Applied to both sides of every comparison so that grounding survives real-world text
    without becoming loose enough to match invented text.
    """
    folded = unicodedata.normalize("NFKC", text)
    for source, target in {**_LIGATURES, **_QUOTES}.items():
        folded = folded.replace(source, target)
    folded = _SOFT_HYPHEN.sub("-", folded)
    # "distri- buted" across a line break is one word.
    folded = _HYPHEN_LINEBREAK.sub(r"\1\2", folded)
    return _WHITESPACE.sub(" ", folded).strip().lower()


def _best_window_ratio(candidate: str, haystack: str) -> tuple[float, int]:
    """Highest similarity between ``candidate`` and any same-length window of ``haystack``.

    Returns ``(ratio, offset)``. Windows step by half the candidate length: fine enough
    that a matching sentence always falls substantially inside one window, coarse enough
    to stay linear in paragraph length.
    """
    length = len(candidate)
    if len(haystack) <= length:
        return SequenceMatcher(None, candidate, haystack).ratio(), 0

    matcher = SequenceMatcher(None, candidate, "")
    step = max(length // 2, 1)
    best_ratio, best_offset = 0.0, 0

    for offset in range(0, len(haystack) - length + step, step):
        matcher.set_seq2(haystack[offset : offset + length])
        # quick_ratio is an upper bound; skip the expensive comparison when even the
        # optimistic score cannot beat what we already have.
        if matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_offset = ratio, offset

    return best_ratio, best_offset
