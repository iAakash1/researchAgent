"""Section detection.

Turns positioned text blocks into a titled, ordered section hierarchy.

A heading is identified by *relative* evidence, never absolute thresholds: font size
compared against the document's own modal body size, boldness, brevity, and section
numbering. A two-column ACM paper and an A4 preprint disagree about what "large" means,
so the baseline is measured per document.

Section names are matched against variant patterns rather than exact strings — "5
Evaluation", "Experimental Results" and "Results and Discussion" all have to reach the
same canonical :class:`SectionKind`, or downstream code would need to know every
author's naming taste.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from researchagent.config.schemas import SectionDetectionConfig
from researchagent.core.logging import get_logger
from researchagent.models.document import Paragraph, Section, SectionKind
from researchagent.models.layout import RawDocument, TextBlock

logger = get_logger(__name__)

# Ordered: the first pattern that matches wins, so specific beats general
# ("related work" must be tested before "work").
_SECTION_PATTERNS: tuple[tuple[SectionKind, re.Pattern[str]], ...] = (
    (SectionKind.ABSTRACT, re.compile(r"^abstract$|^summary$")),
    (SectionKind.RELATED_WORK, re.compile(r"related\s+work|prior\s+work|literature\s+review")),
    (SectionKind.BACKGROUND, re.compile(r"^background|preliminar|motivation")),
    (SectionKind.INTRODUCTION, re.compile(r"^introduction|^overview$")),
    (
        SectionKind.METHODOLOGY,
        # `design` and `model` are deliberately anchored: bare words match ordinary
        # titles like "Threats and Opportunities in Widget Design".
        re.compile(
            r"method|approach|^design\b|system\s+design|architecture|implementation"
            r"|system\s+model"
        ),
    ),
    (SectionKind.EXPERIMENTS, re.compile(r"experiment|^setup\b|study\s+design|benchmark")),
    (SectionKind.EVALUATION, re.compile(r"evaluation|^analysis\b")),
    (SectionKind.RESULTS, re.compile(r"result|finding")),
    (SectionKind.LIMITATIONS, re.compile(r"limitation|threats?\s+to\s+validity")),
    (SectionKind.FUTURE_WORK, re.compile(r"future\s+work|future\s+direction|open\s+problem")),
    (SectionKind.DISCUSSION, re.compile(r"discussion|implication")),
    (SectionKind.CONCLUSION, re.compile(r"conclusion|concluding|closing\s+remark")),
    (SectionKind.ACKNOWLEDGEMENTS, re.compile(r"acknowledg")),
    (SectionKind.REFERENCES, re.compile(r"^references?$|^bibliography$|^works\s+cited$")),
    (SectionKind.APPENDIX, re.compile(r"^appendix|^supplement")),
)

# "3", "3.1", "IV", "A.2" — with or without a trailing separator.
_NUMBERING = re.compile(r"^\s*((\d+(\.\d+)*)|([IVXLC]+)|([A-Z](\.\d+)*))[.)]?\s+(?=\S)")
_TRAILING_PUNCTUATION = re.compile(r"[\s.:]+$")


class HeadingCandidate(BaseModel):
    block_index: int
    title: str
    kind: SectionKind
    level: int
    confidence: float
    page: int


class SectionDetector:
    """Detects headings and groups the blocks between them into sections."""

    name = "section_detector"

    def __init__(self, config: SectionDetectionConfig | None = None) -> None:
        self._config = config or SectionDetectionConfig()

    def detect(self, document: RawDocument) -> tuple[Section, ...]:
        blocks = [block for block in document.blocks if not block.is_blank]
        if not blocks:
            return ()

        body_size = document.body_text_size()
        headings = self._find_headings(blocks, body_size)

        if not headings:
            # No detectable structure. Rather than return nothing, emit the whole
            # document as one OTHER section so downstream stages still have addressable
            # text — and let the section validator report the missing structure.
            logger.warning("no_headings_detected", document_id=document.document_id)
            return (self._single_section(blocks),)

        return self._build_sections(blocks, headings)

    def _find_headings(self, blocks: list[TextBlock], body_size: float) -> list[HeadingCandidate]:
        candidates: list[HeadingCandidate] = []

        for index, block in enumerate(blocks):
            text = block.text.strip()
            if len(text) > self._config.max_heading_chars:
                continue
            if len(text.split()) > self._config.max_heading_words:
                continue

            signals = self._heading_signals(block, body_size)
            if signals is None:
                continue
            confidence = sum(signals.values())
            if confidence < self._config.min_confidence:
                continue

            title = _clean_title(text)
            if not title:
                continue

            kind = classify_section(title)
            # A canonical name is itself strong evidence, even in a plain-looking block.
            if kind is not SectionKind.OTHER:
                confidence = min(1.0, confidence + 0.25)

            candidates.append(
                HeadingCandidate(
                    block_index=index,
                    title=title,
                    kind=kind,
                    level=_heading_level(text),
                    confidence=round(confidence, 4),
                    page=block.page,
                )
            )

        return candidates

    def _heading_signals(self, block: TextBlock, body_size: float) -> dict[str, float] | None:
        """Weighted heading evidence, or None when the block is plainly body text.

        The gate matters more than the weights. Brevity and the absence of a full stop
        are *supporting* signals — on their own they describe half the short lines in any
        paper, and treating them as sufficient turns every table cell and author name
        into a section heading. A heading must first show typographic intent: a larger
        font, boldface, or section numbering.
        """
        text = block.text.strip()
        larger = body_size > 0 and block.style.size >= body_size * self._config.heading_size_ratio
        numbered = bool(_NUMBERING.match(text))
        bold = block.style.bold

        if not (larger or bold or numbered):
            return None

        # Headings are rarely sentences; a trailing full stop usually means prose.
        sentence_like = text.endswith((".", ",", ";")) and not numbered

        return {
            "size": 0.35 if larger else 0.0,
            "bold": 0.25 if bold else 0.0,
            "numbered": 0.25 if numbered else 0.0,
            "short": 0.10 if len(text.split()) <= 8 else 0.0,
            "not_prose": 0.0 if sentence_like else 0.05,
        }

    def _build_sections(
        self, blocks: list[TextBlock], headings: list[HeadingCandidate]
    ) -> tuple[Section, ...]:
        sections: list[Section] = []
        # Content before the first heading is the title block and front matter.
        boundaries = [
            (heading, headings[i + 1].block_index if i + 1 < len(headings) else len(blocks))
            for i, heading in enumerate(headings)
        ]

        parents: dict[int, str] = {}
        for order, (heading, end) in enumerate(boundaries):
            body = blocks[heading.block_index + 1 : end]
            paragraphs = self._to_paragraphs(body)
            section_id = f"s{order:03d}"

            parent_id = next(
                (
                    parents[level]
                    for level in sorted(parents, reverse=True)
                    if level < heading.level
                ),
                None,
            )
            parents[heading.level] = section_id
            for level in [lvl for lvl in parents if lvl > heading.level]:
                del parents[level]

            sections.append(
                Section(
                    id=section_id,
                    kind=heading.kind,
                    title=heading.title,
                    level=heading.level,
                    order=order,
                    parent_id=parent_id,
                    paragraphs=paragraphs,
                    page_start=heading.page,
                    page_end=body[-1].page if body else heading.page,
                    detection_confidence=heading.confidence,
                )
            )

        return tuple(sections)

    def _to_paragraphs(self, blocks: list[TextBlock]) -> tuple[Paragraph, ...]:
        paragraphs: list[Paragraph] = []
        for block in blocks:
            text = " ".join(block.text.split())
            if len(text) < self._config.min_paragraph_chars:
                continue
            paragraphs.append(
                Paragraph(
                    index=len(paragraphs),
                    text=text,
                    page=block.page,
                    bounding_box=block.bounding_box,
                )
            )
        return tuple(paragraphs)

    def _single_section(self, blocks: list[TextBlock]) -> Section:
        return Section(
            id="s000",
            kind=SectionKind.OTHER,
            title="Document",
            level=1,
            order=0,
            paragraphs=self._to_paragraphs(blocks),
            page_start=blocks[0].page,
            page_end=blocks[-1].page,
            detection_confidence=0.0,
        )


def classify_section(title: str) -> SectionKind:
    """Map a printed heading onto the canonical set of section roles."""
    normalised = _NUMBERING.sub("", title.strip()).strip().lower()
    normalised = _TRAILING_PUNCTUATION.sub("", normalised)
    if not normalised:
        return SectionKind.OTHER

    for kind, pattern in _SECTION_PATTERNS:
        if pattern.search(normalised):
            return kind
    return SectionKind.OTHER


def _clean_title(text: str) -> str:
    return _TRAILING_PUNCTUATION.sub("", " ".join(text.split()))


def _heading_level(text: str) -> int:
    """Depth from the section number: '3' is level 1, '3.2.1' is level 3."""
    match = _NUMBERING.match(text)
    if not match:
        return 1
    numbering = match.group(1)
    return min(6, numbering.count(".") + 1) if numbering else 1
