"""Reference and citation extraction.

Two related jobs, kept in one module because they share the marker vocabulary:

* **References** — split the bibliography into entries and recover what structure is
  recoverable. The verbatim ``raw`` string is always kept, because bibliography parsing
  is lossy and v0.7 will want to re-resolve entries with a better parser (or Crossref)
  without re-reading the PDF.
* **Citations** — find in-text markers and link them to those entries. Unmatched markers
  are recorded as unresolved rather than dropped: the resolution rate is a direct,
  observable measure of parse quality and feeds the citation validator's confidence.

No network calls. Resolving references against real indexes is a later release.
"""

from __future__ import annotations

import re

from researchagent.core.logging import get_logger
from researchagent.models.document import Citation, Reference, Section, SectionKind

logger = get_logger(__name__)

# Entry markers. Bibliographies are frequently extracted as one block with the markers
# sitting mid-line rather than at line starts, so `[n]` is matched anywhere.
_BRACKETED_ANYWHERE = re.compile(r"\[(\d{1,3})\]\s*")
_NUMBERED_LINE = re.compile(r"^\s*(\d{1,3})[.)]\s+(?=[A-Z])")

# In-text markers: [12], [3, 4], [1]-[3]. En/em dashes are deliberate — real papers
# print ranges with them.
_CITATION_MARKER = re.compile(r"\[(\d{1,3}(?:\s*[,–—-]\s*\d{1,3})*)\]")  # noqa: RUF001
_MARKER_NUMBERS = re.compile(r"\d{1,3}")

_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_ARXIV = re.compile(r"arXiv[:\s]\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
# "Surname, F." / "F. Surname" runs at the head of an entry.
_AUTHOR_SEGMENT = re.compile(r"^(.{3,200}?)(?:\.\s+|\s{2,})(?=[A-Z“\"])")

_MIN_ENTRY_CHARS = 20
_MAX_ENTRY_CHARS = 2000


class ReferenceExtractor:
    """Splits the references section into structured entries."""

    name = "reference_extractor"

    def extract(self, sections: tuple[Section, ...]) -> tuple[Reference, ...]:
        reference_sections = [s for s in sections if s.kind is SectionKind.REFERENCES]
        if not reference_sections:
            return ()

        entries: list[Reference] = []
        for section in reference_sections:
            for paragraph in section.paragraphs:
                entries.extend(self._entries_from(paragraph.text, paragraph.page, len(entries)))

        logger.debug("references_extracted", count=len(entries))
        return tuple(entries)

    def _entries_from(self, text: str, page: int, offset: int) -> list[Reference]:
        """One block may hold one entry or many, depending on the PDF's layout."""
        references: list[Reference] = []
        for raw, marker in _split_entries(text):
            cleaned = " ".join(raw.split())
            if not _MIN_ENTRY_CHARS <= len(cleaned) <= _MAX_ENTRY_CHARS:
                continue
            index = offset + len(references)
            references.append(
                _parse_entry(cleaned, marker=marker, page=page, reference_id=f"r{index:03d}")
            )
        return references


class CitationExtractor:
    """Finds in-text citation markers and links them to references."""

    name = "citation_extractor"

    def extract(
        self, sections: tuple[Section, ...], references: tuple[Reference, ...]
    ) -> tuple[Citation, ...]:
        by_marker = {ref.marker: ref.id for ref in references if ref.marker}
        citations: list[Citation] = []

        for section in sections:
            if section.kind is SectionKind.REFERENCES:
                continue  # entries cite themselves; not in-text citations
            for paragraph in section.paragraphs:
                for match in _CITATION_MARKER.finditer(paragraph.text):
                    for number in _MARKER_NUMBERS.findall(match.group(1)):
                        citations.append(
                            Citation(
                                id=f"c{len(citations):04d}",
                                marker=f"[{number}]",
                                reference_id=by_marker.get(number),
                                page=paragraph.page,
                                section_id=section.id,
                                paragraph_index=paragraph.index,
                            )
                        )

        resolved = sum(1 for citation in citations if citation.is_resolved)
        logger.debug("citations_extracted", count=len(citations), resolved=resolved)
        return tuple(citations)


def _split_entries(text: str) -> list[tuple[str, str | None]]:
    """Split a block into (entry text, marker) pairs.

    Bibliographies arrive in two shapes and both are common: one block per entry, or one
    block holding the whole list. In the second shape the ``[n]`` markers land mid-line,
    so splitting on line starts alone silently merges entries — which then makes every
    in-text citation unresolvable, because the markers were never recovered.

    Bracketed markers are therefore matched anywhere; numbered styles ("1. Author, ...")
    fall back to line starts, where they are unambiguous.
    """
    markers = list(_BRACKETED_ANYWHERE.finditer(text))
    if markers:
        entries = []
        for index, match in enumerate(markers):
            start = match.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            entries.append((text[start:end], match.group(1)))
        return entries

    lines = text.split("\n")
    numbered_entries: list[tuple[list[str], str | None]] = []
    for line in lines:
        numbered = _NUMBERED_LINE.match(line)
        if numbered:
            numbered_entries.append(([_NUMBERED_LINE.sub("", line)], numbered.group(1)))
        elif numbered_entries:
            numbered_entries[-1][0].append(line)
        else:
            numbered_entries.append(([line], None))

    return [(" ".join(parts), marker) for parts, marker in numbered_entries]


def _parse_entry(raw: str, *, marker: str | None, page: int, reference_id: str) -> Reference:
    """Recover what structure is recoverable. Absent fields stay None — never guessed."""
    doi_match = _DOI.search(raw)
    arxiv_match = _ARXIV.search(raw)
    year_match = _YEAR.search(raw)
    url_match = _URL.search(raw)

    return Reference(
        id=reference_id,
        raw=raw,
        marker=marker,
        title=_extract_title(raw),
        authors=_extract_authors(raw),
        year=int(year_match.group(1)) if year_match else None,
        venue=None,  # venue parsing is unreliable without a trained model; left to v0.7
        doi=doi_match.group(0).rstrip(".,;") if doi_match else None,
        arxiv_id=arxiv_match.group(1) if arxiv_match else None,
        url=url_match.group(0).rstrip(".,;") if url_match else None,
        page=page,
    )


def _extract_title(raw: str) -> str | None:
    """Title is usually the segment after the authors and before the venue.

    Quoted titles are unambiguous; otherwise take the second sentence-like segment. When
    neither pattern fits, return None rather than a guess — a wrong title is worse than
    a missing one for the knowledge graph.
    """
    quoted = re.search(r"[“\"]([^”\"]{10,300})[”\"]", raw)
    if quoted:
        return quoted.group(1).strip()

    without_authors = _AUTHOR_SEGMENT.sub("", raw, count=1)
    if without_authors == raw:
        return None

    candidate = re.split(r"\.\s+(?=[A-Z])|\.\s*In\s+|\.\s*arXiv", without_authors, maxsplit=1)[0]
    candidate = candidate.strip(" .,")
    return candidate if 10 <= len(candidate) <= 300 else None


def _extract_authors(raw: str) -> tuple[str, ...]:
    match = _AUTHOR_SEGMENT.match(raw)
    if not match:
        return ()

    segment = match.group(1)
    if len(segment) > 200:
        return ()

    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", segment)
    authors = [
        " ".join(part.split())
        for part in parts
        if 3 <= len(part.strip()) <= 60 and any(ch.isalpha() for ch in part)
    ]
    return tuple(authors[:20])
