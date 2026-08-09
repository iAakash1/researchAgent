"""Document self-metadata extraction.

Reads what the paper says about *itself* — from its first page and its embedded PDF
fields — with no reference to what Crossref or arXiv claimed.

That independence is the point. Discovery metadata and document metadata are two
witnesses; if the extractor were allowed to peek at the discovered record it would
inherit its errors, and the metadata validator would have nothing left to compare.
"""

from __future__ import annotations

import re

from researchagent.core.logging import get_logger
from researchagent.models.document import PaperMetadata
from researchagent.models.layout import RawDocument, TextBlock

logger = get_logger(__name__)

_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_ARXIV = re.compile(r"arXiv[:\s]\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_ABSTRACT_HEADING = re.compile(r"^\s*abstract\b[.:—–-]?\s*", re.IGNORECASE)  # noqa: RUF001
_EMAIL = re.compile(r"\S+@\S+")

# Author lines are short, comma-separated, and rich in capitalised words.
_AUTHOR_SEPARATOR = re.compile(r",\s*(?:and\s+)?|\s+and\s+|\s*[;·•]\s*")
_AUTHOR_NOISE = re.compile(r"[*†‡§¶#0-9]")

_MAX_TITLE_CHARS = 300
_MAX_AUTHOR_LINE_CHARS = 400
_FIRST_PAGE_BLOCK_LIMIT = 40


class MetadataExtractor:
    """Recovers title, authors, abstract and identifiers from the document itself."""

    name = "metadata_extractor"

    def extract(self, document: RawDocument) -> PaperMetadata:
        if not document.pages:
            return PaperMetadata()

        first_page = document.pages[0]
        blocks = [b for b in first_page.blocks if not b.is_blank][:_FIRST_PAGE_BLOCK_LIMIT]
        front_text = "\n".join(block.text for block in blocks)

        title = self._title(blocks, document)
        metadata = PaperMetadata(
            title=title,
            authors=self._authors(blocks, title),
            abstract=self._abstract(blocks),
            doi=_first(_DOI.search(front_text)),
            arxiv_id=_group(_ARXIV.search(front_text), 1),
            year=self._year(document, front_text),
            keywords=_keywords(document.pdf_metadata.keywords),
        )
        logger.debug(
            "document_metadata_extracted",
            document_id=document.document_id,
            has_title=metadata.title is not None,
            authors=len(metadata.authors),
            has_abstract=metadata.abstract is not None,
        )
        return metadata

    def _title(self, blocks: list[TextBlock], document: RawDocument) -> str | None:
        """The largest text near the top of page one.

        Preferred over the embedded PDF title field, which is frequently the LaTeX
        template's default or a stale working title.
        """
        if not blocks:
            return _clean(document.pdf_metadata.title)

        candidates = [
            block
            for block in blocks[:12]
            if len(block.text) <= _MAX_TITLE_CHARS and len(block.text.strip()) >= 8
        ]
        if not candidates:
            return _clean(document.pdf_metadata.title)

        largest = max(candidates, key=lambda block: (block.style.size, -block.index))
        body_size = document.body_text_size()
        if body_size and largest.style.size <= body_size * 1.05:
            # Nothing stands out typographically; fall back to the embedded field.
            return _clean(document.pdf_metadata.title) or _clean(largest.text)
        return _clean(largest.text)

    def _authors(self, blocks: list[TextBlock], title: str | None) -> tuple[str, ...]:
        """Blocks between the title and the abstract, filtered to name-like fragments."""
        start = 0
        if title:
            start = next(
                (i + 1 for i, block in enumerate(blocks) if _clean(block.text) == title), 0
            )

        collected: list[str] = []
        for block in blocks[start : start + 8]:
            text = " ".join(block.text.split())
            if _ABSTRACT_HEADING.match(text):
                break
            if len(text) > _MAX_AUTHOR_LINE_CHARS or _EMAIL.search(text):
                continue
            collected.extend(_author_names(text))
            if len(collected) >= 20:
                break

        seen: dict[str, str] = {}
        for name in collected:
            seen.setdefault(name.lower(), name)
        return tuple(seen.values())

    def _abstract(self, blocks: list[TextBlock]) -> str | None:
        for index, block in enumerate(blocks):
            text = " ".join(block.text.split())
            if not _ABSTRACT_HEADING.match(text):
                continue

            body = _ABSTRACT_HEADING.sub("", text).strip()
            if len(body) >= 60:
                return body
            # The heading sits in its own block; the abstract is the next one.
            if index + 1 < len(blocks):
                following = " ".join(blocks[index + 1].text.split())
                return following or None
        return None

    def _year(self, document: RawDocument, front_text: str) -> int | None:
        for candidate in (document.pdf_metadata.creation_date or "", front_text):
            match = _YEAR.search(candidate)
            if match:
                return int(match.group(1))
        return None


def _author_names(text: str) -> list[str]:
    names = []
    for part in _AUTHOR_SEPARATOR.split(text):
        cleaned = " ".join(_AUTHOR_NOISE.sub("", part).split())
        words = cleaned.split()
        if not 2 <= len(words) <= 5:
            continue
        if not all(word[0].isupper() for word in words if word):
            continue
        if 4 <= len(cleaned) <= 60:
            names.append(cleaned)
    return names


def _keywords(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    parts = [" ".join(part.split()) for part in re.split(r"[;,]", raw)]
    return tuple(part for part in parts if part)[:15]


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    collapsed = " ".join(value.split())
    return collapsed or None


def _first(match: re.Match[str] | None) -> str | None:
    return match.group(0).rstrip(".,;") if match else None


def _group(match: re.Match[str] | None, index: int) -> str | None:
    return match.group(index) if match else None
