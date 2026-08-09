"""Raw layout: what a PDF loader produces, before any interpretation.

This is deliberately dumb. It says where text is and what it looks like, never what it
means — no section titles, no captions, no references. Interpretation happens in
``services/document/`` where it can be tested without a PDF.

Keeping the two apart is what makes the loader swappable and the detection logic
unit-testable on synthetic pages.
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from researchagent.core.evidence import BoundingBox


class TextStyle(BaseModel):
    """Typographic properties of a run of text — the primary signal for headings."""

    model_config = {"frozen": True}

    size: float = Field(gt=0)
    font: str = ""
    bold: bool = False
    italic: bool = False

    @property
    def is_emphasised(self) -> bool:
        return self.bold or self.italic


class TextBlock(BaseModel):
    """A positioned run of text on one page."""

    model_config = {"frozen": True}

    text: str
    page: int = Field(ge=1)
    index: int = Field(ge=0, description="Reading order within the page")
    bounding_box: BoundingBox
    style: TextStyle

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + 1


class RawPage(BaseModel):
    model_config = {"frozen": True}

    number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: tuple[TextBlock, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if not block.is_blank)

    @property
    def character_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)

    @property
    def is_empty(self) -> bool:
        """A page with no extractable text — the signature of a scanned image."""
        return self.character_count == 0


class PdfMetadata(BaseModel):
    """Metadata embedded in the PDF itself.

    Treated as a claim by the document, not as truth: producers routinely leave the
    template author's name or a working title in these fields, which is exactly why the
    metadata validator cross-checks them against the discovered record.
    """

    model_config = {"frozen": True}

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None


class DocumentFormat(StrEnum):
    PDF = "pdf"


class RawDocument(BaseModel):
    """A loaded document: pages of positioned blocks, plus provenance."""

    model_config = {"frozen": True}

    document_id: str = Field(min_length=1)
    format: DocumentFormat = DocumentFormat.PDF
    pages: tuple[RawPage, ...] = ()
    pdf_metadata: PdfMetadata = Field(default_factory=PdfMetadata)

    # Provenance: identifies the exact bytes parsed, so a re-parse can be detected as
    # unnecessary and a stale document can be spotted after a re-download.
    source_sha256: str = Field(min_length=1)
    source_bytes: int = Field(ge=0)
    loader: str = Field(min_length=1)
    loader_version: str = ""
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def blocks(self) -> list[TextBlock]:
        return [block for page in self.pages for block in page.blocks]

    @property
    def character_count(self) -> int:
        return sum(page.character_count for page in self.pages)

    @property
    def empty_page_count(self) -> int:
        return sum(1 for page in self.pages if page.is_empty)

    def body_text_size(self) -> float:
        """Modal font size weighted by character count — the body-text baseline.

        Headings are identified relative to this rather than against an absolute
        threshold, because a two-column ACM paper and an A4 preprint disagree on what
        "large" means.
        """
        weights: dict[float, int] = {}
        for block in self.blocks:
            if block.is_blank:
                continue
            key = round(block.style.size, 1)
            weights[key] = weights.get(key, 0) + len(block.text)
        if not weights:
            return 0.0
        return max(weights.items(), key=lambda item: item[1])[0]

    def median_block_length(self) -> float:
        lengths = [len(block.text) for block in self.blocks if not block.is_blank]
        return statistics.median(lengths) if lengths else 0.0
