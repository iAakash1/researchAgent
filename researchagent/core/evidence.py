"""Evidence: the traceability primitive.

Every fact the system asserts must be able to answer "where did you get that?" — down to
the page, section and paragraph. Downstream agents consume :class:`Evidence` rather than
raw strings, so the v0.8 verifier can re-open the source and check a claim instead of
asking a model whether it believes itself.

Evidence is frozen. A claim's provenance must mean the same thing at the end of a run as
it did when it was recorded.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class EvidenceKind(StrEnum):
    """What sort of support this evidence provides."""

    # Verbatim text lifted from a document — the strongest kind.
    EXTRACTED_TEXT = "extracted_text"
    # A metadata field read off the artefact itself (PDF title, embedded DOI).
    DOCUMENT_METADATA = "document_metadata"
    # A metadata field asserted by an external index (Crossref, OpenAlex).
    PROVIDER_METADATA = "provider_metadata"
    # A structural observation (this document has 7 pages, 12 sections).
    STRUCTURAL = "structural"
    # Independent sources agreeing — corroboration, not proof.
    CROSS_SOURCE = "cross_source"
    # The documented absence of something looked for; absence of evidence, recorded.
    ABSENCE = "absence"


class BoundingBox(BaseModel):
    """Rectangle in PDF user-space points, origin at the top-left of the page."""

    model_config = {"frozen": True}

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"bounding box corners are inverted: {self}")
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_tuple(cls, values: tuple[float, float, float, float]) -> BoundingBox:
        x0, y0, x1, y1 = values
        return cls(x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))

    def merged_with(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


class SourceLocation(BaseModel):
    """Where in a document something was found.

    Every field below the document is optional because evidence legitimately comes at
    different resolutions — a page-level observation is weaker than a paragraph-level one,
    but recording it honestly beats fabricating precision.
    """

    model_config = {"frozen": True}

    document_id: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section_id: str | None = None
    section_title: str | None = None
    paragraph_index: int | None = Field(default=None, ge=0)
    bounding_box: BoundingBox | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _span_ordered(self) -> SourceLocation:
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end precedes char_start")
        return self

    @property
    def precision(self) -> int:
        """How specific this location is; higher is better. Used to rank evidence."""
        return sum(
            (
                self.page is not None,
                self.section_id is not None,
                self.paragraph_index is not None,
                self.bounding_box is not None,
                self.char_start is not None,
            )
        )

    def describe(self) -> str:
        parts = [self.document_id]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.section_title:
            parts.append(f"§{self.section_title}")
        if self.paragraph_index is not None:
            parts.append(f"¶{self.paragraph_index}")
        return " ".join(parts)


class Evidence(BaseModel):
    """A claim, where it came from, and how strongly it is supported."""

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: EvidenceKind
    claim: str = Field(min_length=1, description="What this evidence supports")
    quote: str | None = Field(
        default=None, description="Verbatim source text; required for EXTRACTED_TEXT"
    )
    location: SourceLocation
    # Confidence lives on the producing ValidationResult rather than here, so evidence
    # stays a statement of fact and scoring stays in one place.
    produced_by: str = Field(min_length=1, description="Component that recorded this")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _quote_required_for_text(self) -> Evidence:
        if self.kind is EvidenceKind.EXTRACTED_TEXT and not self.quote:
            raise ValueError("EXTRACTED_TEXT evidence must carry the verbatim quote")
        return self

    @classmethod
    def from_text(
        cls,
        *,
        claim: str,
        quote: str,
        location: SourceLocation,
        produced_by: str,
    ) -> Evidence:
        return cls(
            kind=EvidenceKind.EXTRACTED_TEXT,
            claim=claim,
            quote=quote,
            location=location,
            produced_by=produced_by,
        )

    @classmethod
    def structural(
        cls, *, claim: str, document_id: str, produced_by: str, page: int | None = None
    ) -> Evidence:
        return cls(
            kind=EvidenceKind.STRUCTURAL,
            claim=claim,
            location=SourceLocation(document_id=document_id, page=page),
            produced_by=produced_by,
        )

    @classmethod
    def absence(cls, *, claim: str, document_id: str, produced_by: str) -> Evidence:
        """Record that something was looked for and not found.

        Kept explicitly so a later stage can distinguish "no abstract in this paper" from
        "nobody checked for an abstract".
        """
        return cls(
            kind=EvidenceKind.ABSENCE,
            claim=claim,
            location=SourceLocation(document_id=document_id),
            produced_by=produced_by,
        )

    def summary(self) -> str:
        return f"{self.claim} [{self.kind.value} @ {self.location.describe()}]"
