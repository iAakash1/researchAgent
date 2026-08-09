"""The canonical research paper representation.

From here on the system stops thinking in PDFs. A :class:`PaperDocument` is what every
later stage consumes — knowledge extraction reads its sections, chunking respects its
section boundaries, the knowledge graph links its references, and verification quotes its
paragraphs back with page numbers.

Two properties make that possible and are load-bearing:

* **Everything is addressable.** Every paragraph knows its page, section and index, so
  any downstream claim can be traced to a :class:`SourceLocation` and re-checked.
* **Everything is immutable.** A stage that could edit the document could quietly launder
  an unsupported claim into the record. New knowledge becomes new objects.

``PaperMetadata`` here is what the *document itself* says, extracted from its own pages —
deliberately separate from the ``Paper`` metadata an index asserted, so the two can be
compared instead of one silently overwriting the other.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from researchagent.core.evidence import BoundingBox, SourceLocation


class SectionKind(StrEnum):
    """Canonical section roles.

    Papers name these dozens of ways ("Evaluation", "Experimental Results", "5 Results");
    the detector maps variants onto this closed set so downstream code can ask for the
    methodology without knowing what this particular author called it.
    """

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    EVALUATION = "evaluation"
    DISCUSSION = "discussion"
    LIMITATIONS = "limitations"
    FUTURE_WORK = "future_work"
    CONCLUSION = "conclusion"
    ACKNOWLEDGEMENTS = "acknowledgements"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"

    @property
    def is_body(self) -> bool:
        """Sections that carry the paper's argument, as opposed to apparatus."""
        return self not in (
            SectionKind.TITLE,
            SectionKind.REFERENCES,
            SectionKind.ACKNOWLEDGEMENTS,
            SectionKind.APPENDIX,
        )


class Paragraph(BaseModel):
    """The smallest addressable unit of prose."""

    model_config = {"frozen": True}

    index: int = Field(ge=0, description="Position within the owning section")
    text: str
    page: int = Field(ge=1)
    bounding_box: BoundingBox | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def location(self, document_id: str, section: Section) -> SourceLocation:
        """Where this paragraph lives — the anchor for any claim drawn from it."""
        return SourceLocation(
            document_id=document_id,
            page=self.page,
            section_id=section.id,
            section_title=section.title,
            paragraph_index=self.index,
            bounding_box=self.bounding_box,
        )


class Section(BaseModel):
    """A titled region of the paper.

    Hierarchy is expressed by ``level`` plus ``parent_id`` rather than nesting, so the
    document stays flat to serialise, cheap to query, and stable when a detector revises
    where a subsection belongs.
    """

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    kind: SectionKind
    title: str
    level: int = Field(default=1, ge=1, le=6)
    order: int = Field(ge=0, description="Reading order across the document")
    parent_id: str | None = None
    paragraphs: tuple[Paragraph, ...] = ()
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    # Confidence that this really is a section boundary of this kind, from the detector.
    detection_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)

    @property
    def word_count(self) -> int:
        return sum(paragraph.word_count for paragraph in self.paragraphs)

    @property
    def is_empty(self) -> bool:
        return not self.paragraphs


class Reference(BaseModel):
    """One entry in the bibliography, as printed.

    ``raw`` is always kept: parsing bibliographies is lossy, and the v0.7 knowledge graph
    will want to re-resolve entries with a better parser without re-reading the PDF.
    """

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    raw: str = Field(min_length=1, description="Verbatim printed entry")
    marker: str | None = Field(default=None, description="e.g. '12' from '[12]'")
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = Field(default=None, ge=1500, le=2200)
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    page: int | None = Field(default=None, ge=1)

    @property
    def is_structured(self) -> bool:
        """Whether anything beyond the raw string was recovered."""
        return bool(self.title or self.authors or self.year or self.doi)


class Citation(BaseModel):
    """An in-text pointer to a reference.

    ``reference_id`` is None when the marker could not be matched. That is recorded
    rather than dropped: the resolution rate is a direct, observable measure of how well
    the document was parsed, and feeds the citation validator's confidence.
    """

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    marker: str = Field(min_length=1, description="As printed, e.g. '[12]'")
    reference_id: str | None = None
    page: int = Field(ge=1)
    section_id: str | None = None
    paragraph_index: int | None = Field(default=None, ge=0)

    @property
    def is_resolved(self) -> bool:
        return self.reference_id is not None


class Figure(BaseModel):
    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    label: str | None = Field(default=None, description="e.g. 'Figure 3'")
    caption: str = ""
    page: int = Field(ge=1)
    bounding_box: BoundingBox | None = None


class Table(BaseModel):
    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    label: str | None = None
    caption: str = ""
    page: int = Field(ge=1)
    bounding_box: BoundingBox | None = None


class Equation(BaseModel):
    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    raw: str
    page: int = Field(ge=1)
    section_id: str | None = None


class Footnote(BaseModel):
    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    marker: str | None = None
    text: str
    page: int = Field(ge=1)


class PaperMetadata(BaseModel):
    """What the document says about itself, read from its own pages and embedded fields.

    Never merged with index metadata. Keeping them separate is what allows the metadata
    validator to notice that Crossref and the PDF disagree about the title.
    """

    model_config = {"frozen": True}

    title: str | None = None
    authors: tuple[str, ...] = ()
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = Field(default=None, ge=1500, le=2200)
    venue: str | None = None
    keywords: tuple[str, ...] = ()


class DocumentStatistics(BaseModel):
    """Counts computed once at assembly. Cheap to store, and the basis of most
    confidence signals — validators must measure, not guess."""

    model_config = {"frozen": True}

    pages: int = Field(ge=0)
    characters: int = Field(ge=0)
    words: int = Field(ge=0)
    sections: int = Field(ge=0)
    paragraphs: int = Field(ge=0)
    figures: int = Field(ge=0)
    tables: int = Field(ge=0)
    references: int = Field(ge=0)
    citations: int = Field(ge=0)
    resolved_citations: int = Field(ge=0)
    empty_pages: int = Field(ge=0)

    @property
    def citation_resolution_rate(self) -> float:
        return self.resolved_citations / self.citations if self.citations else 0.0

    @property
    def characters_per_page(self) -> float:
        return self.characters / self.pages if self.pages else 0.0


class DocumentProvenance(BaseModel):
    """Exactly which bytes were parsed, by what, and when.

    ``source_sha256`` makes re-parsing idempotent and makes a stale document detectable
    after a re-download — reproducibility depends on this, not on timestamps.
    """

    model_config = {"frozen": True}

    source_path: str
    source_sha256: str = Field(min_length=1)
    source_bytes: int = Field(ge=0)
    loader: str
    loader_version: str = ""
    parser_version: str = Field(default="1", description="Version of our own assembly logic")
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PaperDocument(BaseModel):
    """The canonical, immutable representation of one research paper."""

    model_config = {"frozen": True}

    paper_id: str = Field(min_length=1)
    provenance: DocumentProvenance
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)

    sections: tuple[Section, ...] = ()
    figures: tuple[Figure, ...] = ()
    tables: tuple[Table, ...] = ()
    references: tuple[Reference, ...] = ()
    citations: tuple[Citation, ...] = ()
    equations: tuple[Equation, ...] = ()
    footnotes: tuple[Footnote, ...] = ()

    statistics: DocumentStatistics
    # Section ids in reading order; the flat sequence a chunker should follow.
    reading_order: tuple[str, ...] = ()

    def section(self, section_id: str) -> Section | None:
        return next((s for s in self.sections if s.id == section_id), None)

    def sections_of(self, kind: SectionKind) -> tuple[Section, ...]:
        return tuple(section for section in self.sections if section.kind is kind)

    def first_section_of(self, kind: SectionKind) -> Section | None:
        return next((section for section in self.sections if section.kind is kind), None)

    @property
    def abstract(self) -> str | None:
        section = self.first_section_of(SectionKind.ABSTRACT)
        return section.text if section else self.metadata.abstract

    @property
    def body_sections(self) -> tuple[Section, ...]:
        return tuple(section for section in self.sections if section.kind.is_body)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"{s.title}\n{s.text}" for s in self.sections if not s.is_empty)

    def reference(self, reference_id: str) -> Reference | None:
        return next((r for r in self.references if r.id == reference_id), None)

    def paragraph_locations(self) -> list[tuple[Paragraph, SourceLocation]]:
        """Every paragraph with its address — the iteration surface for extraction,
        chunking and verification, so none of them re-derive addressing."""
        return [
            (paragraph, paragraph.location(self.paper_id, section))
            for section in self.sections
            for paragraph in section.paragraphs
        ]
