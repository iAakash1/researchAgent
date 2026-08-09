"""Document assembly.

Composes the detectors into one canonical :class:`PaperDocument`. Owns no detection
logic of its own — it decides the order things run in and computes the statistics that
every validator then measures against.
"""

from __future__ import annotations

from pathlib import Path

from researchagent.core.logging import get_logger
from researchagent.models.document import (
    DocumentProvenance,
    DocumentStatistics,
    PaperDocument,
    Section,
)
from researchagent.models.layout import RawDocument
from researchagent.services.document.figures import FigureTableDetector
from researchagent.services.document.metadata import MetadataExtractor
from researchagent.services.document.references import CitationExtractor, ReferenceExtractor
from researchagent.services.document.sections import SectionDetector

logger = get_logger(__name__)

ASSEMBLER_VERSION = "1"


class DocumentAssembler:
    """Raw layout in, canonical document out."""

    name = "document_assembler"

    def __init__(
        self,
        sections: SectionDetector,
        references: ReferenceExtractor,
        citations: CitationExtractor,
        figures: FigureTableDetector,
        metadata: MetadataExtractor,
    ) -> None:
        self._sections = sections
        self._references = references
        self._citations = citations
        self._figures = figures
        self._metadata = metadata

    def assemble(self, raw: RawDocument, *, source_path: Path) -> PaperDocument:
        sections = self._sections.detect(raw)
        # Citations depend on references, which depend on sections: the order here is a
        # data dependency, not a preference.
        references = self._references.extract(sections)
        citations = self._citations.extract(sections, references)
        figures, tables = self._figures.detect(raw)
        metadata = self._metadata.extract(raw)

        document = PaperDocument(
            paper_id=raw.document_id,
            provenance=DocumentProvenance(
                source_path=str(source_path),
                source_sha256=raw.source_sha256,
                source_bytes=raw.source_bytes,
                loader=raw.loader,
                loader_version=raw.loader_version,
                parser_version=ASSEMBLER_VERSION,
            ),
            metadata=metadata,
            sections=sections,
            figures=figures,
            tables=tables,
            references=references,
            citations=citations,
            statistics=_statistics(raw, sections, figures, tables, references, citations),
            reading_order=tuple(section.id for section in sections),
        )

        logger.info(
            "document_assembled",
            paper_id=document.paper_id,
            pages=document.statistics.pages,
            sections=document.statistics.sections,
            references=document.statistics.references,
            citations=document.statistics.citations,
            figures=document.statistics.figures,
            tables=document.statistics.tables,
        )
        return document


def _statistics(
    raw: RawDocument,
    sections: tuple[Section, ...],
    figures: tuple[object, ...],
    tables: tuple[object, ...],
    references: tuple[object, ...],
    citations: tuple[object, ...],
) -> DocumentStatistics:
    paragraphs = [paragraph for section in sections for paragraph in section.paragraphs]
    return DocumentStatistics(
        pages=raw.page_count,
        characters=raw.character_count,
        words=sum(paragraph.word_count for paragraph in paragraphs),
        sections=len(sections),
        paragraphs=len(paragraphs),
        figures=len(figures),
        tables=len(tables),
        references=len(references),
        citations=len(citations),
        resolved_citations=sum(1 for c in citations if getattr(c, "is_resolved", False)),
        empty_pages=raw.empty_page_count,
    )
