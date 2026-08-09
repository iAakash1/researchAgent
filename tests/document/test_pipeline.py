"""Document intelligence against the real committed PDFs.

The loader and the end-to-end pipeline are tested on actual papers; the detection rules
they feed are tested on synthetic layout in ``test_detection.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchagent.core.events import Event, EventBus, EventType
from researchagent.core.exceptions import (
    DocumentUnreadableError,
    Recoverability,
)
from researchagent.integrations.manual import ManualPaperSource
from researchagent.integrations.pymupdf import PyMuPDFLoader
from researchagent.models.document import SectionKind
from researchagent.models.paper import Paper, PaperIdentifiers, SourceName
from researchagent.repositories.document_repository import JsonDocumentRepository
from researchagent.repositories.paper_repository import JsonPaperRepository
from researchagent.services.document import DocumentIntelligenceService
from researchagent.services.validation.document import PDFValidator

PAPER_01 = "01_[P1]_Metastable_Failures_in_Distributed_Systems.pdf"


@pytest.fixture
def real_paper(manual_source: ManualPaperSource) -> Paper:
    return next(p for p in manual_source.load_all() if p.id == "manual:01")


class TestLoader:
    def test_loads_a_real_pdf_into_positioned_blocks(self, real_paper: Paper) -> None:
        raw = PyMuPDFLoader().load(Path(str(real_paper.local_path)), document_id=real_paper.id)

        assert raw.page_count == 7
        assert raw.character_count > 10_000
        assert raw.empty_page_count == 0
        assert all(block.bounding_box is not None for block in raw.blocks)
        assert raw.pdf_metadata.title == "Metastable Failures in Distributed Systems"

    def test_provenance_identifies_the_exact_bytes(self, real_paper: Paper) -> None:
        """The sha is what makes re-parsing idempotent and staleness detectable."""
        path = Path(str(real_paper.local_path))

        first = PyMuPDFLoader().load(path, document_id="a")
        second = PyMuPDFLoader().load(path, document_id="b")

        assert first.source_sha256 == second.source_sha256
        assert len(first.source_sha256) == 64
        assert first.source_bytes == path.stat().st_size

    def test_body_font_baseline_is_measured_from_the_document(self, real_paper: Paper) -> None:
        raw = PyMuPDFLoader().load(Path(str(real_paper.local_path)), document_id=real_paper.id)

        body = raw.body_text_size()

        assert 8 < body < 12  # typical two-column body text
        assert max(block.style.size for block in raw.blocks) > body

    def test_missing_file_is_a_typed_recoverable_error(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentUnreadableError) as excinfo:
            PyMuPDFLoader().load(tmp_path / "nope.pdf", document_id="x")

        assert excinfo.value.recoverability is Recoverability.RECOVERABLE
        assert excinfo.value.remedy is not None

    def test_a_non_pdf_does_not_escape_as_a_native_error(self, tmp_path: Path) -> None:
        junk = tmp_path / "junk.pdf"
        junk.write_bytes(b"this is definitely not a pdf")

        with pytest.raises(DocumentUnreadableError):
            PyMuPDFLoader().load(junk, document_id="x")

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")

        with pytest.raises(DocumentUnreadableError):
            PyMuPDFLoader().load(empty, document_id="x")

    def test_supports_only_pdfs(self, tmp_path: Path) -> None:
        assert PyMuPDFLoader().supports(tmp_path / "a.pdf") is True
        assert PyMuPDFLoader().supports(tmp_path / "a.txt") is False


class TestPipelineOnRealPapers:
    async def test_produces_a_canonical_document(
        self, document_service: DocumentIntelligenceService, real_paper: Paper
    ) -> None:
        result = await document_service.process([real_paper])

        outcome = result.outcomes[0]
        assert outcome.succeeded is True
        document = outcome.document
        assert document is not None

        kinds = {section.kind for section in document.sections}
        assert SectionKind.ABSTRACT in kinds
        assert SectionKind.INTRODUCTION in kinds
        assert SectionKind.REFERENCES in kinds
        assert document.statistics.pages == 7
        assert document.statistics.references > 10
        assert document.statistics.citations > 0

    async def test_metadata_is_read_from_the_document_itself(
        self, document_service: DocumentIntelligenceService, real_paper: Paper
    ) -> None:
        """The PDF is a second witness; it must not simply echo the index."""
        result = await document_service.process([real_paper])
        document = result.outcomes[0].document
        assert document is not None

        assert document.metadata.title == "Metastable Failures in Distributed Systems"
        assert document.metadata.doi == "10.1145/3458336.3465286"
        assert document.metadata.abstract is not None
        assert "metastable" in document.metadata.abstract.lower()

    async def test_citations_resolve_against_extracted_references(
        self, document_service: DocumentIntelligenceService, real_paper: Paper
    ) -> None:
        result = await document_service.process([real_paper])
        document = result.outcomes[0].document
        assert document is not None

        assert document.statistics.citation_resolution_rate > 0.8

    async def test_every_paragraph_is_addressable(
        self, document_service: DocumentIntelligenceService, real_paper: Paper
    ) -> None:
        """Traceability is the point: any later claim must map to a page and section."""
        result = await document_service.process([real_paper])
        document = result.outcomes[0].document
        assert document is not None

        locations = document.paragraph_locations()
        assert len(locations) > 20
        for _, location in locations:
            assert location.document_id == real_paper.id
            assert location.page is not None
            assert location.section_id is not None
            assert location.paragraph_index is not None

    async def test_confidence_is_grounded_in_observations(
        self, document_service: DocumentIntelligenceService, real_paper: Paper
    ) -> None:
        result = await document_service.process([real_paper])
        verdict = result.outcomes[0].validation
        assert verdict is not None

        assert verdict.confidence.is_grounded is True
        names = {signal.name for signal in verdict.confidence.signals}
        assert {"text_density", "canonical_sections", "citation_resolution"} <= names
        assert all(signal.observation for signal in verdict.confidence.signals)

    async def test_document_is_persisted_with_its_verdict(
        self,
        document_service: DocumentIntelligenceService,
        document_repository: JsonDocumentRepository,
        real_paper: Paper,
    ) -> None:
        await document_service.process([real_paper])

        stored = await document_repository.get(real_paper.id)

        assert stored is not None
        assert stored.is_trusted is True
        assert stored.value.paper_id == real_paper.id
        assert stored.validation.validator == "document_validator"

    async def test_processing_flags_advance(
        self,
        document_service: DocumentIntelligenceService,
        paper_repository: JsonPaperRepository,
        real_paper: Paper,
    ) -> None:
        from researchagent.models.library import PaperRecord

        await paper_repository.save(PaperRecord(paper=real_paper))

        await document_service.process([real_paper])

        record = await paper_repository.get(real_paper.id)
        assert record is not None
        assert record.processing.parsed is True
        assert record.processing.sectioned is True
        assert record.processing.references_extracted is True
        assert record.processing.ready_for_extraction is True
        assert record.processing.stage_reached != "pending"

    async def test_unchanged_documents_are_not_reparsed(
        self, document_service: DocumentIntelligenceService, real_paper: Paper
    ) -> None:
        first = await document_service.process([real_paper])
        second = await document_service.process([real_paper])

        assert first.outcomes[0].duration_ms > second.outcomes[0].duration_ms
        assert second.outcomes[0].document is not None


class TestErrorIsolation:
    async def test_one_bad_pdf_does_not_sink_the_batch(
        self,
        document_service: DocumentIntelligenceService,
        real_paper: Paper,
        tmp_path: Path,
    ) -> None:
        """The invariant that keeps a forty-paper run usable when three are corrupt."""
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"not a pdf at all")
        broken = Paper(
            id="manual:broken",
            title="Corrupt",
            provider=SourceName.MANUAL,
            local_path=corrupt,
            identifiers=PaperIdentifiers(),
        )

        result = await document_service.process([real_paper, broken])

        assert result.succeeded == 1
        assert result.failed == 1
        failure = next(o for o in result.outcomes if not o.succeeded)
        assert failure.error_code == "document_unreadable"
        assert failure.recoverable is True
        assert failure.remedy is not None

    async def test_papers_without_a_local_pdf_are_skipped_not_failed(
        self, document_service: DocumentIntelligenceService
    ) -> None:
        metadata_only = Paper(id="arxiv:1", title="Remote", provider=SourceName.ARXIV)

        result = await document_service.process([metadata_only])

        assert result.outcomes == ()

    async def test_events_narrate_the_pipeline(
        self,
        document_service: DocumentIntelligenceService,
        event_bus: EventBus,
        real_paper: Paper,
    ) -> None:
        seen: list[Event] = []

        async def handler(event: Event) -> None:
            seen.append(event)

        event_bus.subscribe(None, handler)
        await document_service.process([real_paper], run_id="run-1")

        types = {event.type for event in seen}
        assert EventType.DOCUMENT_LOADED in types
        assert EventType.DOCUMENT_PARSED in types
        assert EventType.SECTIONS_DETECTED in types
        assert EventType.REFERENCES_EXTRACTED in types
        assert EventType.VALIDATION_PASSED in types
        assert EventType.EVIDENCE_GENERATED in types
        assert all(event.run_id == "run-1" for event in seen)

    async def test_failure_emits_a_parsing_failed_event(
        self,
        document_service: DocumentIntelligenceService,
        event_bus: EventBus,
        tmp_path: Path,
    ) -> None:
        seen: list[Event] = []

        async def handler(event: Event) -> None:
            seen.append(event)

        event_bus.subscribe(EventType.PARSING_FAILED, handler)
        corrupt = tmp_path / "bad.pdf"
        corrupt.write_bytes(b"nope")

        await document_service.process(
            [Paper(id="x:1", title="Bad", provider=SourceName.MANUAL, local_path=corrupt)]
        )

        assert len(seen) == 1


class TestValidatorsOnRealDocuments:
    def test_scanned_pdf_signature_is_fatal_with_an_ocr_remedy(self, real_paper: Paper) -> None:
        """A page-image PDF must be rejected, not silently yield an empty document."""
        raw = PyMuPDFLoader().load(Path(str(real_paper.local_path)), document_id=real_paper.id)
        no_text = raw.model_copy(
            update={"pages": tuple(page.model_copy(update={"blocks": ()}) for page in raw.pages)}
        )

        verdict = PDFValidator().validate(no_text)

        assert verdict.success is False
        assert verdict.is_fatal is True
        assert "pdf_no_text" in verdict.issue_codes()
        assert any("OCR" in (issue.remedy or "") for issue in verdict.issues)

    def test_metadata_mismatch_is_reported_not_resolved(
        self, document_service: DocumentIntelligenceService, real_paper: Paper
    ) -> None:
        """Zero trust: the index and the PDF are two witnesses, and disagreement is data."""
        wrong = real_paper.model_copy(update={"title": "An Entirely Unrelated Paper About Bees"})

        async def run() -> None:
            result = await document_service.process([wrong])
            verdict = result.outcomes[0].validation
            assert verdict is not None
            assert "metadata_title_mismatch" in verdict.issue_codes()

        import asyncio

        asyncio.run(run())
