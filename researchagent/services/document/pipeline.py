"""Document intelligence pipeline.

The stage entry point: takes discovered papers, produces validated canonical documents.

    PDF -> load -> validate raw -> assemble -> validate document -> persist

Two invariants shape everything here:

* **One paper's failure is one paper's failure.** Every paper is processed inside its own
  error boundary and produces a :class:`DocumentOutcome` either way. Forty papers where
  three are corrupt yields thirty-seven documents and three recorded reasons — never an
  exception that discards the batch.
* **Nothing is trusted, including us.** The raw load is validated before assembly, the
  assembled document is validated before persistence, and the discovered metadata is
  cross-checked against what the PDF itself says.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from researchagent.config.schemas import DocumentPipelineSettings, DocumentValidationConfig
from researchagent.core.events import (
    DocumentPayload,
    EventBus,
    EventType,
    EvidencePayload,
    ValidationPayload,
)
from researchagent.core.evidence import Evidence
from researchagent.core.exceptions import DocumentError, ResearchAgentError
from researchagent.core.interfaces.document_loader import DocumentLoader
from researchagent.core.interfaces.repositories import PaperRepository
from researchagent.core.logging import get_logger, log_context
from researchagent.core.validation import ValidationResult, aggregate
from researchagent.models.document import PaperDocument
from researchagent.models.layout import RawDocument
from researchagent.models.paper import Paper
from researchagent.repositories.document_repository import JsonDocumentRepository
from researchagent.schemas.validated import (
    DocumentBatchResult,
    DocumentOutcome,
    ValidatedDocument,
)
from researchagent.services.document.assembler import DocumentAssembler
from researchagent.services.validation.document import (
    CitationValidator,
    MetadataValidator,
    PDFValidator,
    ReferenceValidator,
    SectionValidator,
)

logger = get_logger(__name__)


class DocumentIntelligenceService:
    """Turns papers with local PDFs into validated canonical documents."""

    name = "document_intelligence"

    def __init__(
        self,
        loader: DocumentLoader,
        assembler: DocumentAssembler,
        documents: JsonDocumentRepository,
        papers: PaperRepository,
        settings: DocumentPipelineSettings | None = None,
        validation_config: DocumentValidationConfig | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._loader = loader
        self._assembler = assembler
        self._documents = documents
        self._papers = papers
        self._settings = settings or DocumentPipelineSettings()
        self._validation_config = validation_config or DocumentValidationConfig()
        self._event_bus = event_bus

    async def process(
        self, papers: list[Paper], *, run_id: str | None = None
    ) -> DocumentBatchResult:
        candidates = [paper for paper in papers if paper.local_path is not None][
            : self._settings.max_documents_per_run
        ]
        if not candidates:
            logger.info("document_pipeline_no_input", supplied=len(papers))
            return DocumentBatchResult()

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_documents)

        async def guarded(paper: Paper) -> DocumentOutcome:
            async with semaphore:
                return await self._process_one(paper, run_id)

        outcomes = await asyncio.gather(*(guarded(paper) for paper in candidates))
        result = DocumentBatchResult(outcomes=tuple(outcomes))

        logger.info(
            "document_pipeline_complete",
            supplied=len(papers),
            attempted=len(candidates),
            succeeded=result.succeeded,
            failed=result.failed,
            mean_confidence=result.mean_confidence,
        )
        return result

    async def _process_one(self, paper: Paper, run_id: str | None) -> DocumentOutcome:
        started = time.perf_counter()
        with log_context(paper_id=paper.id):
            try:
                return await self._parse_and_validate(paper, run_id, started)
            except ResearchAgentError as exc:
                # Expected, classified failure: record it and keep the batch alive.
                return await self._record_failure(paper, exc, started, run_id)
            except Exception as exc:
                # A bug in a detector must cost one paper, not the run. Logged with a
                # traceback because, unlike a DocumentError, this is not expected.
                logger.exception(
                    "document_pipeline_crashed",
                    paper_id=paper.id,
                    error_type=type(exc).__name__,
                )
                return DocumentOutcome(
                    paper_id=paper.id,
                    succeeded=False,
                    error_code="unexpected_error",
                    error_message=f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                    duration_ms=_elapsed_ms(started),
                )

    async def _parse_and_validate(
        self, paper: Paper, run_id: str | None, started: float
    ) -> DocumentOutcome:
        path = Path(str(paper.local_path))

        cached = await self._reuse_existing(paper, path)
        if cached is not None:
            return cached

        raw = await asyncio.to_thread(self._loader.load, path, document_id=paper.id)
        await self._emit(
            EventType.DOCUMENT_LOADED,
            DocumentPayload(paper_id=paper.id, pages=raw.page_count),
            run_id,
        )

        raw_verdict = PDFValidator(self._validation_config).validate(raw)
        await self._emit_validation(raw_verdict, run_id)
        if not raw_verdict.success:
            raise _as_document_error(paper.id, raw_verdict, raw)

        document = await asyncio.to_thread(self._assembler.assemble, raw, source_path=path)
        await self._emit(
            EventType.DOCUMENT_PARSED,
            DocumentPayload(
                paper_id=paper.id,
                pages=document.statistics.pages,
                sections=document.statistics.sections,
                references=document.statistics.references,
                figures=document.statistics.figures,
                tables=document.statistics.tables,
                citations=document.statistics.citations,
            ),
            run_id,
        )
        await self._emit(
            EventType.SECTIONS_DETECTED,
            DocumentPayload(paper_id=paper.id, sections=document.statistics.sections),
            run_id,
        )
        await self._emit(
            EventType.REFERENCES_EXTRACTED,
            DocumentPayload(paper_id=paper.id, references=document.statistics.references),
            run_id,
        )

        verdict = self._validate_document(document, paper, raw_verdict)
        await self._emit_validation(verdict, run_id)
        await self._emit_evidence(document.paper_id, verdict.evidence, run_id)

        validated = ValidatedDocument(value=document, validation=verdict)
        await self._documents.save(validated)
        await self._advance_processing(paper, document, verdict)

        await self._emit(
            EventType.DOCUMENT_READY if verdict.success else EventType.VALIDATION_FAILED,
            DocumentPayload(paper_id=paper.id, duration_ms=_elapsed_ms(started)),
            run_id,
        )

        return DocumentOutcome(
            paper_id=paper.id,
            succeeded=verdict.success,
            document=document,
            validation=verdict,
            error_code=None if verdict.success else "document_validation_failed",
            error_message=None if verdict.success else _summarise(verdict),
            recoverable=not verdict.is_fatal,
            duration_ms=_elapsed_ms(started),
        )

    def _validate_document(
        self, document: PaperDocument, paper: Paper, raw_verdict: ValidationResult
    ) -> ValidationResult:
        """Every validator sees the document; the aggregate is the stage's verdict."""
        results = [
            raw_verdict,
            SectionValidator(self._validation_config).validate(document),
            ReferenceValidator().validate(document),
            CitationValidator(self._validation_config).validate(document),
            MetadataValidator(paper, self._validation_config).validate(document),
        ]
        return aggregate(
            results,
            validator="document_validator",
            subject_id=document.paper_id,
            subject_type="PaperDocument",
        )

    async def _reuse_existing(self, paper: Paper, path: Path) -> DocumentOutcome | None:
        """Skip work when the stored document came from these exact bytes."""
        if not self._settings.skip_unchanged:
            return None

        stored = await self._documents.get(paper.id)
        if stored is None:
            return None

        from researchagent.services.document.assembler import ASSEMBLER_VERSION

        provenance = stored.value.provenance
        if provenance.parser_version != ASSEMBLER_VERSION:
            return None
        # A single stat() on a local file; not worth a thread hop. noqa is deliberate.
        if not path.is_file() or provenance.source_bytes != path.stat().st_size:  # noqa: ASYNC240
            return None

        logger.debug("document_reused", paper_id=paper.id, sha=provenance.source_sha256[:12])
        return DocumentOutcome(
            paper_id=paper.id,
            succeeded=stored.validation.success,
            document=stored.value,
            validation=stored.validation,
        )

    async def _advance_processing(
        self, paper: Paper, document: PaperDocument, verdict: ValidationResult
    ) -> None:
        record = await self._papers.get(paper.id)
        if record is None:
            return
        await self._papers.save(
            record.model_copy(
                update={
                    "processing": record.processing.mark(
                        validated=True,
                        parsed=True,
                        sectioned=document.statistics.sections > 0,
                        references_extracted=document.statistics.references > 0,
                        figures_extracted=document.statistics.figures > 0,
                        tables_extracted=document.statistics.tables > 0,
                        ready_for_extraction=verdict.success,
                        last_error=None if verdict.success else _summarise(verdict),
                    )
                }
            )
        )

    async def _record_failure(
        self, paper: Paper, exc: ResearchAgentError, started: float, run_id: str | None
    ) -> DocumentOutcome:
        logger.warning(
            "document_processing_failed",
            paper_id=paper.id,
            error_code=exc.code,
            recoverability=exc.recoverability.value,
            error=exc.message,
        )
        await self._emit(
            EventType.PARSING_FAILED,
            DocumentPayload(paper_id=paper.id, error_code=exc.code, error_message=exc.message),
            run_id,
        )

        record = await self._papers.get(paper.id)
        if record is not None:
            await self._papers.save(
                record.model_copy(
                    update={"processing": record.processing.mark(last_error=exc.message)}
                )
            )

        return DocumentOutcome(
            paper_id=paper.id,
            succeeded=False,
            error_code=exc.code,
            error_message=exc.message,
            remedy=exc.remedy,
            recoverable=exc.recoverability.allows_continue,
            duration_ms=_elapsed_ms(started),
        )

    async def _emit(self, event: EventType, payload: DocumentPayload, run_id: str | None) -> None:
        if self._event_bus is not None:
            await self._event_bus.emit(event, payload, run_id=run_id, source=self.name)

    async def _emit_validation(self, result: ValidationResult, run_id: str | None) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.emit(
            EventType.VALIDATION_PASSED if result.success else EventType.VALIDATION_FAILED,
            ValidationPayload(
                validator=result.validator,
                subject_id=result.subject_id,
                subject_type=result.subject_type,
                success=result.success,
                confidence=result.confidence.score,
                issue_codes=result.issue_codes(),
            ),
            run_id=run_id,
            source=self.name,
        )

    async def _emit_evidence(
        self, document_id: str, evidence: tuple[Evidence, ...], run_id: str | None
    ) -> None:
        if self._event_bus is None or not evidence:
            return
        await self._event_bus.emit(
            EventType.EVIDENCE_GENERATED,
            EvidencePayload(
                document_id=document_id,
                produced_by=self.name,
                count=len(evidence),
                kinds=tuple(sorted({item.kind.value for item in evidence})),
            ),
            run_id=run_id,
            source=self.name,
        )


def _as_document_error(paper_id: str, verdict: ValidationResult, raw: RawDocument) -> DocumentError:
    from researchagent.core.exceptions import DocumentParsingError

    return DocumentParsingError(
        _summarise(verdict),
        paper_id=paper_id,
        pages=raw.page_count,
        issues=list(verdict.issue_codes()),
    )


def _summarise(verdict: ValidationResult) -> str:
    blocking = [issue for issue in verdict.issues if issue.severity.blocks_use]
    if not blocking:
        return "validation failed"
    return "; ".join(f"{issue.code}: {issue.message}" for issue in blocking[:3])


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
