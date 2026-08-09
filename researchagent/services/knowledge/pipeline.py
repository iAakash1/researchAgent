"""Knowledge intelligence pipeline.

Validated documents in, validated knowledge out:

    PaperDocument -> extractors -> grounding -> per-object validation -> relations
                  -> coverage validation -> persistence

Three properties carried forward from v0.4, because they are what make the pipeline
usable rather than merely impressive:

* **Per-paper isolation.** One paper's failure costs one paper.
* **Per-extractor isolation.** One extractor failing costs one kind of knowledge, not the
  other five — a model that chokes on the results prompt still yields methods and datasets.
* **Rejection is data.** Objects dropped for being ungrounded or invalid are counted and
  reported; the rejection rate is the system's own hallucination measure.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from researchagent.config.schemas import KnowledgePipelineSettings, KnowledgeValidationConfig
from researchagent.core.events import EventBus, EventType, KnowledgePayload, ValidationPayload
from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.interfaces.paper_repository import PaperRepository
from researchagent.core.logging import get_logger, log_context
from researchagent.core.validation import Confidence, ValidationResult, aggregate
from researchagent.models.document import PaperDocument
from researchagent.models.knowledge import (
    ExtractionStats,
    KnowledgeKind,
    KnowledgeObject,
    PaperKnowledge,
)
from researchagent.repositories.document_repository import JsonDocumentRepository
from researchagent.repositories.knowledge_repository import JsonKnowledgeRepository
from researchagent.schemas.knowledge import (
    KnowledgeBatchResult,
    KnowledgeOutcome,
    RejectionReport,
    ValidatedKnowledge,
)
from researchagent.services.knowledge.base import ExtractionOutcome, KnowledgeExtractor
from researchagent.services.knowledge.grounding import EvidenceGrounder
from researchagent.services.knowledge.relations import RelationBuilder
from researchagent.services.validation.knowledge import (
    CompletenessValidator,
    EvidenceValidator,
    KnowledgeCoverageValidator,
    RelationshipValidator,
    ResultValidator,
)
from researchagent.utils.text import normalise

logger = get_logger(__name__)


class KnowledgeIntelligenceService:
    """Turns validated documents into validated knowledge."""

    name = "knowledge_intelligence"

    def __init__(
        self,
        extractors: list[KnowledgeExtractor[Any, Any]],
        relations: RelationBuilder,
        repository: JsonKnowledgeRepository,
        documents: JsonDocumentRepository,
        papers: PaperRepository,
        settings: KnowledgePipelineSettings | None = None,
        validation_config: KnowledgeValidationConfig | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._extractors = extractors
        self._relations = relations
        self._repository = repository
        self._documents = documents
        self._papers = papers
        self._settings = settings or KnowledgePipelineSettings()
        self._validation = validation_config or KnowledgeValidationConfig()
        self._event_bus = event_bus

    @property
    def extractor_names(self) -> list[str]:
        return [extractor.name for extractor in self._extractors]

    async def documents_for(self, state: object) -> list[PaperDocument]:
        """Load the validated documents a run produced.

        Documents are large and belong in the repository, not in workflow state; the
        state carries only the ids the document stage marked ready. Loading here keeps
        the workflow node free of persistence concerns.
        """
        ready: tuple[str, ...] = getattr(
            getattr(state, "documents", None), "ready_for_extraction", ()
        )
        loaded: list[PaperDocument] = []
        for paper_id in ready:
            stored = await self._documents.get(paper_id)
            if stored is None:
                logger.warning("document_missing_for_extraction", paper_id=paper_id)
                continue
            if not stored.is_trusted:
                # Zero trust across stages: a document the previous stage rejected is not
                # a document this stage may reason over.
                logger.info("untrusted_document_skipped", paper_id=paper_id)
                continue
            loaded.append(stored.value)
        return loaded

    async def process(
        self, documents: list[PaperDocument], *, run_id: str | None = None
    ) -> KnowledgeBatchResult:
        selected = documents[: self._settings.max_documents_per_run]
        if not selected:
            logger.info("knowledge_pipeline_no_input", supplied=len(documents))
            return KnowledgeBatchResult()

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_documents)

        async def guarded(document: PaperDocument) -> KnowledgeOutcome:
            async with semaphore:
                return await self._process_one(document, run_id)

        outcomes = await asyncio.gather(*(guarded(document) for document in selected))
        result = KnowledgeBatchResult(outcomes=tuple(outcomes))

        logger.info(
            "knowledge_pipeline_complete",
            documents=len(selected),
            succeeded=result.succeeded,
            failed=result.failed,
            objects=result.total_objects,
            rejected=result.total_rejected,
            grounding_rate=result.grounding_rate,
        )
        return result

    async def _process_one(self, document: PaperDocument, run_id: str | None) -> KnowledgeOutcome:
        started = time.perf_counter()
        with log_context(paper_id=document.paper_id):
            try:
                return await self._extract_and_validate(document, run_id, started)
            except ResearchAgentError as exc:
                logger.warning(
                    "knowledge_extraction_failed",
                    paper_id=document.paper_id,
                    error_code=exc.code,
                    error=exc.message,
                )
                return KnowledgeOutcome(
                    paper_id=document.paper_id,
                    succeeded=False,
                    error_code=exc.code,
                    error_message=exc.message,
                    remedy=exc.remedy,
                    duration_ms=_elapsed_ms(started),
                )
            except Exception as exc:
                logger.exception(
                    "knowledge_pipeline_crashed",
                    paper_id=document.paper_id,
                    error_type=type(exc).__name__,
                )
                return KnowledgeOutcome(
                    paper_id=document.paper_id,
                    succeeded=False,
                    error_code="unexpected_error",
                    error_message=f"{type(exc).__name__}: {exc}",
                    duration_ms=_elapsed_ms(started),
                )

    async def _extract_and_validate(
        self, document: PaperDocument, run_id: str | None, started: float
    ) -> KnowledgeOutcome:
        cached = await self._reuse_existing(document)
        if cached is not None:
            return cached

        grounder = EvidenceGrounder(
            document, similarity_threshold=self._settings.grounding_similarity_threshold
        )

        # Extractors are independent by construction, so they run concurrently; each
        # returns its own outcome rather than raising.
        extraction_outcomes = list(
            await asyncio.gather(
                *(extractor.extract(document, grounder) for extractor in self._extractors)
            )
        )

        grounded = [obj for outcome in extraction_outcomes for obj in outcome.objects]
        ungrounded = sum(outcome.drafts_rejected_ungrounded for outcome in extraction_outcomes)

        accepted, object_verdicts, rejected_codes = self._validate_objects(grounded)
        accepted = _deduplicate(accepted)
        relations = self._relations.build(tuple(accepted))

        stats = ExtractionStats(
            proposed=sum(o.drafts_proposed for o in extraction_outcomes),
            grounded=len(grounded),
            accepted=len(accepted),
            rejected_ungrounded=ungrounded,
            rejected_invalid=len(grounded) - len(accepted),
            rejection_codes=tuple(sorted(set(rejected_codes))),
        )
        knowledge = PaperKnowledge(
            paper_id=document.paper_id,
            document_sha256=document.provenance.source_sha256,
            objects=tuple(accepted),
            relations=relations,
            extraction=stats,
        )
        verdict = self._validate_knowledge(knowledge, extraction_outcomes, object_verdicts)

        validated = ValidatedKnowledge(value=knowledge, validation=verdict)
        await self._repository.save(validated)
        await self._advance_processing(document, verdict)
        await self._emit(document, knowledge, verdict, run_id)

        return KnowledgeOutcome(
            paper_id=document.paper_id,
            succeeded=verdict.success,
            knowledge=knowledge,
            validation=verdict,
            rejections=RejectionReport(
                ungrounded=stats.rejected_ungrounded,
                invalid=stats.rejected_invalid,
                by_code=stats.rejection_codes,
            ),
            drafts_proposed=stats.proposed,
            extractor_errors=tuple(
                f"{o.extractor}: {o.error}" for o in extraction_outcomes if o.error
            ),
            error_code=None if verdict.success else "knowledge_validation_failed",
            error_message=None if verdict.success else _summarise(verdict),
            duration_ms=_elapsed_ms(started),
        )

    def _validate_objects(
        self, proposed: list[KnowledgeObject]
    ) -> tuple[list[KnowledgeObject], list[ValidationResult], list[str]]:
        """Every object faces every applicable validator; failures are dropped, not fixed."""
        evidence_validator = EvidenceValidator(self._validation)
        completeness_validator = CompletenessValidator(self._validation)
        result_validator = ResultValidator()

        accepted: list[KnowledgeObject] = []
        verdicts: list[ValidationResult] = []
        rejected_codes: list[str] = []

        for candidate in proposed:
            checks = [
                evidence_validator.validate(candidate),
                completeness_validator.validate(candidate),
            ]
            if candidate.kind is KnowledgeKind.RESULT:
                checks.append(result_validator.validate(candidate))

            verdict = aggregate(
                checks,
                validator="knowledge_object_validator",
                subject_id=candidate.id,
                subject_type="KnowledgeObject",
            )
            verdicts.append(verdict)

            if not verdict.success:
                rejected_codes.extend(verdict.issue_codes())
                logger.info(
                    "knowledge_object_rejected",
                    object_id=candidate.id,
                    kind=candidate.kind.value,
                    codes=list(verdict.issue_codes()),
                )
                continue

            accepted.append(
                candidate.model_copy(
                    update={
                        "confidence": candidate.confidence.combined_with(verdict.confidence),
                        "validated_by": tuple(check.validator for check in checks),
                    }
                )
            )

        return accepted, verdicts, rejected_codes

    def _validate_knowledge(
        self,
        knowledge: PaperKnowledge,
        extraction_outcomes: list[ExtractionOutcome],
        object_verdicts: list[ValidationResult],
    ) -> ValidationResult:
        checks = [
            RelationshipValidator().validate(knowledge),
            KnowledgeCoverageValidator().validate(knowledge),
        ]
        # Per-extractor grounding rates are the most informative signal available about
        # how much of this knowledge the paper actually supports.
        signals = [
            signal
            for extractor, outcome in zip(self._extractors, extraction_outcomes, strict=True)
            for signal in extractor.confidence_signals(outcome)
        ]
        grounding = ValidationResult.passed(
            validator="extraction_grounding",
            subject_id=knowledge.paper_id,
            subject_type="PaperKnowledge",
            confidence=Confidence.from_signals(signals),
        )

        return aggregate(
            [*checks, grounding, *object_verdicts],
            validator="knowledge_validator",
            subject_id=knowledge.paper_id,
            subject_type="PaperKnowledge",
        )

    async def _reuse_existing(self, document: PaperDocument) -> KnowledgeOutcome | None:
        """Extraction is expensive; skip when the same document bytes were already done."""
        if not self._settings.skip_unchanged:
            return None

        stored = await self._repository.get(document.paper_id)
        if stored is None:
            return None
        if stored.value.document_sha256 != document.provenance.source_sha256:
            return None

        logger.debug("knowledge_reused", paper_id=document.paper_id)
        stats = stored.value.extraction
        return KnowledgeOutcome(
            paper_id=document.paper_id,
            succeeded=stored.validation.success,
            knowledge=stored.value,
            validation=stored.validation,
            # Restored from the artefact, so a cached paper reports the same counters it
            # reported when it was extracted rather than silently contributing a
            # numerator with no denominator.
            drafts_proposed=stats.proposed if stats else 0,
            rejections=RejectionReport(
                ungrounded=stats.rejected_ungrounded,
                invalid=stats.rejected_invalid,
                by_code=stats.rejection_codes,
            )
            if stats
            else RejectionReport(),
        )

    async def _advance_processing(self, document: PaperDocument, verdict: ValidationResult) -> None:
        record = await self._papers.get(document.paper_id)
        if record is None:
            return
        await self._papers.save(
            record.model_copy(
                update={
                    "processing": record.processing.mark(
                        extracted=verdict.success,
                        last_error=None if verdict.success else _summarise(verdict),
                    )
                }
            )
        )

    async def _emit(
        self,
        document: PaperDocument,
        knowledge: PaperKnowledge,
        verdict: ValidationResult,
        run_id: str | None,
    ) -> None:
        if self._event_bus is None:
            return

        await self._event_bus.emit(
            EventType.KNOWLEDGE_EXTRACTED,
            KnowledgePayload(
                paper_id=document.paper_id,
                objects=len(knowledge.objects),
                relations=len(knowledge.relations),
                evidence=knowledge.evidence_count,
                kinds=tuple(kind.value for kind in knowledge.kinds_present),
            ),
            run_id=run_id,
            source=self.name,
        )
        await self._event_bus.emit(
            EventType.VALIDATION_PASSED if verdict.success else EventType.VALIDATION_FAILED,
            ValidationPayload(
                validator=verdict.validator,
                subject_id=verdict.subject_id,
                subject_type=verdict.subject_type,
                success=verdict.success,
                confidence=verdict.confidence.score,
                issue_codes=verdict.issue_codes(),
            ),
            run_id=run_id,
            source=self.name,
        )


def _deduplicate(objects: list[KnowledgeObject]) -> list[KnowledgeObject]:
    """Collapse objects naming the same entity.

    Extractors read overlapping sections, so the same dataset legitimately gets proposed
    from two places. Keeping both would double-count it in v0.7's graph and make the
    v0.8 synthesis claim two papers agree when one paper said it twice.

    The survivor keeps the highest confidence and inherits the other's evidence — more
    provenance for the same fact is strictly better.
    """
    best: dict[tuple[str, str], KnowledgeObject] = {}

    for candidate in objects:
        key = (candidate.kind.value, normalise(candidate.name))
        existing = best.get(key)
        if existing is None:
            best[key] = candidate
            continue

        keeper, other = (
            (existing, candidate)
            if existing.confidence.score >= candidate.confidence.score
            else (candidate, existing)
        )
        merged_evidence = (
            *keeper.evidence,
            *(e for e in other.evidence if e not in keeper.evidence),
        )
        best[key] = keeper.model_copy(update={"evidence": merged_evidence})

    return list(best.values())


def _summarise(verdict: ValidationResult) -> str:
    blocking = [issue for issue in verdict.issues if issue.severity.blocks_use]
    if not blocking:
        return "validation failed"
    return "; ".join(f"{issue.code}: {issue.message}" for issue in blocking[:3])


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
