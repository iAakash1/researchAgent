"""Evidence intelligence pipeline.

Validated knowledge in, validated bundles out:

    PaperKnowledge -> evidence index -> per-question retrieval -> bundles -> persistence

The stage closes a loop that opened in v0.2: the research questions the Planner wrote
become the queries that assemble evidence, so every bundle can name the question it
exists to answer, and the reviewer can ask "which question is still unsupported?".

Nothing here generates text. This release ends with structured, cited, contradiction-aware
context sitting in a repository — deliberately one step short of reasoning over it.
"""

from __future__ import annotations

import time

from researchagent.config.schemas import EvidencePipelineSettings
from researchagent.core.events import BundlePayload, EventBus, EventType, ValidationPayload
from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.interfaces.bundle_repository import BundleRepository
from researchagent.core.interfaces.knowledge_repository import KnowledgeRepository
from researchagent.core.logging import get_logger, log_context
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.knowledge import KnowledgeRelation
from researchagent.models.query import ResearchQuery
from researchagent.models.research import ResearchPlan
from researchagent.schemas.evidence import (
    BundleOutcome,
    EvidenceBatchResult,
    EvidenceIndexReport,
)
from researchagent.services.evidence.builder import EvidenceBundleBuilder
from researchagent.services.evidence.indexer import EvidenceIndexer

logger = get_logger(__name__)


class EvidenceIntelligenceService:
    """Indexes evidence and assembles bundles for a plan's research questions."""

    name = "evidence_intelligence"

    def __init__(
        self,
        indexer: EvidenceIndexer,
        builder: EvidenceBundleBuilder,
        knowledge: KnowledgeRepository,
        bundles: BundleRepository,
        settings: EvidencePipelineSettings | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._indexer = indexer
        self._builder = builder
        self._knowledge = knowledge
        self._bundles = bundles
        self._settings = settings or EvidencePipelineSettings()
        self._event_bus = event_bus

    async def build_bundle(self, query: ResearchQuery) -> EvidenceBundle:
        """Assemble one bundle for an ad-hoc query.

        The same builder the workflow uses, exposed for callers that already know what
        they want to ask — the API today, the reasoning engine from v0.9.
        """
        return await self._builder.build(query)

    async def process(
        self, plan: ResearchPlan, paper_ids: tuple[str, ...], *, run_id: str | None = None
    ) -> EvidenceBatchResult:
        index_report = await self._index(paper_ids, run_id)
        outcomes = await self._build_bundles(plan, paper_ids, run_id)

        result = EvidenceBatchResult(index=index_report, outcomes=tuple(outcomes))
        logger.info(
            "evidence_pipeline_complete",
            papers_indexed=index_report.papers_indexed,
            evidence_records=index_report.evidence_records,
            bundles=len(outcomes),
            trusted_bundles=result.trusted,
            contradictions=result.total_contradictions,
        )
        return result

    async def _index(self, paper_ids: tuple[str, ...], run_id: str | None) -> EvidenceIndexReport:
        """Populate the evidence repository from validated knowledge."""
        indexed = 0
        records = 0
        skipped: list[str] = []

        for paper_id in paper_ids:
            stored = await self._knowledge.get(paper_id)
            if stored is None:
                skipped.append(paper_id)
                continue
            if not stored.is_trusted:
                # Zero trust across stages: knowledge the previous stage rejected does not
                # become retrievable evidence.
                logger.info("untrusted_knowledge_not_indexed", paper_id=paper_id)
                skipped.append(paper_id)
                continue

            with log_context(paper_id=paper_id):
                try:
                    paper_evidence = await self._indexer.index(stored.value, run_id=run_id)
                except ResearchAgentError as exc:
                    logger.warning("evidence_index_failed", paper_id=paper_id, error_code=exc.code)
                    skipped.append(paper_id)
                    continue

            indexed += 1
            records += len(paper_evidence.records)

        return EvidenceIndexReport(
            papers_indexed=indexed,
            papers_skipped=tuple(skipped),
            evidence_records=records,
        )

    async def _build_bundles(
        self, plan: ResearchPlan, paper_ids: tuple[str, ...], run_id: str | None
    ) -> list[BundleOutcome]:
        """One bundle per research question — the loop back to the Planner."""
        relations = await self._relations_for(paper_ids)
        outcomes: list[BundleOutcome] = []

        for question in plan.research_questions[: self._settings.max_bundles_per_run]:
            query = ResearchQuery.for_question(
                question,
                kinds=self._settings.bundle_kinds,
                paper_ids=paper_ids,
                limit=self._settings.max_objects_per_bundle,
            )
            outcomes.append(await self._build_one(query, relations, run_id))

        return outcomes

    async def _build_one(
        self, query: ResearchQuery, relations: tuple[KnowledgeRelation, ...], run_id: str | None
    ) -> BundleOutcome:
        started = time.perf_counter()
        try:
            bundle = await self._builder.build(query, relations=relations)
        except ResearchAgentError as exc:
            logger.warning(
                "bundle_build_failed", question_id=query.question_id, error_code=exc.code
            )
            return BundleOutcome(
                question_id=query.question_id,
                succeeded=False,
                error_code=exc.code,
                error_message=exc.message,
                duration_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            logger.exception(
                "bundle_build_crashed",
                question_id=query.question_id,
                error_type=type(exc).__name__,
            )
            return BundleOutcome(
                question_id=query.question_id,
                succeeded=False,
                error_code="unexpected_error",
                error_message=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(started),
            )

        await self._bundles.save(bundle)
        await self._emit(bundle, run_id)

        return BundleOutcome(
            question_id=query.question_id,
            succeeded=bundle.is_trusted,
            bundle=bundle,
            duration_ms=_elapsed_ms(started),
        )

    async def _relations_for(self, paper_ids: tuple[str, ...]) -> tuple[KnowledgeRelation, ...]:
        relations: list[KnowledgeRelation] = []
        for paper_id in paper_ids:
            stored = await self._knowledge.get(paper_id)
            if stored is not None and stored.is_trusted:
                relations.extend(stored.value.relations)
        return tuple(relations)

    async def _emit(self, bundle: EvidenceBundle, run_id: str | None) -> None:
        if self._event_bus is None:
            return

        await self._event_bus.emit(
            EventType.BUNDLE_CREATED,
            BundlePayload(
                bundle_id=bundle.id,
                question_id=bundle.query.question_id,
                knowledge_objects=len(bundle.knowledge_objects),
                evidence_items=len(bundle.evidence),
                papers=bundle.coverage.paper_count,
                contradictions=len(bundle.contradictions),
                confidence=bundle.confidence.score,
            ),
            run_id=run_id,
            source=self.name,
        )
        if bundle.contradictions:
            await self._event_bus.emit(
                EventType.CONTRADICTION_DETECTED,
                BundlePayload(
                    bundle_id=bundle.id,
                    question_id=bundle.query.question_id,
                    contradictions=len(bundle.contradictions),
                    confidence=bundle.confidence.score,
                ),
                run_id=run_id,
                source=self.name,
            )
        await self._event_bus.emit(
            EventType.VALIDATION_PASSED if bundle.is_trusted else EventType.VALIDATION_FAILED,
            ValidationPayload(
                validator=bundle.validation.validator,
                subject_id=bundle.id,
                subject_type="EvidenceBundle",
                success=bundle.is_trusted,
                confidence=bundle.confidence.score,
                issue_codes=bundle.validation.issue_codes(),
            ),
            run_id=run_id,
            source=self.name,
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
