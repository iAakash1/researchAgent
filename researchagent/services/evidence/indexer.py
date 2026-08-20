"""Evidence indexing.

Populates the evidence repository from validated knowledge, so evidence becomes
retrievable on its own terms rather than only through the objects it supports.

Deduplicates by content: two extractors quoting the same sentence for the same paragraph
observed one fact twice, not two facts. Both associations are kept as links on the single
record, which is exactly the information a bundle needs to say "two independent
extractors found this".
"""

from __future__ import annotations

from researchagent.core.events import EventBus, EventType, EvidencePayload
from researchagent.core.interfaces.repositories import EvidenceRepository
from researchagent.core.logging import get_logger
from researchagent.models.evidence import (
    EvidenceLink,
    EvidenceRecord,
    EvidenceRole,
    PaperEvidence,
    content_hash_for,
)
from researchagent.models.knowledge import PaperKnowledge

logger = get_logger(__name__)


class EvidenceIndexer:
    """Turns a paper's validated knowledge into an independent evidence index."""

    name = "evidence_indexer"

    def __init__(
        self, repository: EvidenceRepository, *, event_bus: EventBus | None = None
    ) -> None:
        self._repository = repository
        self._event_bus = event_bus

    async def index(self, knowledge: PaperKnowledge, *, run_id: str | None = None) -> PaperEvidence:
        by_content: dict[str, EvidenceRecord] = {}

        for obj in knowledge.objects:
            for item in obj.evidence:
                key = content_hash_for(knowledge.paper_id, item)
                link = EvidenceLink(
                    evidence_id=item.id,
                    knowledge_object_id=obj.id,
                    knowledge_kind=obj.kind,
                    role=EvidenceRole.FOUNDING,
                    linked_by=self.name,
                )
                existing = by_content.get(key)
                if existing is None:
                    by_content[key] = EvidenceRecord(
                        evidence=item,
                        paper_id=knowledge.paper_id,
                        document_sha256=knowledge.document_sha256,
                        links=(link,),
                    )
                else:
                    # Same sentence, another fact drawn from it. One record, two links.
                    by_content[key] = existing.linked_to(
                        link.model_copy(update={"evidence_id": existing.id})
                    )

        indexed = PaperEvidence(
            paper_id=knowledge.paper_id,
            document_sha256=knowledge.document_sha256,
            records=tuple(by_content.values()),
        )
        await self._repository.save_paper(indexed)

        logger.info(
            "evidence_index_built",
            paper_id=knowledge.paper_id,
            objects=len(knowledge.objects),
            evidence_records=len(indexed.records),
            deduplicated=knowledge.evidence_count - len(indexed.records),
        )
        if self._event_bus is not None:
            await self._event_bus.emit(
                EventType.EVIDENCE_INDEXED,
                EvidencePayload(
                    document_id=knowledge.paper_id,
                    produced_by=self.name,
                    count=len(indexed.records),
                    kinds=tuple(sorted({record.kind.value for record in indexed.records})),
                ),
                run_id=run_id,
                source=self.name,
            )
        return indexed
