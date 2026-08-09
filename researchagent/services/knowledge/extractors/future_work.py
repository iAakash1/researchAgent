"""Future work extractor: what the paper says should happen next."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence
from researchagent.models.document import SectionKind
from researchagent.models.knowledge import FutureWorkDetails, KnowledgeKind, KnowledgeObject
from researchagent.services.knowledge.base import ExtractionDraft, KnowledgeExtractor, build_object

_MIN_DESCRIPTION_CHARS = 15


class FutureWorkDraft(ExtractionDraft):
    summary: str = ""
    description: str = ""
    direction: str = ""


class FutureWorkBatch(BaseModel):
    future_work: list[FutureWorkDraft] = Field(default_factory=list)


class FutureWorkExtractor(KnowledgeExtractor[FutureWorkDraft, FutureWorkBatch]):
    name: ClassVar[str] = "future_work_extractor"
    kind: ClassVar[KnowledgeKind] = KnowledgeKind.FUTURE_WORK
    prompt_name: ClassVar[str] = "knowledge_future_work"
    source_sections: ClassVar[tuple[SectionKind, ...]] = (
        SectionKind.FUTURE_WORK,
        SectionKind.CONCLUSION,
        SectionKind.DISCUSSION,
    )
    batch_schema: ClassVar[type[BaseModel]] = FutureWorkBatch

    def drafts_of(self, batch: FutureWorkBatch) -> list[FutureWorkDraft]:
        return batch.future_work

    def to_object(
        self, draft: FutureWorkDraft, *, paper_id: str, index: int, evidence: tuple[Evidence, ...]
    ) -> KnowledgeObject | None:
        summary = " ".join(draft.summary.split()) or " ".join(draft.description.split())
        if len(summary) < _MIN_DESCRIPTION_CHARS:
            return None
        return build_object(
            kind=self.kind,
            paper_id=paper_id,
            index=index,
            name=summary[:120],
            description=" ".join(draft.description.split()) or summary,
            details=FutureWorkDetails(direction=" ".join(draft.direction.split()) or None),
            evidence=evidence,
            extracted_by=self.name,
            grounding_note=f"future direction proposed at {evidence[0].location.describe()}",
        )
