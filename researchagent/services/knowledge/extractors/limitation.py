"""Limitation extractor: what the paper says it cannot do.

Limitations are the highest-value knowledge for gap discovery in v0.8 and the easiest to
hallucinate — a plausible-sounding weakness can be written for any paper. Grounding is
therefore the only thing that admits one.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence
from researchagent.models.document import SectionKind
from researchagent.models.knowledge import KnowledgeKind, KnowledgeObject, LimitationDetails
from researchagent.services.knowledge.base import ExtractionDraft, KnowledgeExtractor, build_object

_MIN_DESCRIPTION_CHARS = 15


class LimitationDraft(ExtractionDraft):
    summary: str = ""
    description: str = ""
    affects: str = ""
    acknowledged_by_authors: bool = True


class LimitationBatch(BaseModel):
    limitations: list[LimitationDraft] = Field(default_factory=list)


class LimitationExtractor(KnowledgeExtractor[LimitationDraft, LimitationBatch]):
    name: ClassVar[str] = "limitation_extractor"
    kind: ClassVar[KnowledgeKind] = KnowledgeKind.LIMITATION
    prompt_name: ClassVar[str] = "knowledge_limitation"
    source_sections: ClassVar[tuple[SectionKind, ...]] = (
        SectionKind.LIMITATIONS,
        SectionKind.DISCUSSION,
        SectionKind.CONCLUSION,
        SectionKind.EVALUATION,
    )
    batch_schema: ClassVar[type[BaseModel]] = LimitationBatch

    def drafts_of(self, batch: LimitationBatch) -> list[LimitationDraft]:
        return batch.limitations

    def to_object(
        self, draft: LimitationDraft, *, paper_id: str, index: int, evidence: tuple[Evidence, ...]
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
            details=LimitationDetails(
                acknowledged_by_authors=draft.acknowledged_by_authors,
                affects=" ".join(draft.affects.split()) or None,
            ),
            evidence=evidence,
            extracted_by=self.name,
            grounding_note=(
                f"limitation stated by the authors at {evidence[0].location.describe()}"
            ),
        )
