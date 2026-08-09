"""Method extractor: what the paper *does*."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence
from researchagent.models.document import SectionKind
from researchagent.models.knowledge import KnowledgeKind, KnowledgeObject, MethodDetails
from researchagent.services.knowledge.base import ExtractionDraft, KnowledgeExtractor, build_object


class MethodDraft(ExtractionDraft):
    name: str = ""
    description: str = ""
    category: str = ""
    components: list[str] = Field(default_factory=list)
    is_novel: bool | None = None


class MethodBatch(BaseModel):
    methods: list[MethodDraft] = Field(default_factory=list)


class MethodExtractor(KnowledgeExtractor[MethodDraft, MethodBatch]):
    name: ClassVar[str] = "method_extractor"
    kind: ClassVar[KnowledgeKind] = KnowledgeKind.METHOD
    prompt_name: ClassVar[str] = "knowledge_method"
    source_sections: ClassVar[tuple[SectionKind, ...]] = (
        SectionKind.ABSTRACT,
        SectionKind.METHODOLOGY,
        SectionKind.EXPERIMENTS,
    )
    batch_schema: ClassVar[type[BaseModel]] = MethodBatch

    def drafts_of(self, batch: MethodBatch) -> list[MethodDraft]:
        return batch.methods

    def to_object(
        self, draft: MethodDraft, *, paper_id: str, index: int, evidence: tuple[Evidence, ...]
    ) -> KnowledgeObject | None:
        name = " ".join(draft.name.split())
        if not name:
            return None
        return build_object(
            kind=self.kind,
            paper_id=paper_id,
            index=index,
            name=name,
            description=" ".join(draft.description.split()),
            details=MethodDetails(
                category=" ".join(draft.category.split()) or None,
                components=tuple(c.strip() for c in draft.components if c.strip())[:10],
                is_novel=draft.is_novel,
            ),
            evidence=evidence,
            extracted_by=self.name,
            grounding_note=f"method '{name}' quoted verbatim at {evidence[0].location.describe()}",
        )
