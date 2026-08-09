"""Dataset extractor: what the paper evaluates on."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence
from researchagent.models.document import SectionKind
from researchagent.models.knowledge import DatasetDetails, KnowledgeKind, KnowledgeObject
from researchagent.services.knowledge.base import ExtractionDraft, KnowledgeExtractor, build_object


class DatasetDraft(ExtractionDraft):
    name: str = ""
    description: str = ""
    domain: str = ""
    size: str = ""
    url: str = ""
    is_public: bool | None = None


class DatasetBatch(BaseModel):
    datasets: list[DatasetDraft] = Field(default_factory=list)


class DatasetExtractor(KnowledgeExtractor[DatasetDraft, DatasetBatch]):
    name: ClassVar[str] = "dataset_extractor"
    kind: ClassVar[KnowledgeKind] = KnowledgeKind.DATASET
    prompt_name: ClassVar[str] = "knowledge_dataset"
    source_sections: ClassVar[tuple[SectionKind, ...]] = (
        SectionKind.EXPERIMENTS,
        SectionKind.EVALUATION,
        SectionKind.METHODOLOGY,
        SectionKind.RESULTS,
    )
    batch_schema: ClassVar[type[BaseModel]] = DatasetBatch

    def drafts_of(self, batch: DatasetBatch) -> list[DatasetDraft]:
        return batch.datasets

    def to_object(
        self, draft: DatasetDraft, *, paper_id: str, index: int, evidence: tuple[Evidence, ...]
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
            details=DatasetDetails(
                domain=" ".join(draft.domain.split()) or None,
                size=" ".join(draft.size.split()) or None,
                url=draft.url.strip() or None,
                is_public=draft.is_public,
            ),
            evidence=evidence,
            extracted_by=self.name,
            grounding_note=f"dataset '{name}' named at {evidence[0].location.describe()}",
        )
