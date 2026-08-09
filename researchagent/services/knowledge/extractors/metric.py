"""Metric extractor: how the paper measures success."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence
from researchagent.models.document import SectionKind
from researchagent.models.knowledge import KnowledgeKind, KnowledgeObject, MetricDetails
from researchagent.services.knowledge.base import ExtractionDraft, KnowledgeExtractor, build_object


class MetricDraft(ExtractionDraft):
    name: str = ""
    description: str = ""
    unit: str = ""
    higher_is_better: bool | None = None


class MetricBatch(BaseModel):
    metrics: list[MetricDraft] = Field(default_factory=list)


class MetricExtractor(KnowledgeExtractor[MetricDraft, MetricBatch]):
    name: ClassVar[str] = "metric_extractor"
    kind: ClassVar[KnowledgeKind] = KnowledgeKind.METRIC
    prompt_name: ClassVar[str] = "knowledge_metric"
    source_sections: ClassVar[tuple[SectionKind, ...]] = (
        SectionKind.EVALUATION,
        SectionKind.EXPERIMENTS,
        SectionKind.RESULTS,
    )
    batch_schema: ClassVar[type[BaseModel]] = MetricBatch

    def drafts_of(self, batch: MetricBatch) -> list[MetricDraft]:
        return batch.metrics

    def to_object(
        self, draft: MetricDraft, *, paper_id: str, index: int, evidence: tuple[Evidence, ...]
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
            details=MetricDetails(
                unit=draft.unit.strip() or None, higher_is_better=draft.higher_is_better
            ),
            evidence=evidence,
            extracted_by=self.name,
            grounding_note=f"metric '{name}' reported at {evidence[0].location.describe()}",
        )
