"""Result extractor: the numbers the paper reports.

The highest-stakes extractor in the release. A fabricated accuracy figure is the exact
failure the whole architecture exists to prevent, so a result is kept only when its
number appears verbatim in the grounded quote — the model may not paraphrase a value into
existence.
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence
from researchagent.core.logging import get_logger
from researchagent.models.document import SectionKind
from researchagent.models.knowledge import KnowledgeKind, KnowledgeObject, ResultDetails
from researchagent.services.knowledge.base import ExtractionDraft, KnowledgeExtractor, build_object
from researchagent.services.knowledge.grounding import normalise

logger = get_logger(__name__)

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


class ResultDraft(ExtractionDraft):
    metric_name: str = ""
    dataset_name: str = ""
    value: str = ""
    unit: str = ""
    description: str = ""
    baseline_comparison: str = ""


class ResultBatch(BaseModel):
    results: list[ResultDraft] = Field(default_factory=list)


class ResultExtractor(KnowledgeExtractor[ResultDraft, ResultBatch]):
    name: ClassVar[str] = "result_extractor"
    kind: ClassVar[KnowledgeKind] = KnowledgeKind.RESULT
    prompt_name: ClassVar[str] = "knowledge_result"
    source_sections: ClassVar[tuple[SectionKind, ...]] = (
        SectionKind.RESULTS,
        SectionKind.EVALUATION,
        SectionKind.EXPERIMENTS,
        SectionKind.ABSTRACT,
    )
    batch_schema: ClassVar[type[BaseModel]] = ResultBatch

    def drafts_of(self, batch: ResultBatch) -> list[ResultDraft]:
        return batch.results

    def to_object(
        self, draft: ResultDraft, *, paper_id: str, index: int, evidence: tuple[Evidence, ...]
    ) -> KnowledgeObject | None:
        value = " ".join(draft.value.split())
        metric = " ".join(draft.metric_name.split())
        if not value or not metric:
            # A result without a metric or a number is not a result.
            return None

        quote = evidence[0].quote or ""
        if not _value_appears_in(value, quote):
            # Grounding proved the sentence exists; this proves the *number* does. A
            # model that cites a real sentence while inventing the figure inside it is
            # the subtlest failure mode here, and the one worth spending a check on.
            logger.warning(
                "result_value_not_in_quote",
                paper_id=paper_id,
                metric=metric,
                value=value,
                quote=quote[:160],
            )
            return None

        name = f"{metric}"
        if draft.dataset_name.strip():
            name = f"{metric} on {' '.join(draft.dataset_name.split())}"

        return build_object(
            kind=self.kind,
            paper_id=paper_id,
            index=index,
            name=name,
            description=" ".join(draft.description.split()),
            details=ResultDetails(
                metric_name=metric,
                dataset_name=" ".join(draft.dataset_name.split()) or None,
                value_text=value,
                numeric_value=_first_number(value),
                unit=draft.unit.strip() or None,
                baseline_comparison=" ".join(draft.baseline_comparison.split()) or None,
            ),
            evidence=evidence,
            extracted_by=self.name,
            grounding_note=(
                f"value {value!r} appears in the quoted sentence at "
                f"{evidence[0].location.describe()}"
            ),
        )


def _value_appears_in(value: str, quote: str) -> bool:
    """Whether every number in the claimed value is present in the source sentence."""
    numbers = _NUMBER.findall(value)
    if not numbers:
        # A qualitative value ("outperforms the baseline") is checked as plain text.
        return normalise(value) in normalise(quote)

    haystack = normalise(quote).replace(",", "")
    return all(number.replace(",", "") in haystack for number in numbers)


def _first_number(value: str) -> float | None:
    match = _NUMBER.search(value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None
