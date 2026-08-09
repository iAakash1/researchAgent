"""Contradiction detection.

Deterministic and mechanical. No model is asked whether two findings disagree, because a
model asked that question will produce a confident answer either way and there is nothing
to check it against.

Only disagreements that can be *verified from the extracted fields* are detected:

* two papers reporting materially different numbers for the same metric and dataset
* two papers asserting opposite values for the same boolean attribute

That is a narrow net on purpose. A missed contradiction is a gap; a fabricated one is a
false accusation against a paper, which is worse.

Detected disagreements are never resolved. Both sides are carried with their evidence, so
the reasoning engine and the reviewer see the conflict rather than a manufactured
consensus.
"""

from __future__ import annotations

from researchagent.config.schemas import ContradictionConfig
from researchagent.core.logging import get_logger
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.bundle import Contradiction, ContradictionKind
from researchagent.models.knowledge import (
    DatasetDetails,
    KnowledgeKind,
    KnowledgeObject,
    MethodDetails,
    ResultDetails,
)
from researchagent.utils.text import normalise

logger = get_logger(__name__)


class ContradictionDetector:
    """Finds mechanically checkable disagreements between knowledge objects."""

    name = "contradiction_detector"

    def __init__(self, config: ContradictionConfig | None = None) -> None:
        self._config = config or ContradictionConfig()

    def detect(self, objects: tuple[KnowledgeObject, ...]) -> tuple[Contradiction, ...]:
        found = [
            *self._value_conflicts(objects),
            *self._attribute_conflicts(objects),
        ]
        if found:
            logger.info(
                "contradictions_detected",
                count=len(found),
                cross_paper=sum(1 for item in found if item.is_cross_paper),
            )
        return tuple(found)

    def _value_conflicts(self, objects: tuple[KnowledgeObject, ...]) -> list[Contradiction]:
        """Same metric, same dataset, materially different numbers."""
        results = [
            item
            for item in objects
            if item.kind is KnowledgeKind.RESULT and isinstance(item.details, ResultDetails)
        ]

        grouped: dict[tuple[str, str], list[KnowledgeObject]] = {}
        for item in results:
            details = item.details
            assert isinstance(details, ResultDetails)  # noqa: S101 - filtered above
            if details.numeric_value is None or not details.metric_name:
                continue
            key = (normalise(details.metric_name), normalise(details.dataset_name or ""))
            grouped.setdefault(key, []).append(item)

        conflicts: list[Contradiction] = []
        for (metric, dataset), group in grouped.items():
            for index, left in enumerate(group):
                for right in group[index + 1 :]:
                    conflict = self._compare_values(left, right, metric, dataset)
                    if conflict is not None:
                        conflicts.append(conflict)
        return conflicts

    def _compare_values(
        self, left: KnowledgeObject, right: KnowledgeObject, metric: str, dataset: str
    ) -> Contradiction | None:
        left_details, right_details = left.details, right.details
        assert isinstance(left_details, ResultDetails)  # noqa: S101
        assert isinstance(right_details, ResultDetails)  # noqa: S101

        left_value = left_details.numeric_value
        right_value = right_details.numeric_value
        if left_value is None or right_value is None:
            return None

        largest = max(abs(left_value), abs(right_value), 1e-9)
        difference = abs(left_value - right_value) / largest
        if difference < self._config.numeric_tolerance:
            return None

        return Contradiction(
            id=f"{left.id}--vs--{right.id}",
            kind=ContradictionKind.VALUE_CONFLICT,
            description=(
                f"{metric or 'metric'} on {dataset or 'an unnamed dataset'} is reported as "
                f"{left_details.value_text} and as {right_details.value_text} "
                f"({difference:.0%} apart)"
            ),
            left_object_id=left.id,
            right_object_id=right.id,
            left_paper_id=left.paper_id,
            right_paper_id=right.paper_id,
            left_evidence=left.evidence,
            right_evidence=right.evidence,
            detected_by=self.name,
            confidence=Confidence.from_signals(
                [
                    ConfidenceSignal(
                        name="numeric_divergence",
                        value=min(difference, 1.0),
                        observation=(
                            f"{left_value} and {right_value} differ by {difference:.0%}, "
                            f"beyond the {self._config.numeric_tolerance:.0%} tolerance"
                        ),
                    ),
                    ConfidenceSignal(
                        name="independent_sources",
                        value=1.0 if left.paper_id != right.paper_id else 0.3,
                        observation=(
                            "reported by different papers"
                            if left.paper_id != right.paper_id
                            else "reported twice within one paper"
                        ),
                    ),
                ]
            ),
        )

    def _attribute_conflicts(self, objects: tuple[KnowledgeObject, ...]) -> list[Contradiction]:
        """Opposite boolean claims about the same named entity."""
        conflicts: list[Contradiction] = []

        for kind, attribute in (
            (KnowledgeKind.DATASET, "is_public"),
            (KnowledgeKind.METHOD, "is_novel"),
        ):
            by_name: dict[str, list[KnowledgeObject]] = {}
            for item in objects:
                if item.kind is not kind:
                    continue
                if not isinstance(item.details, DatasetDetails | MethodDetails):
                    continue
                if getattr(item.details, attribute, None) is None:
                    continue
                by_name.setdefault(normalise(item.name), []).append(item)

            for name, group in by_name.items():
                for index, left in enumerate(group):
                    for right in group[index + 1 :]:
                        left_value = getattr(left.details, attribute)
                        right_value = getattr(right.details, attribute)
                        if left_value == right_value:
                            continue
                        # Within one paper this is usually an extraction slip, not a real
                        # disagreement; only cross-paper conflicts are reported.
                        if left.paper_id == right.paper_id:
                            continue
                        conflicts.append(
                            Contradiction(
                                id=f"{left.id}--attr--{right.id}",
                                kind=ContradictionKind.ATTRIBUTE_CONFLICT,
                                description=(
                                    f"{name!r}: {attribute} is stated as {left_value} in "
                                    f"{left.paper_id} and {right_value} in {right.paper_id}"
                                ),
                                left_object_id=left.id,
                                right_object_id=right.id,
                                left_paper_id=left.paper_id,
                                right_paper_id=right.paper_id,
                                left_evidence=left.evidence,
                                right_evidence=right.evidence,
                                detected_by=self.name,
                                confidence=Confidence.from_signals(
                                    [
                                        ConfidenceSignal(
                                            name="attribute_disagreement",
                                            value=1.0,
                                            observation=(
                                                f"{attribute} differs between "
                                                f"{left.paper_id} and {right.paper_id}"
                                            ),
                                        )
                                    ]
                                ),
                            )
                        )
        return conflicts
