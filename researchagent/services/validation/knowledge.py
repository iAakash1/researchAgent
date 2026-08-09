"""Knowledge validators.

Grounding proved a quote exists. These validators ask the next questions: does the
evidence actually support *this* object, is the object internally coherent, and do the
relations between objects hold up?

No extractor trusts another extractor, and nothing trusts the model. Every validator
explains itself — a rejection names the code, the reason and the remedy, and a pass names
the observations behind its confidence.
"""

from __future__ import annotations

from researchagent.config.schemas import KnowledgeValidationConfig
from researchagent.core.evidence import EvidenceKind
from researchagent.core.interfaces.validator import Validator
from researchagent.core.logging import get_logger
from researchagent.core.validation import (
    Confidence,
    ConfidenceSignal,
    ValidationIssue,
    ValidationResult,
)
from researchagent.models.knowledge import (
    KnowledgeKind,
    KnowledgeObject,
    KnowledgeRelation,
    PaperKnowledge,
    ResultDetails,
)
from researchagent.services.knowledge.grounding import normalise

logger = get_logger(__name__)

_SUBJECT_OBJECT = "KnowledgeObject"
_SUBJECT_KNOWLEDGE = "PaperKnowledge"
_SUBJECT_RELATION = "KnowledgeRelation"


class EvidenceValidator(Validator[KnowledgeObject]):
    """Does the evidence attached to this object actually mention it?

    The check grounding cannot make: a quote may be genuinely present in the paper and
    still have nothing to do with the object it was attached to. For objects whose `name`
    is a named entity — a dataset, a metric, a method — the name should appear in the
    supporting sentence.
    """

    name = "evidence_validator"
    subject_type = _SUBJECT_OBJECT

    def __init__(self, config: KnowledgeValidationConfig | None = None) -> None:
        self._config = config or KnowledgeValidationConfig()

    def check(self, subject: KnowledgeObject) -> ValidationResult:
        issues: list[ValidationIssue] = []
        signals: list[ConfidenceSignal] = []

        textual = [item for item in subject.evidence if item.kind is EvidenceKind.EXTRACTED_TEXT]
        if not textual:
            issues.append(
                ValidationIssue.error(
                    "evidence_not_textual",
                    "No verbatim quote supports this object",
                    field="evidence",
                    remedy="Re-extract; only quoted text can support a knowledge claim",
                )
            )

        located = [item for item in subject.evidence if item.location.paragraph_index is not None]
        signals.append(
            ConfidenceSignal(
                name="evidence_precision",
                value=len(located) / len(subject.evidence),
                observation=(
                    f"{len(located)} of {len(subject.evidence)} evidence items resolve to a "
                    "specific paragraph"
                ),
            )
        )

        # Named entities should appear in their own supporting sentence.
        if not subject.kind.is_claim_like:
            mentions = self._name_mentioned(subject)
            if not mentions and self._config.require_name_in_quote:
                issues.append(
                    ValidationIssue.warning(
                        "name_absent_from_evidence",
                        f"{subject.name!r} does not appear in its supporting quote",
                        field="name",
                        remedy="The quote may support a different object than the one named",
                    )
                )
            signals.append(
                ConfidenceSignal(
                    name="name_in_evidence",
                    value=1.0 if mentions else 0.0,
                    observation=(
                        f"the name {subject.name!r} "
                        f"{'appears in' if mentions else 'is absent from'} the supporting quote"
                    ),
                )
            )

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
            evidence=list(subject.evidence),
        )

    @staticmethod
    def _name_mentioned(subject: KnowledgeObject) -> bool:
        name = normalise(subject.name)
        if not name:
            return False
        # Compare on significant tokens: "MIMIC-III dataset" should match a quote saying
        # "MIMIC-III", without demanding the exact phrasing.
        tokens = [token for token in name.split() if len(token) > 2]
        if not tokens:
            tokens = name.split()

        for quote in subject.quotes:
            haystack = normalise(quote)
            if any(token in haystack for token in tokens):
                return True
        return False


class ResultValidator(Validator[KnowledgeObject]):
    """Is a reported number coherent and actually present in its quote?

    Applied on top of the extractor's own check, because a result that reaches the
    knowledge layer with an unsupported figure is the single most damaging thing the
    system could emit.
    """

    name = "result_validator"
    subject_type = _SUBJECT_OBJECT

    def check(self, subject: KnowledgeObject) -> ValidationResult:
        issues: list[ValidationIssue] = []
        signals: list[ConfidenceSignal] = []

        if not isinstance(subject.details, ResultDetails):
            return _not_applicable(self.name, subject)

        details = subject.details
        if not details.metric_name:
            issues.append(
                ValidationIssue.error(
                    "result_without_metric",
                    "A reported value with no metric cannot be interpreted",
                    field="metric_name",
                )
            )

        value_text = details.value_text or ""
        in_quote = any(normalise(value_text) in normalise(q) for q in subject.quotes) or any(
            _digits(value_text) and _digits(value_text) in _digits(q) for q in subject.quotes
        )
        if value_text and not in_quote:
            issues.append(
                ValidationIssue.error(
                    "result_value_unsupported",
                    f"Value {value_text!r} does not appear in the supporting quote",
                    field="value_text",
                    remedy="Discard; the figure was not read from the paper",
                )
            )

        signals.append(
            ConfidenceSignal(
                name="value_in_quote",
                value=1.0 if in_quote else 0.0,
                observation=(
                    f"value {value_text!r} "
                    f"{'found in' if in_quote else 'missing from'} the supporting quote"
                ),
            )
        )
        signals.append(
            ConfidenceSignal(
                name="result_context",
                value=_fraction_present(details.dataset_name, details.unit),
                observation=(
                    f"dataset={'yes' if details.dataset_name else 'no'}, "
                    f"unit={'yes' if details.unit else 'no'}"
                ),
            )
        )

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )


class CompletenessValidator(Validator[KnowledgeObject]):
    """Is the object substantial enough to reason over?

    Rejects the degenerate extractions — a dataset called "dataset", a limitation of four
    words — that pass every other check while carrying no information.
    """

    name = "completeness_validator"
    subject_type = _SUBJECT_OBJECT

    _GENERIC_NAMES = frozenset(
        {
            "dataset",
            "datasets",
            "method",
            "methods",
            "metric",
            "metrics",
            "model",
            "result",
            "results",
            "data",
            "approach",
            "technique",
            "n/a",
            "none",
            "unknown",
        }
    )

    def __init__(self, config: KnowledgeValidationConfig | None = None) -> None:
        self._config = config or KnowledgeValidationConfig()

    def check(self, subject: KnowledgeObject) -> ValidationResult:
        issues: list[ValidationIssue] = []
        name = subject.name.strip()

        if normalise(name) in self._GENERIC_NAMES:
            issues.append(
                ValidationIssue.error(
                    "name_too_generic",
                    f"{name!r} names no specific entity",
                    field="name",
                    remedy="Extractor returned a placeholder rather than a real entity",
                )
            )
        if len(name) < self._config.min_name_chars:
            issues.append(
                ValidationIssue.error(
                    "name_too_short",
                    f"Name {name!r} is too short to identify anything",
                    field="name",
                )
            )
        if subject.kind.is_claim_like and len(subject.description) < self._config.min_claim_chars:
            issues.append(
                ValidationIssue.warning(
                    "claim_underspecified",
                    "Claim-like knowledge carries almost no description",
                    field="description",
                )
            )

        signals = [
            ConfidenceSignal(
                name="specificity",
                value=min(len(name) / 24, 1.0),
                observation=f"name {name!r} is {len(name)} characters",
            ),
            ConfidenceSignal(
                name="description_present",
                value=1.0 if subject.description.strip() else 0.0,
                observation=(
                    "description present" if subject.description.strip() else "no description"
                ),
            ),
        ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )


class RelationshipValidator(Validator[PaperKnowledge]):
    """Do the relations connect objects that exist, with the right types?

    A dangling or mistyped edge is worse than a missing one: v0.7 builds a graph from
    these, and a graph that lies is harder to detect than a graph that is sparse.
    """

    name = "relationship_validator"
    subject_type = _SUBJECT_KNOWLEDGE

    def check(self, subject: PaperKnowledge) -> ValidationResult:
        issues: list[ValidationIssue] = []
        by_id = {item.id: item for item in subject.objects}
        well_typed = 0

        for relation in subject.relations:
            problem = _relation_problem(relation, by_id)
            if problem is None:
                well_typed += 1
                continue
            issues.append(problem)

        signals = [
            ConfidenceSignal(
                name="relation_integrity",
                value=(well_typed / len(subject.relations)) if subject.relations else 1.0,
                observation=(
                    f"{well_typed} of {len(subject.relations)} relations connect existing "
                    "objects with matching kinds"
                    if subject.relations
                    else "no relations were proposed"
                ),
            )
        ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.paper_id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )


class KnowledgeCoverageValidator(Validator[PaperKnowledge]):
    """Did extraction produce a usable picture of the paper?

    Not every paper reports datasets or numbers, so absence is a warning rather than an
    error — but a paper that yielded nothing at all is a signal that upstream parsing or
    the prompts failed, and the reviewer must see it.
    """

    name = "knowledge_coverage_validator"
    subject_type = _SUBJECT_KNOWLEDGE

    def check(self, subject: PaperKnowledge) -> ValidationResult:
        issues: list[ValidationIssue] = []

        if not subject.objects:
            issues.append(
                ValidationIssue.error(
                    "no_knowledge_extracted",
                    "No grounded knowledge could be extracted from this document",
                    remedy="Check section detection and whether the document is a research paper",
                )
            )

        kinds = set(subject.kinds_present)
        signals = [
            ConfidenceSignal(
                name="kind_coverage",
                value=len(kinds) / len(KnowledgeKind),
                observation=(
                    f"{len(kinds)} of {len(KnowledgeKind)} knowledge kinds present: "
                    f"{sorted(k.value for k in kinds)}"
                ),
            ),
            ConfidenceSignal(
                name="evidence_density",
                value=min(subject.evidence_count / max(len(subject.objects) or 1, 1), 1.0),
                observation=(
                    f"{subject.evidence_count} evidence items across {len(subject.objects)} objects"
                ),
            ),
        ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.paper_id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )


def _relation_problem(
    relation: KnowledgeRelation, by_id: dict[str, KnowledgeObject]
) -> ValidationIssue | None:
    subject_object = by_id.get(relation.subject_id)
    object_object = by_id.get(relation.object_id)

    if subject_object is None or object_object is None:
        return ValidationIssue.error(
            "relation_dangling",
            f"{relation.predicate.value} references an object that does not exist",
            field="relations",
            remedy="Drop the relation; its endpoints did not survive validation",
        )
    if subject_object.kind is not relation.predicate.domain:
        return ValidationIssue.error(
            "relation_domain_mismatch",
            f"{relation.predicate.value} expects a {relation.predicate.domain.value} subject, "
            f"got {subject_object.kind.value}",
            field="relations",
        )
    if object_object.kind is not relation.predicate.range:
        return ValidationIssue.error(
            "relation_range_mismatch",
            f"{relation.predicate.value} expects a {relation.predicate.range.value} object, "
            f"got {object_object.kind.value}",
            field="relations",
        )
    return None


def _not_applicable(validator: str, subject: KnowledgeObject) -> ValidationResult:
    """A validator asked about a kind it does not judge abstains rather than passing."""
    return ValidationResult.passed(
        validator=validator,
        subject_id=subject.id,
        subject_type=_SUBJECT_OBJECT,
        confidence=Confidence.unknown(),
    )


def _digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit() or character == ".")


def _fraction_present(*values: str | None) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0
