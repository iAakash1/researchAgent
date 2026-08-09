"""Bundle validators.

A bundle is the last thing checked before context reaches a reasoning engine, so these
validators ask the questions that matter at that boundary: is every claim traceable, and
is the support broad enough to justify a conclusion?

Both expose ``check_*`` methods taking the bundle's parts rather than a finished bundle,
because the builder needs the verdict *before* it can construct one — an EvidenceBundle
requires its validation at construction.
"""

from __future__ import annotations

from researchagent.config.schemas import BundleSettings
from researchagent.core.logging import get_logger
from researchagent.core.validation import (
    Confidence,
    ConfidenceSignal,
    ValidationIssue,
    ValidationResult,
)
from researchagent.models.bundle import BundleCoverage, BundledEvidence
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.query import ResearchQuery

logger = get_logger(__name__)

_SUBJECT = "EvidenceBundle"


class ProvenanceValidator:
    """Is every fact in this bundle traceable back to a page?

    The guarantee the whole architecture exists to provide, checked one last time at the
    point where context leaves the system. A knowledge object without retrievable
    evidence is an assertion, and assertions must not reach a reasoning engine dressed as
    evidence.
    """

    name = "provenance_validator"

    def check_bundle(
        self,
        query: ResearchQuery,
        objects: tuple[KnowledgeObject, ...],
        evidence: tuple[BundledEvidence, ...],
        confidence: Confidence,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []

        located = [item for item in evidence if item.location.page is not None]
        unsupported = [
            item.id
            for item in objects
            if not any(entry.knowledge_object_id == item.id for entry in evidence)
        ]

        if objects and not evidence:
            issues.append(
                ValidationIssue.error(
                    "bundle_without_evidence",
                    f"{len(objects)} knowledge objects carry no retrievable evidence",
                    field="evidence",
                    remedy="Build the evidence index before bundling",
                )
            )
        elif unsupported:
            issues.append(
                ValidationIssue.warning(
                    "objects_without_evidence",
                    f"{len(unsupported)} object(s) have no evidence in this bundle",
                    field="evidence",
                    remedy="These facts cannot be cited; the reasoner must not rely on them",
                )
            )

        if evidence and len(located) < len(evidence):
            issues.append(
                ValidationIssue.warning(
                    "imprecise_provenance",
                    f"{len(evidence) - len(located)} evidence items resolve no further than "
                    "the document",
                    field="evidence",
                )
            )

        signals = [
            ConfidenceSignal(
                name="traceability",
                value=(len(located) / len(evidence)) if evidence else 0.0,
                observation=(
                    f"{len(located)} of {len(evidence)} evidence items resolve to a page"
                    if evidence
                    else "the bundle carries no evidence"
                ),
            )
        ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=query.question_id or query.text[:60],
            subject_type=_SUBJECT,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )


class BundleCoverageValidator:
    """Is the support broad enough to conclude anything from?

    Thin coverage is a warning rather than an error: one paper is a legitimate answer to
    a narrow question. What matters is that the thinness is visible, so the reviewer can
    say "this conclusion rests on a single paper" instead of discovering it later.
    """

    name = "bundle_coverage_validator"

    def __init__(self, settings: BundleSettings | None = None) -> None:
        self._settings = settings or BundleSettings()

    def check_coverage(
        self, query: ResearchQuery, coverage: BundleCoverage, confidence: Confidence
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []

        if coverage.paper_count == 0:
            issues.append(
                ValidationIssue.error(
                    "no_supporting_papers",
                    "No paper contributed evidence for this query",
                    field="coverage",
                    remedy="Widen the query, or discover and parse more literature",
                )
            )
        elif coverage.paper_count < self._settings.min_papers_for_confidence:
            issues.append(
                ValidationIssue.warning(
                    "single_source_coverage",
                    f"Only {coverage.paper_count} paper(s) support this bundle",
                    field="coverage",
                    remedy="Treat conclusions as provisional until corroborated",
                )
            )

        signals = [
            ConfidenceSignal(
                name="paper_breadth",
                value=min(coverage.paper_count / self._settings.corroboration_target, 1.0),
                observation=(
                    f"{coverage.paper_count} of {coverage.papers_considered} candidate papers "
                    "contributed"
                ),
            ),
            ConfidenceSignal(
                name="kind_breadth",
                value=min(len(coverage.kinds_covered) / 3, 1.0),
                observation=(
                    f"knowledge kinds present: "
                    f"{sorted(kind.value for kind in coverage.kinds_covered)}"
                ),
            ),
        ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=query.question_id or query.text[:60],
            subject_type=_SUBJECT,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )
