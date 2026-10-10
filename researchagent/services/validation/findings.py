"""Deterministic checks on research findings.

Run *before* any model is asked for an opinion, and able to reject on their own. The
reviewer's model call is one signal among these, not the gate — a language model asked
"is this good research?" will usually say yes, and a quality gate that can be talked into
approving is not a gate.

Each validator answers one question and returns a ``ValidationResult`` rather than raising:
an unsupported finding is a finding to reject, not an exception to handle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from researchagent.core.validation import (
    Confidence,
    ConfidenceSignal,
    ValidationIssue,
    ValidationResult,
)
from researchagent.models.integrity import IntegrityObservation, IntegrityStatus
from researchagent.models.reasoning import (
    FindingStatus,
    ResearchFinding,
    VerificationResult,
    VerificationVerdict,
)


class CitationValidator:
    """Every finding must carry citations that resolve to real provenance."""

    name = "citation_validator"

    def __init__(self, resolved_evidence_ids: frozenset[str]) -> None:
        self._resolved = resolved_evidence_ids

    def validate(self, finding: ResearchFinding) -> ValidationResult:
        cited = [
            evidence_id for citation in finding.citations for evidence_id in citation.evidence_ids
        ]
        unresolved = [item for item in cited if item not in self._resolved]
        completeness = (len(cited) - len(unresolved)) / len(cited) if cited else 0.0

        issues: list[ValidationIssue] = []
        if not cited:
            issues.append(
                ValidationIssue.error(
                    "no_citations", "finding carries no citations", field="citations"
                )
            )
        if unresolved:
            issues.append(
                ValidationIssue.error(
                    "unresolved_citations",
                    f"{len(unresolved)} cited evidence id(s) do not resolve to a source",
                    field="citations",
                )
            )

        return ValidationResult(
            validator=self.name,
            subject_id=finding.id,
            subject_type="ResearchFinding",
            success=not issues,
            issues=tuple(issues),
            confidence=Confidence(
                score=round(completeness, 4),
                signals=(
                    ConfidenceSignal(
                        name="citation_completeness",
                        value=round(completeness, 4),
                        observation=(
                            f"{len(cited) - len(unresolved)} of {len(cited)} cited ids "
                            "resolve to stored evidence"
                        ),
                    ),
                ),
            ),
        )


class SourceDiversityValidator:
    """A conclusion drawn from one paper is a summary, not a research finding.

    Not fatal on its own — some questions genuinely have one relevant paper in a corpus
    this size — but recorded as a warning so the reviewer can weigh it.
    """

    name = "source_diversity_validator"

    def __init__(self, minimum_papers: int = 2) -> None:
        self._minimum = minimum_papers

    def validate(self, finding: ResearchFinding) -> ValidationResult:
        papers = finding.paper_ids
        diversity = min(len(papers) / self._minimum, 1.0) if self._minimum else 1.0
        issues: list[ValidationIssue] = []
        if len(papers) < self._minimum:
            issues.append(
                ValidationIssue.warning(
                    "single_source",
                    f"supported by {len(papers)} paper(s); {self._minimum} expected",
                    field="citations",
                )
            )
        return ValidationResult(
            validator=self.name,
            subject_id=finding.id,
            subject_type="ResearchFinding",
            success=True,
            issues=tuple(issues),
            confidence=Confidence(
                score=round(diversity, 4),
                signals=(
                    ConfidenceSignal(
                        name="source_diversity",
                        value=round(diversity, 4),
                        observation=f"cited papers: {sorted(papers) or 'none'}",
                    ),
                ),
            ),
        )


class VerificationRequiredValidator:
    """Nothing becomes a result without having been checked.

    This is the rule that stops the reasoning agent's confidence from being the last word
    on its own output.
    """

    name = "verification_required_validator"

    def validate(
        self, finding: ResearchFinding, verification: VerificationResult | None
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if verification is None:
            issues.append(
                ValidationIssue.error(
                    "not_verified", "no verification result for this finding", field="status"
                )
            )
        elif not verification.verdict.accepts:
            issues.append(
                ValidationIssue.error(
                    f"verdict_{verification.verdict.value}",
                    f"verification returned {verification.verdict.value}",
                    field="status",
                )
            )
        if verification is not None and verification.overstatements:
            issues.append(
                ValidationIssue.error(
                    "overclaiming",
                    f"verifier named {len(verification.overstatements)} overstatement(s)",
                    field="statement",
                )
            )

        accepted = verification is not None and verification.verdict.accepts
        return ValidationResult(
            validator=self.name,
            subject_id=finding.id,
            subject_type="ResearchFinding",
            success=accepted and not (verification.overstatements if verification else ()),
            issues=tuple(issues),
            confidence=Confidence(
                score=1.0 if accepted else 0.0,
                signals=(
                    ConfidenceSignal(
                        name="verification_verdict",
                        value=1.0 if accepted else 0.0,
                        observation=(
                            f"verdict={verification.verdict.value}"
                            if verification
                            else "no verification was run"
                        ),
                    ),
                ),
            ),
        )


class ContradictionValidator:
    """A finding the corpus contradicts cannot be accepted, whatever else says otherwise."""

    name = "contradiction_validator"

    def validate(
        self, finding: ResearchFinding, verification: VerificationResult | None
    ) -> ValidationResult:
        contradicted = bool(finding.contradicting) or (
            verification is not None and verification.verdict is VerificationVerdict.CONTRADICTED
        )
        issues: list[ValidationIssue] = []
        if contradicted:
            issues.append(
                ValidationIssue.error(
                    "contradicted",
                    "the corpus contains evidence against this finding",
                    field="contradicting",
                )
            )
        return ValidationResult(
            validator=self.name,
            subject_id=finding.id,
            subject_type="ResearchFinding",
            success=not contradicted,
            issues=tuple(issues),
            confidence=Confidence(
                score=0.0 if contradicted else 1.0,
                signals=(
                    ConfidenceSignal(
                        name="contradiction_free",
                        value=0.0 if contradicted else 1.0,
                        observation=(
                            f"{len(finding.contradicting)} contradicting citation(s) on the finding"
                        ),
                    ),
                ),
            ),
        )


class RetractedSupportValidator:
    """A retracted or withdrawn source may not support a finding, disclosed or not.

    Status comes from integrity observations (registry records), never from a model. A
    finding may still *mention* such a source — among its contradicting citations — but
    only with an explicit :class:`IntegrityDisclosure` for that paper. With no
    observations supplied the check passes: an unchecked source is not a retracted one,
    and the maintenance workflow is what supplies the checks.
    """

    name = "retracted_support_validator"

    def __init__(
        self,
        observations: Mapping[str, IntegrityObservation],
        excluded_statuses: Iterable[IntegrityStatus] = (
            IntegrityStatus.RETRACTED,
            IntegrityStatus.WITHDRAWN,
        ),
    ) -> None:
        self._observations = observations
        self._excluded = frozenset(excluded_statuses)

    def validate(self, finding: ResearchFinding) -> ValidationResult:
        issues: list[ValidationIssue] = []
        supporting = finding.paper_ids
        mentioned = tuple(
            dict.fromkeys(
                pid
                for citation in finding.contradicting
                for pid in citation.paper_ids
                if pid not in supporting
            )
        )
        disclosed = {item.paper_id for item in finding.integrity_disclosures}

        for paper_id in supporting:
            observation = self._flagged(paper_id)
            if observation is not None:
                issues.append(
                    ValidationIssue.error(
                        "retracted_support",
                        f"{paper_id} is {observation.status.value} "
                        f"({_notice(observation)}) and cannot support a finding",
                        field="citations",
                        remedy="Drop the evidence from this source and re-verify the finding",
                    )
                )
        for paper_id in mentioned:
            observation = self._flagged(paper_id)
            if observation is None:
                continue
            if paper_id in disclosed:
                issues.append(
                    ValidationIssue.info(
                        "disclosed_retracted_mention",
                        f"{paper_id} is {observation.status.value}; the finding discloses it",
                        field="integrity_disclosures",
                    )
                )
            else:
                issues.append(
                    ValidationIssue.error(
                        "undisclosed_retracted_mention",
                        f"{paper_id} is {observation.status.value} ({_notice(observation)}) "
                        "and is mentioned without a disclosure",
                        field="integrity_disclosures",
                        remedy="Add an integrity disclosure for this source or remove it",
                    )
                )

        checked = sum(1 for pid in (*supporting, *mentioned) if pid in self._observations)
        return ValidationResult.decide(
            validator=self.name,
            subject_id=finding.id,
            subject_type="ResearchFinding",
            issues=issues,
            confidence=Confidence.from_signals(
                [
                    ConfidenceSignal(
                        name="integrity_checked",
                        value=checked / len(supporting + mentioned)
                        if supporting or mentioned
                        else 0.0,
                        observation=(
                            f"{checked} of {len(supporting) + len(mentioned)} cited paper(s) "
                            "have a registry integrity observation"
                        ),
                    )
                ]
            ),
        )

    def _flagged(self, paper_id: str) -> IntegrityObservation | None:
        observation = self._observations.get(paper_id)
        if observation is None or observation.status not in self._excluded:
            return None
        return observation


def _notice(observation: IntegrityObservation) -> str:
    parts = [observation.registry]
    if observation.notice_doi:
        parts.append(f"notice {observation.notice_doi}")
    if observation.notice_date:
        parts.append(observation.notice_date)
    return ", ".join(parts)


def status_after_review(
    finding: ResearchFinding, verification: VerificationResult | None, accepted: bool
) -> FindingStatus:
    """The single place a finding's final status is decided."""
    if accepted:
        return FindingStatus.VERIFIED
    if verification is None:
        return FindingStatus.SUPPORTED
    if verification.verdict is VerificationVerdict.CONTRADICTED:
        return FindingStatus.REJECTED
    if verification.verdict in (
        VerificationVerdict.INSUFFICIENT_EVIDENCE,
        VerificationVerdict.UNVERIFIABLE,
    ):
        return FindingStatus.INSUFFICIENT_EVIDENCE
    return FindingStatus.SUPPORTED
