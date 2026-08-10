"""Audit trail construction.

Reconstructs how a conclusion came to exist:

    goal -> plan -> question -> retrieval decision -> bundles -> reasoning
         -> finding -> verification -> review -> final status

This is a product feature rather than a debugging aid. A research system whose output
cannot be traced back to a page in a PDF is asking to be believed; one that can be traced
is asking to be checked, which is the only thing that makes it useful.

Built from the session and the repositories — never from logs, so an audit stays available
after the run and survives a process restart.
"""

from __future__ import annotations

from researchagent.core.logging import get_logger
from researchagent.models.reasoning import ResearchFinding
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.repositories.evidence_repository import JsonEvidenceRepository
from researchagent.schemas.reasoning import AuditStep, FindingAudit
from researchagent.schemas.workflow import ResearchState

logger = get_logger(__name__)


class AuditTrailBuilder:
    """Turns a finished session into per-finding provenance chains."""

    name = "audit_trail_builder"

    def __init__(self, bundles: JsonBundleRepository, evidence: JsonEvidenceRepository) -> None:
        self._bundles = bundles
        self._evidence = evidence

    async def build(self, state: ResearchState) -> tuple[FindingAudit, ...]:
        session = state.reasoning
        if session is None:
            return ()
        return tuple([await self.for_finding(state, finding) for finding in session.findings])

    async def for_finding(self, state: ResearchState, finding: ResearchFinding) -> FindingAudit:
        session = state.reasoning
        assert session is not None  # noqa: S101 - callers hold a session

        steps: list[AuditStep] = [
            AuditStep(
                stage="goal",
                actor="user",
                summary=state.goal,
            )
        ]
        if state.plan is not None:
            steps.append(
                AuditStep(
                    stage="plan",
                    actor="planner",
                    summary=state.plan.framing[:200],
                    references=tuple(q.id for q in state.plan.research_questions),
                )
            )
            question = next(
                (q for q in state.plan.research_questions if q.id == finding.question_id), None
            )
            if question is not None:
                steps.append(
                    AuditStep(
                        stage="question",
                        actor="planner",
                        summary=question.question,
                        references=(question.id,),
                    )
                )

        question_state = next(
            (q for q in session.questions if q.question_id == finding.question_id), None
        )
        if question_state is not None:
            steps.append(
                AuditStep(
                    stage="retrieval",
                    actor="retrieval",
                    summary=(
                        f"{question_state.retrieval_attempts} retrieval attempt(s) produced "
                        f"{len(question_state.bundle_ids)} bundle(s)"
                    ),
                    references=question_state.bundle_ids,
                )
            )

        steps.append(
            AuditStep(
                stage="reasoning",
                actor=finding.produced_by,
                summary=finding.statement,
                iteration=finding.iteration,
                references=finding.bundle_ids,
            )
        )

        verification = session.verification_for(finding.id)
        if verification is not None:
            steps.append(
                AuditStep(
                    stage="verification",
                    actor=verification.verified_by,
                    summary=f"{verification.verdict.value}: {verification.reasoning[:160]}",
                    iteration=verification.iteration,
                    references=tuple(citation.bundle_id for citation in verification.supporting),
                )
            )

        review = session.latest_review
        if review is not None:
            outcome = "accepted" if finding.id in review.accepted_findings else "rejected"
            steps.append(
                AuditStep(
                    stage="review",
                    actor=review.reviewed_by,
                    summary=f"{review.decision.value}; this finding {outcome}",
                    iteration=review.iteration,
                    references=tuple(
                        issue.code for issue in review.issues if issue.finding_id == finding.id
                    ),
                )
            )

        provenance = await self._provenance(finding)
        audit = FindingAudit(
            finding_id=finding.id,
            question_id=finding.question_id,
            statement=finding.statement,
            status=finding.status,
            steps=tuple(steps),
            citations=finding.citations,
            provenance=provenance,
            verification=verification,
        )
        if not audit.is_complete:
            # A finding whose chain does not reach a source location is exactly what the
            # reviewer's citation validator rejects; surfacing it here too means an
            # incomplete audit is visible without re-running the review.
            logger.warning(
                "incomplete_audit_trail",
                finding=finding.id,
                citations=len(finding.citations),
                provenance=len(provenance),
            )
        return audit

    async def _provenance(self, finding: ResearchFinding) -> tuple[str, ...]:
        """Resolve every cited evidence id down to its page and paragraph."""
        addresses: list[str] = []
        for citation in finding.citations:
            for evidence_id in citation.evidence_ids:
                record = await self._evidence.get(evidence_id)
                if record is not None:
                    addresses.append(record.evidence.location.describe())
        return tuple(dict.fromkeys(addresses))
