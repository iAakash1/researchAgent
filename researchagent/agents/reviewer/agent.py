"""Reviewer agent: the final gate.

Deliberately not an LLM approve/reject call. The order is:

    deterministic validators -> per-finding accept/reject -> model critique -> decision

The validators can reject on their own and the model cannot overturn them. The model can
only *add* concerns — it sees the checks that already ran and is asked for the part
arithmetic misses: overclaiming, findings that quietly contradict each other,
generalisation from a single condition.

Asking a language model "is this good research?" reliably produces "yes". A gate that can
be talked into approving is not a gate, so the model's opinion is one signal among four
measured ones and never the deciding vote.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.registry import AGENTS
from researchagent.agents.reviewer.prompt import ReviewerPrompt
from researchagent.agents.reviewer.schemas import CritiqueDraft, ReviewerInput, ReviewerOutput
from researchagent.core.exceptions import ResearchAgentError
from researchagent.models.reasoning import (
    ResearchFinding,
    ReviewDecision,
    ReviewIssue,
    ReviewResult,
    VerificationResult,
)
from researchagent.services.validation.findings import (
    CitationValidator,
    ContradictionValidator,
    SourceDiversityValidator,
    VerificationRequiredValidator,
)


@AGENTS.register("reviewer")
class ReviewerAgent(BaseAgent[ReviewerInput, ReviewerOutput]):
    name: ClassVar[str] = "reviewer"
    description: ClassVar[str] = "Final quality gate over a set of research findings"
    input_schema: ClassVar[type[BaseModel]] = ReviewerInput
    output_schema: ClassVar[type[BaseModel]] = ReviewerOutput

    async def execute(self, payload: ReviewerInput, context: AgentContext) -> ReviewerOutput:
        citation = CitationValidator(payload.resolved_evidence_ids)
        diversity = SourceDiversityValidator()
        verified = VerificationRequiredValidator()
        contradiction = ContradictionValidator()

        accepted: list[str] = []
        rejected: list[str] = []
        issues: list[ReviewIssue] = []
        completeness_scores: list[float] = []
        diversity_scores: list[float] = []
        unsupported = 0

        for finding in payload.findings:
            verification = self._verification_for(finding, payload.verifications)
            checks = [
                citation.validate(finding),
                diversity.validate(finding),
                verified.validate(finding, verification),
                contradiction.validate(finding, verification),
            ]
            completeness_scores.append(checks[0].confidence.score)
            diversity_scores.append(checks[1].confidence.score)
            if not checks[0].success:
                unsupported += 1

            blocking = [
                ReviewIssue(
                    code=issue.code,
                    message=issue.message,
                    finding_id=finding.id,
                    blocking=True,
                )
                for check in checks
                for issue in check.errors
            ]
            issues.extend(blocking)
            issues.extend(
                ReviewIssue(
                    code=issue.code, message=issue.message, finding_id=finding.id, blocking=False
                )
                for check in checks
                for issue in check.warnings
            )
            (accepted if not blocking else rejected).append(finding.id)

        critique = await self._critique(payload, issues)
        # The model may reject, never accept: it can move a finding out of `accepted` but
        # nothing it says can move one in.
        for finding_id in critique.overclaiming_finding_ids:
            if finding_id in accepted:
                accepted.remove(finding_id)
                rejected.append(finding_id)
                issues.append(
                    ReviewIssue(
                        code="reviewer_overclaiming",
                        message="reviewer judged this finding to overstate its evidence",
                        finding_id=finding_id,
                        blocking=True,
                    )
                )

        decision = self._decide(payload, accepted, critique)
        result = ReviewResult(
            decision=decision,
            accepted_findings=tuple(accepted),
            rejected_findings=tuple(dict.fromkeys(rejected)),
            issues=tuple(issues),
            evidence_coverage=_mean(completeness_scores),
            citation_completeness=_mean(completeness_scores),
            source_diversity=_mean(diversity_scores),
            unsupported_claim_rate=round(unsupported / len(payload.findings), 4)
            if payload.findings
            else 0.0,
            critique=critique.critique.strip(),
            reviewed_by=self.name,
            iteration=payload.iteration,
        )
        self.logger.info(
            "review_complete",
            decision=decision.value,
            accepted=len(accepted),
            rejected=len(result.rejected_findings),
            blocking_issues=len(result.blocking_issues),
        )
        return ReviewerOutput(result=result)

    def _verification_for(
        self, finding: ResearchFinding, verifications: tuple[VerificationResult, ...]
    ) -> VerificationResult | None:
        matches = [item for item in verifications if item.finding_id == finding.id]
        return matches[-1] if matches else None

    async def _critique(self, payload: ReviewerInput, issues: list[ReviewIssue]) -> CritiqueDraft:
        """The model's contribution. A failure here degrades the review, never blocks it."""
        if not payload.findings:
            return CritiqueDraft(critique="no findings were produced")

        checks_block = "\n".join(
            f"- {issue.finding_id or '-'}: {issue.code} — {issue.message}" for issue in issues
        )
        try:
            return await self.llm.complete_structured(
                ReviewerPrompt(self.prompt).review_messages(payload, checks_block), CritiqueDraft
            )
        except ResearchAgentError as exc:
            self.logger.warning("reviewer_critique_failed", error=exc.code)
            return CritiqueDraft(
                critique="(model critique unavailable; deterministic checks stand)"
            )

    def _decide(
        self, payload: ReviewerInput, accepted: list[str], critique: CritiqueDraft
    ) -> ReviewDecision:
        if not payload.findings:
            return ReviewDecision.REJECT
        if not accepted:
            return ReviewDecision.REJECT
        if critique.recommend_more_evidence and len(accepted) < len(payload.findings):
            return ReviewDecision.REVISE
        return ReviewDecision.ACCEPT


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
