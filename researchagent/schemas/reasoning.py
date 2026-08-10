"""Reasoning stage contracts: the agentic loop's state and its budget.

This is *research memory*, not chat history. Nothing here stores what an agent said —
only what it decided, what it found, and what remains open. That is the difference
between a system that can explain a conclusion and one that can only re-narrate it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from researchagent.config.schemas import ResearchBudget
from researchagent.core.interfaces.tools import ToolCall
from researchagent.models.reasoning import (
    Citation,
    FindingStatus,
    Hypothesis,
    ResearchFinding,
    ReviewResult,
    TerminationReason,
    VerificationResult,
)


class BudgetLedger(BaseModel):
    """What the run has spent. Checked before work, not after."""

    iterations: int = Field(default=0, ge=0)
    retrieval_attempts: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    tokens_by_agent: dict[str, int] = Field(default_factory=dict)
    # LLM calls whose provider reported no usage. Their real cost is unknown, so it is
    # counted rather than assumed to be zero — the difference is what stops a
    # non-reporting provider from making the token budget unenforceable in silence.
    unmeasured_calls: int = Field(default=0, ge=0)

    def exceeded(self, budget: ResearchBudget) -> TerminationReason | None:
        """The first limit this ledger has passed, or None.

        Ordered so the most specific cause is reported: hitting the iteration cap is a
        different story from burning the token budget inside one iteration.
        """
        if self.iterations >= budget.max_iterations:
            return TerminationReason.MAX_ITERATIONS
        if self.total_tokens >= budget.max_total_tokens:
            return TerminationReason.BUDGET_EXHAUSTED
        if self.tool_calls >= budget.max_tool_calls:
            return TerminationReason.BUDGET_EXHAUSTED
        if self.retrieval_attempts >= budget.max_retrieval_attempts:
            return TerminationReason.BUDGET_EXHAUSTED
        if any(spent >= budget.max_tokens_per_agent for spent in self.tokens_by_agent.values()):
            return TerminationReason.BUDGET_EXHAUSTED
        return None

    def with_tokens(self, agent: str, tokens: int) -> BudgetLedger:
        by_agent = dict(self.tokens_by_agent)
        by_agent[agent] = by_agent.get(agent, 0) + tokens
        return self.model_copy(
            update={"total_tokens": self.total_tokens + tokens, "tokens_by_agent": by_agent}
        )


class ReasoningStage(StrEnum):
    """Where in the loop the run currently is."""

    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    REVIEW = "review"
    TERMINATED = "terminated"


class QuestionState(BaseModel):
    """Per-question progress. The loop terminates on this, not on a global feeling."""

    model_config = {"frozen": True}

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    bundle_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    verified_finding_ids: tuple[str, ...] = ()
    retrieval_attempts: int = Field(default=0, ge=0)
    # Set when the loop gave up on this question specifically, with the reason.
    exhausted_reason: str | None = None

    @property
    def is_answered(self) -> bool:
        return bool(self.verified_finding_ids)

    @property
    def is_open(self) -> bool:
        return not self.is_answered and self.exhausted_reason is None


class ReasoningSession(BaseModel):
    """Everything the agentic loop knows. The v0.9 addition to workflow state.

    Held as one nested model rather than a dozen new top-level fields so that the loop's
    memory can be persisted, replayed and audited as a unit.
    """

    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    ledger: BudgetLedger = Field(default_factory=BudgetLedger)

    iteration: int = Field(default=0, ge=0)
    stage: ReasoningStage = ReasoningStage.RETRIEVAL

    questions: tuple[QuestionState, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    findings: tuple[ResearchFinding, ...] = ()
    verifications: tuple[VerificationResult, ...] = ()
    reviews: tuple[ReviewResult, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()

    # Bundle ids the loop has produced, in order. Bundles live in the repository.
    bundle_ids: tuple[str, ...] = ()
    # Questions retrieval could not satisfy — the corpus gaps this run found.
    unresolved_questions: tuple[str, ...] = ()

    terminated: bool = False
    termination_reason: TerminationReason | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _termination_has_a_reason(self) -> ReasoningSession:
        """Never stop silently."""
        if self.terminated and self.termination_reason is None:
            raise ValueError("a terminated session must record why it terminated")
        return self

    @property
    def open_questions(self) -> tuple[QuestionState, ...]:
        return tuple(question for question in self.questions if question.is_open)

    @property
    def verified_findings(self) -> tuple[ResearchFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.status is FindingStatus.VERIFIED
        )

    @property
    def latest_review(self) -> ReviewResult | None:
        return self.reviews[-1] if self.reviews else None

    def verification_for(self, finding_id: str) -> VerificationResult | None:
        """The most recent verdict on a finding."""
        matches = [item for item in self.verifications if item.finding_id == finding_id]
        return matches[-1] if matches else None

    def finding(self, finding_id: str) -> ResearchFinding | None:
        return next((item for item in self.findings if item.id == finding_id), None)


class AuditStep(BaseModel):
    """One link in the chain from goal to conclusion."""

    model_config = {"frozen": True}

    stage: str
    actor: str
    summary: str
    iteration: int = 0
    references: tuple[str, ...] = ()


class FindingAudit(BaseModel):
    """The reconstructible history of a single finding.

    A product feature, not a debug aid: a conclusion nobody can trace back to a page is
    a conclusion nobody should act on.
    """

    model_config = {"frozen": True}

    finding_id: str
    question_id: str
    statement: str
    status: FindingStatus
    steps: tuple[AuditStep, ...] = ()
    citations: tuple[Citation, ...] = ()
    provenance: tuple[str, ...] = ()
    verification: VerificationResult | None = None

    @property
    def is_complete(self) -> bool:
        """Whether the chain reaches actual source locations."""
        return bool(self.citations) and bool(self.provenance)
