"""Reasoning domain objects — the v0.9 canonical layer.

The load-bearing distinction, and the reason this module exists separately from
``models/knowledge.py``:

    KnowledgeObject  = a fact one paper states, extracted and grounded in that paper.
    ResearchFinding  = a conclusion the system drew across several papers.

They must never merge. A KnowledgeObject is true of a document; a ResearchFinding is a
claim about the literature, and it is only as good as the citations under it. Collapsing
them would let a synthesised conclusion inherit the trust that belongs to a located quote.

The invariant that makes a finding checkable is enforced by the model, not by review:

    A ResearchFinding cannot be constructed without at least one Citation.

and a Citation cannot be constructed without an EvidenceBundle id. So the chain

    ResearchFinding -> Citation -> EvidenceBundle -> Evidence -> SourceLocation -> PDF

is total by construction. An agent that produces a claim with nothing under it produces
a ``Hypothesis`` — which is a legitimate output — and never a finding.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from researchagent.core.validation import Confidence


class Citation(BaseModel):
    """Where a claim's support lives.

    Addresses an EvidenceBundle rather than raw text: the bundle is the validated unit,
    and citing it means the whole provenance chain underneath is already established.
    """

    model_config = {"frozen": True}

    bundle_id: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    knowledge_object_ids: tuple[str, ...] = ()
    paper_ids: tuple[str, ...] = ()
    quote: str | None = Field(
        default=None, description="Verbatim supporting text, when the agent identified one"
    )

    @property
    def is_traceable(self) -> bool:
        return bool(self.bundle_id and self.evidence_ids)


class FindingStatus(StrEnum):
    """How far a claim has travelled through the pipeline.

    Deliberately ordered from weakest to strongest. The three middle states are what stop
    the system from presenting a guess and a verified conclusion in the same voice.
    """

    # Proposed by reasoning, not yet checked. May have no supporting evidence at all.
    HYPOTHESIS = "hypothesis"
    # Carries citations, but verification has not run.
    SUPPORTED = "supported"
    # Verification found the evidence sufficient.
    VERIFIED = "verified"
    # Verification found contradicting evidence, or the reviewer rejected it.
    REJECTED = "rejected"
    # Neither supported nor refuted by the corpus. A real answer, not a failure.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    @property
    def is_citable(self) -> bool:
        """Whether this may be presented as a result of the research."""
        return self is FindingStatus.VERIFIED


class Hypothesis(BaseModel):
    """A candidate explanation, explicitly not yet established.

    Modelled separately from a finding so that "the system suspects X" can be recorded
    without ever being mistaken for "the system found X". A hypothesis with no support is
    valid; a finding with no support is not constructible.
    """

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: f"H-{uuid4().hex[:8]}")
    question_id: str = Field(min_length=1, description="The research question that prompted it")
    statement: str = Field(min_length=10)
    rationale: str = ""
    supporting: tuple[Citation, ...] = ()
    contradicting: tuple[Citation, ...] = ()
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_promotable(self) -> bool:
        """Whether this could become a finding: it has something under it."""
        return bool(self.supporting)


class ResearchFinding(BaseModel):
    """A conclusion synthesised from validated evidence across papers.

    Not a KnowledgeObject: nothing in the corpus states this. It is the system's own
    claim, which is exactly why it must carry its citations and why it starts life as
    SUPPORTED rather than VERIFIED.
    """

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: f"F-{uuid4().hex[:8]}")
    question_id: str = Field(min_length=1)
    statement: str = Field(min_length=10)
    reasoning: str = Field(default="", description="How the evidence leads to the statement")

    # Non-empty by construction. This is the anti-fabrication guarantee, and it is the
    # single most important line in the module.
    citations: tuple[Citation, ...] = Field(min_length=1)
    contradicting: tuple[Citation, ...] = ()

    status: FindingStatus = FindingStatus.SUPPORTED
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    # Caveats the reasoning agent stated itself; kept because a finding that names its own
    # limits is more useful than one that does not.
    limitations: tuple[str, ...] = ()

    derived_from_hypothesis: str | None = None
    produced_by: str = Field(min_length=1)
    iteration: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _citations_are_traceable(self) -> ResearchFinding:
        untraceable = [c.bundle_id for c in self.citations if not c.is_traceable]
        if untraceable:
            raise ValueError(f"finding cites bundles with no evidence: {untraceable}")
        return self

    @property
    def bundle_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(citation.bundle_id for citation in self.citations))

    @property
    def paper_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(pid for citation in self.citations for pid in citation.paper_ids)
        )

    @property
    def is_cross_paper(self) -> bool:
        """Whether more than one paper supports this. Single-paper findings are weaker."""
        return len(self.paper_ids) > 1


class VerificationVerdict(StrEnum):
    """The verifier's judgement. Never a boolean: "cannot tell" is a distinct answer."""

    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    # The claim is not the kind of thing this corpus could settle either way.
    UNVERIFIABLE = "unverifiable"

    @property
    def accepts(self) -> bool:
        return self is VerificationVerdict.VERIFIED

    @property
    def wants_more_evidence(self) -> bool:
        """Whether another retrieval round could change this verdict."""
        return self in (
            VerificationVerdict.INSUFFICIENT_EVIDENCE,
            VerificationVerdict.PARTIALLY_SUPPORTED,
        )

    @property
    def wants_rereasoning(self) -> bool:
        """Whether the claim itself should be reconsidered rather than re-evidenced."""
        return self is VerificationVerdict.CONTRADICTED


class VerificationResult(BaseModel):
    """An adversarial check on one finding.

    Carries its own citations: a verdict asserted without evidence is an opinion, and the
    verifier is held to the same standard as the agent it is checking.
    """

    model_config = {"frozen": True}

    finding_id: str = Field(min_length=1)
    verdict: VerificationVerdict
    reasoning: str = Field(default="")
    supporting: tuple[Citation, ...] = ()
    contradicting: tuple[Citation, ...] = ()
    # Claims in the finding the verifier could not tie to any evidence.
    unsupported_claims: tuple[str, ...] = ()
    # Where the finding says more than its evidence does.
    overstatements: tuple[str, ...] = ()
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    verified_by: str = Field(min_length=1)
    iteration: int = Field(default=0, ge=0)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _a_verdict_must_cite(self) -> VerificationResult:
        """VERIFIED and CONTRADICTED are positive claims and need evidence.

        The other three verdicts are statements about *absence*, which is exactly the
        case where there is nothing to cite — requiring citations there would push the
        verifier to invent them.
        """
        if self.verdict is VerificationVerdict.VERIFIED and not self.supporting:
            raise ValueError("a VERIFIED verdict must cite the evidence that verified it")
        if self.verdict is VerificationVerdict.CONTRADICTED and not self.contradicting:
            raise ValueError("a CONTRADICTED verdict must cite the contradicting evidence")
        return self


class ReviewDecision(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class ReviewIssue(BaseModel):
    """One problem the reviewer found, and whether it is disqualifying."""

    model_config = {"frozen": True}

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    finding_id: str | None = None
    blocking: bool = True


class ReviewResult(BaseModel):
    """The final gate. Deterministic checks first; the model is one signal among them."""

    model_config = {"frozen": True}

    decision: ReviewDecision
    accepted_findings: tuple[str, ...] = ()
    rejected_findings: tuple[str, ...] = ()
    issues: tuple[ReviewIssue, ...] = ()
    # What the deterministic validators measured, independent of any model's opinion.
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    source_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    critique: str = ""
    reviewed_by: str = Field(min_length=1)
    iteration: int = Field(default=0, ge=0)
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def blocking_issues(self) -> tuple[ReviewIssue, ...]:
        return tuple(issue for issue in self.issues if issue.blocking)


class TerminationReason(StrEnum):
    """Why the loop stopped. Every run has exactly one; none stops silently."""

    ALL_QUESTIONS_ANSWERED = "all_questions_answered"
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNRESOLVED_CONTRADICTIONS = "unresolved_contradictions"
    VERIFICATION_REPEATEDLY_FAILED = "verification_repeatedly_failed"
    REVIEWER_REJECTED = "reviewer_rejected"
    AGENT_FAILED = "agent_failed"

    @property
    def is_success(self) -> bool:
        return self is TerminationReason.ALL_QUESTIONS_ANSWERED
