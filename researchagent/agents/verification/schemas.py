"""Verification agent contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.reasoning import ResearchFinding, VerificationResult
from researchagent.models.research import ResearchQuestion


class VerificationDraft(BaseModel):
    """The adversarial check, as the model produces it.

    ``verdict`` is a plain string rather than the enum so a model that invents a verdict
    fails resolution loudly instead of failing schema validation opaquely.
    """

    verdict: str = "insufficient_evidence"
    reasoning: str = ""
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    contradicting_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=8)
    overstatements: list[str] = Field(default_factory=list, max_length=8)


class VerificationInput(BaseModel):
    finding: ResearchFinding
    question: ResearchQuestion
    iteration: int = Field(default=0, ge=0)

    model_config = {"arbitrary_types_allowed": True}


class VerificationOutput(BaseModel):
    result: VerificationResult
    # Provenance addresses the verifier actually resolved. Empty means the finding's
    # citations did not lead anywhere real.
    provenance: tuple[str, ...] = ()
