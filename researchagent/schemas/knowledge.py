"""Knowledge stage contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.core.validation import ValidationResult
from researchagent.models.knowledge import PaperKnowledge
from researchagent.schemas.validated import Validated

ValidatedKnowledge = Validated[PaperKnowledge]


class RejectionReport(BaseModel):
    """Why extractions did not become knowledge.

    Kept as a first-class result rather than a log line: the share of proposals a paper's
    own text refused to support is the system's most direct measure of how much the model
    made up, and the reviewer needs to see it.
    """

    model_config = {"frozen": True}

    ungrounded: int = Field(default=0, ge=0, description="Quote not found in the document")
    invalid: int = Field(default=0, ge=0, description="Grounded but failed validation")
    by_code: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return self.ungrounded + self.invalid


class KnowledgeOutcome(BaseModel):
    """What the knowledge stage produced for one paper."""

    model_config = {"frozen": True}

    paper_id: str
    succeeded: bool
    knowledge: PaperKnowledge | None = None
    validation: ValidationResult | None = None
    rejections: RejectionReport = Field(default_factory=RejectionReport)
    drafts_proposed: int = 0
    # One extractor failing is recorded without failing the paper.
    extractor_errors: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    remedy: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)

    @property
    def object_count(self) -> int:
        return len(self.knowledge.objects) if self.knowledge else 0

    @property
    def grounding_rate(self) -> float:
        """Share of proposals that survived grounding and validation."""
        return self.object_count / self.drafts_proposed if self.drafts_proposed else 0.0


class KnowledgeBatchResult(BaseModel):
    model_config = {"frozen": True}

    outcomes: tuple[KnowledgeOutcome, ...] = ()

    @property
    def succeeded(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.succeeded)

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.succeeded)

    @property
    def total_objects(self) -> int:
        return sum(outcome.object_count for outcome in self.outcomes)

    @property
    def total_rejected(self) -> int:
        return sum(outcome.rejections.total for outcome in self.outcomes)

    @property
    def grounding_rate(self) -> float:
        proposed = sum(outcome.drafts_proposed for outcome in self.outcomes)
        return round(self.total_objects / proposed, 4) if proposed else 0.0

    @property
    def knowledge(self) -> list[PaperKnowledge]:
        return [o.knowledge for o in self.outcomes if o.knowledge is not None]
