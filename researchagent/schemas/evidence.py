"""Evidence stage contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.bundle import EvidenceBundle


class EvidenceIndexReport(BaseModel):
    """What indexing did. Skipped papers are named, not merely counted."""

    model_config = {"frozen": True}

    papers_indexed: int = Field(default=0, ge=0)
    papers_skipped: tuple[str, ...] = ()
    evidence_records: int = Field(default=0, ge=0)

    @property
    def deduplication_note(self) -> str:
        return f"{self.evidence_records} evidence records across {self.papers_indexed} papers"


class BundleOutcome(BaseModel):
    """One question's bundle, or the reason there isn't one."""

    model_config = {"frozen": True}

    question_id: str | None
    succeeded: bool
    bundle: EvidenceBundle | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)

    @property
    def bundle_id(self) -> str | None:
        return self.bundle.id if self.bundle else None


class EvidenceBatchResult(BaseModel):
    model_config = {"frozen": True}

    index: EvidenceIndexReport = Field(default_factory=EvidenceIndexReport)
    outcomes: tuple[BundleOutcome, ...] = ()

    @property
    def trusted(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.succeeded)

    @property
    def bundles(self) -> list[EvidenceBundle]:
        return [o.bundle for o in self.outcomes if o.bundle is not None]

    @property
    def total_contradictions(self) -> int:
        return sum(len(bundle.contradictions) for bundle in self.bundles)

    @property
    def unanswered_questions(self) -> tuple[str, ...]:
        """Questions whose bundle carries no evidence — the gaps in the review."""
        return tuple(
            outcome.question_id or "?"
            for outcome in self.outcomes
            if outcome.bundle is None or outcome.bundle.is_empty
        )
