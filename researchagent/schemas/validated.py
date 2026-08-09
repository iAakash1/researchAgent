"""Validated artefacts: the immutable pipeline's currency.

Nothing crosses a stage boundary as a bare business object. It crosses wrapped in the
verdict of the validator that inspected it, so the receiving stage can decide for itself
whether to trust what it was handed — which is the whole zero-trust idea, made
mechanical rather than aspirational.

Stages never mutate what they receive. They produce a new, separately validated artefact:

    Paper -> ValidatedPaper -> RawDocument -> PaperDocument -> ValidatedDocument
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.core.validation import Confidence, ValidationIssue, ValidationResult
from researchagent.models.document import PaperDocument
from researchagent.models.paper import Paper


class Validated[T: BaseModel](BaseModel):
    """An artefact together with the verdict that admitted it."""

    model_config = {"frozen": True}

    value: T
    validation: ValidationResult

    @property
    def is_trusted(self) -> bool:
        """Whether downstream stages may use this without further qualification."""
        return self.validation.success

    @property
    def confidence(self) -> Confidence:
        return self.validation.confidence

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return self.validation.warnings

    def require_trusted(self) -> T:
        """Unwrap, refusing to hand back an artefact that failed validation.

        Call sites that genuinely want a rejected artefact (to report on it, to retry it)
        read ``.value`` explicitly, which makes that choice visible in review.
        """
        if not self.is_trusted:
            from researchagent.core.exceptions import ValidationFailedError

            raise ValidationFailedError(
                "Refusing to use an artefact that failed validation",
                validator=self.validation.validator,
                subject_id=self.validation.subject_id,
                issues=list(self.validation.issue_codes()),
            )
        return self.value


# Concrete aliases. Parametrising at import time keeps Pydantic from rebuilding the
# model on every call and gives these types a name that shows up in API schemas.
ValidatedPaper = Validated[Paper]
ValidatedDocument = Validated[PaperDocument]


class DocumentOutcome(BaseModel):
    """Per-paper result of the document intelligence stage.

    A failed paper produces an outcome, never an exception: one unreadable PDF in forty
    must not end the run, and the reason it failed is itself information the reviewer
    needs.
    """

    model_config = {"frozen": True}

    paper_id: str
    succeeded: bool
    document: PaperDocument | None = None
    validation: ValidationResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    remedy: str | None = None
    recoverable: bool = True
    duration_ms: float = Field(default=0.0, ge=0.0)

    @property
    def is_usable(self) -> bool:
        return self.succeeded and self.document is not None and self.validation is not None


class DocumentBatchResult(BaseModel):
    """What the stage did across every paper it was given."""

    model_config = {"frozen": True}

    outcomes: tuple[DocumentOutcome, ...] = ()

    @property
    def succeeded(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.succeeded)

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.succeeded)

    @property
    def documents(self) -> list[PaperDocument]:
        return [o.document for o in self.outcomes if o.document is not None]

    @property
    def mean_confidence(self) -> float:
        scores = [o.validation.confidence.score for o in self.outcomes if o.validation is not None]
        return round(sum(scores) / len(scores), 6) if scores else 0.0
