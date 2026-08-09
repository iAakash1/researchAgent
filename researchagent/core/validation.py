"""Zero-trust validation primitives.

Every stage in the pipeline validates what the previous stage handed it. The result of
that check is a first-class object, not a boolean and not an exception: downstream stages
need to know *how much* to trust an input, not merely whether it exists.

Confidence is never invented. A :class:`ConfidenceSignal` cannot be constructed without
an ``observation`` — the concrete, checkable fact it was derived from. A score with no
signals is :meth:`Confidence.unknown`, which is honest, rather than a made-up 0.5.

Everything here is frozen. Validation results are evidence about a moment in time; a
later stage that could edit them would defeat the purpose.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from researchagent.core.evidence import Evidence


class Severity(StrEnum):
    """How badly an issue undermines the subject."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

    @property
    def blocks_use(self) -> bool:
        """Whether an issue of this severity makes the subject unusable downstream."""
        return self in (Severity.ERROR, Severity.FATAL)


class ConfidenceLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceSignal(BaseModel):
    """One observable basis for a confidence score.

    ``observation`` is mandatory and is the whole point: it forces every contribution to
    name the measurable fact behind it ("14 of 15 citations resolved to a reference"),
    which is what makes a score auditable and stops confidence from being guessed.
    """

    model_config = {"frozen": True}

    name: str = Field(min_length=1)
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, gt=0.0)
    observation: str = Field(min_length=1, description="The measurable fact behind `value`")


class Confidence(BaseModel):
    """A score plus the signals that produced it."""

    model_config = {"frozen": True}

    score: float = Field(ge=0.0, le=1.0)
    signals: tuple[ConfidenceSignal, ...] = ()

    @classmethod
    def from_signals(cls, signals: list[ConfidenceSignal]) -> Confidence:
        """Weighted mean of the signals. No signals means no basis, not a middling score."""
        if not signals:
            return cls.unknown()
        total_weight = sum(signal.weight for signal in signals)
        weighted = sum(signal.value * signal.weight for signal in signals)
        return cls(score=round(weighted / total_weight, 6), signals=tuple(signals))

    @classmethod
    def unknown(cls) -> Confidence:
        """No observable basis. Distinct from 'observed and found to be bad'."""
        return cls(score=0.0, signals=())

    @classmethod
    def certain(cls, observation: str) -> Confidence:
        """For facts read directly off the artefact, e.g. a page count."""
        return cls.from_signals(
            [ConfidenceSignal(name="direct_observation", value=1.0, observation=observation)]
        )

    @property
    def level(self) -> ConfidenceLevel:
        if not self.signals:
            return ConfidenceLevel.NONE
        if self.score >= 0.8:
            return ConfidenceLevel.HIGH
        if self.score >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @property
    def is_grounded(self) -> bool:
        """Whether the score rests on any observation at all."""
        return bool(self.signals)

    def combined_with(self, other: Confidence) -> Confidence:
        return Confidence.from_signals([*self.signals, *other.signals])

    def explain(self) -> str:
        if not self.signals:
            return "no observations"
        return "; ".join(f"{s.name}={s.value:.2f} ({s.observation})" for s in self.signals)


class ValidationIssue(BaseModel):
    """One problem found during validation.

    ``remedy`` is what turns a report into something actionable — the recommended action
    an operator or a later automated recovery step should take.
    """

    model_config = {"frozen": True}

    code: str = Field(min_length=1, description="Stable machine-readable identifier")
    message: str = Field(min_length=1)
    severity: Severity = Severity.ERROR
    field: str | None = Field(default=None, description="Which part of the subject")
    remedy: str | None = Field(default=None, description="Recommended action")

    @classmethod
    def info(cls, code: str, message: str, **kwargs: object) -> ValidationIssue:
        return cls.model_validate({"code": code, "message": message, "severity": "info", **kwargs})

    @classmethod
    def warning(cls, code: str, message: str, **kwargs: object) -> ValidationIssue:
        return cls.model_validate(
            {"code": code, "message": message, "severity": "warning", **kwargs}
        )

    @classmethod
    def error(cls, code: str, message: str, **kwargs: object) -> ValidationIssue:
        return cls.model_validate({"code": code, "message": message, "severity": "error", **kwargs})

    @classmethod
    def fatal(cls, code: str, message: str, **kwargs: object) -> ValidationIssue:
        return cls.model_validate({"code": code, "message": message, "severity": "fatal", **kwargs})


class ValidationResult(BaseModel):
    """The verdict of one validator on one subject."""

    model_config = {"frozen": True}

    validator: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    subject_type: str = Field(min_length=1)
    success: bool
    confidence: Confidence = Field(default_factory=Confidence.unknown)
    issues: tuple[ValidationIssue, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _success_agrees_with_issues(self) -> ValidationResult:
        # A result cannot claim success while carrying a blocking issue; the two would
        # disagree and downstream code would have to guess which to believe.
        if self.success and any(issue.severity.blocks_use for issue in self.issues):
            raise ValueError(
                "ValidationResult cannot be successful while holding error/fatal issues"
            )
        return self

    @classmethod
    def passed(
        cls,
        *,
        validator: str,
        subject_id: str,
        subject_type: str,
        confidence: Confidence,
        issues: list[ValidationIssue] | None = None,
        evidence: list[Evidence] | None = None,
        duration_ms: float = 0.0,
    ) -> ValidationResult:
        return cls(
            validator=validator,
            subject_id=subject_id,
            subject_type=subject_type,
            success=True,
            confidence=confidence,
            issues=tuple(issues or ()),
            evidence=tuple(evidence or ()),
            duration_ms=duration_ms,
        )

    @classmethod
    def failed(
        cls,
        *,
        validator: str,
        subject_id: str,
        subject_type: str,
        issues: list[ValidationIssue],
        confidence: Confidence | None = None,
        evidence: list[Evidence] | None = None,
        duration_ms: float = 0.0,
    ) -> ValidationResult:
        return cls(
            validator=validator,
            subject_id=subject_id,
            subject_type=subject_type,
            success=False,
            confidence=confidence or Confidence.unknown(),
            issues=tuple(issues),
            evidence=tuple(evidence or ()),
            duration_ms=duration_ms,
        )

    @classmethod
    def decide(
        cls,
        *,
        validator: str,
        subject_id: str,
        subject_type: str,
        confidence: Confidence,
        issues: list[ValidationIssue],
        evidence: list[Evidence] | None = None,
        duration_ms: float = 0.0,
    ) -> ValidationResult:
        """Success is derived from the issues rather than asserted by the caller."""
        blocking = [issue for issue in issues if issue.severity.blocks_use]
        if blocking:
            return cls.failed(
                validator=validator,
                subject_id=subject_id,
                subject_type=subject_type,
                issues=issues,
                confidence=confidence,
                evidence=evidence,
                duration_ms=duration_ms,
            )
        return cls.passed(
            validator=validator,
            subject_id=subject_id,
            subject_type=subject_type,
            confidence=confidence,
            issues=issues,
            evidence=evidence,
            duration_ms=duration_ms,
        )

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.WARNING)

    @property
    def fatal_issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.FATAL)

    @property
    def is_fatal(self) -> bool:
        """Fatal means retrying will not help; the subject itself is unusable."""
        return bool(self.fatal_issues)

    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def merged_with(self, other: ValidationResult, *, validator: str) -> ValidationResult:
        """Combine two verdicts about the same subject into an aggregate one."""
        return ValidationResult(
            validator=validator,
            subject_id=self.subject_id,
            subject_type=self.subject_type,
            success=self.success and other.success,
            confidence=self.confidence.combined_with(other.confidence),
            issues=(*self.issues, *other.issues),
            evidence=(*self.evidence, *other.evidence),
            duration_ms=self.duration_ms + other.duration_ms,
        )


def aggregate(
    results: list[ValidationResult], *, validator: str, subject_id: str, subject_type: str
) -> ValidationResult:
    """Fold many validator verdicts into one. Any blocking issue sinks the aggregate."""
    issues = [issue for result in results for issue in result.issues]
    evidence = [item for result in results for item in result.evidence]
    signals = [signal for result in results for signal in result.confidence.signals]

    return ValidationResult.decide(
        validator=validator,
        subject_id=subject_id,
        subject_type=subject_type,
        confidence=Confidence.from_signals(signals),
        issues=issues,
        evidence=evidence,
        duration_ms=sum(result.duration_ms for result in results),
    )
