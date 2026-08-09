from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.validation import (
    Confidence,
    ConfidenceLevel,
    ConfidenceSignal,
    Severity,
    ValidationIssue,
    ValidationResult,
    aggregate,
)


def signal(name: str = "s", value: float = 1.0, weight: float = 1.0) -> ConfidenceSignal:
    return ConfidenceSignal(
        name=name, value=value, weight=weight, observation=f"{name} measured at {value}"
    )


def result(**overrides: object) -> ValidationResult:
    return ValidationResult.model_validate(
        {
            "validator": "v",
            "subject_id": "doc-1",
            "subject_type": "PaperDocument",
            "success": True,
            "confidence": Confidence.from_signals([signal()]),
        }
        | overrides
    )


class TestConfidence:
    def test_a_signal_cannot_exist_without_an_observation(self) -> None:
        """The whole guard against invented confidence: no basis, no signal."""
        with pytest.raises(ValidationError):
            ConfidenceSignal(name="vibes", value=0.9, observation="")

    def test_no_signals_means_unknown_not_a_middling_guess(self) -> None:
        unknown = Confidence.from_signals([])

        assert unknown.score == 0.0
        assert unknown.level is ConfidenceLevel.NONE
        assert unknown.is_grounded is False

    def test_weighted_mean_of_signals(self) -> None:
        confidence = Confidence.from_signals(
            [signal("a", 1.0, weight=3.0), signal("b", 0.0, weight=1.0)]
        )

        assert confidence.score == 0.75
        assert confidence.is_grounded is True

    def test_levels_follow_the_score(self) -> None:
        assert Confidence.from_signals([signal(value=0.9)]).level is ConfidenceLevel.HIGH
        assert Confidence.from_signals([signal(value=0.6)]).level is ConfidenceLevel.MEDIUM
        assert Confidence.from_signals([signal(value=0.2)]).level is ConfidenceLevel.LOW

    def test_certain_records_its_basis(self) -> None:
        confidence = Confidence.certain("the file has 7 pages")

        assert confidence.score == 1.0
        assert "7 pages" in confidence.explain()

    def test_combining_keeps_every_observation(self) -> None:
        combined = Confidence.from_signals([signal("a")]).combined_with(
            Confidence.from_signals([signal("b", 0.0)])
        )

        assert {s.name for s in combined.signals} == {"a", "b"}
        assert combined.score == 0.5

    def test_explain_lists_the_evidence_for_the_score(self) -> None:
        explanation = Confidence.from_signals([signal("citation_resolution", 0.5)]).explain()

        assert "citation_resolution" in explanation
        assert "measured at 0.5" in explanation


class TestValidationResult:
    def test_success_cannot_coexist_with_a_blocking_issue(self) -> None:
        """A result that claims success while carrying an error is self-contradictory."""
        with pytest.raises(ValidationError):
            result(issues=(ValidationIssue.error("boom", "it broke"),))

    def test_warnings_do_not_block_success(self) -> None:
        passed = result(issues=(ValidationIssue.warning("meh", "not ideal"),))

        assert passed.success is True
        assert len(passed.warnings) == 1

    def test_decide_derives_success_from_the_issues(self) -> None:
        failing = ValidationResult.decide(
            validator="v",
            subject_id="doc-1",
            subject_type="T",
            confidence=Confidence.unknown(),
            issues=[ValidationIssue.error("bad", "nope")],
        )
        passing = ValidationResult.decide(
            validator="v",
            subject_id="doc-1",
            subject_type="T",
            confidence=Confidence.unknown(),
            issues=[ValidationIssue.warning("meh", "eh")],
        )

        assert failing.success is False
        assert passing.success is True

    def test_fatal_is_distinguished_from_error(self) -> None:
        """Fatal means retrying cannot help; error means this attempt failed."""
        fatal = ValidationResult.failed(
            validator="v",
            subject_id="d",
            subject_type="T",
            issues=[ValidationIssue.fatal("scanned", "no text")],
        )
        recoverable = ValidationResult.failed(
            validator="v",
            subject_id="d",
            subject_type="T",
            issues=[ValidationIssue.error("thin", "sparse")],
        )

        assert fatal.is_fatal is True
        assert recoverable.is_fatal is False

    def test_severity_knows_what_blocks_use(self) -> None:
        assert Severity.ERROR.blocks_use is True
        assert Severity.FATAL.blocks_use is True
        assert Severity.WARNING.blocks_use is False
        assert Severity.INFO.blocks_use is False

    def test_results_are_immutable(self) -> None:
        verdict = result()

        with pytest.raises(ValidationError):
            verdict.success = False  # type: ignore[misc]

    def test_issue_codes_are_exposed_for_logging(self) -> None:
        verdict = result(
            issues=(
                ValidationIssue.warning("a", "x"),
                ValidationIssue.info("b", "y"),
            )
        )

        assert verdict.issue_codes() == ("a", "b")


class TestAggregate:
    def test_any_failure_sinks_the_aggregate(self) -> None:
        combined = aggregate(
            [
                result(),
                ValidationResult.failed(
                    validator="other",
                    subject_id="doc-1",
                    subject_type="PaperDocument",
                    issues=[ValidationIssue.error("bad", "nope")],
                ),
            ],
            validator="document_validator",
            subject_id="doc-1",
            subject_type="PaperDocument",
        )

        assert combined.success is False
        assert "bad" in combined.issue_codes()

    def test_signals_and_evidence_are_pooled(self) -> None:
        evidence = Evidence.structural(
            claim="7 pages", document_id="doc-1", produced_by="pdf_validator"
        )
        combined = aggregate(
            [
                result(evidence=(evidence,)),
                result(confidence=Confidence.from_signals([signal("b", 0.0)])),
            ],
            validator="document_validator",
            subject_id="doc-1",
            subject_type="PaperDocument",
        )

        assert len(combined.evidence) == 1
        assert {s.name for s in combined.confidence.signals} == {"s", "b"}
        assert combined.confidence.score == 0.5

    def test_aggregating_nothing_is_an_ungrounded_pass(self) -> None:
        combined = aggregate([], validator="v", subject_id="doc-1", subject_type="PaperDocument")

        assert combined.success is True
        assert combined.confidence.is_grounded is False


class TestEvidence:
    def test_extracted_text_must_carry_its_quote(self) -> None:
        """Evidence that cannot show the source text is not evidence."""
        with pytest.raises(ValidationError):
            Evidence(
                kind="extracted_text",  # type: ignore[arg-type]
                claim="the paper reports 94.3% accuracy",
                location=SourceLocation(document_id="d"),
                produced_by="test",
            )

    def test_location_precision_is_ranked(self) -> None:
        vague = SourceLocation(document_id="d")
        precise = SourceLocation(
            document_id="d", page=3, section_id="s001", paragraph_index=2, char_start=10
        )

        assert precise.precision > vague.precision

    def test_absence_is_recorded_not_omitted(self) -> None:
        """'Looked for and not found' must be distinguishable from 'never checked'."""
        evidence = Evidence.absence(
            claim="no abstract section", document_id="d", produced_by="section_validator"
        )

        assert evidence.kind.value == "absence"
        assert "no abstract" in evidence.summary()

    def test_location_describes_itself_for_citation(self) -> None:
        location = SourceLocation(
            document_id="manual:01", page=4, section_title="Results", paragraph_index=2
        )

        assert location.describe() == "manual:01 p.4 §Results ¶2"
