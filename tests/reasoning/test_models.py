"""The evidence contract, enforced by the models themselves.

These tests exist because the guarantee "a finding cannot be fabricated" has to be
structural. If it lives in a code path, some future call site skips that path.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchagent.models.reasoning import (
    Citation,
    FindingStatus,
    Hypothesis,
    ResearchFinding,
    TerminationReason,
    VerificationResult,
    VerificationVerdict,
)


class TestFindingRequiresEvidence:
    def test_a_finding_cannot_be_constructed_without_citations(self) -> None:
        """The single most important line in the reasoning layer."""
        with pytest.raises(ValidationError):
            ResearchFinding(
                question_id="RQ1",
                statement="Circuit breakers always prevent overload.",
                citations=(),
                produced_by="reasoning",
            )

    def test_a_citation_cannot_be_constructed_without_evidence_ids(self) -> None:
        with pytest.raises(ValidationError):
            Citation(bundle_id="B-1", evidence_ids=())

    def test_a_citation_with_no_bundle_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation(bundle_id="", evidence_ids=("e1",))

    def test_a_finding_starts_supported_not_verified(self, finding: ResearchFinding) -> None:
        """Reasoning does not get to declare its own output verified."""
        assert finding.status is FindingStatus.SUPPORTED
        assert not finding.status.is_citable

    def test_only_verified_findings_are_citable(self) -> None:
        assert FindingStatus.VERIFIED.is_citable
        for status in FindingStatus:
            if status is not FindingStatus.VERIFIED:
                assert not status.is_citable

    def test_a_finding_knows_which_papers_support_it(self, finding: ResearchFinding) -> None:
        assert set(finding.paper_ids) == {"manual:01", "manual:02"}
        assert finding.is_cross_paper


class TestHypothesisIsNotAFinding:
    """A hypothesis may be unsupported; that is the whole point of having the type."""

    def test_a_hypothesis_needs_no_evidence(self) -> None:
        hypothesis = Hypothesis(
            question_id="RQ1", statement="Retry storms may be the dominant trigger."
        )

        assert hypothesis.supporting == ()
        assert not hypothesis.is_promotable

    def test_a_supported_hypothesis_is_promotable(self) -> None:
        hypothesis = Hypothesis(
            question_id="RQ1",
            statement="Retry storms may be the dominant trigger.",
            supporting=(Citation(bundle_id="B-1", evidence_ids=("e1",)),),
        )

        assert hypothesis.is_promotable


class TestVerdictsMustCite:
    def test_a_verified_verdict_without_evidence_is_rejected(self) -> None:
        """A verifier that approves without citing is asserting, not verifying."""
        with pytest.raises(ValidationError, match="must cite"):
            VerificationResult(
                finding_id="F-1",
                verdict=VerificationVerdict.VERIFIED,
                verified_by="verification",
            )

    def test_a_contradicted_verdict_without_evidence_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must cite"):
            VerificationResult(
                finding_id="F-1",
                verdict=VerificationVerdict.CONTRADICTED,
                verified_by="verification",
            )

    @pytest.mark.parametrize(
        "verdict",
        [
            VerificationVerdict.INSUFFICIENT_EVIDENCE,
            VerificationVerdict.UNVERIFIABLE,
            VerificationVerdict.PARTIALLY_SUPPORTED,
        ],
    )
    def test_absence_verdicts_need_no_citations(self, verdict: VerificationVerdict) -> None:
        """Requiring citations for "nothing supports this" would invite inventing them."""
        result = VerificationResult(finding_id="F-1", verdict=verdict, verified_by="verification")

        assert result.verdict is verdict

    def test_verdicts_route_the_loop(self) -> None:
        assert VerificationVerdict.INSUFFICIENT_EVIDENCE.wants_more_evidence
        assert VerificationVerdict.CONTRADICTED.wants_rereasoning
        assert not VerificationVerdict.CONTRADICTED.wants_more_evidence
        assert VerificationVerdict.VERIFIED.accepts


class TestTermination:
    def test_only_answering_the_questions_counts_as_success(self) -> None:
        assert TerminationReason.ALL_QUESTIONS_ANSWERED.is_success
        for reason in TerminationReason:
            if reason is not TerminationReason.ALL_QUESTIONS_ANSWERED:
                assert not reason.is_success
