"""The grounding rate must be a rate.

A corpus run once reported grounding_rate=1.2444. The cause was structural rather than
arithmetic: reused (cached) knowledge contributed its objects to the numerator while
contributing nothing to the denominator, because the proposal counters existed only while
the extractors ran and were never persisted. These tests pin the invariant and the two
mechanisms that now uphold it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchagent.models.knowledge import ExtractionStats, PaperKnowledge
from researchagent.schemas.knowledge import KnowledgeBatchResult, KnowledgeOutcome


def _knowledge(paper_id: str, *, objects: int, stats: ExtractionStats | None) -> PaperKnowledge:
    return PaperKnowledge(
        paper_id=paper_id,
        document_sha256="a" * 64,
        objects=(),
        extraction=stats,
    )


def _outcome(paper_id: str, stats: ExtractionStats | None) -> KnowledgeOutcome:
    return KnowledgeOutcome(
        paper_id=paper_id,
        succeeded=True,
        knowledge=_knowledge(paper_id, objects=0, stats=stats),
        drafts_proposed=stats.proposed if stats else 0,
    )


class TestStatsInvariants:
    """The counters cannot describe an impossible extraction."""

    def test_grounded_may_not_exceed_proposed(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed proposed"):
            ExtractionStats(proposed=5, grounded=6, accepted=6)

    def test_accepted_may_not_exceed_grounded(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed grounded"):
            ExtractionStats(proposed=10, grounded=4, accepted=5)

    def test_negative_counts_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionStats(proposed=-1, grounded=0, accepted=0)

    @pytest.mark.parametrize(
        ("proposed", "grounded", "accepted"),
        [(0, 0, 0), (1, 0, 0), (1, 1, 1), (10, 7, 5), (224, 224, 193)],
    )
    def test_rates_stay_within_bounds(self, proposed: int, grounded: int, accepted: int) -> None:
        stats = ExtractionStats(proposed=proposed, grounded=grounded, accepted=accepted)

        assert 0.0 <= stats.grounding_rate <= 1.0
        assert 0.0 <= stats.acceptance_rate <= 1.0
        assert stats.acceptance_rate <= stats.grounding_rate


class TestBatchAggregate:
    """The corpus-level rate is the number that went wrong; it is bounded now."""

    def test_batch_rate_is_bounded_across_a_mixed_corpus(self) -> None:
        result = KnowledgeBatchResult(
            outcomes=(
                _outcome("p1", ExtractionStats(proposed=47, grounded=47, accepted=47)),
                _outcome("p2", ExtractionStats(proposed=44, grounded=39, accepted=35)),
                _outcome("p3", ExtractionStats(proposed=22, grounded=22, accepted=17)),
            )
        )

        rate = result.grounding_rate
        assert rate is not None
        assert 0.0 <= rate <= 1.0
        assert rate == pytest.approx(108 / 113, abs=1e-4)

    def test_unmeasured_papers_contribute_to_neither_side(self) -> None:
        """The original bug, pinned directly.

        A cached paper with 47 objects and no recorded proposals must not inflate the
        rate above 1.0 — it must be excluded from the ratio and counted as unmeasured.
        """
        result = KnowledgeBatchResult(
            outcomes=(
                _outcome("cached", None),
                _outcome("fresh", ExtractionStats(proposed=10, grounded=8, accepted=6)),
            )
        )

        rate = result.grounding_rate
        assert rate is not None
        assert 0.0 <= rate <= 1.0
        assert rate == pytest.approx(0.8)
        assert result.measured_documents == 1
        assert result.unmeasured_documents == 1

    def test_a_wholly_unmeasured_batch_reports_unknown_not_zero(self) -> None:
        result = KnowledgeBatchResult(outcomes=(_outcome("cached", None),))

        assert result.grounding_rate is None, "unknown is not the same claim as 0.0"
        assert result.measured_documents == 0

    def test_empty_batch_reports_unknown(self) -> None:
        assert KnowledgeBatchResult().grounding_rate is None

    def test_outcome_rate_is_unknown_without_stats(self) -> None:
        assert _outcome("cached", None).grounding_rate is None

    @pytest.mark.parametrize("proposed", [1, 3, 17, 224])
    def test_no_combination_of_papers_can_exceed_one(self, proposed: int) -> None:
        """Every valid ExtractionStats has grounded <= proposed, so the sum does too."""
        result = KnowledgeBatchResult(
            outcomes=tuple(
                _outcome(
                    f"p{index}",
                    ExtractionStats(proposed=proposed, grounded=index % (proposed + 1), accepted=0),
                )
                for index in range(5)
            )
        )

        rate = result.grounding_rate
        assert rate is not None
        assert 0.0 <= rate <= 1.0
