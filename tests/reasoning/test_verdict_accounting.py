"""The verdict histogram must account for every verification.

The v0.9 completion report tallied partially_supported, contradicted, insufficient and
unverifiable but never VERIFIED, so the histogram silently under-counted and a run with
verified findings looked like a run with none.
"""

from __future__ import annotations

import pytest

from researchagent.models.reasoning import (
    Citation,
    VerificationResult,
    VerificationVerdict,
)


def _verification(verdict: VerificationVerdict, finding_id: str) -> VerificationResult:
    supporting = (
        (Citation(bundle_id="B-1", evidence_ids=("e1",)),)
        if verdict is VerificationVerdict.VERIFIED
        else ()
    )
    contradicting = (
        (Citation(bundle_id="B-1", evidence_ids=("e2",)),)
        if verdict is VerificationVerdict.CONTRADICTED
        else ()
    )
    return VerificationResult(
        finding_id=finding_id,
        verdict=verdict,
        supporting=supporting,
        contradicting=contradicting,
        verified_by="verification",
    )


def histogram(results: list[VerificationResult]) -> dict[VerificationVerdict, int]:
    """The same construction the experiment harness uses: buckets come from the enum."""
    counts = dict.fromkeys(VerificationVerdict, 0)
    for item in results:
        counts[item.verdict] += 1
    return counts


class TestVerdictHistogram:
    def test_the_histogram_sums_to_the_number_of_verifications(self) -> None:
        results = [
            _verification(verdict, f"F-{index}")
            for index, verdict in enumerate(VerificationVerdict)
        ]

        counts = histogram(results)

        assert sum(counts.values()) == len(results)

    def test_every_verdict_has_a_bucket(self) -> None:
        """A verdict added to the enum later cannot go uncounted."""
        counts = histogram([])

        assert set(counts) == set(VerificationVerdict)
        assert all(value == 0 for value in counts.values())

    def test_verified_is_counted(self) -> None:
        """The specific omission that produced a misleading completion report."""
        counts = histogram([_verification(VerificationVerdict.VERIFIED, "F-1")])

        assert counts[VerificationVerdict.VERIFIED] == 1
        assert sum(counts.values()) == 1

    @pytest.mark.parametrize("verdict", list(VerificationVerdict))
    def test_no_verdict_is_silently_dropped(self, verdict: VerificationVerdict) -> None:
        counts = histogram([_verification(verdict, "F-1")])

        assert counts[verdict] == 1
        assert sum(counts.values()) == 1


class TestVerdictIsNotAcceptance:
    def test_a_verified_verdict_is_not_the_same_as_an_accepted_finding(self) -> None:
        """Reported separately on purpose: a run that ends before review has verdicts and
        no accepted findings, and collapsing the two would overstate the result."""
        from researchagent.models.reasoning import FindingStatus

        assert VerificationVerdict.VERIFIED.accepts
        assert FindingStatus.SUPPORTED is not FindingStatus.VERIFIED
        assert not FindingStatus.SUPPORTED.is_citable
