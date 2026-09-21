from __future__ import annotations

from researchagent.models.evidence import EvidenceRecord, PaperEvidence
from researchagent.models.reasoning import (
    Citation,
    FindingStatus,
    ResearchFinding,
    ReviewDecision,
    ReviewResult,
    TerminationReason,
    VerificationResult,
    VerificationVerdict,
)
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.repositories.evidence_repository import JsonEvidenceRepository
from researchagent.schemas.reasoning import ReasoningSession
from researchagent.schemas.workflow import EvidenceReport, ResearchState, RunStatus
from researchagent.services.audit import AuditTrailBuilder
from researchagent.services.result import ResearchResultBuilder
from tests.reasoning.conftest import a_bundle


async def test_result_exposes_multi_paper_evidence_and_verified_synthesis(
    bundle_repository: JsonBundleRepository,
    evidence_repository: JsonEvidenceRepository,
) -> None:
    bundle = a_bundle("B-1", ("manual:01", "manual:02"))
    await bundle_repository.save(bundle)
    for paper_id in ("manual:01", "manual:02"):
        items = tuple(item for item in bundle.evidence if item.paper_id == paper_id)
        await evidence_repository.save_paper(
            PaperEvidence(
                paper_id=paper_id,
                document_sha256=f"sha-{paper_id}",
                records=tuple(
                    EvidenceRecord(
                        evidence=item.evidence,
                        paper_id=paper_id,
                        document_sha256=f"sha-{paper_id}",
                    )
                    for item in items
                ),
            )
        )

    citation = Citation(
        bundle_id=bundle.id,
        evidence_ids=tuple(item.evidence.id for item in bundle.evidence),
        paper_ids=("manual:01", "manual:02"),
    )
    finding = ResearchFinding(
        id="F-1",
        question_id="RQ1",
        statement="Two papers report circuit breakers as an overload mitigation.",
        citations=(citation,),
        status=FindingStatus.VERIFIED,
        produced_by="reasoning",
    )
    session = ReasoningSession(
        findings=(finding,),
        verifications=(
            VerificationResult(
                finding_id=finding.id,
                verdict=VerificationVerdict.VERIFIED,
                supporting=(citation,),
                verified_by="verification",
            ),
        ),
        reviews=(
            ReviewResult(
                decision=ReviewDecision.ACCEPT,
                accepted_findings=(finding.id,),
                reviewed_by="reviewer",
            ),
        ),
        terminated=True,
        termination_reason=TerminationReason.ALL_QUESTIONS_ANSWERED,
    )
    state = ResearchState(
        goal="Study overload mitigation across several papers",
        status=RunStatus.COMPLETED,
        evidence=EvidenceReport(bundle_ids=(bundle.id,)),
        reasoning=session,
    )
    builder = ResearchResultBuilder(
        bundle_repository, AuditTrailBuilder(bundle_repository, evidence_repository)
    )

    result = await builder.build(state)

    assert result.final_summary == finding.statement
    assert result.findings[0].supporting_sources == ("manual:01", "manual:02")
    assert len(result.evidence) == 2
    assert len(result.findings[0].provenance) == 2
    assert result.reviewer_status is ReviewDecision.ACCEPT
