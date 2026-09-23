from __future__ import annotations

from researchagent.models.evidence import EvidenceRecord, PaperEvidence
from researchagent.models.paper import Paper, PaperIdentifiers, SourceName
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
from researchagent.schemas.workflow import (
    AcquisitionReport,
    DocumentReport,
    EvidenceReport,
    ResearchState,
    RunStatus,
)
from researchagent.services.audit import AuditTrailBuilder
from researchagent.services.ranking import ScoredPaper
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


async def test_result_chooses_safe_canonical_paper_links(
    bundle_repository: JsonBundleRepository,
    evidence_repository: JsonEvidenceRepository,
) -> None:
    papers = [
        Paper(
            id="doi:10.1145/3600006",
            title="Published paper",
            provider=SourceName.CROSSREF,
            identifiers=PaperIdentifiers(doi="10.1145/3600006"),
            url="https://publisher.example/paper",
            pdf_url="https://publisher.example/paper.pdf",
        ),
        Paper(
            id="arxiv:2401.12345",
            title="Preprint",
            provider=SourceName.ARXIV,
            identifiers=PaperIdentifiers(arxiv_id="2401.12345"),
            url="https://arxiv.org/abs/2401.12345",
            pdf_url="https://arxiv.org/pdf/2401.12345",
        ),
        Paper(
            id="s2:pdf-only",
            title="PDF only",
            provider=SourceName.SEMANTIC_SCHOLAR,
            pdf_url="https://example.org/pdf-only.pdf",
        ),
        Paper(
            id="openalex:missing",
            title="No source URL",
            provider=SourceName.OPENALEX,
        ),
        Paper(
            id="openalex:unsafe",
            title="Unsafe source URL",
            provider=SourceName.OPENALEX,
            url="javascript:alert(1)",
        ),
    ]
    ids = tuple(paper.id for paper in papers)
    state = ResearchState(
        goal="Study safe canonical links for research papers",
        status=RunStatus.COMPLETED,
        candidates=[ScoredPaper(paper=paper, score=0.9) for paper in papers],
        acquisition=AcquisitionReport(
            selected=len(papers), available=len(papers), selected_ids=ids
        ),
        documents=DocumentReport(ready_for_extraction=ids),
    )
    builder = ResearchResultBuilder(
        bundle_repository, AuditTrailBuilder(bundle_repository, evidence_repository)
    )

    result = await builder.build(state)
    by_id = {paper.id: paper for paper in result.papers}

    published = by_id["doi:10.1145/3600006"]
    assert published.doi == "10.1145/3600006"
    assert published.source_url == "https://doi.org/10.1145/3600006"
    assert published.pdf_url == "https://publisher.example/paper.pdf"
    assert by_id["arxiv:2401.12345"].source_url == "https://arxiv.org/abs/2401.12345"
    assert by_id["s2:pdf-only"].source_url == "https://example.org/pdf-only.pdf"
    assert by_id["openalex:missing"].source_url is None
    assert by_id["openalex:unsafe"].source_url is None
