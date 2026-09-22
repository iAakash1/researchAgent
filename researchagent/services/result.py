"""Assemble the persisted run artefacts into one API response."""

from __future__ import annotations

from researchagent.models.bundle import Contradiction, EvidenceBundle
from researchagent.models.reasoning import Citation
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.schemas.result import (
    ContradictionResult,
    EvidenceResult,
    FailedPaperResult,
    FindingResult,
    PaperResult,
    ResearchResult,
)
from researchagent.schemas.workflow import ResearchState
from researchagent.services.audit import AuditTrailBuilder


class ResearchResultBuilder:
    def __init__(self, bundles: JsonBundleRepository, audits: AuditTrailBuilder) -> None:
        self._bundles = bundles
        self._audits = audits

    async def build(self, state: ResearchState) -> ResearchResult:
        bundles = await self._load_bundles(state)
        audits = {item.finding_id: item for item in await self._audits.build(state)}
        processed = set(state.documents.ready_for_extraction if state.documents else ())
        acquisition_failures = {
            item.paper_id: item.reason
            for item in (state.acquisition.failures if state.acquisition else ())
        }
        document_failures = {
            item.paper_id: item.message
            for item in (state.documents.failures if state.documents else ())
        }
        failures = acquisition_failures | document_failures
        selected = set(state.acquisition.selected_ids if state.acquisition else ())
        session = state.reasoning
        review = session.latest_review if session else None
        accepted = set(review.accepted_findings if review else ())

        finding_results: list[FindingResult] = []
        for finding in session.findings if session else ():
            verification = session.verification_for(finding.id) if session else None
            finding_results.append(
                FindingResult(
                    id=finding.id,
                    statement=finding.statement,
                    status=finding.status,
                    verification_status=(
                        verification.verdict.value if verification is not None else None
                    ),
                    reviewer_accepted=finding.id in accepted,
                    supporting_sources=finding.paper_ids,
                    conflicting_sources=_paper_ids(
                        (
                            *finding.contradicting,
                            *(verification.contradicting if verification else ()),
                        )
                    ),
                    citations=finding.citations,
                    provenance=audits[finding.id].provenance if finding.id in audits else (),
                    limitations=finding.limitations,
                )
            )
        findings = tuple(finding_results)
        verified = [item.statement for item in findings if item.reviewer_accepted]
        gaps = tuple(
            dict.fromkeys(
                (
                    *((state.evidence.unanswered_questions) if state.evidence else ()),
                    *((session.unresolved_questions) if session else ()),
                )
            )
        )

        return ResearchResult(
            run_id=state.run_id,
            research_goal=state.goal,
            status=state.status,
            progress_stage=(
                session.stage.value
                if session is not None
                else (state.current_stage.value if state.current_stage else None)
            ),
            discovered_papers=state.discovery.candidates if state.discovery else 0,
            selected_papers=state.acquisition.selected if state.acquisition else 0,
            processed_papers=len(processed),
            papers=tuple(
                PaperResult(
                    id=candidate.paper.id,
                    title=candidate.paper.title,
                    provider=candidate.paper.provider.value,
                    year=candidate.paper.year,
                    url=candidate.paper.url,
                    score=candidate.score,
                    relevance_score=candidate.relevance_score,
                    relevance_decision=candidate.relevance_decision,
                    selected=candidate.paper.id in selected,
                    accessible=candidate.paper.local_path is not None,
                    processed=candidate.paper.id in processed,
                    failure_reason=failures.get(candidate.paper.id),
                )
                for candidate in state.candidates
            ),
            failed_papers=tuple(
                FailedPaperResult(paper_id=paper_id, reason=reason)
                for paper_id, reason in failures.items()
            ),
            findings=findings,
            evidence=_evidence(bundles),
            contradictions=_contradictions(bundles),
            reviewer_status=review.decision if review else None,
            limitations=tuple(
                dict.fromkeys(limit for finding in findings for limit in finding.limitations)
            ),
            research_gaps=gaps,
            final_summary=(
                " ".join(verified)
                if verified
                else "No finding survived verification and reviewer checks for this corpus."
            ),
            failure=state.failure.message if state.failure else None,
            history=tuple(record.stage.value for record in state.history),
        )

    async def _load_bundles(self, state: ResearchState) -> tuple[EvidenceBundle, ...]:
        ids = state.evidence.bundle_ids if state.evidence else ()
        loaded = [bundle for bundle_id in ids if (bundle := await self._bundles.get(bundle_id))]
        return tuple(loaded)


def _paper_ids(citations: tuple[Citation, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            paper_id for citation in citations for paper_id in getattr(citation, "paper_ids", ())
        )
    )


def _evidence(bundles: tuple[EvidenceBundle, ...]) -> tuple[EvidenceResult, ...]:
    items: dict[str, EvidenceResult] = {}
    for bundle in bundles:
        for item in bundle.evidence:
            items.setdefault(
                item.evidence.id,
                EvidenceResult(
                    id=item.evidence.id,
                    paper_id=item.paper_id,
                    quote=item.quote,
                    location=item.location.describe(),
                    stance=item.stance.value,
                    bundle_id=bundle.id,
                ),
            )
    return tuple(items.values())


def _contradictions(bundles: tuple[EvidenceBundle, ...]) -> tuple[ContradictionResult, ...]:
    items: dict[str, Contradiction] = {
        item.id: item for bundle in bundles for item in bundle.contradictions
    }
    return tuple(
        ContradictionResult(
            id=item.id,
            description=item.description,
            left_paper_id=item.left_paper_id,
            right_paper_id=item.right_paper_id,
            left_quotes=tuple(evidence.quote or evidence.claim for evidence in item.left_evidence),
            right_quotes=tuple(
                evidence.quote or evidence.claim for evidence in item.right_evidence
            ),
        )
        for item in items.values()
    )
