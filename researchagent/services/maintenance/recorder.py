"""Persists accepted conclusions with their evidence chain resolved.

The minimum run persistence RACM needs: when a research run finishes, its
reviewer-accepted findings become version-1 conclusions of a saved topic (keyed by the
run id), each citation resolved through the stored bundle — or, failing that, the evidence
repository — to KnowledgeObject, SourceLocation and Paper. Evidence that resolves to no
knowledge object is recorded as unresolved, never attached to something else.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.repositories import (
    BundleRepository,
    ConclusionRepository,
    EvidenceRepository,
    KnowledgeRepository,
    PaperRepository,
)
from researchagent.core.logging import get_logger
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.evidence import EvidenceRole
from researchagent.models.maintenance import (
    ChangeType,
    ConclusionRecord,
    MaintainedTopic,
    SupportLink,
    VerificationSnapshot,
)
from researchagent.models.reasoning import (
    Citation,
    FindingStatus,
    ResearchFinding,
    VerificationResult,
)
from researchagent.schemas.workflow import ResearchState
from researchagent.services.integrity.service import identifiers_from_paper_id

logger = get_logger(__name__)


def conclusion_id_for(topic_id: str, finding_id: str) -> str:
    """Stable across processes: the topic plus the finding that first stated it."""
    return f"C:{topic_id}:{finding_id}"


def snapshot(verification: VerificationResult | None, run_id: str) -> VerificationSnapshot | None:
    if verification is None:
        return None
    return VerificationSnapshot(
        verdict=verification.verdict,
        confidence=verification.confidence,
        verified_by=verification.verified_by,
        run_id=run_id,
        checked_at=verification.checked_at,
    )


class ConclusionRecorder:
    name = "conclusion_recorder"

    def __init__(
        self,
        conclusions: ConclusionRepository,
        bundles: BundleRepository,
        evidence: EvidenceRepository,
        knowledge: KnowledgeRepository,
        papers: PaperRepository | None = None,
    ) -> None:
        self._conclusions = conclusions
        self._bundles = bundles
        self._evidence = evidence
        self._knowledge = knowledge
        self._papers = papers

    async def record_run(self, state: ResearchState) -> tuple[ConclusionRecord, ...]:
        """Save the run as a topic and its accepted findings as version-1 conclusions.

        Idempotent: a conclusion that already exists is left as it is.
        """
        session, plan = state.reasoning, state.plan
        if session is None or plan is None:
            return ()
        accepted = [item for item in session.findings if item.status is FindingStatus.VERIFIED]

        topic = await self._conclusions.get_topic(state.run_id)
        if topic is None:
            topic = MaintainedTopic(
                topic_id=state.run_id,
                goal=state.goal,
                plan=plan,
                known_paper_ids=await self.trusted_paper_ids(),
            )
            await self._conclusions.save_topic(topic)

        recorded: list[ConclusionRecord] = []
        for finding in accepted:
            conclusion_id = conclusion_id_for(topic.topic_id, finding.id)
            if await self._conclusions.latest(conclusion_id) is not None:
                continue
            record = await self.first_version(
                finding,
                topic_id=topic.topic_id,
                run_id=state.run_id,
                verification=session.verification_for(finding.id),
            )
            if record is not None:
                recorded.append(await self._conclusions.append(record))
        logger.info(
            "conclusions_recorded",
            topic_id=topic.topic_id,
            accepted=len(accepted),
            recorded=len(recorded),
        )
        return tuple(recorded)

    async def first_version(
        self,
        finding: ResearchFinding,
        *,
        topic_id: str,
        run_id: str,
        verification: VerificationResult | None,
        change_type: ChangeType | None = None,
    ) -> ConclusionRecord | None:
        supports, unresolved = await self.resolve(finding.citations)
        if not supports:
            logger.warning(
                "conclusion_without_resolvable_support",
                finding_id=finding.id,
                unresolved=len(unresolved),
            )
            return None
        return ConclusionRecord(
            conclusion_id=conclusion_id_for(topic_id, finding.id),
            version=1,
            topic_id=topic_id,
            run_id=run_id,
            question_id=finding.question_id,
            statement=finding.statement,
            change_type=change_type,
            supports=supports,
            unresolved_evidence_ids=unresolved,
            verification=snapshot(verification, run_id),
            source_finding_id=finding.id,
            recorded_at=datetime.now(UTC),
        )

    async def resolve(
        self, citations: Iterable[Citation]
    ) -> tuple[tuple[SupportLink, ...], tuple[str, ...]]:
        """Each cited evidence id down to its knowledge object, location and paper."""
        bundles: dict[str, EvidenceBundle | None] = {}
        links: dict[str, SupportLink] = {}
        unresolved: list[str] = []
        for citation in citations:
            if citation.bundle_id not in bundles:
                bundles[citation.bundle_id] = await self._load_bundle(citation.bundle_id)
            bundle = bundles[citation.bundle_id]
            for evidence_id in citation.evidence_ids:
                if evidence_id in links:
                    continue
                link = await self._link(evidence_id, bundle, citation)
                if link is None:
                    unresolved.append(evidence_id)
                else:
                    links[evidence_id] = link
        return tuple(links.values()), tuple(dict.fromkeys(unresolved))

    async def trusted_paper_ids(self) -> tuple[str, ...]:
        """Papers whose knowledge passed validation — the corpus a topic can draw on."""
        found: list[str] = []
        for key in await self._knowledge.list_ids():
            stored = await self._knowledge.get(key)
            if stored is not None and stored.is_trusted:
                found.append(stored.value.paper_id)
        return tuple(sorted(found))

    async def doi_for(self, paper_id: str) -> str | None:
        if self._papers is not None:
            try:
                record = await self._papers.get(paper_id)
            except RepositoryError:
                record = None
            if record is not None and record.paper.doi:
                return record.paper.doi
        return identifiers_from_paper_id(paper_id).doi

    async def _link(
        self, evidence_id: str, bundle: EvidenceBundle | None, citation: Citation
    ) -> SupportLink | None:
        if bundle is not None:
            item = next((e for e in bundle.evidence if e.evidence.id == evidence_id), None)
            if item is not None and item.knowledge_object_id is not None:
                return SupportLink(
                    evidence_id=evidence_id,
                    knowledge_object_id=item.knowledge_object_id,
                    paper_id=item.paper_id,
                    location=item.location,
                    bundle_id=bundle.id,
                    doi=await self.doi_for(item.paper_id),
                    quote=item.quote,
                    role=EvidenceRole.FOUNDING,
                )

        record = await self._evidence.get(evidence_id)
        if record is None:
            return None
        cited = [oid for oid in citation.knowledge_object_ids if oid in record.knowledge_object_ids]
        object_id = next(iter(cited or record.knowledge_object_ids), None)
        if object_id is None:
            return None
        return SupportLink(
            evidence_id=evidence_id,
            knowledge_object_id=object_id,
            paper_id=record.paper_id,
            location=record.location,
            bundle_id=citation.bundle_id,
            doi=await self.doi_for(record.paper_id),
            quote=record.quote,
            role=EvidenceRole.FOUNDING,
        )

    async def _load_bundle(self, bundle_id: str) -> EvidenceBundle | None:
        try:
            return await self._bundles.get(bundle_id)
        except RepositoryError as exc:
            logger.warning("conclusion_bundle_unreadable", bundle_id=bundle_id, error=exc.message)
            return None
