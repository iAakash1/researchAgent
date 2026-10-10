"""What the verifier is shown when a conclusion lost part of its support.

The existing verification agent is reused unchanged. It is handed a finding whose
citations are exactly the *remaining* support, and one bundle assembled — by the same
builder and validators as every other bundle — from that support alone. Excluded evidence
is not cited and not in the bundle, so the verifier can neither lean on it nor see it.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from researchagent.core.evidence import Evidence
from researchagent.core.exceptions import MaintenanceError
from researchagent.core.interfaces.repositories import (
    BundleRepository,
    EvidenceRepository,
    KnowledgeRepository,
)
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.bundle import BundledEvidence, EvidenceBundle, EvidenceStance
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.maintenance import MaintainedTopic, SupportLink
from researchagent.models.query import QueryIntent, ResearchQuery
from researchagent.models.reasoning import Citation, ResearchFinding
from researchagent.models.research import ResearchQuestion
from researchagent.schemas.maintenance import ReverificationRequest
from researchagent.services.evidence.builder import EvidenceBundleBuilder
from researchagent.services.evidence.contradictions import ContradictionDetector

BUILT_BY = "conclusion_maintenance"


class ReverificationContext(BaseModel):
    model_config = {"frozen": True}

    finding: ResearchFinding
    bundle: EvidenceBundle
    question: ResearchQuestion


class ReverificationContextBuilder:
    def __init__(
        self,
        knowledge: KnowledgeRepository,
        evidence: EvidenceRepository,
        bundles: BundleRepository,
        builder: EvidenceBundleBuilder,
        detector: ContradictionDetector,
    ) -> None:
        self._knowledge = knowledge
        self._evidence = evidence
        self._bundles = bundles
        self._builder = builder
        self._detector = detector

    async def build(
        self, request: ReverificationRequest, topic: MaintainedTopic, *, run_id: str
    ) -> ReverificationContext:
        record = request.record
        try:
            question = topic.plan.question(record.question_id)
        except KeyError as exc:
            raise MaintenanceError(
                "The topic's plan no longer contains this conclusion's question",
                conclusion_id=record.conclusion_id,
                question_id=record.question_id,
            ) from exc

        objects: dict[str, KnowledgeObject] = {}
        bundled: list[BundledEvidence] = []
        for link in request.remaining:
            obj = objects.get(link.knowledge_object_id) or await self._object(link)
            evidence = await self._evidence_for(link, obj)
            if obj is None or evidence is None:
                continue
            objects[obj.id] = obj
            bundled.append(
                BundledEvidence(
                    evidence=evidence,
                    paper_id=link.paper_id,
                    knowledge_object_id=obj.id,
                    stance=EvidenceStance.SUPPORTING,
                    relevance=Confidence.from_signals(
                        [
                            ConfidenceSignal(
                                name="conclusion_support",
                                value=1.0,
                                observation=(
                                    f"{link.role.value} support of {record.conclusion_id} "
                                    f"v{record.version}"
                                ),
                            )
                        ]
                    ),
                )
            )
        if not bundled:
            raise MaintenanceError(
                "None of the remaining support could be loaded for re-verification",
                conclusion_id=record.conclusion_id,
                remaining=len(request.remaining),
            )

        knowledge = tuple(objects.values())
        digest = _digest(run_id, record.conclusion_id)
        bundle = self._builder.assemble(
            ResearchQuery(
                text=record.statement, intent=QueryIntent.TRACE, question_id=record.question_id
            ),
            knowledge,
            tuple(bundled),
            contradictions=_unique((*self._detector.detect(knowledge), *request.contradictions)),
            bundle_id=f"racm:{run_id}:{digest}",
            built_by=BUILT_BY,
        )
        await self._bundles.save(bundle)

        finding = ResearchFinding(
            id=f"RV-{digest}",
            question_id=record.question_id,
            statement=record.statement,
            citations=(
                Citation(
                    bundle_id=bundle.id,
                    evidence_ids=tuple(item.evidence.id for item in bundled),
                    knowledge_object_ids=tuple(objects),
                    paper_ids=tuple(dict.fromkeys(item.paper_id for item in bundled)),
                ),
            ),
            produced_by=BUILT_BY,
        )
        return ReverificationContext(finding=finding, bundle=bundle, question=question)

    async def _object(self, link: SupportLink) -> KnowledgeObject | None:
        stored = await self._knowledge.get(link.paper_id)
        return stored.value.by_id(link.knowledge_object_id) if stored is not None else None

    async def _evidence_for(
        self, link: SupportLink, obj: KnowledgeObject | None
    ) -> Evidence | None:
        record = await self._evidence.get(link.evidence_id)
        if record is not None:
            return record.evidence
        if obj is not None:
            return next((item for item in obj.evidence if item.id == link.evidence_id), None)
        return None


def _digest(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]  # noqa: S324


def _unique[T: BaseModel](items: tuple[T, ...]) -> tuple[T, ...]:
    seen: dict[str, T] = {}
    for item in items:
        seen.setdefault(str(getattr(item, "id", id(item))), item)
    return tuple(seen.values())
