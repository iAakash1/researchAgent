"""Persistence ports.

One abstract repository per artefact the pipeline produces, in the order the pipeline
produces them: papers, documents, knowledge, evidence, bundles — then the artefacts of
conclusion maintenance: integrity observations, conclusions and their diffs.

Services depend on these, never on a concrete store, which is what lets the JSON adapters
in ``researchagent/repositories/`` be swapped for a database without touching a service —
and what lets tests run against real repositories over a temp directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from researchagent.models.bundle import EvidenceBundle
from researchagent.models.evidence import EvidenceLink, EvidenceRecord, PaperEvidence
from researchagent.models.integrity import IntegrityObservation
from researchagent.models.library import PaperRecord
from researchagent.models.maintenance import ConclusionDiff, ConclusionRecord, MaintainedTopic
from researchagent.schemas.knowledge import ValidatedKnowledge
from researchagent.schemas.validated import ValidatedDocument


class PaperRepository(ABC):
    @abstractmethod
    async def get(self, paper_id: str) -> PaperRecord | None: ...

    @abstractmethod
    async def save(self, record: PaperRecord) -> PaperRecord:
        """Insert or update. Implementations merge rather than clobber, so a second
        discovery run cannot erase pipeline flags set by a later stage."""

    @abstractmethod
    async def save_many(self, records: Sequence[PaperRecord]) -> list[PaperRecord]: ...

    @abstractmethod
    async def list_all(self) -> list[PaperRecord]: ...

    @abstractmethod
    async def exists(self, paper_id: str) -> bool: ...

    @abstractmethod
    async def delete(self, paper_id: str) -> bool: ...

    async def find_pending(self, flag: str) -> list[PaperRecord]:
        """Records whose ``ProcessingStatus.<flag>`` is still False.

        The hook every later version uses to find its outstanding work.
        """
        records = await self.list_all()
        return [r for r in records if getattr(r.processing, flag) is False]


class DocumentRepository(ABC):
    """Stores validated canonical documents.

    The verdict is stored with the document, never separately: a document whose
    validation has been lost is a document nothing downstream can decide to trust.
    """

    @abstractmethod
    async def get(self, paper_id: str) -> ValidatedDocument | None: ...

    @abstractmethod
    async def save(self, document: ValidatedDocument) -> ValidatedDocument: ...

    @abstractmethod
    async def exists(self, paper_id: str) -> bool: ...

    @abstractmethod
    async def list_ids(self) -> list[str]: ...

    @abstractmethod
    async def delete(self, paper_id: str) -> bool: ...


class KnowledgeRepository(ABC):
    """Stores validated knowledge, always with the verdict that admitted it."""

    @abstractmethod
    async def get(self, paper_id: str) -> ValidatedKnowledge | None: ...

    @abstractmethod
    async def save(self, knowledge: ValidatedKnowledge) -> ValidatedKnowledge: ...

    @abstractmethod
    async def exists(self, paper_id: str) -> bool: ...

    @abstractmethod
    async def list_ids(self) -> list[str]: ...

    @abstractmethod
    async def delete(self, paper_id: str) -> bool: ...


class EvidenceRepository(ABC):
    @abstractmethod
    async def get_paper(self, paper_id: str) -> PaperEvidence | None: ...

    @abstractmethod
    async def save_paper(self, evidence: PaperEvidence) -> PaperEvidence:
        """Replace a paper's evidence index wholesale.

        Wholesale because evidence is re-derived per document: a re-parse invalidates
        every location at once, and merging stale records with fresh ones would leave
        provenance pointing at paragraphs that moved.
        """

    @abstractmethod
    async def get(self, evidence_id: str) -> EvidenceRecord | None:
        """Look up one item by its own id, without going through knowledge."""

    @abstractmethod
    async def for_objects(self, object_ids: tuple[str, ...]) -> tuple[EvidenceRecord, ...]:
        """Every evidence item linked to any of these knowledge objects."""

    @abstractmethod
    async def search(
        self, terms: tuple[str, ...], *, paper_ids: tuple[str, ...] = (), limit: int = 50
    ) -> tuple[EvidenceRecord, ...]:
        """Lexical search over quoted text.

        Deliberately part of the port: v0.7 replaces this implementation with hybrid
        retrieval, and nothing above this line changes.
        """

    @abstractmethod
    async def link(self, link: EvidenceLink) -> EvidenceLink:
        """Record an association. Adding a link never rewrites the evidence itself."""

    @abstractmethod
    async def list_paper_ids(self) -> list[str]: ...


class BundleRepository(ABC):
    @abstractmethod
    async def get(self, bundle_id: str) -> EvidenceBundle | None: ...

    @abstractmethod
    async def save(self, bundle: EvidenceBundle) -> EvidenceBundle: ...

    @abstractmethod
    async def for_question(self, question_id: str) -> tuple[EvidenceBundle, ...]:
        """Bundles built to answer a specific research question."""

    @abstractmethod
    async def list_ids(self) -> list[str]: ...

    @abstractmethod
    async def delete(self, bundle_id: str) -> bool: ...


class IntegrityObservationRepository(ABC):
    """Append-only cache of integrity observations, newest last, per paper.

    Append-only because an observation is a timestamped fact about what a registry said;
    a later lookup adds to the record rather than rewriting what was seen before.
    """

    @abstractmethod
    async def append(self, observation: IntegrityObservation) -> IntegrityObservation: ...

    @abstractmethod
    async def history(self, paper_id: str) -> tuple[IntegrityObservation, ...]:
        """Every observation for a paper, oldest first."""

    async def latest(self, paper_id: str) -> IntegrityObservation | None:
        history = await self.history(paper_id)
        return history[-1] if history else None

    async def latest_informative(self, paper_id: str) -> IntegrityObservation | None:
        """The newest observation that says something about the paper itself.

        A failed lookup never stands in for an earlier real answer.
        """
        return next(
            (item for item in reversed(await self.history(paper_id)) if item.is_informative),
            None,
        )


class ConclusionRepository(ABC):
    """Saved topics and their conclusions, every version kept.

    ``append`` is the only write for conclusions and refuses anything but the next version
    of a conclusion: history cannot be rewritten through this port.
    """

    @abstractmethod
    async def save_topic(self, topic: MaintainedTopic) -> MaintainedTopic: ...

    @abstractmethod
    async def get_topic(self, topic_id: str) -> MaintainedTopic | None: ...

    @abstractmethod
    async def list_topics(self) -> tuple[MaintainedTopic, ...]: ...

    @abstractmethod
    async def append(self, record: ConclusionRecord) -> ConclusionRecord:
        """Persist the next version. Raises ``RepositoryError`` if it is not the next one."""

    @abstractmethod
    async def history(self, conclusion_id: str) -> tuple[ConclusionRecord, ...]:
        """Every version, oldest first. Empty when the conclusion is unknown."""

    @abstractmethod
    async def for_topic(self, topic_id: str) -> tuple[ConclusionRecord, ...]:
        """The latest version of every conclusion in a topic."""

    async def latest(self, conclusion_id: str) -> ConclusionRecord | None:
        history = await self.history(conclusion_id)
        return history[-1] if history else None


class MaintenanceDiffRepository(ABC):
    @abstractmethod
    async def save(self, diff: ConclusionDiff) -> ConclusionDiff: ...

    @abstractmethod
    async def get(self, diff_id: str) -> ConclusionDiff | None: ...

    @abstractmethod
    async def for_topic(self, topic_id: str) -> tuple[ConclusionDiff, ...]:
        """Every diff for a topic, oldest first."""

    async def latest(self, topic_id: str) -> ConclusionDiff | None:
        diffs = await self.for_topic(topic_id)
        return diffs[-1] if diffs else None
