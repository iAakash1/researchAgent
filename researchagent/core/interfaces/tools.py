"""The research toolbox port.

Agents in this system reason; they do not open sockets. But a retrieval agent that cannot
retrieve is not an agent, so v0.9 gives agents exactly one I/O surface: this port.

The rules it exists to enforce:

* **Named operations, not queries.** There is no ``execute(cypher)`` and no
  ``search(sql)``. Every tool is a domain verb with a typed signature, so a model can
  choose *what* to ask but never *how* the store is asked.
* **Bounded.** Every tool caps its own result count. An agent cannot request the corpus.
* **Typed results.** Tools return domain models, never dicts or free text, so an agent's
  next step is validated the same way its first was.
* **Recorded.** Every call produces a ``ToolCall`` for the audit trail, including
  failures — a tool that errored is part of how a conclusion was reached.

Concrete toolboxes live in ``services/tools/`` and compose the existing repositories and
retrieval services. Nothing here knows about Qdrant, Neo4j or BM25.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from researchagent.models.bundle import Contradiction, EvidenceBundle
from researchagent.models.evidence import EvidenceRecord
from researchagent.models.graph import GraphNode
from researchagent.models.knowledge import KnowledgeObject


class ToolName(StrEnum):
    """The complete vocabulary. An agent naming anything else is rejected, not guessed at."""

    SEARCH_KNOWLEDGE = "search_knowledge"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    BUILD_BUNDLE = "build_bundle"
    SEARCH_GRAPH = "search_graph"
    GET_PROVENANCE = "get_provenance"
    FIND_CONTRADICTIONS = "find_contradictions"
    GET_PAPER_CONTEXT = "get_paper_context"


class ToolCall(BaseModel):
    """One tool invocation, recorded whether it succeeded or not."""

    model_config = {"frozen": True}

    tool: ToolName
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    result_count: int = 0
    latency_ms: float = 0.0
    succeeded: bool = True
    error: str | None = None
    agent: str = ""
    iteration: int = Field(default=0, ge=0)
    called_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeSearchResult(BaseModel):
    model_config = {"frozen": True}

    objects: tuple[KnowledgeObject, ...] = ()
    retrieved_by: str = ""
    degraded: bool = False


class EvidenceSearchResult(BaseModel):
    model_config = {"frozen": True}

    records: tuple[EvidenceRecord, ...] = ()
    degraded: bool = False


class GraphSearchResult(BaseModel):
    model_config = {"frozen": True}

    nodes: tuple[GraphNode, ...] = ()
    # Provenance strings for the relationships traversed, so a graph answer is citable.
    citations: tuple[str, ...] = ()
    available: bool = True


class PaperContext(BaseModel):
    """What a paper says, at the level an agent can reason over without reading the PDF."""

    model_config = {"frozen": True}

    paper_id: str
    title: str = ""
    year: int | None = None
    objects: tuple[KnowledgeObject, ...] = ()
    found: bool = True


class ResearchToolbox(ABC):
    """The only I/O an agent may perform.

    Every method is bounded, typed and recorded. Implementations must never raise for an
    empty result — "nothing found" is an answer an agent needs to be able to act on, and
    an exception would make absence indistinguishable from failure.
    """

    @abstractmethod
    async def search_knowledge(
        self, query: str, *, kinds: tuple[str, ...] = (), limit: int = 10
    ) -> KnowledgeSearchResult:
        """Find knowledge objects relevant to a query, via the configured retriever."""

    @abstractmethod
    async def retrieve_evidence(
        self, knowledge_object_ids: tuple[str, ...], *, limit: int = 20
    ) -> EvidenceSearchResult:
        """Fetch the evidence backing specific knowledge objects."""

    @abstractmethod
    async def build_bundle(
        self, query: str, *, kinds: tuple[str, ...] = (), paper_ids: tuple[str, ...] = ()
    ) -> EvidenceBundle:
        """Assemble a validated EvidenceBundle. The only route to citable context."""

    @abstractmethod
    async def search_graph(
        self, entity_name: str, *, depth: int = 1, limit: int = 25
    ) -> GraphSearchResult:
        """Traverse the knowledge graph around a named entity."""

    @abstractmethod
    async def get_provenance(self, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Resolve evidence ids to human-checkable source addresses."""

    @abstractmethod
    async def find_contradictions(
        self, paper_ids: tuple[str, ...] = ()
    ) -> tuple[Contradiction, ...]:
        """Disagreements across the corpus. Both sides, never resolved."""

    @abstractmethod
    async def get_paper_context(self, paper_id: str) -> PaperContext:
        """Everything validated that one paper states."""

    @property
    @abstractmethod
    def calls(self) -> tuple[ToolCall, ...]:
        """Every call made through this toolbox, in order. The audit trail's raw material."""
