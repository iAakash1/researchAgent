from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from researchagent.agents.base import BaseAgent
from researchagent.agents.registry import AGENTS, build_agent
from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import (
    AgentConfig,
    AgentSpec,
    DocumentsConfig,
    EvidenceConfig,
    GraphConfig,
    KnowledgeConfig,
    ModelCatalog,
    ReasoningConfig,
    RetrievalConfig,
    SourcesConfig,
    WorkflowConfig,
)
from researchagent.container import Container
from researchagent.core.events import EventBus
from researchagent.core.interfaces.llm import (
    CompletionResponse,
    GenerationParams,
    LLMProvider,
    Message,
    ProviderHealth,
    StructuredResult,
    TokenUsage,
    TSchema,
)
from researchagent.core.prompts import PromptLibrary
from researchagent.core.settings import Environment, Settings
from researchagent.integrations.manual import ManualPaperSource
from researchagent.integrations.memory_graph import InMemoryGraphRepository
from researchagent.integrations.memory_store import InMemoryVectorStore
from researchagent.integrations.ollama import NullEmbeddingModel
from researchagent.integrations.pymupdf import PyMuPDFLoader
from researchagent.memory.checkpoints import build_checkpointer
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.repositories.document_repository import JsonDocumentRepository
from researchagent.repositories.evidence_repository import JsonEvidenceRepository
from researchagent.repositories.knowledge_repository import JsonKnowledgeRepository
from researchagent.repositories.paper_repository import JsonPaperRepository
from researchagent.services.audit import AuditTrailBuilder
from researchagent.services.deduplication import PaperDeduplicator
from researchagent.services.discovery_service import DiscoveryService
from researchagent.services.document import (
    CitationExtractor,
    DocumentAssembler,
    DocumentIntelligenceService,
    FigureTableDetector,
    MetadataExtractor,
    ReferenceExtractor,
    SectionDetector,
)
from researchagent.services.evidence import (
    AgreementCrossPaperRetriever,
    ContradictionDetector,
    EvidenceBundleBuilder,
    EvidenceIndexer,
    EvidenceIntelligenceService,
    LexicalKnowledgeRetriever,
    LinkedEvidenceRetriever,
    RepositoryDocumentRetriever,
    StoredBundleRetriever,
)
from researchagent.services.graph.builder import GraphBuilder
from researchagent.services.graph.mapper import GraphMapper
from researchagent.services.graph.queries import GraphQueries
from researchagent.services.graph.validator import GraphValidator
from researchagent.services.knowledge import KnowledgeIntelligenceService, RelationBuilder
from researchagent.services.knowledge.registry import build_extractors
from researchagent.services.llm_service import BoundLLM, LLMService
from researchagent.services.ranking import HeuristicScorer
from researchagent.services.retrieval import KnowledgeIndexer
from researchagent.services.retrieval.registry import build_retrieval_arms, select_active
from researchagent.services.retrieval_service import RetrievalService
from researchagent.services.tools import ServiceToolbox
from researchagent.services.tools.toolbox import ToolBudget
from researchagent.workflows.reasoning_runner import ReasoningRunner
from researchagent.workflows.research import build_research_graph
from researchagent.workflows.runner import WorkflowRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment=Environment.CI,
        config_dir=REPO_ROOT / "config",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def config_loader(settings: Settings) -> ConfigLoader:
    return ConfigLoader(settings.config_dir)


@pytest.fixture
def model_catalog(config_loader: ConfigLoader) -> ModelCatalog:
    return config_loader.load("models", ModelCatalog)


@pytest.fixture
def agent_config(config_loader: ConfigLoader) -> AgentConfig:
    return config_loader.load("agents", AgentConfig)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


class FakeLLMProvider(LLMProvider):
    """Deterministic in-memory provider: no network, no Ollama, no LangChain."""

    name = "fake"

    def __init__(
        self,
        *,
        text: str = "fake response",
        structured: BaseModel | None = None,
        structured_sequence: list[BaseModel] | None = None,
        fail_times: int = 0,
        error: Exception | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        self.text = text
        self.structured = structured
        # Consumed in order, so a multi-phase agent can be driven phase by phase.
        self.structured_sequence = list(structured_sequence or [])
        self.fail_times = fail_times
        self.error = error
        # None models a provider that reports nothing, which is a real case (Ollama
        # structured output before include_raw) and must stay distinguishable from zero.
        self.usage = usage
        self.calls: list[list[Message]] = []
        self.closed = False

    async def complete(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> CompletionResponse:
        self.calls.append(messages)
        self._maybe_fail()
        return CompletionResponse(
            text=self.text,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            latency_ms=1.0,
        )

    async def stream(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        self._maybe_fail()
        for token in self.text.split():
            yield token + " "

    async def complete_structured_with_usage(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> StructuredResult[TSchema]:
        value = await self.complete_structured(messages, model=model, params=params, schema=schema)
        return StructuredResult[TSchema](value=value, usage=self.usage)

    async def complete_structured(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> TSchema:
        self.calls.append(messages)
        self._maybe_fail()
        if self.structured_sequence:
            reply = self.structured_sequence.pop(0)
        elif self.structured is not None:
            reply = self.structured
        else:
            raise AssertionError("FakeLLMProvider needs `structured=` for structured calls")
        return schema.model_validate(reply.model_dump())

    async def health(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, healthy=True, available_models=["fake-model"])

    async def aclose(self) -> None:
        self.closed = True

    def _maybe_fail(self) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error or RuntimeError("fake failure")


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def bound_llm(fake_provider: FakeLLMProvider, model_catalog: ModelCatalog) -> BoundLLM:
    alias = model_catalog.default
    return BoundLLM(alias, model_catalog.spec_for(alias), fake_provider)


@pytest.fixture
def default_agent_spec() -> AgentSpec:
    return AgentSpec()


@pytest.fixture
def llm_service(
    model_catalog: ModelCatalog,
    settings: Settings,
    event_bus: EventBus,
    fake_provider: FakeLLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[LLMService]:
    service = LLMService(model_catalog, settings, event_bus=event_bus)
    monkeypatch.setattr(service, "_provider", lambda _name: fake_provider)
    yield service


@pytest.fixture
def prompt_library() -> PromptLibrary:
    return PromptLibrary(REPO_ROOT / "prompts")


@pytest.fixture
def container(
    settings: Settings,
    config_loader: ConfigLoader,
    model_catalog: ModelCatalog,
    agent_config: AgentConfig,
    event_bus: EventBus,
    prompt_library: PromptLibrary,
    fake_provider: FakeLLMProvider,
    manual_source: ManualPaperSource,
    paper_repository: JsonPaperRepository,
    document_repository: JsonDocumentRepository,
    document_assembler: DocumentAssembler,
    knowledge_repository: JsonKnowledgeRepository,
    evidence_repository: JsonEvidenceRepository,
    bundle_repository: JsonBundleRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Container:
    """A fully wired container. The LLM is faked; the only paper source is the real
    manual library, so no test ever touches a remote index."""
    graph_config = config_loader.load("graph", GraphConfig)
    graph_repository = InMemoryGraphRepository()
    reasoning_config = config_loader.load("reasoning", ReasoningConfig)

    graph_config = config_loader.load("graph", GraphConfig)
    graph_repository = InMemoryGraphRepository()
    reasoning_config = config_loader.load("reasoning", ReasoningConfig)

    service = LLMService(model_catalog, settings, event_bus=event_bus)
    monkeypatch.setattr(service, "_provider", lambda _name: fake_provider)

    documents_config = config_loader.load("documents", DocumentsConfig)
    document_service = DocumentIntelligenceService(
        PyMuPDFLoader(),
        document_assembler,
        document_repository,
        paper_repository,
        documents_config.pipeline,
        documents_config.validation,
        event_bus=event_bus,
    )

    knowledge_config = config_loader.load("knowledge", KnowledgeConfig)
    knowledge_service = KnowledgeIntelligenceService(
        build_extractors(
            knowledge_config.enabled_extractors, service.get("extraction"), prompt_library
        ),
        RelationBuilder(),
        knowledge_repository,
        document_repository,
        paper_repository,
        knowledge_config.pipeline,
        knowledge_config.validation,
        event_bus=event_bus,
    )

    evidence_config = config_loader.load("evidence", EvidenceConfig)
    knowledge_retriever = LexicalKnowledgeRetriever(knowledge_repository, evidence_config.weights)
    evidence_retriever = LinkedEvidenceRetriever(evidence_repository, evidence_config.weights)
    document_retriever = RepositoryDocumentRetriever(document_repository)
    cross_paper_retriever = AgreementCrossPaperRetriever(
        knowledge_retriever, evidence_config.weights
    )
    bundle_retriever = StoredBundleRetriever(bundle_repository)
    retrieval_config = config_loader.load("retrieval", RetrievalConfig)
    # Tests never touch Ollama or Qdrant: a null embedder plus the in-memory store keep
    # the semantic arm constructible and honestly degraded.
    embedding_model = NullEmbeddingModel()
    vector_store = InMemoryVectorStore()
    knowledge_indexer = KnowledgeIndexer(
        embedding_model, vector_store, knowledge_repository, retrieval_config.index
    )
    retrieval_arms = build_retrieval_arms(
        retrieval_config, knowledge_repository, knowledge_retriever, embedding_model, vector_store
    )
    active_knowledge_retriever = select_active(retrieval_arms, retrieval_config)

    evidence_service = EvidenceIntelligenceService(
        EvidenceIndexer(evidence_repository, event_bus=event_bus),
        EvidenceBundleBuilder(
            knowledge_retriever,
            evidence_retriever,
            cross_paper_retriever,
            ContradictionDetector(evidence_config.contradictions),
            evidence_config.bundles,
        ),
        knowledge_repository,
        bundle_repository,
        evidence_config.pipeline,
        event_bus=event_bus,
    )

    discovery_service = DiscoveryService(
        [manual_source],
        PaperDeduplicator(),
        HeuristicScorer(),
        config_loader.load("sources", SourcesConfig).discovery_settings(),
        repository=paper_repository,
    )

    workflow_config = config_loader.load("workflow", WorkflowConfig)
    sources_config = config_loader.load("sources", SourcesConfig)
    planner = build_agent(
        "planner",
        agent_config=agent_config,
        llm_service=service,
        prompts=prompt_library,
        event_bus=event_bus,
    )
    toolbox = ServiceToolbox(
        active_knowledge_retriever,
        evidence_service,
        knowledge_repository,
        evidence_repository,
        paper_repository,
        bundle_repository,
        graph_repository,
        GraphQueries(graph_repository),
        budget=ToolBudget(max_tool_calls=reasoning_config.budget.max_tool_calls),
        event_bus=event_bus,
    )

    def agent_for(
        name: str, iteration: int, tokens_remaining: int | None = None
    ) -> BaseAgent[Any, Any]:
        spec = agent_config.spec_for(name)
        agent_cls = AGENTS.get(name)
        kwargs: dict[str, Any] = {"event_bus": event_bus}
        if name in {"retrieval", "verification"}:
            kwargs["toolbox"] = toolbox.for_agent(name, iteration)
        return agent_cls(
            BoundLLM(spec.model, model_catalog.spec_for(spec.model), fake_provider),
            spec,
            prompt_library,
            **kwargs,
        )

    graph = build_research_graph(
        planner=planner,
        discovery=discovery_service,
        documents=document_service,
        knowledge=knowledge_service,
        evidence=evidence_service,
        checkpointer=build_checkpointer(workflow_config.checkpointer),
    )

    return Container(
        settings=settings,
        config_loader=config_loader,
        model_catalog=model_catalog,
        agent_config=agent_config,
        workflow_config=workflow_config,
        sources_config=sources_config,
        documents_config=documents_config,
        knowledge_config=knowledge_config,
        evidence_config=evidence_config,
        retrieval_config=retrieval_config,
        prompt_library=prompt_library,
        event_bus=event_bus,
        llm_service=service,
        paper_sources=[manual_source],
        paper_repository=paper_repository,
        discovery_service=discovery_service,
        retrieval_service=RetrievalService(
            {manual_source.name: manual_source}, paper_repository, tmp_path / "downloads"
        ),
        document_repository=document_repository,
        document_service=document_service,
        knowledge_repository=knowledge_repository,
        knowledge_service=knowledge_service,
        evidence_repository=evidence_repository,
        bundle_repository=bundle_repository,
        evidence_service=evidence_service,
        knowledge_retriever=knowledge_retriever,
        evidence_retriever=evidence_retriever,
        document_retriever=document_retriever,
        cross_paper_retriever=cross_paper_retriever,
        bundle_retriever=bundle_retriever,
        embedding_model=embedding_model,
        vector_store=vector_store,
        knowledge_indexer=knowledge_indexer,
        retrieval_arms=retrieval_arms,
        active_knowledge_retriever=active_knowledge_retriever,
        # In-memory graph backend: the suite exercises the full graph stack without a
        # running Neo4j, which is what keeps `uv run pytest` offline.
        graph_config=graph_config,
        graph_repository=graph_repository,
        graph_builder=GraphBuilder(
            knowledge_repository,
            graph_repository,
            GraphMapper(),
            GraphValidator(),
            ContradictionDetector(),
            paper_repository,
            event_bus=event_bus,
        ),
        graph_queries=GraphQueries(graph_repository),
        # v0.9. The toolbox is real — it composes the same services — but the LLM behind
        # every agent is faked, so the loop runs offline and deterministically.
        reasoning_config=reasoning_config,
        toolbox=toolbox,
        audit_trail=AuditTrailBuilder(bundle_repository, evidence_repository),
        reasoning_runner=ReasoningRunner(
            agent_for, bundle_repository, reasoning_config, event_bus=event_bus
        ),
        workflow_runner=WorkflowRunner(graph, workflow_config),
    )


MANUAL_LIBRARY = REPO_ROOT / "storage" / "papers" / "raw" / "manual"


@pytest.fixture
def manual_source() -> ManualPaperSource:
    """Points at the real committed collection — this provider is tested against real PDFs."""
    return ManualPaperSource(MANUAL_LIBRARY)


@pytest.fixture
def paper_repository(tmp_path: Path) -> JsonPaperRepository:
    """Always a temp directory: tests must never write into storage/papers/metadata."""
    return JsonPaperRepository(tmp_path / "metadata")


@pytest.fixture
def document_repository(tmp_path: Path) -> JsonDocumentRepository:
    return JsonDocumentRepository(tmp_path / "documents")


@pytest.fixture
def document_assembler() -> DocumentAssembler:
    return DocumentAssembler(
        SectionDetector(),
        ReferenceExtractor(),
        CitationExtractor(),
        FigureTableDetector(),
        MetadataExtractor(),
    )


@pytest.fixture
def document_service(
    document_assembler: DocumentAssembler,
    document_repository: JsonDocumentRepository,
    paper_repository: JsonPaperRepository,
    event_bus: EventBus,
) -> DocumentIntelligenceService:
    return DocumentIntelligenceService(
        PyMuPDFLoader(),
        document_assembler,
        document_repository,
        paper_repository,
        event_bus=event_bus,
    )


@pytest.fixture
def knowledge_repository(tmp_path: Path) -> JsonKnowledgeRepository:
    return JsonKnowledgeRepository(tmp_path / "knowledge")


@pytest.fixture
def knowledge_service(
    bound_llm: BoundLLM,
    prompt_library: PromptLibrary,
    knowledge_repository: JsonKnowledgeRepository,
    document_repository: JsonDocumentRepository,
    paper_repository: JsonPaperRepository,
    event_bus: EventBus,
) -> KnowledgeIntelligenceService:
    return KnowledgeIntelligenceService(
        build_extractors(
            (
                "method_extractor",
                "dataset_extractor",
                "metric_extractor",
                "result_extractor",
                "limitation_extractor",
                "future_work_extractor",
            ),
            bound_llm,
            prompt_library,
        ),
        RelationBuilder(),
        knowledge_repository,
        document_repository,
        paper_repository,
        event_bus=event_bus,
    )


@pytest.fixture
def evidence_repository(tmp_path: Path) -> JsonEvidenceRepository:
    return JsonEvidenceRepository(tmp_path / "evidence")


@pytest.fixture
def bundle_repository(tmp_path: Path) -> JsonBundleRepository:
    return JsonBundleRepository(tmp_path / "bundles")


@pytest.fixture
def knowledge_retriever(knowledge_repository: JsonKnowledgeRepository) -> LexicalKnowledgeRetriever:
    return LexicalKnowledgeRetriever(knowledge_repository)


@pytest.fixture
def evidence_service(
    knowledge_repository: JsonKnowledgeRepository,
    evidence_repository: JsonEvidenceRepository,
    bundle_repository: JsonBundleRepository,
    document_repository: JsonDocumentRepository,
    knowledge_retriever: LexicalKnowledgeRetriever,
    event_bus: EventBus,
) -> EvidenceIntelligenceService:
    cross_paper = AgreementCrossPaperRetriever(knowledge_retriever)
    return EvidenceIntelligenceService(
        EvidenceIndexer(evidence_repository, event_bus=event_bus),
        EvidenceBundleBuilder(
            knowledge_retriever,
            LinkedEvidenceRetriever(evidence_repository),
            cross_paper,
            ContradictionDetector(),
        ),
        knowledge_repository,
        bundle_repository,
        event_bus=event_bus,
    )
