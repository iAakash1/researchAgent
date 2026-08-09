from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from researchagent.agents.registry import build_agent
from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import (
    AgentConfig,
    AgentSpec,
    DocumentsConfig,
    ModelCatalog,
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
    TokenUsage,
    TSchema,
)
from researchagent.core.prompts import PromptLibrary
from researchagent.core.settings import Environment, Settings
from researchagent.integrations.manual import ManualPaperSource
from researchagent.integrations.pymupdf import PyMuPDFLoader
from researchagent.memory.checkpoints import build_checkpointer
from researchagent.repositories.document_repository import JsonDocumentRepository
from researchagent.repositories.paper_repository import JsonPaperRepository
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
from researchagent.services.llm_service import BoundLLM, LLMService
from researchagent.services.ranking import HeuristicScorer
from researchagent.services.retrieval_service import RetrievalService
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
    ) -> None:
        self.text = text
        self.structured = structured
        # Consumed in order, so a multi-phase agent can be driven phase by phase.
        self.structured_sequence = list(structured_sequence or [])
        self.fail_times = fail_times
        self.error = error
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Container:
    """A fully wired container. The LLM is faked; the only paper source is the real
    manual library, so no test ever touches a remote index."""
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
    graph = build_research_graph(
        planner=planner,
        discovery=discovery_service,
        documents=document_service,
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
