from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from researchagent.agents.registry import build_agent
from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, AgentSpec, ModelCatalog, WorkflowConfig
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
from researchagent.memory.checkpoints import build_checkpointer
from researchagent.services.llm_service import BoundLLM, LLMService
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
    monkeypatch: pytest.MonkeyPatch,
) -> Container:
    """A fully wired container whose only fake is the LLM provider."""
    service = LLMService(model_catalog, settings, event_bus=event_bus)
    monkeypatch.setattr(service, "_provider", lambda _name: fake_provider)

    workflow_config = config_loader.load("workflow", WorkflowConfig)
    planner = build_agent(
        "planner",
        agent_config=agent_config,
        llm_service=service,
        prompts=prompt_library,
        event_bus=event_bus,
    )
    graph = build_research_graph(
        planner=planner, checkpointer=build_checkpointer(workflow_config.checkpointer)
    )

    return Container(
        settings=settings,
        config_loader=config_loader,
        model_catalog=model_catalog,
        agent_config=agent_config,
        workflow_config=workflow_config,
        prompt_library=prompt_library,
        event_bus=event_bus,
        llm_service=service,
        workflow_runner=WorkflowRunner(graph, workflow_config),
    )
