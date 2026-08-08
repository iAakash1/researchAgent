from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, AgentSpec, ModelCatalog
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
from researchagent.core.settings import Environment, Settings
from researchagent.services.llm_service import BoundLLM, LLMService

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
        fail_times: int = 0,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.structured = structured
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
        if self.structured is None:
            raise AssertionError("FakeLLMProvider needs `structured=` for structured calls")
        return schema.model_validate(self.structured.model_dump())

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
