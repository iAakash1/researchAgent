from __future__ import annotations

import pytest

from researchagent.config.schemas import ModelCatalog
from researchagent.core.events import Event, EventBus, EventType
from researchagent.core.interfaces.llm import GenerationParams, Message
from researchagent.core.registry import RegistryError
from researchagent.core.settings import Settings
from researchagent.services.llm_service import BoundLLM, LLMService
from tests.conftest import FakeLLMProvider


def test_get_resolves_default_alias(llm_service: LLMService, model_catalog: ModelCatalog) -> None:
    handle = llm_service.get()

    assert handle.alias == model_catalog.default
    assert handle.model == model_catalog.spec_for(model_catalog.default).model_name


def test_get_unknown_alias_raises(llm_service: LLMService) -> None:
    with pytest.raises(KeyError):
        llm_service.get("does-not-exist")


async def test_complete_uses_alias_params(fake_provider: FakeLLMProvider) -> None:
    catalog = ModelCatalog.model_validate(
        {
            "default": "extraction",
            "models": {"extraction": {"model": "qwen3:8b", "params": {"temperature": 0.0}}},
        }
    )
    handle = BoundLLM("extraction", catalog.spec_for("extraction"), fake_provider)

    response = await handle.complete([Message.user("hello")])

    assert response.text == "fake response"
    assert response.usage.total_tokens == 15


async def test_param_override_merges_over_alias_defaults(fake_provider: FakeLLMProvider) -> None:
    catalog = ModelCatalog.model_validate(
        {
            "default": "a",
            "models": {"a": {"model": "m", "params": {"temperature": 0.9, "top_p": 0.5}}},
        }
    )
    handle = BoundLLM("a", catalog.spec_for("a"), fake_provider)

    merged = handle.spec.params.merged_with(GenerationParams(temperature=0.0))

    assert merged.temperature == 0.0
    assert merged.top_p == 0.5  # untouched by the override


async def test_completion_publishes_an_event(fake_provider: FakeLLMProvider) -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.LLM_CALL_COMPLETED, handler)

    catalog = ModelCatalog.model_validate({"default": "a", "models": {"a": {"model": "m"}}})
    handle = BoundLLM("a", catalog.spec_for("a"), fake_provider, event_bus=bus)
    await handle.complete([Message.user("hi")])

    assert received[0].payload["alias"] == "a"
    assert received[0].payload["prompt_tokens"] == 10


async def test_verify_models_available_flags_unpulled_models(
    settings: Settings, fake_provider: FakeLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = ModelCatalog.model_validate(
        {
            "default": "here",
            "models": {"here": {"model": "fake-model"}, "missing": {"model": "not-pulled"}},
        }
    )
    service = LLMService(catalog, settings)
    monkeypatch.setattr(service, "_provider", lambda _name: fake_provider)

    assert await service.verify_models_available() == {"here": True, "missing": False}


async def test_aclose_closes_every_provider(
    settings: Settings, fake_provider: FakeLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = ModelCatalog.model_validate({"default": "a", "models": {"a": {"model": "m"}}})
    service = LLMService(catalog, settings)
    monkeypatch.setattr(service, "_provider", lambda _name: fake_provider)

    service.get("a")
    service._providers["fake"] = fake_provider
    await service.aclose()

    assert fake_provider.closed is True


def test_unknown_provider_name_is_a_registry_error(settings: Settings) -> None:
    catalog = ModelCatalog.model_validate(
        {"default": "a", "models": {"a": {"provider": "vllm", "model": "m"}}}
    )

    with pytest.raises(RegistryError):
        LLMService(catalog, settings).get("a")
