from __future__ import annotations

import pytest
from pydantic import BaseModel

from researchagent.core.exceptions import OutputParsingError, ProviderUnavailableError
from researchagent.core.interfaces.llm import GenerationParams, Message
from researchagent.services.model_router import ModelRouter
from tests.conftest import FakeLLMProvider


class ExampleResult(BaseModel):
    value: str


async def test_successful_primary_does_not_fallback() -> None:
    primary = FakeLLMProvider()
    fallback = FakeLLMProvider(text="fallback")
    router = ModelRouter(
        primary, fallback, primary_provider="deepseek", fallback_model="llama3.1:8b"
    )

    response = await router.complete(
        [Message.user("go")], model="deepseek-flash", params=GenerationParams()
    )

    assert response.text == "fake response"
    assert not fallback.calls
    assert response.metadata.get("fallback_used") is None


async def test_provider_failure_falls_back_to_ollama() -> None:
    primary = FakeLLMProvider(
        fail_times=1,
        error=ProviderUnavailableError("offline", provider="deepseek"),
    )
    fallback = FakeLLMProvider(text="local")
    router = ModelRouter(
        primary, fallback, primary_provider="deepseek", fallback_model="llama3.1:8b"
    )

    response = await router.complete(
        [Message.user("go")], model="deepseek-flash", params=GenerationParams()
    )

    assert response.text == "local"
    assert response.model == "llama3.1:8b"
    assert response.metadata["fallback_used"] is True


async def test_semantic_output_failure_does_not_fallback() -> None:
    primary = FakeLLMProvider(error=OutputParsingError("bad JSON"), fail_times=1)
    fallback = FakeLLMProvider(text="local")
    router = ModelRouter(
        primary, fallback, primary_provider="deepseek", fallback_model="llama3.1:8b"
    )

    with pytest.raises(OutputParsingError):
        await router.complete(
            [Message.user("go")], model="deepseek-flash", params=GenerationParams()
        )
    assert not fallback.calls


async def test_structured_output_failure_does_not_fallback() -> None:
    primary = FakeLLMProvider(error=OutputParsingError("bad schema"), fail_times=1)
    fallback = FakeLLMProvider(structured=ExampleResult(value="local"))
    router = ModelRouter(
        primary, fallback, primary_provider="deepseek", fallback_model="llama3.1:8b"
    )

    with pytest.raises(OutputParsingError):
        await router.complete_structured_with_usage(
            [Message.user("go")],
            model="deepseek-flash",
            params=GenerationParams(),
            schema=ExampleResult,
        )
    assert not fallback.calls


async def test_both_unavailable_reports_both_providers() -> None:
    unavailable = ProviderUnavailableError("offline")
    primary = FakeLLMProvider(error=unavailable, fail_times=1)
    fallback = FakeLLMProvider(error=unavailable, fail_times=1)
    router = ModelRouter(
        primary, fallback, primary_provider="deepseek", fallback_model="llama3.1:8b"
    )

    with pytest.raises(ProviderUnavailableError, match="Primary and fallback") as caught:
        await router.complete(
            [Message.user("go")], model="deepseek-flash", params=GenerationParams()
        )
    assert caught.value.context["primary_provider"] == "deepseek"
    assert caught.value.context["fallback_model"] == "llama3.1:8b"
