"""Unit tests for the Ollama adapter — no server required.

The LangChain chat model is stubbed; what is under test is our translation of
messages, usage metadata and failures, not LangChain itself.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ollama import ResponseError
from pydantic import BaseModel

from researchagent.core.exceptions import (
    OutputParsingError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from researchagent.core.interfaces.llm import GenerationParams, Message
from researchagent.integrations.ollama.provider import OllamaProvider

PARAMS = GenerationParams(temperature=0.0)


class Plan(BaseModel):
    goal: str
    steps: list[str]


class StubChat:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.received: list[Any] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.received.append(messages)
        if self.error is not None:
            raise self.error
        return self.result

    def with_structured_output(
        self, schema: type[BaseModel], *, include_raw: bool = False
    ) -> StubChat:
        return self


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider("http://localhost:11434", request_timeout_seconds=5.0)


def stub(provider: OllamaProvider, chat: StubChat) -> None:
    provider._chat_model = lambda model, params: chat  # type: ignore[assignment,method-assign]


async def test_complete_maps_usage_metadata(provider: OllamaProvider) -> None:
    message = AIMessage(content="hello there")
    message.usage_metadata = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
    stub(provider, StubChat(result=message))

    response = await provider.complete([Message.user("hi")], model="qwen3:8b", params=PARAMS)

    assert response.text == "hello there"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 3
    assert response.provider == "ollama"
    assert response.latency_ms >= 0


async def test_complete_falls_back_to_eval_counts(provider: OllamaProvider) -> None:
    message = AIMessage(
        content="ok",
        response_metadata={"prompt_eval_count": 7, "eval_count": 2, "done_reason": "stop"},
    )
    stub(provider, StubChat(result=message))

    response = await provider.complete([Message.user("hi")], model="m", params=PARAMS)

    assert response.usage.total_tokens == 9
    assert response.metadata["done_reason"] == "stop"


async def test_roles_map_to_langchain_message_types(provider: OllamaProvider) -> None:
    chat = StubChat(result=AIMessage(content="ok"))
    stub(provider, chat)

    await provider.complete(
        [Message.system("you are a planner"), Message.user("plan this")],
        model="m",
        params=PARAMS,
    )

    sent = chat.received[0]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[1], HumanMessage)


async def test_empty_message_list_is_rejected(provider: OllamaProvider) -> None:
    stub(provider, StubChat(result=AIMessage(content="ok")))

    with pytest.raises(ProviderError):
        await provider.complete([], model="m", params=PARAMS)


async def test_structured_output_returns_the_schema(provider: OllamaProvider) -> None:
    stub(provider, StubChat(result=Plan(goal="g", steps=["a"])))

    plan = await provider.complete_structured(
        [Message.user("plan")], model="m", params=PARAMS, schema=Plan
    )

    assert plan.goal == "g"


async def test_invalid_structured_output_is_retryable(provider: OllamaProvider) -> None:
    stub(provider, StubChat(result={"goal": "g"}))  # `steps` missing

    with pytest.raises(OutputParsingError) as excinfo:
        await provider.complete_structured(
            [Message.user("plan")], model="m", params=PARAMS, schema=Plan
        )

    assert excinfo.value.retryable is True


async def test_missing_model_tag_becomes_provider_unavailable(provider: OllamaProvider) -> None:
    stub(provider, StubChat(error=ResponseError("model not found", status_code=404)))

    with pytest.raises(ProviderUnavailableError) as excinfo:
        await provider.complete([Message.user("hi")], model="qwen3:8b", params=PARAMS)

    assert excinfo.value.remedy == "ollama pull qwen3:8b"
    assert excinfo.value.retryable is True


async def test_connection_failure_becomes_provider_unavailable(provider: OllamaProvider) -> None:
    stub(provider, StubChat(error=httpx.ConnectError("connection refused")))

    with pytest.raises(ProviderUnavailableError):
        await provider.complete([Message.user("hi")], model="m", params=PARAMS)


async def test_timeout_becomes_provider_timeout(provider: OllamaProvider) -> None:
    stub(provider, StubChat(error=TimeoutError()))

    with pytest.raises(ProviderTimeoutError):
        await provider.complete([Message.user("hi")], model="m", params=PARAMS)


async def test_unexpected_error_is_wrapped_not_leaked(provider: OllamaProvider) -> None:
    stub(provider, StubChat(error=ValueError("something odd")))

    with pytest.raises(ProviderError) as excinfo:
        await provider.complete([Message.user("hi")], model="m", params=PARAMS)

    assert excinfo.value.context["error_type"] == "ValueError"


async def test_health_reports_unreachable_server(provider: OllamaProvider) -> None:
    async def boom() -> list[str]:
        raise ProviderUnavailableError("down", base_url="http://localhost:11434")

    provider._admin.list_models = boom  # type: ignore[method-assign]

    health = await provider.health()

    assert health.healthy is False
    assert health.provider == "ollama"


async def test_chat_models_are_cached_per_config() -> None:
    provider = OllamaProvider("http://localhost:11434")

    first = provider._chat_model("m", PARAMS)
    same = provider._chat_model("m", PARAMS)
    other = provider._chat_model("m", GenerationParams(temperature=0.9))

    assert first is same
    assert first is not other
    await provider.aclose()
