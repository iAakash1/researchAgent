"""Groq adapter tests.

Every request is served by an httpx MockTransport, so the suite never touches the network
and never needs a key. That is the point: the project is local-first, and `uv run pytest`
must pass on a laptop with no internet and no Groq account.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from researchagent.core.exceptions import (
    ConfigurationError,
    OutputParsingError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from researchagent.core.interfaces.llm import GenerationParams, Message
from researchagent.core.retry import RetryPolicy
from researchagent.core.settings import Settings
from researchagent.integrations.groq import GroqProvider
from researchagent.integrations.registry import build_llm_provider

FAKE_KEY = "test-key-not-a-real-credential"
MODEL = "openai/gpt-oss-120b"


class Finding(BaseModel):
    name: str
    score: float


def _chat_body(content: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def _provider(handler: object, *, attempts: int = 1) -> GroqProvider:
    return GroqProvider(
        api_key=FAKE_KEY,
        retry_policy=RetryPolicy(max_attempts=attempts, initial_delay_seconds=0.0, jitter=False),
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


async def test_completion_normalises_into_the_port_response() -> None:
    """The domain sees CompletionResponse, never a Groq payload."""
    provider = _provider(lambda request: httpx.Response(200, json=_chat_body("hello")))

    response = await provider.complete([Message.user("hi")], model=MODEL, params=GenerationParams())

    assert response.text == "hello"
    assert response.provider == "groq"
    assert response.model == MODEL
    assert response.usage.total_tokens == 18
    assert response.latency_ms > 0.0, "latency must be reported in milliseconds"
    await provider.aclose()


async def test_request_carries_the_key_only_in_the_authorization_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content.decode()
        return httpx.Response(200, json=_chat_body("ok"))

    provider = _provider(handler)
    await provider.complete([Message.user("hi")], model=MODEL, params=GenerationParams())

    assert seen["auth"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in str(seen["body"])
    await provider.aclose()


async def test_generation_params_map_onto_the_openai_surface() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_body("ok"))

    provider = _provider(handler)
    await provider.complete(
        [Message.system("s"), Message.user("u")],
        model=MODEL,
        params=GenerationParams(temperature=0.7, top_p=0.9, seed=42, max_output_tokens=64),
    )

    assert captured["temperature"] == 0.7
    assert captured["seed"] == 42
    assert captured["max_completion_tokens"] == 64
    assert captured["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    await provider.aclose()


async def test_structured_output_validates_against_the_caller_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_body('{"name": "BERT", "score": 0.91}'))

    provider = _provider(handler)
    result = await provider.complete_structured(
        [Message.user("extract")], model=MODEL, params=GenerationParams(), schema=Finding
    )

    assert result == Finding(name="BERT", score=0.91)
    schema = captured["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    assert schema["additionalProperties"] is False, "strict mode rejects open objects"
    assert sorted(schema["required"]) == ["name", "score"]
    await provider.aclose()


async def test_schema_violating_json_raises_a_retryable_parsing_error() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_chat_body('{"name": "x"}')))

    with pytest.raises(OutputParsingError) as caught:
        await provider.complete_structured(
            [Message.user("go")], model=MODEL, params=GenerationParams(), schema=Finding
        )

    assert caught.value.retryable, "a resample can satisfy the schema"
    await provider.aclose()


async def test_malformed_json_body_raises_a_parsing_error() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(OutputParsingError):
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())
    await provider.aclose()


async def test_response_without_choices_raises_rather_than_returning_empty_text() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"usage": {}}))

    with pytest.raises(OutputParsingError):
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())
    await provider.aclose()


@pytest.mark.parametrize("status", [401, 403])
async def test_authentication_failure_is_fatal_and_never_retried(status: int) -> None:
    """Retrying a rejected key burns rate limit and cannot succeed."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "Invalid API Key"}})

    provider = _provider(handler, attempts=3)

    with pytest.raises(ProviderAuthenticationError) as caught:
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())

    assert calls == 1
    assert not caught.value.retryable
    await provider.aclose()


async def test_rate_limit_is_retryable_and_succeeds_on_a_later_attempt() -> None:
    responses = [
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json=_chat_body("recovered")),
    ]
    provider = _provider(lambda request: responses.pop(0), attempts=3)

    response = await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())

    assert response.text == "recovered"
    assert response.metadata["attempts"] == 2
    await provider.aclose()


@pytest.mark.parametrize("status", [429, 413])
async def test_exhausted_token_budget_surfaces_a_retryable_error(status: int) -> None:
    """413 is how Groq reports a tokens-per-minute overage, not a malformed request."""
    provider = _provider(lambda request: httpx.Response(status, json={}), attempts=2)

    with pytest.raises(ProviderRateLimitedError) as caught:
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())

    assert caught.value.retryable
    assert "max_output_tokens" in (caught.value.remedy or "")
    await provider.aclose()


async def test_unknown_model_names_the_remedy() -> None:
    provider = _provider(
        lambda request: httpx.Response(404, json={"error": {"message": "model not found"}})
    )

    with pytest.raises(ProviderUnavailableError) as caught:
        await provider.complete(
            [Message.user("go")], model="gpt-oss-9000", params=GenerationParams()
        )

    assert "config/models.yaml" in (caught.value.remedy or "")
    await provider.aclose()


async def test_server_error_is_provider_unavailable() -> None:
    provider = _provider(lambda request: httpx.Response(503, json={}), attempts=1)

    with pytest.raises(ProviderUnavailableError):
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())
    await provider.aclose()


async def test_client_error_stays_a_plain_provider_error() -> None:
    provider = _provider(
        lambda request: httpx.Response(400, json={"error": {"message": "bad request"}})
    )

    with pytest.raises(ProviderError) as caught:
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())

    assert not isinstance(caught.value, ProviderAuthenticationError)
    await provider.aclose()


async def test_timeout_maps_to_the_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = _provider(handler, attempts=1)

    with pytest.raises(ProviderTimeoutError):
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())
    await provider.aclose()


async def test_network_failure_maps_to_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    provider = _provider(handler, attempts=1)

    with pytest.raises(ProviderUnavailableError):
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())
    await provider.aclose()


async def test_errors_never_carry_the_api_key() -> None:
    """Secret redaction: the key must not reach a message, context or serialised error."""
    provider = _provider(
        lambda request: httpx.Response(401, json={"error": {"message": "Invalid API Key"}}),
        attempts=1,
    )

    with pytest.raises(ProviderAuthenticationError) as caught:
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())

    rendered = f"{caught.value!s}{caught.value!r}{caught.value.context}"
    assert FAKE_KEY not in rendered
    assert "Bearer" not in rendered
    await provider.aclose()


async def test_streaming_yields_content_deltas_only() -> None:
    stream = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    provider = _provider(lambda request: httpx.Response(200, text=stream))

    chunks = [
        chunk
        async for chunk in provider.stream(
            [Message.user("go")], model=MODEL, params=GenerationParams()
        )
    ]

    assert "".join(chunks) == "Hello"
    await provider.aclose()


async def test_health_reports_unhealthy_instead_of_raising() -> None:
    provider = _provider(lambda request: httpx.Response(401, json={}))

    health = await provider.health()

    assert health.healthy is False
    assert health.detail == "credentials rejected"
    await provider.aclose()


async def test_health_lists_available_models_when_reachable() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"data": [{"id": MODEL}]}))

    health = await provider.health()

    assert health.healthy is True
    assert MODEL in health.available_models
    await provider.aclose()


def test_constructing_without_a_key_is_rejected_at_the_boundary() -> None:
    with pytest.raises(ProviderAuthenticationError):
        GroqProvider(api_key="")


class TestProviderSelection:
    """Configuration decides the provider; the domain never does."""

    def test_ollama_remains_the_default(self) -> None:
        provider = build_llm_provider("ollama", Settings())
        assert provider.name == "ollama"

    def test_groq_without_a_key_fails_loudly_rather_than_falling_back(self) -> None:
        settings = Settings(groq_api_key=None)

        with pytest.raises(ConfigurationError) as caught:
            build_llm_provider("groq", settings)

        assert "GROQ_API_KEY" in caught.value.message
        assert caught.value.remedy is not None

    def test_groq_is_selectable_when_a_key_is_present(self) -> None:
        provider = build_llm_provider("groq", Settings(groq_api_key=FAKE_KEY))
        assert provider.name == "groq"


async def test_retry_waits_at_least_as_long_as_the_provider_asked() -> None:
    """A backoff shorter than the advertised retry-after guarantees a wasted attempt."""
    from researchagent.core.retry import _delay_for

    provider = _provider(
        lambda request: httpx.Response(429, json={}, headers={"retry-after": "4"}), attempts=1
    )

    with pytest.raises(ProviderRateLimitedError) as caught:
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())

    assert caught.value.retry_after_seconds == 4.0
    policy = RetryPolicy(max_attempts=3, initial_delay_seconds=0.1, jitter=False)
    assert _delay_for(policy, 2, caught.value) == 4.0
    await provider.aclose()


async def test_absent_retry_after_leaves_the_backoff_curve_alone() -> None:
    from researchagent.core.retry import _delay_for

    provider = _provider(lambda request: httpx.Response(429, json={}), attempts=1)

    with pytest.raises(ProviderRateLimitedError) as caught:
        await provider.complete([Message.user("go")], model=MODEL, params=GenerationParams())

    assert caught.value.retry_after_seconds is None
    policy = RetryPolicy(max_attempts=3, initial_delay_seconds=0.1, jitter=False)
    assert _delay_for(policy, 2, caught.value) == pytest.approx(0.1)
    await provider.aclose()
