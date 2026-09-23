from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from researchagent.agents.planner.schemas import FramingDraft
from researchagent.core.exceptions import (
    ConfigurationError,
    OutputParsingError,
    ProviderAuthenticationError,
)
from researchagent.core.interfaces.llm import GenerationParams, Message
from researchagent.core.retry import RetryPolicy, retry_async
from researchagent.core.settings import Settings
from researchagent.integrations.deepseek import DeepSeekProvider
from researchagent.integrations.registry import build_llm_provider

FAKE_KEY = "deepseek-test-key-not-real"
MODEL = "deepseek-flash"


class Answer(BaseModel):
    answer: str


VALID_FRAMING = {
    "topic": "DeepSeek model routing for research agents",
    "framing": "Evaluate structured generation reliability and routing behavior.",
    "questions": [
        {
            "question": "How reliably does JSON mode satisfy a nested schema?",
            "rationale": "This determines whether validation repair is needed.",
            "priority": "high",
            "keywords": ["JSON mode", "structured output"],
        }
    ],
}


def _body(
    content: str | None = "hello",
    *,
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 5},
            "prompt_cache_hit_tokens": 5,
            "prompt_cache_miss_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


def _provider(handler: object, attempts: int = 1) -> DeepSeekProvider:
    return DeepSeekProvider(
        FAKE_KEY,
        retry_policy=RetryPolicy(max_attempts=attempts, initial_delay_seconds=0, jitter=False),
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


async def test_successful_response_and_usage_are_normalised() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_body()))

    response = await provider.complete(
        [Message.user("hello")], model=MODEL, params=GenerationParams()
    )

    assert response.provider == "deepseek"
    assert response.model == MODEL
    assert response.usage.total_tokens == 28
    assert response.usage.cached_prompt_tokens == 5
    assert response.usage.uncached_prompt_tokens == 15
    assert response.usage.reasoning_tokens == 3
    await provider.aclose()


async def test_thinking_parameters_use_only_documented_values() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_body())

    provider = _provider(handler)
    await provider.complete(
        [Message.user("hello")],
        model=MODEL,
        params=GenerationParams(thinking=True, reasoning_effort="high", temperature=0.7),
    )

    assert captured["thinking"] == {"type": "enabled"}
    assert captured["reasoning_effort"] == "high"
    assert "temperature" not in captured
    await provider.aclose()


async def test_structured_response_uses_json_object_mode() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_body('{"answer":"grounded"}'))

    provider = _provider(handler)
    result = await provider.complete_structured_with_usage(
        [Message.user("answer")], model=MODEL, params=GenerationParams(), schema=Answer
    )

    assert result.value.answer == "grounded"
    assert result.provider == "deepseek"
    assert captured["response_format"] == {"type": "json_object"}
    await provider.aclose()


async def test_valid_framing_draft_passes_strict_validation() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_body(json.dumps(VALID_FRAMING))))

    result = await provider.complete_structured_with_usage(
        [Message.user("frame the review")],
        model=MODEL,
        params=GenerationParams(),
        schema=FramingDraft,
    )

    assert result.value.topic == VALID_FRAMING["topic"]
    assert result.value.questions[0].priority == "high"
    assert result.metadata["repair_used"] is False
    await provider.aclose()


@pytest.mark.parametrize(
    ("invalid", "field"),
    [
        ({"topic": "A topic", "questions": []}, "framing"),
        ({**VALID_FRAMING, "questions": "not a list"}, "questions"),
    ],
)
async def test_invalid_framing_draft_fails_after_one_repair(
    invalid: dict[str, object], field: str
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_body(json.dumps(invalid)))

    provider = _provider(handler)

    with pytest.raises(OutputParsingError) as caught:
        await provider.complete_structured_with_usage(
            [Message.user("frame the review")],
            model=MODEL,
            params=GenerationParams(),
            schema=FramingDraft,
        )

    assert calls == 2
    assert caught.value.context["model"] == MODEL
    assert caught.value.context["repair_attempted"] is True
    assert caught.value.context["validation_errors"][0]["field"] == field
    assert caught.value.retryable is False
    await provider.aclose()


async def test_wrapped_json_is_repaired_without_local_unwrapping() -> None:
    responses = [json.dumps({"result": VALID_FRAMING}), json.dumps(VALID_FRAMING)]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        content = responses[calls]
        calls += 1
        return httpx.Response(200, json=_body(content))

    provider = _provider(handler)
    result = await provider.complete_structured_with_usage(
        [Message.user("frame the review")],
        model=MODEL,
        params=GenerationParams(),
        schema=FramingDraft,
    )

    assert calls == 2
    assert result.value.framing == VALID_FRAMING["framing"]
    assert result.metadata["repair_used"] is True
    assert result.metadata["attempts"] == 2
    assert result.usage.total_tokens == 56
    await provider.aclose()


async def test_fenced_json_uses_model_repair_without_local_string_cleanup() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = (
            f"```json\n{json.dumps(VALID_FRAMING)}\n```"
            if calls == 1
            else json.dumps(VALID_FRAMING)
        )
        return httpx.Response(
            200,
            json=_body(content),
        )

    provider = _provider(handler)
    result = await provider.complete_structured_with_usage(
        [Message.user("frame the review")],
        model=MODEL,
        params=GenerationParams(),
        schema=FramingDraft,
    )

    assert calls == 2
    assert result.value.topic == VALID_FRAMING["topic"]
    assert result.metadata["repair_used"] is True
    await provider.aclose()


async def test_nonempty_malformed_json_uses_one_repair_and_succeeds() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = '{"topic":"broken"' if len(requests) == 1 else json.dumps(VALID_FRAMING)
        return httpx.Response(200, json=_body(content))

    provider = _provider(handler)
    result = await provider.complete_structured_with_usage(
        [Message.user("frame the review")],
        model=MODEL,
        params=GenerationParams(),
        schema=FramingDraft,
    )

    repair_messages = requests[1]["messages"]
    assert isinstance(repair_messages, list)
    feedback = repair_messages[-1]["content"]
    assert "JSON parse error" in feedback
    assert '"line":1' in feedback
    assert result.value.framing == VALID_FRAMING["framing"]
    assert result.metadata["repair_used"] is True
    assert len(requests) == 2
    await provider.aclose()


async def test_malformed_repair_fails_clearly_and_is_not_retried_again() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_body('{"topic":"still broken"'))

    provider = _provider(handler)

    async def operation() -> object:
        return await provider.complete_structured_with_usage(
            [Message.user("frame the review")],
            model=MODEL,
            params=GenerationParams(),
            schema=FramingDraft,
        )

    with pytest.raises(OutputParsingError, match="malformed JSON") as caught:
        await retry_async(
            operation,
            RetryPolicy(max_attempts=3, initial_delay_seconds=0, jitter=False),
            operation_name="agent.planner",
        )

    assert calls == 2
    assert caught.value.context["repair_attempted"] is True
    assert caught.value.context["json_parse_error"]["type"] == "JSONDecodeError"
    assert caught.value.context["json_parse_error"]["line"] == 1
    assert caught.value.context["json_parse_error"]["column"] > 1
    assert caught.value.context["finish_reason"] == "stop"
    assert caught.value.context["assistant_content_present"] is True
    assert caught.value.context["reasoning_content_present"] is False
    assert caught.value.context["truncated"] is False
    assert caught.value.retryable is False
    await provider.aclose()


@pytest.mark.parametrize(("content", "present"), [("", True), (None, False)])
async def test_empty_final_content_is_clear_and_does_not_start_repair(
    content: str | None, present: bool
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json=_body(content, reasoning_content="private reasoning exists"),
        )

    provider = _provider(handler)

    with pytest.raises(OutputParsingError, match="no final assistant content") as caught:
        await provider.complete_structured_with_usage(
            [Message.user("frame the review")],
            model=MODEL,
            params=GenerationParams(thinking=True),
            schema=FramingDraft,
        )

    assert calls == 1
    assert caught.value.context["raw_length"] == 0
    assert caught.value.context["assistant_content_present"] is present
    assert caught.value.context["assistant_content_nonempty"] is False
    assert caught.value.context["reasoning_content_present"] is True
    assert caught.value.context["repair_attempted"] is False
    await provider.aclose()


async def test_thinking_response_uses_documented_final_content_field() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json=_body(
                json.dumps(VALID_FRAMING),
                reasoning_content="This is reasoning, not the final JSON.",
            ),
        )

    provider = _provider(handler)

    result = await provider.complete_structured_with_usage(
        [Message.user("frame the review")],
        model=MODEL,
        params=GenerationParams(thinking=True, reasoning_effort="high", max_output_tokens=2048),
        schema=FramingDraft,
    )

    assert result.value.topic == VALID_FRAMING["topic"]
    assert result.metadata["repair_used"] is False
    assert captured["thinking"] == {"type": "enabled"}
    assert captured["reasoning_effort"] == "high"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 2048
    await provider.aclose()


async def test_finish_reason_length_is_observable_across_repair() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json=_body('{"topic":"cut off"', finish_reason="length"),
            )
        return httpx.Response(200, json=_body(json.dumps(VALID_FRAMING)))

    provider = _provider(handler)
    result = await provider.complete_structured_with_usage(
        [Message.user("frame the review")],
        model=MODEL,
        params=GenerationParams(max_output_tokens=2048),
        schema=FramingDraft,
    )

    assert result.metadata["initial_finish_reason"] == "length"
    assert result.metadata["initial_truncated"] is True
    assert result.metadata["finish_reason"] == "stop"
    assert result.metadata["truncated"] is False
    await provider.aclose()


async def test_repair_request_contains_field_level_validation_feedback() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = (
            json.dumps({"topic": "A topic", "questions": []})
            if len(requests) == 1
            else json.dumps(VALID_FRAMING)
        )
        return httpx.Response(200, json=_body(content))

    provider = _provider(handler)
    result = await provider.complete_structured_with_usage(
        [Message.user("frame the review")],
        model=MODEL,
        params=GenerationParams(),
        schema=FramingDraft,
    )

    repair_messages = requests[1]["messages"]
    assert isinstance(repair_messages, list)
    feedback = repair_messages[-1]["content"]
    assert "Validation errors" in feedback
    assert '"field":"framing"' in feedback
    assert "FramingDraft" in feedback
    assert result.value.framing == VALID_FRAMING["framing"]
    assert result.metadata["repair_used"] is True
    await provider.aclose()


async def test_validation_diagnostics_are_bounded_and_redact_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchagent.integrations.deepseek import provider as provider_module

    class CaptureLogger:
        def __init__(self) -> None:
            self.entries: list[tuple[str, str, dict[str, object]]] = []

        def warning(self, event: str, **values: object) -> None:
            self.entries.append(("warning", event, values))

        def debug(self, event: str, **values: object) -> None:
            self.entries.append(("debug", event, values))

        def info(self, event: str, **values: object) -> None:
            self.entries.append(("info", event, values))

    capture = CaptureLogger()
    monkeypatch.setattr(provider_module, "logger", capture)
    raw_marker = "sk-" + "redaction-test-123456789"
    invalid = json.dumps(
        {
            "topic": "A topic",
            "questions": [],
            "api_key": raw_marker,
            "padding": "x" * 1000,
        }
    )
    provider = _provider(lambda request: httpx.Response(200, json=_body(invalid)))

    with pytest.raises(OutputParsingError):
        await provider.complete_structured_with_usage(
            [Message.user("frame the review")],
            model=MODEL,
            params=GenerationParams(),
            schema=FramingDraft,
        )

    warnings = [entry for entry in capture.entries if entry[0] == "warning"]
    previews = [entry[2]["raw_preview"] for entry in capture.entries if "raw_preview" in entry[2]]
    assert warnings[0][2]["provider"] == "deepseek"
    assert warnings[0][2]["model"] == MODEL
    assert warnings[0][2]["schema"] == "FramingDraft"
    assert warnings[0][2]["validation_errors"][0]["field"] == "framing"
    assert previews
    assert all(raw_marker not in preview for preview in previews)
    assert all("[REDACTED]" in preview for preview in previews)
    assert all(len(preview) <= 800 for preview in previews)
    await provider.aclose()


async def test_retriable_server_failure_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, json={} if calls == 1 else _body())

    provider = _provider(handler, attempts=2)
    response = await provider.complete(
        [Message.user("hello")], model=MODEL, params=GenerationParams()
    )

    assert calls == 2
    assert response.metadata["attempts"] == 2
    await provider.aclose()


async def test_provider_construction_reads_only_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    settings = Settings(_env_file=None)
    provider = build_llm_provider("deepseek", settings)

    assert isinstance(provider, DeepSeekProvider)
    await provider.aclose()


def test_missing_key_is_a_clear_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(_env_file=None, deepseek_api_key=None)

    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        build_llm_provider("deepseek", settings)


async def test_secret_never_appears_in_provider_error() -> None:
    provider = _provider(
        lambda request: httpx.Response(401, json={"error": {"message": "bad key"}})
    )

    with pytest.raises(ProviderAuthenticationError) as caught:
        await provider.complete([Message.user("hello")], model=MODEL, params=GenerationParams())

    assert FAKE_KEY not in f"{caught.value!s}{caught.value!r}{caught.value.context}"
    await provider.aclose()
