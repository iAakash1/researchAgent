"""Groq adapter for the :class:`LLMProvider` port.

Optional external inference. The project stays local-first: Ollama is the default and
nothing here is reachable unless a model alias explicitly names ``provider: groq``.

Groq exposes an OpenAI-compatible surface, so this talks plain HTTP rather than pulling in
another SDK. The API key lives only in this module's memory, is passed only in the
Authorization header, and is never placed in a log field, an exception message, or an
exception's context — see :func:`_fail`.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from researchagent.core.constants import SECONDS_PER_MILLISECOND
from researchagent.core.exceptions import (
    OutputParsingError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from researchagent.core.interfaces.llm import (
    CompletionResponse,
    GenerationParams,
    LLMProvider,
    Message,
    ProviderHealth,
    TokenUsage,
    TSchema,
)
from researchagent.core.logging import get_logger
from researchagent.core.retry import RetryPolicy, retry_async

logger = get_logger(__name__)

_AUTH_STATUSES = frozenset({401, 403})
_MISSING_MODEL_STATUSES = frozenset({404})
_RATE_LIMIT_STATUSES = frozenset({413, 429})


class GroqProvider(LLMProvider):
    """Chat completions through Groq's OpenAI-compatible API.

    ``model`` is supplied by the caller from ``config/models.yaml`` — this class never
    names a model, so moving from ``openai/gpt-oss-120b`` to whatever Groq offers next is
    a YAML edit.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        request_timeout_seconds: float = 120.0,
        retry_policy: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ProviderAuthenticationError(
                "Groq provider constructed without an API key", provider=self.name
            )
        self._base_url = base_url.rstrip("/")
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=3)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(request_timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    async def complete(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> CompletionResponse:
        payload = _payload(messages, model=model, params=params)
        started = time.perf_counter()
        body, attempts = await self._post_with_retry("/chat/completions", payload, model=model)
        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND

        text = _first_choice_text(body, model=model)
        usage = _usage(body)
        logger.info(
            "groq_completion",
            provider=self.name,
            model=model,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            attempts=attempts,
            success=True,
        )
        return CompletionResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            metadata={"attempts": attempts, "finish_reason": _finish_reason(body)},
        )

    def stream(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> AsyncIterator[str]:
        return self._stream(messages, model=model, params=params)

    async def _stream(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> AsyncIterator[str]:
        payload = _payload(messages, model=model, params=params) | {"stream": True}
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= httpx.codes.BAD_REQUEST:
                    await response.aread()
                    _fail(response, model=model)
                async for line in response.aiter_lines():
                    chunk = _sse_delta(line)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Groq streaming request timed out", provider=self.name, model=model
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Could not reach Groq", provider=self.name, model=model, reason=type(exc).__name__
            ) from exc

    async def complete_structured(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> TSchema:
        payload = _payload(messages, model=model, params=params) | {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": _strict_schema(schema),
                    "strict": True,
                },
            }
        }
        body, _ = await self._post_with_retry("/chat/completions", payload, model=model)
        text = _first_choice_text(body, model=model)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            # Retryable by taxonomy: a resample often satisfies the schema.
            raise OutputParsingError(
                "Groq returned JSON that does not satisfy the schema",
                provider=self.name,
                model=model,
                schema=schema.__name__,
                issues=exc.error_count(),
            ) from exc
        except ValueError as exc:
            raise OutputParsingError(
                "Groq returned malformed JSON",
                provider=self.name,
                model=model,
                schema=schema.__name__,
            ) from exc

    async def health(self) -> ProviderHealth:
        """Cheap liveness probe. Must not raise — a dead provider is a reported state."""
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            models = [entry["id"] for entry in response.json().get("data", [])]
        except httpx.HTTPStatusError as exc:
            detail = (
                "credentials rejected"
                if exc.response.status_code in _AUTH_STATUSES
                else f"http {exc.response.status_code}"
            )
            return ProviderHealth(provider=self.name, healthy=False, detail=detail)
        except httpx.HTTPError as exc:
            return ProviderHealth(provider=self.name, healthy=False, detail=type(exc).__name__)
        return ProviderHealth(provider=self.name, healthy=True, available_models=models)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any], *, model: str
    ) -> tuple[dict[str, Any], int]:
        async def attempt() -> dict[str, Any]:
            try:
                response = await self._client.post(path, json=payload)
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(
                    "Groq request timed out", provider=self.name, model=model
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderUnavailableError(
                    "Could not reach Groq",
                    provider=self.name,
                    model=model,
                    reason=type(exc).__name__,
                ) from exc
            if response.status_code >= httpx.codes.BAD_REQUEST:
                _fail(response, model=model)
            try:
                parsed: dict[str, Any] = response.json()
            except ValueError as exc:
                raise OutputParsingError(
                    "Groq returned a non-JSON response body", provider=self.name, model=model
                ) from exc
            return parsed

        # retry_async re-raises non-retryable errors immediately, so an authentication
        # failure costs one request rather than three.
        return await retry_async(attempt, self._retry_policy, operation_name="groq_chat")


def _fail(response: httpx.Response, *, model: str) -> None:
    """Translate an error response into the project taxonomy.

    Only the status code and the provider's own message are propagated. Request headers —
    which carry the bearer token — are never touched.
    """
    status = response.status_code
    detail = _error_message(response)

    if status in _AUTH_STATUSES:
        raise ProviderAuthenticationError(
            "Groq rejected the credentials", provider="groq", status=status
        )
    # Groq reports a tokens-per-minute overage as 413, not 429: the request is "too
    # large" only relative to what is left of this minute's budget. Both are the same
    # condition to a caller, and both clear on their own, so both are retryable — with a
    # remedy that names the config fix for the case where the budget is never enough.
    if status in _RATE_LIMIT_STATUSES:
        raise ProviderRateLimitedError(
            "Groq token budget exhausted",
            provider="groq",
            model=model,
            status=status,
            retry_after=response.headers.get("retry-after"),
            detail=detail,
            remedy=(
                "Wait for the per-minute budget to reset; if it never fits, lower "
                "max_output_tokens for this alias in config/models.yaml"
            ),
        )
    if status in _MISSING_MODEL_STATUSES:
        raise ProviderUnavailableError(
            "Groq does not serve this model",
            provider="groq",
            model=model,
            remedy="Update the model id in config/models.yaml",
            detail=detail,
        )
    if status >= httpx.codes.INTERNAL_SERVER_ERROR:
        raise ProviderUnavailableError(
            "Groq is unavailable", provider="groq", model=model, status=status, detail=detail
        )
    raise ProviderError(
        "Groq rejected the request", provider="groq", model=model, status=status, detail=detail
    )


def _error_message(response: httpx.Response) -> str:
    """The provider's own message, never the request that produced it."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", ""))[:200]
    return str(error or "")[:200]


def _payload(messages: list[Message], *, model: str, params: GenerationParams) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": message.role.value, "content": message.content} for message in messages
        ],
        "temperature": params.temperature,
    }
    optional = {
        "top_p": params.top_p,
        "seed": params.seed,
        "max_completion_tokens": params.max_output_tokens,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    if params.stop:
        payload["stop"] = params.stop
    return payload


def _first_choice_text(body: dict[str, Any], *, model: str) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OutputParsingError("Groq response contained no choices", provider="groq", model=model)
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise OutputParsingError(
            "Groq response contained no message content", provider="groq", model=model
        )
    return content


def _finish_reason(body: dict[str, Any]) -> str | None:
    choices = body.get("choices") or [{}]
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) else None


def _usage(body: dict[str, Any]) -> TokenUsage:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
    )


def _sse_delta(line: str) -> str:
    if not line.startswith("data: "):
        return ""
    data = line.removeprefix("data: ").strip()
    if not data or data == "[DONE]":
        return ""
    try:
        chunk = json.loads(data)
    except ValueError:
        return ""
    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
    return delta if isinstance(delta, str) else ""


def _strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Pydantic's JSON schema, tightened to what strict structured output requires.

    Strict mode rejects open objects and treats every declared property as required, so
    optional fields become nullable rather than absent. The alternative — free-form JSON
    plus a prayer — is what the grounding layer exists to compensate for; better to
    constrain the decoder.
    """
    tightened = _tighten(schema.model_json_schema())
    assert isinstance(tightened, dict)  # noqa: S101 - _tighten preserves the root type
    return tightened


def _tighten(node: Any) -> Any:
    if isinstance(node, list):
        return [_tighten(item) for item in node]
    if not isinstance(node, dict):
        return node

    tightened = {key: _tighten(value) for key, value in node.items()}
    if tightened.get("type") == "object" and "properties" in tightened:
        tightened["additionalProperties"] = False
        tightened["required"] = list(tightened["properties"])
    return tightened
