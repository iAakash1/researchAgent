"""DeepSeek's OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from researchagent.core.constants import SECONDS_PER_MILLISECOND
from researchagent.core.exceptions import (
    OutputParsingError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Recoverability,
)
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
from researchagent.core.logging import get_logger
from researchagent.core.retry import RetryPolicy, retry_async

logger = get_logger(__name__)

_RAW_PREVIEW_LIMIT = 800
_SECRET_PATTERN = re.compile(r"\b(?:sk|key)-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
_JSON_SECRET_PATTERN = re.compile(
    r'("(?:api[_-]?key|authorization|secret|token)"\s*:\s*")[^"]*(")',
    re.IGNORECASE,
)


class _StructuredRepairExhaustedError(OutputParsingError):
    """A targeted repair already failed, so rerunning the whole agent is redundant."""

    recoverability = Recoverability.RECOVERABLE


@dataclass(frozen=True, slots=True)
class _ResponseDetails:
    raw: str
    finish_reason: str | None
    assistant_content_present: bool
    reasoning_content_present: bool

    @property
    def assistant_content_nonempty(self) -> bool:
        return bool(self.raw.strip())

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class DeepSeekProvider(LLMProvider):
    """DeepSeek chat completions without adding a vendor SDK dependency."""

    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        request_timeout_seconds: float = 120.0,
        retry_policy: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ProviderAuthenticationError(
                "DeepSeek provider constructed without an API key", provider=self.name
            )
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=3)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(request_timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            transport=transport,
        )

    async def complete(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> CompletionResponse:
        started = time.perf_counter()
        body, attempts = await self._post(_payload(messages, model=model, params=params), model)
        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
        usage = _usage(body)
        response = CompletionResponse(
            text=_text(body, model),
            model=model,
            provider=self.name,
            usage=usage,
            latency_ms=latency_ms,
            metadata={"attempts": attempts, "finish_reason": _finish_reason(body)},
        )
        logger.info(
            "deepseek_completion",
            provider=self.name,
            model=model,
            latency_ms=round(latency_ms, 1),
            attempts=attempts,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return response

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
                if response.status_code >= 400:
                    await response.aread()
                    _fail(response, model)
                async for line in response.aiter_lines():
                    if chunk := _sse_delta(line):
                        yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "DeepSeek streaming request timed out", provider=self.name, model=model
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Could not reach DeepSeek", provider=self.name, model=model
            ) from exc

    async def complete_structured(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> TSchema:
        return (
            await self.complete_structured_with_usage(
                messages, model=model, params=params, schema=schema
            )
        ).value

    async def complete_structured_with_usage(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> StructuredResult[TSchema]:
        json_schema = schema.model_json_schema()
        schema_instruction = _schema_instruction(schema.__name__, json_schema)
        requested = [*messages, Message.user(schema_instruction)]
        started = time.perf_counter()
        body, attempts = await self._post(
            _payload(requested, model=model, params=params)
            | {"response_format": {"type": "json_object"}},
            model,
        )
        details = _response_details(body, model)
        initial_finish_reason = details.finish_reason
        initial_truncated = details.truncated
        usage = _usage(body)
        repair_used = False
        if not details.assistant_content_nonempty:
            raise _empty_structured_error(model, schema.__name__, details)

        repair_instruction: str | None = None
        try:
            value = _parse_structured(details.raw, schema)
        except json.JSONDecodeError as exc:
            parse_error = _json_parse_error(exc)
            _log_malformed_json(
                model,
                schema.__name__,
                details,
                parse_error,
                repair_attempted=False,
            )
            repair_instruction = _malformed_repair_instruction(
                schema.__name__, parse_error, json_schema
            )
        except ValidationError as exc:
            issues = _validation_issues(exc)
            _log_validation_failure(model, schema.__name__, issues, details, repair_attempted=False)
            repair_instruction = _schema_repair_instruction(schema.__name__, issues, json_schema)

        if repair_instruction is not None:
            repair_messages = [
                *requested,
                Message.assistant(details.raw),
                Message.user(repair_instruction),
            ]
            repair_body, repair_attempts = await self._post(
                _payload(repair_messages, model=model, params=params)
                | {"response_format": {"type": "json_object"}},
                model,
            )
            repaired = _response_details(repair_body, model)
            if not repaired.assistant_content_nonempty:
                raise _empty_structured_error(
                    model, schema.__name__, repaired, repair_attempted=True
                )
            try:
                value = _parse_structured(repaired.raw, schema)
            except json.JSONDecodeError as repair_exc:
                repair_parse_error = _json_parse_error(repair_exc)
                _log_malformed_json(
                    model,
                    schema.__name__,
                    repaired,
                    repair_parse_error,
                    repair_attempted=True,
                )
                raise _malformed_structured_error(
                    model,
                    schema.__name__,
                    repaired,
                    repair_parse_error,
                    repair_attempted=True,
                ) from repair_exc
            except ValidationError as repair_exc:
                repair_issues = _validation_issues(repair_exc)
                _log_validation_failure(
                    model,
                    schema.__name__,
                    repair_issues,
                    repaired,
                    repair_attempted=True,
                )
                raise _structured_validation_error(
                    model,
                    schema.__name__,
                    repair_issues,
                    repaired,
                    repair_attempted=True,
                ) from repair_exc
            body = repair_body
            details = repaired
            attempts += repair_attempts
            usage = usage + _usage(repair_body)
            repair_used = True
        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
        return StructuredResult[TSchema](
            value=value,
            usage=usage,
            provider=self.name,
            model=model,
            latency_ms=latency_ms,
            metadata={
                "attempts": attempts,
                "finish_reason": details.finish_reason,
                "truncated": details.truncated,
                "initial_finish_reason": initial_finish_reason,
                "initial_truncated": initial_truncated,
                "repair_used": repair_used,
            },
        )

    async def health(self) -> ProviderHealth:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            models = [item["id"] for item in response.json().get("data", [])]
        except httpx.HTTPStatusError as exc:
            detail = (
                "credentials rejected"
                if exc.response.status_code in {401, 403}
                else f"http {exc.response.status_code}"
            )
            return ProviderHealth(provider=self.name, healthy=False, detail=detail)
        except httpx.HTTPError as exc:
            return ProviderHealth(provider=self.name, healthy=False, detail=type(exc).__name__)
        return ProviderHealth(provider=self.name, healthy=True, available_models=models)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict[str, Any], model: str) -> tuple[dict[str, Any], int]:
        attempt_count = 0

        async def attempt() -> dict[str, Any]:
            nonlocal attempt_count
            attempt_count += 1
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(
                    "DeepSeek request timed out", provider=self.name, model=model
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderUnavailableError(
                    "Could not reach DeepSeek", provider=self.name, model=model
                ) from exc
            if response.status_code >= 400:
                _fail(response, model)
            try:
                parsed = response.json()
            except ValueError as exc:
                raise OutputParsingError(
                    "DeepSeek returned a non-JSON response", provider=self.name, model=model
                ) from exc
            if not isinstance(parsed, dict):
                raise OutputParsingError(
                    "DeepSeek returned a non-object response", provider=self.name, model=model
                )
            return parsed

        try:
            return await retry_async(attempt, self._retry_policy, operation_name="deepseek_chat")
        except ProviderError as exc:
            exc.context["attempts"] = attempt_count
            raise


def _payload(messages: list[Message], *, model: str, params: GenerationParams) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": item.role.value, "content": item.content} for item in messages],
    }
    if params.thinking is not None:
        payload["thinking"] = {"type": "enabled" if params.thinking else "disabled"}
    if params.reasoning_effort is not None and params.thinking is not False:
        payload["reasoning_effort"] = params.reasoning_effort
    if not params.thinking:
        payload["temperature"] = params.temperature
        if params.top_p is not None:
            payload["top_p"] = params.top_p
    if params.max_output_tokens is not None:
        payload["max_tokens"] = params.max_output_tokens
    if params.stop:
        payload["stop"] = params.stop
    return payload


def _schema_instruction(schema_name: str, schema: dict[str, Any]) -> str:
    example = _schema_example(schema, schema)
    return (
        f"Return JSON only. The root value must be the {schema_name} object itself, with "
        "no wrapper, prose, or Markdown fence. Match this JSON Schema exactly: "
        f"{json.dumps(schema, separators=(',', ':'))}\n"
        f"Example JSON shape: {json.dumps(example, separators=(',', ':'))}"
    )


def _schema_repair_instruction(
    schema_name: str, issues: tuple[dict[str, str], ...], schema: dict[str, Any]
) -> str:
    return (
        f"The previous JSON failed {schema_name} validation. Return a corrected JSON object "
        "only; do not explain the correction, add a wrapper, use Markdown, or invent default "
        f"values. Validation errors: {json.dumps(issues, separators=(',', ':'))}. "
        f"Required schema: {json.dumps(schema, separators=(',', ':'))}. "
        f"Example JSON shape: {json.dumps(_schema_example(schema, schema), separators=(',', ':'))}"
    )


def _malformed_repair_instruction(
    schema_name: str, parse_error: dict[str, str | int], schema: dict[str, Any]
) -> str:
    return (
        f"The previous assistant response was not valid JSON for {schema_name}. Return a "
        "corrected JSON object only; do not explain the correction, add a wrapper, use "
        "Markdown, or invent default values. "
        f"JSON parse error: {json.dumps(parse_error, separators=(',', ':'))}. "
        f"Required schema: {json.dumps(schema, separators=(',', ':'))}. "
        f"Example JSON shape: {json.dumps(_schema_example(schema, schema), separators=(',', ':'))}"
    )


def _schema_example(node: dict[str, Any], root: dict[str, Any]) -> Any:
    reference = node.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        resolved = root.get("$defs", {}).get(reference.rsplit("/", 1)[-1], {})
        return _schema_example(resolved, root) if isinstance(resolved, dict) else None
    choices = node.get("anyOf") or node.get("oneOf")
    if isinstance(choices, list):
        concrete = next(
            (
                choice
                for choice in choices
                if isinstance(choice, dict) and choice.get("type") != "null"
            ),
            {},
        )
        return _schema_example(concrete, root)
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    node_type = node.get("type")
    if node_type == "object" or "properties" in node:
        properties = node.get("properties", {})
        return {
            name: _schema_example(value, root)
            for name, value in properties.items()
            if isinstance(value, dict)
        }
    if node_type == "array":
        items = node.get("items", {})
        return [_schema_example(items, root)] if isinstance(items, dict) else []
    if node_type in {"integer", "number"}:
        return 0
    if node_type == "boolean":
        return False
    return "<string>"


def _validation_issues(error: ValidationError) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "field": ".".join(str(part) for part in issue["loc"]) or "<root>",
            "type": str(issue["type"]),
            "message": str(issue["msg"]),
        }
        for issue in error.errors(include_url=False, include_input=False)
    )


def _parse_structured(raw: str, schema: type[TSchema]) -> TSchema:
    return schema.model_validate(json.loads(raw))


def _json_parse_error(error: json.JSONDecodeError) -> dict[str, str | int]:
    return {
        "type": type(error).__name__,
        "message": error.msg,
        "line": error.lineno,
        "column": error.colno,
        "position": error.pos,
    }


def _log_validation_failure(
    model: str,
    schema_name: str,
    issues: tuple[dict[str, str], ...],
    details: _ResponseDetails,
    *,
    repair_attempted: bool,
) -> None:
    logger.warning(
        "deepseek_structured_validation_failed",
        provider="deepseek",
        model=model,
        schema=schema_name,
        validation_errors=issues,
        **_response_context(details),
        repair_attempted=repair_attempted,
    )
    logger.debug(
        "deepseek_structured_invalid_response",
        provider="deepseek",
        model=model,
        schema=schema_name,
        raw_preview=_safe_preview(details.raw),
        **_response_context(details),
        repair_attempted=repair_attempted,
    )


def _log_malformed_json(
    model: str,
    schema_name: str,
    details: _ResponseDetails,
    parse_error: dict[str, str | int],
    *,
    repair_attempted: bool,
) -> None:
    logger.warning(
        "deepseek_structured_malformed_json",
        provider="deepseek",
        model=model,
        schema=schema_name,
        json_parse_error=parse_error,
        **_response_context(details),
        repair_attempted=repair_attempted,
    )
    logger.debug(
        "deepseek_structured_malformed_response",
        provider="deepseek",
        model=model,
        schema=schema_name,
        raw_preview=_safe_preview(details.raw),
        json_parse_error=parse_error,
        **_response_context(details),
        repair_attempted=repair_attempted,
    )


def _structured_validation_error(
    model: str,
    schema_name: str,
    issues: tuple[dict[str, str], ...],
    details: _ResponseDetails,
    *,
    repair_attempted: bool,
) -> OutputParsingError:
    error_type = _StructuredRepairExhaustedError if repair_attempted else OutputParsingError
    return error_type(
        "DeepSeek returned JSON that does not satisfy the schema",
        provider="deepseek",
        model=model,
        schema=schema_name,
        validation_errors=issues,
        **_response_context(details),
        repair_attempted=repair_attempted,
    )


def _malformed_structured_error(
    model: str,
    schema_name: str,
    details: _ResponseDetails,
    parse_error: dict[str, str | int],
    *,
    repair_attempted: bool,
) -> OutputParsingError:
    error_type = _StructuredRepairExhaustedError if repair_attempted else OutputParsingError
    return error_type(
        "DeepSeek returned malformed JSON",
        provider="deepseek",
        model=model,
        schema=schema_name,
        json_parse_error=parse_error,
        **_response_context(details),
        repair_attempted=repair_attempted,
    )


def _empty_structured_error(
    model: str,
    schema_name: str,
    details: _ResponseDetails,
    *,
    repair_attempted: bool = False,
) -> OutputParsingError:
    context = _response_context(details)
    logger.warning(
        "deepseek_structured_empty_content",
        provider="deepseek",
        model=model,
        schema=schema_name,
        **context,
        repair_attempted=repair_attempted,
    )
    error_type = _StructuredRepairExhaustedError if repair_attempted else OutputParsingError
    return error_type(
        "DeepSeek returned no final assistant content for structured output",
        provider="deepseek",
        model=model,
        schema=schema_name,
        **context,
        repair_attempted=repair_attempted,
    )


def _response_context(details: _ResponseDetails) -> dict[str, object]:
    return {
        "finish_reason": details.finish_reason,
        "raw_length": len(details.raw),
        "assistant_content_present": details.assistant_content_present,
        "assistant_content_nonempty": details.assistant_content_nonempty,
        "reasoning_content_present": details.reasoning_content_present,
        "truncated": details.truncated,
    }


def _safe_preview(raw: str) -> str:
    preview = _SECRET_PATTERN.sub("[REDACTED]", raw[:_RAW_PREVIEW_LIMIT])
    return _JSON_SECRET_PATTERN.sub(r"\1[REDACTED]\2", preview)


def _fail(response: httpx.Response, model: str) -> None:
    status = response.status_code
    detail = _error_message(response)
    context = {"provider": "deepseek", "model": model, "status": status}
    if status in {401, 403}:
        raise ProviderAuthenticationError("DeepSeek rejected the credentials", **context)
    if status == 429:
        raise ProviderRateLimitedError(
            "DeepSeek rate limit reached",
            **context,
            retry_after=response.headers.get("retry-after"),
        )
    if status == 404 or status >= 500:
        raise ProviderUnavailableError("DeepSeek is unavailable", **context, detail=detail)
    raise ProviderError("DeepSeek rejected the request", **context, detail=detail)


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error") if isinstance(body, dict) else None
    return str(error.get("message", "") if isinstance(error, dict) else error or "")[:200]


def _text(body: dict[str, Any], model: str) -> str:
    details = _response_details(body, model)
    if not details.assistant_content_present:
        raise OutputParsingError(
            "DeepSeek response contained no message content",
            provider="deepseek",
            model=model,
            **_response_context(details),
        )
    return details.raw


def _response_details(body: dict[str, Any], model: str) -> _ResponseDetails:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OutputParsingError(
            "DeepSeek response contained no choices", provider="deepseek", model=model
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise OutputParsingError(
            "DeepSeek response contained an invalid choice", provider="deepseek", model=model
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise OutputParsingError(
            "DeepSeek response contained no assistant message",
            provider="deepseek",
            model=model,
        )
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise OutputParsingError(
            "DeepSeek response contained invalid message content",
            provider="deepseek",
            model=model,
        )
    reasoning = message.get("reasoning_content")
    finish_reason = choice.get("finish_reason")
    return _ResponseDetails(
        raw=content or "",
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        assistant_content_present=isinstance(content, str),
        reasoning_content_present=isinstance(reasoning, str) and bool(reasoning.strip()),
    )


def _usage(body: dict[str, Any]) -> TokenUsage:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cached = int(usage.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens", 0)))
    prompt = int(usage.get("prompt_tokens", 0))
    uncached = int(usage.get("prompt_cache_miss_tokens", max(0, prompt - cached)))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=int(usage.get("completion_tokens", 0)),
        cached_prompt_tokens=cached,
        uncached_prompt_tokens=uncached,
        reasoning_tokens=int(completion_details.get("reasoning_tokens", 0)),
    )


def _finish_reason(body: dict[str, Any]) -> str | None:
    choices = body.get("choices") or [{}]
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else None


def _sse_delta(line: str) -> str:
    if not line.startswith("data: "):
        return ""
    data = line.removeprefix("data: ").strip()
    if not data or data == "[DONE]":
        return ""
    try:
        value = json.loads(data).get("choices", [{}])[0].get("delta", {}).get("content")
    except (ValueError, AttributeError, IndexError):
        return ""
    return value if isinstance(value, str) else ""
