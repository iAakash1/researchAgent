"""Ollama adapter for the :class:`LLMProvider` port."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import ValidationError

from researchagent.core.constants import SECONDS_PER_MILLISECOND
from researchagent.core.exceptions import (
    OutputParsingError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ResearchAgentError,
)
from researchagent.core.interfaces.llm import (
    CompletionResponse,
    GenerationParams,
    LLMProvider,
    Message,
    ProviderHealth,
    Role,
    StructuredResult,
    TokenUsage,
    TSchema,
)
from researchagent.core.logging import get_logger
from researchagent.integrations.ollama.client import OllamaAdminClient

logger = get_logger(__name__)

_ROLE_TO_MESSAGE: dict[Role, type[BaseMessage]] = {
    Role.SYSTEM: SystemMessage,
    Role.USER: HumanMessage,
    Role.ASSISTANT: AIMessage,
}


class OllamaProvider(LLMProvider):
    """Local inference through an Ollama server.

    Chat models are cached per (model, decoding params) because each ChatOllama
    instance owns an HTTP client; rebuilding one per call leaks sockets under load.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float = 300.0,
        keep_alive: str = "10m",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout_seconds
        self._keep_alive = keep_alive
        self._admin = OllamaAdminClient(self._base_url)
        self._chat_models: dict[str, ChatOllama] = {}

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
    ) -> CompletionResponse:
        chat = self._chat_model(model, params)
        started = time.perf_counter()

        async def call() -> AIMessage:
            return await chat.ainvoke(self._to_langchain(messages))

        result = await self._guard(call, model=model, operation="complete")
        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND

        response = CompletionResponse(
            text=_text_of(result),
            model=model,
            provider=self.name,
            usage=_usage_of(result),
            latency_ms=latency_ms,
            metadata={
                "done_reason": result.response_metadata.get("done_reason"),
                "stop_reason": result.response_metadata.get("stop_reason"),
            },
        )
        logger.debug(
            "llm_completion",
            provider=self.name,
            model=model,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        return response

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        chat = self._chat_model(model, params)
        try:
            async with asyncio.timeout(self._timeout):
                async for chunk in chat.astream(self._to_langchain(messages)):
                    text = _text_of(chunk)
                    if text:
                        yield text
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                "Ollama stream exceeded the request budget",
                model=model,
                timeout_seconds=self._timeout,
            ) from exc
        except ResearchAgentError:
            raise
        except Exception as exc:
            raise self._classify(exc, model=model, operation="stream") from exc

    async def complete_structured(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> TSchema:
        chat = self._chat_model(model, params)
        structured = chat.with_structured_output(schema)

        async def call() -> Any:
            return await structured.ainvoke(self._to_langchain(messages))

        result = await self._guard(call, model=model, operation="complete_structured")

        if isinstance(result, schema):
            return result
        # Small local models routinely emit JSON that misses required fields; make that
        # a retryable parsing failure rather than a hard crash.
        try:
            return schema.model_validate(result)
        except (ValidationError, TypeError) as exc:
            raise OutputParsingError(
                "Model output did not satisfy the requested schema",
                model=model,
                schema=schema.__name__,
                received_type=type(result).__name__,
            ) from exc

    async def complete_structured_with_usage(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> StructuredResult[TSchema]:
        """Structured output plus token counts.

        ``with_structured_output`` normally discards the underlying message, and with it
        the usage metadata. ``include_raw=True`` keeps both, so a local run can be
        budgeted on real counts rather than an estimate. Models that report no usage
        metadata yield ``usage=None`` — unknown, not zero.
        """
        chat = self._chat_model(model, params)
        structured = chat.with_structured_output(schema, include_raw=True)

        async def call() -> Any:
            return await structured.ainvoke(self._to_langchain(messages))

        result = await self._guard(call, model=model, operation="complete_structured")
        parsed = result.get("parsed") if isinstance(result, dict) else result
        raw = result.get("raw") if isinstance(result, dict) else None

        return StructuredResult[TSchema](
            value=self._as_schema(parsed, schema, model=model), usage=_reported_usage(raw)
        )

    def _as_schema(self, value: Any, schema: type[TSchema], *, model: str) -> TSchema:
        if isinstance(value, schema):
            return value
        try:
            return schema.model_validate(value)
        except (ValidationError, TypeError) as exc:
            raise OutputParsingError(
                "Model output did not satisfy the requested schema",
                model=model,
                schema=schema.__name__,
                received_type=type(value).__name__,
            ) from exc

    async def health(self) -> ProviderHealth:
        try:
            models = await self._admin.list_models()
        except ProviderUnavailableError as exc:
            return ProviderHealth(provider=self.name, healthy=False, detail=str(exc))
        return ProviderHealth(provider=self.name, healthy=True, available_models=models)

    async def aclose(self) -> None:
        await self._admin.aclose()
        self._chat_models.clear()

    def _chat_model(self, model: str, params: GenerationParams) -> ChatOllama:
        cache_key = f"{model}|{params.model_dump_json()}"
        cached = self._chat_models.get(cache_key)
        if cached is not None:
            return cached

        chat = ChatOllama(
            model=model,
            base_url=self._base_url,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repeat_penalty,
            seed=params.seed,
            num_ctx=params.context_window,
            num_predict=params.max_output_tokens,
            stop=params.stop or None,
            keep_alive=self._keep_alive,
            client_kwargs={"timeout": self._timeout},
        )
        self._chat_models[cache_key] = chat
        return chat

    @staticmethod
    def _to_langchain(messages: list[Message]) -> list[BaseMessage]:
        if not messages:
            raise ProviderError("Cannot call an LLM with an empty message list")
        return [_ROLE_TO_MESSAGE[m.role](content=m.content) for m in messages]

    async def _guard(
        self,
        call: Any,
        *,
        model: str,
        operation: str,
    ) -> Any:
        try:
            async with asyncio.timeout(self._timeout):
                return await call()
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                "Ollama request exceeded the configured budget",
                model=model,
                operation=operation,
                timeout_seconds=self._timeout,
            ) from exc
        except ResearchAgentError:
            raise
        except Exception as exc:
            raise self._classify(exc, model=model, operation=operation) from exc

    def _classify(self, exc: Exception, *, model: str, operation: str) -> ProviderError:
        if isinstance(exc, ResponseError):
            # 404 from Ollama means the model tag was never pulled on this host.
            if exc.status_code == httpx.codes.NOT_FOUND:
                return ProviderUnavailableError(
                    "Model is not available on the Ollama host; pull it first",
                    model=model,
                    remedy=f"ollama pull {model}",
                )
            return ProviderError(
                "Ollama returned an error response",
                model=model,
                operation=operation,
                status_code=exc.status_code,
                reason=str(exc),
            )

        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout | ConnectionError):
            return ProviderUnavailableError(
                "Cannot reach the Ollama server",
                base_url=self._base_url,
                model=model,
                reason=str(exc),
            )

        return ProviderError(
            "Unexpected Ollama failure",
            model=model,
            operation=operation,
            error_type=type(exc).__name__,
            reason=str(exc),
        )


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # LangChain may return content blocks; concatenate the textual ones.
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def _usage_of(message: AIMessage) -> TokenUsage:
    usage = message.usage_metadata
    if usage:
        return TokenUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )
    meta = message.response_metadata
    return TokenUsage(
        prompt_tokens=int(meta.get("prompt_eval_count", 0) or 0),
        completion_tokens=int(meta.get("eval_count", 0) or 0),
    )


def _reported_usage(message: Any) -> TokenUsage | None:
    """Token counts a message actually carries, or None when it carries none.

    Distinct from ``_usage_of``, which fills a required field on ``CompletionResponse``
    and so must return a value. Budgeting needs the difference: a model that reports
    nothing has *unknown* cost, and treating unknown as zero is how a token budget
    silently stops being a budget.
    """
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and (
        usage.get("input_tokens") is not None or usage.get("output_tokens") is not None
    ):
        return TokenUsage(
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
        )

    meta = getattr(message, "response_metadata", None)
    if isinstance(meta, dict) and (
        meta.get("prompt_eval_count") is not None or meta.get("eval_count") is not None
    ):
        return TokenUsage(
            prompt_tokens=int(meta.get("prompt_eval_count") or 0),
            completion_tokens=int(meta.get("eval_count") or 0),
        )
    return None
