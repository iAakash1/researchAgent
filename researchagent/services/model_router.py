"""Provider-failure routing behind the existing LLM port."""

from __future__ import annotations

from collections.abc import AsyncIterator

from researchagent.core.exceptions import (
    ConfigurationError,
    ProviderAuthenticationError,
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
    StructuredResult,
    TSchema,
)
from researchagent.core.logging import get_logger

logger = get_logger(__name__)

_FALLBACK_ERRORS = (
    ConfigurationError,
    ProviderAuthenticationError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class ModelRouter(LLMProvider):
    """Use the fallback only when the primary provider cannot serve the call."""

    name = "router"

    def __init__(
        self,
        primary: LLMProvider | ConfigurationError,
        fallback: LLMProvider,
        *,
        primary_provider: str,
        fallback_model: str,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_provider = primary_provider
        self._fallback_model = fallback_model

    async def complete(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> CompletionResponse:
        try:
            primary = self._require_primary(model)
            return await primary.complete(messages, model=model, params=params)
        except _FALLBACK_ERRORS as exc:
            self._log_fallback(model, exc)
            try:
                response = await self._fallback.complete(
                    messages, model=self._fallback_model, params=params
                )
            except _FALLBACK_ERRORS as fallback_error:
                raise self._both_failed(model, exc, fallback_error) from fallback_error
            return response.model_copy(
                update={"metadata": _fallback_metadata(response.metadata, exc)}
            )

    def stream(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> AsyncIterator[str]:
        return self._stream(messages, model=model, params=params)

    async def _stream(
        self, messages: list[Message], *, model: str, params: GenerationParams
    ) -> AsyncIterator[str]:
        try:
            primary = self._require_primary(model)
            async for chunk in primary.stream(messages, model=model, params=params):
                yield chunk
            return
        except _FALLBACK_ERRORS as exc:
            self._log_fallback(model, exc)
            try:
                async for chunk in self._fallback.stream(
                    messages, model=self._fallback_model, params=params
                ):
                    yield chunk
            except _FALLBACK_ERRORS as fallback_error:
                raise self._both_failed(model, exc, fallback_error) from fallback_error

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
        try:
            primary = self._require_primary(model)
            return await primary.complete_structured_with_usage(
                messages, model=model, params=params, schema=schema
            )
        except _FALLBACK_ERRORS as exc:
            self._log_fallback(model, exc)
            try:
                response = await self._fallback.complete_structured_with_usage(
                    messages, model=self._fallback_model, params=params, schema=schema
                )
            except _FALLBACK_ERRORS as fallback_error:
                raise self._both_failed(model, exc, fallback_error) from fallback_error
            return response.model_copy(
                update={
                    "provider": response.provider or self._fallback.name,
                    "model": response.model or self._fallback_model,
                    "metadata": _fallback_metadata(response.metadata, exc),
                }
            )

    async def health(self) -> ProviderHealth:
        if isinstance(self._primary, ConfigurationError):
            return await self._fallback.health()
        return await self._primary.health()

    async def aclose(self) -> None:
        # Lifecycle is owned by LLMService; closing here would double-close shared clients.
        return None

    def _require_primary(self, model: str) -> LLMProvider:
        if isinstance(self._primary, ConfigurationError):
            raise ProviderAuthenticationError(
                "Primary provider is not configured",
                provider=self._primary_provider,
                model=model,
            )
        return self._primary

    def _log_fallback(self, model: str, error: BaseException) -> None:
        logger.warning(
            "llm_provider_fallback",
            primary_provider=self._primary_provider,
            primary_model=model,
            fallback_provider=self._fallback.name,
            fallback_model=self._fallback_model,
            reason=type(error).__name__,
        )

    def _both_failed(
        self, model: str, primary: BaseException, fallback: BaseException
    ) -> ProviderUnavailableError:
        return ProviderUnavailableError(
            "Primary and fallback LLM providers are unavailable",
            primary_provider=self._primary_provider,
            primary_model=model,
            primary_error=type(primary).__name__,
            fallback_provider=self._fallback.name,
            fallback_model=self._fallback_model,
            fallback_error=type(fallback).__name__,
        )


def _fallback_metadata(metadata: dict[str, object], error: BaseException) -> dict[str, object]:
    recorded_attempts = metadata.get("attempts", 1)
    fallback_attempts = recorded_attempts if isinstance(recorded_attempts, int) else 1
    context = getattr(error, "context", {})
    recorded_primary_attempts = context.get("attempts", 1) if isinstance(context, dict) else 1
    primary_attempts = (
        recorded_primary_attempts if isinstance(recorded_primary_attempts, int) else 1
    )
    return {
        **metadata,
        "fallback_used": True,
        "fallback_reason": type(error).__name__,
        "attempts": primary_attempts + fallback_attempts,
    }
