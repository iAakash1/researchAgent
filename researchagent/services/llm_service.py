"""LLM access service.

Agents ask for a *model alias* ("reasoning", "extraction"); the service resolves it
through ``config/models.yaml`` into a provider + model + decoding params. Agents never
see a provider, a base URL or a model tag.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from researchagent.config.schemas import ModelCatalog, ModelSpec
from researchagent.core.events import Event, EventBus, EventType
from researchagent.core.interfaces.llm import (
    CompletionResponse,
    GenerationParams,
    LLMProvider,
    Message,
    ProviderHealth,
    TSchema,
)
from researchagent.core.logging import get_logger
from researchagent.core.settings import Settings
from researchagent.integrations.registry import build_llm_provider

logger = get_logger(__name__)


class BoundLLM:
    """A model alias bound to its provider and default decoding params."""

    def __init__(
        self,
        alias: str,
        spec: ModelSpec,
        provider: LLMProvider,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self.alias = alias
        self.spec = spec
        self._provider = provider
        self._event_bus = event_bus

    @property
    def model(self) -> str:
        return self.spec.model_name

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def complete(
        self,
        messages: list[Message],
        *,
        params: GenerationParams | None = None,
    ) -> CompletionResponse:
        response = await self._provider.complete(
            messages, model=self.model, params=self._params(params)
        )
        await self._emit(response)
        return response

    def stream(
        self,
        messages: list[Message],
        *,
        params: GenerationParams | None = None,
    ) -> AsyncIterator[str]:
        return self._provider.stream(messages, model=self.model, params=self._params(params))

    async def complete_structured(
        self,
        messages: list[Message],
        schema: type[TSchema],
        *,
        params: GenerationParams | None = None,
    ) -> TSchema:
        return await self._provider.complete_structured(
            messages, model=self.model, params=self._params(params), schema=schema
        )

    def _params(self, override: GenerationParams | None) -> GenerationParams:
        return self.spec.params.merged_with(override)

    async def _emit(self, response: CompletionResponse) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event(
                type=EventType.LLM_CALL_COMPLETED,
                source=f"{self.provider_name}:{self.alias}",
                payload={
                    "alias": self.alias,
                    "model": response.model,
                    "latency_ms": response.latency_ms,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
        )


class LLMService:
    """Owns provider lifecycles and hands out :class:`BoundLLM` handles."""

    def __init__(
        self,
        catalog: ModelCatalog,
        settings: Settings,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._catalog = catalog
        self._settings = settings
        self._event_bus = event_bus
        self._providers: dict[str, LLMProvider] = {}

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    def get(self, alias: str | None = None) -> BoundLLM:
        """Resolve ``alias`` (or the catalog default) into a usable handle."""
        resolved = self._catalog.resolve_alias(alias)
        spec = self._catalog.spec_for(resolved)
        return BoundLLM(resolved, spec, self._provider(spec.provider), event_bus=self._event_bus)

    async def health(self) -> list[ProviderHealth]:
        """Probe every provider referenced by the catalog."""
        names = {spec.provider for spec in self._catalog.models.values()}
        return [await self._provider(name).health() for name in sorted(names)]

    async def verify_models_available(self) -> dict[str, bool]:
        """Map each alias to whether its model tag is actually pulled locally."""
        available: dict[str, set[str]] = {}
        for name in {spec.provider for spec in self._catalog.models.values()}:
            health = await self._provider(name).health()
            available[name] = set(health.available_models)

        return {
            alias: spec.model_name in available.get(spec.provider, set())
            for alias, spec in self._catalog.models.items()
        }

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._providers.clear()

    def _provider(self, name: str) -> LLMProvider:
        provider = self._providers.get(name)
        if provider is None:
            provider = build_llm_provider(name, self._settings)
            self._providers[name] = provider
            logger.debug("llm_provider_initialised", provider=name)
        return provider
