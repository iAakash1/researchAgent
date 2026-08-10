"""LLM access service.

Agents ask for a *model alias* ("reasoning", "extraction"); the service resolves it
through ``config/models.yaml`` into a provider + model + decoding params. Agents never
see a provider, a base URL or a model tag.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from researchagent.config.schemas import ModelCatalog, ModelSpec
from researchagent.core.events import EventBus, EventType, LLMCallPayload
from researchagent.core.exceptions import ConfigurationError
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
from researchagent.core.settings import Settings
from researchagent.integrations.registry import build_llm_provider

logger = get_logger(__name__)


class UsageReport(BaseModel):
    """What one handle has spent, and how much of that is actually known.

    ``unmeasured_calls`` is the honesty field. A provider that reports nothing produces
    zero *measured* tokens, which is indistinguishable from a free call unless the count
    of unmeasured calls is carried alongside — and a budget that cannot tell "cheap" from
    "unknown" is not enforcing anything.
    """

    model_config = {"frozen": True}

    usage: TokenUsage = Field(default_factory=TokenUsage)
    calls: int = Field(default=0, ge=0)
    unmeasured_calls: int = Field(default=0, ge=0)

    @property
    def is_complete(self) -> bool:
        """Whether every call in this report reported its cost."""
        return self.unmeasured_calls == 0

    def plus(self, usage: TokenUsage | None) -> UsageReport:
        return UsageReport(
            usage=self.usage + (usage or TokenUsage()),
            calls=self.calls + 1,
            unmeasured_calls=self.unmeasured_calls + (1 if usage is None else 0),
        )


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
        # Accumulated here rather than returned to every call site: agents care about
        # their answer, the loop cares about the bill, and neither should have to thread
        # the other's concern through its own signatures.
        self._usage = UsageReport()

    @property
    def usage(self) -> UsageReport:
        """Everything this handle has spent since it was constructed.

        Handles are built per agent per iteration, so this is exactly one agent's spend
        for one round.
        """
        return self._usage

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
        self._usage = self._usage.plus(response.usage)
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
        result = await self._provider.complete_structured_with_usage(
            messages, model=self.model, params=self._params(params), schema=schema
        )
        self._usage = self._usage.plus(result.usage)
        return result.value

    def _params(self, override: GenerationParams | None) -> GenerationParams:
        return self.spec.params.merged_with(override)

    async def _emit(self, response: CompletionResponse) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.emit(
            EventType.LLM_CALL_COMPLETED,
            LLMCallPayload(
                alias=self.alias,
                model=response.model,
                latency_ms=response.latency_ms,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            ),
            source=f"{self.provider_name}:{self.alias}",
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

    def configured_providers(self) -> tuple[frozenset[str], frozenset[str]]:
        """Split the catalogue's providers into (configured, unconfigured).

        A provider is unconfigured when building it raises a ``ConfigurationError`` — an
        optional remote backend with no credentials. That is an absence, not a failure:
        reporting it as unhealthy would leave a purely local, offline install permanently
        un-ready, which contradicts the local-first default.
        """
        configured: set[str] = set()
        unconfigured: set[str] = set()
        for name in {spec.provider for spec in self._catalog.models.values()}:
            try:
                self._provider(name)
            except ConfigurationError:
                unconfigured.add(name)
            else:
                configured.add(name)
        return frozenset(configured), frozenset(unconfigured)

    def active_aliases(self) -> dict[str, ModelSpec]:
        """Catalogue entries whose provider is usable in this environment."""
        configured, _ = self.configured_providers()
        return {
            alias: spec
            for alias, spec in self._catalog.models.items()
            if spec.provider in configured
        }

    async def health(self) -> list[ProviderHealth]:
        """Probe every configured provider referenced by the catalog."""
        configured, _ = self.configured_providers()
        return [await self._provider(name).health() for name in sorted(configured)]

    async def verify_models_available(self) -> dict[str, bool]:
        """Map each active alias to whether its model tag is actually available."""
        active = self.active_aliases()
        available: dict[str, set[str]] = {}
        for name in {spec.provider for spec in active.values()}:
            health = await self._provider(name).health()
            available[name] = set(health.available_models)

        return {
            alias: spec.model_name in available.get(spec.provider, set())
            for alias, spec in active.items()
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
