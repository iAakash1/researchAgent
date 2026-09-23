"""LLM access service.

Agents ask for a *model alias* ("reasoning", "extraction"); the service resolves it
through ``config/models.yaml`` into a provider + model + decoding params. Agents never
see a provider, a base URL or a model tag.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextvars import ContextVar, Token
from typing import cast

from pydantic import BaseModel, Field

from researchagent.config.schemas import ModelCatalog, ModelPricing, ModelSpec
from researchagent.core.constants import SECONDS_PER_MILLISECOND
from researchagent.core.events import EventBus, EventType, LLMCallPayload
from researchagent.core.exceptions import BudgetExhaustedError, ConfigurationError
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
from researchagent.services.model_router import ModelRouter

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
        pricing: dict[tuple[str, str], ModelPricing] | None = None,
    ) -> None:
        self.alias = alias
        self.spec = spec
        self._provider = provider
        self._event_bus = event_bus
        self._pricing = pricing or {}
        self._agent: ContextVar[str | None] = ContextVar(f"llm_agent_{id(self)}", default=None)
        self._run_id: ContextVar[str | None] = ContextVar(f"llm_run_{id(self)}", default=None)
        # Accumulated here rather than returned to every call site: agents care about
        # their answer, the loop cares about the bill, and neither should have to thread
        # the other's concern through its own signatures.
        self._usage = UsageReport()
        self._token_ceiling: int | None = None

    def with_token_ceiling(self, remaining: int | None) -> BoundLLM:
        """Refuse to start a call once ``remaining`` measured tokens are gone.

        Enforced on *spent* tokens, never on a prediction of what a call will cost:
        estimating a request's size would be inventing usage, which the accounting rules
        forbid. So this is an honest floor — no call begins after the budget is gone —
        and not a claim that a single call cannot overshoot it.
        """
        self._token_ceiling = remaining
        return self

    def bind_context(
        self, *, agent: str, run_id: str
    ) -> tuple[Token[str | None], Token[str | None]]:
        return self._agent.set(agent), self._run_id.set(run_id)

    def reset_context(self, tokens: tuple[Token[str | None], Token[str | None]]) -> None:
        self._agent.reset(tokens[0])
        self._run_id.reset(tokens[1])

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
        self._require_budget()
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
        self._require_budget()
        started = time.perf_counter()
        result = await self._provider.complete_structured_with_usage(
            messages, model=self.model, params=self._params(params), schema=schema
        )
        self._usage = self._usage.plus(result.usage)
        usage = result.usage or TokenUsage()
        await self._emit(
            CompletionResponse(
                text="",
                model=result.model or self.model,
                provider=result.provider or self._provider.name,
                usage=usage,
                latency_ms=(
                    result.latency_ms or (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
                ),
                metadata=result.metadata,
            )
        )
        return result.value

    def _require_budget(self) -> None:
        if self._token_ceiling is None:
            return
        if self._usage.usage.total_tokens >= self._token_ceiling:
            raise BudgetExhaustedError(
                "Token budget exhausted before this call",
                alias=self.alias,
                spent=self._usage.usage.total_tokens,
                ceiling=self._token_ceiling,
            )

    def _params(self, override: GenerationParams | None) -> GenerationParams:
        return self.spec.params.merged_with(override)

    async def _emit(self, response: CompletionResponse) -> None:
        if self._event_bus is None:
            return
        fallback_used = bool(response.metadata.get("fallback_used", False))
        attempts = int(response.metadata.get("attempts", 1))
        await self._event_bus.emit(
            EventType.LLM_CALL_COMPLETED,
            LLMCallPayload(
                alias=self.alias,
                agent=self._agent.get(),
                provider=response.provider,
                model=response.model,
                latency_ms=response.latency_ms,
                attempts=attempts,
                fallback_used=fallback_used,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                estimated_cost_usd=self._estimated_cost(response),
            ),
            run_id=self._run_id.get(),
            source=f"{response.provider}:{self.alias}",
        )

    def _estimated_cost(self, response: CompletionResponse) -> float | None:
        pricing = self._pricing.get((response.provider, response.model))
        if pricing is None:
            return None
        hit_rate = pricing.input_cache_hit
        miss_rate = pricing.input_cache_miss
        output_rate = pricing.output
        if hit_rate is None or miss_rate is None or output_rate is None:
            return None
        usage = response.usage
        cached = usage.cached_prompt_tokens
        uncached = usage.uncached_prompt_tokens or max(0, usage.prompt_tokens - cached)
        return round(
            (cached * hit_rate + uncached * miss_rate + usage.completion_tokens * output_rate)
            / 1_000_000,
            8,
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
        provider = self._provider_for(spec)
        pricing = {
            (entry.provider, entry.model_name): entry.pricing
            for entry in (*self._catalog.models.values(),)
            if entry.pricing is not None
        }
        if self._catalog.fallback is not None and self._catalog.fallback.pricing is not None:
            fallback = self._catalog.fallback
            fallback_pricing = cast(ModelPricing, fallback.pricing)
            pricing[(fallback.provider, fallback.model_name)] = fallback_pricing
        return BoundLLM(
            resolved,
            spec,
            provider,
            event_bus=self._event_bus,
            pricing=pricing,
        )

    def configured_providers(self) -> tuple[frozenset[str], frozenset[str]]:
        """Split the catalogue's providers into (configured, unconfigured).

        A provider is unconfigured when building it raises a ``ConfigurationError`` — an
        optional remote backend with no credentials. That is an absence, not a failure:
        reporting it as unhealthy would leave a purely local, offline install permanently
        un-ready, which contradicts the local-first default.
        """
        configured: set[str] = set()
        unconfigured: set[str] = set()
        specs = list(self._catalog.models.values())
        if self._catalog.fallback is not None:
            specs.append(self._catalog.fallback)
        for name in {spec.provider for spec in specs}:
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

    def _provider_for(self, spec: ModelSpec) -> LLMProvider:
        fallback = self._catalog.fallback
        if fallback is None or spec.provider == fallback.provider:
            return self._provider(spec.provider)
        try:
            primary: LLMProvider | ConfigurationError = self._provider(spec.provider)
        except ConfigurationError as exc:
            primary = exc
        return ModelRouter(
            primary,
            self._provider(fallback.provider),
            primary_provider=spec.provider,
            fallback_model=fallback.model_name,
        )
