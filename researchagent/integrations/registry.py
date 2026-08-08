"""Provider registry and factories.

``config/models.yaml`` names a provider as a string; this module turns that string
into a concrete adapter, so no inner layer imports a vendor SDK.
"""

from __future__ import annotations

from collections.abc import Callable

from researchagent.core.interfaces.llm import LLMProvider
from researchagent.core.registry import Registry
from researchagent.core.settings import Settings
from researchagent.integrations.ollama import OllamaProvider

LLMProviderFactory = Callable[[Settings], LLMProvider]

LLM_PROVIDERS: Registry[LLMProviderFactory] = Registry("llm_provider")


def _build_ollama(settings: Settings) -> LLMProvider:
    return OllamaProvider(
        base_url=settings.ollama.base_url,
        request_timeout_seconds=settings.ollama.request_timeout_seconds,
        keep_alive=settings.ollama.keep_alive,
    )


LLM_PROVIDERS.add("ollama", _build_ollama)


def build_llm_provider(name: str, settings: Settings) -> LLMProvider:
    """Instantiate the adapter registered under ``name``."""
    return LLM_PROVIDERS.get(name)(settings)
