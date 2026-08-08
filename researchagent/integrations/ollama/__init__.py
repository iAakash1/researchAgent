"""Ollama adapter: local inference for the LLM port."""

from researchagent.integrations.ollama.client import OllamaAdminClient
from researchagent.integrations.ollama.provider import OllamaProvider

__all__ = ["OllamaAdminClient", "OllamaProvider"]
