"""Ollama adapter: local inference and local embeddings."""

from researchagent.integrations.ollama.client import OllamaAdminClient
from researchagent.integrations.ollama.embeddings import NullEmbeddingModel, OllamaEmbeddingModel
from researchagent.integrations.ollama.provider import OllamaProvider

__all__ = [
    "NullEmbeddingModel",
    "OllamaAdminClient",
    "OllamaEmbeddingModel",
    "OllamaProvider",
]
