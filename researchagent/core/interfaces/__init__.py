"""Ports (abstract interfaces) that the rest of the system depends on.

One module per outbound dependency. Concrete adapters live in
``researchagent/integrations/`` and are wired by name through configuration, so no
inner layer ever imports a vendor SDK.

Added as each subsystem lands: ``llm`` (v0.1), ``vector_store`` (RAG),
``graph_store`` (knowledge graph), ``paper_source`` (literature discovery).
"""

from researchagent.core.interfaces.llm import (
    CompletionResponse,
    GenerationParams,
    LLMProvider,
    Message,
    ProviderHealth,
    Role,
    TokenUsage,
)

__all__ = [
    "CompletionResponse",
    "GenerationParams",
    "LLMProvider",
    "Message",
    "ProviderHealth",
    "Role",
    "TokenUsage",
]
