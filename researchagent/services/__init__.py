"""Application services: reusable capabilities agents compose.

An agent decides *what* to do; a service knows *how* to do it (call a model, search a
catalogue, embed a chunk). Agents own no I/O.
"""

from researchagent.services.llm_service import BoundLLM, LLMService

__all__ = ["BoundLLM", "LLMService"]
