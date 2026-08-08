"""Exception hierarchy shared by every ResearchAgent subsystem.

Rules:
    * Every raised error carries a stable ``code`` used by the API layer and logs.
    * ``retryable`` drives the retry helper; it is a property of the error, not the caller.
"""

from __future__ import annotations

from typing import Any


class ResearchAgentError(Exception):
    """Base class for all domain errors."""

    code: str = "researchagent_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({rendered})"


class ConfigurationError(ResearchAgentError):
    """Invalid, missing or contradictory configuration."""

    code = "configuration_error"
    http_status = 500


class RegistryError(ConfigurationError):
    """Unknown or duplicated registry key."""

    code = "registry_error"


class ProviderError(ResearchAgentError):
    """An LLM provider failed to serve a request."""

    code = "provider_error"
    http_status = 502


class ProviderTimeoutError(ProviderError):
    """The provider did not answer within the configured budget."""

    code = "provider_timeout"
    http_status = 504
    retryable = True


class ProviderUnavailableError(ProviderError):
    """The provider is unreachable (process down, wrong base_url, model not pulled)."""

    code = "provider_unavailable"
    http_status = 503
    retryable = True


class OutputParsingError(ResearchAgentError):
    """The model returned text that does not satisfy the requested schema."""

    code = "output_parsing_error"
    http_status = 422
    retryable = True


class AgentExecutionError(ResearchAgentError):
    """An agent failed after exhausting its retry policy."""

    code = "agent_execution_error"
    http_status = 500

    def __init__(self, message: str, *, agent: str, **context: Any) -> None:
        super().__init__(message, agent=agent, **context)
        self.agent = agent


class AgentInputError(AgentExecutionError):
    """Payload handed to an agent does not satisfy its input schema."""

    code = "agent_input_error"
    http_status = 400
