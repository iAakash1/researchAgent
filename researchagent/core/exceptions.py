"""Exception hierarchy shared by every ResearchAgent subsystem.

Rules:
    * Every raised error carries a stable ``code`` used by the API layer and logs.
    * ``recoverability`` classifies what a caller should do about it, so recovery is a
      property of the error rather than a guess at the call site.
    * ``remedy`` names the recommended action in plain words.

Generic ``Exception`` is never raised from domain code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Recoverability(StrEnum):
    """What the caller should do when this error surfaces."""

    # The same call may succeed if repeated (timeout, rate limit, flaky parse).
    RETRYABLE = "retryable"
    # Retrying will not help, but the pipeline can continue with reduced scope —
    # e.g. one PDF of forty failed to parse.
    RECOVERABLE = "recoverable"
    # The run cannot meaningfully continue (missing configuration, invalid contract).
    FATAL = "fatal"

    @property
    def allows_retry(self) -> bool:
        return self is Recoverability.RETRYABLE

    @property
    def allows_continue(self) -> bool:
        return self is not Recoverability.FATAL


class ResearchAgentError(Exception):
    """Base class for all domain errors."""

    code: str = "researchagent_error"
    http_status: int = 500
    recoverability: Recoverability = Recoverability.FATAL
    remedy: str | None = None

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    @property
    def retryable(self) -> bool:
        """Kept as the retry helper's predicate; derived from ``recoverability``."""
        return self.recoverability.allows_retry

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "recoverability": self.recoverability.value,
            "remedy": self.remedy,
            "context": self.context,
        }

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


class PromptError(ConfigurationError):
    """A prompt file is missing, malformed, or missing a required variable."""

    code = "prompt_error"


class ProviderError(ResearchAgentError):
    """An LLM provider failed to serve a request."""

    code = "provider_error"
    http_status = 502
    recoverability = Recoverability.RECOVERABLE


class ProviderTimeoutError(ProviderError):
    """The provider did not answer within the configured budget."""

    code = "provider_timeout"
    http_status = 504
    recoverability = Recoverability.RETRYABLE
    remedy = "Retry; raise RESEARCHAGENT_OLLAMA__REQUEST_TIMEOUT_SECONDS if it persists"


class ProviderUnavailableError(ProviderError):
    """The provider is unreachable (process down, wrong base_url, model not pulled)."""

    code = "provider_unavailable"
    http_status = 503
    recoverability = Recoverability.RETRYABLE
    remedy = "Check the provider is running and the model is pulled"


class PaperSourceError(ResearchAgentError):
    """A literature provider failed to serve a request."""

    code = "paper_source_error"
    http_status = 502
    recoverability = Recoverability.RECOVERABLE

    def __init__(self, message: str, *, source: str, **context: Any) -> None:
        super().__init__(message, source=source, **context)
        self.source = source


class SourceUnavailableError(PaperSourceError):
    """Provider unreachable, timed out, or returned a server error."""

    code = "source_unavailable"
    http_status = 503
    recoverability = Recoverability.RETRYABLE
    remedy = "Retry later; discovery continues with the remaining providers"


class SourceRateLimitedError(PaperSourceError):
    """Provider asked us to slow down (HTTP 429)."""

    code = "source_rate_limited"
    http_status = 429
    recoverability = Recoverability.RETRYABLE
    remedy = "Lower requests_per_second for this source in config/sources.yaml"


class SourceResponseError(PaperSourceError):
    """Provider replied, but the payload could not be parsed into a Paper."""

    code = "source_response_error"
    http_status = 502
    recoverability = Recoverability.RECOVERABLE


class PaperNotFoundError(ResearchAgentError):
    """No paper matches the requested identifier."""

    code = "paper_not_found"
    http_status = 404
    recoverability = Recoverability.RECOVERABLE


class RepositoryError(ResearchAgentError):
    """Persistence failed (unreadable record, unwritable path)."""

    code = "repository_error"
    http_status = 500
    recoverability = Recoverability.RECOVERABLE


class OutputParsingError(ResearchAgentError):
    """The model returned text that does not satisfy the requested schema."""

    code = "output_parsing_error"
    http_status = 422
    recoverability = Recoverability.RETRYABLE
    remedy = "Re-prompt; consider a stricter schema or a larger model"


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


class WorkflowExecutionError(ResearchAgentError):
    """A workflow run finished in a failed state."""

    code = "workflow_execution_error"
    http_status = 502


class RunNotFoundError(ResearchAgentError):
    """No checkpoint exists for the requested run id."""

    code = "run_not_found"
    http_status = 404


class DocumentError(ResearchAgentError):
    """A document could not be turned into a canonical representation."""

    code = "document_error"
    http_status = 422
    recoverability = Recoverability.RECOVERABLE

    def __init__(self, message: str, *, paper_id: str, **context: Any) -> None:
        super().__init__(message, paper_id=paper_id, **context)
        self.paper_id = paper_id


class DocumentUnreadableError(DocumentError):
    """The file is missing, empty, encrypted, or not a PDF at all."""

    code = "document_unreadable"
    recoverability = Recoverability.RECOVERABLE
    remedy = "Re-download the PDF; the stored file is not a usable document"


class DocumentParsingError(DocumentError):
    """The PDF opened but structured content could not be extracted."""

    code = "document_parsing_error"
    recoverability = Recoverability.RECOVERABLE
    remedy = "Inspect the PDF; it may be scanned images, which need OCR"


class ValidationFailedError(ResearchAgentError):
    """A validated artefact was rejected by its validator."""

    code = "validation_failed"
    http_status = 422
    recoverability = Recoverability.RECOVERABLE

    def __init__(self, message: str, *, validator: str, subject_id: str, **context: Any) -> None:
        super().__init__(message, validator=validator, subject_id=subject_id, **context)
        self.validator = validator
        self.subject_id = subject_id


class PrerequisiteNotMetError(ResearchAgentError):
    """A workflow stage was reached without the inputs it requires.

    Raised by guards, never by stage bodies: a stage should not have to defend itself
    against being called in the wrong order.
    """

    code = "prerequisite_not_met"
    http_status = 409
    recoverability = Recoverability.FATAL
