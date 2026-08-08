"""LLM provider port.

Agents depend on this abstraction, never on Ollama, LangChain or any vendor SDK.
Swapping Ollama for vLLM/llama.cpp means adding an implementation under
``researchagent/integrations/`` and one line in ``config/models.yaml``.

The DTOs live beside the port because they are the contract itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

TSchema = TypeVar("TSchema", bound=BaseModel)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = {"frozen": True}

    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=Role.ASSISTANT, content=content)


class GenerationParams(BaseModel):
    """Decoding parameters, resolved from ``config/models.yaml`` per model alias."""

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    repeat_penalty: float | None = Field(default=None, gt=0.0)
    seed: int | None = None
    # Context window and max new tokens; None defers to the model's own default.
    context_window: int | None = Field(default=None, ge=512)
    max_output_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] = Field(default_factory=list)

    def merged_with(self, override: GenerationParams | None) -> GenerationParams:
        if override is None:
            return self
        explicit = override.model_dump(exclude_unset=True)
        return self.model_copy(update=explicit)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class CompletionResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(BaseModel):
    provider: str
    healthy: bool
    detail: str | None = None
    available_models: list[str] = Field(default_factory=list)


class LLMProvider(ABC):
    """Vendor-agnostic chat completion port."""

    name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
    ) -> CompletionResponse:
        """Return a single completion. Raises ``ProviderError`` subclasses on failure."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        """Yield incremental text chunks."""

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[Message],
        *,
        model: str,
        params: GenerationParams,
        schema: type[TSchema],
    ) -> TSchema:
        """Return a validated instance of ``schema``.

        Implementations must raise ``OutputParsingError`` (retryable) rather than
        returning partially valid objects.
        """

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Cheap liveness probe; must not raise."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release sockets and background resources."""
