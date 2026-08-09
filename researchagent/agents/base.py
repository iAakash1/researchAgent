"""Agent contract.

``BaseAgent`` owns everything that is identical for every agent — input validation,
retries, timing, log context, event emission, error wrapping — so a concrete agent is
only its reasoning step. Subclasses implement ``execute`` and nothing else.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from researchagent.config.schemas import AgentSpec
from researchagent.core.constants import AGENT_KEY, RUN_ID_KEY, SECONDS_PER_MILLISECOND
from researchagent.core.events import AgentPayload, Event, EventBus, EventType
from researchagent.core.exceptions import (
    AgentExecutionError,
    AgentInputError,
    ResearchAgentError,
)
from researchagent.core.logging import get_logger, log_context
from researchagent.core.prompts import PromptLibrary, PromptTemplate
from researchagent.core.retry import retry_async
from researchagent.services.llm_service import BoundLLM

logger = get_logger(__name__)


class AgentContext(BaseModel):
    """Per-invocation correlation data. Not the workflow state — agents stay stateless."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult[TOutput: BaseModel](BaseModel):
    agent: str
    output: TOutput
    latency_ms: float
    attempts: int = 1


class BaseAgent[TInput: BaseModel, TOutput: BaseModel](ABC):
    """Single-responsibility unit of reasoning.

    Class attributes declare the agent's contract; the orchestrator relies on them to
    route and validate without importing the agent's internals.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]

    def __init__(
        self,
        llm: BoundLLM,
        spec: AgentSpec,
        prompts: PromptLibrary,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._require_contract()
        self.llm = llm
        self.spec = spec
        self._prompts = prompts
        self._prompt: PromptTemplate | None = None
        self._event_bus = event_bus
        self.logger = get_logger(f"agent.{self.name}")

    @property
    def prompt(self) -> PromptTemplate:
        """The agent's prompt at the version pinned in ``config/agents.yaml``.

        Loaded on first use so agents that need no prompt (and their tests) never
        require a prompt file to exist.
        """
        if self._prompt is None:
            self._prompt = self._prompts.load(self.name, self.spec.prompt_version)
        return self._prompt

    async def run(
        self,
        payload: TInput | Mapping[str, Any],
        context: AgentContext,
    ) -> AgentResult[TOutput]:
        """Validate, execute under the retry policy, and report."""
        validated = self._validate_input(payload)
        started = time.perf_counter()

        with log_context(**{AGENT_KEY: self.name, RUN_ID_KEY: context.run_id}):
            await self._emit(EventType.AGENT_STARTED, context, AgentPayload(agent=self.name))
            try:
                output, attempts = await retry_async(
                    lambda: self.execute(validated, context),
                    self.spec.retry,
                    operation_name=f"agent.{self.name}",
                    on_retry=lambda attempt, error: self._on_retry(context, attempt, error),
                )
            except ResearchAgentError as exc:
                latency_ms = self._elapsed_ms(started)
                self.logger.error(
                    "agent_failed",
                    error=str(exc),
                    error_code=exc.code,
                    latency_ms=round(latency_ms, 1),
                )
                await self._emit(
                    EventType.AGENT_FAILED,
                    context,
                    AgentPayload(
                        agent=self.name,
                        error=str(exc),
                        code=exc.code,
                        latency_ms=latency_ms,
                    ),
                )
                raise AgentExecutionError(
                    f"Agent {self.name!r} failed", agent=self.name, cause=exc.code
                ) from exc

            latency_ms = self._elapsed_ms(started)
            self.logger.info("agent_completed", latency_ms=round(latency_ms, 1), attempts=attempts)
            await self._emit(
                EventType.AGENT_COMPLETED,
                context,
                AgentPayload(agent=self.name, latency_ms=latency_ms, attempts=attempts),
            )

        return self._result_model()(
            agent=self.name, output=output, latency_ms=latency_ms, attempts=attempts
        )

    @abstractmethod
    async def execute(self, payload: TInput, context: AgentContext) -> TOutput:
        """The agent's reasoning. Raise a ``ResearchAgentError`` subclass on failure."""

    def _validate_input(self, payload: TInput | Mapping[str, Any]) -> TInput:
        if isinstance(payload, self.input_schema):
            return payload
        try:
            return self.input_schema.model_validate(payload)  # type: ignore[return-value]
        except ValidationError as exc:
            raise AgentInputError(
                "Payload does not satisfy the agent input schema",
                agent=self.name,
                schema=self.input_schema.__name__,
                errors=exc.errors(include_url=False),
            ) from exc

    def _require_contract(self) -> None:
        missing = [
            attribute
            for attribute in ("name", "description", "input_schema", "output_schema")
            if getattr(type(self), attribute, None) is None
        ]
        if missing:
            raise TypeError(
                f"{type(self).__name__} must declare class attributes: {', '.join(missing)}"
            )

    @classmethod
    def _result_model(cls) -> type[AgentResult[Any]]:
        """Concrete ``AgentResult[<output_schema>]``, built once per agent class.

        Without parametrising, Pydantic would validate ``output`` against the bare
        TypeVar bound and silently drop the agent's own fields.
        """
        cached: type[AgentResult[Any]] | None = cls.__dict__.get("_result_model_cache")
        if cached is None:
            output_schema: type[BaseModel] = cls.output_schema
            cached = AgentResult[output_schema]  # type: ignore[valid-type]
            cls._result_model_cache = cached  # type: ignore[attr-defined]
        return cached

    async def _on_retry(self, context: AgentContext, attempt: int, error: BaseException) -> None:
        self.logger.warning("agent_retry", attempt=attempt, error=str(error))
        await self._emit(
            EventType.AGENT_RETRIED,
            context,
            AgentPayload(agent=self.name, attempts=attempt, error=str(error)),
        )

    async def _emit(
        self, event_type: EventType, context: AgentContext, payload: AgentPayload
    ) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            Event(type=event_type, source=self.name, run_id=context.run_id, payload=payload)
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
