from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.config.schemas import AgentSpec
from researchagent.core.events import Event, EventBus, EventType
from researchagent.core.exceptions import AgentExecutionError, AgentInputError, OutputParsingError
from researchagent.core.prompts import PromptLibrary
from researchagent.core.retry import RetryPolicy
from researchagent.services.llm_service import BoundLLM

NO_WAIT = RetryPolicy(
    max_attempts=3, initial_delay_seconds=0.0, max_delay_seconds=0.0, jitter=False
)


class EchoInput(BaseModel):
    goal: str


class EchoOutput(BaseModel):
    plan: str
    steps: list[str]


class EchoAgent(BaseAgent[EchoInput, EchoOutput]):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Test double"
    input_schema: ClassVar[type[BaseModel]] = EchoInput
    output_schema: ClassVar[type[BaseModel]] = EchoOutput

    def __init__(
        self,
        llm: BoundLLM,
        spec: AgentSpec,
        prompts: PromptLibrary,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        super().__init__(llm, spec, prompts, event_bus=event_bus)
        self.fail_times = 0
        self.executions = 0

    async def execute(self, payload: EchoInput, context: AgentContext) -> EchoOutput:
        self.executions += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OutputParsingError("model returned junk")
        return EchoOutput(plan=payload.goal, steps=["a", "b"])


@pytest.fixture
def agent(bound_llm: BoundLLM, event_bus: EventBus, prompt_library: PromptLibrary) -> EchoAgent:
    return EchoAgent(bound_llm, AgentSpec(retry=NO_WAIT), prompt_library, event_bus=event_bus)


async def test_run_validates_and_returns_typed_result(agent: EchoAgent) -> None:
    result = await agent.run({"goal": "study agentic ai"}, AgentContext())

    assert result.agent == "echo"
    assert isinstance(result.output, EchoOutput)
    assert result.output.plan == "study agentic ai"
    assert result.output.steps == ["a", "b"]  # subclass fields survive serialisation
    assert result.attempts == 1
    assert result.latency_ms >= 0


async def test_invalid_payload_raises_before_execution(agent: EchoAgent) -> None:
    with pytest.raises(AgentInputError) as excinfo:
        await agent.run({"wrong_field": 1}, AgentContext())

    assert excinfo.value.agent == "echo"
    assert agent.executions == 0


async def test_already_validated_payload_is_passed_through(agent: EchoAgent) -> None:
    result = await agent.run(EchoInput(goal="x"), AgentContext())
    assert result.output.plan == "x"


async def test_retryable_failure_is_retried(agent: EchoAgent) -> None:
    agent.fail_times = 2

    result = await agent.run({"goal": "x"}, AgentContext())

    assert (result.attempts, agent.executions) == (3, 3)


async def test_exhausted_retries_surface_as_agent_execution_error(agent: EchoAgent) -> None:
    agent.fail_times = 99

    with pytest.raises(AgentExecutionError) as excinfo:
        await agent.run({"goal": "x"}, AgentContext())

    assert excinfo.value.agent == "echo"
    assert excinfo.value.context["cause"] == "output_parsing_error"


async def test_lifecycle_events_are_emitted(
    bound_llm: BoundLLM, prompt_library: PromptLibrary
) -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.subscribe(None, handler)
    agent = EchoAgent(bound_llm, AgentSpec(retry=NO_WAIT), prompt_library, event_bus=bus)
    agent.fail_times = 1
    context = AgentContext()

    await agent.run({"goal": "x"}, context)

    types = [event.type for event in seen]
    assert types == [
        EventType.AGENT_STARTED,
        EventType.AGENT_RETRIED,
        EventType.AGENT_COMPLETED,
    ]
    assert all(event.run_id == context.run_id for event in seen)


async def test_failure_emits_failed_event(
    bound_llm: BoundLLM, prompt_library: PromptLibrary
) -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.subscribe(EventType.AGENT_FAILED, handler)
    agent = EchoAgent(
        bound_llm, AgentSpec(retry=RetryPolicy(max_attempts=1)), prompt_library, event_bus=bus
    )
    agent.fail_times = 1

    with pytest.raises(AgentExecutionError):
        await agent.run({"goal": "x"}, AgentContext())

    assert seen[0].payload.code == "output_parsing_error"  # type: ignore[union-attr]


def test_agent_without_contract_is_rejected(
    bound_llm: BoundLLM, prompt_library: PromptLibrary
) -> None:
    class Incomplete(BaseAgent[EchoInput, EchoOutput]):
        name: ClassVar[str] = "incomplete"

        async def execute(self, payload: EchoInput, context: AgentContext) -> EchoOutput:
            raise NotImplementedError

    with pytest.raises(TypeError, match="input_schema"):
        Incomplete(bound_llm, AgentSpec(), prompt_library)
