"""Tool events reach the bus, and carry no secrets."""

from __future__ import annotations

import pytest

from researchagent.container import Container
from researchagent.core.events import Event, EventType


@pytest.fixture
def captured(container: Container) -> list[Event]:
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    container.event_bus.subscribe(EventType.TOOL_CALLED, handler)
    container.event_bus.subscribe(EventType.TOOL_COMPLETED, handler)
    return seen


class TestToolEvents:
    async def test_a_tool_call_publishes_both_events(
        self, container: Container, captured: list[Event]
    ) -> None:
        toolbox = container.toolbox.for_agent("retrieval", 2)

        await toolbox.search_knowledge("overload mitigation")

        types = [event.type for event in captured]
        assert EventType.TOOL_CALLED in types
        assert EventType.TOOL_COMPLETED in types

    async def test_the_completion_event_carries_the_execution_metadata(
        self, container: Container, captured: list[Event]
    ) -> None:
        toolbox = container.toolbox.for_agent("verification", 3)

        await toolbox.get_provenance(("no-such-evidence",))

        done = next(e for e in captured if e.type is EventType.TOOL_COMPLETED)
        assert done.payload.agent == "verification"
        assert done.payload.tool == "get_provenance"
        assert done.payload.iteration == 3
        assert done.payload.latency_ms >= 0.0
        assert done.payload.succeeded is True

    async def test_events_never_echo_the_arguments(
        self, container: Container, captured: list[Event]
    ) -> None:
        """Counts and status only: an event stream that quoted queries would leak the
        corpus, and eventually whatever a user typed, into the log."""
        toolbox = container.toolbox.for_agent("retrieval", 0)

        await toolbox.search_knowledge("a distinctive secret-looking query string")

        for event in captured:
            rendered = event.model_dump_json()
            assert "distinctive secret-looking" not in rendered
            assert "authorization" not in rendered.lower()

    async def test_a_refused_call_publishes_nothing(self, container: Container) -> None:
        """A call the budget refused did not happen, and must not appear as though it did."""
        from researchagent.core.exceptions import BudgetExhaustedError

        seen: list[Event] = []

        async def handler(event: Event) -> None:
            seen.append(event)

        container.event_bus.subscribe(EventType.TOOL_CALLED, handler)
        container.toolbox.budget.max_tool_calls = 1
        container.toolbox.budget.spent = 1
        toolbox = container.toolbox.for_agent("retrieval", 0)

        with pytest.raises(BudgetExhaustedError):
            await toolbox.search_knowledge("overload")

        assert seen == []
