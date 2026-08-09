from __future__ import annotations

from researchagent.core.events import AgentPayload, Event, EventBus, EventType


async def test_publishes_to_type_and_wildcard_subscribers() -> None:
    bus = EventBus()
    typed: list[Event] = []
    wildcard: list[Event] = []

    async def on_typed(event: Event) -> None:
        typed.append(event)

    async def on_any(event: Event) -> None:
        wildcard.append(event)

    bus.subscribe(EventType.AGENT_STARTED, on_typed)
    bus.subscribe(None, on_any)

    await bus.publish(
        Event(type=EventType.AGENT_STARTED, source="planner", payload=AgentPayload(agent="p"))
    )
    await bus.publish(
        Event(type=EventType.AGENT_FAILED, source="planner", payload=AgentPayload(agent="p"))
    )

    assert len(typed) == 1
    assert len(wildcard) == 2


async def test_failing_subscriber_does_not_break_delivery() -> None:
    bus = EventBus()
    delivered: list[Event] = []

    async def boom(event: Event) -> None:
        raise RuntimeError("subscriber exploded")

    async def good(event: Event) -> None:
        delivered.append(event)

    bus.subscribe(EventType.AGENT_COMPLETED, boom)
    bus.subscribe(EventType.AGENT_COMPLETED, good)

    await bus.publish(Event(type=EventType.AGENT_COMPLETED))

    assert len(delivered) == 1


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    unsubscribe = bus.subscribe(EventType.WORKFLOW_STARTED, handler)
    await bus.publish(Event(type=EventType.WORKFLOW_STARTED))
    unsubscribe()
    await bus.publish(Event(type=EventType.WORKFLOW_STARTED))

    assert len(seen) == 1
    assert bus.subscriber_count(EventType.WORKFLOW_STARTED) == 0


async def test_publish_without_subscribers_is_a_noop() -> None:
    await EventBus().publish(Event(type=EventType.AGENT_STARTED))


async def test_payloads_are_typed_models_not_dicts() -> None:
    """A subscriber that has to guess an event's keys breaks silently on refactors."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Event(type=EventType.AGENT_STARTED, payload={"agent": "planner"})  # type: ignore[arg-type]

    event = Event(type=EventType.AGENT_STARTED, payload=AgentPayload(agent="planner"))
    assert isinstance(event.payload, AgentPayload)
    assert event.payload.agent == "planner"


async def test_emit_is_the_one_line_publish() -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.subscribe(EventType.AGENT_COMPLETED, handler)
    await bus.emit(EventType.AGENT_COMPLETED, AgentPayload(agent="planner"), run_id="r1")

    assert seen[0].run_id == "r1"
