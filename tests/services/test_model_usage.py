from __future__ import annotations

from researchagent.core.events import EventBus, EventType, LLMCallPayload
from researchagent.services.model_usage import ModelUsageTracker


async def test_run_summary_retains_provider_model_fallback_tokens_and_cost() -> None:
    bus = EventBus()
    tracker = ModelUsageTracker(bus)
    await bus.emit(
        EventType.LLM_CALL_COMPLETED,
        LLMCallPayload(
            alias="reasoning",
            agent="verification",
            provider="ollama",
            model="llama3.1:8b",
            latency_ms=21.5,
            attempts=4,
            fallback_used=True,
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            estimated_cost_usd=0.0,
        ),
        run_id="run-1",
    )

    result = tracker.summary("run-1")

    assert result.providers == ("ollama",)
    assert result.models == ("llama3.1:8b",)
    assert result.calls[0].agent == "verification"
    assert result.fallback_used is True
    assert result.total_tokens == 125
    assert result.estimated_cost_usd == 0.0
    tracker.aclose()
