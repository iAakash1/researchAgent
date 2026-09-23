"""Run-scoped model-call accounting assembled from typed events."""

from __future__ import annotations

from collections import defaultdict

from researchagent.core.events import Event, EventBus, EventType, LLMCallPayload
from researchagent.schemas.result import ModelCallResult, ModelUsageResult


class ModelUsageTracker:
    def __init__(self, event_bus: EventBus) -> None:
        self._calls: dict[str, list[ModelCallResult]] = defaultdict(list)
        self._unsubscribe = event_bus.subscribe(EventType.LLM_CALL_COMPLETED, self._capture)

    async def _capture(self, event: Event) -> None:
        if event.run_id is None or not isinstance(event.payload, LLMCallPayload):
            return
        payload = event.payload
        self._calls[event.run_id].append(
            ModelCallResult.model_validate(payload.model_dump(exclude={"alias"}))
        )

    def summary(self, run_id: str) -> ModelUsageResult:
        calls = tuple(self._calls.get(run_id, ()))
        known_costs = [item.estimated_cost_usd for item in calls]
        return ModelUsageResult(
            calls=calls,
            providers=tuple(dict.fromkeys(item.provider for item in calls)),
            models=tuple(dict.fromkeys(item.model for item in calls)),
            fallback_used=any(item.fallback_used for item in calls),
            prompt_tokens=sum(item.prompt_tokens for item in calls),
            completion_tokens=sum(item.completion_tokens for item in calls),
            total_tokens=sum(item.total_tokens for item in calls),
            estimated_cost_usd=(
                round(sum(value for value in known_costs if value is not None), 8)
                if calls and all(value is not None for value in known_costs)
                else None
            ),
        )

    def aclose(self) -> None:
        self._unsubscribe()
