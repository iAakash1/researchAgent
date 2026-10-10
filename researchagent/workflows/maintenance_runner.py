"""Runs the maintenance workflow for one saved topic and returns its diff."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from researchagent.core.exceptions import WorkflowExecutionError
from researchagent.core.logging import get_logger
from researchagent.models.integrity import IntegrityObservation
from researchagent.models.maintenance import ConclusionDiff, MaintainedTopic
from researchagent.schemas.maintenance import MaintenanceRequest, MaintenanceState
from researchagent.schemas.reasoning import ReasoningSession
from researchagent.schemas.workflow import ResearchState, RunStatus
from researchagent.services.maintenance.service import ConclusionMaintenanceService
from researchagent.workflows.maintenance import (
    ConclusionGenerator,
    MaintenanceAgentFactory,
    build_maintenance_graph,
)
from researchagent.workflows.reasoning_runner import ReasoningRunner

logger = get_logger(__name__)

# Node executions: seven nodes, no cycles. Anything above this is a wiring bug.
_RECURSION_LIMIT = 20


def reasoning_generator(runner: ReasoningRunner) -> ConclusionGenerator:
    """The normal reasoning -> verification -> reviewer loop, over a saved topic's plan."""

    async def generate(
        topic: MaintainedTopic, run_id: str, excluded: frozenset[str]
    ) -> ReasoningSession | None:
        state = ResearchState(
            run_id=run_id, goal=topic.goal, plan=topic.plan, status=RunStatus.COMPLETED
        )
        final = await runner.run(state, excluded_paper_ids=excluded)
        return final.reasoning

    return generate


class MaintenanceRunner:
    def __init__(
        self,
        service: ConclusionMaintenanceService,
        agent_for: MaintenanceAgentFactory,
        generate: ConclusionGenerator,
    ) -> None:
        self._service = service
        self._graph: Any = build_maintenance_graph(service, agent_for, generate)

    @property
    def service(self) -> ConclusionMaintenanceService:
        return self._service

    async def run(
        self,
        topic_id: str,
        request: MaintenanceRequest | None = None,
        *,
        injected: Iterable[IntegrityObservation] = (),
        skip_registry: bool = False,
    ) -> ConclusionDiff:
        """One maintenance pass. ``injected``/``skip_registry`` exist for the evaluation
        harness, which flips statuses deliberately instead of asking a registry."""
        await self._service.topic(topic_id)  # unknown topic -> 404 before anything runs
        ask = request or MaintenanceRequest()
        state = MaintenanceState(
            topic_id=topic_id,
            mode=ask.mode or self._service.config.mode,
            force_refresh=ask.force_refresh,
            requested_new_paper_ids=ask.new_paper_ids,
            injected=tuple(injected),
            skip_registry=skip_registry,
        )
        logger.info(
            "maintenance_started", topic_id=topic_id, run_id=state.run_id, mode=state.mode.value
        )
        raw = await self._graph.ainvoke(state, config={"recursion_limit": _RECURSION_LIMIT})
        final = MaintenanceState.model_validate(raw)
        if final.diff is None:
            raise WorkflowExecutionError("Maintenance finished without a diff", run_id=state.run_id)
        return final.diff
