"""Runs the agentic loop.

Owns graph compilation and the one translation LangGraph needs: it returns a dict, the
rest of the system speaks ``ResearchState``. Kept separate from ``WorkflowRunner`` because
the loop can be run on its own — over an already-built corpus, without re-running
discovery and extraction — which is how the real research task is executed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from researchagent.agents.base import BaseAgent
from researchagent.config.schemas import ReasoningConfig
from researchagent.core.events import EventBus
from researchagent.core.logging import get_logger
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.schemas.reasoning import QuestionState, ReasoningSession
from researchagent.schemas.workflow import ResearchState
from researchagent.workflows import agentic
from researchagent.workflows.reasoning import build_reasoning_graph

logger = get_logger(__name__)

# (name, iteration, tokens_remaining) -> agent. `tokens_remaining` is None when the
# caller does not budget tokens, which keeps the factory usable in tests.
AgentFactory = Callable[..., BaseAgent[Any, Any]]


class ReasoningRunner:
    """Compiles and executes the retrieve -> reason -> verify -> review loop."""

    name = "reasoning_runner"

    def __init__(
        self,
        agent_for: AgentFactory,
        bundles: JsonBundleRepository,
        config: ReasoningConfig,
        *,
        event_bus: EventBus | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self._agent_for = agent_for
        self._bundles = bundles
        self._config = config
        self._event_bus = event_bus
        self._graph = build_reasoning_graph(
            retrieval_node=agentic.retrieval_node(agent_for, event_bus=event_bus),
            reasoning_node=agentic.reasoning_node(agent_for, bundles, event_bus=event_bus),
            verification_node=agentic.verification_node(agent_for, bundles, event_bus=event_bus),
            review_node=agentic.review_node(agent_for, bundles, event_bus=event_bus),
            terminate_node=agentic.terminate_node(event_bus=event_bus),
            checkpointer=checkpointer,
        )

    async def run(self, state: ResearchState) -> ResearchState:
        """Execute the loop from a state that already carries a plan.

        The session is seeded here rather than inside a node so the budget comes from
        configuration exactly once, and a caller can override it before starting.
        """
        seeded = state.model_copy(update={"reasoning": state.reasoning or self._new_session(state)})
        # `recursion_limit` is LangGraph's own backstop; the real limit is the budget
        # checked inside the loop. Setting it well above the budget means a hit here
        # signals a routing bug rather than a run that legitimately took many rounds.
        raw = await self._graph.ainvoke(
            seeded, config={"recursion_limit": self._config.budget.max_iterations * 8 + 10}
        )
        final = ResearchState.model_validate(raw)
        session = final.reasoning
        logger.info(
            "reasoning_run_complete",
            run_id=final.run_id,
            iterations=session.iteration if session else 0,
            findings=len(session.findings) if session else 0,
            verified=len(session.verified_findings) if session else 0,
            termination=session.termination_reason.value
            if session and session.termination_reason
            else None,
        )
        return final

    def _new_session(self, state: ResearchState) -> ReasoningSession:
        questions = state.plan.research_questions if state.plan else []
        return ReasoningSession(
            budget=self._config.budget,
            questions=tuple(
                QuestionState(question_id=q.id, question=q.question) for q in questions
            ),
        )
