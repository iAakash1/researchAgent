"""The conclusion-maintenance workflow.

    START -> refresh -> [mode] -> propagate -> reverify -> review -> discover -> finalize -> END
                             \\-> regenerate ---------------------------------/

* **refresh** — integrity of every supporting (and new) paper, against what the topic
  last absorbed;
* **propagate** — deterministic rules over the dependency graph (no model);
* **reverify** — the existing verification agent, on the remaining evidence only, for
  conclusions that lost part of their support;
* **review** — the existing reviewer, with registry integrity status, over re-verified
  conclusions;
* **discover** — the normal reasoning loop for conclusions citing new papers;
* **regenerate** — ``full_rerun`` mode: the whole loop again, the cost baseline;
* **finalize** — append versions, write the diff.

Orchestration only: every decision is a service call; agents are called here because
agents are not allowed to call each other, and services are not allowed to call agents.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.reviewer.schemas import ReviewerInput, ReviewerOutput
from researchagent.agents.verification.agent import VerificationAgent
from researchagent.agents.verification.schemas import VerificationInput, VerificationOutput
from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.logging import get_logger
from researchagent.models.maintenance import (
    ChangeType,
    MaintainedTopic,
    MaintenanceFailure,
    MaintenanceMode,
)
from researchagent.models.reasoning import ResearchFinding, VerificationResult
from researchagent.schemas.maintenance import ConclusionOutcome, MaintenanceState
from researchagent.schemas.reasoning import ReasoningSession
from researchagent.services.maintenance.service import ConclusionMaintenanceService

logger = get_logger(__name__)

# (agent name, papers its tools must not see) -> agent
MaintenanceAgentFactory = Callable[[str, frozenset[str]], BaseAgent[Any, Any]]
# (topic, run id, excluded papers) -> the reasoning loop's session
ConclusionGenerator = Callable[
    [MaintainedTopic, str, frozenset[str]], Awaitable[ReasoningSession | None]
]
Update = dict[str, Any]
Node = Callable[[MaintenanceState], Awaitable[Update]]


def _failure(
    stage: str, exc: ResearchAgentError, conclusion_id: str | None = None
) -> MaintenanceFailure:
    return MaintenanceFailure(
        stage=stage, code=exc.code, message=exc.message, conclusion_id=conclusion_id
    )


def refresh_node(service: ConclusionMaintenanceService) -> Node:
    async def node(state: MaintenanceState) -> Update:
        topic = await service.topic(state.topic_id)
        records = await service.records(topic.topic_id)
        new = (
            await service.new_papers(topic, state.requested_new_paper_ids)
            if state.mode is MaintenanceMode.INCREMENTAL
            else ()
        )
        refreshed = await service.refresh(
            topic,
            records,
            new,
            run_id=state.run_id,
            force=state.force_refresh,
            injected=state.injected,
            skip_registry=state.skip_registry,
        )
        return {
            "topic": topic,
            "records": records,
            "new_paper_ids": new,
            "changes": refreshed.changes,
            "effective": refreshed.effective,
            "lookup_failures": refreshed.lookup_failures,
            "papers_checked": refreshed.papers_checked,
        }

    return node


def route_mode(state: MaintenanceState) -> Literal["propagate", "regenerate"]:
    return "regenerate" if state.mode is MaintenanceMode.FULL_RERUN else "propagate"


def propagate_node(service: ConclusionMaintenanceService) -> Node:
    async def node(state: MaintenanceState) -> Update:
        plan = await service.plan(state)
        return {"plan": plan, "outcomes": plan.outcomes}

    return node


def reverify_node(
    service: ConclusionMaintenanceService, agent_for: MaintenanceAgentFactory
) -> Node:
    async def node(state: MaintenanceState) -> Update:
        plan, topic = state.plan, state.topic
        if plan is None or topic is None or not plan.reverify:
            return {}
        excluded = service.excluded_papers(
            state.effective, service.config.propagation.excluded_statuses
        )
        outcomes: list[ConclusionOutcome] = list(state.outcomes)
        failures = list(state.failures)
        reverified: dict[str, ResearchFinding] = {}
        verifications: list[VerificationResult] = []
        resolved: set[str] = set()

        for request in plan.reverify:
            conclusion_id = request.record.conclusion_id
            if not service.config.propagation.reverify:
                outcomes.append(service.resolve(request, None, run_id=state.run_id))
                continue
            try:
                context = await service.reverification_context(request, topic, run_id=state.run_id)
                agent = agent_for("verification", excluded)
                if isinstance(agent, VerificationAgent):
                    agent.with_bundles((context.bundle,))
                result = await agent.run(
                    VerificationInput(finding=context.finding, question=context.question),
                    AgentContext(run_id=state.run_id),
                )
            except ResearchAgentError as exc:
                failures.append(_failure("reverify", exc, conclusion_id))
                outcomes.append(
                    service.resolve(request, None, run_id=state.run_id, failure=exc.code)
                )
                continue
            output: VerificationOutput = result.output
            outcomes.append(service.resolve(request, output.result, run_id=state.run_id))
            reverified[conclusion_id] = context.finding
            verifications.append(output.result)
            resolved.update(item.evidence.id for item in context.bundle.evidence)

        return {
            "outcomes": tuple(outcomes),
            "failures": tuple(failures),
            "reverified": reverified,
            "verifications": tuple(verifications),
            "resolved_evidence_ids": frozenset(resolved),
        }

    return node


def review_node(service: ConclusionMaintenanceService, agent_for: MaintenanceAgentFactory) -> Node:
    async def node(state: MaintenanceState) -> Update:
        topic = state.topic
        kept = [
            outcome
            for outcome in state.outcomes
            if outcome.change_type is ChangeType.KEPT
            and outcome.record.conclusion_id in state.reverified
        ]
        if topic is None or not kept or not service.config.review.enabled:
            return {}
        findings = tuple(state.reverified[item.record.conclusion_id] for item in kept)
        cited = {pid for finding in findings for pid in finding.paper_ids}
        excluded = service.excluded_papers(
            state.effective, service.config.propagation.excluded_statuses
        )
        try:
            result = await agent_for("reviewer", excluded).run(
                ReviewerInput(
                    goal=topic.goal,
                    questions=tuple(topic.plan.research_questions),
                    findings=findings,
                    verifications=state.verifications,
                    resolved_evidence_ids=state.resolved_evidence_ids,
                    integrity=tuple(
                        state.effective[pid] for pid in sorted(cited) if pid in state.effective
                    ),
                ),
                AgentContext(run_id=state.run_id),
            )
        except ResearchAgentError as exc:
            return {"failures": (*state.failures, _failure("review", exc))}
        output: ReviewerOutput = result.output
        return {
            "review": output.result,
            "outcomes": service.after_review(
                state.outcomes,
                output.result,
                {conclusion_id: finding.id for conclusion_id, finding in state.reverified.items()},
            ),
        }

    return node


def discover_node(service: ConclusionMaintenanceService, generate: ConclusionGenerator) -> Node:
    async def node(state: MaintenanceState) -> Update:
        topic = state.topic
        if (
            topic is None
            or not state.new_paper_ids
            or not service.config.new_papers.generate_conclusions
        ):
            return {}
        excluded = service.excluded_papers(
            state.effective, service.config.propagation.excluded_statuses
        )
        try:
            session = await generate(topic, state.run_id, excluded)
        except ResearchAgentError as exc:
            return {"failures": (*state.failures, _failure("discover", exc))}
        return {
            "new_findings": service.accepted_findings(session, state.records),
            "new_verifications": session.verifications if session else (),
        }

    return node


def regenerate_node(service: ConclusionMaintenanceService, generate: ConclusionGenerator) -> Node:
    async def node(state: MaintenanceState) -> Update:
        topic = state.topic
        if topic is None:
            return {}
        excluded = service.excluded_papers(
            state.effective, service.config.full_rerun.exclude_statuses
        )
        try:
            session = await generate(topic, state.run_id, excluded)
        except ResearchAgentError as exc:
            return {"failures": (*state.failures, _failure("regenerate", exc))}
        return {
            "new_findings": service.accepted_findings(session, ()),
            "new_verifications": session.verifications if session else (),
        }

    return node


def finalize_node(service: ConclusionMaintenanceService) -> Node:
    async def node(state: MaintenanceState) -> Update:
        return {"diff": await service.finalize(state)}

    return node


def build_maintenance_graph(
    service: ConclusionMaintenanceService,
    agent_for: MaintenanceAgentFactory,
    generate: ConclusionGenerator,
) -> CompiledStateGraph[MaintenanceState, Any, Any, Any]:
    graph: StateGraph[MaintenanceState, Any, Any, Any] = StateGraph(MaintenanceState)
    # Typed as Any at the LangGraph boundary, as in workflows/reasoning.py: node bodies
    # return partial updates, which LangGraph's overloads cannot express for a model state.
    nodes: dict[str, Any] = {
        "refresh": refresh_node(service),
        "propagate": propagate_node(service),
        "reverify": reverify_node(service, agent_for),
        "review": review_node(service, agent_for),
        "discover": discover_node(service, generate),
        "regenerate": regenerate_node(service, generate),
        "finalize": finalize_node(service),
    }
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "refresh")
    graph.add_conditional_edges(
        "refresh", route_mode, {"propagate": "propagate", "regenerate": "regenerate"}
    )
    graph.add_edge("propagate", "reverify")
    graph.add_edge("reverify", "review")
    graph.add_edge("review", "discover")
    graph.add_edge("discover", "finalize")
    graph.add_edge("regenerate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
