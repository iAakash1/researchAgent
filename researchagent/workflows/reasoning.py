"""The agentic research loop.

    START -> retrieval -> reasoning -> verification -> [route] -> review -> END
                  ^            ^                          |
                  |            |                          |
                  +-- insufficient evidence --------------+
                               |
                               +-- contradicted ----------+

Where v0.4-v0.8 were a pipeline — each stage running once, in order — this is the first
graph with a cycle in it, and the cycle is what makes the system agentic rather than
merely automated. A verification verdict of INSUFFICIENT_EVIDENCE sends the run back to
retrieve differently; CONTRADICTED sends it back to reason differently.

Three properties keep the cycle honest:

* **It cannot spin.** ``within_budget`` is checked before every round, and the router
  terminates on iteration, token and tool-call limits.
* **It cannot stop silently.** Every exit sets a ``TerminationReason``.
* **It cannot skip a step.** Guards, not the agents, enforce that reasoning has bundles,
  verification has findings, and review has verdicts.

Control flow lives here and only here — no agent knows what runs before or after it.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from researchagent.core.logging import get_logger
from researchagent.models.reasoning import TerminationReason, VerificationVerdict
from researchagent.schemas.reasoning import ReasoningStage
from researchagent.schemas.workflow import ResearchState, WorkflowStage

logger = get_logger(__name__)

ReasoningBranch = Literal["retrieve_more", "reason_again", "review", "terminate"]

RETRIEVE_MORE: ReasoningBranch = "retrieve_more"
REASON_AGAIN: ReasoningBranch = "reason_again"
REVIEW: ReasoningBranch = "review"
TERMINATE: ReasoningBranch = "terminate"


def route_after_verification(state: ResearchState) -> ReasoningBranch:
    """The loop's only branch point.

    Read in priority order: budget first (a run out of money stops regardless of how
    promising the verdicts look), then the verdicts themselves, then progress.
    """
    session = state.reasoning
    if session is None:
        return TERMINATE

    exceeded = session.ledger.exceeded(session.budget)
    if exceeded is not None:
        logger.info(
            "reasoning_budget_reached",
            run_id=state.run_id,
            reason=exceeded.value,
            iteration=session.iteration,
        )
        return TERMINATE

    verified = [
        item for item in session.verifications if item.verdict is VerificationVerdict.VERIFIED
    ]
    if verified:
        # Something survived adversarial checking; hand the body of work to the reviewer.
        return REVIEW

    latest = _latest_verdicts(state)
    if any(verdict.wants_rereasoning for verdict in latest):
        # The claim is wrong, not under-evidenced. More retrieval would not help.
        return REASON_AGAIN
    if any(verdict.wants_more_evidence for verdict in latest):
        return RETRIEVE_MORE
    # Nothing verified and nothing worth another round: let the reviewer record why.
    return REVIEW


def _latest_verdicts(state: ResearchState) -> list[VerificationVerdict]:
    """Verdicts from the most recent round that actually produced any.

    Deliberately keyed on the highest iteration *present in the verdicts* rather than on
    ``session.iteration``: the verification node stamps its results with the current
    iteration and then increments, so comparing against the session counter looks for a
    round that has not happened yet, finds nothing, and makes the retry branches
    unreachable — the loop would exist on paper but never actually iterate.
    """
    session = state.reasoning
    if session is None or not session.verifications:
        return []
    newest = max(item.iteration for item in session.verifications)
    return [item.verdict for item in session.verifications if item.iteration == newest]


def terminal_reason(state: ResearchState) -> TerminationReason:
    """Why the loop stopped. Every path through the graph produces exactly one."""
    session = state.reasoning
    if session is None:
        return TerminationReason.AGENT_FAILED

    exceeded = session.ledger.exceeded(session.budget)
    if exceeded is not None:
        return exceeded

    review = session.latest_review
    if review is not None and review.accepted_findings:
        return TerminationReason.ALL_QUESTIONS_ANSWERED
    if review is not None:
        return TerminationReason.REVIEWER_REJECTED

    verdicts = [item.verdict for item in session.verifications]
    if verdicts and all(verdict is VerificationVerdict.CONTRADICTED for verdict in verdicts):
        return TerminationReason.UNRESOLVED_CONTRADICTIONS
    if verdicts and not any(verdict.accepts for verdict in verdicts):
        return TerminationReason.VERIFICATION_REPEATEDLY_FAILED
    return TerminationReason.INSUFFICIENT_EVIDENCE


def build_reasoning_graph(
    *,
    retrieval_node: Any,
    reasoning_node: Any,
    verification_node: Any,
    review_node: Any,
    terminate_node: Any,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[ResearchState, Any, Any, Any]:
    """Compile the loop. Node bodies live in ``workflows/agentic.py``."""
    graph: StateGraph[ResearchState, Any, Any, Any] = StateGraph(ResearchState)

    graph.add_node(WorkflowStage.RETRIEVAL.value, retrieval_node)
    graph.add_node(WorkflowStage.REASONING.value, reasoning_node)
    graph.add_node(WorkflowStage.VERIFICATION.value, verification_node)
    graph.add_node(WorkflowStage.REVIEW.value, review_node)
    graph.add_node(ReasoningStage.TERMINATED.value, terminate_node)

    graph.add_edge(START, WorkflowStage.RETRIEVAL.value)
    graph.add_edge(WorkflowStage.RETRIEVAL.value, WorkflowStage.REASONING.value)
    graph.add_edge(WorkflowStage.REASONING.value, WorkflowStage.VERIFICATION.value)

    graph.add_conditional_edges(
        WorkflowStage.VERIFICATION.value,
        route_after_verification,
        {
            RETRIEVE_MORE: WorkflowStage.RETRIEVAL.value,
            REASON_AGAIN: WorkflowStage.REASONING.value,
            REVIEW: WorkflowStage.REVIEW.value,
            TERMINATE: ReasoningStage.TERMINATED.value,
        },
    )
    graph.add_edge(WorkflowStage.REVIEW.value, ReasoningStage.TERMINATED.value)
    graph.add_edge(ReasoningStage.TERMINATED.value, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "reasoning_graph_compiled",
        checkpointing=checkpointer is not None,
        stages=[
            stage.value
            for stage in (
                WorkflowStage.RETRIEVAL,
                WorkflowStage.REASONING,
                WorkflowStage.VERIFICATION,
                WorkflowStage.REVIEW,
            )
        ],
    )
    return compiled
