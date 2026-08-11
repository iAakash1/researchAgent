"""Node bodies for the agentic loop.

Each node runs one agent over the open research questions, folds the result into the
``ReasoningSession``, and charges the budget. Nodes own the bookkeeping so agents stay
stateless and unaware of the loop they are inside.

The invariant every node upholds: nothing enters the session that has not been validated
by the agent that produced it. Bundles are re-fetched from the repository rather than
carried in state, because state should hold ids and verdicts, not payloads.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.reasoning.schemas import ReasoningInput, ReasoningOutput
from researchagent.agents.retrieval.schemas import RetrievalInput, RetrievalOutput
from researchagent.agents.reviewer.schemas import ReviewerInput, ReviewerOutput
from researchagent.agents.verification.agent import VerificationAgent
from researchagent.agents.verification.schemas import VerificationInput, VerificationOutput
from researchagent.core.constants import SECONDS_PER_MILLISECOND
from researchagent.core.events import (
    EventBus,
    EventType,
    ReasoningPayload,
)
from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.interfaces.tools import ResearchToolbox
from researchagent.core.logging import get_logger
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.reasoning import (
    FindingStatus,
    ResearchFinding,
    TerminationReason,
    VerificationResult,
    VerificationVerdict,
)
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.schemas.reasoning import (
    BudgetLedger,
    QuestionState,
    ReasoningSession,
    ReasoningStage,
)
from researchagent.schemas.workflow import ResearchState, StageRecord, StageStatus, WorkflowStage
from researchagent.workflows.guards import (
    Guard,
    requires_evidence_bundles,
    requires_findings,
    requires_verification,
    within_budget,
)
from researchagent.workflows.nodes import StateUpdate
from researchagent.workflows.reasoning import terminal_reason

logger = get_logger(__name__)

# Per iteration. A question that needs more than this is a question the corpus cannot
# answer, and more rounds only cost tokens.
MAX_QUESTIONS_PER_ROUND = 3
MAX_FINDINGS_TO_VERIFY = 6


def _session(state: ResearchState) -> ReasoningSession:
    """The session, initialised from the plan on first entry."""
    if state.reasoning is not None:
        return state.reasoning
    questions = state.plan.research_questions if state.plan else []
    return ReasoningSession(
        questions=tuple(QuestionState(question_id=q.id, question=q.question) for q in questions)
    )


def _blocked(session: ReasoningSession, stage: WorkflowStage, reason: str) -> StateUpdate:
    """A guard refused. Recorded, never raised, and never silent.

    The termination reason is derived from the ledger rather than fixed: a stage blocked
    because the budget ran out is a different outcome from one blocked because the corpus
    had nothing to say, and reporting the second when the first happened would hide the
    cost of the run.
    """
    exceeded = session.ledger.exceeded(session.budget)
    logger.info(
        "reasoning_stage_blocked",
        stage=stage.value,
        reason=reason,
        termination=(exceeded or TerminationReason.INSUFFICIENT_EVIDENCE).value,
    )
    return {
        "reasoning": session.model_copy(
            update={
                "terminated": True,
                "termination_reason": exceeded or TerminationReason.INSUFFICIENT_EVIDENCE,
                "stage": ReasoningStage.TERMINATED,
            }
        ),
        "history": [
            StageRecord(
                stage=stage, agent="loop", status=StageStatus.BLOCKED, latency_ms=0.0, note=reason
            )
        ],
    }


def _remaining_tokens(session: ReasoningSession) -> int:
    """What the run may still spend, floored at zero."""
    return max(0, session.budget.max_total_tokens - session.ledger.total_tokens)


def _first_unmet(guards: list[Guard], state: ResearchState) -> str | None:
    for guard in guards:
        result = guard.check(state)
        if not result.allowed:
            return result.reason or guard.name
    return None


def retrieval_node(
    agent_for: Callable[..., BaseAgent[RetrievalInput, RetrievalOutput]],
    *,
    event_bus: EventBus | None = None,
) -> Callable[[ResearchState], Awaitable[StateUpdate]]:
    """Run retrieval for every still-open question."""

    async def node(state: ResearchState) -> StateUpdate:
        session = _session(state)
        started = time.perf_counter()

        blocked = _first_unmet([within_budget()], state.model_copy(update={"reasoning": session}))
        if blocked is not None:
            return _terminated(session, state, blocked)

        agent = agent_for("retrieval", session.iteration, _remaining_tokens(session))
        context = AgentContext(run_id=state.run_id)
        questions = {q.id: q for q in (state.plan.research_questions if state.plan else [])}

        bundle_ids = list(session.bundle_ids)
        unresolved = list(session.unresolved_questions)
        updated: list[QuestionState] = []
        attempts = session.ledger.retrieval_attempts

        for question_state in session.questions:
            if not question_state.is_open or len(updated) >= MAX_QUESTIONS_PER_ROUND:
                updated.append(question_state)
                continue
            question = questions.get(question_state.question_id)
            if question is None:
                updated.append(question_state)
                continue

            try:
                result = await agent.run(
                    RetrievalInput(
                        question=question,
                        goal=state.goal,
                        iteration=session.iteration,
                        previous_bundle_ids=question_state.bundle_ids,
                        gaps=_gaps_for(session, question_state.question_id),
                    ),
                    context,
                )
            except ResearchAgentError as exc:
                logger.warning("retrieval_agent_failed", question=question.id, error=exc.code)
                updated.append(question_state)
                continue

            output: RetrievalOutput = result.output
            attempts += 1
            bundle_ids.extend(output.bundle_ids)
            if not output.sufficient:
                unresolved.extend(output.unresolved)
            updated.append(
                question_state.model_copy(
                    update={
                        "bundle_ids": tuple(
                            dict.fromkeys(question_state.bundle_ids + output.bundle_ids)
                        ),
                        "retrieval_attempts": question_state.retrieval_attempts + 1,
                    }
                )
            )
            await _emit(
                event_bus,
                EventType.RESEARCH_ITERATION_STARTED,
                state,
                ReasoningPayload(
                    agent="retrieval",
                    iteration=session.iteration,
                    question_id=question.id,
                    bundles=len(output.bundle_ids),
                    detail=output.decision.strategy.value,
                ),
            )

        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
        return {
            "reasoning": session.model_copy(
                update={
                    "questions": tuple(updated),
                    "bundle_ids": tuple(dict.fromkeys(bundle_ids)),
                    "unresolved_questions": tuple(dict.fromkeys(unresolved)),
                    "stage": ReasoningStage.REASONING,
                    "ledger": _charge(session, agent, "retrieval").model_copy(
                        update={"retrieval_attempts": attempts}
                    ),
                    "tool_calls": session.tool_calls + _calls(agent),
                }
            ),
            "history": [
                StageRecord(
                    stage=WorkflowStage.RETRIEVAL,
                    agent="retrieval",
                    status=StageStatus.OK,
                    latency_ms=latency_ms,
                )
            ],
        }

    return node


def reasoning_node(
    agent_for: Callable[..., BaseAgent[ReasoningInput, ReasoningOutput]],
    bundles: JsonBundleRepository,
    *,
    event_bus: EventBus | None = None,
) -> Callable[[ResearchState], Awaitable[StateUpdate]]:
    async def node(state: ResearchState) -> StateUpdate:
        session = _session(state)
        probe = state.model_copy(update={"reasoning": session})
        blocked = _first_unmet([within_budget(), requires_evidence_bundles()], probe)
        if blocked is not None:
            return _blocked(session, WorkflowStage.REASONING, blocked)

        started = time.perf_counter()
        agent = agent_for("reasoning", session.iteration, _remaining_tokens(session))
        context = AgentContext(run_id=state.run_id)
        questions = {q.id: q for q in (state.plan.research_questions if state.plan else [])}

        findings = list(session.findings)
        hypotheses = list(session.hypotheses)

        for question_state in session.questions:
            question = questions.get(question_state.question_id)
            if question is None or not question_state.bundle_ids or question_state.is_answered:
                continue
            loaded = await _load_bundles(bundles, question_state.bundle_ids)
            if not loaded:
                continue

            try:
                result = await agent.run(
                    ReasoningInput(
                        question=question,
                        goal=state.goal,
                        bundles=loaded,
                        iteration=session.iteration,
                        critique=_critique_for(session, question_state.question_id),
                    ),
                    context,
                )
            except ResearchAgentError as exc:
                logger.warning("reasoning_agent_failed", question=question.id, error=exc.code)
                continue

            output: ReasoningOutput = result.output
            findings.extend(output.findings)
            hypotheses.extend(output.hypotheses)
            for finding in output.findings:
                await _emit(
                    event_bus,
                    EventType.FINDING_CREATED,
                    state,
                    ReasoningPayload(
                        agent="reasoning",
                        iteration=session.iteration,
                        question_id=question.id,
                        finding_id=finding.id,
                        detail=finding.statement[:120],
                    ),
                )
            if output.discarded_claims:
                logger.info(
                    "claims_discarded_for_missing_citations",
                    question=question.id,
                    discarded=len(output.discarded_claims),
                )

        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
        return {
            "reasoning": session.model_copy(
                update={
                    "findings": tuple(findings),
                    "hypotheses": tuple(hypotheses),
                    "ledger": _charge(session, agent, "reasoning"),
                    "stage": ReasoningStage.VERIFICATION,
                }
            ),
            "history": [
                StageRecord(
                    stage=WorkflowStage.REASONING,
                    agent="reasoning",
                    status=StageStatus.OK,
                    latency_ms=latency_ms,
                )
            ],
        }

    return node


def verification_node(
    agent_for: Callable[..., BaseAgent[Any, Any]],
    bundles: JsonBundleRepository,
    *,
    event_bus: EventBus | None = None,
) -> Callable[[ResearchState], Awaitable[StateUpdate]]:
    async def node(state: ResearchState) -> StateUpdate:
        session = _session(state)
        probe = state.model_copy(update={"reasoning": session})
        blocked = _first_unmet([requires_findings()], probe)
        if blocked is not None:
            return _blocked(session, WorkflowStage.VERIFICATION, blocked)

        started = time.perf_counter()
        agent = agent_for("verification", session.iteration, _remaining_tokens(session))
        context = AgentContext(run_id=state.run_id)
        questions = {q.id: q for q in (state.plan.research_questions if state.plan else [])}

        loaded = await _load_bundles(bundles, session.bundle_ids)
        if isinstance(agent, VerificationAgent):
            # The verifier needs the whole round's evidence, not only what the finding
            # cited — contradicting material is by definition what was left out.
            agent.with_bundles(loaded)

        verifications = list(session.verifications)
        unverified = [
            finding for finding in session.findings if session.verification_for(finding.id) is None
        ][:MAX_FINDINGS_TO_VERIFY]

        for finding in unverified:
            question = questions.get(finding.question_id)
            if question is None:
                continue
            try:
                result = await agent.run(
                    VerificationInput(
                        finding=finding, question=question, iteration=session.iteration
                    ),
                    context,
                )
            except ResearchAgentError as exc:
                logger.warning("verification_agent_failed", finding=finding.id, error=exc.code)
                continue

            output: VerificationOutput = result.output
            verifications.append(output.result)
            await _emit(
                event_bus,
                EventType.FINDING_VERIFIED
                if output.result.verdict.accepts
                else EventType.FINDING_REJECTED,
                state,
                ReasoningPayload(
                    agent="verification",
                    iteration=session.iteration,
                    finding_id=finding.id,
                    verdict=output.result.verdict.value,
                ),
            )

        findings = _apply_verdicts(session.findings, verifications)
        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
        return {
            "reasoning": session.model_copy(
                update={
                    "verifications": tuple(verifications),
                    "findings": findings,
                    "iteration": session.iteration + 1,
                    "ledger": _charge(session, agent, "verification").model_copy(
                        update={"iterations": session.ledger.iterations + 1}
                    ),
                    "stage": ReasoningStage.REVIEW,
                    "tool_calls": session.tool_calls + _calls(agent),
                }
            ),
            "history": [
                StageRecord(
                    stage=WorkflowStage.VERIFICATION,
                    agent="verification",
                    status=StageStatus.OK,
                    latency_ms=latency_ms,
                )
            ],
        }

    return node


def review_node(
    agent_for: Callable[..., BaseAgent[ReviewerInput, ReviewerOutput]],
    bundles: JsonBundleRepository,
    *,
    event_bus: EventBus | None = None,
) -> Callable[[ResearchState], Awaitable[StateUpdate]]:
    async def node(state: ResearchState) -> StateUpdate:
        session = _session(state)
        probe = state.model_copy(update={"reasoning": session})
        blocked = _first_unmet([requires_verification()], probe)
        if blocked is not None:
            return _blocked(session, WorkflowStage.REVIEW, blocked)

        started = time.perf_counter()
        loaded = await _load_bundles(bundles, session.bundle_ids)
        resolved = frozenset(item.evidence.id for bundle in loaded for item in bundle.evidence)

        reviewer = agent_for("reviewer", session.iteration, _remaining_tokens(session))
        result = await reviewer.run(
            ReviewerInput(
                goal=state.goal,
                questions=tuple(state.plan.research_questions) if state.plan else (),
                findings=session.findings,
                verifications=session.verifications,
                resolved_evidence_ids=resolved,
                iteration=session.iteration,
            ),
            AgentContext(run_id=state.run_id),
        )
        output: ReviewerOutput = result.output

        accepted = set(output.result.accepted_findings)
        findings = tuple(
            finding.model_copy(
                update={
                    "status": FindingStatus.VERIFIED
                    if finding.id in accepted
                    else FindingStatus.REJECTED
                }
            )
            if finding.id in accepted or finding.id in set(output.result.rejected_findings)
            else finding
            for finding in session.findings
        )
        questions = tuple(
            question.model_copy(
                update={
                    "verified_finding_ids": tuple(
                        finding.id
                        for finding in findings
                        if finding.question_id == question.question_id
                        and finding.status is FindingStatus.VERIFIED
                    )
                }
            )
            for question in session.questions
        )

        latency_ms = (time.perf_counter() - started) * SECONDS_PER_MILLISECOND
        return {
            "reasoning": session.model_copy(
                update={
                    "ledger": _charge(session, reviewer, "reviewer"),
                    "reviews": (*session.reviews, output.result),
                    "findings": findings,
                    "questions": questions,
                    "stage": ReasoningStage.TERMINATED,
                }
            ),
            "history": [
                StageRecord(
                    stage=WorkflowStage.REVIEW,
                    agent="reviewer",
                    status=StageStatus.OK,
                    latency_ms=latency_ms,
                    note=output.result.decision.value,
                )
            ],
        }

    return node


def terminate_node(
    *, event_bus: EventBus | None = None
) -> Callable[[ResearchState], Awaitable[StateUpdate]]:
    """Record why the loop stopped. The one node every path passes through."""

    async def node(state: ResearchState) -> StateUpdate:
        session = _session(state)
        reason = session.termination_reason or terminal_reason(state)
        logger.info(
            "research_terminated",
            run_id=state.run_id,
            reason=reason.value,
            iterations=session.iteration,
            findings=len(session.findings),
            verified=len(session.verified_findings),
        )
        await _emit(
            event_bus,
            EventType.RESEARCH_TERMINATED,
            state,
            ReasoningPayload(
                agent="loop",
                iteration=session.iteration,
                detail=reason.value,
                findings=len(session.verified_findings),
            ),
        )
        return {
            "reasoning": session.model_copy(
                update={
                    "terminated": True,
                    "termination_reason": reason,
                    "stage": ReasoningStage.TERMINATED,
                }
            )
        }

    return node


def _terminated(session: ReasoningSession, state: ResearchState, reason: str) -> StateUpdate:
    exceeded = session.ledger.exceeded(session.budget) or TerminationReason.BUDGET_EXHAUSTED
    logger.info("reasoning_halted_before_round", reason=reason, termination=exceeded.value)
    return {
        "reasoning": session.model_copy(
            update={
                "terminated": True,
                "termination_reason": exceeded,
                "stage": ReasoningStage.TERMINATED,
            }
        )
    }


def _apply_verdicts(
    findings: tuple[ResearchFinding, ...], verifications: Sequence[VerificationResult]
) -> tuple[ResearchFinding, ...]:
    """Reflect each verdict on its finding, without promoting anything to VERIFIED.

    Promotion is the reviewer's decision alone: a finding the verifier liked is not yet a
    result.
    """
    verdicts: dict[str, VerificationVerdict] = {
        item.finding_id: item.verdict for item in verifications
    }

    updated = []
    for finding in findings:
        verdict = verdicts.get(finding.id)
        if verdict is None:
            updated.append(finding)
        elif verdict is VerificationVerdict.CONTRADICTED:
            updated.append(finding.model_copy(update={"status": FindingStatus.REJECTED}))
        elif verdict.wants_more_evidence:
            updated.append(finding)
        else:
            updated.append(
                finding.model_copy(update={"status": FindingStatus.INSUFFICIENT_EVIDENCE})
            )
    return tuple(updated)


async def _load_bundles(
    repository: JsonBundleRepository, bundle_ids: tuple[str, ...]
) -> tuple[EvidenceBundle, ...]:
    """Bundles live in the repository; state carries only their ids."""
    loaded = []
    for bundle_id in bundle_ids:
        bundle = await repository.get(bundle_id)
        if bundle is not None:
            loaded.append(bundle)
    return tuple(loaded)


def _gaps_for(session: ReasoningSession, question_id: str) -> tuple[str, ...]:
    """What verification said was missing, so the next retrieval widens rather than repeats."""
    gaps: list[str] = []
    for verification in session.verifications:
        finding = session.finding(verification.finding_id)
        if finding is not None and finding.question_id == question_id:
            gaps.extend(verification.unsupported_claims)
    return tuple(dict.fromkeys(gaps))[:5]


def _critique_for(session: ReasoningSession, question_id: str) -> tuple[str, ...]:
    critique: list[str] = []
    for verification in session.verifications:
        finding = session.finding(verification.finding_id)
        if finding is None or finding.question_id != question_id:
            continue
        critique.extend(verification.overstatements)
        if verification.reasoning and not verification.verdict.accepts:
            critique.append(verification.reasoning[:200])
    return tuple(dict.fromkeys(critique))[:5]


def _charge(session: ReasoningSession, agent: object, name: str) -> BudgetLedger:
    """Fold one agent's spend into the session ledger.

    Read off the ``BoundLLM`` the agent used rather than estimated from prompt length:
    when a provider reports usage, that is the number, and when it does not the call is
    counted as unmeasured instead of free.
    """
    llm = getattr(agent, "llm", None)
    report = getattr(llm, "usage", None)
    ledger = session.ledger.model_copy(
        update={"tool_calls": len(session.tool_calls) + len(_calls(agent))}
    )
    if report is None:
        return ledger

    charged = ledger.with_tokens(name, report.usage)
    if not report.is_complete:
        logger.info(
            "unmeasured_llm_calls",
            agent=name,
            unmeasured=report.unmeasured_calls,
            of=report.calls,
        )
    return charged.model_copy(
        update={"unmeasured_calls": charged.unmeasured_calls + report.unmeasured_calls}
    )


def _calls(agent: object) -> tuple[object, ...]:
    """Tool calls the agent made, when it owns a toolbox."""
    toolbox = getattr(agent, "_toolbox", None)
    if isinstance(toolbox, ResearchToolbox):
        return toolbox.calls
    return ()


async def _emit(
    event_bus: EventBus | None,
    event_type: EventType,
    state: ResearchState,
    payload: ReasoningPayload,
) -> None:
    if event_bus is None:
        return
    from researchagent.core.events import Event

    await event_bus.publish(Event(type=event_type, run_id=state.run_id, payload=payload))
