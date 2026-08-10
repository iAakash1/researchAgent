"""The loop: routing, guards, termination and budgets.

The cycle is what makes this agentic and what makes it dangerous. Every test here is
about it stopping.
"""

from __future__ import annotations

from researchagent.config.schemas import ResearchBudget
from researchagent.models.reasoning import (
    Citation,
    ResearchFinding,
    ReviewDecision,
    ReviewResult,
    TerminationReason,
    VerificationResult,
    VerificationVerdict,
)
from researchagent.models.research import ResearchPlan, ResearchQuestion, SearchStrategy
from researchagent.schemas.reasoning import BudgetLedger, QuestionState, ReasoningSession
from researchagent.schemas.workflow import ResearchState
from researchagent.workflows.guards import (
    requires_evidence_bundles,
    requires_findings,
    requires_verification,
    within_budget,
)
from researchagent.workflows.reasoning import (
    REASON_AGAIN,
    RETRIEVE_MORE,
    REVIEW,
    TERMINATE,
    route_after_verification,
    terminal_reason,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        topic="metastable failure",
        framing="How overload-driven failures start and are mitigated in distributed systems",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="Which techniques mitigate overload?",
                rationale="Mitigations differ across systems",
            )
        ],
        strategy=SearchStrategy(queries=["overload mitigation"]),
    )


def _state(session: ReasoningSession | None = None) -> ResearchState:
    return ResearchState(
        goal="study overload-driven failure in distributed systems",
        plan=_plan(),
        reasoning=session,
    )


def _finding(finding_id: str = "F-1") -> ResearchFinding:
    return ResearchFinding(
        id=finding_id,
        question_id="RQ1",
        statement="Circuit breakers are reported as an overload mitigation.",
        citations=(Citation(bundle_id="B-1", evidence_ids=("e1",), paper_ids=("manual:01",)),),
        produced_by="reasoning",
    )


def _verification(verdict: VerificationVerdict, iteration: int = 0) -> VerificationResult:
    supporting = (
        (Citation(bundle_id="B-1", evidence_ids=("e1",)),)
        if verdict is VerificationVerdict.VERIFIED
        else ()
    )
    contradicting = (
        (Citation(bundle_id="B-1", evidence_ids=("e2",)),)
        if verdict is VerificationVerdict.CONTRADICTED
        else ()
    )
    return VerificationResult(
        finding_id="F-1",
        verdict=verdict,
        supporting=supporting,
        contradicting=contradicting,
        verified_by="verification",
        iteration=iteration,
    )


class TestRouting:
    def test_a_verified_finding_goes_to_review(self) -> None:
        session = ReasoningSession(
            findings=(_finding(),), verifications=(_verification(VerificationVerdict.VERIFIED),)
        )

        assert route_after_verification(_state(session)) is REVIEW

    def test_insufficient_evidence_goes_back_to_retrieval(self) -> None:
        session = ReasoningSession(
            findings=(_finding(),),
            verifications=(_verification(VerificationVerdict.INSUFFICIENT_EVIDENCE),),
        )

        assert route_after_verification(_state(session)) is RETRIEVE_MORE

    def test_a_contradicted_finding_goes_back_to_reasoning_not_retrieval(self) -> None:
        """The claim is wrong, not under-evidenced. More retrieval would not help."""
        session = ReasoningSession(
            findings=(_finding(),),
            verifications=(_verification(VerificationVerdict.CONTRADICTED),),
        )

        assert route_after_verification(_state(session)) is REASON_AGAIN

    def test_budget_exhaustion_beats_every_verdict(self) -> None:
        """A run out of budget stops however promising the verdicts look."""
        session = ReasoningSession(
            budget=ResearchBudget(max_iterations=2),
            ledger=BudgetLedger(iterations=2),
            findings=(_finding(),),
            verifications=(_verification(VerificationVerdict.VERIFIED),),
        )

        assert route_after_verification(_state(session)) is TERMINATE

    def test_an_unverifiable_round_still_reaches_the_reviewer(self) -> None:
        """So the reviewer records why nothing was concluded, rather than the loop vanishing."""
        session = ReasoningSession(
            findings=(_finding(),),
            verifications=(_verification(VerificationVerdict.UNVERIFIABLE),),
        )

        assert route_after_verification(_state(session)) is REVIEW

    def test_no_session_terminates(self) -> None:
        assert route_after_verification(_state(None)) is TERMINATE


class TestTermination:
    def test_every_path_yields_exactly_one_reason(self) -> None:
        assert terminal_reason(_state(None)) is TerminationReason.AGENT_FAILED

    def test_accepted_findings_mean_the_questions_were_answered(self) -> None:
        review = ReviewResult(
            decision=ReviewDecision.ACCEPT, accepted_findings=("F-1",), reviewed_by="reviewer"
        )
        session = ReasoningSession(findings=(_finding(),), reviews=(review,))

        assert terminal_reason(_state(session)) is TerminationReason.ALL_QUESTIONS_ANSWERED

    def test_a_reviewer_rejection_is_reported_as_such(self) -> None:
        review = ReviewResult(decision=ReviewDecision.REJECT, reviewed_by="reviewer")
        session = ReasoningSession(findings=(_finding(),), reviews=(review,))

        assert terminal_reason(_state(session)) is TerminationReason.REVIEWER_REJECTED

    def test_all_contradicted_is_an_unresolved_contradiction(self) -> None:
        session = ReasoningSession(
            findings=(_finding(),),
            verifications=(_verification(VerificationVerdict.CONTRADICTED),),
        )

        assert terminal_reason(_state(session)) is TerminationReason.UNRESOLVED_CONTRADICTIONS

    def test_repeated_failure_to_verify_is_reported_distinctly(self) -> None:
        session = ReasoningSession(
            findings=(_finding(),),
            verifications=(
                _verification(VerificationVerdict.PARTIALLY_SUPPORTED),
                _verification(VerificationVerdict.INSUFFICIENT_EVIDENCE),
            ),
        )

        assert terminal_reason(_state(session)) is TerminationReason.VERIFICATION_REPEATEDLY_FAILED

    def test_the_budget_reason_survives_into_termination(self) -> None:
        session = ReasoningSession(
            budget=ResearchBudget(max_iterations=1), ledger=BudgetLedger(iterations=1)
        )

        assert terminal_reason(_state(session)) is TerminationReason.MAX_ITERATIONS


class TestGuards:
    def test_reasoning_refuses_to_run_without_bundles(self) -> None:
        """The v0.9 restatement of the pipeline rule."""
        result = requires_evidence_bundles().check(_state(ReasoningSession()))

        assert not result.allowed
        assert "evidence_bundles" in result.missing

    def test_reasoning_runs_once_bundles_exist(self) -> None:
        session = ReasoningSession(bundle_ids=("B-1",))

        assert requires_evidence_bundles().check(_state(session)).allowed

    def test_verification_refuses_to_run_without_findings(self) -> None:
        assert not requires_findings().check(_state(ReasoningSession())).allowed

    def test_review_refuses_to_run_without_verification(self) -> None:
        """Without this the reviewer would be the only thing between self-assessed
        confidence and an accepted result."""
        session = ReasoningSession(findings=(_finding(),))

        assert not requires_verification().check(_state(session)).allowed

    def test_review_runs_once_something_has_been_verified(self) -> None:
        session = ReasoningSession(
            findings=(_finding(),), verifications=(_verification(VerificationVerdict.VERIFIED),)
        )

        assert requires_verification().check(_state(session)).allowed

    def test_the_budget_guard_blocks_an_exhausted_run(self) -> None:
        session = ReasoningSession(
            budget=ResearchBudget(max_tool_calls=5), ledger=BudgetLedger(tool_calls=5)
        )

        result = within_budget().check(_state(session))

        assert not result.allowed
        assert "budget" in result.missing

    def test_the_budget_guard_allows_a_fresh_run(self) -> None:
        assert within_budget().check(_state(ReasoningSession())).allowed

    def test_guards_never_raise_they_report(self) -> None:
        """A guard that raised would lose the partial results it was protecting."""
        for guard in (
            requires_evidence_bundles(),
            requires_findings(),
            requires_verification(),
            within_budget(),
        ):
            result = guard.check(_state(None))
            assert isinstance(result.allowed, bool)


class TestQuestionProgress:
    def test_a_question_closes_only_on_a_verified_finding(self) -> None:
        question = QuestionState(question_id="RQ1", question="a", finding_ids=("F-1",))

        assert question.is_open, "a finding is not an answer until it is verified"
        assert question.model_copy(update={"verified_finding_ids": ("F-1",)}).is_answered


class TestRoutingUsesTheRoundThatJustRan:
    """Regression: the retry branches must be reachable.

    Verification stamps each verdict with the current iteration and then increments the
    counter. Keying the router off the counter looked for a round that had not happened,
    found nothing, and routed everything to review — the cycle existed on paper but the
    system never actually iterated. Caught by the first real research run, where every
    finding came back INSUFFICIENT_EVIDENCE and the loop reviewed instead of retrieving.
    """

    def test_a_verdict_from_the_finished_round_still_routes_the_retry(self) -> None:
        session = ReasoningSession(
            iteration=1,  # already incremented by the verification node
            findings=(_finding(),),
            verifications=(_verification(VerificationVerdict.INSUFFICIENT_EVIDENCE, iteration=0),),
        )

        assert route_after_verification(_state(session)) is RETRIEVE_MORE

    def test_a_contradiction_from_the_finished_round_still_routes_to_reasoning(self) -> None:
        session = ReasoningSession(
            iteration=1,
            findings=(_finding(),),
            verifications=(_verification(VerificationVerdict.CONTRADICTED, iteration=0),),
        )

        assert route_after_verification(_state(session)) is REASON_AGAIN

    def test_only_the_newest_round_decides(self) -> None:
        """An early contradiction must not outvote a later insufficiency."""
        session = ReasoningSession(
            iteration=2,
            findings=(_finding(),),
            verifications=(
                _verification(VerificationVerdict.CONTRADICTED, iteration=0),
                _verification(VerificationVerdict.INSUFFICIENT_EVIDENCE, iteration=1),
            ),
        )

        assert route_after_verification(_state(session)) is RETRIEVE_MORE


class TestToolCallBudgetIsCharged:
    """The tool-call cap must be checked against a counter something increments.

    The ledger recorded calls in `session.tool_calls` but never advanced
    `ledger.tool_calls`, so `max_tool_calls` was compared against a permanent zero. A real
    run made 52 calls against a cap of 40 without the guard noticing.
    """

    def test_the_guard_fires_once_the_ledger_is_charged(self) -> None:
        from researchagent.config.schemas import ResearchBudget

        session = ReasoningSession(
            budget=ResearchBudget(max_tool_calls=10), ledger=BudgetLedger(tool_calls=10)
        )

        assert session.ledger.exceeded(session.budget) is TerminationReason.BUDGET_EXHAUSTED
        assert not within_budget().check(_state(session)).allowed

    def test_an_uncharged_ledger_would_never_fire(self) -> None:
        """Pins the shape of the bug: calls recorded but not charged look like zero spend."""
        from researchagent.config.schemas import ResearchBudget

        session = ReasoningSession(budget=ResearchBudget(max_tool_calls=1))

        assert session.ledger.tool_calls == 0
        assert within_budget().check(_state(session)).allowed
