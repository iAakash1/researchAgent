"""Session state, budgets and termination."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchagent.config.schemas import ResearchBudget
from researchagent.models.reasoning import TerminationReason
from researchagent.schemas.reasoning import BudgetLedger, QuestionState, ReasoningSession


class TestBudget:
    def test_a_per_agent_budget_larger_than_the_total_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds"):
            ResearchBudget(max_tokens_per_agent=500_000, max_total_tokens=1024)

    def test_an_empty_ledger_is_within_budget(self) -> None:
        assert BudgetLedger().exceeded(ResearchBudget()) is None

    def test_iteration_cap_reports_max_iterations_specifically(self) -> None:
        """A run that ran out of rounds is a different story from one that ran out of tokens."""
        ledger = BudgetLedger(iterations=3)

        assert ledger.exceeded(ResearchBudget(max_iterations=3)) is TerminationReason.MAX_ITERATIONS

    @pytest.mark.parametrize(
        "ledger",
        [
            BudgetLedger(total_tokens=200_000),
            BudgetLedger(tool_calls=40),
            BudgetLedger(retrieval_attempts=8),
            BudgetLedger(tokens_by_agent={"reasoning": 32_000}),
        ],
    )
    def test_every_other_limit_reports_budget_exhausted(self, ledger: BudgetLedger) -> None:
        assert ledger.exceeded(ResearchBudget()) is TerminationReason.BUDGET_EXHAUSTED

    def test_token_accounting_is_per_agent_and_total(self) -> None:
        ledger = BudgetLedger().with_tokens("reasoning", 100).with_tokens("reasoning", 50)

        assert ledger.total_tokens == 150
        assert ledger.tokens_by_agent == {"reasoning": 150}

    def test_the_ledger_is_never_mutated_in_place(self) -> None:
        original = BudgetLedger()

        original.with_tokens("reasoning", 100)

        assert original.total_tokens == 0


class TestSession:
    def test_a_terminated_session_must_say_why(self) -> None:
        """Never stop silently."""
        with pytest.raises(ValidationError, match="why it terminated"):
            ReasoningSession(terminated=True)

    def test_a_question_is_open_until_answered_or_exhausted(self) -> None:
        question = QuestionState(question_id="RQ1", question="what?")

        assert question.is_open
        assert not question.is_answered
        assert not question.model_copy(update={"exhausted_reason": "no evidence"}).is_open
        assert question.model_copy(update={"verified_finding_ids": ("F-1",)}).is_answered

    def test_open_questions_drive_the_loop(self) -> None:
        session = ReasoningSession(
            questions=(
                QuestionState(question_id="RQ1", question="a", verified_finding_ids=("F-1",)),
                QuestionState(question_id="RQ2", question="b"),
            )
        )

        assert [q.question_id for q in session.open_questions] == ["RQ2"]
