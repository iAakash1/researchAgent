"""Hard ceilings, not stopping conditions.

A limit checked only between rounds lets a round begin under it and finish above it — a
real run made 48 tool calls against a cap of 40. These tests pin the difference: the call
that would cross the line does not execute.
"""

from __future__ import annotations

import pytest

from researchagent.container import Container
from researchagent.core.exceptions import BudgetExhaustedError
from researchagent.core.interfaces.llm import Message, TokenUsage
from researchagent.core.interfaces.tools import ToolName
from researchagent.services.llm_service import BoundLLM
from researchagent.services.tools.toolbox import ToolBudget
from tests.conftest import FakeLLMProvider


class TestToolCallCeiling:
    def test_a_budget_of_n_permits_exactly_n_reservations(self) -> None:
        budget = ToolBudget(max_tool_calls=3)

        for _ in range(3):
            budget.reserve(ToolName.SEARCH_KNOWLEDGE)

        with pytest.raises(BudgetExhaustedError):
            budget.reserve(ToolName.SEARCH_KNOWLEDGE)
        assert budget.spent == 3, "the refused call is not counted as spent"

    def test_zero_disables_the_ceiling(self) -> None:
        """Absent configuration must not silently forbid all tool use."""
        budget = ToolBudget(max_tool_calls=0)

        for _ in range(50):
            budget.reserve(ToolName.SEARCH_KNOWLEDGE)

        assert budget.remaining is None

    def test_the_refusal_names_the_limit_it_hit(self) -> None:
        budget = ToolBudget(max_tool_calls=1)
        budget.reserve(ToolName.BUILD_BUNDLE)

        with pytest.raises(BudgetExhaustedError) as caught:
            budget.reserve(ToolName.BUILD_BUNDLE)

        assert caught.value.context["max_tool_calls"] == 1
        assert not caught.value.retryable, "retrying spends a budget that is already gone"

    async def test_at_most_n_tool_calls_execute_against_the_real_toolbox(
        self, container: Container
    ) -> None:
        """The regression the ceiling exists for, end to end through ServiceToolbox."""
        limit = 4
        toolbox = container.toolbox.for_agent("retrieval", 0)
        toolbox.budget.max_tool_calls = limit
        toolbox.budget.spent = 0

        executed = 0
        with pytest.raises(BudgetExhaustedError):
            for _ in range(limit + 6):
                await toolbox.search_knowledge("overload mitigation")
                executed += 1

        assert executed == limit
        assert len(toolbox.calls) == limit, "no call past the ceiling reached the ledger"

    async def test_the_ceiling_is_shared_across_agent_views(self, container: Container) -> None:
        """One run, one budget: two agents cannot each spend the whole allowance."""
        container.toolbox.budget.max_tool_calls = 2
        container.toolbox.budget.spent = 0
        retrieval = container.toolbox.for_agent("retrieval", 0)
        verification = container.toolbox.for_agent("verification", 0)

        await retrieval.search_knowledge("overload")
        await verification.get_provenance(("nope",))

        with pytest.raises(BudgetExhaustedError):
            await retrieval.search_knowledge("more overload")


class TestTokenCeiling:
    async def test_a_spent_handle_refuses_to_start_another_call(self, model_catalog) -> None:
        from researchagent.agents.reviewer.schemas import CritiqueDraft

        provider = FakeLLMProvider(
            structured=CritiqueDraft(), usage=TokenUsage(prompt_tokens=60, completion_tokens=40)
        )
        llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)
        llm.with_token_ceiling(100)

        await llm.complete_structured([Message.user("go")], CritiqueDraft)

        with pytest.raises(BudgetExhaustedError) as caught:
            await llm.complete_structured([Message.user("again")], CritiqueDraft)
        assert caught.value.context["spent"] == 100

    async def test_no_ceiling_means_no_refusal(self, model_catalog) -> None:
        from researchagent.agents.reviewer.schemas import CritiqueDraft

        provider = FakeLLMProvider(
            structured=CritiqueDraft(), usage=TokenUsage(prompt_tokens=10_000, completion_tokens=0)
        )
        llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)

        for _ in range(3):
            await llm.complete_structured([Message.user("go")], CritiqueDraft)

        assert llm.usage.usage.total_tokens == 30_000

    async def test_an_unmeasured_provider_never_trips_the_ceiling(self, model_catalog) -> None:
        """Unknown spend must not be treated as spend; the run stops on other limits."""
        from researchagent.agents.reviewer.schemas import CritiqueDraft

        provider = FakeLLMProvider(structured=CritiqueDraft(), usage=None)
        llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)
        llm.with_token_ceiling(10)

        for _ in range(5):
            await llm.complete_structured([Message.user("go")], CritiqueDraft)

        assert llm.usage.unmeasured_calls == 5
        assert llm.usage.usage.total_tokens == 0
