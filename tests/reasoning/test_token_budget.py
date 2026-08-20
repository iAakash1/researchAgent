"""Token accounting, and proof that the limits actually stop a run.

The gap this closes: the budget declared `max_tokens_per_agent` and `max_total_tokens`,
and the ledger was never charged, so both were compared against a permanent zero. These
tests assert the accounting is real (taken from the provider, never estimated), that
unknown usage stays unknown, and that each limit terminates execution with its own
reason.
"""

from __future__ import annotations

from researchagent.agents.reasoning.schemas import ClaimDraft, ReasoningDraft
from researchagent.agents.retrieval.schemas import RetrievalPlanDraft, SufficiencyDraft
from researchagent.agents.reviewer.schemas import CritiqueDraft
from researchagent.agents.verification.schemas import VerificationDraft
from researchagent.config.schemas import ReasoningConfig, ResearchBudget
from researchagent.container import Container
from researchagent.core.interfaces.llm import GenerationParams, Message, TokenUsage
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.reasoning import TerminationReason
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.schemas.reasoning import BudgetLedger
from researchagent.schemas.workflow import ResearchState
from researchagent.services.llm_service import BoundLLM, UsageReport
from researchagent.workflows.reasoning_runner import ReasoningRunner
from tests.conftest import FakeLLMProvider


def _plan() -> ResearchPlan:
    return ResearchPlan(
        topic="overload mitigation",
        framing="How distributed systems mitigate overload-driven metastable failure",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="Which techniques mitigate overload in distributed systems?",
                rationale="The mitigations and their trade-offs are not settled",
                priority=QuestionPriority.HIGH,
            )
        ],
        strategy=SearchStrategy(queries=["overload mitigation"]),
    )


def _state() -> ResearchState:
    return ResearchState(goal="study how distributed systems mitigate overload", plan=_plan())


async def _runner(
    container: Container,
    bundle: EvidenceBundle,
    *,
    budget: ResearchBudget,
    usage: TokenUsage | None,
) -> ReasoningRunner:
    """A loop whose every agent reports the same per-call usage."""
    from researchagent.agents.registry import agent_class
    from researchagent.core.prompts import PromptLibrary

    await container.bundle_repository.save(bundle)
    evidence_id = bundle.evidence[0].evidence.id

    scripts: dict[str, list[object]] = {
        "retrieval": [RetrievalPlanDraft(queries=["overload"]), SufficiencyDraft(sufficient=True)]
        * 8,
        "reasoning": [
            ReasoningDraft(
                claims=[
                    ClaimDraft(statement="Circuit breakers shed load.", evidence_ids=[evidence_id])
                ]
            )
        ]
        * 8,
        "verification": [VerificationDraft(verdict="insufficient_evidence")] * 16,
        "reviewer": [CritiqueDraft()] * 8,
    }

    class _Toolbox:
        calls = ()

        async def build_bundle(self, query: str, **kwargs: object) -> EvidenceBundle:
            return bundle

        async def get_provenance(self, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(f"{eid} @ manual:01 p.4" for eid in evidence_ids)

        async def search_graph(self, entity_name: str, **kwargs: object) -> object:
            from researchagent.core.interfaces.tools import GraphSearchResult

            return GraphSearchResult(available=False)

        async def find_contradictions(self, paper_ids: tuple[str, ...] = ()) -> tuple[()]:
            return ()

    def agent_for(name: str, iteration: int, tokens_remaining: int | None = None) -> object:
        spec = container.agent_config.spec_for(name)
        provider = FakeLLMProvider(structured_sequence=list(scripts[name]), usage=usage)
        kwargs: dict[str, object] = {}
        if name in {"retrieval", "verification"}:
            kwargs["toolbox"] = _Toolbox()
        return agent_class(name)(
            BoundLLM(spec.model, container.model_catalog.spec_for(spec.model), provider),
            spec,
            PromptLibrary(container.settings.prompts_dir),
            **kwargs,
        )

    config = ReasoningConfig(
        budget=budget,
        loop=container.reasoning_config.loop,
        review=container.reasoning_config.review,
    )
    return ReasoningRunner(agent_for, container.bundle_repository, config)  # type: ignore[arg-type]


class TestUsageReport:
    """Accounting is taken from the provider, never inferred."""

    def test_an_empty_report_is_complete_and_free(self) -> None:
        report = UsageReport()

        assert report.usage.total_tokens == 0
        assert report.is_complete

    def test_reported_usage_accumulates(self) -> None:
        report = (
            UsageReport()
            .plus(TokenUsage(prompt_tokens=10, completion_tokens=5))
            .plus(TokenUsage(prompt_tokens=3, completion_tokens=2))
        )

        assert report.usage.total_tokens == 20
        assert report.calls == 2
        assert report.is_complete

    def test_unreported_usage_is_counted_as_unknown_not_zero(self) -> None:
        """Requirement: never invent a number when the provider reports none."""
        report = UsageReport().plus(None)

        assert report.usage.total_tokens == 0
        assert report.calls == 1
        assert report.unmeasured_calls == 1
        assert not report.is_complete

    def test_mixed_reporting_keeps_the_measured_part_exact(self) -> None:
        report = UsageReport().plus(TokenUsage(prompt_tokens=7, completion_tokens=3)).plus(None)

        assert report.usage.total_tokens == 10, "measured tokens are not diluted by unknowns"
        assert report.unmeasured_calls == 1


class TestBoundLLMAccounting:
    async def test_a_structured_call_charges_the_handle(self, model_catalog) -> None:
        provider = FakeLLMProvider(
            structured=CritiqueDraft(), usage=TokenUsage(prompt_tokens=40, completion_tokens=10)
        )
        llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)

        await llm.complete_structured([Message.user("go")], CritiqueDraft)

        assert llm.usage.usage.total_tokens == 50
        assert llm.usage.is_complete

    async def test_a_provider_reporting_nothing_marks_the_call_unmeasured(
        self, model_catalog
    ) -> None:
        provider = FakeLLMProvider(structured=CritiqueDraft(), usage=None)
        llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)

        await llm.complete_structured([Message.user("go")], CritiqueDraft)

        assert llm.usage.usage.total_tokens == 0
        assert llm.usage.unmeasured_calls == 1
        assert not llm.usage.is_complete

    async def test_a_plain_completion_also_charges(self, model_catalog) -> None:
        provider = FakeLLMProvider(text="hello")
        llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)

        await llm.complete([Message.user("go")], params=GenerationParams())

        assert llm.usage.usage.total_tokens == 15


class TestLedgerEnforcement:
    def test_the_total_limit_reports_budget_exhausted(self) -> None:
        ledger = BudgetLedger(total_tokens=1024)

        assert (
            ledger.exceeded(ResearchBudget(max_total_tokens=1024, max_tokens_per_agent=512))
            is TerminationReason.BUDGET_EXHAUSTED
        )

    def test_the_per_agent_limit_fires_even_when_the_total_has_room(self) -> None:
        """Otherwise one runaway agent hides inside a generous total."""
        ledger = BudgetLedger(total_tokens=600, tokens_by_agent={"reasoning": 600})
        budget = ResearchBudget(max_tokens_per_agent=512, max_total_tokens=200_000)

        assert ledger.exceeded(budget) is TerminationReason.BUDGET_EXHAUSTED

    def test_spend_below_both_limits_is_allowed(self) -> None:
        ledger = BudgetLedger(total_tokens=100, tokens_by_agent={"reasoning": 100})

        assert ledger.exceeded(ResearchBudget()) is None


class TestLimitsTerminateExecution:
    """The point of the whole exercise: the limits must actually stop the loop."""

    async def test_the_total_token_limit_terminates_the_run(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        runner = await _runner(
            container,
            bundle,
            budget=ResearchBudget(
                max_iterations=10, max_tokens_per_agent=1024, max_total_tokens=1024
            ),
            usage=TokenUsage(prompt_tokens=800, completion_tokens=400),
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.terminated
        assert session.termination_reason is TerminationReason.BUDGET_EXHAUSTED
        assert session.ledger.total_tokens >= 1024
        assert session.iteration < 10, "the run stopped short of the iteration cap"

    async def test_the_per_agent_token_limit_terminates_the_run(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        runner = await _runner(
            container,
            bundle,
            budget=ResearchBudget(
                max_iterations=10, max_tokens_per_agent=1024, max_total_tokens=200_000
            ),
            usage=TokenUsage(prompt_tokens=900, completion_tokens=200),
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.terminated
        assert session.termination_reason is TerminationReason.BUDGET_EXHAUSTED
        assert max(session.ledger.tokens_by_agent.values()) >= 1024

    async def test_spend_is_attributed_per_agent(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        runner = await _runner(
            container,
            bundle,
            budget=ResearchBudget(max_iterations=1, max_total_tokens=200_000),
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10),
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert set(session.ledger.tokens_by_agent) >= {"retrieval", "reasoning", "verification"}
        assert session.ledger.total_tokens == sum(session.ledger.tokens_by_agent.values())

    async def test_a_generous_budget_does_not_terminate_early(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        """The limits must bind only when actually reached."""
        runner = await _runner(
            container,
            bundle,
            budget=ResearchBudget(max_iterations=1, max_total_tokens=200_000),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.termination_reason is not TerminationReason.BUDGET_EXHAUSTED

    async def test_a_provider_reporting_no_usage_does_not_fake_a_charge(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        """Unknown spend is recorded as unknown; the loop stops on iterations instead."""
        runner = await _runner(
            container,
            bundle,
            budget=ResearchBudget(
                max_iterations=1, max_tokens_per_agent=1024, max_total_tokens=1024
            ),
            usage=None,
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.ledger.total_tokens == 0, "no usage reported means no tokens invented"
        assert session.ledger.unmeasured_calls > 0, "the unknown spend is still visible"
        assert session.termination_reason is not None
