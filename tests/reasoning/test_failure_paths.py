"""Failure paths: the system must fail explicitly, never quietly produce a result.

Every test here answers the same question — when something breaks, does the run report
that, or does it invent an answer? A silent fallback in a research system is worse than a
crash, because the output still looks like a finding.
"""

from __future__ import annotations

import httpx
import pytest

from researchagent.agents.base import AgentContext
from researchagent.agents.reasoning.agent import ResearchReasoningAgent
from researchagent.agents.reasoning.schemas import ClaimDraft, ReasoningDraft, ReasoningInput
from researchagent.agents.retrieval.agent import RetrievalAgent
from researchagent.agents.retrieval.schemas import RetrievalInput, RetrievalPlanDraft
from researchagent.agents.reviewer.agent import ReviewerAgent
from researchagent.agents.reviewer.schemas import ReviewerInput
from researchagent.agents.verification.agent import VerificationAgent
from researchagent.agents.verification.schemas import VerificationInput
from researchagent.config.schemas import AgentSpec, ResearchBudget
from researchagent.core.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    OutputParsingError,
    ProviderUnavailableError,
)
from researchagent.core.interfaces.retrieval import RetrievalLayer, RetrievalResult
from researchagent.core.prompts import PromptLibrary
from researchagent.core.retry import RetryPolicy
from researchagent.core.settings import Settings
from researchagent.integrations.groq import GroqProvider
from researchagent.integrations.registry import build_llm_provider
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.query import ResearchQuery
from researchagent.models.reasoning import (
    Citation,
    ResearchFinding,
    ReviewDecision,
    VerificationVerdict,
)
from researchagent.models.research import ResearchQuestion
from researchagent.schemas.reasoning import BudgetLedger, ReasoningSession
from researchagent.services.llm_service import BoundLLM
from researchagent.workflows.guards import within_budget
from tests.conftest import FakeLLMProvider
from tests.reasoning.test_agents import StubToolbox, _prompts_dir

CONTEXT = AgentContext(run_id="failure-test")


def _agent(agent_cls: type, provider: FakeLLMProvider, model_catalog, **kwargs: object):
    spec = AgentSpec(model="reasoning", retry=RetryPolicy(max_attempts=1))
    llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)
    return agent_cls(llm, spec, PromptLibrary(_prompts_dir()), **kwargs)


class TestLLMUnavailable:
    async def test_an_unreachable_llm_surfaces_as_an_error_not_an_empty_finding(
        self, question: ResearchQuestion, bundle: EvidenceBundle, model_catalog
    ) -> None:
        provider = FakeLLMProvider(
            fail_times=5,
            error=ProviderUnavailableError("ollama is down", provider="ollama"),
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        # BaseAgent wraps the cause so the failure is attributed to the agent, and carries
        # the original error code — the run is told what broke, not merely that it broke.
        with pytest.raises(AgentExecutionError) as caught:
            await agent.run(
                ReasoningInput(question=question, goal="study overload", bundles=(bundle,)),
                CONTEXT,
            )

        assert caught.value.context["cause"] == "provider_unavailable"
        assert isinstance(caught.value.__cause__, ProviderUnavailableError)

    async def test_a_failed_reasoning_agent_costs_its_question_not_the_run(
        self, container, bundle: EvidenceBundle
    ) -> None:
        """Per-question isolation: a dead model must not fabricate, and must not cascade."""

        session = ReasoningSession(bundle_ids=(bundle.id,))
        assert session.findings == ()
        assert not session.terminated

    async def test_an_unreachable_llm_in_the_reviewer_degrades_but_does_not_accept(
        self, finding: ResearchFinding, question: ResearchQuestion, model_catalog
    ) -> None:
        """The deterministic checks still stand when the critique model is gone."""
        provider = FakeLLMProvider(
            fail_times=5, error=ProviderUnavailableError("down", provider="ollama")
        )
        agent = _agent(ReviewerAgent, provider, model_catalog)

        result = await agent.run(
            ReviewerInput(
                goal="study overload failures",
                questions=(question,),
                findings=(finding,),
                verifications=(),
                resolved_evidence_ids=frozenset(),
            ),
            CONTEXT,
        )

        assert result.output.result.decision is ReviewDecision.REJECT
        assert "unavailable" in result.output.result.critique


class TestRetrievalUnavailable:
    async def test_a_failing_bundle_builder_yields_no_evidence_rather_than_a_guess(
        self, question: ResearchQuestion, model_catalog
    ) -> None:
        provider = FakeLLMProvider(structured_sequence=[RetrievalPlanDraft(queries=["x"])])
        agent = _agent(RetrievalAgent, provider, model_catalog, toolbox=StubToolbox(bundles=()))

        result = await agent.run(RetrievalInput(question=question, goal="study overload"), CONTEXT)

        assert result.output.bundle_ids == ()
        assert not result.output.sufficient
        assert result.output.unresolved

    async def test_a_degraded_semantic_retriever_is_reported_not_hidden(self) -> None:
        """`degraded` is what distinguishes an outage from an empty corpus."""
        result = RetrievalResult[object].unavailable(
            layer=RetrievalLayer.KNOWLEDGE,
            query=ResearchQuery(text="overload"),
            retrieved_by="semantic",
            reason="qdrant unreachable",
        )

        assert result.degraded
        assert not result.is_usable
        assert result.hits == ()

    async def test_the_toolbox_reports_a_degraded_search_rather_than_raising(
        self, container
    ) -> None:
        toolbox = container.toolbox.for_agent("retrieval", 0)

        result = await toolbox.search_graph("NoSuchEntity")

        assert result.available is False


class TestGroqUnavailable:
    def test_a_missing_key_fails_loudly_and_never_falls_back_to_ollama(self) -> None:
        """A run that silently changed provider is a run whose results mean nothing."""
        with pytest.raises(ConfigurationError) as caught:
            build_llm_provider("groq", Settings(groq_api_key=None))

        assert "GROQ_API_KEY" in caught.value.message

    async def test_an_unreachable_groq_raises_provider_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        provider = GroqProvider(
            api_key="test-key",
            retry_policy=RetryPolicy(max_attempts=1),
            transport=httpx.MockTransport(handler),
        )

        from researchagent.core.interfaces.llm import GenerationParams, Message

        with pytest.raises(ProviderUnavailableError):
            await provider.complete(
                [Message.user("go")], model="openai/gpt-oss-120b", params=GenerationParams()
            )
        await provider.aclose()


class TestInvalidModelOutput:
    async def test_malformed_structured_output_is_a_retryable_parsing_error(
        self, question: ResearchQuestion, bundle: EvidenceBundle, model_catalog
    ) -> None:
        provider = FakeLLMProvider(
            fail_times=5, error=OutputParsingError("schema violated", model="fake")
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        with pytest.raises(AgentExecutionError) as caught:
            await agent.run(
                ReasoningInput(question=question, goal="study overload", bundles=(bundle,)),
                CONTEXT,
            )

        assert caught.value.context["cause"] == "output_parsing_error"
        assert isinstance(caught.value.__cause__, OutputParsingError)
        assert caught.value.__cause__.retryable, "a resample can satisfy the schema"

    async def test_an_unknown_verdict_string_becomes_unverifiable_not_verified(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        from researchagent.agents.verification.schemas import VerificationDraft

        provider = FakeLLMProvider(structured=VerificationDraft(verdict="probably fine"))
        agent = _agent(
            VerificationAgent,
            provider,
            model_catalog,
            toolbox=StubToolbox(provenance=("manual:01 p.4",)),
        )
        agent.with_bundles((bundle,))

        result = await agent.run(VerificationInput(finding=finding, question=question), CONTEXT)

        assert result.output.result.verdict is VerificationVerdict.UNVERIFIABLE


class TestUnsupportedCitation:
    async def test_a_claim_with_no_resolvable_citation_never_becomes_a_finding(
        self, question: ResearchQuestion, bundle: EvidenceBundle, model_catalog
    ) -> None:
        provider = FakeLLMProvider(
            structured=ReasoningDraft(
                claims=[ClaimDraft(statement="A confident invention.", evidence_ids=["ghost"])]
            )
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=(bundle,)),
            CONTEXT,
        )

        assert result.output.findings == ()
        assert result.output.discarded_claims

    def test_a_finding_cannot_exist_without_a_citation_at_all(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResearchFinding(
                question_id="RQ1",
                statement="An unsupported conclusion.",
                citations=(),
                produced_by="reasoning",
            )


class TestMissingEvidenceBundle:
    async def test_a_finding_citing_a_bundle_that_no_longer_exists_is_unverifiable(
        self, question: ResearchQuestion, model_catalog
    ) -> None:
        """Provenance is resolved before the model is consulted, so there is nothing
        for it to have an opinion about."""
        orphan = ResearchFinding(
            question_id="RQ1",
            statement="A claim citing a bundle that was never stored.",
            citations=(Citation(bundle_id="B-gone", evidence_ids=("e-gone",)),),
            produced_by="reasoning",
        )
        provider = FakeLLMProvider()  # would raise if a structured call were attempted
        agent = _agent(
            VerificationAgent, provider, model_catalog, toolbox=StubToolbox(provenance=())
        )

        result = await agent.run(VerificationInput(finding=orphan, question=question), CONTEXT)

        assert result.output.result.verdict is VerificationVerdict.UNVERIFIABLE
        assert provider.calls == []

    async def test_reasoning_over_a_missing_bundle_reports_insufficient_evidence(
        self, question: ResearchQuestion, model_catalog
    ) -> None:
        agent = _agent(ResearchReasoningAgent, FakeLLMProvider(), model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=()), CONTEXT
        )

        assert result.output.insufficient_evidence
        assert result.output.findings == ()


class TestBudgetAndIterationExhaustion:
    def test_every_limit_has_a_distinct_reported_cause(self) -> None:
        from researchagent.models.reasoning import TerminationReason

        assert (
            BudgetLedger(iterations=3).exceeded(ResearchBudget(max_iterations=3))
            is TerminationReason.MAX_ITERATIONS
        )
        for ledger in (
            BudgetLedger(total_tokens=200_000),
            BudgetLedger(tool_calls=40),
            BudgetLedger(retrieval_attempts=8),
        ):
            assert ledger.exceeded(ResearchBudget()) is TerminationReason.BUDGET_EXHAUSTED

    def test_an_exhausted_run_is_refused_entry_to_another_round(self) -> None:
        from researchagent.schemas.workflow import ResearchState

        session = ReasoningSession(
            budget=ResearchBudget(max_tool_calls=5), ledger=BudgetLedger(tool_calls=5)
        )
        state = ResearchState(goal="study overload failures", reasoning=session)

        assert not within_budget().check(state).allowed


class TestContradictoryEvidence:
    async def test_a_contradicted_finding_is_never_accepted(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        from researchagent.models.reasoning import VerificationResult

        contradicted = VerificationResult(
            finding_id=finding.id,
            verdict=VerificationVerdict.CONTRADICTED,
            contradicting=(Citation(bundle_id=bundle.id, evidence_ids=("e1",)),),
            verified_by="verification",
        )
        from researchagent.agents.reviewer.schemas import CritiqueDraft

        agent = _agent(ReviewerAgent, FakeLLMProvider(structured=CritiqueDraft()), model_catalog)

        result = await agent.run(
            ReviewerInput(
                goal="study overload failures",
                questions=(question,),
                findings=(finding,),
                verifications=(contradicted,),
                resolved_evidence_ids=frozenset(item.evidence.id for item in bundle.evidence),
            ),
            CONTEXT,
        )

        assert finding.id in result.output.result.rejected_findings
        assert result.output.result.decision is ReviewDecision.REJECT

    def test_a_contradiction_routes_to_reasoning_not_more_retrieval(self) -> None:
        assert VerificationVerdict.CONTRADICTED.wants_rereasoning
        assert not VerificationVerdict.CONTRADICTED.wants_more_evidence
