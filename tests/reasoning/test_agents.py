"""Agent behaviour under scripted models.

The point of every test here is the same: what happens when the model produces something
it should not be trusted with. A fabricated citation, a self-approving verdict, a claim
with nothing behind it.
"""

from __future__ import annotations

from researchagent.agents.base import AgentContext
from researchagent.agents.reasoning.agent import ResearchReasoningAgent
from researchagent.agents.reasoning.schemas import ClaimDraft, ReasoningDraft, ReasoningInput
from researchagent.agents.retrieval.agent import RetrievalAgent
from researchagent.agents.retrieval.schemas import (
    RetrievalInput,
    RetrievalPlanDraft,
    RetrievalStrategy,
    SufficiencyDraft,
)
from researchagent.agents.reviewer.agent import ReviewerAgent
from researchagent.agents.reviewer.schemas import CritiqueDraft, ReviewerInput
from researchagent.agents.verification.agent import VerificationAgent
from researchagent.agents.verification.schemas import VerificationDraft, VerificationInput
from researchagent.config.schemas import AgentSpec
from researchagent.core.exceptions import RepositoryError
from researchagent.core.prompts import PromptLibrary
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.reasoning import (
    Citation,
    ResearchFinding,
    ReviewDecision,
    VerificationVerdict,
)
from researchagent.models.research import ResearchQuestion
from researchagent.services.llm_service import BoundLLM
from tests.conftest import FakeLLMProvider

CONTEXT = AgentContext(run_id="test-run")


class StubToolbox:
    """A toolbox whose every answer is fixed. Records what it was asked."""

    def __init__(
        self,
        *,
        bundles: tuple[EvidenceBundle, ...] = (),
        provenance: tuple[str, ...] = (),
    ) -> None:
        self._bundles = list(bundles)
        self._provenance = provenance
        self.queries: list[str] = []
        self.calls = ()

    async def build_bundle(self, query: str, **kwargs: object) -> EvidenceBundle:
        """Mirrors the real toolbox: a bundle that cannot be built raises a domain error."""
        self.queries.append(query)
        if not self._bundles:
            raise RepositoryError("no evidence available for this query", query=query)
        return self._bundles[min(len(self.queries) - 1, len(self._bundles) - 1)]

    async def search_graph(self, entity_name: str, **kwargs: object) -> object:
        from researchagent.core.interfaces.tools import GraphSearchResult

        return GraphSearchResult(citations=("manual:01 p.4",))

    async def find_contradictions(self, paper_ids: tuple[str, ...] = ()) -> tuple[object, ...]:
        return ()

    async def get_provenance(self, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
        return self._provenance


def _agent(agent_cls: type, provider: FakeLLMProvider, model_catalog, **kwargs: object):
    spec = AgentSpec(model="reasoning", options=kwargs.pop("options", {}))
    llm = BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider)
    return agent_cls(llm, spec, PromptLibrary(_prompts_dir()), **kwargs)


def _prompts_dir():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "prompts"


class TestRetrievalAgent:
    async def test_it_builds_bundles_from_its_own_queries(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        provider = FakeLLMProvider(
            structured_sequence=[
                RetrievalPlanDraft(
                    strategy=RetrievalStrategy.DIRECT, queries=["overload mitigation"]
                ),
                SufficiencyDraft(sufficient=True),
            ]
        )
        toolbox = StubToolbox(bundles=(bundle,))
        agent = _agent(RetrievalAgent, provider, model_catalog, toolbox=toolbox)

        result = await agent.run(
            RetrievalInput(question=question, goal="study overload failures"), CONTEXT
        )

        assert result.output.bundle_ids == (bundle.id,)
        assert toolbox.queries == ["overload mitigation"]
        assert result.output.sufficient

    async def test_an_empty_query_list_falls_back_to_the_question(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        """A model that returns no queries must not silently retrieve nothing."""
        provider = FakeLLMProvider(
            structured_sequence=[RetrievalPlanDraft(queries=[]), SufficiencyDraft(sufficient=True)]
        )
        toolbox = StubToolbox(bundles=(bundle,))
        agent = _agent(RetrievalAgent, provider, model_catalog, toolbox=toolbox)

        await agent.run(RetrievalInput(question=question, goal="study overload"), CONTEXT)

        assert toolbox.queries == [question.question]

    async def test_no_bundles_is_insufficient_whatever_the_model_says(
        self, question: ResearchQuestion, model_catalog
    ) -> None:
        """Arithmetic beats opinion: nothing retrieved cannot be enough."""
        provider = FakeLLMProvider(
            structured_sequence=[
                RetrievalPlanDraft(queries=["anything"]),
                SufficiencyDraft(sufficient=True),
            ]
        )
        agent = _agent(RetrievalAgent, provider, model_catalog, toolbox=StubToolbox(bundles=()))

        result = await agent.run(RetrievalInput(question=question, goal="study overload"), CONTEXT)

        assert not result.output.sufficient
        assert result.output.bundle_ids == ()
        assert result.output.unresolved

    async def test_queries_are_deduplicated_and_capped(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        provider = FakeLLMProvider(
            structured_sequence=[
                RetrievalPlanDraft(queries=["a", "A", "b", "c", "d", "e"]),
                SufficiencyDraft(sufficient=True),
            ]
        )
        toolbox = StubToolbox(bundles=(bundle,))
        agent = _agent(
            RetrievalAgent,
            provider,
            model_catalog,
            toolbox=toolbox,
            options={"max_queries": 3},
        )

        result = await agent.run(RetrievalInput(question=question, goal="study overload"), CONTEXT)

        assert len(result.output.decision.queries) == 3


class TestReasoningAgentCitations:
    """Citation resolution is the anti-fabrication mechanism."""

    async def test_a_claim_citing_real_evidence_becomes_a_finding(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        real_id = bundle.evidence[0].evidence.id
        provider = FakeLLMProvider(
            structured=ReasoningDraft(
                claims=[
                    ClaimDraft(
                        statement="Circuit breakers are used to shed load under overload.",
                        evidence_ids=[real_id],
                        confidence=0.8,
                    )
                ]
            )
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=(bundle,)), CONTEXT
        )

        assert len(result.output.findings) == 1
        assert result.output.findings[0].citations[0].bundle_id == bundle.id
        assert result.output.discarded_claims == ()

    async def test_a_claim_citing_an_invented_id_never_becomes_a_finding(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        """The fabrication case. It becomes a hypothesis, and is counted."""
        provider = FakeLLMProvider(
            structured=ReasoningDraft(
                claims=[
                    ClaimDraft(
                        statement="Circuit breakers eliminate metastable failure entirely.",
                        evidence_ids=["evidence-that-does-not-exist"],
                        confidence=0.95,
                    )
                ]
            )
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=(bundle,)), CONTEXT
        )

        assert result.output.findings == ()
        assert len(result.output.discarded_claims) == 1
        assert len(result.output.hypotheses) == 1
        assert result.output.hypotheses[0].supporting == ()

    async def test_unrelated_evidence_is_never_attached_to_rescue_a_claim(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        """The explicit instruction: do not silently attach unrelated evidence."""
        provider = FakeLLMProvider(
            structured=ReasoningDraft(
                claims=[
                    ClaimDraft(statement="An unsupported claim about caching.", evidence_ids=[])
                ]
            )
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=(bundle,)), CONTEXT
        )

        assert result.output.findings == ()
        assert all(h.supporting == () for h in result.output.hypotheses)

    async def test_partially_fabricated_citations_keep_only_the_real_ones(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        real_id = bundle.evidence[0].evidence.id
        provider = FakeLLMProvider(
            structured=ReasoningDraft(
                claims=[
                    ClaimDraft(
                        statement="Two systems report circuit breakers under overload.",
                        evidence_ids=[real_id, "made-up-1", "made-up-2"],
                    )
                ]
            )
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=(bundle,)), CONTEXT
        )

        cited = [i for c in result.output.findings[0].citations for i in c.evidence_ids]
        assert cited == [real_id]

    async def test_no_bundles_produces_insufficient_evidence_not_a_guess(
        self, question: ResearchQuestion, model_catalog
    ) -> None:
        agent = _agent(ResearchReasoningAgent, FakeLLMProvider(), model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=()), CONTEXT
        )

        assert result.output.insufficient_evidence
        assert result.output.findings == ()

    async def test_confidence_is_capped_by_paper_support_not_model_assertion(
        self, bundle: EvidenceBundle, question: ResearchQuestion, model_catalog
    ) -> None:
        """A model claiming 0.99 from one paper does not get 0.99."""
        single = bundle.model_copy(
            update={
                "knowledge_objects": bundle.knowledge_objects[:1],
                "evidence": bundle.evidence[:1],
            }
        )
        provider = FakeLLMProvider(
            structured=ReasoningDraft(
                claims=[
                    ClaimDraft(
                        statement="Circuit breakers shed load.",
                        evidence_ids=[single.evidence[0].evidence.id],
                        confidence=0.99,
                    )
                ]
            )
        )
        agent = _agent(ResearchReasoningAgent, provider, model_catalog)

        result = await agent.run(
            ReasoningInput(question=question, goal="study overload", bundles=(single,)), CONTEXT
        )

        assert result.output.findings[0].confidence.score <= 0.75
        assert result.output.findings[0].confidence.signals


class TestVerificationAgent:
    async def test_unresolvable_provenance_is_unverifiable_without_asking_the_model(
        self, finding: ResearchFinding, question: ResearchQuestion, model_catalog
    ) -> None:
        """No provenance, nothing to have an opinion about."""
        provider = FakeLLMProvider()  # would raise if a structured call were made
        agent = _agent(
            VerificationAgent, provider, model_catalog, toolbox=StubToolbox(provenance=())
        )

        result = await agent.run(VerificationInput(finding=finding, question=question), CONTEXT)

        assert result.output.result.verdict is VerificationVerdict.UNVERIFIABLE
        assert provider.calls == []

    async def test_a_verified_verdict_with_no_citations_is_downgraded(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        """The rubber-stamp case: approval without evidence is not approval."""
        provider = FakeLLMProvider(
            structured=VerificationDraft(verdict="verified", supporting_evidence_ids=[])
        )
        agent = _agent(
            VerificationAgent,
            provider,
            model_catalog,
            toolbox=StubToolbox(provenance=("manual:01 p.4",)),
        )
        agent.with_bundles((bundle,))

        result = await agent.run(VerificationInput(finding=finding, question=question), CONTEXT)

        assert result.output.result.verdict is VerificationVerdict.PARTIALLY_SUPPORTED

    async def test_a_verdict_that_is_verified_but_overstated_is_downgraded(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        provider = FakeLLMProvider(
            structured=VerificationDraft(
                verdict="verified",
                supporting_evidence_ids=[bundle.evidence[0].evidence.id],
                overstatements=["'always' is not supported"],
            )
        )
        agent = _agent(
            VerificationAgent,
            provider,
            model_catalog,
            toolbox=StubToolbox(provenance=("manual:01 p.4",)),
        )
        agent.with_bundles((bundle,))

        result = await agent.run(VerificationInput(finding=finding, question=question), CONTEXT)

        assert result.output.result.verdict is VerificationVerdict.PARTIALLY_SUPPORTED

    async def test_an_unknown_verdict_string_becomes_unverifiable(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        provider = FakeLLMProvider(structured=VerificationDraft(verdict="looks_great_to_me"))
        agent = _agent(
            VerificationAgent,
            provider,
            model_catalog,
            toolbox=StubToolbox(provenance=("manual:01 p.4",)),
        )
        agent.with_bundles((bundle,))

        result = await agent.run(VerificationInput(finding=finding, question=question), CONTEXT)

        assert result.output.result.verdict is VerificationVerdict.UNVERIFIABLE

    async def test_a_properly_cited_verified_verdict_stands(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        provider = FakeLLMProvider(
            structured=VerificationDraft(
                verdict="verified",
                supporting_evidence_ids=[bundle.evidence[0].evidence.id],
                reasoning="both papers state this directly",
            )
        )
        agent = _agent(
            VerificationAgent,
            provider,
            model_catalog,
            toolbox=StubToolbox(provenance=("manual:01 p.4", "manual:02 p.6")),
        )
        agent.with_bundles((bundle,))

        result = await agent.run(VerificationInput(finding=finding, question=question), CONTEXT)

        assert result.output.result.verdict is VerificationVerdict.VERIFIED
        assert result.output.result.supporting


class TestReviewerAgent:
    async def test_an_unverified_finding_is_rejected_however_good_it_looks(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        """Deterministic gate: nothing becomes a result without having been checked."""
        provider = FakeLLMProvider(structured=CritiqueDraft(critique="reads fine"))
        agent = _agent(ReviewerAgent, provider, model_catalog)

        result = await agent.run(
            ReviewerInput(
                goal="study overload failures",
                questions=(question,),
                findings=(finding,),
                verifications=(),
                resolved_evidence_ids=frozenset(item.evidence.id for item in bundle.evidence),
            ),
            CONTEXT,
        )

        assert result.output.result.decision is ReviewDecision.REJECT
        assert finding.id in result.output.result.rejected_findings
        assert any(issue.code == "not_verified" for issue in result.output.result.issues)

    async def test_a_finding_with_unresolvable_citations_is_rejected(
        self, question: ResearchQuestion, model_catalog
    ) -> None:
        ghost = ResearchFinding(
            question_id="RQ1",
            statement="A claim resting on evidence that does not exist.",
            citations=(Citation(bundle_id="B-1", evidence_ids=("nope",)),),
            produced_by="reasoning",
        )
        provider = FakeLLMProvider(structured=CritiqueDraft())
        agent = _agent(ReviewerAgent, provider, model_catalog)

        result = await agent.run(
            ReviewerInput(
                goal="study overload failures",
                questions=(question,),
                findings=(ghost,),
                resolved_evidence_ids=frozenset(),
            ),
            CONTEXT,
        )

        assert result.output.result.decision is ReviewDecision.REJECT
        assert result.output.result.unsupported_claim_rate == 1.0

    async def test_the_model_can_reject_but_never_accept(
        self,
        finding: ResearchFinding,
        question: ResearchQuestion,
        bundle: EvidenceBundle,
        model_catalog,
    ) -> None:
        """A gate that can be talked into approving is not a gate."""
        from researchagent.models.reasoning import VerificationResult

        verification = VerificationResult(
            finding_id=finding.id,
            verdict=VerificationVerdict.VERIFIED,
            supporting=(Citation(bundle_id=bundle.id, evidence_ids=("e1",)),),
            verified_by="verification",
        )
        provider = FakeLLMProvider(structured=CritiqueDraft(overclaiming_finding_ids=[finding.id]))
        agent = _agent(ReviewerAgent, provider, model_catalog)

        result = await agent.run(
            ReviewerInput(
                goal="study overload failures",
                questions=(question,),
                findings=(finding,),
                verifications=(verification,),
                resolved_evidence_ids=frozenset(item.evidence.id for item in bundle.evidence),
            ),
            CONTEXT,
        )

        assert finding.id in result.output.result.rejected_findings
        assert finding.id not in result.output.result.accepted_findings

    async def test_no_findings_is_a_rejection_not_an_empty_acceptance(
        self, question: ResearchQuestion, model_catalog
    ) -> None:
        agent = _agent(ReviewerAgent, FakeLLMProvider(), model_catalog)

        result = await agent.run(
            ReviewerInput(goal="study overload failures", questions=(question,), findings=()),
            CONTEXT,
        )

        assert result.output.result.decision is ReviewDecision.REJECT
