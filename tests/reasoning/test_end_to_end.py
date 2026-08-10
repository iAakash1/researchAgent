"""The whole loop, driven by scripted models.

Runs the real LangGraph cycle, the real guards, the real citation resolution and the real
reviewer — only the LLM is faked. This is the test that would catch the loop spinning, a
finding reaching the reviewer unverified, or a conclusion with no provenance.
"""

from __future__ import annotations

from researchagent.agents.reasoning.schemas import ClaimDraft, ReasoningDraft
from researchagent.agents.retrieval.schemas import RetrievalPlanDraft, SufficiencyDraft
from researchagent.agents.reviewer.schemas import CritiqueDraft
from researchagent.agents.verification.schemas import VerificationDraft
from researchagent.config.schemas import ReasoningConfig, ResearchBudget
from researchagent.container import Container
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.reasoning import FindingStatus, TerminationReason
from researchagent.models.research import ResearchPlan, ResearchQuestion, SearchStrategy
from researchagent.schemas.workflow import ResearchState
from researchagent.workflows.reasoning_runner import ReasoningRunner


def _plan() -> ResearchPlan:
    return ResearchPlan(
        topic="overload mitigation",
        framing="How distributed systems mitigate overload-driven metastable failure",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="Which techniques mitigate overload in distributed systems?",
                rationale="Mitigations differ and the trade-offs are not settled",
            )
        ],
        strategy=SearchStrategy(queries=["overload mitigation"]),
    )


def _state() -> ResearchState:
    return ResearchState(goal="study how distributed systems mitigate overload", plan=_plan())


async def _runner(
    container: Container,
    bundle: EvidenceBundle,
    scripts: dict[str, list[object]],
    *,
    budget: ResearchBudget | None = None,
) -> ReasoningRunner:
    """Wire a runner whose agents each read their own script."""
    from researchagent.agents.registry import AGENTS
    from researchagent.core.prompts import PromptLibrary
    from researchagent.services.llm_service import BoundLLM
    from tests.conftest import FakeLLMProvider

    await container.bundle_repository.save(bundle)

    class _Toolbox:
        calls = ()

        async def build_bundle(self, query: str, **kwargs: object) -> EvidenceBundle:
            return bundle

        async def search_graph(self, entity_name: str, **kwargs: object) -> object:
            from researchagent.core.interfaces.tools import GraphSearchResult

            return GraphSearchResult(available=False)

        async def find_contradictions(self, paper_ids: tuple[str, ...] = ()) -> tuple[()]:
            return ()

        async def get_provenance(self, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(f"{eid} @ manual:01 p.4" for eid in evidence_ids)

    def agent_for(name: str, iteration: int) -> object:
        spec = container.agent_config.spec_for(name)
        provider = FakeLLMProvider(structured_sequence=list(scripts.get(name, [])))
        kwargs: dict[str, object] = {}
        if name in {"retrieval", "verification"}:
            kwargs["toolbox"] = _Toolbox()
        return AGENTS.get(name)(
            BoundLLM(spec.model, container.model_catalog.spec_for(spec.model), provider),
            spec,
            PromptLibrary(container.settings.prompts_dir),
            **kwargs,
        )

    config = container.reasoning_config
    if budget is not None:
        config = ReasoningConfig(budget=budget, loop=config.loop, review=config.review)
    return ReasoningRunner(agent_for, container.bundle_repository, config)  # type: ignore[arg-type]


class TestHappyPath:
    async def test_a_well_evidenced_claim_becomes_a_verified_finding(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        evidence_id = bundle.evidence[0].evidence.id
        runner = await _runner(
            container,
            bundle,
            {
                "retrieval": [
                    RetrievalPlanDraft(queries=["overload mitigation"]),
                    SufficiencyDraft(sufficient=True),
                ],
                "reasoning": [
                    ReasoningDraft(
                        claims=[
                            ClaimDraft(
                                statement="Circuit breakers are reported as an overload mitigation",
                                evidence_ids=[evidence_id],
                                confidence=0.8,
                            )
                        ]
                    )
                ],
                "verification": [
                    VerificationDraft(
                        verdict="verified",
                        supporting_evidence_ids=[evidence_id],
                        reasoning="both cited papers state this",
                    )
                ],
                "reviewer": [CritiqueDraft(critique="statement matches the evidence")],
            },
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.terminated
        assert session.termination_reason is TerminationReason.ALL_QUESTIONS_ANSWERED
        assert len(session.verified_findings) == 1
        assert session.verified_findings[0].status is FindingStatus.VERIFIED

    async def test_the_audit_trail_reaches_a_source_location(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        """A conclusion nobody can trace to a page is a conclusion nobody should act on."""
        evidence_id = bundle.evidence[0].evidence.id
        runner = await _runner(
            container,
            bundle,
            {
                "retrieval": [
                    RetrievalPlanDraft(queries=["overload"]),
                    SufficiencyDraft(sufficient=True),
                ],
                "reasoning": [
                    ReasoningDraft(
                        claims=[
                            ClaimDraft(
                                statement="Circuit breakers are used to shed load.",
                                evidence_ids=[evidence_id],
                            )
                        ]
                    )
                ],
                "verification": [
                    VerificationDraft(verdict="verified", supporting_evidence_ids=[evidence_id])
                ],
                "reviewer": [CritiqueDraft()],
            },
        )
        final = await runner.run(_state())

        audits = await container.audit_trail.build(final)

        assert audits
        audit = audits[0]
        stages = [step.stage for step in audit.steps]
        assert stages[:4] == ["goal", "plan", "question", "retrieval"]
        assert "reasoning" in stages
        assert "verification" in stages
        assert "review" in stages
        assert audit.citations


class TestRejection:
    async def test_a_fabricated_citation_never_reaches_a_verified_finding(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        """The end-to-end statement of the evidence contract."""
        runner = await _runner(
            container,
            bundle,
            {
                "retrieval": [
                    RetrievalPlanDraft(queries=["overload"]),
                    SufficiencyDraft(sufficient=True),
                ],
                "reasoning": [
                    ReasoningDraft(
                        claims=[
                            ClaimDraft(
                                statement="Circuit breakers eliminate overload failure entirely.",
                                evidence_ids=["fabricated-evidence-id"],
                                confidence=0.99,
                            )
                        ]
                    )
                ],
                "verification": [VerificationDraft(verdict="verified")],
                "reviewer": [CritiqueDraft()],
            },
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.verified_findings == ()
        assert session.terminated
        assert session.termination_reason is not TerminationReason.ALL_QUESTIONS_ANSWERED

    async def test_an_unsupported_claim_produces_a_hypothesis_not_a_finding(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        runner = await _runner(
            container,
            bundle,
            {
                "retrieval": [
                    RetrievalPlanDraft(queries=["overload"]),
                    SufficiencyDraft(sufficient=True),
                ],
                "reasoning": [
                    ReasoningDraft(
                        claims=[
                            ClaimDraft(statement="An entirely unsupported claim.", evidence_ids=[])
                        ]
                    )
                ],
                "verification": [VerificationDraft(verdict="insufficient_evidence")],
                "reviewer": [CritiqueDraft()],
            },
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.findings == ()
        assert session.hypotheses
        assert session.hypotheses[0].supporting == ()


class TestBudgets:
    async def test_the_loop_stops_at_the_iteration_cap(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        """A verifier that never accepts would otherwise cycle forever."""
        evidence_id = bundle.evidence[0].evidence.id
        insufficient = VerificationDraft(verdict="insufficient_evidence")
        runner = await _runner(
            container,
            bundle,
            {
                "retrieval": [
                    RetrievalPlanDraft(queries=["overload"]),
                    SufficiencyDraft(sufficient=False),
                ]
                * 6,
                "reasoning": [
                    ReasoningDraft(
                        claims=[
                            ClaimDraft(
                                statement="A claim that verification keeps refusing.",
                                evidence_ids=[evidence_id],
                            )
                        ]
                    )
                ]
                * 6,
                "verification": [insufficient] * 12,
                "reviewer": [CritiqueDraft()] * 4,
            },
            budget=ResearchBudget(max_iterations=2),
        )

        final = await runner.run(_state())
        session = final.reasoning

        assert session is not None
        assert session.terminated
        assert session.termination_reason is not None
        assert session.iteration <= 2

    async def test_termination_always_records_a_reason(
        self, container: Container, bundle: EvidenceBundle
    ) -> None:
        runner = await _runner(
            container,
            bundle,
            {
                "retrieval": [
                    RetrievalPlanDraft(queries=["overload"]),
                    SufficiencyDraft(sufficient=True),
                ],
                "reasoning": [ReasoningDraft(claims=[])],
                "verification": [VerificationDraft(verdict="insufficient_evidence")],
                "reviewer": [CritiqueDraft()],
            },
        )

        final = await runner.run(_state())

        assert final.reasoning is not None
        assert final.reasoning.terminated
        assert final.reasoning.termination_reason is not None, "never stop silently"
