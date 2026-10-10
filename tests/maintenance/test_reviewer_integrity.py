"""The reviewer rejects retracted or withdrawn support, and undisclosed mentions."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchagent.agents.base import AgentContext
from researchagent.agents.reviewer.agent import ReviewerAgent
from researchagent.agents.reviewer.prompt import ReviewerPrompt
from researchagent.agents.reviewer.schemas import CritiqueDraft, ReviewerInput
from researchagent.config.schemas import AgentSpec, ModelCatalog
from researchagent.core.prompts import PromptLibrary
from researchagent.models.integrity import IntegrityStatus
from researchagent.models.reasoning import (
    Citation,
    IntegrityDisclosure,
    ResearchFinding,
    ReviewDecision,
    VerificationResult,
    VerificationVerdict,
)
from researchagent.models.research import QuestionPriority, ResearchQuestion
from researchagent.services.llm_service import BoundLLM
from researchagent.services.validation.findings import RetractedSupportValidator
from tests.conftest import FakeLLMProvider
from tests.maintenance.conftest import observation, paper

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
P1, P2, P3 = paper("p1"), paper("p2"), paper("p3")
QUESTION = ResearchQuestion(
    id="RQ1",
    question="Which techniques mitigate overload in distributed systems?",
    rationale="Overload drives metastable failure and mitigations differ",
    priority=QuestionPriority.HIGH,
)


def finding(
    *supporting: str,
    mentioned: tuple[str, ...] = (),
    disclosures: tuple[IntegrityDisclosure, ...] = (),
) -> ResearchFinding:
    return ResearchFinding(
        question_id="RQ1",
        statement="Circuit breakers are reported as an overload mitigation by two papers.",
        citations=(
            Citation(
                bundle_id="B-1",
                evidence_ids=tuple(f"E-{pid}" for pid in supporting),
                paper_ids=supporting,
            ),
        ),
        contradicting=(
            (Citation(bundle_id="B-1", evidence_ids=("E-x",), paper_ids=mentioned),)
            if mentioned
            else ()
        ),
        integrity_disclosures=disclosures,
        produced_by="reasoning",
    )


def disclosure(paper_id: str) -> IntegrityDisclosure:
    return IntegrityDisclosure(
        paper_id=paper_id,
        status=IntegrityStatus.RETRACTED,
        statement=f"{paper_id} has been retracted and is cited only to note the disagreement.",
    )


RETRACTED = {P1: observation(P1, IntegrityStatus.RETRACTED)}


def codes(result: object) -> set[str]:
    return set(result.issue_codes())  # type: ignore[attr-defined]


@pytest.mark.parametrize("status", [IntegrityStatus.RETRACTED, IntegrityStatus.WITHDRAWN])
def test_retracted_or_withdrawn_support_is_rejected(status: IntegrityStatus) -> None:
    result = RetractedSupportValidator({P1: observation(P1, status)}).validate(finding(P1, P2))

    assert not result.success
    assert "retracted_support" in codes(result)


def test_disclosure_does_not_make_a_retracted_source_support() -> None:
    result = RetractedSupportValidator(RETRACTED).validate(
        finding(P1, P2, disclosures=(disclosure(P1),))
    )
    assert not result.success


def test_undisclosed_mention_is_rejected_disclosed_mention_passes() -> None:
    validator = RetractedSupportValidator(RETRACTED)

    undisclosed = validator.validate(finding(P2, P3, mentioned=(P1,)))
    disclosed = validator.validate(finding(P2, P3, mentioned=(P1,), disclosures=(disclosure(P1),)))

    assert "undisclosed_retracted_mention" in codes(undisclosed) and not undisclosed.success
    assert disclosed.success and "disclosed_retracted_mention" in codes(disclosed)


@pytest.mark.parametrize(
    "status",
    [IntegrityStatus.EXPRESSION_OF_CONCERN, IntegrityStatus.CORRECTED, IntegrityStatus.UNKNOWN],
)
def test_other_statuses_do_not_reject(status: IntegrityStatus) -> None:
    result = RetractedSupportValidator({P1: observation(P1, status)}).validate(finding(P1, P2))
    assert result.success


def test_no_observations_is_not_a_rejection() -> None:
    assert RetractedSupportValidator({}).validate(finding(P1, P2)).success


async def test_reviewer_agent_rejects_undisclosed_retracted_support(
    model_catalog: ModelCatalog,
) -> None:
    supported = finding(P1, P2)
    verification = VerificationResult(
        finding_id=supported.id,
        verdict=VerificationVerdict.VERIFIED,
        supporting=supported.citations,
        verified_by="verification",
    )
    provider = FakeLLMProvider(structured=CritiqueDraft(critique="reads fine"))
    agent = ReviewerAgent(
        BoundLLM("reasoning", model_catalog.spec_for("reasoning"), provider),
        AgentSpec(model="reasoning", prompt_version="v3"),
        PromptLibrary(PROMPTS),
    )
    payload = ReviewerInput(
        goal="study overload failures",
        questions=(QUESTION,),
        findings=(supported,),
        verifications=(verification,),
        resolved_evidence_ids=frozenset(supported.citations[0].evidence_ids),
    )

    clean = await agent.run(payload, AgentContext(run_id="r"))
    flagged = await agent.run(
        payload.model_copy(update={"integrity": tuple(RETRACTED.values())}),
        AgentContext(run_id="r"),
    )

    assert clean.output.result.decision is ReviewDecision.ACCEPT
    assert flagged.output.result.decision is ReviewDecision.REJECT
    assert supported.id in flagged.output.result.rejected_findings
    assert any(issue.code == "retracted_support" for issue in flagged.output.result.issues)
    # The v3 prompt showed the model the registry status of the cited paper.
    rendered = provider.calls[-1][-1].content
    assert f"{P1}: retracted (crossref, notice 10.5555/notice-p1)" in rendered


def test_reviewer_prompt_v3_is_a_new_version() -> None:
    library = PromptLibrary(PROMPTS)
    assert {"v1", "v2", "v3"} <= set(library.available_versions("reviewer"))
    payload = ReviewerInput(goal="study overload failures", findings=(finding(P1, P2),))
    v2 = ReviewerPrompt(library.load("reviewer", "v2")).review_messages(payload, "")
    assert "integrity" not in v2[-1].content.lower()  # released versions are untouched
