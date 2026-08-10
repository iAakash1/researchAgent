"""Reasoning agent: validated evidence -> research findings.

This is the agent with the most freedom and therefore the most constraint. It may
synthesise across papers, state uncertainty, and refuse to answer. It may not produce a
citation that does not resolve.

Citation resolution is the whole mechanism. The model emits evidence ids as plain
strings; every one is looked up in the bundles the agent was actually given. Ids that do
not resolve are dropped. A claim whose citations all drop is *not* silently attached to
some other evidence — it becomes a hypothesis, or it is discarded and counted. That count
is the agent's own fabrication rate, and it is reported rather than hidden.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.reasoning.prompt import ReasoningPrompt
from researchagent.agents.reasoning.schemas import (
    ClaimDraft,
    ReasoningDraft,
    ReasoningInput,
    ReasoningOutput,
)
from researchagent.agents.registry import AGENTS
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.reasoning import Citation, FindingStatus, Hypothesis, ResearchFinding

MIN_STATEMENT_CHARS = 10


class _EvidenceIndex:
    """Which bundle each evidence id came from, and what it points at.

    Built from the bundles the agent was handed, so a lookup miss is definitive: the id
    was not in this agent's evidence.
    """

    def __init__(self, bundles: tuple[EvidenceBundle, ...]) -> None:
        self._bundle_of: dict[str, str] = {}
        self._object_of: dict[str, str | None] = {}
        self._paper_of: dict[str, str] = {}
        self._quote_of: dict[str, str | None] = {}
        self._objects: set[str] = set()

        for bundle in bundles:
            for item in bundle.evidence:
                self._bundle_of[item.evidence.id] = bundle.id
                self._object_of[item.evidence.id] = item.knowledge_object_id
                self._paper_of[item.evidence.id] = item.paper_id
                self._quote_of[item.evidence.id] = item.evidence.quote
            self._objects.update(obj.id for obj in bundle.knowledge_objects)

    def resolve(self, evidence_ids: list[str], object_ids: list[str]) -> tuple[Citation, ...]:
        """Group resolvable ids into one citation per bundle. Unknown ids vanish."""
        grouped: dict[str, list[str]] = {}
        for evidence_id in dict.fromkeys(evidence_ids):
            bundle_id = self._bundle_of.get(evidence_id.strip())
            if bundle_id is not None:
                grouped.setdefault(bundle_id, []).append(evidence_id.strip())

        known_objects = tuple(
            object_id for object_id in dict.fromkeys(object_ids) if object_id in self._objects
        )
        return tuple(
            Citation(
                bundle_id=bundle_id,
                evidence_ids=tuple(ids),
                knowledge_object_ids=tuple(
                    dict.fromkeys(
                        [self._object_of[i] for i in ids if self._object_of.get(i)]
                        + list(known_objects)
                    )
                ),
                paper_ids=tuple(dict.fromkeys(self._paper_of[i] for i in ids)),
                quote=next((self._quote_of[i] for i in ids if self._quote_of.get(i)), None),
            )
            for bundle_id, ids in grouped.items()
        )

    def unresolved(self, evidence_ids: list[str]) -> list[str]:
        return [item for item in evidence_ids if item.strip() not in self._bundle_of]


@AGENTS.register("reasoning")
class ResearchReasoningAgent(BaseAgent[ReasoningInput, ReasoningOutput]):
    name: ClassVar[str] = "reasoning"
    description: ClassVar[str] = "Synthesises research findings from validated evidence bundles"
    input_schema: ClassVar[type[BaseModel]] = ReasoningInput
    output_schema: ClassVar[type[BaseModel]] = ReasoningOutput

    async def execute(self, payload: ReasoningInput, context: AgentContext) -> ReasoningOutput:
        if not payload.bundles:
            # Guarded upstream, but stated here too: reasoning over nothing produces
            # exactly one honest output.
            return ReasoningOutput(
                question_id=payload.question.id,
                insufficient_evidence=True,
                notes="no evidence bundles were supplied",
            )

        prompt = ReasoningPrompt(self.prompt)
        draft = await self.llm.complete_structured(prompt.reason_messages(payload), ReasoningDraft)
        index = _EvidenceIndex(payload.bundles)

        findings: list[ResearchFinding] = []
        hypotheses: list[Hypothesis] = []
        discarded: list[str] = []
        fabricated = 0

        for claim in draft.claims:
            statement = " ".join(claim.statement.split())
            if len(statement) < MIN_STATEMENT_CHARS:
                continue
            fabricated += len(index.unresolved(claim.evidence_ids))
            citations = index.resolve(claim.evidence_ids, claim.knowledge_object_ids)

            if not citations:
                # The claim survives as a hypothesis; it does not survive as a finding,
                # and no unrelated evidence is attached to rescue it.
                discarded.append(statement)
                hypotheses.append(self._to_hypothesis(claim, statement, payload, ()))
                continue

            findings.append(
                ResearchFinding(
                    question_id=payload.question.id,
                    statement=statement,
                    reasoning=claim.reasoning.strip(),
                    citations=citations,
                    contradicting=index.resolve(claim.contradicting_evidence_ids, []),
                    status=FindingStatus.SUPPORTED,
                    confidence=self._confidence(claim, citations),
                    limitations=tuple(item.strip() for item in claim.limitations if item.strip()),
                    produced_by=self.name,
                    iteration=payload.iteration,
                )
            )

        for claim in draft.open_hypotheses:
            statement = " ".join(claim.statement.split())
            if len(statement) >= MIN_STATEMENT_CHARS:
                hypotheses.append(
                    self._to_hypothesis(
                        claim,
                        statement,
                        payload,
                        index.resolve(claim.evidence_ids, claim.knowledge_object_ids),
                    )
                )

        self.logger.info(
            "reasoning_round",
            question=payload.question.id,
            claims=len(draft.claims),
            findings=len(findings),
            hypotheses=len(hypotheses),
            discarded=len(discarded),
            unresolved_citations=fabricated,
        )
        return ReasoningOutput(
            question_id=payload.question.id,
            findings=tuple(findings),
            hypotheses=tuple(hypotheses),
            discarded_claims=tuple(discarded),
            insufficient_evidence=draft.insufficient_evidence or not findings,
            notes=draft.notes.strip(),
        )

    def _to_hypothesis(
        self,
        claim: ClaimDraft,
        statement: str,
        payload: ReasoningInput,
        supporting: tuple[Citation, ...],
    ) -> Hypothesis:
        return Hypothesis(
            question_id=payload.question.id,
            statement=statement,
            rationale=claim.reasoning.strip(),
            supporting=supporting,
            confidence=Confidence.unknown()
            if not supporting
            else self._confidence(claim, supporting),
        )

    def _confidence(self, claim: ClaimDraft, citations: tuple[Citation, ...]) -> Confidence:
        """Grounded in what was counted, never in what the model asserted alone.

        The model's own number is one signal, and deliberately not the only one: a claim
        resting on two papers is stronger than the same claim resting on one, whatever
        the model says about it.
        """
        papers = {paper for citation in citations for paper in citation.paper_ids}
        evidence_count = sum(len(citation.evidence_ids) for citation in citations)
        return Confidence(
            score=round(min(claim.confidence, 0.5 + 0.25 * min(len(papers), 2)), 4),
            signals=(
                ConfidenceSignal(
                    name="model_stated",
                    value=claim.confidence,
                    observation=f"reasoning agent reported {claim.confidence:.2f} for this claim",
                ),
                ConfidenceSignal(
                    name="evidence_count",
                    value=min(evidence_count / 4, 1.0),
                    observation=f"{evidence_count} evidence item(s) resolved to real bundles",
                ),
                ConfidenceSignal(
                    name="paper_support",
                    value=min(len(papers) / 2, 1.0),
                    observation=f"supported by {len(papers)} paper(s): {sorted(papers)}",
                ),
            ),
        )
