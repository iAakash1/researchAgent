"""Retrieval agent: research question -> validated EvidenceBundles.

The agent decides *what to look for*; the toolbox performs the lookup. It never touches
Qdrant, BM25 or Neo4j, and it cannot construct a query in any query language — it names a
strategy and supplies search terms, and everything else is deterministic.

Two model calls, not one: choosing a strategy and judging sufficiency are different
questions, and asking them together produces a model that rationalises whatever it
already retrieved.

The one thing it may never do is hand back context that did not come from a validated
bundle. Bundles that fail validation are counted and discarded, never used.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ValidationError

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.registry import AGENTS
from researchagent.agents.retrieval.prompt import RetrievalPrompt
from researchagent.agents.retrieval.schemas import (
    RetrievalDecision,
    RetrievalInput,
    RetrievalOutput,
    RetrievalPlanDraft,
    RetrievalStrategy,
    SufficiencyDraft,
)
from researchagent.core.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    ResearchAgentError,
)
from researchagent.core.interfaces.tools import ResearchToolbox
from researchagent.core.prompts import PromptLibrary
from researchagent.models.bundle import EvidenceBundle
from researchagent.services.llm_service import BoundLLM

MAX_QUERIES = 4


class RetrievalOptions(BaseModel):
    max_queries: int = 4
    max_entities: int = 3
    # Below this share of queries returning anything, the round is not sufficient
    # regardless of what the model says about it.
    min_coverage: float = 0.34


@AGENTS.register("retrieval")
class RetrievalAgent(BaseAgent[RetrievalInput, RetrievalOutput]):
    name: ClassVar[str] = "retrieval"
    description: ClassVar[str] = (
        "Decides how to search the corpus and returns validated evidence bundles"
    )
    input_schema: ClassVar[type[BaseModel]] = RetrievalInput
    output_schema: ClassVar[type[BaseModel]] = RetrievalOutput

    def __init__(
        self,
        llm: BoundLLM,
        spec: object,
        prompts: PromptLibrary,
        *,
        toolbox: ResearchToolbox,
        **kwargs: object,
    ) -> None:
        super().__init__(llm, spec, prompts, **kwargs)  # type: ignore[arg-type]
        self._toolbox = toolbox

    async def execute(self, payload: RetrievalInput, context: AgentContext) -> RetrievalOutput:
        options = self._options()
        prompt = RetrievalPrompt(self.prompt)

        draft = await self.llm.complete_structured(
            prompt.plan_messages(payload, options.max_queries), RetrievalPlanDraft
        )
        decision = self._to_decision(draft, payload, options)

        bundles, rejected = await self._gather(decision, options)
        graph_citations = await self._expand_graph(decision, options)
        contradictions = await self._contradictions(decision)

        objects = sum(len(bundle.knowledge_objects) for bundle in bundles)
        evidence = sum(len(bundle.evidence) for bundle in bundles)
        coverage = self._coverage(decision, bundles)

        sufficient, unresolved = await self._judge(payload, bundles, coverage, options, prompt)

        self.logger.info(
            "retrieval_round",
            question=payload.question.id,
            strategy=decision.strategy.value,
            queries=len(decision.queries),
            bundles=len(bundles),
            rejected=len(rejected),
            objects=objects,
            coverage=round(coverage, 3),
            sufficient=sufficient,
        )
        return RetrievalOutput(
            question_id=payload.question.id,
            decision=decision,
            bundle_ids=tuple(bundle.id for bundle in bundles),
            rejected_bundle_ids=tuple(rejected),
            objects_found=objects,
            evidence_found=evidence,
            contradictions_found=contradictions,
            graph_citations=graph_citations,
            coverage=coverage,
            sufficient=sufficient,
            unresolved=unresolved,
        )

    def _options(self) -> RetrievalOptions:
        try:
            return RetrievalOptions.model_validate(self.spec.options)
        except ValidationError as exc:
            raise ConfigurationError(
                "Invalid retrieval options in config/agents.yaml",
                agent=self.name,
                errors=exc.errors(include_url=False),
            ) from exc

    def _to_decision(
        self, draft: RetrievalPlanDraft, payload: RetrievalInput, options: RetrievalOptions
    ) -> RetrievalDecision:
        """Deduplicate, cap, and fall back to the question itself if the model gave nothing."""
        seen: set[str] = set()
        queries: list[str] = []
        for candidate in draft.queries:
            text = " ".join(candidate.split())
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                queries.append(text)
        if not queries:
            queries = [payload.question.question]

        return RetrievalDecision(
            strategy=draft.strategy,
            queries=tuple(queries[: options.max_queries]),
            entities=tuple(dict.fromkeys(draft.entities))[: options.max_entities],
            rationale=draft.rationale.strip(),
        )

    async def _gather(
        self, decision: RetrievalDecision, options: RetrievalOptions
    ) -> tuple[list[EvidenceBundle], list[str]]:
        """Build one bundle per query. Untrusted bundles are discarded, not downgraded."""
        bundles: list[EvidenceBundle] = []
        rejected: list[str] = []
        for query in decision.queries:
            try:
                bundle = await self._toolbox.build_bundle(query)
            except BudgetExhaustedError:
                # Deliberately not swallowed: a budget ceiling that degrades into
                # "skip this query" is a stopping condition wearing a ceiling's name.
                self.logger.info("retrieval_stopped_on_budget", query=query[:60])
                raise
            except ResearchAgentError as exc:
                self.logger.warning("bundle_build_failed", query=query[:60], error=exc.code)
                continue
            if not bundle.is_trusted:
                rejected.append(bundle.id)
                continue
            if bundle.is_empty:
                continue
            bundles.append(bundle)
        return bundles, rejected

    async def _expand_graph(
        self, decision: RetrievalDecision, options: RetrievalOptions
    ) -> tuple[str, ...]:
        """Graph traversal is additive context, never a substitute for evidence."""
        if decision.strategy is not RetrievalStrategy.GRAPH_EXPANSION or not decision.entities:
            return ()
        citations: list[str] = []
        for entity in decision.entities:
            result = await self._toolbox.search_graph(entity)
            citations.extend(result.citations)
        return tuple(dict.fromkeys(citations))

    async def _contradictions(self, decision: RetrievalDecision) -> int:
        if decision.strategy is not RetrievalStrategy.CONTRADICTION_SEEKING:
            return 0
        return len(await self._toolbox.find_contradictions())

    def _coverage(self, decision: RetrievalDecision, bundles: list[EvidenceBundle]) -> float:
        """Share of the agent's own queries that returned usable evidence."""
        if not decision.queries:
            return 0.0
        return round(len(bundles) / len(decision.queries), 4)

    async def _judge(
        self,
        payload: RetrievalInput,
        bundles: list[EvidenceBundle],
        coverage: float,
        options: RetrievalOptions,
        prompt: RetrievalPrompt,
    ) -> tuple[bool, tuple[str, ...]]:
        """Deterministic floor first, model opinion second.

        Nothing retrieved is insufficient as a matter of arithmetic, and no model should
        be given the chance to disagree with that.
        """
        if not bundles:
            return False, ("no evidence retrieved for this question",)

        block = "\n".join(
            f"- [{obj.kind.value}] {obj.name}: {obj.description[:120]}"
            for bundle in bundles
            for obj in bundle.knowledge_objects[:8]
        )
        try:
            verdict = await self.llm.complete_structured(
                prompt.sufficiency_messages(payload, block), SufficiencyDraft
            )
        except ResearchAgentError as exc:
            self.logger.warning("sufficiency_judgement_failed", error=exc.code)
            return coverage >= options.min_coverage, ()

        sufficient = verdict.sufficient and coverage >= options.min_coverage
        return sufficient, tuple(gap.strip() for gap in verdict.missing if gap.strip())
