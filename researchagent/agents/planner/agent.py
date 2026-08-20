"""Planner agent: research goal -> executable research plan.

Reasoning runs in two phases rather than one call:

    1. framing  — restate the topic, define scope, generate research questions
    2. strategy — derive search queries and filters *from those questions*

One call producing the whole plan makes a local 8B model choose between depth and
completeness, and it usually sacrifices both. Splitting also means a failure in phase 2
retries phase 2 only, and gives the strategy step the questions as explicit context.

Everything that can be decided deterministically — ids, ordering, deduplication, limits
— is decided here, not by the model.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ValidationError

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.planner.prompt import PlannerPrompt
from researchagent.agents.planner.schemas import (
    FramingDraft,
    PlannerInput,
    PlannerOptions,
    PlannerOutput,
    QuestionDraft,
    StrategyDraft,
)
from researchagent.core.exceptions import ConfigurationError, OutputParsingError
from researchagent.models.research import (
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)


class PlannerAgent(BaseAgent[PlannerInput, PlannerOutput]):
    name: ClassVar[str] = "planner"
    description: ClassVar[str] = (
        "Turns a research goal into research questions and a search strategy"
    )
    input_schema: ClassVar[type[BaseModel]] = PlannerInput
    output_schema: ClassVar[type[BaseModel]] = PlannerOutput

    async def execute(self, payload: PlannerInput, context: AgentContext) -> PlannerOutput:
        options = self._options()
        prompt = PlannerPrompt(self.prompt)

        framing = await self.llm.complete_structured(
            prompt.framing_messages(payload, options), FramingDraft
        )
        questions = self._to_questions(framing.questions, payload, options)
        self.logger.debug("planner_framing", topic=framing.topic, question_count=len(questions))

        strategy = await self.llm.complete_structured(
            prompt.strategy_messages(payload, framing, questions, options), StrategyDraft
        )

        return PlannerOutput(plan=self._to_plan(framing, questions, strategy, payload, options))

    def _options(self) -> PlannerOptions:
        try:
            return PlannerOptions.model_validate(self.spec.options)
        except ValidationError as exc:
            raise ConfigurationError(
                "Invalid planner options in config/agents.yaml",
                agent=self.name,
                errors=exc.errors(include_url=False),
            ) from exc

    def _to_questions(
        self,
        drafts: list[QuestionDraft],
        payload: PlannerInput,
        options: PlannerOptions,
    ) -> list[ResearchQuestion]:
        """Deduplicate, order by priority, cap, and assign stable RQ ids."""
        limit = payload.constraints.max_research_questions or options.max_research_questions

        seen: set[str] = set()
        unique: list[QuestionDraft] = []
        for draft in drafts:
            text = draft.question.strip()
            fingerprint = " ".join(text.lower().split())
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(draft)

        if not unique:
            # Retryable: the model produced no usable questions, so re-prompting is the
            # correct response rather than emitting an empty plan.
            raise OutputParsingError(
                "Planner produced no usable research questions",
                agent=self.name,
                raw_count=len(drafts),
            )

        # Stable sort keeps the model's ordering within a priority band.
        ordered = sorted(unique, key=lambda draft: draft.priority.rank)[:limit]

        return [
            ResearchQuestion(
                id=f"RQ{index}",
                question=draft.question.strip(),
                rationale=draft.rationale.strip(),
                priority=draft.priority,
                keywords=_clean_list(draft.keywords, options.max_keywords_per_question),
            )
            for index, draft in enumerate(ordered, start=1)
        ]

    def _to_plan(
        self,
        framing: FramingDraft,
        questions: list[ResearchQuestion],
        strategy: StrategyDraft,
        payload: PlannerInput,
        options: PlannerOptions,
    ) -> ResearchPlan:
        queries = _clean_list(strategy.queries, options.max_queries)
        if not queries:
            # Falling back to question keywords keeps a usable plan when only the query
            # list came back empty; the rest of the strategy is still sound.
            queries = _clean_list(
                [q.question for q in questions] + [kw for q in questions for kw in q.keywords],
                options.max_queries,
            )

        try:
            return ResearchPlan(
                topic=framing.topic.strip() or payload.goal.strip(),
                framing=framing.framing.strip(),
                research_questions=questions,
                strategy=SearchStrategy(
                    queries=queries,
                    inclusion_criteria=_clean_list(strategy.inclusion_criteria, 10),
                    exclusion_criteria=_clean_list(strategy.exclusion_criteria, 10),
                    year_from=payload.constraints.year_from,
                ),
                expected_methods=_clean_list(strategy.expected_methods, 15),
                expected_datasets=_clean_list(strategy.expected_datasets, 15),
                evaluation_metrics=_clean_list(strategy.evaluation_metrics, 15),
            )
        except ValidationError as exc:
            raise OutputParsingError(
                "Planner output did not form a valid research plan",
                agent=self.name,
                errors=exc.errors(include_url=False),
            ) from exc


def _clean_list(values: list[str], limit: int) -> list[str]:
    """Strip, drop blanks, deduplicate case-insensitively, preserve order, cap length."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        text = " ".join(value.split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) == limit:
            break
    return cleaned
