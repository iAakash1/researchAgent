"""Message assembly for the Planner.

Kept out of ``agent.py`` so prompt wording and reasoning flow can change independently.
"""

from __future__ import annotations

from researchagent.agents.planner.schemas import FramingDraft, PlannerInput, PlannerOptions
from researchagent.core.interfaces.llm import Message
from researchagent.core.prompts import PromptTemplate
from researchagent.models.research import ResearchQuestion

_NONE = "(none)"


class PlannerPrompt:
    """Renders the Planner's two reasoning phases from a versioned prompt file."""

    def __init__(self, template: PromptTemplate) -> None:
        self._template = template

    @property
    def version(self) -> str:
        return self._template.version

    def framing_messages(self, payload: PlannerInput, options: PlannerOptions) -> list[Message]:
        return [
            Message.system(self._template.section("system")),
            Message.user(
                self._template.render(
                    "framing",
                    goal=payload.goal.strip(),
                    constraints_block=self._constraints_block(payload),
                    feedback_block=self._feedback_block(payload),
                    min_questions=options.min_research_questions,
                    max_questions=options.max_research_questions,
                    max_keywords=options.max_keywords_per_question,
                )
            ),
        ]

    def strategy_messages(
        self,
        payload: PlannerInput,
        framing: FramingDraft,
        questions: list[ResearchQuestion],
        options: PlannerOptions,
    ) -> list[Message]:
        return [
            Message.system(self._template.section("system")),
            Message.user(
                self._template.render(
                    "strategy",
                    topic=framing.topic.strip(),
                    framing=framing.framing.strip(),
                    questions_block=self._questions_block(questions),
                    constraints_block=self._constraints_block(payload),
                    max_queries=options.max_queries,
                )
            ),
        ]

    @staticmethod
    def _questions_block(questions: list[ResearchQuestion]) -> str:
        return "\n".join(
            f"- {q.id} [{q.priority.value}] {q.question}"
            + (f" (keywords: {', '.join(q.keywords)})" if q.keywords else "")
            for q in questions
        )

    @staticmethod
    def _constraints_block(payload: PlannerInput) -> str:
        constraints = payload.constraints
        lines: list[str] = []
        if constraints.year_from is not None:
            lines.append(f"- Only consider work published from {constraints.year_from} onwards.")
        if constraints.focus_areas:
            lines.append(f"- Focus specifically on: {', '.join(constraints.focus_areas)}.")
        if constraints.exclusions:
            lines.append(f"- Explicitly exclude: {', '.join(constraints.exclusions)}.")
        if not lines:
            return "Constraints: none."
        return "Constraints:\n" + "\n".join(lines)

    @staticmethod
    def _feedback_block(payload: PlannerInput) -> str:
        if not payload.feedback:
            return ""
        items = "\n".join(f"- {item}" for item in payload.feedback)
        return (
            "A previous version of this plan was rejected in review. Address every point "
            f"below; do not simply reword the earlier plan.\n{items}"
        )
