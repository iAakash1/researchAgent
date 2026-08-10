"""Message assembly for the Retrieval agent."""

from __future__ import annotations

from researchagent.agents.retrieval.schemas import RetrievalInput
from researchagent.core.interfaces.llm import Message
from researchagent.core.prompts import PromptTemplate

_NONE = "(none)"


class RetrievalPrompt:
    def __init__(self, template: PromptTemplate) -> None:
        self._template = template

    def plan_messages(self, payload: RetrievalInput, max_queries: int) -> list[Message]:
        return [
            Message.system(self._template.section("system")),
            Message.user(
                self._template.render(
                    "plan",
                    goal=payload.goal.strip(),
                    question_id=payload.question.id,
                    question=payload.question.question,
                    rationale=payload.question.rationale,
                    keywords=", ".join(payload.question.keywords) or _NONE,
                    gaps_block=self._gaps_block(payload),
                    max_queries=max_queries,
                )
            ),
        ]

    def sufficiency_messages(self, payload: RetrievalInput, evidence_block: str) -> list[Message]:
        return [
            Message.system(self._template.section("system")),
            Message.user(
                self._template.render(
                    "sufficiency",
                    question_id=payload.question.id,
                    question=payload.question.question,
                    evidence_block=evidence_block or _NONE,
                )
            ),
        ]

    def _gaps_block(self, payload: RetrievalInput) -> str:
        if not payload.gaps:
            return ""
        gaps = "\n".join(f"- {gap}" for gap in payload.gaps)
        return (
            "A previous attempt was judged insufficient. What was missing:\n"
            f"{gaps}\n\nWiden or redirect the search; do not repeat the same queries."
        )
