"""Message assembly for the Reviewer agent."""

from __future__ import annotations

from researchagent.agents.reviewer.schemas import ReviewerInput
from researchagent.core.interfaces.llm import Message
from researchagent.core.prompts import PromptTemplate

_NONE = "(none)"


class ReviewerPrompt:
    def __init__(self, template: PromptTemplate) -> None:
        self._template = template

    def review_messages(self, payload: ReviewerInput, checks_block: str) -> list[Message]:
        return [
            Message.system(self._template.section("system")),
            Message.user(
                self._template.render(
                    "review",
                    goal=payload.goal.strip(),
                    questions_block=self._questions_block(payload),
                    findings_block=self._findings_block(payload),
                    checks_block=checks_block or _NONE,
                )
            ),
        ]

    def _questions_block(self, payload: ReviewerInput) -> str:
        return "\n".join(f"- {q.id}: {q.question}" for q in payload.questions) or _NONE

    def _findings_block(self, payload: ReviewerInput) -> str:
        lines: list[str] = []
        for finding in payload.findings:
            verdict = next(
                (v.verdict.value for v in payload.verifications if v.finding_id == finding.id),
                "not verified",
            )
            lines.append(
                f"- {finding.id} [{finding.question_id}] verdict={verdict} "
                f"papers={list(finding.paper_ids)}\n"
                f'    "{finding.statement}"\n'
                f"    limitations: {'; '.join(finding.limitations) or _NONE}"
            )
        return "\n".join(lines) or _NONE
