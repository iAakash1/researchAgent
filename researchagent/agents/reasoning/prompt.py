"""Message assembly for the Reasoning agent.

The evidence block is the whole contract: the model can only cite what appears here, so
this is also where the citation vocabulary is defined.
"""

from __future__ import annotations

from researchagent.agents.reasoning.schemas import ReasoningInput
from researchagent.core.interfaces.llm import Message
from researchagent.core.prompts import PromptTemplate
from researchagent.models.bundle import EvidenceBundle

_NONE = "(none)"
# Per bundle. Enough context to reason over, small enough that a local model can hold it.
MAX_EVIDENCE_PER_BUNDLE = 12
MAX_QUOTE_CHARS = 1200
MAX_EVIDENCE_CHARS = 12000


class ReasoningPrompt:
    def __init__(self, template: PromptTemplate) -> None:
        self._template = template

    def reason_messages(self, payload: ReasoningInput) -> list[Message]:
        return [
            Message.system(self._template.section("system")),
            Message.user(
                self._template.render(
                    "reason",
                    goal=payload.goal.strip(),
                    question_id=payload.question.id,
                    question=payload.question.question,
                    critique_block=self._critique_block(payload),
                    evidence_block=self._evidence_block(payload.bundles),
                    contradiction_block=self._contradiction_block(payload.bundles),
                )
            ),
        ]

    def _evidence_block(self, bundles: tuple[EvidenceBundle, ...]) -> str:
        lines: list[str] = []
        used = 0
        for bundle in bundles:
            header = f"\n[bundle {bundle.id}]"
            if used + len(header) + bool(lines) > MAX_EVIDENCE_CHARS:
                break
            lines.append(header)
            used += len(header) + (len(lines) > 1)
            for item in bundle.evidence[:MAX_EVIDENCE_PER_BUNDLE]:
                quote = (item.evidence.quote or item.evidence.claim).strip()
                prefix = (
                    f"  evidence_id={item.evidence.id} paper={item.paper_id} "
                    f'object={item.knowledge_object_id or "-"}\n    "'
                )
                available = MAX_EVIDENCE_CHARS - used - len(prefix) - 2
                if available <= 0:
                    return "\n".join(lines)
                line = f'{prefix}{quote[: min(MAX_QUOTE_CHARS, available)]}"'
                lines.append(line)
                used += len(line) + 1
        return "\n".join(lines) or _NONE

    def _contradiction_block(self, bundles: tuple[EvidenceBundle, ...]) -> str:
        conflicts = [item for bundle in bundles for item in bundle.contradictions]
        if not conflicts:
            return ""
        lines = "\n".join(
            f"- {item.description} ({item.left_paper_id} vs {item.right_paper_id})"
            for item in conflicts[:8]
        )
        return f"Known disagreements in this evidence:\n{lines}\n"

    def _critique_block(self, payload: ReasoningInput) -> str:
        if not payload.critique:
            return ""
        items = "\n".join(f"- {item}" for item in payload.critique)
        return (
            "A previous attempt at this question was rejected by verification:\n"
            f"{items}\n\nAddress these specifically. Do not restate the rejected claim.\n"
        )
