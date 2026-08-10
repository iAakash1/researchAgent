"""Message assembly for the Verification agent."""

from __future__ import annotations

from researchagent.agents.verification.schemas import VerificationInput
from researchagent.core.interfaces.llm import Message
from researchagent.core.prompts import PromptTemplate
from researchagent.models.bundle import EvidenceBundle

_NONE = "(none)"


class VerificationPrompt:
    def __init__(self, template: PromptTemplate) -> None:
        self._template = template

    def verify_messages(
        self, payload: VerificationInput, bundles: tuple[EvidenceBundle, ...]
    ) -> list[Message]:
        return [
            Message.system(self._template.section("system")),
            Message.user(
                self._template.render(
                    "verify",
                    question_id=payload.question.id,
                    question=payload.question.question,
                    statement=payload.finding.statement,
                    reasoning=payload.finding.reasoning or _NONE,
                    limitations="; ".join(payload.finding.limitations) or _NONE,
                    evidence_block=self._evidence_block(payload, bundles),
                    contradiction_block=self._contradiction_block(bundles),
                )
            ),
        ]

    def _evidence_block(
        self, payload: VerificationInput, bundles: tuple[EvidenceBundle, ...]
    ) -> str:
        """The cited evidence first, then the rest of the bundle.

        The verifier needs the surrounding evidence too: contradicting material is by
        definition what the finding did *not* cite.
        """
        cited = {
            evidence_id
            for citation in payload.finding.citations
            for evidence_id in citation.evidence_ids
        }
        lines: list[str] = []
        for bundle in bundles:
            for item in bundle.evidence:
                marker = "CITED" if item.evidence.id in cited else "also available"
                quote = (item.evidence.quote or item.evidence.claim).strip()
                lines.append(
                    f"  [{marker}] evidence_id={item.evidence.id} paper={item.paper_id}\n"
                    f'    "{quote[:260]}"'
                )
        return "\n".join(lines) or _NONE

    def _contradiction_block(self, bundles: tuple[EvidenceBundle, ...]) -> str:
        conflicts = [item for bundle in bundles for item in bundle.contradictions]
        if not conflicts:
            return ""
        lines = "\n".join(
            f"- {item.description} ({item.left_paper_id} vs {item.right_paper_id})"
            for item in conflicts[:8]
        )
        return f"Detected disagreements across these papers:\n{lines}\n"
