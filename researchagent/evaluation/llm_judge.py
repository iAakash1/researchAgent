"""Baseline B3: ask a language model whether each source is retracted.

Exists to be measured against, not used. Offline models miss most retractions, which is
why RACM reads integrity off a registry; this baseline quantifies that on the same
corpus. Its answers become *injected* observations tagged with the model alias, so they
can never be mistaken for registry records.
"""

from __future__ import annotations

from pydantic import BaseModel

from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.interfaces.llm import Message
from researchagent.core.interfaces.repositories import PaperRepository
from researchagent.core.logging import get_logger
from researchagent.core.prompts import PromptTemplate
from researchagent.models.integrity import IntegrityObservation, IntegrityStatus, ObservationBasis
from researchagent.services.integrity.service import identifiers_from_paper_id
from researchagent.services.llm_service import BoundLLM

logger = get_logger(__name__)


class RetractionJudgementDraft(BaseModel):
    status: str = "unknown"
    reasoning: str = ""


class LLMRetractionJudge:
    name = "retraction_judge"

    def __init__(
        self, llm: BoundLLM, prompt: PromptTemplate, papers: PaperRepository | None = None
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._papers = papers

    async def __call__(
        self, paper_ids: tuple[str, ...], run_id: str
    ) -> tuple[IntegrityObservation, ...]:
        """Non-current statuses the model asserts. A failed call asserts nothing."""
        tokens = self._llm.bind_context(agent=self.name, run_id=run_id)
        try:
            asserted: list[IntegrityObservation] = []
            for paper_id in paper_ids:
                status, reasoning = await self._ask(paper_id)
                if status in (IntegrityStatus.CURRENT, IntegrityStatus.UNKNOWN):
                    continue
                asserted.append(
                    IntegrityObservation(
                        paper_id=paper_id,
                        identifier=paper_id,
                        status=status,
                        basis=ObservationBasis.INJECTED,
                        registry=f"llm_judge:{self._llm.alias}",
                        observation=(
                            f"model {self._llm.model} asserted {status.value}: {reasoning[:200]}"
                        ),
                    )
                )
            return tuple(asserted)
        finally:
            self._llm.reset_context(tokens)

    async def _ask(self, paper_id: str) -> tuple[IntegrityStatus, str]:
        doi = identifiers_from_paper_id(paper_id).doi or "unknown"
        title = "unknown"
        if self._papers is not None:
            record = await self._papers.get(paper_id)
            if record is not None:
                title, doi = record.paper.title, record.paper.doi or doi
        messages = [
            Message.system(self._prompt.section("system")),
            Message.user(self._prompt.render("judge", paper_id=paper_id, doi=doi, title=title)),
        ]
        try:
            draft = await self._llm.complete_structured(messages, RetractionJudgementDraft)
        except ResearchAgentError as exc:
            logger.warning("retraction_judge_failed", paper_id=paper_id, error=exc.code)
            return IntegrityStatus.UNKNOWN, ""
        try:
            return IntegrityStatus(draft.status.strip().lower()), draft.reasoning
        except ValueError:
            return IntegrityStatus.UNKNOWN, draft.reasoning
