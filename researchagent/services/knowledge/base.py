"""Knowledge extractor contract.

An extractor turns part of a validated :class:`PaperDocument` into knowledge objects of
exactly one kind. It is a *service*, not an agent: it performs a bounded transformation
over data it is handed, and never decides what the system should do next. Agents decide;
extractors extract.

Everything identical across extractors lives here — prompt rendering, the structured
model call, evidence grounding, confidence assembly, per-extractor error isolation — so a
concrete extractor is only its section selection, its draft schema, and how a draft
becomes a knowledge object.

The base enforces the zero-trust rule the whole release rests on: a draft whose quote
cannot be grounded in the document is dropped before a knowledge object is built.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence
from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.interfaces.llm import Message
from researchagent.core.logging import get_logger
from researchagent.core.prompts import PromptLibrary, PromptTemplate
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.document import PaperDocument, Section, SectionKind
from researchagent.models.knowledge import KnowledgeKind, KnowledgeObject, make_knowledge_id
from researchagent.services.knowledge.grounding import EvidenceGrounder
from researchagent.services.llm_service import BoundLLM

logger = get_logger(__name__)

_MAX_SECTION_CHARS = 12_000


class ExtractionDraft(BaseModel):
    """What the model is asked to produce, before anything is believed.

    Every draft carries a ``quote``: the verbatim sentence from the paper that supports
    it. That field is the entire contract between the model and the grounder — without it
    there is nothing to check, and an unverifiable extraction is discarded.

    It is declared **required**, with no default. A field with a default is optional in
    the generated JSON schema, and a model handed an optional field will reliably leave
    it out — which silently turns every extraction into an ungrounded one.
    """

    quote: str = Field(description="Verbatim supporting sentence copied from the paper")


class ExtractionOutcome(BaseModel):
    """What one extractor produced for one document, including what it rejected."""

    model_config = {"frozen": True}

    extractor: str
    kind: KnowledgeKind
    objects: tuple[KnowledgeObject, ...] = ()
    drafts_proposed: int = 0
    drafts_rejected_ungrounded: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def grounding_rate(self) -> float:
        """Share of proposals that survived grounding — a direct hallucination measure."""
        return len(self.objects) / self.drafts_proposed if self.drafts_proposed else 0.0


class KnowledgeExtractor[TDraft: ExtractionDraft, TBatch: BaseModel](ABC):
    """Extracts knowledge objects of one kind from a validated document."""

    name: ClassVar[str]
    kind: ClassVar[KnowledgeKind]
    prompt_name: ClassVar[str]
    # Sections this extractor reads. Feeding the whole paper to every extractor wastes
    # context and invites cross-contamination — limitations do not live in the abstract.
    source_sections: ClassVar[tuple[SectionKind, ...]]
    batch_schema: ClassVar[type[BaseModel]]

    def __init__(
        self, llm: BoundLLM, prompts: PromptLibrary, *, prompt_version: str = "v1"
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._prompt_version = prompt_version
        self._template: PromptTemplate | None = None

    @property
    def prompt(self) -> PromptTemplate:
        if self._template is None:
            self._template = self._prompts.load(self.prompt_name, self._prompt_version)
        return self._template

    @abstractmethod
    def drafts_of(self, batch: TBatch) -> list[TDraft]:
        """Unwrap the model's batch response into individual drafts."""

    @abstractmethod
    def to_object(
        self, draft: TDraft, *, paper_id: str, index: int, evidence: tuple[Evidence, ...]
    ) -> KnowledgeObject | None:
        """Build a knowledge object from a grounded draft.

        Returning None rejects the draft — used when the draft is internally incoherent
        (a result with no value, a dataset with no name) in a way grounding cannot catch.
        """

    async def extract(
        self, document: PaperDocument, grounder: EvidenceGrounder
    ) -> ExtractionOutcome:
        started = time.perf_counter()
        text = self._source_text(document)

        if not text.strip():
            logger.debug(
                "extractor_no_source_text",
                extractor=self.name,
                paper_id=document.paper_id,
                sections=[kind.value for kind in self.source_sections],
            )
            return ExtractionOutcome(extractor=self.name, kind=self.kind)

        try:
            batch = await self._llm.complete_structured(
                self._messages(document, text), self.batch_schema
            )
        except ResearchAgentError as exc:
            # One extractor failing must not cost the paper its other five kinds.
            logger.warning(
                "extractor_failed",
                extractor=self.name,
                paper_id=document.paper_id,
                error_code=exc.code,
                error=exc.message,
            )
            return ExtractionOutcome(
                extractor=self.name,
                kind=self.kind,
                latency_ms=_elapsed_ms(started),
                error=f"{exc.code}: {exc.message}",
            )

        drafts = self.drafts_of(batch)  # type: ignore[arg-type]
        objects, ungrounded = self._ground(drafts, document, grounder)

        outcome = ExtractionOutcome(
            extractor=self.name,
            kind=self.kind,
            objects=tuple(objects),
            drafts_proposed=len(drafts),
            drafts_rejected_ungrounded=ungrounded,
            latency_ms=_elapsed_ms(started),
        )
        logger.info(
            "extraction_complete",
            extractor=self.name,
            paper_id=document.paper_id,
            proposed=outcome.drafts_proposed,
            kept=len(outcome.objects),
            ungrounded=ungrounded,
            grounding_rate=round(outcome.grounding_rate, 3),
        )
        return outcome

    def _ground(
        self, drafts: list[TDraft], document: PaperDocument, grounder: EvidenceGrounder
    ) -> tuple[list[KnowledgeObject], int]:
        """Keep only the drafts the document actually supports."""
        objects: list[KnowledgeObject] = []
        ungrounded = 0

        for index, draft in enumerate(drafts):
            evidence = grounder.evidence_for(
                claim=self._claim_for(draft), quote=draft.quote, produced_by=self.name
            )
            if evidence is None:
                ungrounded += 1
                continue

            candidate = self.to_object(
                draft, paper_id=document.paper_id, index=index, evidence=(evidence,)
            )
            if candidate is None:
                continue
            objects.append(candidate)

        return objects, ungrounded

    def _claim_for(self, draft: TDraft) -> str:
        """What this evidence is being offered in support of."""
        return f"{self.kind.value} extracted by {self.name}"

    def _messages(self, document: PaperDocument, text: str) -> list[Message]:
        return [
            Message.system(self.prompt.section("system")),
            Message.user(
                self.prompt.render(
                    "extract",
                    title=document.metadata.title or document.paper_id,
                    sections=text,
                )
            ),
        ]

    def _source_text(self, document: PaperDocument) -> str:
        """Concatenate this extractor's sections, truncated to a workable context.

        Section titles are kept: the model is told where in the paper it is reading, which
        measurably improves the quotes it returns.
        """
        selected = [
            section
            for section in document.sections
            if section.kind in self.source_sections and not section.is_empty
        ]
        if not selected:
            # Better a degraded read of the body than nothing; the validator will see the
            # weaker confidence that results.
            selected = [s for s in document.body_sections if not s.is_empty]

        return _truncate(selected, _MAX_SECTION_CHARS)

    def confidence_signals(self, outcome: ExtractionOutcome) -> list[ConfidenceSignal]:
        """Per-extractor confidence, grounded in what was actually observed."""
        if not outcome.drafts_proposed:
            return [
                ConfidenceSignal(
                    name=f"{self.kind.value}_presence",
                    value=0.0,
                    observation=f"{self.name} proposed nothing for this document",
                )
            ]
        return [
            ConfidenceSignal(
                name=f"{self.kind.value}_grounding",
                value=outcome.grounding_rate,
                observation=(
                    f"{len(outcome.objects)} of {outcome.drafts_proposed} proposals were "
                    f"located verbatim in the document"
                ),
            )
        ]


def build_object(
    *,
    kind: KnowledgeKind,
    paper_id: str,
    index: int,
    name: str,
    description: str,
    details: object,
    evidence: tuple[Evidence, ...],
    extracted_by: str,
    grounding_note: str,
) -> KnowledgeObject:
    """Assemble a knowledge object with confidence derived from its own grounding."""
    return KnowledgeObject.model_validate(
        {
            "id": make_knowledge_id(paper_id, kind, name, index),
            "kind": kind,
            "paper_id": paper_id,
            "name": name,
            "description": description,
            "details": details,
            "evidence": evidence,
            "confidence": Confidence.from_signals(
                [
                    ConfidenceSignal(
                        name="evidence_grounded",
                        value=1.0,
                        observation=grounding_note,
                    )
                ]
            ),
            "extracted_by": extracted_by,
        }
    )


def _truncate(sections: list[Section], limit: int) -> str:
    parts: list[str] = []
    remaining = limit
    for section in sections:
        block = f"## {section.title}\n{section.text}"
        if len(block) > remaining:
            parts.append(block[:remaining])
            break
        parts.append(block)
        remaining -= len(block)
    return "\n\n".join(parts)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
