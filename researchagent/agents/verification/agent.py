"""Verification agent: adversarial check on one finding.

Structurally prevented from rubber-stamping. Three mechanisms, in order of how much they
matter:

1. **Provenance is resolved before the model is asked anything.** If a finding's citations
   do not lead to real source locations, the verdict is UNVERIFIABLE and no model opinion
   is sought — there is nothing to have an opinion about.
2. **The verifier sees evidence the finding did not cite.** Contradicting material is by
   definition what was left out, so a verifier shown only the citations could never find
   any.
3. **A VERIFIED verdict must cite what verified it**, enforced by ``VerificationResult``
   itself. A model that says "verified" with no citations gets its verdict downgraded to
   PARTIALLY_SUPPORTED rather than accepted.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.registry import AGENTS
from researchagent.agents.verification.prompt import VerificationPrompt
from researchagent.agents.verification.schemas import (
    VerificationDraft,
    VerificationInput,
    VerificationOutput,
)
from researchagent.core.interfaces.tools import ResearchToolbox
from researchagent.core.prompts import PromptLibrary
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.bundle import EvidenceBundle
from researchagent.models.reasoning import Citation, VerificationResult, VerificationVerdict
from researchagent.services.llm_service import BoundLLM


@AGENTS.register("verification")
class VerificationAgent(BaseAgent[VerificationInput, VerificationOutput]):
    name: ClassVar[str] = "verification"
    description: ClassVar[str] = "Adversarially checks a research finding against its evidence"
    input_schema: ClassVar[type[BaseModel]] = VerificationInput
    output_schema: ClassVar[type[BaseModel]] = VerificationOutput

    def __init__(
        self,
        llm: BoundLLM,
        spec: object,
        prompts: PromptLibrary,
        *,
        toolbox: ResearchToolbox,
        bundle_loader: object = None,
        **kwargs: object,
    ) -> None:
        super().__init__(llm, spec, prompts, **kwargs)  # type: ignore[arg-type]
        self._toolbox = toolbox
        self._bundles: dict[str, EvidenceBundle] = {}

    def with_bundles(self, bundles: tuple[EvidenceBundle, ...]) -> VerificationAgent:
        """Supply the bundles this round's findings were drawn from."""
        self._bundles = {bundle.id: bundle for bundle in bundles}
        return self

    async def execute(
        self, payload: VerificationInput, context: AgentContext
    ) -> VerificationOutput:
        finding = payload.finding

        # 1. Provenance first. A citation that leads nowhere cannot be argued with.
        evidence_ids = tuple(
            evidence_id for citation in finding.citations for evidence_id in citation.evidence_ids
        )
        provenance = await self._toolbox.get_provenance(evidence_ids)
        if not provenance:
            return VerificationOutput(
                result=VerificationResult(
                    finding_id=finding.id,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    reasoning=(
                        "The finding's citations do not resolve to any source location, "
                        "so there is nothing to verify against."
                    ),
                    unsupported_claims=(finding.statement,),
                    confidence=Confidence(
                        score=0.0,
                        signals=(
                            ConfidenceSignal(
                                name="provenance_resolved",
                                value=0.0,
                                observation=f"0 of {len(evidence_ids)} cited evidence ids resolved",
                            ),
                        ),
                    ),
                    verified_by=self.name,
                    iteration=payload.iteration,
                ),
            )

        bundles = tuple(
            self._bundles[bundle_id]
            for bundle_id in finding.bundle_ids
            if bundle_id in self._bundles
        )
        prompt = VerificationPrompt(self.prompt)
        draft = await self.llm.complete_structured(
            prompt.verify_messages(payload, bundles), VerificationDraft
        )

        index = {
            item.evidence.id: (bundle.id, item.paper_id)
            for bundle in bundles
            for item in bundle.evidence
        }
        supporting = _citations(draft.supporting_evidence_ids, index)
        contradicting = _citations(draft.contradicting_evidence_ids, index)
        verdict = self._resolve_verdict(draft, supporting, contradicting)

        result = VerificationResult(
            finding_id=finding.id,
            verdict=verdict,
            reasoning=draft.reasoning.strip(),
            supporting=supporting,
            contradicting=contradicting,
            unsupported_claims=tuple(c.strip() for c in draft.unsupported_claims if c.strip()),
            overstatements=tuple(o.strip() for o in draft.overstatements if o.strip()),
            confidence=self._confidence(verdict, provenance, evidence_ids, supporting),
            verified_by=self.name,
            iteration=payload.iteration,
        )
        self.logger.info(
            "finding_verified",
            finding=finding.id,
            verdict=verdict.value,
            supporting=len(supporting),
            contradicting=len(contradicting),
            overstatements=len(result.overstatements),
        )
        return VerificationOutput(result=result, provenance=provenance)

    def _resolve_verdict(
        self,
        draft: VerificationDraft,
        supporting: tuple[Citation, ...],
        contradicting: tuple[Citation, ...],
    ) -> VerificationVerdict:
        """Map the model's string to a verdict, then hold it to its own evidence.

        A model asserting VERIFIED without citing anything is asserting a conclusion in
        exactly the way this agent exists to catch, so it is downgraded rather than
        trusted or rejected outright.
        """
        try:
            verdict = VerificationVerdict(draft.verdict.strip().lower())
        except ValueError:
            self.logger.warning("unknown_verdict", verdict=draft.verdict[:40])
            return VerificationVerdict.UNVERIFIABLE

        if verdict is VerificationVerdict.VERIFIED and not supporting:
            return VerificationVerdict.PARTIALLY_SUPPORTED
        if verdict is VerificationVerdict.CONTRADICTED and not contradicting:
            return VerificationVerdict.INSUFFICIENT_EVIDENCE
        if verdict is VerificationVerdict.VERIFIED and draft.overstatements:
            # It cannot be both verified and overstated.
            return VerificationVerdict.PARTIALLY_SUPPORTED
        return verdict

    def _confidence(
        self,
        verdict: VerificationVerdict,
        provenance: tuple[str, ...],
        evidence_ids: tuple[str, ...],
        supporting: tuple[Citation, ...],
    ) -> Confidence:
        resolved = len(provenance) / len(evidence_ids) if evidence_ids else 0.0
        return Confidence(
            score=round(resolved if verdict.accepts else resolved * 0.5, 4),
            signals=(
                ConfidenceSignal(
                    name="provenance_resolved",
                    value=round(resolved, 4),
                    observation=(
                        f"{len(provenance)} of {len(evidence_ids)} cited evidence ids "
                        "resolved to a source location"
                    ),
                ),
                ConfidenceSignal(
                    name="verifier_support",
                    value=min(len(supporting) / 2, 1.0),
                    observation=f"verifier independently cited {len(supporting)} bundle(s)",
                ),
            ),
        )


def _citations(evidence_ids: list[str], index: dict[str, tuple[str, str]]) -> tuple[Citation, ...]:
    """Same resolution discipline as the reasoning agent: unknown ids vanish."""
    grouped: dict[str, list[str]] = {}
    papers: dict[str, set[str]] = {}
    for raw in dict.fromkeys(evidence_ids):
        found = index.get(raw.strip())
        if found is None:
            continue
        bundle_id, paper_id = found
        grouped.setdefault(bundle_id, []).append(raw.strip())
        papers.setdefault(bundle_id, set()).add(paper_id)
    return tuple(
        Citation(
            bundle_id=bundle_id,
            evidence_ids=tuple(ids),
            paper_ids=tuple(sorted(papers[bundle_id])),
        )
        for bundle_id, ids in grouped.items()
    )
