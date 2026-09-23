"""Objective run metrics for later DeepSeek-versus-Ollama comparisons."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.reasoning import FindingStatus
from researchagent.schemas.result import ResearchResult


class ModelBenchmarkRecord(BaseModel):
    workload: str
    run_id: str
    runtime_ms: float = Field(ge=0.0)
    providers: tuple[str, ...]
    models: tuple[str, ...]
    fallback_used: bool
    findings_produced: int
    findings_verified: int
    insufficient_evidence_verdicts: int
    contradicted_findings: int
    unresolved_citations: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_hosted_cost_usd: float | None

    @classmethod
    def from_result(
        cls, result: ResearchResult, *, workload: str, runtime_ms: float
    ) -> ModelBenchmarkRecord:
        findings = result.findings
        available_evidence = {item.id for item in result.evidence}
        return cls(
            workload=workload,
            run_id=result.run_id,
            runtime_ms=runtime_ms,
            providers=result.model_usage.providers,
            models=result.model_usage.models,
            fallback_used=result.model_usage.fallback_used,
            findings_produced=len(findings),
            findings_verified=sum(
                item.status is FindingStatus.VERIFIED or item.verification_status == "verified"
                for item in findings
            ),
            insufficient_evidence_verdicts=sum(
                item.status is FindingStatus.INSUFFICIENT_EVIDENCE
                or item.verification_status == "insufficient_evidence"
                for item in findings
            ),
            contradicted_findings=sum(
                item.verification_status == "contradicted" for item in findings
            ),
            unresolved_citations=sum(
                not set(citation.evidence_ids).issubset(available_evidence)
                for item in findings
                for citation in item.citations
            ),
            prompt_tokens=result.model_usage.prompt_tokens,
            completion_tokens=result.model_usage.completion_tokens,
            total_tokens=result.model_usage.total_tokens,
            estimated_hosted_cost_usd=result.model_usage.estimated_cost_usd,
        )
