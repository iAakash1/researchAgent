"""Conclusion maintenance endpoints. Thin: every decision is in the workflow and service."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from researchagent.api.dependencies import MaintenanceRunnerDep
from researchagent.core.exceptions import DiffNotFoundError
from researchagent.models.maintenance import ConclusionDiff, ConclusionRecord
from researchagent.schemas.maintenance import MaintenanceRequest

router = APIRouter(prefix="/maintenance", tags=["maintenance"])


class TopicSummary(BaseModel):
    topic_id: str
    goal: str
    known_papers: int
    dependency_graph_version: str | None = None
    updated_at: datetime


class ConclusionHistory(BaseModel):
    conclusion_id: str
    versions: tuple[ConclusionRecord, ...]


@router.get("/topics", response_model=list[TopicSummary])
async def list_topics(runner: MaintenanceRunnerDep) -> list[TopicSummary]:
    """Saved topics: completed research runs whose accepted findings are maintained."""
    return [
        TopicSummary(
            topic_id=topic.topic_id,
            goal=topic.goal,
            known_papers=len(topic.known_paper_ids),
            dependency_graph_version=topic.dependency_graph_version,
            updated_at=topic.updated_at,
        )
        for topic in await runner.service.topics()
    ]


@router.post("/topics/{topic_id}/run", response_model=ConclusionDiff)
async def run_maintenance(
    topic_id: str, runner: MaintenanceRunnerDep, request: MaintenanceRequest | None = None
) -> ConclusionDiff:
    """Refresh integrity, propagate, re-verify what was hit, and return the diff.

    ``mode`` selects ``incremental`` (RACM) or ``full_rerun`` for this request only.
    """
    return await runner.run(topic_id, request)


@router.get("/topics/{topic_id}/diff", response_model=ConclusionDiff)
async def latest_diff(topic_id: str, runner: MaintenanceRunnerDep) -> ConclusionDiff:
    diff = await runner.service.latest_diff(topic_id)
    if diff is None:
        raise DiffNotFoundError("This topic has not been maintained yet", topic_id=topic_id)
    return diff


@router.get("/conclusions/{conclusion_id}/history", response_model=ConclusionHistory)
async def conclusion_history(conclusion_id: str, runner: MaintenanceRunnerDep) -> ConclusionHistory:
    """Every version of one conclusion, oldest first. Versions are never rewritten."""
    return ConclusionHistory(
        conclusion_id=conclusion_id, versions=await runner.service.history(conclusion_id)
    )
