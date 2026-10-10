"""Verifier sensitivity: the injection benchmark repeated across grounding floors.

The grounding similarity floor (``knowledge.pipeline.grounding_similarity_threshold``)
decides at *extraction* time which quotes become evidence, so moving it means a corpus
extracted again at that floor and a topic built on it. That preparation is a live,
model-backed run and is supplied by the caller (``prepare``); the sweep itself only
orders the runs and lines up their metrics.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable

from pydantic import BaseModel

from researchagent.evaluation.retraction import BenchmarkReport, Strategy

# floor -> topic id prepared at that floor
PrepareTopic = Callable[[float], Awaitable[str]]
# (topic id, floor) -> benchmark report
RunBenchmark = Callable[[str, float], Awaitable[BenchmarkReport]]


class SensitivityRow(BaseModel):
    model_config = {"frozen": True}

    floor: float
    topic_id: str
    strategy: Strategy
    invalidation_precision: float | None
    invalidation_recall: float | None
    over_invalidation_rate: float | None
    leakage_rate: float | None
    total_tokens: int


async def sweep(
    floors: Iterable[float], prepare: PrepareTopic, run: RunBenchmark
) -> tuple[SensitivityRow, ...]:
    rows: list[SensitivityRow] = []
    for floor in floors:
        topic_id = await prepare(floor)
        report = await run(topic_id, floor)
        for result in report.results:
            metrics = result.metrics
            rows.append(
                SensitivityRow(
                    floor=floor,
                    topic_id=topic_id,
                    strategy=result.strategy,
                    invalidation_precision=metrics.invalidation_precision,
                    invalidation_recall=metrics.invalidation_recall,
                    over_invalidation_rate=metrics.over_invalidation_rate,
                    leakage_rate=metrics.leakage_rate,
                    total_tokens=result.usage.total_tokens,
                )
            )
    return tuple(rows)
