#!/usr/bin/env python
"""Retraction-injection benchmark over one maintained topic.

    uv run python scripts/run_retraction_benchmark.py --topic <run_id> --label smoke

Flips K supporting papers (seeded; config/evaluation.yaml), runs each strategy on its own
copy of the topic, and writes ``evaluation/results/racm_<label>.json``. No registry is
contacted: flipped statuses are injected. RACM and B4 re-verify with the configured model,
B1/B2 regenerate with the full reasoning loop, and B3 asks a model about every source —
all of these call the configured LLM and may cost money. B3 runs only with
``--with-llm-judge``.

Sensitivity across grounding floors needs one topic per floor, each built from a corpus
extracted at that floor (knowledge.pipeline.grounding_similarity_threshold):

    --floor-topic 0.75=<run_id> --floor-topic 0.85=<run_id> ...
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib

from researchagent.config.schemas import (
    EvaluationConfig,
    IntegrityConfig,
    MaintenanceConfig,
    ReasoningConfig,
)
from researchagent.container import build_container
from researchagent.evaluation.llm_judge import LLMRetractionJudge
from researchagent.evaluation.retraction import (
    BenchmarkReport,
    RetractionInjectionBenchmark,
    Strategy,
    config_digest,
)
from researchagent.evaluation.sensitivity import sweep
from researchagent.workflows.maintenance_runner import reasoning_generator


def _row(report: BenchmarkReport) -> None:
    identity = report.identity
    print(f"\n{report.label}: topic {identity.topic_id}, K={identity.k}, seed {identity.seed}")
    for item in report.plan.injections:
        print(f"  flipped {item.paper_id} -> {item.status.value}")
    print(
        f"{'strategy':<24}{'inv.P':>8}{'inv.R':>8}{'over':>8}{'leak':>8}{'review':>8}{'tokens':>10}{'cost':>12}"
    )
    for result in report.results:
        m = result.metrics

        def cell(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        cost = (
            "n/a"
            if result.usage.estimated_cost_usd is None
            else f"${result.usage.estimated_cost_usd:.6f}"
        )
        print(
            f"{result.strategy.value:<24}{cell(m.invalidation_precision):>8}"
            f"{cell(m.invalidation_recall):>8}{cell(m.over_invalidation_rate):>8}"
            f"{cell(m.leakage_rate):>8}{cell(m.needs_review_correctness):>8}"
            f"{result.usage.total_tokens:>10}{cost:>12}"
        )


async def main(args: argparse.Namespace) -> int:
    container = build_container()
    try:
        loader = container.config_loader
        evaluation = loader.load("evaluation", EvaluationConfig)
        settings = evaluation.retraction_benchmark
        if args.k is not None or args.seed is not None:
            settings = settings.model_copy(
                update={k: v for k, v in (("k", args.k), ("seed", args.seed)) if v is not None}
            )
        strategies = [Strategy(name) for name in (args.strategies or settings.strategies)]
        judge = None
        if Strategy.B3_LLM_JUDGE in strategies:
            if not args.with_llm_judge:
                strategies.remove(Strategy.B3_LLM_JUDGE)
                print("b3_llm_judge skipped: pass --with-llm-judge to ask a model (paid)")
            else:
                judge = LLMRetractionJudge(
                    container.llm_service.get(settings.llm_judge.alias),
                    container.prompt_library.load(
                        "retraction_judge", settings.llm_judge.prompt_version
                    ),
                    container.paper_repository,
                )
        maintenance = container.maintenance
        benchmark = RetractionInjectionBenchmark(
            maintenance.service,
            maintenance.agent_for,
            reasoning_generator(container.reasoning_runner),
            settings,
            judge=judge,
            config_digest=config_digest(
                loader.load("maintenance", MaintenanceConfig),
                loader.load("integrity", IntegrityConfig),
                loader.load("reasoning", ReasoningConfig),
                container.agent_config,
                settings,
            ),
        )
        results_dir = pathlib.Path(settings.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if args.floor_topic:
            topics: dict[float, str] = {}
            for item in args.floor_topic:
                floor_text, topic_id = item.split("=", 1)
                topics[float(floor_text)] = topic_id

            async def prepare(floor: float) -> str:
                # Preparing a topic per floor is a live re-extraction done beforehand;
                # here it is only looked up.
                return topics[floor]

            async def run(topic_id: str, floor: float) -> BenchmarkReport:
                report = await benchmark.run(
                    topic_id,
                    label=f"{args.label}-g{floor:g}",
                    strategies=strategies,
                    grounding_floor=floor,
                )
                (results_dir / f"racm_{report.label}.json").write_text(
                    report.model_dump_json(indent=2)
                )
                _row(report)
                return report

            floors = sorted(topics)
            for row in await sweep(floors, prepare, run):
                print(row.model_dump_json())
            return 0

        if not args.topic:
            print("--topic is required (the run_id of a completed research run)")
            return 2
        report = await benchmark.run(args.topic, label=args.label, strategies=strategies)
        path = results_dir / f"racm_{args.label}.json"
        path.write_text(report.model_dump_json(indent=2))
        _row(report)
        print(f"\nwritten {path}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--topic")
    parser.add_argument("--label", required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--strategies", nargs="*", choices=[s.value for s in Strategy])
    parser.add_argument("--with-llm-judge", action="store_true")
    parser.add_argument("--floor-topic", action="append", default=[], metavar="FLOOR=TOPIC")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
