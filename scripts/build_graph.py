#!/usr/bin/env python
"""Build the knowledge graph from the validated corpus.

    uv run python scripts/build_graph.py [--twice]

The graph is a derived index: everything it holds comes from the knowledge and evidence
repositories, which stay authoritative. Deleting the graph store costs a rebuild, never a
fact — `--twice` demonstrates that by rebuilding and checking nothing duplicated.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from datetime import UTC, datetime

from researchagent.container import build_container

RESULTS_DIR = pathlib.Path("evaluation/results")


async def main(twice: bool) -> int:
    container = build_container()
    try:
        report = await container.graph_builder.build(run_id="graph_build")
        if not report.succeeded and report.error == "no validated knowledge available":
            print("no validated knowledge in the repository — run the corpus pipeline first")
            return 1

        print(f"version           {report.version}")
        print(f"papers            {report.papers}")
        print(f"nodes             {report.nodes}")
        print(f"edges proposed    {report.edges_proposed}")
        print(f"edges accepted    {report.edges_accepted}")
        print(f"edges rejected    {report.edges_rejected}  {report.rejection_reasons or ''}")
        print(f"contradictions    {report.contradictions}")
        print(f"provenance cover  {report.provenance_coverage:.4f}")
        print(f"build time        {report.duration_ms:.0f}ms")

        print("\nnodes by kind")
        for kind, count in sorted(report.stats.nodes_by_kind.items()):
            print(f"  {kind:<14}{count}")
        print("edges by kind")
        for kind, count in sorted(report.stats.edges_by_kind.items()):
            print(f"  {kind:<14}{count}")

        payload: dict[str, object] = {
            "report": report.model_dump(mode="json"),
            "run_at": datetime.now(UTC).isoformat(),
        }

        if twice:
            second = await container.graph_builder.build(run_id="graph_build_repeat")
            versions = await container.graph_repository.versions()
            identical = (
                second.version == report.version
                and second.nodes == report.nodes
                and second.edges_accepted == report.edges_accepted
            )
            print(
                f"\nidempotency: rebuild produced {second.nodes} nodes / "
                f"{second.edges_accepted} edges, {len(versions)} generation(s) stored "
                f"-> {'IDENTICAL' if identical else 'DIVERGED'}"
            )
            payload["idempotent"] = identical
            payload["generations_after_rebuild"] = len(versions)

        # A worked example of the domain queries, on real data.
        version = (await container.graph_repository.versions())[0]
        queries = container.graph_queries
        shared = await queries.entities_across_papers(version, minimum_papers=2)
        print(f"\nentities appearing in 2+ papers: {len(shared)}")
        for entity in shared[:8]:
            names = ", ".join(node.name[:28] for node in entity.shared_by)
            print(f"  {entity.node.kind.value:<10}{entity.node.name[:34]:<36}{names}")

        contradictions = await queries.contradictions(version)
        print(f"\ncontradictions: {len(contradictions)}")
        for pair in contradictions[:5]:
            print(f"  {pair.left.name[:40]} <-> {pair.right.name[:40]}  papers={list(pair.papers)}")

        payload["shared_entities"] = len(shared)
        payload["contradiction_pairs"] = len(contradictions)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RESULTS_DIR / (f"graph_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
        destination.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {destination}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main("--twice" in sys.argv)))
