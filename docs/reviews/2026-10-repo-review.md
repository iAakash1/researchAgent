# Repository review before RACM (2026-10-10)

Scope: `main` at `e4f4a2a` (Merge PR #4, `milestone-3/deepseek-model-routing`) of
`iAakash1/researchAgent`, reviewed before implementing Retraction- and Update-Aware
Conclusion Maintenance (RACM) on `feature/conclusion-maintenance`.

## Sync state

- Local `main` was at `5286e56` (v0.9.1), 12 commits behind `origin/main`. Fast-forwarded
  cleanly to `e4f4a2a`. Working tree was clean.
- **CLAUDE.md is not on `main`.** It was deleted in `7916e09` ("refactor: simplify
  architecture and rewrite README"). This work follows the last committed version
  (`git show 5286e56:CLAUDE.md`).
- `PROJECT_CONTEXT.md` is not in this repository. An untracked copy exists only in the
  other local clone (`~/Desktop/Research_Agent/researchAgent`, origin `Xitiz-01`). It was
  read for context only.

## Milestone 3 (DeepSeek routing): present on `main`

Merged in PR #4 (`08f7475`, merge `e4f4a2a`). All the named artefacts exist:

| Artefact | Path |
|---|---|
| DeepSeek provider (JSON mode, repair, diagnostics) | `researchagent/integrations/deepseek/provider.py` |
| Primary → Ollama fallback router | `researchagent/services/model_router.py` |
| Per-run token/cost accounting from `LLM_CALL_COMPLETED` events | `researchagent/services/model_usage.py` |
| Alias catalogue with pricing | `config/models.yaml` |
| Diagnostic script | `scripts/check_deepseek.py` |
| Tests | `tests/integrations/test_deepseek_provider.py`, `tests/services/test_model_router.py`, `tests/services/test_model_usage.py` |

## What is implemented, per pipeline stage

| Stage | Implementation |
|---|---|
| Planning | `agents/planner/`, `prompts/planner/v2.md`, `workflows/research.py:62` |
| Discovery (arXiv, OpenAlex, Crossref, S2, PubMed, manual) | `services/discovery_service.py`, `integrations/*/source.py`, `integrations/sources.py` |
| Ranking + dedup | `services/ranking.py`, `services/deduplication.py` |
| Acquisition | `services/retrieval_service.py`, `workflows/research.py:107` |
| Document intelligence | `services/document/`, `integrations/pymupdf/loader.py`, `services/validation/document.py` |
| Knowledge extraction + grounding | `services/knowledge/` (six extractors), `services/knowledge/grounding.py`, `services/validation/knowledge.py` |
| Evidence index + bundles + contradictions | `services/evidence/`, `models/evidence.py`, `models/bundle.py` |
| Retrieval arms (deterministic/BM25/semantic/hybrid) | `services/retrieval/`, `services/evidence/retrievers.py` |
| Knowledge graph | `services/graph/`, `models/graph.py`, `repositories/graph_repository.py`, `integrations/neo4j/`, `integrations/memory_graph.py` |
| Agentic loop (retrieval → reasoning → verification → review) | `workflows/reasoning.py`, `workflows/agentic.py`, `workflows/reasoning_runner.py`, `agents/{retrieval,reasoning,verification,reviewer}/` |
| Run service + result + API + UI | `services/research_run.py`, `services/result.py`, `api/routes/research.py`, `researchagent/web/` |
| Evaluation | `evaluation/` (retrieval gold set, benchmark), `scripts/run_experiment.py`, `scripts/compare_runs.py` |

## How the relevant subsystems work today

- **Evidence grounding** (`services/knowledge/grounding.py`). Each quote proposed by an
  extractor is located in the parsed document: an exact match first, then the best
  same-length window above `knowledge.pipeline.grounding_similarity_threshold` (0.85), then
  a section-level match. A quote that is never located produces no `Evidence`, so no
  `KnowledgeObject` can be built from it. `SourceLocation.document_id` equals `paper_id`.
- **Contradiction detection** (`services/evidence/contradictions.py`). This is
  deterministic. It detects two things: `VALUE_CONFLICT`, where results with the same
  metric and dataset differ by at least `evidence.contradictions.numeric_tolerance`; and
  `ATTRIBUTE_CONFLICT`, where papers disagree on `is_public`/`is_novel`, cross-paper only.
  It is used by the bundle builder and the graph builder, and both sides are always kept.
- **Verification agent** (`agents/verification/agent.py`). Provenance is resolved first
  through `toolbox.get_provenance`; if nothing resolves, the verdict is `UNVERIFIABLE` and
  no LLM is called. Otherwise the LLM is shown the cited evidence plus the other evidence
  in the round's bundles (`with_bundles`). A `VERIFIED` verdict with no citations is
  downgraded. The confidence it reports is built from observed signals.
- **Reviewer** (`agents/reviewer/agent.py`). Four deterministic validators run first
  (`services/validation/findings.py`): citations, source diversity, verification
  required, and contradiction. The LLM critique can then only remove findings from the
  accepted set.
- **Graph persistence**. `GraphRepository` port (`core/interfaces/graph_repository.py`)
  with memory/JSON/Neo4j adapters. The graph is a *derived index*: each generation is
  written whole and identified by `GraphVersion` (corpus fingerprint). "Current" means the
  newest generation (`api/routes/graph.py:48`, `services/tools/toolbox.py:268`). Knowledge
  nodes are name-merged across papers, except claim-like kinds. The schema has no
  Conclusion, SourceLocation or Contradiction nodes.
- **Memory/checkpoints** (`memory/checkpoints.py`). These are LangGraph `InMemorySaver`
  checkpoints and do not survive the process.
- **Persistence of runs/papers/conclusions**. Papers, documents, knowledge, evidence and
  bundles are stored in JSON file repositories (`repositories/*`). **Runs and conclusions
  are not persisted**: `ResearchRunService` keeps results in an in-memory dict
  (`services/research_run.py:39`). `PostgresSettings` exists (`core/settings.py`), but no
  Postgres repository or driver exists. `docker-compose.yml` declares Postgres, and no
  code uses it.

## Baseline checks (before any change)

| Check | Result |
|---|---|
| `uv run ruff check .` | pass |
| `uv run ruff format --check .` | pass (276 files) |
| `uv run mypy researchagent --strict` | pass (193 files) |
| `uv run pytest` | **710 passed**, 0 failed |
| `node --test tests/web/test_progress.js` | 5/5 pass |
| `node --check researchagent/web/app.js` | pass |

## Defects and rule violations found (not fixed; none blocks RACM)

1. `core/interfaces/graph_repository.py:89-90`: `@abstractmethod` is applied twice to
   `versions`.
2. `workflows/agentic.py:63-64` hardcodes `MAX_QUESTIONS_PER_ROUND`/`MAX_FINDINGS_TO_VERIFY`.
   `config/reasoning.yaml` `loop.*` is loaded but never read (CLAUDE.md: no hardcoded
   thresholds).
3. `agents/reviewer/agent.py:50`: `SourceDiversityValidator()` ignores
   `reasoning.review.minimum_papers_per_finding`, which is dead config.
4. `workflows/agentic.py:146`: the per-round cap compares `len(updated)`, which also
   counts closed and answered questions. Answered questions therefore use up slots for
   open ones.
5. `agents/reviewer/agent.py:116-117`: `evidence_coverage` is computed from the same list
   as `citation_completeness`, so evidence coverage is never measured separately.
6. `integrations/neo4j/repository.py:312-327`: `_to_edge` drops `provenance.locations`,
   so every edge read back from Neo4j is `is_trusted == False`. Lines 298 and 326 build
   `Confidence(score=…)` without signals, which is a score with no observation.
7. Services and workflows import concrete repositories instead of ports:
   `services/audit.py:20-21`, `services/result.py:11`, `workflows/agentic.py:41`,
   `workflows/reasoning_runner.py:20`.
8. Hardcoded API URLs in adapters: `integrations/crossref/source.py:29`,
   `integrations/openalex/source.py:30`, `integrations/arxiv/source.py:28`,
   `integrations/pubmed/source.py:31`, `integrations/semantic_scholar/source.py:29`
   (CLAUDE.md: URLs belong in config).
9. `integrations/crossref/source.py:72-75`: `get_paper` returns `None` on a missing record,
   but `HttpClient` raises `SourceResponseError` on HTTP 404 first. This contradicts the
   port contract ("`None` when absent").
10. `core/prompts.py:75`: the docstring says surplus variables are errors, but
    `Template.substitute` silently ignores them.
11. `agents/verification/agent.py:51`: unused `bundle_loader` parameter (dead code).
12. `services/graph/builder.py:134`: `key.replace("-", ":", 1)` does not invert
    `storage_key_for` for DOIs. Loading still works, but `papers_skipped` reports ids that
    are not the real ids.
13. `core/settings.py:122`: the comment about `GROQ_API_KEY` sits above `llm_provider`.
14. `tests/conftest.py:269-275`: three fixture lines are duplicated verbatim.
15. CLAUDE.md was deleted on `main` (`7916e09`), even though it is the governing document.

## What RACM reuses

| Need | Existing abstraction |
|---|---|
| Typed errors / recording | `ResearchAgentError` + `Recoverability`; `StageFailure`-style recording |
| Confidence | `Confidence`, `ConfidenceSignal`, `Confidence.unknown()` |
| Conclusion | `ResearchFinding` with `FindingStatus.VERIFIED` after review (the accepted set) |
| Citation resolution | `Citation` → `EvidenceBundle`/`EvidenceRepository` → `EvidenceRecord` (`paper_id`, links → `KnowledgeObject`, `SourceLocation`) |
| HTTP | `integrations/http.py::HttpClient` (rate limit, error mapping, polite `mailto` UA) |
| Graph | `KnowledgeGraph`/`GraphNode`/`GraphEdge`/`GraphProvenance` and the `GraphRepository` port with its three adapters |
| Re-verification | `VerificationAgent` (unchanged), fed pruned findings/bundles |
| Contradictions | `ContradictionDetector` (unchanged) |
| Reviewer | `ReviewerAgent` + a new deterministic validator in `services/validation/findings.py` |
| Cost/tokens | `BoundLLM` → `LLM_CALL_COMPLETED` → `ModelUsageTracker.summary(run_id)` |
| Events | `EventBus` with typed payloads |
| Workflow | LangGraph `StateGraph`, same style as `workflows/reasoning.py` |

## Conflicts between the RACM spec and this architecture

1. **"Follow CLAUDE.md"**, but CLAUDE.md is absent from `main`. Resolution: follow the
   last committed version. It is not restored, because that would be out of scope.
2. **"Reuse the existing Postgres repositories"**: none exist. Resolution: RACM adds
   JSON-file repositories behind new ports (conclusions, integrity observations,
   maintenance diffs). This matches every other persisted artefact in the repo.
3. **"Reuse the existing Neo4j/graph services"**: the graph store holds whole,
   corpus-fingerprinted generations, and its "current" generation is simply the newest
   one. Writing a conclusion-dependency generation into the same store would replace the
   knowledge graph as "current" for `/graph` and the toolbox. Its knowledge nodes are also
   name-merged, so the KnowledgeObject → SourceLocation → Paper chain cannot be expressed
   there. Resolution: extend `NodeKind`/`EdgeKind` with the RACM vocabulary. Build the
   dependency graph as a `KnowledgeGraph` projection and persist it through the *same*
   `GraphRepository` port and adapters, but in a dedicated instance configured in
   `config/maintenance.yaml`, which defaults to JSON under `data/maintenance/graphs`.
   The append-only conclusion repository stays the source of truth. The graph remains a
   derived index, as the existing design requires. A Neo4j backend for the projection
   needs its own database; on Neo4j Community this is a known limitation.
4. **"Saved topics / run persistence"** do not exist. Resolution: the minimum needed.
   When a research run completes, its reviewer-accepted findings are snapshotted as
   version-1 conclusions keyed by `run_id` (the "topic").
5. **"A report citing a retracted source"**: the only report artefact is
   `ResearchResult` (findings + `final_summary`). Resolution: the reviewer check works per
   finding. Supporting evidence from a retracted or withdrawn paper is rejected. Any other
   mention of such a paper requires an explicit typed disclosure on the finding.
6. **Integrity lookups during ordinary runs** would add network calls to every run.
   Resolution: the agentic loop is unchanged. Integrity status comes from the maintenance
   workflow, and the reviewer check is a no-op when no observations are supplied.
