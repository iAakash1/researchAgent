# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-08

Literature discovery and retrieval. The system now finds real papers across five public
indexes plus the local collection, normalises them into one schema, deduplicates, ranks
and persists them.

### Added
- `core/interfaces/paper_source.py` and `core/interfaces/paper_repository.py`: ports for
  literature providers and paper persistence.
- `models/paper.py`: provider-agnostic `Paper`, `Author`, `PaperIdentifiers`
  (DOI/arXiv/OpenAlex/S2/PubMed, normalised); `models/library.py`: `PaperRecord` with
  monotonic `ProcessingStatus` flags — the plug point for parsing, RAG, extraction and
  the knowledge graph.
- Six adapters behind one port: arXiv (Atom XML), OpenAlex (inverted-index abstracts
  reconstructed), Crossref (JATS stripped), Semantic Scholar, PubMed (two-step
  E-utilities), and a manual source over `storage/papers/raw/manual/`.
- `integrations/http.py`: shared client with per-provider rate limiting, uniform error
  mapping and atomic streaming downloads.
- `services/deduplication.py`: DOI → arXiv id → shared identifier → title similarity,
  with conflicting identifiers vetoing a title match. Duplicates are merged, not dropped.
- `services/ranking.py`: `PaperScorer` port plus a lexical `HeuristicScorer` that reports
  a per-signal breakdown; replaced by embedding reranking in v0.5.
- `services/discovery_service.py` (fan-out, partial-failure tolerant) and
  `services/retrieval_service.py` (explicit, separate PDF download).
- `repositories/paper_repository.py`: atomic JSON sidecars under
  `storage/papers/metadata/`; the source PDFs are never touched.
- `workflows/edges.py` and `ServiceNode`: the graph is now
  `planning →(ok)→ discovery → END`, halting on failure.
- `config/sources.yaml`; API `GET /library/papers|summary|sources`,
  `POST /library/retrieve`; `scripts/index_library.py` (`make index`).

### Changed
- `workflows/nodes.py` factored into a shared `StageNode` base with `AgentNode` and
  `ServiceNode`, so deterministic stages get identical bookkeeping without pretending to
  be LLM agents.
- `PaperRecord.processing` flags now merge monotonically on save. Previously a stored
  record's flags always won, which would have silently discarded progress reported by a
  later stage.

[0.3.0]: https://github.com/iAakash1/researchAgent/releases/tag/v0.3.0

## [0.2.0] — 2026-08-08

Planner agent and LangGraph orchestration. The system now takes a research goal and
returns an executable research plan.

### Added
- `core/prompts.py`: versioned prompt files (`prompts/<agent>/<version>.md`) split into
  named sections, rendered with `string.Template` so prompt bodies can contain JSON.
  Missing variables and unknown sections fail loudly.
- `models/research.py`: `ResearchQuestion`, `SearchStrategy`, `ResearchPlan`.
- `schemas/workflow.py`: `ResearchState` — the typed state threaded through the graph,
  with an accumulating stage history.
- `agents/planner/`: two-phase Planner (framing -> strategy). Ids, ordering,
  deduplication and limits are applied deterministically, not left to the model.
- `workflows/`: `AgentNode` adapter (agent -> LangGraph node), the compiled research
  graph, and `WorkflowRunner` with invoke, SSE streaming and checkpoint retrieval.
- `memory/checkpoints.py`: checkpointer registry (`none` / `memory`).
- `config/workflow.yaml`; `planner` entry with options in `config/agents.yaml`.
- API: `POST /research/plan`, `POST /research/plan/stream` (SSE), `GET /research/runs/{id}`.

### Changed
- `BaseAgent` now takes a `PromptLibrary` and exposes `self.prompt`, resolved from the
  agent's name and the `prompt_version` pinned in config. Every agent is prompt-driven,
  so wiring it once in the base removes it from all ten.
- Workflow stages record failures into state instead of raising, so a failed run stays
  checkpointed and inspectable. Unexpected exceptions are captured the same way and
  logged with a traceback.

[0.2.0]: https://github.com/iAakash1/researchAgent/releases/tag/v0.2.0

## [0.1.0] — 2026-08-08

Foundation release. No agents yet — this is the architecture everything else builds on.

### Added
- Layered package structure with inward-pointing dependencies; vendor SDKs confined
  to `integrations/`.
- `core/`: settings (env-driven, nested), structured logging with correlation context,
  exception hierarchy with stable error codes, generic registry, async event bus,
  retry with exponential backoff and full jitter.
- `core/interfaces/llm.py`: vendor-agnostic LLM port (complete / stream / structured).
- `integrations/ollama/`: Ollama adapter with model caching, failure classification
  (missing model tag, unreachable host, timeout) and an admin client for introspection.
- `config/`: typed YAML loading; `config/models.yaml` model catalogue and
  `config/agents.yaml` per-agent wiring, layered over defaults.
- `services/LLMService`: alias-to-model resolution; agents never see a provider.
- `agents/BaseAgent`: input validation, retry policy, timing, log context, lifecycle
  events, error wrapping. Subclasses implement `execute()` only.
- FastAPI app with liveness/readiness probes; readiness reports provider health and
  whether every configured model is actually pulled.
- Docker Compose stack (Ollama, PostgreSQL, Qdrant, Neo4j, API), multi-stage image,
  GitHub Actions CI (ruff, mypy --strict, pytest, image build), pre-commit hooks.

[0.1.0]: https://github.com/iAakash1/researchAgent/releases/tag/v0.1.0
