# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-08-09

Knowledge Intelligence Engine. The system stops reasoning over text and starts reasoning
over `KnowledgeObject` — typed, evidence-backed facts.

### Added
- `models/knowledge.py`: the canonical abstraction. `KnowledgeObject` **cannot be
  constructed without evidence** — enforced by the model, so an unsupported extraction
  never exists even transiently. Typed details per kind via a discriminated union,
  `KnowledgeRelation` with typed domain/range, `PaperKnowledge`.
- `services/knowledge/grounding.py`: the anti-hallucination gate. Every model-supplied
  quote is located in the source document (exact, then windowed fuzzy, then section-level)
  before it becomes evidence. Tolerates ligatures, soft hyphens and column wrapping;
  rejects invention.
- Six specialized extractors — method, dataset, metric, result, limitation, future work —
  each reading its own sections, with its own versioned prompt, behind a shared base that
  owns grounding and error isolation.
- `services/validation/knowledge.py`: evidence, result, completeness, relationship and
  coverage validators. The result validator independently re-checks that a reported number
  appears in its own quote.
- `services/knowledge/relations.py`: relations derived from evidence, never generated —
  field matches from grounded sentences, plus weaker co-location links scored as such.
- `repositories/knowledge_repository.py`, `config/knowledge.yaml`, the
  `knowledge_extraction` workflow stage and its `requires_documents` guard.

### Fixed
- `ExtractionDraft.quote` is required rather than defaulted. As an optional schema field
  the model omitted it entirely, which silently made every extraction ungrounded.
- Knowledge objects naming the same entity are deduplicated across extractors, merging
  their evidence instead of double-counting the fact.

[0.5.0]: https://github.com/iAakash1/researchAgent/releases/tag/v0.5.0

## [0.4.0] — 2026-08-09

Document Intelligence Engine and Zero-Trust Foundation. The system stops thinking in PDFs
and starts thinking in validated `PaperDocument` objects.

### Added — zero-trust foundation
- `core/validation.py`: `ValidationResult`, `ValidationIssue`, `Severity`, and a
  `Confidence` that cannot be constructed without the observations behind it — a
  `ConfidenceSignal` requires its `observation`, and no signals means `unknown()`.
- `core/evidence.py`: `Evidence`, `SourceLocation`, `BoundingBox`. Every asserted fact is
  traceable to document, page, section and paragraph. `EXTRACTED_TEXT` evidence must carry
  its verbatim quote; absence is recorded rather than omitted.
- `core/interfaces/validator.py`: `Validator[T]` port — pure, synchronous, never raises
  for an expected negative.
- `schemas/validated.py`: `Validated[T]`, `ValidatedPaper`, `ValidatedDocument`,
  `DocumentOutcome`. Artefacts cross stage boundaries wrapped in the verdict that admitted
  them.
- `workflows/guards.py`: declarative prerequisites (`requires_plan`, `requires_candidates`,
  `requires_local_pdfs`, `run_not_failed`) checked by `StageNode` before a stage body runs.
- `Recoverability` (retryable / recoverable / fatal) and `remedy` on every domain error.

### Added — document intelligence
- `models/layout.py` (raw positioned blocks) and `models/document.py` (canonical
  `PaperDocument`: sections, paragraphs, figures, tables, references, citations,
  statistics, provenance) — all frozen.
- `integrations/pymupdf/`: the only module importing a PDF library. Produces layout only.
- `services/document/`: section detection (relative typography, canonical-kind mapping),
  reference and citation extraction, figure/table caption detection, self-metadata
  extraction, assembly, and the error-isolating pipeline.
- `services/validation/document.py`: PDF, section, reference, citation and metadata
  validators. The metadata validator cross-checks the PDF against the discovered record
  and reports disagreement instead of preferring one witness.
- `repositories/document_repository.py`, `config/documents.yaml`, and the
  `document_intelligence` workflow stage.

### Changed
- Event payloads are typed models (`AgentPayload`, `DocumentPayload`, `ValidationPayload`,
  …) with `extra="forbid"`; dictionaries no longer cross the event bus.
- `ProcessingStatus` gained the document stages and a `stage_reached` summary.
- `WorkflowRunner` settles the terminal status when the graph stops, so no stage needs to
  know it is last — and a reloaded checkpoint reports the same status the run returned.

[0.4.0]: https://github.com/iAakash1/researchAgent/releases/tag/v0.4.0

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
