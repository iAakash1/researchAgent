# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

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
