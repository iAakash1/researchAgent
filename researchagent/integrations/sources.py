"""Paper source registry and construction.

``config/sources.yaml`` names providers as strings; this module turns those strings into
adapters, each with its own rate-limited HTTP client. Adding a provider is a new adapter
plus one entry here plus one line of YAML — no service or agent changes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from researchagent.config.schemas import SourcesConfig, SourceSettings
from researchagent.core.interfaces.paper_source import PaperSource
from researchagent.core.logging import get_logger
from researchagent.core.registry import Registry
from researchagent.integrations.arxiv import ArxivSource
from researchagent.integrations.crossref import CrossrefSource
from researchagent.integrations.http import HttpClient
from researchagent.integrations.manual import ManualPaperSource
from researchagent.integrations.openalex import OpenAlexSource
from researchagent.integrations.pubmed import PubMedSource
from researchagent.integrations.semantic_scholar import SemanticScholarSource
from researchagent.models.paper import SourceName

logger = get_logger(__name__)

# (settings, contact_email, library_dir) -> adapter
SourceFactory = Callable[[SourceSettings, str | None, Path], PaperSource]

PAPER_SOURCES: Registry[SourceFactory] = Registry("paper_source")


def _http(source: SourceName, settings: SourceSettings, contact_email: str | None) -> HttpClient:
    return HttpClient(
        source.value,
        timeout_seconds=settings.timeout_seconds,
        requests_per_second=settings.requests_per_second,
        contact_email=contact_email,
    )


PAPER_SOURCES.add(
    SourceName.ARXIV.value,
    lambda settings, email, _dir: ArxivSource(_http(SourceName.ARXIV, settings, email)),
)
PAPER_SOURCES.add(
    SourceName.OPENALEX.value,
    lambda settings, email, _dir: OpenAlexSource(_http(SourceName.OPENALEX, settings, email)),
)
PAPER_SOURCES.add(
    SourceName.CROSSREF.value,
    lambda settings, email, _dir: CrossrefSource(_http(SourceName.CROSSREF, settings, email)),
)
PAPER_SOURCES.add(
    SourceName.SEMANTIC_SCHOLAR.value,
    lambda settings, email, _dir: SemanticScholarSource(
        _http(SourceName.SEMANTIC_SCHOLAR, settings, email)
    ),
)
PAPER_SOURCES.add(
    SourceName.PUBMED.value,
    lambda settings, email, _dir: PubMedSource(_http(SourceName.PUBMED, settings, email)),
)
PAPER_SOURCES.add(
    SourceName.MANUAL.value,
    lambda _settings, _email, library_dir: ManualPaperSource(library_dir),
)


def build_paper_source(
    name: SourceName, settings: SourceSettings, config: SourcesConfig, project_root: Path
) -> PaperSource:
    return PAPER_SOURCES.get(name.value)(
        settings, config.contact_email, _resolve(config.manual_library_dir, project_root)
    )


def build_enabled_sources(config: SourcesConfig, project_root: Path) -> list[PaperSource]:
    """Instantiate every provider marked enabled in ``config/sources.yaml``."""
    sources = [
        build_paper_source(name, config.settings_for(name), config, project_root)
        for name in config.enabled_sources()
    ]
    logger.info(
        "paper_sources_built",
        enabled=[source.name.value for source in sources],
        disabled=[name.value for name in SourceName if name not in set(config.enabled_sources())],
    )
    return sources


def _resolve(path: Path, project_root: Path) -> Path:
    """Config paths are relative to the repository root unless given absolutely."""
    return path if path.is_absolute() else project_root / path
