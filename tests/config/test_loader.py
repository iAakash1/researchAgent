from __future__ import annotations

from pathlib import Path

import pytest

from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, ModelCatalog
from researchagent.core.exceptions import ConfigurationError


def write(directory: Path, name: str, body: str) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_and_validates(tmp_path: Path) -> None:
    write(
        tmp_path,
        "models",
        """
        default: main
        models:
          main:
            provider: ollama
            model: qwen3:8b
            params:
              temperature: 0.5
        """,
    )

    catalog = ConfigLoader(tmp_path).load("models", ModelCatalog)

    assert catalog.spec_for("main").model_name == "qwen3:8b"
    assert catalog.spec_for("main").params.temperature == 0.5
    assert catalog.resolve_alias(None) == "main"


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        ConfigLoader(tmp_path).load("models", ModelCatalog)

    assert "models.yaml" in excinfo.value.context["file"]


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    write(tmp_path, "models", "default: [unclosed\n")

    with pytest.raises(ConfigurationError):
        ConfigLoader(tmp_path).load("models", ModelCatalog)


def test_non_mapping_root_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "models", "- a\n- b\n")

    with pytest.raises(ConfigurationError) as excinfo:
        ConfigLoader(tmp_path).load("models", ModelCatalog)

    assert excinfo.value.context["actual_type"] == "list"


def test_default_alias_must_exist(tmp_path: Path) -> None:
    write(
        tmp_path,
        "models",
        """
        default: missing
        models:
          main:
            model: qwen3:8b
        """,
    )

    with pytest.raises(ConfigurationError):
        ConfigLoader(tmp_path).load("models", ModelCatalog)


def test_results_are_cached_until_cleared(tmp_path: Path) -> None:
    write(tmp_path, "models", "default: a\nmodels:\n  a:\n    model: m1\n")
    loader = ConfigLoader(tmp_path)

    first = loader.load("models", ModelCatalog)
    write(tmp_path, "models", "default: b\nmodels:\n  b:\n    model: m2\n")

    assert loader.load("models", ModelCatalog) is first

    loader.clear_cache()
    assert loader.load("models", ModelCatalog).default == "b"


def test_agent_spec_layers_over_defaults(tmp_path: Path) -> None:
    write(
        tmp_path,
        "agents",
        """
        defaults:
          model: reasoning
          timeout_seconds: 300
          retry:
            max_attempts: 3
        agents:
          planner:
            model: fast
        """,
    )

    config = ConfigLoader(tmp_path).load("agents", AgentConfig)
    planner = config.spec_for("planner")

    assert planner.model == "fast"
    assert planner.timeout_seconds == 300  # inherited
    assert planner.retry.max_attempts == 3  # inherited
    assert config.spec_for("unknown").model == "reasoning"


def test_repository_config_is_valid() -> None:
    """The committed config/ must always load — this is the guard against typos."""
    loader = ConfigLoader(Path(__file__).resolve().parents[2] / "config")

    catalog = loader.load("models", ModelCatalog)
    agents = loader.load("agents", AgentConfig)

    assert catalog.default in catalog.models
    assert agents.defaults.model in catalog.models
