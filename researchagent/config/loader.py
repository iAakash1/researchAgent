"""Typed YAML configuration loading.

Each domain owns its own schema and asks the loader for it:

    catalog = loader.load("models", ModelCatalog)

The loader only knows about files and validation, never about what the config means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from researchagent.core.exceptions import ConfigurationError

TModel = TypeVar("TModel", bound=BaseModel)


class ConfigLoader:
    """Loads ``<config_dir>/<name>.yaml`` into a validated Pydantic model."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._cache: dict[tuple[str, type[BaseModel]], BaseModel] = {}

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    def load(self, name: str, schema: type[TModel], *, use_cache: bool = True) -> TModel:
        cache_key = (name, schema)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            assert isinstance(cached, schema)  # noqa: S101 - key is (name, schema)
            return cached

        raw = self._read(name)
        try:
            parsed = schema.model_validate(raw)
        except ValidationError as exc:
            raise ConfigurationError(
                "Configuration file failed validation",
                file=str(self._path(name)),
                schema=schema.__name__,
                errors=exc.errors(include_url=False),
            ) from exc

        self._cache[cache_key] = parsed
        return parsed

    def clear_cache(self) -> None:
        self._cache.clear()

    def _path(self, name: str) -> Path:
        return self._config_dir / f"{name}.yaml"

    def _read(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.is_file():
            raise ConfigurationError(
                "Configuration file not found",
                file=str(path),
                config_dir=str(self._config_dir),
            )
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError("Malformed YAML", file=str(path), reason=str(exc)) from exc

        if content is None:
            return {}
        if not isinstance(content, dict):
            raise ConfigurationError(
                "Configuration root must be a mapping",
                file=str(path),
                actual_type=type(content).__name__,
            )
        return content
