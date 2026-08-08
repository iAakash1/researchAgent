"""Runtime settings: infrastructure endpoints and process-level knobs.

Split of responsibilities:
    * ``Settings``  -> environment-specific values (hosts, ports, secrets, log level).
    * ``config/*.yaml`` -> behavioural values (models, agent wiring, RAG params).

Environment variables use the ``RESEARCHAGENT_`` prefix and ``__`` for nesting,
e.g. ``RESEARCHAGENT_OLLAMA__BASE_URL=http://ollama:11434``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    # How long Ollama keeps a model resident after a request; avoids reload cost
    # between chained agent calls.
    keep_alive: str = "10m"


class PostgresSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "researchagent"
    password: SecretStr = SecretStr("researchagent")
    database: str = "researchagent"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    api_key: SecretStr | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class Neo4jSettings(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("researchagent")
    database: str = "neo4j"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESEARCHAGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE

    project_root: Path = PROJECT_ROOT
    config_dir: Path = PROJECT_ROOT / "config"
    data_dir: Path = PROJECT_ROOT / "data"

    api_host: str = "0.0.0.0"  # noqa: S104 - container-facing service
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)

    @property
    def is_local(self) -> bool:
        return self.environment is Environment.LOCAL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
