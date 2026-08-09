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

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from researchagent.core.exceptions import ConfigurationError

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


class GroqSettings(BaseModel):
    """Optional external inference. Local-first: Ollama remains the default.

    The API key is deliberately absent from this model — it is read from the environment
    as ``GROQ_API_KEY`` and never from YAML, so a config file can be committed safely.
    """

    base_url: str = "https://api.groq.com/openai/v1"
    request_timeout_seconds: float = Field(default=120.0, gt=0)


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
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    data_dir: Path = PROJECT_ROOT / "data"

    api_host: str = "0.0.0.0"  # noqa: S104 - container-facing service
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    # Read from GROQ_API_KEY directly, bypassing the RESEARCHAGENT_ prefix, so it matches
    # the conventional variable name. SecretStr keeps it out of reprs and logs.
    # Coarse, optional overrides for every model alias at once. Unset by default, which
    # is what keeps a fresh checkout local-first.
    llm_provider: str | None = Field(default=None, validation_alias=AliasChoices("LLM_PROVIDER"))
    llm_model: str | None = Field(default=None, validation_alias=AliasChoices("LLM_MODEL"))
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY")
    )
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)

    def require_groq_key(self) -> str:
        """The key, or a clear configuration error naming the fix.

        Never falls back to Ollama: a run that silently used a different provider than the
        one requested is a run whose results cannot be attributed.
        """
        if self.groq_api_key is None:
            raise ConfigurationError(
                "Groq was requested but GROQ_API_KEY is not set",
                remedy="Export GROQ_API_KEY, or set the model alias back to an ollama provider",
            )
        return self.groq_api_key.get_secret_value()

    @property
    def is_local(self) -> bool:
        return self.environment is Environment.LOCAL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
