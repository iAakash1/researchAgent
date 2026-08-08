"""Project-wide constants. No configuration values, no magic numbers elsewhere."""

from __future__ import annotations

from typing import Final

APP_NAME: Final = "ResearchAgent"
APP_SLUG: Final = "researchagent"
APP_VERSION: Final = "0.1.0"
APP_DESCRIPTION: Final = "Local-first multi-agent research intelligence platform"

# Correlation keys bound to structlog context and propagated through the workflow.
RUN_ID_KEY: Final = "run_id"
SESSION_ID_KEY: Final = "session_id"
AGENT_KEY: Final = "agent"

# Model alias used when an agent has no explicit binding in config/agents.yaml.
DEFAULT_MODEL_ALIAS: Final = "default"

SECONDS_PER_MILLISECOND: Final = 1000.0
