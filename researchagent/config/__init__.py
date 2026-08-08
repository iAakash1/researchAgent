"""Configuration loading and its typed schemas.

The YAML files themselves live in the repository-root ``config/`` directory; this
package is the code that reads and validates them.
"""

from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, AgentSpec, ModelCatalog, ModelSpec

__all__ = ["AgentConfig", "AgentSpec", "ConfigLoader", "ModelCatalog", "ModelSpec"]
