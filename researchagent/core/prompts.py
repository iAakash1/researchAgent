"""Versioned prompt loading.

One file per prompt version (``prompts/<agent>/<version>.md``), never edited in place —
comparing v1 against v2 is a diff of two files, which is what makes prompt evaluation
possible later.

A file is split into named sections by ``## <name>`` headings, so a multi-step agent
keeps all of its steps in one diffable unit::

    ## system
    You are a research planning specialist.

    ## framing
    Research goal: ${goal}

Placeholders use ``string.Template`` syntax (``${name}``) rather than ``str.format``
because prompt bodies routinely contain literal JSON braces. A literal ``$`` must be
written ``$$``.
"""

from __future__ import annotations

import re
from pathlib import Path
from string import Template

from researchagent.core.exceptions import PromptError
from researchagent.core.logging import get_logger

logger = get_logger(__name__)

_SECTION_HEADING = re.compile(r"^##[ \t]+(?P<name>[\w.-]+)[ \t]*$", re.MULTILINE)


class PromptTemplate:
    """The parsed sections of one prompt version."""

    def __init__(self, agent: str, version: str, sections: dict[str, str], source: Path) -> None:
        self._agent = agent
        self._version = version
        self._sections = sections
        self._source = source

    @property
    def agent(self) -> str:
        return self._agent

    @property
    def version(self) -> str:
        return self._version

    @property
    def source(self) -> Path:
        return self._source

    def section_names(self) -> tuple[str, ...]:
        return tuple(self._sections)

    def section(self, name: str) -> str:
        try:
            return self._sections[name.lower()]
        except KeyError:
            raise PromptError(
                "Prompt section not found",
                agent=self._agent,
                version=self._version,
                section=name,
                available=sorted(self._sections),
                file=str(self._source),
            ) from None

    def render(self, name: str, /, **variables: object) -> str:
        """Substitute ``${var}`` placeholders in a section.

        Missing or surplus variables are errors: a silently half-rendered prompt is far
        harder to debug than a loud failure at startup.
        """
        template = Template(self.section(name))
        try:
            return template.substitute(variables)
        except KeyError as exc:
            raise PromptError(
                "Prompt is missing a required variable",
                agent=self._agent,
                version=self._version,
                section=name,
                variable=str(exc.args[0]),
                provided=sorted(variables),
            ) from exc
        except ValueError as exc:
            raise PromptError(
                "Malformed placeholder in prompt (use $$ for a literal $)",
                agent=self._agent,
                version=self._version,
                section=name,
                reason=str(exc),
                file=str(self._source),
            ) from exc


class PromptLibrary:
    """Loads and caches prompt versions from the prompts directory."""

    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir
        self._cache: dict[tuple[str, str], PromptTemplate] = {}

    @property
    def prompts_dir(self) -> Path:
        return self._prompts_dir

    def load(self, agent: str, version: str = "v1") -> PromptTemplate:
        key = (agent, version)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        path = self._prompts_dir / agent / f"{version}.md"
        if not path.is_file():
            raise PromptError(
                "Prompt version not found",
                agent=agent,
                version=version,
                file=str(path),
                available=self.available_versions(agent),
            )

        template = PromptTemplate(agent, version, _parse_sections(path), path)
        self._cache[key] = template
        logger.debug(
            "prompt_loaded", agent=agent, version=version, sections=template.section_names()
        )
        return template

    def available_versions(self, agent: str) -> list[str]:
        directory = self._prompts_dir / agent
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.md"))

    def clear_cache(self) -> None:
        self._cache.clear()


def _parse_sections(path: Path) -> dict[str, str]:
    """Split a prompt file on ``## name`` headings.

    Text before the first heading is treated as file-level commentary and dropped, so a
    prompt file can carry a title and authoring notes without polluting the model input.
    """
    text = path.read_text(encoding="utf-8")
    matches = list(_SECTION_HEADING.finditer(text))
    if not matches:
        raise PromptError(
            "Prompt file defines no '## <section>' headings",
            file=str(path),
        )

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group("name").lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            raise PromptError("Prompt section is empty", file=str(path), section=name)
        if name in sections:
            raise PromptError("Duplicate prompt section", file=str(path), section=name)
        sections[name] = body

    return sections
