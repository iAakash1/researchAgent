from __future__ import annotations

from pathlib import Path

import pytest

from researchagent.core.exceptions import PromptError
from researchagent.core.prompts import PromptLibrary

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_prompt(root: Path, agent: str, version: str, body: str) -> Path:
    directory = root / agent
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_parses_sections_and_drops_preamble(tmp_path: Path) -> None:
    write_prompt(
        tmp_path,
        "planner",
        "v1",
        "# authoring notes, not model input\n\n## system\nYou plan.\n\n## framing\nGoal: ${goal}\n",
    )

    template = PromptLibrary(tmp_path).load("planner", "v1")

    assert template.section_names() == ("system", "framing")
    assert template.section("system") == "You plan."
    assert "authoring notes" not in template.section("system")


def test_render_substitutes_variables(tmp_path: Path) -> None:
    write_prompt(tmp_path, "planner", "v1", "## framing\nGoal: ${goal} (${count} questions)\n")

    rendered = PromptLibrary(tmp_path).load("planner").render("framing", goal="RAG", count=3)

    assert rendered == "Goal: RAG (3 questions)"


def test_json_braces_survive_rendering(tmp_path: Path) -> None:
    """str.format would choke here; string.Template is why prompts can show JSON."""
    write_prompt(tmp_path, "planner", "v1", '## framing\nReturn {"topic": "${goal}"}\n')

    rendered = PromptLibrary(tmp_path).load("planner").render("framing", goal="x")

    assert rendered == 'Return {"topic": "x"}'


def test_missing_variable_is_loud(tmp_path: Path) -> None:
    write_prompt(tmp_path, "planner", "v1", "## framing\n${goal} and ${missing}\n")

    with pytest.raises(PromptError) as excinfo:
        PromptLibrary(tmp_path).load("planner").render("framing", goal="x")

    assert excinfo.value.context["variable"] == "missing"


def test_unknown_section_lists_available(tmp_path: Path) -> None:
    write_prompt(tmp_path, "planner", "v1", "## system\nhi\n")

    with pytest.raises(PromptError) as excinfo:
        PromptLibrary(tmp_path).load("planner").section("framing")

    assert excinfo.value.context["available"] == ["system"]


def test_missing_version_reports_available_versions(tmp_path: Path) -> None:
    write_prompt(tmp_path, "planner", "v1", "## system\nhi\n")

    with pytest.raises(PromptError) as excinfo:
        PromptLibrary(tmp_path).load("planner", "v9")

    assert excinfo.value.context["available"] == ["v1"]


def test_file_without_headings_is_rejected(tmp_path: Path) -> None:
    write_prompt(tmp_path, "planner", "v1", "just some text\n")

    with pytest.raises(PromptError):
        PromptLibrary(tmp_path).load("planner")


def test_empty_and_duplicate_sections_are_rejected(tmp_path: Path) -> None:
    write_prompt(tmp_path, "a", "v1", "## system\n\n## framing\nx\n")
    with pytest.raises(PromptError, match="empty"):
        PromptLibrary(tmp_path).load("a")

    write_prompt(tmp_path, "b", "v1", "## system\nx\n\n## system\ny\n")
    with pytest.raises(PromptError, match="Duplicate"):
        PromptLibrary(tmp_path).load("b")


def test_templates_are_cached(tmp_path: Path) -> None:
    write_prompt(tmp_path, "planner", "v1", "## system\nfirst\n")
    library = PromptLibrary(tmp_path)

    first = library.load("planner")
    write_prompt(tmp_path, "planner", "v1", "## system\nsecond\n")

    assert library.load("planner") is first
    library.clear_cache()
    assert library.load("planner").section("system") == "second"


def test_repository_planner_prompt_is_loadable() -> None:
    """Guards the committed prompt against typos and missing sections."""
    template = PromptLibrary(REPO_ROOT / "prompts").load("planner", "v1")

    assert set(template.section_names()) == {"system", "framing", "strategy"}
    rendered = template.render(
        "framing",
        goal="agentic ai",
        constraints_block="Constraints: none.",
        feedback_block="",
        min_questions=3,
        max_questions=5,
        max_keywords=6,
    )
    assert "agentic ai" in rendered
    assert "$" not in rendered  # every placeholder was consumed
