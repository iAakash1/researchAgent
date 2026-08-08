from __future__ import annotations

import pytest

from researchagent.agents.base import AgentContext
from researchagent.agents.planner import PlannerAgent
from researchagent.agents.planner.schemas import (
    FramingDraft,
    PlannerInput,
    PlannerOptions,
    QuestionDraft,
    StrategyDraft,
)
from researchagent.agents.registry import AGENTS, build_agent
from researchagent.config.schemas import AgentConfig, AgentSpec
from researchagent.core.exceptions import AgentExecutionError, ConfigurationError
from researchagent.core.prompts import PromptLibrary
from researchagent.core.retry import RetryPolicy
from researchagent.models.research import QuestionPriority
from researchagent.services.llm_service import BoundLLM, LLMService
from tests.conftest import FakeLLMProvider

NO_RETRY = RetryPolicy(max_attempts=1)


def framing(**overrides: object) -> FramingDraft:
    defaults: dict[str, object] = {
        "topic": "Agentic AI in clinical decision support",
        "framing": "This review covers autonomous multi-agent systems applied to clinical "
        "decision support, excluding purely diagnostic imaging models.",
        "questions": [
            QuestionDraft(
                question="Which agent architectures are used for clinical triage?",
                rationale="Architecture choice drives safety guarantees and auditability.",
                priority=QuestionPriority.HIGH,
                keywords=["agent architecture", "clinical triage"],
            ),
            QuestionDraft(
                question="How is factual grounding evaluated in clinical agents?",
                rationale="Ungrounded output is the dominant safety failure in this field.",
                priority=QuestionPriority.MEDIUM,
                keywords=["grounding", "hallucination"],
            ),
        ],
    }
    return FramingDraft.model_validate(defaults | overrides)


def strategy(**overrides: object) -> StrategyDraft:
    defaults: dict[str, object] = {
        "queries": ["agentic ai clinical decision support", "multi-agent llm healthcare"],
        "inclusion_criteria": ["Reports a quantitative evaluation"],
        "exclusion_criteria": ["Opinion pieces without evaluation"],
        "expected_methods": ["ReAct", "retrieval-augmented generation"],
        "expected_datasets": ["MIMIC-III", "MedQA"],
        "evaluation_metrics": ["accuracy", "factuality"],
    }
    return StrategyDraft.model_validate(defaults | overrides)


def make_agent(
    provider: FakeLLMProvider,
    prompt_library: PromptLibrary,
    *,
    options: dict[str, object] | None = None,
) -> PlannerAgent:
    from researchagent.config.schemas import ModelSpec

    spec = AgentSpec(retry=NO_RETRY, options=options or {})
    llm = BoundLLM("reasoning", ModelSpec(model="fake-model"), provider)
    return PlannerAgent(llm, spec, prompt_library)


async def test_produces_a_complete_plan(prompt_library: PromptLibrary) -> None:
    provider = FakeLLMProvider(structured_sequence=[framing(), strategy()])
    agent = make_agent(provider, prompt_library)

    result = await agent.run(PlannerInput(goal="Agentic AI in healthcare"), AgentContext())
    plan = result.output.plan

    assert plan.topic == "Agentic AI in clinical decision support"
    assert [q.id for q in plan.research_questions] == ["RQ1", "RQ2"]
    assert plan.strategy.queries == [
        "agentic ai clinical decision support",
        "multi-agent llm healthcare",
    ]
    assert plan.expected_datasets == ["MIMIC-III", "MedQA"]
    assert len(provider.calls) == 2  # framing, then strategy


async def test_framing_and_strategy_are_separate_prompted_phases(
    prompt_library: PromptLibrary,
) -> None:
    provider = FakeLLMProvider(structured_sequence=[framing(), strategy()])
    agent = make_agent(provider, prompt_library)

    await agent.run(PlannerInput(goal="Agentic AI in healthcare"), AgentContext())

    framing_prompt = provider.calls[0][-1].content
    strategy_prompt = provider.calls[1][-1].content

    assert "Agentic AI in healthcare" in framing_prompt
    # Phase 2 must see the questions phase 1 produced, or the strategy is unanchored.
    assert "RQ1" in strategy_prompt
    assert "Which agent architectures are used for clinical triage?" in strategy_prompt


async def test_questions_are_ordered_by_priority_and_capped(
    prompt_library: PromptLibrary,
) -> None:
    drafts = [
        QuestionDraft(
            question=f"Low priority question number {i} about clinical agents?",
            rationale="Rationale text that is long enough to pass validation.",
            priority=QuestionPriority.LOW,
        )
        for i in range(3)
    ] + [
        QuestionDraft(
            question="The single most important question about agent safety?",
            rationale="Rationale text that is long enough to pass validation.",
            priority=QuestionPriority.HIGH,
        )
    ]
    provider = FakeLLMProvider(structured_sequence=[framing(questions=drafts), strategy()])
    agent = make_agent(
        provider, prompt_library, options={"min_research_questions": 1, "max_research_questions": 2}
    )

    result = await agent.run(PlannerInput(goal="Agent safety in medicine"), AgentContext())
    questions = result.output.plan.research_questions

    assert len(questions) == 2
    assert questions[0].priority is QuestionPriority.HIGH
    assert questions[0].id == "RQ1"


async def test_duplicate_questions_are_dropped(prompt_library: PromptLibrary) -> None:
    duplicate = QuestionDraft(
        question="Which agent architectures are used for clinical triage?",
        rationale="Architecture choice drives safety guarantees and auditability.",
        priority=QuestionPriority.HIGH,
    )
    near_duplicate = duplicate.model_copy(
        update={"question": "  which  AGENT architectures are used for clinical triage?  "}
    )
    provider = FakeLLMProvider(
        structured_sequence=[framing(questions=[duplicate, near_duplicate]), strategy()]
    )
    agent = make_agent(provider, prompt_library)

    result = await agent.run(PlannerInput(goal="Clinical triage agents"), AgentContext())

    assert len(result.output.plan.research_questions) == 1


async def test_request_constraints_override_configured_limit(
    prompt_library: PromptLibrary,
) -> None:
    provider = FakeLLMProvider(structured_sequence=[framing(), strategy()])
    agent = make_agent(provider, prompt_library, options={"max_research_questions": 5})

    result = await agent.run(
        PlannerInput(
            goal="Agentic AI in healthcare",
            constraints={"max_research_questions": 1, "year_from": 2021},
        ),
        AgentContext(),
    )

    assert len(result.output.plan.research_questions) == 1
    assert result.output.plan.strategy.year_from == 2021


async def test_constraints_and_feedback_reach_the_prompt(
    prompt_library: PromptLibrary,
) -> None:
    provider = FakeLLMProvider(structured_sequence=[framing(), strategy()])
    agent = make_agent(provider, prompt_library)

    await agent.run(
        PlannerInput(
            goal="Agentic AI in healthcare",
            constraints={"year_from": 2022, "focus_areas": ["triage"], "exclusions": ["imaging"]},
            feedback=["Missing comparison against single-agent baselines"],
        ),
        AgentContext(),
    )

    prompt = provider.calls[0][-1].content
    assert "2022" in prompt
    assert "triage" in prompt
    assert "imaging" in prompt
    assert "single-agent baselines" in prompt


async def test_empty_questions_fail_the_run(prompt_library: PromptLibrary) -> None:
    provider = FakeLLMProvider(structured_sequence=[framing(questions=[]), strategy()])
    agent = make_agent(provider, prompt_library)

    with pytest.raises(AgentExecutionError) as excinfo:
        await agent.run(PlannerInput(goal="Agentic AI in healthcare"), AgentContext())

    assert excinfo.value.context["cause"] == "output_parsing_error"


async def test_missing_queries_fall_back_to_question_terms(
    prompt_library: PromptLibrary,
) -> None:
    provider = FakeLLMProvider(structured_sequence=[framing(), strategy(queries=["", "   "])])
    agent = make_agent(provider, prompt_library)

    result = await agent.run(PlannerInput(goal="Agentic AI in healthcare"), AgentContext())

    assert result.output.plan.strategy.queries  # never empty
    assert "Which agent architectures are used for clinical triage?" in (
        result.output.plan.strategy.queries
    )


async def test_blank_and_duplicate_list_items_are_cleaned(
    prompt_library: PromptLibrary,
) -> None:
    provider = FakeLLMProvider(
        structured_sequence=[
            framing(),
            strategy(expected_datasets=["MIMIC-III", "  mimic-iii ", "", "MedQA"]),
        ]
    )
    agent = make_agent(provider, prompt_library)

    result = await agent.run(PlannerInput(goal="Agentic AI in healthcare"), AgentContext())

    assert result.output.plan.expected_datasets == ["MIMIC-III", "MedQA"]


async def test_invalid_options_are_a_configuration_error(
    prompt_library: PromptLibrary,
) -> None:
    provider = FakeLLMProvider(structured_sequence=[framing(), strategy()])
    agent = make_agent(
        provider, prompt_library, options={"min_research_questions": 8, "max_research_questions": 2}
    )

    with pytest.raises(AgentExecutionError) as excinfo:
        await agent.run(PlannerInput(goal="Agentic AI in healthcare"), AgentContext())

    assert excinfo.value.context["cause"] == ConfigurationError.code


def test_planner_is_registered_and_buildable(
    agent_config: AgentConfig, llm_service: LLMService, prompt_library: PromptLibrary
) -> None:
    assert "planner" in AGENTS

    agent = build_agent(
        "planner", agent_config=agent_config, llm_service=llm_service, prompts=prompt_library
    )

    assert isinstance(agent, PlannerAgent)
    # Options in config/agents.yaml must satisfy the agent's own schema.
    assert PlannerOptions.model_validate(agent.spec.options).max_research_questions == 5
    assert agent.prompt.version == "v1"
