"""FastAPI dependency providers.

Routes depend on services, never on the container itself beyond this module.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from researchagent.container import Container
from researchagent.core.settings import Settings
from researchagent.services.llm_service import LLMService
from researchagent.workflows.runner import WorkflowRunner


def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


def get_settings_dep(container: Annotated[Container, Depends(get_container)]) -> Settings:
    return container.settings


def get_llm_service(container: Annotated[Container, Depends(get_container)]) -> LLMService:
    return container.llm_service


def get_workflow_runner(
    container: Annotated[Container, Depends(get_container)],
) -> WorkflowRunner:
    return container.workflow_runner


ContainerDep = Annotated[Container, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
WorkflowRunnerDep = Annotated[WorkflowRunner, Depends(get_workflow_runner)]
