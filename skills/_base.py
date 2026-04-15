from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from vision.capture import ScreenCapture
    from vision.client import VisionClient
    from vision.executor import ActionExecutor
    from data.client import DataClient


@dataclass
class AgentMessage:
    text: str
    images: list[bytes] = field(default_factory=list)
    skill_name: str | None = None
    progress: str | None = None


@dataclass
class SkillResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillContext:
    trigger_source: Literal["chat", "scheduler", "agent"]
    params: dict[str, Any]
    capture: "ScreenCapture"
    vision: "VisionClient"
    executor: "ActionExecutor"
    data: "DataClient"
    reply: Callable[[AgentMessage], Awaitable[None]]


class BaseSkill(ABC):
    name: str
    description: str
    triggers: list[str]
    parameters: type[BaseModel] = BaseModel
    schedule: str | None = None

    @abstractmethod
    async def run(self, ctx: SkillContext) -> SkillResult: ...

    async def on_interrupt(self) -> None:
        pass

    async def on_error(self, e: Exception) -> SkillResult:
        return SkillResult(success=False, message=str(e))
