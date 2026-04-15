from __future__ import annotations
from skills._base import SkillContext


class BaseSceneModule:
    def __init__(self, ctx: SkillContext) -> None:
        self.ctx = ctx
