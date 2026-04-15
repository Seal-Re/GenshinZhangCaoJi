# skills/explore_map.py
from __future__ import annotations

from pydantic import BaseModel

from scene.combat import CombatModule, TeamConfig
from scene.navigation import NavigationModule
from scene.settlement import SettlementModule
from skills._base import AgentMessage, BaseSkill, SkillContext, SkillResult
from config.loader import load_config


class ExploreMapParams(BaseModel):
    region: str = "当前区域"
    max_steps: int = 200


class ExploreMapSkill(BaseSkill):
    name = "explore_map"
    description = "在原神游戏地图上自动探索指定区域，开图、收集宝箱、解锁传送锚点"
    triggers = ["开图", "探索地图", "开蒙德", "开璃月", "开稻妻", "explore map"]
    parameters = ExploreMapParams
    schedule = None

    # Default team rotation — can be overridden via config in future
    _DEFAULT_ROTATION = [
        {"key": "1", "delay": 0.3},
        {"key": "e", "delay": 0.8},
        {"key": "q", "delay": 0.3},
    ]

    async def run(self, ctx: SkillContext) -> SkillResult:
        cfg = load_config()
        region = ctx.params.get("region", "当前区域")
        max_steps = ctx.params.get("max_steps", 200)

        nav = NavigationModule(ctx, minimap_region=cfg.game.minimap_region)
        combat = CombatModule(ctx, TeamConfig(rotation=self._DEFAULT_ROTATION))
        settlement = SettlementModule(ctx)

        await ctx.reply(AgentMessage(
            text=f"开始探索 {region}...",
            skill_name=self.name,
        ))

        step = 0
        while step < max_steps and nav.has_unexplored():
            target = nav.get_next_target()
            if target is None:
                break

            await nav.move_toward(target)

            # Handle combat if enemies present
            frame = ctx.capture.capture()
            if combat.detect_enemy(frame):
                await combat.clear_enemies()
                await settlement.confirm_and_close()

            step += 1
            if step % 20 == 0:
                await ctx.reply(AgentMessage(
                    text=f"探索中 ({step}/{max_steps} 步)",
                    skill_name=self.name,
                    progress=f"步数: {step}",
                ))

        return SkillResult(
            success=True,
            message=f"{region} 探索完成，共执行 {step} 步",
            data={"steps": step, "region": region},
        )
