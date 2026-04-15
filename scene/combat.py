# scene/combat.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field

import cv2
import numpy as np

from scene._base import BaseSceneModule
from skills._base import SkillContext


@dataclass
class TeamConfig:
    # Each action: {"key": str, "delay": float (seconds after press)}
    rotation: list[dict[str, float | str]] = field(default_factory=list)


class CombatModule(BaseSceneModule):
    # Red HP bar detection: HSV hue 0-10 or 170-180, high saturation
    _RED_LOWER_1 = np.array([0, 120, 70], dtype=np.uint8)
    _RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
    _RED_LOWER_2 = np.array([170, 120, 70], dtype=np.uint8)
    _RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)
    _RED_PIXEL_THRESHOLD = 300

    def __init__(self, ctx: SkillContext, team_config: TeamConfig) -> None:
        super().__init__(ctx)
        self.team_config = team_config

    def detect_enemy(self, frame: np.ndarray) -> bool:
        """Detect enemy presence via red HP bar pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self._RED_LOWER_1, self._RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self._RED_LOWER_2, self._RED_UPPER_2)
        combined = cv2.bitwise_or(mask1, mask2)
        return int(cv2.countNonZero(combined)) > self._RED_PIXEL_THRESHOLD

    async def execute_rotation(self) -> None:
        """Execute one full skill rotation cycle."""
        for action in self.team_config.rotation:
            await self.ctx.executor.key_press(str(action["key"]))
            await asyncio.sleep(float(action["delay"]))

    async def clear_enemies(self) -> None:
        """
        Run rotation loop until no enemies detected or settlement screen appears.
        Uses VLM only to detect final settlement — not for per-frame decisions.
        """
        max_iterations = 30
        for _ in range(max_iterations):
            frame = self.ctx.capture.capture()
            if not self.detect_enemy(frame):
                # Double-check: VLM confirms battle is over
                settlement = await self.ctx.vision.understand(
                    frame,
                    "Is there a battle reward/settlement screen visible? Answer only yes or no.",
                )
                if "yes" in settlement.lower():
                    # Dismiss settlement
                    await self.ctx.executor.key_press("f")
                break
            await self.execute_rotation()
