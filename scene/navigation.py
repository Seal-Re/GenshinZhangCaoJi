from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from scene._base import BaseSceneModule
from skills._base import SkillContext


@dataclass
class MapPoint:
    x: int
    y: int


class MinimapAnalyzer:
    # Unexplored areas: gray pixels — low saturation in HSV
    _GRAY_LOWER = np.array([0, 0, 40], dtype=np.uint8)
    _GRAY_UPPER = np.array([180, 50, 210], dtype=np.uint8)
    _MIN_AREA = 15  # ignore tiny noise patches

    def get_unexplored_areas(self, minimap_img: np.ndarray) -> list[MapPoint]:
        """Return centroids of gray (unexplored) regions on the minimap."""
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._GRAY_LOWER, self._GRAY_UPPER)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        points: list[MapPoint] = []
        for c in contours:
            if cv2.contourArea(c) < self._MIN_AREA:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append(MapPoint(cx, cy))
        return points

    def get_player_position(self, minimap_img: np.ndarray) -> MapPoint:
        """Player icon is always near the center of the minimap."""
        h, w = minimap_img.shape[:2]
        return MapPoint(w // 2, h // 2)


class NavigationModule(BaseSceneModule):
    def __init__(
        self,
        ctx: SkillContext,
        minimap_region: tuple[int, int, int, int],
    ) -> None:
        super().__init__(ctx)
        self.analyzer = MinimapAnalyzer()
        self._minimap_region = minimap_region

    def has_unexplored(self) -> bool:
        minimap = self.ctx.capture.capture_minimap(self._minimap_region)
        return len(self.analyzer.get_unexplored_areas(minimap)) > 0

    def get_next_target(self) -> MapPoint | None:
        minimap = self.ctx.capture.capture_minimap(self._minimap_region)
        targets = self.analyzer.get_unexplored_areas(minimap)
        if not targets:
            return None
        # Pick the closest target to player (player is at center of minimap)
        player = self.analyzer.get_player_position(minimap)
        targets.sort(
            key=lambda p: (p.x - player.x) ** 2 + (p.y - player.y) ** 2
        )
        return targets[0]

    async def move_toward(self, target: MapPoint) -> None:
        """Press directional key toward target for 0.5s."""
        h, w = 95, 95  # minimap dimensions
        player = MapPoint(w // 2, h // 2)
        dx = target.x - player.x
        dy = target.y - player.y

        if abs(dx) >= abs(dy):
            key = "d" if dx > 0 else "a"
        else:
            key = "s" if dy > 0 else "w"

        await self.ctx.executor.hold_key(key, 0.5)
