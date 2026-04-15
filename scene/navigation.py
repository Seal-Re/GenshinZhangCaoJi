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
