from __future__ import annotations
from dataclasses import dataclass

import mss
import numpy as np


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height}


class ScreenCapture:
    def __init__(self) -> None:
        self._sct = mss.mss()

    def capture(self) -> np.ndarray:
        """Capture the primary monitor."""
        monitor = self._sct.monitors[1]
        return np.array(self._sct.grab(monitor))

    def capture_region(self, region: Region) -> np.ndarray:
        """Capture a specific screen region."""
        return np.array(self._sct.grab(region.to_dict()))

    def capture_minimap(self, minimap_region: tuple[int, int, int, int]) -> np.ndarray:
        """Convenience method: capture minimap from (left, top, width, height) tuple."""
        left, top, width, height = minimap_region
        return self.capture_region(Region(left, top, width, height))
