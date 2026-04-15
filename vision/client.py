# vision/client.py
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from vision.providers import VLMProvider


@dataclass
class Detection:
    found: bool
    location: tuple[int, int] | None = None
    confidence: float = 0.0


class VisionClient:
    def __init__(self, vlm_provider: VLMProvider) -> None:
        self._vlm = vlm_provider

    def fast_detect(
        self,
        frame: np.ndarray,
        template: np.ndarray,
        threshold: float = 0.8,
    ) -> Detection:
        """OpenCV template matching — sub-50ms, for UI/combat detection."""
        # Ensure both are BGR (3-channel) before matching
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if template.ndim == 3 and template.shape[2] == 4:
            template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

        result = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            return Detection(found=True, location=max_loc, confidence=float(max_val))
        return Detection(found=False, confidence=float(max_val))

    async def understand(self, screenshot: np.ndarray, prompt: str) -> str:
        """VLM semantic analysis — 1-3s, for state/phase understanding."""
        _, buf = cv2.imencode(".png", screenshot)
        return await self._vlm.analyze(buf.tobytes(), prompt)
