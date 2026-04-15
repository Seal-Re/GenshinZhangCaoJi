# scene/settlement.py
from __future__ import annotations

from scene._base import BaseSceneModule


class SettlementModule(BaseSceneModule):
    async def is_settlement_visible(self) -> bool:
        """Use VLM to check if a settlement/reward screen is on screen."""
        frame = self.ctx.capture.capture()
        result = await self.ctx.vision.understand(
            frame,
            "Is there a settlement screen, reward collection screen, or "
            "mission/commission complete screen visible? Answer only yes or no.",
        )
        return "yes" in result.lower()

    async def confirm_and_close(self) -> bool:
        """Confirm settlement if visible. Returns True if confirmed."""
        if await self.is_settlement_visible():
            await self.ctx.executor.key_press("f")
            return True
        return False
