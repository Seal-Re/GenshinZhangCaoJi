# vision/executor.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass

import pyautogui

# Safety: move mouse to top-left corner to abort
pyautogui.FAILSAFE = True
# Small pause between actions to avoid overwhelming game input
pyautogui.PAUSE = 0.05


@dataclass
class Point:
    x: int
    y: int


class ActionExecutor:
    async def click(self, point: Point, button: str = "left") -> None:
        pyautogui.click(point.x, point.y, button=button)
        await asyncio.sleep(0.1)

    async def double_click(self, point: Point) -> None:
        pyautogui.doubleClick(point.x, point.y)
        await asyncio.sleep(0.1)

    async def key_press(self, key: str, duration: float = 0.05) -> None:
        pyautogui.keyDown(key)
        await asyncio.sleep(duration)
        pyautogui.keyUp(key)

    async def hold_key(self, key: str, duration: float) -> None:
        pyautogui.keyDown(key)
        await asyncio.sleep(duration)
        pyautogui.keyUp(key)

    async def move_to(self, point: Point, duration: float = 0.2) -> None:
        pyautogui.moveTo(point.x, point.y, duration=duration)

    async def scroll(self, clicks: int) -> None:
        pyautogui.scroll(clicks)
        await asyncio.sleep(0.1)
