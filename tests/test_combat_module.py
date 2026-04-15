import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock
from scene.combat import CombatModule, TeamConfig
from skills._base import SkillContext


def make_frame_with_red_hp_bar():
    """Frame with red pixels (simulating enemy HP bar)."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Red in BGR: (0, 0, 200) — in HSV this is red hue ~0°
    frame[50:60, 200:400, 0] = 0    # B
    frame[50:60, 200:400, 1] = 0    # G
    frame[50:60, 200:400, 2] = 200  # R
    return frame


def make_empty_frame():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def make_mock_ctx():
    ctx = MagicMock()
    ctx.executor = AsyncMock()
    ctx.vision = MagicMock()
    ctx.vision.understand = AsyncMock(return_value="no")
    ctx.capture = MagicMock()
    ctx.capture.capture.return_value = make_empty_frame()
    ctx.reply = AsyncMock()
    return ctx


def test_detect_enemy_with_red_hp_bar():
    ctx = make_mock_ctx()
    team = TeamConfig(rotation=[{"key": "1", "delay": 0.1}])
    module = CombatModule(ctx, team)
    frame = make_frame_with_red_hp_bar()
    assert module.detect_enemy(frame) is True


def test_detect_enemy_no_enemy():
    ctx = make_mock_ctx()
    team = TeamConfig(rotation=[{"key": "1", "delay": 0.1}])
    module = CombatModule(ctx, team)
    frame = make_empty_frame()
    assert module.detect_enemy(frame) is False


@pytest.mark.asyncio
async def test_execute_rotation_presses_keys():
    ctx = make_mock_ctx()
    team = TeamConfig(rotation=[
        {"key": "1", "delay": 0.1},
        {"key": "e", "delay": 0.1},
    ])
    module = CombatModule(ctx, team)
    await module.execute_rotation()
    assert ctx.executor.key_press.call_count == 2
    keys_pressed = [c[0][0] for c in ctx.executor.key_press.call_args_list]
    assert keys_pressed == ["1", "e"]
