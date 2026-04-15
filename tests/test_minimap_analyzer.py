import numpy as np
import pytest
from scene.navigation import MinimapAnalyzer, MapPoint


def make_minimap_with_gray_patch():
    """
    Synthetic minimap: colored background (explored) + gray patch (unexplored).
    Minimap is 95x95. Gray region at center-right.
    """
    img = np.zeros((95, 95, 3), dtype=np.uint8)
    # Colored (explored) background — green-ish
    img[:, :] = [30, 120, 30]
    # Gray (unexplored) region at (60:80, 60:80) — 20x20 px
    img[60:80, 60:80] = [100, 100, 100]
    return img


def make_all_explored_minimap():
    """All colored — no unexplored areas."""
    img = np.zeros((95, 95, 3), dtype=np.uint8)
    img[:, :] = [30, 120, 30]
    return img


def test_finds_unexplored_gray_region():
    analyzer = MinimapAnalyzer()
    minimap = make_minimap_with_gray_patch()
    points = analyzer.get_unexplored_areas(minimap)
    assert len(points) >= 1
    # The centroid of the gray patch (60:80, 60:80) is around (70, 70)
    cx = points[0].x
    cy = points[0].y
    assert 55 <= cx <= 85
    assert 55 <= cy <= 85


def test_no_unexplored_on_fully_explored_map():
    analyzer = MinimapAnalyzer()
    minimap = make_all_explored_minimap()
    points = analyzer.get_unexplored_areas(minimap)
    assert len(points) == 0


def test_get_player_position_returns_center():
    analyzer = MinimapAnalyzer()
    minimap = np.zeros((95, 95, 3), dtype=np.uint8)
    pos = analyzer.get_player_position(minimap)
    # Player is always approximately at center of minimap
    assert isinstance(pos, MapPoint)
    assert 40 <= pos.x <= 55
    assert 40 <= pos.y <= 55


import asyncio
from unittest.mock import MagicMock, AsyncMock
from scene.navigation import NavigationModule
from skills._base import SkillContext


def make_mock_ctx(minimap_img=None):
    if minimap_img is None:
        minimap_img = make_minimap_with_gray_patch()
    ctx = MagicMock()
    ctx.capture.capture_minimap.return_value = minimap_img
    ctx.executor = AsyncMock()
    ctx.reply = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_navigation_has_unexplored_true():
    ctx = make_mock_ctx(make_minimap_with_gray_patch())
    nav = NavigationModule(ctx, minimap_region=(1820, 30, 95, 95))
    assert nav.has_unexplored() is True


@pytest.mark.asyncio
async def test_navigation_has_unexplored_false():
    ctx = make_mock_ctx(make_all_explored_minimap())
    nav = NavigationModule(ctx, minimap_region=(1820, 30, 95, 95))
    assert nav.has_unexplored() is False


@pytest.mark.asyncio
async def test_move_toward_calls_hold_key():
    ctx = make_mock_ctx()
    nav = NavigationModule(ctx, minimap_region=(1820, 30, 95, 95))
    # Target to the right of center (cx=47) → should press 'd'
    await nav.move_toward(MapPoint(x=80, y=47))
    ctx.executor.hold_key.assert_called_once()
    key_called = ctx.executor.hold_key.call_args[0][0]
    assert key_called == "d"
