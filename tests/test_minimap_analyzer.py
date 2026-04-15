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
