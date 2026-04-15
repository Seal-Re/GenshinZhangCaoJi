import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock
from skills.explore_map import ExploreMapSkill
from skills._base import SkillContext, SkillResult, AgentMessage
from scene.navigation import MapPoint


def make_gray_minimap():
    """Minimap with unexplored gray region."""
    img = np.zeros((95, 95, 3), dtype=np.uint8)
    img[:, :] = [30, 120, 30]      # green explored background
    img[60:80, 60:80] = [100, 100, 100]  # gray unexplored patch
    return img


def make_explored_minimap():
    """Minimap fully explored."""
    img = np.zeros((95, 95, 3), dtype=np.uint8)
    img[:, :] = [30, 120, 30]
    return img


def make_mock_ctx(minimap_sequence=None):
    """
    minimap_sequence: list of ndarrays returned by successive capture_minimap calls.
    Default: [gray, gray, explored] — two moves then done.
    """
    if minimap_sequence is None:
        minimap_sequence = [make_gray_minimap(), make_gray_minimap(), make_explored_minimap()]

    ctx = MagicMock()
    ctx.params = {"region": "测试区域"}
    ctx.trigger_source = "chat"
    ctx.executor = AsyncMock()
    ctx.reply = AsyncMock()
    ctx.capture.capture_minimap.side_effect = minimap_sequence
    ctx.capture.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx.vision.understand = AsyncMock(return_value="no")
    ctx.data = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_explore_map_returns_success():
    ctx = make_mock_ctx()
    skill = ExploreMapSkill()
    result = await skill.run(ctx)
    assert isinstance(result, SkillResult)
    assert result.success is True


@pytest.mark.asyncio
async def test_explore_map_sends_start_reply():
    ctx = make_mock_ctx()
    skill = ExploreMapSkill()
    await skill.run(ctx)
    first_call = ctx.reply.call_args_list[0][0][0]
    assert isinstance(first_call, AgentMessage)
    assert "开始探索" in first_call.text


@pytest.mark.asyncio
async def test_explore_map_moves_toward_unexplored():
    ctx = make_mock_ctx()
    skill = ExploreMapSkill()
    await skill.run(ctx)
    # Should have called hold_key at least once (directional movement)
    assert ctx.executor.hold_key.call_count >= 1
