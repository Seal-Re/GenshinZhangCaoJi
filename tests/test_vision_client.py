# tests/test_vision_client.py
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from vision.client import VisionClient, Detection
from vision.providers import VLMProvider


def make_template_in_frame():
    """Create a frame with a known template embedded at (50, 80)."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    template = np.full((20, 30, 3), 200, dtype=np.uint8)
    frame[80:100, 50:80] = template
    return frame, template


def test_fast_detect_finds_template():
    client = VisionClient(vlm_provider=MagicMock())
    frame, template = make_template_in_frame()
    result = client.fast_detect(frame, template)
    assert result.found is True
    assert result.confidence > 0.9
    # location is (x, y) = (col, row) = (50, 80)
    assert result.location == (50, 80)


def test_fast_detect_returns_not_found():
    client = VisionClient(vlm_provider=MagicMock())
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    template = np.full((20, 30, 3), 200, dtype=np.uint8)
    result = client.fast_detect(frame, template, threshold=0.99)
    assert result.found is False
    assert result.location is None


@pytest.mark.asyncio
async def test_understand_calls_vlm_provider():
    mock_provider = AsyncMock(spec=VLMProvider)
    mock_provider.analyze.return_value = "yes"
    client = VisionClient(vlm_provider=mock_provider)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = await client.understand(frame, "Is there a reward screen?")

    assert result == "yes"
    mock_provider.analyze.assert_called_once()
    call_args = mock_provider.analyze.call_args
    assert isinstance(call_args[0][0], bytes)      # image_bytes
    assert "reward screen" in call_args[0][1]      # prompt
