import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from vision.capture import ScreenCapture, Region


@pytest.fixture
def mock_sct():
    fake_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    fake_frame[100:200, 300:400] = 255  # white region for testing
    mock = MagicMock()
    mock.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
    mock.grab.return_value = fake_frame
    return mock


def test_capture_returns_ndarray(mock_sct):
    with patch("vision.capture.mss.mss", return_value=mock_sct):
        cap = ScreenCapture()
        frame = cap.capture()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3


def test_capture_region_uses_provided_region(mock_sct):
    with patch("vision.capture.mss.mss", return_value=mock_sct):
        cap = ScreenCapture()
        region = Region(left=1820, top=30, width=95, height=95)
        cap.capture_region(region)
        call_args = mock_sct.grab.call_args[0][0]
        assert call_args["left"] == 1820
        assert call_args["top"] == 30
        assert call_args["width"] == 95
        assert call_args["height"] == 95


def test_capture_minimap_from_config(mock_sct):
    with patch("vision.capture.mss.mss", return_value=mock_sct):
        cap = ScreenCapture()
        region = Region(left=1820, top=30, width=95, height=95)
        result = cap.capture_region(region)
        assert isinstance(result, np.ndarray)
