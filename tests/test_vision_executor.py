import pytest
from unittest.mock import patch, AsyncMock, call
from vision.executor import ActionExecutor, Point


@pytest.fixture
def executor():
    return ActionExecutor()


@pytest.mark.asyncio
async def test_click_calls_pyautogui(executor):
    with patch("vision.executor.pyautogui.click") as mock_click:
        await executor.click(Point(100, 200))
        mock_click.assert_called_once_with(100, 200, button="left")


@pytest.mark.asyncio
async def test_click_right_button(executor):
    with patch("vision.executor.pyautogui.click") as mock_click:
        await executor.click(Point(50, 60), button="right")
        mock_click.assert_called_once_with(50, 60, button="right")


@pytest.mark.asyncio
async def test_key_press_calls_down_and_up(executor):
    with patch("vision.executor.pyautogui.keyDown") as mock_down, \
         patch("vision.executor.pyautogui.keyUp") as mock_up:
        await executor.key_press("f")
        mock_down.assert_called_once_with("f")
        mock_up.assert_called_once_with("f")


@pytest.mark.asyncio
async def test_hold_key_duration(executor):
    with patch("vision.executor.pyautogui.keyDown") as mock_down, \
         patch("vision.executor.pyautogui.keyUp") as mock_up, \
         patch("asyncio.sleep") as mock_sleep:
        await executor.hold_key("w", duration=1.5)
        mock_down.assert_called_once_with("w")
        mock_sleep.assert_awaited_once_with(1.5)
        mock_up.assert_called_once_with("w")
