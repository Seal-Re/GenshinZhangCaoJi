# YuanShen Agent Phase 1 — Core Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core game automation foundation — project skeleton, dual-tier vision system (OpenCV + VLM), scene modules (Navigation + Combat + Settlement), and a runnable `ExploreMapSkill` triggered from CLI.

**Architecture:** Three-layer execution model — `Skill Pipeline` composes reusable `Scene Modules`, which use `Action Primitives` (screenshot/keyboard/mouse). Vision has two tiers: fast OpenCV for real-time state, slow VLM for semantic understanding. Config via Pydantic Settings + YAML.

**Tech Stack:** Python 3.11+, langgraph, langchain-anthropic, anthropic, opencv-python, pyautogui, mss, pydantic v2, pydantic-settings, pytest, pytest-asyncio

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata and dependencies |
| `config/settings.yaml` | Default runtime config (LLM, game window, minimap region) |
| `config/loader.py` | Pydantic Settings — load + validate config |
| `skills/_base.py` | `BaseSkill`, `SkillContext`, `SkillResult`, `AgentMessage` |
| `vision/capture.py` | `ScreenCapture` — mss-based screenshot, region crop |
| `vision/executor.py` | `ActionExecutor` — PyAutoGUI keyboard/mouse wrapper |
| `vision/client.py` | `VisionClient` — `fast_detect` (OpenCV) + `understand` (VLM) |
| `vision/providers.py` | `VLMProvider` protocol + `ClaudeVLMProvider` implementation |
| `scene/_base.py` | `BaseSceneModule` — holds `SkillContext` |
| `scene/navigation.py` | `MinimapAnalyzer` + `NavigationModule` |
| `scene/combat.py` | `CombatModule` — enemy detection + macro rotation |
| `scene/settlement.py` | `SettlementModule` — VLM battle-end / reward screen confirmation |
| `data/client.py` | `DataClient` — stub for Phase 1 (Phase 2 adds real API calls) |
| `skills/explore_map.py` | `ExploreMapSkill` — first working Skill |
| `agent/router.py` | `SkillRouter` — maps natural language intent → Skill |
| `agent/core.py` | LangGraph `StateGraph` — main agent loop |
| `main.py` | CLI entry point — `python main.py "开图"` |
| `tests/test_config.py` | Config loading + validation |
| `tests/test_skill_base.py` | BaseSkill interface + SkillContext |
| `tests/test_vision_capture.py` | ScreenCapture region crop (mss mocked) |
| `tests/test_vision_executor.py` | ActionExecutor calls (pyautogui mocked) |
| `tests/test_vision_client.py` | fast_detect with synthetic images |
| `tests/test_minimap_analyzer.py` | MinimapAnalyzer with synthetic minimap images |
| `tests/test_explore_map_skill.py` | ExploreMapSkill state flow (all game I/O mocked) |

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "yuanshen-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "langchain-anthropic>=0.3.0",
    "langgraph>=0.2.0",
    "opencv-python>=4.9.0",
    "pyautogui>=0.9.54",
    "mss>=9.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "numpy>=1.26.0",
    "aiofiles>=23.0.0",
    "apscheduler>=3.10.0",
    "paddlepaddle>=2.6.0",
    "paddleocr>=2.7.0",
    "yt-dlp>=2024.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # optional fallback VLM
GENSHIN_UID=                 # your UID
```

- [ ] **Step 3: Create tests/__init__.py (empty)**

```python
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: All packages install without errors.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml .env.example tests/__init__.py
git commit -m "chore: project setup with dependencies"
```

---

## Task 2: Config System

**Files:**
- Create: `config/__init__.py`
- Create: `config/loader.py`
- Create: `config/settings.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from config.loader import load_config, Settings, GameConfig

def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, Settings)
    assert cfg.llm.default == "claude-sonnet-4-6"

def test_minimap_region_is_four_ints():
    cfg = load_config()
    r = cfg.game.minimap_region
    assert len(r) == 4
    assert all(isinstance(v, int) for v in r)

def test_game_window_title():
    cfg = load_config()
    assert cfg.game.window_title == "原神"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Create config/loader.py**

```python
# config/__init__.py  (empty)
```

```python
# config/loader.py
from __future__ import annotations
from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    default: str = "claude-sonnet-4-6"
    vision_fast: str = "gemini-2.0-flash"
    vision_fallback: str = "gpt-4o"


class GameConfig(BaseModel):
    window_title: str = "原神"
    screenshot_interval_ms: int = 500
    # [left, top, width, height] — calibrate for your resolution
    # Default: 1920x1080, standard UI scale
    minimap_region: tuple[int, int, int, int] = (1820, 30, 95, 95)


class GenshinConfig(BaseModel):
    uid: str = ""
    cookie_path: str = ".secrets/cookie.txt"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )
    llm: LLMConfig = LLMConfig()
    game: GameConfig = GameConfig()
    genshin: GenshinConfig = GenshinConfig()


@lru_cache(maxsize=1)
def load_config() -> Settings:
    return Settings()
```

- [ ] **Step 4: Create config/settings.yaml (reference, not loaded by default)**

```yaml
# config/settings.yaml
# Human-readable reference. Actual config via environment variables or .env
llm:
  default: claude-sonnet-4-6
  vision_fast: gemini-2.0-flash
  vision_fallback: gpt-4o

game:
  window_title: "原神"
  screenshot_interval_ms: 500
  # Adjust minimap_region for your resolution (left, top, width, height)
  minimap_region: [1820, 30, 95, 95]  # default: 1920x1080

genshin:
  uid: ""
  cookie_path: .secrets/cookie.txt
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add config/ tests/test_config.py
git commit -m "feat: config system with Pydantic Settings"
```

---

## Task 3: BaseSkill + Core Types

**Files:**
- Create: `skills/__init__.py`
- Create: `skills/_base.py`
- Create: `tests/test_skill_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_skill_base.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from skills._base import BaseSkill, SkillContext, SkillResult, AgentMessage


class ConcreteSkill(BaseSkill):
    name = "test_skill"
    description = "A test skill"
    triggers = ["test", "testing"]
    schedule = None

    async def run(self, ctx: SkillContext) -> SkillResult:
        await ctx.reply(AgentMessage(text="hello"))
        return SkillResult(success=True, message="done")


@pytest.fixture
def mock_ctx():
    return SkillContext(
        trigger_source="chat",
        params={},
        capture=MagicMock(),
        vision=MagicMock(),
        executor=MagicMock(),
        data=MagicMock(),
        reply=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_skill_run_returns_result(mock_ctx):
    skill = ConcreteSkill()
    result = await skill.run(mock_ctx)
    assert result.success is True
    assert result.message == "done"


@pytest.mark.asyncio
async def test_skill_run_calls_reply(mock_ctx):
    skill = ConcreteSkill()
    await skill.run(mock_ctx)
    mock_ctx.reply.assert_called_once()
    call_arg = mock_ctx.reply.call_args[0][0]
    assert isinstance(call_arg, AgentMessage)
    assert call_arg.text == "hello"


@pytest.mark.asyncio
async def test_skill_on_error_returns_failure(mock_ctx):
    skill = ConcreteSkill()
    result = await skill.on_error(ValueError("oops"))
    assert result.success is False
    assert "oops" in result.message


def test_agent_message_defaults():
    msg = AgentMessage(text="hi")
    assert msg.images == []
    assert msg.skill_name is None
    assert msg.progress is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_skill_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'skills'`

- [ ] **Step 3: Create skills/_base.py**

```python
# skills/__init__.py  (empty)
```

```python
# skills/_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from vision.capture import ScreenCapture
    from vision.client import VisionClient
    from vision.executor import ActionExecutor
    from data.client import DataClient


@dataclass
class AgentMessage:
    text: str
    images: list[bytes] = field(default_factory=list)
    skill_name: str | None = None
    progress: str | None = None


@dataclass
class SkillResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillContext:
    trigger_source: Literal["chat", "scheduler", "agent"]
    params: dict[str, Any]
    capture: "ScreenCapture"
    vision: "VisionClient"
    executor: "ActionExecutor"
    data: "DataClient"
    reply: Callable[[AgentMessage], Awaitable[None]]


class BaseSkill(ABC):
    name: str
    description: str
    triggers: list[str]
    parameters: type[BaseModel] = BaseModel
    schedule: str | None = None

    @abstractmethod
    async def run(self, ctx: SkillContext) -> SkillResult: ...

    async def on_interrupt(self) -> None:
        pass

    async def on_error(self, e: Exception) -> SkillResult:
        return SkillResult(success=False, message=str(e))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_skill_base.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add skills/ tests/test_skill_base.py
git commit -m "feat: BaseSkill, SkillContext, SkillResult types"
```

---

## Task 4: ScreenCapture

**Files:**
- Create: `vision/__init__.py`
- Create: `vision/capture.py`
- Create: `tests/test_vision_capture.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vision_capture.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_vision_capture.py -v
```

Expected: `ModuleNotFoundError: No module named 'vision'`

- [ ] **Step 3: Create vision/capture.py**

```python
# vision/__init__.py  (empty)
```

```python
# vision/capture.py
from __future__ import annotations
from dataclasses import dataclass

import mss
import numpy as np


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height}


class ScreenCapture:
    def __init__(self) -> None:
        self._sct = mss.mss()

    def capture(self) -> np.ndarray:
        """Capture the primary monitor."""
        monitor = self._sct.monitors[1]
        return np.array(self._sct.grab(monitor))

    def capture_region(self, region: Region) -> np.ndarray:
        """Capture a specific screen region."""
        return np.array(self._sct.grab(region.to_dict()))

    def capture_minimap(self, minimap_region: tuple[int, int, int, int]) -> np.ndarray:
        """Convenience method: capture minimap from (left, top, width, height) tuple."""
        left, top, width, height = minimap_region
        return self.capture_region(Region(left, top, width, height))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_vision_capture.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vision/__init__.py vision/capture.py tests/test_vision_capture.py
git commit -m "feat: ScreenCapture with region support"
```

---

## Task 5: ActionExecutor

**Files:**
- Create: `vision/executor.py`
- Create: `tests/test_vision_executor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vision_executor.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_vision_executor.py -v
```

Expected: `ModuleNotFoundError: No module named 'vision.executor'`

- [ ] **Step 3: Create vision/executor.py**

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_vision_executor.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vision/executor.py tests/test_vision_executor.py
git commit -m "feat: ActionExecutor keyboard/mouse wrapper"
```

---

## Task 6: VisionClient (CV tier + VLM tier)

**Files:**
- Create: `vision/providers.py`
- Create: `vision/client.py`
- Create: `tests/test_vision_client.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_vision_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'vision.client'`

- [ ] **Step 3: Create vision/providers.py**

```python
# vision/providers.py
from __future__ import annotations
import base64
from typing import Protocol, runtime_checkable

import anthropic


@runtime_checkable
class VLMProvider(Protocol):
    async def analyze(self, image_bytes: bytes, prompt: str) -> str: ...


class ClaudeVLMProvider:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def analyze(self, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.standard_b64encode(image_bytes).decode()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text
```

- [ ] **Step 4: Create vision/client.py**

```python
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

        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            return Detection(found=True, location=max_loc, confidence=float(max_val))
        return Detection(found=False, confidence=float(max_val))

    async def understand(self, screenshot: np.ndarray, prompt: str) -> str:
        """VLM semantic analysis — 1-3s, for state/phase understanding."""
        _, buf = cv2.imencode(".png", screenshot)
        return await self._vlm.analyze(buf.tobytes(), prompt)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/test_vision_client.py -v
```

Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add vision/providers.py vision/client.py tests/test_vision_client.py
git commit -m "feat: VisionClient with OpenCV fast_detect and VLM understand"
```

---

## Task 7: DataClient Stub + Scene Base

**Files:**
- Create: `data/__init__.py`
- Create: `data/client.py`
- Create: `scene/__init__.py`
- Create: `scene/_base.py`

- [ ] **Step 1: Create data/client.py (stub)**

```python
# data/__init__.py  (empty)
```

```python
# data/client.py
"""
DataClient: unified access to Genshin data APIs.
Phase 1 stub — returns empty results.
Phase 2 will implement: genshin.py (account), enka-py (builds), genshin-db (wiki).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PlayerStats:
    uid: str = ""
    nickname: str = ""
    level: int = 0


class DataClient:
    async def get_player_stats(self) -> PlayerStats:
        """Phase 1 stub."""
        return PlayerStats()

    async def get_character_info(self, name: str) -> dict:
        """Phase 1 stub."""
        return {}
```

- [ ] **Step 2: Create scene/_base.py**

```python
# scene/__init__.py  (empty)
```

```python
# scene/_base.py
from __future__ import annotations
from skills._base import SkillContext


class BaseSceneModule:
    def __init__(self, ctx: SkillContext) -> None:
        self.ctx = ctx
```

- [ ] **Step 3: Run existing tests to confirm nothing broken**

```bash
pytest tests/ -v
```

Expected: All prior tests PASS

- [ ] **Step 4: Commit**

```bash
git add data/ scene/__init__.py scene/_base.py
git commit -m "feat: DataClient stub and BaseSceneModule"
```

---

## Task 8: MinimapAnalyzer

**Files:**
- Create: `scene/navigation.py`
- Create: `tests/test_minimap_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_minimap_analyzer.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_minimap_analyzer.py -v
```

Expected: `ModuleNotFoundError: No module named 'scene.navigation'`

- [ ] **Step 3: Create scene/navigation.py (MinimapAnalyzer only)**

```python
# scene/navigation.py
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from scene._base import BaseSceneModule
from skills._base import SkillContext


@dataclass
class MapPoint:
    x: int
    y: int


class MinimapAnalyzer:
    # Unexplored areas: gray pixels — low saturation in HSV
    _GRAY_LOWER = np.array([0, 0, 40], dtype=np.uint8)
    _GRAY_UPPER = np.array([180, 50, 210], dtype=np.uint8)
    _MIN_AREA = 15  # ignore tiny noise patches

    def get_unexplored_areas(self, minimap_img: np.ndarray) -> list[MapPoint]:
        """Return centroids of gray (unexplored) regions on the minimap."""
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._GRAY_LOWER, self._GRAY_UPPER)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        points: list[MapPoint] = []
        for c in contours:
            if cv2.contourArea(c) < self._MIN_AREA:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append(MapPoint(cx, cy))
        return points

    def get_player_position(self, minimap_img: np.ndarray) -> MapPoint:
        """Player icon is always near the center of the minimap."""
        h, w = minimap_img.shape[:2]
        return MapPoint(w // 2, h // 2)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_minimap_analyzer.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scene/navigation.py tests/test_minimap_analyzer.py
git commit -m "feat: MinimapAnalyzer — unexplored area detection via HSV"
```

---

## Task 9: NavigationModule

**Files:**
- Modify: `scene/navigation.py` (add `NavigationModule`)

- [ ] **Step 1: Add NavigationModule tests to test_minimap_analyzer.py**

```python
# Append to tests/test_minimap_analyzer.py

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from scene.navigation import NavigationModule, MapPoint
from skills._base import SkillContext


def make_mock_ctx(minimap_img=None):
    if minimap_img is None:
        minimap_img = make_minimap_with_gray_patch()
    ctx = MagicMock(spec=SkillContext)
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
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
pytest tests/test_minimap_analyzer.py -v -k "navigation"
```

Expected: `AttributeError` or `ImportError` for `NavigationModule`

- [ ] **Step 3: Add NavigationModule to scene/navigation.py**

```python
# Append to scene/navigation.py (after MinimapAnalyzer class)

class NavigationModule(BaseSceneModule):
    def __init__(
        self,
        ctx: SkillContext,
        minimap_region: tuple[int, int, int, int],
    ) -> None:
        super().__init__(ctx)
        self.analyzer = MinimapAnalyzer()
        self._minimap_region = minimap_region

    def has_unexplored(self) -> bool:
        minimap = self.ctx.capture.capture_minimap(self._minimap_region)
        return len(self.analyzer.get_unexplored_areas(minimap)) > 0

    def get_next_target(self) -> MapPoint | None:
        minimap = self.ctx.capture.capture_minimap(self._minimap_region)
        targets = self.analyzer.get_unexplored_areas(minimap)
        if not targets:
            return None
        # Pick the closest target to player (player is at center of minimap)
        player = self.analyzer.get_player_position(minimap)
        targets.sort(
            key=lambda p: (p.x - player.x) ** 2 + (p.y - player.y) ** 2
        )
        return targets[0]

    async def move_toward(self, target: MapPoint) -> None:
        """Press directional key toward target for 0.5s."""
        h, w = 95, 95  # minimap dimensions
        player = MapPoint(w // 2, h // 2)
        dx = target.x - player.x
        dy = target.y - player.y

        if abs(dx) >= abs(dy):
            key = "d" if dx > 0 else "a"
        else:
            key = "s" if dy > 0 else "w"

        await self.ctx.executor.hold_key(key, 0.5)
```

- [ ] **Step 4: Run all navigation tests**

```bash
pytest tests/test_minimap_analyzer.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scene/navigation.py tests/test_minimap_analyzer.py
git commit -m "feat: NavigationModule — minimap-driven movement"
```

---

## Task 10: CombatModule

**Files:**
- Create: `scene/combat.py`
- Create: `tests/test_combat_module.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_combat_module.py
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock
from scene.combat import CombatModule, TeamConfig
from skills._base import SkillContext


def make_frame_with_red_hp_bar():
    """Frame with red pixels (simulating enemy HP bar)."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # BGR: red = (0, 0, 200) — but in HSV: red hue at 0 or 180
    # Use a clearly red color
    frame[50:60, 200:400, 0] = 0    # B
    frame[50:60, 200:400, 1] = 0    # G
    frame[50:60, 200:400, 2] = 200  # R
    return frame


def make_empty_frame():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def make_mock_ctx():
    ctx = MagicMock(spec=SkillContext)
    ctx.executor = AsyncMock()
    ctx.vision = AsyncMock()
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_combat_module.py -v
```

Expected: `ModuleNotFoundError: No module named 'scene.combat'`

- [ ] **Step 3: Create scene/combat.py**

```python
# scene/combat.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field

import cv2
import numpy as np

from scene._base import BaseSceneModule
from skills._base import SkillContext


@dataclass
class TeamConfig:
    # Each action: {"key": str, "delay": float (seconds after press)}
    rotation: list[dict[str, float | str]] = field(default_factory=list)


class CombatModule(BaseSceneModule):
    # Red HP bar detection: HSV hue 0-10 or 170-180, high saturation
    _RED_LOWER_1 = np.array([0, 120, 70], dtype=np.uint8)
    _RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
    _RED_LOWER_2 = np.array([170, 120, 70], dtype=np.uint8)
    _RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)
    _RED_PIXEL_THRESHOLD = 300

    def __init__(self, ctx: SkillContext, team_config: TeamConfig) -> None:
        super().__init__(ctx)
        self.team_config = team_config

    def detect_enemy(self, frame: np.ndarray) -> bool:
        """Detect enemy presence via red HP bar pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self._RED_LOWER_1, self._RED_UPPER_1)
        mask2 = cv2.inRange(hsv, self._RED_LOWER_2, self._RED_UPPER_2)
        combined = cv2.bitwise_or(mask1, mask2)
        return int(cv2.countNonZero(combined)) > self._RED_PIXEL_THRESHOLD

    async def execute_rotation(self) -> None:
        """Execute one full skill rotation cycle."""
        for action in self.team_config.rotation:
            await self.ctx.executor.key_press(str(action["key"]))
            await asyncio.sleep(float(action["delay"]))

    async def clear_enemies(self) -> None:
        """
        Run rotation loop until no enemies detected or settlement screen appears.
        Uses VLM only to detect final settlement — not for per-frame decisions.
        """
        max_iterations = 30
        for _ in range(max_iterations):
            frame = self.ctx.capture.capture()
            if not self.detect_enemy(frame):
                # Double-check: VLM confirms battle is over
                settlement = await self.ctx.vision.understand(
                    frame,
                    "Is there a battle reward/settlement screen visible? Answer only yes or no.",
                )
                if "yes" in settlement.lower():
                    # Dismiss settlement
                    await self.ctx.executor.key_press("f")
                break
            await self.execute_rotation()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_combat_module.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scene/combat.py tests/test_combat_module.py
git commit -m "feat: CombatModule — enemy detection + macro rotation"
```

---

## Task 11: SettlementModule

**Files:**
- Create: `scene/settlement.py`

- [ ] **Step 1: Create scene/settlement.py**

```python
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
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add scene/settlement.py
git commit -m "feat: SettlementModule — VLM settlement screen confirmation"
```

---

## Task 12: ExploreMapSkill

**Files:**
- Create: `skills/explore_map.py`
- Create: `tests/test_explore_map_skill.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_explore_map_skill.py
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from skills.explore_map import ExploreMapSkill
from skills._base import SkillContext, SkillResult, AgentMessage
from scene.navigation import MapPoint


def make_mock_ctx(has_unexplored_sequence=None):
    """
    has_unexplored_sequence: list of bool values returned on successive calls.
    e.g. [True, True, False] = two moves then done.
    """
    ctx = MagicMock(spec=SkillContext)
    ctx.params = {"region": "测试区域"}
    ctx.trigger_source = "chat"
    ctx.executor = AsyncMock()
    ctx.reply = AsyncMock()

    # Minimap: first two calls return gray patch, third call returns empty
    gray_minimap = np.zeros((95, 95, 3), dtype=np.uint8)
    gray_minimap[:, :] = [30, 120, 30]
    gray_minimap[60:80, 60:80] = [100, 100, 100]

    explored_minimap = np.zeros((95, 95, 3), dtype=np.uint8)
    explored_minimap[:, :] = [30, 120, 30]

    if has_unexplored_sequence is None:
        has_unexplored_sequence = [gray_minimap, gray_minimap, explored_minimap]

    ctx.capture.capture_minimap.side_effect = has_unexplored_sequence
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
    # First reply should announce start
    first_call = ctx.reply.call_args_list[0][0][0]
    assert isinstance(first_call, AgentMessage)
    assert "开始探索" in first_call.text


@pytest.mark.asyncio
async def test_explore_map_moves_toward_unexplored():
    ctx = make_mock_ctx()
    skill = ExploreMapSkill()
    await skill.run(ctx)
    # Should have called hold_key at least once (movement)
    assert ctx.executor.hold_key.call_count >= 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_explore_map_skill.py -v
```

Expected: `ModuleNotFoundError: No module named 'skills.explore_map'`

- [ ] **Step 3: Create skills/explore_map.py**

```python
# skills/explore_map.py
from __future__ import annotations

from pydantic import BaseModel

from scene.combat import CombatModule, TeamConfig
from scene.navigation import NavigationModule
from scene.settlement import SettlementModule
from skills._base import AgentMessage, BaseSkill, SkillContext, SkillResult
from config.loader import load_config


class ExploreMapParams(BaseModel):
    region: str = "当前区域"
    max_steps: int = 200


class ExploreMapSkill(BaseSkill):
    name = "explore_map"
    description = "在原神游戏地图上自动探索指定区域，开图、收集宝箱、解锁传送锚点"
    triggers = ["开图", "探索地图", "开蒙德", "开璃月", "开稻妻", "explore map"]
    parameters = ExploreMapParams
    schedule = None

    # Default team rotation — override via config in future
    _DEFAULT_ROTATION = [
        {"key": "1", "delay": 0.3},
        {"key": "e", "delay": 0.8},
        {"key": "q", "delay": 0.3},
    ]

    async def run(self, ctx: SkillContext) -> SkillResult:
        cfg = load_config()
        region = ctx.params.get("region", "当前区域")
        max_steps = ctx.params.get("max_steps", 200)

        nav = NavigationModule(ctx, minimap_region=cfg.game.minimap_region)
        combat = CombatModule(ctx, TeamConfig(rotation=self._DEFAULT_ROTATION))
        settlement = SettlementModule(ctx)

        await ctx.reply(AgentMessage(
            text=f"开始探索 {region}...",
            skill_name=self.name,
        ))

        step = 0
        while step < max_steps and nav.has_unexplored():
            target = nav.get_next_target()
            if target is None:
                break

            await nav.move_toward(target)

            # Handle combat if enemies present
            frame = ctx.capture.capture()
            if combat.detect_enemy(frame):
                await combat.clear_enemies()
                await settlement.confirm_and_close()

            step += 1
            if step % 20 == 0:
                await ctx.reply(AgentMessage(
                    text=f"探索中 ({step}/{max_steps} 步)",
                    skill_name=self.name,
                    progress=f"步数: {step}",
                ))

        return SkillResult(
            success=True,
            message=f"{region} 探索完成，共执行 {step} 步",
            data={"steps": step, "region": region},
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_explore_map_skill.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add skills/explore_map.py tests/test_explore_map_skill.py
git commit -m "feat: ExploreMapSkill — minimap-driven autonomous exploration"
```

---

## Task 13: Skill Router + Agent Core

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/router.py`
- Create: `agent/core.py`

- [ ] **Step 1: Create agent/router.py**

```python
# agent/__init__.py  (empty)
```

```python
# agent/router.py
from __future__ import annotations

from skills._base import BaseSkill


class SkillRouter:
    def __init__(self, skills: list[BaseSkill]) -> None:
        self._skills = {s.name: s for s in skills}
        self._trigger_map: dict[str, str] = {}
        for skill in skills:
            for trigger in skill.triggers:
                self._trigger_map[trigger.lower()] = skill.name

    def match(self, user_input: str) -> BaseSkill | None:
        """Find skill by trigger keyword or name. Returns None if no match."""
        lower = user_input.lower()
        # Direct name match
        if lower in self._skills:
            return self._skills[lower]
        # Trigger keyword match
        for trigger, skill_name in self._trigger_map.items():
            if trigger in lower:
                return self._skills[skill_name]
        return None

    @property
    def all_skills(self) -> list[BaseSkill]:
        return list(self._skills.values())
```

- [ ] **Step 2: Create agent/core.py**

```python
# agent/core.py
from __future__ import annotations
import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from agent.router import SkillRouter
from config.loader import load_config
from data.client import DataClient
from skills._base import AgentMessage, SkillContext
from skills.explore_map import ExploreMapSkill
from vision.capture import ScreenCapture
from vision.client import VisionClient
from vision.executor import ActionExecutor
from vision.providers import ClaudeVLMProvider


class AgentState(TypedDict):
    user_input: str
    skill_name: str | None
    skill_params: dict
    result_message: str
    reply_messages: list[AgentMessage]


def build_agent() -> tuple[object, SkillRouter]:
    cfg = load_config()

    # Wire up infrastructure
    capture = ScreenCapture()
    executor = ActionExecutor()
    vlm = ClaudeVLMProvider(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=cfg.llm.default,
    )
    vision = VisionClient(vlm_provider=vlm)
    data = DataClient()

    # Register skills
    skills = [ExploreMapSkill()]
    router = SkillRouter(skills)

    async def _route_node(state: AgentState) -> AgentState:
        skill = router.match(state["user_input"])
        if skill is None:
            return {**state, "skill_name": None,
                    "result_message": "未找到匹配的技能，请尝试: " +
                    ", ".join(s.triggers[0] for s in router.all_skills)}
        return {**state, "skill_name": skill.name}

    async def _execute_node(state: AgentState) -> AgentState:
        if state["skill_name"] is None:
            return state
        skill = router._skills[state["skill_name"]]
        replies: list[AgentMessage] = []

        async def collect_reply(msg: AgentMessage) -> None:
            replies.append(msg)

        ctx = SkillContext(
            trigger_source="chat",
            params=state["skill_params"],
            capture=capture,
            vision=vision,
            executor=executor,
            data=data,
            reply=collect_reply,
        )
        result = await skill.run(ctx)
        return {**state, "result_message": result.message,
                "reply_messages": replies}

    graph = StateGraph(AgentState)
    graph.add_node("route", _route_node)
    graph.add_node("execute", _execute_node)
    graph.set_entry_point("route")
    graph.add_edge("route", "execute")
    graph.add_edge("execute", END)

    return graph.compile(), router
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add agent/ 
git commit -m "feat: SkillRouter and LangGraph agent core"
```

---

## Task 14: CLI Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Create main.py**

```python
# main.py
"""
CLI entry point for YuanShen Agent — Phase 1.

Usage:
    python main.py "开图"
    python main.py "探索蒙德地图"
    python main.py --list-skills
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys


async def run_cli(user_input: str) -> None:
    from agent.core import build_agent
    from skills._base import AgentMessage

    print(f"[Agent] 收到指令: {user_input}")
    agent, router = build_agent()

    collected: list[AgentMessage] = []

    result = await agent.ainvoke({
        "user_input": user_input,
        "skill_name": None,
        "skill_params": {},
        "result_message": "",
        "reply_messages": [],
    })

    for msg in result.get("reply_messages", []):
        prefix = f"[{msg.skill_name}]" if msg.skill_name else "[Agent]"
        if msg.progress:
            print(f"{prefix} {msg.text} ({msg.progress})")
        else:
            print(f"{prefix} {msg.text}")

    print(f"[完成] {result['result_message']}")


def list_skills() -> None:
    from skills.explore_map import ExploreMapSkill
    skills = [ExploreMapSkill()]
    print("已注册技能:")
    for s in skills:
        triggers = " / ".join(s.triggers[:3])
        print(f"  {s.name}: {s.description[:50]}...")
        print(f"    触发词: {triggers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YuanShen Agent CLI")
    parser.add_argument("input", nargs="?", help="Natural language command")
    parser.add_argument("--list-skills", action="store_true")
    args = parser.parse_args()

    if args.list_skills:
        list_skills()
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("错误: 请设置 ANTHROPIC_API_KEY 环境变量")
        sys.exit(1)

    asyncio.run(run_cli(args.input))
```

- [ ] **Step 2: Verify CLI help works (no game needed)**

```bash
python main.py --list-skills
```

Expected output:
```
已注册技能:
  explore_map: 在原神游戏地图上自动探索指定区域...
    触发词: 开图 / 探索地图 / 开蒙德
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS, 0 failures

- [ ] **Step 4: Final commit**

```bash
git add main.py
git commit -m "feat: CLI entry point — python main.py '开图'"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - BaseSkill + SkillContext + types → Task 3
  - VisionClient dual-tier (OpenCV + VLM) → Tasks 6-7
  - ActionExecutor (PyAutoGUI) → Task 5
  - ScreenCapture (mss) → Task 4
  - MinimapAnalyzer + NavigationModule → Tasks 8-9
  - CombatModule (CV + Macro) → Task 10
  - SettlementModule (VLM) → Task 11
  - ExploreMapSkill (working Phase 1 skill) → Task 12
  - LangGraph Agent Core + Router → Task 13
  - CLI entry point → Task 14
  - Config system → Task 2

- [x] **Phase 2 items NOT in this plan (by design):**
  - DialogueModule / auto_dialogue
  - DailyCommissionSkill
  - APScheduler scheduling
  - Gradio Web UI
  - VideoRouteLearner
  - NoneBot2 QQ Bot
  - Data Layer (genshin.py / enka-py) — DataClient is a stub in Phase 1

- [x] **Type consistency:** `MapPoint` used in NavigationModule tests and ExploreMapSkill → defined in `scene/navigation.py`; `SkillContext.capture` vs `ctx.capture` — consistent throughout; `TeamConfig` defined in `scene/combat.py` and used in `ExploreMapSkill._DEFAULT_ROTATION` — consistent.

- [x] **No placeholders:** All steps have concrete code.

---

## Phase 2 Plan Preview

Once Phase 1 is working:
- `scene/dialogue.py` — DialogueModule (OCR with PaddleOCR + LLM summary)
- `skills/daily_commission.py` — DailyCommissionSkill
- `skills/auto_dialogue.py` — auto_dialogue Skill
- `agent/scheduler.py` — APScheduler integration
- `interfaces/web_ui.py` — Gradio UI (chat + schedule panel + screenshot stream)
