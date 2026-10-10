# GenshinZhangCaoJi

A Phase 1 prototype of a Genshin Impact automation agent: screenshot capture, OpenCV heuristics and a vision LLM drive keyboard/mouse actions through a small LangGraph state machine.

Status: prototype / Phase 1. Only one skill is registered, and the account-data client is a stub. Trigger words and console output are in Chinese. Windows only (the game is; `pyautogui` and `mss` target a local desktop).

## What is implemented

**Agent graph.** `agent/core.py` builds a `StateGraph(AgentState)` with two nodes — `route` and `execute` — wired `route -> execute -> END`. `route` asks `SkillRouter` to match the user input against skill trigger words; `execute` builds a `SkillContext` (capture, vision, executor, data, reply callback) and awaits the skill. This is a single pass, not a perceive/decide/act loop.

**Skill router.** `agent/router.py` matches by skill name or by substring of any trigger word. One skill is registered in `build_agent()`: `ExploreMapSkill`, whose triggers are `开图`, `探索地图`, `开蒙德`, `开璃月`, `开稻妻` and `explore map`. Unmatched input returns a "no skill found" message.

**Explore loop.** `skills/explore_map.py` runs up to 200 steps: read the minimap region, ask the navigation module for the nearest unexplored point, hold the corresponding WASD key for 0.5 s, capture a full frame and — if the combat module sees enemies — run the rotation and then check the settlement screen. Progress is reported every 20 steps.

**Scene modules** (`scene/`):

- `navigation.py` — `MinimapAnalyzer` thresholds the minimap in HSV for low-saturation grey pixels, takes contour centroids above 15 px as unexplored areas, and treats the minimap centre as the player. `NavigationModule.move_toward` compares the target offset against the centre and picks one of W/A/S/D.
- `combat.py` — `detect_enemy` counts red pixels (two HSV hue bands, 300-pixel threshold) to decide whether an enemy HP bar is on screen. `clear_enemies` presses the fixed rotation `1`, `E`, `Q` with per-key delays until the red-pixel test fails or 30 iterations elapse, then asks the VLM whether a reward screen is visible.
- `settlement.py` — asks the VLM a single yes/no question about a settlement/reward screen and presses `F` if it answers yes.

**Vision** (`vision/`): `capture.py` wraps `mss` for full-screen or region grabs; `client.py` exposes `fast_detect` (OpenCV `matchTemplate`, threshold 0.8) and `understand` (PNG-encodes the frame and asks the provider); `executor.py` wraps `pyautogui` clicks, key presses, held keys, moves and scrolls, with `FAILSAFE` on.

**Providers.** `vision/providers.py` has two implementations of the `VLMProvider` protocol: `ClaudeVLMProvider` (anthropic SDK, base64 image block) and `OpenAIVLMProvider` (any OpenAI-compatible `chat/completions` endpoint, `image_url` data URI). `_build_vlm_provider` picks one at startup from `llm.provider` (`claude` or `openai`) and never switches afterwards. There is no Gemini provider and no timeout/quota fallback between providers.

**Config.** `config/loader.py` is a pydantic-settings `Settings` model reading `.env` with `env_nested_delimiter="__"`. `config/settings.yaml` is a human-readable reference only — the loader never reads it. Defaults: minimap region `(1820, 30, 95, 95)`, window title `原神`, 500 ms screenshot interval.

**Tests.** 29 tests in 8 modules (`test_combat_module`, `test_config`, `test_explore_map_skill`, `test_minimap_analyzer`, `test_skill_base`, `test_vision_capture`, `test_vision_client`, `test_vision_executor`). Capture, vision and executor are constructor-injected, so the suite runs without a game or a display.

## Getting started

```bash
python -m venv .venv && . .venv/bin/activate

# pyproject.toml declares the dependencies, but the hatchling build config has no
# wheel target, so `pip install -e .` cannot select files. Install them directly:
pip install anthropic openai langchain-anthropic langgraph opencv-python pyautogui mss \
            pydantic pydantic-settings numpy aiofiles pytest pytest-asyncio pytest-mock

cp .env.example .env
# ANTHROPIC_API_KEY is required by main.py regardless of which provider is selected.
# The loader uses nested env names, so the OpenAI path needs OPENAI__API_KEY,
# OPENAI__MODEL_ID and OPENAI__BASE_URL (the flat names in .env.example are not read).

python main.py --list-skills     # prints registered skills and trigger words
python main.py '开图'            # run exploration (with the game window open)
```

The minimap bounds default to 1920x1080; on another resolution adjust `minimap_region` in the environment or in `config/settings.yaml` for reference.

## Project layout

```
GenshinZhangCaoJi/
├── main.py                  # CLI entry, asyncio.run(run_cli(...))
├── agent/
│   ├── core.py              # build_agent(): StateGraph + wiring
│   └── router.py            # SkillRouter
├── skills/
│   ├── _base.py             # BaseSkill, SkillContext, SkillResult, AgentMessage
│   └── explore_map.py       # ExploreMapSkill (the only registered skill)
├── scene/
│   ├── _base.py
│   ├── navigation.py        # MinimapAnalyzer + NavigationModule
│   ├── combat.py            # CombatModule
│   └── settlement.py        # SettlementModule
├── vision/
│   ├── capture.py           # ScreenCapture (mss)
│   ├── client.py            # VisionClient: fast_detect + understand
│   ├── executor.py          # ActionExecutor (pyautogui)
│   └── providers.py         # ClaudeVLMProvider, OpenAIVLMProvider
├── config/
│   ├── loader.py            # pydantic-settings
│   └── settings.yaml        # reference values, not loaded
├── data/client.py           # DataClient — stub
└── tests/                   # 8 modules, injected mocks
```

## Limitations

- Phase 1 only. `data/client.py` returns empty `PlayerStats` and `{}`; the planned account and wiki integrations are not written.
- The OpenCV thresholds (grey unexplored areas, red HP bars, template match at 0.8) are uncalibrated outside the 1920x1080 default, and the combat rotation is a hardcoded `1 / E / Q` sequence.
- One skill. Daily commissions, domains, abyss and the video-route learning described in `docs/superpowers/specs/` are not implemented.
- The design spec lists Claude, GPT-4o and Gemini with per-scene routing; only Claude and OpenAI-compatible providers exist, with no fallback.
- `fast_detect` is exercised by tests but is not called anywhere in the gameplay path.

## License

MIT. See `LICENSE`.
