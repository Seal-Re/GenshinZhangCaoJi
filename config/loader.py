from __future__ import annotations
from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    provider: str = "claude"          # "claude" | "openai"
    default: str = "claude-sonnet-4-6"
    vision_fast: str = "gemini-2.0-flash"
    vision_fallback: str = "gpt-4o"


class OpenAIConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_id: str = "gpt-4o"


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
    openai: OpenAIConfig = OpenAIConfig()
    game: GameConfig = GameConfig()
    genshin: GenshinConfig = GenshinConfig()


@lru_cache(maxsize=1)
def load_config() -> Settings:
    return Settings()
