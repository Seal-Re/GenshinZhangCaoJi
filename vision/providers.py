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
