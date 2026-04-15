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
