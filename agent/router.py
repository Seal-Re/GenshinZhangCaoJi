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
