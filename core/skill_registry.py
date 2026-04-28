"""
core/skill_registry.py — Auto-registration and dispatch for all Jarvis2 skills.

Usage:
    registry = SkillRegistry(api_key)
    tool_declarations = registry.get_tool_declarations()
    result = registry.execute("web_search", {"query": "Python 3.13 news"})
"""
from __future__ import annotations

import traceback
from typing import Any

from skills.base import Skill
from skills.open_app import OpenAppSkill
from skills.web_search import WebSearchSkill
from skills.youtube_video import YouTubeSkill
from skills.browser_control import BrowserControlSkill
from skills.tech_researcher import TechResearcherSkill
from skills.music_player import MusicPlayerSkill


class SkillRegistry:
    """Central registry that holds all skills and handles dispatch."""

    def __init__(self, api_key: str) -> None:
        self._skills: dict[str, Skill] = {}
        self._register_all(api_key)

    def _register_all(self, api_key: str) -> None:
        """Instantiate and register every skill by its tool name."""
        skill_classes = [
            OpenAppSkill,
            WebSearchSkill,
            YouTubeSkill,
            BrowserControlSkill,
            TechResearcherSkill,
            MusicPlayerSkill,
        ]
        for cls in skill_classes:
            instance = cls(api_key=api_key)
            name = cls.TOOL_DECLARATION.get("name")
            if not name:
                continue
            self._skills[name] = instance

    def get_tool_declarations(self) -> list[dict[str, Any]]:
        """Return all Gemini function declarations for the Live session config."""
        return [skill.TOOL_DECLARATION for skill in self._skills.values()]

    def execute(self, tool_name: str, params: dict) -> str:
        """
        Dispatch a tool call to the correct skill.

        Returns a string result (or an error message).
        """
        skill = self._skills.get(tool_name)
        if skill is None:
            return f"Unknown tool: '{tool_name}'"

        try:
            result = skill.execute(params)
            return result or "Done."
        except Exception as exc:
            traceback.print_exc()
            return f"Skill '{tool_name}' failed: {exc}"
