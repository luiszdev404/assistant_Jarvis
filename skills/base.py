"""
skills/base.py — Abstract base class for all Jarvis2 skills.

Every skill must:
  1. Subclass Skill
  2. Define TOOL_DECLARATION (dict) with the Gemini function schema
  3. Implement execute(params: dict) -> str
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Skill(ABC):
    """Base class for all Jarvis2 skills."""

    # Subclasses MUST define this as a class-level dict.
    # Schema follows Gemini function_declarations format.
    TOOL_DECLARATION: dict[str, Any] = {}

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    @abstractmethod
    def execute(self, params: dict) -> str:
        """
        Execute the skill with the given parameters.

        Args:
            params: Dictionary of parameters from Gemini function call.

        Returns:
            A string result to send back to the Gemini session.
        """
        ...

    def log(self, msg: str) -> None:
        """Simple stdout logger with skill name prefix."""
        print(f"[{self.__class__.__name__}] {msg}")


def _get_ddgs():
    """Return the DDGS class, preferring the ddgs package over duckduckgo_search."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS
        return DDGS
