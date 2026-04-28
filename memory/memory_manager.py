"""
memory/memory_manager.py — Long-term user memory for Jarvis2.

Persists user facts (name, preferences, projects, etc.) to a JSON file
and formats them for injection into the Gemini system prompt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.settings import MEMORY_PATH


def load_memory() -> dict[str, Any]:
    """Load memory from disk. Returns an empty dict if file does not exist."""
    if not MEMORY_PATH.exists():
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_memory(memory: dict[str, Any]) -> None:
    """Persist the full memory dict to disk."""
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def update_memory(patch: dict[str, Any]) -> None:
    """
    Merge a patch into existing memory and save.

    patch format: {category: {key: {value: ...}}}
    Example: {"identity": {"name": {"value": "Luis"}}}
    """
    memory = load_memory()
    for category, entries in patch.items():
        if category not in memory:
            memory[category] = {}
        for key, data in entries.items():
            memory[category][key] = data
    save_memory(memory)


def format_memory_for_prompt(memory: dict[str, Any]) -> str:
    """Convert memory dict into a readable block for the system prompt."""
    if not memory:
        return ""

    lines = ["[USER MEMORY]"]
    for category, entries in memory.items():
        lines.append(f"\n{category.upper()}:")
        for key, data in entries.items():
            value = data.get("value", data) if isinstance(data, dict) else data
            lines.append(f"  • {key}: {value}")
    lines.append("")
    return "\n".join(lines)
