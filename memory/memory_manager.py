"""
memory/memory_manager.py — Long-term user memory for Jarvis.

Schema per entry:
  {
    "value":      "...",
    "saved_at":   "2026-05-15T19:30:00",   # always present
    "expires_at": "2026-05-22T00:00:00"    # optional, for ephemeral context
  }
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from core.settings import MEMORY_PATH


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_memory() -> dict[str, Any]:
    if not MEMORY_PATH.exists():
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_memory(memory: dict[str, Any]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def update_memory(patch: dict[str, Any], expires_in_days: int | None = None) -> None:
    """Merge patch into memory. Adds saved_at timestamp; optional TTL via expires_in_days."""
    memory = load_memory()
    now = datetime.now().isoformat(timespec="seconds")
    expires_at = (
        (datetime.now() + timedelta(days=expires_in_days)).isoformat(timespec="seconds")
        if expires_in_days
        else None
    )

    for category, entries in patch.items():
        if category not in memory:
            memory[category] = {}
        for key, data in entries.items():
            value = data.get("value", data) if isinstance(data, dict) else data
            entry: dict[str, Any] = {"value": value, "saved_at": now}
            if expires_at:
                entry["expires_at"] = expires_at
            memory[category][key] = entry

    save_memory(memory)


def forget_memory(category: str, key: str | None = None) -> str:
    """Delete a key or entire category. Returns a human-readable status."""
    memory = load_memory()

    if category not in memory:
        return f"No recuerdo nada bajo '{category}'."

    if key is None:
        del memory[category]
        save_memory(memory)
        return f"Olvidé todo lo de '{category}'."

    if key not in memory[category]:
        return f"No recuerdo '{key}' en '{category}'."

    del memory[category][key]
    if not memory[category]:
        del memory[category]
    save_memory(memory)
    return f"Olvidé '{key}' de '{category}'."


def expire_memory() -> int:
    """Remove entries past their expires_at. Called at startup. Returns count removed."""
    memory = load_memory()
    now = datetime.now()
    removed = 0

    empty_cats = []
    for category, entries in memory.items():
        stale = [
            k for k, d in entries.items()
            if isinstance(d, dict) and d.get("expires_at")
            and _parse_dt(d["expires_at"]) < now
        ]
        for k in stale:
            del entries[k]
            removed += 1
        if not entries:
            empty_cats.append(category)

    for cat in empty_cats:
        del memory[cat]

    if removed:
        save_memory(memory)

    return removed


def recall_memory(query: str | None = None, category: str | None = None) -> str:
    """Search memory by keyword or return a category. Returns formatted plain text."""
    memory = load_memory()
    if not memory:
        return "La memoria está vacía."

    if category:
        cat = category.lower()
        if cat not in memory:
            return f"No tengo nada guardado en '{category}'."
        lines = [f"{cat.upper()}:"]
        for key, data in memory[cat].items():
            value = data.get("value", data) if isinstance(data, dict) else data
            saved = (data.get("saved_at", "")[:10] if isinstance(data, dict) else "")
            suffix = f"  [guardado {saved}]" if saved else ""
            lines.append(f"  {key}: {value}{suffix}")
        return "\n".join(lines)

    if query:
        q = query.lower()
        hits = []
        for cat, entries in memory.items():
            for key, data in entries.items():
                value = str(data.get("value", data) if isinstance(data, dict) else data)
                if q in key.lower() or q in value.lower() or q in cat.lower():
                    hits.append(f"[{cat}] {key}: {value}")
        return "\n".join(hits) if hits else f"No encontré nada relacionado con '{query}'."

    return format_memory_for_prompt(memory) or "La memoria está vacía."


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_memory_for_prompt(memory: dict[str, Any]) -> str:
    """Concise block injected into the system prompt. Skips expired entries."""
    if not memory:
        return ""

    now = datetime.now()
    lines = ["[USER MEMORY]"]

    for category, entries in memory.items():
        section = []
        for key, data in entries.items():
            if isinstance(data, dict):
                exp = data.get("expires_at")
                if exp and _parse_dt(exp) < now:
                    continue
                value = data.get("value", "")
            else:
                value = data
            section.append(f"  • {key}: {value}")

        if section:
            lines.append(f"\n{category.upper()}:")
            lines.extend(section)

    if len(lines) == 1:
        return ""

    lines.append("")
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.max
