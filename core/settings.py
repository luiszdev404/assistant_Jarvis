"""
core/settings.py — Jarvis2 global configuration
All paths, model names, and constants live here.
"""
from __future__ import annotations

import json
from pathlib import Path

# ── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
CONFIG_DIR      = BASE_DIR / "config"
API_CONFIG_PATH = CONFIG_DIR / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
MEMORY_PATH     = BASE_DIR / "memory" / "memory.json"

# ── Obsidian vault ───────────────────────────────────────────────────────────
OBSIDIAN_VAULT  = Path.home() / "Documents" / "Obsidian" / "Jarvis-Research"

# ── Gemini models ────────────────────────────────────────────────────────────
LIVE_MODEL      = "models/gemini-2.5-flash-native-audio-preview-12-2025"
TEXT_MODEL      = "gemini-2.5-flash"
LITE_MODEL      = "gemini-2.5-flash-lite"

# ── Audio settings ───────────────────────────────────────────────────────────
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024
VOICE_NAME          = "Charon"


def get_api_key() -> str:
    """Load Gemini API key from config file."""
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def load_system_prompt() -> str:
    """Load system prompt from core/prompt.txt, with a safe fallback."""
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "You are JARVIS, an AI assistant. "
            "Be concise, professional, and always use tools to complete tasks. "
            "Never simulate results — always call the appropriate tool."
        )
