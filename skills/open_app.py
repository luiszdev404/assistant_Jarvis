"""
skills/open_app.py — Launch applications on Linux GNOME.

Strategy (in order):
  1. Direct binary lookup via shutil.which
  2. gtk-launch <desktop-entry-name>
  3. gio launch (GNOME-native fallback)
"""
from __future__ import annotations

import shutil
import subprocess
import time

from skills.base import Skill


# ── App alias table (Linux Fedora GNOME only) ─────────────────────────────────
_APP_MAP: dict[str, str] = {
    # Browsers
    "chrome":              "google-chrome",
    "google chrome":       "google-chrome",
    "chromium":            "chromium-browser",
    "firefox":             "firefox",
    "brave":               "brave",
    "edge":                "microsoft-edge",
    "opera":               "opera",
    # Communication
    "telegram":            "telegram",
    "discord":             "discord",
    "slack":               "slack",
    "zoom":                "zoom",
    "signal":              "signal",
    "whatsapp":            "whatsapp",
    "teams":               "teams",
    # Media
    "spotify":             "spotify",
    "vlc":                 "vlc",
    "mpv":                 "mpv",
    # Dev tools
    "vscode":              "code",
    "visual studio code":  "code",
    "code":                "code",
    "cursor":              "cursor",
    "gitbash":             "bash",
    "postman":             "postman",
    "insomnia":            "insomnia",
    # Terminals
    "terminal":            "gnome-terminal",
    "konsole":             "konsole",
    "alacritty":           "alacritty",
    "kitty":               "kitty",
    # Files & system
    "files":               "nautilus",
    "nautilus":            "nautilus",
    "file manager":        "nautilus",
    "file explorer":       "nautilus",
    "system monitor":      "gnome-system-monitor",
    "task manager":        "gnome-system-monitor",
    "settings":            "gnome-control-center",
    "calculator":          "gnome-calculator",
    # Office / notes
    "obsidian":            "obsidian",
    "notion":              "notion",
    "libreoffice":         "libreoffice",
    "writer":              "libreoffice --writer",
    "calc":                "libreoffice --calc",
    "impress":             "libreoffice --impress",
    "gedit":               "gedit",
    "kate":                "kate",
    # Design / creative
    "gimp":                "gimp",
    "inkscape":            "inkscape",
    "blender":             "blender",
    "figma":               "figma",
    # Gaming
    "steam":               "steam",
    "lutris":              "lutris",
    # Other
    "thunar":              "thunar",
    "timeshift":           "timeshift-gtk",
    "virtualbox":          "virtualbox",
    "vm":                  "virtualbox",
    "anki":                "anki",
    "calibre":             "calibre",
}


def _resolve(app_name: str) -> str:
    """Normalize app name to its executable/command string."""
    key = app_name.lower().strip()
    # Exact match
    if key in _APP_MAP:
        return _APP_MAP[key]
    # Partial match (key is substring of alias or vice versa)
    for alias, cmd in _APP_MAP.items():
        if alias in key or key in alias:
            return cmd
    return app_name  # fallback: use as-is


def _launch(cmd: str) -> bool:
    """
    Attempt to launch a command string.
    Supports commands with flags like 'libreoffice --writer'.
    Each step is tried exactly once; returns immediately on first success
    to prevent multiple instances from opening.
    """
    parts = cmd.split()
    binary = parts[0]

    # 1. Direct binary — if found in PATH, Popen is the only attempt.
    if shutil.which(binary):
        try:
            subprocess.Popen(
                parts,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception as e:
            print(f"[open_app] Direct launch failed: {e}")
            return False  # binary existed but failed — don't try other methods

    # 2. gtk-launch — one attempt with the binary name only.
    try:
        result = subprocess.run(
            ["gtk-launch", binary],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    # 3. gio launch — first matching .desktop file.
    import glob
    for pattern in [
        f"/usr/share/applications/{binary}*.desktop",
        f"/home/*/.local/share/applications/{binary}*.desktop",
        f"/var/lib/flatpak/exports/share/applications/*{binary}*.desktop",
    ]:
        matches = glob.glob(pattern)
        if matches:
            try:
                subprocess.Popen(
                    ["gio", "launch", matches[0]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                pass

    return False


class OpenAppSkill(Skill):
    """Launch any application on Linux Fedora GNOME."""

    TOOL_DECLARATION = {
        "name": "open_app",
        "description": (
            "Opens any application installed on the system. "
            "Use this whenever the user asks to open, launch, or start any app or program. "
            "Always call this tool — never just say you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Name of the application to open (e.g. 'Firefox', 'Obsidian', 'Terminal')",
                }
            },
            "required": ["app_name"],
        },
    }

    def execute(self, params: dict) -> str:
        app_name = params.get("app_name", "").strip()
        if not app_name:
            return "No application name provided."

        resolved = _resolve(app_name)
        self.log(f"Launching '{app_name}' → '{resolved}'")

        if _launch(resolved):
            return f"Opened {app_name}."

        return (
            f"Could not open {app_name}. "
            "It may not be installed or the name may differ."
        )
