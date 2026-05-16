"""
skills/music_player.py — Audio-only music player for Jarvis2.

Plays music by piping yt-dlp audio URLs directly into mpv (no browser,
no graphical interface, no video).

Actions:
  - play    : Search and play a song (yt-dlp + mpv, audio only)
  - pause   : Pause the current track (mpv IPC socket)
  - resume  : Resume a paused track
  - stop    : Stop playback and clear session state
  - next    : Skip to the next track (replays the same query with next result)
  - volume  : Set volume 0-100
  - current : Report what is currently playing

Configuration:
  MPV_OPTIONS = --no-video --quiet   (audio-only, silent output)

The mpv process is stored in module-level session state so it survives
across multiple skill invocations within the same Jarvis session.

Uses the dedicated ``gemini_api_skill`` key to pick the best match from
multiple yt-dlp search results via the Gemini API.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any

from google import genai

from core.settings import LITE_MODEL
from skills.base import Skill

# ── MPV configuration ─────────────────────────────────────────────────────────
MPV_OPTIONS: list[str] = [
    "--no-video",    # audio-only, never open a window
    "--quiet",       # suppress stdout chatter
]

# ── Module-level session state ────────────────────────────────────────────────
# These survive across multiple execute() calls within the same process.
_state_lock = threading.Lock()
_session: dict[str, Any] = {
    "process":      None,   # subprocess.Popen | None
    "title":        None,   # str | None — human-readable track title
    "query":        None,   # str | None — original search query
    "volume":       80,     # int 0-100
    "paused":       False,  # bool
    "ipc_socket":   None,   # str | None — path to mpv IPC socket
}


# ── IPC helpers ───────────────────────────────────────────────────────────────

def _ipc_command(socket_path: str, command: list) -> dict | None:
    """Send a JSON IPC command to mpv and return the response, or None on error."""
    if not socket_path or not os.path.exists(socket_path):
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            sock.connect(socket_path)
            payload = json.dumps({"command": command}) + "\n"
            sock.sendall(payload.encode())
            data = sock.recv(4096)
            return json.loads(data.decode().strip())
    except Exception as e:
        print(f"[MusicPlayer] IPC error: {e}")
        return None


def _set_property(socket_path: str, prop: str, value: Any) -> None:
    _ipc_command(socket_path, ["set_property", prop, value])


def _get_property(socket_path: str, prop: str) -> Any:
    resp = _ipc_command(socket_path, ["get_property", prop])
    if resp and resp.get("error") == "success":
        return resp.get("data")
    return None


# ── Process helpers ───────────────────────────────────────────────────────────

def _is_running() -> bool:
    """Return True if the mpv process is alive."""
    proc = _session.get("process")
    return proc is not None and proc.poll() is None


def _stop_current() -> None:
    """Terminate any running mpv process and clear session state."""
    with _state_lock:
        proc = _session.get("process")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _session["process"]    = None
        _session["title"]      = None
        _session["query"]      = None
        _session["paused"]     = False
        _session["ipc_socket"] = None


# ── yt-dlp helpers ────────────────────────────────────────────────────────────

def _ytdlp_get_entries(search_query: str, max_results: int = 5) -> list[dict]:
    """
    Use yt-dlp to fetch metadata for up to *max_results* entries matching
    *search_query*.  Returns a list of dicts with at least ``title`` and
    ``url`` keys.  Empty list on failure.
    """
    if not shutil.which("yt-dlp"):
        return []

    prefix = f"ytsearch{max_results}:" if max_results > 1 else "ytsearch1:"
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--dump-json",
        "--no-playlist",
        "--flat-playlist",
        f"{prefix}{search_query}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )
        entries = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                url   = data.get("url") or data.get("webpage_url", "")
                title = data.get("title", url)
                if url:
                    # Ensure we have a full YouTube URL
                    if not url.startswith("http"):
                        url = f"https://www.youtube.com/watch?v={url}"
                    entries.append({"title": title, "url": url})
            except json.JSONDecodeError:
                continue
        return entries
    except Exception as e:
        print(f"[MusicPlayer] yt-dlp metadata fetch failed: {e}")
        return []


def _ytdlp_get_audio_url(video_url: str) -> str | None:
    """
    Resolve the best audio-only stream URL for a given video URL using yt-dlp.
    Returns None on failure.
    """
    if not shutil.which("yt-dlp"):
        return None
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "bestaudio",
        "--get-url",
        video_url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )
        url = result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
        return url
    except Exception as e:
        print(f"[MusicPlayer] yt-dlp audio URL resolution failed: {e}")
        return None


def _pick_best_entry(query: str, entries: list[dict], client) -> dict | None:
    """
    Ask Gemini to pick the best match from *entries* for the user's *query*.
    Falls back to the first entry if Gemini fails or there is only one result.
    """
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]

    try:
        numbered = "\n".join(
            f"{i+1}. {e['title']}" for i, e in enumerate(entries)
        )
        prompt = (
            f"A user wants to listen to: \"{query}\"\n\n"
            f"Here are the top search results:\n{numbered}\n\n"
            "Which result (by number) is the best match for the user's request? "
            "Reply with ONLY the number and nothing else."
        )
        response = client.models.generate_content(
            model=LITE_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        idx = int("".join(c for c in raw if c.isdigit())) - 1
        if 0 <= idx < len(entries):
            return entries[idx]
    except Exception as e:
        print(f"[MusicPlayer] Gemini pick failed, using first result: {e}")

    return entries[0]


# ── Skill class ───────────────────────────────────────────────────────────────

class MusicPlayerSkill(Skill):
    """
    Audio-only music player that searches YouTube via yt-dlp and streams
    directly into mpv.  No browser, no video, no graphical window.

    The mpv subprocess is kept alive in module-level session state so it
    can be paused, resumed, or stopped by subsequent commands.
    """

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key=api_key)
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

    TOOL_DECLARATION = {
        "name": "music_player",
        "description": (
            "Plays music audio-only through the speakers using yt-dlp + mpv. "
            "No browser or graphical interface is opened. "
            "Use this for: playing a song, pausing, resuming, stopping, "
            "skipping to the next track, adjusting volume, or asking what is "
            "currently playing. "
            "Examples: 'play Bohemian Rhapsody', 'pause the music', "
            "'turn the volume to 60', 'what song is playing?', 'skip this song'."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "play | pause | resume | stop | next | volume | current  "
                        "(default: play)"
                    ),
                },
                "query": {
                    "type": "STRING",
                    "description": (
                        "Song or artist to search for. Required for 'play'. "
                        "For 'next', reuses the previous query if omitted."
                    ),
                },
                "volume": {
                    "type": "INTEGER",
                    "description": "Volume level 0–100. Required for 'volume' action.",
                },
            },
            "required": ["action"],
        },
    }

    # ── Action: play ─────────────────────────────────────────────────────────

    def _play(self, params: dict) -> str:
        query = params.get("query", "").strip()
        if not query:
            return "Please tell me what song or artist you'd like to listen to."

        # Verify dependencies
        if not shutil.which("yt-dlp"):
            return "yt-dlp is not installed. Run: pip install yt-dlp"
        if not shutil.which("mpv"):
            return "mpv is not installed. Install it with your package manager (e.g. sudo apt install mpv)."

        self.log(f"Searching for: {query}")

        # Fetch up to 5 candidates, then let Gemini pick the best
        entries = _ytdlp_get_entries(query, max_results=5)
        if not entries:
            # Last resort: single result
            entries = _ytdlp_get_entries(query, max_results=1)
        if not entries:
            return f"Could not find any results for: {query}"

        best = _pick_best_entry(query, entries, self._client)
        if not best:
            return f"Could not resolve a playable result for: {query}"

        self.log(f"Selected: {best['title']} ({best['url']})")

        # Resolve the direct audio stream URL
        audio_url = _ytdlp_get_audio_url(best["url"])
        if not audio_url:
            # Fall back: let mpv resolve via yt-dlp itself
            audio_url = best["url"]

        return self._launch_mpv(audio_url, title=best["title"], query=query)

    def _launch_mpv(self, audio_url: str, title: str, query: str) -> str:
        """Stop any existing playback and start mpv with *audio_url*."""
        _stop_current()

        # Temporary IPC socket path
        ipc_path = os.path.join(tempfile.gettempdir(), f"jarvis_mpv_{os.getpid()}.sock")

        cmd = [
            "mpv",
            *MPV_OPTIONS,
            f"--input-ipc-server={ipc_path}",
            f"--volume={_session['volume']}",
            audio_url,
        ]

        self.log(f"Launching mpv: {' '.join(cmd[:6])} ...")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # detach from Jarvis's process group
            )
        except Exception as e:
            return f"Failed to start mpv: {e}"

        with _state_lock:
            _session["process"]    = proc
            _session["title"]      = title
            _session["query"]      = query
            _session["paused"]     = False
            _session["ipc_socket"] = ipc_path

        # Give mpv a moment to create the IPC socket
        for _ in range(10):
            if os.path.exists(ipc_path):
                break
            time.sleep(0.2)

        return f"Now playing: {title}"

    # ── Action: pause ────────────────────────────────────────────────────────

    def _pause(self, _params: dict) -> str:
        if not _is_running():
            return "Nothing is currently playing."
        if _session["paused"]:
            return "Music is already paused."

        sock = _session.get("ipc_socket")
        if sock:
            _set_property(sock, "pause", True)
        else:
            # IPC not available – send SIGSTOP as fallback
            try:
                _session["process"].send_signal(signal.SIGSTOP)
            except Exception:
                pass

        with _state_lock:
            _session["paused"] = True
        return "Music paused."

    # ── Action: resume ───────────────────────────────────────────────────────

    def _resume(self, _params: dict) -> str:
        if not _is_running():
            return "Nothing is currently playing. Use 'play' to start a song."
        if not _session["paused"]:
            return "Music is already playing."

        sock = _session.get("ipc_socket")
        if sock:
            _set_property(sock, "pause", False)
        else:
            try:
                _session["process"].send_signal(signal.SIGCONT)
            except Exception:
                pass

        with _state_lock:
            _session["paused"] = False
        return "Music resumed."

    # ── Action: stop ─────────────────────────────────────────────────────────

    def _stop(self, _params: dict) -> str:
        if not _is_running():
            return "Nothing is currently playing."
        title = _session.get("title", "the current track")
        _stop_current()
        return f"Stopped: {title}"

    # ── Action: next ─────────────────────────────────────────────────────────

    def _next(self, params: dict) -> str:
        """
        Skip to the next result for the same (or a new) query.

        Strategy: re-fetch 5 results, skip the current title if possible,
        and play the next one.  If a new query is provided, treat it as a
        fresh play.
        """
        query = params.get("query", "").strip() or _session.get("query", "")
        if not query:
            return "No query to skip to. Please provide a song name."

        current_title = _session.get("title")

        self.log(f"Fetching next result for: {query}")
        entries = _ytdlp_get_entries(query, max_results=5)
        if not entries:
            return f"Could not find any results for: {query}"

        # Try to find a track that is not the currently playing one
        chosen = None
        if current_title:
            for entry in entries:
                if entry["title"] != current_title:
                    chosen = entry
                    break
        if chosen is None:
            chosen = entries[0]

        self.log(f"Next track: {chosen['title']}")
        audio_url = _ytdlp_get_audio_url(chosen["url"])
        if not audio_url:
            audio_url = chosen["url"]

        return self._launch_mpv(audio_url, title=chosen["title"], query=query)

    # ── Action: volume ───────────────────────────────────────────────────────

    def _volume(self, params: dict) -> str:
        level = params.get("volume")
        if level is None:
            return "Please specify a volume level between 0 and 100."
        try:
            level = int(level)
        except (TypeError, ValueError):
            return "Volume must be a number between 0 and 100."
        level = max(0, min(100, level))

        with _state_lock:
            _session["volume"] = level

        if _is_running():
            sock = _session.get("ipc_socket")
            if sock:
                _set_property(sock, "volume", level)
            else:
                self.log("IPC socket not available; volume will apply on next track.")

        return f"Volume set to {level}%."

    # ── Action: current ──────────────────────────────────────────────────────

    def _current(self, _params: dict) -> str:
        if not _is_running():
            return "Nothing is currently playing."
        title  = _session.get("title", "Unknown")
        paused = _session.get("paused", False)
        vol    = _session.get("volume", 80)

        # Try to get live position from mpv
        sock = _session.get("ipc_socket")
        pos_str = ""
        if sock:
            pos = _get_property(sock, "playback-time")
            dur = _get_property(sock, "duration")
            if pos is not None and dur:
                def _fmt(s: float) -> str:
                    s = int(s)
                    return f"{s // 60}:{s % 60:02d}"
                pos_str = f" ({_fmt(pos)} / {_fmt(dur)})"

        status = "⏸ Paused" if paused else "▶ Playing"
        return (
            f"{status}: {title}{pos_str} | Volume: {vol}%"
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────

    _ACTIONS: dict[str, str] = {
        "play":    "_play",
        "pause":   "_pause",
        "resume":  "_resume",
        "stop":    "_stop",
        "next":    "_next",
        "volume":  "_volume",
        "current": "_current",
    }

    def execute(self, params: dict) -> str:
        action = params.get("action", "play").lower().strip()
        method_name = self._ACTIONS.get(action)
        if not method_name:
            return (
                f"Unknown music action: '{action}'. "
                "Available: play, pause, resume, stop, next, volume, current."
            )
        self.log(f"Action: {action}")
        return getattr(self, method_name)(params) or "Done."
