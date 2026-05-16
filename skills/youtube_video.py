"""
skills/youtube_video.py — YouTube interactions for Jarvis2.

Actions:
  - play      : Search for and open the first non-Shorts video in Brave browser
  - summarize : Fetch transcript and summarize with Gemini (URL via parameter)
  - get_info  : Scrape metadata for a given video URL
  - trending  : Show trending videos for a given region
"""
from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import requests
from google import genai

from core.settings import TEXT_MODEL
from skills.base import Skill


class SkillError(RuntimeError):
    """Raised when a skill cannot complete due to a configuration problem."""


# ── Brave browser detection (resolved once, then cached) ─────────────────────
# Stores the full command prefix, e.g. ["brave-browser"] or ["flatpak", "run", "com.brave.Browser"]
_BRAVE_CMD: list[str] | None = None
_BRAVE_NATIVE_BINS = ["brave", "brave-browser"]
_BRAVE_FLATPAK_ID  = "com.brave.Browser"


def _get_brave_cmd() -> list[str]:
    """
    Return the command list needed to launch Brave browser.
    Tries native binaries first, then Flatpak.  Caches the result.
    Raises SkillError with a clear message if Brave cannot be found.
    """
    global _BRAVE_CMD
    if _BRAVE_CMD is not None:
        return _BRAVE_CMD

    # 1. Native install: brave-browser / brave
    for binary in _BRAVE_NATIVE_BINS:
        if shutil.which(binary):
            _BRAVE_CMD = [binary]
            return _BRAVE_CMD

    # 2. Flatpak install: `flatpak run com.brave.Browser`
    if shutil.which("flatpak"):
        try:
            result = subprocess.run(
                ["flatpak", "list", "--app", "--columns=application"],
                capture_output=True, text=True, timeout=5,
            )
            if _BRAVE_FLATPAK_ID in result.stdout:
                _BRAVE_CMD = ["flatpak", "run", _BRAVE_FLATPAK_ID]
                return _BRAVE_CMD
        except Exception:
            pass

    tried = _BRAVE_NATIVE_BINS + [f"flatpak run {_BRAVE_FLATPAK_ID}"]
    raise SkillError(
        "Brave browser not found. Install it or check the binary name. "
        f"Tried: {', '.join(tried)}"
    )


def _open_url(url: str) -> None:
    """Open a URL in Brave browser (no xdg-open)."""
    cmd = _get_brave_cmd()  # raises SkillError if not found
    try:
        subprocess.Popen(
            [*cmd, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[youtube_video] Brave launch failed: {e}")


# ── YouTube constants ─────────────────────────────────────────────────────────
# Filter: videos only (no Shorts, playlists)
_YT_VIDEO_FILTER = "EgIQAQ%3D%3D"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_video_id(url: str) -> str | None:
    """Extract 11-char video ID from any YouTube URL format."""
    match = re.search(
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([A-Za-z0-9_-]{11})", url
    )
    return match.group(1) if match else None


def _is_valid_youtube_url(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url or ""))


def _scrape_first_video(query: str) -> str | None:
    """Return the URL of the first non-Shorts video for a query."""
    search_url = (
        f"https://www.youtube.com/results"
        f"?search_query={quote_plus(query)}"
        f"&sp={_YT_VIDEO_FILTER}"
    )
    try:
        r    = requests.get(search_url, headers=_HEADERS, timeout=10)
        html = r.text
        video_ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', html)
        seen = set()
        for vid in video_ids:
            if vid in seen:
                continue
            seen.add(vid)
            if f"/shorts/{vid}" in html:
                continue
            return f"https://www.youtube.com/watch?v={vid}"
    except Exception as e:
        print(f"[youtube_video] Scrape failed: {e}")
    return None


def _scrape_video_info(video_id: str) -> dict:
    """Scrape basic metadata from a YouTube video page."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        r    = requests.get(url, headers=_HEADERS, timeout=12)
        html = r.text
        info = {}
        for key, pattern in [
            ("title",    r'"title":\{"runs":\[\{"text":"([^"]+)"'),
            ("channel",  r'"ownerChannelName":"([^"]+)"'),
            ("views",    r'"viewCount":"(\d+)"'),
            ("duration", r'"lengthSeconds":"(\d+)"'),
            ("likes",    r'"label":"([0-9,]+ likes)"'),
        ]:
            match = re.search(pattern, html)
            if match:
                raw = match.group(1)
                if key == "views":
                    info[key] = f"{int(raw):,}"
                elif key == "duration":
                    secs = int(raw)
                    info[key] = f"{secs // 60}:{secs % 60:02d}"
                else:
                    info[key] = raw
        return info
    except Exception as e:
        print(f"[youtube_video] Info scrape failed: {e}")
        return {}


def _scrape_trending(region: str = "US", max_results: int = 8) -> list[dict]:
    """Scrape trending videos for a region."""
    url = f"https://www.youtube.com/feed/trending?gl={region.upper()}"
    try:
        r       = requests.get(url, headers=_HEADERS, timeout=12)
        html    = r.text
        titles  = re.findall(r'"title":\{"runs":\[\{"text":"([^"]+)"\}\]\}', html)
        channels= re.findall(r'"ownerText":\{"runs":\[\{"text":"([^"]+)"', html)
        results, seen = [], set()
        for i, title in enumerate(titles):
            if title in seen or len(title) < 5:
                continue
            seen.add(title)
            channel = channels[i] if i < len(channels) else "Unknown"
            results.append({"rank": len(results) + 1, "title": title, "channel": channel})
            if len(results) >= max_results:
                break
        return results
    except Exception as e:
        print(f"[youtube_video] Trending scrape failed: {e}")
        return []


def _get_transcript(video_id: str) -> str | None:
    """Fetch video transcript using youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        lang_priority   = ["en", "es", "de", "fr", "pt", "it", "ru", "ja", "ko", "ar", "zh"]
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(lang_priority)
        except Exception:
            pass
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(lang_priority)
            except Exception:
                for t in transcript_list:
                    transcript = t
                    break
        if transcript is None:
            return None
        fetched = transcript.fetch()
        return " ".join(entry["text"] for entry in fetched)
    except Exception as e:
        print(f"[youtube_video] Transcript fetch failed: {e}")
        return None


class YouTubeSkill(Skill):
    """Interact with YouTube: play, summarize, get info, trending."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key=api_key)
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

    TOOL_DECLARATION = {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "play | summarize | get_info | trending (default: play)",
                },
                "query": {
                    "type": "STRING",
                    "description": "Search query for 'play' action",
                },
                "url": {
                    "type": "STRING",
                    "description": "Full YouTube video URL for 'summarize' or 'get_info' actions",
                },
                "save": {
                    "type": "BOOLEAN",
                    "description": "Save the summary to Desktop as a .txt file (summarize only)",
                },
                "region": {
                    "type": "STRING",
                    "description": "Country code for trending (e.g. US, MX, ES). Default: US",
                },
            },
            "required": [],
        },
    }

    # ── Action handlers ───────────────────────────────────────────────────────

    def _play(self, params: dict) -> str:
        query = params.get("query", "").strip()
        if not query:
            return "Please tell me what you'd like to watch."

        self.log(f"Searching: {query}")
        video_url = _scrape_first_video(query)

        if video_url:
            self.log(f"Opening: {video_url}")
            _open_url(video_url)
            return f"Playing: {query}"

        # Fallback: open search results page
        fallback = (
            f"https://www.youtube.com/results"
            f"?search_query={quote_plus(query)}"
            f"&sp={_YT_VIDEO_FILTER}"
        )
        _open_url(fallback)
        return f"Opened YouTube search for: {query} (manual selection required)"

    def _summarize(self, params: dict) -> str:
        url = params.get("url", "").strip()
        if not url:
            return "Please provide the YouTube video URL in the 'url' parameter."
        if not _is_valid_youtube_url(url):
            return "That doesn't appear to be a valid YouTube URL."

        video_id = _extract_video_id(url)
        if not video_id:
            return "Could not extract video ID from that URL."

        self.log(f"Fetching transcript for: {url}")
        transcript = _get_transcript(video_id)
        if not transcript:
            return "Could not retrieve a transcript for that video."

        self.log("Summarizing with Gemini...")
        try:
            max_chars = 80_000
            truncated = transcript[:max_chars] + ("..." if len(transcript) > max_chars else "")
            response  = self._client.models.generate_content(
                model=TEXT_MODEL,
                contents=(
                    "You are JARVIS, an AI assistant. "
                    "Summarize this YouTube video transcript clearly and concisely. "
                    "Structure: 1-sentence overview, then 3-5 key bullet points. "
                    "Be direct. Match the language of the transcript.\n\n"
                    f"Transcript:\n{truncated}"
                ),
            )
            summary = response.text.strip()
        except Exception as e:
            return f"Summary generation failed: {e}"

        if params.get("save", False):
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path.home() / "Desktop" / f"youtube_summary_{ts}.txt"
            header   = (
                f"JARVIS — YouTube Summary\n"
                f"{'─' * 50}\n"
                f"URL  : {url}\n"
                f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"{'─' * 50}\n\n"
            )
            filepath.write_text(header + summary, encoding="utf-8")
            try:
                subprocess.Popen(
                    [*_get_brave_cmd(), str(filepath)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            return f"Summary complete. Saved to: {filepath}\n\n{summary}"

        return summary

    def _get_info(self, params: dict) -> str:
        url = params.get("url", "").strip()
        if not url or not _is_valid_youtube_url(url):
            return "Please provide a valid YouTube URL in the 'url' parameter."

        video_id = _extract_video_id(url)
        if not video_id:
            return "Could not extract video ID."

        info = _scrape_video_info(video_id)
        if not info:
            return "Could not retrieve video information."

        lines = [
            f"{key.capitalize()}: {info[key]}"
            for key in ("title", "channel", "views", "duration", "likes")
            if key in info
        ]
        return "\n".join(lines)

    def _trending(self, params: dict) -> str:
        region  = params.get("region", "US").upper()
        videos  = _scrape_trending(region=region, max_results=8)
        if not videos:
            return f"Could not fetch trending videos for region {region}."
        lines = [f"Top trending videos in {region}:"]
        lines += [f"{v['rank']}. {v['title']} — {v['channel']}" for v in videos]
        return "\n".join(lines)

    # ── Main execute ──────────────────────────────────────────────────────────

    _ACTIONS = {
        "play":      "_play",
        "summarize": "_summarize",
        "get_info":  "_get_info",
        "trending":  "_trending",
    }

    def execute(self, params: dict) -> str:
        action = params.get("action", "play").lower().strip()
        method_name = self._ACTIONS.get(action)
        if not method_name:
            return (
                f"Unknown YouTube action: '{action}'. "
                "Available: play, summarize, get_info, trending."
            )
        self.log(f"Action: {action}")
        return getattr(self, method_name)(params) or "Done."
